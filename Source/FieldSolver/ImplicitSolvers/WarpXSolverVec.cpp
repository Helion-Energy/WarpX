/* Copyright 2024 Justin Angus
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"
#include "WarpX.H"

using warpx::fields::FieldType;

WarpXSolverVec::~WarpXSolverVec ()
{
    ClearData();
}

void WarpXSolverVec::ClearData () noexcept
{
    for (auto & lvl : m_array_vec)
    {
        for (int i =0; i<3; ++i)
        {
            delete lvl[i];
        }
    }
    for (auto* scalar : m_scalar_vec)
    {
        delete scalar;
    }
    m_array_vec.clear();
    m_scalar_vec.clear();
    m_multifab_blocks.clear();
    m_multifab_block_specs.clear();
}

void WarpXSolverVec::Define ( WarpX*  a_WarpX,
                              const std::string&  a_vector_type_name,
                              const std::string&  a_scalar_type_name )
{
    Define(a_WarpX, a_vector_type_name, a_scalar_type_name, {}, 1.0, 1.0);
}

void WarpXSolverVec::Define (
    WarpX* a_WarpX,
    const std::string& a_vector_type_name,
    const std::string& a_scalar_type_name,
    const std::vector<MultiFabBlockSpec>& a_multifab_block_specs,
    const RT a_vector_scale,
    const RT a_scalar_scale)
{
    DefineData(
        a_WarpX,
        a_vector_type_name,
        a_scalar_type_name,
        a_multifab_block_specs,
        a_vector_scale,
        a_scalar_scale);

    m_dofs = std::make_shared<WarpXSolverDOF>();
    m_dofs->Define(
        m_WarpX,
        m_num_amr_levels,
        m_vector_type_name,
        m_scalar_type_name,
        m_multifab_block_specs);

    m_is_defined = true;
}

void WarpXSolverVec::Define (const WarpXSolverVec& a_solver_vec)
{
    assertIsDefined(a_solver_vec);

    DefineData(
        a_solver_vec.m_WarpX,
        a_solver_vec.getVectorType(),
        a_solver_vec.getScalarType(),
        a_solver_vec.m_multifab_block_specs,
        a_solver_vec.m_array_scale,
        a_solver_vec.m_scalar_scale);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_solver_vec.m_dofs != nullptr,
        "WarpXSolverVec::Define() source DOF object is a nullptr");
    m_dofs = a_solver_vec.m_dofs;

    m_is_defined = true;
}

void WarpXSolverVec::DefineData (WarpX* a_WarpX,
                                 const std::string& a_vector_type_name,
                                 const std::string& a_scalar_type_name,
                                 const std::vector<MultiFabBlockSpec>&
                                     a_multifab_block_specs,
                                 const RT a_vector_scale,
                                 const RT a_scalar_scale)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !IsDefined(),
        "WarpXSolverVec::Define() called on already defined WarpXSolverVec");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_WarpX != nullptr,
        "WarpXSolverVec::Define() called with a nullptr WarpX object");
    m_WarpX = a_WarpX;

    m_num_amr_levels = 1;

    m_vector_type_name = a_vector_type_name;
    m_scalar_type_name = a_scalar_type_name;
    m_array_scale = a_vector_scale;
    m_scalar_scale = a_scalar_scale;
    m_multifab_block_specs = a_multifab_block_specs;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_array_scale > 0.0,
        "WarpXSolverVec vector field scale must be positive");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_scalar_scale > 0.0,
        "WarpXSolverVec scalar field scale must be positive");

    for (std::size_t i = 0; i < m_multifab_block_specs.size(); ++i) {
        auto const& spec = m_multifab_block_specs[i];
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !spec.name.empty(),
            "WarpXSolverVec MultiFab block names cannot be empty");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            spec.scale > 0.0,
            "WarpXSolverVec MultiFab block scale must be positive for " + spec.name);
        for (std::size_t j = 0; j < i; ++j) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                spec.name != m_multifab_block_specs[j].name,
                "WarpXSolverVec MultiFab block names must be unique: " + spec.name);
        }
    }

    if (m_vector_type_name=="Efield_fp") {
        m_array_type = FieldType::Efield_fp;
    }
    else if (m_vector_type_name=="Bfield_fp") {
        m_array_type = FieldType::Bfield_fp;
    }
    else if (m_vector_type_name=="vector_potential_fp_nodal") {
        m_array_type = FieldType::vector_potential_fp;
    }
    else if (m_vector_type_name!="none") {
        WARPX_ABORT_WITH_MESSAGE(a_vector_type_name+" "
                    +"is not a valid option for array type used in Definining "
                    +"a WarpXSolverVec. Valid array types are: Efield_fp, Bfield_fp, "
                    +"and vector_potential_fp_nodal");
    }

    if (m_scalar_type_name=="phi_fp") {
        m_scalar_type = FieldType::phi_fp;
    }
    else if (m_scalar_type_name!="none") {
        WARPX_ABORT_WITH_MESSAGE(a_scalar_type_name+" "
                    +"is not a valid option for scalar type used in Definining "
                    +"a WarpXSolverVec. Valid scalar types are: phi_fp");
    }

    m_array_vec.resize(m_num_amr_levels);
    m_scalar_vec.resize(m_num_amr_levels);

    // Define the 3D vector field data container
    if (m_array_type != FieldType::None) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            isFieldArray(m_array_type),
            "WarpXSolverVec::Define() called with array_type not an array field");
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            const ablastr::fields::VectorField this_array = m_WarpX->m_fields.get_alldirs(m_vector_type_name, lev);
            for (int n = 0; n < 3; n++) {
                m_array_vec[lev][n] = new amrex::MultiFab( this_array[n]->boxArray(),
                                                           this_array[n]->DistributionMap(),
                                                           this_array[n]->nComp(),
                                                           amrex::IntVect::TheZeroVector() );
            }
        }
    }

    // Define the scalar data container
    if (m_scalar_type != FieldType::None) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !isFieldArray(m_scalar_type),
            "WarpXSolverVec::Define() called with scalar_type not a scalar field ");
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            const amrex::MultiFab* this_mf = m_WarpX->m_fields.get(m_scalar_type_name,lev);
            m_scalar_vec[lev] = new amrex::MultiFab( this_mf->boxArray(),
                                                     this_mf->DistributionMap(),
                                                     this_mf->nComp(),
                                                     amrex::IntVect::TheZeroVector() );
        }
    }

    m_multifab_blocks.reserve(m_multifab_block_specs.size());
    for (auto const& spec : m_multifab_block_specs) {
        MultiFabBlock block;
        block.spec = spec;
        block.data.resize(m_num_amr_levels);
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            amrex::MultiFab const& field = *m_WarpX->m_fields.get(spec.name, lev);
            block.data[lev] = std::make_unique<amrex::MultiFab>(
                field.boxArray(),
                field.DistributionMap(),
                field.nComp(),
                amrex::IntVect::TheZeroVector());
        }
        m_multifab_blocks.push_back(std::move(block));
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_array_type != FieldType::None ||
        m_scalar_type != FieldType::None ||
        !m_multifab_blocks.empty(),
        "WarpXSolverVec must contain at least one field block");
}

void WarpXSolverVec::Copy ( warpx::fields::FieldType  a_array_type,
                            warpx::fields::FieldType  a_scalar_type,
                            bool allow_type_mismatch)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "WarpXSolverVec::Copy() called on undefined WarpXSolverVec");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (a_array_type==m_array_type &&
        a_scalar_type==m_scalar_type) || allow_type_mismatch,
        "WarpXSolverVec::Copy() called with vecs of different types");

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        if (m_array_type != FieldType::None) {
            const ablastr::fields::VectorField this_array = m_WarpX->m_fields.get_alldirs(a_array_type, lev);
            for (int n = 0; n < 3; ++n) {
                amrex::MultiFab::Copy( *m_array_vec[lev][n], *this_array[n], 0, 0, m_ncomp,
                                       amrex::IntVect::TheZeroVector() );
            }
        }
        if (m_scalar_type != FieldType::None) {
            const amrex::MultiFab* this_mf = m_WarpX->m_fields.get(a_scalar_type,lev);
            amrex::MultiFab::Copy( *m_scalar_vec[lev], *this_mf, 0, 0, m_ncomp,
                                   amrex::IntVect::TheZeroVector() );
        }
    }
}

bool WarpXSolverVec::hasMultiFabBlock (const std::string& a_name) const
{
    return std::any_of(
        m_multifab_blocks.begin(),
        m_multifab_blocks.end(),
        [&a_name] (auto const& block) { return block.spec.name == a_name; });
}

amrex::MultiFab& WarpXSolverVec::getMultiFabBlock (
    const std::string& a_name, const int a_lev)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_lev >= 0 && a_lev < m_num_amr_levels,
        "WarpXSolverVec::getMultiFabBlock() level is out of range");
    for (auto& block : m_multifab_blocks) {
        if (block.spec.name == a_name) {
            return *block.data[a_lev];
        }
    }
    WARPX_ABORT_WITH_MESSAGE(
        "WarpXSolverVec does not contain MultiFab block " + a_name);
    return *m_multifab_blocks.front().data[a_lev];
}

const amrex::MultiFab& WarpXSolverVec::getMultiFabBlock (
    const std::string& a_name, const int a_lev) const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_lev >= 0 && a_lev < m_num_amr_levels,
        "WarpXSolverVec::getMultiFabBlock() level is out of range");
    for (auto const& block : m_multifab_blocks) {
        if (block.spec.name == a_name) {
            return *block.data[a_lev];
        }
    }
    WARPX_ABORT_WITH_MESSAGE(
        "WarpXSolverVec does not contain MultiFab block " + a_name);
    return *m_multifab_blocks.front().data[a_lev];
}

void WarpXSolverVec::CopyMultiFabBlocksFromFields ()
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "WarpXSolverVec::CopyMultiFabBlocksFromFields() called on undefined object");
    for (auto& block : m_multifab_blocks) {
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            amrex::MultiFab const& source =
                *m_WarpX->m_fields.get(block.spec.name, lev);
            amrex::MultiFab::Copy(
                *block.data[lev],
                source,
                0, 0, source.nComp(), amrex::IntVect::TheZeroVector());
        }
    }
}

void WarpXSolverVec::CopyMultiFabBlocksToFields () const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "WarpXSolverVec::CopyMultiFabBlocksToFields() called on undefined object");
    for (auto const& block : m_multifab_blocks) {
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            amrex::MultiFab& destination =
                *m_WarpX->m_fields.get(block.spec.name, lev);
            amrex::MultiFab::Copy(
                destination,
                *block.data[lev],
                0, 0, destination.nComp(), amrex::IntVect::TheZeroVector());
        }
    }
}

void WarpXSolverVec::copyFrom ( const amrex::Real* const a_arr)
{
    BL_PROFILE("WarpXSolverVec::copyFrom");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "WarpXSolverVec::CopyFrom() called on undefined WarpXSolverVec");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (m_dofs != nullptr),
        "WarpXSolverVec::CopyFrom() DOF object is a nullptr");
    const amrex::Real array_scale = m_array_scale;
    const amrex::Real scalar_scale = m_scalar_scale;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        if (m_array_type != FieldType::None) {
            for (int n = 0; n < 3; ++n) {
                auto ncomp = m_array_vec[lev][n]->nComp();
                for (amrex::MFIter mfi(*(m_dofs->m_array)[lev][n]); mfi.isValid(); ++mfi) {
                    auto bx = mfi.tilebox();
                    auto data_arr = m_array_vec[lev][n]->array(mfi);
                    auto dof_arr = m_dofs->m_array[lev][n]->const_array(mfi);
                    ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        for (int v = 0; v < ncomp; v++) {
                            const  int dof = dof_arr(i,j,k,2*v); // local
                            if (dof >= 0) {
                                data_arr(i,j,k,v) = array_scale * a_arr[dof];
                            }
                        }
                    });
                }
                m_array_vec[lev][n]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
            }
        }
        if (m_scalar_type != FieldType::None) {
            auto ncomp = m_scalar_vec[lev]->nComp();
            for (amrex::MFIter mfi(*(m_dofs->m_scalar)[lev]); mfi.isValid(); ++mfi) {
                auto bx = mfi.tilebox();
                auto data_arr = m_scalar_vec[lev]->array(mfi);
                auto dof_arr = m_dofs->m_scalar[lev]->const_array(mfi);
                ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    for (int v = 0; v < ncomp; v++) {
                        const int dof = dof_arr(i,j,k,2*v); // local
                        if (dof >= 0) {
                            data_arr(i,j,k,v) = scalar_scale * a_arr[dof];
                        }
                    }
                });
            }
            m_scalar_vec[lev]->FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
        }
        for (std::size_t iblock = 0; iblock < m_multifab_blocks.size(); ++iblock) {
            auto& field = *m_multifab_blocks[iblock].data[lev];
            auto const& dofs = *m_dofs->m_multifab_blocks[iblock].dofs[lev];
            const int ncomp = field.nComp();
            const amrex::Real scale = m_multifab_blocks[iblock].spec.scale;
            for (amrex::MFIter mfi(dofs); mfi.isValid(); ++mfi) {
                auto const bx = mfi.tilebox();
                auto const data_arr = field.array(mfi);
                auto const dof_arr = dofs.const_array(mfi);
                ParallelFor(
                    bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        for (int v = 0; v < ncomp; ++v) {
                            const int dof = dof_arr(i,j,k,2*v);
                            if (dof >= 0) {
                                data_arr(i,j,k,v) = scale * a_arr[dof];
                            }
                        }
                    });
            }
            field.FillBoundaryAndSync(m_WarpX->Geom(lev).periodicity());
        }
    }
}

void WarpXSolverVec::copyTo ( amrex::Real* const a_arr) const
{
    BL_PROFILE("WarpXSolverVec::copyTo");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "WarpXSolverVec::CopyTo() called on undefined WarpXSolverVec");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (m_dofs != nullptr),
        "WarpXSolverVec::CopyTo() DOF object is a nullptr");
    const amrex::Real inverse_array_scale = 1.0 / m_array_scale;
    const amrex::Real inverse_scalar_scale = 1.0 / m_scalar_scale;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        if (m_array_type != FieldType::None) {
            for (int n = 0; n < 3; ++n) {
                auto ncomp = m_array_vec[lev][n]->nComp();
                for (amrex::MFIter mfi(*(m_dofs->m_array)[lev][n]); mfi.isValid(); ++mfi) {
                    auto bx = mfi.tilebox();
                    auto data_arr = m_array_vec[lev][n]->const_array(mfi);
                    auto dof_arr = m_dofs->m_array[lev][n]->const_array(mfi);
                    ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        for (int v = 0; v < ncomp; v++) {
                            const int dof = dof_arr(i,j,k,2*v); // local
                            if (dof >= 0) {
                                a_arr[dof] = inverse_array_scale * data_arr(i,j,k,v);
                            }
                        }
                    });
                }
            }
        }
        if (m_scalar_type != FieldType::None) {
            auto ncomp = m_scalar_vec[lev]->nComp();
            for (amrex::MFIter mfi(*(m_dofs->m_scalar)[lev]); mfi.isValid(); ++mfi) {
                auto bx = mfi.tilebox();
                auto data_arr = m_scalar_vec[lev]->const_array(mfi);
                auto dof_arr = m_dofs->m_scalar[lev]->const_array(mfi);
                ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    for (int v = 0; v < ncomp; v++) {
                        const int dof = dof_arr(i,j,k,2*v); // local
                        if (dof >= 0) {
                            a_arr[dof] = inverse_scalar_scale * data_arr(i,j,k,v);
                        }
                    }
                });
            }
        }
        for (std::size_t iblock = 0; iblock < m_multifab_blocks.size(); ++iblock) {
            auto const& field = *m_multifab_blocks[iblock].data[lev];
            auto const& dofs = *m_dofs->m_multifab_blocks[iblock].dofs[lev];
            const int ncomp = field.nComp();
            const amrex::Real inverse_scale =
                1.0 / m_multifab_blocks[iblock].spec.scale;
            for (amrex::MFIter mfi(dofs); mfi.isValid(); ++mfi) {
                auto const bx = mfi.tilebox();
                auto const data_arr = field.const_array(mfi);
                auto const dof_arr = dofs.const_array(mfi);
                ParallelFor(
                    bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        for (int v = 0; v < ncomp; ++v) {
                            const int dof = dof_arr(i,j,k,2*v);
                            if (dof >= 0) {
                                a_arr[dof] = inverse_scale * data_arr(i,j,k,v);
                            }
                        }
                    });
            }
        }
    }
}

[[nodiscard]] amrex::Real WarpXSolverVec::dotProduct ( const WarpXSolverVec&  a_X ) const
{
    assertIsDefined( a_X );
    assertSameType( a_X );

    amrex::Real result = 0.0;
    const bool local = true;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        if (m_array_type != FieldType::None) {
            for (int n = 0; n < 3; ++n) {
                const amrex::iMultiFab* dotMask = m_WarpX->getFieldDotMaskPointer(m_array_type, lev, ablastr::fields::Direction{n});
                auto rtmp = amrex::MultiFab::Dot( *dotMask,
                                                  *m_array_vec[lev][n], 0,
                                                  *a_X.getArrayVec()[lev][n], 0, 1, 0, local);
                result += rtmp / (m_array_scale * m_array_scale);
            }
        }
        if (m_scalar_type != FieldType::None) {
            const amrex::iMultiFab* dotMask = m_WarpX->getFieldDotMaskPointer(m_scalar_type,lev, ablastr::fields::Direction{0});
            auto rtmp = amrex::MultiFab::Dot( *dotMask,
                                              *m_scalar_vec[lev], 0,
                                              *a_X.getScalarVec()[lev], 0, 1, 0, local);
            result += rtmp / (m_scalar_scale * m_scalar_scale);
        }
        for (std::size_t iblock = 0; iblock < m_multifab_blocks.size(); ++iblock) {
            auto const& block = m_multifab_blocks[iblock];
            auto const& other_block = a_X.m_multifab_blocks[iblock];
            auto const& mask = *m_dofs->m_multifab_blocks[iblock].masks[lev];
            const int ncomp = block.data[lev]->nComp();
            const amrex::Real inverse_scale = 1.0 / block.spec.scale;
            const amrex::Real rtmp = amrex::MultiFab::Dot(
                mask,
                *block.data[lev], 0,
                *other_block.data[lev], 0,
                ncomp, 0, local);
            result += inverse_scale * inverse_scale * rtmp;
        }
    }
    amrex::ParallelAllReduce::Sum(result, amrex::ParallelContext::CommunicatorSub());
    return result;
}
