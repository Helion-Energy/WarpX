/* Copyright 2024 Justin Angus
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_Loop.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>

using warpx::fields::FieldType;

namespace
{
    /**
     * Segment of the fused index space holding a_idx: the last segment whose
     * start offset is <= a_idx (binary search over the nseg+1 offsets).
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int fused_segment (const amrex::Long* a_offsets, const int a_nseg, const amrex::Long a_idx) noexcept
    {
        int lo = 0;
        int hi = a_nseg;
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (a_offsets[mid] <= a_idx) { lo = mid; } else { hi = mid; }
        }
        return lo;
    }

    struct FusedIndex { int i; int j; int k; int n; };

    /**
     * Grid index and component of the a_local-th entry of a segment: i fastest,
     * then the other space dimensions, then the component.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    FusedIndex fused_decode (const amrex::Box& a_box, amrex::Long a_local) noexcept
    {
        const auto lo = a_box.smallEnd();
        const auto len = a_box.length();
        FusedIndex c{0, 0, 0, 0};
        c.i = static_cast<int>(a_local % len[0]);
        a_local /= len[0];
#if (AMREX_SPACEDIM >= 2)
        c.j = static_cast<int>(a_local % len[1]);
        a_local /= len[1];
#endif
#if (AMREX_SPACEDIM == 3)
        c.k = static_cast<int>(a_local % len[2]);
        a_local /= len[2];
#endif
        c.n = static_cast<int>(a_local);
        c.i += lo[0];
#if (AMREX_SPACEDIM >= 2)
        c.j += lo[1];
#endif
#if (AMREX_SPACEDIM == 3)
        c.k += lo[2];
#endif
        return c;
    }
}

int WarpXSolverVec::FusedLevel ()
{
    static int level = -1;
    if (level < 0) {
        level = 0;
        const amrex::ParmParse pp("implicit_evolve");
        pp.query("fused_vector_ops", level);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            level >= 0 && level <= 2,
            "implicit_evolve.fused_vector_ops must be 0, 1 or 2");
    }
    return level;
}

const amrex::Array4<amrex::Real>* WarpXSolverVec::fusedArrays () const
{
    if (!m_fused_arrays_built) {
        BL_PROFILE("WarpXSolverVec::fusedArrays(build)");
        constexpr int lev = 0;
        std::vector<amrex::Array4<amrex::Real>> table;
        table.reserve(m_dofs->m_fused.host_segments.size());
        auto add = [&table] (amrex::MultiFab& mf)
        {
            for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) { table.push_back(mf.array(mfi)); }
        };
        if (m_array_type != FieldType::None) {
            for (int n = 0; n < 3; ++n) { add(*m_array_vec[lev][n]); }
        }
        if (m_scalar_type != FieldType::None) { add(*m_scalar_vec[lev]); }
        for (auto const& block : m_multifab_blocks) { add(*block.data[lev]); }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            table.size() == m_dofs->m_fused.host_segments.size(),
            "WarpXSolverVec: fused layout does not match the vector's MultiFabs");
        m_fused_arrays.resize(table.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, table.begin(), table.end(), m_fused_arrays.begin());
        amrex::Gpu::streamSynchronize();
        m_fused_arrays_built = true;
    }
    return m_fused_arrays.data();
}

void WarpXSolverVec::fusedCopy (const WarpXSolverVec& X)
{
    BL_PROFILE("WarpXSolverVec::fusedCopy");
    auto const& layout = m_dofs->m_fused;
    const auto* segs = layout.segments.data();
    const auto* offs = layout.offsets.data();
    const int nseg = layout.nseg;
    const auto* ya = fusedArrays();
    const auto* xa = X.fusedArrays();
    amrex::ParallelFor(layout.total, [=] AMREX_GPU_DEVICE (amrex::Long idx) noexcept
    {
        const int s = fused_segment(offs, nseg, idx);
        const auto c = fused_decode(segs[s].box, idx - offs[s]);
        ya[s](c.i, c.j, c.k, c.n) = xa[s](c.i, c.j, c.k, c.n);
    });
}

void WarpXSolverVec::fusedPlus (const WarpXSolverVec& X, const RT sign)
{
    BL_PROFILE("WarpXSolverVec::fusedPlus");
    auto const& layout = m_dofs->m_fused;
    const auto* segs = layout.segments.data();
    const auto* offs = layout.offsets.data();
    const int nseg = layout.nseg;
    const auto* ya = fusedArrays();
    const auto* xa = X.fusedArrays();
    const bool add = (sign > 0.0);
    amrex::ParallelFor(layout.total, [=] AMREX_GPU_DEVICE (amrex::Long idx) noexcept
    {
        const int s = fused_segment(offs, nseg, idx);
        const auto c = fused_decode(segs[s].box, idx - offs[s]);
        if (add) { ya[s](c.i, c.j, c.k, c.n) += xa[s](c.i, c.j, c.k, c.n); }
        else     { ya[s](c.i, c.j, c.k, c.n) -= xa[s](c.i, c.j, c.k, c.n); }
    });
}

void WarpXSolverVec::fusedLinComb (const RT a, const WarpXSolverVec& X, const RT b, const WarpXSolverVec& Y)
{
    BL_PROFILE("WarpXSolverVec::fusedLinComb");
    auto const& layout = m_dofs->m_fused;
    const auto* segs = layout.segments.data();
    const auto* offs = layout.offsets.data();
    const int nseg = layout.nseg;
    const auto* da = fusedArrays();
    const auto* xa = X.fusedArrays();
    const auto* yb = Y.fusedArrays();
    amrex::ParallelFor(layout.total, [=] AMREX_GPU_DEVICE (amrex::Long idx) noexcept
    {
        const int s = fused_segment(offs, nseg, idx);
        const auto c = fused_decode(segs[s].box, idx - offs[s]);
        da[s](c.i, c.j, c.k, c.n) = a * xa[s](c.i, c.j, c.k, c.n) + b * yb[s](c.i, c.j, c.k, c.n);
    });
}

void WarpXSolverVec::fusedIncrement (const WarpXSolverVec& X, const RT a)
{
    BL_PROFILE("WarpXSolverVec::fusedIncrement");
    auto const& layout = m_dofs->m_fused;
    const auto* segs = layout.segments.data();
    const auto* offs = layout.offsets.data();
    const int nseg = layout.nseg;
    const auto* ya = fusedArrays();
    const auto* xa = X.fusedArrays();
    amrex::ParallelFor(layout.total, [=] AMREX_GPU_DEVICE (amrex::Long idx) noexcept
    {
        const int s = fused_segment(offs, nseg, idx);
        const auto c = fused_decode(segs[s].box, idx - offs[s]);
        ya[s](c.i, c.j, c.k, c.n) += a * xa[s](c.i, c.j, c.k, c.n);
    });
}

void WarpXSolverVec::fusedScale (const RT a)
{
    BL_PROFILE("WarpXSolverVec::fusedScale");
    auto const& layout = m_dofs->m_fused;
    const auto* segs = layout.segments.data();
    const auto* offs = layout.offsets.data();
    const int nseg = layout.nseg;
    const auto* ya = fusedArrays();
    amrex::ParallelFor(layout.total, [=] AMREX_GPU_DEVICE (amrex::Long idx) noexcept
    {
        const int s = fused_segment(offs, nseg, idx);
        const auto c = fused_decode(segs[s].box, idx - offs[s]);
        ya[s](c.i, c.j, c.k, c.n) *= a;
    });
}

void WarpXSolverVec::fusedSetVal (const RT a)
{
    BL_PROFILE("WarpXSolverVec::fusedSetVal");
    auto const& layout = m_dofs->m_fused;
    const auto* segs = layout.segments.data();
    const auto* offs = layout.offsets.data();
    const int nseg = layout.nseg;
    const auto* ya = fusedArrays();
    amrex::ParallelFor(layout.total, [=] AMREX_GPU_DEVICE (amrex::Long idx) noexcept
    {
        const int s = fused_segment(offs, nseg, idx);
        const auto c = fused_decode(segs[s].box, idx - offs[s]);
        ya[s](c.i, c.j, c.k, c.n) = a;
    });
}

amrex::Real WarpXSolverVec::fusedDot (const WarpXSolverVec& X, const bool a_apply_block_scales) const
{
    BL_PROFILE("WarpXSolverVec::fusedDot");
    auto const& layout = m_dofs->m_fused;
    const auto* segs = layout.segments.data();
    const auto* offs = layout.offsets.data();
    const int nseg = layout.nseg;
    const auto* ya = fusedArrays();
    const auto* xa = X.fusedArrays();
    const bool weighted = a_apply_block_scales;
    amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    reduce_op.eval(layout.total, reduce_data,
        [=] AMREX_GPU_DEVICE (amrex::Long idx) noexcept -> ReduceTuple
        {
            const int s = fused_segment(offs, nseg, idx);
            const auto c = fused_decode(segs[s].box, idx - offs[s]);
            if (!segs[s].mask(c.i, c.j, c.k)) { return {0.0}; }
            const amrex::Real weight = weighted ? segs[s].inv_scale2 : 1.0;
            return {weight * ya[s](c.i, c.j, c.k, c.n) * xa[s](c.i, c.j, c.k, c.n)};
        });
    amrex::Real result = amrex::get<0>(reduce_data.value(reduce_op));
    amrex::ParallelAllReduce::Sum(result, amrex::ParallelContext::CommunicatorSub());
    return result;
}

void WarpXSolverVec::fusedCopyTo (amrex::Real* const a_arr) const
{
    BL_PROFILE("WarpXSolverVec::fusedCopyTo");
    auto const& layout = m_dofs->m_fused;
    const auto* segs = layout.segments.data();
    const auto* offs = layout.offsets.data();
    const int nseg = layout.nseg;
    const auto* ya = fusedArrays();
    amrex::ParallelFor(layout.total, [=] AMREX_GPU_DEVICE (amrex::Long idx) noexcept
    {
        const int s = fused_segment(offs, nseg, idx);
        const auto c = fused_decode(segs[s].box, idx - offs[s]);
        const int dof = segs[s].dof(c.i, c.j, c.k, 2*c.n); // local
        if (dof >= 0) { a_arr[dof] = segs[s].inv_scale * ya[s](c.i, c.j, c.k, c.n); }
    });
}

void WarpXSolverVec::fusedCopyFrom (const amrex::Real* const a_arr)
{
    BL_PROFILE("WarpXSolverVec::fusedCopyFrom");
    auto const& layout = m_dofs->m_fused;
    const auto* segs = layout.segments.data();
    const auto* offs = layout.offsets.data();
    const int nseg = layout.nseg;
    const auto* ya = fusedArrays();
    amrex::ParallelFor(layout.total, [=] AMREX_GPU_DEVICE (amrex::Long idx) noexcept
    {
        const int s = fused_segment(offs, nseg, idx);
        const auto c = fused_decode(segs[s].box, idx - offs[s]);
        const int dof = segs[s].dof(c.i, c.j, c.k, 2*c.n); // local
        if (dof >= 0) { ya[s](c.i, c.j, c.k, c.n) = segs[s].scale * a_arr[dof]; }
    });
    // duplicates of shared grid points (staggered fields) take the owner's value
    constexpr int lev = 0;
    const auto periodicity = m_WarpX->Geom(lev).periodicity();
    if (m_array_type != FieldType::None) {
        for (int n = 0; n < 3; ++n) { m_array_vec[lev][n]->FillBoundaryAndSync(periodicity); }
    }
    if (m_scalar_type != FieldType::None) { m_scalar_vec[lev]->FillBoundaryAndSync(periodicity); }
    for (auto& block : m_multifab_blocks) { block.data[lev]->FillBoundaryAndSync(periodicity); }
}

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
    m_fused_arrays.clear();
    m_fused_arrays_built = false;
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
        m_multifab_block_specs,
        m_array_scale,
        m_scalar_scale);

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
    if (FusedLevel() >= 1 && fusedAvailable()) { fusedCopyFrom(a_arr); return; }
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
    if (FusedLevel() >= 1 && fusedAvailable()) { fusedCopyTo(a_arr); return; }
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
    return dotProduct(a_X, true);
}

[[nodiscard]] amrex::Real WarpXSolverVec::dotProduct ( const WarpXSolverVec&  a_X,
                                                       bool a_apply_block_scales ) const
{
    assertIsDefined( a_X );
    assertSameType( a_X );
    // NOTE for the merge with the component-probe branch (dotProduct(a_X,
    // a_apply_block_scales)): the fused path must receive that flag,
    // fusedDot(a_X, a_apply_block_scales), or the unweighted component norm
    // would silently become the weighted solver norm at fused_vector_ops = 2.
    if (FusedLevel() >= 2 && fusedAvailable()) { return fusedDot(a_X, true); }

    amrex::Real result = 0.0;
    const bool local = true;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        // The weighted branches keep the ORIGINAL arithmetic (a division by
        // the squared scale; multiplying by a reciprocal differs by an ulp,
        // which changes the probe epsilon and hence sensitive Krylov paths).
        if (m_array_type != FieldType::None) {
            for (int n = 0; n < 3; ++n) {
                const amrex::iMultiFab* dotMask = m_WarpX->getFieldDotMaskPointer(m_array_type, lev, ablastr::fields::Direction{n});
                auto rtmp = amrex::MultiFab::Dot( *dotMask,
                                                  *m_array_vec[lev][n], 0,
                                                  *a_X.getArrayVec()[lev][n], 0, 1, 0, local);
                if (a_apply_block_scales) { result += rtmp / (m_array_scale * m_array_scale); }
                else { result += rtmp; }
            }
        }
        if (m_scalar_type != FieldType::None) {
            const amrex::iMultiFab* dotMask = m_WarpX->getFieldDotMaskPointer(m_scalar_type,lev, ablastr::fields::Direction{0});
            auto rtmp = amrex::MultiFab::Dot( *dotMask,
                                              *m_scalar_vec[lev], 0,
                                              *a_X.getScalarVec()[lev], 0, 1, 0, local);
            if (a_apply_block_scales) { result += rtmp / (m_scalar_scale * m_scalar_scale); }
            else { result += rtmp; }
        }
        for (std::size_t iblock = 0; iblock < m_multifab_blocks.size(); ++iblock) {
            auto const& block = m_multifab_blocks[iblock];
            auto const& other_block = a_X.m_multifab_blocks[iblock];
            auto const& mask = *m_dofs->m_multifab_blocks[iblock].masks[lev];
            const int ncomp = block.data[lev]->nComp();
            const amrex::Real rtmp = amrex::MultiFab::Dot(
                mask,
                *block.data[lev], 0,
                *other_block.data[lev], 0,
                ncomp, 0, local);
            if (a_apply_block_scales) {
                const amrex::Real inverse_scale = 1.0 / block.spec.scale;
                result += inverse_scale * inverse_scale * rtmp;
            } else {
                result += rtmp;
            }
        }
    }
    amrex::ParallelAllReduce::Sum(result, amrex::ParallelContext::CommunicatorSub());
    return result;
}

void WarpXSolverVec::abs ()
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(), "WarpXSolverVec::abs() called on undefined WarpXSolverVec");
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        if (m_array_type != FieldType::None) {
            for (int n = 0; n < 3; ++n) { m_array_vec[lev][n]->abs(0, 1, 0); }
        }
        if (m_scalar_type != FieldType::None) { m_scalar_vec[lev]->abs(0, 1, 0); }
        for (auto& block : m_multifab_blocks) {
            block.data[lev]->abs(0, block.data[lev]->nComp(), 0);
        }
    }
}

void WarpXSolverVec::addBlockConstants (const std::vector<amrex::Real>& a_block_values)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "WarpXSolverVec::addBlockConstants() called on undefined WarpXSolverVec");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_block_values.size() == blockNames().size(),
        "WarpXSolverVec::addBlockConstants(): one value per field block is required");
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        std::size_t iv = 0;
        if (m_array_type != FieldType::None) {
            for (int n = 0; n < 3; ++n) { m_array_vec[lev][n]->plus(a_block_values[iv], 0, 1, 0); }
            ++iv;
        }
        if (m_scalar_type != FieldType::None) {
            m_scalar_vec[lev]->plus(a_block_values[iv], 0, 1, 0);
            ++iv;
        }
        for (auto& block : m_multifab_blocks) {
            block.data[lev]->plus(a_block_values[iv], 0, block.data[lev]->nComp(), 0);
            ++iv;
        }
    }
}

void WarpXSolverVec::divideBy (const WarpXSolverVec& a_D)
{
    assertIsDefined( a_D );
    assertSameType( a_D );
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        if (m_array_type != FieldType::None) {
            for (int n = 0; n < 3; ++n) {
                amrex::MultiFab::Divide(*m_array_vec[lev][n], *a_D.getArrayVec()[lev][n],
                                        0, 0, 1, 0);
            }
        }
        if (m_scalar_type != FieldType::None) {
            amrex::MultiFab::Divide(*m_scalar_vec[lev], *a_D.getScalarVec()[lev], 0, 0, 1, 0);
        }
        for (std::size_t iblock = 0; iblock < m_multifab_blocks.size(); ++iblock) {
            auto& block = *m_multifab_blocks[iblock].data[lev];
            amrex::MultiFab::Divide(block, *a_D.m_multifab_blocks[iblock].data[lev],
                                    0, 0, block.nComp(), 0);
        }
    }
}

std::vector<std::string> WarpXSolverVec::blockNames () const
{
    std::vector<std::string> names;
    if (m_array_type != FieldType::None) { names.push_back(m_vector_type_name); }
    if (m_scalar_type != FieldType::None) { names.push_back(m_scalar_type_name); }
    for (auto const& spec : m_multifab_block_specs) { names.push_back(spec.name); }
    return names;
}

std::vector<amrex::Real> WarpXSolverVec::blockScales () const
{
    std::vector<amrex::Real> scales;
    if (m_array_type != FieldType::None) { scales.push_back(m_array_scale); }
    if (m_scalar_type != FieldType::None) { scales.push_back(m_scalar_scale); }
    for (auto const& spec : m_multifab_block_specs) { scales.push_back(spec.scale); }
    return scales;
}

namespace
{
    /**
     * One entry of a fixed-capacity, descending-|value| cell list.
     */
    struct ResidualCell
    {
        amrex::Real value = 0.0;
        int i = 0;
        int j = 0;
        int k = 0;
        int comp = 0;
    };

    void InsertTopCell (std::vector<ResidualCell>& a_cells,
                        const std::size_t a_capacity,
                        const ResidualCell& a_cell)
    {
        if (a_cells.size() == a_capacity &&
            std::abs(a_cell.value) <= std::abs(a_cells.back().value)) {
            return;
        }
        auto const pos = std::find_if(
            a_cells.begin(), a_cells.end(),
            [&a_cell] (const ResidualCell& cell)
            { return std::abs(a_cell.value) > std::abs(cell.value); });
        a_cells.insert(pos, a_cell);
        if (a_cells.size() > a_capacity) { a_cells.resize(a_capacity); }
    }

    /**
     * Scan one MultiFab on the host (device data is staged through a host
     * copy) and merge its largest-|value| cells into a_cells. a_comp_offset
     * shifts the reported component index, so the three MultiFabs of an
     * array-type block can share one component axis.
     */
    void CollectTopCells (const amrex::MultiFab& a_mf,
                          const int a_comp_offset,
                          const std::size_t a_capacity,
                          std::vector<ResidualCell>& a_cells)
    {
        for (amrex::MFIter mfi(a_mf); mfi.isValid(); ++mfi) {
            const auto& fab = a_mf[mfi];
            const amrex::Box& box = fab.box();
            const int ncomp = fab.nComp();
            const auto npts = static_cast<std::size_t>(box.numPts()) *
                              static_cast<std::size_t>(ncomp);
            std::vector<amrex::Real> host_data(npts);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, fab.dataPtr(),
                             fab.dataPtr() + npts, host_data.data());
            const auto host_arr = amrex::makeArray4<const amrex::Real>(
                host_data.data(), box, ncomp);
            amrex::LoopOnCpu(
                box, ncomp,
                [&] (int i, int j, int k, int n)
                {
                    InsertTopCell(a_cells, a_capacity,
                                  {host_arr(i, j, k, n), i, j, k,
                                   n + a_comp_offset});
                });
        }
    }
}

void WarpXSolverVec::ReportBlockNorms (const RT a_share_threshold,
                                       const int a_num_top_cells) const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "WarpXSolverVec::ReportBlockNorms() called on undefined WarpXSolverVec");

    // Per-block squared scaled norms, matching the per-block contributions
    // of dotProduct(*this) so the shares sum to one.
    std::vector<std::string> names;
    std::vector<amrex::Real> norms_sq;
    const bool local = true;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        if (m_array_type != FieldType::None) {
            amrex::Real block_norm_sq = 0.0;
            for (int n = 0; n < 3; ++n) {
                const amrex::iMultiFab* dotMask = m_WarpX->getFieldDotMaskPointer(
                    m_array_type, lev, ablastr::fields::Direction{n});
                block_norm_sq += amrex::MultiFab::Dot(
                    *dotMask, *m_array_vec[lev][n], 0,
                    *m_array_vec[lev][n], 0, 1, 0, local) /
                    (m_array_scale * m_array_scale);
            }
            names.push_back(m_vector_type_name);
            norms_sq.push_back(block_norm_sq);
        }
        if (m_scalar_type != FieldType::None) {
            const amrex::iMultiFab* dotMask = m_WarpX->getFieldDotMaskPointer(
                m_scalar_type, lev, ablastr::fields::Direction{0});
            names.push_back(m_scalar_type_name);
            norms_sq.push_back(amrex::MultiFab::Dot(
                *dotMask, *m_scalar_vec[lev], 0,
                *m_scalar_vec[lev], 0, 1, 0, local) /
                (m_scalar_scale * m_scalar_scale));
        }
        for (std::size_t iblock = 0; iblock < m_multifab_blocks.size(); ++iblock) {
            auto const& block = m_multifab_blocks[iblock];
            auto const& mask = *m_dofs->m_multifab_blocks[iblock].masks[lev];
            names.push_back(block.spec.name);
            norms_sq.push_back(amrex::MultiFab::Dot(
                mask, *block.data[lev], 0, *block.data[lev], 0,
                block.data[lev]->nComp(), 0, local) /
                (block.spec.scale * block.spec.scale));
        }
    }
    amrex::ParallelAllReduce::Sum(norms_sq.data(),
                                  static_cast<int>(norms_sq.size()),
                                  amrex::ParallelContext::CommunicatorSub());
    amrex::Real total_sq = 0.0;
    for (auto const norm_sq : norms_sq) { total_sq += norm_sq; }

    std::stringstream report;
    report << "Newton residual blocks: total "
           << std::scientific << std::setprecision(5) << std::sqrt(total_sq);
    for (std::size_t n = 0; n < names.size(); ++n) {
        const amrex::Real share =
            (total_sq > 0.0) ? norms_sq[n] / total_sq : 0.0;
        report << " | " << names[n] << " "
               << std::scientific << std::setprecision(3)
               << std::sqrt(norms_sq[n])
               << " (" << std::fixed << std::setprecision(4) << share << ")";
    }
    amrex::Print() << report.str() << "\n";

    // For each dominant block, the largest-|value| cells (per rank; the
    // scan skips the DOF ownership masks, so a cell on a shared box face
    // can at most be reported twice).
    if (a_num_top_cells <= 0) { return; }
    const auto capacity = static_cast<std::size_t>(a_num_top_cells);
    for (std::size_t n = 0; n < names.size(); ++n) {
        if (total_sq <= 0.0 || norms_sq[n] / total_sq <= a_share_threshold) {
            continue;
        }
        std::vector<ResidualCell> cells;
        cells.reserve(capacity + 1);
        amrex::Real block_scale = 1.0;
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            if (m_array_type != FieldType::None &&
                names[n] == m_vector_type_name) {
                block_scale = m_array_scale;
                for (int dir = 0; dir < 3; ++dir) {
                    CollectTopCells(*m_array_vec[lev][dir], dir,
                                    capacity, cells);
                }
            }
            if (m_scalar_type != FieldType::None &&
                names[n] == m_scalar_type_name) {
                block_scale = m_scalar_scale;
                CollectTopCells(*m_scalar_vec[lev], 0, capacity, cells);
            }
            for (auto const& block : m_multifab_blocks) {
                if (names[n] == block.spec.name) {
                    block_scale = block.spec.scale;
                    CollectTopCells(*block.data[lev], 0, capacity, cells);
                }
            }
        }
        std::stringstream top_report;
        for (auto const& cell : cells) {
            top_report << "Newton residual top cell: " << names[n]
                       << "[" << cell.comp << "] (" << cell.i << ","
                       << cell.j << "," << cell.k << ") value "
                       << std::scientific << std::setprecision(5)
                       << cell.value << " scaled "
                       << cell.value / block_scale << "\n";
        }
        amrex::AllPrint() << top_report.str();
    }
}
