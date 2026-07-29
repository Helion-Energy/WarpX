/* Copyright 2024 Justin Angus
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WarpXSolverVec_H_
#define WarpXSolverVec_H_

#include "Utils/TextMsg.H"
#include "FieldSolver/ImplicitSolvers/WarpXSolverDOF.H"
#include "Fields.H"

#include <ablastr/utils/SignalHandling.H>
#include <ablastr/warn_manager/WarnManager.H>
#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_IntVect.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <array>
#include <memory>
#include <ostream>
#include <vector>

// forward declaration
class WarpX;

/**
 * \brief
 *  This is a wrapper class around a Vector of pointers to MultiFabs that
 *  contains basic math operators and functionality needed to interact with nonlinear
 *  solvers in WarpX and linear solvers in AMReX, such as GMRES. The size of the
 *  Vector is the number of amr levels. Hardcoded for 1 right now.
 *
 *  A WarpXSolverVec can consist of an array size 3 of MultiFabs (for vector fields
 *  such as E, B, and A) or of a single MultiFab for scalar fields. Both the array
 *  size 3 and scalar fields must be of type warpx::fields::FieldType.
 *  Additionally, a WarpXSolverVec can in general contain both an array size 3 field and a
 *  scalar field. For example, the array size 3 field can be used for the vector potential A
 *  and the scalar field can be used for the scalar potential phi, which is the full state of
 *  unknowns for a Darwin electromagnetic model.
 */
class WarpXSolverVec
{
public:

    WarpXSolverVec() = default;

    WarpXSolverVec(const WarpXSolverVec&) = delete;

    ~WarpXSolverVec();

    using value_type = amrex::Real;
    using RT = value_type;
    using MultiFabBlockSpec = WarpXSolverMultiFabBlockSpec;

    [[nodiscard]] inline bool IsDefined () const { return m_is_defined; }

    void Define ( WarpX*  a_WarpX,
             const std::string&  a_vector_type_name,
             const std::string&  a_scalar_type_name = "none" );

    void Define (
        WarpX* a_WarpX,
        const std::string& a_vector_type_name,
        const std::string& a_scalar_type_name,
        const std::vector<MultiFabBlockSpec>& a_multifab_block_specs,
        RT a_vector_scale = 1.0,
        RT a_scalar_scale = 1.0);

    void Define (const WarpXSolverVec& a_solver_vec);

    [[nodiscard]] RT dotProduct( const WarpXSolverVec&  a_X ) const;

    void Copy ( warpx::fields::FieldType  a_array_type,
                warpx::fields::FieldType  a_scalar_type = warpx::fields::FieldType::None,
                bool allow_type_mismatch = false);

    inline
    void Copy ( const WarpXSolverVec&  a_solver_vec )
    {
        assertIsDefined( a_solver_vec );
        if (IsDefined()) { assertSameType( a_solver_vec ); }
        else { Define(a_solver_vec); }

        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            if (m_array_type != warpx::fields::FieldType::None) {
                for (int n = 0; n < 3; ++n) {
                    const amrex::MultiFab* this_field = a_solver_vec.getArrayVec()[lev][n];
                    amrex::MultiFab::Copy( *m_array_vec[lev][n], *this_field, 0, 0, m_ncomp,
                                           amrex::IntVect::TheZeroVector() );
                }
            }
            if (m_scalar_type != warpx::fields::FieldType::None) {
                const amrex::MultiFab* this_scalar = a_solver_vec.getScalarVec()[lev];
                amrex::MultiFab::Copy( *m_scalar_vec[lev], *this_scalar, 0, 0, m_ncomp,
                                       amrex::IntVect::TheZeroVector() );
            }
            for (std::size_t iblock = 0; iblock < m_multifab_blocks.size(); ++iblock) {
                auto& destination = *m_multifab_blocks[iblock].data[lev];
                amrex::MultiFab::Copy(
                    destination,
                    *a_solver_vec.m_multifab_blocks[iblock].data[lev],
                    0, 0, destination.nComp(), amrex::IntVect::TheZeroVector());
            }
        }
    }

    // Prohibit Copy assignment operator
    WarpXSolverVec& operator= ( const WarpXSolverVec&  a_solver_vec ) = delete;

    // Move assignment operator
    WarpXSolverVec(WarpXSolverVec&&) noexcept = default;
    WarpXSolverVec& operator= ( WarpXSolverVec&&  a_solver_vec ) noexcept
    {
        if (this != &a_solver_vec) {
            ClearData();

            m_array_vec = std::move(a_solver_vec.m_array_vec);
            m_scalar_vec = std::move(a_solver_vec.m_scalar_vec);
            m_array_type = a_solver_vec.m_array_type;
            m_scalar_type = a_solver_vec.m_scalar_type;
            m_vector_type_name = std::move(a_solver_vec.m_vector_type_name);
            m_scalar_type_name = std::move(a_solver_vec.m_scalar_type_name);
            m_array_scale = a_solver_vec.m_array_scale;
            m_scalar_scale = a_solver_vec.m_scalar_scale;
            m_multifab_blocks = std::move(a_solver_vec.m_multifab_blocks);
            m_multifab_block_specs = std::move(a_solver_vec.m_multifab_block_specs);
            m_num_amr_levels = a_solver_vec.m_num_amr_levels;
            m_WarpX = a_solver_vec.m_WarpX;
            m_dofs = std::move(a_solver_vec.m_dofs);
            m_is_defined = a_solver_vec.m_is_defined;

            a_solver_vec.m_array_vec.clear();
            a_solver_vec.m_scalar_vec.clear();
            a_solver_vec.m_multifab_blocks.clear();
            a_solver_vec.m_multifab_block_specs.clear();
            a_solver_vec.m_array_type = warpx::fields::FieldType::None;
            a_solver_vec.m_scalar_type = warpx::fields::FieldType::None;
            a_solver_vec.m_array_scale = 1.0;
            a_solver_vec.m_scalar_scale = 1.0;
            a_solver_vec.m_num_amr_levels = 0;
            a_solver_vec.m_WarpX = nullptr;
            a_solver_vec.m_is_defined = false;
        }
        return *this;
    }

    inline
    void operator+= ( const WarpXSolverVec&  a_solver_vec )
    {
        assertIsDefined( a_solver_vec );
        assertSameType( a_solver_vec );
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            if (m_array_type != warpx::fields::FieldType::None) {
                m_array_vec[lev][0]->plus(*(a_solver_vec.getArrayVec()[lev][0]), 0, 1, 0);
                m_array_vec[lev][1]->plus(*(a_solver_vec.getArrayVec()[lev][1]), 0, 1, 0);
                m_array_vec[lev][2]->plus(*(a_solver_vec.getArrayVec()[lev][2]), 0, 1, 0);
            }
            if (m_scalar_type != warpx::fields::FieldType::None) {
                m_scalar_vec[lev]->plus(*(a_solver_vec.getScalarVec()[lev]), 0, 1, 0);
            }
            for (std::size_t iblock = 0; iblock < m_multifab_blocks.size(); ++iblock) {
                auto& block = *m_multifab_blocks[iblock].data[lev];
                block.plus(*a_solver_vec.m_multifab_blocks[iblock].data[lev],
                           0, block.nComp(), 0);
            }
        }
    }

    inline
    void operator-= (const WarpXSolverVec& a_solver_vec)
    {
        assertIsDefined( a_solver_vec );
        assertSameType( a_solver_vec );
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            if (m_array_type != warpx::fields::FieldType::None) {
                m_array_vec[lev][0]->minus(*(a_solver_vec.getArrayVec()[lev][0]), 0, 1, 0);
                m_array_vec[lev][1]->minus(*(a_solver_vec.getArrayVec()[lev][1]), 0, 1, 0);
                m_array_vec[lev][2]->minus(*(a_solver_vec.getArrayVec()[lev][2]), 0, 1, 0);
            }
            if (m_scalar_type != warpx::fields::FieldType::None) {
                m_scalar_vec[lev]->minus(*(a_solver_vec.getScalarVec()[lev]), 0, 1, 0);
            }
            for (std::size_t iblock = 0; iblock < m_multifab_blocks.size(); ++iblock) {
                auto& block = *m_multifab_blocks[iblock].data[lev];
                block.minus(*a_solver_vec.m_multifab_blocks[iblock].data[lev],
                            0, block.nComp(), 0);
            }
        }
    }

    /**
     * \brief Y = a*X + b*Y
     */
    inline
    void linComb (const RT a, const WarpXSolverVec& X, const RT b, const WarpXSolverVec& Y)
    {
        assertIsDefined( X );
        assertIsDefined( Y );
        assertSameType( X );
        assertSameType( Y );
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            if (m_array_type != warpx::fields::FieldType::None) {
                for (int n = 0; n < 3; n++) {
                    amrex::MultiFab::LinComb(*m_array_vec[lev][n], a, *X.getArrayVec()[lev][n], 0,
                                                                   b, *Y.getArrayVec()[lev][n], 0,
                                                                   0, 1, 0);
                }
            }
            if (m_scalar_type != warpx::fields::FieldType::None) {
                amrex::MultiFab::LinComb(*m_scalar_vec[lev], a, *X.getScalarVec()[lev], 0,
                                                             b, *Y.getScalarVec()[lev], 0,
                                                             0, 1, 0);
            }
            for (std::size_t iblock = 0; iblock < m_multifab_blocks.size(); ++iblock) {
                auto& block = *m_multifab_blocks[iblock].data[lev];
                const int ncomp = block.nComp();
                amrex::MultiFab::LinComb(
                    block,
                    a, *X.m_multifab_blocks[iblock].data[lev], 0,
                    b, *Y.m_multifab_blocks[iblock].data[lev], 0,
                    0, ncomp, 0);
            }
        }
    }

    /**
     * \brief Increment Y by a*X (Y += a*X)
     */
    void increment (const WarpXSolverVec& X, const RT a)
    {
        assertIsDefined( X );
        assertSameType( X );
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            if (m_array_type != warpx::fields::FieldType::None) {
                for (int n = 0; n < 3; n++) {
                    amrex::MultiFab::Saxpy( *m_array_vec[lev][n], a, *X.getArrayVec()[lev][n],
                                            0, 0, 1, amrex::IntVect::TheZeroVector() );
                }
            }
            if (m_scalar_type != warpx::fields::FieldType::None) {
                amrex::MultiFab::Saxpy( *m_scalar_vec[lev], a, *X.getScalarVec()[lev],
                                        0, 0, 1, amrex::IntVect::TheZeroVector() );
            }
            for (std::size_t iblock = 0; iblock < m_multifab_blocks.size(); ++iblock) {
                auto& block = *m_multifab_blocks[iblock].data[lev];
                amrex::MultiFab::Saxpy(
                    block, a, *X.m_multifab_blocks[iblock].data[lev],
                    0, 0, block.nComp(), amrex::IntVect::TheZeroVector());
            }
        }
    }

    /**
     * \brief Scale Y by a (Y *= a)
     */
    inline
    void scale (RT a_a)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            IsDefined(),
            "WarpXSolverVec::scale() called on undefined WarpXSolverVec");
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            if (m_array_type != warpx::fields::FieldType::None) {
                m_array_vec[lev][0]->mult(a_a, 0, 1);
                m_array_vec[lev][1]->mult(a_a, 0, 1);
                m_array_vec[lev][2]->mult(a_a, 0, 1);
            }
            if (m_scalar_type != warpx::fields::FieldType::None) {
                m_scalar_vec[lev]->mult(a_a, 0, 1);
            }
            for (auto& block : m_multifab_blocks) {
                block.data[lev]->mult(a_a, 0, block.data[lev]->nComp());
            }
        }
    }

    inline
    void zero () { setVal(0.0); }

    inline
    void setVal ( const RT  a_val )
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            IsDefined(),
            "WarpXSolverVec::setVal() called on undefined WarpXSolverVec");
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            if (m_array_type != warpx::fields::FieldType::None) {
                m_array_vec[lev][0]->setVal(a_val);
                m_array_vec[lev][1]->setVal(a_val);
                m_array_vec[lev][2]->setVal(a_val);
            }
            if (m_scalar_type != warpx::fields::FieldType::None) {
                m_scalar_vec[lev]->setVal(a_val);
            }
            for (auto& block : m_multifab_blocks) {
                block.data[lev]->setVal(a_val);
            }
        }
    }

    inline
    void assertIsDefined( const WarpXSolverVec&  a_solver_vec ) const
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            a_solver_vec.IsDefined(),
            "WarpXSolverVec::function(X) called with undefined WarpXSolverVec X");
    }

    inline
    void assertSameType( const WarpXSolverVec&  a_solver_vec ) const
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            a_solver_vec.getArrayVecType()==m_array_type &&
            a_solver_vec.getScalarVecType()==m_scalar_type &&
            a_solver_vec.m_array_scale == m_array_scale &&
            a_solver_vec.m_scalar_scale == m_scalar_scale &&
            a_solver_vec.m_multifab_block_specs == m_multifab_block_specs,
            "WarpXSolverVec::function(X) called with WarpXSolverVec X of different type");
    }

    [[nodiscard]] inline RT norm2 () const
    {
        auto const norm = dotProduct(*this);
        return std::sqrt(norm);
    }

    [[nodiscard]] const ablastr::fields::MultiLevelVectorField& getArrayVec() const {return m_array_vec;}
    ablastr::fields::MultiLevelVectorField& getArrayVec() {return m_array_vec;}

    [[nodiscard]] const ablastr::fields::MultiLevelScalarField& getScalarVec() const {return m_scalar_vec;}
    ablastr::fields::MultiLevelScalarField& getScalarVec() {return m_scalar_vec;}

    // solver vector types are type warpx::fields::FieldType
    [[nodiscard]] warpx::fields::FieldType getArrayVecType () const { return m_array_type; }
    [[nodiscard]] warpx::fields::FieldType getScalarVecType () const { return m_scalar_type; }

    // solver vector type names
    [[nodiscard]] std::string getVectorType () const { return m_vector_type_name; }
    [[nodiscard]] std::string getScalarType () const { return m_scalar_type_name; }

    [[nodiscard]] bool hasMultiFabBlock (const std::string& a_name) const;
    [[nodiscard]] amrex::MultiFab& getMultiFabBlock (const std::string& a_name, int a_lev);
    [[nodiscard]] const amrex::MultiFab& getMultiFabBlock (
        const std::string& a_name, int a_lev) const;
    [[nodiscard]] const std::vector<MultiFabBlockSpec>& getMultiFabBlockSpecs () const
    {
        return m_multifab_block_specs;
    }

    void CopyMultiFabBlocksFromFields ();
    void CopyMultiFabBlocksToFields () const;

    // vector sizes
    [[nodiscard]] amrex::Long nDOF_local  () const
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (m_dofs != nullptr),
            "WarpXSolverVec::nDOF_local() DOF object is a nullptr");
        return m_dofs->m_nDoFs_l;
    }
    [[nodiscard]] amrex::Long nDOF_global () const
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (m_dofs != nullptr),
            "WarpXSolverVec::nDOF_global() DOF object is a nullptr");
        return m_dofs->m_nDoFs_g;
    }

    // copy to/from Real-type arrays
    void copyFrom (const amrex::Real*);
    void copyTo   (amrex::Real*) const;

    // return WarpX pointer
    [[nodiscard]] auto getWarpX () const { return m_WarpX; }

    // return the number of AMR levels
    [[nodiscard]] auto numAMRLevels () const { return m_num_amr_levels; }

    // return DOFs object pointer
    [[nodiscard]] inline const auto& getDOFsObject () const { return m_dofs; }

private:

    struct MultiFabBlock
    {
        MultiFabBlockSpec spec;
        amrex::Vector<std::unique_ptr<amrex::MultiFab>> data;
    };

    bool m_is_defined = false;

    ablastr::fields::MultiLevelVectorField m_array_vec;
    ablastr::fields::MultiLevelScalarField m_scalar_vec;
    std::vector<MultiFabBlock> m_multifab_blocks;
    std::vector<MultiFabBlockSpec> m_multifab_block_specs;

    warpx::fields::FieldType m_array_type = warpx::fields::FieldType::None;
    warpx::fields::FieldType m_scalar_type = warpx::fields::FieldType::None;

    std::string m_vector_type_name = "none";
    std::string m_scalar_type_name = "none";
    RT m_array_scale = 1.0;
    RT m_scalar_scale = 1.0;

    static constexpr int m_ncomp = 1;
    int m_num_amr_levels = 1;

    WarpX* m_WarpX = nullptr;

    /**
     * Serialized layout for this nonlinear state. Solver work vectors created
     * with Define(other) share the layout, while independently defined states
     * may use different field blocks and centerings.
     */
    std::shared_ptr<WarpXSolverDOF> m_dofs;

    void DefineData (WarpX* a_WarpX,
                     const std::string& a_vector_type_name,
                     const std::string& a_scalar_type_name,
                     const std::vector<MultiFabBlockSpec>& a_multifab_block_specs,
                     RT a_vector_scale,
                     RT a_scalar_scale);

    void ClearData () noexcept;
};

#endif
