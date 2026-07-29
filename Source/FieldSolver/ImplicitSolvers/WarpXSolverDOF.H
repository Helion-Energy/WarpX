/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WarpXSolverDOF_H_
#define WarpXSolverDOF_H_

#include "Fields.H"
#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_INT.H>
#include <AMReX_REAL.H>

#include <memory>
#include <string>
#include <vector>

// forward declaration
class WarpX;

/**
 * \brief Description of an additional MultiFab block in a nonlinear state.
 *
 * A block can have any number of components and uses the centering of its
 * registered MultiFab. The scale is the characteristic physical magnitude
 * used to normalize serialized values and inner products.
 */
struct WarpXSolverMultiFabBlockSpec
{
    std::string name;
    amrex::Real scale = 1.0;

    bool operator== (const WarpXSolverMultiFabBlockSpec&) const = default;
};

struct WarpXSolverMultiFabDOFBlock
{
    std::string name;
    amrex::Vector<std::unique_ptr<amrex::iMultiFab>> dofs;
    amrex::Vector<std::unique_ptr<amrex::iMultiFab>> masks;
};

/**
 * \brief
 * This is contains the local and global DOF data that maps each grid location and
 * component to its position in a serialized vector.
 */
struct WarpXSolverDOF
{
    amrex::Vector<std::array<std::unique_ptr<amrex::iMultiFab>,3>> m_array;
    amrex::Vector<std::unique_ptr<amrex::iMultiFab>> m_scalar;
    std::vector<WarpXSolverMultiFabDOFBlock> m_multifab_blocks;

    warpx::fields::FieldType m_array_type = warpx::fields::FieldType::None;
    warpx::fields::FieldType m_scalar_type = warpx::fields::FieldType::None;

    amrex::Long m_nDoFs_l = 0; /*!< Local nDOF */
    amrex::Long m_nDoFs_g = 0; /*!< Global nDOF */

    void Define (
        WarpX*,
        int,
        const std::string&,
        const std::string&,
        const std::vector<WarpXSolverMultiFabBlockSpec>& = {});

    void fill_local_dof (amrex::iMultiFab& dof, amrex::iMultiFab const& mask);
    void fill_global_dof ();
};

#endif
