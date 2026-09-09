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

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuContainers.H>
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
 * \brief One valid box of one MultiFab of a nonlinear state, as a segment of
 * the fused one-dimensional index space (see WarpXSolverFusedLayout).
 */
struct WarpXSolverFusedSegment
{
    amrex::Box box;                    //!< valid box of the FAB
    amrex::Long offset = 0;            //!< first index of the segment in the fused space
    int ncomp = 1;                     //!< components of the MultiFab
    amrex::Real scale = 1.0;           //!< block scale (deserialization)
    amrex::Real inv_scale = 1.0;       //!< 1 / block scale (serialization)
    amrex::Real inv_scale2 = 1.0;      //!< 1 / block scale^2 (inner products)
    amrex::Array4<int const> mask;     //!< ownership mask (dot mask / owner mask)
    amrex::Array4<int const> dof;      //!< {local, global} DOF ids per component
};

/**
 * \brief Fused-kernel layout of a nonlinear state (level 0 only).
 *
 * Every valid box of every MultiFab of the state (the three components of
 * the array-type field, the scalar field, the extra MultiFab blocks, in
 * that order, FABs in MFIter order) is one segment of a single 1D index
 * space of ncomp * npts entries. With it a vector operation over the whole
 * state is ONE kernel launch and an inner product one launch plus one
 * reduction, instead of one launch (and, for inner products, one host
 * synchronization) per MultiFab. Built by WarpXSolverDOF::Define; the
 * per-vector data pointers (Array4 tables in the same segment order) are
 * cached by each WarpXSolverVec.
 */
struct WarpXSolverFusedLayout
{
    amrex::Long total = 0;                                  //!< entries in the fused space
    int nseg = 0;
    std::vector<WarpXSolverFusedSegment> host_segments;     //!< host copy
    amrex::Gpu::DeviceVector<WarpXSolverFusedSegment> segments; //!< device copy
    amrex::Gpu::DeviceVector<amrex::Long> offsets;          //!< nseg + 1 segment starts (search table)
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

    WarpXSolverFusedLayout m_fused; /*!< fused-kernel layout (level 0) */

    void Define (
        WarpX*,
        int,
        const std::string&,
        const std::string&,
        const std::vector<WarpXSolverMultiFabBlockSpec>& = {},
        amrex::Real a_vector_scale = 1.0,
        amrex::Real a_scalar_scale = 1.0);

    void define_fused_layout (
        WarpX*,
        const std::vector<WarpXSolverMultiFabBlockSpec>&,
        amrex::Real a_vector_scale,
        amrex::Real a_scalar_scale);

    void fill_local_dof (amrex::iMultiFab& dof, amrex::iMultiFab const& mask);
    void fill_global_dof ();
};

#endif
