/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_IMPLICIT_MHD_WALL_MASK_H_
#define WARPX_IMPLICIT_MHD_WALL_MASK_H_

#include "NonlinearSolvers/MHDResistiveStencil.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>

/**
 * \brief Stair-step PEC (perfectly conducting) shaped-wall mask for the
 * theta-implicit MHD solver (RZ only).
 *
 * The machine wall is a revolved poloidal polyline r_wall(z) (CSV file,
 * "z, r" rows, single-valued in z; near-vertical sections are expressed
 * with epsilon-offset duplicate z values, as in the run32 embedded-
 * boundary CSVs). Everything on or outside the polyline (r >= r_wall(z),
 * the run32 "recessed EB" convention: the conductor surface sits AT the
 * polyline) is treated as solid perfect conductor:
 *
 * - Every electric-field component located in the masked region is a
 *   tangential-E degree of freedom of the stair-step conductor surface or
 *   an interior-metal value (by the Yee staggering, an E component that
 *   lies exactly ON a stair face is always tangential to it), and is
 *   projected to the conductor condition at the END of every Ohm's-law
 *   assembly, i.e. inside every residual evaluation: total tangential
 *   E = 0. Under the split-field external-A scheme the evolved fields are
 *   the plasma response, so the projection writes E_plasma = -E_external
 *   (total E zero); without external fields it writes zero. The map is
 *   affine in the state with the same sparsity as a zeroed row, so
 *   matrix-free Jacobian probes see the wall exactly (JFNK-exact).
 *
 * - B in the masked region then evolves as dB/dt = -curl E with the
 *   masked E: cells whose complete edge set is masked keep their TOTAL
 *   field frozen at its initial value (dB_plasma/dt = +curl E_ext =
 *   -dB_ext/dt exactly, discrete-curl identical), which is the perfect-
 *   conductor eddy response -- flux through every closed contour inside
 *   the metal is conserved, and the interior of the machine sees the
 *   shielded (image-corrected) field.
 *
 * - The fluid is deliberately untouched: the wall region is expected to
 *   sit in the dust/vacuum fill (frozen momentum, E -> eta J), which is
 *   the fluid-side analog of run32's wall particle scraper.
 *
 * The preconditioner sees the identical mask: the resistive stencil
 * emission (MHDResistiveStencil.H) drops the same masked E rows through
 * the WallMaskView, so the matrix-free Chebyshev application and the
 * banded/direct assemblies stay consistent with the residual.
 *
 * Inputs (all under implicit_mhd.):
 * - wall_model = none (default; bit-identical to no wall) | pec
 * - wall_polyline_file = <csv path> ("z, r" rows; resolved against
 *   AMREX_INPUTS_FILE_PREFIX like the inputs file itself when relative)
 */
class ImplicitMHDWallMask
{
public:
    /** Parse the wall inputs and, when active, build the static
     * first-masked-radial-index tables per electric-field staggering over
     * the ghost-extended axial range. Aborts outside RZ, on periodic z,
     * or on a malformed polyline. */
    void Define (const amrex::Geometry& geom, const amrex::IntVect& ngrow);

    [[nodiscard]] bool IsActive () const { return m_active; }

    /** Device view consumed by the preconditioner stencil emission; the
     * pointers are pre-offset so the axial index of the residual grid is
     * used directly (valid over [-ng, nz+ng] nodal, [-ng, nz-1+ng] cell). */
    [[nodiscard]] warpx::mhd_pc::WallMaskView View () const;

    /** Apply the conductor projection to the assembled electric field:
     * at every masked location set E = -E_external (when the split
     * external fields are present) or E = 0. Covers the domain ghosts
     * (the external registers carry trusted domain-ghost values). */
    void ProjectElectricField (ablastr::fields::VectorField const& efield,
                               ablastr::fields::VectorField const* efield_external,
                               const amrex::Geometry& geom) const;

private:
    bool m_active = false;
    int m_ng = 0;   //!< axial ghost extension of the index tables
    int m_nz = 0;   //!< axial cells

    /** First masked radial index per axial index (sentinel: never masked),
     * one table per electric-field staggering; storage index j + m_ng.
     * MANAGED memory: the preconditioner's stencil emission runs on the
     * device (Chebyshev, banded) AND on the host (the direct solver's
     * sparse-row assembly), and both read these tables through the same
     * WallMaskView pointers. */
    amrex::Gpu::ManagedVector<int> m_first_masked_er; //!< (r-cell, z-node)
    amrex::Gpu::ManagedVector<int> m_first_masked_et; //!< (r-node, z-node)
    amrex::Gpu::ManagedVector<int> m_first_masked_ez; //!< (r-node, z-cell)
};

#endif // WARPX_IMPLICIT_MHD_WALL_MASK_H_
