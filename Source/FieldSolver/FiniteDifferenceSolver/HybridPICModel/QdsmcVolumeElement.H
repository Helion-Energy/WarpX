/* Copyright 2024-2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_QDSMC_VOLUME_ELEMENT_H_
#define WARPX_QDSMC_VOLUME_ELEMENT_H_

#include "Utils/WarpXConst.H"

#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_REAL.H>
#include <AMReX_SPACE.H>

/** \file QdsmcVolumeElement.H
 *
 * THE control-volume element for every energy integral in the hybrid
 * electron-energy machinery: the per-stage energy budget
 * (qdsmc_energy_budget), the cumulative audit tallies, and the
 * HybridDissipation reduced diagnostic.
 *
 * It lives in its own header because those are TWO INDEPENDENT LEDGERS of
 * the same physical quantity, and their agreement is the acceptance test
 * for both. That test is only meaningful if they integrate against the
 * identical volume element, so there is exactly one definition of it and
 * both include this file. Do not inline a copy of this arithmetic at a call
 * site: a second copy is a second convention waiting to drift.
 *
 * In Cartesian geometry the element is the constant cell volume prod(dx).
 * In RZ it is the cylindrical 2 pi r dr dz, with two details that matter:
 *
 *   - the radius is taken at the grid point's own centring (nodal in r for
 *     the nodal T_e / rho / entropy fields, cell-centred for cell-centred
 *     tallies), and
 *   - the on-axis nodal ring owns [0, dr/2] and is weighted at its VOLUME
 *     CENTROID r = dr/8, not at r = 0, which would give it zero weight and
 *     silently drop the axis column from every energy integral.
 *
 * Historical note: the budget reductions omitted the 2 pi r factor entirely
 * and were therefore integrals of u dr dz reported in joules. On a
 * production-like RZ geometry that reads low by a large, profile-dependent
 * factor. Every RZ budget number produced before that fix is wrong by it.
 */

/** Control-volume element [m^3] of a single grid point.
 *
 * Construct on the host with MakeQdsmcVolumeElement and capture by value
 * into a device lambda; call with the RADIAL index (the first index) in RZ,
 * which is ignored in Cartesian geometry. */
struct QdsmcVolumeElement
{
    /** prod(dx) [m^3]; the whole answer in Cartesian geometry. */
    amrex::Real cartesian = amrex::Real(1.0);
    /** RZ only: domain lower radius, cell sizes, first radial index of the
     *  domain, and whether the field is nodal in r. */
    amrex::Real rmin = amrex::Real(0.0);
    amrex::Real dr = amrex::Real(0.0);
    amrex::Real dz = amrex::Real(0.0);
    int dom_lo_r = 0;
    bool nodal_r = true;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator() ([[maybe_unused]] int const i) const
    {
        using namespace amrex::literals;
#if defined(WARPX_DIM_RZ)
        amrex::Real r_vol = nodal_r ? (rmin + i*dr) : (rmin + (i + 0.5_rt)*dr);
        // The nodal axis ring spans [0, dr/2]; weight it at its volume
        // centroid so it neither vanishes nor counts as a full ring.
        if (nodal_r && i == dom_lo_r && rmin == 0.0_rt) { r_vol = 0.125_rt*dr; }
        return 2.0_rt*MathConst::pi*r_vol*dr*dz;
#else
        return cartesian;
#endif
    }
};

/** Build the volume element for \p geom at the centring of \p ixtype. */
inline QdsmcVolumeElement
MakeQdsmcVolumeElement (amrex::Geometry const & geom,
                        amrex::IndexType const ixtype)
{
    QdsmcVolumeElement v;
    auto const dx = geom.CellSizeArray();
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { v.cartesian *= dx[d]; }
    v.rmin = geom.ProbLo(0);
    v.dr = dx[0];
    v.dz = dx[AMREX_SPACEDIM - 1];
    v.dom_lo_r = geom.Domain().smallEnd(0);
    v.nodal_r = ixtype.nodeCentered(0);
    return v;
}

#endif // WARPX_QDSMC_VOLUME_ELEMENT_H_
