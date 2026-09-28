/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_HYBRID_END_REGION_H_
#define WARPX_HYBRID_END_REGION_H_

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>
#include <cmath>

/** Geometric buffer at the two physical z ends. Width zero disables a side.
 *  The same profile is sampled on each E component and at thermal nodes.
 */
struct HybridEndRegion
{
    amrex::GpuArray<amrex::Real, 2> width{0., 0.}, rolloff{0., 0.};
    amrex::Real resistivity = 0.;
    bool holmstrom = false;
    amrex::Real zlo = 0., zhi = 0., dz = 1.;
    int index_lo = 0;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE amrex::Real
    Weight (int i, int j, int k, amrex::GpuArray<int, 3> const &nodal) const noexcept
    {
        constexpr int axial = AMREX_SPACEDIM - 1;
        int const index[3] = {i, j, k};
        amrex::Real const z = zlo + (index[axial] - index_lo + .5 * (1 - nodal[axial])) * dz;
        amrex::Real const distance[2] = {z - zlo, zhi - z};
        amrex::Real result = 0.;
        for (int side = 0; side < 2; ++side)
        {
            if (width[side] > 0.)
            {
                amrex::Real const value =
                    rolloff[side] > 0.
                        ? .5 * (1. + std::tanh ((width[side] - distance[side]) / rolloff[side]))
                        : (distance[side] <= width[side] ? 1. : 0.);
                result = amrex::max (result, value);
            }
        }
        return result;
    }
};
#endif
