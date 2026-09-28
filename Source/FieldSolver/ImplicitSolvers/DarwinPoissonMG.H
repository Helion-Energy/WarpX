/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DARWIN_POISSON_MG_H_
#define WARPX_DARWIN_POISSON_MG_H_

#include <AMReX_Geometry.H>
#include <AMReX_MLLinOp.H>

#include <algorithm>

/** Configure only the multigrid hierarchy, leaving the finest operator unchanged.
 *  The direction with the largest cell width is retained; the other directions
 *  coarsen until their largest width would exceed it, or the requested cap is met.
 *  AMReX can stop earlier when the boxes cannot be coarsened further. */
inline int
ConfigureDarwinPoissonSemicoarsening (amrex::LPInfo& info, amrex::Geometry const& geom,
                                      int const max_levels, int retained_direction = -1)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        max_levels >= 0 && max_levels <= 30,
        "Darwin Poisson semicoarsening levels must be between 0 and 30");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        retained_direction >= -1 && retained_direction < AMREX_SPACEDIM,
        "Darwin Poisson semicoarsening direction is outside the grid dimensions");
    if (max_levels == 0)
    {
        return 0;
    }
#if AMREX_SPACEDIM == 1
    amrex::Abort("Darwin Poisson semicoarsening requires at least two grid dimensions");
    return 0;
#else
    auto const dx = geom.CellSizeArray();
    if (retained_direction < 0)
    {
        retained_direction = 0;
        for (int d = 1; d < AMREX_SPACEDIM; ++d)
        {
            if (dx[d] > dx[retained_direction])
            {
                retained_direction = d;
            }
        }
    }
    amrex::Real coarsened_width = 0.;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        if (d != retained_direction)
        {
            coarsened_width = std::max(coarsened_width, dx[d]);
        }
    }
    int levels = 0;
    while (levels < max_levels && 2. * coarsened_width <= dx[retained_direction])
    {
        coarsened_width *= 2.;
        ++levels;
    }
    if (levels > 0)
    {
        info.setSemicoarsening(true)
            .setSemicoarseningDirection(retained_direction)
            .setMaxSemicoarseningLevel(levels);
    }
    return levels;
#endif
}
#endif
