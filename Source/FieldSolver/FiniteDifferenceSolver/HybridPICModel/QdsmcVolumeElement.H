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

/** Physical dual-cell volume for the hybrid electron-energy ledgers.
 * Nodal boundary cells are clipped at non-periodic domain faces. In RZ
 * the axis node owns the disk [0, dr/2], and the outer radial node owns
 * only its interior half ring. Periodic nodal images each use a full cell;
 * callers deduplicate them with an owner mask or sum_unique.
 */
struct QdsmcVolumeElement
{
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lo{}, hi{}, dx{};
    amrex::GpuArray<int, AMREX_SPACEDIM> domain_lo{}, nodal{}, periodic{};

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    lower (int const d, int const index) const
    {
        amrex::Real const x = lo[d] + (amrex::Real(index - domain_lo[d]) +
                                       (nodal[d] ? amrex::Real(0.0) : amrex::Real(0.5))) *
                                          dx[d];
        return periodic[d] ? x - amrex::Real(0.5) * dx[d]
                           : amrex::max(x - amrex::Real(0.5) * dx[d], lo[d]);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    upper (int const d, int const index) const
    {
        amrex::Real const x = lo[d] + (amrex::Real(index - domain_lo[d]) +
                                       (nodal[d] ? amrex::Real(0.0) : amrex::Real(0.5))) *
                                          dx[d];
        return periodic[d] ? x + amrex::Real(0.5) * dx[d]
                           : amrex::min(x + amrex::Real(0.5) * dx[d], hi[d]);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    width (int const d, int const index) const
    {
        return upper(d, index) - lower(d, index);
    }

    // Integral r dr, with the 2*pi angular factor left out for radial fluxes.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    radial_measure (int const i) const
    {
        amrex::Real const l = lower(0, i), u = upper(0, i);
        return amrex::Real(0.5) * (u * u - l * l);
    }

    // Area of a physical domain face divided by the adjacent node volume.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    wall_area_over_volume (int const d, int const side, int const index) const
    {
#if defined(WARPX_DIM_RZ)
        if (d == 0)
        {
            return (side == 0 ? lo[0] : hi[0]) / radial_measure(index);
        }
#else
        amrex::ignore_unused(side);
#endif
        return 1.0 / width(d, index);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    operator()(int const i, int const j, int const k = 0) const
    {
        int const index[3] = {i, j, k};
        amrex::Real volume = 1.0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
#if defined(WARPX_DIM_RZ)
            if (d == 0)
            {
                volume *= 2.0 * MathConst::pi * radial_measure(i);
            }
            else
#endif
            {
                volume *= width(d, index[d]);
            }
        }
        return volume;
    }
};

inline QdsmcVolumeElement
MakeQdsmcVolumeElement (amrex::Geometry const& geom, amrex::IndexType const ixtype)
{
    QdsmcVolumeElement v;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        v.lo[d] = geom.ProbLo(d);
        v.hi[d] = geom.ProbHi(d);
        v.dx[d] = geom.CellSize(d);
        v.domain_lo[d] = geom.Domain().smallEnd(d);
        v.nodal[d] = ixtype.nodeCentered(d);
        v.periodic[d] = geom.isPeriodic(d);
    }
    return v;
}

#endif // WARPX_QDSMC_VOLUME_ELEMENT_H_
