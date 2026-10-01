/* Copyright 2026 WarpX contributors
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_BOUNDARY_PROBABILITY_H_
#define WARPX_BOUNDARY_PROBABILITY_H_

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

namespace warpx::particles
{
    /** One decision per wall encounter; validated endpoints consume no RNG. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool absorb_at_fractional_boundary (amrex::Real fraction,
                                       amrex::RandomEngine const& engine) noexcept
    {
        return fraction == amrex::Real(1.0) ||
            (fraction > amrex::Real(0.0) && amrex::Random(engine) < fraction);
    }
}

#endif
