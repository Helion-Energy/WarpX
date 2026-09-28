/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_CURRENT_TOLERANCE_H_
#define WARPX_NATIVE_CURRENT_TOLERANCE_H_

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

// A fixed physical current floor in A/m^2, plus the existing local arithmetic
// allowance. Freeze the floor before the solve; never fit it to a trial defect.
struct NativeCurrentTolerance
{
    amrex::Real absolute = 0.;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Bound (amrex::Real roundoff) const
    {
        if (!std::isfinite(absolute) || absolute < 0. ||
            !std::isfinite(roundoff) || roundoff < 0.) {
            return std::numeric_limits<amrex::Real>::quiet_NaN();
        }
        // Preserve the original arithmetic-only policy exactly when disabled.
        if (absolute == 0.) { return roundoff; }
        amrex::Real const total = absolute + roundoff;
        return std::nextafter(total, std::numeric_limits<amrex::Real>::infinity());
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool Accepts (amrex::Real error, amrex::Real roundoff) const
    {
        amrex::Real const bound = Bound(roundoff);
        return std::isfinite(error) && std::isfinite(bound) && std::abs(error) <= bound;
    }
};
#endif
