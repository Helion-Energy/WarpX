/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_PRESCRIBED_TIME_PROFILE_H_
#define WARPX_PRESCRIBED_TIME_PROFILE_H_

#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <cmath>
#include <limits>

namespace warpx::external {
// Borrowed immutable executors. Their owner must outlive all device work.
// f is dimensionless, derivative is 1/s, increment(t,h) is f(t+h)-f(t).
// The analytic derivative/increment semantics are an explicit caller contract;
// finite sample agreement cannot prove an arbitrary parser derivative.
struct PrescribedTimeProfile {
    amrex::ParserExecutor<1> value;
    amrex::ParserExecutor<1> derivative;
    amrex::ParserExecutor<2> increment;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real MidpointCurvature (amrex::Real t, amrex::Real h) const noexcept {
        return 0.5*increment(t,h)-increment(t,0.5*h);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool ConsistentIncrement (amrex::Real t, amrex::Real h) const noexcept {
        auto const a=value(t), b=value(t+h), d=increment(t,h);
        auto const reverse=increment(t+h,-h), slope=derivative(t);
        if (!(std::isfinite(a) && std::isfinite(b) && std::isfinite(d) &&
              std::isfinite(reverse) && std::isfinite(slope))) { return false; }
        auto const bound=64.*std::numeric_limits<amrex::Real>::epsilon()*
            (std::abs(a)+std::abs(b)+std::abs(d));
        return std::abs((b-a)-d)<=bound && std::abs(d+reverse)<=bound &&
            increment(t,0.)==0.;
    }
};
} // namespace warpx::external
#endif
