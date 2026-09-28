/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_IMPLICIT_GATHER_SAFETY_H_
#define WARPX_IMPLICIT_GATHER_SAFETY_H_

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_REAL.H>
#include <cmath>
#include <limits>

namespace warpx::particles {
enum class ImplicitGatherIssue {
    None,
    NonFinitePosition,
    InvalidLayout,
    StencilOutOfBounds,
    NonFiniteField,
    NonFiniteMomentum,
    Count
};
struct ImplicitGatherResult {
    ImplicitGatherIssue issue = ImplicitGatherIssue::None;
    int component = -1;
    int direction = -1;
    AMREX_GPU_HOST_DEVICE explicit
    operator bool () const {
        return issue == ImplicitGatherIssue::None;
    }
};
AMREX_GPU_HOST_DEVICE inline bool
FiniteParticleVector (amrex::ParticleReal x, amrex::ParticleReal y,
                      amrex::ParticleReal z) {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

// The coordinates and rounding below deliberately match doGatherShapeN<N,0>,
// including its truncation toward zero. Check the full loop support, even where
// a shape weight is exactly zero: the native loop still reads that grid value.
// No field is read, and no floating-to-integer conversion precedes validation.
AMREX_GPU_HOST_DEVICE inline ImplicitGatherResult
CheckMomentumGather (
    amrex::ParticleReal xp, amrex::ParticleReal yp, amrex::ParticleReal zp,
    amrex::GpuArray<amrex::Array4<amrex::Real const>, 6> const& fields,
    amrex::GpuArray<amrex::IndexType, 6> const& types, amrex::XDim3 const& dinv,
    amrex::XDim3 const& origin, amrex::Dim3 const& lo, int modes, int order) {
    if (!FiniteParticleVector(xp, yp, zp)) {
        return {ImplicitGatherIssue::NonFinitePosition};
    }
    if (order < 1 || order > 4) {
        return {ImplicitGatherIssue::InvalidLayout};
    }
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> q{};
    amrex::GpuArray<int, AMREX_SPACEDIM> lower{};
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
    amrex::Real const radius = std::sqrt(xp * xp + yp * yp);
    q[0] = (radius - origin.x) * dinv.x;
    lower[0] = lo.x;
#elif defined(WARPX_DIM_RSPHERE)
    amrex::Real const radius = std::sqrt(xp * xp + yp * yp + zp * zp);
    q[0] = (radius - origin.x) * dinv.x;
    lower[0] = lo.x;
#elif !defined(WARPX_DIM_1D_Z)
    q[0] = (xp - origin.x) * dinv.x;
    lower[0] = lo.x;
#endif
#if defined(WARPX_DIM_3D)
    q[1] = (yp - origin.y) * dinv.y;
    lower[1] = lo.y;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    q[WARPX_ZINDEX] = (zp - origin.z) * dinv.z;
    lower[WARPX_ZINDEX] = AMREX_SPACEDIM == 3 ? lo.z : lo.y;
#elif defined(WARPX_DIM_1D_Z)
    q[0] = (zp - origin.z) * dinv.z;
    lower[0] = lo.x;
#endif
    int required_components = 1;
#ifdef WARPX_DIM_RZ
    if (modes < 1 || modes > (std::numeric_limits<int>::max() - 1) / 2) {
        return {ImplicitGatherIssue::InvalidLayout};
    }
    required_components = 2 * modes - 1;
#else
    amrex::ignore_unused(modes);
#endif
    for (int f = 0; f < 6; ++f) {
        auto const& a = fields[f];
        if (!a || a.nComp() < required_components) {
            return {ImplicitGatherIssue::InvalidLayout, f};
        }
        auto const begin = amrex::lbound(a), end = amrex::ubound(a);
        amrex::GpuArray<int, 3> b{begin.x, begin.y, begin.z};
        amrex::GpuArray<int, 3> e{end.x, end.y, end.z};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            // Compute_shape_factor uses q+.5 for even shape orders.
            amrex::Real const centered =
                q[d] - amrex::Real(.5) * (1 - types[f][d]);
            amrex::Real const anchor =
                centered + (order % 2 == 0 ? amrex::Real(.5) : amrex::Real(0));
            if (!std::isfinite(anchor)) {
                return {ImplicitGatherIssue::NonFinitePosition, f, d};
            }
            // Widen the comparison to double so a single-precision rounded
            // INT_MAX cannot pass the check and make the cast undefined.
            if (double(anchor) < double(std::numeric_limits<int>::min()) + 8 ||
                double(anchor) > double(std::numeric_limits<int>::max()) - 8) {
                return {ImplicitGatherIssue::StencilOutOfBounds, f, d};
            }
            auto const index = static_cast<int>(anchor);
            auto const first =
                static_cast<long long>(lower[d]) + index - order / 2;
            auto const last = first + order;
            if (first < b[d] || last > e[d]) {
                return {ImplicitGatherIssue::StencilOutOfBounds, f, d};
            }
        }
        for (int d = AMREX_SPACEDIM; d < 3; ++d) {
            if (b[d] > 0 || e[d] < 0) {
                return {ImplicitGatherIssue::InvalidLayout, f, d};
            }
        }
    }
    return {};
}
} // namespace warpx::particles
#endif
