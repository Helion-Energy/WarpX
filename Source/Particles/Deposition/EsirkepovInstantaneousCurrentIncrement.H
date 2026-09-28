/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_ESIRKEPOV_INSTANTANEOUS_CURRENT_INCREMENT_H_
#define WARPX_ESIRKEPOV_INSTANTANEOUS_CURRENT_INCREMENT_H_

#include "EsirkepovCurrentIncrement.H"
#include "EsirkepovCurrentRate.H"

#include <cmath>
#include <limits>

namespace warpx::particles {
/** Correlated instantaneous states, with no timestep or history convention.
 * Position is in tile-local grid coordinates. Active velocity components are
 * grid-coordinate rates [1/s]: 0,2 for Dim2 and 0,1,2 for Dim3. For Dim2 only,
 * azimuthal_velocity is the physical out-of-plane velocity [m/s]. Unused Dim2
 * component1 and unused Dim3 azimuthal_velocity do not enter the map.
 * The caller must supply correlated increments, not differences of previously
 * rounded absolute positions or velocities.
 */
struct EsirkepovInstantaneousPointIncrement {
    amrex::GpuArray<EsirkepovIncrement, 3> position{}, velocity{};
    EsirkepovIncrement azimuthal_velocity{};
};

namespace instantaneous_increment_detail {
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool FinitePair (EsirkepovIncrement const& p) noexcept
{
    return std::isfinite(p.value) && std::isfinite(p.delta) &&
           std::isfinite(p.value + p.delta);
}

template<int N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool CommonPointSupport (EsirkepovIncrement const& x, int& center, int& first) noexcept
{
    // Bound conversion and subsequent support-index arithmetic before casting.
    double const upper = double(std::numeric_limits<int>::max() - 8);
    if (!FinitePair(x) || x.value < 0. || x.value > upper ||
        x.value + x.delta < 0. || x.value + x.delta > upper) {
        return false;
    }
    center = static_cast<int>(x.value + (N % 2 == 0 ? .5 : 0.));
    first = center - N / 2;
    double const lower = double(center) - (N % 2 == 0 ? .5 : 0.);
    // Endpoint addition is used only for the support decision. It never
    // supplies the increment in the polynomial evaluation below.
    double const values[2] = {x.value, x.value + x.delta};
    for (double q : values) {
        double const guard = 64. * std::numeric_limits<double>::epsilon() *
                             (1. + std::abs(q));
        if (!(q > lower + guard && q < lower + 1. - guard)) { return false; }
    }
    return true;
}
} // namespace instantaneous_increment_detail

/** Unweighted SAME-Esirkepov h->0 current and its finite change at a mesh index.
 * Value uses the exact pinned native EsirkepovCurrentRate base evaluation.
 * Delta evaluates the equal polynomial through finite-increment algebra; it
 * includes every finite product cross term, not merely a directional tangent.
 * It therefore resolves changes hidden by subtracting rounded native values.
 * The small represented-arithmetic remainder between value+delta and a second
 * rounded native evaluation is not asserted to vanish.
 *
 * Caller owns q*w*inverse transverse areas (normal), q*w*inverse volume (Dim2
 * azimuthal), native RZ metric scaling, summation and boundary images. This
 * helper neither deposits nor changes particles/fields. Common support is
 * required; unresolved knots/support changes, nonfinite input or nonfinite
 * output return false with all outputs zero. No fallback is implicit.
 */
template<int N, int Dim> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool EsirkepovInstantaneousCurrentPair (
    EsirkepovInstantaneousPointIncrement const& point,
    amrex::GpuArray<int, 3> const& node,
    amrex::GpuArray<EsirkepovIncrement, 3>& current) noexcept
{
    static_assert(N >= 1 && N <= 4);
    static_assert(Dim == 2 || Dim == 3);
    current = {};
    amrex::GpuArray<double, 3> x{}, v{}, acceleration{}, base{}, unused_rate{};
    amrex::GpuArray<int, 3> safe_node{};
    EsirkepovIncrement nodal[3]{{1., 0.}, {1., 0.}, {1., 0.}};
    EsirkepovIncrement face[3]{};
    bool unchanged = true;
    for (int d = 0; d < 3; ++d) {
        if constexpr (Dim == 2) { if (d == 1) { continue; } }
        int center = 0, first = 0;
        if (!instantaneous_increment_detail::FinitePair(point.velocity[d]) ||
            !instantaneous_increment_detail::CommonPointSupport<N>(
                point.position[d], center, first)) {
            return false;
        }
        x[d] = point.position[d].value;
        v[d] = point.velocity[d].value;
        unchanged = unchanged && point.position[d].delta == 0. &&
                    point.velocity[d].delta == 0.;
        // Clamp only an out-of-support query index, not a physical position.
        // This retains identically zero shape values and avoids int overflow
        // in the native prefix evaluator for arbitrarily distant mesh indices.
        safe_node[d] = node[d] < first ? first - 1 :
                       node[d] > first + N ? first + N + 1 : node[d];
        int const slot = safe_node[d] - first;
        auto const local = point.position[d] - double(center);
        nodal[d] = esirkepov_increment_detail::Shape<N>(slot, local);
        if constexpr (N == 1) {
            face[d] = slot == 0 ? EsirkepovIncrement{1., 0.} : EsirkepovIncrement{};
        } else {
            // -sum_{k<=slot} S'_N,k is exactly the order-(N-1) face shape.
            auto const face_local = local + (N % 2 == 0 ? .5 : -.5);
            face[d] = esirkepov_increment_detail::Shape<N - 1>(slot, face_local);
        }
    }
    double transverse = 0.;
    if constexpr (Dim == 2) {
        if (!instantaneous_increment_detail::FinitePair(point.azimuthal_velocity)) {
            return false;
        }
        transverse = point.azimuthal_velocity.value;
        unchanged = unchanged && point.azimuthal_velocity.delta == 0.;
    }
    if (!EsirkepovCurrentRate<N, Dim>(x, v, acceleration, safe_node,
                                     transverse, 0., base, unused_rate)) {
        return false;
    }
    amrex::GpuArray<EsirkepovIncrement, 3> candidate{};
    for (int d = 0; d < 3; ++d) {
        if constexpr (Dim == 2) {
            if (d == 1) {
                candidate[d] = point.azimuthal_velocity * nodal[0] * nodal[2];
                continue;
            }
        }
        int const e = (d + 1) % 3, f = (d + 2) % 3;
        candidate[d] = face[d] * point.velocity[d] * nodal[e] * nodal[f];
    }
    for (int d = 0; d < 3; ++d) {
        candidate[d].value = base[d];
        if (unchanged) { candidate[d].delta = 0.; }
        if (!std::isfinite(candidate[d].value) || !std::isfinite(candidate[d].delta) ||
            !std::isfinite(candidate[d].value + candidate[d].delta)) {
            return false;
        }
    }
    current = candidate;
    return true;
}
} // namespace warpx::particles
#endif
