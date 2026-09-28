/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_EULERIAN_THERMAL_TRANSPORT_H_
#define WARPX_EULERIAN_THERMAL_TRANSPORT_H_
#include "EulerianThermalConduction.H"

// Scalar reconstruction and central electron channel extracted from
// resistive_mhd@592e2b3dfe9469eabab7bd8c6696a07ce1d68277,
// ThetaImplicitMHD_K.H:1836-1985, 3190-3193, 3257-3258, 3592-3595.
// Darwin supplies one conservative electron face velocity (m/s), so the
// MHD left/right velocities coincide. No ion-fluid wave speed is invented.
namespace warpx::thermal::mhd_transport {
using mhd::smooth_max;
using mhd::smooth_min;
struct TransportParameters {
    int fluid_reconstruction =
        0; // none, median, vanAlbada, unlimited, SMART, smooth SMART
    amrex::Real reconstruction_kappa = 0.01;
    amrex::Real density_floor = 0; // electron NUMBER density [m^-3]
    amrex::Real central_dissipation_entropy = 0;
};
constexpr int reconstruction_none = 0;
constexpr int reconstruction_median = 1;
constexpr int reconstruction_vanalbada = 2;
constexpr int reconstruction_unlimited = 3;
// SMART (Gaskell & Lau 1988), hard and kink-smoothed (see smart_slope
// and smart_slope_smooth above). Unlike the modes above these are
// UPWIND-BIASED: the slope depends on which of the two one-sided
// differences is the donor's upwind one, so reconstruct_face_pair builds
// the two face states with opposite donor roles.
constexpr int reconstruction_smart = 4;
constexpr int reconstruction_smart_smooth = 5;

// Relative guard of the limiter widths: guard = this times the RMS of
// the three stencil values entering one slope. Its only job is to keep
// the smoothing width (median) and the van Albada denominator strictly
// positive; it is deliberately far below the difference scale of any
// resolved structure, so it contributes a relative residual error of
// order guard/4 = 2.5e-7 and never drives the limiter.
constexpr amrex::Real reconstruction_relative_guard = 1.0e-6;

// Width of the reconstruction positivity floor, as a fraction of the
// floor. It is DECOUPLED from the floor -- unlike smooth_positive_floor,
// whose width IS the floor -- because a value sitting only a few floors
// up is exactly the halo pedestal band, and smooth_positive_floor
// inflates it by percent-level amounts there (12% at value = 2 floor).
// With width = f * floor the RELATIVE inflation well above the knee is
// (f/2)^2 (floor/value)^2, i.e. 2.5e-7 (floor/value)^2 at f = 1e-3:
// 2.5e-7 at the floor's own scale, ~1.6e-8 across a production halo
// pedestal band (value ~ 4 floors), and below round-off in the bulk --
// while the map stays C-infinity and the floor stays a hard lower
// bound. The width still exceeds the matrix-free probe scale
// (~1e-8 |u|) by five orders of magnitude at the floor, which is what
// the smoothing has to beat.
constexpr amrex::Real reconstruction_floor_width_fraction = 1.0e-3;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
smart_slope (const amrex::Real difference_up,
             const amrex::Real difference_down) noexcept {
    if (difference_up * difference_down <= 0.0) {
        return 0.0;
    }
    const amrex::Real sign = (difference_up > 0.0) ? 1.0 : -1.0;
    const amrex::Real up = sign * difference_up;
    const amrex::Real down = sign * difference_down;
    return sign *
           std::min(std::min(2.0 * down, 0.25 * up + 0.75 * down), 4.0 * up);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
smart_slope_smooth (const amrex::Real difference_up,
                    const amrex::Real difference_down, const amrex::Real guard,
                    const amrex::Real width) noexcept {
    const amrex::Real denominator =
        difference_up * difference_up + guard * guard;
    if (denominator <= 0.0) {
        return 0.0;
    }
    const amrex::Real ratio = difference_up * difference_down / denominator;
    const amrex::Real psi = smooth_max(
        smooth_min(smooth_min(2.0 * ratio, 0.25 + 0.75 * ratio, width), 4.0,
                   width),
        0.0, width);
    return difference_up * psi;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
smooth_median3 (const amrex::Real a, const amrex::Real b, const amrex::Real c,
                const amrex::Real width) noexcept {
    const amrex::Real pair_mean = 0.5 * (a + b);
    const amrex::Real difference = a - b;
    const amrex::Real half_span =
        0.5 * std::sqrt(difference * difference + width * width);
    const amrex::Real offset = pair_mean - c;
    const amrex::Real low = offset - half_span;
    const amrex::Real high = offset + half_span;
    const amrex::Real width_squared = width * width;
    return pair_mean + 0.5 * (std::sqrt(low * low + width_squared) -
                              std::sqrt(high * high + width_squared));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
reconstruction_positive_floor (const amrex::Real value,
                               const amrex::Real floor) noexcept {
    if (floor <= 0.0) {
        // No floor to enforce (the deck opted out); the limiter's own
        // neighbour bound is then the only guarantee.
        return value;
    }
    const amrex::Real width = reconstruction_floor_width_fraction * floor;
    const amrex::Real excess = value - floor;
    return floor + 0.5 * (excess + std::sqrt(excess * excess + width * width));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
limited_slope (const amrex::Real difference_up,
               const amrex::Real difference_down, const amrex::Real guard,
               const TransportParameters& parameters) noexcept {
    if (parameters.fluid_reconstruction == reconstruction_unlimited) {
        return 0.5 * (difference_up + difference_down);
    }
    if (parameters.fluid_reconstruction == reconstruction_smart) {
        // SMART, hard: difference_up must be the DONOR's upwind-side
        // difference (see reconstruct_face_pair).
        return smart_slope(difference_up, difference_down);
    }
    if (parameters.fluid_reconstruction == reconstruction_smart_smooth) {
        // SMART, kinks rounded in ratio space at dimensionless width
        // reconstruction_kappa.
        return smart_slope_smooth(difference_up, difference_down, guard,
                                  parameters.reconstruction_kappa);
    }
    const amrex::Real up_squared = difference_up * difference_up;
    const amrex::Real down_squared = difference_down * difference_down;
    const amrex::Real guard_squared = guard * guard;
    if (parameters.fluid_reconstruction == reconstruction_vanalbada) {
        // van Albada (1982) averaged slope: exactly the centered slope
        // when the two differences agree, exactly zero when they are
        // equal and opposite, and a rational function of the data with
        // a strictly positive denominator -- no |.|, no sqrt, no kink.
        const amrex::Real denominator =
            up_squared + down_squared + 2.0 * guard_squared;
        if (denominator <= 0.0) {
            return 0.0;
        }
        return (difference_down * (up_squared + guard_squared) +
                difference_up * (down_squared + guard_squared)) /
               denominator;
    }
    // median: minmod(d_up, d_down) = med(0, d_up, d_down), smoothed.
    const amrex::Real kappa = parameters.reconstruction_kappa;
    const amrex::Real width =
        std::sqrt(kappa * kappa * (up_squared + down_squared) + guard_squared);
    return smooth_median3(difference_up, difference_down, 0.0, width);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
reconstruct_face_pair (const amrex::Real far_left, const amrex::Real left,
                       const amrex::Real right, const amrex::Real far_right,
                       const TransportParameters& parameters,
                       amrex::Real& face_left,
                       amrex::Real& face_right) noexcept {
    const amrex::Real difference_left = left - far_left;
    const amrex::Real difference_middle = right - left;
    const amrex::Real difference_right = far_right - right;
    const amrex::Real guard_left =
        reconstruction_relative_guard *
        std::sqrt(far_left * far_left + left * left + right * right);
    const amrex::Real guard_right =
        reconstruction_relative_guard *
        std::sqrt(left * left + right * right + far_right * far_right);
    face_left = left + 0.5 * limited_slope(difference_left, difference_middle,
                                           guard_left, parameters);
    // The symmetric modes (median, vanalbada, unlimited) are even in
    // their two differences, so the argument order is immaterial and the
    // legacy call below is kept verbatim (bit-identical). The SMART
    // modes are upwind-biased: for the RIGHT cell's left-face state the
    // right cell is the donor of a leftward flow, its upwind-side
    // difference is (right - far_right) and its downwind-side difference
    // is (left - right); with the slope odd under the joint sign flip,
    //     right + (1/2) L(right - far_right, left - right)
    //   = right - (1/2) L(difference_right, difference_middle).
    // This is also what makes the reconstruction mirror-covariant
    // (reversing the stencil swaps the two face states), the property
    // the RZ mirror-parity test holds the limiter to.
    const bool upwind_biased =
        (parameters.fluid_reconstruction == reconstruction_smart ||
         parameters.fluid_reconstruction == reconstruction_smart_smooth);
    face_right =
        upwind_biased
            ? right - 0.5 * limited_slope(difference_right, difference_middle,
                                          guard_right, parameters)
            : right - 0.5 * limited_slope(difference_middle, difference_right,
                                          guard_right, parameters);
}

struct ElectronFaceFlux {
    amrex::Real energy;   // U u [W/m^2]
    amrex::Real velocity; // same face u in -P div(u) [m/s]
};
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ElectronFaceFlux
CentralElectronFlux (amrex::Real left, amrex::Real right, amrex::Real velocity,
                     amrex::Real penalty_speed,
                     amrex::Real entropy_coefficient) noexcept {
    return {0.5 * (left * velocity + right * velocity) -
                0.5 * entropy_coefficient * penalty_speed * (right - left),
            velocity};
}
} // namespace warpx::thermal::mhd_transport
#endif
