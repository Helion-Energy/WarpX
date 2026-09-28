/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_EULERIAN_THERMAL_WALL_H_
#define WARPX_EULERIAN_THERMAL_WALL_H_
#include "QdsmcConductionWall.H"
#include <limits>

namespace warpx::thermal {

struct LegFaceParameters {
    amrex::Real bath_ev;
    amrex::Real length;
    amrex::Real half_cell;
    amrex::Real flux_limit;
    bool sonic;
    amrex::Real gamma;
    amrex::Real ion_mass;
    amrex::Real temperature_floor_ev;
};
struct LegFaceResult {
    amrex::Real flux = 0; // outward W/m^2, never a finite-dt temperature reset
    amrex::Real interface_ev = 0;
    amrex::Real half_conductance = 0; // W/(m^2 K)
    amrex::Real leg_conductance = 0;
    amrex::Real uncapped_flux = 0;
    amrex::Real cap = 0;
};

// Maintained physical law: parallel kappa(n,Te[eV],t) in both the unresolved
// leg and its access half-cell, as in the legacy leg. The interface has zero
// heat capacity; its nonlinear flux balance is solved by a bounded bisection.
// n is electron number density, shared across this unresolved interface. No
// species mass is inferred; only a sonic cap uses the caller's effective mi.
// This is a new cell-face discretization of that law, NOT LegWallTemperature.
template <typename Kappa>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE LegFaceResult
EvaluateLegFace (amrex::Real temperature_ev, amrex::Real density,
                 amrex::Real time, LegFaceParameters const& p,
                 Kappa const& kappa) noexcept {
    using amrex::Real;
    LegFaceResult out;
    out.interface_ev = p.bath_ev;
    if (!(std::isfinite(temperature_ev) && temperature_ev > 0 &&
          std::isfinite(density) && density > 0)) {
        out.flux = std::numeric_limits<Real>::quiet_NaN();
        return out;
    }
    if (temperature_ev <= amrex::max(p.bath_ev, p.temperature_floor_ev)) {
        return out; // exactly drain-only, including at the bath/floor
    }
    Real const kc = kappa(density, temperature_ev, time);
    if (!(std::isfinite(kc) && kc >= 0)) {
        out.flux = std::numeric_limits<Real>::quiet_NaN();
        return out;
    }
    Real lo = p.bath_ev, hi = temperature_ev;
    for (int iteration = 0; iteration < 56; ++iteration) {
        Real const tf = lo + 0.5 * (hi - lo);
        Real const kf = kappa(density, tf, time);
        if (!(std::isfinite(kf) && kf >= 0)) {
            out.flux = std::numeric_limits<Real>::quiet_NaN();
            return out;
        }
        Real const gh = 0.5 * (kc + kf) / p.half_cell;
        Real const gl = kf / p.length;
        if (gh * (temperature_ev - tf) > gl * (tf - p.bath_ev)) {
            lo = tf;
        } else {
            hi = tf;
        }
    }
    out.interface_ev = lo + 0.5 * (hi - lo);
    Real const kf = kappa(density, out.interface_ev, time);
    out.half_conductance = 0.5 * (kc + kf) / p.half_cell;
    out.leg_conductance = kf / p.length;
    Real const sum = out.half_conductance + out.leg_conductance;
    // Series form avoids cancellation in Tc-Tinterface when L >> dx.
    out.uncapped_flux =
        sum > 0
            ? (out.half_conductance / sum) * out.leg_conductance *
                  (temperature_ev - p.bath_ev) * PhysConst::q_e / PhysConst::kb
            : 0;
    out.flux = out.uncapped_flux;
    if (p.flux_limit > 0) {
        Real const kelvin = temperature_ev * PhysConst::q_e / PhysConst::kb;
        out.cap = p.flux_limit * density * PhysConst::q_e * temperature_ev *
                  WallCapSpeed(kelvin, p.sonic, p.gamma, p.ion_mass);
        // The maintained wall law is a HARD cap, unlike the bulk smooth cap.
        out.flux = amrex::min(out.flux, out.cap);
    }
    return out;
}

// Generic parser laws have no analytic derivative API. A scale-relative
// centred difference differentiates the SAME physical interface and cap law.
// Its branch derivative at a kink is the symmetric generalized derivative.
// This is evaluated only at wall faces; the interior FD4 PC stays unchanged.
template <typename Kappa>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
LegFaceTemperatureDerivative (amrex::Real temperature_ev, amrex::Real density,
                              amrex::Real time, LegFaceParameters const& p,
                              Kappa const& kappa) noexcept {
    if (temperature_ev <= amrex::max(p.bath_ev, p.temperature_floor_ev)) {
        return 0;
    }
    amrex::Real const delta =
        6.e-6 * amrex::max(temperature_ev, p.temperature_floor_ev);
    return (EvaluateLegFace(temperature_ev + delta, density, time, p, kappa)
                .flux -
            EvaluateLegFace(temperature_ev - delta, density, time, p, kappa)
                .flux) /
           (2 * delta); // W/(m^2 eV)
}
// Capped reservoir at the actual cell face. Heating is allowed unless the
// caller explicitly requests a drain-only reservoir. The cap limits cooling.
template <typename KappaParallel, typename KappaPerpendicular>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
EvaluateReservoirFace (amrex::Real te, amrex::Real density, amrex::Real time,
                       LegFaceParameters const& p, amrex::Real normal_fraction,
                       bool drain_only, KappaParallel const& parallel,
                       KappaPerpendicular const& perpendicular) noexcept {
    if (!(std::isfinite(te) && te > 0 && std::isfinite(density) &&
          density > 0)) {
        return std::numeric_limits<amrex::Real>::quiet_NaN();
    }
    if (drain_only && te <= p.bath_ev) {
        return 0;
    }
    auto const kp = parallel(density, te, time);
    auto const kt = perpendicular(density, te, time);
    if (!(std::isfinite(kp) && std::isfinite(kt) && kp >= 0 && kt >= 0)) {
        return std::numeric_limits<amrex::Real>::quiet_NaN();
    }
    auto const knn = kt + (kp - kt) * normal_fraction;
    auto flux =
        knn / p.half_cell * (te - p.bath_ev) * PhysConst::q_e / PhysConst::kb;
    if (p.flux_limit > 0 && flux > 0) {
        auto const kelvin = te * PhysConst::q_e / PhysConst::kb;
        auto const cap = p.flux_limit * density * PhysConst::q_e * te *
                         WallCapSpeed(kelvin, p.sonic, p.gamma, p.ion_mass);
        flux = amrex::min(flux, cap);
    }
    return flux;
}
} // namespace warpx::thermal
#endif
