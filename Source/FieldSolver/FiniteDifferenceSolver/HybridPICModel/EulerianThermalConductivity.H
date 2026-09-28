/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_EULERIAN_THERMAL_CONDUCTIVITY_H_
#define WARPX_EULERIAN_THERMAL_CONDUCTIVITY_H_

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>
#include <algorithm>
#include <cmath>
#include <limits>

namespace warpx::thermal {

// Material-law compatibility; these options do not replace the MHD face
// stencil, its magnetic regularization, or its mixed-flux free-streaming cap.
struct ThermalConductivityOptions {
    bool isotropic = false; // conduct at the perpendicular rate everywhere
    amrex::Real isotropic_B = -1.0; // [T], <=0 disables the weak-B blend
    // Legacy hybrid FD uses the parallel rate in every direction at EXACT
    // B=0 if both isotropy options are off. The MHD default instead keeps its
    // supplied perpendicular channel and smooth magnetic floor. Opt in only
    // when adapting that legacy constitutive convention.
    bool unmagnetized_parallel = false;
};

struct ThermalConductivityPair {
    amrex::Real parallel, perpendicular; // W/(m K)
};

// Both executors take n [m^-3], Te [eV], t [s]. They may already include
// model constants and hard min(kappa,1.5 kB chi_max n) caps. Apply those exact
// laws first, then the maintained isotropy transform. Never cap twice or
// silently change the model's 3/2 heat-capacity constant to another gamma.
// Invalid live coefficients are rejected rather than clipped to zero.
template <typename Parallel, typename Perpendicular>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ThermalConductivityPair
EvaluateThermalConductivity (amrex::Real n, amrex::Real temperature_ev,
                             amrex::Real time, amrex::Real magnetic_squared,
                             ThermalConductivityOptions const& options,
                             Parallel const& parallel,
                             Perpendicular const& perpendicular) noexcept {
    amrex::Real kp = parallel(n, temperature_ev, time);
    amrex::Real kt = perpendicular(n, temperature_ev, time);
    if (!(std::isfinite(kp) && kp >= 0 && std::isfinite(kt) && kt >= 0 &&
          std::isfinite(magnetic_squared) && magnetic_squared >= 0)) {
        auto const invalid = std::numeric_limits<amrex::Real>::quiet_NaN();
        return {invalid, invalid};
    }
    if (options.isotropic) {
        kp = kt;
    } else if (options.isotropic_B > 0) {
        // Equivalent to B²/(B²+Biso²), avoiding an overflowing sum.
        amrex::Real const s =
            magnetic_squared > 0
                ? 1 / (1 + options.isotropic_B * options.isotropic_B /
                               magnetic_squared)
                : 0;
        kp = kt + (kp - kt) * s;
    } else if (options.unmagnetized_parallel && magnetic_squared == 0) {
        kt = kp;
    }
    return {kp, kt};
}

// Callable component of the same law for temperature-dependent physical
// reservoir rows and their local derivative. Parser storage remains borrowed.
template <typename Parallel, typename Perpendicular>
struct ThermalConductivityComponent {
    Parallel parallel;
    Perpendicular perpendicular;
    ThermalConductivityOptions options;
    amrex::Real magnetic_squared;
    int component;
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    operator()(amrex::Real n, amrex::Real temperature_ev,
               amrex::Real time) const noexcept {
        auto const value = EvaluateThermalConductivity(
            n, temperature_ev, time, magnetic_squared, options, parallel,
            perpendicular);
        return component == 0 ? value.parallel : value.perpendicular;
    }
};

struct ThermalConductionDensityOptions {
    amrex::Real number_density_floor = 0; // [m^-3], must be strictly positive
    bool halo_unfreeze = false;
    bool has_pedestal = false; // a zero-valued pedestal still changes the gate
};
struct ThermalConductionDensity {
    amrex::Real effective_number_density;
    bool active;
    bool valid;
};

// Pure maintained density/open-set law at ONE caller-selected sample. Raw
// and pedestal densities are electron NUMBER densities, never mass density.
// It does not transfer nodal state to cells: floor/max and restriction do not
// commute. A caller must explicitly derive and qualify its mask transfer.
// Applying this helper never changes accepted U, temperature, or any tally.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ThermalConductionDensity
EvaluateThermalConductionDensity (
    amrex::Real raw_number_density, amrex::Real pedestal_number_density,
    ThermalConductionDensityOptions const& options) noexcept {
    amrex::Real const total =
        raw_number_density +
        (options.has_pedestal ? pedestal_number_density : 0);
    if (!(std::isfinite(raw_number_density) &&
          std::isfinite(pedestal_number_density) &&
          pedestal_number_density >= 0 && std::isfinite(total) &&
          std::isfinite(options.number_density_floor) &&
          options.number_density_floor > 0)) {
        return {std::numeric_limits<amrex::Real>::quiet_NaN(), false, false};
    }
    amrex::Real const threshold = options.halo_unfreeze || options.has_pedestal
                                      ? 0
                                      : options.number_density_floor;
    return {std::max(total, options.number_density_floor), total > threshold,
            true};
}
} // namespace warpx::thermal
#endif
