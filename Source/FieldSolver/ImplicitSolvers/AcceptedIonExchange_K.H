/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_ACCEPTED_ION_EXCHANGE_K_H_
#define WARPX_ACCEPTED_ION_EXCHANGE_K_H_
#include "Particles/Algorithms/KineticEnergy.H"
#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <cmath>
namespace warpx::thermal {
struct IonExchangeKick {
    amrex::GpuArray<amrex::ParticleReal, 3> momentum{};
    amrex::Real expected_relaxation = 0, expected_redirect = 0;
    amrex::Real realized_relaxation = 0, realized_redirect = 0,
                realized_total = 0;
};
// Exact native expm1 OU arithmetic, with an explicit supplied normal vector.
// The caller is solely responsible for drawing it AFTER acceptance. RZ drift
// rotates to Cartesian at each endpoint particle; stored proper momenta are
// updated directly, preserving the legacy (nonrelativistic OU) convention.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE IonExchangeKick
ApplyIonExchangeKick (
    amrex::GpuArray<amrex::ParticleReal, 3> const& u,
    amrex::GpuArray<amrex::ParticleReal, 3> const& cylindrical_drift,
    amrex::ParticleReal theta, amrex::ParticleReal nu,
    amrex::ParticleReal temperature, amrex::ParticleReal redirect_energy,
    amrex::ParticleReal mass, amrex::ParticleReal weight, amrex::Real dt,
    amrex::GpuArray<amrex::ParticleReal, 3> const& normal) {
    using P = amrex::ParticleReal;
    P const c = std::cos(theta), s = std::sin(theta);
    amrex::GpuArray<P, 3> const drift{
        cylindrical_drift[0] * c - cylindrical_drift[1] * s,
        cylindrical_drift[0] * s + cylindrical_drift[1] * c,
        cylindrical_drift[2]};
    P const nu_dt = nu * dt;
    P const drag = -std::expm1(-nu_dt);
    P const thermal_energy =
        -PhysConst::kb * temperature * std::expm1(-2 * nu_dt);
    P const variance = (thermal_energy + redirect_energy) / mass;
    P const sigma = std::sqrt(amrex::max(P(0), variance));
    P const relaxation_sigma =
        std::sqrt(amrex::max(P(0), thermal_energy / mass));
    IonExchangeKick result;
    amrex::GpuArray<P, 3> relaxation{};
    amrex::Real mean2 = 0, initial2 = 0;
    for (int d = 0; d < 3; ++d) {
        P const deterministic = -drag * (u[d] - drift[d]);
        P const mean = u[d] + deterministic;
        // Preserve legacy `u += -drag*(u-ue) + sigma*R` operation order.
        result.momentum[d] = u[d] + (deterministic + sigma * normal[d]);
        relaxation[d] = u[d] + (deterministic + relaxation_sigma * normal[d]);
        mean2 += amrex::Real(mean) * mean;
        initial2 += amrex::Real(u[d]) * u[d];
    }
    result.expected_relaxation =
        weight * (.5 * mass * (mean2 - initial2) + 1.5 * thermal_energy);
    result.expected_redirect = weight * 1.5 * redirect_energy;
    amrex::Real const before =
        Algorithms::KineticEnergy<amrex::Real>(u[0], u[1], u[2], mass);
    amrex::Real const relaxed = Algorithms::KineticEnergy<amrex::Real>(
        relaxation[0], relaxation[1], relaxation[2], mass);
    amrex::Real const after = Algorithms::KineticEnergy<amrex::Real>(
        result.momentum[0], result.momentum[1], result.momentum[2], mass);
    result.realized_relaxation = weight * (relaxed - before);
    result.realized_redirect = weight * (after - relaxed);
    result.realized_total = weight * (after - before);
    return result;
}
} // namespace warpx::thermal
#endif
