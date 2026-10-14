/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_BRAGINSKII_CONDUCTIVITY_H_
#define WARPX_BRAGINSKII_CONDUCTIVITY_H_
#include "Utils/WarpXConst.H"
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Algorithm.H>
#include <AMReX_REAL.H>
#include <algorithm>
#include <cmath>
#include <limits>

/** Z=1 electron transport (trace alpha approximation). Density is physical
 * electron density in m^-3, temperature eV, B tesla, kappa W/(m K).
 * Real electron mass is used, independent of the inertial regularization.
 * NRL 2023 pp.34,36: tau=3.44e5 T^1.5/(n[cm^-3] lnLambda) seconds.
 * NRL cold/hot Z=1 log branches meet at 10 eV (rounded fits); lnLambda is
 * bounded below by 2, explicitly identifying the weak-coupling regularization.
 * A positive fixed_log selects an explicitly fixed Coulomb logarithm instead.
 * Full-Hall factors: Braginskii/PlasmaPy Z=1 interpolation. No 1/B^2 singularity.
 */
struct BraginskiiConductivity {
    amrex::Real perpendicular = 0.0;
    amrex::Real parallel_physical = 0.0;
    amrex::Real parallel_used = 0.0;
    amrex::Real hall = 0.0;
    amrex::Real coulomb_log = 0.0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE BraginskiiConductivity
EvaluateBraginskiiConductivity (amrex::Real const density,
                               amrex::Real const temperature,
                               amrex::Real const B,
                               amrex::Real const max_anisotropy,
                               amrex::Real const fixed_log = 0.0)
{
    BraginskiiConductivity c;
    // Empty physical cells have no carriers. Numerical pedestal is not n.
    if (density <= 0.0 || temperature <= 0.0) { return c; }
    amrex::Real const log_n_cm = std::log(density) - std::log(1.e6);
    amrex::Real const log_t = std::log(temperature);
    c.coulomb_log = fixed_log > 0.0 ? fixed_log :
        amrex::max(amrex::Real(2.0), temperature < 10.0
                    ? 23.0 - 0.5 * log_n_cm + 1.5 * log_t
                    : 24.0 - 0.5 * log_n_cm + log_t);
    // Cancel density in n*tau before floating-point evaluation. The
    // physical density may be positive but arbitrarily small in deposited
    // particle tails; forming tau first can overflow and produce 0*inf.
    amrex::Real const n_tau = 3.44e11 * temperature * std::sqrt(temperature) /
                             c.coulomb_log;
    amrex::Real const base = PhysConst::kb *
        (PhysConst::q_e * temperature) * n_tau / PhysConst::m_e;
    c.parallel_physical = base * (11.92 / 3.7703);
    amrex::Real const omega_n_tau = (PhysConst::q_e / PhysConst::m_e) *
                                   std::abs(B) * n_tau;
    if (omega_n_tau <= density) {
        // B=0 must not form 0*infinite collision time in an empty tail.
        c.hall = omega_n_tau / density;
        amrex::Real const x2 = c.hall * c.hall;
        c.perpendicular = base * (4.664*x2 + 11.92) /
                          (x2*x2 + 14.79*x2 + 3.7703);
    } else {
        // Evaluate the magnetized factor from n/(omega*n_tau), including its
        // zero-density limit. Hall is diagnostic and may exceed Real range;
        // the transport coefficients remain finite without a density floor.
        amrex::Real const inv = density / omega_n_tau;
        c.hall = inv > 0.0 ? 1.0 / inv : std::numeric_limits<amrex::Real>::infinity();
        amrex::Real const y = inv * inv;
        amrex::Real const shape =
            (4.664 + 11.92*y) / (1.0 + 14.79*y + 3.7703*y*y);
        c.perpendicular = ((base * inv) * inv) * shape;
        // Apply the anisotropy cap before a tiny perpendicular coefficient
        // rounds to zero, preserving a representable capped parallel value.
        c.parallel_used = amrex::min(c.parallel_physical,
            (((base * inv) * max_anisotropy) * inv) * shape);
        return c;
    }
    c.parallel_used = amrex::min(c.parallel_physical, max_anisotropy*c.perpendicular);
    return c;
}
#endif
