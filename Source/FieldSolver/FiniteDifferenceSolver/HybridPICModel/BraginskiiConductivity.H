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
    amrex::Real const tau = 3.44e11 * temperature * std::sqrt(temperature) /
                           (density * c.coulomb_log);
    c.hall = PhysConst::q_e * B * tau / PhysConst::m_e;
    amrex::Real const base = density * PhysConst::kb *
        (PhysConst::q_e * temperature) * tau / PhysConst::m_e;
    c.parallel_physical = base * (11.92 / 3.7703);
    amrex::Real factor;
    if (c.hall <= 1.0) {
        amrex::Real const x2 = c.hall * c.hall;
        factor = (4.664 * x2 + 11.92) / (x2*x2 + 14.79*x2 + 3.7703);
    } else {
        // Reciprocal form avoids x^4 overflow in strongly magnetized cells.
        amrex::Real const inv = 1.0 / c.hall;
        amrex::Real const y = inv * inv;
        factor = y * (4.664 + 11.92*y) / (1.0 + 14.79*y + 3.7703*y*y);
    }
    c.perpendicular = base * factor;
    c.parallel_used = amrex::min(c.parallel_physical, max_anisotropy*c.perpendicular);
    return c;
}
#endif
