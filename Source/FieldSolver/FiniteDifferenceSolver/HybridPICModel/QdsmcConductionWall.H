/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_QDSMC_CONDUCTION_WALL_H_
#define WARPX_QDSMC_CONDUCTION_WALL_H_

#include "Utils/WarpXConst.H"
#include <AMReX_Algorithm.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>
#include <cmath>

namespace warpx::thermal {
using namespace amrex::literals;

/** Pure point laws of the EXISTING accepted-substep boundary map. These
 * functions have no field, tally, ownership or RNG access. The caller supplies
 * density/gate/geometry and applies returned changes only on acceptance.
 * They are not by themselves implicit residual rows: a Dirichlet constraint,
 * a capped drain and a finite-dt leg relaxation have different mathematics.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
WallCapSpeed (amrex::Real const T, bool const sonic, amrex::Real const gamma,
              amrex::Real const ion_mass) {
    return sonic ? std::sqrt(gamma * PhysConst::kb * T / ion_mass)
                 : std::sqrt(PhysConst::kb * T / PhysConst::m_e);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
IsothermalWallTemperature (amrex::Real const T, amrex::Real const wall_T,
                           amrex::Real const cap_dt_area_over_volume,
                           bool const sonic, amrex::Real const gamma,
                           amrex::Real const ion_mass) {
    if (cap_dt_area_over_volume > 0.0_rt && T > wall_T) {
        auto const speed = WallCapSpeed(T, sonic, gamma, ion_mass);
        return amrex::max(wall_T,
                          T - cap_dt_area_over_volume * speed * T / 1.5_rt);
    }
    return wall_T;
}

/** du_flux = prescribed inward heat flux * dt * wall area / dual volume. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
PrescribedWallTemperatureChange (amrex::Real const T, amrex::Real const ne,
                                 amrex::Real const du_flux) {
    return amrex::max(du_flux / (1.5_rt * PhysConst::kb * ne), -T);
}

struct LegTarget {
    amrex::Real temperature = 0.0_rt;
    amrex::Real interior_conductance = 0.0_rt;
    amrex::Real leg_conductance = 0.0_rt;
};

/** Exact finite-iteration series target of the legacy leg map. Densities are
 * already floored; ne_face is their harmonic mean. face_open is the original
 * two-node density/coverage gate, not a new temperature mask. The caller must
 * preserve the drain gate (interior T on open face, wall T on closed face).
 */
template <typename ParallelKappa>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE LegTarget
EvaluateLegTarget (amrex::Real const T, amrex::Real const inner_T,
                   amrex::Real const ne_wall, amrex::Real const ne_inner,
                   amrex::Real const ne_face, bool const face_open,
                   amrex::Real const dx, amrex::Real const inner_area_ratio,
                   amrex::Real const length, amrex::Real const wall_T,
                   amrex::Real const min_T, int const iterations,
                   ParallelKappa const& kappa, amrex::Real const time) {
    amrex::Real const inner_eV =
        amrex::max(inner_T, 0.0_rt) * PhysConst::kb / PhysConst::q_e;
    amrex::Real const kappa_inner =
        amrex::max(kappa(ne_inner, inner_eV, time), 0.0_rt);
    LegTarget result;
    result.temperature = amrex::max(T, min_T);
    for (int it = 0; it < iterations; ++it) {
        amrex::Real const face_eV =
            result.temperature * PhysConst::kb / PhysConst::q_e;
        amrex::Real const kappa_face_node =
            amrex::max(kappa(ne_wall, face_eV, time), 0.0_rt);
        amrex::Real const kappa_face =
            face_open ? ne_face * 0.5_rt *
                            (kappa_inner / ne_inner + kappa_face_node / ne_wall)
                      : 0.0_rt;
        result.interior_conductance = kappa_face * inner_area_ratio / dx;
        result.leg_conductance = kappa_face_node / length;
        auto const sum = result.interior_conductance + result.leg_conductance;
        result.temperature = sum > 0.0_rt
                                 ? (result.interior_conductance * inner_T +
                                    result.leg_conductance * wall_T) /
                                       sum
                                 : wall_T;
        result.temperature = amrex::max(result.temperature, min_T);
    }
    return result;
}

/** Existing finite-dt row map, including drain-only behavior and cap.
 * It contains G_int as well as G_leg; do not replace it by -G_leg*(T-Twall)
 * and call that equivalent. The instantaneous FD RHS already has interior
 * flux, so the overlap must be resolved before defining an implicit wall row.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
LegWallTemperature (amrex::Real const T, LegTarget const target,
                    amrex::Real const ne_floored,
                    amrex::Real const area_over_volume, amrex::Real const dt,
                    amrex::Real const cap_dt_area_over_volume, bool const sonic,
                    amrex::Real const gamma, amrex::Real const ion_mass) {
    amrex::Real const capacity =
        1.5_rt * PhysConst::kb * ne_floored / area_over_volume;
    auto const relax =
        std::exp(-dt * (target.interior_conductance + target.leg_conductance) /
                 capacity);
    amrex::Real out = target.temperature + (T - target.temperature) * relax;
    if (out >= T) {
        return T;
    }
    if (cap_dt_area_over_volume > 0.0_rt) {
        auto const speed = WallCapSpeed(T, sonic, gamma, ion_mass);
        out = amrex::max(out, T - cap_dt_area_over_volume * speed * T / 1.5_rt);
    }
    return out;
}

/** Right derivative at dt=0 of LegWallTemperature, holding its target fixed.
 * Caller still applies the original source-temperature gate. This is the
 * legacy-map generator for analysis/qualification, NOT an approved physical
 * Robin residual. cap_area_over_volume excludes dt and is zero when disabled.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
LegWallRate (amrex::Real const T, LegTarget const target,
             amrex::Real const ne_floored, amrex::Real const area_over_volume,
             amrex::Real const cap_area_over_volume, bool const sonic,
             amrex::Real const gamma, amrex::Real const ion_mass) {
    auto const capacity =
        1.5_rt * PhysConst::kb * ne_floored / area_over_volume;
    auto rate = amrex::min(
        (target.temperature - T) *
            (target.interior_conductance + target.leg_conductance) / capacity,
        0.0_rt);
    if (cap_area_over_volume > 0.0_rt) {
        rate = amrex::max(rate, -cap_area_over_volume *
                                    WallCapSpeed(T, sonic, gamma, ion_mass) *
                                    T / 1.5_rt);
    }
    return rate;
}
} // namespace warpx::thermal
#endif
