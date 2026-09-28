/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_THERMAL_STAGE_INITIAL_GUESS_H_
#define WARPX_THERMAL_STAGE_INITIAL_GUESS_H_
#include "Utils/WarpXConst.H"
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <cmath>
#include <limits>

namespace warpx::thermal {
struct ThermalStageGuessOptions {
    amrex::Real theta = 0.5;
    amrex::Real gamma = 5. / 3.;
    amrex::Real temperature_floor_ev = 0.;
    // An interior initial guess leaves room for finite-difference Jv probes.
    // This changes only an infeasible guess, never a physical floor or root.
    amrex::Real relative_margin = 1.e-4;
};
struct ThermalStageGuessPoint {
    amrex::Real energy;
    bool changed;
    bool valid;
};

// Prepare only a trial seed. Accepted old U is immutable. A feasible seed is
// returned bit-for-bit; otherwise move into both stage and affine endpoint
// constraints by a relative representability margin, never clip an accepted
// solution or modify its energy equation. No physical floor is inserted when
// the configured floor is zero. An unrepresentable finite guess rejects.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ThermalStageGuessPoint
PrepareThermalStageGuess (amrex::Real old, amrex::Real seed, amrex::Real nstage,
                          amrex::Real nend,
                          ThermalStageGuessOptions const& o) noexcept {
    using R = amrex::Real;
    if (!(std::isfinite(old) && old > 0 && std::isfinite(seed) &&
          std::isfinite(nstage) && nstage > 0 && std::isfinite(nend) &&
          nend > 0 && std::isfinite(o.theta) && o.theta > 0 && o.theta <= 1 &&
          std::isfinite(o.gamma) && o.gamma > 1 &&
          std::isfinite(o.temperature_floor_ev) &&
          o.temperature_floor_ev >= 0 && std::isfinite(o.relative_margin) &&
          o.relative_margin > 0)) {
        return {seed, false, false};
    }
    R const floor = PhysConst::q_e * o.temperature_floor_ev / (o.gamma - 1.);
    R const oldpart = (1. - o.theta) * old;
    if (!std::isfinite(floor) || (o.temperature_floor_ev > 0 && floor == 0.)) {
        return {seed, false, false};
    }
    auto feasible = [=] AMREX_GPU_HOST_DEVICE(R u) {
        R const stage = u / nstage;
        R const endpoint = (u - oldpart) / o.theta;
        R const end = endpoint / nend;
        return std::isfinite(u) && std::isfinite(stage) && stage > 0 &&
               stage >= floor && std::isfinite(endpoint) &&
               std::isfinite(end) && end > 0 && end >= floor;
    };
    if (feasible(seed)) {
        return {seed, false, true};
    }
    R const stage_bound = nstage * floor;
    R const end_bound = oldpart + o.theta * nend * floor;
    R const bound = stage_bound > end_bound ? stage_bound : end_bound;
    R const scale = old > bound ? old : bound;
    if (!std::isfinite(bound) || !std::isfinite(scale)) {
        return {seed, false, false};
    }
    R candidate = bound + o.relative_margin * scale;
    // nextafter also covers an underflowed margin. The bounded retry checks
    // actual divisions/cancellation used by the production stage, not a proxy.
    for (int attempt = 0; attempt < 4; ++attempt) {
        if (candidate > bound && feasible(candidate)) {
            return {candidate, true, true};
        }
        candidate =
            std::nextafter(candidate, std::numeric_limits<R>::infinity());
    }
    return {seed, false, false};
}

// Matching one-component cell layouts. Reads valid inputs only. All output is
// published transactionally after validation. output may alias seed, but must
// not alias accepted old U or either density. Called to prepare an initial
// Newton guess only; never inside a residual/Jv or acceptance receipt.
bool MakeThermalStageInitialGuess (amrex::MultiFab& output,
                                   const amrex::MultiFab& old,
                                   const amrex::MultiFab& seed,
                                   const amrex::MultiFab& stage_density,
                                   const amrex::MultiFab& endpoint_density,
                                   ThermalStageGuessOptions options);
} // namespace warpx::thermal
#endif
