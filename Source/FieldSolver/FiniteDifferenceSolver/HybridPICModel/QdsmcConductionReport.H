/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_QDSMC_CONDUCTION_REPORT_H_
#define WARPX_QDSMC_CONDUCTION_REPORT_H_
#include <AMReX_REAL.H>
#include <array>
#include <string>

/** One FD conduction call. Energies are joules; physical heat is positive
 * outward. Domain channels are 2*dimension+side; channel 6 is the EB ring.
 * Failed calls publish no state or cumulative heat. This per-call diagnostic
 * is recomputed after restart, not an invented legacy cumulative history. */
struct QdsmcConductionReport {
    bool completed = false;
    std::string failure;
    amrex::Real requested_time = 0.0, completed_time = 0.0;
    int attempts = 0, accepted = 0, s_max = 0;
    std::array<amrex::Real, 7> outward_heat{};
    std::array<amrex::Real, 7> entry_heat{}; // positive into constrained state
    amrex::Real floor_heat = 0.0, floor_raw_heat = 0.0, floor_count = 0.0;
    // Capacity, physical n_eff, legacy bulk, legacy band, covered subset,
    // and closed subset. Subsets are not extra reservoirs.
    std::array<amrex::Real, 6> energy_before{}, energy_after{};
    amrex::Real residual = 0.0;
};
#endif
