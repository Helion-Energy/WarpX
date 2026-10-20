/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_QDSMC_CONDUCTION_REPORT_H_
#define WARPX_QDSMC_CONDUCTION_REPORT_H_
#include <AMReX_REAL.H>
#include <array>
#include <string>
#include <limits>

/** One FD conduction call. Energies are joules; physical heat is positive
 * outward. Domain channels are 2*dimension+side; channel 6 is the EB ring.
 * Failed calls publish no state or cumulative heat. This per-call diagnostic
 * is recomputed after restart, not an invented legacy cumulative history. */
struct QdsmcConductionReport {
    bool completed = false;
    std::string failure;
    amrex::Real requested_time = 0.0, completed_time = 0.0;
    int attempts = 0, accepted = 0, s_max = 0;
    int retries = 0, rejected_work = 0, admissibility_retries = 0;
    // Last finite rejected stage; bounds include reservoirs and imposed fluxes.
    amrex::Real rejected_temperature_min = 0.0, rejected_temperature_max = 0.0;
    amrex::Real temperature_bound_min = 0.0, temperature_bound_max = 0.0;
    amrex::Real attempted_dt = 0.0, required_cap = 0.0, observed_cap = 0.0;
    amrex::Real rejected_dt = 0.0, rejected_required_cap = 0.0, rejected_observed_cap = 0.0;
    std::array<amrex::Real, 7> outward_heat{};
    std::array<amrex::Real, 7> entry_heat{}; // positive into constrained state
    amrex::Real floor_heat = 0.0, floor_raw_heat = 0.0, floor_count = 0.0;
    // Capacity, physical n_eff, legacy bulk, legacy band, covered subset,
    // and closed subset. Subsets are not extra reservoirs.
    std::array<amrex::Real, 6> energy_before{}, energy_after{};
    amrex::Real residual = 0.0;
    // Sampled valid owned open nodes, final accepted state, every configured interval.
    // n_phys, n_capacity [m^-3], chi_perp, chi_par_phys, chi_par_used [m^2/s], Hall.
    bool physical_coefficients_sampled = false;
    std::array<amrex::Real, 6> coefficient_min{}, coefficient_max{};
    amrex::Real coefficient_nodes = 0.0, parallel_capped_nodes = 0.0;
    // Positive Hall overflow is permitted by the inverse-Hall transport branch.
    // Only its reported bound is saturated, with explicit metadata; this is
    // neither a physical cap nor a finite measurement. Other nonfinite values
    // remain invalid. Call once after the global coefficient reductions.
    bool hall_min_overflow = false, hall_max_overflow = false;
    void EncodeHallOverflow () {
        auto encode = [] (amrex::Real& bound, bool& overflow) {
            overflow = bound == std::numeric_limits<amrex::Real>::infinity();
            if (overflow) { bound = std::numeric_limits<amrex::Real>::max(); }
        };
        encode(coefficient_min[5], hall_min_overflow);
        encode(coefficient_max[5], hall_max_overflow);
    }
};
#endif
