/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_THERMAL_SOURCE_ACCEPTANCE_H_
#define WARPX_THERMAL_SOURCE_ACCEPTANCE_H_

#include "EulerianThermalSources.H"
#include <array>
#include <cmath>

namespace warpx::thermal {
enum class ThermalSourceAcceptanceStatus {
    Ready,
    NonfiniteLedger,
    InvalidAbsoluteWork,
    UnassignableAppliedWork
};

inline constexpr std::array<const char*, SourceComponent::Count>
    ThermalSourceChannelNames{"electron_total",
                              "joule",
                              "relaxation",
                              "viscous",
                              "hyperresistive",
                              "external_sink",
                              "stopping",
                              "declined_joule_gate",
                              "declined_redirect",
                              "declined_stopping",
                              "viscous_work",
                              "viscous_strain",
                              "declined_viscous_work",
                              "declined_hyperresistive",
                              "unassignable_viscous_work_abs",
                              "unassignable_hyperresistive_abs",
                              "declined_joule_work",
                              "unassignable_joule_work_abs"};

// Inspect an ALREADY reduced converged-step ledger before irreversible work.
// This pure policy adds no reduction, RNG access, accepted write, or tally.
// Match the maintained explicit exact-zero orphan policy: positive/negative
// unassigned edges must not cancel. Signed cooling in delivered channels is
// valid.
inline ThermalSourceAcceptanceStatus
InspectThermalSourceAcceptance (const ThermalSourceLedger& ledger,
                               amrex::Real verified_joule_error=0) {
    // Optional independently verified empty-vacuum numerical receipt only.
    // Default and every other applied-work channel retain the exact policy.
    if (!std::isfinite(verified_joule_error) || verified_joule_error<0) {
        return ThermalSourceAcceptanceStatus::InvalidAbsoluteWork;
    }
    for (auto const value : ledger.energy) {
        if (!std::isfinite(value)) {
            return ThermalSourceAcceptanceStatus::NonfiniteLedger;
        }
    }
    for (auto const* values : {&ledger.ion_relaxation, &ledger.ion_redirect}) {
        for (auto const value : *values) {
            if (!std::isfinite(value)) {
                return ThermalSourceAcceptanceStatus::NonfiniteLedger;
            }
        }
    }
    for (int const channel : {SourceComponent::UnassignableViscousWorkAbs,
                              SourceComponent::UnassignableHyperResistiveAbs,
                              SourceComponent::UnassignableJouleWorkAbs}) {
        if (ledger.energy[channel] < 0) {
            return ThermalSourceAcceptanceStatus::InvalidAbsoluteWork;
        }
        amrex::Real const allowance=channel==SourceComponent::UnassignableJouleWorkAbs
            ? verified_joule_error : 0.;
        if (ledger.energy[channel] > allowance) {
            return ThermalSourceAcceptanceStatus::UnassignableAppliedWork;
        }
    }
    return ThermalSourceAcceptanceStatus::Ready;
}
} // namespace warpx::thermal
#endif
