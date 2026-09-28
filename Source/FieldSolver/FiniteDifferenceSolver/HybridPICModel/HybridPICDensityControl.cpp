/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/KineticThermalMoments.H"
#include "HybridPICModel.H"
#include "WarpX.H"
#include <AMReX_Print.H>
#include <iomanip>
#include <limits>
using warpx::fields::FieldType;
using namespace warpx::thermal;

void
HybridPICModel::InitializeDensityControl () {
    auto& c = m_density_control;
    c.initial_number_floor = c.number_floor = m_n_floor;
    c.initial_gate_floor = c.gate_floor = m_qdsmc_n_floor;
    c.initial_representation_floor = c.representation_floor =
        m_qdsmc_te_n_floor;
    c.pedestal = m_density_pedestal;
    c.pedestal_tracks_floor = m_density_pedestal_track_floor;
    c.pedestal_taper_cells = m_density_pedestal_eb_taper_cells;
    c.pedestal_profile = m_density_pedestal_expression;
    c.halo_unfreeze = m_qdsmc_halo_unfreeze;
    if (UsesEulerianElectronEnergy()) {
        c.Validate();
    }
}
void
HybridPICModel::AssertDensityControlState () const {
    if (!UsesEulerianElectronEnergy()) {
        return;
    }
    m_density_control.Validate();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_n_floor == m_density_control.number_floor &&
            m_qdsmc_n_floor == m_density_control.gate_floor &&
            m_qdsmc_te_n_floor == m_density_control.representation_floor,
        "Eulerian density floors must change through accepted-boundary "
        "setters");
}
std::uint64_t
HybridPICModel::DensityControlEpoch () const {
    AssertDensityControlState();
    return m_density_control.epoch;
}
void
HybridPICModel::BeginDensityControlStep () {
    AssertDensityControlState();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_density_control_step_open &&
            !m_density_controller_needs_registration,
        "Density control requires an accepted boundary and restored controller "
        "registration");
    m_density_control_step_open = true;
}
void
HybridPICModel::EndDensityControlStep () {
    m_density_control_step_open = false;
}
void
HybridPICModel::SetHybridDensityFloor (amrex::Real value) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(value) && value > 0.,
        "Hybrid density floor must be finite and positive");
    if (!UsesEulerianElectronEnergy()) {
        m_n_floor = value;
        if (m_density_pedestal_track_floor) {
            m_density_pedestal_stale = true;
        }
        return;
    }
    AssertDensityControlState();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_density_control_step_open && !HasElectronThermalTrial(0),
        "Density floor changes require an accepted step boundary");
    if (value == m_n_floor) {
        return;
    }
    auto& w = WarpX::GetInstance();
    DensityControlInventory inventory{true, 0., 0., 0.};
    if (m_eulerian_energy_initialized[0]) {
        auto& u = *w.m_fields.get(FieldType::hybrid_electron_energy_fp, 0);
        auto const& rho = *w.m_fields.get(FieldType::rho_fp, 0);
        auto const geometry = ElectronThermalGeometry(0);
        auto old_options = EulerianMomentOptions();
        auto new_options = old_options;
        new_options.number_density_floor = value;
        KineticThermalMoments old(geometry, u.boxArray(), u.DistributionMap(),
                                  old_options);
        KineticThermalMoments middle(geometry, u.boxArray(),
                                     u.DistributionMap(), new_options);
        KineticThermalMoments next(geometry, u.boxArray(), u.DistributionMap(),
                                   new_options);
        KineticThermalStateView state{rho};
        state.pedestal = DensityPedestal(0);
        AMREX_ALWAYS_ASSERT(old.Evaluate(state) && middle.Evaluate(state));
        amrex::MultiFab candidate_pedestal;
        if (m_density_pedestal_track_floor) {
            candidate_pedestal.define(state.pedestal->boxArray(),
                                      state.pedestal->DistributionMap(), 1, 0);
            candidate_pedestal.setVal(value * PhysConst::q_e);
            state.pedestal = &candidate_pedestal;
        }
        AMREX_ALWAYS_ASSERT(next.Evaluate(state));
        amrex::MultiFab candidate(u.boxArray(), u.DistributionMap(), 1, 0);
        inventory = RemapElectronDensityControl(
            geometry, u, old.NumberDensity(), middle.NumberDensity(),
            next.NumberDensity(), candidate);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            inventory.valid, "Invalid accepted density-control remap");
        amrex::MultiFab::Copy(u, candidate, 0, 0, 1, 0);
    }
    auto& c = m_density_control;
    m_n_floor = c.number_floor = value;
    m_qdsmc_te_n_floor = c.representation_floor = value;
    if (m_density_pedestal_track_floor) {
        m_density_pedestal_stale = true;
    }
    if (!c.controller_configuration.empty()) {
        c.controller_last = value;
    }
    ++c.epoch;
    c.floor_inventory_joule += inventory.floor_joule;
    c.pedestal_inventory_joule += inventory.pedestal_joule;
    if (m_eulerian_energy_initialized[0]) {
        RefreshEulerianElectronThermodynamics(
            0, *w.m_fields.get(FieldType::rho_fp, 0));
    }
    AssertDensityControlState();
    amrex::Print() << std::setprecision(17)
                   << "Eulerian density-control inventory [J]: epoch="
                   << c.epoch << " floor=" << inventory.floor_joule
                   << " pedestal=" << inventory.pedestal_joule
                   << " total=" << inventory.total_joule
                   << " cumulative_floor=" << c.floor_inventory_joule
                   << " cumulative_pedestal=" << c.pedestal_inventory_joule
                   << "\n";
}
void
HybridPICModel::SetQdsmcDensityFloor (amrex::Real value) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(value) && value >= 0.,
        "QDSMC density gate must be finite and nonnegative");
    if (!UsesEulerianElectronEnergy()) {
        m_qdsmc_n_floor = value;
        return;
    }
    AssertDensityControlState();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(value > 0.,
                                     "Eulerian density gate must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_density_control_step_open && !HasElectronThermalTrial(0),
        "Density gate changes require an accepted step boundary");
    if (value != m_qdsmc_n_floor) {
        m_qdsmc_n_floor = m_density_control.gate_floor = value;
        ++m_density_control.epoch;
    }
}
const ElectronDensityControl&
HybridPICModel::ConfigureDensityFloorController (
    const std::string& configuration) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        UsesEulerianElectronEnergy() && !configuration.empty() &&
            !m_density_control_step_open,
        "Persistent density controller requires accepted Eulerian state");
    auto& c = m_density_control;
    if (c.controller_configuration.empty()) {
        c.controller_configuration = configuration;
        c.controller_last = m_n_floor;
    } else {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            c.controller_configuration == configuration,
            "Restart density controller configuration differs from original "
            "tracking arguments");
    }
    c.Validate();
    m_density_controller_needs_registration = false;
    return c;
}
void
HybridPICModel::CommitDensityFloorController (const std::string& configuration,
                                              amrex::Real last, amrex::Real ema,
                                              bool has_ema, std::int64_t step,
                                              amrex::Real gate) {
    auto const& current = ConfigureDensityFloorController(configuration);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        step > current.controller_step && std::isfinite(ema) && ema >= 0. &&
            (has_ema || ema == 0.) &&
            (gate == -1. || (std::isfinite(gate) && gate > 0.)),
        "Invalid or repeated accepted density-controller update");
    SetHybridDensityFloor(last);
    if (gate >= 0.) {
        SetQdsmcDensityFloor(gate);
    }
    auto& c = m_density_control;
    c.controller_last = last;
    c.controller_ema = ema;
    c.controller_has_ema = has_ema;
    c.controller_step = step;
    c.Validate();
}
std::string
HybridPICModel::DensityControlCheckpointMetadata () const {
    AssertDensityControlState();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_density_control_step_open &&
            !m_density_controller_needs_registration,
        "Density-control checkpoint requires a complete accepted controller "
        "state");
    return m_density_control.Encode();
}
void
HybridPICModel::RestoreDensityControlCheckpointMetadata (
    const std::string& metadata) {
    AMREX_ALWAYS_ASSERT(UsesEulerianElectronEnergy() &&
                        !m_density_control_step_open);
    auto restored = ElectronDensityControl::Decode(metadata);
    restored.ValidateConfiguration(m_density_control);
    m_density_control = std::move(restored);
    m_n_floor = m_density_control.number_floor;
    m_qdsmc_n_floor = m_density_control.gate_floor;
    m_qdsmc_te_n_floor = m_density_control.representation_floor;
    m_density_pedestal_stale = true;
    m_density_controller_needs_registration =
        !m_density_control.controller_configuration.empty();
    AssertDensityControlState();
}
