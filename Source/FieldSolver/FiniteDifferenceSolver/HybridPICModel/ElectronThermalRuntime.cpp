/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "ElectronThermalRuntime.H"
#include "Utils/WarpXConst.H"
#include <AMReX_MFIter.H>
#include <AMReX_Print.H>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace warpx::thermal {
namespace {
ElectronEnergyMode
decode_mode (const std::string& name) {
    if (name == "legacy") {
        return ElectronEnergyMode::Legacy;
    }
    if (name == "decoupled_jfnk") {
        return ElectronEnergyMode::DecoupledJFNK;
    }
    if (name == "coupled_jfnk") {
        return ElectronEnergyMode::CoupledJFNK;
    }
    amrex::Abort("hybrid_pic_model.electron_energy_mode must be legacy, "
                 "decoupled_jfnk or coupled_jfnk");
    return ElectronEnergyMode::Legacy;
}
void
check_trial (const amrex::MultiFab& trial, const amrex::MultiFab& accepted) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        &trial != &accepted && trial.boxArray() == accepted.boxArray() &&
            trial.DistributionMap() == accepted.DistributionMap() &&
            trial.nComp() == accepted.nComp() &&
            trial.ixType().nodeCentered() &&
            trial.nGrowVect().allGE(accepted.nGrowVect()),
        "Electron thermal trial needs independent nodal storage, matching "
        "layout and all required ghosts");
}
} // namespace
ElectronEnergyMode
ParseElectronEnergyMode (const amrex::ParmParse& pp, bool& solve) {
    std::string name = "legacy";
    pp.query("electron_energy_mode", name);
    auto const mode = decode_mode(name);
    bool const supplied = pp.query("solve_electron_energy_equation", solve);
    if (mode != ElectronEnergyMode::Legacy) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            !supplied || solve, "electron_energy_mode JFNK conflicts with "
                                "solve_electron_energy_equation=0");
        solve = true;
    }
    return mode;
}
const char*
ElectronEnergyModeName (ElectronEnergyMode mode) {
    switch (mode) {
    case ElectronEnergyMode::Legacy:
        return "legacy";
    case ElectronEnergyMode::DecoupledJFNK:
        return "decoupled_jfnk";
    case ElectronEnergyMode::CoupledJFNK:
        return "coupled_jfnk";
    }
    amrex::Abort("invalid electron energy mode");
    return "invalid";
}
void
ValidateElectronEnergyRuntime (ElectronEnergyMode mode,
                               const ElectronEnergyRuntimeConfig& c) {
    if (mode == ElectronEnergyMode::Legacy) {
        return;
    }
#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_3D)
    amrex::Abort("Eulerian JFNK electron-energy runtime requires RZ or Cartesian 3D; "
                 "legacy geometry paths are unchanged");
#endif
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        c.darwin && c.evolve_scheme == "theta_implicit_hybrid",
        "Eulerian JFNK electron energy requires Darwin theta_implicit_hybrid");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !c.embedded_boundary && c.max_level == 0,
        "Eulerian JFNK electron energy requires one level and no EB");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(c.gamma) && c.gamma > 1.,
        "Eulerian electron internal energy requires gamma>1");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        c.smooth_floor_width == 0. && c.number_floor > 0. &&
            std::isfinite(c.number_floor) &&
            c.representation_floor == c.number_floor,
        "Eulerian moment adapter requires hard common Ohm/energy density floor "
        "(n_floor_smooth_width=0 and qdsmc_te_n_floor=n_floor)");
}
void
ElectronThermalTrialViews::Set (int lev, const amrex::MultiFab& temperature,
                                const amrex::MultiFab& pressure,
                                const amrex::MultiFab& accepted_temperature,
                                const amrex::MultiFab& accepted_pressure) {
    AMREX_ALWAYS_ASSERT(lev >= 0);
    check_trial(temperature, accepted_temperature);
    check_trial(pressure, accepted_pressure);
    m_views[lev] = {&temperature, &pressure};
}
const amrex::MultiFab&
ElectronThermalTrialViews::Temperature (int lev,
                                        const amrex::MultiFab& accepted) const {
    auto const it = m_views.find(lev);
    return it == m_views.end() ? accepted : *it->second.temperature;
}
const amrex::MultiFab&
ElectronThermalTrialViews::Pressure (int lev,
                                     const amrex::MultiFab& accepted) const {
    auto const it = m_views.find(lev);
    return it == m_views.end() ? accepted : *it->second.pressure;
}
ElectronThermalModelConfig
ElectronThermalModelConfig::Resolve (const amrex::ParmParse& pp,
                                     bool relaxation, bool joule_redirect) {
    ElectronThermalModelConfig c;
    // Resolve is called only for the Eulerian modes. Pair electron relaxation
    // with the accepted finite-step OU thermal expectation by default; legacy
    // modes retain their original source law and never enter this resolver.
    if (relaxation) { c.relaxation_partner = "population_bounded_nr"; }
    pp.query("expected_ou_audit", c.expected_ou_convention);
    pp.query("expected_ou_thermal_partner", c.relaxation_partner);
    pp.query("expected_ou_relativistic_reference", c.relativistic_reference);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        c.relaxation_partner == "off" || c.PopulationPartner(),
        "expected_ou_thermal_partner must be off or population_bounded_nr");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!c.relativistic_reference || c.PopulationPartner(),
        "expected_ou_relativistic_reference requires the population thermal partner");
    if (c.PopulationPartner()) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(relaxation &&
            (c.expected_ou_convention == "off" || c.expected_ou_convention == "bounded_nr"),
            "Population OU thermal partner requires relaxation and bounded_nr audit convention");
        c.expected_ou_convention = "bounded_nr";
    }
    if (c.ExpectedAudit()) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(relaxation || joule_redirect,
            "Expected OU source requires an active accepted exchange channel");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(c.expected_ou_convention == "relativistic" ||
            c.expected_ou_convention == "bounded_nr",
            "expected_ou_audit must be off, relativistic or bounded_nr");
        c.expected_options.mode = c.expected_ou_convention == "relativistic"
            ? ExpectedIonEnergyMode::RelativisticQuadrature
            : ExpectedIonEnergyMode::NonrelativisticBounded;
        pp.query("expected_ou_absolute_tolerance_joule", c.expected_options.absolute_tolerance_joule);
        pp.query("expected_ou_relative_tolerance", c.expected_options.relative_component_tolerance);
    }
    c.Validate();
    if (relaxation && !c.PopulationPartner()) {
        amrex::Print() << "[hybrid] expected_ou_thermal_partner=off selects the "
            "diagnostic/reference requested-rate electron source; it does not pair "
            "the finite-step expected ion OU thermal exchange.\n";
    }
    return c;
}
void
ElectronThermalModelConfig::Validate () const {
    ElectronThermalModelConfig const defaults;
    auto const mode = expected_options.mode;
    bool const convention =
        (expected_ou_convention == "off" && mode == ExpectedIonEnergyMode::Unspecified) ||
        (expected_ou_convention == "bounded_nr" && mode == ExpectedIonEnergyMode::NonrelativisticBounded) ||
        (expected_ou_convention == "relativistic" && mode == ExpectedIonEnergyMode::RelativisticQuadrature);
    bool const valid =
        (relaxation_partner == "off" || PopulationPartner()) && convention &&
        (!PopulationPartner() || expected_ou_convention == "bounded_nr") &&
        (!relativistic_reference || PopulationPartner()) &&
        std::isfinite(expected_options.absolute_tolerance_joule) &&
        expected_options.absolute_tolerance_joule >= 0. &&
        std::isfinite(expected_options.relative_component_tolerance) &&
        expected_options.relative_component_tolerance >= 0. &&
        (ExpectedAudit() ||
         (expected_options.absolute_tolerance_joule == defaults.expected_options.absolute_tolerance_joule &&
          expected_options.relative_component_tolerance == defaults.expected_options.relative_component_tolerance)) &&
        source_semantics == defaults.source_semantics &&
        endpoint_absorption_policy == defaults.endpoint_absorption_policy &&
        boundary_history_semantics == defaults.boundary_history_semantics;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(valid,
        "Invalid Eulerian thermal model contract or unsupported source/boundary semantics");
}
std::string
ElectronThermalModelConfig::EncodePhysicalContract () const {
    Validate();
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
        << "thermal_model 1\nrelaxation_partner " << relaxation_partner
        << "\nexpected_ou_convention " << expected_ou_convention
        << "\nexpected_ou_absolute_tolerance_joule " << expected_options.absolute_tolerance_joule
        << "\nexpected_ou_relative_tolerance " << expected_options.relative_component_tolerance
        << "\nsource_semantics " << source_semantics
        << "\nendpoint_absorption_policy " << endpoint_absorption_policy
        << "\nboundary_history_semantics " << boundary_history_semantics << '\n';
    return out.str();
}
ElectronThermalModelConfig
ElectronThermalModelConfig::DecodePhysicalContract (std::istream& in) {
    ElectronThermalModelConfig c;
    std::string label;
    int version = 0;
    in >> label >> version;
    bool okay = label == "thermal_model" && version == 1;
    in >> label >> c.relaxation_partner;
    okay = okay && label == "relaxation_partner";
    in >> label >> c.expected_ou_convention;
    okay = okay && label == "expected_ou_convention";
    in >> label >> c.expected_options.absolute_tolerance_joule;
    okay = okay && label == "expected_ou_absolute_tolerance_joule";
    in >> label >> c.expected_options.relative_component_tolerance;
    okay = okay && label == "expected_ou_relative_tolerance";
    in >> label >> c.source_semantics;
    okay = okay && label == "source_semantics";
    in >> label >> c.endpoint_absorption_policy;
    okay = okay && label == "endpoint_absorption_policy";
    in >> label >> c.boundary_history_semantics;
    okay = okay && label == "boundary_history_semantics";
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(okay && bool(in),
        "Invalid or missing Eulerian thermal model checkpoint contract");
    c.expected_options.mode = c.expected_ou_convention == "off"
        ? ExpectedIonEnergyMode::Unspecified
        : c.expected_ou_convention == "bounded_nr"
            ? ExpectedIonEnergyMode::NonrelativisticBounded
            : ExpectedIonEnergyMode::RelativisticQuadrature;
    c.Validate();
    return c;
}
void
ElectronThermalModelConfig::ValidateRestart (const ElectronThermalModelConfig& r) const {
    Validate();
    r.Validate();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(relaxation_partner == r.relaxation_partner,
        "Eulerian restart changes the physical electron relaxation partner");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(expected_ou_convention == r.expected_ou_convention &&
        expected_options.absolute_tolerance_joule == r.expected_options.absolute_tolerance_joule &&
        expected_options.relative_component_tolerance == r.expected_options.relative_component_tolerance,
        "Eulerian restart changes the expected OU convention or accuracy budget");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(source_semantics == r.source_semantics &&
        endpoint_absorption_policy == r.endpoint_absorption_policy &&
        boundary_history_semantics == r.boundary_history_semantics,
        "Eulerian restart changes the source or endpoint boundary semantics");
}
std::string
ElectronEnergyMetadata::Encode () const {
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
        << "WarpXElectronEnergy 2\nmode " << ElectronEnergyModeName(mode)
        << "\ncoordinate U_e_cell\nunits J_per_m3\ngamma " << gamma
        << "\nnumber_floor " << number_floor << "\ncorrected_axis "
        << corrected_axis << '\n' << thermal_model.EncodePhysicalContract();
    return out.str();
}
ElectronEnergyMetadata
ElectronEnergyMetadata::Decode (const std::string& text) {
    std::istringstream in(text);
    std::string magic, label, name, coordinate, units;
    int version = 0;
    ElectronEnergyMetadata result;
    in >> magic >> version >> label >> name;
    bool okay =
        magic == "WarpXElectronEnergy" && version == 2 && label == "mode";
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(okay,
        "invalid Eulerian electron energy checkpoint metadata: restart requires complete version 2 metadata; version 1 has no physical model contract");
    result.mode = decode_mode(name);
    in >> label >> coordinate;
    okay = okay && label == "coordinate" && coordinate == "U_e_cell";
    in >> label >> units;
    okay = okay && label == "units" && units == "J_per_m3";
    in >> label >> result.gamma;
    okay = okay && label == "gamma";
    in >> label >> result.number_floor;
    okay = okay && label == "number_floor";
    in >> label >> result.corrected_axis;
    okay = okay && label == "corrected_axis";
    okay = okay && bool(in) && std::isfinite(result.gamma) &&
           result.gamma > 1. && std::isfinite(result.number_floor) &&
           result.number_floor > 0.;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(okay,
        "invalid Eulerian electron energy checkpoint metadata");
    result.thermal_model = ElectronThermalModelConfig::DecodePhysicalContract(in);
    in >> std::ws;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        in.eof(),
        "invalid Eulerian electron energy checkpoint metadata");
    return result;
}
void
ElectronEnergyMetadata::ValidateRestart (
    const ElectronEnergyMetadata& requested) const {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        mode != ElectronEnergyMode::Legacy &&
            requested.mode != ElectronEnergyMode::Legacy &&
            gamma == requested.gamma &&
            number_floor == requested.number_floor &&
            corrected_axis == requested.corrected_axis,
        "Eulerian electron energy restart requires matching U coordinate, "
        "gamma, density floor and axis convention");
    thermal_model.ValidateRestart(requested.thermal_model);
    // Decoupled/coupled switches preserve U when this physical contract matches.
}
bool
SeedCellElectronEnergy (KineticThermalMoments& moments,
                        KineticThermalStateView density,
                        const InitialElectronProfile& profile,
                        amrex::Real gamma, amrex::MultiFab& energy) {
    AMREX_ALWAYS_ASSERT(gamma > 1. && std::isfinite(gamma));
    AMREX_ALWAYS_ASSERT(
        std::isfinite(profile.temperature_kelvin) &&
        profile.temperature_kelvin >= 0. && std::isfinite(profile.gamma) &&
        profile.number_floor > 0. && std::isfinite(profile.number_floor) &&
        (profile.gamma == 1. || (profile.reference_density > 0. &&
                                 std::isfinite(profile.reference_density))));
    density.energy = nullptr;
    density.ion_current = {};
    density.current = {};
    if (!moments.Evaluate(density)) {
        return false;
    }
    auto const& capacity = moments.NodalNumberDensity();
    amrex::MultiFab node_energy(capacity.boxArray(), capacity.DistributionMap(),
                                1, 0);
    auto const p = profile;
    bool const use_ped = density.pedestal && p.include_pedestal;
    int const component = density.charge_component;
    for (amrex::MFIter mfi(node_energy, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
        auto const rho = density.charge.const_array(mfi);
        auto const n = capacity.const_array(mfi);
        amrex::Array4<const amrex::Real> ped;
        if (use_ped) {
            ped = density.pedestal->const_array(mfi);
        }
        auto const u = node_energy.array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                               int k) {
            amrex::Real const seed_n = amrex::max(
                (rho(i, j, k, component) + (use_ped ? ped(i, j, k) : 0.)) /
                    PhysConst::q_e,
                p.number_floor);
            amrex::Real const temperature =
                p.gamma == 1.
                    ? p.temperature_kelvin
                    : p.temperature_kelvin *
                          std::pow(seed_n / p.reference_density, p.gamma - 1.);
            u(i, j, k) =
                n(i, j, k) * PhysConst::kb * temperature / (gamma - 1.);
        });
    }
    if (node_energy.contains_nan() || node_energy.contains_inf()) {
        return false;
    }
    moments.RestrictNativeMoment(node_energy, 0, energy);
    return true;
}
} // namespace warpx::thermal
