/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ThetaImplicitMHD.H"
#include "ThetaImplicitMHD_K.H"

#include "BoundaryConditions/GreensFunctionOpenBC.H"
#include "Circuit/CircuitCoupler.H"
#include "Circuit/CircuitCoupling.H"
#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "Python/callbacks.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_Array4.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuPrint.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace amrex::literals;
using warpx::fields::FieldType;

namespace
{
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
shift_index (int& i, int& j, int& k, const int physical_direction, const int offset) noexcept
{
#if defined(WARPX_DIM_3D)
    if (physical_direction == 0) {
        i += offset;
    } else if (physical_direction == 1) {
        j += offset;
    } else {
        k += offset;
    }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    amrex::ignore_unused(k);
    if (physical_direction == 0) {
        i += offset;
    } else if (physical_direction == 2) {
        j += offset;
    }
#elif defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(j, k);
    if (physical_direction == 2) {
        i += offset;
    }
#else
    amrex::ignore_unused(i, j, k, physical_direction, offset);
#endif
}

amrex::GpuArray<int, 3> field_staggering (const amrex::MultiFab& field)
{
    amrex::GpuArray<int, 3> staggering = {1, 1, 1};
    const amrex::IntVect index_type = field.ixType().toIntVect();
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
        staggering[direction] = index_type[direction];
    }
    return staggering;
}

amrex::GpuArray<int, 3> cell_staggering ()
{
    amrex::GpuArray<int, 3> staggering = {1, 1, 1};
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
        staggering[direction] = 0;
    }
    return staggering;
}

amrex::GpuArray<int, 3> active_physical_directions ()
{
#if defined(WARPX_DIM_3D)
    return {1, 1, 1};
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    return {1, 0, 1};
#elif defined(WARPX_DIM_1D_Z)
    return {0, 0, 1};
#else
    return {0, 0, 0};
#endif
}

amrex::GpuArray<amrex::Real, 3> physical_inverse_cell_size (const amrex::Geometry& geometry)
{
#if defined(WARPX_DIM_3D)
    const auto inverse_cell_size = geometry.InvCellSizeArray();
    return {inverse_cell_size[0], inverse_cell_size[1], inverse_cell_size[2]};
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    const auto inverse_cell_size = geometry.InvCellSizeArray();
    return {inverse_cell_size[0], 0.0_rt, inverse_cell_size[1]};
#elif defined(WARPX_DIM_1D_Z)
    const auto inverse_cell_size = geometry.InvCellSizeArray();
    return {0.0_rt, 0.0_rt, inverse_cell_size[0]};
#else
    amrex::ignore_unused(geometry);
    return {0.0_rt, 0.0_rt, 0.0_rt};
#endif
}
} // namespace

ThetaImplicitMHD::ThetaImplicitMHD () : m_ion_charge_to_mass(PhysConst::q_e / PhysConst::m_p)
{
    const amrex::ParmParse pp("implicit_mhd");

    utils::parser::getWithParser(pp, "reference_mass_density", m_reference_mass_density);
    utils::parser::getWithParser(pp, "reference_magnetic_field", m_reference_magnetic_field);
    const bool has_reference_velocity =
        utils::parser::queryWithParser(pp, "reference_velocity", m_reference_velocity);
    utils::parser::queryWithParser(pp, "reference_ion_pressure", m_reference_ion_pressure);
    utils::parser::queryWithParser(pp, "ion_charge_to_mass", m_ion_charge_to_mass);
    utils::parser::queryWithParser(pp, "gamma_e", m_gamma_e);
    utils::parser::queryWithParser(pp, "gamma_i", m_gamma_i);
    const bool has_mass_density_floor =
        utils::parser::queryWithParser(pp, "mass_density_floor", m_mass_density_floor);
    utils::parser::queryWithParser(pp, "electron_pressure_floor", m_electron_pressure_floor);
    utils::parser::queryWithParser(pp, "ion_pressure_floor", m_ion_pressure_floor);
    const bool has_ion_temperature_floor = utils::parser::queryWithParser(
        pp, "ion_temperature_floor", m_ion_temperature_floor);
    utils::parser::queryWithParser(pp, "electron_temperature_floor",
                                   m_electron_temperature_floor);
    utils::parser::queryWithParser(pp, "vacuum_mass_density", m_vacuum_mass_density);
    utils::parser::queryWithParser(pp, "vacuum_drag_rate", m_vacuum_drag_rate);
    utils::parser::queryWithParser(pp, "halo_pedestal_fraction",
                                   m_halo_pedestal_fraction);
    utils::parser::queryWithParser(pp, "halo_pedestal_drag_rate",
                                   m_halo_pedestal_drag_rate);
    utils::parser::queryWithParser(pp, "halo_pedestal_energy_rate",
                                   m_halo_pedestal_energy_rate);
    utils::parser::queryWithParser(pp, "floor_consistency_rate",
                                   m_floor_consistency_rate);
    utils::parser::queryWithParser(pp, "floor_consistency_width_fraction",
                                   m_floor_consistency_width_fraction);
    pp.query("floor_ledger_file", m_floor_ledger_file);
    // The reference code's density eater (see ApplyDensityEater).
    utils::parser::queryWithParser(pp, "density_eater_rate",
                                   m_density_eater_rate);
    utils::parser::queryWithParser(pp, "density_eater_target_fraction",
                                   m_density_eater_target_fraction);
    utils::parser::queryWithParser(pp, "density_eater_reference_density",
                                   m_density_eater_reference_density);
    utils::parser::queryWithParser(pp, "density_eater_reference_peak_fraction",
                                   m_density_eater_reference_peak_fraction);
    pp.query("density_eater_band", m_density_eater_band);
    utils::parser::queryWithParser(pp, "density_eater_band_cells",
                                   m_density_eater_band_cells);
#if defined(WARPX_DIM_RZ)
    // The reference code's default: the psi > 0 closed-flux gate is always on.
    m_density_eater_flux_sign = 1;
#endif
    utils::parser::queryWithParser(pp, "density_eater_flux_sign",
                                   m_density_eater_flux_sign);
    pp.query("density_eater_ledger_file", m_density_eater_ledger_file);
    utils::parser::queryWithParser(pp, "vacuum_resistivity_diffusivity",
                                   m_vacuum_resistivity_diffusivity);
    // Reference-code-style dynamic reference density (see
    // m_vacuum_reference_peak_fraction).
    utils::parser::queryWithParser(pp, "vacuum_reference_peak_fraction",
                                   m_vacuum_reference_peak_fraction);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_vacuum_reference_peak_fraction >= 0.0_rt &&
            m_vacuum_reference_peak_fraction < 1.0_rt,
        "implicit_mhd.vacuum_reference_peak_fraction must be in [0, 1) "
        "(0 disables the dynamic reference; the reference code uses 0.1)");
    // Reference-code-style Ohm-current Joule quench (see m_joule_ohm_current).
    pp.query("joule_ohm_current", m_joule_ohm_current);
    // Sentinel -1 = "not set": defaulted to the global implicit_evolve.theta
    // in Define(), where theta is parsed.
    utils::parser::queryWithParser(pp, "resistive_theta", m_resistive_theta);
    // Same sentinel convention for the conduction-stage centering.
    utils::parser::queryWithParser(pp, "conduction_theta", m_conduction_theta);
    utils::parser::queryWithParser(pp, "positivity_safety", m_positivity_safety);
    pp.query("external_field_iteration", m_external_field_iteration);
    {
        // Scope of the "externalcoiltheta" circuit hook: "residual"
        // fires it on every residual evaluation (bit-identical default),
        // "newton" only at accepted Newton iterates (see the
        // m_circuit_hook_newton_scope member documentation).
        std::string circuit_hook_scope = "residual";
        pp.query("circuit_hook_scope", circuit_hook_scope);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            circuit_hook_scope == "residual" ||
                circuit_hook_scope == "newton",
            "implicit_mhd.circuit_hook_scope must be 'residual' or "
            "'newton'");
        m_circuit_hook_newton_scope = (circuit_hook_scope == "newton");
    }
    {
        // Which coupler drives the circuit at the hook firing points:
        // "python" (default, bit-identical) executes the
        // externalcoiltheta/externalcoilfinish callbacks; "native" drives
        // the C++ circuit-coupling engine in-process (see the
        // m_circuit_native member documentation).
        std::string circuit_driver = "python";
        pp.query("circuit_driver", circuit_driver);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            circuit_driver == "python" || circuit_driver == "native",
            "implicit_mhd.circuit_driver must be 'python' or 'native'");
        m_circuit_native = (circuit_driver == "native");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_circuit_native || m_external_field_iteration,
            "implicit_mhd.circuit_driver = native requires "
            "implicit_mhd.external_field_iteration = 1 (the native driver "
            "replaces the python hooks at the same firing points)");
    }
    pp.query("fluid_flux", m_fluid_flux);
    utils::parser::queryWithParser(pp, "viscosity", m_viscosity);
    // Reference-code-style wall-row viscosity mask (see m_wall_viscosity_mask);
    // the active-wall requirement is asserted in Define, where the wall
    // mask is built.
    pp.query("wall_viscosity_mask", m_wall_viscosity_mask);
    utils::parser::queryWithParser(pp, "wall_viscosity_mask_width",
                                   m_wall_viscosity_mask_width);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_wall_viscosity_mask_width >= 1,
        "implicit_mhd.wall_viscosity_mask_width must be at least one "
        "fluid cell inside the masked wall contour");
    // Thermal diffusivities: legacy numeric key (bit-identical constant
    // fast path) or the parser signature (rho,Te,Ti,J,t), not both. Same
    // symbol conventions as plasma_resistivity(rho,Te,J,t) plus Ti [K]
    // from the recovered ion pressure (0 outside total_energy); see the
    // header for the face evaluation points.
    {
        const bool has_ion_const = utils::parser::queryWithParser(
            pp, "thermal_diffusivity_ion", m_thermal_diffusivity_ion);
        std::string expression;
        if (pp.query("thermal_diffusivity_ion(rho,Te,Ti,J,t)", expression)) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !has_ion_const,
                "Specify either implicit_mhd.thermal_diffusivity_ion or "
                "thermal_diffusivity_ion(rho,Te,Ti,J,t), not both");
            m_chi_ion_expression = expression;
            m_chi_ion_parser =
                std::make_unique<amrex::Parser>(utils::parser::makeParser(
                    expression, {"rho", "Te", "Ti", "J", "t"}));
            m_chi_ion = m_chi_ion_parser->compile<5>();
            m_chi_ion_is_parser = true;
        }
        const bool has_electron_const = utils::parser::queryWithParser(
            pp, "thermal_diffusivity_electron",
            m_thermal_diffusivity_electron);
        if (pp.query("thermal_diffusivity_electron(rho,Te,Ti,J,t)",
                     expression)) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !has_electron_const,
                "Specify either implicit_mhd.thermal_diffusivity_electron "
                "or thermal_diffusivity_electron(rho,Te,Ti,J,t), not both");
            m_chi_electron_expression = expression;
            m_chi_electron_parser =
                std::make_unique<amrex::Parser>(utils::parser::makeParser(
                    expression, {"rho", "Te", "Ti", "J", "t"}));
            m_chi_electron = m_chi_electron_parser->compile<5>();
            m_chi_electron_is_parser = true;
        }
    }
    utils::parser::queryWithParser(pp, "conduction_flux_limit_factor",
                                   m_conduction_flux_limit_factor);
    {
        std::string conduction_coefficient_state = "theta";
        pp.query("conduction_coefficient_state",
                 conduction_coefficient_state);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            conduction_coefficient_state == "theta" ||
                conduction_coefficient_state == "step_old",
            "implicit_mhd.conduction_coefficient_state must be 'theta' "
            "(coefficients live at the theta stage, the default) or "
            "'step_old' (coefficients frozen at the step-old state, a "
            "per-solve constant -- Newton sees linear diffusion)");
        m_conduction_coefficient_step_old =
            (conduction_coefficient_state == "step_old");
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_conduction_flux_limit_factor >= 0.0_rt,
        "implicit_mhd.conduction_flux_limit_factor cannot be negative "
        "(0 disables the free-streaming conduction limiter)");
    pp.query("thermal_conduction_model", m_thermal_conduction_model);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_thermal_conduction_model == "isotropic" ||
            m_thermal_conduction_model == "braginskii",
        "implicit_mhd.thermal_conduction_model must be isotropic or "
        "braginskii");
    m_conduction_braginskii = (m_thermal_conduction_model == "braginskii");
    utils::parser::queryWithParser(pp, "conduction_coulomb_log",
                                   m_conduction_coulomb_log);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_conduction_coulomb_log > 0.0_rt,
        "implicit_mhd.conduction_coulomb_log must be positive");
    const bool has_conduction_chi_min = utils::parser::queryWithParser(
        pp, "conduction_chi_min", m_conduction_chi_min);
    const bool has_conduction_chi_max = utils::parser::queryWithParser(
        pp, "conduction_chi_max", m_conduction_chi_max);
    // The clamps only enter the Braginskii coefficient evaluation: set
    // with the isotropic model they would be silent no-ops.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_conduction_braginskii ||
            (!has_conduction_chi_min && !has_conduction_chi_max),
        "implicit_mhd.conduction_chi_min/max clamp the Braginskii "
        "chi_par/chi_perp and require "
        "implicit_mhd.thermal_conduction_model = braginskii");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_conduction_chi_min >= 0.0_rt && m_conduction_chi_max >= 0.0_rt,
        "implicit_mhd.conduction_chi_min/max cannot be negative "
        "(0 disables the clamp)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_conduction_chi_max == 0.0_rt ||
            m_conduction_chi_max > m_conduction_chi_min,
        "implicit_mhd.conduction_chi_max must exceed conduction_chi_min");
    // Quasi-shorting cross-field boost (see the header): additive
    // chi_perp keyed to the pseudo-entropy excess above the load
    // envelope. Braginskii only -- the parser diffusivities carry the
    // equivalent physics in their own chi_qs expressions.
    utils::parser::queryWithParser(pp, "conduction_qs_chi",
                                   m_conduction_qs_chi);
    utils::parser::queryWithParser(pp, "conduction_qs_onset",
                                   m_conduction_qs_onset);
    const bool has_conduction_qs_reference_temperature =
        utils::parser::queryWithParser(
            pp, "conduction_qs_reference_temperature",
            m_conduction_qs_reference_temperature);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_conduction_qs_chi >= 0.0_rt,
        "implicit_mhd.conduction_qs_chi cannot be negative "
        "(0 disables the quasi-shorting boost)");
    if (m_conduction_qs_chi > 0.0_rt) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_conduction_braginskii,
            "implicit_mhd.conduction_qs_chi requires "
            "implicit_mhd.thermal_conduction_model = braginskii (with "
            "the parser diffusivities, carry the quasi-shorting term "
            "inside the deck-level thermal_diffusivity_* chi_qs "
            "expressions instead)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_conduction_qs_onset > 1.0_rt,
            "implicit_mhd.conduction_qs_onset must exceed 1: the ramp "
            "is centered ABOVE the load envelope (a ramp centered on "
            "the envelope leaks half its width onto every on-adiabat "
            "cell)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            has_conduction_qs_reference_temperature &&
                m_conduction_qs_reference_temperature > 0.0_rt,
            "implicit_mhd.conduction_qs_chi requires a positive "
            "implicit_mhd.conduction_qs_reference_temperature (the "
            "load-envelope temperature T0, in eV)");
    }
    // Density-keyed halo boost of the Braginskii chi_perp (see
    // m_conduction_halo_boost): the reference code's low-density perp-chi boost,
    // keyed to the same reference density as the field-eta vacuum boost.
    utils::parser::queryWithParser(pp, "conduction_halo_boost",
                                   m_conduction_halo_boost);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_conduction_halo_boost >= 0.0_rt,
        "implicit_mhd.conduction_halo_boost cannot be negative "
        "(0 disables the halo perp-chi boost)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_conduction_halo_boost == 0.0_rt || m_conduction_braginskii,
        "implicit_mhd.conduction_halo_boost boosts the Braginskii "
        "chi_perp and requires implicit_mhd.thermal_conduction_model = "
        "braginskii (with the parser diffusivities, carry the halo "
        "boost inside the deck-level thermal_diffusivity_* expressions "
        "instead)");
    utils::parser::queryWithParser(pp, "pressure_corner_width_fraction",
                                   m_pressure_corner_width_fraction);
    pp.query("r_open_fluid", m_r_open_fluid);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_r_open_fluid == "outflow" || m_r_open_fluid == "reflect" ||
            m_r_open_fluid == "absorb",
        "implicit_mhd.r_open_fluid must be outflow, reflect, or absorb");
    pp.query("z_boundary_fluid", m_z_boundary_fluid);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_z_boundary_fluid == "neumann" ||
            m_z_boundary_fluid == "wall_temperature" ||
            m_z_boundary_fluid == "outflow",
        "implicit_mhd.z_boundary_fluid must be neumann, wall_temperature, "
        "or outflow");
    pp.query("z_lo_boundary_fluid", m_z_lo_boundary_fluid);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_z_lo_boundary_fluid.empty() || m_z_lo_boundary_fluid == "symmetry",
        "implicit_mhd.z_lo_boundary_fluid must be empty (inherit "
        "z_boundary_fluid) or symmetry");
    const bool has_z_wall_temperature = utils::parser::queryWithParser(
        pp, "z_wall_temperature", m_z_wall_temperature);
    if (m_z_boundary_fluid == "wall_temperature") {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            has_z_wall_temperature && m_z_wall_temperature > 0.0_rt,
            "implicit_mhd.z_boundary_fluid = wall_temperature requires a "
            "positive implicit_mhd.z_wall_temperature (in eV)");
    } else {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !has_z_wall_temperature,
            "implicit_mhd.z_wall_temperature requires "
            "implicit_mhd.z_boundary_fluid = wall_temperature");
    }
    utils::parser::queryWithParser(pp, "absorb_ledger_interval",
                                   m_absorb_ledger_interval);
    pp.query("wall_ledger_file", m_wall_ledger_file);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_absorb_ledger_interval >= 0,
        "implicit_mhd.absorb_ledger_interval cannot be negative");
    pp.query("absorb_ledger_file", m_absorb_ledger_file);
    pp.query("hllc_signal_closure", m_hllc_signal_closure);
    utils::parser::queryWithParser(pp, "hllc_contact_blend", m_hllc_contact_blend);
    pp.query("ion_closure", m_ion_closure);
    utils::parser::queryWithParser(pp, "dual_energy_internal_cutoff",
                                   m_dual_energy_internal_cutoff);
    utils::parser::queryWithParser(pp, "dual_energy_sync_threshold",
                                   m_dual_energy_sync_threshold);
    pp.query("evolve_ion_fluid", m_evolve_ion_fluid);
    pp.query("include_joule_heating", m_include_joule_heating);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_reference_mass_density > 0.0_rt,
                                     "implicit_mhd.reference_mass_density must be positive");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_reference_magnetic_field > 0.0_rt,
                                     "implicit_mhd.reference_magnetic_field must be positive");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_ion_charge_to_mass > 0.0_rt,
                                     "implicit_mhd.ion_charge_to_mass must be positive");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_gamma_e > 1.0_rt,
                                     "implicit_mhd.gamma_e must be greater than one");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_gamma_i > 0.0_rt, "implicit_mhd.gamma_i must be positive");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_reference_ion_pressure >= 0.0_rt,
                                     "implicit_mhd.reference_ion_pressure cannot be negative");

    if (!has_reference_velocity) {
        m_reference_velocity =
            m_reference_magnetic_field / std::sqrt(PhysConst::mu0 * m_reference_mass_density);
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_reference_velocity > 0.0_rt,
                                     "implicit_mhd.reference_velocity must be positive");
    if (!has_mass_density_floor) {
        m_mass_density_floor = 1.0e-12_rt * m_reference_mass_density;
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_mass_density_floor > 0.0_rt,
                                     "implicit_mhd.mass_density_floor must be positive");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_electron_pressure_floor >= 0.0_rt,
                                     "implicit_mhd.electron_pressure_floor cannot be negative");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_fluid_flux == "centered" || m_fluid_flux == "rusanov" ||
            m_fluid_flux == "hllc" || m_fluid_flux == "hlld" ||
            m_fluid_flux == "central",
        "implicit_mhd.fluid_flux must be centered, rusanov, hllc, hlld, "
        "or central");
    m_use_hlld = (m_fluid_flux == "hlld");
    m_use_central = (m_fluid_flux == "central");
    m_use_recast = m_use_hlld || m_use_central;
    // hlld is NOT a production flux (central + viscosity is): the hlld
    // path stays only as kernel regression coverage and must be opted
    // into explicitly, so no production deck can reach it silently.
    pp.query("allow_hlld", m_allow_hlld);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_use_hlld || m_allow_hlld,
        "implicit_mhd.fluid_flux = hlld is not a production flux (use "
        "central, with implicit_mhd.viscosity); it remains available for "
        "kernel regression tests only -- set implicit_mhd.allow_hlld = "
        "true to opt in");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_z_boundary_fluid == "neumann" || m_use_recast,
        "implicit_mhd.z_boundary_fluid = wall_temperature/outflow requires "
        "the conservative-form recast (implicit_mhd.fluid_flux = hlld or "
        "central)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_z_lo_boundary_fluid.empty() || m_use_recast,
        "implicit_mhd.z_lo_boundary_fluid = symmetry requires the "
        "conservative-form recast (implicit_mhd.fluid_flux = hlld or "
        "central)");
#if !defined(WARPX_DIM_1D_Z) && !defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_use_recast,
        "implicit_mhd.fluid_flux = hlld/central (the conservative-form "
        "recast) currently supports 1D Cartesian and cylindrical RZ "
        "geometry only");
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_viscosity >= 0.0_rt,
        "implicit_mhd.viscosity cannot be negative");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_viscosity == 0.0_rt || m_use_recast,
        "implicit_mhd.viscosity requires the conservative-form recast "
        "(implicit_mhd.fluid_flux = hlld or central): the viscous face "
        "flux is only wired into the recast face-flux registers");
    // The viscous work is paired into the conservative E_i channel; the
    // cgl internal-energy blocks track no kinetic energy, so the heating
    // the momentum diffusion implies would silently vanish there.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_viscosity == 0.0_rt || m_ion_closure != "cgl",
        "implicit_mhd.viscosity is not supported with "
        "implicit_mhd.ion_closure = cgl (the viscous stress work is only "
        "paired into the total_energy ion-energy channel)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_thermal_diffusivity_ion >= 0.0_rt &&
            m_thermal_diffusivity_electron >= 0.0_rt,
        "implicit_mhd.thermal_diffusivity_ion/electron cannot be negative");
    // Braginskii conduction supplies its own chi_par/chi_perp from the
    // face state; a simultaneous constant/parser diffusivity would be a
    // silent contradiction. The channels it activates are the electron
    // channel always and the ion channel under total_energy (the only
    // closure whose ion-energy register consumes the flux).
    if (m_conduction_braginskii) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_chi_ion_is_parser && !m_chi_electron_is_parser &&
                m_thermal_diffusivity_ion == 0.0_rt &&
                m_thermal_diffusivity_electron == 0.0_rt,
            "implicit_mhd.thermal_conduction_model = braginskii computes "
            "its own chi_par/chi_perp: do not also set "
            "implicit_mhd.thermal_diffusivity_ion/electron");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_use_recast,
            "implicit_mhd.thermal_conduction_model = braginskii requires "
            "the conservative-form recast (implicit_mhd.fluid_flux = hlld "
            "or central): the conductive face flux is only wired into the "
            "recast face-flux registers");
        // Strictly positive pressure floors keep the face temperatures
        // (hence the collision times) strictly positive and anchor the
        // smooth bhat-direction floor at the field-energy scale
        // mu0 p_floor, so the tensor stays C-infinity for the JFNK
        // probes (ion_pressure_floor > 0 is enforced by total_energy).
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_electron_pressure_floor > 0.0_rt,
            "implicit_mhd.thermal_conduction_model = braginskii requires "
            "a positive implicit_mhd.electron_pressure_floor (it keeps "
            "the face Te strictly positive for the collision times and "
            "anchors the smooth bhat regularization)");
    }
    const bool has_ion_conduction =
        m_chi_ion_is_parser || m_thermal_diffusivity_ion > 0.0_rt ||
        (m_conduction_braginskii && (m_ion_closure == "total_energy" ||
                                     m_ion_closure == "dual_energy"));
    const bool has_electron_conduction =
        m_chi_electron_is_parser ||
        m_thermal_diffusivity_electron > 0.0_rt || m_conduction_braginskii;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (!has_ion_conduction && !has_electron_conduction) || m_use_recast,
        "implicit_mhd.thermal_diffusivity_ion/electron require the "
        "conservative-form recast (implicit_mhd.fluid_flux = hlld or "
        "central): the conductive face flux is only wired into the recast "
        "face-flux registers");
    // The conductive flux enters the total_energy ion-energy channel;
    // under the barotropic and cgl closures that register is never
    // consumed, so a positive chi_i would be a silent no-op (and the Ti
    // parser symbol needs the recovered ion pressure).
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !has_ion_conduction || m_ion_closure == "total_energy" ||
            m_ion_closure == "dual_energy",
        "implicit_mhd.thermal_diffusivity_ion requires "
        "implicit_mhd.ion_closure = total_energy or dual_energy (the ion "
        "conductive flux is only wired into the total/internal ion-energy "
        "channels)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_conduction_flux_limit_factor == 0.0_rt ||
            (has_ion_conduction || has_electron_conduction),
        "implicit_mhd.conduction_flux_limit_factor requires a nonzero "
        "thermal diffusivity (nothing to limit)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_pressure_corner_width_fraction >= 0.0_rt,
        "implicit_mhd.pressure_corner_width_fraction cannot be negative");
    if (m_use_central) {
        // Central alone is nonlinearly unstable: the flux carries no
        // Riemann dissipation, so explicit viscosity must supply it.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_viscosity > 0.0_rt,
            "implicit_mhd.fluid_flux = central requires a positive "
            "implicit_mhd.viscosity: the central flux has no Riemann "
            "dissipation, so the explicit viscous face flux must provide "
            "the nonlinear stabilization");
    }
    utils::parser::queryWithParser(pp, "hlld_kappa_signal",
                                   m_hlld_kappa_signal);
    utils::parser::queryWithParser(pp, "hlld_kappa_contact",
                                   m_hlld_kappa_contact);
    utils::parser::queryWithParser(pp, "hlld_kappa_bn", m_hlld_kappa_bn);
    utils::parser::queryWithParser(pp, "hlld_kappa_denominator",
                                   m_hlld_kappa_denominator);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hlld_kappa_signal >= 0.0_rt && m_hlld_kappa_contact >= 0.0_rt &&
            m_hlld_kappa_bn >= 0.0_rt && m_hlld_kappa_denominator >= 0.0_rt,
        "implicit_mhd.hlld_kappa_* smoothing widths cannot be negative");
    pp.query("hlld_fan_closure", m_hlld_fan_closure);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hlld_fan_closure == "consistent" ||
            m_hlld_fan_closure == "barotropic",
        "implicit_mhd.hlld_fan_closure must be consistent or barotropic");
    pp.query("hlld_ion_energy_flux", m_hlld_ion_energy_flux);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hlld_ion_energy_flux == "star" || m_hlld_ion_energy_flux == "llf",
        "implicit_mhd.hlld_ion_energy_flux must be star or llf");
    pp.query("hlld_all_speed", m_hlld_all_speed);
    if (m_hlld_fan_closure == "barotropic") {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_use_hlld && m_ion_closure == "total_energy",
            "implicit_mhd.hlld_fan_closure = barotropic requires "
            "implicit_mhd.fluid_flux = hlld and implicit_mhd.ion_closure = "
            "total_energy (it decouples the wave-fan structure from the "
            "E_i unknown; the barotropic closure already has an "
            "E_i-independent fan)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_reference_ion_pressure > 0.0_rt,
            "implicit_mhd.hlld_fan_closure = barotropic requires a "
            "positive implicit_mhd.reference_ion_pressure (it sets the "
            "polytropic fan ion pressure)");
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_positivity_safety > 0.0_rt && m_positivity_safety < 1.0_rt,
        "implicit_mhd.positivity_safety must be greater than zero and less than one");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_halo_pedestal_fraction >= 0.0_rt && m_halo_pedestal_fraction < 1.0_rt,
        "implicit_mhd.halo_pedestal_fraction must be in [0, 1)");
    if (m_halo_pedestal_fraction > 0.0_rt) {
        // The pedestal band is held by the donor drain gates, which the
        // conservative-form recast applies to EVERY face flux -- the
        // gate machinery itself imposes the C-infinity smoothed donor
        // selection post-flux, so the central flux is gated exactly
        // like the Riemann fluxes (the historical Riemann-only assert
        // predated the shared recast kernel; the central invariance is
        // gated by test_1d_theta_implicit_mhd_halo_pedestal_central).
        // Only the legacy non-recast centered flux lacks the gates.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_fluid_flux != "centered",
            "implicit_mhd.halo_pedestal_fraction requires a recast fluid "
            "flux (central, rusanov, hllc, or hlld): the pedestal band is "
            "held by the recast face-flux donor drain gates, which the "
            "legacy centered flux does not have");
        // "Well above": the pedestal must displace the halo operating
        // point off the Newton admissibility bound, so the pedestal base
        // (its value when the reference density is the peak) must exceed
        // the positivity floor.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_halo_pedestal_fraction * m_reference_mass_density >
                m_mass_density_floor,
            "implicit_mhd.halo_pedestal_fraction * reference_mass_density "
            "must exceed implicit_mhd.mass_density_floor (the pedestal "
            "must sit above the positivity guard so the halo band rides "
            "an interior point of the admissible set)");
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_halo_pedestal_drag_rate >= 0.0_rt,
        "implicit_mhd.halo_pedestal_drag_rate cannot be negative");
    // The drag weight is keyed to the density pedestal (identically zero
    // without one), so an explicit positive rate without a pedestal is a
    // configuration error.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_halo_pedestal_drag_rate == 0.0_rt ||
            m_halo_pedestal_fraction > 0.0_rt,
        "implicit_mhd.halo_pedestal_drag_rate requires a positive "
        "implicit_mhd.halo_pedestal_fraction (the drag relaxes the "
        "pedestal band, which does not exist without a pedestal)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_halo_pedestal_energy_rate >= 0.0_rt,
        "implicit_mhd.halo_pedestal_energy_rate cannot be negative");
    // Same mask as the drag: identically zero without a pedestal, so an
    // explicit positive rate without one is a configuration error.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_halo_pedestal_energy_rate == 0.0_rt ||
            m_halo_pedestal_fraction > 0.0_rt,
        "implicit_mhd.halo_pedestal_energy_rate requires a positive "
        "implicit_mhd.halo_pedestal_fraction (the relaxation drains the "
        "pedestal band's ion energy, which does not exist without a "
        "pedestal)");
    // The barotropic closure evolves no ion energy block: nothing to
    // relax, so an explicit positive rate is a configuration error.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_halo_pedestal_energy_rate == 0.0_rt ||
            m_ion_closure == "total_energy" ||
            m_ion_closure == "dual_energy" || m_ion_closure == "cgl",
        "implicit_mhd.halo_pedestal_energy_rate requires "
        "implicit_mhd.ion_closure = total_energy, dual_energy, or cgl "
        "(the barotropic closure evolves no ion energy block to relax)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_floor_consistency_rate >= 0.0_rt,
        "implicit_mhd.floor_consistency_rate cannot be negative");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_floor_consistency_width_fraction > 0.0_rt &&
            m_floor_consistency_width_fraction <= 0.5_rt,
        "implicit_mhd.floor_consistency_width_fraction must be in "
        "(0, 0.5]: the bound-riding supply capacity is rate_eff * "
        "width/2 and the exact-zero guarantee moves to "
        "(1 + 2 f) x bound");
    // The ledger is the source's conservation instrument: a file without
    // the source is a configuration error, not a silent no-op.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_floor_ledger_file.empty() || m_floor_consistency_rate > 0.0_rt,
        "implicit_mhd.floor_ledger_file requires a positive "
        "implicit_mhd.floor_consistency_rate (the ledger books the "
        "floor-consistency supply, which does not exist without it)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_density_eater_rate >= 0.0_rt && m_density_eater_rate <= 1.0_rt,
        "implicit_mhd.density_eater_rate must be in [0, 1] (the per-step "
        "fraction of the excess above target removed; the reference code uses 0.2)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_density_eater_target_fraction > 0.0_rt,
        "implicit_mhd.density_eater_target_fraction must be positive "
        "(the reference code uses 0.01: target = en0/100)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_density_eater_reference_peak_fraction >= 0.0_rt &&
            m_density_eater_reference_peak_fraction < 1.0_rt,
        "implicit_mhd.density_eater_reference_peak_fraction must be in "
        "[0, 1) (0 freezes the reference at the base; the reference code uses 0.1)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_density_eater_band.empty() || m_density_eater_band == "z_lo" ||
            m_density_eater_band == "z_center",
        "implicit_mhd.density_eater_band must be 'z_lo' or 'z_center' "
        "(unset = z_lo when the z_lo pmc mirror is active, else "
        "z_center: the reference code's sym_bc dispatch)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_density_eater_flux_sign >= -1 && m_density_eater_flux_sign <= 1,
        "implicit_mhd.density_eater_flux_sign must be -1, 0, or 1");
#if !defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_density_eater_flux_sign == 0,
        "implicit_mhd.density_eater_flux_sign requires RZ (the "
        "closed-flux gate is a poloidal-flux integral; there is no "
        "poloidal flux in 1D)");
#endif
    if (m_density_eater_rate > 0.0_rt) {
        // The target must stay an admissible density: the eater pulls
        // cells DOWN onto it, so a target below the positivity floor
        // would break the end-of-step floor assertions. The dynamic
        // Reference only RAISES the target, so checking the base
        // suffices.
        const amrex::Real eater_reference_base =
            m_density_eater_reference_density > 0.0_rt
                ? m_density_eater_reference_density
                : m_reference_mass_density;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_density_eater_target_fraction * eater_reference_base >=
                m_mass_density_floor,
            "implicit_mhd.density_eater_target_fraction x the eater "
            "reference density must be at or above "
            "implicit_mhd.mass_density_floor (the eater relaxes cells "
            "onto the target; the reference code keeps it at 10x the floor)");
    }
    // Same contract as the floor ledger: a ledger file without the
    // mechanism is a configuration error, not a silent no-op.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_density_eater_ledger_file.empty() || m_density_eater_rate > 0.0_rt,
        "implicit_mhd.density_eater_ledger_file requires a positive "
        "implicit_mhd.density_eater_rate (the ledger books the eater "
        "removal, which does not exist without it)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_vacuum_resistivity_diffusivity >= 0.0_rt,
        "implicit_mhd.vacuum_resistivity_diffusivity cannot be negative");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_ion_closure == "barotropic" || m_ion_closure == "total_energy" ||
            m_ion_closure == "dual_energy" || m_ion_closure == "cgl",
        "implicit_mhd.ion_closure must be barotropic, total_energy, "
        "dual_energy, or cgl");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_ion_pressure_floor >= 0.0_rt,
                                     "implicit_mhd.ion_pressure_floor cannot be negative");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_ion_temperature_floor >= 0.0_rt,
        "implicit_mhd.ion_temperature_floor cannot be negative");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_electron_temperature_floor >= 0.0_rt,
        "implicit_mhd.electron_temperature_floor cannot be negative");
    if (m_ion_closure != "total_energy" && m_ion_closure != "dual_energy" &&
        m_ion_closure != "cgl") {
        // The ion temperature floor bounds the evolved ion energy blocks;
        // the barotropic closure ties the ion temperature to the density
        // and evolves no such block, so the room-temperature default is
        // inert and an explicit request is a configuration error.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !has_ion_temperature_floor || m_ion_temperature_floor == 0.0_rt,
            "implicit_mhd.ion_temperature_floor requires an ion closure "
            "that evolves an ion energy block (implicit_mhd.ion_closure = "
            "total_energy, dual_energy, or cgl)");
        m_ion_temperature_floor = 0.0_rt;
    }
    utils::parser::queryWithParser(pp, "cgl_relaxation_scale",
                                   m_cgl_relaxation_scale);
    utils::parser::queryWithParser(pp, "cgl_coulomb_log", m_cgl_coulomb_log);
    utils::parser::queryWithParser(pp, "cgl_instability_scale",
                                   m_cgl_instability_scale);
    utils::parser::queryWithParser(pp, "cgl_instability_width",
                                   m_cgl_instability_width);
    utils::parser::queryWithParser(pp, "cgl_null_scale", m_cgl_null_scale);
    utils::parser::queryWithParser(pp, "electron_ion_equilibration",
                                   m_electron_ion_equilibration);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_electron_ion_equilibration >= 0.0_rt,
        "implicit_mhd.electron_ion_equilibration cannot be negative");
    // The exchange relaxes the electron energy against an evolved ion
    // energy block; the barotropic closure ties the ion temperature to
    // the density and has no block to receive the pair term.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_electron_ion_equilibration == 0.0_rt ||
            m_ion_closure == "total_energy" ||
            m_ion_closure == "dual_energy" || m_ion_closure == "cgl",
        "implicit_mhd.electron_ion_equilibration requires "
        "implicit_mhd.ion_closure = total_energy, dual_energy, or cgl "
        "(the exchange deposits into an evolved ion energy block)");
    pp.query("z_outflow_no_reflux", m_z_outflow_no_reflux);
    // The outflow mode is the smooth superset of the no-reflux clamp
    // (and wall_temperature keeps the passive momentum copy by design),
    // so stacking the hard clamp on top is a configuration error.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_z_outflow_no_reflux || m_z_boundary_fluid == "neumann",
        "implicit_mhd.z_outflow_no_reflux requires "
        "implicit_mhd.z_boundary_fluid = neumann (outflow is its "
        "C-infinity superset; wall_temperature keeps the passive "
        "momentum copy)");
    if (m_ion_closure == "cgl") {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_cgl_instability_scale >= 0.0_rt &&
                m_cgl_instability_width > 0.0_rt,
            "implicit_mhd.cgl_instability_scale cannot be negative and "
            "cgl_instability_width must be positive");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_cgl_null_scale >= 0.0_rt,
            "implicit_mhd.cgl_null_scale cannot be negative");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_use_hlld,
            "implicit_mhd.ion_closure = cgl requires implicit_mhd.fluid_flux "
            "= hlld (the CGL channels, work terms, and deviation stress are "
            "only wired into the conservative-form face-flux path)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_evolve_ion_fluid,
            "implicit_mhd.ion_closure = cgl requires "
            "implicit_mhd.evolve_ion_fluid = true");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_ion_pressure_floor > 0.0_rt,
            "implicit_mhd.ion_closure = cgl requires a positive "
            "implicit_mhd.ion_pressure_floor (it sets the U_par/U_perp "
            "positivity floors and anchors the smooth regularizations of "
            "the deviation stress and the isotropization rate)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_cgl_relaxation_scale >= 0.0_rt,
            "implicit_mhd.cgl_relaxation_scale cannot be negative");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_cgl_coulomb_log > 0.0_rt,
            "implicit_mhd.cgl_coulomb_log must be positive");
    }
    if (m_ion_closure == "total_energy") {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_evolve_ion_fluid,
            "implicit_mhd.ion_closure = total_energy requires "
            "implicit_mhd.evolve_ion_fluid = true");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_ion_pressure_floor > 0.0_rt,
            "implicit_mhd.ion_closure = total_energy requires a positive "
            "implicit_mhd.ion_pressure_floor (it sets the internal-energy "
            "floor of the recovered ion pressure)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_gamma_i > 1.0_rt,
            "implicit_mhd.ion_closure = total_energy requires "
            "implicit_mhd.gamma_i greater than one");
    }
    if (m_ion_closure == "dual_energy") {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_use_recast,
            "implicit_mhd.ion_closure = dual_energy requires "
            "implicit_mhd.fluid_flux = hlld or central (the U_i advection "
            "channel, the blended-pressure loaders, and the pointwise PdV/"
            "heating sources are only wired into the conservative-form "
            "face-flux path)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_evolve_ion_fluid,
            "implicit_mhd.ion_closure = dual_energy requires "
            "implicit_mhd.evolve_ion_fluid = true");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_ion_pressure_floor > 0.0_rt,
            "implicit_mhd.ion_closure = dual_energy requires a positive "
            "implicit_mhd.ion_pressure_floor (it sets the internal-energy "
            "floors of both the recovered and the auxiliary ion pressure, "
            "and guards the kinetic-fraction division)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_gamma_i > 1.0_rt,
            "implicit_mhd.ion_closure = dual_energy requires "
            "implicit_mhd.gamma_i greater than one");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_dual_energy_internal_cutoff >= 0.0_rt &&
                m_dual_energy_internal_cutoff < 1.0_rt,
            "implicit_mhd.dual_energy_internal_cutoff must be in [0, 1) "
            "(0 disables the low-internal fk cutoff)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_dual_energy_sync_threshold > 0.0_rt &&
                m_dual_energy_sync_threshold <= 1.0_rt,
            "implicit_mhd.dual_energy_sync_threshold must be in (0, 1]");
    } else {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_dual_energy_internal_cutoff == 0.0_rt,
            "implicit_mhd.dual_energy_internal_cutoff requires "
            "implicit_mhd.ion_closure = dual_energy");
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hllc_signal_closure == "consistent" ||
            m_hllc_signal_closure == "barotropic",
        "implicit_mhd.hllc_signal_closure must be consistent or barotropic");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hllc_contact_blend >= 0.0_rt,
        "implicit_mhd.hllc_contact_blend cannot be negative");
    if (m_hllc_signal_closure == "barotropic") {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_fluid_flux == "hllc" && m_ion_closure == "total_energy",
            "implicit_mhd.hllc_signal_closure = barotropic requires "
            "implicit_mhd.fluid_flux = hllc and implicit_mhd.ion_closure = "
            "total_energy (it decouples the HLLC wave-speed estimates from "
            "the E_i unknown; the barotropic closure already has "
            "E_i-independent wave speeds)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_reference_ion_pressure > 0.0_rt,
            "implicit_mhd.hllc_signal_closure = barotropic requires a "
            "positive implicit_mhd.reference_ion_pressure (it sets the "
            "polytropic signal-speed ion pressure)");
    }
    if (m_hllc_contact_blend > 0.0_rt) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_fluid_flux == "hllc",
            "implicit_mhd.hllc_contact_blend requires "
            "implicit_mhd.fluid_flux = hllc");
    }

    std::string mass_density_expression;
    std::string electron_pressure_expression;
    std::string ion_pressure_expression = "0.0";
    std::string velocity_x_expression = "0.0";
    std::string velocity_y_expression = "0.0";
    std::string velocity_z_expression = "0.0";
    utils::parser::Store_parserString(pp, "mass_density(x,y,z)", mass_density_expression);
    utils::parser::Store_parserString(pp, "electron_pressure(x,y,z)", electron_pressure_expression);
    if (m_ion_closure == "total_energy" || m_ion_closure == "dual_energy" ||
        m_ion_closure == "cgl") {
        utils::parser::Store_parserString(pp, "ion_pressure(x,y,z)", ion_pressure_expression);
    } else {
        utils::parser::Query_parserString(pp, "ion_pressure(x,y,z)", ion_pressure_expression);
    }
    // cgl: optional initial anisotropy A = p_perp/p_par with the
    // effective pressure pinned to ion_pressure, i.e.
    // p_par = 3 p_i/(1 + 2A) and p_perp = A p_par.
    std::string ion_anisotropy_expression = "1.0";
    utils::parser::Query_parserString(pp, "ion_pressure_anisotropy(x,y,z)",
                                      ion_anisotropy_expression);
    utils::parser::Query_parserString(pp, "velocity_x(x,y,z)", velocity_x_expression);
    utils::parser::Query_parserString(pp, "velocity_y(x,y,z)", velocity_y_expression);
    utils::parser::Query_parserString(pp, "velocity_z(x,y,z)", velocity_z_expression);

    const amrex::Vector<std::string> variables = {"x", "y", "z"};
    m_mass_density_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(mass_density_expression, variables));
    m_velocity_x_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(velocity_x_expression, variables));
    m_velocity_y_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(velocity_y_expression, variables));
    m_velocity_z_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(velocity_z_expression, variables));
    m_electron_pressure_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(electron_pressure_expression, variables));
    m_ion_pressure_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(ion_pressure_expression, variables));
    m_ion_anisotropy_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(ion_anisotropy_expression, variables));
    m_mass_density = m_mass_density_parser->compile<3>();
    m_velocity_x = m_velocity_x_parser->compile<3>();
    m_velocity_y = m_velocity_y_parser->compile<3>();
    m_velocity_z = m_velocity_z_parser->compile<3>();
    m_electron_pressure = m_electron_pressure_parser->compile<3>();
    m_ion_pressure = m_ion_pressure_parser->compile<3>();
    m_ion_anisotropy = m_ion_anisotropy_parser->compile<3>();
}

void ThetaImplicitMHD::AllocateLevelMFs (ablastr::fields::MultiFabRegister& fields, const int lev,
                                         const amrex::BoxArray& ba,
                                         const amrex::DistributionMapping& dm) const
{
    const amrex::IntVect guard_cells(2);
    constexpr bool remake = true;
    constexpr bool redistribute_on_remake = true;
    constexpr bool checkpoint_restart = true;

    fields.alloc_init(MassDensityName, lev, ba, dm, 1, guard_cells, 0.0_rt, remake,
                      redistribute_on_remake, checkpoint_restart);
    fields.alloc_init(MomentumDensityName, lev, ba, dm, 3, guard_cells, 0.0_rt, remake,
                      redistribute_on_remake, checkpoint_restart);
    fields.alloc_init(ElectronEnergyName, lev, ba, dm, 1, guard_cells, 0.0_rt, remake,
                      redistribute_on_remake, checkpoint_restart);
    // Always allocated (zeroed and unused with the barotropic closure) so
    // flux-kernel signatures and diagnostics are uniform across closures.
    fields.alloc_init(IonEnergyName, lev, ba, dm, 1, guard_cells, 0.0_rt, remake,
                      redistribute_on_remake, checkpoint_restart);
    // CGL closure blocks (zeroed and unused with the other closures).
    fields.alloc_init(IonParallelEnergyName, lev, ba, dm, 1, guard_cells, 0.0_rt, remake,
                      redistribute_on_remake, checkpoint_restart);
    fields.alloc_init(IonPerpEnergyName, lev, ba, dm, 1, guard_cells, 0.0_rt, remake,
                      redistribute_on_remake, checkpoint_restart);
    // Dual-energy closure block (zeroed and unused with the other
    // closures); checkpointed like the other fluid blocks so U_i
    // round-trips through restarts.
    fields.alloc_init(IonInternalEnergyName, lev, ba, dm, 1, guard_cells, 0.0_rt, remake,
                      redistribute_on_remake, checkpoint_restart);
    // Single-component momentum views for field diagnostics (a
    // 3-component register block cannot pass through fields_to_plot);
    // refreshed by PublishMomentumComponents, never read by the solver.
    fields.alloc_init(MomentumDiag0Name, lev, ba, dm, 1, amrex::IntVect(0), 0.0_rt);
    fields.alloc_init(MomentumDiag1Name, lev, ba, dm, 1, amrex::IntVect(0), 0.0_rt);
    fields.alloc_init(MomentumDiag2Name, lev, ba, dm, 1, amrex::IntVect(0), 0.0_rt);

    fields.alloc_init(TotalCurrentCCName, lev, ba, dm, 3, guard_cells, 0.0_rt);
    fields.alloc_init(MagneticFieldCCName, lev, ba, dm, 3, guard_cells, 0.0_rt);
    // Cell-centered stage-E gather of the Ohm-current Joule quench
    // (joule_ohm_current). Allocated unconditionally (the register remake
    // machinery wants a static field list); filled only when the quench
    // is active, and only read at valid cells (no ghosts).
    fields.alloc_init(OhmElectricFieldCCName, lev, ba, dm, 3, amrex::IntVect(0),
                      0.0_rt);
    // Cell-centered T_e scratch of the temperature-primary nodal closure
    // fill (FillFluidSources); ghosts computed in place from the ghosted
    // moments, never FillBoundary'd.
    fields.alloc_init(ElectronTemperatureCCName, lev, ba, dm, 1, guard_cells,
                      0.0_rt);
    fields.alloc_init(FieldResistivityCCName, lev, ba, dm, 1, amrex::IntVect(0), 0.0_rt);
#if defined(WARPX_DIM_1D_Z)
    // The transverse E components share the z-nodal staggering (and the
    // same interpolated density), so one register serves both; the
    // cell-centered Ez never enters the 1D resistive curl-curl.
    fields.alloc_init(FieldResistivityE0Name, lev,
                      amrex::convert(ba, amrex::IntVect(1)), dm, 1,
                      amrex::IntVect(0), 0.0_rt);
    fields.alloc_init(HyperResistivityE0Name, lev,
                      amrex::convert(ba, amrex::IntVect(1)), dm, 1,
                      amrex::IntVect(0), 0.0_rt);
    fields.alloc_init(HallCoefficientE0Name, lev,
                      amrex::convert(ba, amrex::IntVect(1)), dm, 3,
                      amrex::IntVect(0), 0.0_rt);
    fields.alloc_init(InertiaCoefficientE0Name, lev,
                      amrex::convert(ba, amrex::IntVect(1)), dm, 1,
                      amrex::IntVect(0), 0.0_rt);
#elif defined(WARPX_DIM_RZ)
    fields.alloc_init(FieldResistivityE0Name, lev,
                      amrex::convert(ba, amrex::IntVect(0, 1)), dm, 1,
                      amrex::IntVect(0), 0.0_rt);
    fields.alloc_init(FieldResistivityE1Name, lev,
                      amrex::convert(ba, amrex::IntVect(1, 1)), dm, 1,
                      amrex::IntVect(0), 0.0_rt);
    fields.alloc_init(FieldResistivityE2Name, lev,
                      amrex::convert(ba, amrex::IntVect(1, 0)), dm, 1,
                      amrex::IntVect(0), 0.0_rt);
    fields.alloc_init(HyperResistivityE0Name, lev,
                      amrex::convert(ba, amrex::IntVect(0, 1)), dm, 1,
                      amrex::IntVect(0), 0.0_rt);
    fields.alloc_init(HyperResistivityE1Name, lev,
                      amrex::convert(ba, amrex::IntVect(1, 1)), dm, 1,
                      amrex::IntVect(0), 0.0_rt);
    fields.alloc_init(HyperResistivityE2Name, lev,
                      amrex::convert(ba, amrex::IntVect(1, 0)), dm, 1,
                      amrex::IntVect(0), 0.0_rt);
    fields.alloc_init(HallCoefficientE0Name, lev,
                      amrex::convert(ba, amrex::IntVect(0, 1)), dm, 3,
                      amrex::IntVect(0), 0.0_rt);
    fields.alloc_init(HallCoefficientE1Name, lev,
                      amrex::convert(ba, amrex::IntVect(1, 1)), dm, 3,
                      amrex::IntVect(0), 0.0_rt);
    fields.alloc_init(HallCoefficientE2Name, lev,
                      amrex::convert(ba, amrex::IntVect(1, 0)), dm, 3,
                      amrex::IntVect(0), 0.0_rt);
    fields.alloc_init(InertiaCoefficientE0Name, lev,
                      amrex::convert(ba, amrex::IntVect(0, 1)), dm, 1,
                      amrex::IntVect(0), 0.0_rt);
    fields.alloc_init(InertiaCoefficientE1Name, lev,
                      amrex::convert(ba, amrex::IntVect(1, 1)), dm, 1,
                      amrex::IntVect(0), 0.0_rt);
    fields.alloc_init(InertiaCoefficientE2Name, lev,
                      amrex::convert(ba, amrex::IntVect(1, 0)), dm, 1,
                      amrex::IntVect(0), 0.0_rt);
#endif

#if defined(WARPX_DIM_1D_Z)
    // hlld face-flux register (z-faces = z-nodal in 1D). Allocated
    // unconditionally: the register remake machinery wants a static field
    // list, and the cost of the unused MultiFabs is trivial.
    fields.alloc_init(FaceFluxZName, lev,
                      amrex::convert(ba, amrex::IntVect::TheNodeVector()), dm,
                      FaceFluxComponent::count, amrex::IntVect(0), 0.0_rt);
#elif defined(WARPX_DIM_RZ)
    // hlld face-flux registers: r-faces are (r-nodal, z-cc) -- the Br/Ez
    // staggering -- and z-faces are (r-cc, z-nodal) -- the Bz/Er
    // staggering. One ghost face in the transverse direction so the
    // corner UCT EMF can read both adjacent faces of each family.
    fields.alloc_init(FaceFluxRName, lev,
                      amrex::convert(ba, amrex::IntVect(1, 0)), dm,
                      FaceFluxComponent::count, amrex::IntVect(0, 1), 0.0_rt);
    fields.alloc_init(FaceFluxZName, lev,
                      amrex::convert(ba, amrex::IntVect(0, 1)), dm,
                      FaceFluxComponent::count, amrex::IntVect(1, 0), 0.0_rt);
#endif

    fields.alloc_init(OldMassDensityName, lev, ba, dm, 1, guard_cells, 0.0_rt);
    fields.alloc_init(OldMomentumDensityName, lev, ba, dm, 3, guard_cells, 0.0_rt);
    fields.alloc_init(OldElectronEnergyName, lev, ba, dm, 1, guard_cells, 0.0_rt);
    fields.alloc_init(OldIonEnergyName, lev, ba, dm, 1, guard_cells, 0.0_rt);
    fields.alloc_init(OldIonParallelEnergyName, lev, ba, dm, 1, guard_cells, 0.0_rt);
    fields.alloc_init(OldIonPerpEnergyName, lev, ba, dm, 1, guard_cells, 0.0_rt);
    fields.alloc_init(OldIonInternalEnergyName, lev, ba, dm, 1, guard_cells, 0.0_rt);
}

void ThetaImplicitMHD::Define (WarpX* const warpx, const bool from_restart)
{
    BL_PROFILE("ThetaImplicitMHD::Define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_is_defined, "ThetaImplicitMHD object is already defined");
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    WARPX_ABORT_WITH_MESSAGE(
        "theta_implicit_mhd supports Cartesian and cylindrical RZ geometry only");
#endif

    m_WarpX = warpx;
    m_num_amr_levels = 1;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_WarpX->maxLevel() == 0,
                                     "theta_implicit_mhd currently supports one AMR level");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!EB::enabled(),
                                     "theta_implicit_mhd does not yet include "
                                     "embedded-boundary fluid fluxes");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_WarpX->get_load_balance_intervals().isActivated(),
        "theta_implicit_mhd does not yet remake its composite solver state after "
        "runtime load balancing; algo.load_balance_intervals must be disabled");
#if defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::n_rz_azimuthal_modes == 1,
        "theta_implicit_mhd in RZ supports the axisymmetric m=0 mode only");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_WarpX->Geom(0).isPeriodic(0),
        "theta_implicit_mhd in RZ requires a non-periodic radial direction");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC ||
            WarpX::field_boundary_hi[0] == FieldBoundaryType::Open,
        "theta_implicit_mhd in RZ requires the upper radial field boundary "
        "to be PEC (conducting wall; the fluid is reflected there) or open "
        "(Green's-function free-space coupling; the fluid gets zero-gradient "
        "outflow ghosts there)");
    m_r_open = (WarpX::field_boundary_hi[0] == FieldBoundaryType::Open);
    if (m_r_open) {
        // The Green's-function ghost fill runs inside ApplyBfieldBoundary,
        // which UpdateMagneticFieldAndApplyBCs executes in every residual
        // evaluation: the ghost psi is an instantaneous LINEAR map of the
        // current-iterate interior J_theta, so Newton converges boundary
        // and interior self-consistently and matrix-free Jacobian probes
        // see the coupling exactly (OPEN_BC_GREENS_DESIGN.md Sec. 6).
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            GreensFunctionOpenBC::IsActive(),
            "theta_implicit_mhd with boundary.field_hi = open on r requires "
            "the Green's-function open BC to be active (RZ, hybrid solver)");
    }
    if (m_r_open_fluid == "absorb") {
        // The absorbing wall is realized through the wall-face flux
        // registers (the one-way valve on the advective channels), so it
        // exists on the conservative-form path only, and it is only
        // meaningful when the field boundary is actually open (a PEC
        // wall reflects the fluid by construction).
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_r_open && m_use_recast,
            "implicit_mhd.r_open_fluid = absorb requires the open radial "
            "field boundary (boundary.field_hi = open on r) and "
            "implicit_mhd.fluid_flux = hlld or central");
    }
    if (!m_WarpX->Geom(0).isPeriodic(1)) {
        // Outflow (Neumann) axial ends: the solver fills the z domain
        // ghosts itself with zero-gradient values. A z face may instead be
        // open (Green's-function free-space cap): the FLUID moments keep
        // the Neumann outflow ghosts, but the field/current z-ghost fills
        // must not overwrite the Green's values on that face (the fill
        // runs inside ApplyBfieldBoundary in every residual evaluation, so
        // the ghost psi is an instantaneous LINEAR map of the
        // current-iterate interior currents and JFNK probes see it
        // exactly, same as the open radial face).
        m_z_neumann = true;
        m_z_lo_open = (WarpX::field_boundary_lo[1] == FieldBoundaryType::Open);
        m_z_hi_open = (WarpX::field_boundary_hi[1] == FieldBoundaryType::Open);
        // A z_lo PMC face is the mirror-symmetry plane: tangential B and
        // normal E vanish there, which is exactly the electromagnetic
        // parity of the z-mirror (Br, B_theta odd; Bz even; E_r, E_theta
        // even; E_z odd). The solver applies the parity ghost fills
        // itself (the upstream PMC apply only reaches ng_fieldgather
        // ghosts, zero in particle-free MHD runs, and no Yee component
        // is nodal-normal at a z face); the fluid pairing is asserted
        // below.
        m_z_lo_pmc = (WarpX::field_boundary_lo[1] == FieldBoundaryType::PMC);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (WarpX::field_boundary_lo[1] == FieldBoundaryType::None ||
             m_z_lo_open || m_z_lo_pmc) &&
                (WarpX::field_boundary_hi[1] == FieldBoundaryType::None || m_z_hi_open),
            "theta_implicit_mhd with non-periodic z requires "
            "boundary.field_lo/hi = none in z (the solver applies its own "
            "Neumann outflow ghost fills) or open (Green's-function "
            "free-space cap for the fields; the fluid keeps outflow "
            "ghosts); z_lo may instead be pmc (mirror-symmetry plane, "
            "paired with implicit_mhd.z_lo_boundary_fluid = symmetry)");
        if (m_z_lo_open || m_z_hi_open) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                GreensFunctionOpenBC::IsActive(),
                "theta_implicit_mhd with boundary.field_lo/hi = open on z "
                "requires the Green's-function open BC to be active (RZ, "
                "hybrid solver)");
        }
    }
    // The mirror plane must be configured whole or not at all: fluid
    // symmetry ghosts against non-PMC fields (or PMC fields against
    // outflow fluid ghosts) would leak through the plane.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (m_z_lo_boundary_fluid == "symmetry") == m_z_lo_pmc,
        "the z_lo mirror is a pairing: implicit_mhd.z_lo_boundary_fluid = "
        "symmetry requires boundary.field_lo = pmc on z, and a z_lo pmc "
        "field boundary requires implicit_mhd.z_lo_boundary_fluid = "
        "symmetry");
    // The selectable z-end fluid ghosts only exist where the solver
    // fills axial domain ghosts at all; on periodic z they would be a
    // silent no-op.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_z_boundary_fluid == "neumann" || m_z_neumann,
        "implicit_mhd.z_boundary_fluid = wall_temperature/outflow requires "
        "non-periodic z boundaries (boundary.field_lo/hi = none or open "
        "in z)");
    if (m_density_eater_band.empty()) {
        // The reference code's sym_bc dispatch (ntb.f90:783-792): the eater band
        // sits at the mirror plane on a half-domain, else about the
        // domain z-center.
        m_density_eater_band = m_z_lo_pmc ? "z_lo" : "z_center";
    }
    // Stair-step shaped wall (implicit_mhd.wall_model = pec |
    // pec_response | dielectric): a static mask built from the revolved
    // wall polyline (run32's EB analog for the implicit path). The
    // conductor modes apply an affine projection of the assembled Ohm E
    // in every residual evaluation (JFNK-exact), mirrored by the
    // preconditioner's resistive stencil emission; the dielectric
    // standoff shares the identical FLUID contract but leaves the field
    // untouched (EM-transparent; see ImplicitMHDWallMask). Default none
    // is bit-identical to no wall.
    {
        using ablastr::fields::Direction;
        m_wall_mask.Define(
            m_WarpX->Geom(0),
            m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, 0)
                ->nGrowVect());
        if (m_wall_mask.IsActive()) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_use_recast,
                "implicit_mhd.wall_model = pec/pec_response/dielectric "
                "requires the conservative-form recast "
                "(implicit_mhd.fluid_flux = hlld or central): the wall acts "
                "on the solver-assembled Ohm electric field and the recast "
                "flux kernels");
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_wall_viscosity_mask || m_wall_mask.IsActive(),
            "implicit_mhd.wall_viscosity_mask zeroes the viscous face "
            "coefficient along the shaped-wall contour and requires an "
            "active implicit_mhd.wall_model");
        if (m_wall_mask.GetThermalBC() !=
            ImplicitMHDWallMask::ThermalBC::none) {
            // The exterior clamp parks the band at the floor image and
            // the flux kernels divide by max(rho, floor) when loading
            // band cell states: a zero floor would make the clamped
            // band 0/0.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_mass_density_floor > 0.0_rt,
                "implicit_mhd.wall_thermal_bc (the rigid-conductor fluid "
                "freeze) requires a positive "
                "implicit_mhd.mass_density_floor: the shaped-wall "
                "exterior clamp parks the band at the floor image");
        }
        // The temperature wall exchanges heat conductively: without a
        // conduction channel the reservoir is unreachable and the mode
        // would be a silent no-op (zero_flux without conduction is
        // trivially satisfied and stays legal -- the masked-cell Joule
        // gate still applies).
        if (m_wall_mask.GetThermalBC() ==
            ImplicitMHDWallMask::ThermalBC::temperature) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_conduction_braginskii || m_chi_ion_is_parser ||
                    m_chi_electron_is_parser ||
                    m_thermal_diffusivity_ion > 0.0_rt ||
                    m_thermal_diffusivity_electron > 0.0_rt,
                "implicit_mhd.wall_thermal_bc = temperature requires a "
                "conduction channel (thermal_diffusivity_ion/electron, "
                "constant or parser, or thermal_conduction_model = "
                "braginskii): the wall reservoir exchanges conductively");
        }
    }
#else
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_WarpX->Geom(0).isPeriodic(direction),
            "theta_implicit_mhd currently requires periodic field boundaries");
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_z_boundary_fluid == "neumann",
        "implicit_mhd.z_boundary_fluid = wall_temperature/outflow requires "
        "cylindrical RZ geometry (non-periodic z ends)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_z_lo_boundary_fluid.empty(),
        "implicit_mhd.z_lo_boundary_fluid = symmetry requires cylindrical "
        "RZ geometry (a non-periodic z_lo end)");
#endif

    m_hybrid_pic_model = m_WarpX->get_pointer_HybridPICModel();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_hybrid_pic_model != nullptr,
                                     "theta_implicit_mhd requires algo.maxwell_solver = hybrid");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(m_fluid_flux == "hllc" && m_hybrid_pic_model->m_include_hall_term),
        "implicit_mhd.fluid_flux = hllc requires include_hall_term = false: "
        "the electron energy is advected with the ion contact wave, which "
        "assumes u_e = u_i at the fluid faces");
    if (m_use_recast) {
        // Hall MHD in the recast: the ideal face induction flux keeps the
        // ION velocity and the solver-assembled Ohm's law adds the edge
        // Hall EMF (J x B)/rho_q, so E = -u_e x B exactly (see
        // AssembleOhmElectricField). The central flux supports it (its
        // electron-energy channel advects with u_e, reducing bitwise to
        // the ion velocity when Hall is off); the HLLD fan does not:
        // like hllc, its electron energy rides the ION contact wave of
        // the star construction, and its corner-EMF dissipation is
        // scaled by the ion fan's rotational speeds, which know nothing
        // of whistlers.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !(m_use_hlld && m_hybrid_pic_model->m_include_hall_term),
            "implicit_mhd.fluid_flux = hlld requires include_hall_term = "
            "false: the electron energy is advected with the ion contact "
            "wave of the HLLD star construction, which assumes u_e = u_i "
            "(fluid_flux = central supports Hall MHD in the recast)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_hybrid_pic_model->m_include_electron_pressure_term,
            "implicit_mhd.fluid_flux = hlld/central requires "
            "include_electron_pressure_term = false: the solver-assembled "
            "Ohm's law is E = -u x B [+ J x B/rho_q] + eta J "
            "- eta_H laplacian(J) (the electron pressure acts through the "
            "fluid work terms, not an Ohm gradient)");
    }
    if (m_hybrid_pic_model->m_include_electron_inertia) {
        // Electron inertia (the reactive whistler cap, Angus et al.):
        // rides the solver-assembled Ohm's law and completes
        // E = -u_e x B - (m_e_eff/e) D u_e/Dt, so it requires the Hall
        // term (without J x B/rho_q the Ohm law carries no
        // electron-scale branch for the cap to regularize) and the
        // central recast flux (the same u_e-consistency argument that
        // gates Hall out of the hlld fan).
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_hybrid_pic_model->m_include_hall_term,
            "hybrid_pic_model.include_electron_inertia requires "
            "include_hall_term = true under theta_implicit_mhd: the "
            "inertial cap regularizes the Hall/whistler branch");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_use_central,
            "hybrid_pic_model.include_electron_inertia requires "
            "implicit_mhd.fluid_flux = central (the Hall-capable recast "
            "flux; hlld/hllc advect the electron energy with the ion "
            "contact wave)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !from_restart,
            "hybrid_pic_model.include_electron_inertia does not support "
            "restarts yet (the electron-current history is not "
            "checkpointed)");
        // Recast-mode inertia assembly (see HybridPICModel.H): the
        // effective electron mass resolves from the FLUID ion mass
        // m_ion = q_e/(q/m) of the quasi-neutral single-ion fluid --
        // the state has no particle species for the hybrid path's
        // lightest-ion scan.
        m_hybrid_pic_model->m_inertia_recast_mode = true;
        const amrex::Real ion_mass = PhysConst::q_e / m_ion_charge_to_mass;
        m_hybrid_pic_model->m_electron_inertia_mass =
            (m_hybrid_pic_model->m_reduced_electron_mass_ratio > 0.0_rt)
                ? ion_mass /
                      m_hybrid_pic_model->m_reduced_electron_mass_ratio
                : PhysConst::m_e;
        // Dust gate (electron_inertia_linear_below): the recast converts
        // the mass-density threshold to the charge-density image the
        // nodal assembly compares against with the SAME fluid
        // charge-to-mass ratio FillFluidSources uses to fill rho_fp from
        // the mass-density state.
        m_hybrid_pic_model->m_electron_inertia_gate_rhoq =
            m_hybrid_pic_model->m_electron_inertia_linear_below *
            m_ion_charge_to_mass;
        amrex::Print() << "[ThetaImplicitMHD] electron inertia: m_e_eff = "
                       << m_hybrid_pic_model->m_electron_inertia_mass
                       << " kg (mass ratio "
                       << m_hybrid_pic_model->m_reduced_electron_mass_ratio
                       << ")";
        if (m_hybrid_pic_model->m_electron_inertia_linear_below > 0.0_rt) {
            amrex::Print()
                << " (linear below rho = "
                << m_hybrid_pic_model->m_electron_inertia_linear_below
                << " kg/m3)";
        }
        amrex::Print() << "\n";
    }
    // Wall-mask banner, seam-guard line: report the per-family count of
    // live rows at which the Hall/inertia/hyper Ohm terms are zeroed
    // (see ImplicitMHDWallMask and the seam guard in
    // AssembleOhmElectricField), whenever any of the guarded terms is
    // active alongside the shaped CONDUCTING wall. The dielectric
    // standoff never guards (nothing is pinned, so the stencils see a
    // continuous live field): no seam-guard line prints there.
    if (m_wall_mask.IsActive() && !m_wall_mask.IsDielectric() &&
        (m_hybrid_pic_model->m_include_hall_term ||
         m_hybrid_pic_model->m_include_electron_inertia ||
         m_hybrid_pic_model->m_include_hyper_resistivity_term)) {
        const auto guarded = m_wall_mask.SeamGuardedRowCounts();
        amrex::Print() << "ImplicitMHDWallMask: seam guard active "
                          "(Hall/inertia/hyper Ohm terms zeroed at "
                          "wall-seam-adjacent live rows): "
                       << guarded[0] << " E_r, " << guarded[1]
                       << " E_theta, " << guarded[2] << " E_z rows\n";
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_hybrid_pic_model->m_solve_electron_energy_equation,
        "theta_implicit_mhd advances electron energy inside JFNK; "
        "hybrid_pic_model.solve_electron_energy_equation must be false");
    if (m_hybrid_pic_model->m_add_external_fields) {
        // Split-field external vector-potential support: Bfield_fp and
        // Efield_fp hold TOTAL fields between steps (for diagnostics,
        // checkpoints, and the t=0 add in HybridPICInitializeRhoJandB) and
        // the PLASMA RESPONSE only inside the nonlinear solve. Ohm's law
        // adds B_ext for the J x B term and subtracts E_ext from the
        // computed E; the rho-floor gate on that subtraction is a
        // state-dependent discontinuity that breaks matrix-free Jacobian
        // probes, so it is made unconditional here.
        m_hybrid_pic_model->m_external_e_subtraction_unconditional = true;
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_external_field_iteration || m_hybrid_pic_model->m_add_external_fields,
        "implicit_mhd.external_field_iteration requires "
        "hybrid_pic_model.add_external_fields");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_hybrid_pic_model->m_has_external_current,
        "theta_implicit_mhd does not yet support hybrid external currents");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_hybrid_pic_model->m_holmstrom_vacuum_region,
        "theta_implicit_mhd implements its own vacuum-region cell switching "
        "(implicit_mhd.vacuum_mass_density); the hybrid Holmstrom switch "
        "makes Ohm's law discontinuous, which breaks the JFNK Jacobian");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_vacuum_resistivity_diffusivity == 0.0_rt || m_use_recast,
        "implicit_mhd.vacuum_resistivity_diffusivity requires "
        "implicit_mhd.fluid_flux = hlld or central: the density-keyed "
        "vacuum resistivity boosts the solver-assembled Ohm field advance "
        "only (Joule heating keeps the un-boosted user resistivity)");

    if (m_vacuum_mass_density > 0.0_rt) {
        // Holmstrom-style vacuum cell switching for the ion fluid: cells
        // below the threshold are passive dust whose momentum is frozen
        // (see ComputeFluidRHS). With the standard-MHD Ohm switches (no
        // Hall, no electron-pressure gradient) the frozen dust velocity
        // makes E relax to eta J in the vacuum region without any switch
        // in Ohm's law itself. The threshold must sit strictly above the
        // fluid positivity floor.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_vacuum_mass_density > m_mass_density_floor,
            "implicit_mhd.vacuum_mass_density (the vacuum cell-switch "
            "threshold) must exceed implicit_mhd.mass_density_floor");
    }

    // The Ohm's-law division guard (hybrid_pic_model.n_floor) and the fluid
    // admissibility floor (implicit_mhd.mass_density_floor) are DELIBERATELY
    // independent. Raising the fluid floor to the Ohm guard pins every cell
    // of a no-pedestal halo at its admissibility bound, and the resulting
    // tens of thousands of projected direction components stagnate the
    // bounded Newton solve. Ohm's law, the eta/T_e evaluations, and the
    // -(u x B) velocity reconstruction floor their own inputs at the hybrid
    // n_floor (see OhmMassDensityFloor); the fluid floor stays a pure
    // positivity/div-by-zero guard.

    if (!m_use_recast &&
        m_hybrid_pic_model->m_include_hall_term !=
            m_hybrid_pic_model->m_include_electron_pressure_term)
    {
        // E-state path only: the recast asserts the Ohm electron-pressure
        // term off and carries the electron pressure through the fluid
        // work terms instead, so Hall-on there is the standard recast
        // Hall-MHD exchange, not this exploratory split.
        ablastr::warn_manager::WMRecordWarning(
            "ThetaImplicitMHD",
            "Using different Hall and electron-pressure switches is an "
            "exploratory extended-MHD model and does not preserve the continuum "
            "total-energy "
            "exchange of either Hall MHD (both true) or standard resistive MHD "
            "(both false).");
    }
    if (m_include_joule_heating &&
        m_hybrid_pic_model->m_include_hyper_resistivity_term)
    {
        ablastr::warn_manager::WMRecordWarning(
            "ThetaImplicitMHD",
            "Electron heating currently includes eta*|J|^2 but not the "
            "hyper-resistive magnetic-energy loss. Total-energy accounting is "
            "therefore incomplete when plasma_hyper_resistivity is nonzero.");
    }
    if (!from_restart) {
        InitializeFluidState();
    }

    const amrex::Real electric_field_scale = m_reference_velocity * m_reference_magnetic_field;
    const amrex::Real momentum_scale = m_reference_mass_density * m_reference_velocity;
    const amrex::Real energy_scale =
        m_reference_magnetic_field * m_reference_magnetic_field / PhysConst::mu0;
    std::vector<WarpXSolverVec::MultiFabBlockSpec> fluid_blocks = {
        {MassDensityName, m_reference_mass_density},
        {MomentumDensityName, momentum_scale},
        {ElectronEnergyName, energy_scale}};
    if (m_ion_closure == "total_energy") {
        fluid_blocks.push_back({IonEnergyName, energy_scale});
    } else if (m_ion_closure == "dual_energy") {
        // Dual-energy: the conservative E_i block (assembled exactly as
        // under total_energy) plus the auxiliary internal-energy block.
        fluid_blocks.push_back({IonEnergyName, energy_scale});
        fluid_blocks.push_back({IonInternalEnergyName, energy_scale});
    } else if (m_ion_closure == "cgl") {
        fluid_blocks.push_back({IonParallelEnergyName, energy_scale});
        fluid_blocks.push_back({IonPerpEnergyName, energy_scale});
    }
    if (m_use_recast) {
        // Conservative-form recast: B^{n+theta} is the JFNK array block
        // (E is derived from the face EMF and eta J each residual).
        m_state.Define(m_WarpX, "Bfield_fp", "none", fluid_blocks,
                       m_reference_magnetic_field);
    } else {
        m_state.Define(m_WarpX, "Efield_fp", "none", fluid_blocks,
                       electric_field_scale);
    }
    m_state_old.Define(m_state);
    m_state.Copy(m_use_recast ? FieldType::Bfield_fp : FieldType::Efield_fp);
    m_state.CopyMultiFabBlocksFromFields();
    m_state_old.Copy(m_state);

    using ablastr::fields::Direction;
    for (int direction = 0; direction < 3; ++direction) {
        const auto& magnetic_field =
            m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{direction}, 0);
        m_WarpX->m_fields.alloc_init(FieldType::B_old, Direction{direction}, 0,
                                     magnetic_field->boxArray(), magnetic_field->DistributionMap(),
                                     magnetic_field->nComp(), magnetic_field->nGrowVect(), 0.0_rt);
    }

    const amrex::ParmParse pp("implicit_evolve");
    pp.query("theta", m_theta);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_theta >= 0.5_rt && m_theta <= 1.0_rt,
                                     "implicit_evolve.theta must be between 0.5 and 1");
    if (m_resistive_theta < 0.0_rt) {
        // Default: the dissipative Ohm terms keep the global centering.
        m_resistive_theta = m_theta;
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_resistive_theta >= 0.5_rt && m_resistive_theta <= 1.0_rt,
        "implicit_mhd.resistive_theta must be between 0.5 and 1");
    if (m_resistive_theta != m_theta) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_use_recast,
            "implicit_mhd.resistive_theta different from "
            "implicit_evolve.theta requires implicit_mhd.fluid_flux = "
            "hlld or central (the resistive-stage current is wired into "
            "the solver-assembled Ohm's law only)");
        // Resistive-stage current registers: J^n captured once per step
        // and the per-residual theta_r-weighted scratch, both at the
        // native Yee staggering of the plasma current.
        using ablastr::fields::Direction;
        for (int direction = 0; direction < 3; ++direction) {
            const auto& plasma_current = m_WarpX->m_fields.get(
                FieldType::hybrid_current_fp_plasma, Direction{direction}, 0);
            m_WarpX->m_fields.alloc_init(
                OldPlasmaCurrentName, Direction{direction}, 0,
                plasma_current->boxArray(), plasma_current->DistributionMap(),
                plasma_current->nComp(), plasma_current->nGrowVect(), 0.0_rt);
            m_WarpX->m_fields.alloc_init(
                ResistiveStageCurrentName, Direction{direction}, 0,
                plasma_current->boxArray(), plasma_current->DistributionMap(),
                plasma_current->nComp(), plasma_current->nGrowVect(), 0.0_rt);
        }
    }
    if (m_conduction_theta < 0.0_rt) {
        // Default: the conductive fluxes keep the global centering.
        m_conduction_theta = m_theta;
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_conduction_theta >= 0.5_rt && m_conduction_theta <= 1.0_rt,
        "implicit_mhd.conduction_theta must be between 0.5 and 1");
    if (m_conduction_theta != m_theta) {
        // No stage registers needed (unlike resistive_theta): the stage
        // energies are extrapolated in-kernel from arrays the conductive
        // flux already reads. Guard silent no-ops only.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_conduction_braginskii || m_chi_ion_is_parser ||
                m_chi_electron_is_parser ||
                m_thermal_diffusivity_ion > 0.0_rt ||
                m_thermal_diffusivity_electron > 0.0_rt,
            "implicit_mhd.conduction_theta different from "
            "implicit_evolve.theta requires an active thermal-conduction "
            "channel (only the conductive face fluxes consume the "
            "shifted centering)");
    }
    parseNonlinearSolverParams(pp);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_nlsolver_type == NonlinearSolverType::newton,
        "theta_implicit_mhd currently requires "
        "implicit_evolve.nonlinear_solver = newton so fluid-positivity bounds are "
        "also applied to matrix-free Jacobian probes");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_use_mass_matrices, "particle mass matrices are not applicable to theta_implicit_mhd");
    m_nlsolver->Define(m_state, this);
    const PreconditionerType preconditioner_type =
        m_nlsolver->GetPreconditionerType();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        preconditioner_type == PreconditionerType::none ||
            preconditioner_type == PreconditionerType::pc_mhd_block,
        "theta_implicit_mhd supports jacobian.pc_type = none or pc_mhd_block");
#if defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        preconditioner_type == PreconditionerType::none || m_use_recast,
        "pc_mhd_block in RZ requires implicit_mhd.fluid_flux = hlld or "
        "central: the E-based block preconditioner has no cylindrical "
        "metric terms");
#endif
    if (preconditioner_type == PreconditionerType::pc_mhd_block &&
        !m_use_recast) {
        // The E-based operators do not apply to the conservative-form
        // recast; under hlld the preconditioner is the Stage-1 identity-B +
        // signal-diffusion form, which handles any ion closure, any
        // resistivity, and the Open/None boundaries the recast supports.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::grid_type == ablastr::utils::enums::GridType::Staggered,
            "pc_mhd_block with the E-based state currently requires "
            "warpx.grid_type = staggered");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_hybrid_pic_model->m_include_hyper_resistivity_term,
            "pc_mhd_block with the E-based state currently requires zero "
            "plasma_hyper_resistivity");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_hybrid_pic_model->m_has_per_species_eta,
            "pc_mhd_block does not yet support per-species resistivity");
        const amrex::Parser resistivity_parser = utils::parser::makeParser(
            m_hybrid_pic_model->m_eta_expression, {"rho", "J", "t"});
        const std::set<std::string> resistivity_symbols =
            resistivity_parser.symbols();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            resistivity_symbols.count("rho") == 0 &&
                resistivity_symbols.count("J") == 0,
            "pc_mhd_block with the E-based state currently requires "
            "plasma_resistivity to be constant or time-only (no rho or J "
            "dependence)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_fluid_flux == "centered",
            "pc_mhd_block with the E-based state currently requires "
            "implicit_mhd.fluid_flux = centered");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_ion_closure == "barotropic",
            "pc_mhd_block with the E-based state currently requires "
            "implicit_mhd.ion_closure = barotropic; total_energy runs "
            "unpreconditioned");
    }

    FillFluidSources(m_state);
    m_is_defined = true;
}

amrex::Real
ThetaImplicitMHD::OhmMassDensityFloor () const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_pic_model != nullptr,
        "ThetaImplicitMHD Ohm density guard requested before Define()");
    return m_hybrid_pic_model->m_n_floor * PhysConst::q_e / m_ion_charge_to_mass;
}

amrex::Real
ThetaImplicitMHD::VacuumReferenceMassDensity () const
{
    // The static Ohm guard is the exact fraction = 0 limit
    // (m_vacuum_reference_mass_density stays 0), so the default is
    // bit-identical; with the dynamic reference active the max keeps a
    // globally decaying state from dragging the reference below the
    // guard (the reference code's en0 = max(en00, 0.1 max(en)) composition).
    return std::max(OhmMassDensityFloor(), m_vacuum_reference_mass_density);
}

void ThetaImplicitMHD::RefreshVacuumReferenceDensity (const int step)
{
    if (m_vacuum_reference_peak_fraction <= 0.0_rt) {
        return;
    }
    // Global step-old density peak (MultiFab::max is an all-rank
    // reduction), from the state at OneStep refresh time -- the state
    // that becomes m_state_old, so every residual/Jacobian evaluation of
    // this solve keys the boosts to the same frozen reference.
    const amrex::Real density_peak =
        m_state.getMultiFabBlock(MassDensityName, 0).max(0);
    m_vacuum_reference_mass_density =
        m_vacuum_reference_peak_fraction * density_peak;
    const amrex::Real reference = VacuumReferenceMassDensity();
    if (m_vacuum_reference_printed < 0.0_rt ||
        std::abs(reference - m_vacuum_reference_printed) >
            0.01_rt * m_vacuum_reference_printed) {
        amrex::Print() << "ThetaImplicitMHD: step " << step + 1
                       << " vacuum reference density = " << reference
                       << " kg/m^3 (" << m_vacuum_reference_peak_fraction
                       << " x peak " << density_peak << ", Ohm guard "
                       << OhmMassDensityFloor() << ")\n";
        m_vacuum_reference_printed = reference;
    }
}

amrex::Real
ThetaImplicitMHD::GetMHDReferenceResistivityForPC (const amrex::Real time) const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_pic_model != nullptr,
        "ThetaImplicitMHD reference resistivity requested before Define()");
    const amrex::Real reference_charge_density =
        m_ion_charge_to_mass * m_reference_mass_density;
    // Reference Te [K] for the (rho, Te, J, t) parser: the hybrid seed
    // temperature (hybrid_pic_model.elec_temp, stored in J), matching the
    // uniform value hybrid_electron_temperature_fp is initialized with.
    const amrex::Real reference_temperature =
        m_hybrid_pic_model->m_elec_temp / PhysConst::kb;
    return m_hybrid_pic_model->m_eta(reference_charge_density,
                                     reference_temperature, 0.0_rt, time);
}

const amrex::MultiFab*
ThetaImplicitMHD::GetMHDFieldResistivityCCForPC (const amrex::Real time) const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_pic_model != nullptr,
        "ThetaImplicitMHD field resistivity requested before Define()");
    // Same eta_field formula as the residual's Ohm assembly (see
    // AssembleOhmElectricField and vacuum_keyed_resistivity in
    // ThetaImplicitMHD_K.H): the user eta floored at the Ohm charge-density
    // guard, boosted by the density-keyed vacuum resistivity with divisions
    // guarded at the positivity floor. Evaluated at zero current magnitude
    // (the GetMHDReferenceResistivityForPC precedent; the stiff vacuum boost
    // carries no J dependence) from the cell-centered mass density frozen at
    // the preconditioner update state (rho_q = (q/m) rho for the
    // quasi-neutral single-ion fluid).
    const amrex::MultiFab& density = *m_WarpX->m_fields.get(MassDensityName, 0);
    amrex::MultiFab& resistivity =
        *m_WarpX->m_fields.get(FieldResistivityCCName, 0);
    const auto eta = m_hybrid_pic_model->m_eta;
    const amrex::Real charge_to_mass = m_ion_charge_to_mass;
    const amrex::Real charge_density_floor =
        m_ion_charge_to_mass * OhmMassDensityFloor();
    const amrex::Real vacuum_division_guard =
        m_ion_charge_to_mass * m_mass_density_floor;
    const amrex::Real vacuum_eta_scale =
        PhysConst::mu0 * m_vacuum_resistivity_diffusivity;
    // Vacuum-boost reference: the per-step frozen dynamic reference
    // (VacuumReferenceMassDensity; the static Ohm guard when the
    // dynamic fraction is off), matching the residual's Ohm assembly so
    // the PC sees the identical eta_field.
    const amrex::Real vacuum_reference_charge_density =
        m_ion_charge_to_mass * VacuumReferenceMassDensity();
    // Te [K] frozen at the preconditioner update state: the cell-centered
    // temperature-primary scratch, refreshed by the last FillFluidSources
    // (the PC never differentiates eta; the standard frozen-eta lag).
    const amrex::MultiFab& temperature_cc =
        *m_WarpX->m_fields.get(ElectronTemperatureCCName, 0);
    // Wall-band eta override (see the residual's Ohm assembly): the
    // cell-centered fill only feeds the resistive block's interval
    // bound through its maximum, so masked cells take
    // max(composed, override) here -- never less than either the
    // overridden band-interior edges or the composed interface edges
    // they neighbor.
    const WallBandEtaOverrideView wall_band_view =
        m_wall_mask.BandEtaOverrideView();
    const int* const band_override_cc = wall_band_view.first_band_cc;
    const amrex::Real band_eta_override = wall_band_view.eta_override;
    for (amrex::MFIter mfi(resistivity); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto eta_field = resistivity.array(mfi);
        const auto rho = density.const_array(mfi);
        const auto te_cc = temperature_cc.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            const amrex::Real charge_density_raw =
                charge_to_mass * rho(i, j, k);
            const amrex::Real charge_density_value =
                std::max(charge_density_raw, charge_density_floor);
            amrex::Real value =
                theta_implicit_mhd::vacuum_keyed_resistivity(
                    eta(charge_density_value, te_cc(i, j, k), 0.0_rt, time),
                    charge_density_raw, vacuum_reference_charge_density,
                    vacuum_division_guard, vacuum_eta_scale);
            if (band_override_cc != nullptr && i >= band_override_cc[j]) {
                value = std::max(value, band_eta_override);
            }
            eta_field(i, j, k) = value;
        });
    }
    return &resistivity;
}

amrex::Array<const amrex::MultiFab*, 3>
ThetaImplicitMHD::GetMHDFieldResistivityEdgeForPC (const amrex::Real time) const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_pic_model != nullptr,
        "ThetaImplicitMHD field resistivity requested before Define()");
    // eta_field on the electric-field staggerings with the SAME density
    // interpolation the residual's Ohm assembly uses: nodal rho_fp is the
    // arithmetic average of the neighboring cell-centered densities, and
    // the face values average the neighboring nodes (see
    // AssembleOhmElectricField). Evaluating eta AT the interpolated
    // density matters: the vacuum boost is quadratic in 1/rho, so
    // averaging cell etas instead misrepresents the operator by order
    // unity across halo density gradients. Evaluated at zero current
    // magnitude like the cell-centered variant.
    const amrex::MultiFab& density = *m_WarpX->m_fields.get(MassDensityName, 0);
    // Te [K] frozen at the preconditioner update state, interpolated to the
    // staggering points with the same stencils as the density (eta AT the
    // interpolated arguments, matching the residual's Ohm assembly).
    const amrex::MultiFab& temperature_cc =
        *m_WarpX->m_fields.get(ElectronTemperatureCCName, 0);
    const auto eta = m_hybrid_pic_model->m_eta;
    const amrex::Real charge_to_mass = m_ion_charge_to_mass;
    const amrex::Real charge_density_floor =
        m_ion_charge_to_mass * OhmMassDensityFloor();
    const amrex::Real vacuum_division_guard =
        m_ion_charge_to_mass * m_mass_density_floor;
    const amrex::Real vacuum_eta_scale =
        PhysConst::mu0 * m_vacuum_resistivity_diffusivity;
    // Vacuum-boost reference: the per-step frozen dynamic reference
    // (see the cell-centered fill above).
    const amrex::Real vacuum_reference_charge_density =
        m_ion_charge_to_mass * VacuumReferenceMassDensity();
#if defined(WARPX_DIM_1D_Z)
    amrex::MultiFab& node_resistivity =
        *m_WarpX->m_fields.get(FieldResistivityE0Name, 0);
    for (amrex::MFIter mfi(node_resistivity); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto eta_field = node_resistivity.array(mfi);
        const auto rho = density.const_array(mfi);
        const auto te_cc = temperature_cc.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            const amrex::Real charge_density_raw =
                charge_to_mass * 0.5_rt * (rho(i - 1, j, k) + rho(i, j, k));
            const amrex::Real charge_density_value =
                std::max(charge_density_raw, charge_density_floor);
            const amrex::Real temperature =
                0.5_rt * (te_cc(i - 1, j, k) + te_cc(i, j, k));
            eta_field(i, j, k) =
                theta_implicit_mhd::vacuum_keyed_resistivity(
                    eta(charge_density_value, temperature, 0.0_rt, time),
                    charge_density_raw, vacuum_reference_charge_density,
                    vacuum_division_guard, vacuum_eta_scale);
        });
    }
    // Ex and Ey share the z-nodal staggering and the same interpolated
    // density; the cell-centered Ez never enters the 1D resistive
    // curl-curl.
    return {&node_resistivity, &node_resistivity, nullptr};
#elif defined(WARPX_DIM_RZ)
    amrex::MultiFab& radial_resistivity =
        *m_WarpX->m_fields.get(FieldResistivityE0Name, 0);
    amrex::MultiFab& azimuthal_resistivity =
        *m_WarpX->m_fields.get(FieldResistivityE1Name, 0);
    amrex::MultiFab& axial_resistivity =
        *m_WarpX->m_fields.get(FieldResistivityE2Name, 0);
    // Wall-band eta override at the SAME band-interior rows as the
    // residual's Ohm assembly (see AssembleOhmElectricField): the PC's
    // stencil emission consumes these edge fills, so replacing here is
    // what keeps residual and preconditioner exact twins in the band.
    const WallBandEtaOverrideView wall_band_view =
        m_wall_mask.BandEtaOverrideView();
    const int* const band_override_er = wall_band_view.first_band_er;
    const int* const band_override_et = wall_band_view.first_band_et;
    const int* const band_override_ez = wall_band_view.first_band_ez;
    const amrex::Real band_eta_override = wall_band_view.eta_override;
    for (amrex::MFIter mfi(azimuthal_resistivity); mfi.isValid(); ++mfi) {
        const auto eta_radial = radial_resistivity.array(mfi);
        const auto eta_azimuthal = azimuthal_resistivity.array(mfi);
        const auto eta_axial = axial_resistivity.array(mfi);
        const auto rho = density.const_array(mfi);
        const auto te_cc = temperature_cc.const_array(mfi);
        // E_theta corners: the nodal rho_fp value itself.
        amrex::ParallelFor(
            amrex::convert(mfi.validbox(), amrex::IntVect(1, 1)),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const auto node_density = [&] (const int in, const int jn) {
                    return 0.25_rt * (rho(in - 1, jn - 1, k) + rho(in, jn - 1, k) +
                                      rho(in - 1, jn, k) + rho(in, jn, k));
                };
                const auto node_temperature = [&] (const int in, const int jn) {
                    return 0.25_rt * (te_cc(in - 1, jn - 1, k) + te_cc(in, jn - 1, k) +
                                      te_cc(in - 1, jn, k) + te_cc(in, jn, k));
                };
                const amrex::Real charge_density_raw =
                    charge_to_mass * node_density(i, j);
                const amrex::Real charge_density_value =
                    std::max(charge_density_raw, charge_density_floor);
                amrex::Real value =
                    theta_implicit_mhd::vacuum_keyed_resistivity(
                        eta(charge_density_value, node_temperature(i, j),
                            0.0_rt, time),
                        charge_density_raw, vacuum_reference_charge_density,
                        vacuum_division_guard, vacuum_eta_scale);
                if (band_override_et != nullptr && i >= band_override_et[j]) {
                    value = band_eta_override;
                }
                eta_azimuthal(i, j, k) = value;
            });
        // Er z-faces: nodes averaged in r.
        amrex::ParallelFor(
            amrex::convert(mfi.validbox(), amrex::IntVect(0, 1)),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const auto node_density = [&] (const int in, const int jn) {
                    return 0.25_rt * (rho(in - 1, jn - 1, k) + rho(in, jn - 1, k) +
                                      rho(in - 1, jn, k) + rho(in, jn, k));
                };
                const auto node_temperature = [&] (const int in, const int jn) {
                    return 0.25_rt * (te_cc(in - 1, jn - 1, k) + te_cc(in, jn - 1, k) +
                                      te_cc(in - 1, jn, k) + te_cc(in, jn, k));
                };
                const amrex::Real charge_density_raw =
                    charge_to_mass * 0.5_rt *
                    (node_density(i, j) + node_density(i + 1, j));
                const amrex::Real charge_density_value =
                    std::max(charge_density_raw, charge_density_floor);
                const amrex::Real temperature = 0.5_rt *
                    (node_temperature(i, j) + node_temperature(i + 1, j));
                amrex::Real value =
                    theta_implicit_mhd::vacuum_keyed_resistivity(
                        eta(charge_density_value, temperature, 0.0_rt, time),
                        charge_density_raw, vacuum_reference_charge_density,
                        vacuum_division_guard, vacuum_eta_scale);
                if (band_override_er != nullptr && i >= band_override_er[j]) {
                    value = band_eta_override;
                }
                eta_radial(i, j, k) = value;
            });
        // Ez r-faces: nodes averaged in z.
        amrex::ParallelFor(
            amrex::convert(mfi.validbox(), amrex::IntVect(1, 0)),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const auto node_density = [&] (const int in, const int jn) {
                    return 0.25_rt * (rho(in - 1, jn - 1, k) + rho(in, jn - 1, k) +
                                      rho(in - 1, jn, k) + rho(in, jn, k));
                };
                const auto node_temperature = [&] (const int in, const int jn) {
                    return 0.25_rt * (te_cc(in - 1, jn - 1, k) + te_cc(in, jn - 1, k) +
                                      te_cc(in - 1, jn, k) + te_cc(in, jn, k));
                };
                const amrex::Real charge_density_raw =
                    charge_to_mass * 0.5_rt *
                    (node_density(i, j) + node_density(i, j + 1));
                const amrex::Real charge_density_value =
                    std::max(charge_density_raw, charge_density_floor);
                const amrex::Real temperature = 0.5_rt *
                    (node_temperature(i, j) + node_temperature(i, j + 1));
                amrex::Real value =
                    theta_implicit_mhd::vacuum_keyed_resistivity(
                        eta(charge_density_value, temperature, 0.0_rt, time),
                        charge_density_raw, vacuum_reference_charge_density,
                        vacuum_division_guard, vacuum_eta_scale);
                if (band_override_ez != nullptr && i >= band_override_ez[j]) {
                    value = band_eta_override;
                }
                eta_axial(i, j, k) = value;
            });
    }
    return {&radial_resistivity, &azimuthal_resistivity, &axial_resistivity};
#else
    amrex::ignore_unused(density, temperature_cc, eta, charge_to_mass,
                         charge_density_floor, vacuum_division_guard,
                         vacuum_eta_scale, vacuum_reference_charge_density,
                         time);
    return {nullptr, nullptr, nullptr};
#endif
}

bool
ThetaImplicitMHD::GetMHDIncludeHyperResistivityForPC () const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_pic_model != nullptr,
        "ThetaImplicitMHD hyper-resistivity flag requested before Define()");
    return m_hybrid_pic_model->m_include_hyper_resistivity_term;
}

amrex::Array<const amrex::MultiFab*, 3>
ThetaImplicitMHD::GetMHDHyperResistivityEdgeForPC (const amrex::Real time) const
{
    // eta_H carries no explicit time argument in the (rho, B) parser.
    amrex::ignore_unused(time);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_pic_model != nullptr,
        "ThetaImplicitMHD hyper-resistivity requested before Define()");
    if (!m_hybrid_pic_model->m_include_hyper_resistivity_term) {
        return {nullptr, nullptr, nullptr};
    }
    // eta_H(rho_q, |B|) evaluated AT the electric-field staggerings with
    // the residual's own frozen arguments (see AssembleOhmElectricField):
    // the charge density interpolated with the same stencils as the eta
    // edge fill and floored at the Ohm guard, and |B| from the
    // cell-centered field (including any external contribution) with the
    // residual's averaging, read only when the parser depends on B.
    const amrex::MultiFab& density = *m_WarpX->m_fields.get(MassDensityName, 0);
    const amrex::MultiFab& magnetic_cc =
        *m_WarpX->m_fields.get(MagneticFieldCCName, 0);
    const auto eta_h = m_hybrid_pic_model->m_eta_h;
    const bool needs_b = m_hybrid_pic_model->m_hyper_resistivity_has_B_dependence;
    const amrex::Real charge_to_mass = m_ion_charge_to_mass;
    const amrex::Real charge_density_floor =
        m_ion_charge_to_mass * OhmMassDensityFloor();
#if defined(WARPX_DIM_1D_Z)
    amrex::MultiFab& node_hyper = *m_WarpX->m_fields.get(HyperResistivityE0Name, 0);
    for (amrex::MFIter mfi(node_hyper); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto hyper = node_hyper.array(mfi);
        const auto rho = density.const_array(mfi);
        const auto b_cc = magnetic_cc.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            const amrex::Real charge_density_value =
                std::max(charge_to_mass * 0.5_rt * (rho(i - 1, j, k) + rho(i, j, k)),
                         charge_density_floor);
            amrex::Real magnetic_magnitude = 0.0_rt;
            if (needs_b) {
                amrex::Real magnetic_squared = 0.0_rt;
                for (int component = 0; component < 3; ++component) {
                    const amrex::Real value = 0.5_rt * (b_cc(i - 1, j, k, component) +
                                                        b_cc(i, j, k, component));
                    magnetic_squared += value * value;
                }
                magnetic_magnitude = std::sqrt(magnetic_squared);
            }
            hyper(i, j, k) = eta_h(charge_density_value, magnetic_magnitude);
        });
    }
    // Ex and Ey share the z-nodal staggering; the cell-centered Ez never
    // enters the 1D curl.
    return {&node_hyper, &node_hyper, nullptr};
#elif defined(WARPX_DIM_RZ)
    amrex::MultiFab& radial_hyper = *m_WarpX->m_fields.get(HyperResistivityE0Name, 0);
    amrex::MultiFab& azimuthal_hyper =
        *m_WarpX->m_fields.get(HyperResistivityE1Name, 0);
    amrex::MultiFab& axial_hyper = *m_WarpX->m_fields.get(HyperResistivityE2Name, 0);
    for (amrex::MFIter mfi(azimuthal_hyper); mfi.isValid(); ++mfi) {
        const auto hyper_radial = radial_hyper.array(mfi);
        const auto hyper_azimuthal = azimuthal_hyper.array(mfi);
        const auto hyper_axial = axial_hyper.array(mfi);
        const auto rho = density.const_array(mfi);
        const auto b_cc = magnetic_cc.const_array(mfi);
        // E_theta corners: the nodal rho_fp value; |B| from the four
        // surrounding cells (the residual's corner average).
        amrex::ParallelFor(
            amrex::convert(mfi.validbox(), amrex::IntVect(1, 1)),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const auto node_density = [&] (const int in, const int jn) {
                    return 0.25_rt * (rho(in - 1, jn - 1, k) + rho(in, jn - 1, k) +
                                      rho(in - 1, jn, k) + rho(in, jn, k));
                };
                const amrex::Real charge_density_value =
                    std::max(charge_to_mass * node_density(i, j),
                             charge_density_floor);
                amrex::Real magnetic_magnitude = 0.0_rt;
                if (needs_b) {
                    amrex::Real magnetic_squared = 0.0_rt;
                    for (int component = 0; component < 3; ++component) {
                        const amrex::Real value =
                            0.25_rt * (b_cc(i - 1, j - 1, k, component) +
                                       b_cc(i, j - 1, k, component) +
                                       b_cc(i - 1, j, k, component) +
                                       b_cc(i, j, k, component));
                        magnetic_squared += value * value;
                    }
                    magnetic_magnitude = std::sqrt(magnetic_squared);
                }
                hyper_azimuthal(i, j, k) =
                    eta_h(charge_density_value, magnetic_magnitude);
            });
        // Er z-faces: nodes averaged in r; |B| from the two axially
        // adjacent cells (the residual's z-face average).
        amrex::ParallelFor(
            amrex::convert(mfi.validbox(), amrex::IntVect(0, 1)),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const auto node_density = [&] (const int in, const int jn) {
                    return 0.25_rt * (rho(in - 1, jn - 1, k) + rho(in, jn - 1, k) +
                                      rho(in - 1, jn, k) + rho(in, jn, k));
                };
                const amrex::Real charge_density_value =
                    std::max(charge_to_mass * 0.5_rt *
                                 (node_density(i, j) + node_density(i + 1, j)),
                             charge_density_floor);
                amrex::Real magnetic_magnitude = 0.0_rt;
                if (needs_b) {
                    amrex::Real magnetic_squared = 0.0_rt;
                    for (int component = 0; component < 3; ++component) {
                        const amrex::Real value =
                            0.5_rt * (b_cc(i, j - 1, k, component) +
                                      b_cc(i, j, k, component));
                        magnetic_squared += value * value;
                    }
                    magnetic_magnitude = std::sqrt(magnetic_squared);
                }
                hyper_radial(i, j, k) =
                    eta_h(charge_density_value, magnetic_magnitude);
            });
        // Ez r-faces: nodes averaged in z; |B| from the two radially
        // adjacent cells (the residual's r-face average; the below-axis
        // cell-centered ghosts carry the fills of the recast).
        amrex::ParallelFor(
            amrex::convert(mfi.validbox(), amrex::IntVect(1, 0)),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const auto node_density = [&] (const int in, const int jn) {
                    return 0.25_rt * (rho(in - 1, jn - 1, k) + rho(in, jn - 1, k) +
                                      rho(in - 1, jn, k) + rho(in, jn, k));
                };
                const amrex::Real charge_density_value =
                    std::max(charge_to_mass * 0.5_rt *
                                 (node_density(i, j) + node_density(i, j + 1)),
                             charge_density_floor);
                amrex::Real magnetic_magnitude = 0.0_rt;
                if (needs_b) {
                    amrex::Real magnetic_squared = 0.0_rt;
                    for (int component = 0; component < 3; ++component) {
                        const amrex::Real value =
                            0.5_rt * (b_cc(i - 1, j, k, component) +
                                      b_cc(i, j, k, component));
                        magnetic_squared += value * value;
                    }
                    magnetic_magnitude = std::sqrt(magnetic_squared);
                }
                hyper_axial(i, j, k) =
                    eta_h(charge_density_value, magnetic_magnitude);
            });
    }
    return {&radial_hyper, &azimuthal_hyper, &axial_hyper};
#else
    amrex::ignore_unused(density, magnetic_cc, eta_h, needs_b, charge_to_mass,
                         charge_density_floor);
    return {nullptr, nullptr, nullptr};
#endif
}

amrex::GpuArray<amrex::Real, 3>
ThetaImplicitMHD::GetMHDReferenceMagneticFieldForPC () const
{
    const amrex::MultiFab& magnetic_field =
        *m_WarpX->m_fields.get(MagneticFieldCCName, 0);
    const amrex::Real inverse_number_of_cells =
        1.0_rt / static_cast<amrex::Real>(magnetic_field.boxArray().numPts());
    return {inverse_number_of_cells * magnetic_field.sum(0, false),
            inverse_number_of_cells * magnetic_field.sum(1, false),
            inverse_number_of_cells * magnetic_field.sum(2, false)};
}

const amrex::MultiFab*
ThetaImplicitMHD::GetMHDMagneticFieldCCForPC () const
{
    // Filled (with ghosts) by FillCellCenteredElectromagneticFields() during
    // the residual evaluation at the preconditioner's update state; includes
    // the external contribution under the split-field scheme.
    return m_WarpX->m_fields.get(MagneticFieldCCName, 0);
}

bool
ThetaImplicitMHD::GetMHDIncludeHallTermForPC () const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_pic_model != nullptr,
        "ThetaImplicitMHD Hall configuration requested before Define()");
    return m_hybrid_pic_model->m_include_hall_term;
}

amrex::Array<const amrex::MultiFab*, 3>
ThetaImplicitMHD::GetMHDHallCoefficientEdgeForPC () const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_pic_model != nullptr,
        "ThetaImplicitMHD Hall coefficients requested before Define()");
    if (!m_hybrid_pic_model->m_include_hall_term) {
        return {nullptr, nullptr, nullptr};
    }
    // Frozen Hall coefficient vector Hb = B/rho_q on the electric-field
    // staggerings, with the residual Ohm assembly's OWN interpolations
    // (see the Hall blocks of AssembleOhmElectricField): the
    // cell-centered TOTAL B (external included, below-axis ghosts
    // carrying the m = 0 parities) averaged with the eta_H stencils, over
    // the charge density interpolated with the eta stencils and guarded
    // by the same C-infinity smooth floor at the Ohm guard. Read by the
    // preconditioner's Hall/whistler rows (MHDResistiveStencil.H).
    const amrex::MultiFab& density = *m_WarpX->m_fields.get(MassDensityName, 0);
    const amrex::MultiFab& magnetic_cc =
        *m_WarpX->m_fields.get(MagneticFieldCCName, 0);
    const amrex::Real charge_to_mass = m_ion_charge_to_mass;
    const amrex::Real charge_density_floor =
        m_ion_charge_to_mass * OhmMassDensityFloor();
#if defined(WARPX_DIM_1D_Z)
    amrex::MultiFab& node_hall = *m_WarpX->m_fields.get(HallCoefficientE0Name, 0);
    for (amrex::MFIter mfi(node_hall); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto hall = node_hall.array(mfi);
        const auto rho = density.const_array(mfi);
        const auto b_cc = magnetic_cc.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            const amrex::Real charge_density_raw =
                charge_to_mass * 0.5_rt * (rho(i - 1, j, k) + rho(i, j, k));
            const amrex::Real hall_charge_density =
                theta_implicit_mhd::smooth_positive_floor(
                    charge_density_raw, charge_density_floor);
            for (int component = 0; component < 3; ++component) {
                hall(i, j, k, component) =
                    0.5_rt * (b_cc(i - 1, j, k, component) +
                              b_cc(i, j, k, component)) /
                    hall_charge_density;
            }
        });
    }
    // Ex and Ey share the z-nodal staggering; the cell-centered Ez never
    // enters the 1D curl.
    return {&node_hall, &node_hall, nullptr};
#elif defined(WARPX_DIM_RZ)
    amrex::MultiFab& radial_hall = *m_WarpX->m_fields.get(HallCoefficientE0Name, 0);
    amrex::MultiFab& azimuthal_hall =
        *m_WarpX->m_fields.get(HallCoefficientE1Name, 0);
    amrex::MultiFab& axial_hall = *m_WarpX->m_fields.get(HallCoefficientE2Name, 0);
    for (amrex::MFIter mfi(azimuthal_hall); mfi.isValid(); ++mfi) {
        const auto hall_radial = radial_hall.array(mfi);
        const auto hall_azimuthal = azimuthal_hall.array(mfi);
        const auto hall_axial = axial_hall.array(mfi);
        const auto rho = density.const_array(mfi);
        const auto b_cc = magnetic_cc.const_array(mfi);
        // E_theta corners: the nodal charge density; B from the four
        // surrounding cells (the residual's corner average).
        amrex::ParallelFor(
            amrex::convert(mfi.validbox(), amrex::IntVect(1, 1)),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const auto node_density = [&] (const int in, const int jn) {
                    return 0.25_rt * (rho(in - 1, jn - 1, k) + rho(in, jn - 1, k) +
                                      rho(in - 1, jn, k) + rho(in, jn, k));
                };
                const amrex::Real hall_charge_density =
                    theta_implicit_mhd::smooth_positive_floor(
                        charge_to_mass * node_density(i, j),
                        charge_density_floor);
                for (int component = 0; component < 3; ++component) {
                    hall_azimuthal(i, j, k, component) =
                        0.25_rt * (b_cc(i - 1, j - 1, k, component) +
                                   b_cc(i, j - 1, k, component) +
                                   b_cc(i - 1, j, k, component) +
                                   b_cc(i, j, k, component)) /
                        hall_charge_density;
                }
            });
        // Er z-faces: nodes averaged in r; B from the two axially
        // adjacent cells (the residual's z-face average).
        amrex::ParallelFor(
            amrex::convert(mfi.validbox(), amrex::IntVect(0, 1)),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const auto node_density = [&] (const int in, const int jn) {
                    return 0.25_rt * (rho(in - 1, jn - 1, k) + rho(in, jn - 1, k) +
                                      rho(in - 1, jn, k) + rho(in, jn, k));
                };
                const amrex::Real hall_charge_density =
                    theta_implicit_mhd::smooth_positive_floor(
                        charge_to_mass * 0.5_rt *
                            (node_density(i, j) + node_density(i + 1, j)),
                        charge_density_floor);
                for (int component = 0; component < 3; ++component) {
                    hall_radial(i, j, k, component) =
                        0.5_rt * (b_cc(i, j - 1, k, component) +
                                  b_cc(i, j, k, component)) /
                        hall_charge_density;
                }
            });
        // Ez r-faces: nodes averaged in z; B from the two radially
        // adjacent cells (the residual's r-face average; the below-axis
        // cell-centered ghosts carry the m = 0 parities, so Hb_r and
        // Hb_theta vanish exactly on the axis face).
        amrex::ParallelFor(
            amrex::convert(mfi.validbox(), amrex::IntVect(1, 0)),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const auto node_density = [&] (const int in, const int jn) {
                    return 0.25_rt * (rho(in - 1, jn - 1, k) + rho(in, jn - 1, k) +
                                      rho(in - 1, jn, k) + rho(in, jn, k));
                };
                const amrex::Real hall_charge_density =
                    theta_implicit_mhd::smooth_positive_floor(
                        charge_to_mass * 0.5_rt *
                            (node_density(i, j) + node_density(i, j + 1)),
                        charge_density_floor);
                for (int component = 0; component < 3; ++component) {
                    hall_axial(i, j, k, component) =
                        0.5_rt * (b_cc(i - 1, j, k, component) +
                                  b_cc(i, j, k, component)) /
                        hall_charge_density;
                }
            });
    }
    return {&radial_hall, &azimuthal_hall, &axial_hall};
#else
    amrex::ignore_unused(density, magnetic_cc, charge_to_mass,
                         charge_density_floor);
    return {nullptr, nullptr, nullptr};
#endif
}

bool
ThetaImplicitMHD::GetMHDIncludeElectronInertiaForPC () const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_pic_model != nullptr,
        "ThetaImplicitMHD electron-inertia flag requested before Define()");
    return m_hybrid_pic_model->m_include_electron_inertia;
}

amrex::Real
ThetaImplicitMHD::GetMHDElectronInertiaStencilScaleForPC () const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_pic_model != nullptr,
        "ThetaImplicitMHD electron-inertia scale requested before Define()");
    return m_hybrid_pic_model->ElectronInertiaStencilScale(m_theta);
}

amrex::Array<const amrex::MultiFab*, 3>
ThetaImplicitMHD::GetMHDElectronInertiaCoefficientEdgeForPC () const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_pic_model != nullptr,
        "ThetaImplicitMHD electron-inertia coefficients requested before "
        "Define()");
    if (!m_hybrid_pic_model->m_include_electron_inertia) {
        return {nullptr, nullptr, nullptr};
    }
    // Frozen electron-inertia coefficient C_i = m_e_eff/(e rho_q) on the
    // electric-field staggerings: the local (diagonal-like) inertial mass
    // of the frozen dJe/dt response (see MHDResistiveStencil.H), with the
    // charge density interpolated by the eta stencils and floored at the
    // Ohm guard exactly like the residual's nodal 1/rho division (a hard
    // max, matching ComputeElectronInertiaNodal). In 1D the transverse
    // edges ARE the assembly nodes, so the emitted rows carry the exact
    // frozen response.
    const amrex::MultiFab& density = *m_WarpX->m_fields.get(MassDensityName, 0);
    const amrex::Real charge_to_mass = m_ion_charge_to_mass;
    const amrex::Real charge_density_floor =
        m_ion_charge_to_mass * OhmMassDensityFloor();
    const amrex::Real me_over_e =
        m_hybrid_pic_model->m_electron_inertia_mass / PhysConst::q_e;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        me_over_e > 0.0_rt,
        "ThetaImplicitMHD electron-inertia mass not resolved (Define() "
        "order)");
#if defined(WARPX_DIM_1D_Z)
    amrex::MultiFab& node_inertia =
        *m_WarpX->m_fields.get(InertiaCoefficientE0Name, 0);
    for (amrex::MFIter mfi(node_inertia); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto inertia = node_inertia.array(mfi);
        const auto rho = density.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            const amrex::Real charge_density_raw =
                charge_to_mass * 0.5_rt * (rho(i - 1, j, k) + rho(i, j, k));
            inertia(i, j, k) = me_over_e /
                amrex::max(charge_density_raw, charge_density_floor);
        });
    }
    // Ex and Ey share the z-nodal staggering; the cell-centered Ez never
    // enters the 1D curl.
    return {&node_inertia, &node_inertia, nullptr};
#elif defined(WARPX_DIM_RZ)
    amrex::MultiFab& radial_inertia =
        *m_WarpX->m_fields.get(InertiaCoefficientE0Name, 0);
    amrex::MultiFab& azimuthal_inertia =
        *m_WarpX->m_fields.get(InertiaCoefficientE1Name, 0);
    amrex::MultiFab& axial_inertia =
        *m_WarpX->m_fields.get(InertiaCoefficientE2Name, 0);
    for (amrex::MFIter mfi(azimuthal_inertia); mfi.isValid(); ++mfi) {
        const auto inertia_radial = radial_inertia.array(mfi);
        const auto inertia_azimuthal = azimuthal_inertia.array(mfi);
        const auto inertia_axial = axial_inertia.array(mfi);
        const auto rho = density.const_array(mfi);
        // E_theta corners: the nodal charge density.
        amrex::ParallelFor(
            amrex::convert(mfi.validbox(), amrex::IntVect(1, 1)),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const auto node_density = [&] (const int in, const int jn) {
                    return 0.25_rt * (rho(in - 1, jn - 1, k) + rho(in, jn - 1, k) +
                                      rho(in - 1, jn, k) + rho(in, jn, k));
                };
                inertia_azimuthal(i, j, k) = me_over_e /
                    amrex::max(charge_to_mass * node_density(i, j),
                               charge_density_floor);
            });
        // Er z-faces: nodes averaged in r.
        amrex::ParallelFor(
            amrex::convert(mfi.validbox(), amrex::IntVect(0, 1)),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const auto node_density = [&] (const int in, const int jn) {
                    return 0.25_rt * (rho(in - 1, jn - 1, k) + rho(in, jn - 1, k) +
                                      rho(in - 1, jn, k) + rho(in, jn, k));
                };
                inertia_radial(i, j, k) = me_over_e /
                    amrex::max(charge_to_mass * 0.5_rt *
                                   (node_density(i, j) + node_density(i + 1, j)),
                               charge_density_floor);
            });
        // Ez r-faces: nodes averaged in z.
        amrex::ParallelFor(
            amrex::convert(mfi.validbox(), amrex::IntVect(1, 0)),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const auto node_density = [&] (const int in, const int jn) {
                    return 0.25_rt * (rho(in - 1, jn - 1, k) + rho(in, jn - 1, k) +
                                      rho(in - 1, jn, k) + rho(in, jn, k));
                };
                inertia_axial(i, j, k) = me_over_e /
                    amrex::max(charge_to_mass * 0.5_rt *
                                   (node_density(i, j) + node_density(i, j + 1)),
                               charge_density_floor);
            });
    }
    return {&radial_inertia, &azimuthal_inertia, &axial_inertia};
#else
    amrex::ignore_unused(density, charge_to_mass, charge_density_floor,
                         me_over_e);
    return {nullptr, nullptr, nullptr};
#endif
}

void ThetaImplicitMHD::PrintParameters () const
{
    if (!m_WarpX->Verbose()) {
        return;
    }
    // Dust-gate note on the electron-inertia banner line (see
    // hybrid_pic_model.electron_inertia_linear_below), only when active.
    std::string inertia_gate_note;
    if (m_hybrid_pic_model->m_electron_inertia_linear_below > 0.0_rt) {
        std::ostringstream gate_stream;
        gate_stream << " (linear below rho = "
                    << m_hybrid_pic_model->m_electron_inertia_linear_below
                    << " kg/m3)";
        inertia_gate_note = gate_stream.str();
    }
    amrex::Print() << "\n"
                   << "-----------------------------------------------------------\n"
                   << "-------- THETA IMPLICIT SINGLE-FLUID MHD PARAMETERS -------\n"
                   << "-----------------------------------------------------------\n"
                   << "Theta:                         " << m_theta << "\n"
                   << "Ion charge-to-mass [C/kg]:     " << m_ion_charge_to_mass << "\n"
                   << "Electron gamma:                " << m_gamma_e << "\n"
                   << "Ion gamma:                     " << m_gamma_i << "\n"
                   << "Mass-density floor [kg/m^3]:   " << m_mass_density_floor << "\n"
                   << "Ohm density guard [kg/m^3]:    " << OhmMassDensityFloor() << "\n"
                   << "Ion temperature floor [K]:     " << m_ion_temperature_floor << "\n"
                   << "Electron temp. floor [K]:      " << m_electron_temperature_floor << "\n"
                   << "Hall term:                     "
                   << m_hybrid_pic_model->m_include_hall_term
                   << "\n"
                   << "Electron-inertia Ohm term:     "
                   << m_hybrid_pic_model->m_include_electron_inertia
                   << inertia_gate_note << "\n"
                   << "Effective electron mass [kg]:  "
                   << m_hybrid_pic_model->m_electron_inertia_mass << "\n"
                   << "Electron-pressure Ohm term:    "
                   << m_hybrid_pic_model->m_include_electron_pressure_term << "\n"
                   << "External vector potential:     "
                   << m_hybrid_pic_model->m_add_external_fields << "\n"
                   // Circuit-hook scope, only when the coupling is live.
                   << (m_external_field_iteration
                           ? std::string(
                                 "Circuit hook scope:            ") +
                                 (m_circuit_hook_newton_scope ? "newton"
                                                              : "residual") +
                                 "\n"
                           : std::string{})
                   << "Evolve ion fluid:              " << m_evolve_ion_fluid << "\n"
                   << "Fluid flux:                    " << m_fluid_flux << "\n"
                   << "Shaped conducting-wall mask:   "
                   << m_wall_mask.ModeName() << "\n"
                   << "Wall thermal BC:               "
                   << m_wall_mask.ThermalBCName()
                   << (m_wall_mask.GetThermalBC() ==
                               ImplicitMHDWallMask::ThermalBC::temperature
                           ? " (T_wall = " +
                                 std::to_string(
                                     m_wall_mask.WallTemperature_eV()) +
                                 " eV)"
                           : std::string{})
                   << "\n"
                   << "Open-r fluid wall:             " << m_r_open_fluid << "\n"
                   << "Z-end fluid boundary:          " << m_z_boundary_fluid << "\n"
                   // Boot-gate line of the z_lo mirror (verbatim contract).
                   << (m_z_lo_pmc
                           ? "Z-lo fluid boundary:           symmetry "
                             "(mirror; PMC fields)\n"
                           : "")
                   << "Z wall temperature [eV]:       " << m_z_wall_temperature << "\n"
                   << "HLLC signal closure:           " << m_hllc_signal_closure << "\n"
                   << "HLLC contact blend:            " << m_hllc_contact_blend << "\n";
    if (m_use_hlld) {
        amrex::Print() << "HLLD fan closure:              " << m_hlld_fan_closure << "\n"
                       << "HLLD ion-energy flux:          " << m_hlld_ion_energy_flux << "\n"
                       << "HLLD all-speed correction:     " << (m_hlld_all_speed ? "true" : "false") << "\n"
                       << "HLLD kappa signal/contact:     " << m_hlld_kappa_signal
                       << " " << m_hlld_kappa_contact << "\n"
                       << "HLLD kappa bn/denominator:     " << m_hlld_kappa_bn
                       << " " << m_hlld_kappa_denominator << "\n";
    }
    amrex::Print()
                   << "Ion closure:                   " << m_ion_closure << "\n"
                   << "Viscosity [m2/s]:              " << m_viscosity << "\n"
                   << "Wall viscosity mask:           "
                   << (m_wall_viscosity_mask
                           ? "ON (width " +
                                 std::to_string(m_wall_viscosity_mask_width) +
                                 " cells)"
                           : std::string("off"))
                   << "\n"
                   << "Thermal diffusivity i/e [m2/s]: "
                   << (m_chi_ion_is_parser
                           ? std::string("chi_i(rho,Te,Ti,J,t) = ") +
                                 m_chi_ion_expression
                           : std::to_string(m_thermal_diffusivity_ion))
                   << " | "
                   << (m_chi_electron_is_parser
                           ? std::string("chi_e(rho,Te,Ti,J,t) = ") +
                                 m_chi_electron_expression
                           : std::to_string(m_thermal_diffusivity_electron))
                   << "\n"
                   << "Conduction flux limiter:       "
                   << (m_conduction_flux_limit_factor > 0.0_rt
                           ? "ON, chi/(1 + |q|/(f q_fs)), f = " +
                                 std::to_string(
                                     m_conduction_flux_limit_factor)
                           : std::string("OFF (factor 0)"))
                   << "\n"
                   << "Conduction coefficient state:  "
                   << (m_conduction_coefficient_step_old ? "step_old"
                                                         : "theta")
                   << "\n"
                   << "Thermal conduction model:      "
                   << m_thermal_conduction_model
                   << (m_conduction_braginskii
                           ? " (lnLambda = " +
                                 std::to_string(m_conduction_coulomb_log) +
                                 ", chi min/max [m2/s] = " +
                                 std::to_string(m_conduction_chi_min) +
                                 " / " +
                                 (m_conduction_chi_max > 0.0_rt
                                      ? std::to_string(m_conduction_chi_max)
                                      : std::string("off")) +
                                 (m_conduction_qs_chi > 0.0_rt
                                      ? ", qs chi = " +
                                            std::to_string(
                                                m_conduction_qs_chi) +
                                            " @ onset " +
                                            std::to_string(
                                                m_conduction_qs_onset)
                                      : std::string{}) +
                                 ")"
                           : std::string{})
                   << "\n"
                   << "Pressure corner width fraction: "
                   << m_pressure_corner_width_fraction << "\n"
                   << "Positivity step safety:        " << m_positivity_safety << "\n"
                   << "Joule heating:                 " << m_include_joule_heating << "\n"
                   << "Joule Ohm-current quench:      " << m_joule_ohm_current << "\n"
                   << "e-i equilibration scale:       "
                   << m_electron_ion_equilibration
                   << (m_electron_ion_equilibration > 0.0_rt
                           ? " (the reference code's eq_brate rate, step-old frozen)"
                           : " (off)")
                   << "\n"
                   << "Resistive theta:               " << m_resistive_theta << "\n"
                   << "Conduction theta:              " << m_conduction_theta
                   << "\n"
                   << "Vacuum eta diffusivity [m2/s]: "
                   << m_vacuum_resistivity_diffusivity << "\n"
                   << "Vacuum reference peak fraction: "
                   << m_vacuum_reference_peak_fraction
                   << (m_vacuum_reference_peak_fraction > 0.0_rt
                           ? " (dynamic en0, refreshed per step)"
                           : " (static Ohm-guard reference)")
                   << "\n"
                   << "Conduction halo boost [m2/s]:  "
                   << m_conduction_halo_boost << "\n"
                   << "Halo pedestal fraction:        " << m_halo_pedestal_fraction
                   << "\n"
                   << "Halo pedestal drag rate [1/s]: " << m_halo_pedestal_drag_rate
                   << "\n"
                   << "Halo pedestal energy rate [1/s]: "
                   << m_halo_pedestal_energy_rate << "\n";
    if (m_floor_consistency_rate > 0.0_rt) {
        amrex::Print() << "Floor consistency rate [1/s]:  "
                       << m_floor_consistency_rate
                       << " (per-solve cap 1/(theta dt))\n"
                       << "Floor consistency width:       "
                       << m_floor_consistency_width_fraction
                       << " of the bound (capacity rate_eff*w/2)\n"
                       << "Floor consistency ledger:      "
                       << (m_floor_ledger_file.empty() ? "(none)"
                                                       : m_floor_ledger_file)
                       << "\n";
    }
    if (m_density_eater_rate > 0.0_rt) {
        amrex::Print() << "Density eater rate [1/step]:   "
                       << m_density_eater_rate << "\n"
                       << "Density eater target fraction: "
                       << m_density_eater_target_fraction << " of "
                       << (m_density_eater_reference_density > 0.0_rt
                               ? m_density_eater_reference_density
                               : m_reference_mass_density)
                       << " kg/m^3"
                       << (m_density_eater_reference_peak_fraction > 0.0_rt
                               ? " (dynamic en0, refreshed per step)"
                               : " (static reference)")
                       << "\n"
                       << "Density eater band:            "
                       << m_density_eater_band << " ("
                       << (m_density_eater_band_cells > 0
                               ? std::to_string(m_density_eater_band_cells)
                               : std::string("the reference code's grid rule"))
                       << " cells), flux gate "
                       << m_density_eater_flux_sign << "\n"
                       << "Density eater ledger:          "
                       << (m_density_eater_ledger_file.empty()
                               ? "(none)"
                               : m_density_eater_ledger_file)
                       << "\n";
    }
    if (m_ion_closure == "dual_energy") {
        amrex::Print() << "Dual-energy internal cutoff:   "
                       << m_dual_energy_internal_cutoff << "\n"
                       << "Dual-energy sync threshold:    "
                       << m_dual_energy_sync_threshold << "\n";
    }
    if (m_ion_closure == "cgl") {
        amrex::Print() << "CGL relaxation scale:          " << m_cgl_relaxation_scale << "\n"
                       << "CGL Coulomb logarithm:         " << m_cgl_coulomb_log << "\n"
                       << "CGL instability scale/width:   " << m_cgl_instability_scale
                       << " " << m_cgl_instability_width << "\n"
                       << "CGL null-blend scale:          " << m_cgl_null_scale << "\n";
    }
    PrintBaseImplicitSolverParameters();
    m_nlsolver->PrintParams();
    amrex::Print() << "-----------------------------------------------------------\n\n";
}

void ThetaImplicitMHD::InitializeFluidState ()
{
    amrex::MultiFab& density = *m_WarpX->m_fields.get(MassDensityName, 0);
    amrex::MultiFab& momentum = *m_WarpX->m_fields.get(MomentumDensityName, 0);
    amrex::MultiFab& electron_energy = *m_WarpX->m_fields.get(ElectronEnergyName, 0);
    amrex::MultiFab& ion_energy = *m_WarpX->m_fields.get(IonEnergyName, 0);
    amrex::MultiFab& ion_parallel_energy =
        *m_WarpX->m_fields.get(IonParallelEnergyName, 0);
    amrex::MultiFab& ion_perp_energy =
        *m_WarpX->m_fields.get(IonPerpEnergyName, 0);
    amrex::MultiFab& ion_internal_energy =
        *m_WarpX->m_fields.get(IonInternalEnergyName, 0);

    const auto cell_size = m_WarpX->Geom(0).CellSizeArray();
    const auto lower = m_WarpX->Geom(0).ProbLoArray();
    const auto mass_density = m_mass_density;
    const auto velocity_x = m_velocity_x;
    const auto velocity_y = m_velocity_y;
    const auto velocity_z = m_velocity_z;
    const auto electron_pressure = m_electron_pressure;
    const auto ion_pressure = m_ion_pressure;
    const auto ion_anisotropy = m_ion_anisotropy;
    const amrex::Real density_floor = m_mass_density_floor;
    const amrex::Real pressure_floor = m_electron_pressure_floor;
    const amrex::Real gamma_e_minus_one = m_gamma_e - 1.0_rt;
    const bool dual_energy_closure = m_ion_closure == "dual_energy";
    // Dual-energy fills the conservative E_i exactly as total_energy
    // does, plus the consistent auxiliary internal energy U_i.
    const bool total_energy_closure =
        m_ion_closure == "total_energy" || dual_energy_closure;
    const bool cgl_closure = m_ion_closure == "cgl";
    const amrex::Real ion_pressure_floor = m_ion_pressure_floor;
    const amrex::Real gamma_i_minus_one = m_gamma_i - 1.0_rt;

    for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto density_array = density.array(mfi);
        const auto momentum_array = momentum.array(mfi);
        const auto energy_array = electron_energy.array(mfi);
        const auto ion_energy_array = ion_energy.array(mfi);
        const auto ion_parallel_array = ion_parallel_energy.array(mfi);
        const auto ion_perp_array = ion_perp_energy.array(mfi);
        const auto ion_internal_array = ion_internal_energy.array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
#if defined(WARPX_DIM_3D)
            const amrex::Real x = lower[0] + (i + 0.5_rt) * cell_size[0];
            const amrex::Real y = lower[1] + (j + 0.5_rt) * cell_size[1];
            const amrex::Real z = lower[2] + (k + 0.5_rt) * cell_size[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                const amrex::Real x = lower[0] + (i + 0.5_rt) * cell_size[0];
                const amrex::Real y = 0.0_rt;
                const amrex::Real z = lower[1] + (j + 0.5_rt) * cell_size[1];
#else
                const amrex::Real x = 0.0_rt;
                const amrex::Real y = 0.0_rt;
                const amrex::Real z = lower[0] + (i + 0.5_rt) * cell_size[0];
#endif
            const amrex::Real rho = std::max(mass_density(x, y, z), density_floor);
            const amrex::Real vx = velocity_x(x, y, z);
            const amrex::Real vy = velocity_y(x, y, z);
            const amrex::Real vz = velocity_z(x, y, z);
            density_array(i, j, k) = rho;
            momentum_array(i, j, k, 0) = rho * vx;
            momentum_array(i, j, k, 1) = rho * vy;
            momentum_array(i, j, k, 2) = rho * vz;
            energy_array(i, j, k) =
                std::max(electron_pressure(x, y, z), pressure_floor) / gamma_e_minus_one;
            if (total_energy_closure) {
                ion_energy_array(i, j, k) =
                    std::max(ion_pressure(x, y, z), ion_pressure_floor) / gamma_i_minus_one +
                    0.5_rt * rho * (vx * vx + vy * vy + vz * vz);
            }
            if (dual_energy_closure) {
                // Consistent split-energy load: U_i is the internal part
                // of the E_i fill above, so fk-blended and recovered
                // pressures agree exactly at t = 0.
                ion_internal_array(i, j, k) =
                    std::max(ion_pressure(x, y, z), ion_pressure_floor) /
                    gamma_i_minus_one;
            }
            if (cgl_closure) {
                // Bi-Maxwellian initialization with the effective
                // pressure pinned to ion_pressure and the optional
                // anisotropy A = p_perp/p_par (default 1, isotropic):
                // p_par = 3 p_i/(1 + 2A), p_perp = A p_par; U_par =
                // p_par/2, U_perp = p_perp. No kinetic term -- the CGL
                // blocks are pure internal energies (the kinetic energy
                // lives in the momentum block only).
                const amrex::Real p_i =
                    std::max(ion_pressure(x, y, z), ion_pressure_floor);
                const amrex::Real anisotropy =
                    std::max(ion_anisotropy(x, y, z), 0.0_rt);
                const amrex::Real p_par =
                    3.0_rt * p_i / (1.0_rt + 2.0_rt * anisotropy);
                ion_parallel_array(i, j, k) = std::max(
                    0.5_rt * p_par, 0.5_rt * ion_pressure_floor);
                ion_perp_array(i, j, k) =
                    std::max(anisotropy * p_par, ion_pressure_floor);
            }
        });
    }

    density.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    momentum.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    electron_energy.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    ion_energy.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    ion_parallel_energy.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    ion_perp_energy.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    ion_internal_energy.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    ApplyFluidDomainBoundaries(density, momentum, electron_energy, ion_energy,
                               ion_parallel_energy, ion_perp_energy,
                               ion_internal_energy);
    PublishMomentumComponents();
}

void ThetaImplicitMHD::ApplyFluidDomainBoundaries (amrex::MultiFab& density,
                                                   amrex::MultiFab& momentum,
                                                   amrex::MultiFab& electron_energy,
                                                   amrex::MultiFab& ion_energy,
                                                   amrex::MultiFab& ion_parallel_energy,
                                                   amrex::MultiFab& ion_perp_energy,
                                                   amrex::MultiFab& ion_internal_energy) const
{
#if defined(WARPX_DIM_RZ)
    // Fill radial domain ghost cells: mirror parity across the axis (odd u_r
    // and u_theta, even scalars and u_z) and, at r_max, either a reflecting
    // no-normal-flow conducting wall (odd u_r, even everything else; the
    // PEC field boundary) or zero-gradient outflow ghosts (the open
    // Green's-function field boundary). Together with the r-weighted face
    // fluxes the mirror yields zero mass/energy flux and the correct
    // pressure force at the wall. Periodic z is handled by
    // FillBoundaryAndSync; non-periodic z gets zero-gradient (outflow)
    // ghosts first, so the radial fill below also propagates them into
    // the corner ghosts.
    if (m_z_neumann) {
        ApplyNeumannZDomainGhosts(density);
        ApplyNeumannZDomainGhosts(momentum);
        if (m_z_outflow_no_reflux) {
            // No-reflux outflow (opt-in): the zero-gradient ghosts are
            // passive and admit inflow when the interior pulls inward
            // (e.g. mirror flux compression dragging the end columns
            // through the throats); clamping the ghost AXIAL momentum
            // to the outward sign permits drainage but forbids the
            // boundary feeding plasma back. Scalars stay zero-gradient.
            const amrex::Box domain = amrex::convert(
                m_WarpX->Geom(0).Domain(), momentum.ixType().toIntVect());
            const int domain_lo = domain.smallEnd(1);
            const int domain_hi = domain.bigEnd(1);
            // The z_lo mirror face is a zero-flux symmetry plane, not an
            // outflow end: its ghost rows carry the parity image (filled
            // below) and must not be sign-clamped.
            const bool skip_z_lo = m_z_lo_pmc;
            for (amrex::MFIter mfi(momentum); mfi.isValid(); ++mfi) {
                const amrex::Box grown =
                    amrex::grow(mfi.validbox(), momentum.nGrowVect());
                if (grown.smallEnd(1) >= domain_lo &&
                    grown.bigEnd(1) <= domain_hi) {
                    continue;
                }
                const auto arr = momentum.array(mfi);
                amrex::ParallelFor(grown,
                                   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (j < domain_lo && !skip_z_lo) {
                        arr(i, j, k, 2) = std::min(arr(i, j, k, 2), 0.0_rt);
                    } else if (j > domain_hi) {
                        arr(i, j, k, 2) = std::max(arr(i, j, k, 2), 0.0_rt);
                    }
                });
            }
        }
        ApplyNeumannZDomainGhosts(electron_energy);
        ApplyNeumannZDomainGhosts(ion_energy);
        ApplyNeumannZDomainGhosts(ion_parallel_energy);
        ApplyNeumannZDomainGhosts(ion_perp_energy);
        ApplyNeumannZDomainGhosts(ion_internal_energy);
        if (m_z_boundary_fluid != "neumann") {
            // Selectable z-end fluid ghosts (recast path). The Neumann
            // fill above stays the base image -- same density and
            // momentum, so the end-face fan carries interior signal
            // speeds -- and the selected mode overrides the channel
            // that makes the plain clamp unphysical:
            //  * outflow: the ghost AXIAL momentum is C-infinity
            //    rectified to its OUTGOING part -- the exact recipe of
            //    the r_max absorbing wall, the smoothed twin of the
            //    z_outflow_no_reflux clamp -- so the ends advect mass
            //    and energy out at interior values but never feed
            //    plasma back (the Neumann fill is zero-flux for the
            //    energies and an injector under end-ward pull).
            //  * wall_temperature: the ghost ENERGIES are set to
            //    z_wall_temperature at the ghost density -- the
            //    electron ghost carries the wall internal energy, the
            //    ion ghost the wall internal energy PLUS the kinetic
            //    energy of the copied ghost momentum, so the condition
            //    acts on the THERMAL content only. The end faces then
            //    exchange advectively against a T_wall reservoir (the
            //    z analog of the r-wall temperature anchoring).
            // Runs before the radial pass below so the corner ghosts
            // inherit the overridden rows, in EVERY residual
            // evaluation: the fill is part of the JFNK residual and
            // must stay C-infinity in the state (no hard flow-
            // direction branches; the rectifier exists for exactly
            // that).
            const bool z_outflow = (m_z_boundary_fluid == "outflow");
            const bool dual_energy_closure =
                (m_ion_closure == "dual_energy");
            // Dual-energy: the E_i ghost takes the total_energy recipe
            // (wall internal + kinetic of the copied momentum) and the
            // U_i ghost carries the wall internal energy alone.
            const bool total_energy_closure =
                (m_ion_closure == "total_energy") || dual_energy_closure;
            const bool cgl_closure = (m_ion_closure == "cgl");
            // n kB T_wall = rho (q/m) T_wall[eV] for the quasi-neutral
            // single-ion fluid (n = rho (q/m)/e and kB T = e T[eV]).
            const amrex::Real wall_pressure_per_density =
                m_ion_charge_to_mass * m_z_wall_temperature;
            const amrex::Real z_gamma_e = m_gamma_e;
            const amrex::Real z_gamma_i = m_gamma_i;
            const amrex::Real z_rectifier_kappa = m_hlld_kappa_signal;
            const amrex::Real z_electron_energy_floor =
                m_electron_pressure_floor / (m_gamma_e - 1.0_rt);
            const amrex::Real z_density_floor = m_mass_density_floor;
            const amrex::Box& cc_domain = m_WarpX->Geom(0).Domain();
            const int z_lo = cc_domain.smallEnd(1);
            const int z_hi = cc_domain.bigEnd(1);
            // On a z_lo mirror the override applies at z_hi only: the
            // z_lo ghost rows keep the symmetry-plane parity image
            // (filled below), never an outflow/reservoir state.
            const bool skip_z_lo = m_z_lo_pmc;
            for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
                const amrex::Box grown =
                    amrex::grow(mfi.validbox(), density.nGrowVect());
                if (grown.smallEnd(1) >= z_lo && grown.bigEnd(1) <= z_hi) {
                    continue;
                }
                const auto rho = density.array(mfi);
                const auto mom = momentum.array(mfi);
                const auto energy = electron_energy.array(mfi);
                const auto ion_e = ion_energy.array(mfi);
                const auto ion_par = ion_parallel_energy.array(mfi);
                const auto ion_perp = ion_perp_energy.array(mfi);
                const auto ion_int = ion_internal_energy.array(mfi);
                amrex::ParallelFor(grown,
                                   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if ((j >= z_lo && j <= z_hi) || (skip_z_lo && j < z_lo)) {
                        return;
                    }
                    if (z_outflow) {
                        // rectifier width = the acoustic momentum scale
                        // kappa_signal sqrt(gamma_e p_e rho) of the
                        // (copied) end cell, exactly as at the r_max
                        // absorbing wall: genuine outflow passes
                        // asymptotically unchanged and the map stays
                        // smooth for the JFNK probes
                        const amrex::Real width =
                            z_rectifier_kappa *
                            std::sqrt(z_gamma_e * (z_gamma_e - 1.0_rt) *
                                      std::max(energy(i, j, k),
                                               z_electron_energy_floor) *
                                      std::max(rho(i, j, k),
                                               z_density_floor));
                        const amrex::Real outward =
                            (j > z_hi) ? 1.0_rt : -1.0_rt;
                        const amrex::Real axial = mom(i, j, k, 2);
                        mom(i, j, k, 2) =
                            axial * 0.5_rt *
                            (1.0_rt +
                             outward * theta_implicit_mhd::smooth_sign(
                                           axial, width));
                    } else {
                        const amrex::Real wall_pressure =
                            wall_pressure_per_density * rho(i, j, k);
                        energy(i, j, k) =
                            wall_pressure / (z_gamma_e - 1.0_rt);
                        if (total_energy_closure) {
                            amrex::Real kinetic_energy = 0.0_rt;
                            for (int component = 0; component < 3;
                                 ++component) {
                                kinetic_energy +=
                                    mom(i, j, k, component) *
                                    mom(i, j, k, component);
                            }
                            kinetic_energy *=
                                0.5_rt /
                                std::max(rho(i, j, k), z_density_floor);
                            ion_e(i, j, k) =
                                wall_pressure / (z_gamma_i - 1.0_rt) +
                                kinetic_energy;
                            if (dual_energy_closure) {
                                ion_int(i, j, k) =
                                    wall_pressure / (z_gamma_i - 1.0_rt);
                            }
                        } else if (cgl_closure) {
                            // isotropic wall state p_par = p_perp =
                            // p_wall: U_par = p_par/2, U_perp = p_perp
                            // (pure internal energies, no kinetic term
                            // in the CGL blocks)
                            ion_par(i, j, k) = 0.5_rt * wall_pressure;
                            ion_perp(i, j, k) = wall_pressure;
                        }
                        // barotropic: no evolved ion energy block
                    }
                });
            }
        }
        if (m_z_lo_pmc) {
            // Mirror-symmetry plane at z_lo (z_lo_boundary_fluid =
            // symmetry, paired with the pmc field boundary): the z_lo
            // ghost rows are replaced by the exact linear reflection of
            // the interior -- even scalars and tangential momentum, ODD
            // axial momentum -- INSTEAD of the Neumann base image, so
            // the z_lo boundary faces see symmetric Riemann states (zero
            // mass/energy advective flux, pure pressure normal-momentum
            // flux) and the conduction stage sees a zero normal
            // temperature gradient. The fill is linear in the state
            // (C-infinity for the JFNK probes), runs in EVERY residual
            // evaluation, and precedes the radial pass below so the
            // corner ghosts inherit the mirrored rows. z_hi keeps the
            // Neumann image and the z_boundary_fluid override above.
            ApplyMirrorZLoDomainGhosts(density, {1, 1, 1});
            ApplyMirrorZLoDomainGhosts(momentum, {1, 1, -1});
            ApplyMirrorZLoDomainGhosts(electron_energy, {1, 1, 1});
            ApplyMirrorZLoDomainGhosts(ion_energy, {1, 1, 1});
            ApplyMirrorZLoDomainGhosts(ion_parallel_energy, {1, 1, 1});
            ApplyMirrorZLoDomainGhosts(ion_perp_energy, {1, 1, 1});
            ApplyMirrorZLoDomainGhosts(ion_internal_energy, {1, 1, 1});
        }
    }
    const amrex::Box& domain = m_WarpX->Geom(0).Domain();
    const int domain_lo = domain.smallEnd(0);
    const int domain_hi = domain.bigEnd(0);
    // The absorbing wall (r_open_fluid = absorb) keeps the outflow ghost
    // recipe -- the zero-gradient clamp is the impedance-MATCHED image
    // (no thermodynamic jump at the wall face, so the wall-face fan
    // carries wall-plasma signal speeds only) -- with ONE difference:
    // the ghost NORMAL momentum is C-infinity rectified to its outward
    // part (the smoothed twin of the z-outflow no-reflux clamp), so the
    // ghost image never advances toward the domain. Plasma incident on
    // the wall is admitted at its signal-limited rate, but a wall band
    // retreating from the wall separates from a HELD ghost instead of
    // dragging a refilling one back in: absorbed, not stored, and never
    // returned. The rectifier width is the acoustic momentum scale
    // kappa_signal * sqrt(gamma_e p_e rho) of the wall cell, so genuine
    // outflow passes asymptotically unchanged while the map stays
    // smooth for the JFNK probes. (A vacuum-density absorber image
    // receding at v_A was tried and retired: its pedestal-density fan
    // speeds, ~30x the plasma Alfven speed, poison the wall-ring
    // corner-EMF dissipation and the contact-speed estimate the moment
    // the drain gates open -- the deterministic scrape-off-arrival
    // blowup of the FRC mirror benchmark; a flux-register one-way valve
    // was tried and retired too: its rectifier is non-monotone in the
    // flux, an anti-dissipative response band that froze the Newton
    // line search at scrape-off arrival. Both are reproduced by
    // test_rz_theta_implicit_mhd_absorb_scrape_off.)
    const bool r_open = m_r_open && (m_r_open_fluid != "reflect");
    const bool r_absorb = m_r_open && (m_r_open_fluid == "absorb");
    const amrex::Real rectifier_kappa = m_hlld_kappa_signal;
    const amrex::Real gamma_e = m_gamma_e;
    const amrex::Real electron_energy_floor =
        m_electron_pressure_floor / (m_gamma_e - 1.0_rt);
    const amrex::Real density_floor = m_mass_density_floor;
    for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
        const amrex::Box grown = amrex::grow(mfi.validbox(), density.nGrowVect());
        if (grown.smallEnd(0) >= domain_lo && grown.bigEnd(0) <= domain_hi) {
            continue;
        }
        const auto rho = density.array(mfi);
        const auto mom = momentum.array(mfi);
        const auto energy = electron_energy.array(mfi);
        const auto ion_e = ion_energy.array(mfi);
        const auto ion_par = ion_parallel_energy.array(mfi);
        const auto ion_perp = ion_perp_energy.array(mfi);
        const auto ion_int = ion_internal_energy.array(mfi);
        amrex::ParallelFor(grown, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if (i < domain_lo) {
                const int mirror = 2 * domain_lo - 1 - i;
                rho(i, j, k) = rho(mirror, j, k);
                energy(i, j, k) = energy(mirror, j, k);
                ion_e(i, j, k) = ion_e(mirror, j, k);
                ion_par(i, j, k) = ion_par(mirror, j, k);
                ion_perp(i, j, k) = ion_perp(mirror, j, k);
                ion_int(i, j, k) = ion_int(mirror, j, k);
                mom(i, j, k, 0) = -mom(mirror, j, k, 0);
                mom(i, j, k, 1) = -mom(mirror, j, k, 1);
                mom(i, j, k, 2) = mom(mirror, j, k, 2);
            } else if (i > domain_hi) {
                // open boundary: clamp to the edge cell (zero-gradient
                // outflow; the absorbing wall rectifies the clamp's
                // normal momentum, see the r_absorb comment above); PEC
                // wall: reflect with odd normal momentum
                const int mirror =
                    r_open ? domain_hi : 2 * domain_hi + 1 - i;
                rho(i, j, k) = rho(mirror, j, k);
                energy(i, j, k) = energy(mirror, j, k);
                ion_e(i, j, k) = ion_e(mirror, j, k);
                ion_par(i, j, k) = ion_par(mirror, j, k);
                ion_perp(i, j, k) = ion_perp(mirror, j, k);
                ion_int(i, j, k) = ion_int(mirror, j, k);
                amrex::Real normal_momentum =
                    r_open ? mom(mirror, j, k, 0) : -mom(mirror, j, k, 0);
                if (r_absorb) {
                    const amrex::Real width =
                        rectifier_kappa *
                        std::sqrt(gamma_e * (gamma_e - 1.0_rt) *
                                  std::max(energy(mirror, j, k),
                                           electron_energy_floor) *
                                  std::max(rho(mirror, j, k),
                                           density_floor));
                    normal_momentum =
                        normal_momentum * 0.5_rt *
                        (1.0_rt + theta_implicit_mhd::smooth_sign(
                                      normal_momentum, width));
                }
                mom(i, j, k, 0) = normal_momentum;
                mom(i, j, k, 1) = mom(mirror, j, k, 1);
                mom(i, j, k, 2) = mom(mirror, j, k, 2);
            }
        });
    }
#else
    amrex::ignore_unused(density, momentum, electron_energy, ion_energy,
                         ion_parallel_energy, ion_perp_energy,
                         ion_internal_energy);
#endif
}

void ThetaImplicitMHD::ApplyNeumannZDomainGhosts (amrex::MultiFab& mf,
                                                  const int a_open_face_keep_rows) const
{
#if defined(WARPX_DIM_RZ)
    // Zero-gradient extrapolation into the axial domain ghosts, staggering
    // aware: cell-centered-in-z data clamps to the boundary cell (zero
    // gradient across the boundary face) while NODAL-in-z data mirrors
    // EVENLY across the boundary node itself, ghost(j) = f(2 jb - j) --
    // the clamp would leave a spurious half-gradient at the end node,
    // where a centered derivative reads (f_1 - f_0)/(2 dz) instead of 0
    // (feeding e.g. a spurious end-node grad_z of the nodal electron
    // pressure/temperature, charge density, and currents). Same
    // convention as the radial fill (ApplyScalarRadialDomainGhosts:
    // nodal-in-r mirrors across the boundary node). Used for every field
    // the residual stencils read when the z ends are outflow rather
    // than periodic; FillBoundary leaves those ghosts untouched. On an
    // OPEN z face the first a_open_face_keep_rows ghost rows already hold
    // free-space (Green's-consistent) values and stay untouched; deeper
    // rows clamp/mirror about the outermost kept row, so nothing
    // re-imposes the z-invariant continuation the open cap replaces.
    if (!m_z_neumann) {
        return;
    }
    const amrex::Box domain =
        amrex::convert(m_WarpX->Geom(0).Domain(), mf.ixType().toIntVect());
    const int clamp_lo =
        domain.smallEnd(1) - (m_z_lo_open ? a_open_face_keep_rows : 0);
    const int clamp_hi =
        domain.bigEnd(1) + (m_z_hi_open ? a_open_face_keep_rows : 0);
    const bool nodal_z = mf.ixType().nodeCentered(1);
    const int ncomp = mf.nComp();
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const amrex::Box grown = amrex::grow(mfi.validbox(), mf.nGrowVect());
        if (grown.smallEnd(1) >= clamp_lo && grown.bigEnd(1) <= clamp_hi) {
            continue;
        }
        const auto arr = mf.array(mfi);
        amrex::ParallelFor(grown, ncomp,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
            // mirror indices stay inside [clamp_lo, clamp_hi] (deep
            // ghosts of a thin domain fall back to the far anchor row),
            // so the written ghost rows never source each other
            if (j < clamp_lo) {
                const int source =
                    nodal_z ? amrex::min(2 * clamp_lo - j, clamp_hi)
                            : clamp_lo;
                arr(i, j, k, n) = arr(i, source, k, n);
            } else if (j > clamp_hi) {
                const int source =
                    nodal_z ? amrex::max(2 * clamp_hi - j, clamp_lo)
                            : clamp_hi;
                arr(i, j, k, n) = arr(i, source, k, n);
            }
        });
    }
#else
    amrex::ignore_unused(mf, a_open_face_keep_rows);
#endif
}

void ThetaImplicitMHD::ApplyMirrorZLoDomainGhosts (
    amrex::MultiFab& mf, const amrex::GpuArray<int, 3>& a_component_parity) const
{
#if defined(WARPX_DIM_RZ)
    // Mirror-parity fill of the z_lo (PMC/symmetry-plane) domain ghosts:
    // the exact linear reflection of the interior, so the flux/EMF/Ohm
    // stencils see the half domain as the upper half of a mirror-
    // symmetric full domain. Staggering aware: cell-centered-in-z data
    // reflects across the boundary FACE (ghost z_lo - 1 - g takes
    // parity * value at z_lo + g; the Neumann pass's clamp is only the
    // g = 0 image), nodal-in-z data reflects across the boundary NODE
    // (ghost z_lo - g takes parity * value at z_lo + g, identical to
    // the Neumann pass's even mirror for parity +1) and an odd nodal
    // component is zeroed ON the plane node. Runs AFTER the Neumann
    // pass of the same field, which keeps ownership of z_hi. Deep
    // ghosts of a degenerate-thin domain fall back to the far anchor
    // row like the Neumann fill (never sourcing other written ghosts).
    if (!m_z_lo_pmc) {
        return;
    }
    const amrex::Box domain =
        amrex::convert(m_WarpX->Geom(0).Domain(), mf.ixType().toIntVect());
    const int plane = domain.smallEnd(1);
    const int domain_hi = domain.bigEnd(1);
    const bool nodal_z = mf.ixType().nodeCentered(1);
    const int ncomp = mf.nComp();
    AMREX_ALWAYS_ASSERT(ncomp <= 3);
    const amrex::GpuArray<int, 3> parity = a_component_parity;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const amrex::Box grown = amrex::grow(mfi.validbox(), mf.nGrowVect());
        if (grown.smallEnd(1) > plane) {
            continue;
        }
        const auto arr = mf.array(mfi);
        amrex::ParallelFor(grown, ncomp,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
            if (j < plane) {
                const int source =
                    nodal_z ? amrex::min(2 * plane - j, domain_hi)
                            : amrex::min(2 * plane - 1 - j, domain_hi);
                arr(i, j, k, n) =
                    static_cast<amrex::Real>(parity[n]) * arr(i, source, k, n);
            } else if (nodal_z && j == plane && parity[n] < 0) {
                arr(i, j, k, n) = 0.0_rt;
            }
        });
    }
#else
    amrex::ignore_unused(mf, a_component_parity);
#endif
}

void ThetaImplicitMHD::PublishMomentumComponents () const
{
    // Diagnostic-only views; refreshed at initialization and at every
    // accepted step end (never inside the nonlinear solve).
    const amrex::MultiFab& momentum =
        *m_WarpX->m_fields.get(MomentumDensityName, 0);
    const std::array<const char*, 3> names = {
        MomentumDiag0Name, MomentumDiag1Name, MomentumDiag2Name};
    for (int component = 0; component < 3; ++component) {
        amrex::MultiFab& view = *m_WarpX->m_fields.get(names[component], 0);
        amrex::MultiFab::Copy(view, momentum, component, 0, 1, 0);
    }
}

void ThetaImplicitMHD::ApplyScalarRadialDomainGhosts (amrex::MultiFab& mf) const
{
#if defined(WARPX_DIM_RZ)
    // Radial domain-ghost fill for derived cell/nodal SCALARS (electron
    // pressure and temperature): even (m = 0) mirror across the axis and,
    // at r_max, the ghost recipe of the FLUID boundary -- the even
    // reflect (Neumann) image for the conducting wall and the zero-flux
    // reflect wall (so no spurious radial pressure gradient can reach the
    // wall-ring Ohm/E evaluations), or zero-gradient clamp ghosts for
    // open outflow. FillBoundaryAndSync leaves non-periodic domain ghosts
    // untouched, so without this pass they hold stale values. Staggering
    // aware: nodal-in-r data mirrors across the boundary NODE. Axial
    // ghosts must already be filled (periodic or Neumann), so this pass
    // also covers the corner ghosts.
    const amrex::Box domain =
        amrex::convert(m_WarpX->Geom(0).Domain(), mf.ixType().toIntVect());
    const int domain_lo = domain.smallEnd(0);
    const int domain_hi = domain.bigEnd(0);
    // Nodal-in-r data mirrors across the boundary node itself (offset 0);
    // cell-centered data mirrors across the face (offset 1).
    const int cc_offset = mf.ixType().cellCentered(0) ? 1 : 0;
    // The absorbing wall keeps the outflow (zero-gradient clamp) recipe
    // for the derived scalars: the wall-ring Ohm/E evaluations must see
    // no spurious radial pressure gradient (the field-side boundary is
    // unchanged by the fluid absorber).
    const bool r_outflow = m_r_open && (m_r_open_fluid != "reflect");
    const int ncomp = mf.nComp();
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const amrex::Box grown = amrex::grow(mfi.validbox(), mf.nGrowVect());
        if (grown.smallEnd(0) >= domain_lo && grown.bigEnd(0) <= domain_hi) {
            continue;
        }
        const auto arr = mf.array(mfi);
        amrex::ParallelFor(grown, ncomp,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
            if (i < domain_lo) {
                arr(i, j, k, n) = arr(2 * domain_lo - cc_offset - i, j, k, n);
            } else if (i > domain_hi) {
                const int mirror =
                    r_outflow ? domain_hi : 2 * domain_hi + cc_offset - i;
                arr(i, j, k, n) = arr(mirror, j, k, n);
            }
        });
    }
#else
    amrex::ignore_unused(mf);
#endif
}

void ThetaImplicitMHD::AddExternalFieldsToTotals (const amrex::Real sign) const
{
    using ablastr::fields::Direction;
    for (int direction = 0; direction < 3; ++direction) {
        amrex::MultiFab& electric_field =
            *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{direction}, 0);
        const amrex::MultiFab& electric_field_external = *m_WarpX->m_fields.get(
            FieldType::hybrid_E_fp_external, Direction{direction}, 0);
        amrex::MultiFab::Saxpy(electric_field, sign, electric_field_external, 0, 0, 1,
                               electric_field.nGrowVect());

        amrex::MultiFab& magnetic_field =
            *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{direction}, 0);
        const amrex::MultiFab& magnetic_field_external = *m_WarpX->m_fields.get(
            FieldType::hybrid_B_fp_external, Direction{direction}, 0);
        amrex::MultiFab::Saxpy(magnetic_field, sign, magnetic_field_external, 0, 0, 1,
                               magnetic_field.nGrowVect());
    }
}

CircuitCoupler& ThetaImplicitMHD::NativeCircuitCoupler () const
{
    CircuitCoupling* const coupling = m_WarpX->get_pointer_CircuitCoupling();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        coupling != nullptr && coupling->Coupler() != nullptr,
        "implicit_mhd.circuit_driver = native requires the circuit "
        "coupling engine: declare circuit.coils and circuit.engine "
        "(typically 'external' with circuit.plugin_library)");
    return *coupling->Coupler();
}

void ThetaImplicitMHD::SaveMagneticField ()
{
    using ablastr::fields::Direction;
    for (int direction = 0; direction < 3; ++direction) {
        const amrex::MultiFab& magnetic_field =
            *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{direction}, 0);
        amrex::MultiFab& magnetic_field_old =
            *m_WarpX->m_fields.get(FieldType::B_old, Direction{direction}, 0);
        amrex::MultiFab::Copy(magnetic_field_old, magnetic_field, 0, 0, magnetic_field.nComp(),
                              magnetic_field.nGrowVect());
    }
}

int ThetaImplicitMHD::OneStep (const amrex::Real start_time, const amrex::Real dt, const int step)
{
    BL_PROFILE("ThetaImplicitMHD::OneStep()");

    m_dt = dt;
    m_circuit_hook_calls = 0;
    // Native circuit driver: (re)open the coupling step lazily at the
    // first qualifying residual evaluation. A step replayed after a
    // failed solve re-fires BeginStep with the same t0 -- the engine
    // re-snapshots its last ACCEPTED state, the contractual replay.
    m_circuit_step_open = false;
    SaveEoldMultifab();

    if (m_hybrid_pic_model->m_add_external_fields) {
        // Enter the plasma-response frame: subtract the STORED external
        // fields (the exact values added back at the end of the previous
        // step, or at t=0 by HybridPICInitializeRhoJandB) so B_old and the
        // solver state carry the plasma response only. Then refresh the
        // externals at theta-time: the Ohm's-law constraint of this step
        // uses B_ext^{n+theta} and E_ext^{n+theta}. E_old (saved above)
        // keeps total fields for diagnostics.
        AddExternalFieldsToTotals(-1.0_rt);
        m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
            start_time + m_theta * m_dt, m_dt);
    }

    SaveMagneticField();

    if (m_resistive_theta != m_theta) {
        // Capture the beginning-of-step plasma current J^n = curl B^n/mu0
        // for the resistive-stage extrapolation (see m_resistive_theta):
        // under the split-field scheme Bfield_fp holds the plasma
        // response here, matching the residual-time current computation.
        using ablastr::fields::Direction;
        const auto magnetic_field =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, 0);
        m_hybrid_pic_model->CalculatePlasmaCurrent(magnetic_field,
                                                   m_WarpX->GetEBUpdateEFlag());
        for (int direction = 0; direction < 3; ++direction) {
            amrex::MultiFab& plasma_current = *m_WarpX->m_fields.get(
                FieldType::hybrid_current_fp_plasma, Direction{direction}, 0);
            // open z caps keep the curl-of-Green's-B ghost row (computed
            // one row into the ghosts by CalculatePlasmaCurrent)
            ApplyNeumannZDomainGhosts(plasma_current, 1);
            if (direction == 2) {
                // z_lo mirror: J_z is ODD (see ComputeRHS).
                ApplyMirrorZLoDomainGhosts(plasma_current, {-1, -1, -1});
            }
            amrex::MultiFab& old_current = *m_WarpX->m_fields.get(
                OldPlasmaCurrentName, Direction{direction}, 0);
            amrex::MultiFab::Copy(old_current, plasma_current, 0, 0, 1,
                                  old_current.nGrowVect());
        }
    }

    m_state.Copy(m_use_recast ? FieldType::Bfield_fp : FieldType::Efield_fp);
    m_state.CopyMultiFabBlocksFromFields();
    if (!m_loaded_state_sanitized) {
        // Python fluid loaders (beforeInitEsolve callbacks) overwrite the
        // parser fill AFTER InitializeFluidState's floor clamp; raise a
        // below-floor loaded state to admissibility once before the first
        // bounded solve (see SanitizeLoadedState). The shaped-wall
        // exterior clamp runs AFTER the raise (so its fixed image is the
        // band's final state) and is idempotent on restarted states that
        // already carry it.
        SanitizeLoadedState();
        ClampWallExteriorState();
        m_loaded_state_sanitized = true;
    }
    RefreshHaloPedestal();
    RefreshVacuumReferenceDensity(step);
    m_state_old.Copy(m_state);

    // Ghosted beginning-of-step fluid state for the flux kernels' floor
    // limiters (theta-extrapolated end-of-step donor gating needs old
    // NEIGHBOR values; the solver-vec old blocks are zero-ghost).
    {
        std::vector<std::pair<const char*, const char*>> pairs = {
            {MassDensityName, OldMassDensityName},
            {MomentumDensityName, OldMomentumDensityName},
            {ElectronEnergyName, OldElectronEnergyName},
            {IonEnergyName, OldIonEnergyName}};
        if (m_ion_closure == "cgl") {
            pairs.emplace_back(IonParallelEnergyName,
                               OldIonParallelEnergyName);
            pairs.emplace_back(IonPerpEnergyName, OldIonPerpEnergyName);
        }
        if (m_ion_closure == "dual_energy") {
            pairs.emplace_back(IonInternalEnergyName,
                               OldIonInternalEnergyName);
        }
        for (const auto& [source_name, destination_name] : pairs) {
            const amrex::MultiFab& source =
                *m_WarpX->m_fields.get(source_name, 0);
            amrex::MultiFab& destination =
                *m_WarpX->m_fields.get(destination_name, 0);
            amrex::MultiFab::Copy(destination, source, 0, 0,
                                  destination.nComp(), 0);
            destination.FillBoundaryAndSync(
                m_WarpX->Geom(0).periodicity());
        }
        ApplyFluidDomainBoundaries(
            *m_WarpX->m_fields.get(OldMassDensityName, 0),
            *m_WarpX->m_fields.get(OldMomentumDensityName, 0),
            *m_WarpX->m_fields.get(OldElectronEnergyName, 0),
            *m_WarpX->m_fields.get(OldIonEnergyName, 0),
            *m_WarpX->m_fields.get(OldIonParallelEnergyName, 0),
            *m_WarpX->m_fields.get(OldIonPerpEnergyName, 0),
            *m_WarpX->m_fields.get(OldIonInternalEnergyName, 0));
    }

    if (m_ion_closure == "dual_energy" &&
        m_dual_energy_internal_cutoff > 0.0_rt) {
        // Per-step frozen input of the low-internal fk cutoff (the reference code's
        // mix = -2 MAXVAL(wio)): the global max of the STEP-OLD U_i,
        // frozen for the whole nonlinear solve so the cutoff mask is a
        // per-solve constant for the matrix-free Jacobian probes.
        m_dual_energy_internal_old_max =
            m_state_old.getMultiFabBlock(IonInternalEnergyName, 0).max(0);
    }

    if (m_hybrid_pic_model->m_include_electron_inertia) {
        // Electron-inertia drho/dt leg: freeze the step-start NODAL charge
        // density EXACTLY from the continuity state -- the ghosted step-old
        // density through rho_fp's own cell-to-node interpolation (see
        // FillFluidSources) -- rather than capturing a first-evaluation
        // image; the recast state is exact where hybrid deposits are not.
        const amrex::MultiFab& old_density =
            *m_WarpX->m_fields.get(OldMassDensityName, 0);
        amrex::MultiFab& rho_n_frozen =
            *m_WarpX->m_fields.get("hybrid_rho_n_frozen", 0);
        const auto cell_stag = cell_staggering();
        const auto rho_stag = field_staggering(rho_n_frozen);
        const auto coarsening = amrex::GpuArray<int, 3>{1, 1, 1};
        const amrex::Real charge_to_mass = m_ion_charge_to_mass;
        for (amrex::MFIter mfi(rho_n_frozen); mfi.isValid(); ++mfi) {
            const amrex::Box box =
                mfi.tilebox(rho_n_frozen.ixType().toIntVect());
            const auto destination = rho_n_frozen.array(mfi);
            const auto source = old_density.const_array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                destination(i, j, k) =
                    charge_to_mass *
                    ablastr::coarsen::sample::Interp(source, cell_stag,
                                                     rho_stag, coarsening,
                                                     i, j, k, 0);
            });
        }
        rho_n_frozen.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    }

    m_nlsolver->Solve(m_state, m_state_old, start_time, m_dt, step);
    const int exit_status = m_nlsolver->GetExitStatus();
    if (exit_status < 0) {
        return exit_status;
    }

    UpdateWarpXFields(m_state, start_time);
#if defined(WARPX_DIM_RZ)
    if (m_use_recast &&
        ((m_r_open && m_r_open_fluid == "absorb") ||
         m_wall_mask.GetThermalBC() !=
             ImplicitMHDWallMask::ThermalBC::none)) {
        // Book the absorbing wall's export and/or the shaped wall's
        // deposition from the accepted theta state (UpdateWarpXFields
        // just refilled the theta-stage fields), before
        // FinishStateUpdate extrapolates the state to t^{n+1}.
        AccumulateAbsorbedWallLedger(m_dt, step,
                                     start_time + m_theta * m_dt);
    }
#endif
    if (m_floor_consistency_rate > 0.0_rt) {
        // Book the floor-consistency supply from the accepted theta state
        // (m_state), before FinishStateUpdate extrapolates it to t^{n+1}.
        AccumulateFloorConsistencySupplyLedger(m_dt, step);
    }
    m_WarpX->reduced_diags->ComputeDiagsMidStep(step);
    FinishStateUpdate(start_time + m_dt, step);
    if (m_external_field_iteration) {
        // Sync-tax instrument: how many circuit-hook firings the step
        // cost (python round-trips, or native engine advances; the
        // end-of-step commit is separate and fires once in either
        // scope).
        amrex::Print() << "ThetaImplicitMHD: step " << step + 1
                       << " externalcoiltheta calls = "
                       << m_circuit_hook_calls << " (circuit_hook_scope = "
                       << (m_circuit_hook_newton_scope ? "newton"
                                                       : "residual")
                       << ", circuit_driver = "
                       << (m_circuit_native ? "native" : "python")
                       << ")\n";
    }
    ExecutePythonCallback("afterEpush");
    return exit_status;
}

void ThetaImplicitMHD::UpdateWarpXFields (const WarpXSolverVec& state, const amrex::Real start_time)
{
    const amrex::Real theta_time = start_time + m_theta * m_dt;
    if (m_use_recast) {
        // Conservative form: the state array block IS B^{n+theta}; E is
        // assembled from the face EMF and eta J later in the residual.
        m_WarpX->SetMagneticFieldAndApplyBCs(state, theta_time);
    } else {
        m_WarpX->SetElectricFieldAndApplyBCs(state, theta_time);

        const auto& magnetic_field_old =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, 0);
        m_WarpX->UpdateMagneticFieldAndApplyBCs(magnetic_field_old,
                                                m_theta * m_dt, start_time);
    }

    if (m_z_neumann) {
        using ablastr::fields::Direction;
        for (int direction = 0; direction < 3; ++direction) {
            // E has no Green's-fill counterpart and its cap ghosts feed no
            // residual stencil (the ghost-B rows a curl-E write could reach
            // are overwritten by the Green's fill), so it keeps the plain
            // Neumann clamp everywhere. B on an open cap was just filled by
            // the Green's map inside ApplyBfieldBoundary: every ghost row is
            // kept there (the fill covers the full ghost width). On a z_lo
            // PMC (mirror) face the Neumann image is replaced by the
            // parity image: E_z is ODD cell-centered in z; Br and B_theta
            // are ODD cell-centered in z; the remaining components are
            // EVEN and nodal in z, where the Neumann fill already IS the
            // even mirror.
            amrex::MultiFab& electric_component =
                *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{direction}, 0);
            ApplyNeumannZDomainGhosts(electric_component);
            if (direction == 2) {
                ApplyMirrorZLoDomainGhosts(electric_component, {-1, -1, -1});
            }
            amrex::MultiFab& magnetic_component =
                *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{direction}, 0);
            ApplyNeumannZDomainGhosts(magnetic_component,
                                      magnetic_component.nGrowVect()[1]);
            if (direction != 2) {
                ApplyMirrorZLoDomainGhosts(magnetic_component, {-1, -1, -1});
            }
        }
    }

    FillFluidSources(state);
}

void ThetaImplicitMHD::FillFluidSources (const WarpXSolverVec& state)
{
    for (auto const& block_spec : state.getMultiFabBlockSpecs()) {
        amrex::MultiFab& destination = *m_WarpX->m_fields.get(block_spec.name, 0);
        const amrex::MultiFab& source = state.getMultiFabBlock(block_spec.name, 0);
        amrex::MultiFab::Copy(destination, source, 0, 0, destination.nComp(), 0);
        destination.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    }
    ApplyFluidDomainBoundaries(*m_WarpX->m_fields.get(MassDensityName, 0),
                               *m_WarpX->m_fields.get(MomentumDensityName, 0),
                               *m_WarpX->m_fields.get(ElectronEnergyName, 0),
                               *m_WarpX->m_fields.get(IonEnergyName, 0),
                               *m_WarpX->m_fields.get(IonParallelEnergyName, 0),
                               *m_WarpX->m_fields.get(IonPerpEnergyName, 0),
                               *m_WarpX->m_fields.get(IonInternalEnergyName, 0));

    const amrex::MultiFab& density = *m_WarpX->m_fields.get(MassDensityName, 0);
    const amrex::MultiFab& momentum = *m_WarpX->m_fields.get(MomentumDensityName, 0);
    const amrex::MultiFab& electron_energy = *m_WarpX->m_fields.get(ElectronEnergyName, 0);
    amrex::MultiFab& charge_density = *m_WarpX->m_fields.get(FieldType::rho_fp, 0);
    auto ion_current = m_WarpX->m_fields.get_alldirs(FieldType::current_fp, 0);
    amrex::MultiFab& electron_pressure =
        *m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, 0);
    amrex::MultiFab& electron_temperature =
        *m_WarpX->m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);

    const auto cell_stag = cell_staggering();
    const auto coarsening = amrex::GpuArray<int, 3>{1, 1, 1};
    const auto rho_stag = field_staggering(charge_density);
    const auto pressure_stag = field_staggering(electron_pressure);
    const amrex::Real charge_to_mass = m_ion_charge_to_mass;
    const amrex::Real gamma_e_minus_one = m_gamma_e - 1.0_rt;
    const amrex::Real pressure_floor = m_electron_pressure_floor;
    // Ohm's-law guard: T_e feeds the eta("Te") evaluation, so its density
    // denominator floors at the hybrid n_floor, not the (typically much
    // lower) fluid positivity floor.
    const amrex::Real charge_density_floor =
        OhmMassDensityFloor() * m_ion_charge_to_mass;

    charge_density.setVal(0.0_rt);
    for (amrex::MFIter mfi(charge_density); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.tilebox(charge_density.ixType().toIntVect());
        const auto destination = charge_density.array(mfi);
        const auto source = density.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            destination(i, j, k) =
                charge_to_mass * ablastr::coarsen::sample::Interp(source, cell_stag, rho_stag,
                                                                  coarsening, i, j, k, 0);
        });
    }
    charge_density.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    // Nodal-in-z EVEN scalar: on a z_lo mirror face the Neumann nodal
    // fill is already the exact even reflection (same for the nodal Te
    // and rebuilt Pe below), so no parity pass is needed.
    ApplyNeumannZDomainGhosts(charge_density);
    // Radial domain ghosts too (axis parity mirror; wall reflect or open
    // clamp at r_max): the electron-pressure ghost rebuild below reads the
    // nodal density in every domain-ghost ring, and Pe = n_f kB Te must
    // hold there against the SAME mirrored/clamped n image the fills give
    // the temperature.
    ApplyScalarRadialDomainGhosts(charge_density);

    for (int component = 0; component < 3; ++component) {
        const auto current_stag = field_staggering(*ion_current[component]);
        ion_current[component]->setVal(0.0_rt);
        for (amrex::MFIter mfi(*ion_current[component]); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.tilebox(ion_current[component]->ixType().toIntVect());
            const auto destination = ion_current[component]->array(mfi);
            const auto source = momentum.const_array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                destination(i, j, k) = charge_to_mass * ablastr::coarsen::sample::Interp(
                                                            source, cell_stag, current_stag,
                                                            coarsening, i, j, k, component);
            });
        }
        ion_current[component]->FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
        ApplyNeumannZDomainGhosts(*ion_current[component]);
        if (component == 2) {
            // z_lo mirror: J is a polar vector, so J_z (cell-centered in
            // z) is ODD across the plane; J_r and J_theta are EVEN and
            // nodal in z, already exact under the Neumann nodal mirror.
            ApplyMirrorZLoDomainGhosts(*ion_current[component], {-1, -1, -1});
        }
    }

    // TEMPERATURE-primary nodal electron closure fields:
    //   (1) T_e is formed NATIVELY on the cell grid, Te_cell =
    //       p_cell/(n_cell_f kB), from the SAME floored pressure recovery
    //       as the flux kernels (p_cell = max((gamma_e - 1) U_e, p_floor),
    //       see load_cell_state) over the Ohm-floored cell density -- no
    //       interpolation enters the ratio;
    //   (2) Te_nodal = Interp(Te_cell -> node): the interpolant acts on
    //       the smooth bounded RATIO. The previous pressure-primary fill
    //       interpolated the n T product and divided by the interpolated
    //       density, but Interp(n T) != Interp(n) Interp(T) and the
    //       mismatch does not cancel at density gradients -- a spurious
    //       Ohm grad-Pe electric field wherever n has structure
    //       (separatrix, exhaust, wall bands);
    //   (3) Pe_nodal = n_nodal_f kB Te_nodal, rebuilt co-located with the
    //       SAME floored discrete density image the Ohm grad-Pe/(e n)
    //       denominator uses (the OhmMassDensityFloor convention), so the
    //       isothermal limit reduces discretely to E = -kB Te grad(n_f)/
    //       (e n_f) INCLUDING the floor band. Matches the hybrid trees'
    //       QDSMCFillElectronPressureFromTe convention (Pe derived from
    //       n and Te, never the reverse).
    amrex::MultiFab& electron_temperature_cc =
        *m_WarpX->m_fields.get(ElectronTemperatureCCName, 0);
    for (amrex::MFIter mfi(electron_temperature_cc); mfi.isValid(); ++mfi) {
        // Ghosts computed in place from the ghosted moments (filled by
        // ApplyFluidDomainBoundaries above), so the node interpolation and
        // the ghost passes below see boundary-consistent cell temperatures
        // without a FillBoundary of the scratch.
        const amrex::Box grown =
            amrex::grow(mfi.validbox(), electron_temperature_cc.nGrowVect());
        const auto temperature_cc = electron_temperature_cc.array(mfi);
        const auto energy = electron_energy.const_array(mfi);
        const auto rho_cell = density.const_array(mfi);
        amrex::ParallelFor(grown, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const amrex::Real cell_pressure = std::max(
                gamma_e_minus_one * energy(i, j, k), pressure_floor);
            const amrex::Real number_density =
                std::max(charge_to_mass * rho_cell(i, j, k),
                         charge_density_floor) /
                PhysConst::q_e;
            temperature_cc(i, j, k) =
                cell_pressure / (number_density * PhysConst::kb);
        });
    }

    for (amrex::MFIter mfi(electron_pressure); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.tilebox(electron_pressure.ixType().toIntVect());
        const auto pressure = electron_pressure.array(mfi);
        const auto temperature = electron_temperature.array(mfi);
        const auto temperature_cc = electron_temperature_cc.const_array(mfi);
        const auto rho = charge_density.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const amrex::Real temperature_value =
                ablastr::coarsen::sample::Interp(temperature_cc, cell_stag,
                                                 pressure_stag, coarsening, i,
                                                 j, k, 0);
            temperature(i, j, k) = temperature_value;
            const amrex::Real number_density =
                std::max(rho(i, j, k), charge_density_floor) / PhysConst::q_e;
            pressure(i, j, k) =
                number_density * PhysConst::kb * temperature_value;
        });
    }
    electron_pressure.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    electron_temperature.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    ApplyNeumannZDomainGhosts(electron_temperature);
    // Radial domain ghosts (axis parity; Neumann reflect image or outflow
    // clamp at r_max) so every ghost the Ohm/E-field paths could read
    // carries the zero-normal-gradient wall temperature instead of the
    // stale values FillBoundaryAndSync leaves at non-periodic domain edges.
    ApplyScalarRadialDomainGhosts(electron_temperature);
    // The PRESSURE domain ghosts are REBUILT as Pe = n_f kB Te from the
    // mirrored/clamped temperature and density images (never mirrored
    // directly), so the co-location identity of step (3) holds exactly in
    // every ghost ring too. With identical fill recipes for n and Te this
    // equals the direct mirror value-for-value; rebuilding makes the
    // contract structural rather than coincidental.
#if defined(WARPX_DIM_RZ)
    if (m_z_neumann || !m_WarpX->Geom(0).isPeriodic(0)) {
        const amrex::Box nodal_domain = amrex::convert(
            m_WarpX->Geom(0).Domain(), electron_pressure.ixType().toIntVect());
        const int r_lo = nodal_domain.smallEnd(0);
        const int r_hi = nodal_domain.bigEnd(0);
        const int z_lo = nodal_domain.smallEnd(1);
        const int z_hi = nodal_domain.bigEnd(1);
        const bool fill_z = m_z_neumann;
        for (amrex::MFIter mfi(electron_pressure); mfi.isValid(); ++mfi) {
            const amrex::Box grown =
                amrex::grow(mfi.validbox(), electron_pressure.nGrowVect());
            const auto pressure = electron_pressure.array(mfi);
            const auto temperature = electron_temperature.const_array(mfi);
            const auto rho = charge_density.const_array(mfi);
            amrex::ParallelFor(grown,
                               [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const bool outside_r = (i < r_lo || i > r_hi);
                const bool outside_z = fill_z && (j < z_lo || j > z_hi);
                if (!outside_r && !outside_z) {
                    return;
                }
                const amrex::Real number_density =
                    std::max(rho(i, j, k), charge_density_floor) /
                    PhysConst::q_e;
                pressure(i, j, k) =
                    number_density * PhysConst::kb * temperature(i, j, k);
            });
        }
    }
#endif
}

void ThetaImplicitMHD::FillCellCenteredElectromagneticFields ()
{
    const auto total_current =
        m_WarpX->m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0);
    const auto magnetic_field = m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, 0);
    amrex::MultiFab& total_current_cc = *m_WarpX->m_fields.get(TotalCurrentCCName, 0);
    amrex::MultiFab& magnetic_field_cc = *m_WarpX->m_fields.get(MagneticFieldCCName, 0);

    const auto cell_stag = cell_staggering();
    const auto coarsening = amrex::GpuArray<int, 3>{1, 1, 1};
    // On an open z cap the face-flux/EMF kernels read the first
    // cell-centered ghost row: interpolate it directly from the
    // Green's-filled B and the curl-B plasma current (both carry
    // free-space values one row into the cap ghosts, the current from
    // CalculatePlasmaCurrent's grown tileboxes) instead of letting the
    // Neumann pass clamp it from the interior plane.
    const amrex::Box& cc_domain = m_WarpX->Geom(0).Domain();
    auto grow_into_open_caps = [&] (amrex::Box box) {
        bool grew = false;
        if (m_z_lo_open && box.smallEnd(1) == cc_domain.smallEnd(1)) {
            box.growLo(1, 1);
            grew = true;
        }
        if (m_z_hi_open && box.bigEnd(1) == cc_domain.bigEnd(1)) {
            box.growHi(1, 1);
            grew = true;
        }
        if (grew) {
            // the corner-EMF stencils read the computed ghost row one
            // column past the box edges; the interp sources are defined
            // there (grown-tilebox currents, boundary-filled B)
            box.grow(0, 1);
        }
        return box;
    };
    for (int component = 0; component < 3; ++component) {
        const auto current_stag = field_staggering(*total_current[component]);
        const auto magnetic_stag = field_staggering(*magnetic_field[component]);
        for (amrex::MFIter mfi(total_current_cc); mfi.isValid(); ++mfi) {
            const amrex::Box box = grow_into_open_caps(mfi.validbox());
            const auto current_cc = total_current_cc.array(mfi);
            const auto magnetic_cc = magnetic_field_cc.array(mfi);
            const auto current = total_current[component]->const_array(mfi);
            const auto magnetic = magnetic_field[component]->const_array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                current_cc(i, j, k, component) = ablastr::coarsen::sample::Interp(
                    current, current_stag, cell_stag, coarsening, i, j, k, 0);
                magnetic_cc(i, j, k, component) = ablastr::coarsen::sample::Interp(
                    magnetic, magnetic_stag, cell_stag, coarsening, i, j, k, 0);
            });
        }
    }
    // Under the split-field external-A scheme, Bfield_fp holds only the
    // plasma response during the solve; the fluid J x B force needs the
    // total field, so add the cell-centered external B here. The plasma
    // current above is correct as-is: curl B_ext vanishes inside the domain
    // (the coil currents live outside it).
    if (m_hybrid_pic_model->m_add_external_fields) {
        const auto external_field =
            m_WarpX->m_fields.get_alldirs(FieldType::hybrid_B_fp_external, 0);
        for (int component = 0; component < 3; ++component) {
            const auto external_stag = field_staggering(*external_field[component]);
            for (amrex::MFIter mfi(magnetic_field_cc); mfi.isValid(); ++mfi) {
                const amrex::Box box = grow_into_open_caps(mfi.validbox());
                const auto magnetic_cc = magnetic_field_cc.array(mfi);
                const auto external = external_field[component]->const_array(mfi);
                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    magnetic_cc(i, j, k, component) += ablastr::coarsen::sample::Interp(
                        external, external_stag, cell_stag, coarsening, i, j, k, 0);
                });
            }
        }
    }

    total_current_cc.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    magnetic_field_cc.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    ApplyNeumannZDomainGhosts(total_current_cc, 1);
    ApplyNeumannZDomainGhosts(magnetic_field_cc, 1);
    // z_lo mirror: the cell-centered interpolants take the parity image
    // instead of the Neumann clamp -- J is a polar vector (J_r, J_theta
    // even; J_z odd), B an axial vector (Br, B_theta odd; Bz even). With
    // the split-field external drive folded into B_cc above, the deck's
    // external field must itself respect the z = 0 mirror (symmetric
    // coil set); the fill imposes the parity on the total. Runs before
    // the radial passes below so the corner ghosts inherit.
    ApplyMirrorZLoDomainGhosts(total_current_cc, {1, 1, -1});
    ApplyMirrorZLoDomainGhosts(magnetic_field_cc, {-1, -1, 1});
#if defined(WARPX_DIM_RZ)
    {
        // The r_max domain ghosts of the cell-centered total current are
        // reachable by no fill above: FillBoundaryAndSync skips
        // non-periodic domain edges and the Neumann pass only covers z,
        // so they would keep their allocation value (zero) forever.
        // Nothing reads them while the MHD Ohm's law runs Hall-off, but
        // the Hall / electron-inertia extensions interpolate J_cc through
        // the wall ring, and an alloc-zero ghost there would inject a
        // spurious full-J wall jump -- the exact twin of the external-B
        // ghost gap fixed in ExternalVectorPotential. Fill them with the
        // staggering-aware even (zero-normal-gradient) clamp of the
        // outermost valid ring; running after the axial pass makes the
        // corner ghosts consistent too. The axis side is left to the
        // kernels' own parity handling.
        const amrex::Box current_domain = amrex::convert(
            m_WarpX->Geom(0).Domain(), total_current_cc.ixType().toIntVect());
        const int radial_domain_hi = current_domain.bigEnd(0);
        const int ncomp = total_current_cc.nComp();
        for (amrex::MFIter mfi(total_current_cc); mfi.isValid(); ++mfi) {
            const amrex::Box grown =
                amrex::grow(mfi.validbox(), total_current_cc.nGrowVect());
            if (grown.bigEnd(0) <= radial_domain_hi) {
                continue;
            }
            const auto arr = total_current_cc.array(mfi);
            amrex::ParallelFor(grown, ncomp,
                               [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                if (i > radial_domain_hi) {
                    arr(i, j, k, n) = arr(radial_domain_hi, j, k, n);
                }
            });
        }
    }
#endif
    if (m_use_recast) {
        // The recast face kernels read cell-centered B in the radial
        // domain ghosts (the wall face's outer donor cell, and the
        // transverse ghost faces feeding the corner UCT EMF).
        ApplyMagneticCCDomainGhosts(magnetic_field_cc);
    }
}

void ThetaImplicitMHD::FillCellCenteredOhmElectricField ()
{
    // Gather the SOLVED stage E -- the same Efield_fp the Faraday update
    // consumes, assembled fresh on every residual evaluation (Jacobian
    // probes included) -- to cell centers for the Ohm-current Joule
    // quench (see m_joule_ohm_current). Only valid cells are filled: the
    // interpolation stencils read at most one nodal index above the cell,
    // which the staggered valid boxes always carry, and the fluid RHS
    // kernels read the gather at valid cells only.
    const auto electric_field =
        m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, 0);
    amrex::MultiFab& electric_field_cc =
        *m_WarpX->m_fields.get(OhmElectricFieldCCName, 0);

    const auto cell_stag = cell_staggering();
    const auto coarsening = amrex::GpuArray<int, 3>{1, 1, 1};
    for (int component = 0; component < 3; ++component) {
        const auto electric_stag = field_staggering(*electric_field[component]);
        for (amrex::MFIter mfi(electric_field_cc); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto electric_cc = electric_field_cc.array(mfi);
            const auto electric = electric_field[component]->const_array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                electric_cc(i, j, k, component) = ablastr::coarsen::sample::Interp(
                    electric, electric_stag, cell_stag, coarsening, i, j, k, 0);
            });
        }
    }
    // Under the split-field external-A scheme Efield_fp holds only the
    // plasma response (E_ext is subtracted for the Faraday bookkeeping of
    // the evolved response B; see AssembleOhmElectricField). Ohm's law --
    // whose resistive part the quench divides by -- holds for the TOTAL
    // field, so add the cell-centered external E back, mirroring the
    // B_ext fold of FillCellCenteredElectromagneticFields. At wall-masked
    // (pec) locations the projection wrote -E_ext, so the gather
    // correctly reads total E = 0 inside the conductor.
    if (m_hybrid_pic_model->m_add_external_fields) {
        const auto external_field =
            m_WarpX->m_fields.get_alldirs(FieldType::hybrid_E_fp_external, 0);
        for (int component = 0; component < 3; ++component) {
            const auto external_stag = field_staggering(*external_field[component]);
            for (amrex::MFIter mfi(electric_field_cc); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                const auto electric_cc = electric_field_cc.array(mfi);
                const auto external = external_field[component]->const_array(mfi);
                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    electric_cc(i, j, k, component) += ablastr::coarsen::sample::Interp(
                        external, external_stag, cell_stag, coarsening, i, j, k, 0);
                });
            }
        }
    }
}

void ThetaImplicitMHD::ComputeRHS (WarpXSolverVec& rhs, const WarpXSolverVec& state,
                                   const amrex::Real start_time, const int nonlinear_iteration,
                                   const bool from_jacobian)
{
    BL_PROFILE("ThetaImplicitMHD::ComputeRHS()");
    amrex::ignore_unused(nonlinear_iteration);

    UpdateWarpXFields(state, start_time);

    const auto magnetic_field = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, 0);
    m_hybrid_pic_model->CalculatePlasmaCurrent(magnetic_field, m_WarpX->GetEBUpdateEFlag());
    if (m_z_neumann) {
        using ablastr::fields::Direction;
        for (int direction = 0; direction < 3; ++direction) {
            amrex::MultiFab& current_component = *m_WarpX->m_fields.get(
                FieldType::hybrid_current_fp_plasma, Direction{direction}, 0);
            ApplyNeumannZDomainGhosts(current_component, 1);
            if (direction == 2) {
                // z_lo mirror: J_z (cell-centered in z) is ODD; J_r and
                // J_theta are even nodal-in-z, exact under the Neumann
                // nodal mirror.
                ApplyMirrorZLoDomainGhosts(current_component, {-1, -1, -1});
            }
        }
    }

    if (m_external_field_iteration &&
        (!m_circuit_hook_newton_scope ||
         (!from_jacobian && m_residual_is_newton_iterate))) {
        // Circuit-in-the-residual coupling: python measures the
        // reciprocity flux linkage of THIS iterate's plasma current
        // (hybrid_current_fp_plasma, just computed), re-advances the
        // restored circuit against it, and pushes updated coil scale
        // segments; the externals are then refreshed at theta-time so the
        // Ohm's law and fluid J x B below see circuit-consistent fields.
        // The map is smooth in the state, so the matrix-free Jacobian
        // probes see the coupled plasma-circuit physics and Newton
        // converges both together (the segregated lagged alternative is
        // unstable at strong coil-plasma coupling).
        //
        // circuit_hook_scope = newton instead fires the hook ONLY at
        // accepted Newton iterates: Jacobian probes and line-search
        // trials skip both the circuit round-trip and the external-field
        // refresh, reusing the coil scales (and the theta-time external
        // fields computed from them) cached here at the last iterate
        // evaluation. The Jacobian then sees a per-iterate lagged
        // circuit (quasi-Newton); the converged answer is unchanged
        // because convergence is still tested on this live-coupled
        // iterate residual.
        if (m_circuit_native) {
            // Native driver: measure this iterate's linkages with the
            // engine's batched device probes (feeding the plasma current
            // just computed above -- no python, no per-coil pulls, no
            // full-grid re-curl) and re-advance the engine from the
            // committed interval entry to the THETA-STAGE time
            // (accept = false: switches and stage swaps stay latched at
            // the committed state). The engine pushes the realized coil
            // scales as segments; the shared refresh below realizes them
            // on the external fields exactly like the python path.
            CircuitCoupler& coupler = NativeCircuitCoupler();
            coupler.MeasureLinkages(false);
            if (!m_circuit_step_open) {
                coupler.BeginStepMeasured(start_time, m_dt);
                m_circuit_step_open = true;
            }
            coupler.EvaluateInterval(start_time, start_time + m_theta * m_dt,
                                     false);
        } else {
            ExecutePythonCallback("externalcoiltheta");
        }
        ++m_circuit_hook_calls;
        m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
            start_time + m_theta * m_dt, m_dt);
    }

    FillCellCenteredElectromagneticFields();

    const amrex::Real theta_time = start_time + m_theta * m_dt;
    if (m_use_recast) {
        // Conservative form: one face-flux evaluation (HLLD fan or
        // central) per face feeds the fluid divergences AND the ideal
        // EMF; E = EMF + eta J is derived, and the array-block residual
        // is the theta-implicit Faraday update evaluated with the exact
        // Yee curl (div B preserved to round-off):
        // rhs_B = (B_old - theta dt curl E) - B_old.
        if (m_hybrid_pic_model->m_include_electron_inertia) {
            // Electron inertia: assemble the nodal inertial field from the
            // theta-stage state ahead of the Ohm assembly (which adds it
            // per E component). REACTIVE staging: it reads the theta-stage
            // plasma/ion currents and the theta-stage continuity density --
            // never the resistive_theta extrapolation, and it is NOT
            // shifted by conduction_theta (both shifts over-center
            // DISSIPATIVE channels; applied to the inertial reactance they
            // would damp the capped whistler first order in dt). Refreshed
            // on EVERY evaluation including Jacobian probes, so the term is
            // exactly linear-consistent for JFNK (the Je histories and the
            // rho^n leg are per-step frozen).
            m_hybrid_pic_model->ComputeElectronInertiaNodal(m_theta, m_dt,
                                                            from_jacobian);
        }
        ComputeFaceFluxes(theta_time);
        AssembleOhmElectricField(theta_time, true);
        const auto& magnetic_field_old =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, 0);
        m_WarpX->UpdateMagneticFieldAndApplyBCs(magnetic_field_old,
                                                m_theta * m_dt, start_time);
        rhs.Copy(FieldType::Bfield_fp);
    } else {
        const auto electric_field =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, 0);
        const auto ion_current =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, 0);
        const auto charge_density =
            m_WarpX->m_fields.get_mr_levels(FieldType::rho_fp, 0);

        m_hybrid_pic_model->HybridPICSolveE(
            electric_field, ion_current, magnetic_field, charge_density,
            m_WarpX->GetEBUpdateEFlag(), true, true, theta_time, false);

        rhs.Copy(FieldType::Efield_fp);
    }
    for (int component = 0; component < 3; ++component) {
        rhs.getArrayVec()[0][component]->minus(*m_state_old.getArrayVec()[0][component], 0, 1, 0);
    }
    // Shaped-wall FIELD freeze (implicit_mhd.wall_field_freeze): every
    // evolved magnetic-field face whose complete Faraday stencil lies in
    // the masked band is an exact identity row, F = B - B^n -- the field
    // twin of the fluid wall_live identities. Zeroing the rhs here (after
    // the m_state_old subtraction, above) makes the row's Jacobian
    // diagonal exactly one with no state coupling; the tables are
    // geometry-static, so JFNK probes see constant structure, and with
    // the Newton initial guess at m_state_old the Krylov vectors stay
    // exactly zero on the frozen faces -- the exterior plasma-response B
    // is bit-frozen at its boot/restart value. The recast is guaranteed
    // here: wall_model requires it at Define, and the freeze requires an
    // active wall_model.
    const warpx::mhd_pc::WallFieldFreezeView field_freeze =
        m_wall_mask.FieldFreezeView();
    if (field_freeze.active) {
        const amrex::Array<const int*, 3> first_frozen = {
            field_freeze.first_frozen_br, field_freeze.first_frozen_bt,
            field_freeze.first_frozen_bz};
        for (int component = 0; component < 3; ++component) {
            amrex::MultiFab& rhs_component =
                *rhs.getArrayVec()[0][component];
            const int* const AMREX_RESTRICT first = first_frozen[component];
            for (amrex::MFIter mfi(rhs_component); mfi.isValid(); ++mfi) {
                const auto rhs_array = rhs_component.array(mfi);
                amrex::ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        if (i >= first[j]) {
                            rhs_array(i, j, k) = 0.0_rt;
                        }
                    });
            }
        }
    }
    if (m_joule_ohm_current && m_include_joule_heating) {
        // Ohm-current Joule quench: refresh the cell-centered gather of
        // the just-assembled stage E on EVERY residual evaluation, so the
        // quenched deposit below is exactly the map the matrix-free
        // Jacobian probes differentiate. Bit-identical OFF path: nothing
        // is gathered and the deposit keeps the curl-B current.
        FillCellCenteredOhmElectricField();
    }
    ComputeFluidRHS(rhs, theta_time);
}

ThetaImplicitMHD::AdmissibilityBounds
ThetaImplicitMHD::MakeAdmissibilityBounds () const
{
    const bool dual_energy_closure = m_ion_closure == "dual_energy";
    const bool total_energy_closure =
        m_ion_closure == "total_energy" || dual_energy_closure;
    const bool cgl_closure = m_ion_closure == "cgl";
    const amrex::Real energy_floor =
        m_electron_pressure_floor / (m_gamma_e - 1.0_rt);
    const amrex::Real ion_energy_floor =
        total_energy_closure ? m_ion_pressure_floor / (m_gamma_i - 1.0_rt)
                             : 0.0_rt;
    // Temperature floors as pressure-per-mass-density coefficients,
    // p_T(rho) = rho (q/m)/e kB T_floor = n kB T_floor for the
    // quasi-neutral single-ion fluid (n_e = n_i = rho (q/m)/e).
    const amrex::Real electron_pressure_per_density =
        m_ion_charge_to_mass / PhysConst::q_e * PhysConst::kb *
        m_electron_temperature_floor;
    const amrex::Real ion_pressure_per_density =
        m_ion_charge_to_mass / PhysConst::q_e * PhysConst::kb *
        m_ion_temperature_floor;
    // Block 3 is U_perp under cgl (pressure-valued bounds) and the
    // auxiliary internal energy U_i under dual_energy (internal-energy-
    // valued bounds, the exact twins of the E_i block's internal image).
    return {
        {m_mass_density_floor, energy_floor,
         cgl_closure ? 0.5_rt * m_ion_pressure_floor : ion_energy_floor,
         dual_energy_closure ? ion_energy_floor : m_ion_pressure_floor},
        {0.0_rt, electron_pressure_per_density / (m_gamma_e - 1.0_rt),
         cgl_closure
             ? 0.5_rt * ion_pressure_per_density
             : (total_energy_closure
                    ? ion_pressure_per_density / (m_gamma_i - 1.0_rt)
                    : 0.0_rt),
         dual_energy_closure
             ? ion_pressure_per_density / (m_gamma_i - 1.0_rt)
             : ion_pressure_per_density}};
}

amrex::Real
ThetaImplicitMHD::LimitSolverStep (const WarpXSolverVec& state,
                                   const WarpXSolverVec& direction,
                                   const amrex::Real requested_step) const
{
    if (requested_step == 0.0_rt) {
        return requested_step;
    }

    // v1 positivity for the total-energy closure: bound E_i at the INTERNAL
    // energy floor U_i_floor. This is exact where the kinetic energy is
    // small and merely conservative (never permissive) where it is not; the
    // pressure clamp in load_cell_state protects the KE-dominated corner.
    // The CGL blocks are pure internal energies, so their bounds are exact:
    // U_par >= p_i_floor/2 and U_perp >= p_i_floor.
    //
    // Each energy block's floor is the max of the absolute value and the
    // temperature floor's n kB T equivalent evaluated with the STEP-OLD
    // density (frozen for the whole solve, keeping the bound linear even
    // though the density is itself a Newton unknown). The temperature
    // part is a one-way RATCHET: it binds exactly those cells whose
    // step-old value already satisfied it, so colder-than-floor initial
    // data is held by the absolute floors alone (never lifted or
    // deadlocked) while any cell at or above the floor can never cool
    // through it.
    const bool dual_energy_closure = m_ion_closure == "dual_energy";
    const bool total_energy_closure =
        m_ion_closure == "total_energy" || dual_energy_closure;
    const bool cgl_closure = m_ion_closure == "cgl";
    const amrex::Real theta = m_theta;

    const int num_blocks = (cgl_closure || dual_energy_closure)
                               ? 4
                               : (total_energy_closure ? 3 : 2);
    const std::array<const char*, 4> block_names = {
        MassDensityName, ElectronEnergyName,
        cgl_closure ? IonParallelEnergyName : IonEnergyName,
        dual_energy_closure ? IonInternalEnergyName : IonPerpEnergyName};
    const AdmissibilityBounds bounds = MakeAdmissibilityBounds();
    const amrex::MultiFab& old_density_mf =
        m_state_old.getMultiFabBlock(MassDensityName, 0);

    amrex::Real step_fraction = 1.0_rt;
    for (int block = 0; block < num_blocks; ++block) {
        const amrex::Real floor = bounds.floors[block];
        const amrex::Real temperature_coefficient =
            bounds.temperature_coefficients[block];
        const amrex::MultiFab& value_mf =
            state.getMultiFabBlock(block_names[block], 0);
        const amrex::MultiFab& delta_mf =
            direction.getMultiFabBlock(block_names[block], 0);
        const amrex::MultiFab& old_mf =
            m_state_old.getMultiFabBlock(block_names[block], 0);
        amrex::ReduceOps<amrex::ReduceOpMin> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (amrex::MFIter mfi(value_mf); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto value = value_mf.const_array(mfi);
            const auto delta = delta_mf.const_array(mfi);
            const auto old_value = old_mf.const_array(mfi);
            const auto old_density = old_density_mf.const_array(mfi);
            reduce_op.eval(
                box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    const amrex::Real temperature_bound =
                        temperature_coefficient * old_density(i, j, k);
                    const amrex::Real cell_floor =
                        old_value(i, j, k) >= temperature_bound
                            ? std::max(floor, temperature_bound)
                            : floor;
                    return {theta_implicit_mhd::admissible_step_fraction(
                        value(i, j, k), old_value(i, j, k), delta(i, j, k),
                        requested_step, cell_floor, theta)};
                });
        }
        step_fraction = std::min(
            step_fraction, amrex::get<0>(reduce_data.value(reduce_op)));
    }

    amrex::ParallelAllReduce::Min(step_fraction,
                                  amrex::ParallelContext::CommunicatorSub());
    if (step_fraction <= 1.0e-8_rt) {
        // The global clamp has effectively zeroed the Newton update: some
        // cell sits at its admissibility bound with a downward direction,
        // and the solve is about to lock. Report the offending cells
        // (block, index, state, old state, proposed change, floor) so
        // floor lockups are diagnosable from the log. Block ids follow
        // block_names: 0 = rho, 1 = U_e, 2 = E_i (total_energy or
        // dual_energy) or U_par (cgl), 3 = U_perp (cgl) or U_i
        // (dual_energy).
        constexpr amrex::Real report_threshold = 1.0e-8;
        for (int block = 0; block < num_blocks; ++block) {
            const amrex::Real floor = bounds.floors[block];
            const amrex::Real temperature_coefficient =
                bounds.temperature_coefficients[block];
            const int block_id = block;
            const amrex::MultiFab& value_mf =
                state.getMultiFabBlock(block_names[block], 0);
            const amrex::MultiFab& delta_mf =
                direction.getMultiFabBlock(block_names[block], 0);
            const amrex::MultiFab& old_mf =
                m_state_old.getMultiFabBlock(block_names[block], 0);
            for (amrex::MFIter mfi(value_mf); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                const auto value = value_mf.const_array(mfi);
                const auto delta = delta_mf.const_array(mfi);
                const auto old_value = old_mf.const_array(mfi);
                const auto old_density = old_density_mf.const_array(mfi);
                amrex::ParallelFor(
                    box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        const amrex::Real temperature_bound =
                            temperature_coefficient * old_density(i, j, k);
                        const amrex::Real cell_floor =
                            old_value(i, j, k) >= temperature_bound
                                ? std::max(floor, temperature_bound)
                                : floor;
                        if (theta_implicit_mhd::admissible_step_fraction(
                                value(i, j, k), old_value(i, j, k),
                                delta(i, j, k), requested_step, cell_floor,
                                theta) <= report_threshold) {
                            AMREX_DEVICE_PRINTF(
                                "LimitSolverStep: block %d pinned at "
                                "(%d,%d,%d): value %.6e old %.6e change "
                                "%.6e floor %.6e\n",
                                block_id, i, j, k, value(i, j, k),
                                old_value(i, j, k),
                                delta(i, j, k) * requested_step,
                                cell_floor);
                        }
                    });
            }
        }
        amrex::Gpu::streamSynchronize();
    }
    if (step_fraction < 1.0_rt) {
        step_fraction *= m_positivity_safety;
    }
    return requested_step * step_fraction;
}

amrex::Real
ThetaImplicitMHD::ProjectAndLimitSolverStep (const WarpXSolverVec& state,
                                             WarpXSolverVec& direction,
                                             const amrex::Real requested_step) const
{
    if (requested_step == 0.0_rt) {
        return requested_step;
    }
    // Componentwise projection onto the admissible box: clamp each
    // descending direction component so that ITS cell lands no lower
    // than its admissibility bound (the theta-stage image of the
    // positivity floor, (1-theta) old + theta floor) at the full
    // requested step. A GLOBAL step fraction lets one near-floor cell
    // with a large descent component throttle the entire update to
    // nothing and deadlock the line search (observed both for a cell
    // resting exactly on its bound and for one approaching it at
    // fraction ~1e-7); after this per-cell clamp the global fraction is
    // one, near-floor cells land exactly on their bounds, and every
    // backtracked candidate stays admissible because it is a convex
    // combination of the current value and the bound. This is the
    // projected (bound-constrained) Newton direction; the backtracking
    // Armijo search then acts on the projected direction, which is
    // standard practice. Only the Newton-update path calls this;
    // Jacobian probe vectors are never mutated.
    const bool dual_energy_closure = m_ion_closure == "dual_energy";
    const bool total_energy_closure =
        m_ion_closure == "total_energy" || dual_energy_closure;
    const bool cgl_closure = m_ion_closure == "cgl";
    const amrex::Real theta = m_theta;

    const int num_blocks = (cgl_closure || dual_energy_closure)
                               ? 4
                               : (total_energy_closure ? 3 : 2);
    const std::array<const char*, 4> block_names = {
        MassDensityName, ElectronEnergyName,
        cgl_closure ? IonParallelEnergyName : IonEnergyName,
        dual_energy_closure ? IonInternalEnergyName : IonPerpEnergyName};
    const AdmissibilityBounds bounds = MakeAdmissibilityBounds();
    const amrex::MultiFab& old_density_mf =
        m_state_old.getMultiFabBlock(MassDensityName, 0);
    const bool report_projections =
        (std::getenv("WARPX_MHD_REPORT_PROJECTIONS") != nullptr);
    // Record this solve's Newton active set: a per-block clamp mask (1 =
    // this component was projected onto its bound) plus global per-block
    // counts, overwritten on every call. FreeResidualNorm and
    // PinnedComponentReport consume them on the line-search failure
    // path; the counts also feed the always-on projection report below.
    m_projected_components = 0;
    m_projected_per_block = {0, 0, 0, 0};
    for (int block = 0; block < num_blocks; ++block) {
        const amrex::Real floor = bounds.floors[block];
        const amrex::Real temperature_coefficient =
            bounds.temperature_coefficients[block];
        const int block_id = block;
        const amrex::MultiFab& value_mf =
            state.getMultiFabBlock(block_names[block], 0);
        const amrex::MultiFab& old_mf =
            m_state_old.getMultiFabBlock(block_names[block], 0);
        amrex::MultiFab& delta_mf =
            direction.getMultiFabBlock(block_names[block], 0);
        amrex::iMultiFab& mask_mf = m_projection_masks[block];
        if (!mask_mf.ok()) {
            // Allocated once: the fluid box layout is fixed after Define.
            mask_mf.define(value_mf.boxArray(), value_mf.DistributionMap(),
                           1, 0);
        }
        mask_mf.setVal(0);
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Long> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (amrex::MFIter mfi(value_mf); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto value = value_mf.const_array(mfi);
            const auto old_value = old_mf.const_array(mfi);
            const auto old_density = old_density_mf.const_array(mfi);
            const auto delta = delta_mf.array(mfi);
            const auto mask = mask_mf.array(mfi);
            reduce_op.eval(
                box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                    const amrex::Real temperature_bound =
                        temperature_coefficient * old_density(i, j, k);
                    const amrex::Real cell_floor =
                        old_value(i, j, k) >= temperature_bound
                            ? std::max(floor, temperature_bound)
                            : floor;
                    const amrex::Real bound =
                        (1.0 - theta) * old_value(i, j, k) +
                        theta * cell_floor;
                    // Clamp descending components to land a small MARGIN
                    // above the bound (never exactly on it): limited
                    // Jacobian probes then always retain an admissible
                    // perturbation at floor-resident cells, avoiding the
                    // eps -> 0 probe abort. Inside the margin band the
                    // component is zeroed (no lifting: refill is the
                    // physics' job, not the projection's).
                    const amrex::Real margin =
                        1.0e-6 * (std::abs(old_value(i, j, k)) + cell_floor);
                    const amrex::Real target = std::min(
                        bound + margin - value(i, j, k), 0.0_rt);
                    const amrex::Real change =
                        delta(i, j, k) * requested_step;
                    if (change < target) {
                        if (report_projections) {
                            AMREX_DEVICE_PRINTF(
                                "ProjectDirection: block %d at (%d,%d,%d): "
                                "value %.6e old %.6e change %.6e floor %.6e\n",
                                block_id, i, j, k, value(i, j, k),
                                old_value(i, j, k), change, cell_floor);
                        }
                        delta(i, j, k) = target / requested_step;
                        mask(i, j, k) = 1;
                        return {1};
                    }
                    return {0};
                });
        }
        m_projected_per_block[block] =
            amrex::get<0>(reduce_data.value(reduce_op));
    }
    amrex::ParallelAllReduce::Sum(
        m_projected_per_block.data(),
        static_cast<int>(m_projected_per_block.size()),
        amrex::ParallelContext::CommunicatorSub());
    for (int block = 0; block < num_blocks; ++block) {
        m_projected_components += m_projected_per_block[block];
    }
    if (m_projected_components > 0) {
        amrex::Print() << "Newton: projected " << m_projected_components
                       << " direction components onto admissibility bounds ("
                       << PinnedComponentReport() << ")\n";
    }
    return LimitSolverStep(state, direction, requested_step);
}

amrex::Real
ThetaImplicitMHD::FreeResidualNorm (const WarpXSolverVec& residual,
                                    amrex::Real& pinned_defect,
                                    amrex::Long& num_pinned) const
{
    // Fast path -- empty active set: the free norm IS the full norm.
    // This guarantees bit-identical failure-path behavior whenever the
    // last projection clamped nothing.
    num_pinned = m_projected_components;
    pinned_defect = 0.0_rt;
    if (m_projected_components == 0) {
        return residual.norm2();
    }

    const bool dual_energy_closure = m_ion_closure == "dual_energy";
    const bool total_energy_closure =
        m_ion_closure == "total_energy" || dual_energy_closure;
    const bool cgl_closure = m_ion_closure == "cgl";
    const int num_blocks = (cgl_closure || dual_energy_closure)
                               ? 4
                               : (total_energy_closure ? 3 : 2);
    const std::array<const char*, 4> block_names = {
        MassDensityName, ElectronEnergyName,
        cgl_closure ? IonParallelEnergyName : IonEnergyName,
        dual_energy_closure ? IonInternalEnergyName : IonPerpEnergyName};

    // Pinned-defect squared norm: the residual restricted to the clamped
    // components, in the SAME per-block scaling as
    // WarpXSolverVec::dotProduct, so free^2 + pinned^2 = norm2()^2
    // exactly. The fluid blocks are cell centered -- the valid boxes
    // partition the domain, so no DOF ownership mask is needed.
    amrex::Real pinned_sq = 0.0_rt;
    for (int block = 0; block < num_blocks; ++block) {
        const amrex::iMultiFab& mask_mf = m_projection_masks[block];
        if (!mask_mf.ok()) {
            continue;
        }
        const amrex::MultiFab& residual_mf =
            residual.getMultiFabBlock(block_names[block], 0);
        amrex::Real block_scale = 1.0_rt;
        for (const auto& spec : residual.getMultiFabBlockSpecs()) {
            if (spec.name == block_names[block]) {
                block_scale = spec.scale;
            }
        }
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (amrex::MFIter mfi(residual_mf); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto residual_arr = residual_mf.const_array(mfi);
            const auto mask = mask_mf.const_array(mfi);
            reduce_op.eval(
                box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                    return {mask(i, j, k) != 0
                                ? residual_arr(i, j, k) * residual_arr(i, j, k)
                                : 0.0_rt};
                });
        }
        const amrex::Real inverse_scale = 1.0_rt / block_scale;
        pinned_sq += inverse_scale * inverse_scale *
                     amrex::get<0>(reduce_data.value(reduce_op));
    }
    amrex::ParallelAllReduce::Sum(pinned_sq,
                                  amrex::ParallelContext::CommunicatorSub());

    const amrex::Real full_sq = residual.dotProduct(residual);
    pinned_defect = std::sqrt(pinned_sq);
    // max() guards round-off: the pinned part is a subset of the full sum.
    return std::sqrt(std::max(full_sq - pinned_sq, 0.0_rt));
}

std::string ThetaImplicitMHD::PinnedComponentReport () const
{
    if (m_projected_components == 0) {
        return {};
    }
    const bool dual_energy_closure = m_ion_closure == "dual_energy";
    const bool total_energy_closure =
        m_ion_closure == "total_energy" || dual_energy_closure;
    const bool cgl_closure = m_ion_closure == "cgl";
    const int num_blocks = (cgl_closure || dual_energy_closure)
                               ? 4
                               : (total_energy_closure ? 3 : 2);
    const std::array<const char*, 4> block_labels = {
        "mass", "electron_energy",
        cgl_closure ? "ion_par_energy" : "ion_energy",
        dual_energy_closure ? "ion_internal_energy" : "ion_perp_energy"};
    std::string report;
    for (int block = 0; block < num_blocks; ++block) {
        if (!report.empty()) {
            report += ", ";
        }
        report += block_labels[block];
        report += ' ';
        report += std::to_string(m_projected_per_block[block]);
    }
    return report;
}

theta_implicit_mhd::FluxParameters ThetaImplicitMHD::MakeFluxParameters () const
{
    theta_implicit_mhd::FluxParameters flux_parameters = {
        m_mass_density_floor,
        m_ion_charge_to_mass,
        m_gamma_e,
        m_gamma_i,
        m_reference_mass_density,
        m_reference_ion_pressure,
        m_electron_pressure_floor,
        m_ion_pressure_floor,
        m_hybrid_pic_model->m_include_hall_term,
        m_fluid_flux == "rusanov",
        m_fluid_flux == "hllc",
        // dual_energy keeps the total-energy machinery live (the E_i
        // recovery, its donor gates, and its flux channel are assembled
        // exactly as under total_energy); the dual flag below adds the
        // blended-pressure loaders and the U_i channel on top.
        m_ion_closure == "total_energy" || m_ion_closure == "dual_energy",
        // Under hlld the barotropic fan closure decouples the WHOLE
        // dissipation structure from E_i: the signal bounds (Davis fast
        // speeds, which parametrize every channel's upwind dissipation
        // and the corner-EMF alphas) as well as S_M and the star states.
        // The physical fluxes keep the consistent p_i(E_i) either way.
        m_hllc_signal_closure == "barotropic" ||
            (m_use_hlld && m_hlld_fan_closure == "barotropic"),
        m_hllc_contact_blend,
        m_theta};
    // Only read by the hlld kernels (the default 1.0 is unit-agnostic for
    // the host harness); harmless for the other flux paths.
    flux_parameters.mu0 = PhysConst::mu0;
    flux_parameters.hlld_kappa_signal = m_hlld_kappa_signal;
    flux_parameters.hlld_kappa_contact = m_hlld_kappa_contact;
    flux_parameters.hlld_kappa_bn = m_hlld_kappa_bn;
    flux_parameters.hlld_kappa_denominator = m_hlld_kappa_denominator;
    flux_parameters.hlld_barotropic_fan =
        (m_hlld_fan_closure == "barotropic");
    flux_parameters.hlld_ion_energy_star =
        (m_hlld_ion_energy_flux == "star");
    flux_parameters.hlld_all_speed = m_hlld_all_speed;
    // total_energy_closure (set above from m_ion_closure) stays false
    // under cgl: load_cell_state_cgl overwrites the polytropic pressure
    // with p_eff and the E_i machinery stays dormant.
    flux_parameters.cgl_closure = (m_ion_closure == "cgl");
    flux_parameters.dual_energy_closure = (m_ion_closure == "dual_energy");
    flux_parameters.dual_energy_internal_cutoff =
        m_dual_energy_internal_cutoff;
    flux_parameters.dual_energy_internal_old_max =
        m_dual_energy_internal_old_max;
    flux_parameters.pressure_corner_width_fraction =
        m_pressure_corner_width_fraction;
    // Per-step frozen pedestal state (0 while the pedestal is off):
    // anchors the per-block drain gates and the halo source taper.
    flux_parameters.halo_pedestal = m_halo_pedestal_density;
    flux_parameters.halo_pedestal_electron_energy =
        m_halo_pedestal_electron_energy;
    flux_parameters.halo_pedestal_ion_internal = m_halo_pedestal_ion_internal;
    flux_parameters.halo_pedestal_ion_parallel = m_halo_pedestal_ion_parallel;
    flux_parameters.halo_pedestal_ion_perp = m_halo_pedestal_ion_perp;
    return flux_parameters;
}

void ThetaImplicitMHD::ComputeFluidRHS (WarpXSolverVec& rhs, const amrex::Real time) const
{
    if (m_use_recast) {
        // Conservative form: read the precomputed per-face fluxes (one
        // flux evaluation per face; the same evaluation feeds the ideal
        // EMF) instead of re-evaluating fluxes cell-by-cell.
        ComputeFluidRHSFromFaceFluxes(rhs, time);
        return;
    }
    // Defensive: the CGL and dual-energy closures are only wired into the
    // recast face-flux path above (and the input parser asserts a recast
    // flux), so the E-based fluid RHS below must never see them.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_ion_closure != "cgl",
        "implicit_mhd.ion_closure = cgl requires implicit_mhd.fluid_flux = "
        "hlld");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_ion_closure != "dual_energy",
        "implicit_mhd.ion_closure = dual_energy requires "
        "implicit_mhd.fluid_flux = hlld or central");

    const amrex::MultiFab& density = *m_WarpX->m_fields.get(MassDensityName, 0);
    const amrex::MultiFab& momentum = *m_WarpX->m_fields.get(MomentumDensityName, 0);
    const amrex::MultiFab& electron_energy = *m_WarpX->m_fields.get(ElectronEnergyName, 0);
    const amrex::MultiFab& ion_energy = *m_WarpX->m_fields.get(IonEnergyName, 0);
    const amrex::MultiFab& current = *m_WarpX->m_fields.get(TotalCurrentCCName, 0);
    const amrex::MultiFab& magnetic_field = *m_WarpX->m_fields.get(MagneticFieldCCName, 0);
    // Holmstrom cell switching uses the beginning-of-step state so the
    // vacuum mask is frozen during the Newton solve; differentiating the
    // blend through the live state puts O(1/width) entries in the Jacobian
    // that the (unpreconditioned) linear solver cannot handle.
    const amrex::MultiFab& old_density =
        *m_WarpX->m_fields.get(OldMassDensityName, 0);
    const amrex::MultiFab& old_momentum =
        *m_WarpX->m_fields.get(OldMomentumDensityName, 0);
    const amrex::MultiFab& old_electron_energy =
        *m_WarpX->m_fields.get(OldElectronEnergyName, 0);
    const amrex::MultiFab& old_ion_energy =
        *m_WarpX->m_fields.get(OldIonEnergyName, 0);

    const bool total_energy_closure = m_ion_closure == "total_energy";
    amrex::MultiFab& density_rhs = rhs.getMultiFabBlock(MassDensityName, 0);
    amrex::MultiFab& momentum_rhs = rhs.getMultiFabBlock(MomentumDensityName, 0);
    amrex::MultiFab& electron_energy_rhs = rhs.getMultiFabBlock(ElectronEnergyName, 0);
    amrex::MultiFab* const ion_energy_rhs =
        total_energy_closure ? &rhs.getMultiFabBlock(IonEnergyName, 0) : nullptr;

    const auto inverse_cell_size = physical_inverse_cell_size(m_WarpX->Geom(0));
    const auto active = active_physical_directions();
#if defined(WARPX_DIM_RZ)
    const amrex::Real radial_cell_size = m_WarpX->Geom(0).CellSize(0);
    const amrex::Real radial_lower = m_WarpX->Geom(0).ProbLo(0);
#endif
    // Also read by the electron-ion equilibration's live ion pressure,
    // so hoisted out of the RZ geometric-source guard.
    const amrex::Real gamma_i_minus_one = m_gamma_i - 1.0_rt;
    const amrex::Real ion_energy_floor =
        total_energy_closure ? m_ion_pressure_floor / (m_gamma_i - 1.0_rt)
                             : 0.0_rt;
    const amrex::Real theta_dt = m_theta * m_dt;
    // theta-extrapolation weight for the floor limiters' end-of-step
    // gating (see face_flux): U^{n+1} = U^theta (1 + ext) - U^n ext.
    const amrex::Real extrapolation_weight = (1.0_rt - m_theta) / m_theta;
    const amrex::Real density_floor = m_mass_density_floor;
    const amrex::Real charge_to_mass = m_ion_charge_to_mass;
    const amrex::Real gamma_e_minus_one = m_gamma_e - 1.0_rt;
    const amrex::Real pressure_floor = m_electron_pressure_floor;
    const bool evolve_ion_fluid = m_evolve_ion_fluid;
    const bool include_joule_heating = m_include_joule_heating;
    // Electron-ion equilibration (see m_electron_ion_equilibration; the
    // The reference code's eq_brate exchange): per-solve-uniform branch, so the OFF
    // path performs no arithmetic at all. The rate coefficient folds
    // The reference code's 4.75e-15 constant, the m_p/m_i mass ratio, and the
    // user scale; the cap is the theta-scheme monotonicity bound of the
    // pair-difference decay (the reference code's backward Euler needs none).
    const bool ei_equilibration = m_electron_ion_equilibration > 0.0_rt;
    const amrex::Real ei_rate_coefficient =
        m_electron_ion_equilibration * 4.75e-15_rt * PhysConst::m_p *
        m_ion_charge_to_mass / PhysConst::q_e;
    const amrex::Real ei_rate_cap =
        m_theta < 1.0_rt
            ? 1.0_rt / ((m_gamma_e + m_gamma_i - 2.0_rt) *
                        (1.0_rt - m_theta) * m_dt)
            : std::numeric_limits<amrex::Real>::max();
    const amrex::Real ei_inverse_ion_mass =
        m_ion_charge_to_mass / PhysConst::q_e;
    const amrex::Real ei_ion_pressure_floor = m_ion_pressure_floor;
    // Holmstrom-style vacuum cell switching: below the threshold the fluid
    // is passive dust with frozen momentum (conservative advection of mass
    // and energy remains, so exchange with neighboring plasma cells is
    // intact, and arriving momentum is absorbed).
    const bool holmstrom_vacuum = m_vacuum_mass_density > 0.0_rt;
    const amrex::Real vacuum_mass_density = m_vacuum_mass_density;
    const amrex::Real vacuum_drag_rate = m_vacuum_drag_rate;
    // Halo source taper (see ComputeFluidRHSFromFaceFluxes): reactive
    // work sources taper C^1-smoothly to zero below twice the pedestal;
    // identically 1 when the pedestal is off.
    const amrex::Real halo_pedestal = m_halo_pedestal_density;
    // Pedestal-band velocity relaxation (see m_halo_pedestal_drag_rate):
    // engages with the complement of the halo source taper, i.e. full
    // rate at the pedestal and exactly zero at/above twice it.
    const amrex::Real halo_pedestal_drag_rate = m_halo_pedestal_drag_rate;
    // Pedestal-band ion-energy relaxation (see
    // m_halo_pedestal_energy_rate): same mask complement as the drag.
    // The internal-energy target is the per-solve frozen pedestal image,
    // clamped at the positivity floor so the drain can never demand an
    // inadmissible state.
    const amrex::Real halo_pedestal_energy_rate = m_halo_pedestal_energy_rate;
    const amrex::Real halo_pedestal_ion_internal =
        std::max(m_halo_pedestal_ion_internal, ion_energy_floor);
    // Floor-consistency relaxation source (see m_floor_consistency_rate
    // and floor_consistency_deficit): one-sided per-cell supply at the
    // SAME theta-stage admissibility bounds the Newton projection
    // enforces, per bounded block, capped at 1/(theta dt) so the
    // per-solve supplied increment never exceeds the deficit scale plus
    // half a rectifier width (no overshoot). Bounds and rate are
    // per-solve constants (frozen-coefficient idiom): JFNK-exact. The
    // OFF path adds nothing at all -- bit-identical by construction.
    const bool floor_consistency = m_floor_consistency_rate > 0.0_rt;
    const amrex::Real floor_supply_rate =
        std::min(m_floor_consistency_rate, 1.0_rt / theta_dt);
    const amrex::Real fc_width = m_floor_consistency_width_fraction;
    const amrex::Real theta = m_theta;
    const AdmissibilityBounds admissibility = MakeAdmissibilityBounds();
    const amrex::Real fc_mass_floor = admissibility.floors[0];
    const amrex::Real fc_mass_coefficient =
        admissibility.temperature_coefficients[0];
    const amrex::Real fc_electron_floor = admissibility.floors[1];
    const amrex::Real fc_electron_coefficient =
        admissibility.temperature_coefficients[1];
    const amrex::Real fc_ion_floor = admissibility.floors[2];
    const amrex::Real fc_ion_coefficient =
        admissibility.temperature_coefficients[2];
    const auto eta = m_hybrid_pic_model->m_eta;
    // Joule heating evaluates eta at the SAME hybrid-floored density as
    // Ohm's law, so the electron heating rate matches the field's eta J.
    const amrex::Real eta_density_floor = OhmMassDensityFloor();
    // Reference-code-style Ohm-current Joule quench (implicit_mhd.joule_ohm_current,
    // see m_joule_ohm_current): in cells where the FIELD advance is
    // diffusion dominated -- dt eta_field / mu0 > min(dx)^2, precomputed
    // here as eta_field > mu0 min(dx)^2 / dt -- the deposit's |J|^2 is
    // replaced by |E|^2 / eta_field^2 with E the solved stage electric
    // field (FillCellCenteredOhmElectricField) and eta_field the SAME
    // floored/boosted composition the field solve uses: the site's own
    // user eta through vacuum_keyed_resistivity keyed to the Ohm guard,
    // then the wall-band override folded through its maximum (the
    // cell-centered convention of GetMHDFieldResistivityCCForPC). The
    // COEFFICIENT keeps the un-boosted user eta. Binary per-cell switch,
    // exactly as the reference code.
    const bool joule_ohm_current =
        m_joule_ohm_current && include_joule_heating;
    const amrex::Real joule_min_cell_size =
        1.0_rt / std::max({inverse_cell_size[0], inverse_cell_size[1],
                           inverse_cell_size[2]});
    const amrex::Real joule_quench_eta_threshold =
        joule_ohm_current ? PhysConst::mu0 * joule_min_cell_size *
                                joule_min_cell_size / m_dt
                          : std::numeric_limits<amrex::Real>::max();
    const amrex::Real joule_vacuum_division_guard =
        charge_to_mass * m_mass_density_floor;
    const amrex::Real joule_vacuum_eta_scale =
        PhysConst::mu0 * m_vacuum_resistivity_diffusivity;
    // Vacuum-boost reference of the quench's eta_field (and hence of the
    // diffusion-dominance criterion): the per-step frozen dynamic
    // Reference (VacuumReferenceMassDensity), matching the field
    // advance. The eta_joule ARGUMENT floor keeps the static Ohm guard.
    const amrex::Real joule_vacuum_reference =
        charge_to_mass * VacuumReferenceMassDensity();
    const WallBandEtaOverrideView joule_band_view =
        m_wall_mask.BandEtaOverrideView();
    const int* const joule_band_cc = joule_band_view.first_band_cc;
    const amrex::Real joule_band_eta = joule_band_view.eta_override;
    const amrex::MultiFab& ohm_electric_cc =
        *m_WarpX->m_fields.get(OhmElectricFieldCCName, 0);
    const theta_implicit_mhd::FluxParameters flux_parameters =
        MakeFluxParameters();

    for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto rho = density.const_array(mfi);
        const auto rho_old = old_density.const_array(mfi);
        const auto mom_old = old_momentum.const_array(mfi);
        const auto energy_old = old_electron_energy.const_array(mfi);
        const auto ion_e_old = old_ion_energy.const_array(mfi);
        const auto mom = momentum.const_array(mfi);
        const auto energy = electron_energy.const_array(mfi);
        const auto ion_e = ion_energy.const_array(mfi);
        const auto j_plasma = current.const_array(mfi);
        const auto magnetic = magnetic_field.const_array(mfi);
        const auto e_ohm = ohm_electric_cc.const_array(mfi);
        const auto rho_increment = density_rhs.array(mfi);
        const auto momentum_increment = momentum_rhs.array(mfi);
        const auto energy_increment = electron_energy_rhs.array(mfi);
        const auto ion_energy_increment =
            total_energy_closure ? ion_energy_rhs->array(mfi)
                                 : amrex::Array4<amrex::Real>{};

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real divergence_momentum = 0.0_rt;
            amrex::Real divergence_energy_flux = 0.0_rt;
            amrex::Real divergence_ion_energy_flux = 0.0_rt;
            amrex::Real divergence_electron_velocity = 0.0_rt;
            amrex::Real divergence_momentum_flux[3] = {0.0_rt, 0.0_rt, 0.0_rt};

            for (int direction = 0; direction < 3; ++direction) {
                if (!active[direction]) {
                    continue;
                }
                int ih = i;
                int jh = j;
                int kh = k;
                int il = i;
                int jl = j;
                int kl = k;
                shift_index(ih, jh, kh, direction, 1);
                shift_index(il, jl, kl, direction, -1);

                const auto high_flux = theta_implicit_mhd::face_flux(
                    rho, mom, energy, ion_e, j_plasma, rho_old, mom_old,
                    energy_old, ion_e_old, i, j, k, ih, jh, kh,
                    direction, flux_parameters);
                const auto low_flux = theta_implicit_mhd::face_flux(
                    rho, mom, energy, ion_e, j_plasma, rho_old, mom_old,
                    energy_old, ion_e_old, il, jl, kl, i, j, k,
                    direction, flux_parameters);
                const amrex::Real derivative_scale = inverse_cell_size[direction];

                // Face-area weights for the finite-volume divergence: unity in
                // Cartesian geometry; in RZ the radial faces are weighted by
                // r_face / r_center so that (w_hi F_hi - w_lo F_lo) / dr is the
                // cylindrical divergence (1/r) d(r F)/dr. The innermost face
                // sits at r = 0 and drops out automatically.
                amrex::Real weight_high = 1.0_rt;
                amrex::Real weight_low = 1.0_rt;
#if defined(WARPX_DIM_RZ)
                if (direction == 0) {
                    const amrex::Real radial_center =
                        radial_lower + (i + 0.5_rt) * radial_cell_size;
                    weight_high =
                        (radial_lower + (i + 1.0_rt) * radial_cell_size) / radial_center;
                    weight_low = (radial_lower + i * radial_cell_size) / radial_center;
                }
#endif

                divergence_momentum +=
                    derivative_scale *
                    (weight_high * high_flux.mass - weight_low * low_flux.mass);

                for (int component = 0; component < 3; ++component) {
                    divergence_momentum_flux[component] +=
                        derivative_scale * (weight_high * high_flux.momentum[component] -
                                            weight_low * low_flux.momentum[component]);
                }

                divergence_energy_flux +=
                    derivative_scale * (weight_high * high_flux.electron_energy -
                                        weight_low * low_flux.electron_energy);
                if (total_energy_closure) {
                    divergence_ion_energy_flux +=
                        derivative_scale * (weight_high * high_flux.ion_energy -
                                            weight_low * low_flux.ion_energy);
                }
                divergence_electron_velocity +=
                    derivative_scale * (weight_high * high_flux.electron_velocity -
                                        weight_low * low_flux.electron_velocity);
            }

#if defined(WARPX_DIM_RZ)
            {
                // Cylindrical (m=0) geometric source terms of the momentum
                // tensor divergence that are not expressible as face
                // differences: (div T)_r includes -T_theta_theta / r and
                // (div T)_theta includes +T_theta_r / r, with
                // T = rho u u + (P_i + P_e) I.
                const amrex::Real radial_center =
                    radial_lower + (i + 0.5_rt) * radial_cell_size;
                const amrex::Real inverse_radius = 1.0_rt / radial_center;
                const amrex::Real safe_density =
                    std::max(rho(i, j, k), density_floor);
                const amrex::Real velocity_r = mom(i, j, k, 0) / safe_density;
                const amrex::Real velocity_theta = mom(i, j, k, 1) / safe_density;
                amrex::Real pressure_i;
                if (total_energy_closure) {
                    const amrex::Real kinetic_energy =
                        0.5_rt *
                        (mom(i, j, k, 0) * mom(i, j, k, 0) +
                         mom(i, j, k, 1) * mom(i, j, k, 1) +
                         mom(i, j, k, 2) * mom(i, j, k, 2)) /
                        safe_density;
                    pressure_i =
                        gamma_i_minus_one *
                        std::max(ion_e(i, j, k) - kinetic_energy, ion_energy_floor);
                } else {
                    pressure_i = theta_implicit_mhd::ion_pressure(safe_density,
                                                                  flux_parameters);
                }
                const amrex::Real total_pressure =
                    pressure_i +
                    std::max(gamma_e_minus_one * energy(i, j, k), pressure_floor);
                divergence_momentum_flux[0] -=
                    inverse_radius *
                    (total_pressure + mom(i, j, k, 1) * velocity_theta);
                divergence_momentum_flux[1] +=
                    inverse_radius * mom(i, j, k, 1) * velocity_r;
            }
#endif

            // Smooth plasma/vacuum blend of the beginning-of-step density
            // (frozen per step: cell switching). The weight scales the whole
            // fluid update: vacuum cells are static passive dust. Because the
            // weight shrinks with the (frozen) density, rarefaction in the
            // blend band is self-limiting -- a draining cell's update rate
            // falls off exponentially with each step, so the state never
            // reaches the positivity floor and the bounded Newton update
            // never locks. A plasma front compressing into the vacuum region
            // still propagates: blend-band cells gain mass at a reduced rate,
            // which raises their weight on the following step.
            const amrex::Real plasma_weight =
                holmstrom_vacuum
                    ? 0.5_rt * (1.0_rt + std::tanh(
                                             (rho_old(i, j, k) - vacuum_mass_density) /
                                             (0.3_rt * vacuum_mass_density)))
                    : 1.0_rt;

            rho_increment(i, j, k) =
                evolve_ion_fluid
                    ? -theta_dt * plasma_weight * divergence_momentum
                    : 0.0_rt;

            const amrex::Real jx = j_plasma(i, j, k, 0);
            const amrex::Real jy = j_plasma(i, j, k, 1);
            const amrex::Real jz = j_plasma(i, j, k, 2);
            const amrex::Real bx = magnetic(i, j, k, 0);
            const amrex::Real by = magnetic(i, j, k, 1);
            const amrex::Real bz = magnetic(i, j, k, 2);
            const amrex::Real j_cross_b[3] = {jy * bz - jz * by, jz * bx - jx * bz,
                                              jx * by - jy * bx};
            // Linear velocity relaxation of vacuum dust: blend-band cells
            // feel a residual force with almost no inertia, so without
            // friction they run away to many Alfven speeds. The drag is
            // smooth and diagonal (JFNK-friendly) and bounds the dust at a
            // terminal velocity F/(rho nu) instead.
            const amrex::Real vacuum_drag =
                vacuum_drag_rate * (1.0_rt - plasma_weight);
            // Pedestal-band drag (see m_halo_pedestal_drag_rate): the
            // mass gates hold the band's density flux at zero, but its
            // momentum rows still integrate the surrounding truncation
            // forcing; the diagonal drag bounds the band at a terminal
            // velocity instead of letting that inconsistency pin the
            // Newton residual. Keyed to the step-old density like the
            // halo source taper (a per-solve constant mask).
            const amrex::Real halo_drag =
                halo_pedestal_drag_rate *
                (1.0_rt - theta_implicit_mhd::floor_outflow_limiter(
                              rho_old(i, j, k), halo_pedestal));
            for (int component = 0; component < 3; ++component) {
                momentum_increment(i, j, k, component) =
                    evolve_ion_fluid
                        ? theta_dt * (plasma_weight *
                                          (-divergence_momentum_flux[component] +
                                           j_cross_b[component]) -
                                      (vacuum_drag + halo_drag) *
                                          mom(i, j, k, component))
                        : 0.0_rt;
            }

            const amrex::Real pressure_e =
                std::max(gamma_e_minus_one * energy(i, j, k), pressure_floor);
            const amrex::Real current_magnitude = std::sqrt(jx * jx + jy * jy + jz * jz);
            const amrex::Real charge_density =
                charge_to_mass * std::max(rho(i, j, k), eta_density_floor);
            // Te [K] of the (rho, Te, J, t) parser: the same
            // temperature-primary cell ratio as FillFluidSources, from the
            // already-recovered pressure over the already-floored density.
            const amrex::Real temperature_e =
                pressure_e * PhysConst::q_e / (charge_density * PhysConst::kb);
            amrex::Real joule_heating = 0.0_rt;
            if (include_joule_heating) {
                const amrex::Real eta_joule = eta(
                    charge_density, temperature_e, current_magnitude, time);
                amrex::Real square_current =
                    current_magnitude * current_magnitude;
                if (joule_ohm_current) {
                    // Ohm-current quench (see the host constants): the
                    // cell-centered twin of the field advance's
                    // eta_field, from the site's own eta_joule.
                    amrex::Real eta_field =
                        theta_implicit_mhd::vacuum_keyed_resistivity(
                            eta_joule, charge_to_mass * rho(i, j, k),
                            joule_vacuum_reference,
                            joule_vacuum_division_guard,
                            joule_vacuum_eta_scale);
                    if (joule_band_cc != nullptr && i >= joule_band_cc[j]) {
                        eta_field = std::max(eta_field, joule_band_eta);
                    }
                    if (eta_field > joule_quench_eta_threshold) {
                        const amrex::Real e_x = e_ohm(i, j, k, 0);
                        const amrex::Real e_y = e_ohm(i, j, k, 1);
                        const amrex::Real e_z = e_ohm(i, j, k, 2);
                        square_current =
                            (e_x * e_x + e_y * e_y + e_z * e_z) /
                            (eta_field * eta_field);
                    }
                }
                joule_heating = eta_joule * square_current;
            }
            amrex::Real pressure_work =
                -pressure_e * divergence_electron_velocity;
            const bool guard_floors =
                flux_parameters.use_hllc || flux_parameters.use_rusanov;
            const amrex::Real energy_end =
                energy(i, j, k) * (1.0_rt + extrapolation_weight) -
                energy_old(i, j, k) * extrapolation_weight;
            if (guard_floors && pressure_work < 0.0_rt) {
                // Riemann fluxes: close the expansion (pdV) cooling as U_e
                // approaches its floor -- the face limiter guards only
                // the advective flux, and this non-conservative source
                // can otherwise demand sub-floor states in rarefactions.
                // Gate on the theta-extrapolated end-of-step value, like
                // the face limiters (the admissibility bound protects the
                // end-of-step state, not the theta stage).
                pressure_work *= theta_implicit_mhd::floor_outflow_limiter(
                    energy_end, pressure_floor / gamma_e_minus_one);
            }
            // Halo source taper: pedestal-band cells are numerical mass
            // with no reactive response of their own.
            const amrex::Real halo_source_taper =
                theta_implicit_mhd::floor_outflow_limiter(
                    rho_old(i, j, k), halo_pedestal);
            pressure_work *= halo_source_taper;
            energy_increment(i, j, k) =
                theta_dt * plasma_weight *
                (-divergence_energy_flux + pressure_work + joule_heating);
            // Electron-ion equilibration (see the host constants above
            // and m_electron_ion_equilibration; the reference code's eq_brate
            // exchange): the STEP-OLD frozen Spitzer rate times the LIVE
            // linear pair term (p_i - p_e) -- +Q to the electron energy
            // here, -Q to the ion energy below, under the SAME
            // symmetric envelope (plasma_weight, halo taper), so
            // U_e + E_i changes by exactly zero. Appended after the base
            // increments under a per-solve-uniform branch: the OFF path
            // performs no arithmetic at all.
            amrex::Real equilibration_heating = 0.0_rt;
            if (ei_equilibration) {
                const amrex::Real ei_number_density =
                    std::max(rho_old(i, j, k), density_floor) *
                    ei_inverse_ion_mass;
                const amrex::Real ei_te_ev =
                    std::max(gamma_e_minus_one * energy_old(i, j, k),
                             pressure_floor) /
                    (ei_number_density * PhysConst::q_e);
                const amrex::Real nu_ei =
                    theta_implicit_mhd::electron_ion_equilibration_rate(
                        ei_number_density, ei_te_ev, ei_rate_coefficient,
                        ei_rate_cap);
                amrex::Real ei_kinetic = 0.0_rt;
                for (int component = 0; component < 3; ++component) {
                    ei_kinetic += mom(i, j, k, component) *
                                  mom(i, j, k, component);
                }
                ei_kinetic *=
                    0.5_rt / std::max(rho(i, j, k), density_floor);
                const amrex::Real pressure_i_live = std::max(
                    gamma_i_minus_one * (ion_e(i, j, k) - ei_kinetic),
                    ei_ion_pressure_floor);
                equilibration_heating = halo_source_taper * nu_ei *
                                        (pressure_i_live - pressure_e);
                energy_increment(i, j, k) +=
                    theta_dt * plasma_weight * equilibration_heating;
            }

            if (total_energy_closure) {
                const amrex::Real safe_density =
                    std::max(rho(i, j, k), density_floor);
                amrex::Real kinetic_energy = 0.0_rt;
                for (int component = 0; component < 3; ++component) {
                    kinetic_energy += mom(i, j, k, component) *
                                      mom(i, j, k, component);
                }
                kinetic_energy *= 0.5_rt / safe_density;
                // End-of-step internal proxy for the sink gates, from the
                // theta-extrapolated E_i, momentum, and density.
                amrex::Real kinetic_end = 0.0_rt;
                for (int component = 0; component < 3; ++component) {
                    const amrex::Real mom_end =
                        mom(i, j, k, component) *
                            (1.0_rt + extrapolation_weight) -
                        mom_old(i, j, k, component) * extrapolation_weight;
                    kinetic_end += mom_end * mom_end;
                }
                kinetic_end *=
                    0.5_rt /
                    std::max(rho(i, j, k) * (1.0_rt + extrapolation_weight) -
                                 rho_old(i, j, k) * extrapolation_weight,
                             density_floor);
                const amrex::Real internal_proxy_end =
                    ion_e(i, j, k) * (1.0_rt + extrapolation_weight) -
                    ion_e_old(i, j, k) * extrapolation_weight - kinetic_end;
                // Work done by the Lorentz force on the ion fluid, from the
                // same cell-centered J x B the momentum equation uses, so
                // field-to-fluid energy exchange is discretely consistent.
                amrex::Real lorentz_work = 0.0_rt;
                for (int component = 0; component < 3; ++component) {
                    lorentz_work +=
                        mom(i, j, k, component) * j_cross_b[component];
                }
                lorentz_work /= safe_density;
                // Enthalpy-form electron-pressure coupling: the E_i face
                // flux carries the total pressure, u (E_i + P_i + P_e), so
                // the conservative remainder of the electron-pressure work
                // is + P_e div(u_e), evaluated with the SAME clamped
                // pressure and the SAME face-averaged velocity divergence
                // as the electron equation's -P_e div(u_e) work above. The
                // pair cancels identically cell-by-cell (away from the
                // floor limiters), so sum(E_i + U_e) changes only through
                // telescoping face fluxes and is conserved exactly on a
                // periodic mesh.
                amrex::Real ion_pressure_work =
                    pressure_e * divergence_electron_velocity;
                if (guard_floors && ion_pressure_work < 0.0_rt) {
                    // close the compression sink as U_i approaches
                    // its floor, mirroring the electron pdV treatment.
                    ion_pressure_work *=
                        theta_implicit_mhd::floor_outflow_limiter(
                            internal_proxy_end, ion_energy_floor);
                }
                if (guard_floors && lorentz_work < 0.0_rt) {
                    // Likewise close a decelerating Lorentz sink near the
                    // floor: a persistently negative u.(JxB) in a low-E_i
                    // cell (halo fluid braking against the separatrix
                    // force) otherwise demands sub-floor states that lock
                    // the Newton line search.
                    lorentz_work *=
                        theta_implicit_mhd::floor_outflow_limiter(
                            internal_proxy_end, ion_energy_floor);
                }
                // Halo source taper hoisted above the electron pdV term.
                ion_pressure_work *= halo_source_taper;
                lorentz_work *= halo_source_taper;
                // Kinetic-energy drain matched to the pedestal-band
                // momentum drag: E_i carries the kinetic energy under
                // this closure, so the drag must remove exactly the
                // kinetic decay -nu |m|^2/rho it causes (at theta = 1/2
                // the pairing is discretely exact) -- otherwise the
                // relaxed band would read as spurious internal heating.
                // Outside plasma_weight, like the momentum drag term.
                const amrex::Real drag_kinetic_drain =
                    halo_drag * 2.0_rt * kinetic_energy;
                // Pedestal-band ion-energy relaxation (see
                // m_halo_pedestal_energy_rate): drains ONLY the internal
                // part E_i - |m|^2/(2 rho) toward the frozen pedestal
                // image -- the momentum drag owns the kinetic channel
                // (its matched drain above), so the composition is
                // triangular in (KE, e_int) and neither term
                // double-counts the other. ONE-SIDED (rectified): the
                // C^1 gate closes the term where the internal part sits
                // at or below the pedestal image, so it can never act as
                // a SOURCE -- a two-sided form was measured to pump the
                // KE-dominated wall-band cells (theta-stage E_i < KE,
                // the big-minus-big corners) toward their growing
                // KE-following target at the relaxation rate, a secular
                // runaway of the band's conservative E_i stock. Full
                // exact rate at/above twice the image, like the drain
                // gates. NARROW density window (see the face-flux path
                // for the measurement): full at/below 1.125 rho_ped,
                // exactly zero at/above 1.25 rho_ped -- an octave inside
                // the drag mask, keyed to the step-old density. Outside
                // plasma_weight, like the drag terms.
                const amrex::Real internal_energy =
                    ion_e(i, j, k) - kinetic_energy;
                const amrex::Real energy_relax_drain =
                    halo_pedestal_energy_rate *
                    (1.0_rt -
                     theta_implicit_mhd::floor_outflow_limiter(
                         rho_old(i, j, k) - halo_pedestal,
                         0.125_rt * halo_pedestal)) *
                    (internal_energy - halo_pedestal_ion_internal) *
                    theta_implicit_mhd::floor_outflow_limiter(
                        internal_energy, halo_pedestal_ion_internal);
                ion_energy_increment(i, j, k) =
                    theta_dt *
                    (plasma_weight *
                         (-divergence_ion_energy_flux + lorentz_work +
                          ion_pressure_work) -
                     drag_kinetic_drain - energy_relax_drain);
                if (ei_equilibration) {
                    // Electron-ion equilibration counterpart (see the
                    // electron row above): the identical product with
                    // the opposite sign -- exact pair conservation.
                    ion_energy_increment(i, j, k) -=
                        theta_dt * plasma_weight * equilibration_heating;
                }
            }

            // Floor-consistency supply (see the host constants above and
            // floor_consistency_deficit for the one-sidedness /
            // JFNK-exactness / exact-zero-above-bound / no-overshoot
            // guarantees): appended AFTER the base increments under a
            // per-solve-uniform branch, so the OFF path performs no
            // arithmetic at all -- bit-identical including the sign of
            // zero. Booked per step by
            // AccumulateFloorConsistencySupplyLedger.
            if (floor_consistency) {
                if (evolve_ion_fluid) {
                    rho_increment(i, j, k) +=
                        theta_dt * floor_supply_rate *
                        theta_implicit_mhd::floor_consistency_deficit(
                            rho(i, j, k), rho_old(i, j, k),
                            rho_old(i, j, k), fc_mass_floor,
                            fc_mass_coefficient, theta, fc_width);
                }
                energy_increment(i, j, k) +=
                    theta_dt * floor_supply_rate *
                    theta_implicit_mhd::floor_consistency_deficit(
                        energy(i, j, k), energy_old(i, j, k),
                        rho_old(i, j, k), fc_electron_floor,
                        fc_electron_coefficient, theta, fc_width);
                if (total_energy_closure) {
                    ion_energy_increment(i, j, k) +=
                        theta_dt * floor_supply_rate *
                        theta_implicit_mhd::floor_consistency_deficit(
                            ion_e(i, j, k), ion_e_old(i, j, k),
                            rho_old(i, j, k), fc_ion_floor,
                            fc_ion_coefficient, theta, fc_width);
                }
            }
        });
    }
}

void ThetaImplicitMHD::ComputeFaceFluxes (const amrex::Real a_time)
{
#if defined(WARPX_DIM_1D_Z)
    ComputeDirectionalFaceFluxes(*m_WarpX->m_fields.get(FaceFluxZName, 0), 2,
                                 a_time);
#elif defined(WARPX_DIM_RZ)
    ComputeDirectionalFaceFluxes(*m_WarpX->m_fields.get(FaceFluxRName, 0), 0,
                                 a_time);
    ComputeDirectionalFaceFluxes(*m_WarpX->m_fields.get(FaceFluxZName, 0), 2,
                                 a_time);
#else
    amrex::ignore_unused(a_time);
    WARPX_ABORT_WITH_MESSAGE(
        "ThetaImplicitMHD::ComputeFaceFluxes() requires 1D or RZ geometry");
#endif
}

void ThetaImplicitMHD::ComputeDirectionalFaceFluxes (
    amrex::MultiFab& face_flux_mf, const int normal_direction,
    const amrex::Real a_time)
{
#if defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_RZ)
    using ablastr::fields::Direction;
    const amrex::MultiFab& density = *m_WarpX->m_fields.get(MassDensityName, 0);
    const amrex::MultiFab& momentum =
        *m_WarpX->m_fields.get(MomentumDensityName, 0);
    const amrex::MultiFab& electron_energy =
        *m_WarpX->m_fields.get(ElectronEnergyName, 0);
    const amrex::MultiFab& ion_energy =
        *m_WarpX->m_fields.get(IonEnergyName, 0);
    const amrex::MultiFab& old_density =
        *m_WarpX->m_fields.get(OldMassDensityName, 0);
    const amrex::MultiFab& old_momentum =
        *m_WarpX->m_fields.get(OldMomentumDensityName, 0);
    const amrex::MultiFab& old_electron_energy =
        *m_WarpX->m_fields.get(OldElectronEnergyName, 0);
    const amrex::MultiFab& old_ion_energy =
        *m_WarpX->m_fields.get(OldIonEnergyName, 0);
    const amrex::MultiFab& ion_parallel_energy =
        *m_WarpX->m_fields.get(IonParallelEnergyName, 0);
    const amrex::MultiFab& ion_perp_energy =
        *m_WarpX->m_fields.get(IonPerpEnergyName, 0);
    const amrex::MultiFab& old_ion_parallel_energy =
        *m_WarpX->m_fields.get(OldIonParallelEnergyName, 0);
    const amrex::MultiFab& old_ion_perp_energy =
        *m_WarpX->m_fields.get(OldIonPerpEnergyName, 0);
    const amrex::MultiFab& ion_internal_energy =
        *m_WarpX->m_fields.get(IonInternalEnergyName, 0);
    const amrex::MultiFab& old_ion_internal_energy =
        *m_WarpX->m_fields.get(OldIonInternalEnergyName, 0);
    const amrex::MultiFab& current_cc =
        *m_WarpX->m_fields.get(TotalCurrentCCName, 0);
    const amrex::MultiFab& magnetic_cc =
        *m_WarpX->m_fields.get(MagneticFieldCCName, 0);
    // Single-valued normal field on the face: the normal component of the
    // Yee-staggered B lives exactly on its faces (Bz on z-faces in 1D; Br
    // on r-faces and Bz on z-faces in RZ), so the native staggered value
    // (plasma + external under the split-field scheme) is the face value
    // the Riemann fan needs.
    const amrex::MultiFab& magnetic_face = *m_WarpX->m_fields.get(
        FieldType::Bfield_fp, Direction{normal_direction}, 0);
    const amrex::MultiFab* const external_face =
        m_hybrid_pic_model->m_add_external_fields
            ? m_WarpX->m_fields.get(FieldType::hybrid_B_fp_external,
                                    Direction{normal_direction}, 0)
            : nullptr;

    const theta_implicit_mhd::FluxParameters parameters = MakeFluxParameters();
    const bool add_external = (external_face != nullptr);
    const int normal = normal_direction;
    const bool use_central = m_use_central;
    // Explicit ion viscosity of the recast face fluxes: the normal-
    // gradient stress uses the plain cell spacing along the face normal
    // (the cylindrical metric enters only through the r-weighted
    // divergence that consumes these registers, exactly as for the
    // advective channels).
    const amrex::Real viscosity = m_viscosity;
    const bool add_viscosity = (viscosity > 0.0_rt);
    // Wall-row viscosity mask (implicit_mhd.wall_viscosity_mask; the
    // The reference code's 'skin'/'bndy' slip rows of step.f90 disip): a non-null
    // table zeroes the viscous face coefficient -- the momentum stress
    // AND its heating work, which share this one assembly -- at every
    // face either of whose adjacent cells lies within
    // wall_viscosity_mask_width cells (Chebyshev distance over the
    // stair-step contour) of the masked region. Static geometry, so the
    // branches are constants for the JFNK probes.
    const int* const AMREX_RESTRICT wall_viscosity_first_masked =
        (add_viscosity && m_wall_viscosity_mask && m_wall_mask.IsActive())
            ? m_wall_mask.FirstMaskedCellCentered()
            : nullptr;
    const int wall_viscosity_width = m_wall_viscosity_mask_width;
    // Thermal conduction shares the viscous flux's spacing convention.
    const amrex::Real chi_ion = m_thermal_diffusivity_ion;
    const amrex::Real chi_electron = m_thermal_diffusivity_electron;
    // Parser diffusivities chi(rho, Te, Ti, J, t) and the free-streaming
    // limiter share a donor-averaged face state (see the header): the
    // Ohm-floored face charge density, the temperature-primary face
    // Te/Ti ratios from the SAME CellState pressures the physical fluxes
    // use, and the face-averaged cell-centered total current. The
    // constant, limiter-off path is bit-identical to the legacy code.
    const bool chi_ion_is_parser = m_chi_ion_is_parser;
    const bool chi_electron_is_parser = m_chi_electron_is_parser;
    const auto chi_ion_parser = m_chi_ion;
    const auto chi_electron_parser = m_chi_electron;
    const bool chi_any_parser = chi_ion_is_parser || chi_electron_is_parser;
    const amrex::Real conduction_limit = m_conduction_flux_limit_factor;
    const bool braginskii = m_conduction_braginskii;
    const bool chi_needs_state =
        chi_any_parser || conduction_limit > 0.0_rt || braginskii;
    const amrex::Real chi_charge_to_mass = m_ion_charge_to_mass;
    const amrex::Real chi_charge_floor =
        m_ion_charge_to_mass * OhmMassDensityFloor();
    const amrex::Real conduction_ion_mass =
        PhysConst::q_e / m_ion_charge_to_mass;
    // Ion conduction channel: alive under total_energy AND dual_energy;
    // under dual_energy the pressure/temperature inputs are the blended
    // values (the live theta values arrive blended through CellState; the
    // frozen/old recoveries below carry an explicit blend twin).
    const bool chi_dual_energy = (m_ion_closure == "dual_energy");
    const bool chi_total_energy =
        (m_ion_closure == "total_energy") || chi_dual_energy;
    const amrex::Real face_time = a_time;
    const bool chi_coeff_old = m_conduction_coefficient_step_old;
    // Conduction-stage centering (implicit_mhd.conduction_theta): every
    // ENERGY argument whose difference/gradient drives a conductive flux
    // is the exact extrapolation
    //     e^{n+theta_c} = (theta_c/theta) e^{n+theta}
    //                     + (1 - theta_c/theta) e^n,
    // the conduction twin of the resistive-stage current (see
    // PrepareResistiveStageCurrents). conduction_stage = false keeps the
    // theta path bit-identical (no stage arithmetic is performed).
    const bool conduction_stage = (m_conduction_theta != m_theta);
    const amrex::Real stage_new_weight = m_conduction_theta / m_theta;
    const amrex::Real stage_old_weight = 1.0_rt - stage_new_weight;
    const bool add_conduction =
        braginskii ||
        (chi_ion > 0.0_rt || chi_electron > 0.0_rt || chi_any_parser);
    // Braginskii (1965) anisotropic conduction closure
    // (implicit_mhd.thermal_conduction_model = braginskii): per channel
    //   q_n = -rho_f [chi_perp de/dn
    //                 + (chi_par - chi_perp) bhat_n (bhat . grad e)],
    // with e the SAME specific internal energies the isotropic path
    // diffuses. chi_par = 3.16 kB Te tau_e / m_e (electron) and
    // 3.9 kB Ti tau_i / m_i (ion), with the Braginskii Z = 1 collision
    // times (SI)
    //   tau_e = 6 sqrt(2) pi^{3/2} eps0^2 sqrt(m_e) (kB Te)^{3/2}
    //           / (n e^4 lnLambda),
    //   tau_i = 12 pi^{3/2} eps0^2 sqrt(m_i) (kB Ti)^{3/2}
    //           / (n e^4 lnLambda),
    // evaluated from the donor-averaged face state (the same n, Te, Ti
    // the parser diffusivities see). chi_perp/chi_par follows the
    // Braginskii magnetization fits in x = (Omega tau)^2 (see the fit
    // ratios below), exactly 1 at x = 0 so the unmagnetized limit
    // reproduces the isotropic flux at chi = chi_par. Everything stays
    // C-infinity in the state: x is polynomial in B, the collision
    // times are smooth in the strictly floored face pressures, and
    // bhat bhat is regularized by the smooth |b|^2 floor at the
    // field-energy scale mu0 p_floor (the CGL small_b2 idiom) -- below
    // it the anisotropy is physically negligible and only the
    // (arbitrary) direction is smoothed away.
    const amrex::Real brag_tau_shared =
        std::pow(MathConst::pi, amrex::Real(1.5)) * PhysConst::epsilon_0 *
        PhysConst::epsilon_0 /
        (std::pow(PhysConst::q_e, 4) * m_conduction_coulomb_log);
    const amrex::Real brag_tau_e_coefficient =
        6.0_rt * std::sqrt(2.0_rt) * std::sqrt(PhysConst::m_e) *
        brag_tau_shared;
    const amrex::Real brag_tau_i_coefficient =
        12.0_rt * std::sqrt(conduction_ion_mass) * brag_tau_shared;
    const amrex::Real brag_omega_e_coefficient =
        PhysConst::q_e / PhysConst::m_e;
    const amrex::Real brag_omega_i_coefficient = m_ion_charge_to_mass;
    // Braginskii Z = 1 perpendicular fits, normalized to chi_par so the
    // x -> 0 limit is exactly 1: with x = (Omega tau)^2,
    //   electron: (gamma_1' x + gamma_0') / (x^2 + delta_1 x + delta_0)
    //             over gamma_0'/delta_0, coefficients gamma_1' = 4.664,
    //             gamma_0' = 11.92, delta_1 = 14.79, delta_0 = 3.7703;
    //   ion:      (2 x + 2.645) / (x^2 + 2.70 x + 0.677)
    //             over 2.645/0.677.
    const amrex::Real brag_e_numerator_1 = 4.664_rt / 11.92_rt;
    const amrex::Real brag_e_denominator_2 = 1.0_rt / 3.7703_rt;
    const amrex::Real brag_e_denominator_1 = 14.79_rt / 3.7703_rt;
    const amrex::Real brag_i_numerator_1 = 2.0_rt / 2.645_rt;
    const amrex::Real brag_i_denominator_2 = 1.0_rt / 0.677_rt;
    const amrex::Real brag_i_denominator_1 = 2.70_rt / 0.677_rt;
    const amrex::Real brag_small_b2 =
        PhysConst::mu0 *
        (m_electron_pressure_floor + m_ion_pressure_floor);
    const amrex::Real brag_chi_min = m_conduction_chi_min;
    const amrex::Real brag_chi_max = m_conduction_chi_max;
    // Quasi-shorting cross-field boost (implicit_mhd.conduction_qs_*):
    // an ADDITIVE Braginskii chi_perp keyed to the pseudo-entropy excess
    // above the load envelope (see the header and the qs_boost lambda in
    // the kernel). T0 is stored in eV; the face temperatures are kelvin.
    const bool add_qs =
        m_conduction_braginskii && m_conduction_qs_chi > 0.0_rt;
    const amrex::Real qs_chi = m_conduction_qs_chi;
    const amrex::Real qs_onset = m_conduction_qs_onset;
    const amrex::Real qs_width = 0.3_rt * (m_conduction_qs_onset - 1.0_rt);
    const amrex::Real qs_inverse_t0 =
        (m_conduction_qs_reference_temperature > 0.0_rt)
            ? PhysConst::kb / (m_conduction_qs_reference_temperature *
                               PhysConst::q_e)
            : 0.0_rt;
    const amrex::Real qs_envelope_density = m_reference_mass_density;
    // Smooth density guard of the entropy argument, at the same Ohm
    // density scale the Braginskii coefficient inputs are floored at.
    const amrex::Real qs_density_guard = OhmMassDensityFloor();
    // Density-keyed halo boost of the Braginskii chi_perp
    // (implicit_mhd.conduction_halo_boost; the reference code's low-density
    // perp-chi boost, ntb.f90 t_cond ~584): chi_perp is composed with
    // chi_halo = D_boost (rho_ref/rho)^2 through the
    // vacuum_keyed_resistivity quadrature smooth max, keyed to the same
    // per-step frozen reference as the field-eta vacuum boost and with
    // the density division guarded at the positivity floor. The boost
    // input is the coefficient-state face charge density (raw, not
    // Ohm-floored: the boost must grow below the guard), so it follows
    // conduction_coefficient_state exactly like the base chi.
    const bool add_halo_boost =
        m_conduction_braginskii && m_conduction_halo_boost > 0.0_rt;
    const amrex::Real halo_boost_chi = m_conduction_halo_boost;
    const amrex::Real halo_boost_reference =
        m_ion_charge_to_mass * VacuumReferenceMassDensity();
    const amrex::Real halo_boost_guard =
        m_ion_charge_to_mass * m_mass_density_floor;
    // Stair-step wall thermal boundary (implicit_mhd.wall_thermal_bc):
    // the conducting-wall mask is electromagnetic only, so without this
    // the conduction operator exchanges blindly across the stair-step
    // interface against conductor cells riding the floor. zero_flux
    // insulates the interface faces exactly; temperature exchanges
    // against a T_wall reservoir by presenting the wall specific
    // internal energies on the masked side (the EB analog of the
    // z_boundary_fluid = wall_temperature ghost fill). Faces between
    // two masked cells are interior metal and carry no conduction in
    // either mode. The mask is static geometry (never a function of the
    // state), so the branches are C-infinity for the JFNK probes.
    // Inactive (nullptr table) outside RZ and under wall_thermal_bc =
    // none, where this block compiles to dead constants.
    const auto wall_thermal_mode = m_wall_mask.GetThermalBC();
    // wall_mechanics: the rigid-conductor contract is active -- interface
    // faces present the ABSORB IMAGE of the interior state (see the
    // kernel), independent of whether conduction is on. wall_thermal
    // additionally routes the interface conduction (skip/drain).
    const bool wall_mechanics =
        (wall_thermal_mode != ImplicitMHDWallMask::ThermalBC::none);
    const bool wall_thermal = add_conduction && wall_mechanics;
    const bool wall_dirichlet =
        (wall_thermal_mode == ImplicitMHDWallMask::ThermalBC::temperature);
    const int* const AMREX_RESTRICT wall_first_masked_cc =
        wall_mechanics ? m_wall_mask.FirstMaskedCellCentered() : nullptr;
    // The table covers z cells [-ng, nz - 1 + ng] of the MASK's ghost
    // width; kernel boxes may reach one cell past it at the z ends, so
    // reads are clamped (constant continuation, like the polyline).
    const int wall_mask_z_lo = -m_wall_mask.GhostCells();
    const int wall_mask_z_hi =
        m_wall_mask.AxialCells() - 1 + m_wall_mask.GhostCells();
    // n kB T_wall = rho (q/m) T_wall[eV] for the quasi-neutral
    // single-ion fluid (see the z wall_temperature ghost fill): the
    // wall-reservoir SPECIFIC internal energies are density-free.
    const amrex::Real wall_e_spec_electron =
        m_ion_charge_to_mass * m_wall_mask.WallTemperature_eV() /
        (m_gamma_e - 1.0_rt);
    const amrex::Real wall_e_spec_ion =
        m_ion_charge_to_mass * m_wall_mask.WallTemperature_eV() /
        (m_gamma_i - 1.0_rt);
    const amrex::Real wall_te_kelvin = m_wall_mask.WallTemperature_eV() *
                                       PhysConst::q_e / PhysConst::kb;
    // The interface exchange is ONE-SIDED (a drain toward T_wall, C^1
    // gated on the interior specific energy against the wall value) and
    // ALWAYS free-streaming limited (factor = the global
    // conduction_flux_limit_factor when set, else 1). A two-sided,
    // unlimited Dirichlet exchange was measured fatal on the formation
    // ladder (rr12_S0w, frozen at the 13.3 us contact epoch): the
    // reservoir HEATED the entire near-floor dust rim of the machine
    // toward T_wall on the sub-dt cell-diffusion time (median ring Te
    // 2.5 -> 47 eV, a moving hot boundary layer on every wall face),
    // while the un-capped drain ran at ~18x the electron free-streaming
    // flux at the 6 keV contact hot spots -- both are exactly the
    // conduction-type Newton hostility the TC arms die of, switched on
    // at every wall face at once.
    const amrex::Real wall_conduction_limit =
        (m_conduction_flux_limit_factor > 0.0_rt)
            ? m_conduction_flux_limit_factor
            : 1.0_rt;
#if defined(WARPX_DIM_1D_Z)
    const amrex::Real inverse_normal_size =
        1.0_rt / m_WarpX->Geom(0).CellSize(0);
#else
    const amrex::Real inverse_normal_size =
        1.0_rt / m_WarpX->Geom(0).CellSize(normal_direction == 0 ? 0 : 1);
#endif
#if defined(WARPX_DIM_RZ)
    // In-plane tangential spacing of the Braginskii corner stencil:
    // dr for z-faces, dz for r-faces (the theta gradient vanishes for
    // m = 0). 1D has no in-plane tangent.
    const amrex::Real inverse_tangential_size =
        1.0_rt / m_WarpX->Geom(0).CellSize(normal_direction == 0 ? 1 : 0);
    // The axis face has zero area: its fluid channels drop out of the
    // r-weighted divergence and the axis-corner EMF is set by parity, so
    // the kernel is skipped there (its left cell lies below the axis).
    const bool radial_faces = (normal_direction == 0);
    const amrex::Real radial_lower = m_WarpX->Geom(0).ProbLo(0);
    const amrex::Real radial_cell_size = m_WarpX->Geom(0).CellSize(0);
    // Zero-flux wall (open FIELD boundary + reflect FLUID ghosts): the
    // r_max domain face is a rigid no-flow wall for the fluid while the
    // field stays free-space. The fan there otherwise mixes inconsistent
    // ghosts -- mirror fluid states against the OPEN staggered B_n (the
    // Green's-function face value, nonzero where a PEC image would pin
    // B_n = 0) -- and the nonzero B_n opens the Alfven fan's tangential
    // Maxwell stress -B_n B_t/mu0 and energy channels at the wall,
    // a persistent unphysical E_i drain of the last cell ring that pins
    // it on the energy floor (the i = i_max projection storm of the FRC
    // open-wall hold). When reflect is selected, every ADVECTIVE fluid
    // channel of the wall face is therefore set EXACTLY to zero and the
    // tangential Maxwell stress takes its PEC-image value (zero) in both
    // the momentum flux and the work register; the normal channel keeps
    // the open fan (the free-space field acts through the face) and the
    // induction and signal channels keep the OPEN-field fan: the Ohm/
    // Faraday path retains the true open boundary condition.
    const int radial_wall_face = m_WarpX->Geom(0).Domain().bigEnd(0) + 1;
    const bool reflect_wall =
        radial_faces && m_r_open && (m_r_open_fluid == "reflect");
    // Absorbing wall (open field + absorb fluid): the wall-face fan sees
    // the impedance-matched (zero-gradient clamp) ghost image with a
    // RECTIFIED normal velocity (see ApplyFluidDomainBoundaries) -- the
    // same wall-plasma signal speeds as the outflow boundary, but the
    // ghost never advances toward the domain, so the face admits
    // incident plasma at its signal-limited rate and cannot drive a
    // refill. No flux-register surgery here: the fan and the donor
    // gates apply unchanged (the pedestal band still cannot drain), and
    // the reflect path's zero-flux override does NOT apply -- the
    // absorber is a physical sink by design, tallied by the
    // absorbed-mass/energy ledger (see AccumulateAbsorbedWallLedger).
#endif
    constexpr int flux_mass = FaceFluxComponent::mass;
    constexpr int flux_momentum = FaceFluxComponent::momentum_x;
    constexpr int flux_magnetic = FaceFluxComponent::magnetic_x;
    constexpr int flux_electron_energy = FaceFluxComponent::electron_energy;
    constexpr int flux_ion_energy = FaceFluxComponent::ion_energy;
    constexpr int flux_ion_parallel_energy =
        FaceFluxComponent::ion_parallel_energy;
    constexpr int flux_ion_perp_energy = FaceFluxComponent::ion_perp_energy;
    constexpr int flux_ion_internal_energy =
        FaceFluxComponent::ion_internal_energy;
    constexpr int flux_electron_velocity =
        FaceFluxComponent::electron_velocity;
    constexpr int flux_induction_t1 = FaceFluxComponent::induction_t1;
    constexpr int flux_induction_t2 = FaceFluxComponent::induction_t2;
    constexpr int flux_signal_left = FaceFluxComponent::signal_left;
    constexpr int flux_signal_right = FaceFluxComponent::signal_right;
    constexpr int flux_alfven_left = FaceFluxComponent::signal_alfven_left;
    constexpr int flux_alfven_right = FaceFluxComponent::signal_alfven_right;

    for (amrex::MFIter mfi(face_flux_mf); mfi.isValid(); ++mfi) {
        // Grow in the transverse direction(s): the corner UCT EMF reads
        // both adjacent faces of each family, including one ghost face at
        // grid boundaries; the kernel inputs are ghosted deeply enough.
        const amrex::Box box =
            amrex::grow(mfi.validbox(), face_flux_mf.nGrowVect());
        const auto rho = density.const_array(mfi);
        const auto mom = momentum.const_array(mfi);
        const auto energy = electron_energy.const_array(mfi);
        const auto ion_e = ion_energy.const_array(mfi);
        const auto rho_old = old_density.const_array(mfi);
        const auto mom_old = old_momentum.const_array(mfi);
        const auto energy_old = old_electron_energy.const_array(mfi);
        const auto ion_e_old = old_ion_energy.const_array(mfi);
        const auto upar = ion_parallel_energy.const_array(mfi);
        const auto uperp = ion_perp_energy.const_array(mfi);
        const auto upar_old = old_ion_parallel_energy.const_array(mfi);
        const auto uperp_old = old_ion_perp_energy.const_array(mfi);
        const auto ion_int = ion_internal_energy.const_array(mfi);
        const auto ion_int_old = old_ion_internal_energy.const_array(mfi);
        const auto j_cc = current_cc.const_array(mfi);
        const auto b_cc = magnetic_cc.const_array(mfi);
        const auto bn_staggered = magnetic_face.const_array(mfi);
        const auto bn_external =
            add_external ? external_face->const_array(mfi)
                         : amrex::Array4<const amrex::Real>{};
        const auto flux_arr = face_flux_mf.array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // The face at index n separates cells n-1 (left) and n (right)
            // along the normal direction.
            int il = i;
            int jl = j;
            int kl = k;
            shift_index(il, jl, kl, normal, -1);
#if defined(WARPX_DIM_RZ)
            if (radial_faces) {
                const amrex::Real face_radius =
                    radial_lower + i * radial_cell_size;
                if (face_radius == 0.0_rt) {
                    for (int component = 0;
                         component < FaceFluxComponent::count; ++component) {
                        flux_arr(i, j, k, component) = 0.0_rt;
                    }
                    return;
                }
            }
#endif
            const auto load_state = [=] (const int ic, const int jc,
                                         const int kc) {
                if (parameters.cgl_closure) {
                    return theta_implicit_mhd::load_cell_state_hlld_cgl(
                        rho, mom, energy, ion_e, upar, uperp, j_cc, b_cc,
                        ic, jc, kc, normal, parameters);
                }
                if (parameters.dual_energy_closure) {
                    // Blended-pressure loader: every pressure consumer of
                    // the fan (momentum flux, signal bounds, conduction
                    // temperatures) sees the fk blend; the E_i channel
                    // machinery keeps the total_energy assembly.
                    return theta_implicit_mhd::load_cell_state_hlld_dual(
                        rho, mom, energy, ion_e, ion_int, ion_int_old,
                        j_cc, b_cc, ic, jc, kc, normal, parameters);
                }
                return theta_implicit_mhd::load_cell_state_hlld(
                    rho, mom, energy, ion_e, j_cc, b_cc, ic, jc, kc,
                    normal, parameters);
            };
            auto left = load_state(il, jl, kl);
            auto right = load_state(i, j, k);

            // Wall face classification against the cell-centered mask
            // (see the host constants above the loop): exactly one
            // masked cell = stair-step interface face, both masked =
            // interior metal. The table's z index is clamped to its
            // stored range (constant continuation, like the polyline).
            // Always false when the thermal wall is off (nullptr table
            // never read); the mask is static geometry, so every branch
            // below is C-infinity in the state.
            bool wall_left_masked = false;
            bool wall_right_masked = false;
            if (wall_mechanics) {
                const int jzl = std::max(wall_mask_z_lo,
                                         std::min(wall_mask_z_hi, jl));
                const int jzr = std::max(wall_mask_z_lo,
                                         std::min(wall_mask_z_hi, j));
                wall_left_masked = (il >= wall_first_masked_cc[jzl]);
                wall_right_masked = (i >= wall_first_masked_cc[jzr]);
            }
            const bool wall_interface =
                (wall_left_masked != wall_right_masked);
            // Donor indices for the positivity gates: interface faces
            // gate BOTH sides on the interior donor (the masked side
            // presents the interior's absorb image below, so its frozen
            // band arrays are not this face's donor).
            int donor_il = il, donor_jl = jl, donor_kl = kl;
            int donor_ir = i, donor_jr = j, donor_kr = k;
            if (wall_interface) {
                if (wall_right_masked) {
                    donor_ir = il; donor_jr = jl; donor_kr = kl;
                } else {
                    donor_il = i; donor_jl = j; donor_kl = k;
                }
                // Absorbing DIELECTRIC image (the r_max absorbing-wall
                // idea applied per stair face, with a no-injection
                // amendment): the masked side presents the INTERIOR
                // state with its normal momentum replaced by the
                // INTO-WALL smooth absolute value m |m|/(|m|+...) --
                // C-infinity, exactly zero at stagnation. On approach
                // the image matches the interior (the stair admits
                // incident plasma at its signal-limited rate: a
                // frozen-dust image instead stagnates a supersonic
                // contact jet against a rigid corner, converting ram
                // into a keV-scale E_i pocket at the stair steps -- the
                // rr12 v2 S0w corpse, 116 keV one cell inside the cone
                // step corner at first wall contact). On RETREAT the
                // image MIRRORS the interior, so the face flux closes:
                // a dielectric machine wall supplies no plasma, ever --
                // the held-ghost exhaust recipe would inject at half
                // the retreat rate through the central average. E_i is
                // copied verbatim (the r_max absorber's ghost carries
                // the incident kinetic energy unadjusted); the thermal
                // reservoir acts ONLY through the conduction drain
                // below.
                auto& image = wall_right_masked ? right : left;
                const auto& interior_state =
                    wall_right_masked ? left : right;
                image = interior_state;
                const amrex::Real rectifier_width =
                    parameters.hlld_kappa_signal *
                    std::sqrt(
                        parameters.gamma_e *
                        (parameters.gamma_e - 1.0_rt) *
                        std::max(interior_state.electron_energy,
                                 parameters.electron_pressure_floor /
                                     (parameters.gamma_e - 1.0_rt)) *
                        interior_state.safe_density);
                const amrex::Real into_wall_sign =
                    wall_right_masked ? 1.0_rt : -1.0_rt;
                // m * smooth_sign(m, w) is the C-infinity |m| with an
                // exact zero at m = 0 (no spurious O(w) suction on
                // quiescent faces).
                const amrex::Real normal_momentum =
                    into_wall_sign * interior_state.momentum[normal] *
                    theta_implicit_mhd::smooth_sign(
                        interior_state.momentum[normal], rectifier_width);
                image.momentum[normal] = normal_momentum;
                image.ion_velocity[normal] =
                    normal_momentum / image.safe_density;
                image.electron_velocity_normal =
                    interior_state.electron_velocity_normal +
                    (image.ion_velocity[normal] -
                     interior_state.ion_velocity[normal]);
                image.wave_speed =
                    std::max(std::abs(image.ion_velocity[normal]) +
                                 image.sound_speed,
                             std::abs(image.electron_velocity_normal));
                image.fast_wave_speed =
                    std::abs(image.ion_velocity[normal]) +
                    image.fast_speed;
            }

            amrex::Real bn_face = bn_staggered(i, j, k);
            if (add_external) {
                bn_face += bn_external(i, j, k);
            }
            auto flux =
                use_central
                    ? theta_implicit_mhd::central_flux(left, right, bn_face,
                                                       normal, parameters)
                    : theta_implicit_mhd::hlld_flux(left, right, bn_face,
                                                    normal, parameters);

            // Donor-gated positivity guards on the advected mass and
            // energy channels, gating on the theta-extrapolated
            // end-of-step donor values -- identical policy to face_flux
            // (momentum, stress, and induction channels have no floors).
            // The donor SIDE is selected by a C-infinity smoothed flux
            // sign: near-stagnant faces whose two donors carry different
            // limiter values (a floored halo cell against a healthy
            // neighbor) would otherwise present a derivative kink at
            // every zero crossing, which defeats the Newton line search
            // on near-floor equilibria. The blend width scales with the
            // face signal span times the donor magnitudes, so an exactly
            // zero flux stays exactly zero (static contacts remain
            // machine-preserved) and a strong flux keeps its pure donor.
            const amrex::Real ext =
                (1.0_rt - parameters.theta) / parameters.theta;
            const amrex::Real signal_span =
                0.5_rt * (flux.signal_right - flux.signal_left);
            const auto donor_blend = [=] (const amrex::Real flux_value,
                                          const amrex::Real limiter_left,
                                          const amrex::Real limiter_right,
                                          const amrex::Real value_scale) {
                const amrex::Real width = parameters.hlld_kappa_signal *
                                          signal_span * value_scale;
                const amrex::Real left_weight =
                    0.5_rt * (1.0_rt + theta_implicit_mhd::smooth_sign(
                                           flux_value, width));
                return left_weight * limiter_left +
                       (1.0_rt - left_weight) * limiter_right;
            };
            const auto donor_end = [=] (const amrex::Array4<const amrex::Real>& now,
                                        const amrex::Array4<const amrex::Real>& old,
                                        const int id, const int jd, const int kd) {
                return now(id, jd, kd) * (1.0_rt + ext) - old(id, jd, kd) * ext;
            };
            // The mass gate anchors at the halo pedestal when active
            // (see FluxParameters::halo_pedestal): outflow from a donor
            // closes smoothly at rho_ped, holding the pedestal band as a
            // dynamically invariant set while the Newton admissibility
            // bound stays at the far-lower positivity floor -- no cell
            // ever operates on a bound.
            const amrex::Real mass_gate_floor = std::max(
                parameters.density_floor, parameters.halo_pedestal);
            flux.mass *= donor_blend(
                flux.mass,
                theta_implicit_mhd::floor_outflow_limiter(
                    donor_end(rho, rho_old, donor_il, donor_jl, donor_kl),
                    mass_gate_floor),
                theta_implicit_mhd::floor_outflow_limiter(
                    donor_end(rho, rho_old, donor_ir, donor_jr, donor_kr),
                    mass_gate_floor),
                0.5_rt * (left.safe_density + right.safe_density));
            // The energy gates anchor at their pedestal values when the
            // pedestal is active (see FluxParameters::halo_pedestal*):
            // the pedestal is an f-scaled image of the peak STATE, so
            // every block's band is held off its bound the same way the
            // mass band is.
            const amrex::Real electron_energy_floor = std::max(
                parameters.electron_pressure_floor /
                    (parameters.gamma_e - 1.0_rt),
                parameters.halo_pedestal_electron_energy);
            flux.electron_energy *= donor_blend(
                flux.electron_energy,
                theta_implicit_mhd::floor_outflow_limiter(
                    donor_end(energy, energy_old, donor_il, donor_jl,
                              donor_kl),
                    electron_energy_floor),
                theta_implicit_mhd::floor_outflow_limiter(
                    donor_end(energy, energy_old, donor_ir, donor_jr,
                              donor_kr),
                    electron_energy_floor),
                0.5_rt * (left.electron_energy + right.electron_energy) +
                    electron_energy_floor);
            if (parameters.total_energy_closure) {
                const auto ion_internal_end = [=] (const int id, const int jd,
                                                   const int kd) {
                    const amrex::Real ei_end =
                        donor_end(ion_e, ion_e_old, id, jd, kd);
                    amrex::Real kinetic_end = 0.0_rt;
                    for (int component = 0; component < 3; ++component) {
                        const amrex::Real mom_end =
                            mom(id, jd, kd, component) * (1.0_rt + ext) -
                            mom_old(id, jd, kd, component) * ext;
                        kinetic_end += mom_end * mom_end;
                    }
                    kinetic_end *=
                        0.5_rt / std::max(donor_end(rho, rho_old, id, jd, kd),
                                          parameters.density_floor);
                    return ei_end - kinetic_end;
                };
                const amrex::Real ion_energy_floor = std::max(
                    parameters.ion_pressure_floor /
                        (parameters.gamma_i - 1.0_rt),
                    parameters.halo_pedestal_ion_internal);
                flux.ion_energy *= donor_blend(
                    flux.ion_energy,
                    theta_implicit_mhd::floor_outflow_limiter(
                        ion_internal_end(donor_il, donor_jl, donor_kl),
                        ion_energy_floor),
                    theta_implicit_mhd::floor_outflow_limiter(
                        ion_internal_end(donor_ir, donor_jr, donor_kr),
                        ion_energy_floor),
                    0.5_rt * (left.ion_energy + right.ion_energy) +
                        ion_energy_floor);
            }
            if (parameters.cgl_closure) {
                // CGL internal-energy channels: same donor-gated guards,
                // gating on the U fields directly (U_par and U_perp are
                // pure internal energies -- no kinetic subtraction), with
                // U_par floored at p_i_floor/2 and U_perp at p_i_floor.
                const amrex::Real parallel_floor = std::max(
                    0.5_rt * parameters.ion_pressure_floor,
                    parameters.halo_pedestal_ion_parallel);
                flux.ion_parallel_energy *= donor_blend(
                    flux.ion_parallel_energy,
                    theta_implicit_mhd::floor_outflow_limiter(
                        donor_end(upar, upar_old, donor_il, donor_jl,
                                  donor_kl),
                        parallel_floor),
                    theta_implicit_mhd::floor_outflow_limiter(
                        donor_end(upar, upar_old, donor_ir, donor_jr,
                                  donor_kr),
                        parallel_floor),
                    0.5_rt * (left.ion_parallel_energy +
                              right.ion_parallel_energy) +
                        parallel_floor);
                const amrex::Real perp_floor =
                    std::max(parameters.ion_pressure_floor,
                             parameters.halo_pedestal_ion_perp);
                flux.ion_perp_energy *= donor_blend(
                    flux.ion_perp_energy,
                    theta_implicit_mhd::floor_outflow_limiter(
                        donor_end(uperp, uperp_old, donor_il, donor_jl,
                                  donor_kl),
                        perp_floor),
                    theta_implicit_mhd::floor_outflow_limiter(
                        donor_end(uperp, uperp_old, donor_ir, donor_jr,
                                  donor_kr),
                        perp_floor),
                    0.5_rt *
                            (left.ion_perp_energy + right.ion_perp_energy) +
                        perp_floor);
            }
            if (parameters.dual_energy_closure) {
                // Dual-energy auxiliary internal channel: same donor-gated
                // guard, gating on U_i directly (a pure internal energy --
                // no kinetic subtraction), floored at the internal-energy
                // image of the ion pressure floor and anchored at the SAME
                // internal pedestal image as the E_i gate.
                const amrex::Real internal_gate_floor = std::max(
                    parameters.ion_pressure_floor /
                        (parameters.gamma_i - 1.0_rt),
                    parameters.halo_pedestal_ion_internal);
                flux.ion_internal_energy *= donor_blend(
                    flux.ion_internal_energy,
                    theta_implicit_mhd::floor_outflow_limiter(
                        donor_end(ion_int, ion_int_old, donor_il, donor_jl,
                                  donor_kl),
                        internal_gate_floor),
                    theta_implicit_mhd::floor_outflow_limiter(
                        donor_end(ion_int, ion_int_old, donor_ir, donor_jr,
                                  donor_kr),
                        internal_gate_floor),
                    0.5_rt * (left.ion_internal_energy +
                              right.ion_internal_energy) +
                        internal_gate_floor);
            }

            // Explicit ion viscosity (implicit_mhd.viscosity): the
            // normal-gradient viscous stress on the SAME face registers,
            // evaluated at the theta-stage states like every other flux
            // term. The momentum stress and its velocity-weighted work
            // are added as an exactly conservative pair AFTER the donor
            // gates (which only scale the advective channels): gating one
            // member of the pair without the other converts stress work
            // into a spurious energy source/sink. The reflect wall face
            // passes no viscous flux either -- the override below zeroes
            // the tangential pair, and the normal member must not survive
            // alone.
            // Wall-row viscosity mask (see the host constants): the
            // The reference code's slip rows -- a face is a slip face when either
            // adjacent cell sits within wall_viscosity_mask_width cells
            // (Chebyshev distance over the stair-step tables) of the
            // masked contour; the zeroed coefficient removes the
            // momentum stress AND its heating work together (the
            // conservative pair must never split).
            bool viscosity_slip_face = false;
            if (wall_viscosity_first_masked != nullptr) {
                const auto near_wall = [&] (const int ic, const int jc) {
                    for (int dj = -wall_viscosity_width;
                         dj <= wall_viscosity_width; ++dj) {
                        const int jz = std::max(
                            wall_mask_z_lo,
                            std::min(wall_mask_z_hi, jc + dj));
                        if (ic >= wall_viscosity_first_masked[jz] -
                                       wall_viscosity_width) {
                            return true;
                        }
                    }
                    return false;
                };
                viscosity_slip_face = near_wall(il, jl) || near_wall(i, j);
            }
            if (add_viscosity && !viscosity_slip_face
#if defined(WARPX_DIM_RZ)
                && !(reflect_wall && i == radial_wall_face)
#endif
            ) {
                const amrex::Real face_density =
                    0.5_rt * (left.density + right.density);
                amrex::Real viscous_work = 0.0_rt;
                for (int component = 0; component < 3; ++component) {
                    const amrex::Real viscous_stress =
                        -face_density * viscosity *
                        (right.ion_velocity[component] -
                         left.ion_velocity[component]) *
                        inverse_normal_size;
                    flux.momentum[component] += viscous_stress;
                    viscous_work += 0.5_rt *
                                    (left.ion_velocity[component] +
                                     right.ion_velocity[component]) *
                                    viscous_stress;
                }
                flux.ion_energy += viscous_work;
            }

            // Thermal conduction (implicit_mhd.thermal_diffusivity_* or
            // thermal_conduction_model = braginskii, which replaces the
            // scalar flux with the anisotropic tensor form -- see the
            // braginskii host constants above):
            // the conductive internal-energy flux -chi rho_f d(e_spec)/dn
            // (q = -kappa grad T with kappa = chi rho c_v) on the SAME
            // face registers, placed exactly like the viscous terms --
            // after the donor gates (which only scale the advective
            // channels) and inside the zero-flux wall mask. Pure energy
            // diffusion: unlike the viscous stress there is no momentum
            // twin to pair, so no channel-splitting hazard exists. The
            // specific internal energies come from the SAME CellState
            // pressures the physical fluxes use (the smooth-floored
            // recovered p_i(E_i) under total_energy: ion_internal is
            // exactly p_i/(gamma_i - 1)).
            // Interior-metal faces never conduct; interface faces
            // conduct only in the temperature (Dirichlet) mode (the
            // masked flags were classified above, before the absorb
            // image). wall_thermal is false when conduction is off, so
            // the skip only ever engages alongside an active channel.
            const bool wall_skip_conduction =
                wall_thermal &&
                ((wall_left_masked && wall_right_masked) ||
                 (wall_interface && !wall_dirichlet));

            if (add_conduction && !wall_skip_conduction
#if defined(WARPX_DIM_RZ)
                && !(reflect_wall && i == radial_wall_face)
#endif
            ) {
                // Frozen-coefficient option (implicit_mhd.
                // conduction_coefficient_state = step_old): the chi
                // COEFFICIENT inputs -- the rho_f multiplier, the face
                // charge density, and the face temperatures feeding the
                // parser diffusivities, the Braginskii coefficients,
                // and the free-streaming caps -- come from the STEP-OLD
                // fields, per-solve constants like the rho_old-keyed
                // source masks: Newton then sees LINEAR diffusion in
                // the energies. Probe-measured motivation: every
                // Newton-hostile conduction incident on the formation
                // ladder is a state-dependent chi inside the residual
                // (30 vs 322 steps/min in the isolation probes);
                // constant coefficients are cheap at any amplitude.
                // The FLUX keeps the live theta-state specific-energy
                // gradient; the parser J input stays live (theta j_cc,
                // no old-current register). Hard floors are fine here:
                // frozen inputs are constants w.r.t. the Newton state,
                // so residual smoothness is unaffected. Default theta
                // is bit-identical.
                amrex::Real coeff_density_left = left.density;
                amrex::Real coeff_density_right = right.density;
                amrex::Real coeff_pe_left = left.electron_pressure;
                amrex::Real coeff_pe_right = right.electron_pressure;
                amrex::Real coeff_pi_left = left.ion_pressure;
                amrex::Real coeff_pi_right = right.ion_pressure;
                if (chi_coeff_old) {
                    coeff_density_left = rho_old(il, jl, kl);
                    coeff_density_right = rho_old(i, j, k);
                    coeff_pe_left = std::max(
                        (parameters.gamma_e - 1.0_rt) *
                            energy_old(il, jl, kl),
                        parameters.electron_pressure_floor);
                    coeff_pe_right = std::max(
                        (parameters.gamma_e - 1.0_rt) *
                            energy_old(i, j, k),
                        parameters.electron_pressure_floor);
                    if (chi_total_energy) {
                        amrex::Real ke_left = 0.0_rt;
                        amrex::Real ke_right = 0.0_rt;
                        for (int component = 0; component < 3;
                             ++component) {
                            ke_left += mom_old(il, jl, kl, component) *
                                       mom_old(il, jl, kl, component);
                            ke_right += mom_old(i, j, k, component) *
                                        mom_old(i, j, k, component);
                        }
                        ke_left *= 0.5_rt /
                                   std::max(rho_old(il, jl, kl),
                                            parameters.density_floor);
                        ke_right *= 0.5_rt /
                                    std::max(rho_old(i, j, k),
                                             parameters.density_floor);
                        if (chi_dual_energy) {
                            // Frozen coefficients take the SAME blend the
                            // live path sees, evaluated at the step-old
                            // state (a per-solve constant, so smoothness
                            // is moot but consistency is not).
                            coeff_pi_left =
                                theta_implicit_mhd::
                                    dual_energy_blended_pressure(
                                        ion_e_old(il, jl, kl), ke_left,
                                        ion_int_old(il, jl, kl),
                                        ion_int_old(il, jl, kl),
                                        parameters);
                            coeff_pi_right =
                                theta_implicit_mhd::
                                    dual_energy_blended_pressure(
                                        ion_e_old(i, j, k), ke_right,
                                        ion_int_old(i, j, k),
                                        ion_int_old(i, j, k), parameters);
                        } else {
                            coeff_pi_left = std::max(
                                (parameters.gamma_i - 1.0_rt) *
                                    (ion_e_old(il, jl, kl) - ke_left),
                                parameters.ion_pressure_floor);
                            coeff_pi_right = std::max(
                                (parameters.gamma_i - 1.0_rt) *
                                    (ion_e_old(i, j, k) - ke_right),
                                parameters.ion_pressure_floor);
                        }
                    }
                }
                // Inside this block a masked side implies a Dirichlet
                // interface face (interior metal and zero_flux faces
                // were skipped above): the masked side presents the
                // wall reservoir, and the face density is the INTERIOR
                // side's -- chi rho is the exchange conductivity and
                // the conductor's near-floor fill must not choke it.
                const amrex::Real face_density =
                    wall_left_masked
                        ? coeff_density_right
                        : (wall_right_masked
                               ? coeff_density_left
                               : 0.5_rt * (coeff_density_left +
                                           coeff_density_right));
                // Donor-averaged face state for the parser diffusivities
                // and the free-streaming limiter (unused, and skipped, on
                // the constant/limiter-off path). Temperatures are the
                // temperature-primary face ratios p_face/(n_f kB); Ti is
                // 0 outside total_energy (where the ion channel is
                // disallowed anyway). Dirichlet interface faces average
                // the interior side against the T_wall reservoir.
                // Interface faces need the face state even when the
                // parser/limiter path is globally off: the wall drain is
                // always free-streaming limited.
                const bool wall_face =
                    wall_left_masked || wall_right_masked;
                amrex::Real face_charge_density = 0.0_rt;
                amrex::Real face_te = 0.0_rt;
                amrex::Real face_ti = 0.0_rt;
                amrex::Real face_jmag = 0.0_rt;
                if (chi_needs_state || wall_face) {
                    face_charge_density =
                        std::max(chi_charge_to_mass * face_density,
                                 chi_charge_floor);
                    const amrex::Real inverse_nkb =
                        PhysConst::q_e /
                        (face_charge_density * PhysConst::kb);
                    if (wall_face) {
                        const amrex::Real interior_pe =
                            wall_left_masked ? coeff_pe_right
                                             : coeff_pe_left;
                        face_te = 0.5_rt * (interior_pe * inverse_nkb +
                                            wall_te_kelvin);
                        if (chi_total_energy) {
                            const amrex::Real interior_pi =
                                wall_left_masked ? coeff_pi_right
                                                 : coeff_pi_left;
                            face_ti = 0.5_rt * (interior_pi * inverse_nkb +
                                                wall_te_kelvin);
                        }
                    } else {
                        face_te = 0.5_rt *
                                  (coeff_pe_left + coeff_pe_right) *
                                  inverse_nkb;
                        if (chi_total_energy) {
                            face_ti =
                                0.5_rt *
                                (coeff_pi_left + coeff_pi_right) *
                                inverse_nkb;
                        }
                    }
                    if (chi_any_parser) {
                        amrex::Real jsq = 0.0_rt;
                        for (int component = 0; component < 3; ++component) {
                            const amrex::Real face_current =
                                0.5_rt * (j_cc(il, jl, kl, component) +
                                          j_cc(i, j, k, component));
                            jsq += face_current * face_current;
                        }
                        face_jmag = std::sqrt(jsq);
                    }
                }
                // Conduction-stage energy arguments (implicit_mhd.
                // conduction_theta != implicit_evolve.theta): the
                // per-side specific internal energies that drive every
                // conductive flux below -- the normal gradients, the
                // wall-drain interior energy, and (with the weights
                // folded into the gradient) the Braginskii tangential
                // stencil -- are the exact extrapolation
                //     e^{n+theta_c} = (theta_c/theta) e^{n+theta}
                //                     + (1 - theta_c/theta) e^n.
                // The old terms use the chi_coeff_old recipes (hard
                // floors are fine on per-solve constants); the theta
                // terms are the SAME CellState values the default path
                // reads, so the stage value is LINEAR in the theta-stage
                // energy arguments and matrix-free Jacobian probes see
                // the shifted centering exactly -- nothing is
                // re-evaluated nonlinearly at the stage value. The
                // free-streaming caps and the wall-drain gate consume
                // the SAME stage energies (one consistent conduction
                // stage): with theta_c != theta the limiter inputs are
                // stage values, NOT theta or end-of-step values. The chi
                // COEFFICIENTS keep conduction_coefficient_state.
                const amrex::Real inverse_gamma_e_minus_one =
                    1.0_rt / (parameters.gamma_e - 1.0_rt);
                amrex::Real e_spec_electron_left =
                    left.electron_pressure * inverse_gamma_e_minus_one /
                    left.safe_density;
                amrex::Real e_spec_electron_right =
                    right.electron_pressure * inverse_gamma_e_minus_one /
                    right.safe_density;
                amrex::Real e_spec_ion_left =
                    left.ion_internal / left.safe_density;
                amrex::Real e_spec_ion_right =
                    right.ion_internal / right.safe_density;
                // Free-streaming-cap temperatures: the coefficient-state
                // face values by default; stage values (from the stage
                // pressures over the SAME coefficient-state face charge
                // density) when the conduction stage is shifted. Wall
                // faces keep the coefficient-state reservoir average --
                // the drain's sign/cap structure is unchanged, only its
                // e_int argument shifts.
                amrex::Real cap_te = face_te;
                amrex::Real cap_ti = face_ti;
                if (conduction_stage) {
                    const amrex::Real rho_old_left = std::max(
                        rho_old(il, jl, kl), parameters.density_floor);
                    const amrex::Real rho_old_right = std::max(
                        rho_old(i, j, k), parameters.density_floor);
                    const amrex::Real pe_old_left = std::max(
                        (parameters.gamma_e - 1.0_rt) *
                            energy_old(il, jl, kl),
                        parameters.electron_pressure_floor);
                    const amrex::Real pe_old_right = std::max(
                        (parameters.gamma_e - 1.0_rt) * energy_old(i, j, k),
                        parameters.electron_pressure_floor);
                    e_spec_electron_left =
                        stage_new_weight * e_spec_electron_left +
                        stage_old_weight *
                            (pe_old_left * inverse_gamma_e_minus_one /
                             rho_old_left);
                    e_spec_electron_right =
                        stage_new_weight * e_spec_electron_right +
                        stage_old_weight *
                            (pe_old_right * inverse_gamma_e_minus_one /
                             rho_old_right);
                    amrex::Real pi_old_left = 0.0_rt;
                    amrex::Real pi_old_right = 0.0_rt;
                    if (chi_total_energy) {
                        amrex::Real ke_left = 0.0_rt;
                        amrex::Real ke_right = 0.0_rt;
                        for (int component = 0; component < 3;
                             ++component) {
                            ke_left += mom_old(il, jl, kl, component) *
                                       mom_old(il, jl, kl, component);
                            ke_right += mom_old(i, j, k, component) *
                                        mom_old(i, j, k, component);
                        }
                        ke_left *= 0.5_rt / rho_old_left;
                        ke_right *= 0.5_rt / rho_old_right;
                        if (chi_dual_energy) {
                            // Stage-old energies carry the same blend as
                            // the live path (per-solve constants).
                            pi_old_left =
                                theta_implicit_mhd::
                                    dual_energy_blended_pressure(
                                        ion_e_old(il, jl, kl), ke_left,
                                        ion_int_old(il, jl, kl),
                                        ion_int_old(il, jl, kl),
                                        parameters);
                            pi_old_right =
                                theta_implicit_mhd::
                                    dual_energy_blended_pressure(
                                        ion_e_old(i, j, k), ke_right,
                                        ion_int_old(i, j, k),
                                        ion_int_old(i, j, k), parameters);
                        } else {
                            pi_old_left = std::max(
                                (parameters.gamma_i - 1.0_rt) *
                                    (ion_e_old(il, jl, kl) - ke_left),
                                parameters.ion_pressure_floor);
                            pi_old_right = std::max(
                                (parameters.gamma_i - 1.0_rt) *
                                    (ion_e_old(i, j, k) - ke_right),
                                parameters.ion_pressure_floor);
                        }
                        const amrex::Real inverse_gamma_i_minus_one =
                            1.0_rt / (parameters.gamma_i - 1.0_rt);
                        e_spec_ion_left =
                            stage_new_weight * e_spec_ion_left +
                            stage_old_weight *
                                (pi_old_left * inverse_gamma_i_minus_one /
                                 rho_old_left);
                        e_spec_ion_right =
                            stage_new_weight * e_spec_ion_right +
                            stage_old_weight *
                                (pi_old_right * inverse_gamma_i_minus_one /
                                 rho_old_right);
                    }
                    if (!wall_face && chi_needs_state) {
                        // Stage cap temperatures: face-averaged stage
                        // pressures, hard-floored so the thermal speeds
                        // stay defined on extrapolated probe states.
                        const amrex::Real inverse_nkb =
                            PhysConst::q_e /
                            (face_charge_density * PhysConst::kb);
                        cap_te =
                            std::max(0.5_rt * (stage_new_weight *
                                                   (left.electron_pressure +
                                                    right.electron_pressure) +
                                               stage_old_weight *
                                                   (pe_old_left +
                                                    pe_old_right)),
                                     parameters.electron_pressure_floor) *
                            inverse_nkb;
                        if (chi_total_energy) {
                            cap_ti =
                                std::max(0.5_rt *
                                             (stage_new_weight *
                                                  (left.ion_pressure +
                                                   right.ion_pressure) +
                                              stage_old_weight *
                                                  (pi_old_left +
                                                   pi_old_right)),
                                         parameters.ion_pressure_floor) *
                                inverse_nkb;
                        }
                    }
                }
                // Braginskii face geometry (static branch on the host
                // model flag): the face field takes the single-valued
                // staggered B_n and the cell-averaged tangential TOTAL
                // B; |b|^2 is smooth-floored at the field-energy scale
                // for the bhat bhat direction only (the magnetization
                // x = (Omega tau)^2 uses the RAW |b|^2, polynomial in B
                // and exactly 0 at B = 0). The in-plane tangential
                // specific-energy gradients use the standard 4-cell
                // corner stencil; the fluid/parity/domain ghost fills
                // are 2 deep, which covers the transverse-grown kernel
                // box (the axis row reads the parity-mirrored scalars,
                // like the CGL stress stencil). Wall interface faces
                // skip the tangential machinery: they keep the
                // one-sided isotropic drain below with the tensor's nn
                // projection as its scalar chi -- the tensor is never
                // extended across the wall interface.
                amrex::Real brag_b2 = 0.0_rt;
                amrex::Real brag_b2_dir = 1.0_rt;
                amrex::Real brag_bn = 0.0_rt;
                amrex::Real brag_bt = 0.0_rt;
                amrex::Real brag_grad_t_electron = 0.0_rt;
                amrex::Real brag_grad_t_ion = 0.0_rt;
                if (braginskii) {
                    const int tangent1 = (normal + 1) % 3;
                    const int tangent2 = (normal + 2) % 3;
                    const amrex::Real bt1 =
                        0.5_rt * (left.magnetic[tangent1] +
                                  right.magnetic[tangent1]);
                    const amrex::Real bt2 =
                        0.5_rt * (left.magnetic[tangent2] +
                                  right.magnetic[tangent2]);
                    brag_bn = bn_face;
                    brag_b2 = brag_bn * brag_bn + bt1 * bt1 + bt2 * bt2;
                    brag_b2_dir = theta_implicit_mhd::smooth_positive_floor(
                        brag_b2, brag_small_b2);
#if defined(WARPX_DIM_RZ)
                    // In-plane tangent: r (= tangent1) for z-faces,
                    // z (= tangent2) for r-faces; the theta gradient
                    // vanishes for m = 0.
                    brag_bt = (normal == 0) ? bt2 : bt1;
                    if (!wall_face) {
                        const int tangential = (normal == 0) ? 2 : 0;
                        int ipl = il, jpl = jl, kpl = kl;
                        int ipr = i, jpr = j, kpr = k;
                        int iml = il, jml = jl, kml = kl;
                        int imr = i, jmr = j, kmr = k;
                        shift_index(ipl, jpl, kpl, tangential, 1);
                        shift_index(ipr, jpr, kpr, tangential, 1);
                        shift_index(iml, jml, kml, tangential, -1);
                        shift_index(imr, jmr, kmr, tangential, -1);
                        // Neighbor specific internal energies mirror the
                        // CellState recipes exactly (floored electron
                        // pressure; smooth-floored p_i(E_i) recovery).
                        const auto electron_e_spec =
                            [=] (const int ic, const int jc, const int kc) {
                                const amrex::Real pressure = std::max(
                                    (parameters.gamma_e - 1.0_rt) *
                                        energy(ic, jc, kc),
                                    parameters.electron_pressure_floor);
                                return pressure /
                                       ((parameters.gamma_e - 1.0_rt) *
                                        std::max(rho(ic, jc, kc),
                                                 parameters.density_floor));
                            };
                        brag_grad_t_electron =
                            0.25_rt * inverse_tangential_size *
                            (electron_e_spec(ipl, jpl, kpl) +
                             electron_e_spec(ipr, jpr, kpr) -
                             electron_e_spec(iml, jml, kml) -
                             electron_e_spec(imr, jmr, kmr));
                        if (conduction_stage) {
                            // Stage extrapolation of the tangential
                            // samples, folded into the gradient (the
                            // stencil is linear in the samples); the
                            // old samples use the chi_coeff_old
                            // hard-floored recipe, per-solve constants.
                            const auto electron_e_spec_old =
                                [=] (const int ic, const int jc,
                                     const int kc) {
                                    const amrex::Real pressure = std::max(
                                        (parameters.gamma_e - 1.0_rt) *
                                            energy_old(ic, jc, kc),
                                        parameters
                                            .electron_pressure_floor);
                                    return pressure /
                                           ((parameters.gamma_e -
                                             1.0_rt) *
                                            std::max(
                                                rho_old(ic, jc, kc),
                                                parameters
                                                    .density_floor));
                                };
                            brag_grad_t_electron =
                                stage_new_weight * brag_grad_t_electron +
                                stage_old_weight *
                                    (0.25_rt * inverse_tangential_size *
                                     (electron_e_spec_old(ipl, jpl, kpl) +
                                      electron_e_spec_old(ipr, jpr, kpr) -
                                      electron_e_spec_old(iml, jml, kml) -
                                      electron_e_spec_old(imr, jmr,
                                                          kmr)));
                        }
                        if (chi_total_energy) {
                            const auto ion_e_spec =
                                [=] (const int ic, const int jc,
                                     const int kc) {
                                    const amrex::Real safe_density =
                                        std::max(rho(ic, jc, kc),
                                                 parameters.density_floor);
                                    amrex::Real kinetic = 0.0_rt;
                                    for (int component = 0; component < 3;
                                         ++component) {
                                        kinetic +=
                                            mom(ic, jc, kc, component) *
                                            mom(ic, jc, kc, component);
                                    }
                                    kinetic *= 0.5_rt / safe_density;
                                    if (chi_dual_energy) {
                                        // Blended specific internal
                                        // energy, mirroring the dual
                                        // CellState recipe exactly.
                                        return theta_implicit_mhd::
                                                   dual_energy_blended_pressure(
                                                       ion_e(ic, jc, kc),
                                                       kinetic,
                                                       ion_int(ic, jc, kc),
                                                       ion_int_old(ic, jc,
                                                                   kc),
                                                       parameters) /
                                               ((parameters.gamma_i -
                                                 1.0_rt) *
                                                safe_density);
                                    }
                                    const amrex::Real internal_floor =
                                        parameters.ion_pressure_floor /
                                        (parameters.gamma_i - 1.0_rt);
                                    const amrex::Real excess =
                                        ion_e(ic, jc, kc) - kinetic -
                                        internal_floor;
                                    const amrex::Real corner_width =
                                        std::max(
                                            internal_floor,
                                            parameters
                                                    .pressure_corner_width_fraction *
                                                kinetic);
                                    const amrex::Real internal =
                                        internal_floor +
                                        0.5_rt *
                                            (excess +
                                             std::sqrt(
                                                 excess * excess +
                                                 corner_width *
                                                     corner_width));
                                    return internal / safe_density;
                                };
                            brag_grad_t_ion =
                                0.25_rt * inverse_tangential_size *
                                (ion_e_spec(ipl, jpl, kpl) +
                                 ion_e_spec(ipr, jpr, kpr) -
                                 ion_e_spec(iml, jml, kml) -
                                 ion_e_spec(imr, jmr, kmr));
                            if (conduction_stage) {
                                // Stage extrapolation (see the electron
                                // stencil above).
                                const auto ion_e_spec_old =
                                    [=] (const int ic, const int jc,
                                         const int kc) {
                                        const amrex::Real
                                            safe_density_old = std::max(
                                                rho_old(ic, jc, kc),
                                                parameters.density_floor);
                                        amrex::Real kinetic = 0.0_rt;
                                        for (int component = 0;
                                             component < 3; ++component) {
                                            kinetic +=
                                                mom_old(ic, jc, kc,
                                                        component) *
                                                mom_old(ic, jc, kc,
                                                        component);
                                        }
                                        kinetic *=
                                            0.5_rt / safe_density_old;
                                        if (chi_dual_energy) {
                                            return theta_implicit_mhd::
                                                       dual_energy_blended_pressure(
                                                           ion_e_old(ic, jc,
                                                                     kc),
                                                           kinetic,
                                                           ion_int_old(
                                                               ic, jc, kc),
                                                           ion_int_old(
                                                               ic, jc, kc),
                                                           parameters) /
                                                   ((parameters.gamma_i -
                                                     1.0_rt) *
                                                    safe_density_old);
                                        }
                                        const amrex::Real internal_floor =
                                            parameters.ion_pressure_floor /
                                            (parameters.gamma_i - 1.0_rt);
                                        return std::max(
                                                   ion_e_old(ic, jc, kc) -
                                                       kinetic,
                                                   internal_floor) /
                                               safe_density_old;
                                    };
                                brag_grad_t_ion =
                                    stage_new_weight * brag_grad_t_ion +
                                    stage_old_weight *
                                        (0.25_rt *
                                         inverse_tangential_size *
                                         (ion_e_spec_old(ipl, jpl, kpl) +
                                          ion_e_spec_old(ipr, jpr, kpr) -
                                          ion_e_spec_old(iml, jml, kml) -
                                          ion_e_spec_old(imr, jmr,
                                                         kmr)));
                            }
                        }
                    }
#endif
                }
                // Smooth clamps of the Braginskii chi_par/chi_perp
                // (0 = off): smooth-max floor at chi_min; C^2 soft cap
                // at chi_max with knee width chi_max/10 (exact
                // pass-through below 0.9 chi_max). Both maps are
                // monotone, so chi_perp <= chi_par is preserved.
                const auto brag_clamp = [=] (amrex::Real chi_value) {
                    chi_value = theta_implicit_mhd::smooth_positive_floor(
                        chi_value, brag_chi_min);
                    if (brag_chi_max > 0.0_rt) {
                        chi_value = theta_implicit_mhd::soft_upper_clip(
                            chi_value, brag_chi_max, 0.1_rt * brag_chi_max);
                    }
                    return chi_value;
                };
                // Quasi-shorting cross-field boost (implicit_mhd.
                // conduction_qs_chi, braginskii only): ADDITIVE chi_perp
                // keyed to the pseudo-entropy excess
                //     s = (T/T0) (rho0/rho_guarded)^{2/3},
                //     rho_guarded = sqrt(rho^2 + rho_guard^2),
                // above the load envelope, ramped by the C-infinity
                // smooth-max 0.5 ((s - onset) + sqrt((s - onset)^2 +
                // w^2)) with w = 0.3 (onset - 1) -- centered ABOVE the
                // envelope (onset > 1): a ramp centered ON it leaks w/2
                // of the amplitude onto every on-adiabat cell (measured
                // fatal in production). The s inputs (the face
                // temperature and the coefficient-state face density)
                // follow conduction_coefficient_state like every other
                // Braginskii coefficient input; the chi_min/max clamp
                // applies AFTER the addition.
                const auto qs_boost =
                    [=] (const amrex::Real face_temperature) {
                        const amrex::Real guarded_density = std::sqrt(
                            face_density * face_density +
                            qs_density_guard * qs_density_guard);
                        const amrex::Real density_ratio =
                            qs_envelope_density / guarded_density;
                        const amrex::Real entropy =
                            face_temperature * qs_inverse_t0 *
                            std::cbrt(density_ratio * density_ratio);
                        const amrex::Real excess = entropy - qs_onset;
                        return qs_chi * 0.5_rt *
                               (excess + std::sqrt(excess * excess +
                                                   qs_width * qs_width));
                    };
                if ((braginskii && chi_total_energy) || chi_ion > 0.0_rt ||
                    chi_ion_is_parser) {
                    amrex::Real chi_ion_face =
                        chi_ion_is_parser
                            ? chi_ion_parser(face_charge_density, face_te,
                                             face_ti, face_jmag, face_time)
                            : chi_ion;
                    amrex::Real brag_chi_par_ion = 0.0_rt;
                    amrex::Real brag_chi_perp_ion = 0.0_rt;
                    if (braginskii) {
                        // chi_par_i = 3.9 kB Ti tau_i / m_i and the ion
                        // perpendicular fit, Braginskii (1965) Z = 1.
                        const amrex::Real kb_ti = PhysConst::kb * face_ti;
                        const amrex::Real face_number_density =
                            face_charge_density / PhysConst::q_e;
                        const amrex::Real tau_ion =
                            brag_tau_i_coefficient * kb_ti *
                            std::sqrt(kb_ti) / face_number_density;
                        const amrex::Real chi_par_raw =
                            3.9_rt * kb_ti * tau_ion / conduction_ion_mass;
                        const amrex::Real omega_tau =
                            brag_omega_i_coefficient * tau_ion;
                        const amrex::Real x =
                            omega_tau * omega_tau * brag_b2;
                        const amrex::Real chi_perp_raw =
                            chi_par_raw *
                            (brag_i_numerator_1 * x + 1.0_rt) /
                            ((brag_i_denominator_2 * x +
                              brag_i_denominator_1) *
                                 x +
                             1.0_rt);
                        brag_chi_par_ion = brag_clamp(chi_par_raw);
                        // Quasi-shorting boost of the ION channel: s is
                        // keyed on the ion temperature with the SAME
                        // envelope T0 -- the broken-surface shorting
                        // acts on the channel whose own thermal content
                        // breaks the envelope, and a separate ion T0
                        // would be an uncalibratable second knob.
                        amrex::Real chi_perp_ion_value =
                            add_qs ? chi_perp_raw + qs_boost(face_ti)
                                   : chi_perp_raw;
                        // Density-keyed halo boost (see the host
                        // constants): applied to the unclamped perp
                        // value, BEFORE the chi_min/max clamps -- the
                        // The reference code's ntb.f90 t_cond order (boost at ~584,
                        // clamps at ~592-593), so chi_max still caps
                        // the boosted halo diffusivity.
                        if (add_halo_boost) {
                            chi_perp_ion_value =
                                theta_implicit_mhd::vacuum_keyed_resistivity(
                                    chi_perp_ion_value,
                                    chi_charge_to_mass * face_density,
                                    halo_boost_reference,
                                    halo_boost_guard, halo_boost_chi);
                        }
                        brag_chi_perp_ion = brag_clamp(chi_perp_ion_value);
                        // Wall interface faces keep the one-sided
                        // isotropic drain with the tensor's nn
                        // projection as its scalar chi.
                        chi_ion_face =
                            brag_chi_perp_ion +
                            (brag_chi_par_ion - brag_chi_perp_ion) *
                                brag_bn * brag_bn / brag_b2_dir;
                    }
                    amrex::Real conductive_flux;
                    if (wall_face) {
                        // One-sided rectified wall drain (see the host
                        // constants): zero at/below the wall value, C^1
                        // full above twice it -- the reservoir cools the
                        // interior toward T_wall but never heats it.
                        // The interior e_int is the conduction-stage
                        // value; the drain's sign/cap structure is
                        // unchanged.
                        const amrex::Real interior_e_spec =
                            wall_left_masked ? e_spec_ion_right
                                             : e_spec_ion_left;
                        amrex::Real drain =
                            chi_ion_face * face_density *
                            (interior_e_spec - wall_e_spec_ion) *
                            theta_implicit_mhd::floor_outflow_limiter(
                                interior_e_spec, wall_e_spec_ion) *
                            inverse_normal_size;
                        // free-streaming cap, ALWAYS on at the wall face
                        const amrex::Real thermal_speed = std::sqrt(
                            PhysConst::kb * face_ti / conduction_ion_mass);
                        const amrex::Real free_streaming_flux =
                            face_charge_density / PhysConst::q_e *
                            PhysConst::kb * face_ti * thermal_speed;
                        drain /= 1.0_rt +
                                 std::abs(drain) / (wall_conduction_limit *
                                                    free_streaming_flux);
                        // +n flux toward a right-side wall, -n toward a
                        // left-side wall (matches the two-sided sign).
                        conductive_flux =
                            wall_right_masked ? drain : -drain;
                    } else if (braginskii) {
                        // Anisotropic tensor flux: the normal gradient
                        // uses the SAME (conduction-stage) specific
                        // energies as the isotropic path; the tangential
                        // gradient is the corner-stencil value above.
                        const amrex::Real gradient_normal =
                            (e_spec_ion_right - e_spec_ion_left) *
                            inverse_normal_size;
                        conductive_flux =
                            -face_density *
                            (brag_chi_perp_ion * gradient_normal +
                             (brag_chi_par_ion - brag_chi_perp_ion) *
                                 brag_bn *
                                 (brag_bn * gradient_normal +
                                  brag_bt * brag_grad_t_ion) /
                                 brag_b2_dir);
                        if (conduction_limit > 0.0_rt) {
                            // the same free-streaming harmonic cap as
                            // the isotropic path, applied to the TOTAL
                            // (normal + tangential) conductive flux;
                            // the cap temperature is the conduction-
                            // stage value (see cap_ti above)
                            const amrex::Real thermal_speed = std::sqrt(
                                PhysConst::kb * cap_ti /
                                conduction_ion_mass);
                            const amrex::Real free_streaming_flux =
                                face_charge_density / PhysConst::q_e *
                                PhysConst::kb * cap_ti * thermal_speed;
                            conductive_flux /=
                                1.0_rt + std::abs(conductive_flux) /
                                             (conduction_limit *
                                              free_streaming_flux);
                        }
                    } else {
                        conductive_flux =
                            -chi_ion_face * face_density *
                            (e_spec_ion_right - e_spec_ion_left) *
                            inverse_normal_size;
                        if (conduction_limit > 0.0_rt) {
                            // free-streaming cap q_fs = n kB Ti v_ti,
                            // v_ti = sqrt(kB Ti/m_i): the smooth harmonic
                            // form q/(1 + |q|/(f q_fs)), no branches;
                            // Ti is the conduction-stage cap value
                            const amrex::Real thermal_speed = std::sqrt(
                                PhysConst::kb * cap_ti /
                                conduction_ion_mass);
                            const amrex::Real free_streaming_flux =
                                face_charge_density / PhysConst::q_e *
                                PhysConst::kb * cap_ti * thermal_speed;
                            conductive_flux /=
                                1.0_rt + std::abs(conductive_flux) /
                                             (conduction_limit *
                                              free_streaming_flux);
                        }
                    }
                    flux.ion_energy += conductive_flux;
                    if (chi_dual_energy) {
                        // Conduction is a purely internal-energy exchange:
                        // the auxiliary U_i channel receives the identical
                        // face flux the conservative E_i channel books
                        // (its internal-only counterpart is itself).
                        flux.ion_internal_energy += conductive_flux;
                    }
                }
                if (braginskii || chi_electron > 0.0_rt ||
                    chi_electron_is_parser) {
                    amrex::Real chi_electron_face =
                        chi_electron_is_parser
                            ? chi_electron_parser(face_charge_density,
                                                  face_te, face_ti,
                                                  face_jmag, face_time)
                            : chi_electron;
                    amrex::Real brag_chi_par_electron = 0.0_rt;
                    amrex::Real brag_chi_perp_electron = 0.0_rt;
                    if (braginskii) {
                        // chi_par_e = 3.16 kB Te tau_e / m_e and the
                        // electron perpendicular fit, Braginskii (1965)
                        // Z = 1.
                        const amrex::Real kb_te = PhysConst::kb * face_te;
                        const amrex::Real face_number_density =
                            face_charge_density / PhysConst::q_e;
                        const amrex::Real tau_electron =
                            brag_tau_e_coefficient * kb_te *
                            std::sqrt(kb_te) / face_number_density;
                        const amrex::Real chi_par_raw =
                            3.16_rt * kb_te * tau_electron /
                            PhysConst::m_e;
                        const amrex::Real omega_tau =
                            brag_omega_e_coefficient * tau_electron;
                        const amrex::Real x =
                            omega_tau * omega_tau * brag_b2;
                        const amrex::Real chi_perp_raw =
                            chi_par_raw *
                            (brag_e_numerator_1 * x + 1.0_rt) /
                            ((brag_e_denominator_2 * x +
                              brag_e_denominator_1) *
                                 x +
                             1.0_rt);
                        brag_chi_par_electron = brag_clamp(chi_par_raw);
                        // Quasi-shorting boost (see the ion channel and
                        // the qs_boost lambda): additive chi_perp, keyed
                        // on the electron temperature, clamped after.
                        amrex::Real chi_perp_electron_value =
                            add_qs ? chi_perp_raw + qs_boost(face_te)
                                   : chi_perp_raw;
                        // Density-keyed halo boost, pre-clamp like the
                        // ion channel (the reference code's ntb.f90 order).
                        if (add_halo_boost) {
                            chi_perp_electron_value =
                                theta_implicit_mhd::vacuum_keyed_resistivity(
                                    chi_perp_electron_value,
                                    chi_charge_to_mass * face_density,
                                    halo_boost_reference,
                                    halo_boost_guard, halo_boost_chi);
                        }
                        brag_chi_perp_electron =
                            brag_clamp(chi_perp_electron_value);
                        // Wall interface faces keep the one-sided
                        // isotropic drain with the tensor's nn
                        // projection as its scalar chi.
                        chi_electron_face =
                            brag_chi_perp_electron +
                            (brag_chi_par_electron -
                             brag_chi_perp_electron) *
                                brag_bn * brag_bn / brag_b2_dir;
                    }
                    amrex::Real conductive_flux;
                    if (wall_face) {
                        // One-sided rectified wall drain (see the ion
                        // channel above): the interior e_int is the
                        // conduction-stage value, the structure is
                        // unchanged.
                        const amrex::Real interior_e_spec =
                            wall_left_masked ? e_spec_electron_right
                                             : e_spec_electron_left;
                        amrex::Real drain =
                            chi_electron_face * face_density *
                            (interior_e_spec - wall_e_spec_electron) *
                            theta_implicit_mhd::floor_outflow_limiter(
                                interior_e_spec, wall_e_spec_electron) *
                            inverse_normal_size;
                        const amrex::Real thermal_speed = std::sqrt(
                            PhysConst::kb * face_te / PhysConst::m_e);
                        const amrex::Real free_streaming_flux =
                            face_charge_density / PhysConst::q_e *
                            PhysConst::kb * face_te * thermal_speed;
                        drain /= 1.0_rt +
                                 std::abs(drain) / (wall_conduction_limit *
                                                    free_streaming_flux);
                        conductive_flux =
                            wall_right_masked ? drain : -drain;
                    } else if (braginskii) {
                        // Anisotropic tensor flux (see the ion channel).
                        const amrex::Real gradient_normal =
                            (e_spec_electron_right -
                             e_spec_electron_left) *
                            inverse_normal_size;
                        conductive_flux =
                            -face_density *
                            (brag_chi_perp_electron * gradient_normal +
                             (brag_chi_par_electron -
                              brag_chi_perp_electron) *
                                 brag_bn *
                                 (brag_bn * gradient_normal +
                                  brag_bt * brag_grad_t_electron) /
                                 brag_b2_dir);
                        if (conduction_limit > 0.0_rt) {
                            // cap at the conduction-stage temperature
                            const amrex::Real thermal_speed = std::sqrt(
                                PhysConst::kb * cap_te / PhysConst::m_e);
                            const amrex::Real free_streaming_flux =
                                face_charge_density / PhysConst::q_e *
                                PhysConst::kb * cap_te * thermal_speed;
                            conductive_flux /=
                                1.0_rt + std::abs(conductive_flux) /
                                             (conduction_limit *
                                              free_streaming_flux);
                        }
                    } else {
                        conductive_flux =
                            -chi_electron_face * face_density *
                            (e_spec_electron_right -
                             e_spec_electron_left) *
                            inverse_normal_size;
                        if (conduction_limit > 0.0_rt) {
                            // cap at the conduction-stage temperature
                            const amrex::Real thermal_speed = std::sqrt(
                                PhysConst::kb * cap_te / PhysConst::m_e);
                            const amrex::Real free_streaming_flux =
                                face_charge_density / PhysConst::q_e *
                                PhysConst::kb * cap_te * thermal_speed;
                            conductive_flux /=
                                1.0_rt + std::abs(conductive_flux) /
                                             (conduction_limit *
                                              free_streaming_flux);
                        }
                    }
                    flux.electron_energy += conductive_flux;
                }
            }

#if defined(WARPX_DIM_RZ)
            if (reflect_wall && i == radial_wall_face) {
                // Zero-flux wall (see the reflect_wall comment above):
                // every ADVECTIVE fluid channel of the wall face -- mass,
                // tangential momentum advection, electron/ion energies,
                // U_par/U_perp, and the electron pdV velocity -- is
                // EXACTLY zero, applied after the donor gating so no
                // limiter can reintroduce a leak. The tangential Maxwell
                // stress -B_n B_t/mu0 (the Alfven-fan torque channel the
                // open B_n opens against the mirror states -- the E_i
                // work drain of the last ring) takes its PEC-image value
                // (zero) in BOTH the momentum flux and the magnetic work
                // register, keeping the field-to-fluid work discretely
                // paired. The NORMAL channel keeps the open fan in both
                // registers: the free-space field genuinely acts on the
                // wall band through the face (ram + total pressure -
                // B_n^2 tension), and swapping it for the PEC image
                // overcompresses the sealed last ring until the ring
                // behind it deadlocks on the E_i floor (150-consecutive
                // frozen Newton solves on the FRC open-wall wedge). The
                // override is state-independent in structure, so the
                // residual stays smooth for the JFNK probes.
                flux.mass = 0.0_rt;
                flux.momentum[1] = 0.0_rt;
                flux.momentum[2] = 0.0_rt;
                flux.momentum_magnetic[1] = 0.0_rt;
                flux.momentum_magnetic[2] = 0.0_rt;
                flux.electron_energy = 0.0_rt;
                flux.ion_energy = 0.0_rt;
                flux.ion_parallel_energy = 0.0_rt;
                flux.ion_perp_energy = 0.0_rt;
                flux.ion_internal_energy = 0.0_rt;
                flux.electron_velocity = 0.0_rt;
            }
#endif
            flux_arr(i, j, k, flux_mass) = flux.mass;
            for (int component = 0; component < 3; ++component) {
                flux_arr(i, j, k, flux_momentum + component) =
                    flux.momentum[component];
                flux_arr(i, j, k, flux_magnetic + component) =
                    flux.momentum_magnetic[component];
            }
            flux_arr(i, j, k, flux_electron_energy) = flux.electron_energy;
            flux_arr(i, j, k, flux_ion_energy) = flux.ion_energy;
            // Zero under the non-cgl closures (hlld_flux leaves the
            // struct defaults untouched there).
            flux_arr(i, j, k, flux_ion_parallel_energy) =
                flux.ion_parallel_energy;
            flux_arr(i, j, k, flux_ion_perp_energy) = flux.ion_perp_energy;
            // Zero under the non-dual closures (the kernels leave the
            // struct default untouched there).
            flux_arr(i, j, k, flux_ion_internal_energy) =
                flux.ion_internal_energy;
            flux_arr(i, j, k, flux_electron_velocity) = flux.electron_velocity;
            flux_arr(i, j, k, flux_induction_t1) = flux.induction_t1;
            flux_arr(i, j, k, flux_induction_t2) = flux.induction_t2;
            flux_arr(i, j, k, flux_signal_left) = flux.signal_left;
            flux_arr(i, j, k, flux_signal_right) = flux.signal_right;
            flux_arr(i, j, k, flux_alfven_left) = flux.signal_alfven_left;
            flux_arr(i, j, k, flux_alfven_right) = flux.signal_alfven_right;
        });
    }
#else
    amrex::ignore_unused(face_flux_mf, normal_direction);
    WARPX_ABORT_WITH_MESSAGE(
        "ThetaImplicitMHD::ComputeDirectionalFaceFluxes() requires 1D or RZ "
        "geometry");
#endif
}

void ThetaImplicitMHD::AccumulateAbsorbedWallLedger (const amrex::Real dt,
                                                     const int step,
                                                     const amrex::Real a_time)
{
#if defined(WARPX_DIM_RZ)
    using ablastr::fields::Direction;
    // The absorbing wall is a physical sink by design: integrate the
    // r-weighted wall-face fluxes so the conservation ledger stays
    // honest. The theta-state radial face fluxes are recomputed exactly
    // as the residual computed them (the register may hold a rejected
    // line-search trial or a Jacobian probe at solver exit):
    // UpdateWarpXFields(m_state, ...) has just refilled the fluid
    // sources and the theta-stage B, so only the plasma current, the
    // cell-centered fields, and the radial flux pass need re-running.
    // For a converged solve U^{n+1} - U^n = dt * RHS(theta state), so
    // the counters match the domain's mass/energy change through the
    // wall to the nonlinear solver tolerance (frozen/stagnated solves
    // violate this at the residual level, as they do all conservation).
    const auto magnetic_field =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, 0);
    m_hybrid_pic_model->CalculatePlasmaCurrent(magnetic_field,
                                               m_WarpX->GetEBUpdateEFlag());
    if (m_z_neumann) {
        for (int direction = 0; direction < 3; ++direction) {
            amrex::MultiFab& current_component = *m_WarpX->m_fields.get(
                FieldType::hybrid_current_fp_plasma, Direction{direction}, 0);
            ApplyNeumannZDomainGhosts(current_component, 1);
            if (direction == 2) {
                // z_lo mirror: J_z is ODD (see ComputeRHS).
                ApplyMirrorZLoDomainGhosts(current_component, {-1, -1, -1});
            }
        }
    }
    FillCellCenteredElectromagneticFields();
    amrex::MultiFab& face_flux_r = *m_WarpX->m_fields.get(FaceFluxRName, 0);
    ComputeDirectionalFaceFluxes(face_flux_r, 0, a_time);

    const amrex::Box face_domain = amrex::convert(
        m_WarpX->Geom(0).Domain(), face_flux_r.ixType().toIntVect());
    amrex::Box wall_plane = face_domain;
    wall_plane.setSmall(0, face_domain.bigEnd(0));
    constexpr int flux_mass = FaceFluxComponent::mass;
    constexpr int flux_electron_energy = FaceFluxComponent::electron_energy;
    constexpr int flux_ion_energy = FaceFluxComponent::ion_energy;
    constexpr int flux_ion_parallel_energy =
        FaceFluxComponent::ion_parallel_energy;
    constexpr int flux_ion_perp_energy = FaceFluxComponent::ion_perp_energy;
    // Fluid energy channels present under the active closure: U_e always;
    // the conservative E_i (internal + kinetic) under total_energy AND
    // dual_energy (the auxiliary U_i channel is bookkeeping, not a
    // conserved energy -- booking it would double-count E_i); the purely
    // internal U_par/U_perp pair under cgl. (Under barotropic the fan's
    // E_i register is a dormant pseudo-channel and is excluded.)
    const bool total_energy_closure = (m_ion_closure == "total_energy") ||
                                      (m_ion_closure == "dual_energy");
    const bool cgl_closure = (m_ion_closure == "cgl");
    const bool book_absorb = m_r_open && (m_r_open_fluid == "absorb");
    const bool book_wall = m_wall_mask.GetThermalBC() !=
                           ImplicitMHDWallMask::ThermalBC::none;

    if (book_absorb) {
        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (amrex::MFIter mfi(face_flux_r); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox() & wall_plane;
            if (box.isEmpty()) {
                continue;
            }
            const auto flux_arr = face_flux_r.const_array(mfi);
            reduce_op.eval(
                box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                    amrex::Real energy_flux =
                        flux_arr(i, j, k, flux_electron_energy);
                    if (total_energy_closure) {
                        energy_flux += flux_arr(i, j, k, flux_ion_energy);
                    } else if (cgl_closure) {
                        energy_flux +=
                            flux_arr(i, j, k, flux_ion_parallel_energy) +
                            flux_arr(i, j, k, flux_ion_perp_energy);
                    }
                    return {flux_arr(i, j, k, flux_mass), energy_flux};
                });
        }
        auto sums = reduce_data.value(reduce_op);
        // Wall-face annulus area per z-cell, 2 pi r_wall dz: with the
        // cylindrical divergence weights (r_face/r_center)/dr, the
        // interior faces telescope out of the r-weighted domain totals
        // and the change per step is exactly
        // dt * sum_j 2 pi dz r_wall F_wall(j).
        const amrex::Real wall_radius =
            m_WarpX->Geom(0).ProbLo(0) +
            face_domain.bigEnd(0) * m_WarpX->Geom(0).CellSize(0);
        const amrex::Real face_area = 2.0_rt * MathConst::pi * wall_radius *
                                      m_WarpX->Geom(0).CellSize(1);
        amrex::Real step_totals[2] = {dt * face_area * amrex::get<0>(sums),
                                      dt * face_area * amrex::get<1>(sums)};
        amrex::ParallelAllReduce::Sum(
            step_totals, 2, amrex::ParallelContext::CommunicatorSub());
        m_absorbed_wall_mass += step_totals[0];
        m_absorbed_wall_energy += step_totals[1];
    }

    if (book_wall) {
        // Shaped-wall deposition: what crosses a stair-step interface
        // face is gone from the fluid (the masked band's increments are
        // zero), so the interface fluxes ARE the wall load -- advective
        // capture plus the one-sided conductive drain, booked with the
        // exact per-face annulus areas (2 pi r_face dz for r-normal
        // faces, 2 pi r_center dr for z-normal faces). Signs are taken
        // INTO the wall. The z-direction register is recomputed at the
        // same accepted theta state.
        amrex::MultiFab& face_flux_z =
            *m_WarpX->m_fields.get(FaceFluxZName, 0);
        ComputeDirectionalFaceFluxes(face_flux_z, 2, a_time);

        const int* const AMREX_RESTRICT fm =
            m_wall_mask.FirstMaskedCellCentered();
        const int mask_z_lo = -m_wall_mask.GhostCells();
        const int mask_z_hi =
            m_wall_mask.AxialCells() - 1 + m_wall_mask.GhostCells();
        const amrex::Real radial_lower = m_WarpX->Geom(0).ProbLo(0);
        const amrex::Real dr = m_WarpX->Geom(0).CellSize(0);
        const amrex::Real dz = m_WarpX->Geom(0).CellSize(1);
        const amrex::Real two_pi = 2.0_rt * MathConst::pi;

        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        // Face-type validboxes SHARE their boundary faces between
        // adjacent boxes: without unique ownership a stair interface
        // face sitting on a grid seam is booked once per box (caught by
        // the ctest closure gate with the seam parked on the step
        // ledge). The nodal owner masks give each face to exactly one
        // box.
        const auto owner_r =
            face_flux_r.OwnerMask(m_WarpX->Geom(0).periodicity());
        for (amrex::MFIter mfi(face_flux_r); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox() & face_domain;
            if (box.isEmpty()) {
                continue;
            }
            const auto flux_arr = face_flux_r.const_array(mfi);
            const auto own = owner_r->const_array(mfi);
            reduce_op.eval(
                box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                    const int jc =
                        std::max(mask_z_lo, std::min(mask_z_hi, j));
                    const bool left_masked = (i - 1 >= fm[jc]);
                    const bool right_masked = (i >= fm[jc]);
                    if (left_masked == right_masked || !own(i, j, k)) {
                        return {0.0_rt, 0.0_rt};
                    }
                    // +n flux enters a right-side wall; -n a left-side.
                    const amrex::Real sign =
                        right_masked ? 1.0_rt : -1.0_rt;
                    const amrex::Real area =
                        two_pi * (radial_lower + i * dr) * dz;
                    amrex::Real energy_flux =
                        flux_arr(i, j, k, flux_electron_energy);
                    if (total_energy_closure) {
                        energy_flux += flux_arr(i, j, k, flux_ion_energy);
                    } else if (cgl_closure) {
                        energy_flux +=
                            flux_arr(i, j, k, flux_ion_parallel_energy) +
                            flux_arr(i, j, k, flux_ion_perp_energy);
                    }
                    return {sign * area * flux_arr(i, j, k, flux_mass),
                            sign * area * energy_flux};
                });
        }
        const amrex::Box z_face_domain = amrex::convert(
            m_WarpX->Geom(0).Domain(), face_flux_z.ixType().toIntVect());
        const auto owner_z =
            face_flux_z.OwnerMask(m_WarpX->Geom(0).periodicity());
        for (amrex::MFIter mfi(face_flux_z); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox() & z_face_domain;
            if (box.isEmpty()) {
                continue;
            }
            const auto flux_arr = face_flux_z.const_array(mfi);
            const auto own = owner_z->const_array(mfi);
            reduce_op.eval(
                box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                    const int jl =
                        std::max(mask_z_lo, std::min(mask_z_hi, j - 1));
                    const int jr =
                        std::max(mask_z_lo, std::min(mask_z_hi, j));
                    const bool left_masked = (i >= fm[jl]);
                    const bool right_masked = (i >= fm[jr]);
                    if (left_masked == right_masked || !own(i, j, k)) {
                        return {0.0_rt, 0.0_rt};
                    }
                    const amrex::Real sign =
                        right_masked ? 1.0_rt : -1.0_rt;
                    const amrex::Real area =
                        two_pi * (radial_lower + (i + 0.5_rt) * dr) * dr;
                    amrex::Real energy_flux =
                        flux_arr(i, j, k, flux_electron_energy);
                    if (total_energy_closure) {
                        energy_flux += flux_arr(i, j, k, flux_ion_energy);
                    } else if (cgl_closure) {
                        energy_flux +=
                            flux_arr(i, j, k, flux_ion_parallel_energy) +
                            flux_arr(i, j, k, flux_ion_perp_energy);
                    }
                    return {sign * area * flux_arr(i, j, k, flux_mass),
                            sign * area * energy_flux};
                });
        }
        auto sums = reduce_data.value(reduce_op);
        amrex::Real step_totals[2] = {dt * amrex::get<0>(sums),
                                      dt * amrex::get<1>(sums)};
        amrex::ParallelAllReduce::Sum(
            step_totals, 2, amrex::ParallelContext::CommunicatorSub());
        m_shaped_wall_mass += step_totals[0];
        m_shaped_wall_energy += step_totals[1];
    }

    if (m_absorb_ledger_interval > 0 &&
        (step + 1) % m_absorb_ledger_interval == 0) {
        if (book_absorb) {
            amrex::Print().SetPrecision(17)
                << "MHD absorb wall ledger: step " << step + 1
                << " absorbed mass [kg] = " << m_absorbed_wall_mass
                << " absorbed energy [J] = " << m_absorbed_wall_energy
                << "\n";
        }
        if (book_wall) {
            amrex::Print().SetPrecision(17)
                << "MHD shaped wall ledger: step " << step + 1
                << " deposited mass [kg] = " << m_shaped_wall_mass
                << " deposited energy [J] = " << m_shaped_wall_energy
                << "\n";
        }
        if (amrex::ParallelDescriptor::IOProcessor()) {
            // Truncate at the first write of the run (a stale file from a
            // previous run in the same directory would otherwise keep
            // accumulating appended rows), append afterwards.
            if (book_absorb && !m_absorb_ledger_file.empty()) {
                std::ofstream ledger(m_absorb_ledger_file,
                                     m_absorb_ledger_started
                                         ? std::ios::app
                                         : std::ios::trunc);
                m_absorb_ledger_started = true;
                ledger.precision(17);
                ledger << step + 1 << " " << m_absorbed_wall_mass << " "
                       << m_absorbed_wall_energy << "\n";
            }
            if (book_wall && !m_wall_ledger_file.empty()) {
                std::ofstream ledger(m_wall_ledger_file,
                                     m_wall_ledger_started
                                         ? std::ios::app
                                         : std::ios::trunc);
                m_wall_ledger_started = true;
                ledger.precision(17);
                ledger << step + 1 << " " << m_shaped_wall_mass << " "
                       << m_shaped_wall_energy << "\n";
            }
        }
    }
#else
    amrex::ignore_unused(dt, step);
#endif
}

void ThetaImplicitMHD::AccumulateFloorConsistencySupplyLedger (
    const amrex::Real dt, const int step)
{
    // The supply is a pure state-local source: unlike the wall ledgers no
    // flux recompute is needed -- re-evaluate the SAME rectified deficit
    // the residual kernels applied, at the accepted theta state (m_state)
    // against the frozen step-old state (m_state_old). For a converged
    // solve U^{n+1} - U^n = dt * RHS(theta state), so the booked amounts
    // dt * rate_eff * deficit match the supply's share of the state
    // change to the nonlinear solver tolerance (frozen/stagnated solves
    // violate this at the residual level, as they do all conservation).
    const bool dual_energy_closure = m_ion_closure == "dual_energy";
    const bool total_energy_closure =
        m_ion_closure == "total_energy" || dual_energy_closure;
    const bool cgl_closure = m_ion_closure == "cgl";
    const int num_blocks = (cgl_closure || dual_energy_closure)
                               ? 4
                               : (total_energy_closure ? 3 : 2);
    const std::array<const char*, 4> block_names = {
        MassDensityName, ElectronEnergyName,
        cgl_closure ? IonParallelEnergyName : IonEnergyName,
        dual_energy_closure ? IonInternalEnergyName : IonPerpEnergyName};
    const AdmissibilityBounds bounds = MakeAdmissibilityBounds();
    const amrex::Real theta = m_theta;
    const amrex::Real theta_dt = m_theta * dt;
    const amrex::Real rate_eff =
        std::min(m_floor_consistency_rate, 1.0_rt / theta_dt);
    const amrex::MultiFab& old_density_mf =
        m_state_old.getMultiFabBlock(MassDensityName, 0);
    // A frozen ion fluid gets no mass increment in the kernels, so
    // nothing is supplied to it either.
    const int first_block = m_evolve_ion_fluid ? 0 : 1;

    // Geometry cell measure: product of the cell sizes, with the RZ
    // radial annulus weight 2 pi r_center applied per cell in-kernel, so
    // the booked units are kg and J (in 1D per unit cross-section,
    // kg/m^2 and J/m^2 -- the geometry's own measure).
    amrex::Real cell_volume = 1.0_rt;
    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
        cell_volume *= m_WarpX->Geom(0).CellSize(dim);
    }
    const amrex::Real fc_width = m_floor_consistency_width_fraction;
#if defined(WARPX_DIM_RZ)
    const amrex::Real radial_lower = m_WarpX->Geom(0).ProbLo(0);
    const amrex::Real radial_cell_size = m_WarpX->Geom(0).CellSize(0);
#endif
    // Rigid-conductor wall exclusion, mirroring the residual kernels'
    // wall_live factor.
    const bool wall_thermal_freeze =
        (m_wall_mask.GetThermalBC() != ImplicitMHDWallMask::ThermalBC::none);
    const int* const AMREX_RESTRICT wall_first_masked_cc =
        wall_thermal_freeze ? m_wall_mask.FirstMaskedCellCentered() : nullptr;
    const int wall_mask_z_lo = -m_wall_mask.GhostCells();
    const int wall_mask_z_hi =
        m_wall_mask.AxialCells() - 1 + m_wall_mask.GhostCells();

    // Dual-energy: the auxiliary U_i block (block 3) receives its own
    // supply in the residual, but U_i is bookkeeping, not a conserved
    // quantity -- booking it as energy would double-count against the
    // conservative E_i block. Only the conserved blocks are ledgered.
    const int num_booked_blocks =
        dual_energy_closure ? 3 : num_blocks;
    amrex::Real step_totals[2] = {0.0_rt, 0.0_rt}; // supplied mass, energy
    for (int block = first_block; block < num_booked_blocks; ++block) {
        const amrex::Real floor = bounds.floors[block];
        const amrex::Real temperature_coefficient =
            bounds.temperature_coefficients[block];
        const amrex::MultiFab& value_mf =
            m_state.getMultiFabBlock(block_names[block], 0);
        const amrex::MultiFab& old_mf =
            m_state_old.getMultiFabBlock(block_names[block], 0);
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (amrex::MFIter mfi(value_mf); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto value = value_mf.const_array(mfi);
            const auto old_value = old_mf.const_array(mfi);
            const auto old_density = old_density_mf.const_array(mfi);
            reduce_op.eval(
                box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                    if (wall_thermal_freeze) {
                        const int mask_jz = std::max(
                            wall_mask_z_lo, std::min(wall_mask_z_hi, j));
                        if (i >= wall_first_masked_cc[mask_jz]) {
                            return {0.0_rt};
                        }
                    }
                    amrex::Real measure = 1.0_rt;
#if defined(WARPX_DIM_RZ)
                    measure = 2.0_rt * MathConst::pi *
                              (radial_lower +
                               (i + 0.5_rt) * radial_cell_size);
#endif
                    return {measure *
                            theta_implicit_mhd::floor_consistency_deficit(
                                value(i, j, k), old_value(i, j, k),
                                old_density(i, j, k), floor,
                                temperature_coefficient, theta, fc_width)};
                });
        }
        const amrex::Real block_total =
            dt * rate_eff * cell_volume *
            amrex::get<0>(reduce_data.value(reduce_op));
        step_totals[block == 0 ? 0 : 1] += block_total;
    }
    amrex::ParallelAllReduce::Sum(step_totals, 2,
                                  amrex::ParallelContext::CommunicatorSub());
    m_floor_supplied_mass += step_totals[0];
    m_floor_supplied_energy += step_totals[1];

    if (!m_floor_ledger_file.empty() &&
        amrex::ParallelDescriptor::IOProcessor()) {
        // Truncate at the first write of the run (a stale file from a
        // previous run in the same directory would otherwise keep
        // accumulating appended rows), append afterwards.
        std::ofstream ledger(m_floor_ledger_file, m_floor_ledger_started
                                                      ? std::ios::app
                                                      : std::ios::trunc);
        m_floor_ledger_started = true;
        ledger.precision(17);
        ledger << step + 1 << " " << m_floor_supplied_mass << " "
               << m_floor_supplied_energy << "\n";
    }
}

bool ThetaImplicitMHD::PrepareResistiveStageCurrents (
    const bool at_resistive_stage) const
{
    if (!at_resistive_stage || m_resistive_theta == m_theta) {
        return false;
    }
    // Resistive-stage current (see m_resistive_theta): J = curl B/mu0 is
    // LINEAR in B, so the end-of-step current is the exact extrapolation
    // J^{n+1} = (J^{n+theta} - (1-theta) J^n)/theta and the theta_r-stage
    // current is the linear combination
    //     J^{n+theta_r} = (theta_r/theta) J^{n+theta}
    //                     + (1 - theta_r/theta) J^n,
    // linear in the Newton iterate, so matrix-free Jacobian probes see
    // the shifted centering exactly. Ghosts are combined too (the Ohm
    // stencils read neighbor currents).
    using ablastr::fields::Direction;
    const amrex::Real new_weight = m_resistive_theta / m_theta;
    const amrex::Real old_weight = 1.0_rt - new_weight;
    for (int direction = 0; direction < 3; ++direction) {
        const amrex::MultiFab& plasma_current = *m_WarpX->m_fields.get(
            FieldType::hybrid_current_fp_plasma, Direction{direction}, 0);
        const amrex::MultiFab& old_current = *m_WarpX->m_fields.get(
            OldPlasmaCurrentName, Direction{direction}, 0);
        amrex::MultiFab& stage_current = *m_WarpX->m_fields.get(
            ResistiveStageCurrentName, Direction{direction}, 0);
        amrex::MultiFab::LinComb(stage_current, new_weight, plasma_current, 0,
                                 old_weight, old_current, 0, 0, 1,
                                 stage_current.nGrowVect());
    }
    return true;
}

void ThetaImplicitMHD::AssembleOhmElectricField (const amrex::Real time,
                                                 const bool at_resistive_stage) const
{
#if defined(WARPX_DIM_1D_Z)
    using ablastr::fields::Direction;
    // Resistive/Hall-MHD Ohm's law, E = -u x B [+ J x B/rho_q] + eta J,
    // assembled at the native Yee staggering. The ideal part of the tangential
    // components is the SAME Riemann induction flux that closes the fluid
    // update (Ex and Ey live on z-faces in 1D): F_{B_t1} = -EMF_t2 and
    // F_{B_t2} = +EMF_t1, so Ey = -F_Bx and Ex = +F_By. Ez (cell
    // centered) does not enter the 1D curl and is assembled from the
    // cell-centered fields for diagnostics. Under the split-field
    // external-A scheme the evolved B is the plasma response, so E_ext is
    // subtracted (dB_ext/dt = -curl E_ext is carried by the externals).
    amrex::MultiFab& electric_field_x =
        *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, 0);
    amrex::MultiFab& electric_field_y =
        *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{1}, 0);
    amrex::MultiFab& electric_field_z =
        *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{2}, 0);
    // Dissipative terms read the resistive-stage current when the shifted
    // centering is active (see PrepareResistiveStageCurrents); everything
    // current-related in this assembly is dissipative, so the swap covers
    // eta J, the vacuum boost's |J| argument, and the eta_H stencils.
    const bool resistive_stage =
        PrepareResistiveStageCurrents(at_resistive_stage);
    const amrex::MultiFab& current_x =
        *(resistive_stage
              ? m_WarpX->m_fields.get(ResistiveStageCurrentName, Direction{0}, 0)
              : m_WarpX->m_fields.get(FieldType::hybrid_current_fp_plasma,
                                      Direction{0}, 0));
    const amrex::MultiFab& current_y =
        *(resistive_stage
              ? m_WarpX->m_fields.get(ResistiveStageCurrentName, Direction{1}, 0)
              : m_WarpX->m_fields.get(FieldType::hybrid_current_fp_plasma,
                                      Direction{1}, 0));
    const amrex::MultiFab& current_z =
        *(resistive_stage
              ? m_WarpX->m_fields.get(ResistiveStageCurrentName, Direction{2}, 0)
              : m_WarpX->m_fields.get(FieldType::hybrid_current_fp_plasma,
                                      Direction{2}, 0));
    // The Hall EMF is REACTIVE (it rotates B without dissipating energy),
    // so it keeps the global theta staging of the ideal EMF: it reads the
    // theta-stage plasma current directly, never the resistive-stage
    // extrapolation above (that shift exists to over-center the
    // DISSIPATIVE channels; applied to the whistler rotation it would add
    // first-order numerical damping exactly where the theta method is
    // chosen for its energy-conserving centering).
    const bool include_hall = m_hybrid_pic_model->m_include_hall_term;
    const amrex::MultiFab& hall_current_x = *m_WarpX->m_fields.get(
        FieldType::hybrid_current_fp_plasma, Direction{0}, 0);
    const amrex::MultiFab& hall_current_y = *m_WarpX->m_fields.get(
        FieldType::hybrid_current_fp_plasma, Direction{1}, 0);
    const amrex::MultiFab& hall_current_z = *m_WarpX->m_fields.get(
        FieldType::hybrid_current_fp_plasma, Direction{2}, 0);
    // Electron inertia: the nodal inertial field assembled by
    // ComputeElectronInertiaNodal from the SAME theta-stage state
    // (reactive like the Hall EMF: never the resistive-stage
    // extrapolation, never the conduction_theta shift).
    const bool include_inertia =
        m_hybrid_pic_model->m_include_electron_inertia;
    const amrex::MultiFab* const inertia_nodal_mf = include_inertia
        ? m_WarpX->m_fields.get("hybrid_E_inertial_nodal", 0) : nullptr;
    const amrex::MultiFab& charge_density =
        *m_WarpX->m_fields.get(FieldType::rho_fp, 0);
    // Te [K] of the (rho, Te, J, t) parser: the temperature-primary nodal
    // register (same staggering as rho_fp), read/averaged with exactly the
    // rho_q stencils at every eta evaluation point of this assembly.
    const amrex::MultiFab& electron_temperature =
        *m_WarpX->m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
    const amrex::MultiFab& density = *m_WarpX->m_fields.get(MassDensityName, 0);
    const amrex::MultiFab& momentum =
        *m_WarpX->m_fields.get(MomentumDensityName, 0);
    const amrex::MultiFab& magnetic_cc =
        *m_WarpX->m_fields.get(MagneticFieldCCName, 0);
    const amrex::MultiFab& face_flux_mf =
        *m_WarpX->m_fields.get(FaceFluxZName, 0);

    const auto eta = m_hybrid_pic_model->m_eta;
    const auto eta_h = m_hybrid_pic_model->m_eta_h;
    const bool include_hyper_resistivity =
        m_hybrid_pic_model->m_include_hyper_resistivity_term;
    const bool hyper_resistivity_needs_b =
        m_hybrid_pic_model->m_hyper_resistivity_has_B_dependence;
    const amrex::Real axial_cell_size = m_WarpX->Geom(0).CellSize(0);
    const amrex::Real inverse_dz2 =
        1.0_rt / (axial_cell_size * axial_cell_size);
    // Ohm's-law division guards: the eta inputs and the -(u x B) velocity
    // reconstruction floor at the hybrid n_floor (mass-density equivalent),
    // independent of the fluid admissibility floor. In particular the dust
    // regime relies on this: frozen dust momentum over the Ohm guard keeps
    // u ~ 0 there, so E relaxes to eta J in the vacuum region.
    const amrex::Real density_floor = OhmMassDensityFloor();
    const amrex::Real charge_density_floor =
        m_ion_charge_to_mass * density_floor;
    // Density-keyed vacuum resistivity of the FIELD advance (see
    // vacuum_keyed_resistivity in ThetaImplicitMHD_K.H): keyed to the
    // vacuum reference (the Ohm guard, raised to the per-step frozen
    // dynamic reference under vacuum_reference_peak_fraction), divisions
    // guarded at the (far lower) positivity floor so the boost stays
    // uncapped over the reachable density range. Joule heating keeps the
    // un-boosted user eta.
    const amrex::Real vacuum_eta_scale =
        PhysConst::mu0 * m_vacuum_resistivity_diffusivity;
    const amrex::Real vacuum_division_guard =
        m_ion_charge_to_mass * m_mass_density_floor;
    const amrex::Real vacuum_reference_charge_density =
        m_ion_charge_to_mass * VacuumReferenceMassDensity();
    constexpr int flux_induction_t1 = FaceFluxComponent::induction_t1;
    constexpr int flux_induction_t2 = FaceFluxComponent::induction_t2;

    // Ex and Ey (z-nodal, i.e. on the z-faces).
    for (amrex::MFIter mfi(electric_field_x); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto electric_x = electric_field_x.array(mfi);
        const auto electric_y = electric_field_y.array(mfi);
        const auto j_x = current_x.const_array(mfi);
        const auto j_y = current_y.const_array(mfi);
        const auto j_z = current_z.const_array(mfi);
        const auto jh_x = hall_current_x.const_array(mfi);
        const auto jh_y = hall_current_y.const_array(mfi);
        const auto jh_z = hall_current_z.const_array(mfi);
        amrex::Array4<const amrex::Real> ei_nodal;
        if (inertia_nodal_mf) { ei_nodal = inertia_nodal_mf->const_array(mfi); }
        const auto rho_q = charge_density.const_array(mfi);
        const auto te_nodal = electron_temperature.const_array(mfi);
        const auto b_cc = magnetic_cc.const_array(mfi);
        const auto flux_arr = face_flux_mf.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            const amrex::Real jx = j_x(i, j, k);
            const amrex::Real jy = j_y(i, j, k);
            const amrex::Real jz = 0.5_rt * (j_z(i - 1, j, k) + j_z(i, j, k));
            const amrex::Real current_magnitude =
                std::sqrt(jx * jx + jy * jy + jz * jz);
            const amrex::Real charge_density_value =
                std::max(rho_q(i, j, k), charge_density_floor);
            const amrex::Real resistivity =
                theta_implicit_mhd::vacuum_keyed_resistivity(
                    eta(charge_density_value, te_nodal(i, j, k),
                        current_magnitude, time),
                    rho_q(i, j, k), vacuum_reference_charge_density,
                    vacuum_division_guard, vacuum_eta_scale);
            electric_x(i, j, k) =
                flux_arr(i, j, k, flux_induction_t2) + resistivity * jx;
            electric_y(i, j, k) =
                -flux_arr(i, j, k, flux_induction_t1) + resistivity * jy;
            if (include_hall) {
                // Hall EMF E += (J x B)/rho_q at the theta stage (see the
                // staging note at the register captures): the face
                // Riemann induction flux keeps the ION velocity, so this
                // edge term completes E = -u_e x B exactly. Same
                // staggered current interpolations as the |J|
                // regularization, same cell-centered B averaging as the
                // eta_H evaluation (total field, external included), and
                // a C-infinity smooth floor on the charge-density
                // division at the Ohm guard (no hard state branches on
                // the Jacobian probe path).
                const amrex::Real jhx = jh_x(i, j, k);
                const amrex::Real jhy = jh_y(i, j, k);
                const amrex::Real jhz =
                    0.5_rt * (jh_z(i - 1, j, k) + jh_z(i, j, k));
                const amrex::Real bx =
                    0.5_rt * (b_cc(i - 1, j, k, 0) + b_cc(i, j, k, 0));
                const amrex::Real by =
                    0.5_rt * (b_cc(i - 1, j, k, 1) + b_cc(i, j, k, 1));
                const amrex::Real bz =
                    0.5_rt * (b_cc(i - 1, j, k, 2) + b_cc(i, j, k, 2));
                const amrex::Real hall_charge_density =
                    theta_implicit_mhd::smooth_positive_floor(
                        rho_q(i, j, k), charge_density_floor);
                electric_x(i, j, k) +=
                    (jhy * bz - jhz * by) / hall_charge_density;
                electric_y(i, j, k) +=
                    (jhz * bx - jhx * bz) / hall_charge_density;
            }
            if (include_inertia) {
                // Electron-inertia field, NATIVE on the transverse z-node
                // staggering (the nodal assembly grid is the Ex/Ey grid in
                // 1D, so the emitted PC rows carry the exact frozen
                // response).
                electric_x(i, j, k) += ei_nodal(i, j, k, 0);
                electric_y(i, j, k) += ei_nodal(i, j, k, 1);
            }
            if (include_hyper_resistivity) {
                // E -= eta_H laplacian(J), same operator as the hybrid
                // solver; the floored charge density keeps the eta_H
                // parametrization smooth for matrix-free Jacobian probes.
                amrex::Real magnetic_magnitude = 0.0_rt;
                if (hyper_resistivity_needs_b) {
                    const amrex::Real bx =
                        0.5_rt * (b_cc(i - 1, j, k, 0) + b_cc(i, j, k, 0));
                    const amrex::Real by =
                        0.5_rt * (b_cc(i - 1, j, k, 1) + b_cc(i, j, k, 1));
                    const amrex::Real bz =
                        0.5_rt * (b_cc(i - 1, j, k, 2) + b_cc(i, j, k, 2));
                    magnetic_magnitude =
                        std::sqrt(bx * bx + by * by + bz * bz);
                }
                const amrex::Real hyper_resistivity =
                    eta_h(charge_density_value, magnetic_magnitude);
                const amrex::Real laplacian_jx =
                    (j_x(i - 1, j, k) - 2.0_rt * jx + j_x(i + 1, j, k)) *
                    inverse_dz2;
                const amrex::Real laplacian_jy =
                    (j_y(i - 1, j, k) - 2.0_rt * jy + j_y(i + 1, j, k)) *
                    inverse_dz2;
                electric_x(i, j, k) -= hyper_resistivity * laplacian_jx;
                electric_y(i, j, k) -= hyper_resistivity * laplacian_jy;
            }
        });
    }

    // Ez (cell centered): purely diagnostic in 1D (no curl contribution).
    for (amrex::MFIter mfi(electric_field_z); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto electric_z = electric_field_z.array(mfi);
        const auto j_x = current_x.const_array(mfi);
        const auto j_y = current_y.const_array(mfi);
        const auto j_z = current_z.const_array(mfi);
        const auto jh_x = hall_current_x.const_array(mfi);
        const auto jh_y = hall_current_y.const_array(mfi);
        amrex::Array4<const amrex::Real> ei_nodal;
        if (inertia_nodal_mf) { ei_nodal = inertia_nodal_mf->const_array(mfi); }
        const auto rho_q = charge_density.const_array(mfi);
        const auto te_nodal = electron_temperature.const_array(mfi);
        const auto rho = density.const_array(mfi);
        const auto mom = momentum.const_array(mfi);
        const auto b_cc = magnetic_cc.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            const amrex::Real jx = 0.5_rt * (j_x(i, j, k) + j_x(i + 1, j, k));
            const amrex::Real jy = 0.5_rt * (j_y(i, j, k) + j_y(i + 1, j, k));
            const amrex::Real jz = j_z(i, j, k);
            const amrex::Real current_magnitude =
                std::sqrt(jx * jx + jy * jy + jz * jz);
            const amrex::Real charge_density_raw =
                0.5_rt * (rho_q(i, j, k) + rho_q(i + 1, j, k));
            const amrex::Real charge_density_value =
                std::max(charge_density_raw, charge_density_floor);
            const amrex::Real temperature_e =
                0.5_rt * (te_nodal(i, j, k) + te_nodal(i + 1, j, k));
            const amrex::Real resistivity =
                theta_implicit_mhd::vacuum_keyed_resistivity(
                    eta(charge_density_value, temperature_e,
                        current_magnitude, time),
                    charge_density_raw, vacuum_reference_charge_density,
                    vacuum_division_guard, vacuum_eta_scale);
            const amrex::Real safe_density =
                std::max(rho(i, j, k), density_floor);
            const amrex::Real velocity_x = mom(i, j, k, 0) / safe_density;
            const amrex::Real velocity_y = mom(i, j, k, 1) / safe_density;
            electric_z(i, j, k) =
                -(velocity_x * b_cc(i, j, k, 1) -
                  velocity_y * b_cc(i, j, k, 0)) +
                resistivity * jz;
            if (include_hall) {
                // Hall EMF (diagnostic in 1D like the ideal part above):
                // theta-stage plasma current, native cell-centered B,
                // smooth-floored charge density (see the Ex/Ey kernel).
                const amrex::Real jhx =
                    0.5_rt * (jh_x(i, j, k) + jh_x(i + 1, j, k));
                const amrex::Real jhy =
                    0.5_rt * (jh_y(i, j, k) + jh_y(i + 1, j, k));
                const amrex::Real hall_charge_density =
                    theta_implicit_mhd::smooth_positive_floor(
                        charge_density_raw, charge_density_floor);
                electric_z(i, j, k) +=
                    (jhx * b_cc(i, j, k, 1) - jhy * b_cc(i, j, k, 0)) /
                    hall_charge_density;
            }
            if (include_inertia) {
                // Diagnostic in 1D like the ideal part above: the
                // cell-centered Ez averages the two adjacent nodes (the
                // rho_q stencil of this kernel).
                electric_z(i, j, k) += 0.5_rt *
                    (ei_nodal(i, j, k, 2) + ei_nodal(i + 1, j, k, 2));
            }
            if (include_hyper_resistivity) {
                amrex::Real magnetic_magnitude = 0.0_rt;
                if (hyper_resistivity_needs_b) {
                    const amrex::Real bx = b_cc(i, j, k, 0);
                    const amrex::Real by = b_cc(i, j, k, 1);
                    const amrex::Real bz = b_cc(i, j, k, 2);
                    magnetic_magnitude =
                        std::sqrt(bx * bx + by * by + bz * bz);
                }
                const amrex::Real hyper_resistivity =
                    eta_h(charge_density_value, magnetic_magnitude);
                const amrex::Real laplacian_jz =
                    (j_z(i - 1, j, k) - 2.0_rt * jz + j_z(i + 1, j, k)) *
                    inverse_dz2;
                electric_z(i, j, k) -= hyper_resistivity * laplacian_jz;
            }
        });
    }

    if (m_hybrid_pic_model->m_add_external_fields) {
        for (int direction = 0; direction < 3; ++direction) {
            amrex::MultiFab& electric_field = *m_WarpX->m_fields.get(
                FieldType::Efield_fp, Direction{direction}, 0);
            const amrex::MultiFab& electric_field_external =
                *m_WarpX->m_fields.get(FieldType::hybrid_E_fp_external,
                                       Direction{direction}, 0);
            amrex::MultiFab::Subtract(electric_field,
                                      electric_field_external, 0, 0, 1, 0);
        }
    }

    for (int direction = 0; direction < 3; ++direction) {
        m_WarpX->m_fields
            .get(FieldType::Efield_fp, Direction{direction}, 0)
            ->FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    }
#elif defined(WARPX_DIM_RZ)
    using ablastr::fields::Direction;
    // Resistive/Hall-MHD Ohm's law at the native RZ (m = 0) Yee
    // staggering. Er and Ez are DIRECT Riemann induction fluxes: Er lives
    // on z-faces (F_{B_theta} = +EMF_r there) and Ez on r-faces
    // (F_{B_theta} = -EMF_z there), so the Btheta update
    // dBtheta/dt = dEz/dr - dEr/dz is exactly the conservative difference
    // of the Btheta face fluxes. Etheta lives on cell corners and is
    // assembled with the smoothed UCT-HLL (Londrillo--Del Zanna) corner
    // EMF: an upwind-weighted average of the four adjacent cell-centered
    // ideal EMFs plus dissipation on the native staggered Br (z-jump) and
    // Bz (r-jump), with alpha weights from the stored face signal bounds.
    // Etheta(axis) = 0 by m = 0 parity. Under the split-field external-A
    // scheme the evolved B is the plasma response, so E_ext is
    // subtracted at the end.
    amrex::MultiFab& electric_field_r =
        *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, 0);
    amrex::MultiFab& electric_field_theta =
        *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{1}, 0);
    amrex::MultiFab& electric_field_z =
        *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{2}, 0);
    // Dissipative terms read the resistive-stage current when the shifted
    // centering is active (see PrepareResistiveStageCurrents); everything
    // current-related in this assembly is dissipative, so the swap covers
    // eta J, the vacuum boost's |J| argument, and the eta_H stencils.
    const bool resistive_stage =
        PrepareResistiveStageCurrents(at_resistive_stage);
    const amrex::MultiFab& current_r =
        *(resistive_stage
              ? m_WarpX->m_fields.get(ResistiveStageCurrentName, Direction{0}, 0)
              : m_WarpX->m_fields.get(FieldType::hybrid_current_fp_plasma,
                                      Direction{0}, 0));
    const amrex::MultiFab& current_theta =
        *(resistive_stage
              ? m_WarpX->m_fields.get(ResistiveStageCurrentName, Direction{1}, 0)
              : m_WarpX->m_fields.get(FieldType::hybrid_current_fp_plasma,
                                      Direction{1}, 0));
    const amrex::MultiFab& current_z =
        *(resistive_stage
              ? m_WarpX->m_fields.get(ResistiveStageCurrentName, Direction{2}, 0)
              : m_WarpX->m_fields.get(FieldType::hybrid_current_fp_plasma,
                                      Direction{2}, 0));
    // The Hall EMF is REACTIVE (it rotates B without dissipating energy),
    // so it keeps the global theta staging of the ideal EMF: it reads the
    // theta-stage plasma current directly, never the resistive-stage
    // extrapolation above (that shift exists to over-center the
    // DISSIPATIVE channels; applied to the whistler rotation it would add
    // first-order numerical damping exactly where the theta method is
    // chosen for its energy-conserving centering).
    const bool include_hall = m_hybrid_pic_model->m_include_hall_term;
    const amrex::MultiFab& hall_current_r = *m_WarpX->m_fields.get(
        FieldType::hybrid_current_fp_plasma, Direction{0}, 0);
    const amrex::MultiFab& hall_current_theta = *m_WarpX->m_fields.get(
        FieldType::hybrid_current_fp_plasma, Direction{1}, 0);
    const amrex::MultiFab& hall_current_z = *m_WarpX->m_fields.get(
        FieldType::hybrid_current_fp_plasma, Direction{2}, 0);
    // Electron inertia: the nodal (corner) inertial field assembled by
    // ComputeElectronInertiaNodal from the SAME theta-stage state
    // (reactive like the Hall EMF: never the resistive-stage
    // extrapolation, never the conduction_theta shift).
    const bool include_inertia =
        m_hybrid_pic_model->m_include_electron_inertia;
    const amrex::MultiFab* const inertia_nodal_mf = include_inertia
        ? m_WarpX->m_fields.get("hybrid_E_inertial_nodal", 0) : nullptr;
    const amrex::MultiFab& charge_density =
        *m_WarpX->m_fields.get(FieldType::rho_fp, 0);
    // Te [K] of the (rho, Te, J, t) parser: the temperature-primary nodal
    // register (same staggering as rho_fp), read/averaged with exactly the
    // rho_q stencils at every eta evaluation point of this assembly.
    const amrex::MultiFab& electron_temperature =
        *m_WarpX->m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
    const amrex::MultiFab& density = *m_WarpX->m_fields.get(MassDensityName, 0);
    const amrex::MultiFab& momentum =
        *m_WarpX->m_fields.get(MomentumDensityName, 0);
    const amrex::MultiFab& magnetic_cc =
        *m_WarpX->m_fields.get(MagneticFieldCCName, 0);
    const amrex::MultiFab& face_flux_r =
        *m_WarpX->m_fields.get(FaceFluxRName, 0);
    const amrex::MultiFab& face_flux_z =
        *m_WarpX->m_fields.get(FaceFluxZName, 0);
    const amrex::MultiFab& magnetic_r =
        *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, 0);
    const amrex::MultiFab& magnetic_z =
        *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{2}, 0);
    const bool add_external = m_hybrid_pic_model->m_add_external_fields;
    const amrex::MultiFab* const external_r =
        add_external ? m_WarpX->m_fields.get(FieldType::hybrid_B_fp_external,
                                             Direction{0}, 0)
                     : nullptr;
    const amrex::MultiFab* const external_z =
        add_external ? m_WarpX->m_fields.get(FieldType::hybrid_B_fp_external,
                                             Direction{2}, 0)
                     : nullptr;

    const auto eta = m_hybrid_pic_model->m_eta;
    const auto eta_h = m_hybrid_pic_model->m_eta_h;
    const bool include_hyper_resistivity =
        m_hybrid_pic_model->m_include_hyper_resistivity_term;
    const bool hyper_resistivity_needs_b =
        m_hybrid_pic_model->m_hyper_resistivity_has_B_dependence;
    // Ohm's-law division guards: the eta inputs and the -(u x B) velocity
    // reconstruction floor at the hybrid n_floor (mass-density equivalent),
    // independent of the fluid admissibility floor. In particular the dust
    // regime relies on this: frozen dust momentum over the Ohm guard keeps
    // u ~ 0 there, so E relaxes to eta J in the vacuum region.
    const amrex::Real density_floor = OhmMassDensityFloor();
    const amrex::Real charge_density_floor =
        m_ion_charge_to_mass * density_floor;
    // Density-keyed vacuum resistivity of the FIELD advance (see
    // vacuum_keyed_resistivity in ThetaImplicitMHD_K.H): keyed to the
    // vacuum reference (the Ohm guard, raised to the per-step frozen
    // dynamic reference under vacuum_reference_peak_fraction), divisions
    // guarded at the (far lower) positivity floor so the boost stays
    // uncapped over the reachable density range. Joule heating keeps the
    // un-boosted user eta.
    const amrex::Real vacuum_eta_scale =
        PhysConst::mu0 * m_vacuum_resistivity_diffusivity;
    const amrex::Real vacuum_division_guard =
        m_ion_charge_to_mass * m_mass_density_floor;
    const amrex::Real vacuum_reference_charge_density =
        m_ion_charge_to_mass * VacuumReferenceMassDensity();
    const amrex::Real kappa_signal = m_hlld_kappa_signal;
    const amrex::Real kappa_denominator = m_hlld_kappa_denominator;
    const amrex::Real radial_lower = m_WarpX->Geom(0).ProbLo(0);
    const amrex::Real radial_cell_size = m_WarpX->Geom(0).CellSize(0);
    const amrex::Real axial_cell_size = m_WarpX->Geom(0).CellSize(1);
    const amrex::Real inverse_dr2 =
        1.0_rt / (radial_cell_size * radial_cell_size);
    const amrex::Real inverse_dz2 =
        1.0_rt / (axial_cell_size * axial_cell_size);
    const int domain_lo_r = m_WarpX->Geom(0).Domain().smallEnd(0);
    // Wall-seam guard on the non-ideal Ohm edge terms (see
    // ImplicitMHDWallMask): at live rows whose Hall/inertia/hyper
    // stencils reach into the masked band those stencils ingest the
    // conducting wall's SURFACE current (curl B across the stair seam is
    // drive-scale, not plasma) and masked-band field values over the
    // Ohm-floored near-wall density -- a spurious drive-powered EMF pump
    // at the seam. The three terms are zeroed at seam-adjacent rows (the
    // ideal and eta J parts are untouched: electron-frame boundary-layer
    // physics at a rigid conductor is below grid, so ideal + resistive is
    // the correct wall-adjacent Ohm contract). The tables are
    // geometry-static (JFNK probes see constant structure) and the
    // preconditioner's stencil emission drops the same contributions
    // through the same tables. Null when no shaped wall is active AND
    // under wall_model = dielectric (bit-identical to no guard): the
    // EM-transparent standoff pins nothing, so the stencils see a
    // continuous live field and there is no drive-scale surface current
    // to guard against -- the seam-contamination mechanism was specific
    // to the pinned-response conductor contracts.
    const warpx::mhd_pc::WallMaskView wall_seam_view = m_wall_mask.View();
    const int* const seam_guard_er = wall_seam_view.first_guarded_er;
    const int* const seam_guard_et = wall_seam_view.first_guarded_et;
    const int* const seam_guard_ez = wall_seam_view.first_guarded_ez;
    // Wall-band resistivity override (implicit_mhd.wall_band_eta_override,
    // see ImplicitMHDWallMask): the field-advance eta is REPLACED by
    // the constant override at band-INTERIOR rows -- locations whose
    // complete cell-centered interpolation neighborhood is masked, so
    // the stair-interface E rows (which bound live cells) keep the
    // composed physical eta -- making the frozen band current-free by
    // construction (mu0 L^2 / eta_band << the drive time; no
    // intermediate-eta shell) with dEta/dState = 0 there (the
    // current-keyed anomalous terms explode at the clamped floor
    // density; their Jacobian stiffness froze the production arm's
    // Newton). The tables are geometry-static and the override is a
    // constant, so JFNK probes see constant band rows; the
    // preconditioner ingests the identically overridden eta through
    // the solver-side PC fills (GetMHDFieldResistivityCC/EdgeForPC),
    // keeping residual and PC exact twins. Null tables when inactive
    // (bit-identical to no override). Joule heating keeps the composed
    // eta (masked cells receive no fluid increments anyway).
    const WallBandEtaOverrideView wall_band_view =
        m_wall_mask.BandEtaOverrideView();
    const int* const band_override_er = wall_band_view.first_band_er;
    const int* const band_override_et = wall_band_view.first_band_et;
    const int* const band_override_ez = wall_band_view.first_band_ez;
    const amrex::Real band_eta_override = wall_band_view.eta_override;
    constexpr int flux_induction_t1 = FaceFluxComponent::induction_t1;
    constexpr int flux_induction_t2 = FaceFluxComponent::induction_t2;
    constexpr int flux_signal_left = FaceFluxComponent::signal_left;
    constexpr int flux_signal_right = FaceFluxComponent::signal_right;
    constexpr int flux_alfven_left = FaceFluxComponent::signal_alfven_left;
    constexpr int flux_alfven_right = FaceFluxComponent::signal_alfven_right;

    // Er on z-faces (r-cc, z-nodal): direct induction flux + eta J_r.
    for (amrex::MFIter mfi(electric_field_r); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto electric_r = electric_field_r.array(mfi);
        const auto zface = face_flux_z.const_array(mfi);
        const auto j_r = current_r.const_array(mfi);
        const auto j_theta = current_theta.const_array(mfi);
        const auto j_z = current_z.const_array(mfi);
        const auto jh_theta = hall_current_theta.const_array(mfi);
        const auto jh_z = hall_current_z.const_array(mfi);
        amrex::Array4<const amrex::Real> ei_nodal;
        if (inertia_nodal_mf) { ei_nodal = inertia_nodal_mf->const_array(mfi); }
        const auto rho_q = charge_density.const_array(mfi);
        const auto te_nodal = electron_temperature.const_array(mfi);
        const auto b_cc = magnetic_cc.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            const amrex::Real jr = j_r(i, j, k);
            // J_theta is corner-staggered (r-nodal, z-nodal).
            const amrex::Real jt =
                0.5_rt * (j_theta(i, j, k) + j_theta(i + 1, j, k));
            const amrex::Real jz =
                0.25_rt * (j_z(i, j, k) + j_z(i + 1, j, k) +
                           j_z(i, j - 1, k) + j_z(i + 1, j - 1, k));
            const amrex::Real current_magnitude =
                std::sqrt(jr * jr + jt * jt + jz * jz);
            const amrex::Real charge_density_raw =
                0.5_rt * (rho_q(i, j, k) + rho_q(i + 1, j, k));
            const amrex::Real charge_density_value =
                std::max(charge_density_raw, charge_density_floor);
            const amrex::Real temperature_e =
                0.5_rt * (te_nodal(i, j, k) + te_nodal(i + 1, j, k));
            amrex::Real resistivity =
                theta_implicit_mhd::vacuum_keyed_resistivity(
                    eta(charge_density_value, temperature_e,
                        current_magnitude, time),
                    charge_density_raw, vacuum_reference_charge_density,
                    vacuum_division_guard, vacuum_eta_scale);
            if (band_override_er != nullptr && i >= band_override_er[j]) {
                // Wall-band eta override (see the capture comment):
                // REPLACES the composed eta, dEta/dState = 0.
                resistivity = band_eta_override;
            }
            electric_r(i, j, k) =
                zface(i, j, k, flux_induction_t2) + resistivity * jr;
            // Wall-seam guard (see the capture comment): the Hall,
            // electron-inertia and hyper-resistive edge terms are zeroed
            // at rows whose stencils reach into the masked band.
            const bool seam_guarded =
                (seam_guard_er != nullptr) && (i >= seam_guard_er[j]);
            if (include_hall && !seam_guarded) {
                // Hall EMF E_r += (J_theta B_z - J_z B_theta)/rho_q at the
                // theta stage (see the staging note at the register
                // captures): the face Riemann induction flux keeps the
                // ION velocity, so the edge term completes E = -u_e x B
                // exactly. Same staggered current interpolations as the
                // |J| regularization above, the eta_H z-face averaging of
                // the cell-centered TOTAL B, and a C-infinity smooth
                // floor on the charge-density division at the Ohm guard.
                const amrex::Real jht =
                    0.5_rt * (jh_theta(i, j, k) + jh_theta(i + 1, j, k));
                const amrex::Real jhz =
                    0.25_rt * (jh_z(i, j, k) + jh_z(i + 1, j, k) +
                               jh_z(i, j - 1, k) + jh_z(i + 1, j - 1, k));
                const amrex::Real bt =
                    0.5_rt * (b_cc(i, j - 1, k, 1) + b_cc(i, j, k, 1));
                const amrex::Real bz =
                    0.5_rt * (b_cc(i, j - 1, k, 2) + b_cc(i, j, k, 2));
                const amrex::Real hall_charge_density =
                    theta_implicit_mhd::smooth_positive_floor(
                        charge_density_raw, charge_density_floor);
                electric_r(i, j, k) +=
                    (jht * bz - jhz * bt) / hall_charge_density;
            }
            if (include_inertia && !seam_guarded) {
                // Electron-inertia field: the corner (nodal) assembly
                // averaged in r to the z-face (the rho_q stencil of this
                // kernel).
                electric_r(i, j, k) += 0.5_rt *
                    (ei_nodal(i, j, k, 0) + ei_nodal(i + 1, j, k, 0));
            }
            if (include_hyper_resistivity && !seam_guarded) {
                // E_r -= eta_H (laplacian J)_r with the m = 0 cylindrical
                // vector Laplacian, discretely matching the hybrid solver;
                // r is cell centered here, so no axis special case.
                amrex::Real magnetic_magnitude = 0.0_rt;
                if (hyper_resistivity_needs_b) {
                    const amrex::Real br =
                        0.5_rt * (b_cc(i, j - 1, k, 0) + b_cc(i, j, k, 0));
                    const amrex::Real bt =
                        0.5_rt * (b_cc(i, j - 1, k, 1) + b_cc(i, j, k, 1));
                    const amrex::Real bz =
                        0.5_rt * (b_cc(i, j - 1, k, 2) + b_cc(i, j, k, 2));
                    magnetic_magnitude =
                        std::sqrt(br * br + bt * bt + bz * bz);
                }
                const amrex::Real hyper_resistivity =
                    eta_h(charge_density_value, magnetic_magnitude);
                const amrex::Real radius =
                    radial_lower + (i + 0.5_rt) * radial_cell_size;
                const amrex::Real laplacian_jr =
                    ((radius + 0.5_rt * radial_cell_size) *
                         (j_r(i + 1, j, k) - jr) -
                     (radius - 0.5_rt * radial_cell_size) *
                         (jr - j_r(i - 1, j, k))) *
                        inverse_dr2 / radius +
                    (j_r(i, j - 1, k) - 2.0_rt * jr + j_r(i, j + 1, k)) *
                        inverse_dz2 -
                    jr / (radius * radius);
                electric_r(i, j, k) -= hyper_resistivity * laplacian_jr;
            }
        });
    }

    // Ez on r-faces (r-nodal, z-cc): direct induction flux + eta J_z. At
    // the axis the face-flux row is zero by construction, leaving the
    // parity-exact Ez(axis) = eta J_z.
    for (amrex::MFIter mfi(electric_field_z); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto electric_z = electric_field_z.array(mfi);
        const auto rface = face_flux_r.const_array(mfi);
        const auto j_r = current_r.const_array(mfi);
        const auto j_theta = current_theta.const_array(mfi);
        const auto j_z = current_z.const_array(mfi);
        const auto jh_r = hall_current_r.const_array(mfi);
        const auto jh_theta = hall_current_theta.const_array(mfi);
        amrex::Array4<const amrex::Real> ei_nodal;
        if (inertia_nodal_mf) { ei_nodal = inertia_nodal_mf->const_array(mfi); }
        const auto rho_q = charge_density.const_array(mfi);
        const auto te_nodal = electron_temperature.const_array(mfi);
        const auto b_cc = magnetic_cc.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // r-index of the lower cell column; clamped at the physical
            // axis where no below-axis staggered current exists (the
            // clamp only affects the |J| regularization argument of eta).
            const int il = (i - 1 < domain_lo_r) ? i : i - 1;
            const amrex::Real jr =
                0.25_rt * (j_r(il, j, k) + j_r(i, j, k) +
                           j_r(il, j + 1, k) + j_r(i, j + 1, k));
            // J_theta is corner-staggered (r-nodal, z-nodal).
            const amrex::Real jt =
                0.5_rt * (j_theta(i, j, k) + j_theta(i, j + 1, k));
            const amrex::Real jz = j_z(i, j, k);
            const amrex::Real current_magnitude =
                std::sqrt(jr * jr + jt * jt + jz * jz);
            const amrex::Real charge_density_raw =
                0.5_rt * (rho_q(i, j, k) + rho_q(i, j + 1, k));
            const amrex::Real charge_density_value =
                std::max(charge_density_raw, charge_density_floor);
            const amrex::Real temperature_e =
                0.5_rt * (te_nodal(i, j, k) + te_nodal(i, j + 1, k));
            amrex::Real resistivity =
                theta_implicit_mhd::vacuum_keyed_resistivity(
                    eta(charge_density_value, temperature_e,
                        current_magnitude, time),
                    charge_density_raw, vacuum_reference_charge_density,
                    vacuum_division_guard, vacuum_eta_scale);
            if (band_override_ez != nullptr && i >= band_override_ez[j]) {
                // Wall-band eta override (see the capture comment):
                // REPLACES the composed eta, dEta/dState = 0.
                resistivity = band_eta_override;
            }
            electric_z(i, j, k) =
                -rface(i, j, k, flux_induction_t1) + resistivity * jz;
            // Wall-seam guard (see the capture comment).
            const bool seam_guarded =
                (seam_guard_ez != nullptr) && (i >= seam_guard_ez[j]);
            if (include_hall && !seam_guarded) {
                // Hall EMF E_z += (J_r B_theta - J_theta B_r)/rho_q at
                // the theta stage (see the Er kernel). The eta_H r-face
                // averaging of the cell-centered TOTAL B carries the
                // below-axis parity ghosts, so B_theta and B_r vanish
                // EXACTLY on the axis face and the parity-exact
                // Ez(axis) = eta J_z of the recast is preserved (the
                // clamped J_r average below is multiplied by that exact
                // zero there).
                const amrex::Real jhr =
                    0.25_rt * (jh_r(il, j, k) + jh_r(i, j, k) +
                               jh_r(il, j + 1, k) + jh_r(i, j + 1, k));
                const amrex::Real jht =
                    0.5_rt * (jh_theta(i, j, k) + jh_theta(i, j + 1, k));
                const amrex::Real br =
                    0.5_rt * (b_cc(i - 1, j, k, 0) + b_cc(i, j, k, 0));
                const amrex::Real bt =
                    0.5_rt * (b_cc(i - 1, j, k, 1) + b_cc(i, j, k, 1));
                const amrex::Real hall_charge_density =
                    theta_implicit_mhd::smooth_positive_floor(
                        charge_density_raw, charge_density_floor);
                electric_z(i, j, k) +=
                    (jhr * bt - jht * br) / hall_charge_density;
            }
            if (include_inertia && !seam_guarded) {
                // Electron-inertia field: the corner (nodal) assembly
                // averaged in z to the r-face (the rho_q stencil of this
                // kernel).
                electric_z(i, j, k) += 0.5_rt *
                    (ei_nodal(i, j, k, 2) + ei_nodal(i, j + 1, k, 2));
            }
            if (include_hyper_resistivity && !seam_guarded) {
                // E_z -= eta_H (laplacian J)_z. On axis the geometric
                // radial part reduces to Drr by the m = 0 symmetry of
                // J_z (same treatment as the hybrid solver); the b_cc
                // below-axis radial ghosts carry the axis parities.
                amrex::Real magnetic_magnitude = 0.0_rt;
                if (hyper_resistivity_needs_b) {
                    const amrex::Real br =
                        0.5_rt * (b_cc(i - 1, j, k, 0) + b_cc(i, j, k, 0));
                    const amrex::Real bt =
                        0.5_rt * (b_cc(i - 1, j, k, 1) + b_cc(i, j, k, 1));
                    const amrex::Real bz =
                        0.5_rt * (b_cc(i - 1, j, k, 2) + b_cc(i, j, k, 2));
                    magnetic_magnitude =
                        std::sqrt(br * br + bt * bt + bz * bz);
                }
                const amrex::Real hyper_resistivity =
                    eta_h(charge_density_value, magnetic_magnitude);
                const amrex::Real radius =
                    radial_lower + i * radial_cell_size;
                amrex::Real laplacian_jz =
                    (j_z(i, j - 1, k) - 2.0_rt * jz + j_z(i, j + 1, k)) *
                    inverse_dz2;
                if (radius > 0.5_rt * radial_cell_size) {
                    laplacian_jz +=
                        ((radius + 0.5_rt * radial_cell_size) *
                             (j_z(i + 1, j, k) - jz) -
                         (radius - 0.5_rt * radial_cell_size) *
                             (jz - j_z(i - 1, j, k))) *
                        inverse_dr2 / radius;
                } else {
                    laplacian_jz +=
                        (j_z(i - 1, j, k) - 2.0_rt * jz + j_z(i + 1, j, k)) *
                        inverse_dr2;
                }
                electric_z(i, j, k) -= hyper_resistivity * laplacian_jz;
            }
        });
    }

    // Etheta on corners: smoothed UCT-HLL + eta J_theta; zero on axis.
    for (amrex::MFIter mfi(electric_field_theta); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto electric_theta = electric_field_theta.array(mfi);
        const auto rface = face_flux_r.const_array(mfi);
        const auto zface = face_flux_z.const_array(mfi);
        const auto rho = density.const_array(mfi);
        const auto mom = momentum.const_array(mfi);
        const auto b_cc = magnetic_cc.const_array(mfi);
        const auto br_stag = magnetic_r.const_array(mfi);
        const auto bz_stag = magnetic_z.const_array(mfi);
        const auto br_ext = add_external ? external_r->const_array(mfi)
                                         : amrex::Array4<const amrex::Real>{};
        const auto bz_ext = add_external ? external_z->const_array(mfi)
                                         : amrex::Array4<const amrex::Real>{};
        const auto j_r = current_r.const_array(mfi);
        const auto j_theta = current_theta.const_array(mfi);
        const auto j_z = current_z.const_array(mfi);
        const auto jh_r = hall_current_r.const_array(mfi);
        const auto jh_z = hall_current_z.const_array(mfi);
        amrex::Array4<const amrex::Real> ei_nodal;
        if (inertia_nodal_mf) { ei_nodal = inertia_nodal_mf->const_array(mfi); }
        const auto rho_q = charge_density.const_array(mfi);
        const auto te_nodal = electron_temperature.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            const amrex::Real corner_radius =
                radial_lower + i * radial_cell_size;
            if (corner_radius == 0.0_rt) {
                electric_theta(i, j, k) = 0.0_rt;
                return;
            }
            // Smoothed one-sided signal magnitudes, combined over the two
            // adjacent faces of each family: alpha+ from the right-going
            // bound, alpha- from the left-going bound. Evaluated for two
            // channel pairs: the fast Davis bounds set the upwind WEIGHTS
            // of the four-state average (their span is bounded below by
            // the fast scale, so the weights are always well defined),
            // while the fan's rotational bounds S_M -+ c_An set the
            // DISSIPATION coefficients (UCT-HLLD): tangential-field jumps
            // are carried by the rotational discontinuities, so a static
            // pressure-balanced equilibrium (S_M = 0, and c_An* -> 0 with
            // B_n) suffers no corner diffusion instead of being destroyed
            // at the fast-speed scale.
            const auto alpha_pair =
                [=] (const amrex::Array4<const amrex::Real>& faces,
                     const int i1, const int j1, const int i2, const int j2,
                     const int channel_left, const int channel_right,
                     amrex::Real& alpha_plus, amrex::Real& alpha_minus) {
                const amrex::Real sl1 = faces(i1, j1, k, channel_left);
                const amrex::Real sr1 = faces(i1, j1, k, channel_right);
                const amrex::Real sl2 = faces(i2, j2, k, channel_left);
                const amrex::Real sr2 = faces(i2, j2, k, channel_right);
                const amrex::Real w1 = kappa_signal * (sr1 - sl1);
                const amrex::Real w2 = kappa_signal * (sr2 - sl2);
                const amrex::Real w = 0.5_rt * (w1 + w2);
                const amrex::Real plus1 =
                    0.5_rt * (sr1 + std::sqrt(sr1 * sr1 + w1 * w1));
                const amrex::Real plus2 =
                    0.5_rt * (sr2 + std::sqrt(sr2 * sr2 + w2 * w2));
                const amrex::Real minus1 =
                    0.5_rt * (-sl1 + std::sqrt(sl1 * sl1 + w1 * w1));
                const amrex::Real minus2 =
                    0.5_rt * (-sl2 + std::sqrt(sl2 * sl2 + w2 * w2));
                alpha_plus = theta_implicit_mhd::smooth_max(plus1, plus2, w);
                alpha_minus =
                    theta_implicit_mhd::smooth_max(minus1, minus2, w);
            };
            amrex::Real alpha_r_plus;
            amrex::Real alpha_r_minus;
            alpha_pair(rface, i, j - 1, i, j, flux_signal_left,
                       flux_signal_right, alpha_r_plus, alpha_r_minus);
            amrex::Real alpha_z_plus;
            amrex::Real alpha_z_minus;
            alpha_pair(zface, i - 1, j, i, j, flux_signal_left,
                       flux_signal_right, alpha_z_plus, alpha_z_minus);
            amrex::Real alfven_r_plus;
            amrex::Real alfven_r_minus;
            alpha_pair(rface, i, j - 1, i, j, flux_alfven_left,
                       flux_alfven_right, alfven_r_plus, alfven_r_minus);
            amrex::Real alfven_z_plus;
            amrex::Real alfven_z_minus;
            alpha_pair(zface, i - 1, j, i, j, flux_alfven_left,
                       flux_alfven_right, alfven_z_plus, alfven_z_minus);

            // Cell-centered ideal EMF states of the four cells around the
            // corner: E_theta = -(u x B)_theta = u_r B_z - u_z B_r.
            const auto emf_state = [=] (const int ic, const int jc) {
                const amrex::Real safe_density =
                    std::max(rho(ic, jc, k), density_floor);
                const amrex::Real velocity_r =
                    mom(ic, jc, k, 0) / safe_density;
                const amrex::Real velocity_z =
                    mom(ic, jc, k, 2) / safe_density;
                return velocity_r * b_cc(ic, jc, k, 2) -
                       velocity_z * b_cc(ic, jc, k, 0);
            };
            const amrex::Real emf_mm = emf_state(i - 1, j - 1);
            const amrex::Real emf_mp = emf_state(i - 1, j);
            const amrex::Real emf_pm = emf_state(i, j - 1);
            const amrex::Real emf_pp = emf_state(i, j);

            const amrex::Real sum_r = alpha_r_plus + alpha_r_minus;
            const amrex::Real sum_z = alpha_z_plus + alpha_z_minus;
            const amrex::Real average =
                (alpha_r_plus * (alpha_z_plus * emf_mm +
                                 alpha_z_minus * emf_mp) +
                 alpha_r_minus * (alpha_z_plus * emf_pm +
                                  alpha_z_minus * emf_pp)) /
                (sum_r * sum_z);

            // Dissipation on the native staggered TOTAL fields: +z-jump
            // of Br (from the z-face Riemann limit of E_theta = -F_Br)
            // and -r-jump of Bz (from the r-face limit E_theta = +F_Bz).
            // The coefficients are built from the ROTATIONAL alphas (the
            // scale at which the face fan itself dissipates tangential
            // field), with the denominator anchored by the fast span so
            // the coefficient goes to zero smoothly (never 0/0) as the
            // rotational bounds collapse onto a static contact.
            amrex::Real br_low = br_stag(i, j - 1, k);
            amrex::Real br_high = br_stag(i, j, k);
            amrex::Real bz_low = bz_stag(i - 1, j, k);
            amrex::Real bz_high = bz_stag(i, j, k);
            if (add_external) {
                br_low += br_ext(i, j - 1, k);
                br_high += br_ext(i, j, k);
                bz_low += bz_ext(i - 1, j, k);
                bz_high += bz_ext(i, j, k);
            }
            const amrex::Real coefficient_z =
                alfven_z_plus * alfven_z_minus /
                (alfven_z_plus + alfven_z_minus +
                 kappa_denominator * sum_z);
            const amrex::Real coefficient_r =
                alfven_r_plus * alfven_r_minus /
                (alfven_r_plus + alfven_r_minus +
                 kappa_denominator * sum_r);
            const amrex::Real dissipation =
                coefficient_z * (br_high - br_low) -
                coefficient_r * (bz_high - bz_low);

            // J_theta is corner-staggered: native value, no interpolation.
            const amrex::Real jt_corner = j_theta(i, j, k);
            const amrex::Real jr_corner =
                0.5_rt * (j_r(i - 1, j, k) + j_r(i, j, k));
            const amrex::Real jz_corner =
                0.5_rt * (j_z(i, j - 1, k) + j_z(i, j, k));
            const amrex::Real current_magnitude =
                std::sqrt(jr_corner * jr_corner + jt_corner * jt_corner +
                          jz_corner * jz_corner);
            const amrex::Real charge_density_value =
                std::max(rho_q(i, j, k), charge_density_floor);
            amrex::Real resistivity =
                theta_implicit_mhd::vacuum_keyed_resistivity(
                    eta(charge_density_value, te_nodal(i, j, k),
                        current_magnitude, time),
                    rho_q(i, j, k), vacuum_reference_charge_density,
                    vacuum_division_guard, vacuum_eta_scale);
            if (band_override_et != nullptr && i >= band_override_et[j]) {
                // Wall-band eta override (see the capture comment):
                // REPLACES the composed eta, dEta/dState = 0.
                resistivity = band_eta_override;
            }
            electric_theta(i, j, k) =
                average + dissipation + resistivity * jt_corner;
            // Wall-seam guard (see the capture comment): at the first
            // live corner rows inside a wall step corner the half-sum
            // corner currents and the four-cell b_cc average below read
            // straight across the stair seam -- the measured
            // drive-powered E_theta pump of the formation-section exit
            // step (rr14f).
            const bool seam_guarded =
                (seam_guard_et != nullptr) && (i >= seam_guard_et[j]);
            if (include_hall && !seam_guarded) {
                // Hall EMF E_theta += (J_z B_r - J_r B_z)/rho_q at the
                // theta stage (see the Er kernel): half-sum corner
                // currents (the |J| regularization's own stencils), the
                // eta_H four-cell corner averaging of the cell-centered
                // TOTAL B, the nodal charge density with the C-infinity
                // smooth Ohm floor. The axis corner returned exactly
                // zero above, preserving the m = 0 parity.
                const amrex::Real jhr =
                    0.5_rt * (jh_r(i - 1, j, k) + jh_r(i, j, k));
                const amrex::Real jhz =
                    0.5_rt * (jh_z(i, j - 1, k) + jh_z(i, j, k));
                const amrex::Real br = 0.25_rt *
                    (b_cc(i - 1, j - 1, k, 0) + b_cc(i, j - 1, k, 0) +
                     b_cc(i - 1, j, k, 0) + b_cc(i, j, k, 0));
                const amrex::Real bz = 0.25_rt *
                    (b_cc(i - 1, j - 1, k, 2) + b_cc(i, j - 1, k, 2) +
                     b_cc(i - 1, j, k, 2) + b_cc(i, j, k, 2));
                const amrex::Real hall_charge_density =
                    theta_implicit_mhd::smooth_positive_floor(
                        rho_q(i, j, k), charge_density_floor);
                electric_theta(i, j, k) +=
                    (jhz * br - jhr * bz) / hall_charge_density;
            }
            if (include_inertia && !seam_guarded) {
                // Electron-inertia field, NATIVE on the corner staggering
                // (the nodal assembly grid); the axis corner returned
                // exactly zero above, preserving the m = 0 parity.
                electric_theta(i, j, k) += ei_nodal(i, j, k, 1);
            }
            if (include_hyper_resistivity && !seam_guarded) {
                // E_theta -= eta_H (laplacian J)_theta at the corner
                // (J_theta's native staggering, so the stencil needs no
                // interpolation); the axis corner returned zero above.
                amrex::Real magnetic_magnitude = 0.0_rt;
                if (hyper_resistivity_needs_b) {
                    const amrex::Real br = 0.25_rt *
                        (b_cc(i - 1, j - 1, k, 0) + b_cc(i, j - 1, k, 0) +
                         b_cc(i - 1, j, k, 0) + b_cc(i, j, k, 0));
                    const amrex::Real bt = 0.25_rt *
                        (b_cc(i - 1, j - 1, k, 1) + b_cc(i, j - 1, k, 1) +
                         b_cc(i - 1, j, k, 1) + b_cc(i, j, k, 1));
                    const amrex::Real bz = 0.25_rt *
                        (b_cc(i - 1, j - 1, k, 2) + b_cc(i, j - 1, k, 2) +
                         b_cc(i - 1, j, k, 2) + b_cc(i, j, k, 2));
                    magnetic_magnitude =
                        std::sqrt(br * br + bt * bt + bz * bz);
                }
                const amrex::Real hyper_resistivity =
                    eta_h(charge_density_value, magnetic_magnitude);
                const amrex::Real laplacian_jt =
                    ((corner_radius + 0.5_rt * radial_cell_size) *
                         (j_theta(i + 1, j, k) - jt_corner) -
                     (corner_radius - 0.5_rt * radial_cell_size) *
                         (jt_corner - j_theta(i - 1, j, k))) *
                        inverse_dr2 / corner_radius +
                    (j_theta(i, j - 1, k) - 2.0_rt * jt_corner +
                     j_theta(i, j + 1, k)) *
                        inverse_dz2 -
                    jt_corner / (corner_radius * corner_radius);
                electric_theta(i, j, k) -=
                    hyper_resistivity * laplacian_jt;
            }
        });
    }

    if (add_external) {
        for (int direction = 0; direction < 3; ++direction) {
            amrex::MultiFab& electric_field = *m_WarpX->m_fields.get(
                FieldType::Efield_fp, Direction{direction}, 0);
            const amrex::MultiFab& electric_field_external =
                *m_WarpX->m_fields.get(FieldType::hybrid_E_fp_external,
                                       Direction{direction}, 0);
            amrex::MultiFab::Subtract(electric_field,
                                      electric_field_external, 0, 0, 1, 0);
        }
    }

    for (int direction = 0; direction < 3; ++direction) {
        m_WarpX->m_fields
            .get(FieldType::Efield_fp, Direction{direction}, 0)
            ->FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    }
    m_WarpX->ApplyEfieldBoundary(0, PatchType::fine, time);
    if (m_z_neumann) {
        for (int direction = 0; direction < 3; ++direction) {
            amrex::MultiFab& electric_component = *m_WarpX->m_fields.get(
                FieldType::Efield_fp, Direction{direction}, 0);
            ApplyNeumannZDomainGhosts(electric_component);
            if (direction == 2) {
                // z_lo mirror: E_z (cell-centered in z) is ODD; E_r and
                // E_theta are even nodal-in-z, exact under the Neumann
                // nodal mirror.
                ApplyMirrorZLoDomainGhosts(electric_component, {-1, -1, -1});
            }
        }
    }
    if (m_wall_mask.IsActive()) {
        // Stair-step conducting shaped wall: project the conductor
        // condition onto the assembled E LAST, so no other boundary fill
        // overwrites it. wall_model = pec acts on the TOTAL field --
        // under the split-field external-A scheme the plasma-response E
        // is set to -E_ext at every masked location (total tangential
        // E = 0), so the masked-cell Faraday update cancels the external
        // drive exactly and the TOTAL flux through every contour inside
        // the metal stays frozen at its initial value (the perfect-
        // conductor eddy response). wall_model = pec_response pins the
        // PLASMA-RESPONSE field only (E_plasma = 0 at masked locations,
        // the prescribed drive is transparent): the run32 EB parity
        // contract for fitted waveform composites that already embed the
        // machine's wall response. Both maps are affine in the state:
        // matrix-free Jacobian probes see exactly the zeroed rows the
        // preconditioner's stencil emission drops (the masked values are
        // state-independent constants in either contract). wall_model =
        // dielectric performs NOTHING inside this call (the
        // EM-transparent standoff pins no field; only its FLUID
        // contract is active), so the assembled E is identical to the
        // wall_model = none field operator.
        const ablastr::fields::VectorField efield =
            m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, 0);
        ablastr::fields::VectorField efield_external{};
        if (add_external) {
            efield_external = m_WarpX->m_fields.get_alldirs(
                FieldType::hybrid_E_fp_external, 0);
        }
        m_wall_mask.ProjectElectricField(
            efield, add_external ? &efield_external : nullptr,
            m_WarpX->Geom(0));
    }
#else
    amrex::ignore_unused(time, at_resistive_stage);
    WARPX_ABORT_WITH_MESSAGE(
        "ThetaImplicitMHD::AssembleOhmElectricField() requires 1D or RZ "
        "geometry");
#endif
}

void ThetaImplicitMHD::ComputeFluidRHSFromFaceFluxes (WarpXSolverVec& rhs,
                                                      const amrex::Real time) const
{
#if defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_RZ)
    const amrex::MultiFab& density = *m_WarpX->m_fields.get(MassDensityName, 0);
    const amrex::MultiFab& momentum =
        *m_WarpX->m_fields.get(MomentumDensityName, 0);
    const amrex::MultiFab& electron_energy =
        *m_WarpX->m_fields.get(ElectronEnergyName, 0);
    const amrex::MultiFab& ion_energy =
        *m_WarpX->m_fields.get(IonEnergyName, 0);
    const amrex::MultiFab& old_density =
        *m_WarpX->m_fields.get(OldMassDensityName, 0);
    const amrex::MultiFab& old_momentum =
        *m_WarpX->m_fields.get(OldMomentumDensityName, 0);
    const amrex::MultiFab& old_electron_energy =
        *m_WarpX->m_fields.get(OldElectronEnergyName, 0);
    const amrex::MultiFab& old_ion_energy =
        *m_WarpX->m_fields.get(OldIonEnergyName, 0);
    const amrex::MultiFab& ion_parallel_energy =
        *m_WarpX->m_fields.get(IonParallelEnergyName, 0);
    const amrex::MultiFab& ion_perp_energy =
        *m_WarpX->m_fields.get(IonPerpEnergyName, 0);
    const amrex::MultiFab& old_ion_parallel_energy =
        *m_WarpX->m_fields.get(OldIonParallelEnergyName, 0);
    const amrex::MultiFab& old_ion_perp_energy =
        *m_WarpX->m_fields.get(OldIonPerpEnergyName, 0);
    const amrex::MultiFab& ion_internal_energy =
        *m_WarpX->m_fields.get(IonInternalEnergyName, 0);
    const amrex::MultiFab& old_ion_internal_energy =
        *m_WarpX->m_fields.get(OldIonInternalEnergyName, 0);
    const amrex::MultiFab& current = *m_WarpX->m_fields.get(TotalCurrentCCName, 0);
    const amrex::MultiFab& magnetic_cc =
        *m_WarpX->m_fields.get(MagneticFieldCCName, 0);
    const amrex::MultiFab& face_flux_mf =
        *m_WarpX->m_fields.get(FaceFluxZName, 0);
#if defined(WARPX_DIM_RZ)
    const amrex::MultiFab& face_flux_r_mf =
        *m_WarpX->m_fields.get(FaceFluxRName, 0);
#endif

    const bool dual_energy_closure = m_ion_closure == "dual_energy";
    const bool total_energy_closure =
        m_ion_closure == "total_energy" || dual_energy_closure;
    const bool cgl_closure = m_ion_closure == "cgl";
    amrex::MultiFab& density_rhs = rhs.getMultiFabBlock(MassDensityName, 0);
    amrex::MultiFab& momentum_rhs = rhs.getMultiFabBlock(MomentumDensityName, 0);
    amrex::MultiFab& electron_energy_rhs =
        rhs.getMultiFabBlock(ElectronEnergyName, 0);
    amrex::MultiFab* const ion_energy_rhs =
        total_energy_closure ? &rhs.getMultiFabBlock(IonEnergyName, 0) : nullptr;
    amrex::MultiFab* const ion_parallel_energy_rhs =
        cgl_closure ? &rhs.getMultiFabBlock(IonParallelEnergyName, 0)
                    : nullptr;
    amrex::MultiFab* const ion_perp_energy_rhs =
        cgl_closure ? &rhs.getMultiFabBlock(IonPerpEnergyName, 0) : nullptr;
    amrex::MultiFab* const ion_internal_energy_rhs =
        dual_energy_closure
            ? &rhs.getMultiFabBlock(IonInternalEnergyName, 0)
            : nullptr;

    const auto inverse_cell_size = physical_inverse_cell_size(m_WarpX->Geom(0));
    const amrex::Real inverse_dz = inverse_cell_size[2];
    const amrex::Real theta_dt = m_theta * m_dt;
    const amrex::Real extrapolation_weight = (1.0_rt - m_theta) / m_theta;
    const amrex::Real density_floor = m_mass_density_floor;
    const amrex::Real charge_to_mass = m_ion_charge_to_mass;
    const amrex::Real gamma_e_minus_one = m_gamma_e - 1.0_rt;
    const amrex::Real pressure_floor = m_electron_pressure_floor;
    const amrex::Real ion_energy_floor =
        total_energy_closure ? m_ion_pressure_floor / (m_gamma_i - 1.0_rt)
                             : 0.0_rt;
    const bool evolve_ion_fluid = m_evolve_ion_fluid;
    const bool include_joule_heating = m_include_joule_heating;
    // Electron-ion equilibration (see m_electron_ion_equilibration and
    // the non-recast twin in ComputeFluidRHS -- the residual is one map
    // regardless of the flux path): per-solve-uniform branch, so the
    // OFF path performs no arithmetic at all. The rate coefficient
    // folds the reference code's 4.75e-15 constant, the m_p/m_i mass ratio, and
    // the user scale; the cap is the theta-scheme monotonicity bound of
    // the pair-difference decay (the reference code's backward Euler needs none).
    const bool ei_equilibration = m_electron_ion_equilibration > 0.0_rt;
    const amrex::Real ei_rate_coefficient =
        m_electron_ion_equilibration * 4.75e-15_rt * PhysConst::m_p *
        m_ion_charge_to_mass / PhysConst::q_e;
    const amrex::Real ei_rate_cap =
        m_theta < 1.0_rt
            ? 1.0_rt / ((m_gamma_e + m_gamma_i - 2.0_rt) *
                        (1.0_rt - m_theta) * m_dt)
            : std::numeric_limits<amrex::Real>::max();
    const amrex::Real work_kappa = m_hlld_kappa_signal;
    const amrex::Real electron_energy_floor_rate =
        m_electron_pressure_floor / (m_gamma_e - 1.0_rt) / theta_dt;
    const amrex::Real ion_energy_floor_rate = ion_energy_floor / theta_dt;
    // --- Dual-energy closure constants: U_i's internal floor is the
    // SAME internal image of the ion pressure floor as E_i's recovery
    // floor (ion_energy_floor above, which total_energy_closure keeps
    // set under dual). The viscous heating source deposits the
    // positive-definite face dissipation rho_f nu |du/dn|^2 half to each
    // adjacent cell -- the internal-only counterpart of the conservative
    // stress-work pair the E_i face channel carries (the reference code's step_wio
    // per-edge nu (dv)^2 deposit).
    const amrex::Real dual_ion_internal_floor =
        dual_energy_closure ? ion_energy_floor : 0.0_rt;
    const bool add_viscous_heating =
        dual_energy_closure && m_viscosity > 0.0_rt;
    const amrex::Real viscosity = m_viscosity;
    // --- CGL closure constants (all host-precomputed and strictly
    // positive, so every in-kernel regularization is C-infinity). ---
    const amrex::Real ion_pressure_floor = m_ion_pressure_floor;
    const amrex::Real ion_parallel_floor_rate =
        0.5_rt * m_ion_pressure_floor / theta_dt;
    const amrex::Real ion_perp_floor_rate = m_ion_pressure_floor / theta_dt;
    const amrex::Real ion_mass = PhysConst::q_e / m_ion_charge_to_mass;
    const amrex::Real inverse_ion_mass = 1.0_rt / ion_mass;
    const amrex::Real cgl_inverse_mu0 = 1.0_rt / PhysConst::mu0;
    const amrex::Real cgl_instability_scale = m_cgl_instability_scale;
    const amrex::Real cgl_instability_width = m_cgl_instability_width;
    // Magnetization-weighted null blend (Stage B+): w = 1/(1+(r_L/dx)^2)
    // written as Omega_ci^2 dx^2 / (Omega_ci^2 dx^2 + v_thi^2), with dx
    // the smallest resolved cell size (absent dimensions report inverse
    // size 0). The denominator is strictly positive through the smooth
    // pressure floor in v_thi^2, so w is C-infinity and exactly 0 at
    // B = 0.
    const amrex::Real cgl_null_scale = m_cgl_null_scale;
    const amrex::Real cgl_null_inverse_length = std::max(
        {inverse_cell_size[0], inverse_cell_size[1], inverse_cell_size[2]});
    const amrex::Real cgl_null_omega2_coefficient =
        (m_ion_charge_to_mass / cgl_null_inverse_length) *
        (m_ion_charge_to_mass / cgl_null_inverse_length);
    // Braginskii ion-ion collision rate, nu_ii = sqrt(2) n_i e^4 lnLambda
    // / (12 pi^{3/2} sqrt(m_i) eps0^2 (k T_i)^{3/2}), scaled by
    // cgl_relaxation_scale; the kernel evaluates
    // nu_iso = prefactor * n_i / kT_eff^{3/2}.
    const amrex::Real cgl_relaxation_prefactor =
        m_cgl_relaxation_scale * std::sqrt(2.0_rt) *
        std::pow(PhysConst::q_e, 4) * m_cgl_coulomb_log /
        (12.0_rt * std::pow(MathConst::pi, amrex::Real(1.5)) *
         std::sqrt(ion_mass) * PhysConst::epsilon_0 * PhysConst::epsilon_0);
    // Strictly positive temperature anchor for the smooth kT_eff floor:
    // the FLOOR pressure at the REFERENCE density, which is far below
    // any operating temperature (the earlier p_floor/n_floor anchor is
    // the reference temperature itself when both floors are the same
    // fraction of their references, and the softplus then inflates
    // healthy kT by 1.5x, biasing nu_iso low by 1.5^{3/2} -- caught by
    // the calibrated relaxation CI test). Any strictly positive anchor
    // is admissible; it only bounds nu_iso and keeps T^{-3/2} smooth.
    const amrex::Real cgl_temperature_floor =
        m_ion_pressure_floor * ion_mass / m_reference_mass_density;
    // Direction regularization of bhat bhat = b b / smooth-floored |b|^2:
    // below the field-energy scale mu0 p_i_floor the deviation force and
    // the anisotropy work are physically negligible, so only the
    // (arbitrary) field direction is smoothed out there.
    const amrex::Real small_b2 = PhysConst::mu0 * m_ion_pressure_floor;
    const bool holmstrom_vacuum = m_vacuum_mass_density > 0.0_rt;
    const amrex::Real vacuum_mass_density = m_vacuum_mass_density;
    const amrex::Real vacuum_drag_rate = m_vacuum_drag_rate;
    // Halo source taper (inert at pedestal 0, where the limiter is
    // identically 1): the reactive work terms and the CGL relaxation
    // exchange taper C^1-smoothly to zero below twice the pedestal.
    // Across the pedestal band those pointwise sources are noise driven
    // -- the fight maps of the FRC benchmark ladder showed the
    // floor-riding halo pinning tens of thousands of Newton-direction
    // components per solve through exactly these channels -- while the
    // physics there is numerical pedestal mass, so zeroing its reactive
    // response is the consistent counterpart of the offset-density view
    // (the pedestal carries no internal dynamics of its own). Keyed to
    // the STEP-OLD density like the vacuum cell switch, so the taper is
    // a per-solve constant mask.
    const amrex::Real halo_pedestal = m_halo_pedestal_density;
    // Pedestal-band velocity relaxation (see m_halo_pedestal_drag_rate):
    // engages with weight 1 - halo_source_taper, i.e. full rate at the
    // pedestal and exactly zero at/above twice it.
    const amrex::Real halo_pedestal_drag_rate = m_halo_pedestal_drag_rate;
    // Pedestal-band ion-energy relaxation (see
    // m_halo_pedestal_energy_rate): same mask complement as the drag.
    // The targets are the per-solve frozen pedestal images, clamped at
    // the corresponding positivity floors so the drain can never demand
    // an inadmissible state.
    const amrex::Real halo_pedestal_energy_rate = m_halo_pedestal_energy_rate;
    const amrex::Real halo_pedestal_ion_internal =
        std::max(m_halo_pedestal_ion_internal, ion_energy_floor);
    const amrex::Real halo_pedestal_ion_parallel =
        std::max(m_halo_pedestal_ion_parallel,
                 0.5_rt * m_ion_pressure_floor);
    const amrex::Real halo_pedestal_ion_perp =
        std::max(m_halo_pedestal_ion_perp, m_ion_pressure_floor);
    // Floor-consistency relaxation source (see m_floor_consistency_rate
    // and floor_consistency_deficit): one-sided per-cell supply at the
    // SAME theta-stage admissibility bounds the Newton projection
    // enforces, per bounded block (rho, U_e, E_i under total_energy,
    // U_par and U_perp under cgl), capped at 1/(theta dt) so the
    // per-solve supplied increment never exceeds the deficit scale plus
    // half a rectifier width (no overshoot). Bounds and rate are
    // per-solve constants (frozen-coefficient idiom): JFNK-exact. The
    // OFF path adds nothing at all -- bit-identical by construction.
    // Wall-masked cells are excluded through wall_live, like every
    // other source.
    const bool floor_consistency = m_floor_consistency_rate > 0.0_rt;
    const amrex::Real floor_supply_rate =
        std::min(m_floor_consistency_rate, 1.0_rt / theta_dt);
    const amrex::Real fc_width = m_floor_consistency_width_fraction;
    const amrex::Real theta = m_theta;
    const AdmissibilityBounds admissibility = MakeAdmissibilityBounds();
    const amrex::Real fc_mass_floor = admissibility.floors[0];
    const amrex::Real fc_mass_coefficient =
        admissibility.temperature_coefficients[0];
    const amrex::Real fc_electron_floor = admissibility.floors[1];
    const amrex::Real fc_electron_coefficient =
        admissibility.temperature_coefficients[1];
    const amrex::Real fc_ion_floor = admissibility.floors[2];
    const amrex::Real fc_ion_coefficient =
        admissibility.temperature_coefficients[2];
    const amrex::Real fc_perp_floor = admissibility.floors[3];
    const amrex::Real fc_perp_coefficient =
        admissibility.temperature_coefficients[3];
    const auto eta = m_hybrid_pic_model->m_eta;
    // Joule heating evaluates eta at the SAME hybrid-floored density as
    // Ohm's law, so the electron heating rate matches the field's eta J.
    // It deliberately does NOT see the vacuum resistivity boost
    // (vacuum_keyed_resistivity): vacuum field diffusion never heats
    // plasma.
    const amrex::Real eta_density_floor = OhmMassDensityFloor();
    // Reference-code-style Ohm-current Joule quench (implicit_mhd.joule_ohm_current):
    // identical composition to the ComputeFluidRHS site (the E-based
    // fluid RHS), see the comment there -- the two sites MUST stay exact
    // twins so the residual is one map regardless of the flux path.
    const bool joule_ohm_current =
        m_joule_ohm_current && include_joule_heating;
    const amrex::Real joule_min_cell_size =
        1.0_rt / std::max({inverse_cell_size[0], inverse_cell_size[1],
                           inverse_cell_size[2]});
    const amrex::Real joule_quench_eta_threshold =
        joule_ohm_current ? PhysConst::mu0 * joule_min_cell_size *
                                joule_min_cell_size / m_dt
                          : std::numeric_limits<amrex::Real>::max();
    const amrex::Real joule_vacuum_division_guard =
        charge_to_mass * m_mass_density_floor;
    const amrex::Real joule_vacuum_eta_scale =
        PhysConst::mu0 * m_vacuum_resistivity_diffusivity;
    // Vacuum-boost reference of the quench's eta_field (and hence of the
    // diffusion-dominance criterion): the per-step frozen dynamic
    // Reference (VacuumReferenceMassDensity), matching the field
    // advance. The eta_joule ARGUMENT floor keeps the static Ohm guard.
    const amrex::Real joule_vacuum_reference =
        charge_to_mass * VacuumReferenceMassDensity();
    const WallBandEtaOverrideView joule_band_view =
        m_wall_mask.BandEtaOverrideView();
    const int* const joule_band_cc = joule_band_view.first_band_cc;
    const amrex::Real joule_band_eta = joule_band_view.eta_override;
    const amrex::MultiFab& ohm_electric_cc =
        *m_WarpX->m_fields.get(OhmElectricFieldCCName, 0);
    // Rigid-conductor fluid freeze (implicit_mhd.wall_thermal_bc): under
    // either thermal wall mode the masked conductor cells are NOT fluid
    // -- every increment inside them is zero, so the band keeps its
    // bounded load state forever and the stair interface faces drain the
    // interior one-sidedly (the fluid analog of run32's wall particle
    // scraper). This is what keeps the wall band from running away: the
    // PEC surface current otherwise meets the (anomalous) plasma
    // resistivity at near-floor density -- an unbounded Joule source
    // inside the metal (the wall-on ladder crash mechanism) -- and a
    // conservative booking of the interface conduction drain would
    // recreate the same hot-band signal-speed spike from the other
    // side. Opt-in with the thermal BC: wall_model alone stays
    // bit-identical to the electromagnetic-only wall.
    const bool wall_thermal_freeze =
        (m_wall_mask.GetThermalBC() != ImplicitMHDWallMask::ThermalBC::none);
    const int* const AMREX_RESTRICT wall_first_masked_cc =
        wall_thermal_freeze ? m_wall_mask.FirstMaskedCellCentered() : nullptr;
    const int wall_mask_z_lo = -m_wall_mask.GhostCells();
    const int wall_mask_z_hi =
        m_wall_mask.AxialCells() - 1 + m_wall_mask.GhostCells();
    // Also read by the dual-energy blend in 1D, so hoisted out of the RZ
    // block below.
    const theta_implicit_mhd::FluxParameters flux_parameters =
        MakeFluxParameters();
    // Also read by the electron-ion equilibration's live ion pressure,
    // so hoisted out of the RZ block below.
    const amrex::Real gamma_i_minus_one = m_gamma_i - 1.0_rt;
#if defined(WARPX_DIM_RZ)
    const amrex::Real inverse_dr = inverse_cell_size[0];
    const amrex::Real radial_lower = m_WarpX->Geom(0).ProbLo(0);
    const amrex::Real radial_cell_size = m_WarpX->Geom(0).CellSize(0);
    const amrex::Real inverse_mu0 = 1.0_rt / PhysConst::mu0;
    // Zero-flux wall (open field + reflect fluid): pointwise WORK/stress
    // evaluations at the last cell ring that read the cell-centered
    // radial B ghost use the PEC/reflecting image value (B_r odd, so the
    // ghost B_r flips sign) instead of the open zero-gradient fill; the
    // field itself keeps the true open boundary for the Ohm/Faraday
    // path. Consumed by the CGL deviation-stress radial derivative
    // below; the total-energy work terms are covered by the wall-face
    // flux override in ComputeDirectionalFaceFluxes.
    const int radial_domain_hi = m_WarpX->Geom(0).Domain().bigEnd(0);
    const bool wall_image = m_r_open && (m_r_open_fluid == "reflect");
#endif
    constexpr int flux_mass = FaceFluxComponent::mass;
    constexpr int flux_momentum = FaceFluxComponent::momentum_x;
    constexpr int flux_magnetic = FaceFluxComponent::magnetic_x;
    constexpr int flux_electron_energy = FaceFluxComponent::electron_energy;
    constexpr int flux_ion_energy = FaceFluxComponent::ion_energy;
    constexpr int flux_ion_parallel_energy =
        FaceFluxComponent::ion_parallel_energy;
    constexpr int flux_ion_perp_energy = FaceFluxComponent::ion_perp_energy;
    constexpr int flux_ion_internal_energy =
        FaceFluxComponent::ion_internal_energy;
    constexpr int flux_electron_velocity =
        FaceFluxComponent::electron_velocity;

    for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto rho = density.const_array(mfi);
        const auto rho_old = old_density.const_array(mfi);
        const auto mom = momentum.const_array(mfi);
        const auto mom_old = old_momentum.const_array(mfi);
        const auto energy = electron_energy.const_array(mfi);
        const auto energy_old = old_electron_energy.const_array(mfi);
        const auto ion_e = ion_energy.const_array(mfi);
        const auto ion_e_old = old_ion_energy.const_array(mfi);
        const auto upar = ion_parallel_energy.const_array(mfi);
        const auto uperp = ion_perp_energy.const_array(mfi);
        const auto upar_old = old_ion_parallel_energy.const_array(mfi);
        const auto uperp_old = old_ion_perp_energy.const_array(mfi);
        const auto ion_int = ion_internal_energy.const_array(mfi);
        const auto ion_int_old = old_ion_internal_energy.const_array(mfi);
        const auto j_plasma = current.const_array(mfi);
        const auto b_cc = magnetic_cc.const_array(mfi);
        const auto e_ohm = ohm_electric_cc.const_array(mfi);
        const auto flux_arr = face_flux_mf.const_array(mfi);
#if defined(WARPX_DIM_RZ)
        const auto rflux_arr = face_flux_r_mf.const_array(mfi);
#endif
        const auto rho_increment = density_rhs.array(mfi);
        const auto momentum_increment = momentum_rhs.array(mfi);
        const auto energy_increment = electron_energy_rhs.array(mfi);
        const auto ion_energy_increment =
            total_energy_closure ? ion_energy_rhs->array(mfi)
                                 : amrex::Array4<amrex::Real>{};
        const auto ion_parallel_increment =
            cgl_closure ? ion_parallel_energy_rhs->array(mfi)
                        : amrex::Array4<amrex::Real>{};
        const auto ion_perp_increment =
            cgl_closure ? ion_perp_energy_rhs->array(mfi)
                        : amrex::Array4<amrex::Real>{};
        const auto ion_internal_increment =
            dual_energy_closure ? ion_internal_energy_rhs->array(mfi)
                                : amrex::Array4<amrex::Real>{};

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Finite-volume divergences from the precomputed face fluxes:
            // the low z-face of a cell shares its index; the high face is
            // one step along z (and likewise in r for the RZ faces, with
            // the r_face/r_center weights of the cylindrical divergence).
            int izh = i;
            int jzh = j;
            int kzh = k;
            shift_index(izh, jzh, kzh, 2, 1);
            amrex::Real divergence_momentum =
                inverse_dz * (flux_arr(izh, jzh, kzh, flux_mass) -
                              flux_arr(i, j, k, flux_mass));
            amrex::Real divergence_momentum_flux[3];
            amrex::Real magnetic_force[3];
            for (int component = 0; component < 3; ++component) {
                divergence_momentum_flux[component] =
                    inverse_dz *
                    (flux_arr(izh, jzh, kzh, flux_momentum + component) -
                     flux_arr(i, j, k, flux_momentum + component));
                // The magnetic force is minus the divergence of the
                // Maxwell-stress flux channels: the discrete J x B the
                // ion-energy work must pair with.
                magnetic_force[component] =
                    -inverse_dz *
                    (flux_arr(izh, jzh, kzh, flux_magnetic + component) -
                     flux_arr(i, j, k, flux_magnetic + component));
            }
            amrex::Real divergence_energy_flux =
                inverse_dz * (flux_arr(izh, jzh, kzh, flux_electron_energy) -
                              flux_arr(i, j, k, flux_electron_energy));
            amrex::Real divergence_ion_energy_flux =
                inverse_dz * (flux_arr(izh, jzh, kzh, flux_ion_energy) -
                              flux_arr(i, j, k, flux_ion_energy));
            amrex::Real divergence_ion_parallel_flux =
                inverse_dz *
                (flux_arr(izh, jzh, kzh, flux_ion_parallel_energy) -
                 flux_arr(i, j, k, flux_ion_parallel_energy));
            amrex::Real divergence_ion_perp_flux =
                inverse_dz * (flux_arr(izh, jzh, kzh, flux_ion_perp_energy) -
                              flux_arr(i, j, k, flux_ion_perp_energy));
            amrex::Real divergence_ion_internal_flux =
                inverse_dz *
                (flux_arr(izh, jzh, kzh, flux_ion_internal_energy) -
                 flux_arr(i, j, k, flux_ion_internal_energy));
            amrex::Real divergence_electron_velocity =
                inverse_dz * (flux_arr(izh, jzh, kzh, flux_electron_velocity) -
                              flux_arr(i, j, k, flux_electron_velocity));
#if defined(WARPX_DIM_RZ)
            {
                const amrex::Real radial_center =
                    radial_lower + (i + 0.5_rt) * radial_cell_size;
                const amrex::Real weight_high =
                    (radial_lower + (i + 1.0_rt) * radial_cell_size) /
                    radial_center;
                const amrex::Real weight_low =
                    (radial_lower + i * radial_cell_size) / radial_center;
                divergence_momentum +=
                    inverse_dr *
                    (weight_high * rflux_arr(i + 1, j, k, flux_mass) -
                     weight_low * rflux_arr(i, j, k, flux_mass));
                for (int component = 0; component < 3; ++component) {
                    divergence_momentum_flux[component] +=
                        inverse_dr *
                        (weight_high *
                             rflux_arr(i + 1, j, k, flux_momentum + component) -
                         weight_low *
                             rflux_arr(i, j, k, flux_momentum + component));
                    magnetic_force[component] -=
                        inverse_dr *
                        (weight_high *
                             rflux_arr(i + 1, j, k, flux_magnetic + component) -
                         weight_low *
                             rflux_arr(i, j, k, flux_magnetic + component));
                }
                divergence_energy_flux +=
                    inverse_dr *
                    (weight_high * rflux_arr(i + 1, j, k, flux_electron_energy) -
                     weight_low * rflux_arr(i, j, k, flux_electron_energy));
                divergence_ion_energy_flux +=
                    inverse_dr *
                    (weight_high * rflux_arr(i + 1, j, k, flux_ion_energy) -
                     weight_low * rflux_arr(i, j, k, flux_ion_energy));
                divergence_ion_parallel_flux +=
                    inverse_dr *
                    (weight_high *
                         rflux_arr(i + 1, j, k, flux_ion_parallel_energy) -
                     weight_low *
                         rflux_arr(i, j, k, flux_ion_parallel_energy));
                divergence_ion_perp_flux +=
                    inverse_dr *
                    (weight_high *
                         rflux_arr(i + 1, j, k, flux_ion_perp_energy) -
                     weight_low * rflux_arr(i, j, k, flux_ion_perp_energy));
                divergence_ion_internal_flux +=
                    inverse_dr *
                    (weight_high *
                         rflux_arr(i + 1, j, k, flux_ion_internal_energy) -
                     weight_low *
                         rflux_arr(i, j, k, flux_ion_internal_energy));
                divergence_electron_velocity +=
                    inverse_dr *
                    (weight_high *
                         rflux_arr(i + 1, j, k, flux_electron_velocity) -
                     weight_low * rflux_arr(i, j, k, flux_electron_velocity));

                // Cylindrical (m = 0) geometric source terms that are not
                // expressible as face differences, now with the magnetic
                // stress parts: (div Pi)_r includes -Pi_theta_theta / r
                // and (div Pi)_theta includes +Pi_theta_r / r, with
                // Pi = rho u u + (p_i + p_e + B^2/2mu0) I - B B / mu0.
                // The magnetic parts also enter the magnetic force so the
                // ion-energy work pairs with the SAME discrete operator.
                const amrex::Real inverse_radius = 1.0_rt / radial_center;
                const amrex::Real safe_density =
                    std::max(rho(i, j, k), density_floor);
                const amrex::Real velocity_r = mom(i, j, k, 0) / safe_density;
                const amrex::Real velocity_theta =
                    mom(i, j, k, 1) / safe_density;
                amrex::Real pressure_i;
                if (dual_energy_closure) {
                    // Blended pressure, the exact expression of the dual
                    // cell loader, so this geometric source matches the
                    // fan's momentum flux.
                    amrex::Real kinetic_energy = 0.0_rt;
                    for (int component = 0; component < 3; ++component) {
                        kinetic_energy += mom(i, j, k, component) *
                                          mom(i, j, k, component);
                    }
                    kinetic_energy *= 0.5_rt / safe_density;
                    pressure_i =
                        theta_implicit_mhd::dual_energy_blended_pressure(
                            ion_e(i, j, k), kinetic_energy,
                            ion_int(i, j, k), ion_int_old(i, j, k),
                            flux_parameters);
                } else if (total_energy_closure) {
                    amrex::Real kinetic_energy = 0.0_rt;
                    for (int component = 0; component < 3; ++component) {
                        kinetic_energy += mom(i, j, k, component) *
                                          mom(i, j, k, component);
                    }
                    kinetic_energy *= 0.5_rt / safe_density;
                    // Same C-infinity smooth internal-energy floor AND
                    // the same KE-scaled corner width as the kernel's
                    // p_i(E_i) recovery (pressure_corner_width_fraction;
                    // fraction 0 keeps the legacy floor width): a
                    // second, un-widened corner in the same cells would
                    // undercut the widened kernel corner near the floor.
                    const amrex::Real excess =
                        ion_e(i, j, k) - kinetic_energy - ion_energy_floor;
                    const amrex::Real corner_width = std::max(
                        ion_energy_floor,
                        flux_parameters.pressure_corner_width_fraction *
                            kinetic_energy);
                    pressure_i =
                        gamma_i_minus_one *
                        (ion_energy_floor +
                         0.5_rt * (excess +
                                   std::sqrt(excess * excess +
                                             corner_width * corner_width)));
                } else if (cgl_closure) {
                    // Effective isotropic pressure with the SAME smooth
                    // floors as the kernel's p_eff, so this geometric
                    // source matches the fan's momentum flux; the
                    // trace-free deviation part enters through the
                    // separate deviation-stress divergence below.
                    pressure_i =
                        (theta_implicit_mhd::smooth_positive_floor(
                             2.0_rt * upar(i, j, k), ion_pressure_floor) +
                         2.0_rt * theta_implicit_mhd::smooth_positive_floor(
                                      uperp(i, j, k), ion_pressure_floor)) /
                        3.0_rt;
                } else {
                    pressure_i = theta_implicit_mhd::ion_pressure(
                        safe_density, flux_parameters);
                }
                const amrex::Real total_pressure =
                    pressure_i + std::max(gamma_e_minus_one * energy(i, j, k),
                                          pressure_floor);
                divergence_momentum_flux[0] -=
                    inverse_radius *
                    (total_pressure + mom(i, j, k, 1) * velocity_theta);
                divergence_momentum_flux[1] +=
                    inverse_radius * mom(i, j, k, 1) * velocity_r;

                const amrex::Real br = b_cc(i, j, k, 0);
                const amrex::Real bt = b_cc(i, j, k, 1);
                const amrex::Real bz = b_cc(i, j, k, 2);
                const amrex::Real magnetic_theta_theta =
                    (0.5_rt * (br * br + bt * bt + bz * bz) - bt * bt) *
                    inverse_mu0;
                const amrex::Real magnetic_theta_r = -bt * br * inverse_mu0;
                divergence_momentum_flux[0] -=
                    inverse_radius * magnetic_theta_theta;
                divergence_momentum_flux[1] +=
                    inverse_radius * magnetic_theta_r;
                magnetic_force[0] += inverse_radius * magnetic_theta_theta;
                magnetic_force[1] -= inverse_radius * magnetic_theta_r;
            }
#endif

            // Stage B+ magnetization weight of the gyrotropic closure,
            // w = Omega_ci^2 dx^2 / (Omega_ci^2 dx^2 + v_thi^2), i.e.
            // w = 1/(1 + (r_L/dx)^2): where the ion gyroradius is not
            // resolved -- in particular at field nulls, where the RAW
            // (unfloored) b2 drives w to exactly 0 -- the gyrotropic
            // stress and work degenerate C-infinity-smoothly to
            // isotropic MHD with p_eff. Only defined here; evaluated
            // solely inside the cgl blocks below.
            const auto null_weight = [=] (const int ic, const int jc,
                                          const int kc) {
                amrex::Real b2n = 0.0_rt;
                for (int component = 0; component < 3; ++component) {
                    b2n += b_cc(ic, jc, kc, component) *
                           b_cc(ic, jc, kc, component);
                }
                const amrex::Real pressure_eff =
                    (2.0_rt * upar(ic, jc, kc) +
                     2.0_rt * uperp(ic, jc, kc)) / 3.0_rt;
                const amrex::Real thermal_speed2 =
                    theta_implicit_mhd::smooth_positive_floor(
                        pressure_eff, ion_pressure_floor) /
                    std::max(rho(ic, jc, kc), density_floor);
                const amrex::Real magnetization2 =
                    cgl_null_omega2_coefficient * b2n;
                return magnetization2 / (magnetization2 + thermal_speed2);
            };

            // Marginal-stability clamp of the TRANSMITTED anisotropy
            // (Stage B+): kinetic instabilities cap the deviation stress
            // a plasma can sustain at the firehose marginal value
            // Delta p = +B^2/mu0 (effective tension reaches zero, never
            // negative) and the mirror marginal value
            // Delta p = -(B^2/mu0) p_par/(2 p_perp). The Stage B bounded
            // RELAXATION drives U_par/U_perp toward this band but cannot
            // stop a transient super-marginal excursion from transmitting
            // unbounded force through the dissipation-free
            // central-difference stress divergence (the ~53 t_ci FRC
            // wall-channel runaway reached |Delta p| ~ 1e3 x marginal).
            // C^2 tanh soft clips (knee widths from cgl_instability_width)
            // pass the stable interior through EXACTLY -- an equilibrium
            // inside the band acquires no spurious stress offset -- and
            // the mirror bound is self-limiting: it shrinks as
            // p_par/p_perp collapses. The energy work terms use the SAME
            // clamped difference through the p_eff/Delta p decomposition,
            // keeping the total internal work exactly -P_blend : grad u.
            const auto marginal_pressure_difference =
                [=] (const int ic, const int jc, const int kc) {
                    const amrex::Real raw =
                        2.0_rt * upar(ic, jc, kc) - uperp(ic, jc, kc);
                    amrex::Real b2n = 0.0_rt;
                    for (int component = 0; component < 3; ++component) {
                        b2n += b_cc(ic, jc, kc, component) *
                               b_cc(ic, jc, kc, component);
                    }
                    const amrex::Real magnetic_pressure =
                        theta_implicit_mhd::smooth_positive_floor(
                            b2n, small_b2) * cgl_inverse_mu0;
                    const amrex::Real parallel_floored =
                        theta_implicit_mhd::smooth_positive_floor(
                            2.0_rt * upar(ic, jc, kc), ion_pressure_floor);
                    const amrex::Real perp_floored =
                        theta_implicit_mhd::smooth_positive_floor(
                            uperp(ic, jc, kc), ion_pressure_floor);
                    const amrex::Real mirror_magnitude =
                        magnetic_pressure * parallel_floored /
                        (2.0_rt * perp_floored);
                    const amrex::Real firehose_clamped =
                        theta_implicit_mhd::soft_upper_clip(
                            raw, magnetic_pressure,
                            cgl_instability_width * magnetic_pressure);
                    return -theta_implicit_mhd::soft_upper_clip(
                        -firehose_clamped, mirror_magnitude,
                        cgl_instability_width * mirror_magnitude);
                };

            if (cgl_closure) {
                // Anisotropic (CGL) deviation stress, magnetization
                // weighted (Stage B+):
                // PI_dev = w (p_par - p_perp)(bhat bhat - I/3): the fan's
                // momentum flux already carries the EFFECTIVE isotropic
                // pressure p_eff = (p_par + 2 p_perp)/3, so only the
                // trace-free deviation enters here, as a pointwise
                // second-order central-difference divergence of the
                // cell-centered products (ghost neighbors are filled each
                // residual). bhat_a bhat_b is regularized C-infinity as
                // b_a b_b / smooth_positive_floor(|b|^2, mu0 p_i_floor):
                // below that field-energy scale the deviation force is
                // physically negligible and only the (arbitrary) field
                // direction is smoothed away. p_par - p_perp is the
                // marginal-stability CLAMPED difference (smooth in the
                // unknowns; see marginal_pressure_difference above). The
                // weight w makes the momentum stress exactly the P_blend
                // whose work the energy blocks integrate below.
                const auto pi_dev = [=] (const int ic, const int jc,
                                         const int kc, const int a,
                                         const int b) {
                    const amrex::Real pressure_difference =
                        marginal_pressure_difference(ic, jc, kc);
                    amrex::Real b2 = 0.0_rt;
                    for (int component = 0; component < 3; ++component) {
                        b2 += b_cc(ic, jc, kc, component) *
                              b_cc(ic, jc, kc, component);
                    }
                    amrex::Real field_a = b_cc(ic, jc, kc, a);
                    amrex::Real field_b = b_cc(ic, jc, kc, b);
#if defined(WARPX_DIM_RZ)
                    // Zero-flux wall: the outer radial ghost's B_r takes
                    // the PEC image sign for this STRESS evaluation only
                    // (the open ghost fill is zero-gradient; the image
                    // B_r is odd, tangentials even, and |B|^2 -- hence
                    // the floor, the clamp, and the null weight -- is
                    // sign-invariant).
                    if (wall_image && ic > radial_domain_hi) {
                        if (a == 0) { field_a = -field_a; }
                        if (b == 0) { field_b = -field_b; }
                    }
#endif
                    const amrex::Real bhat_ab =
                        field_a * field_b /
                        theta_implicit_mhd::smooth_positive_floor(b2,
                                                                  small_b2);
                    return null_weight(ic, jc, kc) * pressure_difference *
                           (bhat_ab -
                            (a == b ? (1.0_rt / 3.0_rt) : 0.0_rt));
                };
#if defined(WARPX_DIM_1D_Z)
                // 1D: only d/dz of the PI_dev z-row contributes.
                for (int component = 0; component < 3; ++component) {
                    divergence_momentum_flux[component] +=
                        0.5_rt * inverse_dz *
                        (pi_dev(i + 1, j, k, 2, component) -
                         pi_dev(i - 1, j, k, 2, component));
                }
#elif defined(WARPX_DIM_RZ)
                // RZ (m = 0): (div PI)_c = (1/r) d(r PI_rc)/dr
                // + d(PI_zc)/dz plus the geometric terms
                // -PI_theta_theta / r (r component) and +PI_theta_r / r
                // (theta component), mirroring the magnetic geometric
                // sources above. The radial derivative uses the SIGNED
                // neighbor radii, so the axis cell (i = 0) reads the
                // parity-mirrored ghost at r = -dr/2 and the difference
                // is the smooth even/odd extension across the axis.
                const amrex::Real radius_center =
                    radial_lower + (i + 0.5_rt) * radial_cell_size;
                const amrex::Real radius_high =
                    radial_lower + (i + 1.5_rt) * radial_cell_size;
                const amrex::Real radius_low =
                    radial_lower + (i - 0.5_rt) * radial_cell_size;
                const amrex::Real inverse_radius = 1.0_rt / radius_center;
                for (int component = 0; component < 3; ++component) {
                    divergence_momentum_flux[component] +=
                        0.5_rt * inverse_dr * inverse_radius *
                            (radius_high *
                                 pi_dev(i + 1, j, k, 0, component) -
                             radius_low *
                                 pi_dev(i - 1, j, k, 0, component)) +
                        0.5_rt * inverse_dz *
                            (pi_dev(i, j + 1, k, 2, component) -
                             pi_dev(i, j - 1, k, 2, component));
                }
                divergence_momentum_flux[0] -=
                    inverse_radius * pi_dev(i, j, k, 1, 1);
                divergence_momentum_flux[1] +=
                    inverse_radius * pi_dev(i, j, k, 1, 0);
#endif
            }

            const amrex::Real plasma_weight =
                holmstrom_vacuum
                    ? 0.5_rt * (1.0_rt + std::tanh(
                                             (rho_old(i, j, k) - vacuum_mass_density) /
                                             (0.3_rt * vacuum_mass_density)))
                    : 1.0_rt;
            // C^1 halo source taper (see the halo_pedestal host constant):
            // 1 above twice the pedestal, 0 at/below it, applied to the
            // reactive work and relaxation sources below. Identically 1
            // when the pedestal is off.
            const amrex::Real halo_source_taper =
                theta_implicit_mhd::floor_outflow_limiter(rho_old(i, j, k),
                                                          halo_pedestal);

            // Thermal-wall fluid freeze (see the host constants above):
            // under wall_thermal_bc every fluid increment inside a masked
            // conductor cell is zero -- the metal is a rigid absorbing
            // body, not fluid. The stair interface faces then drain the
            // INTERIOR one-sidedly (what crosses is gone: mass, momentum,
            // enthalpy -- the fluid analog of run32's wall particle
            // scraper) while the band keeps its bounded load state
            // forever: no Joule pile-up of the PEC surface current
            // against the anomalous resistivity at near-floor density,
            // no drained-heat accumulation, no E_i-minus-KE corner. A
            // static-geometry 0/1 scale, C-infinity for the JFNK probes
            // like every wall projection.
            amrex::Real wall_live = 1.0_rt;
            if (wall_thermal_freeze) {
                const int mask_jz = std::max(wall_mask_z_lo,
                                             std::min(wall_mask_z_hi, j));
                if (i >= wall_first_masked_cc[mask_jz]) {
                    wall_live = 0.0_rt;
                }
            }

            rho_increment(i, j, k) =
                evolve_ion_fluid
                    ? wall_live * -theta_dt * plasma_weight *
                          divergence_momentum
                    : 0.0_rt;

            const amrex::Real vacuum_drag =
                vacuum_drag_rate * (1.0_rt - plasma_weight);
            // Pedestal-band drag (see m_halo_pedestal_drag_rate): the
            // mass gates hold the band's density flux at zero, but its
            // momentum rows still integrate the surrounding truncation
            // forcing; the diagonal drag bounds the band at a terminal
            // velocity instead of letting that inconsistency pin the
            // Newton residual. Same pattern as the vacuum drag.
            const amrex::Real halo_drag =
                halo_pedestal_drag_rate * (1.0_rt - halo_source_taper);
            for (int component = 0; component < 3; ++component) {
                // The Maxwell stress inside the total momentum flux
                // replaces the pointwise J x B force.
                momentum_increment(i, j, k, component) =
                    evolve_ion_fluid
                        ? wall_live * theta_dt *
                              (plasma_weight *
                                   (-divergence_momentum_flux[component]) -
                               (vacuum_drag + halo_drag) *
                                   mom(i, j, k, component))
                        : 0.0_rt;
            }

            const amrex::Real jx = j_plasma(i, j, k, 0);
            const amrex::Real jy = j_plasma(i, j, k, 1);
            const amrex::Real jz = j_plasma(i, j, k, 2);
            const amrex::Real pressure_e =
                std::max(gamma_e_minus_one * energy(i, j, k), pressure_floor);
            const amrex::Real current_magnitude =
                std::sqrt(jx * jx + jy * jy + jz * jz);
            const amrex::Real charge_density =
                charge_to_mass * std::max(rho(i, j, k), eta_density_floor);
            // Te [K] of the (rho, Te, J, t) parser: the same
            // temperature-primary cell ratio as FillFluidSources, from the
            // already-recovered pressure over the already-floored density.
            const amrex::Real temperature_e =
                pressure_e * PhysConst::q_e / (charge_density * PhysConst::kb);
            amrex::Real joule_heating = 0.0_rt;
            if (include_joule_heating) {
                const amrex::Real eta_joule = eta(
                    charge_density, temperature_e, current_magnitude, time);
                amrex::Real square_current =
                    current_magnitude * current_magnitude;
                if (joule_ohm_current) {
                    // Ohm-current quench (see the host constants): the
                    // cell-centered twin of the field advance's
                    // eta_field, from the site's own eta_joule.
                    amrex::Real eta_field =
                        theta_implicit_mhd::vacuum_keyed_resistivity(
                            eta_joule, charge_to_mass * rho(i, j, k),
                            joule_vacuum_reference,
                            joule_vacuum_division_guard,
                            joule_vacuum_eta_scale);
                    if (joule_band_cc != nullptr && i >= joule_band_cc[j]) {
                        eta_field = std::max(eta_field, joule_band_eta);
                    }
                    if (eta_field > joule_quench_eta_threshold) {
                        const amrex::Real e_x = e_ohm(i, j, k, 0);
                        const amrex::Real e_y = e_ohm(i, j, k, 1);
                        const amrex::Real e_z = e_ohm(i, j, k, 2);
                        square_current =
                            (e_x * e_x + e_y * e_y + e_z * e_z) /
                            (eta_field * eta_field);
                    }
                }
                joule_heating = eta_joule * square_current;
            }
            amrex::Real pressure_work =
                -pressure_e * divergence_electron_velocity;
            const amrex::Real energy_end =
                energy(i, j, k) * (1.0_rt + extrapolation_weight) -
                energy_old(i, j, k) * extrapolation_weight;
            // C-infinity smoothed drain gates: blend full and
            // floor-limited work by the smoothed work sign, with the
            // width set by the local flux-divergence scale (anchored
            // strictly positive by the floor rate). The hard
            // if (work < 0) gate is a derivative kink that sits exactly
            // where near-floor halo work terms fluctuate around zero,
            // which defeats the Newton line search on magnetized
            // floor-density equilibria.
            const auto drain_gate = [=] (const amrex::Real work,
                                         const amrex::Real divergence_scale,
                                         const amrex::Real floor_rate,
                                         const amrex::Real limiter) {
                const amrex::Real width =
                    work_kappa *
                    std::sqrt(divergence_scale * divergence_scale +
                              floor_rate * floor_rate);
                const amrex::Real positive_weight =
                    0.5_rt *
                    (1.0_rt + theta_implicit_mhd::smooth_sign(work, width));
                return work * (positive_weight +
                               (1.0_rt - positive_weight) * limiter);
            };
            pressure_work = halo_source_taper *
                            drain_gate(pressure_work, divergence_energy_flux,
                                       electron_energy_floor_rate,
                                       theta_implicit_mhd::floor_outflow_limiter(
                                           energy_end,
                                           pressure_floor / gamma_e_minus_one));
            energy_increment(i, j, k) =
                wall_live * theta_dt * plasma_weight *
                (-divergence_energy_flux + pressure_work + joule_heating);
            // Electron-ion equilibration (see the host constants above
            // and m_electron_ion_equilibration; the reference code's eq_brate
            // exchange): the STEP-OLD frozen Spitzer rate times the LIVE
            // linear pair term (p_i - p_e) -- +Q to the electron energy
            // here, -Q to the ion channel of the active closure below,
            // under the SAME symmetric envelope (wall_live,
            // plasma_weight, halo taper), so the species pair sum
            // changes by exactly zero. Appended after the base
            // increments under a per-solve-uniform branch: the OFF path
            // performs no arithmetic at all.
            amrex::Real equilibration_heating = 0.0_rt;
            if (ei_equilibration) {
                const amrex::Real ei_number_density =
                    std::max(rho_old(i, j, k), density_floor) *
                    inverse_ion_mass;
                const amrex::Real ei_te_ev =
                    std::max(gamma_e_minus_one * energy_old(i, j, k),
                             pressure_floor) /
                    (ei_number_density * PhysConst::q_e);
                const amrex::Real nu_ei =
                    theta_implicit_mhd::electron_ion_equilibration_rate(
                        ei_number_density, ei_te_ev, ei_rate_coefficient,
                        ei_rate_cap);
                amrex::Real pressure_i_live;
                if (cgl_closure) {
                    // p_eff = (p_par + 2 p_perp)/3 with p_par = 2 U_par
                    // and p_perp = U_perp.
                    pressure_i_live = (2.0_rt * upar(i, j, k) +
                                       2.0_rt * uperp(i, j, k)) /
                                      3.0_rt;
                } else {
                    amrex::Real ei_kinetic = 0.0_rt;
                    for (int component = 0; component < 3; ++component) {
                        ei_kinetic += mom(i, j, k, component) *
                                      mom(i, j, k, component);
                    }
                    ei_kinetic *=
                        0.5_rt / std::max(rho(i, j, k), density_floor);
                    // dual: the SAME kinetic-fraction blended pressure
                    // every other consumer sees (the dynamics' single
                    // source of truth for the ion temperature).
                    pressure_i_live =
                        dual_energy_closure
                            ? theta_implicit_mhd::
                                  dual_energy_blended_pressure(
                                      ion_e(i, j, k), ei_kinetic,
                                      ion_int(i, j, k),
                                      ion_int_old(i, j, k),
                                      flux_parameters)
                            : gamma_i_minus_one *
                                  (ion_e(i, j, k) - ei_kinetic);
                }
                pressure_i_live =
                    std::max(pressure_i_live, ion_pressure_floor);
                equilibration_heating = halo_source_taper * nu_ei *
                                        (pressure_i_live - pressure_e);
                energy_increment(i, j, k) += wall_live * theta_dt *
                                             plasma_weight *
                                             equilibration_heating;
            }

            if (total_energy_closure) {
                const amrex::Real safe_density =
                    std::max(rho(i, j, k), density_floor);
                amrex::Real kinetic_end = 0.0_rt;
                for (int component = 0; component < 3; ++component) {
                    const amrex::Real mom_end =
                        mom(i, j, k, component) *
                            (1.0_rt + extrapolation_weight) -
                        mom_old(i, j, k, component) * extrapolation_weight;
                    kinetic_end += mom_end * mom_end;
                }
                kinetic_end *=
                    0.5_rt /
                    std::max(rho(i, j, k) * (1.0_rt + extrapolation_weight) -
                                 rho_old(i, j, k) * extrapolation_weight,
                             density_floor);
                const amrex::Real internal_proxy_end =
                    ion_e(i, j, k) * (1.0_rt + extrapolation_weight) -
                    ion_e_old(i, j, k) * extrapolation_weight - kinetic_end;
                // Work done by the magnetic force on the ion fluid, from
                // the SAME discrete stress-flux difference the momentum
                // equation uses, so field-to-fluid energy exchange is
                // discretely consistent.
                amrex::Real lorentz_work = 0.0_rt;
                for (int component = 0; component < 3; ++component) {
                    lorentz_work +=
                        mom(i, j, k, component) * magnetic_force[component];
                }
                lorentz_work /= safe_density;
                amrex::Real ion_pressure_work =
                    pressure_e * divergence_electron_velocity;
                const amrex::Real ion_limiter =
                    theta_implicit_mhd::floor_outflow_limiter(
                        internal_proxy_end, ion_energy_floor);
                ion_pressure_work =
                    halo_source_taper *
                    drain_gate(ion_pressure_work, divergence_ion_energy_flux,
                               ion_energy_floor_rate, ion_limiter);
                lorentz_work =
                    halo_source_taper *
                    drain_gate(lorentz_work, divergence_ion_energy_flux,
                               ion_energy_floor_rate, ion_limiter);
                // Kinetic-energy drain matched to the pedestal-band
                // momentum drag: E_i carries the kinetic energy under
                // this closure, so the drag must remove exactly the
                // kinetic decay -nu |m|^2/rho it causes (at theta = 1/2
                // the pairing is discretely exact) -- otherwise the
                // relaxed band would read as spurious internal heating.
                // Outside plasma_weight, like the momentum drag term.
                amrex::Real momentum_square = 0.0_rt;
                for (int component = 0; component < 3; ++component) {
                    momentum_square += mom(i, j, k, component) *
                                       mom(i, j, k, component);
                }
                const amrex::Real drag_kinetic_drain =
                    halo_drag * momentum_square / safe_density;
                // Pedestal-band ion-energy relaxation (see
                // m_halo_pedestal_energy_rate): drains ONLY the internal
                // part E_i - |m|^2/(2 rho) toward the frozen pedestal
                // image -- the momentum drag owns the kinetic channel
                // (its matched drain above), so the composition is
                // triangular in (KE, e_int) and neither term
                // double-counts the other. ONE-SIDED (rectified): the
                // C^1 gate closes the term where the internal part sits
                // at or below the pedestal image, so it can never act as
                // a SOURCE -- a two-sided form was measured to pump the
                // KE-dominated wall-band cells (theta-stage E_i < KE,
                // the big-minus-big corners) toward their growing
                // KE-following target at the relaxation rate, a secular
                // runaway of the band's conservative E_i stock (ambient
                // Newton norm x30 by step 660, NaN blow-up at 678 on the
                // FRC benchmark ladder). Full exact rate at/above twice
                // the image, like the drain gates. NARROW density
                // window, an octave inside the drag mask: full at/below
                // 1.125 rho_ped, exactly zero at/above 1.25 rho_ped,
                // keyed to the step-old density. The drag's 2x window is
                // WRONG for the energy channel: the fingerprinted
                // gate-pinned accretion population sits at 1.0-1.2
                // rho_ped, while the wall's LIVE hot boundary layer
                // (E_i stock 20-400x the pedestal image) cycles through
                // 1.3-1.8 rho_ped -- a 2x-window drain rips that layer
                // down to the image cell-by-cell (measured: 100x
                // cell-to-cell E_i contrast at the wall and a secularly
                // growing ambient residual, realization-dependent
                // ignition). Velocity relaxation is inert on the
                // quasi-static layer, so the drag keeps its wide mask.
                // Outside plasma_weight, like the drag terms.
                const amrex::Real internal_energy =
                    ion_e(i, j, k) -
                    0.5_rt * momentum_square / safe_density;
                const amrex::Real energy_relax_drain =
                    halo_pedestal_energy_rate *
                    (1.0_rt -
                     theta_implicit_mhd::floor_outflow_limiter(
                         rho_old(i, j, k) - halo_pedestal,
                         0.125_rt * halo_pedestal)) *
                    (internal_energy - halo_pedestal_ion_internal) *
                    theta_implicit_mhd::floor_outflow_limiter(
                        internal_energy, halo_pedestal_ion_internal);
                ion_energy_increment(i, j, k) =
                    wall_live * theta_dt *
                    (plasma_weight *
                         (-divergence_ion_energy_flux + lorentz_work +
                          ion_pressure_work) -
                     drag_kinetic_drain - energy_relax_drain);
                if (ei_equilibration) {
                    // Electron-ion equilibration counterpart (see the
                    // electron row above): the identical product with
                    // the opposite sign -- exact pair conservation.
                    // Under dual_energy this books the internal change
                    // into the conservative E_i stock; the auxiliary
                    // U_i mirror is below.
                    ion_energy_increment(i, j, k) -=
                        wall_live * theta_dt * plasma_weight *
                        equilibration_heating;
                }
            }

            if (dual_energy_closure) {
                // Auxiliary internal-energy equation (the reference code's step_wio):
                //   dU_i/dt + div(U_i u) = -p_blend div u + Q_visc
                //                          - relaxation drains,
                // with p_blend the SAME kinetic-fraction blend the
                // momentum flux and the wave fan see (the pressure the
                // dynamics feels), div u the central-difference cell
                // divergence (the CGL work-term convention, RZ metric
                // included), and Q_visc the positive-definite face
                // dissipation of the explicit viscosity -- the
                // internal-only counterpart of the conservative
                // stress-work pair the E_i face channel carries.
                // KE-specific terms (Lorentz work, the electron pdV
                // pairing, the drag's kinetic drain) belong to E_i only;
                // Joule heating deposits into the electron energy alone
                // in this solver, so U_i gets no share (neither does
                // E_i). The E_i block above is untouched: it evolves
                // exactly as under total_energy.
                const amrex::Real safe_density =
                    std::max(rho(i, j, k), density_floor);
                const auto velocity = [=] (const int ic, const int jc,
                                           const int kc,
                                           const int component) {
                    return mom(ic, jc, kc, component) /
                           std::max(rho(ic, jc, kc), density_floor);
                };
#if defined(WARPX_DIM_1D_Z)
                const amrex::Real divergence_velocity =
                    0.5_rt * inverse_dz *
                    (velocity(i + 1, j, k, 2) - velocity(i - 1, j, k, 2));
#elif defined(WARPX_DIM_RZ)
                // div u = du_r/dr + u_r/r + du_z/dz (m = 0); the axis
                // cell reads the parity-mirrored radial ghost, like the
                // CGL velocity gradients.
                const amrex::Real inverse_radius =
                    1.0_rt /
                    (radial_lower + (i + 0.5_rt) * radial_cell_size);
                const amrex::Real divergence_velocity =
                    0.5_rt * inverse_dr *
                        (velocity(i + 1, j, k, 0) -
                         velocity(i - 1, j, k, 0)) +
                    (mom(i, j, k, 0) / safe_density) * inverse_radius +
                    0.5_rt * inverse_dz *
                        (velocity(i, j + 1, k, 2) -
                         velocity(i, j - 1, k, 2));
#endif
                amrex::Real kinetic_energy = 0.0_rt;
                for (int component = 0; component < 3; ++component) {
                    kinetic_energy += mom(i, j, k, component) *
                                      mom(i, j, k, component);
                }
                kinetic_energy *= 0.5_rt / safe_density;
                const amrex::Real pressure_blend =
                    theta_implicit_mhd::dual_energy_blended_pressure(
                        ion_e(i, j, k), kinetic_energy, ion_int(i, j, k),
                        ion_int_old(i, j, k), flux_parameters);
                // Blended PdV work, drain-gated exactly like the other
                // pointwise energy sources: theta-extrapolated
                // end-of-step limiter value, width from the local
                // flux-divergence scale anchored by the floor rate.
                const amrex::Real internal_end =
                    ion_int(i, j, k) * (1.0_rt + extrapolation_weight) -
                    ion_int_old(i, j, k) * extrapolation_weight;
                const amrex::Real internal_limiter =
                    theta_implicit_mhd::floor_outflow_limiter(
                        internal_end, dual_ion_internal_floor);
                amrex::Real pdv_work =
                    -pressure_blend * divergence_velocity;
                pdv_work = halo_source_taper *
                           drain_gate(pdv_work,
                                      divergence_ion_internal_flux,
                                      ion_energy_floor_rate,
                                      internal_limiter);
                // Viscous heating: the per-face dissipation
                // rho_f nu |du/dn|^2, half to each adjacent cell (the
                // The reference code's per-edge deposit) -- non-negative, so no drain
                // gate; tapered across the pedestal band like every
                // other reactive source.
                amrex::Real viscous_heating = 0.0_rt;
                if (add_viscous_heating) {
                    const auto face_dissipation =
                        [=] (const int in, const int jn, const int kn,
                             const amrex::Real inverse_spacing) {
                            const amrex::Real face_density =
                                0.5_rt *
                                (rho(i, j, k) + rho(in, jn, kn));
                            amrex::Real shear = 0.0_rt;
                            for (int component = 0; component < 3;
                                 ++component) {
                                const amrex::Real du =
                                    (velocity(in, jn, kn, component) -
                                     velocity(i, j, k, component)) *
                                    inverse_spacing;
                                shear += du * du;
                            }
                            return face_density * viscosity * shear;
                        };
#if defined(WARPX_DIM_1D_Z)
                    viscous_heating =
                        0.5_rt *
                        (face_dissipation(i + 1, j, k, inverse_dz) +
                         face_dissipation(i - 1, j, k, inverse_dz));
#elif defined(WARPX_DIM_RZ)
                    viscous_heating =
                        0.5_rt *
                        (face_dissipation(i + 1, j, k, inverse_dr) +
                         face_dissipation(i - 1, j, k, inverse_dr) +
                         face_dissipation(i, j + 1, k, inverse_dz) +
                         face_dissipation(i, j - 1, k, inverse_dz));
#endif
                    viscous_heating *= halo_source_taper;
                }
                // Pedestal-band internal-energy relaxation: U_i is the
                // internal channel natively, so it drains toward the
                // SAME frozen internal pedestal image as E_i's internal
                // part, with the same one-sided gate and the same
                // narrow density window.
                const amrex::Real internal_relax_drain =
                    halo_pedestal_energy_rate *
                    (1.0_rt -
                     theta_implicit_mhd::floor_outflow_limiter(
                         rho_old(i, j, k) - halo_pedestal,
                         0.125_rt * halo_pedestal)) *
                    (ion_int(i, j, k) - halo_pedestal_ion_internal) *
                    theta_implicit_mhd::floor_outflow_limiter(
                        ion_int(i, j, k), halo_pedestal_ion_internal);
                ion_internal_increment(i, j, k) =
                    wall_live * theta_dt *
                    (plasma_weight *
                         (-divergence_ion_internal_flux + pdv_work +
                          viscous_heating) -
                     internal_relax_drain);
                if (ei_equilibration) {
                    // Electron-ion equilibration: U_i receives the
                    // SAME internal-only exchange as E_i above (the
                    // The reference code's step_tm split relaxes the internal
                    // temperatures; KE is untouched), keeping the dual
                    // pair consistent.
                    ion_internal_increment(i, j, k) -=
                        wall_live * theta_dt * plasma_weight *
                        equilibration_heating;
                }
            }

            if (cgl_closure) {
                // Neither the lorentz_work nor the ion_pressure_work
                // machinery of the total-energy closure applies here: the
                // magnetic force does no work on internal energies (it
                // acts through the momentum block only) and the electron
                // pdV pairing is a total-energy bookkeeping device.
                const amrex::Real safe_density =
                    std::max(rho(i, j, k), density_floor);
                // Cell-centered velocity of this cell and its neighbors
                // (ghosts are filled every residual) for the central-
                // difference velocity gradients of the CGL work terms.
                const auto velocity = [=] (const int ic, const int jc,
                                           const int kc,
                                           const int component) {
                    return mom(ic, jc, kc, component) /
                           std::max(rho(ic, jc, kc), density_floor);
                };
                // bhat_a bhat_b with the same C-infinity regularization
                // as the deviation stress above.
                amrex::Real b2 = 0.0_rt;
                for (int component = 0; component < 3; ++component) {
                    b2 += b_cc(i, j, k, component) *
                          b_cc(i, j, k, component);
                }
                const amrex::Real inverse_b2 =
                    1.0_rt /
                    theta_implicit_mhd::smooth_positive_floor(b2, small_b2);
#if defined(WARPX_DIM_1D_Z)
                // 1D: nabla has only the z row, so div u = du_z/dz and
                // bb : grad u = bhat_z sum_j bhat_j du_j/dz.
                amrex::Real gradient_z[3];
                for (int component = 0; component < 3; ++component) {
                    gradient_z[component] =
                        0.5_rt * inverse_dz *
                        (velocity(i + 1, j, k, component) -
                         velocity(i - 1, j, k, component));
                }
                const amrex::Real divergence_velocity = gradient_z[2];
                amrex::Real parallel_gradient = 0.0_rt;
                for (int component = 0; component < 3; ++component) {
                    parallel_gradient += b_cc(i, j, k, 2) *
                                         b_cc(i, j, k, component) *
                                         gradient_z[component];
                }
                parallel_gradient *= inverse_b2;
#elif defined(WARPX_DIM_RZ)
                // Axisymmetric cylindrical grad u, with components
                // A_ij = nabla_i u_j (i = derivative direction) for
                // u = u_r e_r + u_t e_t + u_z e_z and d/dtheta = 0:
                //   nabla_r u_j = du_j/dr,  nabla_z u_j = du_j/dz,
                //   nabla_t u_r = (1/r) du_r/dt - u_t/r = -u_t/r,
                //   nabla_t u_t = (1/r) du_t/dt + u_r/r = +u_r/r,
                //   nabla_t u_z = 0,
                // so div u = A_rr + A_tt + A_zz = du_r/dr + u_r/r
                // + du_z/dz and bb : grad u = sum_ij bhat_i bhat_j A_ij.
                // The axis cell (i = 0) sits at r = dr/2 > 0 and its i-1
                // neighbor is the parity-mirrored radial ghost, so both
                // 1/r terms and the central r derivative stay smooth.
                amrex::Real gradient_r[3];
                amrex::Real gradient_z[3];
                for (int component = 0; component < 3; ++component) {
                    gradient_r[component] =
                        0.5_rt * inverse_dr *
                        (velocity(i + 1, j, k, component) -
                         velocity(i - 1, j, k, component));
                    gradient_z[component] =
                        0.5_rt * inverse_dz *
                        (velocity(i, j + 1, k, component) -
                         velocity(i, j - 1, k, component));
                }
                const amrex::Real inverse_radius =
                    1.0_rt /
                    (radial_lower + (i + 0.5_rt) * radial_cell_size);
                const amrex::Real velocity_r =
                    mom(i, j, k, 0) / safe_density;
                const amrex::Real velocity_theta =
                    mom(i, j, k, 1) / safe_density;
                const amrex::Real divergence_velocity =
                    gradient_r[0] + velocity_r * inverse_radius +
                    gradient_z[2];
                const amrex::Real br = b_cc(i, j, k, 0);
                const amrex::Real bt = b_cc(i, j, k, 1);
                const amrex::Real bz = b_cc(i, j, k, 2);
                const amrex::Real parallel_gradient =
                    (br * (br * gradient_r[0] + bt * gradient_r[1] +
                           bz * gradient_r[2]) +
                     bt * (br * (-velocity_theta * inverse_radius) +
                           bt * (velocity_r * inverse_radius)) +
                     bz * (br * gradient_z[0] + bt * gradient_z[1] +
                           bz * gradient_z[2])) *
                    inverse_b2;
#endif
                // CGL work terms (internal-energy form, no heat flux),
                // magnetization blended and marginal-clamped (Stage B+),
                // written in the p_eff/Delta p decomposition
                // p_par = p_eff + (2/3) dp, p_perp = p_eff - (1/3) dp
                // with dp the marginal-stability CLAMPED difference:
                //   dU_par/dt  + div(U_par u)  =
                //       -w (p_eff + (2/3) dp)(bb : grad u)
                //       - (1-w)(1/3) p_eff div u,
                //   dU_perp/dt + div(U_perp u) =
                //       -w (p_eff - (1/3) dp)(div u - bb : grad u)
                //       - (1-w)(2/3) p_eff div u.
                // The total internal work is exactly -P_blend : grad u
                // with P_blend = p_eff I + w dp (bhat bhat - I/3) -- the
                // SAME weighted, clamped stress the momentum divergence
                // integrates -- and the (1/3, 2/3) split keeps isotropy
                // a fixed point of pure compression at w = 0.
                const amrex::Real pressure_parallel =
                    2.0_rt * upar(i, j, k);
                const amrex::Real pressure_perp = uperp(i, j, k);
                const amrex::Real number_density =
                    safe_density * inverse_ion_mass;
                const amrex::Real pressure_effective =
                    (pressure_parallel + 2.0_rt * pressure_perp) / 3.0_rt;
                const amrex::Real magnetization_weight =
                    null_weight(i, j, k);
                const amrex::Real pressure_difference_clamped =
                    marginal_pressure_difference(i, j, k);
                const amrex::Real isotropic_work =
                    -pressure_effective * divergence_velocity;
                amrex::Real parallel_work =
                    magnetization_weight *
                        (-(pressure_effective +
                           (2.0_rt / 3.0_rt) * pressure_difference_clamped) *
                         parallel_gradient) +
                    (1.0_rt - magnetization_weight) *
                        (1.0_rt / 3.0_rt) * isotropic_work;
                amrex::Real perp_work =
                    magnetization_weight *
                        (-(pressure_effective -
                           (1.0_rt / 3.0_rt) * pressure_difference_clamped) *
                         (divergence_velocity - parallel_gradient)) +
                    (1.0_rt - magnetization_weight) *
                        (2.0_rt / 3.0_rt) * isotropic_work;
                // Ion-ion isotropization at nu_iso = scale * nu_ii
                // (Braginskii), with kT_eff = p_eff/n_i smooth-floored at
                // the strictly positive floor-state anchor:
                //   R_par = +(nu_iso/3)(p_perp - p_par) = -R_perp,
                // conserving U_par + U_perp exactly (up to the separate
                // drain gates below, which only reduce a draining side
                // near its floor).
                const amrex::Real temperature =
                    theta_implicit_mhd::smooth_positive_floor(
                        pressure_effective / number_density,
                        cgl_temperature_floor);
                const amrex::Real nu_collisional =
                    cgl_relaxation_prefactor * number_density /
                    (temperature * std::sqrt(temperature));
                // Instability-bounded relaxation (Stage B): past the
                // firehose bound, p_par - p_perp > B^2/mu0, or the
                // mirror bound, beta_perp (p_perp/p_par - 1) > 1,
                // kinetic instabilities isotropize on cyclotron
                // timescales that no collisional rate represents; the
                // rate gains cgl_instability_scale * Omega_ci through
                // C-infinity switches of relative width
                // cgl_instability_width in the dimensionless threshold
                // measures, holding the anisotropy near marginal
                // stability (relaxation drives toward isotropy, which
                // exits both unstable regions). b2 is the |B|^2 already
                // assembled above; both measures reuse the smooth
                // small_b2 and pressure-floor regularizations.
                const amrex::Real b2_floored =
                    theta_implicit_mhd::smooth_positive_floor(b2,
                                                              small_b2);
                const amrex::Real magnetic_pressure =
                    b2_floored * cgl_inverse_mu0;
                const amrex::Real firehose_measure =
                    (pressure_parallel - pressure_perp) /
                        magnetic_pressure -
                    1.0_rt;
                const amrex::Real parallel_pressure_floored =
                    theta_implicit_mhd::smooth_positive_floor(
                        pressure_parallel, ion_pressure_floor);
                const amrex::Real mirror_measure =
                    2.0_rt * pressure_perp *
                        (pressure_perp - pressure_parallel) /
                        (magnetic_pressure * parallel_pressure_floored) -
                    1.0_rt;
                const amrex::Real firehose_switch =
                    0.5_rt * (1.0_rt + theta_implicit_mhd::smooth_sign(
                                           firehose_measure,
                                           cgl_instability_width));
                const amrex::Real mirror_switch =
                    0.5_rt * (1.0_rt + theta_implicit_mhd::smooth_sign(
                                           mirror_measure,
                                           cgl_instability_width));
                const amrex::Real cyclotron_frequency =
                    charge_to_mass * std::sqrt(b2_floored);
                // Stage B+ null relaxation: unmagnetized ions mix over a
                // cell at their thermal transit rate v_thi/dx, so where
                // w -> 0 (gyroradius unresolved, in particular field
                // nulls, where the Omega_ci-scaled instability rates
                // vanish by construction) the anisotropy relaxes at
                // cgl_null_scale times that rate. Same smooth floors as
                // the weight itself.
                const amrex::Real thermal_speed = std::sqrt(
                    theta_implicit_mhd::smooth_positive_floor(
                        pressure_effective, ion_pressure_floor) /
                    safe_density);
                const amrex::Real nu_null =
                    cgl_null_scale * thermal_speed *
                    cgl_null_inverse_length *
                    (1.0_rt - magnetization_weight);
                const amrex::Real nu_iso =
                    nu_collisional +
                    cgl_instability_scale * cyclotron_frequency *
                        (firehose_switch + mirror_switch) +
                    nu_null;
                amrex::Real parallel_relaxation =
                    (nu_iso / 3.0_rt) *
                    (pressure_perp - pressure_parallel);
                amrex::Real perp_relaxation = -parallel_relaxation;
                // Smoothed drain gates on the pointwise sources, exactly
                // like the total-energy ion gates: theta-extrapolated
                // end-of-step limiter values, widths from the local
                // flux-divergence scale anchored by the floor rates.
                const amrex::Real parallel_end =
                    upar(i, j, k) * (1.0_rt + extrapolation_weight) -
                    upar_old(i, j, k) * extrapolation_weight;
                const amrex::Real perp_end =
                    uperp(i, j, k) * (1.0_rt + extrapolation_weight) -
                    uperp_old(i, j, k) * extrapolation_weight;
                const amrex::Real parallel_limiter =
                    theta_implicit_mhd::floor_outflow_limiter(
                        parallel_end, 0.5_rt * ion_pressure_floor);
                const amrex::Real perp_limiter =
                    theta_implicit_mhd::floor_outflow_limiter(
                        perp_end, ion_pressure_floor);
                parallel_work =
                    halo_source_taper *
                    drain_gate(parallel_work, divergence_ion_parallel_flux,
                               ion_parallel_floor_rate, parallel_limiter);
                parallel_relaxation =
                    halo_source_taper *
                    drain_gate(parallel_relaxation,
                               divergence_ion_parallel_flux,
                               ion_parallel_floor_rate, parallel_limiter);
                perp_work =
                    halo_source_taper *
                    drain_gate(perp_work, divergence_ion_perp_flux,
                               ion_perp_floor_rate, perp_limiter);
                perp_relaxation =
                    halo_source_taper *
                    drain_gate(perp_relaxation, divergence_ion_perp_flux,
                               ion_perp_floor_rate, perp_limiter);
                // Pedestal-band ion-energy relaxation (see
                // m_halo_pedestal_energy_rate): U_par and U_perp are
                // purely internal under this closure (the kinetic energy
                // lives in the momentum block alone), so they relax
                // toward their frozen pedestal images directly -- no
                // kinetic bookkeeping to share with the momentum drag.
                // ONE-SIDED (rectified) like the total_energy form: the
                // C^1 gates close each drain where its energy sits at or
                // below the pedestal image, so the term can never act as
                // a source. Same NARROW density window as the
                // total_energy form (see above): full at/below 1.125
                // rho_ped, exactly zero at/above 1.25 rho_ped -- the
                // live wall boundary layer must not be drained. Outside
                // plasma_weight, like the drag terms.
                const amrex::Real halo_energy_relax =
                    halo_pedestal_energy_rate *
                    (1.0_rt -
                     theta_implicit_mhd::floor_outflow_limiter(
                         rho_old(i, j, k) - halo_pedestal,
                         0.125_rt * halo_pedestal));
                const amrex::Real parallel_relax_drain =
                    halo_energy_relax *
                    (upar(i, j, k) - halo_pedestal_ion_parallel) *
                    theta_implicit_mhd::floor_outflow_limiter(
                        upar(i, j, k), halo_pedestal_ion_parallel);
                const amrex::Real perp_relax_drain =
                    halo_energy_relax *
                    (uperp(i, j, k) - halo_pedestal_ion_perp) *
                    theta_implicit_mhd::floor_outflow_limiter(
                        uperp(i, j, k), halo_pedestal_ion_perp);
                ion_parallel_increment(i, j, k) =
                    wall_live * theta_dt *
                    (plasma_weight *
                         (-divergence_ion_parallel_flux + parallel_work +
                          parallel_relaxation) -
                     parallel_relax_drain);
                ion_perp_increment(i, j, k) =
                    wall_live * theta_dt *
                    (plasma_weight *
                         (-divergence_ion_perp_flux + perp_work +
                          perp_relaxation) -
                     perp_relax_drain);
                if (ei_equilibration) {
                    // Electron-ion equilibration counterpart:
                    // isotropic-temperature deposit, -Q/3 to U_par and
                    // -2Q/3 to U_perp (the same (1/3, 2/3) split as the
                    // w = 0 isotropic work terms: equal dT_par = dT_perp
                    // with U_par = n k T_par / 2, U_perp = n k T_perp).
                    // The thirds sum to the electron row's -Q exactly.
                    const amrex::Real equilibration_share =
                        wall_live * theta_dt * plasma_weight *
                        equilibration_heating / 3.0_rt;
                    ion_parallel_increment(i, j, k) -=
                        equilibration_share;
                    ion_perp_increment(i, j, k) -=
                        2.0_rt * equilibration_share;
                }
            }

            // Floor-consistency supply (see the host constants above and
            // floor_consistency_deficit for the one-sidedness /
            // JFNK-exactness / exact-zero-above-bound / no-overshoot
            // guarantees): appended AFTER the base increments under a
            // per-solve-uniform branch, so the OFF path performs no
            // arithmetic at all -- bit-identical including the sign of
            // zero. Wall-frozen cells are excluded (wall_live), like
            // every other source. Booked per step by
            // AccumulateFloorConsistencySupplyLedger. The block-2 bound
            // constants are closure-keyed by MakeAdmissibilityBounds
            // (E_i under total_energy, U_par under cgl).
            if (floor_consistency) {
                const amrex::Real supply_scale =
                    wall_live * theta_dt * floor_supply_rate;
                if (evolve_ion_fluid) {
                    rho_increment(i, j, k) +=
                        supply_scale *
                        theta_implicit_mhd::floor_consistency_deficit(
                            rho(i, j, k), rho_old(i, j, k),
                            rho_old(i, j, k), fc_mass_floor,
                            fc_mass_coefficient, theta, fc_width);
                }
                energy_increment(i, j, k) +=
                    supply_scale *
                    theta_implicit_mhd::floor_consistency_deficit(
                        energy(i, j, k), energy_old(i, j, k),
                        rho_old(i, j, k), fc_electron_floor,
                        fc_electron_coefficient, theta, fc_width);
                if (total_energy_closure) {
                    ion_energy_increment(i, j, k) +=
                        supply_scale *
                        theta_implicit_mhd::floor_consistency_deficit(
                            ion_e(i, j, k), ion_e_old(i, j, k),
                            rho_old(i, j, k), fc_ion_floor,
                            fc_ion_coefficient, theta, fc_width);
                }
                if (dual_energy_closure) {
                    // The auxiliary U_i block is bounded like the others
                    // (block 3 of MakeAdmissibilityBounds under this
                    // closure), so it gets its own supply; it is NOT
                    // booked in the ledger (bookkeeping, not conserved).
                    ion_internal_increment(i, j, k) +=
                        supply_scale *
                        theta_implicit_mhd::floor_consistency_deficit(
                            ion_int(i, j, k), ion_int_old(i, j, k),
                            rho_old(i, j, k), fc_perp_floor,
                            fc_perp_coefficient, theta, fc_width);
                }
                if (cgl_closure) {
                    ion_parallel_increment(i, j, k) +=
                        supply_scale *
                        theta_implicit_mhd::floor_consistency_deficit(
                            upar(i, j, k), upar_old(i, j, k),
                            rho_old(i, j, k), fc_ion_floor,
                            fc_ion_coefficient, theta, fc_width);
                    ion_perp_increment(i, j, k) +=
                        supply_scale *
                        theta_implicit_mhd::floor_consistency_deficit(
                            uperp(i, j, k), uperp_old(i, j, k),
                            rho_old(i, j, k), fc_perp_floor,
                            fc_perp_coefficient, theta, fc_width);
                }
            }
        });
    }
#else
    amrex::ignore_unused(rhs, time);
    WARPX_ABORT_WITH_MESSAGE(
        "ThetaImplicitMHD::ComputeFluidRHSFromFaceFluxes() requires 1D or "
        "RZ geometry");
#endif
}

void ThetaImplicitMHD::ApplyMagneticCCDomainGhosts (amrex::MultiFab& mf) const
{
#if defined(WARPX_DIM_RZ)
    // Radial domain-ghost fill for the cell-centered magnetic field read
    // by the hlld face kernels: axis mirror with m = 0 parities (odd B_r
    // and B_theta, even B_z) and, at r_max, either the perfect-conductor
    // image (odd B_r -- the normal component vanishes at the wall -- and
    // even tangentials) or zero-gradient ghosts for the open
    // (Green's-function) boundary. Axial ghosts must already be filled
    // (periodic or Neumann), so the radial pass also covers the corners.
    const amrex::Box& domain = m_WarpX->Geom(0).Domain();
    const int domain_lo = domain.smallEnd(0);
    const int domain_hi = domain.bigEnd(0);
    const bool r_open = m_r_open;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const amrex::Box grown = amrex::grow(mfi.validbox(), mf.nGrowVect());
        if (grown.smallEnd(0) >= domain_lo && grown.bigEnd(0) <= domain_hi) {
            continue;
        }
        const auto field = mf.array(mfi);
        amrex::ParallelFor(grown, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if (i < domain_lo) {
                const int mirror = 2 * domain_lo - 1 - i;
                field(i, j, k, 0) = -field(mirror, j, k, 0);
                field(i, j, k, 1) = -field(mirror, j, k, 1);
                field(i, j, k, 2) = field(mirror, j, k, 2);
            } else if (i > domain_hi) {
                const int mirror =
                    r_open ? domain_hi : 2 * domain_hi + 1 - i;
                field(i, j, k, 0) =
                    r_open ? field(mirror, j, k, 0) : -field(mirror, j, k, 0);
                field(i, j, k, 1) = field(mirror, j, k, 1);
                field(i, j, k, 2) = field(mirror, j, k, 2);
            }
        });
    }
#else
    amrex::ignore_unused(mf);
#endif
}

void ThetaImplicitMHD::SanitizeLoadedState ()
{
    // One-time admissibility raise of the beginning-of-run fluid state.
    //
    // InitializeFluidState clamps the parser-filled state at the
    // positivity floors, but Python fluid loaders (installed via the
    // beforeInitEsolve callback, e.g. Grad-Shafranov equilibrium loads)
    // overwrite the implicit_mhd_* fields AFTER that clamp and may load
    // an unfloored state arbitrarily far below the floors (a natural
    // halo decays tens of decades below any usable floor). The bounded
    // Newton solve assumes an admissible start: below-bound cells make
    // the direction projection pin huge populations and stagnate the
    // first solve until FinishStateUpdate aborts on the positivity
    // assertions. Raise below-floor cells to their bounds here, landing
    // the same 1e-6 (|value| + bound) SLACK MARGIN above the bound as
    // the direction projection and the end-of-step restorations (a cell
    // resting exactly on its bound makes both Jacobian probe signs
    // inadmissible). The added mass/energy is the same class of tracked
    // non-conservation as the positivity floors, confined to loaded
    // below-floor cells.
    const auto& periodicity = m_WarpX->Geom(0).periodicity();

    auto raise_scalar_block = [&](const char* block_name,
                                  const amrex::Real block_floor) {
        amrex::MultiFab& block = m_state.getMultiFabBlock(block_name, 0);
        const amrex::Real before_min = block.min(0);
        if (before_min >= block_floor) {
            return;
        }
        for (amrex::MFIter mfi(block); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto arr = block.array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const amrex::Real margin =
                    1.0e-6_rt * (std::abs(arr(i, j, k)) + block_floor);
                arr(i, j, k) = std::max(arr(i, j, k), block_floor + margin);
            });
        }
        block.FillBoundaryAndSync(periodicity);
        amrex::Print() << "ThetaImplicitMHD: raised loaded " << block_name
                       << " (min " << before_min << ") to its floor "
                       << block_floor << " + slack\n";
    };

    raise_scalar_block(MassDensityName, m_mass_density_floor);
    raise_scalar_block(ElectronEnergyName,
                       m_electron_pressure_floor / (m_gamma_e - 1.0_rt));

    if (m_ion_closure == "total_energy" || m_ion_closure == "dual_energy") {
        // E_i >= KE + U_i_floor with the (raised) density in the KE
        // denominator -- the same bound FinishStateUpdate restores at
        // every accepted step end. Under dual_energy the auxiliary
        // internal block gets the plain scalar raise below as well.
        amrex::MultiFab& ion_energy_block =
            m_state.getMultiFabBlock(IonEnergyName, 0);
        const amrex::MultiFab& density_block =
            m_state.getMultiFabBlock(MassDensityName, 0);
        const amrex::MultiFab& momentum_block =
            m_state.getMultiFabBlock(MomentumDensityName, 0);
        const amrex::Real internal_floor =
            m_ion_pressure_floor / (m_gamma_i - 1.0_rt);
        const amrex::Real density_floor = m_mass_density_floor;
        const amrex::Real before_min = ion_energy_block.min(0);
        for (amrex::MFIter mfi(ion_energy_block); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto rho = density_block.const_array(mfi);
            const auto mom = momentum_block.const_array(mfi);
            const auto ion_e = ion_energy_block.array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                amrex::Real kinetic_energy = 0.0_rt;
                for (int component = 0; component < 3; ++component) {
                    kinetic_energy +=
                        mom(i, j, k, component) * mom(i, j, k, component);
                }
                kinetic_energy *=
                    0.5_rt / std::max(rho(i, j, k), density_floor);
                const amrex::Real bound = kinetic_energy + internal_floor;
                const amrex::Real margin =
                    1.0e-6_rt * (std::abs(ion_e(i, j, k)) + bound);
                ion_e(i, j, k) = std::max(ion_e(i, j, k), bound + margin);
            });
        }
        ion_energy_block.FillBoundaryAndSync(periodicity);
        amrex::Print() << "ThetaImplicitMHD: raised loaded " << IonEnergyName
                       << " (min " << before_min
                       << ") to KE + its internal floor + slack\n";
        if (m_ion_closure == "dual_energy") {
            raise_scalar_block(IonInternalEnergyName,
                               m_ion_pressure_floor /
                                   (m_gamma_i - 1.0_rt));
        }
    } else if (m_ion_closure == "cgl") {
        raise_scalar_block(IonParallelEnergyName,
                           0.5_rt * m_ion_pressure_floor);
        raise_scalar_block(IonPerpEnergyName, m_ion_pressure_floor);
    }

    m_state.CopyMultiFabBlocksToFields();
}

void ThetaImplicitMHD::ClampWallExteriorState ()
{
#if defined(WARPX_DIM_RZ)
    // Rigid-vacuum exterior clamp of the shaped-wall band (the
    // band-hygiene contract; see ImplicitMHDWallMask): in every
    // wall_live-freezing mode, SET every masked fluid cell to the fixed
    // clamp image -- whatever the IC (parser or Python loader) put
    // outside the contour is scraped at t = 0. Measured failure without
    // it (formation ladder, T5): a reference-matched IC left
    // above-vacuum-threshold density in the frozen band, which stayed
    // electrically conductive under the dielectric standoff -- band
    // |J_theta| 20x the interior median, current-keyed anomalous-eta
    // blow-up, wholesale Newton soft-caps, partial screening of the
    // coil drive.
    //
    // The image is a FIXED value per block (never a function of the
    // current state), so the clamp is bit-exactly idempotent: restarted
    // states that already carry it are unchanged, and the frozen-row
    // identities plus the band exclusions of RefreshHaloPedestal and
    // the FinishStateUpdate floor restorations preserve it bit-exactly
    // afterwards. Each value sits the standard admissibility SLACK
    // MARGIN above its bound -- the 1e-6 (|value| + bound) convention
    // evaluated at value = bound, i.e. bound * (1 + 2e-6) -- so the
    // band is admissible for the bounded Newton solve and satisfies the
    // end-of-step positivity assertions. Energies land on their
    // floor-consistent values at the configured BACKGROUND temperature:
    // under wall_thermal_bc = temperature that is the wall reservoir
    // T_wall itself (the band is parked AT the reservoir, so the
    // one-sided interface drain starts at exactly zero gradient), else
    // the temperature floors when set (through the admissibility
    // temperature coefficients at the clamp density), else the
    // pressure floors -- always the max, so the image never violates a
    // bound.
    if (m_wall_mask.GetThermalBC() == ImplicitMHDWallMask::ThermalBC::none) {
        return;
    }
    const bool dual_energy_closure = m_ion_closure == "dual_energy";
    const bool total_energy_closure =
        m_ion_closure == "total_energy" || dual_energy_closure;
    const bool cgl_closure = m_ion_closure == "cgl";
    const AdmissibilityBounds bounds = MakeAdmissibilityBounds();
    const amrex::Real slack = 2.0e-6_rt;
    const amrex::Real density_image =
        m_mass_density_floor * (1.0_rt + slack);
    // Wall-reservoir pressure at the clamp density (temperature mode):
    // p_wall = n_image kB T_wall = rho_image (q/m) T_wall[eV] for the
    // quasi-neutral single-ion fluid (both conduction channels use the
    // same reservoir).
    const amrex::Real wall_pressure =
        (m_wall_mask.GetThermalBC() ==
         ImplicitMHDWallMask::ThermalBC::temperature)
            ? density_image * m_ion_charge_to_mass *
                  m_wall_mask.WallTemperature_eV()
            : 0.0_rt;
    const auto image_of = [&] (const amrex::Real floor,
                               const amrex::Real coefficient,
                               const amrex::Real wall_bound) {
        return std::max({floor, coefficient * density_image, wall_bound}) *
               (1.0_rt + slack);
    };
    const amrex::Real electron_image = image_of(
        bounds.floors[1], bounds.temperature_coefficients[1],
        wall_pressure / (m_gamma_e - 1.0_rt));
    // Zero momentum, so the total-energy image is purely internal; the
    // CGL blocks are internal energies natively (U_par bounds at p/2,
    // U_perp at p).
    const amrex::Real ion_image = image_of(
        bounds.floors[2], bounds.temperature_coefficients[2],
        cgl_closure ? 0.5_rt * wall_pressure
                    : wall_pressure / (m_gamma_i - 1.0_rt));
    const amrex::Real ion_perp_image = image_of(
        bounds.floors[3], bounds.temperature_coefficients[3],
        wall_pressure);

    amrex::MultiFab& density_block =
        m_state.getMultiFabBlock(MassDensityName, 0);
    amrex::MultiFab& momentum_block =
        m_state.getMultiFabBlock(MomentumDensityName, 0);
    amrex::MultiFab& electron_energy_block =
        m_state.getMultiFabBlock(ElectronEnergyName, 0);
    amrex::MultiFab* const ion_energy_block =
        total_energy_closure ? &m_state.getMultiFabBlock(IonEnergyName, 0)
                             : nullptr;
    amrex::MultiFab* const ion_parallel_block =
        cgl_closure ? &m_state.getMultiFabBlock(IonParallelEnergyName, 0)
                    : nullptr;
    amrex::MultiFab* const ion_perp_block =
        cgl_closure ? &m_state.getMultiFabBlock(IonPerpEnergyName, 0)
                    : nullptr;
    // Dual-energy: the auxiliary U_i image is the SAME internal value as
    // the (zero-momentum, hence purely internal) E_i image, so the
    // clamped band is blend-consistent by construction.
    amrex::MultiFab* const ion_internal_block =
        dual_energy_closure
            ? &m_state.getMultiFabBlock(IonInternalEnergyName, 0)
            : nullptr;

    const int* const AMREX_RESTRICT wall_fm =
        m_wall_mask.FirstMaskedCellCentered();
    const int wall_mz_lo = -m_wall_mask.GhostCells();
    const int wall_mz_hi =
        m_wall_mask.AxialCells() - 1 + m_wall_mask.GhostCells();
    const amrex::Real radial_lower = m_WarpX->Geom(0).ProbLo(0);
    const amrex::Real dr = m_WarpX->Geom(0).CellSize(0);
    const amrex::Real dz = m_WarpX->Geom(0).CellSize(1);
    const amrex::Real two_pi_dr_dz = 2.0_rt * MathConst::pi * dr * dz;

    // Scrape banner reduction: clamped-cell count (any component
    // actually changed; bit-zero on a restarted state that already
    // carries the clamp) and the NET mass removed, with the exact
    // 2 pi r dr dz cell volumes.
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<amrex::Real, amrex::Long> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    for (amrex::MFIter mfi(density_block); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto rho = density_block.array(mfi);
        const auto mom = momentum_block.array(mfi);
        const auto energy = electron_energy_block.array(mfi);
        const auto ion_e = ion_energy_block
                               ? ion_energy_block->array(mfi)
                               : amrex::Array4<amrex::Real>{};
        const auto ion_par = ion_parallel_block
                                 ? ion_parallel_block->array(mfi)
                                 : amrex::Array4<amrex::Real>{};
        const auto ion_perp = ion_perp_block
                                  ? ion_perp_block->array(mfi)
                                  : amrex::Array4<amrex::Real>{};
        const auto ion_int = ion_internal_block
                                 ? ion_internal_block->array(mfi)
                                 : amrex::Array4<amrex::Real>{};
        reduce_op.eval(
            box, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                const int jc =
                    std::max(wall_mz_lo, std::min(wall_mz_hi, j));
                if (i < wall_fm[jc]) {
                    return {0.0_rt, amrex::Long(0)};
                }
                const amrex::Real volume =
                    two_pi_dr_dz * (radial_lower + (i + 0.5_rt) * dr);
                const amrex::Real mass_removed =
                    (rho(i, j, k) - density_image) * volume;
                bool changed = rho(i, j, k) != density_image;
                rho(i, j, k) = density_image;
                for (int component = 0; component < 3; ++component) {
                    changed = changed ||
                              (mom(i, j, k, component) != 0.0_rt);
                    mom(i, j, k, component) = 0.0_rt;
                }
                changed = changed || (energy(i, j, k) != electron_image);
                energy(i, j, k) = electron_image;
                if (ion_e) {
                    changed = changed || (ion_e(i, j, k) != ion_image);
                    ion_e(i, j, k) = ion_image;
                    if (ion_int) {
                        changed = changed ||
                                  (ion_int(i, j, k) != ion_image);
                        ion_int(i, j, k) = ion_image;
                    }
                } else if (ion_par) {
                    changed = changed ||
                              (ion_par(i, j, k) != ion_image) ||
                              (ion_perp(i, j, k) != ion_perp_image);
                    ion_par(i, j, k) = ion_image;
                    ion_perp(i, j, k) = ion_perp_image;
                }
                return {mass_removed, amrex::Long(changed ? 1 : 0)};
            });
    }
    auto sums = reduce_data.value(reduce_op);
    amrex::Real mass_removed = amrex::get<0>(sums);
    amrex::Long cells_clamped = amrex::get<1>(sums);
    amrex::ParallelAllReduce::Sum(mass_removed,
                                  amrex::ParallelContext::CommunicatorSub());
    amrex::ParallelAllReduce::Sum(cells_clamped,
                                  amrex::ParallelContext::CommunicatorSub());

    const auto& periodicity = m_WarpX->Geom(0).periodicity();
    density_block.FillBoundaryAndSync(periodicity);
    momentum_block.FillBoundaryAndSync(periodicity);
    electron_energy_block.FillBoundaryAndSync(periodicity);
    if (ion_energy_block) {
        ion_energy_block->FillBoundaryAndSync(periodicity);
        if (ion_internal_block) {
            ion_internal_block->FillBoundaryAndSync(periodicity);
        }
    } else if (ion_parallel_block) {
        ion_parallel_block->FillBoundaryAndSync(periodicity);
        ion_perp_block->FillBoundaryAndSync(periodicity);
    }
    m_state.CopyMultiFabBlocksToFields();

    amrex::Print().SetPrecision(17)
        << "ThetaImplicitMHD: shaped-wall exterior clamp: "
        << cells_clamped << " cells scraped to the rigid vacuum image, "
        << "net mass removed [kg] = " << mass_removed << "\n";
    if (!m_wall_ledger_file.empty() &&
        amrex::ParallelDescriptor::IOProcessor()) {
        // Header row of the shaped-wall ledger ('#'-comment, so the
        // "step mass energy" consumers skip it): the t = 0 scrape,
        // machine-readable for the ctest gate. First write of the run,
        // so it owns the truncation.
        std::ofstream ledger(m_wall_ledger_file,
                             m_wall_ledger_started ? std::ios::app
                                                   : std::ios::trunc);
        m_wall_ledger_started = true;
        ledger.precision(17);
        ledger << "# exterior clamp: cells " << cells_clamped
               << " mass_removed_kg " << mass_removed << "\n";
    }
#endif
}

void ThetaImplicitMHD::RefreshHaloPedestal ()
{
    if (m_halo_pedestal_fraction <= 0.0_rt) {
        return;
    }
    const bool dual_energy_closure = m_ion_closure == "dual_energy";
    const bool total_energy_closure =
        m_ion_closure == "total_energy" || dual_energy_closure;
    const bool cgl_closure = m_ion_closure == "cgl";
    amrex::MultiFab& density_block = m_state.getMultiFabBlock(MassDensityName, 0);
    amrex::MultiFab& electron_energy_block =
        m_state.getMultiFabBlock(ElectronEnergyName, 0);
    // Dynamic pedestal STATE -- an f-scaled image of the instantaneous
    // peak state, one value per fluid block; the density is never below
    // the same fraction of the reference density (so a globally decaying
    // state cannot drag the pedestal into the positivity guard). A few
    // global reductions per step; FROZEN for the whole nonlinear solve
    // so the drain-gate anchors and the source taper are per-solve
    // constants (no O(1/width) Jacobian coupling through the pedestal
    // itself).
    const amrex::Real density_peak = density_block.max(0);
    m_halo_pedestal_density =
        m_halo_pedestal_fraction *
        std::max(density_peak, m_reference_mass_density);
    m_halo_pedestal_electron_energy =
        m_halo_pedestal_fraction * electron_energy_block.max(0);
    m_halo_pedestal_ion_internal = 0.0_rt;
    m_halo_pedestal_ion_parallel = 0.0_rt;
    m_halo_pedestal_ion_perp = 0.0_rt;
    const amrex::Real pedestal = m_halo_pedestal_density;
    if (total_energy_closure) {
        // Internal-energy peak: E_i is conservative, so subtract the
        // kinetic part cell by cell before reducing.
        const amrex::MultiFab& momentum_block =
            m_state.getMultiFabBlock(MomentumDensityName, 0);
        const amrex::MultiFab& ion_energy_block =
            m_state.getMultiFabBlock(IonEnergyName, 0);
        const amrex::Real density_floor = m_mass_density_floor;
        amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (amrex::MFIter mfi(ion_energy_block); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto rho = density_block.const_array(mfi);
            const auto mom = momentum_block.const_array(mfi);
            const auto ion_e = ion_energy_block.const_array(mfi);
            reduce_op.eval(
                box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                    amrex::Real kinetic_energy = 0.0_rt;
                    for (int component = 0; component < 3; ++component) {
                        kinetic_energy += mom(i, j, k, component) *
                                          mom(i, j, k, component);
                    }
                    kinetic_energy *=
                        0.5_rt / std::max(rho(i, j, k), density_floor);
                    return {ion_e(i, j, k) - kinetic_energy};
                });
        }
        amrex::Real internal_peak =
            amrex::get<0>(reduce_data.value(reduce_op));
        amrex::ParallelAllReduce::Max(
            internal_peak, amrex::ParallelContext::CommunicatorSub());
        m_halo_pedestal_ion_internal =
            m_halo_pedestal_fraction * std::max(internal_peak, 0.0_rt);
    } else if (cgl_closure) {
        m_halo_pedestal_ion_parallel =
            m_halo_pedestal_fraction *
            m_state.getMultiFabBlock(IonParallelEnergyName, 0).max(0);
        m_halo_pedestal_ion_perp =
            m_halo_pedestal_fraction *
            m_state.getMultiFabBlock(IonPerpEnergyName, 0).max(0);
    }
    if (density_block.min(0) >= pedestal) {
        // Nothing sub-pedestal: the per-block drain gates held the band
        // (they anchor at the CURRENT pedestal values each solve, so a
        // slowly moving pedestal needs no re-raise).
        return;
    }
    // Raise every sub-pedestal cell ONTO the pedestal STATE: density to
    // rho_ped and, in the same (pre-raise sub-pedestal density) band,
    // each energy block to at least its pedestal value -- pedestal
    // plasma carries pedestal-consistent energies, never floor energies.
    // A pedestal band whose energies rest on the sanitize floors is the
    // same bound-resident population in different variables (measured:
    // the mass-only pedestal left 16k U_e + 3k E_i components projected
    // per solve and the line search frozen from step 2). The drain
    // gates keep the band dynamically invariant afterwards, so this
    // raise re-triggers only where the dynamic pedestal itself rose
    // (peak growth under compression) -- the injected mass/energy is
    // the same class of tracked non-conservation as the positivity
    // floors, confined to sub-pedestal halo cells. Landing exactly ON
    // the pedestal is safe (unlike the admissibility bounds, which get
    // a slack margin): the pedestal is an RHS-level gate anchor, not a
    // constraint of the bounded Newton solve -- the admissibility
    // bounds stay at the far-lower positivity floors, so raised cells
    // are interior points with full two-sided probe headroom.
    amrex::MultiFab& momentum_block =
        m_state.getMultiFabBlock(MomentumDensityName, 0);
    // The ion energy blocks exist in the solver state only under their
    // respective closures.
    amrex::MultiFab* const ion_energy_block =
        total_energy_closure ? &m_state.getMultiFabBlock(IonEnergyName, 0)
                             : nullptr;
    amrex::MultiFab* const ion_parallel_block =
        cgl_closure ? &m_state.getMultiFabBlock(IonParallelEnergyName, 0)
                    : nullptr;
    amrex::MultiFab* const ion_perp_block =
        cgl_closure ? &m_state.getMultiFabBlock(IonPerpEnergyName, 0)
                    : nullptr;
    // Dual-energy: the auxiliary U_i raises onto the SAME internal
    // pedestal image as E_i's internal part.
    amrex::MultiFab* const ion_internal_block =
        dual_energy_closure
            ? &m_state.getMultiFabBlock(IonInternalEnergyName, 0)
            : nullptr;
    const amrex::Real electron_energy_pedestal =
        m_halo_pedestal_electron_energy;
    const amrex::Real ion_internal_pedestal = m_halo_pedestal_ion_internal;
    const amrex::Real ion_parallel_pedestal = m_halo_pedestal_ion_parallel;
    const amrex::Real ion_perp_pedestal = m_halo_pedestal_ion_perp;
    // The rigid-conductor band is NOT fluid: under an active thermal
    // wall the masked cells keep their frozen load state, so the raise
    // must skip them -- without this the refresh repopulates the band
    // onto the peak-keyed pedestal image every step (measured on the
    // recalibrated formation ladder: the band tracked ~36 eV instead
    // of its load value; bounded but a contract violation).
    const bool wall_freeze = m_wall_mask.GetThermalBC() !=
                             ImplicitMHDWallMask::ThermalBC::none;
    const int* const AMREX_RESTRICT wall_fm =
        wall_freeze ? m_wall_mask.FirstMaskedCellCentered() : nullptr;
    const int wall_mz_lo = -m_wall_mask.GhostCells();
    const int wall_mz_hi =
        m_wall_mask.AxialCells() - 1 + m_wall_mask.GhostCells();
    for (amrex::MFIter mfi(density_block); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto rho = density_block.array(mfi);
        const auto mom = momentum_block.const_array(mfi);
        const auto energy = electron_energy_block.array(mfi);
        const auto ion_e = ion_energy_block
                               ? ion_energy_block->array(mfi)
                               : amrex::Array4<amrex::Real>{};
        const auto ion_par = ion_parallel_block
                                 ? ion_parallel_block->array(mfi)
                                 : amrex::Array4<amrex::Real>{};
        const auto ion_perp = ion_perp_block
                                  ? ion_perp_block->array(mfi)
                                  : amrex::Array4<amrex::Real>{};
        const auto ion_int = ion_internal_block
                                 ? ion_internal_block->array(mfi)
                                 : amrex::Array4<amrex::Real>{};
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if (wall_freeze) {
                const int jz = std::max(wall_mz_lo,
                                        std::min(wall_mz_hi, j));
                if (i >= wall_fm[jz]) {
                    return;
                }
            }
            if (rho(i, j, k) >= pedestal) {
                return;
            }
            rho(i, j, k) = pedestal;
            energy(i, j, k) =
                std::max(energy(i, j, k), electron_energy_pedestal);
            if (total_energy_closure) {
                // Pedestal E_i = kinetic part (at the RAISED density;
                // momentum is untouched) + the internal pedestal.
                amrex::Real kinetic_energy = 0.0_rt;
                for (int component = 0; component < 3; ++component) {
                    kinetic_energy += mom(i, j, k, component) *
                                      mom(i, j, k, component);
                }
                kinetic_energy *= 0.5_rt / pedestal;
                ion_e(i, j, k) =
                    std::max(ion_e(i, j, k),
                             kinetic_energy + ion_internal_pedestal);
                if (ion_int) {
                    ion_int(i, j, k) = std::max(ion_int(i, j, k),
                                                ion_internal_pedestal);
                }
            } else if (cgl_closure) {
                ion_par(i, j, k) =
                    std::max(ion_par(i, j, k), ion_parallel_pedestal);
                ion_perp(i, j, k) =
                    std::max(ion_perp(i, j, k), ion_perp_pedestal);
            }
        });
    }
    const auto& periodicity = m_WarpX->Geom(0).periodicity();
    density_block.FillBoundaryAndSync(periodicity);
    electron_energy_block.FillBoundaryAndSync(periodicity);
    if (total_energy_closure) {
        ion_energy_block->FillBoundaryAndSync(periodicity);
        if (ion_internal_block) {
            ion_internal_block->FillBoundaryAndSync(periodicity);
        }
    } else if (cgl_closure) {
        ion_parallel_block->FillBoundaryAndSync(periodicity);
        ion_perp_block->FillBoundaryAndSync(periodicity);
    }
    // Mirror the raise into the registered field MultiFabs: the ghosted
    // beginning-of-step copies (and the first residual's sources) are
    // built from those.
    m_state.CopyMultiFabBlocksToFields();
}

void ThetaImplicitMHD::ApplyDensityEater (const int step)
{
    if (m_density_eater_rate <= 0.0_rt) {
        return;
    }
    if (!m_evolve_ion_fluid) {
        // The reference code's fluid = 0 skips step_en entirely (ntb.f90:71-75): the
        // eater never runs on a frozen fluid.
        return;
    }

    // ---- Target: target_fraction x max(base, peak_fraction x step-old
    // global density peak) -- the reference code's en0/100 with the en0_upd = 1
    // dynamic reference en0 = MAX(en00, 0.1 MAXVAL(en)) refreshed at
    // step start (step.f90:218-225; m_state_old still holds the
    // step-start state here).
    const amrex::Real reference_base =
        m_density_eater_reference_density > 0.0_rt
            ? m_density_eater_reference_density
            : m_reference_mass_density;
    amrex::Real reference = reference_base;
    if (m_density_eater_reference_peak_fraction > 0.0_rt) {
        // MultiFab::max is an all-rank reduction.
        const amrex::Real density_peak =
            m_state_old.getMultiFabBlock(MassDensityName, 0).max(0);
        reference =
            std::max(reference_base,
                     m_density_eater_reference_peak_fraction * density_peak);
    }
    const amrex::Real target = m_density_eater_target_fraction * reference;

    // ---- Band (the reference code's ntb.f90:782-792, vertex planes mapped to cell
    // planes): z_lo = the first 2 + nz/500 planes at the mirror plane
    // (sym_bc = 1: ngrd = 1 + nz/500, planes 1 .. 1 + ngrd), z_center =
    // the planes within 1 + nz/1000 of the domain z-center.
    const amrex::Box domain = m_WarpX->Geom(0).Domain();
#if defined(WARPX_DIM_1D_Z)
    constexpr int z_dir = 0;
#else
    constexpr int z_dir = 1;
#endif
    const int axial_cells = domain.length(z_dir);
    int band_lo = domain.smallEnd(z_dir);
    int band_hi = domain.bigEnd(z_dir);
    if (m_density_eater_band == "z_lo") {
        const int band_cells = m_density_eater_band_cells > 0
                                   ? m_density_eater_band_cells
                                   : 2 + axial_cells / 500;
        band_hi = band_lo + band_cells - 1;
    } else {
        const int half_width = m_density_eater_band_cells > 0
                                   ? m_density_eater_band_cells
                                   : 1 + axial_cells / 1000;
        const int center = domain.smallEnd(z_dir) + axial_cells / 2;
        band_lo = center - half_width;
        band_hi = center + half_width;
    }
    band_lo = std::max(band_lo, domain.smallEnd(z_dir));
    band_hi = std::min(band_hi, domain.bigEnd(z_dir));
    amrex::Box band_box = domain;
    band_box.setSmall(z_dir, band_lo);
    band_box.setBig(z_dir, band_hi);

    amrex::MultiFab& density_block =
        m_state.getMultiFabBlock(MassDensityName, 0);

#if defined(WARPX_DIM_RZ)
    // ---- Closed-flux gate (the reference code's psi(k) > 0, ntb.f90:799): psi(r, z)
    // = integral_0^r Bz r' dr' per band plane, from the theta-stage
    // TOTAL Bz (Bfield_fp holds the plasma response under the
    // split-field drive, so the stored external Bz is added back;
    // The reference code gates on the STEP-OLD psi -- the eater runs in step_en
    // before pstep -- so an intra-step field stage is faithful). The
    // radial prefix sum spans ranks: gather the per-cell contributions
    // to a host profile, reduce, and integrate to the cell centers.
    amrex::Gpu::DeviceVector<amrex::Real> psi_gate_device;
    const amrex::Real* psi_gate = nullptr;
    const int band_planes = band_hi - band_lo + 1;
    const int radial_lo = domain.smallEnd(0);
    const int radial_cells = domain.length(0);
    if (m_density_eater_flux_sign != 0) {
        using ablastr::fields::Direction;
        amrex::Gpu::DeviceVector<amrex::Real> contribution_device(
            static_cast<std::size_t>(radial_cells) * band_planes, 0.0_rt);
        amrex::Real* const contribution_ptr = contribution_device.data();
        const amrex::MultiFab& bz_face =
            *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{2}, 0);
        const amrex::MultiFab* const bz_external =
            m_hybrid_pic_model->m_add_external_fields
                ? m_WarpX->m_fields.get(FieldType::hybrid_B_fp_external,
                                        Direction{2}, 0)
                : nullptr;
        const amrex::Real radial_lower = m_WarpX->Geom(0).ProbLo(0);
        const amrex::Real radial_cell_size = m_WarpX->Geom(0).CellSize(0);
        const int gate_band_lo = band_lo;
        const int gate_band_planes = band_planes;
        const int gate_radial_lo = radial_lo;
        for (amrex::MFIter mfi(density_block); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox() & band_box;
            if (box.isEmpty()) {
                continue;
            }
            const auto bz = bz_face.const_array(mfi);
            const auto bz_ext = bz_external
                                    ? bz_external->const_array(mfi)
                                    : amrex::Array4<const amrex::Real>{};
            amrex::ParallelFor(box,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    // Cell-centered total Bz from the z-face pair.
                    amrex::Real bz_center =
                        0.5_rt * (bz(i, j, k) + bz(i, j + 1, k));
                    if (bz_ext) {
                        bz_center += 0.5_rt * (bz_ext(i, j, k) +
                                               bz_ext(i, j + 1, k));
                    }
                    const amrex::Real r_center =
                        radial_lower +
                        (i - gate_radial_lo + 0.5_rt) * radial_cell_size;
                    contribution_ptr[(i - gate_radial_lo) *
                                         gate_band_planes +
                                     (j - gate_band_lo)] =
                        bz_center * r_center * radial_cell_size;
                });
        }
        amrex::Vector<amrex::Real> contribution(
            static_cast<std::size_t>(radial_cells) * band_planes, 0.0_rt);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                         contribution_device.begin(),
                         contribution_device.end(), contribution.begin());
        amrex::ParallelAllReduce::Sum(
            contribution.data(), static_cast<int>(contribution.size()),
            amrex::ParallelContext::CommunicatorSub());
        amrex::Vector<amrex::Real> psi(contribution.size(), 0.0_rt);
        const amrex::Real flux_sign =
            static_cast<amrex::Real>(m_density_eater_flux_sign);
        for (int plane = 0; plane < band_planes; ++plane) {
            amrex::Real accumulated = 0.0_rt;
            for (int ir = 0; ir < radial_cells; ++ir) {
                const amrex::Real cell_flux =
                    contribution[static_cast<std::size_t>(ir) * band_planes +
                                 plane];
                psi[static_cast<std::size_t>(ir) * band_planes + plane] =
                    flux_sign * (accumulated + 0.5_rt * cell_flux);
                accumulated += cell_flux;
            }
        }
        psi_gate_device.resize(psi.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, psi.begin(), psi.end(),
                         psi_gate_device.begin());
        psi_gate = psi_gate_device.data();
    }
#endif

    // ---- Eater pass, fused with the removal bookkeeping. The reference code
    // ntb.f90:794-805:
    //     en_lim = 0.8 MAX(en, en0/100) + 0.2 (en0/100)
    //     en     = MIN(en, en_lim)
    // with ONLY the density state touched: velocities are the reference code's
    // momentum state (untouched), so the momentum DENSITY scales with
    // the mass; wio/wik are untouched energy DENSITIES (the reference shot
    // mix = -1 closure), so E_i, U_i, and the CGL pair are invariant
    // (the per-particle ion temperature RISES as the removed mass gives
    // up nothing); te is the reference code's electron state (flg_wie false), so
    // U_e scales with the mass (electron-temperature-preserving).
    amrex::MultiFab& momentum_block =
        m_state.getMultiFabBlock(MomentumDensityName, 0);
    amrex::MultiFab& electron_energy_block =
        m_state.getMultiFabBlock(ElectronEnergyName, 0);
    const amrex::Real rate = m_density_eater_rate;
    const amrex::Real electron_energy_floor =
        m_electron_pressure_floor / (m_gamma_e - 1.0_rt);
    // Rigid-conductor wall exclusion (the reference code's x_kind < 1 CYCLE at
    // ntb.f90:798: masked cells are not fluid).
    const bool wall_freeze = m_wall_mask.GetThermalBC() !=
                             ImplicitMHDWallMask::ThermalBC::none;
    const int* const AMREX_RESTRICT wall_fm =
        wall_freeze ? m_wall_mask.FirstMaskedCellCentered() : nullptr;
    const int wall_mz_lo = -m_wall_mask.GhostCells();
    const int wall_mz_hi =
        m_wall_mask.AxialCells() - 1 + m_wall_mask.GhostCells();
    // Geometry cell measure (booked units kg and J; per unit
    // cross-section in 1D), the floor-ledger convention.
    amrex::Real cell_volume = 1.0_rt;
    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
        cell_volume *= m_WarpX->Geom(0).CellSize(dim);
    }
#if defined(WARPX_DIM_RZ)
    const amrex::Real ledger_radial_lower = m_WarpX->Geom(0).ProbLo(0);
    const amrex::Real ledger_radial_cell_size = m_WarpX->Geom(0).CellSize(0);
#endif
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    for (amrex::MFIter mfi(density_block); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox() & band_box;
        if (box.isEmpty()) {
            continue;
        }
        const auto rho = density_block.array(mfi);
        const auto mom = momentum_block.array(mfi);
        const auto electron_energy = electron_energy_block.array(mfi);
        reduce_op.eval(
            box, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                if (wall_freeze) {
                    const int jc =
                        std::max(wall_mz_lo, std::min(wall_mz_hi, j));
                    if (i >= wall_fm[jc]) {
                        return {0.0_rt, 0.0_rt};
                    }
                }
#if defined(WARPX_DIM_RZ)
                if (psi_gate != nullptr &&
                    psi_gate[(i - radial_lo) * band_planes +
                             (j - band_lo)] <= 0.0_rt) {
                    return {0.0_rt, 0.0_rt};
                }
#endif
                const amrex::Real density_old = rho(i, j, k);
                const amrex::Real density_limit =
                    (1.0_rt - rate) * std::max(density_old, target) +
                    rate * target;
                if (density_old <= density_limit) {
                    // At or below target: the MIN leaves the cell alone.
                    return {0.0_rt, 0.0_rt};
                }
                const amrex::Real scale = density_limit / density_old;
                rho(i, j, k) = density_limit;
                for (int component = 0; component < 3; ++component) {
                    mom(i, j, k, component) *= scale;
                }
                const amrex::Real electron_old = electron_energy(i, j, k);
                amrex::Real electron_new = scale * electron_old;
                // Keep the absolute electron positivity floor with the
                // standard slack margin (the temperature-proportional
                // bound scales with rho, so it is preserved by scale).
                electron_new = std::max(
                    electron_new,
                    electron_energy_floor +
                        1.0e-6_rt * (electron_new + electron_energy_floor));
                electron_energy(i, j, k) = electron_new;
                amrex::Real measure = 1.0_rt;
#if defined(WARPX_DIM_RZ)
                measure = 2.0_rt * MathConst::pi *
                          (ledger_radial_lower +
                           (i - radial_lo + 0.5_rt) *
                               ledger_radial_cell_size);
#endif
                return {measure * (density_old - density_limit),
                        measure * (electron_old - electron_new)};
            });
    }
    ReduceTuple removal_totals = reduce_data.value(reduce_op);
    amrex::Real step_totals[2] = {
        cell_volume * amrex::get<0>(removal_totals),
        cell_volume * amrex::get<1>(removal_totals)};
    amrex::ParallelAllReduce::Sum(step_totals, 2,
                                  amrex::ParallelContext::CommunicatorSub());
    m_eater_removed_mass += step_totals[0];
    m_eater_removed_energy += step_totals[1];

    const auto& periodicity = m_WarpX->Geom(0).periodicity();
    density_block.FillBoundaryAndSync(periodicity);
    momentum_block.FillBoundaryAndSync(periodicity);
    electron_energy_block.FillBoundaryAndSync(periodicity);

    if (!m_density_eater_ledger_file.empty() &&
        amrex::ParallelDescriptor::IOProcessor()) {
        // Truncate at the first write of the run, append afterwards
        // (the floor-ledger convention).
        std::ofstream ledger(m_density_eater_ledger_file,
                             m_density_eater_ledger_started
                                 ? std::ios::app
                                 : std::ios::trunc);
        m_density_eater_ledger_started = true;
        ledger.precision(17);
        ledger << step + 1 << " " << m_eater_removed_mass << " "
               << m_eater_removed_energy << "\n";
    }
}

void ThetaImplicitMHD::FinishStateUpdate (const amrex::Real end_time, const int step)
{
    const amrex::Real inverse_theta = 1.0_rt / m_theta;
    m_state.linComb(inverse_theta, m_state, 1.0_rt - inverse_theta, m_state_old);

    // Shaped-wall FIELD freeze: carry the frozen exterior faces through
    // the theta -> t^{n+1} extrapolation bit-exactly. The converged state
    // already holds B^theta = B^n there (exact identity rows), and the
    // linComb above maps identical values back to themselves bit-exactly
    // for theta in {1/2, 1} -- but can round an ULP at other theta
    // values, a per-step ratchet on faces that must stay bit-static.
    // Restore the step-old values outright.
    {
        const warpx::mhd_pc::WallFieldFreezeView field_freeze =
            m_wall_mask.FieldFreezeView();
        if (field_freeze.active) {
            const amrex::Array<const int*, 3> first_frozen = {
                field_freeze.first_frozen_br, field_freeze.first_frozen_bt,
                field_freeze.first_frozen_bz};
            for (int component = 0; component < 3; ++component) {
                amrex::MultiFab& face_block =
                    *m_state.getArrayVec()[0][component];
                const amrex::MultiFab& face_block_old =
                    *m_state_old.getArrayVec()[0][component];
                const int* const AMREX_RESTRICT first =
                    first_frozen[component];
                for (amrex::MFIter mfi(face_block); mfi.isValid(); ++mfi) {
                    const auto face = face_block.array(mfi);
                    const auto face_old = face_block_old.const_array(mfi);
                    amrex::ParallelFor(mfi.validbox(),
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                            if (i >= first[j]) {
                                face(i, j, k) = face_old(i, j, k);
                            }
                        });
                }
            }
        }
    }

    // The reference code's density eater (see ApplyDensityEater): runs on the
    // committed t^{n+1} state BEFORE the floor restorations, matching
    // The reference code's order within a step -- density_eater at the end of
    // step_en (ntb.f90:202), then the mixmaster energy/temperature
    // update and the step_tm floors on the eaten density. The
    // restorations below therefore key their density-dependent bounds
    // to the eaten density, exactly the state the next solve freezes.
    ApplyDensityEater(step);

    // End-of-step floor restorations below evaluate the temperature
    // floors' density-dependent bounds with the END-OF-STEP density: it is
    // the next solve's frozen step-old density, so restoring onto ITS
    // bound guarantees the next Newton solve starts admissible even where
    // the density rose during this step (the in-solve bounds, frozen at
    // this step's old density, cannot see that rise). Like the in-solve
    // bounds, the temperature part is a one-way RATCHET, gated per cell
    // on the STEP-OLD value satisfying the step-old bound: cells that
    // were already colder than the floor (e.g. colder-than-floor initial
    // data) are never lifted -- the floor prevents cooling through it,
    // it does not inject heat.
    const AdmissibilityBounds bounds = MakeAdmissibilityBounds();

    // Shaped-wall band exclusion (the band-hygiene contract, see
    // ClampWallExteriorState): the masked band is not fluid -- it sits
    // bit-exactly on the rigid vacuum clamp image, whose values rest a
    // fixed slack above their bounds. The restorations below re-land a
    // per-cell margin computed FROM the current value, which on a
    // bound-resident cell is strictly increasing (bound + 1e-6 (|v| +
    // bound) > v at v = bound (1 + 2e-6)): without the skip the band
    // would ratchet upward a few ULP-scale increments per step, i.e.
    // re-create energy outside the wall and break the bit-static
    // contract.
    const bool wall_freeze = m_wall_mask.GetThermalBC() !=
                             ImplicitMHDWallMask::ThermalBC::none;
    const int* const AMREX_RESTRICT wall_fm =
        wall_freeze ? m_wall_mask.FirstMaskedCellCentered() : nullptr;
    const int wall_mz_lo = -m_wall_mask.GhostCells();
    const int wall_mz_hi =
        m_wall_mask.AxialCells() - 1 + m_wall_mask.GhostCells();

    if (m_electron_temperature_floor > 0.0_rt) {
        // Electron-temperature floor restoration at the accepted step end:
        // U_e >= max(p_e_floor, n kB T_e_floor)/(gamma_e - 1). The
        // restoration lands the same 1e-6 (|value| + bound) SLACK MARGIN
        // above the bound as the direction projection, never exactly on
        // it (see the CGL restoration below for why).
        const amrex::MultiFab& density_block =
            m_state.getMultiFabBlock(MassDensityName, 0);
        const amrex::MultiFab& old_density_block =
            m_state_old.getMultiFabBlock(MassDensityName, 0);
        const amrex::MultiFab& old_energy_block =
            m_state_old.getMultiFabBlock(ElectronEnergyName, 0);
        amrex::MultiFab& electron_energy_block =
            m_state.getMultiFabBlock(ElectronEnergyName, 0);
        const amrex::Real energy_per_density =
            bounds.temperature_coefficients[1];
        for (amrex::MFIter mfi(electron_energy_block); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto rho = density_block.const_array(mfi);
            const auto rho_old = old_density_block.const_array(mfi);
            const auto energy_old = old_energy_block.const_array(mfi);
            const auto energy_array = electron_energy_block.array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                if (wall_freeze) {
                    // Wall band exclusion (see the capture comment).
                    const int jc =
                        std::max(wall_mz_lo, std::min(wall_mz_hi, j));
                    if (i >= wall_fm[jc]) {
                        return;
                    }
                }
                if (energy_old(i, j, k) <
                    energy_per_density * rho_old(i, j, k)) {
                    return; // ratchet: was below the floor, do not lift
                }
                // Only the temperature part needs restoring: the CONSTANT
                // absolute floor survives the theta extrapolation on its
                // own, so it is left untouched here (near-absolute-floor
                // cells keep their pre-temperature-floor behavior).
                const amrex::Real cell_floor =
                    energy_per_density * rho(i, j, k);
                const amrex::Real margin =
                    1.0e-6_rt *
                    (std::abs(energy_array(i, j, k)) + cell_floor);
                energy_array(i, j, k) = std::max(energy_array(i, j, k),
                                                 cell_floor + margin);
            });
        }
        electron_energy_block.FillBoundaryAndSync(
            m_WarpX->Geom(0).periodicity());
    }

    if (m_ion_closure == "total_energy" || m_ion_closure == "dual_energy") {
        // Dual-energy synchronization (the standard sync step, without the
        // auxiliary internal-energy equation): in kinetic-dominated cells
        // (FRC end jets) the conservative E_i can drift below KE + U_i
        // floor, where the internal energy is a meaningless small
        // difference of large numbers -- the clamped pressure then
        // decouples from the state and degrades the next Newton solve
        // until the line search fails. Restore E_i >= KE + U_i_floor at
        // every accepted step end; the added energy is the same class of
        // tracked non-conservation as the positivity floors, confined to
        // floored cells.
        const amrex::MultiFab& density_block =
            m_state.getMultiFabBlock(MassDensityName, 0);
        const amrex::MultiFab& momentum_block =
            m_state.getMultiFabBlock(MomentumDensityName, 0);
        amrex::MultiFab& ion_energy_block =
            m_state.getMultiFabBlock(IonEnergyName, 0);
        const amrex::MultiFab& old_density_block =
            m_state_old.getMultiFabBlock(MassDensityName, 0);
        const amrex::MultiFab& old_momentum_block =
            m_state_old.getMultiFabBlock(MomentumDensityName, 0);
        const amrex::MultiFab& old_ion_energy_block =
            m_state_old.getMultiFabBlock(IonEnergyName, 0);
        const amrex::Real internal_floor =
            m_ion_pressure_floor / (m_gamma_i - 1.0_rt);
        // Ion-temperature floor: the restored internal part is bounded by
        // max(p_i_floor, n kB T_i_floor)/(gamma_i - 1); the temperature
        // part is ratchet-gated on the step-old INTERNAL energy (the
        // quantity this restoration bounds) having satisfied the step-old
        // temperature bound.
        const amrex::Real internal_floor_per_density =
            bounds.temperature_coefficients[2];
        const amrex::Real density_floor = m_mass_density_floor;
        for (amrex::MFIter mfi(ion_energy_block); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto rho = density_block.const_array(mfi);
            const auto mom = momentum_block.const_array(mfi);
            const auto rho_old = old_density_block.const_array(mfi);
            const auto mom_old = old_momentum_block.const_array(mfi);
            const auto ion_e_old = old_ion_energy_block.const_array(mfi);
            const auto ion_e = ion_energy_block.array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                if (wall_freeze) {
                    // Wall band exclusion (see the capture comment).
                    const int jc =
                        std::max(wall_mz_lo, std::min(wall_mz_hi, j));
                    if (i >= wall_fm[jc]) {
                        return;
                    }
                }
                amrex::Real kinetic_energy = 0.0_rt;
                amrex::Real kinetic_energy_old = 0.0_rt;
                for (int component = 0; component < 3; ++component) {
                    kinetic_energy +=
                        mom(i, j, k, component) * mom(i, j, k, component);
                    kinetic_energy_old +=
                        mom_old(i, j, k, component) *
                        mom_old(i, j, k, component);
                }
                kinetic_energy *=
                    0.5_rt / std::max(rho(i, j, k), density_floor);
                kinetic_energy_old *=
                    0.5_rt / std::max(rho_old(i, j, k), density_floor);
                const bool ratchet_engaged =
                    ion_e_old(i, j, k) - kinetic_energy_old >=
                    internal_floor_per_density * rho_old(i, j, k);
                const amrex::Real internal_bound =
                    ratchet_engaged
                        ? std::max(internal_floor,
                                   internal_floor_per_density * rho(i, j, k))
                        : internal_floor;
                ion_e(i, j, k) = std::max(ion_e(i, j, k),
                                          kinetic_energy + internal_bound);
            });
        }
        ion_energy_block.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
        if (m_ion_closure == "dual_energy") {
            // Dual-energy re-sync at the accepted step end, the port of
            // The reference code's mixmaster temperature update (ntb.f90:102-106,
            // which rewrites BOTH wio and wik from the blended pressure
            // every step):
            //  (1) E_i := KE + p_blend/(gamma_i - 1) in EVERY live cell
            //      -- in thermal cells the blend IS the recovery and
            //      this is the identity, while in KE-dominated cells it
            //      drains the step's E_i-minus-KE cancellation drift
            //      into the (1 - fk) U_i anchor. Without this rewrite
            //      the drift ACCUMULATES in the conservative E_i across
            //      steps and the blend re-imports fk >= 1 - 1/gamma_i
            //      (~0.4) of the accumulated total each residual --
            //      measured on the cold supersonic advection gate as
            //      dual heating locked at fk x the total_energy
            //      artifact. With it, E_i's internal content is a
            //      damped AR(1) (pole fk) anchored on U_i's clean
            //      internal evolution: bounded at O(one step's
            //      truncation), the reference code's cold-axis behavior.
            //  (2) U_i := E_i - KE where fk > dual_energy_sync_threshold
            //      (thermal cells, where the recovery is well
            //      conditioned): the standard Enzo-style sync, so the
            //      auxiliary variable cannot drift in thermal regions.
            // The fk/cutoff mask inputs are the SAME per-step frozen
            // step-old values as the residual. Runs AFTER the E_i
            // restoration above; the rewrite preserves admissibility by
            // construction (p_blend >= p_i_floor through its smooth
            // floors), and the U_i floor restoration lands the standard
            // slack margin (a synced cell resting exactly on its bound
            // would otherwise abort both Jacobian probe signs).
            const theta_implicit_mhd::FluxParameters sync_parameters =
                MakeFluxParameters();
            const amrex::Real sync_threshold = m_dual_energy_sync_threshold;
            const amrex::Real inverse_gamma_i_minus_one =
                1.0_rt / (m_gamma_i - 1.0_rt);
            amrex::MultiFab& internal_block =
                m_state.getMultiFabBlock(IonInternalEnergyName, 0);
            const amrex::MultiFab& old_internal_block =
                m_state_old.getMultiFabBlock(IonInternalEnergyName, 0);
            const amrex::Real internal_abs_floor = bounds.floors[3];
            const amrex::Real internal_coefficient =
                bounds.temperature_coefficients[3];
            for (amrex::MFIter mfi(internal_block); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                const auto rho = density_block.const_array(mfi);
                const auto mom = momentum_block.const_array(mfi);
                const auto rho_old = old_density_block.const_array(mfi);
                const auto ion_e = ion_energy_block.array(mfi);
                const auto internal_old =
                    old_internal_block.const_array(mfi);
                const auto internal = internal_block.array(mfi);
                amrex::ParallelFor(
                    box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        if (wall_freeze) {
                            // Wall band exclusion (see the capture
                            // comment).
                            const int jc = std::max(
                                wall_mz_lo, std::min(wall_mz_hi, j));
                            if (i >= wall_fm[jc]) {
                                return;
                            }
                        }
                        amrex::Real kinetic_energy = 0.0_rt;
                        for (int component = 0; component < 3;
                             ++component) {
                            kinetic_energy +=
                                mom(i, j, k, component) *
                                mom(i, j, k, component);
                        }
                        kinetic_energy *=
                            0.5_rt /
                            std::max(rho(i, j, k), density_floor);
                        const amrex::Real pressure_blend =
                            theta_implicit_mhd::
                                dual_energy_blended_pressure(
                                    ion_e(i, j, k), kinetic_energy,
                                    internal(i, j, k),
                                    internal_old(i, j, k),
                                    sync_parameters);
                        // (1) mixmaster internal re-sync of E_i.
                        ion_e(i, j, k) =
                            kinetic_energy +
                            pressure_blend * inverse_gamma_i_minus_one;
                        // (2) Enzo-style thermal-cell sync of U_i.
                        const amrex::Real fk =
                            theta_implicit_mhd::
                                dual_energy_kinetic_fraction(
                                    ion_e(i, j, k), kinetic_energy,
                                    internal_old(i, j, k),
                                    sync_parameters);
                        if (fk > sync_threshold) {
                            internal(i, j, k) =
                                ion_e(i, j, k) - kinetic_energy;
                        }
                        // Floor restoration with the standard slack
                        // margin and the one-way temperature ratchet
                        // (the U_i twin of the E_i restoration above).
                        const bool ratchet_engaged =
                            internal_old(i, j, k) >=
                            internal_coefficient * rho_old(i, j, k);
                        const amrex::Real cell_floor =
                            ratchet_engaged
                                ? std::max(internal_abs_floor,
                                           internal_coefficient *
                                               rho(i, j, k))
                                : internal_abs_floor;
                        const amrex::Real margin =
                            1.0e-6_rt *
                            (std::abs(internal(i, j, k)) + cell_floor);
                        internal(i, j, k) = std::max(
                            internal(i, j, k), cell_floor + margin);
                    });
            }
            ion_energy_block.FillBoundaryAndSync(
                m_WarpX->Geom(0).periodicity());
            internal_block.FillBoundaryAndSync(
                m_WarpX->Geom(0).periodicity());
        }
    } else if (m_ion_closure == "cgl") {
        // Floor restoration at the accepted step end (round-off
        // insurance; the Newton admissibility bounds already hold the
        // extrapolated end state at or above the floors, and the CGL
        // blocks are pure internal energies so no kinetic-energy sync is
        // needed): U_par >= p_i_floor/2 and U_perp >= p_i_floor. The
        // restoration lands a SLACK MARGIN above the bound -- the same
        // 1e-6 (|value| + floor) convention as the direction projection
        // -- never exactly on it: a cell resting exactly on its floor
        // makes the value-minus-bound distance zero at the next step, and
        // once two such cells carry opposite-sign probe components BOTH
        // probe signs become inadmissible and the matrix-free Jacobian
        // aborts (measured on the bounded CGL free-boundary hold at
        // 57 t_ci, where a violent transient floored a large population).
        // The per-block floors carry the ion-temperature floor's
        // density-dependent bound: U_par >= max(p_i_floor, n kB T_i)/2
        // and U_perp >= max(p_i_floor, n kB T_i), with the temperature
        // part ratchet-gated on the step-old value.
        const amrex::MultiFab& density_block =
            m_state.getMultiFabBlock(MassDensityName, 0);
        const amrex::MultiFab& old_density_block =
            m_state_old.getMultiFabBlock(MassDensityName, 0);
        const std::array<std::tuple<const char*, amrex::Real, amrex::Real>, 2>
            blocks = {{{IonParallelEnergyName, bounds.floors[2],
                        bounds.temperature_coefficients[2]},
                       {IonPerpEnergyName, bounds.floors[3],
                        bounds.temperature_coefficients[3]}}};
        for (const auto& [block_name, floor, temperature_coefficient] :
             blocks) {
            amrex::MultiFab& energy_block =
                m_state.getMultiFabBlock(block_name, 0);
            const amrex::MultiFab& old_energy_block =
                m_state_old.getMultiFabBlock(block_name, 0);
            const amrex::Real block_floor = floor;
            const amrex::Real block_coefficient = temperature_coefficient;
            for (amrex::MFIter mfi(energy_block); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                const auto rho = density_block.const_array(mfi);
                const auto rho_old = old_density_block.const_array(mfi);
                const auto energy_old = old_energy_block.const_array(mfi);
                const auto energy_array = energy_block.array(mfi);
                amrex::ParallelFor(
                    box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        if (wall_freeze) {
                            // Wall band exclusion (see the capture
                            // comment).
                            const int jc = std::max(
                                wall_mz_lo, std::min(wall_mz_hi, j));
                            if (i >= wall_fm[jc]) {
                                return;
                            }
                        }
                        const bool ratchet_engaged =
                            energy_old(i, j, k) >=
                            block_coefficient * rho_old(i, j, k);
                        const amrex::Real cell_floor =
                            ratchet_engaged
                                ? std::max(block_floor,
                                           block_coefficient * rho(i, j, k))
                                : block_floor;
                        const amrex::Real margin =
                            1.0e-6_rt *
                            (std::abs(energy_array(i, j, k)) + cell_floor);
                        energy_array(i, j, k) =
                            std::max(energy_array(i, j, k),
                                     cell_floor + margin);
                    });
            }
            energy_block.FillBoundaryAndSync(
                m_WarpX->Geom(0).periodicity());
        }
    }

    m_state.CopyMultiFabBlocksToFields();

    const amrex::MultiFab& density = *m_WarpX->m_fields.get(MassDensityName, 0);
    const amrex::MultiFab& electron_energy =
        *m_WarpX->m_fields.get(ElectronEnergyName, 0);
    const amrex::Real density_tolerance =
        64.0_rt * std::numeric_limits<amrex::Real>::epsilon() *
        m_reference_mass_density;
    const amrex::Real energy_scale = m_reference_magnetic_field *
                                     m_reference_magnetic_field /
                                     PhysConst::mu0;
    const amrex::Real energy_tolerance =
        64.0_rt * std::numeric_limits<amrex::Real>::epsilon() * energy_scale;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        density.min(0) >= m_mass_density_floor - density_tolerance,
        "theta_implicit_mhd produced a final mass density below its positivity floor");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        electron_energy.min(0) >=
            m_electron_pressure_floor / (m_gamma_e - 1.0_rt) - energy_tolerance,
        "theta_implicit_mhd produced a final electron energy below its positivity floor");
    if (m_ion_closure == "total_energy" || m_ion_closure == "dual_energy") {
        const amrex::MultiFab& ion_energy =
            *m_WarpX->m_fields.get(IonEnergyName, 0);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            ion_energy.min(0) >=
                m_ion_pressure_floor / (m_gamma_i - 1.0_rt) - energy_tolerance,
            "theta_implicit_mhd produced a final ion total energy below its "
            "internal-energy floor");
        if (m_ion_closure == "dual_energy") {
            const amrex::MultiFab& ion_internal =
                *m_WarpX->m_fields.get(IonInternalEnergyName, 0);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                ion_internal.min(0) >=
                    m_ion_pressure_floor / (m_gamma_i - 1.0_rt) -
                        energy_tolerance,
                "theta_implicit_mhd produced a final ion internal energy "
                "below its positivity floor");
        }
    } else if (m_ion_closure == "cgl") {
        const amrex::MultiFab& parallel_energy =
            *m_WarpX->m_fields.get(IonParallelEnergyName, 0);
        const amrex::MultiFab& perp_energy =
            *m_WarpX->m_fields.get(IonPerpEnergyName, 0);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            parallel_energy.min(0) >=
                0.5_rt * m_ion_pressure_floor - energy_tolerance,
            "theta_implicit_mhd produced a final ion parallel energy below "
            "its positivity floor");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            perp_energy.min(0) >= m_ion_pressure_floor - energy_tolerance,
            "theta_implicit_mhd produced a final ion perpendicular energy "
            "below its positivity floor");
    }

    if (m_use_recast) {
        // The linComb above already extrapolated the state (including its
        // B array block) to t^{n+1}; publish it to Bfield_fp with BCs.
        m_WarpX->SetMagneticFieldAndApplyBCs(m_state, end_time);
    } else {
        const auto& magnetic_field_old =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, 0);
        m_WarpX->FinishMagneticFieldAndApplyBCs(magnetic_field_old, m_theta,
                                                end_time);
    }
    if (m_z_neumann) {
        using ablastr::fields::Direction;
        for (int direction = 0; direction < 3; ++direction) {
            amrex::MultiFab& magnetic_component =
                *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{direction}, 0);
            ApplyNeumannZDomainGhosts(magnetic_component,
                                      magnetic_component.nGrowVect()[1]);
            if (direction != 2) {
                // z_lo mirror: Br and B_theta (cell-centered in z) are
                // ODD; Bz is even nodal-in-z, exact under the Neumann
                // nodal mirror.
                ApplyMirrorZLoDomainGhosts(magnetic_component, {-1, -1, -1});
            }
        }
    }
    FillFluidSources(m_state);
    PublishMomentumComponents();

    // E is an algebraic Ohm-law variable rather than an independently evolved
    // endpoint state. Recompute it from final B, rho, momentum, and Ue instead
    // of retaining the theta extrapolation, which generally would not satisfy
    // Ohm's law at t^{n+1}.
    const auto magnetic_field =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, 0);
    m_hybrid_pic_model->CalculatePlasmaCurrent(magnetic_field,
                                               m_WarpX->GetEBUpdateEFlag());
    if (m_z_neumann) {
        using ablastr::fields::Direction;
        for (int direction = 0; direction < 3; ++direction) {
            amrex::MultiFab& current_component = *m_WarpX->m_fields.get(
                FieldType::hybrid_current_fp_plasma, Direction{direction}, 0);
            ApplyNeumannZDomainGhosts(current_component, 1);
            if (direction == 2) {
                // z_lo mirror: J_z is ODD (see ComputeRHS).
                ApplyMirrorZLoDomainGhosts(current_component, {-1, -1, -1});
            }
        }
    }

    if (m_hybrid_pic_model->m_add_external_fields) {
        if (m_external_field_iteration) {
            // Final circuit pass against the accepted end-of-step plasma
            // current, leaving the circuit state converged at t^{n+1}.
            if (m_circuit_native) {
                // The single accept = true advance of the step, over the
                // full committed interval [t^n, t^{n+1}] on the accepted
                // final state (Bfield_fp and hybrid_current_fp_plasma
                // hold the plasma-response frame here; J was just
                // recomputed above): discontinuous engine transitions
                // latch NOW, in committed time. FinishStep closes the
                // engine's per-step bookkeeping.
                CircuitCoupler& coupler = NativeCircuitCoupler();
                coupler.MeasureLinkages(false);
                if (m_circuit_step_open) {
                    coupler.EvaluateInterval(end_time - m_dt, end_time,
                                             true);
                    coupler.FinishStep();
                    m_circuit_step_open = false;
                }
            } else {
                ExecutePythonCallback("externalcoilfinish");
            }
        }
        // Refresh the external fields at t^{n+1} for the final Ohm's-law
        // recompute below. These same stored values are added back to the
        // totals at the end of this function and subtracted again at the
        // start of the next step, so the split remains exact.
        m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
            end_time, m_dt);
    }
    if (m_use_recast) {
        // Recompute the end-of-step Ohm E (diagnostics and coupling) from
        // the accepted final state: the same face-EMF assembly the
        // residual used, evaluated at t^{n+1} -- with the INSTANTANEOUS
        // current in the dissipative terms (at_resistive_stage = false),
        // so the published E satisfies Ohm's law at t^{n+1} rather than
        // retaining the intra-step resistive-stage weighting.
        FillCellCenteredElectromagneticFields();
        ComputeFaceFluxes(end_time);
        // NOTE: the end-of-step assembly reads the SAVED theta-stage
        // inertial field (never reassembled here), so the published E's
        // inertia contribution is half-step stale -- diagnostic only, the
        // B advance never consumes the published E. The history rotation
        // below must therefore run AFTER this assembly.
        AssembleOhmElectricField(end_time, false);
        if (m_hybrid_pic_model->m_include_electron_inertia) {
            // Rotate the per-step nodal Je history via the
            // theta-extrapolation of the converged theta-stage assembly
            // (one assembly family end to end: differencing across
            // assembly conventions injects spurious derivatives at 1/dt).
            m_hybrid_pic_model->RotateElectronInertiaHistory(m_theta);
        }
    } else {
        const auto electric_field =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, 0);
        const auto ion_current =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, 0);
        const auto charge_density =
            m_WarpX->m_fields.get_mr_levels(FieldType::rho_fp, 0);
        m_hybrid_pic_model->HybridPICSolveE(
            electric_field, ion_current, magnetic_field, charge_density,
            m_WarpX->GetEBUpdateEFlag(), true, true, end_time, false);
    }
    if (m_z_neumann) {
        using ablastr::fields::Direction;
        for (int direction = 0; direction < 3; ++direction) {
            amrex::MultiFab& electric_component = *m_WarpX->m_fields.get(
                FieldType::Efield_fp, Direction{direction}, 0);
            ApplyNeumannZDomainGhosts(electric_component);
            if (direction == 2) {
                // z_lo mirror: E_z is ODD (see AssembleOhmElectricField).
                ApplyMirrorZLoDomainGhosts(electric_component, {-1, -1, -1});
            }
        }
    }
    if (!m_use_recast) {
        m_state.Copy(FieldType::Efield_fp);
    }

    if (m_hybrid_pic_model->m_add_external_fields) {
        // Leave total fields in Efield_fp/Bfield_fp between steps for
        // diagnostics, checkpoints, and coupling. The solver state (copied
        // just above) and hybrid_current_fp_plasma stay plasma-response
        // only; the latter is what circuit flux-linkage diagnostics dot
        // against the coil vector potentials.
        AddExternalFieldsToTotals(1.0_rt);
    }
}
