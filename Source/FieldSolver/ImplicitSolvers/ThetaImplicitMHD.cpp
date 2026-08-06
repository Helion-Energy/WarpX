/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ThetaImplicitMHD.H"
#include "ThetaImplicitMHD_K.H"

#include "BoundaryConditions/GreensFunctionOpenBC.H"
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
#include <limits>
#include <set>
#include <string>
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
    utils::parser::queryWithParser(pp, "vacuum_mass_density", m_vacuum_mass_density);
    utils::parser::queryWithParser(pp, "vacuum_drag_rate", m_vacuum_drag_rate);
    utils::parser::queryWithParser(pp, "positivity_safety", m_positivity_safety);
    pp.query("external_field_iteration", m_external_field_iteration);
    pp.query("fluid_flux", m_fluid_flux);
    pp.query("r_open_fluid", m_r_open_fluid);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_r_open_fluid == "outflow" || m_r_open_fluid == "reflect",
        "implicit_mhd.r_open_fluid must be outflow or reflect");
    pp.query("hllc_signal_closure", m_hllc_signal_closure);
    utils::parser::queryWithParser(pp, "hllc_contact_blend", m_hllc_contact_blend);
    pp.query("ion_closure", m_ion_closure);
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
            m_fluid_flux == "hllc" || m_fluid_flux == "hlld",
        "implicit_mhd.fluid_flux must be centered, rusanov, hllc, or hlld");
    m_use_hlld = (m_fluid_flux == "hlld");
#if !defined(WARPX_DIM_1D_Z) && !defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_use_hlld,
        "implicit_mhd.fluid_flux = hlld (the conservative-form recast) "
        "currently supports 1D Cartesian and cylindrical RZ geometry only");
#endif
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
        m_ion_closure == "barotropic" || m_ion_closure == "total_energy" ||
            m_ion_closure == "cgl",
        "implicit_mhd.ion_closure must be barotropic, total_energy, or cgl");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_ion_pressure_floor >= 0.0_rt,
                                     "implicit_mhd.ion_pressure_floor cannot be negative");
    utils::parser::queryWithParser(pp, "cgl_relaxation_scale",
                                   m_cgl_relaxation_scale);
    utils::parser::queryWithParser(pp, "cgl_coulomb_log", m_cgl_coulomb_log);
    utils::parser::queryWithParser(pp, "cgl_instability_scale",
                                   m_cgl_instability_scale);
    utils::parser::queryWithParser(pp, "cgl_instability_width",
                                   m_cgl_instability_width);
    utils::parser::queryWithParser(pp, "cgl_null_scale", m_cgl_null_scale);
    pp.query("z_outflow_no_reflux", m_z_outflow_no_reflux);
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
    if (m_ion_closure == "total_energy" || m_ion_closure == "cgl") {
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

    fields.alloc_init(TotalCurrentCCName, lev, ba, dm, 3, guard_cells, 0.0_rt);
    fields.alloc_init(MagneticFieldCCName, lev, ba, dm, 3, guard_cells, 0.0_rt);

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
    if (!m_WarpX->Geom(0).isPeriodic(1)) {
        // Outflow (Neumann) axial ends: the solver fills all z domain
        // ghosts itself with zero-gradient values, so WarpX's own field
        // boundary application must stay out of the way.
        m_z_neumann = true;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::field_boundary_lo[1] == FieldBoundaryType::None &&
                WarpX::field_boundary_hi[1] == FieldBoundaryType::None,
            "theta_implicit_mhd with non-periodic z requires "
            "boundary.field_lo/hi = none in z (the solver applies its own "
            "Neumann outflow ghost fills)");
    }
#else
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_WarpX->Geom(0).isPeriodic(direction),
            "theta_implicit_mhd currently requires periodic field boundaries");
    }
#endif

    m_hybrid_pic_model = m_WarpX->get_pointer_HybridPICModel();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_hybrid_pic_model != nullptr,
                                     "theta_implicit_mhd requires algo.maxwell_solver = hybrid");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(m_fluid_flux == "hllc" && m_hybrid_pic_model->m_include_hall_term),
        "implicit_mhd.fluid_flux = hllc requires include_hall_term = false: "
        "the electron energy is advected with the ion contact wave, which "
        "assumes u_e = u_i at the fluid faces");
    if (m_use_hlld) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_hybrid_pic_model->m_include_hall_term,
            "implicit_mhd.fluid_flux = hlld requires include_hall_term = "
            "false: the ideal EMF is the single-fluid Riemann induction "
            "flux, u x B with u_e = u_i");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_hybrid_pic_model->m_include_electron_pressure_term,
            "implicit_mhd.fluid_flux = hlld requires "
            "include_electron_pressure_term = false: the solver-assembled "
            "Ohm's law is E = -u x B + eta J - eta_H laplacian(J)");
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

    // Ohm's law and the fluid equations must regularize the same charge
    // density. Use the stricter requested floor and mirror it into both paths.
    const amrex::Real hybrid_mass_density_floor =
        m_hybrid_pic_model->m_n_floor * PhysConst::q_e / m_ion_charge_to_mass;
    m_mass_density_floor =
        std::max(m_mass_density_floor, hybrid_mass_density_floor);
    m_hybrid_pic_model->m_n_floor =
        m_ion_charge_to_mass * m_mass_density_floor / PhysConst::q_e;

    if (m_hybrid_pic_model->m_include_hall_term !=
        m_hybrid_pic_model->m_include_electron_pressure_term)
    {
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
    } else if (m_ion_closure == "cgl") {
        fluid_blocks.push_back({IonParallelEnergyName, energy_scale});
        fluid_blocks.push_back({IonPerpEnergyName, energy_scale});
    }
    if (m_use_hlld) {
        // Conservative-form recast: B^{n+theta} is the JFNK array block
        // (E is derived from the Riemann EMF and eta J each residual).
        m_state.Define(m_WarpX, "Bfield_fp", "none", fluid_blocks,
                       m_reference_magnetic_field);
    } else {
        m_state.Define(m_WarpX, "Efield_fp", "none", fluid_blocks,
                       electric_field_scale);
    }
    m_state_old.Define(m_state);
    m_state.Copy(m_use_hlld ? FieldType::Bfield_fp : FieldType::Efield_fp);
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
        preconditioner_type == PreconditionerType::none || m_use_hlld,
        "pc_mhd_block in RZ requires implicit_mhd.fluid_flux = hlld: the "
        "E-based block preconditioner has no cylindrical metric terms");
#endif
    if (preconditioner_type == PreconditionerType::pc_mhd_block &&
        !m_use_hlld) {
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
ThetaImplicitMHD::GetMHDReferenceResistivityForPC (const amrex::Real time) const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_pic_model != nullptr,
        "ThetaImplicitMHD reference resistivity requested before Define()");
    const amrex::Real reference_charge_density =
        m_ion_charge_to_mass * m_reference_mass_density;
    return m_hybrid_pic_model->m_eta(reference_charge_density, 0.0_rt, time);
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

void ThetaImplicitMHD::PrintParameters () const
{
    if (!m_WarpX->Verbose()) {
        return;
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
                   << "Hall term:                     "
                   << m_hybrid_pic_model->m_include_hall_term
                   << "\n"
                   << "Electron-pressure Ohm term:    "
                   << m_hybrid_pic_model->m_include_electron_pressure_term << "\n"
                   << "External vector potential:     "
                   << m_hybrid_pic_model->m_add_external_fields << "\n"
                   << "Evolve ion fluid:              " << m_evolve_ion_fluid << "\n"
                   << "Fluid flux:                    " << m_fluid_flux << "\n"
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
                   << "Positivity step safety:        " << m_positivity_safety << "\n"
                   << "Joule heating:                 " << m_include_joule_heating << "\n";
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
    const bool total_energy_closure = m_ion_closure == "total_energy";
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
    ApplyFluidDomainBoundaries(density, momentum, electron_energy, ion_energy,
                               ion_parallel_energy, ion_perp_energy);
}

void ThetaImplicitMHD::ApplyFluidDomainBoundaries (amrex::MultiFab& density,
                                                   amrex::MultiFab& momentum,
                                                   amrex::MultiFab& electron_energy,
                                                   amrex::MultiFab& ion_energy,
                                                   amrex::MultiFab& ion_parallel_energy,
                                                   amrex::MultiFab& ion_perp_energy) const
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
                    if (j < domain_lo) {
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
    }
    const amrex::Box& domain = m_WarpX->Geom(0).Domain();
    const int domain_lo = domain.smallEnd(0);
    const int domain_hi = domain.bigEnd(0);
    const bool r_open = m_r_open && (m_r_open_fluid == "outflow");
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
        amrex::ParallelFor(grown, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if (i < domain_lo) {
                const int mirror = 2 * domain_lo - 1 - i;
                rho(i, j, k) = rho(mirror, j, k);
                energy(i, j, k) = energy(mirror, j, k);
                ion_e(i, j, k) = ion_e(mirror, j, k);
                ion_par(i, j, k) = ion_par(mirror, j, k);
                ion_perp(i, j, k) = ion_perp(mirror, j, k);
                mom(i, j, k, 0) = -mom(mirror, j, k, 0);
                mom(i, j, k, 1) = -mom(mirror, j, k, 1);
                mom(i, j, k, 2) = mom(mirror, j, k, 2);
            } else if (i > domain_hi) {
                // open boundary: clamp to the edge cell (zero-gradient
                // outflow); PEC wall: reflect with odd normal momentum
                const int mirror =
                    r_open ? domain_hi : 2 * domain_hi + 1 - i;
                rho(i, j, k) = rho(mirror, j, k);
                energy(i, j, k) = energy(mirror, j, k);
                ion_e(i, j, k) = ion_e(mirror, j, k);
                ion_par(i, j, k) = ion_par(mirror, j, k);
                ion_perp(i, j, k) = ion_perp(mirror, j, k);
                mom(i, j, k, 0) =
                    r_open ? mom(mirror, j, k, 0) : -mom(mirror, j, k, 0);
                mom(i, j, k, 1) = mom(mirror, j, k, 1);
                mom(i, j, k, 2) = mom(mirror, j, k, 2);
            }
        });
    }
#else
    amrex::ignore_unused(density, momentum, electron_energy, ion_energy,
                         ion_parallel_energy, ion_perp_energy);
#endif
}

void ThetaImplicitMHD::ApplyNeumannZDomainGhosts (amrex::MultiFab& mf) const
{
#if defined(WARPX_DIM_RZ)
    // Zero-gradient extrapolation into the axial domain ghosts, staggering
    // aware (nodal-in-z data clamps to the boundary node). Used for every
    // field the residual stencils read when the z ends are outflow rather
    // than periodic; FillBoundary leaves those ghosts untouched.
    if (!m_z_neumann) {
        return;
    }
    const amrex::Box domain =
        amrex::convert(m_WarpX->Geom(0).Domain(), mf.ixType().toIntVect());
    const int domain_lo = domain.smallEnd(1);
    const int domain_hi = domain.bigEnd(1);
    const int ncomp = mf.nComp();
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const amrex::Box grown = amrex::grow(mfi.validbox(), mf.nGrowVect());
        if (grown.smallEnd(1) >= domain_lo && grown.bigEnd(1) <= domain_hi) {
            continue;
        }
        const auto arr = mf.array(mfi);
        amrex::ParallelFor(grown, ncomp,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
            if (j < domain_lo) {
                arr(i, j, k, n) = arr(i, domain_lo, k, n);
            } else if (j > domain_hi) {
                arr(i, j, k, n) = arr(i, domain_hi, k, n);
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

    m_state.Copy(m_use_hlld ? FieldType::Bfield_fp : FieldType::Efield_fp);
    m_state.CopyMultiFabBlocksFromFields();
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
            *m_WarpX->m_fields.get(OldIonPerpEnergyName, 0));
    }

    m_nlsolver->Solve(m_state, m_state_old, start_time, m_dt, step);
    const int exit_status = m_nlsolver->GetExitStatus();
    if (exit_status < 0) {
        return exit_status;
    }

    UpdateWarpXFields(m_state, start_time);
    m_WarpX->reduced_diags->ComputeDiagsMidStep(step);
    FinishStateUpdate(start_time + m_dt);
    ExecutePythonCallback("afterEpush");
    return exit_status;
}

void ThetaImplicitMHD::UpdateWarpXFields (const WarpXSolverVec& state, const amrex::Real start_time)
{
    const amrex::Real theta_time = start_time + m_theta * m_dt;
    if (m_use_hlld) {
        // Conservative form: the state array block IS B^{n+theta}; E is
        // assembled from the Riemann EMF and eta J later in the residual.
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
            ApplyNeumannZDomainGhosts(
                *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{direction}, 0));
            ApplyNeumannZDomainGhosts(
                *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{direction}, 0));
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
                               *m_WarpX->m_fields.get(IonPerpEnergyName, 0));

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
    const amrex::Real charge_density_floor = m_mass_density_floor * m_ion_charge_to_mass;

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
    ApplyNeumannZDomainGhosts(charge_density);

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
    }

    for (amrex::MFIter mfi(electron_pressure); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.tilebox(electron_pressure.ixType().toIntVect());
        const auto pressure = electron_pressure.array(mfi);
        const auto temperature = electron_temperature.array(mfi);
        const auto energy = electron_energy.const_array(mfi);
        const auto rho = charge_density.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const amrex::Real pressure_value = std::max(
                gamma_e_minus_one * ablastr::coarsen::sample::Interp(
                                        energy, cell_stag, pressure_stag, coarsening, i, j, k, 0),
                pressure_floor);
            pressure(i, j, k) = pressure_value;
            const amrex::Real number_density =
                std::max(rho(i, j, k), charge_density_floor) / PhysConst::q_e;
            temperature(i, j, k) = pressure_value / (number_density * PhysConst::kb);
        });
    }
    electron_pressure.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    electron_temperature.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    ApplyNeumannZDomainGhosts(electron_pressure);
    ApplyNeumannZDomainGhosts(electron_temperature);
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
    for (int component = 0; component < 3; ++component) {
        const auto current_stag = field_staggering(*total_current[component]);
        const auto magnetic_stag = field_staggering(*magnetic_field[component]);
        for (amrex::MFIter mfi(total_current_cc); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
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
                const amrex::Box box = mfi.validbox();
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
    ApplyNeumannZDomainGhosts(total_current_cc);
    ApplyNeumannZDomainGhosts(magnetic_field_cc);
    if (m_use_hlld) {
        // The hlld face kernels read cell-centered B in the radial domain
        // ghosts (the wall face's outer donor cell, and the transverse
        // ghost faces feeding the corner UCT EMF).
        ApplyMagneticCCDomainGhosts(magnetic_field_cc);
    }
}

void ThetaImplicitMHD::ComputeRHS (WarpXSolverVec& rhs, const WarpXSolverVec& state,
                                   const amrex::Real start_time, const int nonlinear_iteration,
                                   const bool from_jacobian)
{
    BL_PROFILE("ThetaImplicitMHD::ComputeRHS()");
    amrex::ignore_unused(nonlinear_iteration, from_jacobian);

    UpdateWarpXFields(state, start_time);

    const auto magnetic_field = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, 0);
    m_hybrid_pic_model->CalculatePlasmaCurrent(magnetic_field, m_WarpX->GetEBUpdateEFlag());
    if (m_z_neumann) {
        using ablastr::fields::Direction;
        for (int direction = 0; direction < 3; ++direction) {
            ApplyNeumannZDomainGhosts(*m_WarpX->m_fields.get(
                FieldType::hybrid_current_fp_plasma, Direction{direction}, 0));
        }
    }

    if (m_external_field_iteration) {
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
        ExecutePythonCallback("externalcoiltheta");
        m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
            start_time + m_theta * m_dt, m_dt);
    }

    FillCellCenteredElectromagneticFields();

    const amrex::Real theta_time = start_time + m_theta * m_dt;
    if (m_use_hlld) {
        // Conservative form: one HLLD Riemann solution per face feeds the
        // fluid divergences AND the ideal EMF; E = EMF + eta J is derived,
        // and the array-block residual is the theta-implicit Faraday
        // update evaluated with the exact Yee curl (div B preserved to
        // round-off): rhs_B = (B_old - theta dt curl E) - B_old.
        ComputeFaceFluxes();
        AssembleOhmElectricField(theta_time);
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
    ComputeFluidRHS(rhs, theta_time);
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
    const bool total_energy_closure = m_ion_closure == "total_energy";
    const bool cgl_closure = m_ion_closure == "cgl";
    const amrex::Real theta = m_theta;
    const amrex::Real density_floor = m_mass_density_floor;
    const amrex::Real energy_floor =
        m_electron_pressure_floor / (m_gamma_e - 1.0_rt);
    const amrex::Real ion_energy_floor =
        total_energy_closure ? m_ion_pressure_floor / (m_gamma_i - 1.0_rt)
                             : 0.0_rt;

    const int num_blocks = cgl_closure ? 4 : (total_energy_closure ? 3 : 2);
    const std::array<const char*, 4> block_names = {
        MassDensityName, ElectronEnergyName,
        cgl_closure ? IonParallelEnergyName : IonEnergyName,
        IonPerpEnergyName};
    const std::array<amrex::Real, 4> block_floors = {
        density_floor, energy_floor,
        cgl_closure ? 0.5_rt * m_ion_pressure_floor : ion_energy_floor,
        m_ion_pressure_floor};

    amrex::Real step_fraction = 1.0_rt;
    for (int block = 0; block < num_blocks; ++block) {
        const amrex::Real floor = block_floors[block];
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
            reduce_op.eval(
                box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    return {theta_implicit_mhd::admissible_step_fraction(
                        value(i, j, k), old_value(i, j, k), delta(i, j, k),
                        requested_step, floor, theta)};
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
        // block_names: 0 = rho, 1 = U_e, 2 = E_i (total_energy) or U_par
        // (cgl), 3 = U_perp (cgl).
        constexpr amrex::Real report_threshold = 1.0e-8;
        for (int block = 0; block < num_blocks; ++block) {
            const amrex::Real floor = block_floors[block];
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
                amrex::ParallelFor(
                    box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        if (theta_implicit_mhd::admissible_step_fraction(
                                value(i, j, k), old_value(i, j, k),
                                delta(i, j, k), requested_step, floor,
                                theta) <= report_threshold) {
                            AMREX_DEVICE_PRINTF(
                                "LimitSolverStep: block %d pinned at "
                                "(%d,%d,%d): value %.6e old %.6e change "
                                "%.6e floor %.6e\n",
                                block_id, i, j, k, value(i, j, k),
                                old_value(i, j, k),
                                delta(i, j, k) * requested_step, floor);
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
    const bool total_energy_closure = m_ion_closure == "total_energy";
    const bool cgl_closure = m_ion_closure == "cgl";
    const amrex::Real theta = m_theta;
    const amrex::Real density_floor = m_mass_density_floor;
    const amrex::Real energy_floor =
        m_electron_pressure_floor / (m_gamma_e - 1.0_rt);
    const amrex::Real ion_energy_floor =
        total_energy_closure ? m_ion_pressure_floor / (m_gamma_i - 1.0_rt)
                             : 0.0_rt;

    const int num_blocks = cgl_closure ? 4 : (total_energy_closure ? 3 : 2);
    const std::array<const char*, 4> block_names = {
        MassDensityName, ElectronEnergyName,
        cgl_closure ? IonParallelEnergyName : IonEnergyName,
        IonPerpEnergyName};
    const std::array<amrex::Real, 4> block_floors = {
        density_floor, energy_floor,
        cgl_closure ? 0.5_rt * m_ion_pressure_floor : ion_energy_floor,
        m_ion_pressure_floor};
    const bool report_projections =
        (std::getenv("WARPX_MHD_REPORT_PROJECTIONS") != nullptr);
    amrex::Long projected_components = 0;
    for (int block = 0; block < num_blocks; ++block) {
        const amrex::Real floor = block_floors[block];
        const int block_id = block;
        const amrex::MultiFab& value_mf =
            state.getMultiFabBlock(block_names[block], 0);
        const amrex::MultiFab& old_mf =
            m_state_old.getMultiFabBlock(block_names[block], 0);
        amrex::MultiFab& delta_mf =
            direction.getMultiFabBlock(block_names[block], 0);
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Long> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (amrex::MFIter mfi(value_mf); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const auto value = value_mf.const_array(mfi);
            const auto old_value = old_mf.const_array(mfi);
            const auto delta = delta_mf.array(mfi);
            reduce_op.eval(
                box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                    const amrex::Real bound =
                        (1.0 - theta) * old_value(i, j, k) + theta * floor;
                    // Clamp descending components to land a small MARGIN
                    // above the bound (never exactly on it): limited
                    // Jacobian probes then always retain an admissible
                    // perturbation at floor-resident cells, avoiding the
                    // eps -> 0 probe abort. Inside the margin band the
                    // component is zeroed (no lifting: refill is the
                    // physics' job, not the projection's).
                    const amrex::Real margin =
                        1.0e-6 * (std::abs(old_value(i, j, k)) + floor);
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
                                old_value(i, j, k), change, floor);
                        }
                        delta(i, j, k) = target / requested_step;
                        return {1};
                    }
                    return {0};
                });
        }
        projected_components += amrex::get<0>(reduce_data.value(reduce_op));
    }
    amrex::ParallelAllReduce::Sum(projected_components,
                                  amrex::ParallelContext::CommunicatorSub());
    if (projected_components > 0) {
        amrex::Print() << "Newton: projected " << projected_components
                       << " direction components onto admissibility bounds\n";
    }
    return LimitSolverStep(state, direction, requested_step);
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
        m_ion_closure == "total_energy",
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
    return flux_parameters;
}

void ThetaImplicitMHD::ComputeFluidRHS (WarpXSolverVec& rhs, const amrex::Real time) const
{
    if (m_use_hlld) {
        // Conservative form: read the precomputed per-face Riemann fluxes
        // (one HLLD solution per face; the same solution feeds the ideal
        // EMF) instead of re-evaluating fluxes cell-by-cell.
        ComputeFluidRHSFromFaceFluxes(rhs, time);
        return;
    }
    // Defensive: the CGL closure is only wired into the hlld face-flux
    // path above (and the input parser asserts fluid_flux = hlld), so the
    // E-based fluid RHS below must never see it.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_ion_closure != "cgl",
        "implicit_mhd.ion_closure = cgl requires implicit_mhd.fluid_flux = "
        "hlld");

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
    const amrex::Real gamma_i_minus_one = m_gamma_i - 1.0_rt;
#endif
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
    // Holmstrom-style vacuum cell switching: below the threshold the fluid
    // is passive dust with frozen momentum (conservative advection of mass
    // and energy remains, so exchange with neighboring plasma cells is
    // intact, and arriving momentum is absorbed).
    const bool holmstrom_vacuum = m_vacuum_mass_density > 0.0_rt;
    const amrex::Real vacuum_mass_density = m_vacuum_mass_density;
    const amrex::Real vacuum_drag_rate = m_vacuum_drag_rate;
    const auto eta = m_hybrid_pic_model->m_eta;
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
            for (int component = 0; component < 3; ++component) {
                momentum_increment(i, j, k, component) =
                    evolve_ion_fluid
                        ? theta_dt * (plasma_weight *
                                          (-divergence_momentum_flux[component] +
                                           j_cross_b[component]) -
                                      vacuum_drag * mom(i, j, k, component))
                        : 0.0_rt;
            }

            const amrex::Real pressure_e =
                std::max(gamma_e_minus_one * energy(i, j, k), pressure_floor);
            const amrex::Real current_magnitude = std::sqrt(jx * jx + jy * jy + jz * jz);
            const amrex::Real charge_density =
                charge_to_mass * std::max(rho(i, j, k), density_floor);
            const amrex::Real joule_heating = include_joule_heating
                                                  ? eta(charge_density, current_magnitude, time) *
                                                        current_magnitude * current_magnitude
                                                  : 0.0_rt;
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
            energy_increment(i, j, k) =
                theta_dt * plasma_weight *
                (-divergence_energy_flux + pressure_work + joule_heating);

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
                ion_energy_increment(i, j, k) =
                    theta_dt * plasma_weight *
                    (-divergence_ion_energy_flux + lorentz_work +
                     ion_pressure_work);
            }
        });
    }
}

void ThetaImplicitMHD::ComputeFaceFluxes ()
{
#if defined(WARPX_DIM_1D_Z)
    ComputeDirectionalFaceFluxes(*m_WarpX->m_fields.get(FaceFluxZName, 0), 2);
#elif defined(WARPX_DIM_RZ)
    ComputeDirectionalFaceFluxes(*m_WarpX->m_fields.get(FaceFluxRName, 0), 0);
    ComputeDirectionalFaceFluxes(*m_WarpX->m_fields.get(FaceFluxZName, 0), 2);
#else
    WARPX_ABORT_WITH_MESSAGE(
        "ThetaImplicitMHD::ComputeFaceFluxes() requires 1D or RZ geometry");
#endif
}

void ThetaImplicitMHD::ComputeDirectionalFaceFluxes (
    amrex::MultiFab& face_flux_mf, const int normal_direction)
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
#if defined(WARPX_DIM_RZ)
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
#endif
    constexpr int flux_mass = FaceFluxComponent::mass;
    constexpr int flux_momentum = FaceFluxComponent::momentum_x;
    constexpr int flux_magnetic = FaceFluxComponent::magnetic_x;
    constexpr int flux_electron_energy = FaceFluxComponent::electron_energy;
    constexpr int flux_ion_energy = FaceFluxComponent::ion_energy;
    constexpr int flux_ion_parallel_energy =
        FaceFluxComponent::ion_parallel_energy;
    constexpr int flux_ion_perp_energy = FaceFluxComponent::ion_perp_energy;
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
            const auto left =
                parameters.cgl_closure
                    ? theta_implicit_mhd::load_cell_state_hlld_cgl(
                          rho, mom, energy, ion_e, upar, uperp, j_cc, b_cc,
                          il, jl, kl, normal, parameters)
                    : theta_implicit_mhd::load_cell_state_hlld(
                          rho, mom, energy, ion_e, j_cc, b_cc, il, jl, kl,
                          normal, parameters);
            const auto right =
                parameters.cgl_closure
                    ? theta_implicit_mhd::load_cell_state_hlld_cgl(
                          rho, mom, energy, ion_e, upar, uperp, j_cc, b_cc,
                          i, j, k, normal, parameters)
                    : theta_implicit_mhd::load_cell_state_hlld(
                          rho, mom, energy, ion_e, j_cc, b_cc, i, j, k,
                          normal, parameters);
            amrex::Real bn_face = bn_staggered(i, j, k);
            if (add_external) {
                bn_face += bn_external(i, j, k);
            }
            auto flux = theta_implicit_mhd::hlld_flux(left, right, bn_face,
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
            flux.mass *= donor_blend(
                flux.mass,
                theta_implicit_mhd::floor_outflow_limiter(
                    donor_end(rho, rho_old, il, jl, kl),
                    parameters.density_floor),
                theta_implicit_mhd::floor_outflow_limiter(
                    donor_end(rho, rho_old, i, j, k),
                    parameters.density_floor),
                0.5_rt * (left.safe_density + right.safe_density));
            const amrex::Real electron_energy_floor =
                parameters.electron_pressure_floor /
                (parameters.gamma_e - 1.0_rt);
            flux.electron_energy *= donor_blend(
                flux.electron_energy,
                theta_implicit_mhd::floor_outflow_limiter(
                    donor_end(energy, energy_old, il, jl, kl),
                    electron_energy_floor),
                theta_implicit_mhd::floor_outflow_limiter(
                    donor_end(energy, energy_old, i, j, k),
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
                const amrex::Real ion_energy_floor =
                    parameters.ion_pressure_floor /
                    (parameters.gamma_i - 1.0_rt);
                flux.ion_energy *= donor_blend(
                    flux.ion_energy,
                    theta_implicit_mhd::floor_outflow_limiter(
                        ion_internal_end(il, jl, kl), ion_energy_floor),
                    theta_implicit_mhd::floor_outflow_limiter(
                        ion_internal_end(i, j, k), ion_energy_floor),
                    0.5_rt * (left.ion_energy + right.ion_energy) +
                        ion_energy_floor);
            }
            if (parameters.cgl_closure) {
                // CGL internal-energy channels: same donor-gated guards,
                // gating on the U fields directly (U_par and U_perp are
                // pure internal energies -- no kinetic subtraction), with
                // U_par floored at p_i_floor/2 and U_perp at p_i_floor.
                const amrex::Real parallel_floor =
                    0.5_rt * parameters.ion_pressure_floor;
                flux.ion_parallel_energy *= donor_blend(
                    flux.ion_parallel_energy,
                    theta_implicit_mhd::floor_outflow_limiter(
                        donor_end(upar, upar_old, il, jl, kl),
                        parallel_floor),
                    theta_implicit_mhd::floor_outflow_limiter(
                        donor_end(upar, upar_old, i, j, k), parallel_floor),
                    0.5_rt * (left.ion_parallel_energy +
                              right.ion_parallel_energy) +
                        parallel_floor);
                const amrex::Real perp_floor = parameters.ion_pressure_floor;
                flux.ion_perp_energy *= donor_blend(
                    flux.ion_perp_energy,
                    theta_implicit_mhd::floor_outflow_limiter(
                        donor_end(uperp, uperp_old, il, jl, kl), perp_floor),
                    theta_implicit_mhd::floor_outflow_limiter(
                        donor_end(uperp, uperp_old, i, j, k), perp_floor),
                    0.5_rt *
                            (left.ion_perp_energy + right.ion_perp_energy) +
                        perp_floor);
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

void ThetaImplicitMHD::AssembleOhmElectricField (const amrex::Real time) const
{
#if defined(WARPX_DIM_1D_Z)
    using ablastr::fields::Direction;
    // Standard resistive-MHD Ohm's law, E = -u x B + eta J, assembled at
    // the native Yee staggering. The ideal part of the tangential
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
    const amrex::MultiFab& current_x = *m_WarpX->m_fields.get(
        FieldType::hybrid_current_fp_plasma, Direction{0}, 0);
    const amrex::MultiFab& current_y = *m_WarpX->m_fields.get(
        FieldType::hybrid_current_fp_plasma, Direction{1}, 0);
    const amrex::MultiFab& current_z = *m_WarpX->m_fields.get(
        FieldType::hybrid_current_fp_plasma, Direction{2}, 0);
    const amrex::MultiFab& charge_density =
        *m_WarpX->m_fields.get(FieldType::rho_fp, 0);
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
    const amrex::Real charge_density_floor =
        m_ion_charge_to_mass * m_mass_density_floor;
    const amrex::Real density_floor = m_mass_density_floor;
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
        const auto rho_q = charge_density.const_array(mfi);
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
                eta(charge_density_value, current_magnitude, time);
            electric_x(i, j, k) =
                flux_arr(i, j, k, flux_induction_t2) + resistivity * jx;
            electric_y(i, j, k) =
                -flux_arr(i, j, k, flux_induction_t1) + resistivity * jy;
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
        const auto rho_q = charge_density.const_array(mfi);
        const auto rho = density.const_array(mfi);
        const auto mom = momentum.const_array(mfi);
        const auto b_cc = magnetic_cc.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            const amrex::Real jx = 0.5_rt * (j_x(i, j, k) + j_x(i + 1, j, k));
            const amrex::Real jy = 0.5_rt * (j_y(i, j, k) + j_y(i + 1, j, k));
            const amrex::Real jz = j_z(i, j, k);
            const amrex::Real current_magnitude =
                std::sqrt(jx * jx + jy * jy + jz * jz);
            const amrex::Real charge_density_value = std::max(
                0.5_rt * (rho_q(i, j, k) + rho_q(i + 1, j, k)),
                charge_density_floor);
            const amrex::Real resistivity =
                eta(charge_density_value, current_magnitude, time);
            const amrex::Real safe_density =
                std::max(rho(i, j, k), density_floor);
            const amrex::Real velocity_x = mom(i, j, k, 0) / safe_density;
            const amrex::Real velocity_y = mom(i, j, k, 1) / safe_density;
            electric_z(i, j, k) =
                -(velocity_x * b_cc(i, j, k, 1) -
                  velocity_y * b_cc(i, j, k, 0)) +
                resistivity * jz;
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
    // Standard resistive-MHD Ohm's law at the native RZ (m = 0) Yee
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
    const amrex::MultiFab& current_r = *m_WarpX->m_fields.get(
        FieldType::hybrid_current_fp_plasma, Direction{0}, 0);
    const amrex::MultiFab& current_theta = *m_WarpX->m_fields.get(
        FieldType::hybrid_current_fp_plasma, Direction{1}, 0);
    const amrex::MultiFab& current_z = *m_WarpX->m_fields.get(
        FieldType::hybrid_current_fp_plasma, Direction{2}, 0);
    const amrex::MultiFab& charge_density =
        *m_WarpX->m_fields.get(FieldType::rho_fp, 0);
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
    const amrex::Real charge_density_floor =
        m_ion_charge_to_mass * m_mass_density_floor;
    const amrex::Real density_floor = m_mass_density_floor;
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
        const auto rho_q = charge_density.const_array(mfi);
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
            const amrex::Real charge_density_value = std::max(
                0.5_rt * (rho_q(i, j, k) + rho_q(i + 1, j, k)),
                charge_density_floor);
            const amrex::Real resistivity =
                eta(charge_density_value, current_magnitude, time);
            electric_r(i, j, k) =
                zface(i, j, k, flux_induction_t2) + resistivity * jr;
            if (include_hyper_resistivity) {
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
        const auto rho_q = charge_density.const_array(mfi);
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
            const amrex::Real charge_density_value = std::max(
                0.5_rt * (rho_q(i, j, k) + rho_q(i, j + 1, k)),
                charge_density_floor);
            const amrex::Real resistivity =
                eta(charge_density_value, current_magnitude, time);
            electric_z(i, j, k) =
                -rface(i, j, k, flux_induction_t1) + resistivity * jz;
            if (include_hyper_resistivity) {
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
        const auto rho_q = charge_density.const_array(mfi);
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
            const amrex::Real resistivity =
                eta(charge_density_value, current_magnitude, time);
            electric_theta(i, j, k) =
                average + dissipation + resistivity * jt_corner;
            if (include_hyper_resistivity) {
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
            ApplyNeumannZDomainGhosts(*m_WarpX->m_fields.get(
                FieldType::Efield_fp, Direction{direction}, 0));
        }
    }
#else
    amrex::ignore_unused(time);
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
    const amrex::MultiFab& current = *m_WarpX->m_fields.get(TotalCurrentCCName, 0);
    const amrex::MultiFab& magnetic_cc =
        *m_WarpX->m_fields.get(MagneticFieldCCName, 0);
    const amrex::MultiFab& face_flux_mf =
        *m_WarpX->m_fields.get(FaceFluxZName, 0);
#if defined(WARPX_DIM_RZ)
    const amrex::MultiFab& face_flux_r_mf =
        *m_WarpX->m_fields.get(FaceFluxRName, 0);
#endif

    const bool total_energy_closure = m_ion_closure == "total_energy";
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
    const amrex::Real work_kappa = m_hlld_kappa_signal;
    const amrex::Real electron_energy_floor_rate =
        m_electron_pressure_floor / (m_gamma_e - 1.0_rt) / theta_dt;
    const amrex::Real ion_energy_floor_rate = ion_energy_floor / theta_dt;
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
    const auto eta = m_hybrid_pic_model->m_eta;
#if defined(WARPX_DIM_RZ)
    const amrex::Real inverse_dr = inverse_cell_size[0];
    const amrex::Real radial_lower = m_WarpX->Geom(0).ProbLo(0);
    const amrex::Real radial_cell_size = m_WarpX->Geom(0).CellSize(0);
    const amrex::Real gamma_i_minus_one = m_gamma_i - 1.0_rt;
    const amrex::Real inverse_mu0 = 1.0_rt / PhysConst::mu0;
    const theta_implicit_mhd::FluxParameters flux_parameters =
        MakeFluxParameters();
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
        const auto j_plasma = current.const_array(mfi);
        const auto b_cc = magnetic_cc.const_array(mfi);
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
                if (total_energy_closure) {
                    amrex::Real kinetic_energy = 0.0_rt;
                    for (int component = 0; component < 3; ++component) {
                        kinetic_energy += mom(i, j, k, component) *
                                          mom(i, j, k, component);
                    }
                    kinetic_energy *= 0.5_rt / safe_density;
                    // Same C-infinity smooth internal-energy floor as the
                    // kernel's p_i(E_i) recovery: halo cells pinned at the
                    // floor sit exactly on a hard max() kink otherwise.
                    pressure_i =
                        gamma_i_minus_one *
                        theta_implicit_mhd::smooth_positive_floor(
                            ion_e(i, j, k) - kinetic_energy,
                            ion_energy_floor);
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

            rho_increment(i, j, k) =
                evolve_ion_fluid
                    ? -theta_dt * plasma_weight * divergence_momentum
                    : 0.0_rt;

            const amrex::Real vacuum_drag =
                vacuum_drag_rate * (1.0_rt - plasma_weight);
            for (int component = 0; component < 3; ++component) {
                // The Maxwell stress inside the total momentum flux
                // replaces the pointwise J x B force.
                momentum_increment(i, j, k, component) =
                    evolve_ion_fluid
                        ? theta_dt *
                              (plasma_weight *
                                   (-divergence_momentum_flux[component]) -
                               vacuum_drag * mom(i, j, k, component))
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
                charge_to_mass * std::max(rho(i, j, k), density_floor);
            const amrex::Real joule_heating =
                include_joule_heating
                    ? eta(charge_density, current_magnitude, time) *
                          current_magnitude * current_magnitude
                    : 0.0_rt;
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
            pressure_work = drain_gate(
                pressure_work, divergence_energy_flux,
                electron_energy_floor_rate,
                theta_implicit_mhd::floor_outflow_limiter(
                    energy_end, pressure_floor / gamma_e_minus_one));
            energy_increment(i, j, k) =
                theta_dt * plasma_weight *
                (-divergence_energy_flux + pressure_work + joule_heating);

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
                    drain_gate(ion_pressure_work, divergence_ion_energy_flux,
                               ion_energy_floor_rate, ion_limiter);
                lorentz_work =
                    drain_gate(lorentz_work, divergence_ion_energy_flux,
                               ion_energy_floor_rate, ion_limiter);
                ion_energy_increment(i, j, k) =
                    theta_dt * plasma_weight *
                    (-divergence_ion_energy_flux + lorentz_work +
                     ion_pressure_work);
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
                parallel_work = drain_gate(
                    parallel_work, divergence_ion_parallel_flux,
                    ion_parallel_floor_rate, parallel_limiter);
                parallel_relaxation = drain_gate(
                    parallel_relaxation, divergence_ion_parallel_flux,
                    ion_parallel_floor_rate, parallel_limiter);
                perp_work =
                    drain_gate(perp_work, divergence_ion_perp_flux,
                               ion_perp_floor_rate, perp_limiter);
                perp_relaxation =
                    drain_gate(perp_relaxation, divergence_ion_perp_flux,
                               ion_perp_floor_rate, perp_limiter);
                ion_parallel_increment(i, j, k) =
                    theta_dt * plasma_weight *
                    (-divergence_ion_parallel_flux + parallel_work +
                     parallel_relaxation);
                ion_perp_increment(i, j, k) =
                    theta_dt * plasma_weight *
                    (-divergence_ion_perp_flux + perp_work +
                     perp_relaxation);
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

void ThetaImplicitMHD::FinishStateUpdate (const amrex::Real end_time)
{
    const amrex::Real inverse_theta = 1.0_rt / m_theta;
    m_state.linComb(inverse_theta, m_state, 1.0_rt - inverse_theta, m_state_old);

    if (m_ion_closure == "total_energy") {
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
        const amrex::Real internal_floor =
            m_ion_pressure_floor / (m_gamma_i - 1.0_rt);
        const amrex::Real density_floor = m_mass_density_floor;
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
                ion_e(i, j, k) = std::max(ion_e(i, j, k),
                                          kinetic_energy + internal_floor);
            });
        }
        ion_energy_block.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
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
        const std::array<std::pair<const char*, amrex::Real>, 2> blocks = {
            {{IonParallelEnergyName, 0.5_rt * m_ion_pressure_floor},
             {IonPerpEnergyName, m_ion_pressure_floor}}};
        for (const auto& [block_name, floor] : blocks) {
            amrex::MultiFab& energy_block =
                m_state.getMultiFabBlock(block_name, 0);
            const amrex::Real block_floor = floor;
            for (amrex::MFIter mfi(energy_block); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                const auto energy_array = energy_block.array(mfi);
                amrex::ParallelFor(
                    box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        const amrex::Real margin =
                            1.0e-6_rt *
                            (std::abs(energy_array(i, j, k)) + block_floor);
                        energy_array(i, j, k) =
                            std::max(energy_array(i, j, k),
                                     block_floor + margin);
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
    if (m_ion_closure == "total_energy") {
        const amrex::MultiFab& ion_energy =
            *m_WarpX->m_fields.get(IonEnergyName, 0);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            ion_energy.min(0) >=
                m_ion_pressure_floor / (m_gamma_i - 1.0_rt) - energy_tolerance,
            "theta_implicit_mhd produced a final ion total energy below its "
            "internal-energy floor");
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

    if (m_use_hlld) {
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
            ApplyNeumannZDomainGhosts(
                *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{direction}, 0));
        }
    }
    FillFluidSources(m_state);

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
            ApplyNeumannZDomainGhosts(*m_WarpX->m_fields.get(
                FieldType::hybrid_current_fp_plasma, Direction{direction}, 0));
        }
    }

    if (m_hybrid_pic_model->m_add_external_fields) {
        if (m_external_field_iteration) {
            // Final circuit pass against the accepted end-of-step plasma
            // current, leaving the circuit state converged at t^{n+1}.
            ExecutePythonCallback("externalcoilfinish");
        }
        // Refresh the external fields at t^{n+1} for the final Ohm's-law
        // recompute below. These same stored values are added back to the
        // totals at the end of this function and subtracted again at the
        // start of the next step, so the split remains exact.
        m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
            end_time, m_dt);
    }
    if (m_use_hlld) {
        // Recompute the end-of-step Ohm E (diagnostics and coupling) from
        // the accepted final state: the same Riemann-EMF assembly the
        // residual used, evaluated at t^{n+1}.
        FillCellCenteredElectromagneticFields();
        ComputeFaceFluxes();
        AssembleOhmElectricField(end_time);
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
            ApplyNeumannZDomainGhosts(
                *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{direction}, 0));
        }
    }
    if (!m_use_hlld) {
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
