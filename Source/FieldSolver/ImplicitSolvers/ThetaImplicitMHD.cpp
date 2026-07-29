/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ThetaImplicitMHD.H"
#include "ThetaImplicitMHD_K.H"

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
#include <AMReX_MFIter.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>
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
#elif defined(WARPX_DIM_XZ)
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
#elif defined(WARPX_DIM_XZ)
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
#elif defined(WARPX_DIM_XZ)
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
    utils::parser::queryWithParser(pp, "positivity_safety", m_positivity_safety);
    pp.query("fluid_flux", m_fluid_flux);
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
        m_fluid_flux == "centered" || m_fluid_flux == "rusanov",
        "implicit_mhd.fluid_flux must be centered or rusanov");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_positivity_safety > 0.0_rt && m_positivity_safety < 1.0_rt,
        "implicit_mhd.positivity_safety must be greater than zero and less than one");

    std::string mass_density_expression;
    std::string electron_pressure_expression;
    std::string velocity_x_expression = "0.0";
    std::string velocity_y_expression = "0.0";
    std::string velocity_z_expression = "0.0";
    utils::parser::Store_parserString(pp, "mass_density(x,y,z)", mass_density_expression);
    utils::parser::Store_parserString(pp, "electron_pressure(x,y,z)", electron_pressure_expression);
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
    m_mass_density = m_mass_density_parser->compile<3>();
    m_velocity_x = m_velocity_x_parser->compile<3>();
    m_velocity_y = m_velocity_y_parser->compile<3>();
    m_velocity_z = m_velocity_z_parser->compile<3>();
    m_electron_pressure = m_electron_pressure_parser->compile<3>();
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

    fields.alloc_init(TotalCurrentCCName, lev, ba, dm, 3, guard_cells, 0.0_rt);
    fields.alloc_init(MagneticFieldCCName, lev, ba, dm, 3, guard_cells, 0.0_rt);
}

void ThetaImplicitMHD::Define (WarpX* const warpx, const bool from_restart)
{
    BL_PROFILE("ThetaImplicitMHD::Define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_is_defined, "ThetaImplicitMHD object is already defined");
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    WARPX_ABORT_WITH_MESSAGE("theta_implicit_mhd currently supports Cartesian geometry only");
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
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_WarpX->Geom(0).isPeriodic(direction),
            "theta_implicit_mhd currently requires periodic field boundaries");
    }

    m_hybrid_pic_model = m_WarpX->get_pointer_HybridPICModel();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_hybrid_pic_model != nullptr,
                                     "theta_implicit_mhd requires algo.maxwell_solver = hybrid");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_hybrid_pic_model->m_solve_electron_energy_equation,
        "theta_implicit_mhd advances electron energy inside JFNK; "
        "hybrid_pic_model.solve_electron_energy_equation must be false");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_hybrid_pic_model->m_add_external_fields,
        "theta_implicit_mhd does not yet support hybrid external fields");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_hybrid_pic_model->m_has_external_current,
        "theta_implicit_mhd does not yet support hybrid external currents");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_hybrid_pic_model->m_holmstrom_vacuum_region,
        "theta_implicit_mhd does not yet support the Holmstrom vacuum-region "
        "model");

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
    const std::vector<WarpXSolverVec::MultiFabBlockSpec> fluid_blocks = {
        {MassDensityName, m_reference_mass_density},
        {MomentumDensityName, momentum_scale},
        {ElectronEnergyName, energy_scale}};
    m_state.Define(m_WarpX, "Efield_fp", "none", fluid_blocks, electric_field_scale);
    m_state_old.Define(m_state);
    m_state.Copy(FieldType::Efield_fp);
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
    if (preconditioner_type == PreconditionerType::pc_mhd_block) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::grid_type == ablastr::utils::enums::GridType::Staggered,
            "pc_mhd_block currently requires warpx.grid_type = staggered");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_hybrid_pic_model->m_include_hall_term,
            "pc_mhd_block currently requires "
            "hybrid_pic_model.include_hall_term = false");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_hybrid_pic_model->m_include_electron_pressure_term,
            "pc_mhd_block currently requires "
            "hybrid_pic_model.include_electron_pressure_term = false");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_hybrid_pic_model->m_include_hyper_resistivity_term,
            "pc_mhd_block currently requires zero plasma_hyper_resistivity");
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
            "pc_mhd_block currently requires plasma_resistivity to be "
            "constant or time-only (no rho or J dependence)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_fluid_flux == "centered",
            "pc_mhd_block currently requires implicit_mhd.fluid_flux = centered");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_evolve_ion_fluid,
            "pc_mhd_block currently requires implicit_mhd.evolve_ion_fluid = true");
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
                   << "Evolve ion fluid:              " << m_evolve_ion_fluid << "\n"
                   << "Fluid flux:                    " << m_fluid_flux << "\n"
                   << "Positivity step safety:        " << m_positivity_safety << "\n"
                   << "Joule heating:                 " << m_include_joule_heating << "\n";
    PrintBaseImplicitSolverParameters();
    m_nlsolver->PrintParams();
    amrex::Print() << "-----------------------------------------------------------\n\n";
}

void ThetaImplicitMHD::InitializeFluidState ()
{
    amrex::MultiFab& density = *m_WarpX->m_fields.get(MassDensityName, 0);
    amrex::MultiFab& momentum = *m_WarpX->m_fields.get(MomentumDensityName, 0);
    amrex::MultiFab& electron_energy = *m_WarpX->m_fields.get(ElectronEnergyName, 0);

    const auto cell_size = m_WarpX->Geom(0).CellSizeArray();
    const auto lower = m_WarpX->Geom(0).ProbLoArray();
    const auto mass_density = m_mass_density;
    const auto velocity_x = m_velocity_x;
    const auto velocity_y = m_velocity_y;
    const auto velocity_z = m_velocity_z;
    const auto electron_pressure = m_electron_pressure;
    const amrex::Real density_floor = m_mass_density_floor;
    const amrex::Real pressure_floor = m_electron_pressure_floor;
    const amrex::Real gamma_e_minus_one = m_gamma_e - 1.0_rt;

    for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto density_array = density.array(mfi);
        const auto momentum_array = momentum.array(mfi);
        const auto energy_array = electron_energy.array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
#if defined(WARPX_DIM_3D)
            const amrex::Real x = lower[0] + (i + 0.5_rt) * cell_size[0];
            const amrex::Real y = lower[1] + (j + 0.5_rt) * cell_size[1];
            const amrex::Real z = lower[2] + (k + 0.5_rt) * cell_size[2];
#elif defined(WARPX_DIM_XZ)
                const amrex::Real x = lower[0] + (i + 0.5_rt) * cell_size[0];
                const amrex::Real y = 0.0_rt;
                const amrex::Real z = lower[1] + (j + 0.5_rt) * cell_size[1];
#else
                const amrex::Real x = 0.0_rt;
                const amrex::Real y = 0.0_rt;
                const amrex::Real z = lower[0] + (i + 0.5_rt) * cell_size[0];
#endif
            const amrex::Real rho = std::max(mass_density(x, y, z), density_floor);
            density_array(i, j, k) = rho;
            momentum_array(i, j, k, 0) = rho * velocity_x(x, y, z);
            momentum_array(i, j, k, 1) = rho * velocity_y(x, y, z);
            momentum_array(i, j, k, 2) = rho * velocity_z(x, y, z);
            energy_array(i, j, k) =
                std::max(electron_pressure(x, y, z), pressure_floor) / gamma_e_minus_one;
        });
    }

    density.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    momentum.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    electron_energy.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
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
    SaveMagneticField();

    m_state.Copy(FieldType::Efield_fp);
    m_state.CopyMultiFabBlocksFromFields();
    m_state_old.Copy(m_state);

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
    m_WarpX->SetElectricFieldAndApplyBCs(state, theta_time);

    const auto& magnetic_field_old = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, 0);
    m_WarpX->UpdateMagneticFieldAndApplyBCs(magnetic_field_old, m_theta * m_dt, start_time);

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
    total_current_cc.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
    magnetic_field_cc.FillBoundaryAndSync(m_WarpX->Geom(0).periodicity());
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
    FillCellCenteredElectromagneticFields();

    const auto electric_field = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, 0);
    const auto ion_current = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, 0);
    const auto charge_density = m_WarpX->m_fields.get_mr_levels(FieldType::rho_fp, 0);

    const amrex::Real theta_time = start_time + m_theta * m_dt;
    m_hybrid_pic_model->HybridPICSolveE(
        electric_field, ion_current, magnetic_field, charge_density,
        m_WarpX->GetEBUpdateEFlag(), true, true, theta_time, false);

    rhs.Copy(FieldType::Efield_fp);
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

    const amrex::MultiFab& density = state.getMultiFabBlock(MassDensityName, 0);
    const amrex::MultiFab& energy =
        state.getMultiFabBlock(ElectronEnergyName, 0);
    const amrex::MultiFab& density_direction =
        direction.getMultiFabBlock(MassDensityName, 0);
    const amrex::MultiFab& energy_direction =
        direction.getMultiFabBlock(ElectronEnergyName, 0);
    const amrex::MultiFab& old_density =
        m_state_old.getMultiFabBlock(MassDensityName, 0);
    const amrex::MultiFab& old_energy =
        m_state_old.getMultiFabBlock(ElectronEnergyName, 0);

    amrex::ReduceOps<amrex::ReduceOpMin> reduce_op;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    const amrex::Real theta = m_theta;
    const amrex::Real density_floor = m_mass_density_floor;
    const amrex::Real energy_floor =
        m_electron_pressure_floor / (m_gamma_e - 1.0_rt);

    for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto rho = density.const_array(mfi);
        const auto electron_energy = energy.const_array(mfi);
        const auto delta_rho = density_direction.const_array(mfi);
        const auto delta_energy = energy_direction.const_array(mfi);
        const auto rho_old = old_density.const_array(mfi);
        const auto energy_old = old_energy.const_array(mfi);
        reduce_op.eval(
            box, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                const amrex::Real density_fraction =
                    theta_implicit_mhd::admissible_step_fraction(
                        rho(i, j, k), rho_old(i, j, k), delta_rho(i, j, k),
                        requested_step, density_floor, theta);
                const amrex::Real energy_fraction =
                    theta_implicit_mhd::admissible_step_fraction(
                        electron_energy(i, j, k), energy_old(i, j, k),
                        delta_energy(i, j, k), requested_step, energy_floor,
                        theta);
                return {std::min(density_fraction, energy_fraction)};
            });
    }

    amrex::Real step_fraction = amrex::get<0>(reduce_data.value(reduce_op));
    amrex::ParallelAllReduce::Min(step_fraction,
                                  amrex::ParallelContext::CommunicatorSub());
    if (step_fraction < 1.0_rt) {
        step_fraction *= m_positivity_safety;
    }
    return requested_step * step_fraction;
}

void ThetaImplicitMHD::ComputeFluidRHS (WarpXSolverVec& rhs, const amrex::Real time) const
{
    const amrex::MultiFab& density = *m_WarpX->m_fields.get(MassDensityName, 0);
    const amrex::MultiFab& momentum = *m_WarpX->m_fields.get(MomentumDensityName, 0);
    const amrex::MultiFab& electron_energy = *m_WarpX->m_fields.get(ElectronEnergyName, 0);
    const amrex::MultiFab& current = *m_WarpX->m_fields.get(TotalCurrentCCName, 0);
    const amrex::MultiFab& magnetic_field = *m_WarpX->m_fields.get(MagneticFieldCCName, 0);

    amrex::MultiFab& density_rhs = rhs.getMultiFabBlock(MassDensityName, 0);
    amrex::MultiFab& momentum_rhs = rhs.getMultiFabBlock(MomentumDensityName, 0);
    amrex::MultiFab& electron_energy_rhs = rhs.getMultiFabBlock(ElectronEnergyName, 0);

    const auto inverse_cell_size = physical_inverse_cell_size(m_WarpX->Geom(0));
    const auto active = active_physical_directions();
    const amrex::Real theta_dt = m_theta * m_dt;
    const amrex::Real density_floor = m_mass_density_floor;
    const amrex::Real charge_to_mass = m_ion_charge_to_mass;
    const amrex::Real gamma_e_minus_one = m_gamma_e - 1.0_rt;
    const amrex::Real pressure_floor = m_electron_pressure_floor;
    const bool include_hall_term = m_hybrid_pic_model->m_include_hall_term;
    const bool evolve_ion_fluid = m_evolve_ion_fluid;
    const bool include_joule_heating = m_include_joule_heating;
    const auto eta = m_hybrid_pic_model->m_eta;
    const theta_implicit_mhd::FluxParameters flux_parameters = {
        density_floor,
        charge_to_mass,
        m_gamma_e,
        m_gamma_i,
        m_reference_mass_density,
        m_reference_ion_pressure,
        pressure_floor,
        include_hall_term,
        m_fluid_flux == "rusanov"};

    for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto rho = density.const_array(mfi);
        const auto mom = momentum.const_array(mfi);
        const auto energy = electron_energy.const_array(mfi);
        const auto j_plasma = current.const_array(mfi);
        const auto magnetic = magnetic_field.const_array(mfi);
        const auto rho_increment = density_rhs.array(mfi);
        const auto momentum_increment = momentum_rhs.array(mfi);
        const auto energy_increment = electron_energy_rhs.array(mfi);

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real divergence_momentum = 0.0_rt;
            amrex::Real divergence_energy_flux = 0.0_rt;
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
                    rho, mom, energy, j_plasma, i, j, k, ih, jh, kh, direction,
                    flux_parameters);
                const auto low_flux = theta_implicit_mhd::face_flux(
                    rho, mom, energy, j_plasma, il, jl, kl, i, j, k, direction,
                    flux_parameters);
                const amrex::Real derivative_scale = inverse_cell_size[direction];

                divergence_momentum +=
                    derivative_scale * (high_flux.mass - low_flux.mass);

                for (int component = 0; component < 3; ++component) {
                    divergence_momentum_flux[component] +=
                        derivative_scale *
                        (high_flux.momentum[component] - low_flux.momentum[component]);
                }

                divergence_energy_flux +=
                    derivative_scale * (high_flux.electron_energy -
                                        low_flux.electron_energy);
                divergence_electron_velocity +=
                    derivative_scale *
                    (high_flux.electron_velocity - low_flux.electron_velocity);
            }

            rho_increment(i, j, k) = evolve_ion_fluid ? -theta_dt * divergence_momentum : 0.0_rt;

            const amrex::Real jx = j_plasma(i, j, k, 0);
            const amrex::Real jy = j_plasma(i, j, k, 1);
            const amrex::Real jz = j_plasma(i, j, k, 2);
            const amrex::Real bx = magnetic(i, j, k, 0);
            const amrex::Real by = magnetic(i, j, k, 1);
            const amrex::Real bz = magnetic(i, j, k, 2);
            const amrex::Real j_cross_b[3] = {jy * bz - jz * by, jz * bx - jx * bz,
                                              jx * by - jy * bx};
            for (int component = 0; component < 3; ++component) {
                momentum_increment(i, j, k, component) =
                    evolve_ion_fluid
                        ? theta_dt *
                              (-divergence_momentum_flux[component] + j_cross_b[component])
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
            energy_increment(i, j, k) =
                theta_dt * (-divergence_energy_flux - pressure_e * divergence_electron_velocity +
                            joule_heating);
        });
    }
}

void ThetaImplicitMHD::FinishStateUpdate (const amrex::Real end_time)
{
    const amrex::Real inverse_theta = 1.0_rt / m_theta;
    m_state.linComb(inverse_theta, m_state, 1.0_rt - inverse_theta, m_state_old);
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

    const auto& magnetic_field_old = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, 0);
    m_WarpX->FinishMagneticFieldAndApplyBCs(magnetic_field_old, m_theta, end_time);
    FillFluidSources(m_state);

    // E is an algebraic Ohm-law variable rather than an independently evolved
    // endpoint state. Recompute it from final B, rho, momentum, and Ue instead
    // of retaining the theta extrapolation, which generally would not satisfy
    // Ohm's law at t^{n+1}.
    const auto magnetic_field =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, 0);
    m_hybrid_pic_model->CalculatePlasmaCurrent(magnetic_field,
                                               m_WarpX->GetEBUpdateEFlag());
    const auto electric_field =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, 0);
    const auto ion_current =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, 0);
    const auto charge_density =
        m_WarpX->m_fields.get_mr_levels(FieldType::rho_fp, 0);
    m_hybrid_pic_model->HybridPICSolveE(
        electric_field, ion_current, magnetic_field, charge_density,
        m_WarpX->GetEBUpdateEFlag(), true, true, end_time, false);
    m_state.Copy(FieldType::Efield_fp);
}
