/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ThetaImplicitMHD.H"

#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX_Array4.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <cmath>
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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
ion_pressure (const amrex::Real rho, const amrex::Real rho_floor, const amrex::Real rho_reference,
              const amrex::Real pressure_reference, const amrex::Real gamma_i) noexcept
{
    if (pressure_reference == 0.0_rt) {
        return 0.0_rt;
    }
    return pressure_reference * std::pow(std::max(rho, rho_floor) / rho_reference, gamma_i);
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
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_nlsolver_type == NonlinearSolverType::newton ||
                                         m_nlsolver_type == NonlinearSolverType::petsc_snes,
                                     "theta_implicit_mhd requires a JFNK nonlinear solver "
                                     "(implicit_evolve.nonlinear_solver = newton or petsc_snes)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_use_mass_matrices, "particle mass matrices are not applicable to theta_implicit_mhd");
    m_nlsolver->Define(m_state, this);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_nlsolver->GetPreconditionerType() == PreconditionerType::none,
        "theta_implicit_mhd does not yet provide a block preconditioner; "
        "jacobian.pc_type must be none");

    FillFluidSources(m_state);
    m_is_defined = true;
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
                   << "Hall term:                     " << m_hybrid_pic_model->m_include_hall_term
                   << "\n"
                   << "Electron-pressure Ohm term:    "
                   << m_hybrid_pic_model->m_include_electron_pressure_term << "\n"
                   << "Evolve ion fluid:              " << m_evolve_ion_fluid << "\n"
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
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
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
    m_hybrid_pic_model->HybridPICSolveE(electric_field, ion_current, magnetic_field, charge_density,
                                        m_WarpX->GetEBUpdateEFlag(), true, true);

    rhs.Copy(FieldType::Efield_fp);
    for (int component = 0; component < 3; ++component) {
        rhs.getArrayVec()[0][component]->minus(*m_state_old.getArrayVec()[0][component], 0, 1, 0);
    }
    ComputeFluidRHS(rhs, start_time + m_theta * m_dt);
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
    const amrex::Real gamma_i = m_gamma_i;
    const amrex::Real reference_density = m_reference_mass_density;
    const amrex::Real reference_ion_pressure = m_reference_ion_pressure;
    const amrex::Real pressure_floor = m_electron_pressure_floor;
    const bool include_hall_term = m_hybrid_pic_model->m_include_hall_term;
    const bool evolve_ion_fluid = m_evolve_ion_fluid;
    const bool include_joule_heating = m_include_joule_heating;
    const auto eta = m_hybrid_pic_model->m_eta;

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
            amrex::Real electron_pressure_gradient[3] = {0.0_rt, 0.0_rt, 0.0_rt};

            for (int direction = 0; direction < 3; ++direction) {
                if (!active[direction]) {
                    continue;
                }
                int ip = i;
                int jp = j;
                int kp = k;
                int im = i;
                int jm = j;
                int km = k;
                shift_index(ip, jp, kp, direction, 1);
                shift_index(im, jm, km, direction, -1);
                const amrex::Real derivative_scale = 0.5_rt * inverse_cell_size[direction];

                const amrex::Real rho_plus = std::max(rho(ip, jp, kp), density_floor);
                const amrex::Real rho_minus = std::max(rho(im, jm, km), density_floor);
                const amrex::Real pressure_i_plus = ion_pressure(
                    rho_plus, density_floor, reference_density, reference_ion_pressure, gamma_i);
                const amrex::Real pressure_i_minus = ion_pressure(
                    rho_minus, density_floor, reference_density, reference_ion_pressure, gamma_i);

                divergence_momentum +=
                    derivative_scale * (mom(ip, jp, kp, direction) - mom(im, jm, km, direction));

                for (int component = 0; component < 3; ++component) {
                    const amrex::Real flux_plus =
                        mom(ip, jp, kp, direction) * mom(ip, jp, kp, component) / rho_plus +
                        (component == direction ? pressure_i_plus : 0.0_rt);
                    const amrex::Real flux_minus =
                        mom(im, jm, km, direction) * mom(im, jm, km, component) / rho_minus +
                        (component == direction ? pressure_i_minus : 0.0_rt);
                    divergence_momentum_flux[component] +=
                        derivative_scale * (flux_plus - flux_minus);
                }

                const amrex::Real pressure_e_plus =
                    std::max(gamma_e_minus_one * energy(ip, jp, kp), pressure_floor);
                const amrex::Real pressure_e_minus =
                    std::max(gamma_e_minus_one * energy(im, jm, km), pressure_floor);
                electron_pressure_gradient[direction] =
                    derivative_scale * (pressure_e_plus - pressure_e_minus);

                const amrex::Real charge_density_plus = charge_to_mass * rho_plus;
                const amrex::Real charge_density_minus = charge_to_mass * rho_minus;
                const amrex::Real electron_velocity_plus =
                    mom(ip, jp, kp, direction) / rho_plus -
                    (include_hall_term ? j_plasma(ip, jp, kp, direction) / charge_density_plus
                                       : 0.0_rt);
                const amrex::Real electron_velocity_minus =
                    mom(im, jm, km, direction) / rho_minus -
                    (include_hall_term ? j_plasma(im, jm, km, direction) / charge_density_minus
                                       : 0.0_rt);
                divergence_energy_flux +=
                    derivative_scale * (energy(ip, jp, kp) * electron_velocity_plus -
                                        energy(im, jm, km) * electron_velocity_minus);
                divergence_electron_velocity +=
                    derivative_scale * (electron_velocity_plus - electron_velocity_minus);
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
                        ? theta_dt * (-divergence_momentum_flux[component] + j_cross_b[component] -
                                      electron_pressure_gradient[component])
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
    m_WarpX->SetElectricFieldAndApplyBCs(m_state, end_time);

    const auto& magnetic_field_old = m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, 0);
    m_WarpX->FinishMagneticFieldAndApplyBCs(magnetic_field_old, m_theta, end_time);
    FillFluidSources(m_state);
}
