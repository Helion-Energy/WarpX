/* Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *          S. Eric Clark (Helion Energy)
 *          Prabhat Kumar (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "HybridPICModel.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/utils/Communication.H>

#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLNodeTensorLaplacian.H>
#include <ablastr/warn_manager/WarnManager.H>

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/ElectrostaticSolvers/PoissonBoundaryHandler.H"
#include "Python/callbacks.H"
#include "Fields.H"
#include "Fluids/QdsmcParticleContainer.H"
#include "Particles/MultiParticleContainer.H"
#include "ExternalVectorPotential.H"
#include "WarpX.H"

#include <AMReX_Random.H>

#include <string>
#include <vector>

using namespace amrex;
using warpx::fields::FieldType;

HybridPICModel::HybridPICModel ()
{
    ReadParameters();
}

HybridPICModel::~HybridPICModel () = default;

void HybridPICModel::ReadParameters ()
{
    const ParmParse pp_hybrid("hybrid_pic_model");

    // The B-field update is subcycled to improve stability - the number
    // of sub steps can be specified by the user.
    utils::parser::queryWithParser(pp_hybrid, "substeps", m_substeps);
    if (m_substeps % 2 != 0) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPIC",
            "hybrid_pic_model.substeps must be divisible by 2. "
            "The value " + std::to_string(m_substeps) + " is not valid. "
            "Automatically adjusting to " + std::to_string(m_substeps + 1) + ".",
            ablastr::warn_manager::WarnPriority::medium);
        m_substeps += 1;
    }

    // read rkf45 intervals
    std::vector<std::string> rkf45_intervals_string_vec = {"0"};
    pp_hybrid.queryarr("use_rkf45", rkf45_intervals_string_vec);
    m_rkf45_intervals = ablastr::utils::text::IntervalsParser(rkf45_intervals_string_vec);
    utils::parser::queryWithParser(pp_hybrid, "substep_rtol", m_substep_rtol);
    utils::parser::queryWithParser(pp_hybrid, "substep_atol", m_substep_atol);
    utils::parser::queryWithParser(pp_hybrid, "substep_safety", m_substep_safety);
    utils::parser::queryWithParser(pp_hybrid, "substep_max_growth", m_substep_max_growth);
    pp_hybrid.query("max_substep_attempts", m_max_substep_attempts);

    utils::parser::queryWithParser(pp_hybrid, "holmstrom_vacuum_region", m_holmstrom_vacuum_region);

    // The hybrid model requires an electron temperature, reference density
    // and exponent to be given. These values will be used to calculate the
    // electron pressure according to p = n0 * Te * (n/n0)^gamma
    utils::parser::queryWithParser(pp_hybrid, "gamma", m_gamma);
    if (!utils::parser::queryWithParser(pp_hybrid, "elec_temp", m_elec_temp)) {
        Abort("hybrid_pic_model.elec_temp must be specified when using the hybrid solver");
    }
    const bool n0_ref_given = utils::parser::queryWithParser(pp_hybrid, "n0_ref", m_n0_ref);
    if (m_gamma != 1.0 && !n0_ref_given) {
        Abort("hybrid_pic_model.n0_ref should be specified if hybrid_pic_model.gamma != 1");
    }

    pp_hybrid.query("plasma_resistivity(rho,J,t)", m_eta_expression);
    pp_hybrid.query("qdsmc_conduction", m_qdsmc_conduction);
    pp_hybrid.query("qdsmc_conduction_kappa(T,n)", m_qdsmc_kappa_expression);
    pp_hybrid.query("qdsmc_conduction_substeps", m_qdsmc_conduction_substeps);
    utils::parser::queryWithParser(pp_hybrid,
        "qdsmc_conduction_flux_limiter", m_qdsmc_conduction_flux_limiter);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_qdsmc_conduction == "off" || m_qdsmc_conduction == "isotropic"
            || m_qdsmc_conduction == "parallel",
        "hybrid_pic_model.qdsmc_conduction must be 'off', 'isotropic' or "
        "'parallel'");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_qdsmc_conduction_substeps >= 1,
        "hybrid_pic_model.qdsmc_conduction_substeps must be >= 1");
    pp_hybrid.query("plasma_hyper_resistivity(rho,B)", m_eta_h_expression);

    utils::parser::queryWithParser(pp_hybrid, "n_floor", m_n_floor);

    // Master gate for the electron-energy equation. When enabled, K_e is
    // advected each step by fictitious Lagrangian particles moving with V_e
    // (see Phys. Plasmas 31, 012902 (2024)); T_e is recovered from K_e and n_e
    // via the polytropic relation; operator-split source terms are added;
    // Pe = n_e k_B T_e is emitted for the Ohm's-law E-solve. Default off
    // preserves the legacy algebraic adiabatic closure.
    pp_hybrid.query("solve_electron_energy_equation",
                    m_solve_electron_energy_equation);
    pp_hybrid.query("qdsmc_n_floor", m_qdsmc_n_floor);

    pp_hybrid.query("implicit_push_excludes_resistive_field",
                    m_implicit_push_excludes_resistive_field);

    // Darwin (magnetoinductive) field split, consumed by the theta-implicit
    // hybrid solver (see the member documentation in HybridPICModel.H).
    pp_hybrid.query("darwin", m_darwin);
    if (m_darwin) {
        utils::parser::queryWithParser(
            pp_hybrid, "darwin_poisson_relative_tolerance", m_darwin_poisson_rtol);
        utils::parser::queryWithParser(
            pp_hybrid, "darwin_poisson_absolute_tolerance", m_darwin_poisson_atol);
        utils::parser::queryWithParser(
            pp_hybrid, "darwin_poisson_max_iterations", m_darwin_poisson_max_iters);
        pp_hybrid.query("darwin_poisson_verbosity", m_darwin_poisson_verbosity);
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        WARPX_ABORT_WITH_MESSAGE(
            "hybrid_pic_model.darwin is not supported in the 1D radial geometries.");
#endif

        // Vacuum vector-potential recovery (see the member documentation in
        // HybridPICModel.H).
        pp_hybrid.query("darwin_vacuum_recovery", m_darwin_vacuum_recovery);
        if (m_darwin_vacuum_recovery) {
            pp_hybrid.query("darwin_vacuum_recovery_mask",
                            m_darwin_vacuum_recovery_mask);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_darwin_vacuum_recovery_mask == "vacuum"
                || m_darwin_vacuum_recovery_mask == "transition"
                || m_darwin_vacuum_recovery_mask == "global",
                "hybrid_pic_model.darwin_vacuum_recovery_mask must be one of "
                "'vacuum', 'transition', 'global'");
            pp_hybrid.query("darwin_vacuum_recovery_cadence",
                            m_darwin_vacuum_recovery_cadence);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_darwin_vacuum_recovery_cadence == "half"
                || m_darwin_vacuum_recovery_cadence == "full",
                "hybrid_pic_model.darwin_vacuum_recovery_cadence must be "
                "'half' or 'full'");
            pp_hybrid.query("darwin_vacuum_recovery_components",
                            m_darwin_vacuum_recovery_components);
            utils::parser::queryWithParser(
                pp_hybrid, "darwin_vacuum_recovery_density_fraction",
                m_darwin_vacuum_recovery_density_fraction);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_darwin_vacuum_recovery_components == "flux"
                || m_darwin_vacuum_recovery_components == "all",
                "hybrid_pic_model.darwin_vacuum_recovery_components must "
                "be 'flux' or 'all'");
            utils::parser::queryWithParser(
                pp_hybrid, "darwin_vacuum_recovery_relative_tolerance",
                m_darwin_vacrec_rtol);
            utils::parser::queryWithParser(
                pp_hybrid, "darwin_vacuum_recovery_absolute_tolerance",
                m_darwin_vacrec_atol);
            utils::parser::queryWithParser(
                pp_hybrid, "darwin_vacuum_recovery_max_iterations",
                m_darwin_vacrec_max_iters);
            pp_hybrid.query("darwin_vacuum_recovery_verbosity",
                            m_darwin_vacrec_verbosity);
#if defined(WARPX_DIM_1D_Z)
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.darwin_vacuum_recovery is not supported "
                "in 1D (the nodal FD vector-Poisson operator has no 1D "
                "form, and the boundary-driven vacuum problem it targets "
                "has no 1D analog).");
#endif
        }
    }
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_solve_electron_energy_equation,
        "hybrid_pic_model.solve_electron_energy_equation is not supported in "
        "RCYLINDER/RSPHERE geometries yet.");
#endif

    // Resistive electron-heating source (Phys. Plasmas 31, 012902 (2024), Eq. 12):
    //   S_e = Sigma_s nu_{s,e} n_s m_s |V_s - V_e|^2,  nu_{s,e} = Z_s e^2 eta n_e / m_s
    // added per cell to T_e by QDSMCAddJouleHeating, using the e-i relative
    // drift J_plasma/(e n_e) and rho_fp(_s). Reduces to eta J^2 in single species.
    // Independent of whether HybridResistiveDrag is registered.
    // Default off; only consulted when solve_electron_energy_equation is on.
    pp_hybrid.query("include_joule_heating", m_include_joule_heating);

    // Te-threshold Joule redirection: heat electrons where Te < threshold,
    // deposit the Joule energy to ions where Te >= threshold.
    pp_hybrid.query("redirect_joule_to_ions", m_joule_redirect_to_ions);
    utils::parser::queryWithParser(pp_hybrid, "joule_redirect_Te_threshold", m_joule_redirect_Te_eV);

    // Electron-ion thermal equilibration (Q_ei) on T_e:
    //   Q_ei = 3 n_e k_B nu_ei (T_e - T_i),  applied per ion species weighted by
    //   n_s/n_e, cooling T_e toward T_i. nu_ei[1/s] comes from the
    //   electron_ion_relaxation_rate(rho,Te,Ti,t) parser (rho [C/m^3], Te,Ti [eV]).
    //   The matching ion heating is deposited conservatively, so the exchange
    //   conserves energy. Default off; only consulted when
    //   solve_electron_energy_equation is on.
    pp_hybrid.query("include_temperature_relaxation", m_include_temperature_relaxation);
    pp_hybrid.query("electron_ion_relaxation_rate(rho,Te,Ti,t)", m_nu_ei_expression);

    // Determine, from the input deck alone, which optional per-species
    // machinery is needed (the particle containers do not exist yet at
    // parameter-parse time, but the species names are available):
    //   * a per-species resistivity overlay parser for any species,
    //   * a hybrid_resistive_drag collision on any species,
    //   * the electron-energy equation (its Joule and Q_ei sources read the
    //     per-species charge densities).
    // These flags gate the per-species field allocations, the per-species
    // rho/J deposition and the fluid-velocity computations, so hybrid-PIC
    // runs that use none of these features carry zero extra cost.
    {
        std::vector<std::string> species_names;
        const ParmParse pp_particles("particles");
        pp_particles.queryarr("species_names", species_names);
        for (auto const & spec_name : species_names) {
            std::string expr;
            if (pp_hybrid.query(
                "plasma_resistivity_" + spec_name + "(rho_s,rho,Te,J,J_s,B,t)",
                expr)) {
                m_has_per_species_eta = true;
            }
        }

        bool has_resistive_drag = false;
        std::vector<std::string> collision_names;
        const ParmParse pp_collisions("collisions");
        pp_collisions.queryarr("collision_names", collision_names);
        for (auto const & coll_name : collision_names) {
            const ParmParse pp_coll(coll_name);
            std::string coll_type;
            pp_coll.query("type", coll_type);
            if (coll_type == "hybrid_resistive_drag") { has_resistive_drag = true; }
        }

        m_need_fluid_velocities   = m_has_per_species_eta || has_resistive_drag;
        m_need_per_species_fields = m_need_fluid_velocities
                                  || m_solve_electron_energy_equation;
    }

    // convert electron temperature from eV to J
    m_elec_temp *= PhysConst::q_e;

    // external currents
    pp_hybrid.query("Jx_external_grid_function(x,y,z,t)", m_Jx_ext_grid_function);
    pp_hybrid.query("Jy_external_grid_function(x,y,z,t)", m_Jy_ext_grid_function);
    pp_hybrid.query("Jz_external_grid_function(x,y,z,t)", m_Jz_ext_grid_function);

    // check if external currents are specified
    if ((m_Jx_ext_grid_function == "0.0") &&
        (m_Jy_ext_grid_function == "0.0") &&
        (m_Jz_ext_grid_function == "0.0"))
    {
        m_has_external_current = false;
    }

    // external fields
    pp_hybrid.query("add_external_fields", m_add_external_fields);

    if (m_add_external_fields) {
        m_external_vector_potential = std::make_unique<ExternalVectorPotential>();
    }
}

void HybridPICModel::AllocateLevelMFs (
    ablastr::fields::MultiFabRegister & fields,
    int lev, const BoxArray& ba, const DistributionMapping& dm,
    const int ncomps,
    const IntVect& ngJ, const IntVect& ngRho,
    const IntVect& ngEB,
    const IntVect& jx_nodal_flag,
    const IntVect& jy_nodal_flag,
    const IntVect& jz_nodal_flag,
    const IntVect& rho_nodal_flag,
    const IntVect& Ex_nodal_flag,
    const IntVect& Ey_nodal_flag,
    const IntVect& Ez_nodal_flag,
    const IntVect& Bx_nodal_flag,
    const IntVect& By_nodal_flag,
    const IntVect& Bz_nodal_flag) const
{
    using ablastr::fields::Direction;

    // The "hybrid_electron_pressure_fp" multifab stores the electron pressure
    // consumed by the Ohm's-law E-solve. With solve_electron_energy_equation
    // off, it is computed from the algebraic adiabatic closure each step. With
    // it on, Pe = n_e k_B T_e is emitted by QDSMCFillElectronPressureFromTe
    // at the end of each QDSMC entropy-transport step.
    fields.alloc_init(FieldType::hybrid_electron_pressure_fp,
        lev, amrex::convert(ba, rho_nodal_flag),
        dm, ncomps, ngRho, 0.0_rt);

    // Electron temperature T_e (Kelvin). Allocated unconditionally (cheap)
    // so the Te diagnostic functor can always read it: with the energy
    // equation on it is the QDSMC state variable, otherwise it mirrors the
    // algebraic closure's implied temperature (CalculateElectronPressure).
    fields.alloc_init(FieldType::hybrid_electron_temperature_fp,
        lev, amrex::convert(ba, rho_nodal_flag),
        dm, ncomps, ngRho, 0.0_rt);

    // QDSMC electron-energy-equation working fields, only touched (and
    // therefore only allocated) when the energy equation is solved:
    //   * hybrid_entropy_fp              : K_e = T_e * n_e^(1-gamma)
    //   * hybrid_qdsmc_weights_fp        : scratch for deposited N_e
    //   * hybrid_electron_velocity_fp    : three-component V_e on a NODAL
    //     grid, computed each step from V_e = -(J_plasma - J_i)/(q_e n_e)
    //     and consumed by the QDSMC particle SetV step to advect the
    //     entropy carriers.
    if (m_solve_electron_energy_equation) {
        fields.alloc_init(FieldType::hybrid_entropy_fp,
            lev, amrex::convert(ba, rho_nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);
        fields.alloc_init(FieldType::hybrid_qdsmc_weights_fp,
            lev, amrex::convert(ba, rho_nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);
        fields.alloc_init(FieldType::hybrid_electron_velocity_fp, Direction{0},
            lev, amrex::convert(ba, rho_nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);
        fields.alloc_init(FieldType::hybrid_electron_velocity_fp, Direction{1},
            lev, amrex::convert(ba, rho_nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);
        fields.alloc_init(FieldType::hybrid_electron_velocity_fp, Direction{2},
            lev, amrex::convert(ba, rho_nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);

        // Theta-implicit hybrid saved states: T_e^n and J_plasma(B^n).
        // Only written on the implicit path; allocated with the rest of the
        // energy-equation fields for simplicity.
        fields.alloc_init(FieldType::hybrid_electron_temperature_old_fp,
            lev, amrex::convert(ba, rho_nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);
        fields.alloc_init(FieldType::hybrid_current_fp_plasma_old, Direction{0},
            lev, amrex::convert(ba, jx_nodal_flag),
            dm, ncomps, ngJ, 0.0_rt);
        fields.alloc_init(FieldType::hybrid_current_fp_plasma_old, Direction{1},
            lev, amrex::convert(ba, jy_nodal_flag),
            dm, ncomps, ngJ, 0.0_rt);
        fields.alloc_init(FieldType::hybrid_current_fp_plasma_old, Direction{2},
            lev, amrex::convert(ba, jz_nodal_flag),
            dm, ncomps, ngJ, 0.0_rt);
    }

    // Darwin split fields (only with hybrid_pic_model.darwin = 1):
    //   * hybrid_A_fp / hybrid_A_old_fp : magnetic vector potential on the E
    //     staggering, advanced as A^{n+1} = A^n - dt E_T^{n+theta};
    //   * hybrid_B_static_fp            : the time-independent part of B
    //     (B(t) = B_static + curl A with A(0) = 0, so B_static = B(0));
    //   * hybrid_E_long_fp / _old_fp    : longitudinal field E_L = grad(phi);
    //   * hybrid_phi_darwin_fp          : the constraint potential (nodal).
    if (m_darwin) {
        const amrex::IntVect E_stag[3] = {Ex_nodal_flag, Ey_nodal_flag, Ez_nodal_flag};
        const amrex::IntVect B_stag[3] = {Bx_nodal_flag, By_nodal_flag, Bz_nodal_flag};
        for (int dir = 0; dir < 3; ++dir) {
            fields.alloc_init("hybrid_A_fp", Direction{dir},
                lev, amrex::convert(ba, E_stag[dir]), dm, ncomps, ngEB, 0.0_rt);
            fields.alloc_init("hybrid_A_old_fp", Direction{dir},
                lev, amrex::convert(ba, E_stag[dir]), dm, ncomps, ngEB, 0.0_rt);
            fields.alloc_init("hybrid_E_long_fp", Direction{dir},
                lev, amrex::convert(ba, E_stag[dir]), dm, ncomps, ngEB, 0.0_rt);
            fields.alloc_init("hybrid_E_long_old_fp", Direction{dir},
                lev, amrex::convert(ba, E_stag[dir]), dm, ncomps, ngEB, 0.0_rt);
            fields.alloc_init("hybrid_B_static_fp", Direction{dir},
                lev, amrex::convert(ba, B_stag[dir]), dm, ncomps, ngEB, 0.0_rt);
        }
        fields.alloc_init("hybrid_phi_darwin_fp",
            lev, amrex::convert(ba, amrex::IntVect::TheNodeVector()),
            dm, ncomps, ngRho, 0.0_rt);

        // Vacuum A recovery: the nodal correction fields dA (persistent —
        // the previous solution warm-starts each MLMG solve) and the
        // edge-staggered scratch for the field-implied current source.
        if (m_darwin_vacuum_recovery) {
            for (int dir = 0; dir < 3; ++dir) {
                fields.alloc_init("hybrid_A_vac_dA_nodal", Direction{dir},
                    lev, amrex::convert(ba, amrex::IntVect::TheNodeVector()),
                    dm, ncomps, ngRho, 0.0_rt);
                fields.alloc_init("hybrid_J_vac_fp", Direction{dir},
                    lev, amrex::convert(ba, E_stag[dir]), dm, ncomps, ngEB,
                    0.0_rt);
                fields.alloc_init("hybrid_E_vac_target_fp", Direction{dir},
                    lev, amrex::convert(ba, E_stag[dir]), dm, ncomps, ngEB,
                    0.0_rt);
                // A^{n-1} for the BDF2 endpoint reconstruction of the band
                // field. Zero is the EXACT pre-history under the gauge-free
                // initialization (A = 0 for t <= 0, B_static carries B(0)),
                // so BDF2 is valid from the first step.
                fields.alloc_init("hybrid_A_vac_nm1_fp", Direction{dir},
                    lev, amrex::convert(ba, E_stag[dir]), dm, ncomps, ngEB,
                    0.0_rt);
            }
        }
    }

    // The "hybrid_rho_fp_temp" multifab is used to store the ion charge density
    // interpolated or extrapolated to appropriate timesteps.
    fields.alloc_init(FieldType::hybrid_rho_fp_temp,
        lev, amrex::convert(ba, rho_nodal_flag),
        dm, ncomps, ngRho, 0.0_rt);

    // The "hybrid_current_fp_temp" multifab is used to store the ion current density
    // interpolated or extrapolated to appropriate timesteps.
    fields.alloc_init(FieldType::hybrid_current_fp_temp, Direction{0},
        lev, amrex::convert(ba, jx_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_current_fp_temp, Direction{1},
        lev, amrex::convert(ba, jy_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_current_fp_temp, Direction{2},
        lev, amrex::convert(ba, jz_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);

    // The "hybrid_current_fp_plasma" multifab stores the total plasma current calculated
    // as the curl of B minus any external current.
    fields.alloc_init(FieldType::hybrid_current_fp_plasma, Direction{0},
        lev, amrex::convert(ba, jx_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_current_fp_plasma, Direction{1},
        lev, amrex::convert(ba, jy_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_current_fp_plasma, Direction{2},
        lev, amrex::convert(ba, jz_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt);

    // Per-species ion fields - one set per charged species. current_fp_s
    // and rho_fp_s are deposited from particles and accumulated into the
    // global current_fp / rho_fp; Vs_fp_s is the bulk velocity J_s/rho_s,
    // used by the resistive-drag operator to shift each species' particles
    // toward V_e without collapsing the thermal moment. Only allocated when
    // a feature that consumes them is active (see m_need_per_species_fields
    // and m_need_fluid_velocities).
    if (m_need_per_species_fields) {
        auto const & mypc = WarpX::GetInstance().GetPartContainer();
        for (auto const & spec : mypc.GetSpeciesNames()) {
            if (mypc.GetParticleContainerFromName(spec).getCharge() == 0._prt) { continue; }
            fields.alloc_init("current_fp_" + spec, Direction{0},
                lev, amrex::convert(ba, jx_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
            fields.alloc_init("current_fp_" + spec, Direction{1},
                lev, amrex::convert(ba, jy_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
            fields.alloc_init("current_fp_" + spec, Direction{2},
                lev, amrex::convert(ba, jz_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
            fields.alloc_init("rho_fp_" + spec,
                lev, amrex::convert(ba, rho_nodal_flag), dm, ncomps, ngRho, 0.0_rt);
            if (m_need_fluid_velocities) {
                fields.alloc_init("Vs_fp_" + spec, Direction{0},
                    lev, amrex::convert(ba, jx_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
                fields.alloc_init("Vs_fp_" + spec, Direction{1},
                    lev, amrex::convert(ba, jy_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
                fields.alloc_init("Vs_fp_" + spec, Direction{2},
                    lev, amrex::convert(ba, jz_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
            }
        }
    }

    // Electron fluid velocity V_e on the grid, V_e = (J_i - J_plasma)/rho.
    // Face-staggered like J for direct gather by the resistive-drag operator.
    if (m_need_fluid_velocities) {
        fields.alloc_init("Ve_fp", Direction{0},
            lev, amrex::convert(ba, jx_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
        fields.alloc_init("Ve_fp", Direction{1},
            lev, amrex::convert(ba, jy_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
        fields.alloc_init("Ve_fp", Direction{2},
            lev, amrex::convert(ba, jz_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
    }

    // Per-species resistive overlay added to Ohm's-law E (see
    // ComputeResistiveOverlay). Computed once per step and read by every
    // (subcycled) E-solve; staggered like J (== E component staggering).
    if (m_has_per_species_eta) {
        fields.alloc_init("hybrid_eta_overlay_fp", Direction{0},
            lev, amrex::convert(ba, jx_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
        fields.alloc_init("hybrid_eta_overlay_fp", Direction{1},
            lev, amrex::convert(ba, jy_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
        fields.alloc_init("hybrid_eta_overlay_fp", Direction{2},
            lev, amrex::convert(ba, jz_nodal_flag), dm, ncomps, ngJ, 0.0_rt);
    }

    // the external current density multifab matches the current staggering and
    // one ghost cell is used since we interpolate the current to a nodal grid
    if (m_has_external_current) {
        fields.alloc_init(FieldType::hybrid_current_fp_external, Direction{0},
            lev, amrex::convert(ba, jx_nodal_flag),
            dm, ncomps, IntVect(1), 0.0_rt);
        fields.alloc_init(FieldType::hybrid_current_fp_external, Direction{1},
            lev, amrex::convert(ba, jy_nodal_flag),
            dm, ncomps, IntVect(1), 0.0_rt);
        fields.alloc_init(FieldType::hybrid_current_fp_external, Direction{2},
            lev, amrex::convert(ba, jz_nodal_flag),
            dm, ncomps, IntVect(1), 0.0_rt);
    }

    if (m_add_external_fields) {
        m_external_vector_potential->AllocateLevelMFs(
            fields,
            lev, ba, dm,
            ncomps, ngEB,
            Ex_nodal_flag, Ey_nodal_flag, Ez_nodal_flag,
            Bx_nodal_flag, By_nodal_flag, Bz_nodal_flag
        );
    }

#ifdef WARPX_DIM_RZ
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (ncomps == 1),
        "Ohm's law solver only support m = 0 azimuthal mode at present.");
#endif
}

void HybridPICModel::InitData (const ablastr::fields::MultiFabRegister& fields)
{
    m_resistivity_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_eta_expression, {"rho","J","t"}));
    m_eta = m_resistivity_parser->compile<3>();

    if (m_qdsmc_conduction != "off") {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_solve_electron_energy_equation,
            "hybrid_pic_model.qdsmc_conduction requires "
            "hybrid_pic_model.solve_electron_energy_equation = true");
        m_qdsmc_kappa_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_qdsmc_kappa_expression, {"T","n"}));
        m_qdsmc_kappa = m_qdsmc_kappa_parser->compile<2>();
    }
    const std::set<std::string> resistivity_symbols = m_resistivity_parser->symbols();
    m_resistivity_has_J_dependence += resistivity_symbols.count("J");

    // Electron-ion energy-equilibration rate nu_ei(rho,Te,Ti,t) for the Q_ei term.
    m_nu_ei_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_nu_ei_expression, {"rho","Te","Ti","t"}));
    m_nu_ei = m_nu_ei_parser->compile<4>();

    // --- Per-species resistivity overlay (Phys. Plasmas 31, 012902 (2024), Eq. 10) ---
    // Optional. For any charged species {spec} the user may supply
    //   hybrid_pic_model.plasma_resistivity_{spec}(rho_s,rho,Te,J,J_s,B,t)="..."
    // This parser overlays species-resolved physics on top of the global
    // m_eta parser; the total per-species effective resistivity used by
    // Ohm's law, the Joule-heating source, and the resistive-drag operator
    // is the sum eta_global + eta_per_species_s (so existing single-eta
    // input scripts keep their exact behaviour with eta_per_species_s = 0).
    // Done here in InitData rather than ReadParameters because species
    // names are not available at parameter-parse time.
    {
        const amrex::ParmParse pp_hybrid_init("hybrid_pic_model");
        auto const & mypc = WarpX::GetInstance().GetPartContainer();
        std::vector<std::string> species_with_per_eta;
        for (auto const & spec_name : mypc.GetSpeciesNames()) {
            if (mypc.GetParticleContainerFromName(spec_name).getCharge() == 0._prt) {
                continue;
            }
            std::string const param_name =
                "plasma_resistivity_" + spec_name + "(rho_s,rho,Te,J,J_s,B,t)";
            std::string expr;
            if (pp_hybrid_init.query(param_name, expr)) {
                auto parser = std::make_unique<amrex::Parser>(
                    utils::parser::makeParser(
                        expr, {"rho_s","rho","Te","J","J_s","B","t"}));
                m_eta_per_species[spec_name] = parser->compile<7>();
                m_per_species_resistivity_parser[spec_name] = std::move(parser);
                species_with_per_eta.push_back(spec_name);
                m_has_per_species_eta = true;
            }
        }

        // Startup summary on rank 0: which species use only the global
        // eta, and which additionally have a per-species overlay.
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n[HybridPICModel] Resistivity configuration\n";
            amrex::Print() << "  global plasma_resistivity(rho,J,t) = "
                           << m_eta_expression << "\n";
            if (m_has_per_species_eta) {
                amrex::Print() << "  per-species overlays (eta_s_eff = "
                                  "eta_global + eta_per_species):\n";
                for (auto const & spec_name : mypc.GetSpeciesNames()) {
                    if (mypc.GetParticleContainerFromName(spec_name).getCharge() == 0._prt) {
                        continue;
                    }
                    auto it = m_per_species_resistivity_parser.find(spec_name);
                    if (it != m_per_species_resistivity_parser.end()) {
                        std::string expr;
                        pp_hybrid_init.query(
                            "plasma_resistivity_" + spec_name +
                             "(rho_s,rho,Te,J,J_s,B,t)", expr);
                        amrex::Print() << "    " << spec_name
                                       << "  : " << expr << "\n";
                    } else {
                        amrex::Print() << "    " << spec_name
                                       << "  : (global only)\n";
                    }
                }
            } else {
                amrex::Print() << "  per-species overlays              : none "
                                  "(all charged species use the global parser)\n";
            }
            amrex::Print() << "\n";
        }
    }

    m_include_hyper_resistivity_term = (m_eta_h_expression != "0.0");
    m_hyper_resistivity_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_eta_h_expression, {"rho","B"}));
    m_eta_h = m_hyper_resistivity_parser->compile<2>();
    const std::set<std::string> hyper_resistivity_symbols = m_hyper_resistivity_parser->symbols();
    m_hyper_resistivity_has_B_dependence += hyper_resistivity_symbols.count("B");

    if (m_has_external_current) {
        m_J_external_parser[0] = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_Jx_ext_grid_function,{"x","y","z","t"}));
        m_J_external_parser[1] = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_Jy_ext_grid_function,{"x","y","z","t"}));
        m_J_external_parser[2] = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_Jz_ext_grid_function,{"x","y","z","t"}));
        m_J_external[0] = m_J_external_parser[0]->compile<4>();
        m_J_external[1] = m_J_external_parser[1]->compile<4>();
        m_J_external[2] = m_J_external_parser[2]->compile<4>();

        // check if the external current parsers depend on time
        for (int i=0; i<3; i++) {
            const std::set<std::string> J_ext_symbols = m_J_external_parser[i]->symbols();
            m_external_current_has_time_dependence += J_ext_symbols.count("t");
        }
    }

    auto& warpx = WarpX::GetInstance();
    using ablastr::fields::Direction;

    // Get the grid staggering of the fields involved in calculating E
    amrex::IntVect Jx_stag = fields.get(FieldType::current_fp, Direction{0}, 0)->ixType().toIntVect();
    amrex::IntVect Jy_stag = fields.get(FieldType::current_fp, Direction{1}, 0)->ixType().toIntVect();
    amrex::IntVect Jz_stag = fields.get(FieldType::current_fp, Direction{2}, 0)->ixType().toIntVect();
    amrex::IntVect Bx_stag = fields.get(FieldType::Bfield_fp, Direction{0}, 0)->ixType().toIntVect();
    amrex::IntVect By_stag = fields.get(FieldType::Bfield_fp, Direction{1}, 0)->ixType().toIntVect();
    amrex::IntVect Bz_stag = fields.get(FieldType::Bfield_fp, Direction{2}, 0)->ixType().toIntVect();
    amrex::IntVect Ex_stag = fields.get(FieldType::Efield_fp, Direction{0}, 0)->ixType().toIntVect();
    amrex::IntVect Ey_stag = fields.get(FieldType::Efield_fp, Direction{1}, 0)->ixType().toIntVect();
    amrex::IntVect Ez_stag = fields.get(FieldType::Efield_fp, Direction{2}, 0)->ixType().toIntVect();

    // copy data to device
    for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        Jx_IndexType[idim]    = Jx_stag[idim];
        Jy_IndexType[idim]    = Jy_stag[idim];
        Jz_IndexType[idim]    = Jz_stag[idim];
        Bx_IndexType[idim]    = Bx_stag[idim];
        By_IndexType[idim]    = By_stag[idim];
        Bz_IndexType[idim]    = Bz_stag[idim];
        Ex_IndexType[idim]    = Ex_stag[idim];
        Ey_IndexType[idim]    = Ey_stag[idim];
        Ez_IndexType[idim]    = Ez_stag[idim];
    }

    // Below we set all the unused dimensions to have nodal values for J, B & E
    // since these values will be interpolated onto a nodal grid - if this is
    // not done the Interp function returns nonsense values.
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_1D_Z) || \
    defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    Jx_IndexType[2]    = 1;
    Jy_IndexType[2]    = 1;
    Jz_IndexType[2]    = 1;
    Bx_IndexType[2]    = 1;
    By_IndexType[2]    = 1;
    Bz_IndexType[2]    = 1;
    Ex_IndexType[2]    = 1;
    Ey_IndexType[2]    = 1;
    Ez_IndexType[2]    = 1;
#endif
#if defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    Jx_IndexType[1]    = 1;
    Jy_IndexType[1]    = 1;
    Jz_IndexType[1]    = 1;
    Bx_IndexType[1]    = 1;
    By_IndexType[1]    = 1;
    Bz_IndexType[1]    = 1;
    Ex_IndexType[1]    = 1;
    Ey_IndexType[1]    = 1;
    Ez_IndexType[1]    = 1;
#endif

    if (m_has_external_current) {
        // Initialize external current - note that this approach skips the check
        // if the current is time dependent which is what needs to be done to
        // write time independent fields on the first step.
        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            warpx.ComputeExternalFieldOnGridUsingParser(
                FieldType::hybrid_current_fp_external,
                m_J_external[0],
                m_J_external[1],
                m_J_external[2],
                lev, PatchType::fine,
                warpx.GetEBUpdateEFlag());
        }
    }

    if (m_add_external_fields) {
        m_external_vector_potential->InitData();
    }

    // Seed T_e with the uniform value parsed from <hybrid>.elec_temp (in
    // Joules after ReadParameters, so dividing by k_B gives Kelvin). Done
    // unconditionally so the iter-0 diag dump -- which WarpX::InitData()
    // flushes BEFORE the first call to HybridPICInitializeRhoJandB -- sees
    // a meaningful T_e rather than the zero-initialized allocation. With
    // the energy equation on, this is the starting K_e value the QDSMC
    // particles will read on the first step; with it off, the diagnostic
    // value gets overwritten each step by CalculateElectronPressure.
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        amrex::MultiFab & Te_mf = *warpx.m_fields.get(
            FieldType::hybrid_electron_temperature_fp, lev);
        Te_mf.setVal(m_elec_temp / PhysConst::kb);
    }

    // QDSMC: lazy-construct the fictitious-particle container and lay one
    // particle per cell.
    if (m_solve_electron_energy_equation) {
        m_qdsmc_pc = std::make_unique<QdsmcParticleContainer>(&warpx);
        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            m_qdsmc_pc->InitParticles(lev);
        }
        if (m_qdsmc_conduction != "off") {
            // One-pass conduction nodes get their own container (the
            // advection markers above are persistent).
            m_qdsmc_cond_pc = std::make_unique<QdsmcParticleContainer>(&warpx);
        }
    }
}

void HybridPICModel::GetCurrentExternal ()
{
    if (!m_external_current_has_time_dependence) { return; }

    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        warpx.ComputeExternalFieldOnGridUsingParser(
            FieldType::hybrid_current_fp_external,
            m_J_external[0],
            m_J_external[1],
            m_J_external[2],
            lev, PatchType::fine,
            warpx.GetEBUpdateEFlag());
    }
}

void HybridPICModel::CalculatePlasmaCurrent (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E) const
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        CalculatePlasmaCurrent(Bfield[lev], eb_update_E[lev], lev);
    }
}

void HybridPICModel::CalculatePlasmaCurrent (
    ablastr::fields::VectorField const& Bfield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
    const int lev) const
{
    ABLASTR_PROFILE("HybridPICModel::CalculatePlasmaCurrent()");

    auto& warpx = WarpX::GetInstance();
    ablastr::fields::VectorField current_fp_plasma = warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    warpx.get_pointer_fdtd_solver_fp(lev)->CalculateCurrentAmpere(
        current_fp_plasma, Bfield, eb_update_E, lev
    );

    if (m_has_external_current) {
        // Subtract external current from "Ampere" current calculated above. Note
        // we need to include 1 ghost cell since later we will interpolate the
        // plasma current to a nodal grid.
        ablastr::fields::VectorField current_fp_external = warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_external, lev);
        for (int i=0; i<3; i++) {
            current_fp_plasma[i]->minus(*current_fp_external[i], 0, 1, 1);
        }
    }
}

void HybridPICModel::HybridPICSolveE (
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    const bool solve_for_Faraday,
    const std::optional<bool> include_resistivity) const
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        HybridPICSolveE(
            Efield[lev], Jfield[lev], Bfield[lev], *rhofield[lev],
            eb_update_E[lev], lev, solve_for_Faraday, include_resistivity
        );
    }
    // Allow execution of Python callback after E-field push
    ExecutePythonCallback("afterEpush");
}

void HybridPICModel::HybridPICSolveE (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
    const int lev, const bool solve_for_Faraday,
    const std::optional<bool> include_resistivity) const
{
    ABLASTR_PROFILE("WarpX::HybridPICSolveE()");

    HybridPICSolveE(
        Efield, Jfield, Bfield, rhofield, eb_update_E, lev,
        PatchType::fine, solve_for_Faraday, include_resistivity
    );
    if (lev > 0)
    {
        amrex::Abort(Utils::TextMsg::Err(
        "HybridPICSolveE: Only one level implemented for hybrid-PIC solver."));
    }
}

void HybridPICModel::HybridPICSolveE (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
    const int lev, PatchType patch_type,
    const bool solve_for_Faraday,
    const std::optional<bool> include_resistivity) const
{
    auto& warpx = WarpX::GetInstance();

    ablastr::fields::VectorField current_fp_plasma = warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    auto* const electron_pressure_fp = warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);

    // Solve E field in regular cells
    // The resistive (and hyper-resistive) terms default to the historical
    // coupling with solve_for_Faraday; the theta-implicit solver overrides
    // this to assemble the full Ohm E (grad(Pe) and eta*J together).
    const bool incl_eta = include_resistivity.value_or(solve_for_Faraday);
    warpx.get_pointer_fdtd_solver_fp(lev)->HybridPICSolveE(
        Efield, current_fp_plasma, Jfield, Bfield, rhofield,
        *electron_pressure_fp, eb_update_E, lev, this, solve_for_Faraday,
        incl_eta
    );
    amrex::Real const time = warpx.gett_old(0) + warpx.getdt(0);
    warpx.ApplyEfieldBoundary(lev, patch_type, time);
}

void HybridPICModel::CalculateElectronPressure() const
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        CalculateElectronPressure(lev);
    }
}

void HybridPICModel::CalculateElectronPressure(const int lev) const
{
    ABLASTR_PROFILE("WarpX::CalculateElectronPressure()");

    auto& warpx = WarpX::GetInstance();
    ablastr::fields::ScalarField electron_pressure_fp = warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    ablastr::fields::ScalarField rho_fp = warpx.m_fields.get(FieldType::rho_fp, lev);

    // Calculate the electron pressure using rho^{n+1}.
    FillElectronPressureMF(
        *electron_pressure_fp,
        *rho_fp
    );
    warpx.ApplyElectronPressureBoundary(lev, PatchType::fine);

    // Diagnostic-only: mirror the algebraic closure's implied electron
    // temperature into hybrid_electron_temperature_fp so the Te diag dump
    // is meaningful even when solve_electron_energy_equation is off. With
    // the polytropic Pe = n0_ref^(-gamma) * (rho/q_e)^gamma * k_B * Te_ref,
    // the per-cell implied T_e is just Pe / (n_e * k_B). When the energy
    // equation is on, this path is skipped and T_e is owned by QDSMC
    // (without the gate, the first-step HybridPICInitializeRhoJandB call
    // would silently overwrite any user-loaded initial T_e profile).
    if (!m_solve_electron_energy_equation) {
        amrex::MultiFab       & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
        amrex::MultiFab const & Pe  = *electron_pressure_fp;
        amrex::MultiFab const & rho = *rho_fp;
        auto const rho_floor = PhysConst::q_e * m_n_floor;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Array4<amrex::Real>       const & Te_arr  = Te.array(mfi);
            amrex::Array4<amrex::Real const> const & Pe_arr  = Pe.const_array(mfi);
            amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);
            amrex::Box const & tbox = mfi.tilebox();
            amrex::ParallelFor(tbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real const rho_val = std::max(rho_arr(i,j,k), rho_floor);
                amrex::Real const ne      = rho_val / PhysConst::q_e;
                Te_arr(i,j,k) = Pe_arr(i,j,k) / (ne * PhysConst::kb);
            });
        }
    }

    ablastr::utils::communication::FillBoundary(
        *electron_pressure_fp,
        WarpX::do_single_precision_comms,
        warpx.Geom(lev).periodicity(),
        true);
}

void HybridPICModel::CalculateElectronFluidVelocity () const
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        CalculateElectronFluidVelocity(lev);
    }
}

void HybridPICModel::CalculateElectronFluidVelocity (const int lev) const
{
    ABLASTR_PROFILE("WarpX::CalculateElectronFluidVelocity()");
    using namespace ablastr::coarsen::sample;

    auto & warpx = WarpX::GetInstance();
    ablastr::fields::VectorField Ve = warpx.m_fields.get_alldirs("Ve_fp", lev);
    ablastr::fields::VectorField Ji = warpx.m_fields.get_alldirs(FieldType::current_fp, lev);
    ablastr::fields::VectorField J  = warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    amrex::MultiFab const & rho_field = *warpx.m_fields.get(FieldType::rho_fp, lev);

    auto const Jx_stag = Jx_IndexType;
    auto const Jy_stag = Jy_IndexType;
    auto const Jz_stag = Jz_IndexType;
    amrex::GpuArray<int, 3> const nodal = {1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};
    auto const rho_floor = m_n_floor * PhysConst::q_e;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Ve[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Vex = Ve[0]->array(mfi);
        Array4<Real> const& Vey = Ve[1]->array(mfi);
        Array4<Real> const& Vez = Ve[2]->array(mfi);
        Array4<Real const> const& Jix = Ji[0]->const_array(mfi);
        Array4<Real const> const& Jiy = Ji[1]->const_array(mfi);
        Array4<Real const> const& Jiz = Ji[2]->const_array(mfi);
        Array4<Real const> const& Jx  = J[0]->const_array(mfi);
        Array4<Real const> const& Jy  = J[1]->const_array(mfi);
        Array4<Real const> const& Jz  = J[2]->const_array(mfi);
        Array4<Real const> const& rho = rho_field.const_array(mfi);

        Box const& tx = mfi.tilebox(Ve[0]->ixType().toIntVect());
        Box const& ty = mfi.tilebox(Ve[1]->ixType().toIntVect());
        Box const& tz = mfi.tilebox(Ve[2]->ixType().toIntVect());

        amrex::ParallelFor(tx, ty, tz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                Real const rho_val = std::max(Interp(rho, nodal, Jx_stag, coarsen, i, j, k, 0), rho_floor);
                Vex(i, j, k) = (Jix(i, j, k) - Jx(i, j, k)) / rho_val;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                Real const rho_val = std::max(Interp(rho, nodal, Jy_stag, coarsen, i, j, k, 0), rho_floor);
                Vey(i, j, k) = (Jiy(i, j, k) - Jy(i, j, k)) / rho_val;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                Real const rho_val = std::max(Interp(rho, nodal, Jz_stag, coarsen, i, j, k, 0), rho_floor);
                Vez(i, j, k) = (Jiz(i, j, k) - Jz(i, j, k)) / rho_val;
            }
        );
    }

    // FillBoundary first so the filter sees valid ghost values, then apply
    // the same binomial filter used on J (suppresses grid-scale noise that
    // would otherwise be injected into particles by the gather inside the
    // drag operator). Final FillBoundary refreshes ghosts after filtering.
    for (int idim = 0; idim < 3; ++idim) {
        ablastr::utils::communication::FillBoundary(
            *Ve[idim], WarpX::do_single_precision_comms,
            warpx.Geom(lev).periodicity(), true);
    }
    if (WarpX::use_filter) {
        warpx.ApplyFilterMF(
            warpx.m_fields.get_mr_levels_alldirs("Ve_fp", warpx.finestLevel()), lev);
        for (int idim = 0; idim < 3; ++idim) {
            ablastr::utils::communication::FillBoundary(
                *Ve[idim], WarpX::do_single_precision_comms,
                warpx.Geom(lev).periodicity(), true);
        }
    }
}

void HybridPICModel::CalculateIonFluidVelocity () const
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        CalculateIonFluidVelocity(lev);
    }
}

void HybridPICModel::CalculateIonFluidVelocity (const int lev) const
{
    ABLASTR_PROFILE("WarpX::CalculateIonFluidVelocity()");
    using namespace ablastr::coarsen::sample;

    auto & warpx = WarpX::GetInstance();
    auto const & mypc = warpx.GetPartContainer();

    auto const Jx_stag = Jx_IndexType;
    auto const Jy_stag = Jy_IndexType;
    auto const Jz_stag = Jz_IndexType;
    amrex::GpuArray<int, 3> const nodal = {1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};
    auto const rho_floor = m_n_floor * PhysConst::q_e;

    for (auto const & spec : mypc.GetSpeciesNames()) {
        if (mypc.GetParticleContainerFromName(spec).getCharge() == 0._prt) { continue; }
        ablastr::fields::VectorField Vs = warpx.m_fields.get_alldirs("Vs_fp_" + spec, lev);
        ablastr::fields::VectorField Js = warpx.m_fields.get_alldirs("current_fp_" + spec, lev);
        amrex::MultiFab const & rho_s = *warpx.m_fields.get("rho_fp_" + spec, lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*Vs[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
            Array4<Real> const& Vsx = Vs[0]->array(mfi);
            Array4<Real> const& Vsy = Vs[1]->array(mfi);
            Array4<Real> const& Vsz = Vs[2]->array(mfi);
            Array4<Real const> const& Jsx = Js[0]->const_array(mfi);
            Array4<Real const> const& Jsy = Js[1]->const_array(mfi);
            Array4<Real const> const& Jsz = Js[2]->const_array(mfi);
            Array4<Real const> const& rho = rho_s.const_array(mfi);

            Box const& tx = mfi.tilebox(Vs[0]->ixType().toIntVect());
            Box const& ty = mfi.tilebox(Vs[1]->ixType().toIntVect());
            Box const& tz = mfi.tilebox(Vs[2]->ixType().toIntVect());

            amrex::ParallelFor(tx, ty, tz,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    Real const rho_val = std::max(Interp(rho, nodal, Jx_stag, coarsen, i, j, k, 0), rho_floor);
                    Vsx(i, j, k) = Jsx(i, j, k) / rho_val;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    Real const rho_val = std::max(Interp(rho, nodal, Jy_stag, coarsen, i, j, k, 0), rho_floor);
                    Vsy(i, j, k) = Jsy(i, j, k) / rho_val;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    Real const rho_val = std::max(Interp(rho, nodal, Jz_stag, coarsen, i, j, k, 0), rho_floor);
                    Vsz(i, j, k) = Jsz(i, j, k) / rho_val;
                }
            );
        }

        for (int idim = 0; idim < 3; ++idim) {
            ablastr::utils::communication::FillBoundary(
                *Vs[idim], WarpX::do_single_precision_comms,
                warpx.Geom(lev).periodicity(), true);
        }
        // Same J-style binomial filter as in CalculateElectronFluidVelocity.
        if (WarpX::use_filter) {
            warpx.ApplyFilterMF(
                warpx.m_fields.get_mr_levels_alldirs("Vs_fp_" + spec, warpx.finestLevel()),
                lev);
            for (int idim = 0; idim < 3; ++idim) {
                ablastr::utils::communication::FillBoundary(
                    *Vs[idim], WarpX::do_single_precision_comms,
                    warpx.Geom(lev).periodicity(), true);
            }
        }
    }
}

void HybridPICModel::ComputeResistiveOverlay (
    int const lev,
    amrex::MultiFab & overlay_x,
    amrex::MultiFab & overlay_y,
    amrex::MultiFab & overlay_z) const
{
    ABLASTR_PROFILE("HybridPICModel::ComputeResistiveOverlay()");

    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // Result is the per-species friction contribution added to Ohm's-law E:
    //   overlay_d(i,j,k) = Sigma_s [ eta_s_per * rho_s * rho / rho_sum * (V_s_d - V_e_d) ]
    // summed over charged species with a registered per-species parser.
    // eta_s_per is evaluated at (rho_s, rho, T_e [K], |J|, |J_s|, |B|, t) per cell
    // at the d-component staggering. We always zero the output first; if
    // no per-species parsers are registered we return immediately (single-eta
    // path is then exactly recovered when the caller adds a field of zeros).

    overlay_x.setVal(0.0_rt);
    overlay_y.setVal(0.0_rt);
    overlay_z.setVal(0.0_rt);

    if (!m_has_per_species_eta) { return; }

    auto & warpx = WarpX::GetInstance();
    auto const & mypc = warpx.GetPartContainer();
    auto const species_names = mypc.GetSpeciesNames();
    auto const t_new = warpx.gett_new(lev);

    amrex::MultiFab const & rho_total =
        *warpx.m_fields.get(FieldType::rho_fp, lev);
    amrex::MultiFab const & Te_K =
        *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    ablastr::fields::VectorField J_plasma =
        warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    ablastr::fields::VectorField B_fp =
        warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev);
    ablastr::fields::VectorField Ve_fp =
        warpx.m_fields.get_alldirs("Ve_fp", lev);

    auto const rho_floor = PhysConst::q_e * m_n_floor;

    // Precompute rho_sum = Sigma_t rho_fp_t over all charged species. This is
    // the unscaled "raw deposit" sum, paired with rho_s_raw in the same
    // form so the 2pi*r RZ factor cancels in the species-fraction ratio.
    amrex::MultiFab rho_sum(rho_total.boxArray(), rho_total.DistributionMap(),
                            1, rho_total.nGrowVect());
    rho_sum.setVal(0.0_rt);
    for (auto const & spec_name : species_names) {
        if (mypc.GetParticleContainerFromName(spec_name).getCharge() == 0._prt) {
            continue;
        }
        amrex::MultiFab const & rho_s =
            *warpx.m_fields.get("rho_fp_" + spec_name, lev);
        amrex::MultiFab::Add(rho_sum, rho_s, 0, 0, 1, rho_total.nGrowVect());
    }

    amrex::GpuArray<int, 3> const & Jx_stag = Jx_IndexType;
    amrex::GpuArray<int, 3> const & Jy_stag = Jy_IndexType;
    amrex::GpuArray<int, 3> const & Jz_stag = Jz_IndexType;
    amrex::GpuArray<int, 3> const & Bx_stag = Bx_IndexType;
    amrex::GpuArray<int, 3> const & By_stag = By_IndexType;
    amrex::GpuArray<int, 3> const & Bz_stag = Bz_IndexType;
    amrex::GpuArray<int, 3> const nodal   = {1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};

    // Single kernel pass per (species, direction) pair. Each pass reads
    // the species's parser executor by value (amrex::ParserExecutor is
    // device-callable), interpolates the global and per-species scalar
    // fields to the d-component staggering, evaluates the parser, and
    // accumulates the contribution into overlay_<d>. The d-component
    // values of V_s and V_e are read directly because Vs_fp and Ve_fp are
    // both at the d staggering by construction; only the cross components
    // of J, J_s and B need interp to compute magnitudes.
    for (auto const & spec_name : species_names) {
        auto & pc = mypc.GetParticleContainerFromName(spec_name);
        if (pc.getCharge() == 0._prt) { continue; }

        auto eta_s_per_it = m_eta_per_species.find(spec_name);
        if (eta_s_per_it == m_eta_per_species.end()) { continue; }
        auto const eta_s_per = eta_s_per_it->second;

        amrex::MultiFab const & rho_s_mf =
            *warpx.m_fields.get("rho_fp_" + spec_name, lev);
        ablastr::fields::VectorField Vs_fp =
            warpx.m_fields.get_alldirs("Vs_fp_" + spec_name, lev);
        ablastr::fields::VectorField Js_fp =
            warpx.m_fields.get_alldirs("current_fp_" + spec_name, lev);

        // Lambda: accumulate this species's overlay contribution into a
        // single output multifab at the given d staggering.
        auto accumulate_one_direction = [&] (
            amrex::MultiFab       & out,
            amrex::GpuArray<int,3>  const d_stag,
            int                     d_idx)
        {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(out, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Array4<amrex::Real>       const & out_arr     = out.array(mfi);
                amrex::Array4<amrex::Real const> const & rho_arr     = rho_total.const_array(mfi);
                amrex::Array4<amrex::Real const> const & rhos_arr    = rho_s_mf.const_array(mfi);
                amrex::Array4<amrex::Real const> const & rhosum_arr  = rho_sum.const_array(mfi);
                amrex::Array4<amrex::Real const> const & Te_arr      = Te_K.const_array(mfi);
                amrex::Array4<amrex::Real const> const & Jpx_arr     = J_plasma[0]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & Jpy_arr     = J_plasma[1]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & Jpz_arr     = J_plasma[2]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & Jsx_arr     = Js_fp[0]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & Jsy_arr     = Js_fp[1]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & Jsz_arr     = Js_fp[2]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & Bx_arr      = B_fp[0]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & By_arr      = B_fp[1]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & Bz_arr      = B_fp[2]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & Vsd_arr     = Vs_fp[d_idx]->const_array(mfi);
                amrex::Array4<amrex::Real const> const & Ved_arr     = Ve_fp[d_idx]->const_array(mfi);

                amrex::Box const & tbox = mfi.tilebox(out.ixType().toIntVect());
                amrex::ParallelFor(tbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    using ablastr::coarsen::sample::Interp;
                    amrex::Real const rho_val = Interp(rho_arr, nodal, d_stag, coarsen, i, j, k, 0);
                    if (rho_val <= rho_floor) { return; }

                    amrex::Real const rhos_val   = Interp(rhos_arr,   nodal, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const rhosum_val = std::max(
                        Interp(rhosum_arr, nodal, d_stag, coarsen, i, j, k, 0), rho_floor);
                    amrex::Real const Te_val     = Interp(Te_arr,     nodal, d_stag, coarsen, i, j, k, 0);

                    // |J|, |J_s|, |B| at d staggering. Interp every Yee
                    // component (the d-component interp from its own
                    // staggering to itself is identity, so this is safe).
                    amrex::Real const jpx = Interp(Jpx_arr, Jx_stag, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const jpy = Interp(Jpy_arr, Jy_stag, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const jpz = Interp(Jpz_arr, Jz_stag, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const Jmag = std::sqrt(jpx*jpx + jpy*jpy + jpz*jpz);

                    amrex::Real const jsx = Interp(Jsx_arr, Jx_stag, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const jsy = Interp(Jsy_arr, Jy_stag, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const jsz = Interp(Jsz_arr, Jz_stag, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const Jsmag = std::sqrt(jsx*jsx + jsy*jsy + jsz*jsz);

                    amrex::Real const bx = Interp(Bx_arr, Bx_stag, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const by = Interp(By_arr, By_stag, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const bz = Interp(Bz_arr, Bz_stag, d_stag, coarsen, i, j, k, 0);
                    amrex::Real const Bmag = std::sqrt(bx*bx + by*by + bz*bz);

                    amrex::Real const dv_d  = Vsd_arr(i,j,k) - Ved_arr(i,j,k);
                    amrex::Real const eta_s = eta_s_per(rhos_val, rho_val, Te_val,
                                                        Jmag, Jsmag, Bmag, t_new);
                    out_arr(i,j,k) += eta_s * rhos_val * rho_val / rhosum_val * dv_d;
                });
            }
        };

        accumulate_one_direction(overlay_x, Jx_stag, 0);
        accumulate_one_direction(overlay_y, Jy_stag, 1);
        accumulate_one_direction(overlay_z, Jz_stag, 2);
    }

    overlay_x.FillBoundary(warpx.Geom(lev).periodicity());
    overlay_y.FillBoundary(warpx.Geom(lev).periodicity());
    overlay_z.FillBoundary(warpx.Geom(lev).periodicity());
}


void HybridPICModel::FillElectronPressureMF (
    amrex::MultiFab& Pe_field,
    amrex::MultiFab const& rho_field
) const
{
    const auto n0_ref = m_n0_ref;
    const auto elec_temp = m_elec_temp;
    const auto gamma = m_gamma;

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(Pe_field, TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {
        // Extract field data for this grid/tile
        Array4<Real const> const& rho = rho_field.const_array(mfi);
        Array4<Real> const& Pe = Pe_field.array(mfi);

        // Extract tileboxes for which to loop
        const Box& tilebox  = mfi.tilebox();

        ParallelFor(tilebox, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            Pe(i, j, k) = ElectronPressure::get_pressure(
                n0_ref, elec_temp, gamma, rho(i, j, k)
            );
        });
    }
}

// =============================================================================
// Darwin longitudinal-field constraint
// =============================================================================

void HybridPICModel::ComputeDarwinELong (
    ablastr::fields::MultiLevelScalarField const& rho,
    amrex::Real const t)
{
    ABLASTR_PROFILE("HybridPICModel::ComputeDarwinELong()");

    using ablastr::fields::Direction;

    auto & warpx = WarpX::GetInstance();
    int const finest = warpx.finestLevel();

#if defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::ncomps == 1,
        "hybrid_pic_model.darwin supports only the m = 0 azimuthal mode in RZ");
#endif

    if (!m_darwin_bc_handler) {
        m_darwin_bc_handler = std::make_unique<PoissonBoundaryHandler>();
        m_darwin_bc_handler->DefinePhiBCs(warpx.Geom(0));
    }

    // Per-direction nodal-to-staggered gradient offsets: component d of the
    // gradient is nonzero only when dimension d is represented on the grid.
    // (In 2D/RZ the y/theta component vanishes for the in-plane/m=0 fields,
    // in 1D only the z component survives.)
#if defined(WARPX_DIM_3D)
    const amrex::IntVect grad_off[3] = {
        amrex::IntVect(1,0,0), amrex::IntVect(0,1,0), amrex::IntVect(0,0,1)};
    const int grad_dim[3] = {0, 1, 2};
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    const amrex::IntVect grad_off[3] = {
        amrex::IntVect(1,0), amrex::IntVect(0,0), amrex::IntVect(0,1)};
    const int grad_dim[3] = {0, -1, 1};
#elif defined(WARPX_DIM_1D_Z)
    const amrex::IntVect grad_off[3] = {
        amrex::IntVect(0), amrex::IntVect(0), amrex::IntVect(1)};
    const int grad_dim[3] = {-1, -1, 0};
#else
    const amrex::IntVect grad_off[3] = {
        amrex::IntVect(0), amrex::IntVect(0), amrex::IntVect(0)};
    const int grad_dim[3] = {-1, -1, -1};
#endif

    amrex::Real const rho_floor = PhysConst::q_e * m_n_floor;

    // Scratch (nodal, phi-shaped) for the constraint right-hand side div(S).
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> rhs_store(finest + 1);

    for (int lev = 0; lev <= finest; ++lev)
    {
        amrex::Geometry const & geom = warpx.Geom(lev);
        auto const dx_arr = geom.CellSizeArray();

        amrex::MultiFab const & Pe = *warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
        ablastr::fields::VectorField E_long =
            warpx.m_fields.get_alldirs("hybrid_E_long_fp", lev);

        // Step 1: the ambipolar source S = -grad(Pe)/(q_e n_e) on the E
        // staggering, assembled into the E_long fabs (overwritten with
        // grad(phi) below).
        for (int dir = 0; dir < 3; ++dir) {
            amrex::MultiFab & Sd = *E_long[dir];
            int const d = grad_dim[dir];
            if (d < 0) { Sd.setVal(0.0_rt); continue; }
            amrex::Real const inv_dx = 1.0_rt / dx_arr[d];
            amrex::IntVect const off = grad_off[dir];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(Sd, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                auto const & S_arr   = Sd.array(mfi);
                auto const & Pe_arr  = Pe.const_array(mfi);
                auto const & rho_arr = rho[lev]->const_array(mfi);
                amrex::ParallelFor(mfi.tilebox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
                    amrex::IntVect const ivp = iv + off;
                    amrex::Real const rho_edge = amrex::max(
                        0.5_rt*(rho_arr(iv) + rho_arr(ivp)), rho_floor);
                    S_arr(iv) = -(Pe_arr(ivp) - Pe_arr(iv)) * inv_dx / rho_edge;
                });
            }
            // Zero-extend past non-periodic physical boundaries (setBndry
            // zeroes all ghosts, FillBoundary then restores the interior and
            // periodic ones): the divergence below reads the ghost layer at
            // the walls, and the nodal Dirichlet phi solve constrains the
            // wall nodes anyway. Proper per-boundary-type E_L conditions are
            // a follow-up.
            Sd.setBndry(0.0_rt);
            Sd.FillBoundary(geom.periodicity());
        }

        // Step 2: nodal div(S) via the geometry-aware finite-difference
        // divergence.
        amrex::MultiFab const & phi_shape = *warpx.m_fields.get("hybrid_phi_darwin_fp", lev);
        rhs_store[lev] = std::make_unique<amrex::MultiFab>(
            phi_shape.boxArray(), phi_shape.DistributionMap(),
            phi_shape.nComp(), phi_shape.nGrowVect());
        warpx.get_pointer_fdtd_solver_fp(lev)->ComputeDivE(E_long, *rhs_store[lev]);
    }

    // Step 3: MLMG solve of laplacian(phi) = div(S) with the
    // finite-difference nodal operator (MLEBNodeFDLaplacian; the EB factory
    // and a Dirichlet body value are attached when embedded boundaries are
    // active), whose stencil is EXACTLY the composition of the divergence
    // and gradient used here. Operator consistency is essential: with a
    // different Laplacian stencil (e.g. the variational nodal operator the
    // electrostatic solver uses in Cartesian geometry, which coincides with
    // the FD composite only in 1D), the projection leaves a spurious
    // stencil-mismatch remainder in E_T = E - E_L at the amplitude of the
    // pressure-gradient noise, and the vector-potential update integrates
    // it into B at O(dt/dx). Boundary types follow the electrostatic
    // boundary-type mapping (m_darwin_bc_handler); Dirichlet values are 0.
    amrex::ignore_unused(t);
    for (int lev = 0; lev <= finest; ++lev)
    {
        amrex::LPInfo const info;
        std::unique_ptr<amrex::MLNodeLinOp> linop;
#if defined(WARPX_DIM_1D_Z)
        // The FD nodal operator does not support 1D; the tensor nodal
        // operator reduces to the same 3-point stencil there.
        linop = std::make_unique<amrex::MLNodeTensorLaplacian>(
            amrex::Vector<amrex::Geometry>{warpx.Geom(lev)},
            amrex::Vector<amrex::BoxArray>{warpx.boxArray(lev)},
            amrex::Vector<amrex::DistributionMapping>{warpx.DistributionMap(lev)},
            info);
#else
        auto linop_fd = std::make_unique<amrex::MLEBNodeFDLaplacian>();
#if defined(AMREX_USE_EB)
        if (EB::enabled()) {
            linop_fd->define(
                amrex::Vector<amrex::Geometry>{warpx.Geom(lev)},
                amrex::Vector<amrex::BoxArray>{warpx.boxArray(lev)},
                amrex::Vector<amrex::DistributionMapping>{warpx.DistributionMap(lev)},
                info,
                amrex::Vector<amrex::EBFArrayBoxFactory const*>{&warpx.fieldEBFactory(lev)});
            // Embedded conductors are equipotential: phi takes a Dirichlet
            // value on the body so E_L has no tangential component along
            // the surface and vanishes inside (the enclosed field is
            // frozen through the vector potential, held at the gauge zero
            // in covered cells).
            linop_fd->setEBDirichlet(0.0_rt);
        } else
#endif
        {
            linop_fd->define(
                amrex::Vector<amrex::Geometry>{warpx.Geom(lev)},
                amrex::Vector<amrex::BoxArray>{warpx.boxArray(lev)},
                amrex::Vector<amrex::DistributionMapping>{warpx.DistributionMap(lev)},
                info);
        }
#if defined(WARPX_DIM_RZ)
        linop_fd->setRZ(true);
        linop_fd->setSigma({0._rt, 1._rt});
#else
        linop_fd->setSigma({AMREX_D_DECL(1._rt, 1._rt, 1._rt)});
#endif
        linop = std::move(linop_fd);
#endif
        linop->setDomainBC(m_darwin_bc_handler->lobc, m_darwin_bc_handler->hibc);

        amrex::MultiFab & phi = *warpx.m_fields.get("hybrid_phi_darwin_fp", lev);
        amrex::MLMG mlmg(*linop);
        mlmg.setVerbose(m_darwin_poisson_verbosity);
        mlmg.setMaxIter(m_darwin_poisson_max_iters);
        mlmg.solve({&phi}, {rhs_store[lev].get()},
                   m_darwin_poisson_rtol, m_darwin_poisson_atol);
    }

    // Step 4: E_L = +grad(phi) on the E staggering (the longitudinal
    // projection of S: laplacian(phi) = div(S) => grad(phi) = S_L).
    for (int lev = 0; lev <= finest; ++lev)
    {
        amrex::Geometry const & geom = warpx.Geom(lev);
        auto const dx_arr = geom.CellSizeArray();
        amrex::MultiFab & phi = *warpx.m_fields.get("hybrid_phi_darwin_fp", lev);
        phi.setBndry(0.0_rt);
        phi.FillBoundary(geom.periodicity());
        ablastr::fields::VectorField E_long =
            warpx.m_fields.get_alldirs("hybrid_E_long_fp", lev);

        for (int dir = 0; dir < 3; ++dir) {
            amrex::MultiFab & Ed = *E_long[dir];
            int const d = grad_dim[dir];
            if (d < 0) { Ed.setVal(0.0_rt); continue; }
            amrex::Real const inv_dx = 1.0_rt / dx_arr[d];
            amrex::IntVect const off = grad_off[dir];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(Ed, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                auto const & E_arr   = Ed.array(mfi);
                auto const & phi_arr = phi.const_array(mfi);
                amrex::ParallelFor(mfi.tilebox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
                    E_arr(iv) = (phi_arr(iv + off) - phi_arr(iv)) * inv_dx;
                });
            }
            // As for S above: zero physical ghosts, since E_L is added into
            // the (ghost-including) push field.
            Ed.setBndry(0.0_rt);
            Ed.FillBoundary(geom.periodicity());
        }

        if (m_darwin_poisson_verbosity > 0) {
            amrex::Print() << "[darwin] lev " << lev
                << " max|E_L| = (" << E_long[0]->norminf()
                << ", " << E_long[1]->norminf()
                << ", " << E_long[2]->norminf()
                << ") max|phi| = " << phi.norminf() << "\n";
        }
    }
}

void HybridPICModel::ComputeVacuumARecovery (bool a_from_jacobian)
{
#if defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    // Blocked at parse time (see ReadParameters); the FD nodal operator has
    // no 1D form.
    WARPX_ABORT_WITH_MESSAGE(
        "hybrid_pic_model.darwin_vacuum_recovery is not supported in 1D "
        "geometries.");
#else
    ABLASTR_PROFILE("HybridPICModel::ComputeVacuumARecovery()");

    using ablastr::fields::Direction;
    using namespace ablastr::coarsen::sample;

    auto & warpx = WarpX::GetInstance();

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(warpx.finestLevel() == 0,
        "hybrid_pic_model.darwin_vacuum_recovery supports a single level only");
#if defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::ncomps == 1,
        "hybrid_pic_model.darwin_vacuum_recovery supports only the m = 0 "
        "azimuthal mode in RZ");
#endif

    constexpr int lev = 0;
    amrex::Geometry const & geom = warpx.Geom(lev);
    const bool trace = (std::getenv("WARPX_DEBUG_VACREC") != nullptr);

    // Mask policy on the step-entry charge density (rho^n, component 0 of
    // rho_fp): frozen across the step, so the mask cannot flicker between
    // nonlinear-solver iterations.
    amrex::Real const rho_floor = PhysConst::q_e * m_n_floor
        * m_darwin_vacuum_recovery_density_fraction;
    const int mask_mode = (m_darwin_vacuum_recovery_mask == "global") ? 2
        : ((m_darwin_vacuum_recovery_mask == "transition") ? 1 : 0);

    ablastr::fields::VectorField A =
        warpx.m_fields.get_alldirs("hybrid_A_fp", lev);
    ablastr::fields::VectorField B =
        warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev);
    ablastr::fields::VectorField Jvac =
        warpx.m_fields.get_alldirs("hybrid_J_vac_fp", lev);
    ablastr::fields::VectorField dA =
        warpx.m_fields.get_alldirs("hybrid_A_vac_dA_nodal", lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp, lev);

    // The field-implied current J_imp = curl(curl A)/mu0 on the E staggering.
    // Bfield_fp is curl scratch here (the caller rebuilds B = B_static +
    // curl A right after the recovery); zero-extension past non-periodic
    // walls mirrors the E_L source convention.
    // FD-Jacobian probes must not difference through the iterative solve
    // (its rtol-level noise is amplified by the 1/(theta dt) scaling of the
    // band field rows): they reuse the correction from the last
    // non-Jacobian evaluation, keeping the probed map smooth and affine.
    if (!a_from_jacobian) {
    warpx.get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(
        B, A, warpx.GetEBUpdateBFlag()[lev], lev);
    for (int dir = 0; dir < 3; ++dir) {
        B[dir]->setBndry(0.0_rt);
        B[dir]->FillBoundary(geom.periodicity());
    }
    warpx.get_pointer_fdtd_solver_fp(lev)->CalculateCurrentAmpere(
        Jvac, B, warpx.GetEBUpdateEFlag()[lev], lev);
    for (int dir = 0; dir < 3; ++dir) {
        Jvac[dir]->FillBoundary(geom.periodicity());
    }
    } // !a_from_jacobian

    amrex::GpuArray<int, 3> const coarsen_rr = {1, 1, 1};
    amrex::GpuArray<int, 3> const nodal = {1, 1, 1};
    const amrex::GpuArray<int, 3> A_stag[3] =
        {Ex_IndexType, Ey_IndexType, Ez_IndexType};

    // Domain BCs for the correction: homogeneous Dirichlet on non-periodic
    // faces (the evolved A already carries the boundary pin there, so the
    // correction vanishes), periodic where periodic, Neumann at the RZ axis
    // (required by the RZ FD nodal operator).
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lobc, hibc;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (geom.isPeriodic(d)) {
            lobc[d] = amrex::LinOpBCType::Periodic;
            hibc[d] = amrex::LinOpBCType::Periodic;
        } else {
            lobc[d] = amrex::LinOpBCType::Dirichlet;
            hibc[d] = amrex::LinOpBCType::Dirichlet;
        }
    }
#if defined(WARPX_DIM_RZ)
    lobc[0] = amrex::LinOpBCType::Neumann;
#endif

    // Nodal domain box and periodicity flags for the boundary-node guard in
    // the RHS assembly (Dirichlet rows carry no source).
    const amrex::Box domain_nd = amrex::convert(
        geom.Domain(), amrex::IntVect::TheNodeVector());
    amrex::GpuArray<int, 3> dlo{{0, 0, 0}};
    amrex::GpuArray<int, 3> dhi{{0, 0, 0}};
    amrex::GpuArray<int, 3> per{{1, 1, 1}};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        dlo[d] = domain_nd.smallEnd(d);
        dhi[d] = domain_nd.bigEnd(d);
        per[d] = geom.isPeriodic(d) ? 1 : 0;
    }

    // "flux": only the out-of-plane component (index 1: A_theta in RZ, A_y
    // in 2D) is recovered — it carries the boundary-driven flux, and its
    // nodal operator (with the RZ metric term) is discretely consistent
    // with the edge curl-curl source. The in-plane corrections are NOT
    // consistent near the RZ axis (O(1) 1/r stencil mismatch in the first
    // rows) and pump an axis-localized instability there; they stay on the
    // raw dynamics unless "all" is requested. In 3D there is no
    // distinguished component and "flux" recovers all three.
#if defined(WARPX_DIM_3D)
    const bool flux_only = false;
#else
    const bool flux_only = (m_darwin_vacuum_recovery_components == "flux");
#endif

    for (int dir = 0; dir < 3; ++dir) {
        if (flux_only && dir != 1) { continue; }
        amrex::MultiFab & dAd = *dA[dir];
        amrex::MultiFab const & Jd = *Jvac[dir];
        amrex::GpuArray<int, 3> const Jstag = A_stag[dir];

        if (a_from_jacobian) {
            // Frozen correction: nothing to apply if it is identically zero
            // (also preserves the exact-identity property bit-for-bit).
            if (dAd.norminf(0) == 0.0_rt) { continue; }
        } else {
        // Nodal right-hand side: +mu0 * J_imp, source-masked at the
        // (native-nodal) rho, zero on Dirichlet boundary nodes.
        amrex::MultiFab rhs(dAd.boxArray(), dAd.DistributionMap(),
                            dAd.nComp(), amrex::IntVect(0));
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const & rhs_arr = rhs.array(mfi);
            auto const & J_arr   = Jd.const_array(mfi);
            auto const & rho_arr = rho.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                const int idx[3] = {i, j, k};
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    if (!per[d] && (idx[d] <= dlo[d] || idx[d] >= dhi[d])) {
                        rhs_arr(i, j, k) = 0.0_rt;
                        return;
                    }
                }
                amrex::Real const rho_val = rho_arr(i, j, k, 0);
                const bool in_mask = (mask_mode == 2)
                    || (mask_mode == 0 && rho_val < rho_floor)
                    || (mask_mode == 1 && rho_val > 0.0_rt
                        && rho_val < rho_floor);
                rhs_arr(i, j, k) = in_mask
                    ? PhysConst::mu0
                        * Interp(J_arr, Jstag, nodal, coarsen_rr, i, j, k, 0)
                    : 0.0_rt;
            });
        }

        // Empty mask (or a fully-relaxed correction): the solution is
        // identically the incoming dA; skip the solve and, when dA is also
        // zero, skip the A update entirely so the recovery is exactly the
        // identity (bit-for-bit) in no-vacuum configurations.
        if (rhs.norminf(0) == 0.0_rt && dAd.norminf(0) == 0.0_rt) {
            if (trace) {
                amrex::Print() << "[vac-recover] dir " << dir
                    << " empty mask, skipped\n";
            }
            continue;
        }

        amrex::LPInfo const info;
        auto linop = std::make_unique<amrex::MLEBNodeFDLaplacian>();
#if defined(AMREX_USE_EB)
        if (EB::enabled()) {
            linop->define(
                amrex::Vector<amrex::Geometry>{geom},
                amrex::Vector<amrex::BoxArray>{warpx.boxArray(lev)},
                amrex::Vector<amrex::DistributionMapping>{warpx.DistributionMap(lev)},
                info,
                amrex::Vector<amrex::EBFArrayBoxFactory const*>{&warpx.fieldEBFactory(lev)});
            // dA = 0 inside embedded conductors: both the evolved and the
            // recovered A hold the gauge zero there.
            linop->setEBDirichlet(0.0_rt);
        } else
#endif
        {
            linop->define(
                amrex::Vector<amrex::Geometry>{geom},
                amrex::Vector<amrex::BoxArray>{warpx.boxArray(lev)},
                amrex::Vector<amrex::DistributionMapping>{warpx.DistributionMap(lev)},
                info);
        }
#if defined(WARPX_DIM_RZ)
        linop->setRZ(true);
        linop->setSigma({0._rt, 1._rt});
        if (dir < 2) {
            // The RZ vector Laplacian carries a -1/r^2 metric term for the
            // r and theta components (none for z).
            linop->setAlpha(1._rt);
        }
#else
        linop->setSigma({AMREX_D_DECL(1._rt, 1._rt, 1._rt)});
#endif
        linop->setDomainBC(lobc, hibc);

        // Previous correction as the initial guess (warm start); its
        // Dirichlet boundary nodes are zero by construction and stay so.
        dAd.setBndry(0.0_rt);
        dAd.FillBoundary(geom.periodicity());

        if (trace) {
            amrex::Print() << "[vac-recover] dir " << dir
                << " solving: |rhs|max = " << rhs.norminf(0)
                << " |dA_in|max = " << dAd.norminf(0) << "\n";
        }
        amrex::MLMG mlmg(*linop);
        mlmg.setVerbose(trace ? std::max(m_darwin_vacrec_verbosity, 2)
                              : m_darwin_vacrec_verbosity);
        mlmg.setMaxIter(m_darwin_vacrec_max_iters);
        mlmg.setThrowException(true);
        // The default coarsest-level BiCGStab fails on the RZ vector
        // Laplacian's -alpha/r^2 rows and its failures amplify through the
        // V-cycles; plain smoothing at the bottom is robust here (the
        // coarse level is tiny and every solve is warm-started).
        mlmg.setBottomSolver(amrex::BottomSolver::smoother);
        amrex::Real resid = 0.0_rt;
        bool solve_ok = true;
        std::string fail_msg;
        try {
            resid = mlmg.solve({&dAd}, {&rhs},
                               m_darwin_vacrec_rtol, m_darwin_vacrec_atol);
        } catch (std::exception const & e) {
            // A failed solve leaves dA mid-iteration: reset it so the
            // recovery degrades to the identity for this component (and
            // the next evaluation restarts from scratch) instead of
            // aborting the nonlinear iteration.
            dAd.setVal(0.0_rt);
            solve_ok = false;
            fail_msg = e.what();
        }
        if (trace) {
            amrex::IntVect const rhs_loc = rhs.maxIndex(0);
            amrex::IntVect const dA_loc = dAd.maxIndex(0);
            amrex::Print() << "[vac-recover] dir " << dir
                << " |rhs|max = " << rhs.norminf(0) << " @ " << rhs_loc
                << " |dA|max = " << dAd.norminf(0) << " @ " << dA_loc
                << " resid = " << resid
                << (solve_ok ? "" : (" (SOLVE FAILED: " + fail_msg + ")"))
                << "\n";
        }
        if (!solve_ok) { continue; }
        dAd.FillBoundary(geom.periodicity());
        } // !a_from_jacobian

        // A += dA at the component's edge points, replace-masked; edges
        // frozen by the embedded-boundary flags keep A = 0 (the recovered
        // and evolved fields agree there by construction). The caller
        // re-applies the A boundary values afterwards.
        amrex::MultiFab & Ad = *A[dir];
        amrex::GpuArray<int, 3> const Astag = A_stag[dir];
        const amrex::iMultiFab* eb_flag =
            (!warpx.GetEBUpdateEFlag().empty()
             && warpx.GetEBUpdateEFlag()[lev][dir] != nullptr)
            ? warpx.GetEBUpdateEFlag()[lev][dir].get() : nullptr;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Ad, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const & a_arr   = Ad.array(mfi);
            auto const & dA_arr  = dAd.const_array(mfi);
            auto const & rho_arr = rho.const_array(mfi);
            amrex::Array4<int const> eb;
            if (eb_flag) { eb = eb_flag->const_array(mfi); }
            const bool use_eb = (eb_flag != nullptr);
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (use_eb && eb(i, j, k) == 0) { return; }
                amrex::Real const rho_val =
                    Interp(rho_arr, nodal, Astag, coarsen_rr, i, j, k, 0);
                const bool in_mask = (mask_mode == 2)
                    || (mask_mode == 0 && rho_val < rho_floor)
                    || (mask_mode == 1 && rho_val > 0.0_rt
                        && rho_val < rho_floor);
                if (!in_mask) { return; }
                a_arr(i, j, k) +=
                    Interp(dA_arr, nodal, Astag, coarsen_rr, i, j, k, 0);
            });
        }
        Ad.FillBoundary(geom.periodicity());
    }
#endif
}

void HybridPICModel::ApplyVacuumFaradayE (amrex::Real a_dt_eff, bool a_add_E_long,
                                          bool a_from_jacobian, bool a_bdf2)
{
#if defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    amrex::ignore_unused(a_dt_eff, a_add_E_long, a_bdf2);
    WARPX_ABORT_WITH_MESSAGE(
        "hybrid_pic_model.darwin_vacuum_recovery is not supported in 1D "
        "geometries.");
#else
    ABLASTR_PROFILE("HybridPICModel::ApplyVacuumFaradayE()");

    using ablastr::fields::Direction;
    using namespace ablastr::coarsen::sample;

    auto & warpx = WarpX::GetInstance();
    constexpr int lev = 0;
    amrex::Geometry const & geom = warpx.Geom(lev);
    const bool trace = (std::getenv("WARPX_DEBUG_VACREC") != nullptr);
    if (trace) {
        amrex::Print() << "[vac-faraday] enter dt_eff = " << a_dt_eff
            << " add_EL = " << a_add_E_long << "\n";
    }

    amrex::Real const rho_floor = PhysConst::q_e * m_n_floor
        * m_darwin_vacuum_recovery_density_fraction;
    const int mask_mode = (m_darwin_vacuum_recovery_mask == "global") ? 2
        : ((m_darwin_vacuum_recovery_mask == "transition") ? 1 : 0);
#if defined(WARPX_DIM_3D)
    const bool flux_only = false;
#else
    const bool flux_only = (m_darwin_vacuum_recovery_components == "flux");
#endif
    amrex::Real const inv_dt = 1.0_rt / a_dt_eff;

    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp, lev);
    amrex::GpuArray<int, 3> const coarsen_rr = {1, 1, 1};
    amrex::GpuArray<int, 3> const nodal = {1, 1, 1};
    const amrex::GpuArray<int, 3> A_stag[3] =
        {Ex_IndexType, Ey_IndexType, Ez_IndexType};

    for (int dir = 0; dir < 3; ++dir) {
        if (flux_only && dir != 1) { continue; }
        amrex::MultiFab & E = *warpx.m_fields.get(
            FieldType::Efield_fp, Direction{dir}, lev);
        amrex::MultiFab const & A = *warpx.m_fields.get(
            "hybrid_A_fp", Direction{dir}, lev);
        amrex::MultiFab const & A_old = *warpx.m_fields.get(
            "hybrid_A_old_fp", Direction{dir}, lev);
        amrex::MultiFab const * EL = a_add_E_long
            ? warpx.m_fields.get("hybrid_E_long_fp", Direction{dir}, lev)
            : nullptr;
        amrex::MultiFab & Etgt = *warpx.m_fields.get(
            "hybrid_E_vac_target_fp", Direction{dir}, lev);
        amrex::MultiFab const & Anm1 = *warpx.m_fields.get(
            "hybrid_A_vac_nm1_fp", Direction{dir}, lev);
        amrex::GpuArray<int, 3> const Astag = A_stag[dir];
        const amrex::iMultiFab* eb_flag =
            (!warpx.GetEBUpdateEFlag().empty()
             && warpx.GetEBUpdateEFlag()[lev][dir] != nullptr)
            ? warpx.GetEBUpdateEFlag()[lev][dir].get() : nullptr;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(E, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const & e_arr    = E.array(mfi);
            auto const & tgt_arr  = Etgt.array(mfi);
            auto const & a_arr    = A.const_array(mfi);
            auto const & aold_arr = A_old.const_array(mfi);
            auto const & anm1_arr = Anm1.const_array(mfi);
            auto const & rho_arr  = rho.const_array(mfi);
            const bool from_jac = a_from_jacobian;
            const bool bdf2 = a_bdf2;
            amrex::Array4<amrex::Real const> el_arr;
            if (EL) { el_arr = EL->const_array(mfi); }
            const bool add_el = (EL != nullptr);
            amrex::Array4<int const> eb;
            if (eb_flag) { eb = eb_flag->const_array(mfi); }
            const bool use_eb = (eb_flag != nullptr);
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (use_eb && eb(i, j, k) == 0) { return; }
                amrex::Real const rho_val =
                    Interp(rho_arr, nodal, Astag, coarsen_rr, i, j, k, 0);
                const bool in_mask = (mask_mode == 2)
                    || (mask_mode == 0 && rho_val < rho_floor)
                    || (mask_mode == 1 && rho_val > 0.0_rt
                        && rho_val < rho_floor);
                if (!in_mask) { return; }
                if (from_jac) {
                    // Probe evaluations use the target stored by the last
                    // non-Jacobian evaluation: the band rows differentiate
                    // to the identity, and the outer Newton iteration lags
                    // the recovery (no differencing through the iterative
                    // solve).
                    e_arr(i, j, k) = tgt_arr(i, j, k)
                        + (add_el ? el_arr(i, j, k) : 0.0_rt);
                } else {
                    // theta-stage: the scheme's own difference quotient
                    // (algebraically the midpoint-centered full-step rate at
                    // theta = 1/2). End of step: BDF2, the second-order
                    // endpoint derivative of the recovered A chain,
                    //   E^{n+1} = -(3A^{n+1} - 4A^n + A^{n-1})/(2 dt).
                    amrex::Real const efar = bdf2
                        ? -(3.0_rt * a_arr(i, j, k)
                            - 4.0_rt * aold_arr(i, j, k)
                            + anm1_arr(i, j, k)) * (0.5_rt * inv_dt)
                        : -(a_arr(i, j, k) - aold_arr(i, j, k)) * inv_dt;
                    tgt_arr(i, j, k) = efar;
                    e_arr(i, j, k) = efar
                        + (add_el ? el_arr(i, j, k) : 0.0_rt);
                }
            });
        }
        E.FillBoundary(geom.periodicity());
        if (trace) {
            amrex::Print() << "[vac-faraday] dir " << dir
                << " |E|max = " << E.norminf(0) << "\n";
        }
    }
#endif
}


// =============================================================================
// QDSMC electron-energy-equation orchestration
// =============================================================================
//
// All four methods below are NO-OPs when m_solve_electron_energy_equation is false; they are
// invoked from HybridPICEvolveFields only when QDSMC is enabled. They operate
// on the level-`lev` MultiFabs of WarpX's MultiFabRegister and use the same
// Yee->nodal interpolation (`ablastr::coarsen::sample::Interp`) as the rest
// of the hybrid solver.

void HybridPICModel::QDSMCInitializeUe (int const lev) const
{
    auto & warpx = WarpX::GetInstance();
    // Explicit-path defaults: rho and J_i at n+1/2 from the "temp" fabs, the
    // plasma current at its current (single) state.
    ablastr::fields::VectorField const J_plasma =
        warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    QDSMCInitializeUe(lev,
        *warpx.m_fields.get(FieldType::hybrid_rho_fp_temp, lev),
        warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_temp, lev),
        J_plasma, J_plasma, 1.0_rt);
}

void HybridPICModel::QDSMCInitializeUe (int const lev,
    amrex::MultiFab const & rho_in,
    ablastr::fields::VectorField const & J_i,
    ablastr::fields::VectorField const & Jp_new,
    ablastr::fields::VectorField const & Jp_old,
    amrex::Real const jp_interp) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCInitializeUe()");

    using ablastr::fields::Direction;

    auto & warpx = WarpX::GetInstance();
    amrex::Geometry const & geom = warpx.Geom(lev);
    amrex::Periodicity const & period = geom.periodicity();

    // V_e and rho live at the nodal grid; the currents are Yee-staggered.
    amrex::MultiFab       & Vex = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{0}, lev);
    amrex::MultiFab       & Vey = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{1}, lev);
    amrex::MultiFab       & Vez = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{2}, lev);

    amrex::MultiFab const & rho_temp = rho_in;

    amrex::Real const rho_floor = PhysConst::q_e * m_n_floor;

    amrex::GpuArray<int, 3> const & Jx_stag = Jx_IndexType;
    amrex::GpuArray<int, 3> const & Jy_stag = Jy_IndexType;
    amrex::GpuArray<int, 3> const & Jz_stag = Jz_IndexType;
    amrex::GpuArray<int, 3> const nodal     = {1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen   = {1, 1, 1};

    Vex.setVal(0.0_rt);
    Vey.setVal(0.0_rt);
    Vez.setVal(0.0_rt);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Vex, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real const> const & rho_arr = rho_temp.const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jpx     = Jp_new[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jpy     = Jp_new[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jpz     = Jp_new[2]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jpx0    = Jp_old[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jpy0    = Jp_old[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jpz0    = Jp_old[2]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jix     = J_i[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jiy     = J_i[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jiz     = J_i[2]->const_array(mfi);
        amrex::Array4<amrex::Real>       const & Vex_arr = Vex.array(mfi);
        amrex::Array4<amrex::Real>       const & Vey_arr = Vey.array(mfi);
        amrex::Array4<amrex::Real>       const & Vez_arr = Vez.array(mfi);

        amrex::Box const & tbox = mfi.tilebox();

        amrex::Real const c_interp = jp_interp;

        amrex::ParallelFor(tbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (rho_arr(i,j,k) <= rho_floor) { return; }

            amrex::Real const rho_val = rho_arr(i,j,k);

            // Plasma current interpolated in time between the saved (old) and
            // current (new) states; identical fields make this a no-op.
            auto const jxn = ablastr::coarsen::sample::Interp(Jpx, Jx_stag, nodal, coarsen, i, j, k, 0);
            auto const jyn = ablastr::coarsen::sample::Interp(Jpy, Jy_stag, nodal, coarsen, i, j, k, 0);
            auto const jzn = ablastr::coarsen::sample::Interp(Jpz, Jz_stag, nodal, coarsen, i, j, k, 0);
            auto const jxo = ablastr::coarsen::sample::Interp(Jpx0, Jx_stag, nodal, coarsen, i, j, k, 0);
            auto const jyo = ablastr::coarsen::sample::Interp(Jpy0, Jy_stag, nodal, coarsen, i, j, k, 0);
            auto const jzo = ablastr::coarsen::sample::Interp(Jpz0, Jz_stag, nodal, coarsen, i, j, k, 0);
            amrex::Real const jx = jxo + c_interp * (jxn - jxo);
            amrex::Real const jy = jyo + c_interp * (jyn - jyo);
            amrex::Real const jz = jzo + c_interp * (jzn - jzo);
            auto const jix = ablastr::coarsen::sample::Interp(Jix, Jx_stag, nodal, coarsen, i, j, k, 0);
            auto const jiy = ablastr::coarsen::sample::Interp(Jiy, Jy_stag, nodal, coarsen, i, j, k, 0);
            auto const jiz = ablastr::coarsen::sample::Interp(Jiz, Jz_stag, nodal, coarsen, i, j, k, 0);

            // V_e = -(J_plasma - J_i) / (q_e * n_e) = -(J_plasma - J_i) / rho_val
            Vex_arr(i,j,k) = -(jx - jix) / rho_val;
            Vey_arr(i,j,k) = -(jy - jiy) / rho_val;
            Vez_arr(i,j,k) = -(jz - jiz) / rho_val;
        });
    }

    Vex.FillBoundary(Vex.nGrowVect(), period);
    Vey.FillBoundary(Vey.nGrowVect(), period);
    Vez.FillBoundary(Vez.nGrowVect(), period);
}


void HybridPICModel::QDSMCInitializeKe (int const lev) const
{
    auto & warpx = WarpX::GetInstance();
    QDSMCInitializeKe(lev, *warpx.m_fields.get(FieldType::hybrid_rho_fp_temp, lev));
}

void HybridPICModel::QDSMCInitializeKe (int const lev, amrex::MultiFab const & rho_in) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCInitializeKe()");

    auto & warpx = WarpX::GetInstance();
    amrex::Periodicity const & period = warpx.Geom(lev).periodicity();

    amrex::MultiFab       & Ke  = *warpx.m_fields.get(FieldType::hybrid_entropy_fp,                lev);
    amrex::MultiFab const & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp,   lev);
    amrex::MultiFab const & rho = rho_in;

    Ke.setVal(0.0_rt);

    auto const gamma     = m_gamma;
    auto const rho_floor = PhysConst::q_e * m_n_floor;
    // Conversion factor used by helion to keep K_e numerically O(1): T_e in K
    // is multiplied by (k_B / q_e) so K_e ends up scaled in eV-equivalent.
    auto const kb_over_qe = PhysConst::kb / PhysConst::q_e;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Ke, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real>       const & Ke_arr  = Ke.array(mfi);
        amrex::Array4<amrex::Real const> const & Te_arr  = Te.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);

        amrex::Box const tbox = amrex::convert(mfi.tilebox(), Ke.ixType().toIntVect());
        amrex::Box       box  = tbox;
        box.grow(Ke.nGrowVect());

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (rho_arr(i,j,k) <= rho_floor) { return; }
            amrex::Real const ne = rho_arr(i,j,k) / PhysConst::q_e;
            Ke_arr(i,j,k) = Te_arr(i,j,k) * std::pow(ne, 1.0_rt - gamma) * kb_over_qe;
        });
    }

    Ke.FillBoundary(Ke.nGrowVect(), period);
}


namespace {
    // Total field variance of Te about its own mean: for a Gaussian bump
    // of width sigma on a uniform background, Var ~ 1/sigma^2(t), so
    // 1/Var grows linearly at the transport rate -- an estimator-free
    // in-step probe.
    void PrintTeVariance (char const* tag)
    {
        if (std::getenv("WARPX_COND_VARTRACE") == nullptr) { return; }
        auto & warpx = WarpX::GetInstance();
        amrex::MultiFab const & Te = *warpx.m_fields.get(
            warpx::fields::FieldType::hybrid_electron_temperature_fp, 0);
        auto const npts = static_cast<amrex::Real>(Te.boxArray().numPts());
        amrex::Real const mean = Te.sum(0) / npts;
        amrex::Real const m2 = amrex::MultiFab::Dot(Te, 0, Te, 0, 1, 0) / npts;
        amrex::Print() << "[var-trace] " << tag << " var "
            << m2 - mean*mean << " mean " << mean << "\n";
    }
}

void HybridPICModel::QdsmcConductionPass (amrex::Real const h,
                                          warpx::fields::FieldType const rho_type,
                                          int const rho_comp) const
{
    ABLASTR_PROFILE("HybridPICModel::QdsmcConductionPass()");

    auto & warpx = WarpX::GetInstance();
    amrex::Real const n_floor_rec = m_qdsmc_n_floor;
    auto const kappa = m_qdsmc_kappa;
    // Recovery variant: the conservative difference is the default; the
    // ratio-delta + conservation projection (noise-cancelling,
    // ledger-projected) is selectable for the composition study.
    const bool use_ratio = (std::getenv("WARPX_COND_RATIO_PROJ") != nullptr);

    amrex::Real const h_sub = h / m_qdsmc_conduction_substeps;

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        amrex::Geometry const & geom = warpx.Geom(lev);
        auto const dxi = geom.InvCellSizeArray();

        amrex::MultiFab & Te = *warpx.m_fields.get(
            FieldType::hybrid_electron_temperature_fp, lev);
        amrex::MultiFab const & rho = *warpx.m_fields.get(rho_type, lev);

        // Work fields on the nodal T_e layout. The guard width must match
        // the registered fields' deposition guards: the shape-1 scatter
        // writes one node beyond the grown tile box, past a 1-ghost
        // allocation.
        amrex::IntVect const ng = Te.nGrowVect();
        amrex::MultiFab U (Te.boxArray(), Te.DistributionMap(), 1, ng);
        amrex::MultiFab N (Te.boxArray(), Te.DistributionMap(), 1, ng);
        amrex::MultiFab U0 (Te.boxArray(), Te.DistributionMap(), 1, ng);
        amrex::MultiFab N0 (Te.boxArray(), Te.DistributionMap(), 1, ng);
        amrex::MultiFab Nk (Te.boxArray(), Te.DistributionMap(), 1, ng);
        amrex::MultiFab dT (Te.boxArray(), Te.DistributionMap(), 1, ng);
        amrex::MultiFab D (Te.boxArray(), Te.DistributionMap(), 1, ng);
        amrex::MultiFab gDx (Te.boxArray(), Te.DistributionMap(), 1, ng);
        amrex::MultiFab gDy (Te.boxArray(), Te.DistributionMap(), 1, ng);
        amrex::MultiFab gDz (Te.boxArray(), Te.DistributionMap(), 1, ng);

        // Parallel mode: nodal unit field direction and the six-component
        // symmetric kernel tensor W = D b b (stored as xx,xy,xz,yy,yz,zz)
        // whose divergence is the Ito drift.
        const bool parallel_mode = (m_qdsmc_conduction == "parallel");
        amrex::MultiFab bhx, bhy, bhz, W;
        if (parallel_mode) {
            bhx.define(Te.boxArray(), Te.DistributionMap(), 1, ng);
            bhy.define(Te.boxArray(), Te.DistributionMap(), 1, ng);
            bhz.define(Te.boxArray(), Te.DistributionMap(), 1, ng);
            W.define(Te.boxArray(), Te.DistributionMap(), 6, ng);
        }

        if (parallel_mode) {
            // Nodal unit b from the staggered magnetic field; cells with
            // |B| below the floor get a zero vector (deposit in place).
            using ablastr::fields::Direction;
            amrex::MultiFab const & Bx = *warpx.m_fields.get(
                FieldType::Bfield_fp, Direction{0}, lev);
            amrex::MultiFab const & By = *warpx.m_fields.get(
                FieldType::Bfield_fp, Direction{1}, lev);
            amrex::MultiFab const & Bz = *warpx.m_fields.get(
                FieldType::Bfield_fp, Direction{2}, lev);
            amrex::GpuArray<int, 3> const bx_stag{{
                Bx.ixType().nodeCentered(0), Bx.ixType().nodeCentered(1),
                AMREX_SPACEDIM == 3 ? Bx.ixType().nodeCentered(2) : 0}};
            amrex::ignore_unused(bx_stag);
            constexpr amrex::Real b_floor = 1.0e-12_rt;
            for (amrex::MFIter mfi(Te, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Box const box = mfi.tilebox();
                auto const bxs = Bx.const_array(mfi);
                auto const bys = By.const_array(mfi);
                auto const bzs = Bz.const_array(mfi);
                auto const hx = bhx.array(mfi);
                auto const hy = bhy.array(mfi);
                auto const hz = bhz.array(mfi);
                auto to_stag = [] (amrex::IntVect const& iv) {
                    amrex::GpuArray<int, 3> a{{0, 0, 0}};
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) { a[d] = iv[d]; }
                    return a;
                };
                amrex::GpuArray<int,3> const sx = to_stag(Bx.ixType().toIntVect());
                amrex::GpuArray<int,3> const sy = to_stag(By.ixType().toIntVect());
                amrex::GpuArray<int,3> const sz = to_stag(Bz.ixType().toIntVect());
                amrex::GpuArray<int,3> const sn = to_stag(Te.ixType().toIntVect());
                amrex::GpuArray<int,3> const cr{{1,1,1}};
                amrex::ParallelFor(box,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    using ablastr::coarsen::sample::Interp;
                    amrex::Real const bx_n = Interp(bxs, sx, sn, cr, i, j, k, 0);
                    amrex::Real const by_n = Interp(bys, sy, sn, cr, i, j, k, 0);
                    amrex::Real const bz_n = Interp(bzs, sz, sn, cr, i, j, k, 0);
                    amrex::Real const bm = std::sqrt(
                        bx_n*bx_n + by_n*by_n + bz_n*bz_n);
                    if (bm > b_floor) {
                        hx(i,j,k) = bx_n / bm;
                        hy(i,j,k) = by_n / bm;
                        hz(i,j,k) = bz_n / bm;
                    } else {
                        hx(i,j,k) = 0.0_rt;
                        hy(i,j,k) = 0.0_rt;
                        hz(i,j,k) = 0.0_rt;
                    }
                });
            }
            bhx.FillBoundary(geom.periodicity());
            bhy.FillBoundary(geom.periodicity());
            bhz.FillBoundary(geom.periodicity());
        }

        for (int sub = 0; sub < m_qdsmc_conduction_substeps; ++sub)
        {
            // Nodal energy density and diffusivity from the pass state.
            for (amrex::MFIter mfi(Te, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Box const box = mfi.growntilebox(ng);
                auto const te_arr  = Te.const_array(mfi);
                auto const rho_arr = rho.const_array(mfi, rho_comp);
                auto const u_arr = U.array(mfi);
                auto const d_arr = D.array(mfi);
                auto const nn_arr = N.array(mfi);
                amrex::ParallelFor(box,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    amrex::Real const n = amrex::max(
                        rho_arr(i,j,k) / PhysConst::q_e, n_floor_rec);
                    amrex::Real const T = amrex::max(te_arr(i,j,k), 0.0_rt);
                    u_arr(i,j,k) = 1.5_rt * n * PhysConst::kb * T;
                    nn_arr(i,j,k) = n;
                    d_arr(i,j,k) = (2.0_rt/3.0_rt) * kappa(T, n)
                                   / (n * PhysConst::kb);
                });
            }
            U.FillBoundary(geom.periodicity());
            N.FillBoundary(geom.periodicity());
            D.FillBoundary(geom.periodicity());

            if (m_qdsmc_conduction_flux_limiter > 0.0_rt) {
                // Free-streaming limiter: harmonic blend of the Spitzer
                // flux against q_fs = alpha n k_B T v_th,e, applied as a
                // per-node diffusivity reduction
                // D_eff = D / (1 + (3/2) D |grad T| / (alpha T v_th,e)).
                amrex::Real const alpha = m_qdsmc_conduction_flux_limiter;
                for (amrex::MFIter mfi(D, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    amrex::Box const box = mfi.tilebox();
                    auto const te_arr = Te.const_array(mfi);
                    auto const d_arr = D.array(mfi);
                    amrex::ParallelFor(box,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        amrex::Real gx2 = 0.0_rt;
#if defined(WARPX_DIM_3D)
                        amrex::Real const tx = 0.5_rt*dxi[0]*(te_arr(i+1,j,k)-te_arr(i-1,j,k));
                        amrex::Real const ty = 0.5_rt*dxi[1]*(te_arr(i,j+1,k)-te_arr(i,j-1,k));
                        amrex::Real const tz = 0.5_rt*dxi[2]*(te_arr(i,j,k+1)-te_arr(i,j,k-1));
                        gx2 = tx*tx + ty*ty + tz*tz;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                        amrex::Real const tx = 0.5_rt*dxi[0]*(te_arr(i+1,j,k)-te_arr(i-1,j,k));
                        amrex::Real const tz = 0.5_rt*dxi[1]*(te_arr(i,j+1,k)-te_arr(i,j-1,k));
                        gx2 = tx*tx + tz*tz;
#else
                        amrex::Real const tz = 0.5_rt*dxi[0]*(te_arr(i+1,j,k)-te_arr(i-1,j,k));
                        gx2 = tz*tz;
#endif
                        amrex::Real const T = amrex::max(te_arr(i,j,k), 1.0_rt);
                        amrex::Real const vth = std::sqrt(
                            PhysConst::kb * T / PhysConst::m_e);
                        d_arr(i,j,k) = d_arr(i,j,k)
                            / (1.0_rt + 1.5_rt * d_arr(i,j,k)
                               * std::sqrt(gx2) / (alpha * T * vth));
                    });
                }
                D.FillBoundary(geom.periodicity());
            }

            if (parallel_mode) {
                // W = D b b, then drift_i = sum_j d_j W_ij by central
                // differences.
                for (amrex::MFIter mfi(D, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    amrex::Box const box = mfi.growntilebox(ng);
                    auto const d_arr = D.const_array(mfi);
                    auto const hx = bhx.const_array(mfi);
                    auto const hy = bhy.const_array(mfi);
                    auto const hz = bhz.const_array(mfi);
                    auto const w = W.array(mfi);
                    amrex::ParallelFor(box,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        amrex::Real const d = d_arr(i,j,k);
                        w(i,j,k,0) = d * hx(i,j,k) * hx(i,j,k);
                        w(i,j,k,1) = d * hx(i,j,k) * hy(i,j,k);
                        w(i,j,k,2) = d * hx(i,j,k) * hz(i,j,k);
                        w(i,j,k,3) = d * hy(i,j,k) * hy(i,j,k);
                        w(i,j,k,4) = d * hy(i,j,k) * hz(i,j,k);
                        w(i,j,k,5) = d * hz(i,j,k) * hz(i,j,k);
                    });
                }
                for (amrex::MFIter mfi(D, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    amrex::Box const box = mfi.tilebox();
                    auto const w = W.const_array(mfi);
                    auto const gx = gDx.array(mfi);
                    auto const gy = gDy.array(mfi);
                    auto const gz = gDz.array(mfi);
                    amrex::ParallelFor(box,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
#if defined(WARPX_DIM_3D)
                        gx(i,j,k) = 0.5_rt*dxi[0]*(w(i+1,j,k,0)-w(i-1,j,k,0))
                                  + 0.5_rt*dxi[1]*(w(i,j+1,k,1)-w(i,j-1,k,1))
                                  + 0.5_rt*dxi[2]*(w(i,j,k+1,2)-w(i,j,k-1,2));
                        gy(i,j,k) = 0.5_rt*dxi[0]*(w(i+1,j,k,1)-w(i-1,j,k,1))
                                  + 0.5_rt*dxi[1]*(w(i,j+1,k,3)-w(i,j-1,k,3))
                                  + 0.5_rt*dxi[2]*(w(i,j,k+1,4)-w(i,j,k-1,4));
                        gz(i,j,k) = 0.5_rt*dxi[0]*(w(i+1,j,k,2)-w(i-1,j,k,2))
                                  + 0.5_rt*dxi[1]*(w(i,j+1,k,4)-w(i,j-1,k,4))
                                  + 0.5_rt*dxi[2]*(w(i,j,k+1,5)-w(i,j,k-1,5));
#elif defined(WARPX_DIM_XZ)
                        // In-plane derivatives only: d_x and d_z (index j).
                        gx(i,j,k) = 0.5_rt*dxi[0]*(w(i+1,j,k,0)-w(i-1,j,k,0))
                                  + 0.5_rt*dxi[1]*(w(i,j+1,k,2)-w(i,j-1,k,2));
                        gy(i,j,k) = 0.5_rt*dxi[0]*(w(i+1,j,k,1)-w(i-1,j,k,1))
                                  + 0.5_rt*dxi[1]*(w(i,j+1,k,4)-w(i,j-1,k,4));
                        gz(i,j,k) = 0.5_rt*dxi[0]*(w(i+1,j,k,2)-w(i-1,j,k,2))
                                  + 0.5_rt*dxi[1]*(w(i,j+1,k,5)-w(i,j-1,k,5));
#else
                        gx(i,j,k) = 0.5_rt*dxi[0]*(w(i,j+1,k,2)-w(i,j-1,k,2));
                        amrex::ignore_unused(gx);
                        gy(i,j,k) = 0.0_rt;
                        gz(i,j,k) = 0.5_rt*dxi[0]*(w(i,j+1,k,5)-w(i,j-1,k,5));
#endif
                    });
                }
            } else {
            // grad(D) by nodal central differences (Ito drift). Valid nodes
            // only; the one-sided domain edge is handled by the ghost fill
            // (periodic wrap or the last interior value via FillBoundary +
            // the reflection treatment of the kick itself).
            for (amrex::MFIter mfi(D, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Box const box = mfi.tilebox();
                auto const d_arr = D.const_array(mfi);
                auto const gx = gDx.array(mfi);
                auto const gy = gDy.array(mfi);
                auto const gz = gDz.array(mfi);
                amrex::ParallelFor(box,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
#if defined(WARPX_DIM_3D)
                    gx(i,j,k) = 0.5_rt*dxi[0]*(d_arr(i+1,j,k) - d_arr(i-1,j,k));
                    gy(i,j,k) = 0.5_rt*dxi[1]*(d_arr(i,j+1,k) - d_arr(i,j-1,k));
                    gz(i,j,k) = 0.5_rt*dxi[2]*(d_arr(i,j,k+1) - d_arr(i,j,k-1));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    gx(i,j,k) = 0.5_rt*dxi[0]*(d_arr(i+1,j,k) - d_arr(i-1,j,k));
                    gy(i,j,k) = 0.0_rt;
                    gz(i,j,k) = 0.5_rt*dxi[1]*(d_arr(i,j+1,k) - d_arr(i,j-1,k));
#else
                    // 1D: the single index runs along z.
                    gx(i,j,k) = 0.0_rt;
                    gy(i,j,k) = 0.0_rt;
                    gz(i,j,k) = 0.5_rt*dxi[0]*(d_arr(i+1,j,k) - d_arr(i-1,j,k));
#endif
                });
            }
            }
            gDx.FillBoundary(geom.periodicity());
            gDy.FillBoundary(geom.periodicity());
            gDz.FillBoundary(geom.periodicity());

            // Delta form: the kicked and unkicked node sets carry
            // IDENTICAL contents, so the difference of their raw energy
            // deposits is exactly conservative (both sum to the same
            // total) and shares every remap, seam and sampling factor --
            // isolating the pure diffusive transport (without it the
            // gather-deposit roundtrip's own smoothing dominates for
            // kernels narrower than a cell; recovering RATIOS instead
            // breaks conservation at strong temperature contrast).
            const amrex::MultiFab* bpx = parallel_mode ? &bhx : nullptr;
            const amrex::MultiFab* bpy = parallel_mode ? &bhy : nullptr;
            const amrex::MultiFab* bpz = parallel_mode ? &bhz : nullptr;
            m_qdsmc_cond_pc->SpawnConductionNodes(lev, h_sub, U, N, D,
                                                  gDx, gDy, gDz, false,
                                                  bpx, bpy, bpz);
            m_qdsmc_cond_pc->DepositConductionEnergy(lev, U0);
            m_qdsmc_cond_pc->DepositConductionCount(lev, N0);
            m_qdsmc_cond_pc->SpawnConductionNodes(lev, h_sub, U, N, D,
                                                  gDx, gDy, gDz, true,
                                                  bpx, bpy, bpz);
            m_qdsmc_cond_pc->DepositConductionEnergy(lev, U);
            m_qdsmc_cond_pc->DepositConductionCount(lev, Nk);

            if (std::getenv("WARPX_COND_TRACE") != nullptr) {
                amrex::Real const dmax = D.max(0);
                amrex::Real const nbar = rho.sum(rho_comp)
                    / (PhysConst::q_e * rho.boxArray().numPts());
                amrex::Print() << "[cond-trace] sub " << sub
                    << " Dmax " << dmax
                    << " sig/dx " << std::sqrt(2.0*dmax*h_sub)
                        * geom.InvCellSize(0)
                    << " nbar " << nbar
                    << " dU " << (U.sum(0) - U0.sum(0))
                    << " Te [" << Te.min(0) << "," << Te.max(0) << "]\n";
            }

            for (amrex::MFIter mfi(Te, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Box const box = mfi.tilebox();
                auto const u_arr   = U.const_array(mfi);
                auto const rho_arr = rho.const_array(mfi, rho_comp);
                auto const te_arr  = Te.array(mfi);
                auto const u0_arr = U0.const_array(mfi);
                auto const n0_arr = N0.const_array(mfi);
                auto const nk_arr = Nk.const_array(mfi);
                auto const dt_arr = dT.array(mfi);
                amrex::ParallelFor(box,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    // Ratio-delta recovery: the difference of the kicked
                    // and unkicked energy-per-count estimates. Each ratio
                    // carries the SAME gathered density realization and
                    // deposit smoothing in numerator and denominator, so
                    // the PIC density noise cancels (a raw energy-deposit
                    // difference moves the noise-modulated content and
                    // injects density-noise-squared temperature noise per
                    // pass). Nodes whose deposited count is far below the
                    // local density carry no reliable increment (boundary
                    // and axis nodes in the radial geometries): zero.
                    amrex::Real const count_floor = 0.1_rt * amrex::max(
                        rho_arr(i,j,k) / PhysConst::q_e, n_floor_rec);
                    if (n0_arr(i,j,k) > count_floor
                        && nk_arr(i,j,k) > count_floor) {
                        if (use_ratio) {
                            dt_arr(i,j,k) = (2.0_rt/3.0_rt) / PhysConst::kb
                                * (u_arr(i,j,k) / nk_arr(i,j,k)
                                   - u0_arr(i,j,k) / n0_arr(i,j,k));
                        } else {
                            // conservative difference (default): exactly
                            // conservative; carries the density noise in
                            // the increment (see the ratio variant).
                            dt_arr(i,j,k) = (2.0_rt/3.0_rt) / PhysConst::kb
                                * (u_arr(i,j,k) - u0_arr(i,j,k))
                                / n0_arr(i,j,k);
                        }
                    } else {
                        dt_arr(i,j,k) = 0.0_rt;
                    }
                });
            }

            // Conservation projection (ratio variant only): the ratio
            // estimates are noise-clean but do not telescope, and strong
            // contrasts grow energy without a ledger constraint. Rescale
            // the cooling part of the increment so the density-weighted
            // sum vanishes exactly.
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
            amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
            for (amrex::MFIter mfi(Te, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Box const box = mfi.tilebox();
                auto const dt_arr  = dT.const_array(mfi);
                auto const rho_arr = rho.const_array(mfi, rho_comp);
                reduce_op.eval(box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    -> amrex::GpuTuple<amrex::Real, amrex::Real>
                {
                    amrex::Real const n = amrex::max(
                        rho_arr(i,j,k) / PhysConst::q_e, n_floor_rec);
                    amrex::Real const v = n * dt_arr(i,j,k);
                    return {amrex::max(v, 0.0_rt), amrex::min(v, 0.0_rt)};
                });
            }
            auto rtuple = reduce_data.value(reduce_op);
            amrex::Real pos = amrex::get<0>(rtuple);
            amrex::Real neg = amrex::get<1>(rtuple);
            amrex::ParallelDescriptor::ReduceRealSum(pos);
            amrex::ParallelDescriptor::ReduceRealSum(neg);
            amrex::Real const alpha =
                (use_ratio && neg < 0.0_rt) ? (pos / (-neg)) : 1.0_rt;

            for (amrex::MFIter mfi(Te, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Box const box = mfi.tilebox();
                auto const dt_arr = dT.const_array(mfi);
                auto const te_arr = Te.array(mfi);
                amrex::ParallelFor(box,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    amrex::Real const d = dt_arr(i,j,k);
                    amrex::Real const dc = (d > 0.0_rt) ? d : alpha * d;
                    te_arr(i,j,k) = amrex::max(te_arr(i,j,k) + dc, 0.0_rt);
                });
            }
            Te.FillBoundary(geom.periodicity());
        }
    }
}

void HybridPICModel::QdsmcConductionFusedIncrement (amrex::Real const dt,
                                                    amrex::MultiFab & Karr,
                                                    warpx::fields::FieldType const rho_type,
                                                    int const rho_comp) const
{
    ABLASTR_PROFILE("HybridPICModel::QdsmcConductionFusedIncrement()");

    auto & warpx = WarpX::GetInstance();
    int const lev = 0;
    amrex::Real const n_floor_rec = m_qdsmc_n_floor;
    auto const kappa = m_qdsmc_kappa;
    // The stage stores K in eV-equivalent units: T[K] * (k_B / q_e).
    auto const kb_over_qe = PhysConst::kb / PhysConst::q_e;

    amrex::Geometry const & geom = warpx.Geom(lev);
    auto const dxi = geom.InvCellSizeArray();

    amrex::MultiFab & Te = *warpx.m_fields.get(
        FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(rho_type, lev);

    amrex::IntVect const ng = Te.nGrowVect();
    amrex::MultiFab U (Te.boxArray(), Te.DistributionMap(), 1, ng);
    amrex::MultiFab N (Te.boxArray(), Te.DistributionMap(), 1, ng);
    amrex::MultiFab U0 (Te.boxArray(), Te.DistributionMap(), 1, ng);
    amrex::MultiFab D (Te.boxArray(), Te.DistributionMap(), 1, ng);
    amrex::MultiFab gDx (Te.boxArray(), Te.DistributionMap(), 1, ng);
    amrex::MultiFab gDy (Te.boxArray(), Te.DistributionMap(), 1, ng);
    amrex::MultiFab gDz (Te.boxArray(), Te.DistributionMap(), 1, ng);

    amrex::Real const h_sub = dt / m_qdsmc_conduction_substeps;
    for (int sub = 0; sub < m_qdsmc_conduction_substeps; ++sub)
    {
        for (amrex::MFIter mfi(Te, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box const box = mfi.growntilebox(ng);
            auto const te_arr  = Te.const_array(mfi);
            auto const rho_arr = rho.const_array(mfi, rho_comp);
            auto const u_arr = U.array(mfi);
            auto const nn_arr = N.array(mfi);
            auto const d_arr = D.array(mfi);
            amrex::ParallelFor(box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real const n = amrex::max(
                    rho_arr(i,j,k) / PhysConst::q_e, n_floor_rec);
                amrex::Real const T = amrex::max(te_arr(i,j,k), 0.0_rt);
                // K*N-unit energy density: T[eV-equivalent] * n.
                u_arr(i,j,k) = kb_over_qe * T * n;
                nn_arr(i,j,k) = n;
                d_arr(i,j,k) = (2.0_rt/3.0_rt) * kappa(T, n)
                               / (n * PhysConst::kb);
            });
        }
        U.FillBoundary(geom.periodicity());
        N.FillBoundary(geom.periodicity());
        D.FillBoundary(geom.periodicity());

        for (amrex::MFIter mfi(D, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box const box = mfi.tilebox();
            auto const d_arr = D.const_array(mfi);
            auto const gx = gDx.array(mfi);
            auto const gy = gDy.array(mfi);
            auto const gz = gDz.array(mfi);
            amrex::ParallelFor(box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
#if defined(WARPX_DIM_3D)
                gx(i,j,k) = 0.5_rt*dxi[0]*(d_arr(i+1,j,k) - d_arr(i-1,j,k));
                gy(i,j,k) = 0.5_rt*dxi[1]*(d_arr(i,j+1,k) - d_arr(i,j-1,k));
                gz(i,j,k) = 0.5_rt*dxi[2]*(d_arr(i,j,k+1) - d_arr(i,j,k-1));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                gx(i,j,k) = 0.5_rt*dxi[0]*(d_arr(i+1,j,k) - d_arr(i-1,j,k));
                gy(i,j,k) = 0.0_rt;
                gz(i,j,k) = 0.5_rt*dxi[1]*(d_arr(i,j+1,k) - d_arr(i,j-1,k));
#else
                gx(i,j,k) = 0.0_rt;
                gy(i,j,k) = 0.0_rt;
                gz(i,j,k) = 0.5_rt*dxi[0]*(d_arr(i+1,j,k) - d_arr(i-1,j,k));
#endif
            });
        }
        gDx.FillBoundary(geom.periodicity());
        gDy.FillBoundary(geom.periodicity());
        gDz.FillBoundary(geom.periodicity());

        m_qdsmc_cond_pc->SpawnConductionNodes(lev, h_sub, U, N, D,
                                              gDx, gDy, gDz, false);
        m_qdsmc_cond_pc->DepositConductionEnergy(lev, U0);
        m_qdsmc_cond_pc->SpawnConductionNodes(lev, h_sub, U, N, D,
                                              gDx, gDy, gDz, true);
        m_qdsmc_cond_pc->DepositConductionEnergy(lev, U);

        // The increment rides the stage's deposited K*N: on the LAST
        // substep add into Karr (the stage recovery divides by its own
        // deposited N once, sharing the remap); earlier substeps must
        // compose through Te directly.
        if (sub + 1 == m_qdsmc_conduction_substeps) {
            amrex::MultiFab::Subtract(U, U0, 0, 0, 1, amrex::IntVect(0));
            amrex::MultiFab::Add(Karr, U, 0, 0, 1, amrex::IntVect(0));
        } else {
            for (amrex::MFIter mfi(Te, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Box const box = mfi.tilebox();
                auto const u_arr   = U.const_array(mfi);
                auto const u0_arr  = U0.const_array(mfi);
                auto const rho_arr = rho.const_array(mfi, rho_comp);
                auto const te_arr  = Te.array(mfi);
                amrex::ParallelFor(box,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    amrex::Real const n = amrex::max(
                        rho_arr(i,j,k) / PhysConst::q_e, n_floor_rec);
                    amrex::Real const dT = (u_arr(i,j,k) - u0_arr(i,j,k))
                        / (kb_over_qe * n);
                    te_arr(i,j,k) = amrex::max(te_arr(i,j,k) + dT, 0.0_rt);
                });
            }
            Te.FillBoundary(geom.periodicity());
        }
    }
}

void HybridPICModel::QDSMCUpdateTe (int const lev) const
{
    auto & warpx = WarpX::GetInstance();
    QDSMCUpdateTe(lev, *warpx.m_fields.get(FieldType::rho_fp, lev));
}

void HybridPICModel::QDSMCUpdateTe (int const lev, amrex::MultiFab const & rho_new) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCUpdateTe()");

    auto & warpx = WarpX::GetInstance();
    amrex::Geometry const & geom = warpx.Geom(lev);
    amrex::Periodicity const & period = geom.periodicity();

    // After the QDSMC scatter, weights_fp ~= n_e (density) and entropy_fp ~=
    // K_e * N_e (entropy weighted by count, summed). Recover T_e_new:
    //
    //   K_e_new = entropy_fp / (weights_fp * V_cell)
    //   T_e_new = K_e_new / (n_e_new^(1-gamma) * k_B / q_e)
    //
    // n_e_new comes from rho_fp (post-deposit, post-particle-push).

    auto const dx_arr = geom.CellSizeArray();
    amrex::Real cell_volume = 1.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { cell_volume *= dx_arr[d]; }

    amrex::MultiFab       & Te      = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & Ke      = *warpx.m_fields.get(FieldType::hybrid_entropy_fp,              lev);
    amrex::MultiFab const & weights = *warpx.m_fields.get(FieldType::hybrid_qdsmc_weights_fp,        lev);
    amrex::MultiFab const & rho     = rho_new;

    // Note: T_e is NOT zeroed here. Cells that received no QDSMC weight or
    // are below the density floor keep their previous T_e -- zeroing them
    // would erase valid state (and seed K_e = 0 into neighbors on the next
    // step) whenever a cell momentarily receives no deposit.

    auto const gamma      = m_gamma;
    auto const n_floor    = m_qdsmc_n_floor;
    auto const kb_over_qe = PhysConst::kb / PhysConst::q_e;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real>       const & Te_arr      = Te.array(mfi);
        amrex::Array4<amrex::Real const> const & Ke_arr      = Ke.const_array(mfi);
        amrex::Array4<amrex::Real const> const & weights_arr = weights.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr     = rho.const_array(mfi);

        amrex::Box const tbox = amrex::convert(mfi.tilebox(), Te.ixType().toIntVect());
        amrex::Box       box  = tbox;
        box.grow(Te.nGrowVect());

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (rho_arr(i,j,k) <= 0.0_rt) { return; }
            amrex::Real const ne = rho_arr(i,j,k) / PhysConst::q_e;
            amrex::Real const w  = weights_arr(i,j,k) * cell_volume;
            if ((w <= 0.0_rt) || (ne <= n_floor)) { return; }
            Te_arr(i,j,k) = Ke_arr(i,j,k)
                          / std::pow(ne, 1.0_rt - gamma)
                          / w
                          / kb_over_qe;
        });
    }

    Te.FillBoundary(Te.nGrowVect(), period);
}


void HybridPICModel::QDSMCAddJouleHeating (int const lev, amrex::Real const dt,
                                           amrex::MultiFab * const redirect_E) const
{
    auto & warpx = WarpX::GetInstance();
    QDSMCAddJouleHeating(lev, dt,
        *warpx.m_fields.get(FieldType::rho_fp, lev), redirect_E);
}

void HybridPICModel::QDSMCAddJouleHeating (int const lev, amrex::Real const dt,
                                           amrex::MultiFab const & rho_in,
                                           amrex::MultiFab * const redirect_E) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCAddJouleHeating()");

    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // Per-cell resistive electron-heating source (Phys. Plasmas 31, 012902 (2024), Eq. 12).
    // With the e-i relative drift V_s - V_e = J_plasma/(e n_e) and the
    // eta-derived rate nu_{s,e} = Z_s e^2 eta n_e / m_s, the source is
    //
    //   S_e = e^2 eta n_e Sum_s Z_s n_s |J_plasma/(e n_e)|^2
    //
    // which collapses to eta J^2 for a single species. Computed on the grid from
    // rho_fp, rho_fp_s and the plasma current -- no per-particle scatter.

    auto & warpx = WarpX::GetInstance();
    amrex::Periodicity const & period = warpx.Geom(lev).periodicity();

    amrex::MultiFab       & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho = rho_in;
    ablastr::fields::VectorField J_plasma =
        warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    // B field on Yee staggering; needed (as |B|) only when a per-species
    // resistivity parser is registered for a species visited in this loop.
    // The Yee->nodal interp is cheap; we always compute it inside the kernel
    // because the branch fast-paths to skip the parser call when not needed.
    ablastr::fields::VectorField B_fp =
        warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev);

    auto const gamma_minus_1 = m_gamma - 1.0_rt;
    auto const rho_floor     = PhysConst::q_e * m_n_floor;
    auto const eta           = m_eta;
    auto const t_new         = warpx.gett_new(0);

    amrex::GpuArray<int, 3> const & Jx_stag = Jx_IndexType;
    amrex::GpuArray<int, 3> const & Jy_stag = Jy_IndexType;
    amrex::GpuArray<int, 3> const & Jz_stag = Jz_IndexType;
    amrex::GpuArray<int, 3> const & Bx_stag = Bx_IndexType;
    amrex::GpuArray<int, 3> const & By_stag = By_IndexType;
    amrex::GpuArray<int, 3> const & Bz_stag = Bz_IndexType;
    amrex::GpuArray<int, 3> const nodal     = {1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen   = {1, 1, 1};

    // Te-threshold Joule redirection: in cells with Te >= threshold the Joule
    // heat is written into redirect_E (per charged ion species, the m_i-independent
    // energy E_s [J]) for QDSMCApplyIonHeating to deposit on the ions, rather than
    // added to T_e.
    bool const do_redirect = (redirect_E != nullptr);
    auto const K_per_eV    = PhysConst::q_e / PhysConst::kb;        // T[eV]*this = T[K]
    amrex::Real const Te_thresh_K = m_joule_redirect_Te_eV * K_per_eV;

    auto & mypc = warpx.GetPartContainer();

    // Loop over every charged ion species and accumulate its per-cell
    // contribution to S_e into T_e directly. Each species contributes
    //   dT_e_s = dt (gamma-1) * Z_s e^2 eta n_s |V_s - V_e|^2 / k_B
    // (the n_e factor in nu_{s,e} cancels the 1/n_e from the T_e update).
    // V_s is computed inline from THIS STEP's current_fp_s / rho_fp_s;
    // these are both deposited WITHOUT RZ volume scaling (per
    // HybridPICDepositRhoAndJ -- apply_boundary_and_scale_volume=false),
    // so their ratio V_s = J_s/rho_s gives correct m/s with the 2pi*r
    // factors cancelling.
    //
    // n_s in the heating coefficient, however, must be a TRUE density
    // (1/m^3). Reading rho_fp_s/q_e directly gives (2pi*r) x n_s_true
    // in RZ. We recover the correct n_s from the species charge fraction
    //
    //   f_s = rho_fp_s / Sigma_t rho_fp_t   =   Z_s n_s / n_e   (unitless,
    //                                            2pi*r factor cancels)
    //   n_s = f_s * n_e / Z_s
    //
    // where n_e comes from the volume-scaled total rho_fp. Works in
    // any dimensionality (in Cartesian the 2pi*r is just 1).
    auto const species_names = mypc.GetSpeciesNames();

    // Build Sigma_t rho_fp_t (unscaled per-species charge densities) so we
    // can compute the species fraction f_s = rho_fp_s / rhos_sum per
    // cell inside the species loop.
    amrex::MultiFab rhos_sum(rho.boxArray(), rho.DistributionMap(), 1, rho.nGrowVect());
    rhos_sum.setVal(0.0_rt);
    for (auto const & spec_name : species_names) {
        auto & pc = mypc.GetParticleContainerFromName(spec_name);
        if (pc.getCharge() == 0._prt) { continue; }
        amrex::MultiFab const & rho_s =
            *warpx.m_fields.get("rho_fp_" + spec_name, lev);
        amrex::MultiFab::Add(rhos_sum, rho_s, 0, 0, 1, rho.nGrowVect());
    }

    // Charged-species component index for redirect_E (matches QDSMCApplyIonHeating).
    int ion_comp = -1;
    for (auto const & spec_name : species_names) {
        auto & pc = mypc.GetParticleContainerFromName(spec_name);
        if (pc.getCharge() == 0._prt) { continue; }
        ++ion_comp;

        amrex::Real const Z_s = pc.getCharge() / PhysConst::q_e;

        ablastr::fields::VectorField Js =
            warpx.m_fields.get_alldirs("current_fp_" + spec_name, lev);
        amrex::MultiFab const & rho_s =
            *warpx.m_fields.get("rho_fp_" + spec_name, lev);

        // Per-species resistivity overlay parser, if registered. The
        // captured executor is only called from inside the kernel when
        // has_eta_per is true; default-constructed ParserExecutor is
        // never invoked.
        auto const eta_per_it     = m_eta_per_species.find(spec_name);
        bool const has_eta_per    = (eta_per_it != m_eta_per_species.end());
        amrex::ParserExecutor<7> eta_s_per{};
        if (has_eta_per) { eta_s_per = eta_per_it->second; }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Array4<amrex::Real>       const & Te_arr     = Te.array(mfi);
            amrex::Array4<amrex::Real const> const & rho_arr    = rho.const_array(mfi);
            amrex::Array4<amrex::Real const> const & rhos_arr   = rho_s.const_array(mfi);
            amrex::Array4<amrex::Real const> const & rhosum_arr = rhos_sum.const_array(mfi);
            amrex::Array4<amrex::Real const> const & Jpx        = J_plasma[0]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Jpy        = J_plasma[1]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Jpz        = J_plasma[2]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Jsx        = Js[0]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Jsy        = Js[1]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Jsz        = Js[2]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Bx_arr     = B_fp[0]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & By_arr     = B_fp[1]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Bz_arr     = B_fp[2]->const_array(mfi);

            // Redirect output (default Array4 when redirect off -> never indexed
            // because do_redirect gates the write).
            amrex::Array4<amrex::Real> redirect_arr;
            if (do_redirect) { redirect_arr = redirect_E->array(mfi); }

            amrex::Box const & tbox = mfi.tilebox();
            amrex::ParallelFor(tbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real const rho_val = rho_arr(i,j,k);
                if (rho_val <= rho_floor) { return; }
                // n_e (m^-3) from the volume-scaled total rho_fp.
                amrex::Real const ne = rho_val / PhysConst::q_e;
                // Species charge fraction f_s = rho_fp_s / Sigma_t rho_fp_t
                // = Z_s n_s / n_e (unitless; RZ 2pi*r factor cancels). Then
                // the true per-species number density:
                //   n_s = f_s * n_e / Z_s
                amrex::Real const rhos_val_raw  = rhos_arr(i,j,k);
                amrex::Real const rhos_sum_val  = std::max(rhosum_arr(i,j,k), rho_floor);
                amrex::Real const f_s           = rhos_val_raw / rhos_sum_val;
                amrex::Real const ns            = f_s * ne / Z_s;

                // |J| at the nodal grid (where Te lives). Used by the
                // global eta parser and forwarded to per-species parsers.
                auto const jx = ablastr::coarsen::sample::Interp(Jpx, Jx_stag, nodal, coarsen, i, j, k, 0);
                auto const jy = ablastr::coarsen::sample::Interp(Jpy, Jy_stag, nodal, coarsen, i, j, k, 0);
                auto const jz = ablastr::coarsen::sample::Interp(Jpz, Jz_stag, nodal, coarsen, i, j, k, 0);
                amrex::Real const Jmag = std::sqrt(jx*jx + jy*jy + jz*jz);

                // eta_global: same Ohm's-law parser the E-solve uses, evaluated
                // per cell. This makes the per-cell heat reduce to eta J^2
                // exactly in single species (when no per-species overlay).
                amrex::Real eta_s_eff = eta(rho_val, Jmag, t_new);

                // e-i relative drift = J_plasma/(e n_e), from the nodal plasma
                // current and n_e. Energy-consistent with the eta*J dissipation
                // in Ohm's law; reduces to eta*|J|^2 for a single species.
                amrex::Real const inv_ene = 1.0_rt / (PhysConst::q_e * ne);
                amrex::Real const dvx = jx * inv_ene;
                amrex::Real const dvy = jy * inv_ene;
                amrex::Real const dvz = jz * inv_ene;
                amrex::Real const dv2 = dvx*dvx + dvy*dvy + dvz*dvz;

                // eta_per_species overlay (Phys. Plasmas 31, 012902 (2024)). Only evaluated when
                // a per-species parser is registered for this species --
                // for the other species we just see eta_s_eff = eta_global.
                if (has_eta_per) {
                    amrex::Real const Te_K_val = Te_arr(i,j,k);
                    // J_s interpolated to nodal -- only needed for the parser's
                    // |J_s| argument.
                    auto const jsx_nodal = ablastr::coarsen::sample::Interp(Jsx, Jx_stag, nodal, coarsen, i, j, k, 0);
                    auto const jsy_nodal = ablastr::coarsen::sample::Interp(Jsy, Jy_stag, nodal, coarsen, i, j, k, 0);
                    auto const jsz_nodal = ablastr::coarsen::sample::Interp(Jsz, Jz_stag, nodal, coarsen, i, j, k, 0);
                    amrex::Real const Jsmag    = std::sqrt(
                        jsx_nodal*jsx_nodal + jsy_nodal*jsy_nodal + jsz_nodal*jsz_nodal);
                    auto const bx = ablastr::coarsen::sample::Interp(Bx_arr, Bx_stag, nodal, coarsen, i, j, k, 0);
                    auto const by = ablastr::coarsen::sample::Interp(By_arr, By_stag, nodal, coarsen, i, j, k, 0);
                    auto const bz = ablastr::coarsen::sample::Interp(Bz_arr, Bz_stag, nodal, coarsen, i, j, k, 0);
                    amrex::Real const Bmag = std::sqrt(bx*bx + by*by + bz*bz);
                    eta_s_eff += eta_s_per(rhos_val_raw, rho_val, Te_K_val,
                                           Jmag, Jsmag, Bmag, t_new);
                }

                // Per-species contribution to S_e at this cell.
                //   nu_{s,e} n_s m_s |V_s - V_e|^2 = Z_s e^2 eta_s_eff n_e n_s |dV|^2
                // Dividing by (n_e k_B) for the T_e update:
                //   dT_e_s = dt (gamma-1) * Z_s e^2 eta_s_eff n_s |dV|^2 / k_B
                amrex::Real const dTe_s = dt * gamma_minus_1
                               * Z_s * PhysConst::q_e * PhysConst::q_e
                               * eta_s_eff * ns * dv2 / PhysConst::kb;
                // Te-threshold redirection: below threshold heat electrons (the
                // usual Joule deposit); at/above it write this species'
                // m_i-independent redirected energy E_s = (2/3) n_e Z_s e^2 eta
                // |dV|^2 dt [J] into its component for the ion-heating step.
                if (do_redirect && Te_arr(i,j,k) >= Te_thresh_K) {
                    redirect_arr(i,j,k,ion_comp) = (2.0_rt/3.0_rt) * ne
                        * Z_s * PhysConst::q_e * PhysConst::q_e * eta_s_eff * dv2 * dt;
                } else {
                    Te_arr(i,j,k) += dTe_s;
                }
            });
        }
    }

    Te.FillBoundary(Te.nGrowVect(), period);
}


void HybridPICModel::QDSMCAddTemperatureRelaxation (int const lev, amrex::Real const dt,
    std::map<std::string, amrex::MultiFab*> const & Ti_dep_by_species) const
{
    auto & warpx = WarpX::GetInstance();
    QDSMCAddTemperatureRelaxation(lev, dt,
        *warpx.m_fields.get(FieldType::rho_fp, lev), Ti_dep_by_species);
}

void HybridPICModel::QDSMCAddTemperatureRelaxation (int const lev, amrex::Real const dt,
    amrex::MultiFab const & rho_in,
    std::map<std::string, amrex::MultiFab*> const & Ti_dep_by_species) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCAddTemperatureRelaxation()");

    using warpx::fields::FieldType;

    // Electron-ion thermal-equilibration sink, summed over ion species s:
    //   Q_ei = Sigma_s 3 n_s k_B nu_ei (T_e - T_i_s),    dU_e/dt += -Q_ei.
    // With U_e = n_e k_B T_e/(gamma-1), the per-cell T_e obeys
    //   dT_e/dt = -(gamma-1) 3 Sigma_s (n_s/n_e) nu_ei (T_e - T_i_s),
    // where n_s/n_e = f_s/Z_s, f_s = rho_fp_s/Sigma_t rho_fp_t. T_e is stored in
    // Kelvin; T_i (deposited per species, cell-centered, in eV) is interpolated
    // to the nodal T_e grid and converted to K. This is the electron-side sink;
    // QDSMCApplyIonHeating deposits the matching ion heating so the pair
    // conserves energy.
    auto & warpx = WarpX::GetInstance();
    amrex::Periodicity const & period = warpx.Geom(lev).periodicity();

    amrex::MultiFab       & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho = rho_in;

    auto const gamma_minus_1 = m_gamma - 1.0_rt;
    auto const rho_floor     = PhysConst::q_e * m_n_floor;
    auto const nu_ei         = m_nu_ei;
    auto const t_new         = warpx.gett_new(0);
    auto const K_per_eV      = PhysConst::q_e / PhysConst::kb;   // T[eV] * this = T[K]
    // Floor on T_e in the nu_ei rate argument so pow(Te,-1.5) stays finite.
    amrex::Real const Te_floor_eV = 1.e-3_rt;

    amrex::GpuArray<int, 3> const nodal   = {1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};

    auto & mypc = warpx.GetPartContainer();
    auto const species_names = mypc.GetSpeciesNames();

    // Sigma_t rho_fp_t (unscaled per-species charge densities) -> species fraction.
    amrex::MultiFab rhos_sum(rho.boxArray(), rho.DistributionMap(), 1, rho.nGrowVect());
    rhos_sum.setVal(0.0_rt);
    for (auto const & spec_name : species_names) {
        auto & pc = mypc.GetParticleContainerFromName(spec_name);
        if (pc.getCharge() == 0._prt) { continue; }
        amrex::MultiFab const & rho_s = *warpx.m_fields.get("rho_fp_" + spec_name, lev);
        amrex::MultiFab::Add(rhos_sum, rho_s, 0, 0, 1, rho.nGrowVect());
    }

    // Cell-centered field box array (for staging the deposited T_i with a guard
    // cell so it can be interpolated to the nodal T_e grid).
    amrex::BoxArray const cc_ba = amrex::convert(Te.boxArray(), amrex::IntVect::TheCellVector());

    for (auto const & spec_name : species_names) {
        auto & pc = mypc.GetParticleContainerFromName(spec_name);
        if (pc.getCharge() == 0._prt) { continue; }
        amrex::Real const Z_s = pc.getCharge() / PhysConst::q_e;

        amrex::MultiFab const & rho_s = *warpx.m_fields.get("rho_fp_" + spec_name, lev);

        // Per-cell ion temperature [eV] (NGP velocity-variance deposit, done once
        // by the caller and shared via Ti_dep_by_species), moved onto the field's
        // cell-centered grid with one guard cell so the cc->nodal interpolation
        // has its neighbours at box edges.
        amrex::MultiFab const & Ti_dep = *Ti_dep_by_species.at(spec_name);
        amrex::MultiFab Ti_cc(cc_ba, Te.DistributionMap(), 1, 1);
        Ti_cc.setVal(0.0_rt);
        Ti_cc.ParallelCopy(Ti_dep, 0, 0, 1, amrex::IntVect::TheZeroVector(),
                           amrex::IntVect::TheZeroVector());
        Ti_cc.FillBoundary(warpx.Geom(lev).periodicity());
        // Ti_cc is cell-centered in the real dimensions; the unused (2D/1D)
        // dimensions are set NODAL so they match the nodal destination grid in
        // Interp (sf==sc there -> np=1, no out-of-bounds k=-1 read). Mirrors the
        // unused-dimension handling for J/B/E above.
        amrex::GpuArray<int, 3> cc_stag = {0, 0, 0};
        for (int d = AMREX_SPACEDIM; d < 3; ++d) { cc_stag[d] = 1; }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Array4<amrex::Real>       const & Te_arr     = Te.array(mfi);
            amrex::Array4<amrex::Real const> const & rho_arr    = rho.const_array(mfi);
            amrex::Array4<amrex::Real const> const & rhos_arr   = rho_s.const_array(mfi);
            amrex::Array4<amrex::Real const> const & rhosum_arr = rhos_sum.const_array(mfi);
            amrex::Array4<amrex::Real const> const & Ti_arr     = Ti_cc.const_array(mfi);

            amrex::Box const & tbox = mfi.tilebox();
            amrex::ParallelFor(tbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real const rho_val = rho_arr(i,j,k);
                if (rho_val <= rho_floor) { return; }
                amrex::Real const rhos_sum_val = std::max(rhosum_arr(i,j,k), rho_floor);
                amrex::Real const f_s = rhos_arr(i,j,k) / rhos_sum_val;   // = Z_s n_s/n_e

                amrex::Real const Ti_eV = ablastr::coarsen::sample::Interp(
                    Ti_arr, cc_stag, nodal, coarsen, i, j, k, 0);
                amrex::Real const Te_K  = Te_arr(i,j,k);
                amrex::Real const Te_eV = Te_K / K_per_eV;
                amrex::Real const Ti_K  = Ti_eV * K_per_eV;

                amrex::Real const nu = nu_ei(rho_val, amrex::max(Te_eV, Te_floor_eV), Ti_eV, t_new);
                // Exact exponential integration of dT_e/dt = -alpha nu (T_e - T_i),
                // with alpha = (gamma-1) 3 n_s/n_e and n_s/n_e = f_s/Z_s.
                amrex::Real const alpha = gamma_minus_1 * 3.0_rt * (f_s / Z_s);
                Te_arr(i,j,k) = Ti_K + (Te_K - Ti_K) * std::exp(-alpha * nu * dt);
            });
        }
    }

    Te.FillBoundary(Te.nGrowVect(), period);
}


void HybridPICModel::QDSMCApplyIonHeating (int const lev, amrex::Real const dt,
                                           amrex::MultiFab const * const redirect_E,
                                           std::map<std::string, amrex::MultiFab*> const * const Ti_dep_by_species) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCApplyIonHeating()");

    using warpx::fields::FieldType;

    // Stochastic Ornstein-Uhlenbeck ion-heating operator delivering both e-i energy
    // channels per particle over dt:
    //   v_p <- u_e + (v_p - u_e) exp(-nu_ei dt) + sig R,   R ~ N(0,1) per component.
    // Q_ei (when do_relax) sets the drag toward the electron fluid u_e and the thermal
    // diffusion sig^2 = k_B T_e/m_i (1 - exp(-2 nu_ei dt)). The Te-threshold redirect
    // (when do_redir) adds pure-diffusion heating E_s/m_i, with the per-species
    // redirected energy E_s [J] read from redirect_E. Both channels are per-species
    // correct (own mass, own T_i, own redirect_E comp).
    auto & warpx = WarpX::GetInstance();

    bool const do_relax = m_include_temperature_relaxation;
    bool const do_redir = (redirect_E != nullptr);
    if (!do_relax && !do_redir) { return; }

    amrex::MultiFab const & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp, lev);
    ablastr::fields::VectorField Ve =
        warpx.m_fields.get_alldirs(FieldType::hybrid_electron_velocity_fp, lev);

    auto const rho_floor = PhysConst::q_e * m_n_floor;
    auto const nu_ei     = m_nu_ei;
    auto const t_new     = warpx.gett_new(0);
    auto const K_per_eV  = PhysConst::q_e / PhysConst::kb;   // T[eV]*this = T[K]
    // Floor on T_e in the nu_ei rate argument so pow(Te,-1.5) stays finite.
    amrex::Real const Te_floor_eV = 1.e-3_rt;

    // Nodal->cc interpolation staggers (unused dims set cc-like).
    amrex::GpuArray<int, 3> nodal_src = {1, 1, 1};
    for (int d = AMREX_SPACEDIM; d < 3; ++d) { nodal_src[d] = 0; }
    amrex::GpuArray<int, 3> const cc_dst  = {0, 0, 0};
    amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};

    amrex::BoxArray const cc_ba = amrex::convert(Te.boxArray(), amrex::IntVect::TheCellVector());

    auto & mypc = warpx.GetPartContainer();
    auto const species_names = mypc.GetSpeciesNames();

    // Charged-species component index for redirect_E (matches QDSMCAddJouleHeating:
    // incremented for every charged species before the mass check).
    int ion_comp = -1;
    for (auto const & spec_name : species_names) {
        auto & pc = mypc.GetParticleContainerFromName(spec_name);
        if (pc.getCharge() == 0._prt) { continue; }
        ++ion_comp;
        auto const m_i = pc.getMass();
        if (m_i <= 0._prt) { continue; }

        // Ion temperature [eV] (NGP) -- only needed as the nu_ei parser argument
        // (Q_ei drag/diffusion). Skipped when only the redirect is active. When
        // relaxation is on, T_i was deposited once by the caller and is shared via
        // Ti_dep_by_species (QDSMCAddTemperatureRelaxation ran just before with no
        // intervening ion motion).
        amrex::MultiFab Ti_cc(cc_ba, Te.DistributionMap(), 1, 0);
        Ti_cc.setVal(0.0_rt);
        if (do_relax) {
            amrex::MultiFab const & Ti_dep = *(Ti_dep_by_species->at(spec_name));
            Ti_cc.ParallelCopy(Ti_dep, 0, 0, 1, amrex::IntVect::TheZeroVector(),
                               amrex::IntVect::TheZeroVector());
        }

        // Per-cell drag-diffusion coefficients on the cc field grid:
        //   0 = nu_ei [1/s], 1-3 = u_e [m/s], 4 = T_e [K], 5 = redirected dTe [K].
        // Defaults (0) leave inactive / below-floor cells as no-ops.
        amrex::MultiFab coef(cc_ba, Te.DistributionMap(), 6, 0);
        coef.setVal(0.0_rt);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(coef, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Array4<amrex::Real>       const & coef_arr = coef.array(mfi);
            amrex::Array4<amrex::Real const> const & rho_arr  = rho.const_array(mfi);
            amrex::Array4<amrex::Real const> const & Te_arr   = Te.const_array(mfi);
            amrex::Array4<amrex::Real const> const & Ti_arr   = Ti_cc.const_array(mfi);
            amrex::Array4<amrex::Real const> const & Vex_arr  = Ve[0]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Vey_arr  = Ve[1]->const_array(mfi);
            amrex::Array4<amrex::Real const> const & Vez_arr  = Ve[2]->const_array(mfi);
            amrex::Array4<amrex::Real const> redirect_arr;
            if (do_redir) { redirect_arr = redirect_E->const_array(mfi); }

            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real const rho_val = ablastr::coarsen::sample::Interp(
                    rho_arr, nodal_src, cc_dst, coarsen, i, j, k, 0);
                if (rho_val <= rho_floor) { return; }

                amrex::Real const Te_K = ablastr::coarsen::sample::Interp(
                    Te_arr, nodal_src, cc_dst, coarsen, i, j, k, 0);
                coef_arr(i,j,k,4) = Te_K;

                if (do_relax) {
                    amrex::Real const Ti_eV = Ti_arr(i,j,k);
                    coef_arr(i,j,k,0) = nu_ei(rho_val, amrex::max(Te_K / K_per_eV, Te_floor_eV), Ti_eV, t_new);
                    coef_arr(i,j,k,1) = ablastr::coarsen::sample::Interp(
                        Vex_arr, nodal_src, cc_dst, coarsen, i, j, k, 0);
                    coef_arr(i,j,k,2) = ablastr::coarsen::sample::Interp(
                        Vey_arr, nodal_src, cc_dst, coarsen, i, j, k, 0);
                    coef_arr(i,j,k,3) = ablastr::coarsen::sample::Interp(
                        Vez_arr, nodal_src, cc_dst, coarsen, i, j, k, 0);
                }
                if (do_redir) {
                    // E_s for this species = redirect_E component ion_comp.
                    coef_arr(i,j,k,5) = ablastr::coarsen::sample::Interp(
                        redirect_arr, nodal_src, cc_dst, coarsen, i, j, k, ion_comp);
                }
            });
        }

        // Stage the coefficients on the particle grid for NGP lookup.
        auto const & pba = pc.ParticleBoxArray(lev);
        auto const & pdm = pc.ParticleDistributionMap(lev);
        amrex::MultiFab coef_p(pba, pdm, 6, 0);
        coef_p.setVal(0.0_rt);
        coef_p.ParallelCopy(coef, 0, 0, 6);

        // Apply the drag-diffusion update to each ion (NGP cell lookup).
        auto const plo = warpx.Geom(lev).ProbLoArray();
        auto const dxi = warpx.Geom(lev).InvCellSizeArray();
        auto const kb  = PhysConst::kb;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (WarpXParIter pti(pc, lev); pti.isValid(); ++pti)
        {
            long const np = pti.numParticles();
            auto & tile = pti.GetParticleTile();
            auto ptd = tile.getParticleTileData();
            amrex::ParticleReal* AMREX_RESTRICT uxp = pti.GetAttribs(PIdx::ux).dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT uyp = pti.GetAttribs(PIdx::uy).dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT uzp = pti.GetAttribs(PIdx::uz).dataPtr();

            amrex::Array4<amrex::Real const> const & coef_arr = coef_p.const_array(pti);

            amrex::ParallelForRNG(np,
                [=] AMREX_GPU_DEVICE (long ip, amrex::RandomEngine const& engine)
            {
                auto const p = WarpXParticleContainer::ParticleType(ptd, ip);
                const auto [ii, jj, kk] = amrex::getParticleCell(p, plo, dxi).dim3();
                amrex::ParticleReal const nu   = coef_arr(ii,jj,kk,0);
                amrex::ParticleReal const Te_K = coef_arr(ii,jj,kk,4);
                amrex::ParticleReal const E_s  = coef_arr(ii,jj,kk,5);

                // Ornstein-Uhlenbeck drag and variance (Q_ei diffusion + redirect E_s).
                amrex::ParticleReal const nu_dt = nu * dt;
                amrex::ParticleReal const drag = -std::expm1(-nu_dt);            // 1 - exp(-nu dt)
                amrex::ParticleReal const sig2 =
                    (-kb * Te_K * std::expm1(-2._prt * nu_dt) + E_s) / m_i;
                if (drag <= 0._prt && sig2 <= 0._prt) { return; }

                amrex::ParticleReal const uex = coef_arr(ii,jj,kk,1);
                amrex::ParticleReal const uey = coef_arr(ii,jj,kk,2);
                amrex::ParticleReal const uez = coef_arr(ii,jj,kk,3);
                amrex::ParticleReal const sig = std::sqrt(amrex::max(0._prt, sig2));
                uxp[ip] += -drag*(uxp[ip]-uex) + sig*amrex::RandomNormal(0._prt, 1._prt, engine);
                uyp[ip] += -drag*(uyp[ip]-uey) + sig*amrex::RandomNormal(0._prt, 1._prt, engine);
                uzp[ip] += -drag*(uzp[ip]-uez) + sig*amrex::RandomNormal(0._prt, 1._prt, engine);
            });
        }
    }
}


void HybridPICModel::QDSMCFillElectronPressureFromTe (int const lev) const
{
    auto & warpx = WarpX::GetInstance();
    QDSMCFillElectronPressureFromTe(lev, *warpx.m_fields.get(FieldType::rho_fp, lev));
}

void HybridPICModel::QDSMCFillElectronPressureFromTe (int const lev,
    amrex::MultiFab const & rho_in) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCFillElectronPressureFromTe()");

    auto & warpx = WarpX::GetInstance();

    amrex::MultiFab       & Pe  = *warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    amrex::MultiFab const & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho = rho_in;

    auto const rho_floor = PhysConst::q_e * m_n_floor;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Pe, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real>       const & Pe_arr  = Pe.array(mfi);
        amrex::Array4<amrex::Real const> const & Te_arr  = Te.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);

        amrex::Box const & tbox = mfi.tilebox();
        amrex::ParallelFor(tbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real const rho_val = std::max(rho_arr(i,j,k), rho_floor);
            amrex::Real const ne      = rho_val / PhysConst::q_e;
            Pe_arr(i,j,k) = ne * PhysConst::kb * Te_arr(i,j,k);
        });
    }

    // The Ohm's-law kernels difference Pe across box edges, so the domain
    // boundary must be applied and the ghosts filled (mirrors
    // CalculateElectronPressure).
    warpx.ApplyElectronPressureBoundary(lev, PatchType::fine);
    Pe.FillBoundary(warpx.Geom(lev).periodicity());
}


void HybridPICModel::AdvanceElectronEnergyQDSMC (amrex::Real const dt) const
{
    ABLASTR_PROFILE("HybridPICModel::AdvanceElectronEnergyQDSMC()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_qdsmc_pc != nullptr,
        "AdvanceElectronEnergyQDSMC called with "
        "solve_electron_energy_equation=true but the "
        "QDSMC particle container was not constructed (InitData not run?)");

    auto & warpx = WarpX::GetInstance();

    // Strang first half of the electron thermal conduction (Phase 3a):
    // cond(dt/2) o [advect + sources](dt) o cond(dt/2). The pass updates
    // T_e in place, so the entropy seed below picks up the conducted state.
    const bool cond_fused = (std::getenv("WARPX_COND_FUSED") != nullptr);
    if (m_qdsmc_conduction != "off" && !cond_fused) {
        PrintTeVariance("step-entry");
        QdsmcConductionPass(0.5_rt*dt, FieldType::hybrid_rho_fp_temp, 0);
        PrintTeVariance("post-prehalf");
    }

    // J_plasma at the current B (B^{n+1/2} from the last Faraday substep) is
    // needed for V_e. The downstream final-state E-solve also recomputes
    // J_plasma later, so this call is redundant work in some configurations
    // but keeps the QDSMC sequence self-contained.
    CalculatePlasmaCurrent(
        warpx.m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, warpx.finestLevel()),
        warpx.GetEBUpdateEFlag());

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        // Step 1: grid-side initialization at t = n
        QDSMCInitializeUe(lev);
        QDSMCInitializeKe(lev);


        using ablastr::fields::Direction;
        amrex::MultiFab const & Vex = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{0}, lev);
        amrex::MultiFab const & Vey = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{1}, lev);
        amrex::MultiFab const & Vez = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{2}, lev);
        amrex::MultiFab const & Ke  = *warpx.m_fields.get(FieldType::hybrid_entropy_fp,           lev);
        amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::hybrid_rho_fp_temp,          lev);
        amrex::MultiFab       & Karr_out    = *warpx.m_fields.get(FieldType::hybrid_entropy_fp,        lev);
        amrex::MultiFab       & weights_out = *warpx.m_fields.get(FieldType::hybrid_qdsmc_weights_fp, lev);

        // Step 2: load each QDSMC particle with V_e and (K_e * N_e, N_e) from
        // its home cell.
        m_qdsmc_pc->SetV(lev, Vex, Vey, Vez);
        m_qdsmc_pc->SetK(lev, Ke, rho);

        // Step 3: forward-Euler push by dt; redistribute so particles end up
        // in their new tile.
        m_qdsmc_pc->PushX(lev, dt);

        // Step 4: scatter the carried entropy and weight onto the grid (each
        // call zeroes its target field, then deposits, then SumBoundary).
        m_qdsmc_pc->DepositK(lev, Karr_out);
        m_qdsmc_pc->DepositField(lev, weights_out);

        // PROTOTYPE: fused conduction increment into the stage's deposit
        // (shares the stage's single remap; see
        // QdsmcConductionFusedIncrement).
        if (m_qdsmc_conduction != "off"
            && std::getenv("WARPX_COND_FUSED") != nullptr) {
            QdsmcConductionFusedIncrement(dt, Karr_out,
                                          FieldType::hybrid_rho_fp_temp, 0);
        }

        // Step 5: recover T_e^{n+1} from (deposited K*N) / (deposited N) and
        // the updated n_e (from rho_fp = rho^{n+1}).
        QDSMCUpdateTe(lev);


        // Step 6: Joule-heating source on T_e (Phys. Plasmas 31, 012902 (2024), Eq. 12), per-cell from
        // rho_fp(_s), the plasma current, and the Ohm's-law eta parser. With the
        // Te-threshold redirect on, the above-threshold heat is staged in
        // ion_redirect_E (per-charged-species energy, J) for the ion-heating step.
        bool redirect_active = m_include_joule_heating && m_joule_redirect_to_ions;
        int n_ion_species = 0;
        if (redirect_active) {
            auto & mpc = warpx.GetPartContainer();
            for (auto const & nm : mpc.GetSpeciesNames()) {
                if (mpc.GetParticleContainerFromName(nm).getCharge() != 0._prt) { ++n_ion_species; }
            }
            if (n_ion_species == 0) { redirect_active = false; }
        }
        amrex::MultiFab ion_redirect_E;
        if (redirect_active) {
            amrex::MultiFab const & Te_mf =
                *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
            ion_redirect_E.define(Te_mf.boxArray(), Te_mf.DistributionMap(), n_ion_species, 0);
            ion_redirect_E.setVal(0.0_rt);
        }
        if (m_include_joule_heating) {
            QDSMCAddJouleHeating(lev, dt, redirect_active ? &ion_redirect_E : nullptr);
        }

        // Steps 6b/6c both need each charged species' T_i when Q_ei relaxation is
        // on. Deposit it ONCE here (the expensive per-particle NGP temperature
        // reduction) and share it: the electron sink (6b) and the ion-heating
        // operator (6c) run back-to-back with no intervening ion motion, so the
        // deposited T_i is identical for both.
        std::map<std::string, amrex::MultiFab*> Ti_dep_by_species;
        // Owns the per-species cell-centered scalar T_i built from the shape-aware
        // deposition; must outlive the QDSMCAddTemperatureRelaxation /
        // QDSMCApplyIonHeating calls that read it through Ti_dep_by_species.
        std::map<std::string, std::unique_ptr<amrex::MultiFab>> Ti_scalar_owned;
        if (m_include_temperature_relaxation) {
            QDSMCBuildTiDeposits(lev, Ti_scalar_owned, Ti_dep_by_species);
        }

        // Step 6b: electron-ion thermal-equilibration (Q_ei) sink on T_e
        // (cools T_e toward each ion species' T_i).
        if (m_include_temperature_relaxation) {
            QDSMCAddTemperatureRelaxation(lev, dt, Ti_dep_by_species);
        }

        // Step 6c: stochastic drag-diffusion ion-heating operator -- delivers the Q_ei
        // conjugate (when relaxation is on) and/or the redirected Joule energy
        // (when the redirect is on), so the ions are heated by one mechanism.
        if (m_include_temperature_relaxation || redirect_active) {
            QDSMCApplyIonHeating(lev, dt, redirect_active ? &ion_redirect_E : nullptr,
                                 m_include_temperature_relaxation ? &Ti_dep_by_species : nullptr);
        }

        // Strang second half of the electron thermal conduction, on the
        // post-advection density rho^{n+1} (the pass loops levels
        // internally; the single-level guard mirrors the solver-wide
        // finest_level == 0 restriction).
        if (m_qdsmc_conduction != "off" && lev == 0 && !cond_fused) {
            PrintTeVariance("post-stage");
            if (std::getenv("WARPX_COND_SKIP_POST") == nullptr) {
                QdsmcConductionPass(0.5_rt*dt, FieldType::rho_fp, 0);
            }
            PrintTeVariance("post-posthalf");
        }

        // Step 7: emit P_e = n_e * k_B * T_e for the downstream Ohm's-law solve.
        QDSMCFillElectronPressureFromTe(lev);

        // Step 8: reset particles to home positions (and zero velocity /
        // weight / entropy) so the next step starts with a clean grid.
        m_qdsmc_pc->ResetParticles(lev);
    }
}


void HybridPICModel::QDSMCBuildTiDeposits (int const lev,
    std::map<std::string, std::unique_ptr<amrex::MultiFab>> & owned,
    std::map<std::string, amrex::MultiFab*> & by_name) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCBuildTiDeposits()");

    using ablastr::fields::Direction;

    auto & warpx = WarpX::GetInstance();

    amrex::GpuArray<int, 3> const Tr_stag = Jx_IndexType;
    amrex::GpuArray<int, 3> const Tt_stag = Jy_IndexType;
    amrex::GpuArray<int, 3> const Tz_stag = Jz_IndexType;
    amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};
    // Cell-centered target in the real dimensions; in collapsed dimensions
    // (index >= AMREX_SPACEDIM, e.g. theta in RZ or y in 2D) match the source
    // staggering so Interp does not read the out-of-bounds neighbour there.
    amrex::GpuArray<int, 3> cc_r = {0, 0, 0};
    amrex::GpuArray<int, 3> cc_t = {0, 0, 0};
    amrex::GpuArray<int, 3> cc_z = {0, 0, 0};
    for (int d = AMREX_SPACEDIM; d < 3; ++d) {
        cc_r[d] = Tr_stag[d]; cc_t[d] = Tt_stag[d]; cc_z[d] = Tz_stag[d];
    }

    auto & mpc_ti = warpx.GetPartContainer();
    for (auto const & nm : mpc_ti.GetSpeciesNames()) {
        auto & pc = mpc_ti.GetParticleContainerFromName(nm);
        if (pc.getCharge() == 0._prt) { continue; }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(pc.getTemperatureDepositionFlag(),
            "QDSMC include_temperature_relaxation requires the ion species to set "
            "do_temperature_deposition = 1 (for shape-consistent T_i).");

        // Shape-aware ion temperature (particle-shape order, consistent with
        // charge/current) in the Yee-staggered 3-component T_<nm> vector field
        // (Tr,Tt,Tz), deposited in Kelvin by the caller's DepositTemperatures
        // pass and read here. Fill guard cells so the cell-centered
        // interpolation below reads finite neighbours at box/domain edges.
        auto const T_vf = warpx.m_fields.get_mr_levels_alldirs("T_" + nm, warpx.finestLevel());
        for (int idim = 0; idim < 3; ++idim) {
            T_vf[lev][Direction{idim}]->FillBoundary(warpx.Geom(lev).periodicity());
        }

        // Collapse the staggered vector to a cell-centered scalar
        // T_i = (Tr + Tt + Tz)/3 by interpolating each component to CC
        // (Path A: accept the CC interpolation for shape consistency).
        amrex::MultiFab const & Te_ref =
            *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
        amrex::BoxArray const cc_ba =
            amrex::convert(Te_ref.boxArray(), amrex::IntVect::TheCellVector());
        auto Ti_s = std::make_unique<amrex::MultiFab>(
            cc_ba, Te_ref.DistributionMap(), 1, 0);
        Ti_s->setVal(0.0_rt);

        // AccumulateVelocitiesAndComputeTemperature writes T_<nm> in Kelvin;
        // the Q_ei consumers (and the nu_ei parser) expect T_i in eV, matching
        // the previous NGP deposit. Convert K -> eV below.
        amrex::Real const K_per_eV = PhysConst::q_e / PhysConst::kb;

        amrex::MultiFab const & Tr = *T_vf[lev][Direction{0}];
        amrex::MultiFab const & Tt = *T_vf[lev][Direction{1}];
        amrex::MultiFab const & Tz = *T_vf[lev][Direction{2}];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(*Ti_s, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box const & bx = mfi.tilebox();
            amrex::Array4<amrex::Real>       const & Ti_arr = Ti_s->array(mfi);
            amrex::Array4<amrex::Real const> const & Tr_arr = Tr.const_array(mfi);
            amrex::Array4<amrex::Real const> const & Tt_arr = Tt.const_array(mfi);
            amrex::Array4<amrex::Real const> const & Tz_arr = Tz.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real const tr = ablastr::coarsen::sample::Interp(
                    Tr_arr, Tr_stag, cc_r, coarsen, i, j, k, 0);
                amrex::Real const tt = ablastr::coarsen::sample::Interp(
                    Tt_arr, Tt_stag, cc_t, coarsen, i, j, k, 0);
                amrex::Real const tz = ablastr::coarsen::sample::Interp(
                    Tz_arr, Tz_stag, cc_z, coarsen, i, j, k, 0);
                Ti_arr(i, j, k) = (tr + tt + tz) / (3._rt * K_per_eV);  // K -> eV
            });
        }
        by_name[nm] = Ti_s.get();
        owned[nm] = std::move(Ti_s);
    }
}


void HybridPICModel::QDSMCFillElectronPressureTheta (int const lev, amrex::Real const theta) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCFillElectronPressureTheta()");

    auto & warpx = WarpX::GetInstance();

    amrex::MultiFab       & Pe      = *warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    amrex::MultiFab const & Te      = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & Te_old  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_old_fp, lev);
    amrex::MultiFab const & rho     = *warpx.m_fields.get(FieldType::rho_fp, lev);
    amrex::MultiFab const & rho_old = *warpx.m_fields.get(FieldType::hybrid_rho_fp_temp, lev);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rho.nComp() >= 2,
        "QDSMCFillElectronPressureTheta requires the two time-level rho components "
        "allocated by the theta-implicit hybrid scheme");

    auto const rho_floor = PhysConst::q_e * m_n_floor;
    auto const th = theta;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Pe, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real>       const & Pe_arr      = Pe.array(mfi);
        amrex::Array4<amrex::Real const> const & Te_arr      = Te.const_array(mfi);
        amrex::Array4<amrex::Real const> const & Te_old_arr  = Te_old.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr     = rho.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rho_old_arr = rho_old.const_array(mfi);

        amrex::Box const & tbox = mfi.tilebox();
        int const c_half = rho.nComp()/2;  // second time level (mode 0)
        amrex::ParallelFor(tbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // rho^{n+theta} extrapolated from rho^n (saved at step start in
            // hybrid_rho_fp_temp) and rho^{n+1/2} (the current-iterate
            // midpoint deposit in the second time-level component):
            //   rho^{n+theta} = (1-2 theta) rho^n + 2 theta rho^{n+1/2}.
            amrex::Real const rho_theta = (1.0_rt - 2.0_rt*th) * rho_old_arr(i,j,k)
                                        +           2.0_rt*th  * rho_arr(i,j,k,c_half);
            amrex::Real const rho_val = std::max(rho_theta, rho_floor);
            amrex::Real const ne      = rho_val / PhysConst::q_e;
            amrex::Real const Te_th   = (1.0_rt - th) * Te_old_arr(i,j,k) + th * Te_arr(i,j,k);
            Pe_arr(i,j,k) = ne * PhysConst::kb * Te_th;
        });
    }

    // See QDSMCFillElectronPressureFromTe: the Ohm's-law kernels difference
    // Pe across box edges.
    warpx.ApplyElectronPressureBoundary(lev, PatchType::fine);
    Pe.FillBoundary(warpx.Geom(lev).periodicity());
}


void HybridPICModel::QDSMCSaveImplicitStepStart () const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCSaveImplicitStepStart()");

    using ablastr::fields::Direction;

    auto & warpx = WarpX::GetInstance();

    m_qdsmc_Ti_owned.clear();
    m_qdsmc_Ti_by_name.clear();

    // T_i^n deposits, frozen for the whole step (used by the in-residual Q_ei
    // sink and the post-solve ion-heating kick).
    if (m_include_temperature_relaxation) {
        warpx.GetPartContainer().DepositTemperatures(warpx.m_fields, 0.0_rt);
    }

    // rho^n = rho(x^n), deposited once now (particles are at t^n). During the
    // nonlinear iteration the rho_fp components hold midpoint-position
    // deposits, so the true start-of-step density is not recoverable from
    // them; the (otherwise unused on this path) hybrid_rho_fp_temp fab stores
    // it for the whole step.
    //
    // MultiParticleContainer::DepositCharge applies the radial inverse-volume
    // scaling itself but leaves the per-species deposits LOCAL: shape-spread
    // contributions land in guard cells and boundary-shared nodes hold
    // partial sums until a SumBoundary. The K_e seed and the rho^{n+1}
    // extrapolation read absolute node values, so fold the guards and fill
    // ghosts for the grown-box consumers (no additional volume scaling here
    // -- applying it twice puts a spurious 1/r on the density).
    auto rho_n_levels = warpx.m_fields.get_mr_levels(FieldType::hybrid_rho_fp_temp, warpx.finestLevel());
    warpx.GetPartContainer().DepositCharge(rho_n_levels, 0.0_rt);
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        amrex::MultiFab & rho_n_mf = *rho_n_levels[lev];
        ablastr::utils::communication::SumBoundary(
            rho_n_mf, 0, rho_n_mf.nComp(), rho_n_mf.nGrowVect(), rho_n_mf.nGrowVect(),
            WarpX::do_single_precision_comms, warpx.Geom(lev).periodicity());
        // Same wall fold the main loop applies to rho_fp (reflect the
        // shape-spill past reflecting/PEC walls back onto the boundary
        // nodes); without it the boundary-node density is half-valued.
        warpx.ApplyRhofieldBoundary(lev, &rho_n_mf, PatchType::fine);
        rho_n_mf.FillBoundary(warpx.Geom(lev).periodicity());
    }

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        // T_e^n
        amrex::MultiFab       & Te_old = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_old_fp, lev);
        amrex::MultiFab const & Te     = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
        amrex::MultiFab::Copy(Te_old, Te, 0, 0, Te.nComp(), Te.nGrowVect());

        // J_plasma(B^n): the caller refreshed hybrid_current_fp_plasma from
        // B^n immediately before this call.
        for (int dir = 0; dir < 3; ++dir) {
            amrex::MultiFab       & Jp_old = *warpx.m_fields.get(FieldType::hybrid_current_fp_plasma_old, Direction{dir}, lev);
            amrex::MultiFab const & Jp     = *warpx.m_fields.get(FieldType::hybrid_current_fp_plasma,     Direction{dir}, lev);
            amrex::MultiFab::Copy(Jp_old, Jp, 0, 0, Jp.nComp(), Jp.nGrowVect());
        }

        if (m_include_temperature_relaxation) {
            QDSMCBuildTiDeposits(lev, m_qdsmc_Ti_owned[lev], m_qdsmc_Ti_by_name[lev]);
        }
    }
}


void HybridPICModel::AdvanceElectronEnergyQDSMCTheta (amrex::Real const dt,
                                                      amrex::Real const theta,
                                                      bool const refresh_species_deposits) const
{
    ABLASTR_PROFILE("HybridPICModel::AdvanceElectronEnergyQDSMCTheta()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_qdsmc_pc != nullptr,
        "AdvanceElectronEnergyQDSMCTheta called with "
        "solve_electron_energy_equation=true but the "
        "QDSMC particle container was not constructed (InitData not run?)");

    using ablastr::fields::Direction;

    auto & warpx = WarpX::GetInstance();

    // Per-species charge deposits feeding the multi-species source terms
    // (species charge fractions in the Joule and Q_ei kernels). Refreshed
    // once per Newton iteration; frozen during Jacobian evaluations so the
    // finite-difference linearization sees fixed species fractions.
    if (refresh_species_deposits &&
        (m_include_joule_heating || m_include_temperature_relaxation)) {
        auto & mypc = warpx.GetPartContainer();
        for (auto const & nm : mypc.GetSpeciesNames()) {
            auto & pc = mypc.GetParticleContainerFromName(nm);
            if (pc.getCharge() == 0._prt) { continue; }
            pc.DepositCharge(warpx.m_fields.get_mr_levels("rho_fp_" + nm, warpx.finestLevel()),
                             /*local*/false, /*reset*/true,
                             /*apply_boundary_and_scale_volume*/false,
                             /*interpolate_across_levels*/false);
        }
    }

    // Midpoint-position density deposited through the same
    // MultiParticleContainer path as rho^n (QDSMCSaveImplicitStepStart) and
    // rho^{n+1} (QDSMCFinishImplicitStep). The nonlinear-solver deposit in
    // rho_fp treats the boundary nodes differently in the radial geometries
    // (see m_qdsmc_rho_mid), and the seed/recovery algebra must not mix
    // density conventions: T_e ~ K (n_recover / n_seed)^(gamma-1) at a
    // stationary node, so a convention mismatch acts as a steady spurious
    // boundary heat source. The ions sit at the current-iterate midpoint
    // positions here, which is exactly the state this deposit captures.
    ablastr::fields::MultiLevelScalarField rho_mid_levels;
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        amrex::MultiFab const & rho_fp = *warpx.m_fields.get(FieldType::rho_fp, lev);
        auto & slot = m_qdsmc_rho_mid[lev];
        if (!slot) {
            slot = std::make_unique<amrex::MultiFab>(
                rho_fp.boxArray(), rho_fp.DistributionMap(), 1, rho_fp.nGrowVect());
        }
        rho_mid_levels.push_back(slot.get());
    }
    warpx.GetPartContainer().DepositCharge(rho_mid_levels, 0.0_rt);
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        amrex::MultiFab & rho_mid_mf = *rho_mid_levels[lev];
        ablastr::utils::communication::SumBoundary(
            rho_mid_mf, 0, rho_mid_mf.nComp(), rho_mid_mf.nGrowVect(), rho_mid_mf.nGrowVect(),
            WarpX::do_single_precision_comms, warpx.Geom(lev).periodicity());
        warpx.ApplyRhofieldBoundary(lev, &rho_mid_mf, PatchType::fine);
        rho_mid_mf.FillBoundary(warpx.Geom(lev).periodicity());
    }

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        // Step 0: re-entrancy -- every residual evaluation restarts the
        // energy stage from the saved T_e^n, and returns the markers to
        // their homes (also required so the home-cell gathers below read
        // tile-local data).
        amrex::MultiFab       & Te     = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
        amrex::MultiFab const & Te_old = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_old_fp, lev);
        amrex::MultiFab::Copy(Te, Te_old, 0, 0, Te.nComp(), Te.nGrowVect());
        m_qdsmc_pc->ResetParticles(lev);

        // Strang first half of the electron thermal conduction, from the
        // restored T_e^n on rho^n (re-entrant: runs inside every residual
        // evaluation with iterate-state coefficients; the pass loops levels
        // internally, single-level guard as elsewhere).
        if (m_qdsmc_conduction != "off" && lev == 0) {
            QdsmcConductionPass(0.5_rt*dt, FieldType::hybrid_rho_fp_temp, 0);
        }

        // Charge-density states: rho^n was deposited once at step start into
        // hybrid_rho_fp_temp (see QDSMCSaveImplicitStepStart); rho^{n+1/2} is
        // component 1 of rho_fp, deposited at the current-iterate midpoint
        // particle positions during PreRHSOp. (Component 0 holds a
        // previous-iterate midpoint deposit, NOT rho^n.)
        amrex::MultiFab & rho_fp = *warpx.m_fields.get(FieldType::rho_fp, lev);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rho_fp.nComp() >= 2,
            "AdvanceElectronEnergyQDSMCTheta requires the two time-level rho components");
        amrex::MultiFab const & rho_n = *warpx.m_fields.get(FieldType::hybrid_rho_fp_temp, lev);
        // Second time level starts at nComp/2 (mode 0 of that level in RZ
        // multimode; identical to component 1 in the m=0-only case).
        amrex::MultiFab rho_half(rho_fp, amrex::make_alias, rho_fp.nComp()/2, 1);
        // Midpoint density of the seed's own deposit convention (see above);
        // all K_e/T_e recovery pairings below use this family, while the
        // Ohm's-law-side quantities (V_e, sources, the emitted pressure)
        // keep the solver's rho_fp states.
        amrex::MultiFab const & rho_mid = *rho_mid_levels[lev];
        // Extrapolated end-of-step density iterate. The GEOMETRIC form
        // rho^{n+1} = rho_mid^2 / rho^n is used instead of the linear
        // 2 rho_mid - rho^n: both are second-order for smooth density,
        // but the two deposits carry different shape-noise realizations
        // and the linear combination amplifies their difference into a
        // dt-INDEPENDENT recovery bias (measured as a flat 6e-3 offset in
        // the theta = 1 temporal-order study), while the per-node ratio
        // cancels the common deposition-shape convention exactly.
        amrex::MultiFab rho_np1(rho_fp.boxArray(), rho_fp.DistributionMap(), 1, rho_fp.nGrowVect());
        {
            amrex::Real const rho_floor_x = PhysConst::q_e * m_n_floor * 1.0e-6_rt;
            for (amrex::MFIter mfi(rho_np1, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Box const box = mfi.growntilebox(rho_np1.nGrowVect());
                auto const mid = rho_mid.const_array(mfi);
                auto const rn  = rho_n.const_array(mfi);
                auto const out = rho_np1.array(mfi);
                amrex::ParallelFor(box,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    out(i,j,k) = mid(i,j,k) * mid(i,j,k)
                        / amrex::max(rn(i,j,k), rho_floor_x);
                });
            }
        }

        // Step 1: midpoint electron fluid velocity
        //   V_e^{n+1/2} = -(J_plasma^{n+1/2} - J_i^{n+1/2}) / rho^{n+1/2},
        // with J_i^{n+1/2} the implicit particle deposit (current_fp) and the
        // plasma current interpolated between J_plasma(B^n) and
        // J_plasma(B^{n+theta}) with weight 1/(2 theta).
        QDSMCInitializeUe(lev, rho_half,
            warpx.m_fields.get_alldirs(FieldType::current_fp, lev),
            warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev),
            warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma_old, lev),
            0.5_rt/theta);

        // Step 2: seed the entropy from the endpoint state (T_e^n, rho^n)
        QDSMCInitializeKe(lev, rho_n);

        amrex::MultiFab const & Vex = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{0}, lev);
        amrex::MultiFab const & Vey = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{1}, lev);
        amrex::MultiFab const & Vez = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{2}, lev);
        amrex::MultiFab const & Ke  = *warpx.m_fields.get(FieldType::hybrid_entropy_fp,           lev);
        amrex::MultiFab       & Karr_out    = *warpx.m_fields.get(FieldType::hybrid_entropy_fp,        lev);
        amrex::MultiFab       & weights_out = *warpx.m_fields.get(FieldType::hybrid_qdsmc_weights_fp, lev);

        m_qdsmc_pc->SetV(lev, Vex, Vey, Vez);
        m_qdsmc_pc->SetK(lev, Ke, rho_n);

        if (theta == 0.5_rt) {
            // Time-centered path: the residual only needs midpoint-state
            // quantities, so recover the temperature AT the temporal midpoint
            // from midpoint-position deposits and the midpoint density --
            // no time extrapolation of the density anywhere. (Recovering
            // T^{n+1} with the extrapolated 2 rho^{n+1/2} - rho^n instead
            // carries a one-sided deposition-shape bias, O(dt^2) per step,
            // that contaminates the scheme with a global O(dt) error.)
            //
            // Step 3a: markers to the temporal midpoint of the characteristic
            m_qdsmc_pc->PushX(lev, dt, 0.5_rt);

            // Step 4a: midpoint scatter and recovery: T_e^{n+1/2} from the
            // midpoint entropy deposit and the midpoint density deposit
            // (same deposit convention as the seed)
            m_qdsmc_pc->DepositK(lev, Karr_out);
            m_qdsmc_pc->DepositField(lev, weights_out);
            QDSMCUpdateTe(lev, rho_mid);

            // Step 5a: half-step deterministic sources with midpoint
            // coefficient states, so the emitted pressure is the midpoint
            // value including sources. The stochastic ion-heating
            // realization (Q_ei conjugate) is applied once, post-solve, in
            // QDSMCFinishImplicitStep; the Te-threshold Joule redirect is
            // not yet supported on the implicit path (guarded at input
            // parsing).
            if (m_include_joule_heating) {
                QDSMCAddJouleHeating(lev, 0.5_rt*dt, rho_half, nullptr);
            }
            if (m_include_temperature_relaxation) {
                QDSMCAddTemperatureRelaxation(lev, 0.5_rt*dt, rho_half, m_qdsmc_Ti_by_name[lev]);
            }

            // Step 6a: emit Pe^{n+1/2} = n^{n+1/2} k_B T_e^{n+1/2} for the
            // Ohm's-law E-solve of this residual evaluation. The markers are
            // left at the midpoint; QDSMCFinishImplicitStep completes the
            // characteristic and recovers T_e^{n+1} once, post-solve.
            QDSMCFillElectronPressureFromTe(lev, rho_half);
        } else {
            // Dissipative (theta > 1/2) path: Pe^{n+theta} requires the
            // end-of-step temperature iterate, recovered with the
            // time-extrapolated density. The extrapolation carries a small
            // deposition-shape bias (see above); acceptable here since
            // theta > 1/2 biasing is itself first-order in time.
            //
            // Step 3: second-order midpoint characteristic push: move to the
            // temporal midpoint, resample V_e there, then complete the full
            // step from home with the midpoint velocity.
            m_qdsmc_pc->PushX(lev, dt, 0.5_rt);
            m_qdsmc_pc->GatherVAtCurrentPosition(lev, Vex, Vey, Vez);
            m_qdsmc_pc->PushX(lev, dt, 1.0_rt);

            // Step 4: scatter and recover T_e^{n+1} with the end-of-step density
            m_qdsmc_pc->DepositK(lev, Karr_out);
            m_qdsmc_pc->DepositField(lev, weights_out);
            QDSMCUpdateTe(lev, rho_np1);

            // Step 5: full-step deterministic sources with midpoint
            // coefficient states.
            if (m_include_joule_heating) {
                QDSMCAddJouleHeating(lev, dt, rho_half, nullptr);
            }
            if (m_include_temperature_relaxation) {
                QDSMCAddTemperatureRelaxation(lev, dt, rho_half, m_qdsmc_Ti_by_name[lev]);
            }

            // Strang second half of the electron thermal conduction, on
            // the midpoint density rho^{n+1/2} (rho_fp component 1).
            if (m_qdsmc_conduction != "off" && lev == 0) {
                QdsmcConductionPass(0.5_rt*dt, FieldType::rho_fp, 1);
            }

            // Step 6: emit Pe^{n+theta} for the Ohm's-law E-solve of this
            // residual evaluation.
            QDSMCFillElectronPressureTheta(lev, theta);
        }
    }
}


void HybridPICModel::QDSMCFinishImplicitStep (amrex::Real const dt, amrex::Real const theta) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCFinishImplicitStep()");

    using ablastr::fields::Direction;

    auto & warpx = WarpX::GetInstance();

    // On the time-centered path the residual leaves the markers at the
    // temporal midpoint of the characteristic; complete the push with the
    // converged midpoint electron velocity and scatter the entropy at the
    // end-of-step positions before recovering T_e^{n+1}.
    if (theta == 0.5_rt) {
        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            amrex::MultiFab const & Vex = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{0}, lev);
            amrex::MultiFab const & Vey = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{1}, lev);
            amrex::MultiFab const & Vez = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{2}, lev);
            amrex::MultiFab & Karr_out    = *warpx.m_fields.get(FieldType::hybrid_entropy_fp,        lev);
            amrex::MultiFab & weights_out = *warpx.m_fields.get(FieldType::hybrid_qdsmc_weights_fp, lev);
            m_qdsmc_pc->GatherVAtCurrentPosition(lev, Vex, Vey, Vez);
            m_qdsmc_pc->PushX(lev, dt, 1.0_rt);
            m_qdsmc_pc->DepositK(lev, Karr_out);
            m_qdsmc_pc->DepositField(lev, weights_out);
        }
    }

    // True end-of-step charge density: the ions are at x^{n+1} now, so a
    // fresh deposit replaces the extrapolated 2 rho^{n+1/2} - rho^n iterate
    // used inside the residual. Recovering T_e^{n+1} from this deposit keeps
    // the polytropic reconstruction correlated with the same PIC deposit
    // noise every other consumer of rho^{n+1} sees (the in-residual
    // extrapolation both amplifies the deposit noise and decorrelates it).
    // hybrid_rho_fp_temp (rho^n) is no longer needed this step, so reuse it.
    //
    // FinishImplicitParticleUpdate just extrapolated the positions to x^{n+1}
    // without redistributing, so particles can sit outside their tiles'
    // deposit-safe range; redistribute first (the main loop redistributes
    // again later, which is harmless).
    warpx.GetPartContainer().Redistribute();
    auto rho_end_levels = warpx.m_fields.get_mr_levels(FieldType::hybrid_rho_fp_temp, warpx.finestLevel());
    warpx.GetPartContainer().DepositCharge(rho_end_levels, 0.0_rt);
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        // DepositCharge already applied the radial inverse-volume scaling;
        // the guard fold, the wall reflection and the ghost fill complete
        // the same treatment the main loop gives rho_fp.
        amrex::MultiFab & rho_end_mf = *rho_end_levels[lev];
        ablastr::utils::communication::SumBoundary(
            rho_end_mf, 0, rho_end_mf.nComp(), rho_end_mf.nGrowVect(), rho_end_mf.nGrowVect(),
            WarpX::do_single_precision_comms, warpx.Geom(lev).periodicity());
        warpx.ApplyRhofieldBoundary(lev, &rho_end_mf, PatchType::fine);
        rho_end_mf.FillBoundary(warpx.Geom(lev).periodicity());
    }

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        // Stochastic ion-heating realization (Q_ei conjugate), applied once
        // with converged states: ions are at (x,u)^{n+1}, the electron fluid
        // velocity holds the converged midpoint state, and T_i is the deposit
        // frozen at t^n. Kept out of the residual so finite-difference
        // Jacobian-vector products stay smooth.
        if (m_include_temperature_relaxation) {
            QDSMCApplyIonHeating(lev, dt, nullptr, &m_qdsmc_Ti_by_name[lev]);
        }

        // Final T_e^{n+1} recovery from the last residual evaluation's marker
        // deposits (still on the grid) and the true end-of-step density. The
        // recovery overwrites the transported temperature, so the
        // deterministic sources are re-applied with the same converged
        // midpoint coefficient states the residual used, then
        // Pe^{n+1} = n^{n+1} k_B T_e^{n+1} is emitted for the next step /
        // diagnostics.
        amrex::MultiFab const & rho_end = *rho_end_levels[lev];
        QDSMCUpdateTe(lev, rho_end);
        amrex::MultiFab & rho_fp = *warpx.m_fields.get(FieldType::rho_fp, lev);
        amrex::MultiFab rho_half(rho_fp, amrex::make_alias, rho_fp.nComp()/2, 1);
        if (m_include_joule_heating) {
            QDSMCAddJouleHeating(lev, dt, rho_half, nullptr);
        }
        if (m_include_temperature_relaxation) {
            QDSMCAddTemperatureRelaxation(lev, dt, rho_half, m_qdsmc_Ti_by_name[lev]);
        }
        QDSMCFillElectronPressureFromTe(lev, rho_end);

        // Return the markers to their homes for the next step.
        m_qdsmc_pc->ResetParticles(lev);
    }

    m_qdsmc_Ti_owned.clear();
    m_qdsmc_Ti_by_name.clear();
}


void HybridPICModel::BfieldEvolve (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    int step, amrex::Real dt_half, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        BfieldEvolve(
            Bfield, Efield, Jfield, rhofield, eb_update_E,
            step, dt_half, lev, subcycling_half, ng, nodal_sync
        );
    }
}

void HybridPICModel::BfieldEvolve (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    int step, amrex::Real dt_half, int lev, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    bool use_rkf45 = DoRKF45(step);
    // Make copies of the current B-field multifabs (at t = n) since the
    // starting B-field is needed for the integration logic.
    // We also store the initial B-field from the start of this integration step
    // (i.e., a static copy) in case we need to fully reset the Bfield (needed
    // for RK4).
    std::array< MultiFab, 3 > B_old;
    for (int ii = 0; ii < 3; ii++)
    {
        B_old[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 2, ng
        );
        MultiFab::Copy(B_old[ii], *Bfield[lev][ii], 0, 0, 1, ng);
        // the values at index 1 will be kept static through the integration steps
        MultiFab::Copy(B_old[ii], *Bfield[lev][ii], 0, 1, 1, ng);
    }

    amrex::Real dt_sub = dt_half / (m_substeps / 2._rt);
    amrex::Real t = 0._rt;
    int n_attempts = 0;
    int n_accepted = 0;

    // Step the magnetic field forward (from t -> t + dt_half) using the user
    // specified integration scheme. The loop is set up such that the timestep
    // for a given step (dt_sub) can be modified within the loop, i.e.,
    // adaptive timestepping.
    while (t < dt_half)
    {
        // Adjust size of the last substep, so as to land exactly at t+dt_half.
        if (t + dt_sub > dt_half) { dt_sub = dt_half - t; }
        bool step_succeeded = true;
        amrex::Real step_change_factor = 1.0_rt;

        if (use_rkf45) {
            const amrex::Real error = BfieldEvolveRKF45(
                Bfield, Efield, Jfield, rhofield, eb_update_E, B_old,
                dt_sub, lev, subcycling_half, ng, nodal_sync
            );

            step_change_factor = m_substep_safety * std::pow(error + 1.e-10_rt, -0.2_rt);
            step_succeeded = (error <= 1._rt);

        } else {
            BfieldEvolveRK4(
                Bfield, Efield, Jfield, rhofield, eb_update_E, B_old,
                dt_sub, lev, subcycling_half, ng, nodal_sync
            );

            // Check that the B-field does not have nan or inf values
            for (int idim = 0; idim < 3; ++idim) {
                step_succeeded = step_succeeded && Bfield[lev][idim]->is_finite(/*local=*/true);
            }
            amrex::ParallelDescriptor::ReduceBoolAnd(step_succeeded);

            if (!step_succeeded) {
                ablastr::warn_manager::WMRecordWarning(
                    "HybridPIC",
                    "NaN or Inf value encountered in the B-field during RK4 "
                    "substepping. Restarting this step using RKF45.",
                    ablastr::warn_manager::WarnPriority::medium);

                // restart this full step and this time use RKF45
                t = 0._rt;
                n_accepted = 0;
                // reset B_old to original one
                for (int ii = 0; ii < 3; ii++) {
                    MultiFab::Copy(B_old[ii], B_old[ii], 1, 0, 1, ng);
                }
                use_rkf45 = true;
            }
        }

        if (step_succeeded) {
            // update time tracker and accepted steps number
            t += dt_sub;
            ++n_accepted;
            // update B_old to the current Bfield
            for (int ii = 0; ii < 3; ii++) {
                MultiFab::Copy(B_old[ii], *Bfield[lev][ii], 0, 0, 1, ng);
            }
            dt_sub *= std::min(m_substep_max_growth, step_change_factor);
        } else {
            // reset Bfield to B_old before trying the integration again
            for (int ii = 0; ii < 3; ii++) {
                MultiFab::Copy(*Bfield[lev][ii], B_old[ii], 0, 0, 1, ng);
            }
            dt_sub *= std::max(0.1_rt, step_change_factor);
        }

        if (++n_attempts > m_max_substep_attempts) { break; }
    }

    // Adjust the number of substeps. This affects both the next RKF45 or RK4 step.
    // The adjustment is made to jump to more required substeps or slowly decrease
    // if m_substeps is too large (using 95% of the current m_substeps value and
    // 5% of the lower, new value).
    if (m_substeps < 2*n_attempts) {
        m_substeps = 2*n_attempts;
    } else {
        m_substeps = 2 * int(std::ceil(0.475 * m_substeps + 0.05 * n_attempts));
    }

    if (WarpX::GetInstance().Verbose()) {
        amrex::Print() << "B-field update "
            << (subcycling_half == SubcyclingHalf::FirstHalf ? "1st" : "2nd") << " half"
            << ": " << n_accepted << " accepted, "
            << (n_attempts - n_accepted) << " rejected substeps"
            << " (dt_sub_final/dt_half = " << dt_sub / dt_half << ")\n";
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        n_attempts <= m_max_substep_attempts,
        "BfieldEvolve: exceeded max substep attempts;"
        "consider relaxing hybrid_pic_model.substep_rtol/substep_atol."
    );
}

void HybridPICModel::BfieldEvolveRK4 (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    std::array<amrex::MultiFab, 3>& B_old,
    amrex::Real dt, int lev, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    // Create multifabs for each direction to store the Runge-Kutta intermediate terms.
    // Each multifab has 2 components for the different terms that need to be stored.
    std::array< MultiFab, 3 > K;
    for (int ii = 0; ii < 3; ii++)
    {
        K[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 2,
            Bfield[lev][ii]->nGrowVect()
        );
    }

    // The Runge-Kutta scheme begins here.
    // Step 1:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, eb_update_E,
        0.5_rt*dt, subcycling_half, ng, nodal_sync
    );

    // The Bfield is now given by:
    // B_new = B_old + 0.5 * dt * [-curl x E(B_old)] = B_old + 0.5 * dt * K0.
    for (int ii = 0; ii < 3; ii++)
    {
        // Extract 0.5 * dt * K0 for each direction into index 0 of K.
        MultiFab::LinComb(
            K[ii], 1._rt, *Bfield[lev][ii], 0, -1._rt, B_old[ii], 0, 0, 1, ng
        );
    }

    // Step 2:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, eb_update_E,
        0.5_rt*dt, subcycling_half, ng, nodal_sync
    );

    // The Bfield is now given by:
    //   B_new = B_old + 0.5 * dt * K0 + 0.5 * dt * [-curl x E(B_old + 0.5 * dt * K1)]
    //         = B_old + 0.5 * dt * K0 + 0.5 * dt * K1
    //
    // Subtract 0.5 * dt * K0 from the Bfield to get
    //   B_new = B_old + 0.5 * dt * K1.
    // Extract 0.5 * dt * K1 and write into index 1 of K.

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const &Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const &By = Bfield[lev][1]->array(mfi);
        Array4<Real> const &Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const &Kx = K[0].array(mfi);
        Array4<Real> const &Ky = K[1].array(mfi);
        Array4<Real> const &Kz = K[2].array(mfi);
        Array4<Real const> const &Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const &By_old = B_old[1].const_array(mfi);
        Array4<Real const> const &Bz_old = B_old[2].const_array(mfi);

        // Extract tileboxes for which to loop
        Box const& tjx  = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy  = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz  = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);

        amrex::ParallelFor(tjx, tjy, tjz,
            // x calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Bx(i, j, k) -= Kx(i, j, k, 0);
                Kx(i, j, k, 1) = Bx(i, j, k) - Bx_old(i, j, k);
            },

            // y calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                By(i, j, k) -= Ky(i, j, k, 0);
                Ky(i, j, k, 1) = By(i, j, k) - By_old(i, j, k);
            },

            // z calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Bz(i, j, k) -= Kz(i, j, k, 0);
                Kz(i, j, k, 1) = Bz(i, j, k) - Bz_old(i, j, k);
            }
        );
    }

    // Step 3:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, eb_update_E,
        dt, subcycling_half, ng, nodal_sync
    );

    // The Bfield is now given by:
    // B_new = B_old + 0.5 * dt * K1 + dt * [-curl  x E(B_old + 0.5 * dt * K1)]
    //       = B_old + 0.5 * dt * K1 + dt * K2
    for (int ii = 0; ii < 3; ii++)
    {
        // Subtract 0.5 * dt * K1 from the Bfield for each direction to get
        // B_new = B_old + dt * K2.
        MultiFab::Subtract(*Bfield[lev][ii], K[ii], 1, 0, 1, ng);
    }

    // Step 4:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, eb_update_E,
        0.5_rt*dt, subcycling_half, ng, nodal_sync
    );

    // The Bfield is now given by:
    //   B_new = B_old + dt * K2 + 0.5 * dt * [-curl x E(B_old + dt * K2)]
    //         = B_old + dt * K2 + 0.5 * dt * K3
    // and
    //   index 0 of K = 0.5 * dt * K0
    //   index 1 of K = 0.5 * dt * K1
    //
    // We calculate:
    //   K = 0.5 * dt * K0 + dt * K1 + dt * K2 + 0.5 * dt * K3
    // then update B with the Runge-Kutta sum:
    //   B = B_old + 1/3 * K

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const &Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const &By = Bfield[lev][1]->array(mfi);
        Array4<Real> const &Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const &Kx = K[0].array(mfi);
        Array4<Real> const &Ky = K[1].array(mfi);
        Array4<Real> const &Kz = K[2].array(mfi);
        Array4<Real const> const &Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const &By_old = B_old[1].const_array(mfi);
        Array4<Real const> const &Bz_old = B_old[2].const_array(mfi);

        // Extract tileboxes for which to loop
        Box const& tjx  = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy  = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz  = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);

        amrex::ParallelFor(tjx, tjy, tjz,
            // Bx calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Kx(i, j, k, 0) += Bx(i, j, k) - Bx_old(i, j, k) + 2.0_rt * Kx(i, j, k, 1);
                Bx(i, j, k) = Bx_old(i, j, k) + Kx(i, j, k, 0) / 3.0_rt;
            },

            // By calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Ky(i, j, k, 0) += By(i, j, k) - By_old(i, j, k) + 2.0_rt * Ky(i, j, k, 1);
                By(i, j, k) = By_old(i, j, k) + Ky(i, j, k, 0) / 3.0_rt;
            },

            // Bz calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Kz(i, j, k, 0) += Bz(i, j, k) - Bz_old(i, j, k) + 2.0_rt * Kz(i, j, k, 1);
                Bz(i, j, k) = Bz_old(i, j, k) + Kz(i, j, k, 0) / 3.0_rt;
            }
        );
    }
}

amrex::Real HybridPICModel::BfieldEvolveRKF45 (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    std::array<amrex::MultiFab, 3>& B_old,
    amrex::Real dt, int lev, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    // Fehlberg RKF45 Butcher tableau coefficients
    constexpr amrex::Real a21 = 1._rt/4._rt;
    constexpr amrex::Real a31 = 3._rt/32._rt,      a32 = 9._rt/32._rt;
    constexpr amrex::Real a41 = 1932._rt/2197._rt,  a42 = -7200._rt/2197._rt, a43 = 7296._rt/2197._rt;
    constexpr amrex::Real a51 = 439._rt/216._rt,    a52 = -8._rt,
                          a53 = 3680._rt/513._rt,    a54 = -845._rt/4104._rt;
    constexpr amrex::Real a61 = -8._rt/27._rt,      a62 = 2._rt,
                          a63 = -3544._rt/2565._rt,  a64 = 1859._rt/4104._rt,  a65 = -11._rt/40._rt;
    // 4th-order solution weights (k2 and k6 terms are zero in Fehlberg's formula)
    constexpr amrex::Real b1 = 25._rt/216._rt,  b3 = 1408._rt/2565._rt,
                          b4 = 2197._rt/4104._rt, b5 = -1._rt/5._rt;
    // Error = B5 - B4 weights: h*(e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6)
    constexpr amrex::Real e1 =  1._rt/360._rt,    e3 = -128._rt/4275._rt,
                          e4 = -2197._rt/75240._rt, e5 = 1._rt/50._rt, e6 = 2._rt/55._rt;

    // K: 5 components per field direction stored as:
    //   comp 0 = h*k1, comp 1 = h*k2 (overwritten with h*k6 after stage 6),
    //   comp 2 = h*k3, comp 3 = h*k4, comp 4 = h*k5
    std::array<MultiFab, 3> K;
    std::array<MultiFab, 3> err_scratch;
    for (int ii = 0; ii < 3; ii++)
    {
        K[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 5,
            Bfield[lev][ii]->nGrowVect()
        );
        err_scratch[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 1,
            amrex::IntVect(0)
        );
    }

    // ---- Stage 1: B = B_old, FieldPush, K[comp0] = h*k1 fused with Stage 2 B-update ----
    FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                dt, subcycling_half, ng, nodal_sync);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const& By = Bfield[lev][1]->array(mfi);
        Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const& Kx = K[0].array(mfi);
        Array4<Real> const& Ky = K[1].array(mfi);
        Array4<Real> const& Kz = K[2].array(mfi);
        Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const& By_old = B_old[1].const_array(mfi);
        Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
        Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
        amrex::ParallelFor(tjx, tjy, tjz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Bx(i, j, k) - Bx_old(i, j, k);
                Kx(i, j, k, 0) = k1;
                Bx(i, j, k) = Bx_old(i, j, k) + a21*k1;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = By(i, j, k) - By_old(i, j, k);
                Ky(i, j, k, 0) = k1;
                By(i, j, k) = By_old(i, j, k) + a21*k1;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Bz(i, j, k) - Bz_old(i, j, k);
                Kz(i, j, k, 0) = k1;
                Bz(i, j, k) = Bz_old(i, j, k) + a21*k1;
            }
        );
    }

    // ---- Stage 2: FieldPush, K[comp1] = h*k2 fused with Stage 3 B-update ----
    FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                dt, subcycling_half, ng, nodal_sync);
    // Stage 2 K[1]-readback fused with Stage 3 B-update.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const& By = Bfield[lev][1]->array(mfi);
        Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const& Kx = K[0].array(mfi);
        Array4<Real> const& Ky = K[1].array(mfi);
        Array4<Real> const& Kz = K[2].array(mfi);
        Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const& By_old = B_old[1].const_array(mfi);
        Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
        Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
        amrex::ParallelFor(tjx, tjy, tjz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kx(i, j, k, 0);
                amrex::Real const k2 = Bx(i, j, k) - Bx_old(i, j, k) - a21*k1;
                Kx(i, j, k, 1) = k2;
                Bx(i, j, k) = Bx_old(i, j, k) + a31*k1 + a32*k2;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Ky(i, j, k, 0);
                amrex::Real const k2 = By(i, j, k) - By_old(i, j, k) - a21*k1;
                Ky(i, j, k, 1) = k2;
                By(i, j, k) = By_old(i, j, k) + a31*k1 + a32*k2;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kz(i, j, k, 0);
                amrex::Real const k2 = Bz(i, j, k) - Bz_old(i, j, k) - a21*k1;
                Kz(i, j, k, 1) = k2;
                Bz(i, j, k) = Bz_old(i, j, k) + a31*k1 + a32*k2;
            }
        );
    }

    // ---- Stage 3: FieldPush, then K[comp2] = h*k3 fused with Stage 4 B-update ----
    FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                dt, subcycling_half, ng, nodal_sync);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const& By = Bfield[lev][1]->array(mfi);
        Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const& Kx = K[0].array(mfi);
        Array4<Real> const& Ky = K[1].array(mfi);
        Array4<Real> const& Kz = K[2].array(mfi);
        Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const& By_old = B_old[1].const_array(mfi);
        Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
        Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
        amrex::ParallelFor(tjx, tjy, tjz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kx(i, j, k, 0);
                amrex::Real const k2 = Kx(i, j, k, 1);
                amrex::Real const k3 = Bx(i, j, k) - Bx_old(i, j, k) - a31*k1 - a32*k2;
                Kx(i, j, k, 2) = k3;
                Bx(i, j, k) = Bx_old(i, j, k) + a41*k1 + a42*k2 + a43*k3;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Ky(i, j, k, 0);
                amrex::Real const k2 = Ky(i, j, k, 1);
                amrex::Real const k3 = By(i, j, k) - By_old(i, j, k) - a31*k1 - a32*k2;
                Ky(i, j, k, 2) = k3;
                By(i, j, k) = By_old(i, j, k) + a41*k1 + a42*k2 + a43*k3;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kz(i, j, k, 0);
                amrex::Real const k2 = Kz(i, j, k, 1);
                amrex::Real const k3 = Bz(i, j, k) - Bz_old(i, j, k) - a31*k1 - a32*k2;
                Kz(i, j, k, 2) = k3;
                Bz(i, j, k) = Bz_old(i, j, k) + a41*k1 + a42*k2 + a43*k3;
            }
        );
    }

    // ---- Stage 4: FieldPush, then K[comp3] = h*k4 fused with Stage 5 B-update ----
    FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                dt, subcycling_half, ng, nodal_sync);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const& By = Bfield[lev][1]->array(mfi);
        Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const& Kx = K[0].array(mfi);
        Array4<Real> const& Ky = K[1].array(mfi);
        Array4<Real> const& Kz = K[2].array(mfi);
        Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const& By_old = B_old[1].const_array(mfi);
        Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
        Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
        amrex::ParallelFor(tjx, tjy, tjz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kx(i, j, k, 0);
                amrex::Real const k2 = Kx(i, j, k, 1);
                amrex::Real const k3 = Kx(i, j, k, 2);
                amrex::Real const k4 = Bx(i, j, k) - Bx_old(i, j, k)
                                        - a41*k1 - a42*k2 - a43*k3;
                Kx(i, j, k, 3) = k4;
                Bx(i, j, k) = Bx_old(i, j, k) + a51*k1 + a52*k2 + a53*k3 + a54*k4;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Ky(i, j, k, 0);
                amrex::Real const k2 = Ky(i, j, k, 1);
                amrex::Real const k3 = Ky(i, j, k, 2);
                amrex::Real const k4 = By(i, j, k) - By_old(i, j, k)
                                        - a41*k1 - a42*k2 - a43*k3;
                Ky(i, j, k, 3) = k4;
                By(i, j, k) = By_old(i, j, k) + a51*k1 + a52*k2 + a53*k3 + a54*k4;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kz(i, j, k, 0);
                amrex::Real const k2 = Kz(i, j, k, 1);
                amrex::Real const k3 = Kz(i, j, k, 2);
                amrex::Real const k4 = Bz(i, j, k) - Bz_old(i, j, k)
                                        - a41*k1 - a42*k2 - a43*k3;
                Kz(i, j, k, 3) = k4;
                Bz(i, j, k) = Bz_old(i, j, k) + a51*k1 + a52*k2 + a53*k3 + a54*k4;
            }
        );
    }

    // ---- Stage 5: FieldPush, then K[comp4] = h*k5 fused with Stage 6 B-update ----
    FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                dt, subcycling_half, ng, nodal_sync);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const& By = Bfield[lev][1]->array(mfi);
        Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const& Kx = K[0].array(mfi);
        Array4<Real> const& Ky = K[1].array(mfi);
        Array4<Real> const& Kz = K[2].array(mfi);
        Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const& By_old = B_old[1].const_array(mfi);
        Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
        Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
        amrex::ParallelFor(tjx, tjy, tjz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kx(i, j, k, 0);
                amrex::Real const k2 = Kx(i, j, k, 1);
                amrex::Real const k3 = Kx(i, j, k, 2);
                amrex::Real const k4 = Kx(i, j, k, 3);
                amrex::Real const k5 = Bx(i, j, k) - Bx_old(i, j, k)
                                        - a51*k1 - a52*k2 - a53*k3 - a54*k4;
                Kx(i, j, k, 4) = k5;
                Bx(i, j, k) = Bx_old(i, j, k)
                            + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Ky(i, j, k, 0);
                amrex::Real const k2 = Ky(i, j, k, 1);
                amrex::Real const k3 = Ky(i, j, k, 2);
                amrex::Real const k4 = Ky(i, j, k, 3);
                amrex::Real const k5 = By(i, j, k) - By_old(i, j, k)
                                        - a51*k1 - a52*k2 - a53*k3 - a54*k4;
                Ky(i, j, k, 4) = k5;
                By(i, j, k) = By_old(i, j, k)
                            + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kz(i, j, k, 0);
                amrex::Real const k2 = Kz(i, j, k, 1);
                amrex::Real const k3 = Kz(i, j, k, 2);
                amrex::Real const k4 = Kz(i, j, k, 3);
                amrex::Real const k5 = Bz(i, j, k) - Bz_old(i, j, k)
                                        - a51*k1 - a52*k2 - a53*k3 - a54*k4;
                Kz(i, j, k, 4) = k5;
                Bz(i, j, k) = Bz_old(i, j, k)
                            + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5;
            }
        );
    }

    // ---- Stage 6: FieldPush, then K[comp1] = h*k6 (overwrites h*k2) fused with B4 + error ----
    FieldPush(Bfield, Efield, Jfield, rhofield, eb_update_E,
                dt, subcycling_half, ng, nodal_sync);
    // K[comp1] is overwritten here: reads h*k2 (old value) then writes h*k6 in each cell.
    // k6, B4 assembly (b2=0, so k2 is not needed for B4), and error assembly are fused into
    // one ParallelFor per direction. B4 is updated over ghost+valid cells; error is written
    // only for valid cells (err_scratch has no ghost), guarded by a box check.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
        Array4<Real> const& By = Bfield[lev][1]->array(mfi);
        Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
        Array4<Real> const& Kx = K[0].array(mfi);
        Array4<Real> const& Ky = K[1].array(mfi);
        Array4<Real> const& Kz = K[2].array(mfi);
        Array4<Real> const& error_x = err_scratch[0].array(mfi);
        Array4<Real> const& error_y = err_scratch[1].array(mfi);
        Array4<Real> const& error_z = err_scratch[2].array(mfi);
        Array4<Real const> const& Bx_old = B_old[0].const_array(mfi);
        Array4<Real const> const& By_old = B_old[1].const_array(mfi);
        Array4<Real const> const& Bz_old = B_old[2].const_array(mfi);
        Box const& tjx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect());
        Box const& tjy = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect());
        Box const& tjz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect());
        Box const& tjx_ng = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect(), ng);
        Box const& tjy_ng = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect(), ng);
        Box const& tjz_ng = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect(), ng);
        amrex::ParallelFor(tjx_ng, tjy_ng, tjz_ng,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kx(i, j, k, 0);
                amrex::Real const k2 = Kx(i, j, k, 1);
                amrex::Real const k3 = Kx(i, j, k, 2);
                amrex::Real const k4 = Kx(i, j, k, 3);
                amrex::Real const k5 = Kx(i, j, k, 4);
                amrex::Real const k6 = Bx(i, j, k) - Bx_old(i, j, k)
                                        - a61*k1 - a62*k2 - a63*k3 - a64*k4 - a65*k5;
                Kx(i, j, k, 1) = k6;
                Bx(i, j, k) = Bx_old(i, j, k) + b1*k1 + b3*k3 + b4*k4 + b5*k5;
                if (tjx.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
                    error_x(i, j, k) = e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6;
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Ky(i, j, k, 0);
                amrex::Real const k2 = Ky(i, j, k, 1);
                amrex::Real const k3 = Ky(i, j, k, 2);
                amrex::Real const k4 = Ky(i, j, k, 3);
                amrex::Real const k5 = Ky(i, j, k, 4);
                amrex::Real const k6 = By(i, j, k) - By_old(i, j, k)
                                        - a61*k1 - a62*k2 - a63*k3 - a64*k4 - a65*k5;
                Ky(i, j, k, 1) = k6;
                By(i, j, k) = By_old(i, j, k) + b1*k1 + b3*k3 + b4*k4 + b5*k5;
                if (tjy.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
                    error_y(i, j, k) = e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6;
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                amrex::Real const k1 = Kz(i, j, k, 0);
                amrex::Real const k2 = Kz(i, j, k, 1);
                amrex::Real const k3 = Kz(i, j, k, 2);
                amrex::Real const k4 = Kz(i, j, k, 3);
                amrex::Real const k5 = Kz(i, j, k, 4);
                amrex::Real const k6 = Bz(i, j, k) - Bz_old(i, j, k)
                                        - a61*k1 - a62*k2 - a63*k3 - a64*k4 - a65*k5;
                Kz(i, j, k, 1) = k6;
                Bz(i, j, k) = Bz_old(i, j, k) + b1*k1 + b3*k3 + b4*k4 + b5*k5;
                if (tjz.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
                    error_z(i, j, k) = e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6;
                }
            }
        );
    }

    // ---- Error norm and adaptive step control ----
    // Compute local maxima first, then one combined AllReduce for both norms.
    amrex::Real err_norm = 0._rt;
    amrex::Real B4_norm  = 0._rt;
    for (int ii = 0; ii < 3; ii++) {
        err_norm = std::max(err_norm, err_scratch[ii].norm0(/*comp=*/0, /*nghost=*/0, /*local=*/true));
        B4_norm  = std::max(B4_norm,  Bfield[lev][ii]->norm0(/*comp=*/0, /*nghost=*/0, /*local=*/true));
    }
    amrex::ParallelDescriptor::ReduceRealMax({err_norm, B4_norm});
    return err_norm / (m_substep_atol + m_substep_rtol * B4_norm);
}


void HybridPICModel::FieldPush (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelVectorField const& Jfield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
    amrex::Real dt, SubcyclingHalf subcycling_half,
    IntVect ng, std::optional<bool> nodal_sync )
{
    auto& warpx = WarpX::GetInstance();

    amrex::Real const t_old = warpx.gett_old(0);

    // Calculate J = curl x B / mu0 - J_ext
    CalculatePlasmaCurrent(Bfield, eb_update_E);
    // Calculate the E-field from Ohm's law
    HybridPICSolveE(Efield, Jfield, Bfield, rhofield, eb_update_E, true);
    // Call FillBoundary if a collocated grid is used
    if (Bz_IndexType[0] == Ez_IndexType[0]) {
        warpx.FillBoundaryE(ng, nodal_sync);
    }

    // Push forward the B-field using Faraday's law
    warpx.EvolveB(dt, subcycling_half, t_old);
    warpx.FillBoundaryB(ng, nodal_sync);
}
