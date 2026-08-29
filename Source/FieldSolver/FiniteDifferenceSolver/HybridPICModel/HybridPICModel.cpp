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

#include <AMReX_MLCurlCurl.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLNodeABecLaplacian.H>
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

#include <cmath>
#include <limits>
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
    utils::parser::queryWithParser(pp_hybrid, "holmstrom_transition_width", m_holmstrom_transition_width);
    pp_hybrid.query("include_hall_term", m_include_hall_term);
    pp_hybrid.query("include_electron_pressure_term", m_include_electron_pressure_term);
    pp_hybrid.query("pec_conductor_wall_rows", m_pec_conductor_wall_rows);

    // Thin resistive-shell wall at the RZ r-max PEC boundary (see the
    // member documentation of m_resistive_wall for the model and the
    // shell L/R timescale). Independent of pec_conductor_wall_rows.
    pp_hybrid.query("resistive_wall", m_resistive_wall);
    utils::parser::queryWithParser(pp_hybrid, "wall_resistivity",
                                   m_wall_resistivity);
    utils::parser::queryWithParser(pp_hybrid, "wall_thickness",
                                   m_wall_thickness);
    utils::parser::queryWithParser(pp_hybrid, "wall_bz_ext", m_wall_bz_ext);
#if !defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_resistive_wall,
        "hybrid_pic_model.resistive_wall is only implemented for the RZ "
        "r-max wall");
#endif
    if (m_resistive_wall) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_wall_resistivity > 0.,
            "hybrid_pic_model.wall_resistivity must be > 0");
    }

    // Opt-in conformal (enlarged-cell/ECT) embedded-boundary wall.
    pp_hybrid.query("use_conformal_eb", m_use_conformal_eb);
    if (m_use_conformal_eb) {
#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_XZ)
        WARPX_ABORT_WITH_MESSAGE(
            "hybrid_pic_model.use_conformal_eb is only supported in 3D and 2D (XZ) "
            "Cartesian geometry");
#endif
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(EB::enabled(),
            "hybrid_pic_model.use_conformal_eb requires embedded boundaries to be enabled");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::grid_type == ablastr::utils::enums::GridType::Staggered,
            "hybrid_pic_model.use_conformal_eb requires warpx.grid_type = staggered");
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                WarpX::field_boundary_lo[idim] != FieldBoundaryType::PML &&
                WarpX::field_boundary_hi[idim] != FieldBoundaryType::PML,
                "hybrid_pic_model.use_conformal_eb is not compatible with PML boundaries");
        }
        pp_hybrid.query("conformal_wall_model", m_conformal_wall_model);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_conformal_wall_model == "conductor"
                || m_conformal_wall_model == "transparent",
            "hybrid_pic_model.conformal_wall_model must be 'conductor' or "
            "'transparent'");
        m_conformal_wall_conductor = (m_conformal_wall_model == "conductor");
    }

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
    pp_hybrid.query("joule_heating_resistivity(rho,J,t)",
                    m_eta_heating_expression);
    utils::parser::queryWithParser(pp_hybrid, "joule_heating_n_min",
                                   m_joule_heating_n_min);
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
    pp_hybrid.query("hyper_resistivity_curlcurl", m_hyper_resistivity_curlcurl);

    utils::parser::queryWithParser(pp_hybrid, "n_floor", m_n_floor);
    utils::parser::queryWithParser(pp_hybrid, "n_floor_smooth_width",
                                   m_n_floor_smooth_width);
    utils::parser::queryWithParser(pp_hybrid, "electron_inertia_floor_taper",
                                   m_electron_inertia_floor_taper);
    pp_hybrid.query("electron_inertia_extrapolated_history",
                    m_electron_inertia_extrapolated_history);
    pp_hybrid.query("electron_inertia_djedt_only",
                    m_electron_inertia_djedt_only);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_n_floor_smooth_width >= 0.,
        "hybrid_pic_model.n_floor_smooth_width must be >= 0");

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

    // E-solve form: "e_form" (default) = the legacy algebraic solve with
    // the floored density division; "curlcurl_form" = the division-free
    // multiplied-through solve (Amano, J. Comput. Phys. 275, 197 (2014);
    // Hewett & Nielson, J. Comput. Phys. 29, 219 (1978)) in which the
    // dJe/dt leg's dependence on E becomes the elliptic operator
    // (beta + curl curl) E = beta-scaled numerator, with beta =
    // mu0 e^2 n / m_e_eff UNfloored -- no density pedestal anywhere in
    // the E assembly. Currently the toroidal (E_theta) sector in RZ m=0
    // (a scalar nodal Helmholtz; the poloidal stream-scalar and E_L
    // recovery stages follow). Requires the theta-implicit hybrid scheme
    // (the elliptic solve is amortized over the large implicit step; the
    // explicit advance never runs elliptic solves) and electron inertia
    // in the djedt_only form (whose derivation this solve shares with
    // the pc_curl_curl_mlmg hybrid mode).
    {
        std::string esolve = "e_form";
        pp_hybrid.query("esolve", esolve);
        if (esolve == "curlcurl_form") { m_esolve_curlcurl = true; }
        else if (esolve == "tensor_form") { m_esolve_tensor = true; }
        else if (esolve == "amano_form") {
            // backward-compatibility guard: the value was retired, not
            // aliased -- decks must name the operator identity explicitly
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.esolve = amano_form was renamed to "
                "curlcurl_form; update the input deck");
        }
        else if (esolve != "e_form") {
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.esolve must be 'e_form', "
                "'curlcurl_form' or 'tensor_form'");
        }
        if (m_esolve_tensor) {
#if !defined(WARPX_DIM_1D_Z) && !defined(WARPX_DIM_RZ)
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.esolve = tensor_form supports 1D_Z and "
                "RZ (m = 0) only");
#endif
            std::string scheme;
            const amrex::ParmParse pp_algo("algo");
            pp_algo.query("evolve_scheme", scheme);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                scheme == "theta_implicit_hybrid",
                "hybrid_pic_model.esolve = tensor_form requires "
                "algo.evolve_scheme = theta_implicit_hybrid (the elliptic "
                "solve is amortized over the large implicit step)");
            utils::parser::queryWithParser(
                pp_hybrid, "tensor_mass_alpha", m_tensor_alpha);
            utils::parser::queryWithParser(
                pp_hybrid, "tensor_n_min", m_tensor_n_min);
            pp_hybrid.query("tensor_verify", m_esolve_tensor_verify);
        }
        if (m_esolve_curlcurl) {
#if !defined(WARPX_DIM_RZ)
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.esolve = curlcurl_form is currently "
                "RZ-only (the toroidal m = 0 sector)");
#endif
            // One-shot discrete verification of the poloidal curl-curl
            // assembly (see SolveEPolCurlCurlRZ).
            pp_hybrid.query("esolve_pol_verify", m_esolve_pol_verify);
            // Per-step-frozen density for the poloidal stage's operator
            // content and gates (see the member documentation; default on
            // -- the iron rule: residual-consumed binary decisions must be
            // a pure function of the iterate or bit-frozen per step).
            pp_hybrid.query("curlcurl_pol_frozen_rho",
                            m_curlcurl_pol_frozen_rho);
            // One-count membership anchor of the curlcurl_form gates
            // (see the member documentation; 0 = legacy strictly-zero
            // gating, bit-identical).
            utils::parser::queryWithParser(
                pp_hybrid, "curlcurl_n_min", m_curlcurl_n_min);
            std::string scheme;
            const amrex::ParmParse pp_algo("algo");
            pp_algo.query("evolve_scheme", scheme);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                scheme == "theta_implicit_hybrid",
                "hybrid_pic_model.esolve = curlcurl_form requires "
                "algo.evolve_scheme = theta_implicit_hybrid (the elliptic "
                "solve is amortized over the large implicit step; the "
                "explicit advance does not run elliptic solves)");
            std::string pc_type;
            const amrex::ParmParse pp_jac("jacobian");
            pp_jac.query("pc_type", pc_type);
            if (pc_type == "pc_block_banded") {
                ablastr::warn_manager::WMRecordWarning(
                    "HybridPICModel",
                    "jacobian.pc_type = pc_block_banded assembles the e_form "
                    "(divided) Jacobian rows, which do not model the "
                    "curlcurl_form residual: measured convergence is "
                    "identical to running with no preconditioner while "
                    "paying the factor/apply cost. Prefer pc_type = none "
                    "with esolve = curlcurl_form until a curlcurl-aware "
                    "variant lands.",
                    ablastr::warn_manager::WarnPriority::high);
            }
        }
    }

    // Electron inertia in the generalized Ohm's law (see the member
    // documentation in HybridPICModel.H).
    pp_hybrid.query("include_electron_inertia", m_include_electron_inertia);
    if (m_include_electron_inertia) {
        utils::parser::queryWithParser(
            pp_hybrid, "reduced_electron_mass_ratio",
            m_reduced_electron_mass_ratio);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_reduced_electron_mass_ratio >= 0.0,
            "hybrid_pic_model.reduced_electron_mass_ratio must be >= 0 "
            "(0 selects the physical electron mass)");
        pp_hybrid.query("electron_inertia_bdf2", m_electron_inertia_bdf2);
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        WARPX_ABORT_WITH_MESSAGE(
            "hybrid_pic_model.include_electron_inertia is not supported in "
            "the 1D radial geometries.");
#endif
    }
    if (m_esolve_curlcurl) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_include_electron_inertia && m_electron_inertia_djedt_only
                && !m_electron_inertia_bdf2,
            "hybrid_pic_model.esolve = curlcurl_form requires "
            "include_electron_inertia = 1 with electron_inertia_djedt_only "
            "= 1 and electron_inertia_bdf2 = 0 (the dJe/dt leg IS the "
            "elliptic operator; the derivation is shared with the "
            "pc_curl_curl_mlmg hybrid mode)");
    }
    if (m_esolve_tensor) {
        // The closed-form Je elimination IS the electron-inertia treatment
        // (the dJe/dt leg becomes the curl-curl operator, the gyration and
        // friction the pointwise tensor); the additive inertia term would
        // double-count it.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_include_electron_inertia,
            "hybrid_pic_model.esolve = tensor_form eliminates the electron "
            "momentum equation in closed form; set include_electron_inertia "
            "= 0 (the additive inertia term would double-count the dJe/dt "
            "leg)");
        utils::parser::queryWithParser(
            pp_hybrid, "reduced_electron_mass_ratio",
            m_reduced_electron_mass_ratio);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_reduced_electron_mass_ratio >= 0.0,
            "hybrid_pic_model.reduced_electron_mass_ratio must be >= 0 "
            "(0 selects the physical electron mass)");
    }

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
            utils::parser::queryWithParser(
                pp_hybrid, "darwin_vacuum_recovery_relaxation_time",
                m_darwin_vacrec_relax_time);
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
            pp_hybrid.query("darwin_vacuum_recovery_frozen_mask",
                            m_darwin_vacuum_recovery_frozen_mask);
            pp_hybrid.query("darwin_vacuum_recovery_operator",
                            m_darwin_vacuum_recovery_operator);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_darwin_vacuum_recovery_operator == "poisson"
                || m_darwin_vacuum_recovery_operator == "curlcurl",
                "hybrid_pic_model.darwin_vacuum_recovery_operator must be "
                "'poisson' or 'curlcurl'");
            utils::parser::queryWithParser(
                pp_hybrid, "darwin_vacuum_recovery_curlcurl_beta",
                m_darwin_vacrec_cc_beta_rel);
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_darwin_vacuum_recovery_operator != "curlcurl",
                "hybrid_pic_model.darwin_vacuum_recovery_operator = "
                "curlcurl requires Cartesian geometry (amrex::MLCurlCurl "
                "carries no cylindrical metric; the RZ recovery stays on "
                "the fixed-point-exact 'poisson' map).");
#endif
            // Live-probe mode (see the member documentation): env-gated,
            // and forced on by circuit-in-the-residual solvers.
            m_vacuum_recovery_live_probes =
                (std::getenv("WARPX_VACREC_LIVE_PROBES") != nullptr);
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
                // Edge-staggered persistent correction for the "curlcurl"
                // recovery operator (warm start across evaluations).
                if (m_darwin_vacuum_recovery_operator == "curlcurl") {
                    fields.alloc_init("hybrid_A_vac_dA_edge", Direction{dir},
                        lev, amrex::convert(ba, E_stag[dir]), dm, ncomps,
                        ngEB, 0.0_rt);
                }
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
            // Frozen mask density (see the member documentation): the
            // per-step snapshot of the committed entry rho the recovery
            // and Faraday-overwrite masks read instead of the live
            // (iterate-dependent) rho_fp component 0.
            if (m_darwin_vacuum_recovery_frozen_mask) {
                fields.alloc_init("hybrid_rho_vacmask_fp",
                    lev, amrex::convert(ba, rho_nodal_flag), dm, 1, ngRho,
                    0.0_rt);
            }
        }
    }

    // Electron inertia: nodal 3-component Je history (per-step levels;
    // zero init = the exact pre-history for quiescent starts) and the
    // per-evaluation nodal scratch for the assembled inertial field.
    if (m_include_electron_inertia) {
        // One ghost cell throughout: the histories are read at valid nodes
        // only, the assembled field is consumed by valid-contained Interp
        // stencils, and the theta-save copies exactly this width -- wider
        // ghosts would never be maintained.
        const amrex::IntVect nd = amrex::IntVect::TheNodeVector();
        const amrex::IntVect ng1 = amrex::IntVect(1);
        fields.alloc_init("hybrid_Je_n_nodal",
            lev, amrex::convert(ba, nd), dm, 3, ng1, 0.0_rt);
        fields.alloc_init("hybrid_Je_nm1_nodal",
            lev, amrex::convert(ba, nd), dm, 3, ng1, 0.0_rt);
        fields.alloc_init("hybrid_E_inertial_nodal",
            lev, amrex::convert(ba, nd), dm, 3, ng1, 0.0_rt);
        fields.alloc_init("hybrid_Je_theta_nodal",
            lev, amrex::convert(ba, nd), dm, 3, ng1, 0.0_rt);
        // Step-start density frozen for the drho/dt leg: during the
        // nonlinear iteration rho_fp component 0 is a midpoint-position
        // deposit (only the first evaluation of a step sees rho(x^n)
        // there), so the rate term needs its own per-step-frozen copy --
        // captured in ComputeElectronInertiaNodal on the same deposit
        // family the midpoint component comes from. One mode: the inertia
        // term asserts m = 0 in RZ.
        fields.alloc_init("hybrid_rho_n_frozen",
            lev, amrex::convert(ba, rho_nodal_flag), dm, 1, ngRho, 0.0_rt);
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

    // Exact-operator hyper-resistivity scratch (m_hyper_resistivity_curlcurl):
    // curl J on the B (face) staggering and the composed
    // E_H = curl(eta_H curl J) back on the J (edge) staggering, recomputed
    // per Ohm-solve call. Gate on the expression, not
    // m_include_hyper_resistivity_term: that flag is derived in InitData,
    // after this allocation runs.
    if ((m_eta_h_expression != "0.0") && m_hyper_resistivity_curlcurl) {
        const amrex::IntVect B_stag_hr[3] =
            {Bx_nodal_flag, By_nodal_flag, Bz_nodal_flag};
        const amrex::IntVect J_stag_hr[3] =
            {jx_nodal_flag, jy_nodal_flag, jz_nodal_flag};
        for (int dir = 0; dir < 3; ++dir) {
            fields.alloc_init("hybrid_hyperres_curlJ_fp", Direction{dir},
                lev, amrex::convert(ba, B_stag_hr[dir]), dm, ncomps, ngJ,
                0.0_rt);
            fields.alloc_init("hybrid_hyperres_E_fp", Direction{dir},
                lev, amrex::convert(ba, J_stag_hr[dir]), dm, ncomps, ngJ,
                0.0_rt);
        }
    }

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

    // Joule-heating resistivity split (see the member doc): unset
    // aliases the E-solve executor exactly -- the default path is
    // bit-identical, including any J/t dependence.
    if (m_eta_heating_expression.empty()) {
        m_eta_heating = m_eta;
    } else {
        m_heating_resistivity_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_eta_heating_expression,
                                      {"rho","J","t"}));
        m_eta_heating = m_heating_resistivity_parser->compile<3>();
    }

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

void HybridPICModel::ZeroConductorEdges (
    ablastr::fields::VectorField const& field,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update,
    const int lev) const
{
    // Constitutive PEC on the conformal (ECT) wall: zero every masked (fully
    // covered) edge and every cut edge (open length < full). There is no wall
    // physics on the zeroed set -- E = 0 in a perfect conductor and the
    // hybrid carries no surface currents.
    auto& warpx = WarpX::GetInstance();
    const auto dx = warpx.Geom(lev).CellSizeArray();
    const ablastr::fields::VectorField edge_lengths =
        warpx.m_fields.get_alldirs(FieldType::edge_lengths, lev);
    // Full edges carry exactly their cell size after ScaleEdges; anything
    // shorter is cut. The tolerance absorbs EB-geometry round-off.
    amrex::GpuArray<amrex::Real, 3> l_full{0.0_rt, 0.0_rt, 0.0_rt};
    for (int d = 0; d < 3; ++d) {
        const int gd = amrex::min(d, AMREX_SPACEDIM - 1);
        l_full[d] = (1.0_rt - 1.e-6_rt) * dx[gd];
    }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*field[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Array4<amrex::Real> const& Fx = field[0]->array(mfi);
        amrex::Array4<amrex::Real> const& Fy = field[1]->array(mfi);
        amrex::Array4<amrex::Real> const& Fz = field[2]->array(mfi);
        amrex::Array4<amrex::Real const> const& lx = edge_lengths[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& ly = edge_lengths[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& lz = edge_lengths[2]->const_array(mfi);
        amrex::Array4<int const> const& ux = eb_update[0]->const_array(mfi);
        amrex::Array4<int const> const& uy = eb_update[1]->const_array(mfi);
        amrex::Array4<int const> const& uz = eb_update[2]->const_array(mfi);
        const amrex::Box tx = mfi.tilebox(field[0]->ixType().toIntVect());
        const amrex::Box ty = mfi.tilebox(field[1]->ixType().toIntVect());
        const amrex::Box tz = mfi.tilebox(field[2]->ixType().toIntVect());
        amrex::ParallelFor(tx, ty, tz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                if (ux(i,j,k) == 0 || lx(i,j,k) < l_full[0]) { Fx(i,j,k) = 0.0_rt; }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                if (uy(i,j,k) == 0 || ly(i,j,k) < l_full[1]) { Fy(i,j,k) = 0.0_rt; }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                if (uz(i,j,k) == 0 || lz(i,j,k) < l_full[2]) { Fz(i,j,k) = 0.0_rt; }
            });
    }
}

void HybridPICModel::ZeroCoveredFaces (
    ablastr::fields::VectorField const& field,
    const int lev) const
{
    // Zero fully covered faces (open area = 0); cut faces keep their value,
    // since their open part carries real flux.
    auto& warpx = WarpX::GetInstance();
    const ablastr::fields::VectorField face_areas =
        warpx.m_fields.get_alldirs(FieldType::face_areas, lev);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*field[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Array4<amrex::Real> const& Fx = field[0]->array(mfi);
        amrex::Array4<amrex::Real> const& Fy = field[1]->array(mfi);
        amrex::Array4<amrex::Real> const& Fz = field[2]->array(mfi);
        amrex::Array4<amrex::Real const> const& Sx = face_areas[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Sy = face_areas[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Sz = face_areas[2]->const_array(mfi);
        const amrex::Box tx = mfi.tilebox(field[0]->ixType().toIntVect());
        const amrex::Box ty = mfi.tilebox(field[1]->ixType().toIntVect());
        const amrex::Box tz = mfi.tilebox(field[2]->ixType().toIntVect());
        amrex::ParallelFor(tx, ty, tz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                if (Sx(i,j,k) <= 0.0_rt) { Fx(i,j,k) = 0.0_rt; }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                if (Sy(i,j,k) <= 0.0_rt) { Fy(i,j,k) = 0.0_rt; }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                if (Sz(i,j,k) <= 0.0_rt) { Fz(i,j,k) = 0.0_rt; }
            });
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

    // Conformal wall, constitutive PEC: the hybrid carries no wall (surface)
    // currents, so J = 0 on every covered and cut edge; cut faces evolve only
    // through their fully-open edges.
    if (EB::enabled() && m_use_conformal_eb && m_conformal_wall_conductor) {
        ZeroConductorEdges(current_fp_plasma, eb_update_E, lev);
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

#if defined(WARPX_DIM_RZ)
    // Conductor-row treatment of the r-max PEC wall: the conductor row
    // carries no plasma moments (see the m_pec_conductor_wall_rows member
    // documentation for the full closure stack). Zero the tangential
    // current components on the wall node plane (and beyond) for BOTH the
    // Ampere and ion currents, so the electron current J_e = J - J_i the
    // Ohm kernels assemble carries no wall sheet from the deposit-fold /
    // one-sided-curl parity mismatch. Conductor-surface current lives in
    // the conductor, not in the plasma Ohm terms.
    //
    // Ordering: this block runs inside every residual evaluation of the
    // theta-implicit solver, strictly AFTER the moment deposition and its
    // guard folds (ThetaImplicitHybrid::ComputeRHS -> PreRHSOp:
    // PushParticlesandDeposit -> ApplyInverseVolumeScalingTo{Current,
    // Charge}Density -> SyncCurrentAndRho, whose SyncRho/SyncCurrent
    // SumBoundary and ApplyRhofieldBoundary/ApplyJfieldBoundary PEC
    // reflect-folds are the last writers of the wall rows) and immediately
    // BEFORE the Ohm kernels consume the moments -- so no fold can
    // re-populate the conductor row behind this treatment. The explicit
    // hybrid's subcycled E-solves route through this same wrapper.
    if (m_pec_conductor_wall_rows
        && WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC)
    {
        const int iwall = warpx.Geom(lev).Domain().bigEnd(0) + 1;
        for (auto* J : {current_fp_plasma[1], current_fp_plasma[2],
                        Jfield[1], Jfield[2]}) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(*J, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                amrex::Box bx = mfi.growntilebox();
                if (bx.bigEnd(0) < iwall) { continue; }
                bx.setSmall(0, iwall);
                auto const& arr = J->array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    arr(i, j, k) = 0.0_rt;
                });
            }
        }

        // ne-Dirichlet wall moment: the conductor row carries no plasma
        // charge either. Zero the deposited (nodal) charge density the Ohm
        // divide consumes (comp 0 of the rhofield handed to the kernels) on
        // the wall node plane, and odd-image the ghost rows beyond it
        // (ghost = -interior mirror), so any staggered interpolation at or
        // beyond the wall row sees a raw density <= 0 and the gate-on-raw
        // divide in the Ohm kernel returns zero Hall/motional force there
        // instead of enE/rho_floor (the floored divide alone would turn the
        // zeroed numerator row into a 1/rho_floor-amplified response).
        // The deposit is re-made in every residual evaluation, so this
        // mutation never leaks across evaluations; the const_cast marks the
        // wall-moment boundary-condition application point on a field the
        // kernels otherwise consume read-only.
        auto& rho_mut = const_cast<amrex::MultiFab&>(rhofield);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(rho_mut, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box bx = mfi.growntilebox();
            if (bx.bigEnd(0) < iwall) { continue; }
            bx.setSmall(0, iwall);
            auto const& arr = rho_mut.array(mfi);
            // Writes land at i >= iwall only and reads mirror rows at
            // i < iwall only, so the loop body is race-free; overlapping
            // grown tiles re-write identical values (idempotent).
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                arr(i, j, k, 0) = (i == iwall)
                    ? 0.0_rt : -arr(2*iwall - i, j, k, 0);
            });
        }
    }
#endif

    // Solve E field in regular cells
    // The resistive (and hyper-resistive) terms default to the historical
    // coupling with solve_for_Faraday; the theta-implicit solver overrides
    // this to assemble the full Ohm E (grad(Pe) and eta*J together).
    const bool incl_eta = include_resistivity.value_or(solve_for_Faraday);
    if (m_esolve_tensor) {
        // tensor_form: the stateless Je elimination replaces the FD Ohm
        // kernel entirely (Hall/motional, inertia, resistive friction all
        // live in the pointwise tensor and the measured RHS; see
        // m_esolve_tensor).
#if defined(WARPX_DIM_RZ)
        SolveETensorRZ(Efield, Jfield, Bfield, rhofield,
                       *electron_pressure_fp, lev, solve_for_Faraday,
                       incl_eta);
#else
        SolveETensor1D(Efield, Jfield, Bfield, rhofield,
                       *electron_pressure_fp, lev, solve_for_Faraday,
                       incl_eta);
#endif
    } else {
    // Exact-operator hyper-resistivity (m_hyper_resistivity_curlcurl):
    // precompute E_H = +curl(eta_H curl J) of the SAME plasma current the
    // Ohm kernels consume (after the conductor-wall row zeroing above).
    // First curl edge -> face (ComputeCurlA); eta_H is applied INSIDE the
    // outer curl, at the face points, which is the energy-sign-definite
    // form for spatially varying eta_H (E_H.J integrates to
    // eta_H |curl J|^2 >= 0); the scaled field is zero-extended past
    // non-periodic walls (the E_L source convention); second curl
    // face -> edge through the Ampere closure (curl/mu0), scaled back by
    // mu0. The kernels then add E_H in place of the truncated
    // -eta_H lap J.
    if (m_include_hyper_resistivity_term && m_hyper_resistivity_curlcurl
        && incl_eta) {
        auto * fdtd = warpx.get_pointer_fdtd_solver_fp(lev);
        ablastr::fields::VectorField curlJ =
            warpx.m_fields.get_alldirs("hybrid_hyperres_curlJ_fp", lev);
        ablastr::fields::VectorField E_H =
            warpx.m_fields.get_alldirs("hybrid_hyperres_E_fp", lev);
        fdtd->ComputeCurlA(curlJ, current_fp_plasma,
                           warpx.GetEBUpdateBFlag()[lev], lev);
        {
            using namespace ablastr::coarsen::sample;
            const amrex::GpuArray<int, 3> nodal{1, 1, 1};
            const amrex::GpuArray<int, 3> cr{1, 1, 1};
            const amrex::GpuArray<int, 3> B_stag[3] =
                {Bx_IndexType, By_IndexType, Bz_IndexType};
            const auto eta_h = m_eta_h;
            const bool has_B = m_hyper_resistivity_has_B_dependence;
            for (int dir = 0; dir < 3; ++dir) {
                const amrex::GpuArray<int, 3> stag = B_stag[dir];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (amrex::MFIter mfi(*curlJ[dir], amrex::TilingIfNotGPU());
                     mfi.isValid(); ++mfi) {
                    auto const & w = curlJ[dir]->array(mfi);
                    auto const & rho_arr = rhofield.const_array(mfi);
                    auto const & bx = Bfield[0]->const_array(mfi);
                    auto const & by = Bfield[1]->const_array(mfi);
                    auto const & bz = Bfield[2]->const_array(mfi);
                    const amrex::GpuArray<int, 3> sbx = Bx_IndexType;
                    const amrex::GpuArray<int, 3> sby = By_IndexType;
                    const amrex::GpuArray<int, 3> sbz = Bz_IndexType;
                    amrex::ParallelFor(mfi.tilebox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        const amrex::Real rho_f =
                            Interp(rho_arr, nodal, stag, cr, i, j, k, 0);
                        amrex::Real btot = 0.0_rt;
                        if (has_B) {
                            const amrex::Real bxv =
                                Interp(bx, sbx, stag, cr, i, j, k, 0);
                            const amrex::Real byv =
                                Interp(by, sby, stag, cr, i, j, k, 0);
                            const amrex::Real bzv =
                                Interp(bz, sbz, stag, cr, i, j, k, 0);
                            btot = std::sqrt(bxv*bxv + byv*byv + bzv*bzv);
                        }
                        w(i, j, k) *= eta_h(rho_f, btot);
                    });
                }
            }
        }
        for (int dir = 0; dir < 3; ++dir) {
            curlJ[dir]->setBndry(0.0_rt);
            curlJ[dir]->FillBoundary(warpx.Geom(lev).periodicity());
        }
        fdtd->CalculateCurrentAmpere(E_H, curlJ, eb_update_E, lev);
        for (int dir = 0; dir < 3; ++dir) {
            E_H[dir]->mult(PhysConst::mu0);
            E_H[dir]->FillBoundary(warpx.Geom(lev).periodicity());
        }
    }
    warpx.get_pointer_fdtd_solver_fp(lev)->HybridPICSolveE(
        Efield, current_fp_plasma, Jfield, Bfield, rhofield,
        *electron_pressure_fp, eb_update_E, lev, this, solve_for_Faraday,
        incl_eta
    );

    // curlcurl_form (division-free) toroidal sector: the kernel above left
    // the multiplied-through numerator in m_num_theta and only the
    // E-valued resistive/hyper-resistive accumulations in E_theta. Fold
    // those into the numerator (times e n, unfloored), solve the
    // multiplied-through Helmholtz system for E_theta, and apply the
    // deferred external-field subtraction unconditionally (no vacuum
    // branch exists on this path).
    if (m_esolve_curlcurl) {
        amrex::MultiFab & Etheta = *Efield[1];
        amrex::MultiFab & num = *m_num_theta;
        const int mid = rhofield.nComp()/2;
        // Response-frame source: transform the multiplied-through equation
        // BEFORE solving, (beta + curlcurl) E_p = rhs - (beta + curlcurl)
        // E_ext. In-domain curlcurl E_ext = -s'(t) curlcurl A = mu0 J_coil
        // = 0 exactly (the filaments are outside the domain), so the
        // transform reduces to subtracting rho(x) E_ext from the numerator
        // -- division-free, floor-free, and vanishing in vacuum: the
        // inductive drive pushes electrons only where electrons exist. The
        // solved field IS the plasma response (the solver-state frame); a
        // post-solve subtraction would instead force the response to
        // -E_ext across the vacuum region, where the screened solve
        // correctly returns ~0 but E_ext is curl-full.
        amrex::MultiFab const * Et_ext = nullptr;
        if (m_add_external_fields && !m_external_unified) {
            Et_ext = warpx.m_fields.get(FieldType::hybrid_E_fp_external,
                                        ablastr::fields::Direction{1}, lev);
        }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(num, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto const & n_arr = num.array(mfi);
            auto const & e_arr = Etheta.const_array(mfi);
            auto const & r_arr = rhofield.const_array(mfi);
            amrex::Array4<amrex::Real const> xt;
            if (Et_ext) { xt = Et_ext->const_array(mfi); }
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // nodal density from the cell-centered midpoint deposit
                amrex::Real const rho_n = amrex::max(0.0_rt,
                    0.25_rt*(r_arr(i, j, k, mid)
                           + r_arr(i-1, j, k, mid)
                           + r_arr(i, j-1, k, mid)
                           + r_arr(i-1, j-1, k, mid)));
                n_arr(i, j, k) += rho_n * e_arr(i, j, k);
                if (xt) { n_arr(i, j, k) -= rho_n * xt(i, j, k); }
            });
        }
        SolveEThetaCurlCurlRZ(Etheta, num, rhofield, mid, lev);

        // Stage 2: the poloidal (E_r, E_z) pair through the coupled
        // curl-curl solve (the identity form: a component Laplacian
        // equals it only at zero discrete divergence and otherwise
        // spuriously screens E_L at density-contrast edges). Fold the
        // kernel's E-valued resistive accumulations into each numerator
        // with the staggering-consistent unfloored density and subtract
        // the (usually zero: A_theta drives have no poloidal E_ext)
        // response-frame external term; the solve runs once on both
        // native staggerings.
        //
        // Poloidal-stage density source: the per-step-frozen rho^n
        // snapshot (see m_curlcurl_pol_frozen_rho) once it has been
        // captured this step (ComputeElectronInertiaNodal runs before
        // this solve in every residual evaluation); the live midpoint
        // deposit otherwise (initialization solves, knob off). The fold
        // weights, beta rows, RHS membership test, and gamma gate must
        // all read the SAME density: mixed sources reopen the gradient
        // null family the gate closes.
        amrex::MultiFab const * rho_pol = &rhofield;
        int rho_pol_comp = mid;
        if (m_curlcurl_pol_frozen_rho && m_inertia_rho_n_captured) {
            rho_pol = warpx.m_fields.get("hybrid_rho_n_frozen", lev);
            rho_pol_comp = 0;
        }
        for (int c = 0; c < 2; ++c) {
            const int dir = (c == 0) ? 0 : 2;
            amrex::MultiFab & Ec = *Efield[dir];
            amrex::MultiFab & numc = *m_num_pol[c];
            amrex::MultiFab const * Ex_ext = nullptr;
            if (m_add_external_fields && !m_external_unified) {
                Ex_ext = warpx.m_fields.get(
                    FieldType::hybrid_E_fp_external,
                    ablastr::fields::Direction{dir}, lev);
            }
            const bool is_er = (c == 0);
            // Clamp the staggered density average into the cell-centered
            // domain: boundary-node rows otherwise read unfilled rho
            // ghosts (the E_z axis node is a LIVE row; the clamp mirrors
            // SolveEPolCurlCurlRZ's own coefficient assembly). Periodic z
            // ghosts ARE validly filled, so the z clamp disarms there.
            amrex::Box const dom_cc = warpx.Geom(lev).Domain();
            const int ilo_cc = dom_cc.smallEnd(0);
            const int ihi_cc = dom_cc.bigEnd(0);
            const bool z_per_fold = warpx.Geom(lev).isPeriodic(1);
            const int jlo_cc = z_per_fold
                ? std::numeric_limits<int>::lowest()/2 : dom_cc.smallEnd(1);
            const int jhi_cc = z_per_fold
                ? std::numeric_limits<int>::max()/2 : dom_cc.bigEnd(1);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(numc, TilingIfNotGPU());
                 mfi.isValid(); ++mfi)
            {
                auto const & n_arr = numc.array(mfi);
                auto const & e_arr = Ec.const_array(mfi);
                auto const & r_arr = rho_pol->const_array(mfi);
                const int rc_pol = rho_pol_comp;
                amrex::Array4<amrex::Real const> xt;
                if (Ex_ext) { xt = Ex_ext->const_array(mfi); }
                amrex::ParallelFor(mfi.tilebox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    amrex::Real rho_p;
                    if (is_er) {
                        int const jm = amrex::max(j - 1, jlo_cc);
                        int const jp = amrex::min(j, jhi_cc);
                        rho_p = 0.5_rt*(r_arr(i, jp, k, rc_pol)
                                      + r_arr(i, jm, k, rc_pol));
                    } else {
                        int const im = amrex::max(i - 1, ilo_cc);
                        int const ip = amrex::min(i, ihi_cc);
                        rho_p = 0.5_rt*(r_arr(ip, j, k, rc_pol)
                                      + r_arr(im, j, k, rc_pol));
                    }
                    rho_p = amrex::max(0.0_rt, rho_p);
                    n_arr(i, j, k) += rho_p * e_arr(i, j, k);
                    if (xt) { n_arr(i, j, k) -= rho_p * xt(i, j, k); }
                });
            }
        }
        SolveEPolCurlCurlRZ(*Efield[0], *Efield[2],
                            *m_num_pol[0], *m_num_pol[1],
                            *rho_pol, rho_pol_comp, lev);
    }
    } // end !m_esolve_tensor

    amrex::Real const time = warpx.gett_old(0) + warpx.getdt(0);
    warpx.ApplyEfieldBoundary(lev, patch_type, time);

    // Conformal wall, constitutive PEC: the Ohm E is algebraic in B, so the
    // wall condition is imposed directly -- E = 0 on every covered and cut
    // edge (tangential E vanishes at the wall at the cut-edge level).
    if (EB::enabled() && m_use_conformal_eb && m_conformal_wall_conductor) {
        ZeroConductorEdges(Efield, eb_update_E, lev);
    }

    // Thin resistive-shell wall (RZ r-max): overwrite the tangential wall
    // rows the PEC kernel just zeroed with the shell value computed from
    // THIS evaluation's B iterate (the Bfield handed to the Ohm kernels),
    // ghosts imaged odd about that value. Composition order at this site:
    // pec_conductor_wall_rows moment-zeroing (Ohm INPUTS, top of this
    // function) -> Ohm kernels -> ApplyEfieldBoundary (PEC image) ->
    // resistive wall LAST, so it owns the tangential wall row + r-hi ghosts
    // while E_r keeps the PEC normal image. This runs inside every residual
    // evaluation of the theta-implicit solver, making the wall a relaxation
    // (Robin) row of the residual by construction; the explicit hybrid's
    // subcycled E-solves route through this same wrapper and get the
    // per-substep shell value.
    ApplyResistiveWallE(Efield, Bfield, lev);
}

void HybridPICModel::ApplyResistiveWallE (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Bfield,
    const int lev) const
{
#if defined(WARPX_DIM_RZ)
    if (!m_resistive_wall
        || WarpX::field_boundary_hi[0] != FieldBoundaryType::PEC) {
        return;
    }

    auto& warpx = WarpX::GetInstance();
    // Wall node plane (E_theta and E_z are nodal in r).
    const int iwall = warpx.Geom(lev).Domain().bigEnd(0) + 1;
    const amrex::Real dr = warpx.Geom(lev).CellSize(0);
    const amrex::Real delta = (m_wall_thickness > 0.) ? m_wall_thickness : dr;
    // E_t(wall) = (eta_w/delta) K_t with K from the tangential-B jump:
    //   K_theta = +(B_z(in) - B_z,ext)/mu0,  K_z = -B_theta(in)/mu0,
    // so E_t = fac * (jump), fac = eta_w/(delta*mu0)  [m/s].
    const amrex::Real fac =
        m_wall_resistivity / (delta * PhysConst::mu0);
    const amrex::Real bz_ext = m_wall_bz_ext;

    // comp 1 (E_theta) reads B_z; comp 2 (E_z) reads B_theta. Both B
    // sources are cell-centered in r (last interior row at iwall-1) and
    // share the E component's z staggering (B_z node-in-z with E_theta,
    // B_theta cell-in-z with E_z), so the j index passes through.
    for (int comp : {1, 2}) {
        amrex::MultiFab& E = *Efield[comp];
        amrex::MultiFab const& B = (comp == 1) ? *Bfield[2] : *Bfield[1];
        const amrex::Real sgn = (comp == 1) ? 1.0_rt : -1.0_rt;
        const amrex::Real sub = (comp == 1) ? bz_ext : 0.0_rt;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(E, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box bx = mfi.growntilebox();
            if (bx.bigEnd(0) < iwall) { continue; }
            // Guard against fabs whose E ghosts reach the wall but whose B
            // fab box does not contain the last interior B row.
            if (B[mfi].box().bigEnd(0) < iwall - 1) { continue; }
            bx.setSmall(0, iwall);
            auto const& E_arr = E.array(mfi);
            auto const& B_arr = B.const_array(mfi);
            // Writes land at i >= iwall only; reads come from B (untouched)
            // and from E interior mirrors at i < iwall only, so iterations
            // are independent; overlapping grown tiles re-write identical
            // values (idempotent).
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const amrex::Real ew =
                    sgn * fac * (B_arr(iwall - 1, j, k, 0) - sub);
                E_arr(i, j, k, 0) = (i == iwall)
                    ? ew : 2.0_rt * ew - E_arr(2*iwall - i, j, k, 0);
            });
        }
    }
#else
    amrex::ignore_unused(Efield, Bfield, lev);
#endif
}

void HybridPICModel::ImposeResistiveWallIterateRowsE (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Esrc,
    const int lev) const
{
#if defined(WARPX_DIM_RZ)
    if (!m_resistive_wall
        || WarpX::field_boundary_hi[0] != FieldBoundaryType::PEC) {
        return;
    }

    auto& warpx = WarpX::GetInstance();
    const int iwall = warpx.Geom(lev).Domain().bigEnd(0) + 1;

    for (int comp : {1, 2}) {
        amrex::MultiFab& E = *Efield[comp];
        amrex::MultiFab const& Es = *Esrc[comp];

        // Pass 1: restore the VALID wall-node row from the source iterate.
        // Only valid rows are read from Esrc -- its ghost layers are not
        // guaranteed current (solver-vector linear algebra touches valid
        // data only).
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(E, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box bx = mfi.tilebox();
            if (bx.bigEnd(0) < iwall) { continue; }
            bx.setSmall(0, iwall);
            auto const& E_arr = E.array(mfi);
            auto const& Es_arr = Es.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                E_arr(i, j, k, 0) = Es_arr(i, j, k, 0);
            });
        }

        // Propagate the restored wall row into the z-periodic ghost columns
        // (the PEC kernel zeroed the wall node there too, and the ghost
        // image below reads the wall value at ghost j).
        E.FillBoundary(warpx.Geom(lev).periodicity());

        // Pass 2: image the r-hi ghost band odd about the wall value
        // (ghost = 2*E_wall - interior mirror). Writes land at i > iwall
        // only; reads come from i <= iwall only, so iterations are
        // independent; overlapping grown tiles re-write identical values.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(E, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box bx = mfi.growntilebox();
            if (bx.bigEnd(0) < iwall + 1) { continue; }
            bx.setSmall(0, iwall + 1);
            auto const& E_arr = E.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                E_arr(i, j, k, 0) = 2.0_rt * E_arr(iwall, j, k, 0)
                                  - E_arr(2*iwall - i, j, k, 0);
            });
        }
    }
#else
    amrex::ignore_unused(Efield, Esrc, lev);
#endif
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
    auto& warpx = WarpX::GetInstance();
    CalculateElectronPressure(
        lev, *warpx.m_fields.get(FieldType::rho_fp, lev));
}

void HybridPICModel::CalculateElectronPressure(const int lev,
                                               amrex::MultiFab const& rho_mf) const
{
    ABLASTR_PROFILE("WarpX::CalculateElectronPressure()");

    auto& warpx = WarpX::GetInstance();
    ablastr::fields::ScalarField electron_pressure_fp = warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);

    // Calculate the electron pressure from component 0 of the given density
    // (the current time level on the explicit path; under the theta-implicit
    // scheme the in-solve evaluations see midpoint-position deposits, and
    // CalculateElectronPressureAtStepEnd re-evaluates with a true
    // end-of-step deposit after the solve).
    FillElectronPressureMF(
        *electron_pressure_fp,
        rho_mf
    );
    // Conformal wall: Dirichlet Pe at the PEC surface (odd reflection). The
    // resulting grad(Pe) across the wall supplies the allowable normal E in
    // Ohm's law, unlike a Neumann fill, which would pin it to zero.
    if (EB::enabled() && m_use_conformal_eb && m_conformal_wall_conductor) {
        warpx::hybrid::ApplyEBBoundaryToNodalScalar(
            *electron_pressure_fp,
            *warpx.m_fields.get(FieldType::distance_to_eb, lev),
            warpx.Geom(lev),
            /*odd=*/true);
    }
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
        amrex::MultiFab const & rho = rho_mf;
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

void HybridPICModel::CalculateElectronPressureAtStepEnd () const
{
    ABLASTR_PROFILE("HybridPICModel::CalculateElectronPressureAtStepEnd()");

    auto& warpx = WarpX::GetInstance();

    // True end-of-step charge density for the closure evaluation: during the
    // nonlinear iteration the rho_fp components hold midpoint-position
    // deposits (the correct theta-level density for the residual), and
    // nothing after the solve re-evaluates the closure, so the dumped Pe/Te
    // -- and the values the next step's pre-solve consumers see (the
    // resistive-drag collision operator's T_e) -- would otherwise lag the
    // ion state by half a step. That label error is invisible in the
    // dynamics but first order in every Te-based convergence metric
    // (closure-ladder measurement: Te order ~1.0 vs rho ~2.0).
    //
    // Deposit through the MultiParticleContainer path into the
    // (implicit-path-idle) hybrid_rho_fp_temp fab, then fold the guards,
    // reflect at the walls and fill ghosts -- the same treatment the main
    // loop gives rho_fp. rho_fp itself is deliberately left untouched: the
    // Darwin entry block reads its component 0 (and the as-left Pe) for the
    // E_L^n solve, and re-labeling that input is a separate decision.
    //
    // Deliberately NO Redistribute before this deposit, unlike the
    // QDSMCFinishImplicitStep end recovery: redistributing here reorders
    // the particles ahead of the next step's deposits, and the last-bit
    // float reassociation that follows grows chaotically -- measured as a
    // ~1e-3 trajectory shift over the 50-step RZ em-modes test, breaking
    // bit comparability of every closure-path implicit run (and the pinned
    // regression data) for what is a diagnostics-only fix. The step-end
    // extrapolated positions deposit from their home tiles through the
    // guard cells, which the SumBoundary below folds back home -- the same
    // posture as the per-species temperature deposits that already run
    // un-redistributed at this point in the step. (Pathological one-step
    // guard-range crossers are a documented pre-existing hazard of that
    // posture; test decks use inert heavy ions to avoid it.)
    auto rho_end_levels = warpx.m_fields.get_mr_levels(
        FieldType::hybrid_rho_fp_temp, warpx.finestLevel());
    warpx.GetPartContainer().DepositCharge(rho_end_levels, 0.0_rt);
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        amrex::MultiFab & rho_end_mf = *rho_end_levels[lev];
        ablastr::utils::communication::SumBoundary(
            rho_end_mf, 0, rho_end_mf.nComp(), rho_end_mf.nGrowVect(), rho_end_mf.nGrowVect(),
            WarpX::do_single_precision_comms, warpx.Geom(lev).periodicity());
        warpx.ApplyRhofieldBoundary(lev, &rho_end_mf, PatchType::fine);
        rho_end_mf.FillBoundary(warpx.Geom(lev).periodicity());
        CalculateElectronPressure(lev, rho_end_mf);
    }
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
            // The wall's odd rho reflection mirrors the density negative
            // inside the conductor: clamp the equation-of-state input.
            Pe(i, j, k) = ElectronPressure::get_pressure(
                n0_ref, elec_temp, gamma, amrex::max(rho(i, j, k), 0._rt)
            );
        });
    }
}

// =============================================================================
// Darwin longitudinal-field constraint
// =============================================================================

void HybridPICModel::SolveEThetaCurlCurlRZ (amrex::MultiFab& Etheta,
                                            amrex::MultiFab const& num,
                                            amrex::MultiFab const& rho,
                                            int const rho_comp,
                                            int const lev) const
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(Etheta, num, rho, rho_comp, lev);
    WARPX_ABORT_WITH_MESSAGE(
        "SolveEThetaCurlCurlRZ: the curlcurl_form toroidal solve is RZ-only");
#else
    ABLASTR_PROFILE("HybridPICModel::SolveEThetaCurlCurlRZ()");

    using namespace amrex;

    auto & warpx = WarpX::GetInstance();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(lev == 0,
        "hybrid_pic_model.esolve = curlcurl_form supports a single level");
    Geometry const & geom = warpx.Geom(lev);
    auto const dx_arr = geom.CellSizeArray();
    Real const dr = dx_arr[0];
    Real const rmin = geom.ProbLo(0);

    Real const me_eff = m_electron_inertia_mass;
    // operator scale: beta = mu0 e^2 n / m_e_eff = mu0 e rho / m_e_eff
    Real const beta_fac = PhysConst::mu0 * PhysConst::q_e / me_eff;
    // rhs scale applied by the caller convention: rhs = mu0 e/m_e * num;
    // the metric multiply-through by r happens here for both sides.
    Real const rhs_fac = PhysConst::mu0 * PhysConst::q_e / me_eff;
    // One-count RHS membership anchor (hybrid_pic_model.curlcurl_n_min;
    // 0 = legacy ungated rows): sub-one-count rows would otherwise solve
    // E_theta from interpolation-bleed content against the bare 1/r
    // screening -- the unfloored division resurfacing on deposit-tail
    // rows. No gamma family here (the toroidal sector has no gradient
    // null space); the row keeps its (regular) operator.
    Real const rho_1c = PhysConst::q_e * m_curlcurl_n_min;

    // Coefficients: nodal acoef = r*beta + 1/r (the 1/r is the metric-
    // absorbed 1/r^2 vector-Helmholtz term; at the axis node the row is a
    // Dirichlet regularity boundary -- cap r away from zero only to keep
    // the masked row finite), cell-centered bcoef = r.
    BoxArray const & ba = warpx.boxArray(lev);
    DistributionMapping const & dm = warpx.DistributionMap(lev);
    MultiFab acoef(amrex::convert(ba, IntVect::TheNodeVector()), dm, 1, 0);
    MultiFab bcoef(ba, dm, 1, 1);
    MultiFab rhs(amrex::convert(ba, IntVect::TheNodeVector()), dm, 1, 0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(acoef, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const nbx = mfi.tilebox();
        auto const & a_arr = acoef.array(mfi);
        auto const & r_arr = rhs.array(mfi);
        auto const & rho_arr = rho.const_array(mfi);
        auto const & num_arr = num.const_array(mfi);
        Real const rcap = 0.125_rt * dr;
        amrex::ParallelFor(nbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real const r = amrex::max(rmin + i*dr, rcap);
            // node-interpolated midpoint density, UNfloored (clamped at
            // zero: negative deposits must not flip the operator sign)
            Real const rho_n = amrex::max(0.0_rt,
                0.25_rt*(rho_arr(i, j, k, rho_comp)
                       + rho_arr(i-1, j, k, rho_comp)
                       + rho_arr(i, j-1, k, rho_comp)
                       + rho_arr(i-1, j-1, k, rho_comp)));
            a_arr(i, j, k) = r*beta_fac*rho_n + 1.0_rt/r;
            r_arr(i, j, k) = (rho_1c <= 0.0_rt || rho_n > rho_1c)
                ? r*rhs_fac*num_arr(i, j, k) : 0.0_rt;
        });
    }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(bcoef, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const cbx = mfi.growntilebox(1);
        auto const & b_arr = bcoef.array(mfi);
        amrex::ParallelFor(cbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            b_arr(i, j, k) = rmin + (i + 0.5_rt)*dr;
        });
    }

    // Boundary types: axis = Dirichlet 0 (m = 0 regularity of a theta
    // vector component), r_hi = Dirichlet 0 (PEC tangential E); z periodic
    // or PEC-class Dirichlet. The Green's open face has no mapping yet --
    // documented follow-up.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC,
        "hybrid_pic_model.esolve = curlcurl_form currently requires a PEC "
        "wall at r_hi (the Green's open face mapping is a follow-up)");
    const bool z_periodic = geom.isPeriodic(1);
    if (!z_periodic) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::field_boundary_lo[1] == FieldBoundaryType::PEC
                && WarpX::field_boundary_hi[1] == FieldBoundaryType::PEC,
            "hybrid_pic_model.esolve = curlcurl_form: non-periodic z requires "
            "PEC z boundaries");
    }

    // Preconditioned conjugate gradients on the exact nodal stencil.
    // (MLNodeABecLaplacian hard-aborts in EB-enabled builds and no AMReX
    // nodal operator carries a varying zeroth-order coefficient there, so
    // the operator is applied by hand: it is SPD -- acoef >= 1/r > 0 and
    // the conservative 5-point r-sigma Laplacian with Dirichlet rows --
    // and Jacobi-preconditioned CG converges in O(N_r) iterations at the
    // validation scales this stage targets. A factored (block-banded
    // direct) backend is the production follow-up.)
    Box const dom_nd = amrex::convert(geom.Domain(), IntVect::TheNodeVector());
    auto const dlo = amrex::lbound(dom_nd);
    auto const dhi = amrex::ubound(dom_nd);
    Real const inv_dr2 = 1.0_rt/(dr*dr);
    Real const dz = dx_arr[1];
    Real const inv_dz2 = 1.0_rt/(dz*dz);

    // A p (with Dirichlet rows as identity), reading p's ghosts
    auto apply_op = [&](MultiFab& out, MultiFab& in)
    {
        in.FillBoundary(geom.periodicity());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(out, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto const & o = out.array(mfi);
            auto const & p = in.const_array(mfi);
            auto const & a = acoef.const_array(mfi);
            auto const & b = bcoef.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                const bool dir_row = (i <= dlo.x) || (i >= dhi.x)
                    || (!z_periodic && (j <= dlo.y || j >= dhi.y));
                if (dir_row) { o(i,j,k) = p(i,j,k); return; }
                Real const lap =
                    (b(i, j, k)*(p(i+1,j,k) - p(i,j,k))
                     - b(i-1, j, k)*(p(i,j,k) - p(i-1,j,k)))*inv_dr2
                  + (0.5_rt*(b(i,j,k) + b(i-1,j,k)))
                    *((p(i,j+1,k) - p(i,j,k)) - (p(i,j,k) - p(i,j-1,k)))
                    *inv_dz2;
                o(i,j,k) = a(i,j,k)*p(i,j,k) - lap;
            });
        }
    };

    // Dirichlet rows of the RHS carry the boundary values (zero), and the
    // initial guess is the incoming E_theta (warm start from the previous
    // evaluation) with its Dirichlet rows zeroed.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const & r_arr = rhs.array(mfi);
        auto const & e_arr = Etheta.array(mfi);
        amrex::ParallelFor(mfi.tilebox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            const bool dir_row = (i <= dlo.x) || (i >= dhi.x)
                || (!z_periodic && (j <= dlo.y || j >= dhi.y));
            if (dir_row) { r_arr(i,j,k) = 0.0_rt; e_arr(i,j,k) = 0.0_rt; }
        });
    }

    auto const ng0 = IntVect(0);
    MultiFab Ax(rhs.boxArray(), dm, 1, 0);
    MultiFab res(rhs.boxArray(), dm, 1, 0);
    MultiFab z_mf(rhs.boxArray(), dm, 1, 0);
    MultiFab p_mf(rhs.boxArray(), dm, 1, IntVect(1));
    MultiFab diag(rhs.boxArray(), dm, 1, 0);

    // Jacobi diagonal: a + 2 sigma_avg (1/dr^2 + 1/dz^2)-form
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(diag, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const & d = diag.array(mfi);
        auto const & a = acoef.const_array(mfi);
        auto const & b = bcoef.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real const bs = b(i,j,k) + b(i-1,j,k);
            d(i,j,k) = a(i,j,k) + bs*inv_dr2 + bs*inv_dz2;
        });
    }

    // r0 = rhs - A x0
    apply_op(Ax, Etheta);
    MultiFab::LinComb(res, 1.0_rt, rhs, 0, -1.0_rt, Ax, 0, 0, 1, ng0);
    MultiFab::Copy(z_mf, res, 0, 0, 1, ng0);
    MultiFab::Divide(z_mf, diag, 0, 0, 1, 0);
    p_mf.setVal(0.0_rt);
    MultiFab::Copy(p_mf, z_mf, 0, 0, 1, ng0);
    Real rz_old = MultiFab::Dot(res, 0, z_mf, 0, 1, 0);
    Real const rhs_norm = std::sqrt(MultiFab::Dot(rhs, 0, rhs, 0, 1, 0));
    // Tight tolerance: this solve runs inside every residual evaluation
    // including finite-difference Jacobian probes (the Darwin E_L
    // precedent) -- solver noise above the probe scale poisons secants.
    Real const tol = 1.0e-11_rt * amrex::max(rhs_norm, 1.0e-300_rt);
    int const max_cg = 2000;
    int it = 0;
    for (; it < max_cg; ++it) {
        Real const res_norm = std::sqrt(MultiFab::Dot(res, 0, res, 0, 1, 0));
        if (res_norm <= tol) { break; }
        apply_op(Ax, p_mf);
        Real const pAp = MultiFab::Dot(p_mf, 0, Ax, 0, 1, 0);
        Real const alpha = rz_old / pAp;
        MultiFab::Saxpy(Etheta, alpha, p_mf, 0, 0, 1, ng0);
        MultiFab::Saxpy(res, -alpha, Ax, 0, 0, 1, ng0);
        MultiFab::Copy(z_mf, res, 0, 0, 1, ng0);
        MultiFab::Divide(z_mf, diag, 0, 0, 1, 0);
        Real const rz_new = MultiFab::Dot(res, 0, z_mf, 0, 1, 0);
        Real const beta_cg = rz_new / rz_old;
        rz_old = rz_new;
        MultiFab::LinComb(p_mf, 1.0_rt, z_mf, 0, beta_cg, p_mf, 0, 0, 1, ng0);
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(it < max_cg,
        "SolveEThetaCurlCurlRZ: CG failed to converge");

    Etheta.FillBoundary(geom.periodicity());
#endif
}

namespace {
#if defined(WARPX_DIM_RZ)
    /** Node volume (without the common dz) for the poloidal vacuum-closure
     *  energy (1/2) sum gamma Vn (div E)^2 in SolveEPolCurlCurlRZ:
     *  interior r_n dr; domain-end half cells carry the exact integral of
     *  r dr over the existing half (axis at rmin = 0: dr^2/8). A free
     *  function: capturing a device lambda inside another extended lambda
     *  is CUDA-illegal. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real PolNodeVol (int i, amrex::Real rmin, amrex::Real dr,
                            int ihi_nd)
    {
        if (i == 0) { return (rmin + 0.25*dr)*0.5*dr; }
        if (i == ihi_nd) { return (rmin + i*dr - 0.25*dr)*0.5*dr; }
        return (rmin + i*dr)*dr;
    }
#endif
}

void HybridPICModel::SolveEPolCurlCurlRZ (amrex::MultiFab& Er,
                                          amrex::MultiFab& Ez,
                                          amrex::MultiFab const& num_r,
                                          amrex::MultiFab const& num_z,
                                          amrex::MultiFab const& rho,
                                          int const rho_comp,
                                          int const lev) const
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(Er, Ez, num_r, num_z, rho, rho_comp, lev);
    WARPX_ABORT_WITH_MESSAGE(
        "SolveEPolCurlCurlRZ: the curlcurl_form poloidal solve is RZ-only");
#else
    ABLASTR_PROFILE("HybridPICModel::SolveEPolCurlCurlRZ()");
    using namespace amrex;

    auto & warpx = WarpX::GetInstance();
    Geometry const & geom = warpx.Geom(lev);
    auto const dx_arr = geom.CellSizeArray();
    Real const dr = dx_arr[0];
    Real const dz = dx_arr[1];
    Real const rmin = geom.ProbLo(0);
    Real const beta_fac = PhysConst::mu0 * PhysConst::q_e
        / m_electron_inertia_mass;
    Real const rhs_fac = beta_fac;
    // One-count membership anchor (hybrid_pic_model.curlcurl_n_min;
    // 0 = legacy strictly-zero gating). Sub-anchor rows keep their tiny
    // screening in the operator but take a zeroed RHS and the gamma
    // div E = 0 completion as content -- the SolveETensorRZ recipe (a
    // statistics guard, never a density modification).
    Real const rho_1c = PhysConst::q_e * m_curlcurl_n_min;
    const bool z_periodic = geom.isPeriodic(1);
    Real const inv_dr = 1.0_rt/dr;
    Real const inv_dz = 1.0_rt/dz;
    Real const inv_dz2 = inv_dz*inv_dz;

    // Staggered domain bounds: E_r (cc r, nodal z), E_z (nodal r, cc z).
    Box const dom_r = amrex::convert(geom.Domain(), Er.ixType().toIntVect());
    Box const dom_z = amrex::convert(geom.Domain(), Ez.ixType().toIntVect());
    auto const rlo = amrex::lbound(dom_r);
    auto const rhi = amrex::ubound(dom_r);
    auto const zlo = amrex::lbound(dom_z);
    auto const zhi = amrex::ubound(dom_z);

    auto const & dm = Er.DistributionMap();

    // Row scalings (control volumes without the common dz): V_r = r_c dr,
    // V_z = r_n dr (E_z axis node: dr^2/8). beta from the staggered
    // UNfloored density (the caller passes the per-step-frozen rho^n
    // snapshot by default, the live midpoint deposit with
    // curlcurl_pol_frozen_rho = 0 -- see the member doc), clamped at zero
    // (negative deposits must not flip the operator sign). RHS rows with
    // zero staggered density are zeroed: the multiplied-through equation
    // is 0 = 0 without electrons, and the interpolation-bleed numerator
    // there is exactly the content inconsistent with the curl-curl
    // gradient null space (see the header doc).
    MultiFab acoef_r(Er.boxArray(), dm, 1, 0);
    MultiFab acoef_z(Ez.boxArray(), dm, 1, 0);
    MultiFab rhs_r(Er.boxArray(), dm, 1, 0);
    MultiFab rhs_z(Ez.boxArray(), dm, 1, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(acoef_r, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const & a_arr = acoef_r.array(mfi);
        auto const & b_arr = rhs_r.array(mfi);
        auto const & rho_arr = rho.const_array(mfi);
        auto const & num_arr = num_r.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real const V = (rmin + (i + 0.5_rt)*dr)*dr;
            // (cc r, nodal z): average the two z-neighbor cells; the
            // non-periodic cap rows read a z ghost but are Dirichlet
            // identity rows whose coefficients are never consumed
            Real const rho_p = amrex::max(0.0_rt,
                0.5_rt*(rho_arr(i, j, k, rho_comp)
                      + rho_arr(i, j-1, k, rho_comp)));
            a_arr(i, j, k) = beta_fac*rho_p*V;
            b_arr(i, j, k) = (rho_p > rho_1c)
                ? V*rhs_fac*num_arr(i, j, k) : 0.0_rt;
        });
    }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(acoef_z, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const & a_arr = acoef_z.array(mfi);
        auto const & b_arr = rhs_z.array(mfi);
        auto const & rho_arr = rho.const_array(mfi);
        auto const & num_arr = num_z.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real const V = (i == zlo.x)
                ? 0.125_rt*dr*dr : (rmin + i*dr)*dr;
            // (nodal r, cc z): average the two r-neighbor cells (clamped
            // into the domain at the axis; the wall node reads an r ghost
            // but is a Dirichlet identity row)
            Real const rho_p = amrex::max(0.0_rt,
                0.5_rt*(rho_arr(i, j, k, rho_comp)
                      + rho_arr(amrex::max(i-1, zlo.x), j, k, rho_comp)));
            a_arr(i, j, k) = beta_fac*rho_p*V;
            b_arr(i, j, k) = (rho_p > rho_1c)
                ? V*rhs_fac*num_arr(i, j, k) : 0.0_rt;
        });
    }

    // Dirichlet rows (PEC tangential E): zero RHS and warm start; the
    // matching residual/search-direction entries then stay exactly zero
    // through CG, so the identity rows never couple into the free rows.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs_r, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const & b_arr = rhs_r.array(mfi);
        auto const & e_arr = Er.array(mfi);
        amrex::ParallelFor(mfi.tilebox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::ignore_unused(i);
            if (!z_periodic && (j <= rlo.y || j >= rhi.y)) {
                b_arr(i,j,k) = 0.0_rt; e_arr(i,j,k) = 0.0_rt;
            }
        });
    }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs_z, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const & b_arr = rhs_z.array(mfi);
        auto const & e_arr = Ez.array(mfi);
        amrex::ParallelFor(mfi.tilebox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::ignore_unused(j);
            if (i >= zhi.x) {
                b_arr(i,j,k) = 0.0_rt; e_arr(i,j,k) = 0.0_rt;
            }
        });
    }

    // Vacuum-closure gate: gamma = 1 on strictly-vacuum NODES (nodal
    // unfloored density <= 0), 0 elsewhere. Where the unfloored density
    // vanishes the multiplied-through equation is 0 = 0 and the bare
    // curl-curl rows are singular along discrete gradients (a null
    // family the Jacobi preconditioner reinjects, random-walking junk
    // into the vacuum E that edge particles then gather; measured as CG
    // grinding at 10k-17k iterations and intermittent Newton divergence
    // on a vacuum-annulus column deck). The physical closure is the
    // vacuum Gauss constraint div E = 0, imposed below as the energy-form
    // -grad(gamma div .) completion. The axis and wall nodes use their
    // half-cell FV divergence over the EXISTING faces only (axis: the
    // inner face has zero metric area; wall: normal E lives half a cell
    // inside, the PEC surface-charge face is outside the half cell) --
    // gating them too is what closes the wall-line gradient family
    // (phi free along the wall nodes otherwise). Non-periodic cap nodes
    // stay ungated (their E_r rows are Dirichlet; a vacuum region
    // touching the caps would retain a null family -- documented
    // limitation, not exercised by the periodic-z decks). Deep vacuum
    // carries the full vector Laplacian (curlcurl - grad div) and the
    // system is SPD on the free rows, while every row carrying plasma
    // keeps the pure identity form (the density-sheet rows couple to the
    // closure only as its boundary data).
    Box const dom_nd = amrex::convert(geom.Domain(),
                                      IntVect::TheNodeVector());
    auto const nlo = amrex::lbound(dom_nd);
    auto const nhi = amrex::ubound(dom_nd);
    MultiFab gam(amrex::convert(warpx.boxArray(lev),
                                IntVect::TheNodeVector()),
                 dm, 1, 1);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(gam, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const & g = gam.array(mfi);
        auto const & rho_arr = rho.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (!z_periodic && (j <= nlo.y || j >= nhi.y)) {
                g(i,j,k) = 0.0_rt; return;
            }
            Real rho_n;
            if (i <= nlo.x) {
                rho_n = 0.5_rt*(rho_arr(i, j, k, rho_comp)
                              + rho_arr(i, j-1, k, rho_comp));
            } else if (i >= nhi.x) {
                rho_n = 0.5_rt*(rho_arr(i-1, j, k, rho_comp)
                              + rho_arr(i-1, j-1, k, rho_comp));
            } else {
                rho_n = 0.25_rt*(rho_arr(i, j, k, rho_comp)
                               + rho_arr(i-1, j, k, rho_comp)
                               + rho_arr(i, j-1, k, rho_comp)
                               + rho_arr(i-1, j-1, k, rho_comp));
            }
            // anchored gate: sub-one-count nodes take the div E = 0
            // completion (rho_1c = 0 reproduces the strictly-zero gate)
            g(i,j,k) = (rho_n > rho_1c) ? 0.0_rt : 1.0_rt;
        });
    }
    gam.setBndry(0.0_rt);
    gam.FillBoundary(geom.periodicity());

    int const ihi_nd = nhi.x;

    // Fused block operator: row = a E + V (curlcurl E)|_pol
    // - V grad(gamma div E)|_pol, W and the gated FV divergence
    // recomputed inline from the input pair (one ghost each); Dirichlet
    // rows are identity. The E_r row carries no radial second difference
    // of E_r and no 1/r^2 term ((curlcurl)_r = -d_z(d_z E_r - d_r E_z));
    // the E_z row carries no axial second difference of E_z. Natural
    // curl rows drop the missing W cell (E_z axis: W(-1); every W cell
    // of the E_r wall row exists). Cross-coefficient symmetry on the
    // free rows follows from the energy form (header doc). The wall
    // nodes' divergence reads the (identically zero) Dirichlet E_z wall
    // DOFs, which is harmless.
    auto apply_op = [&](MultiFab& out_r, MultiFab& out_z,
                        MultiFab& in_r, MultiFab& in_z)
    {
        in_r.FillBoundary(geom.periodicity());
        in_z.FillBoundary(geom.periodicity());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(out_r, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto const & o = out_r.array(mfi);
            auto const & pr = in_r.const_array(mfi);
            auto const & pz = in_z.const_array(mfi);
            auto const & a = acoef_r.const_array(mfi);
            auto const & g = gam.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (!z_periodic && (j <= rlo.y || j >= rhi.y)) {
                    o(i,j,k) = pr(i,j,k); return;
                }
                Real const rc = rmin + (i + 0.5_rt)*dr;
                Real const wp = (pr(i,j+1,k) - pr(i,j,k))*inv_dz
                              - (pz(i+1,j,k) - pz(i,j,k))*inv_dr;
                Real const wm = (pr(i,j,k) - pr(i,j-1,k))*inv_dz
                              - (pz(i+1,j-1,k) - pz(i,j-1,k))*inv_dr;
                Real dd = 0.0_rt;
                if (g(i,j,k) > 0.0_rt) {
                    // gated FV divergence at node (i, j)
                    Real f = rc*pr(i,j,k);
                    if (i > 0) {
                        f -= (rmin + (i - 0.5_rt)*dr)*pr(i-1,j,k);
                    }
                    dd += f/PolNodeVol(i, rmin, dr, ihi_nd)
                        + (pz(i,j,k) - pz(i,j-1,k))*inv_dz;
                }
                if (g(i+1,j,k) > 0.0_rt) {
                    // gated FV divergence at node (i+1, j); the outer
                    // flux is dropped at the wall node (half cell)
                    Real f = -rc*pr(i,j,k);
                    if (i + 1 <= rhi.x) {
                        f += (rmin + (i + 1.5_rt)*dr)*pr(i+1,j,k);
                    }
                    dd -= f/PolNodeVol(i+1, rmin, dr, ihi_nd)
                        + (pz(i+1,j,k) - pz(i+1,j-1,k))*inv_dz;
                }
                o(i,j,k) = a(i,j,k)*pr(i,j,k) - rc*dr*(wp - wm)*inv_dz
                    + rc*dd;
            });
        }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(out_z, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto const & o = out_z.array(mfi);
            auto const & pr = in_r.const_array(mfi);
            auto const & pz = in_z.const_array(mfi);
            auto const & a = acoef_z.const_array(mfi);
            auto const & g = gam.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (i >= zhi.x) { o(i,j,k) = pz(i,j,k); return; }
                Real const rcp = rmin + (i + 0.5_rt)*dr;
                Real const rcm = rmin + (i - 0.5_rt)*dr;
                Real const wp = (pr(i,j+1,k) - pr(i,j,k))*inv_dz
                              - (pz(i+1,j,k) - pz(i,j,k))*inv_dr;
                Real cc = rcp*wp;
                if (i > zlo.x) {
                    Real const wm = (pr(i-1,j+1,k) - pr(i-1,j,k))*inv_dz
                                  - (pz(i,j,k) - pz(i-1,j,k))*inv_dr;
                    cc -= rcm*wm;
                }
                if (g(i,j,k) > 0.0_rt || g(i,j+1,k) > 0.0_rt) {
                    Real const Vn = PolNodeVol(i, rmin, dr, ihi_nd);
                    Real dd = 0.0_rt;
                    if (g(i,j,k) > 0.0_rt) {
                        Real f = rcp*pr(i,j,k);
                        if (i > 0) { f -= rcm*pr(i-1,j,k); }
                        dd += f/Vn
                            + (pz(i,j,k) - pz(i,j-1,k))*inv_dz;
                    }
                    if (g(i,j+1,k) > 0.0_rt) {
                        Real f = rcp*pr(i,j+1,k);
                        if (i > 0) { f -= rcm*pr(i-1,j+1,k); }
                        dd -= f/Vn
                            + (pz(i,j+1,k) - pz(i,j,k))*inv_dz;
                    }
                    cc += Vn*dd*inv_dz;
                }
                o(i,j,k) = a(i,j,k)*pz(i,j,k) + cc;
            });
        }
    };

    // Jacobi diagonals: the exact diagonal of the fused rows (Dirichlet
    // rows get 1; their residual entries are identically zero).
    MultiFab diag_r(Er.boxArray(), dm, 1, 0);
    MultiFab diag_z(Ez.boxArray(), dm, 1, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(diag_r, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const & d = diag_r.array(mfi);
        auto const & a = acoef_r.const_array(mfi);
        auto const & g = gam.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (!z_periodic && (j <= rlo.y || j >= rhi.y)) {
                d(i,j,k) = 1.0_rt; return;
            }
            Real const rc = rmin + (i + 0.5_rt)*dr;
            Real dv = 0.0_rt;
            if (g(i,j,k) > 0.0_rt) { dv += rc*rc/PolNodeVol(i, rmin, dr, ihi_nd); }
            if (g(i+1,j,k) > 0.0_rt) { dv += rc*rc/PolNodeVol(i+1, rmin, dr, ihi_nd); }
            d(i,j,k) = a(i,j,k) + 2.0_rt*rc*dr*inv_dz2 + dv;
        });
    }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(diag_z, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const & d = diag_z.array(mfi);
        auto const & a = acoef_z.const_array(mfi);
        auto const & g = gam.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (i >= zhi.x) { d(i,j,k) = 1.0_rt; return; }
            Real rdiag = rmin + (i + 0.5_rt)*dr;
            if (i > zlo.x) { rdiag += rmin + (i - 0.5_rt)*dr; }
            Real const gsum = ((g(i,j,k) > 0.0_rt) ? 1.0_rt : 0.0_rt)
                            + ((g(i,j+1,k) > 0.0_rt) ? 1.0_rt : 0.0_rt);
            d(i,j,k) = a(i,j,k) + rdiag*inv_dr
                + gsum*PolNodeVol(i, rmin, dr, ihi_nd)*inv_dz2;
        });
    }

    // One-shot discrete verification (hybrid_pic_model.esolve_pol_verify):
    // fused rows on a deterministic pseudo-random poloidal field vs the
    // directly composed curl(curl .) and -grad(gamma div .) through
    // STORED cell-centered W and nodal D fields -- the composition reads
    // the stored ghosts filled by the boundary exchange (periodic z) and
    // zeroed r ghosts (axis/wall convention), so it checks the fused
    // kernel's staggering, metric factors, and boundary gating
    // independently of its inline arithmetic.
    if (m_esolve_pol_verify && !m_esolve_pol_verified) {
        m_esolve_pol_verified = true;
        MultiFab Fr(Er.boxArray(), dm, 1, IntVect(1));
        MultiFab Fz(Ez.boxArray(), dm, 1, IntVect(1));
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Fr, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto const & fr = Fr.array(mfi);
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (!z_periodic && (j <= rlo.y || j >= rhi.y)) {
                    fr(i,j,k) = 0.0_rt; return;
                }
                // periodic z identifies the duplicated nodal DOFs j = lo
                // and j = hi: hash the wrapped index so the test field is
                // single-valued (an inconsistent pair would make the
                // fused/composed paths legitimately disagree through the
                // ghost exchange)
                int jw = j;
                if (z_periodic) {
                    int const nzc = rhi.y - rlo.y;
                    jw = rlo.y + (((j - rlo.y) % nzc) + nzc) % nzc;
                }
                auto h = static_cast<unsigned int>(i*73856093)
                       ^ static_cast<unsigned int>(jw*19349663)
                       ^ 0x9e3779b9u;
                h = h*1103515245u + 12345u;
                fr(i,j,k) = Real((h >> 8) & 0xFFFFu)/32768.0_rt - 1.0_rt;
            });
        }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Fz, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto const & fz = Fz.array(mfi);
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (i >= zhi.x) { fz(i,j,k) = 0.0_rt; return; }
                auto h = static_cast<unsigned int>(i*73856093)
                       ^ static_cast<unsigned int>(j*19349663)
                       ^ 0x85ebca6bu;
                h = h*1103515245u + 12345u;
                fz(i,j,k) = Real((h >> 8) & 0xFFFFu)/32768.0_rt - 1.0_rt;
            });
        }
        MultiFab Ar(Er.boxArray(), dm, 1, 0);
        MultiFab Az(Ez.boxArray(), dm, 1, 0);
        apply_op(Ar, Az, Fr, Fz);
        MultiFab W(warpx.boxArray(lev), dm, 1, IntVect(1));
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(W, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto const & w = W.array(mfi);
            auto const & fr = Fr.const_array(mfi);
            auto const & fz = Fz.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                w(i,j,k) = (fr(i,j+1,k) - fr(i,j,k))*inv_dz
                         - (fz(i+1,j,k) - fz(i,j,k))*inv_dr;
            });
        }
        W.setBndry(0.0_rt);
        W.FillBoundary(geom.periodicity());
        // stored gated divergence D = gamma div F on nodes (the closure's
        // composed reference)
        MultiFab D(gam.boxArray(), dm, 1, 1);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(D, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto const & dv = D.array(mfi);
            auto const & fr = Fr.const_array(mfi);
            auto const & fz = Fz.const_array(mfi);
            auto const & g = gam.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (!(g(i,j,k) > 0.0_rt)) { dv(i,j,k) = 0.0_rt; return; }
                Real f = 0.0_rt;
                if (i <= rhi.x) {
                    f += (rmin + (i + 0.5_rt)*dr)*fr(i,j,k);
                }
                if (i > 0) {
                    f -= (rmin + (i - 0.5_rt)*dr)*fr(i-1,j,k);
                }
                dv(i,j,k) = f/PolNodeVol(i, rmin, dr, ihi_nd)
                    + (fz(i,j,k) - fz(i,j-1,k))*inv_dz;
            });
        }
        D.setBndry(0.0_rt);
        D.FillBoundary(geom.periodicity());
        MultiFab Rr(Er.boxArray(), dm, 1, 0);
        MultiFab Rz(Ez.boxArray(), dm, 1, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Rr, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto const & o = Rr.array(mfi);
            auto const & fr = Fr.const_array(mfi);
            auto const & w = W.const_array(mfi);
            auto const & a = acoef_r.const_array(mfi);
            auto const & dv = D.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (!z_periodic && (j <= rlo.y || j >= rhi.y)) {
                    o(i,j,k) = 0.0_rt; return;
                }
                Real const rc = rmin + (i + 0.5_rt)*dr;
                o(i,j,k) = a(i,j,k)*fr(i,j,k)
                    - rc*dr*(w(i,j,k) - w(i,j-1,k))*inv_dz
                    + rc*(dv(i,j,k) - dv(i+1,j,k));
            });
        }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Rz, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto const & o = Rz.array(mfi);
            auto const & fz = Fz.const_array(mfi);
            auto const & w = W.const_array(mfi);
            auto const & a = acoef_z.const_array(mfi);
            auto const & dv = D.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (i >= zhi.x) { o(i,j,k) = 0.0_rt; return; }
                Real const rcp = rmin + (i + 0.5_rt)*dr;
                Real cc = rcp*w(i,j,k);
                if (i > zlo.x) {
                    cc -= (rmin + (i - 0.5_rt)*dr)*w(i-1,j,k);
                }
                cc += PolNodeVol(i, rmin, dr, ihi_nd)
                    *(dv(i,j,k) - dv(i,j+1,k))*inv_dz;
                o(i,j,k) = a(i,j,k)*fz(i,j,k) + cc;
            });
        }
        MultiFab::Subtract(Ar, Rr, 0, 0, 1, 0);
        MultiFab::Subtract(Az, Rz, 0, 0, 1, 0);
        Real const err_r = Ar.norminf(0)
            / amrex::max(Rr.norminf(0), 1.0e-300_rt);
        Real const err_z = Az.norminf(0)
            / amrex::max(Rz.norminf(0), 1.0e-300_rt);
        amrex::Print() << "[SolveEPolCurlCurlRZ] discrete verify (fused vs "
            "composed curl(curl .)): max rel row error r-sector = " << err_r
            << ", z-sector = " << err_z << "\n";
    }

    // Jacobi-preconditioned CG on the coupled two-field system. rhs is
    // orthogonal to any residual curl-curl null content by the beta = 0
    // row zeroing above; the closure makes the free rows SPD wherever
    // gamma covers the vacuum (see the gate comment).
    auto const ng0 = IntVect(0);
    MultiFab Ax_r(Er.boxArray(), dm, 1, 0);
    MultiFab Ax_z(Ez.boxArray(), dm, 1, 0);
    MultiFab res_r(Er.boxArray(), dm, 1, 0);
    MultiFab res_z(Ez.boxArray(), dm, 1, 0);
    MultiFab zz_r(Er.boxArray(), dm, 1, 0);
    MultiFab zz_z(Ez.boxArray(), dm, 1, 0);
    MultiFab p_r(Er.boxArray(), dm, 1, IntVect(1));
    MultiFab p_z(Ez.boxArray(), dm, 1, IntVect(1));

    apply_op(Ax_r, Ax_z, Er, Ez);
    MultiFab::LinComb(res_r, 1.0_rt, rhs_r, 0, -1.0_rt, Ax_r, 0, 0, 1, ng0);
    MultiFab::LinComb(res_z, 1.0_rt, rhs_z, 0, -1.0_rt, Ax_z, 0, 0, 1, ng0);
    MultiFab::Copy(zz_r, res_r, 0, 0, 1, ng0);
    MultiFab::Divide(zz_r, diag_r, 0, 0, 1, 0);
    MultiFab::Copy(zz_z, res_z, 0, 0, 1, ng0);
    MultiFab::Divide(zz_z, diag_z, 0, 0, 1, 0);
    p_r.setVal(0.0_rt);
    p_z.setVal(0.0_rt);
    MultiFab::Copy(p_r, zz_r, 0, 0, 1, ng0);
    MultiFab::Copy(p_z, zz_z, 0, 0, 1, ng0);
    Real rz_old = MultiFab::Dot(res_r, 0, zz_r, 0, 1, 0)
                + MultiFab::Dot(res_z, 0, zz_z, 0, 1, 0);
    Real const rhs_norm = std::sqrt(
        MultiFab::Dot(rhs_r, 0, rhs_r, 0, 1, 0)
      + MultiFab::Dot(rhs_z, 0, rhs_z, 0, 1, 0));
    // Tight tolerance: this solve runs inside every residual evaluation
    // including finite-difference Jacobian probes (the Darwin E_L
    // precedent) -- solver noise above the probe scale poisons secants.
    Real const tol = 1.0e-11_rt * amrex::max(rhs_norm, 1.0e-300_rt);
    // Iteration budget: a half-vacuum column deck (32 x 128) measures
    // multi-thousand Jacobi-CG iterations per cold solve -- the vacuum
    // block is a bare vector Laplacian and plain Jacobi does not see its
    // long-wavelength modes. Uniform-plasma decks converge in O(100)
    // (beta-dominant rows). The factored (block-banded direct) backend is
    // the production follow-up, as for the toroidal stage.
    int const max_cg = 25000;
    int it = 0;
    for (; it < max_cg; ++it) {
        Real const res_norm = std::sqrt(
            MultiFab::Dot(res_r, 0, res_r, 0, 1, 0)
          + MultiFab::Dot(res_z, 0, res_z, 0, 1, 0));
        if (res_norm <= tol) { break; }
        apply_op(Ax_r, Ax_z, p_r, p_z);
        Real const pAp = MultiFab::Dot(p_r, 0, Ax_r, 0, 1, 0)
                       + MultiFab::Dot(p_z, 0, Ax_z, 0, 1, 0);
        Real const alpha = rz_old / pAp;
        MultiFab::Saxpy(Er, alpha, p_r, 0, 0, 1, ng0);
        MultiFab::Saxpy(Ez, alpha, p_z, 0, 0, 1, ng0);
        MultiFab::Saxpy(res_r, -alpha, Ax_r, 0, 0, 1, ng0);
        MultiFab::Saxpy(res_z, -alpha, Ax_z, 0, 0, 1, ng0);
        MultiFab::Copy(zz_r, res_r, 0, 0, 1, ng0);
        MultiFab::Divide(zz_r, diag_r, 0, 0, 1, 0);
        MultiFab::Copy(zz_z, res_z, 0, 0, 1, ng0);
        MultiFab::Divide(zz_z, diag_z, 0, 0, 1, 0);
        Real const rz_new = MultiFab::Dot(res_r, 0, zz_r, 0, 1, 0)
                          + MultiFab::Dot(res_z, 0, zz_z, 0, 1, 0);
        Real const beta_cg = rz_new / rz_old;
        rz_old = rz_new;
        MultiFab::LinComb(p_r, 1.0_rt, zz_r, 0, beta_cg, p_r, 0, 0, 1, ng0);
        MultiFab::LinComb(p_z, 1.0_rt, zz_z, 0, beta_cg, p_z, 0, 0, 1, ng0);
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(it < max_cg,
        "SolveEPolCurlCurlRZ: CG failed to converge");
    Er.FillBoundary(geom.periodicity());
    Ez.FillBoundary(geom.periodicity());
#endif
}

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
        // grad(phi) below). With electron inertia the inertial field joins
        // the source, so its longitudinal part lands in E_L and the
        // transverse state stays clean of gradient content (the structural
        // cancellation this projection exists for). NOTE: the zero
        // extension past non-periodic walls (below) truncates the inertial
        // part of S at the wall layer; on wall-bounded decks with a
        // floored halo at the wall this can leave a wall-sheet residue in
        // E_T at the inertial-term amplitude -- watch wall-adjacent B in
        // validations (proper per-boundary-type E_L conditions remain the
        // documented follow-up).
        amrex::MultiFab const * Ei_nodal = m_include_electron_inertia
            ? warpx.m_fields.get("hybrid_E_inertial_nodal", lev) : nullptr;
        const amrex::GpuArray<int, 3> E_stag_arr[3] =
            {Ex_IndexType, Ey_IndexType, Ez_IndexType};
        amrex::GpuArray<int, 3> const coarsen_rr = {1, 1, 1};
        amrex::GpuArray<int, 3> const nodal_st = {1, 1, 1};
        using namespace ablastr::coarsen::sample;
        for (int dir = 0; dir < 3; ++dir) {
            amrex::MultiFab & Sd = *E_long[dir];
            int const d = grad_dim[dir];
            if (d < 0) { Sd.setVal(0.0_rt); continue; }
            amrex::Real const inv_dx = 1.0_rt / dx_arr[d];
            amrex::IntVect const off = grad_off[dir];
            amrex::GpuArray<int, 3> const Estag = E_stag_arr[dir];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(Sd, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                auto const & S_arr   = Sd.array(mfi);
                auto const & Pe_arr  = Pe.const_array(mfi);
                auto const & rho_arr = rho[lev]->const_array(mfi);
                amrex::Array4<amrex::Real const> ei;
                if (Ei_nodal) { ei = Ei_nodal->const_array(mfi); }
                const bool add_inertia = (Ei_nodal != nullptr);
                const int comp = dir;
                amrex::ParallelFor(mfi.tilebox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
                    amrex::IntVect const ivp = iv + off;
                    amrex::Real const rho_edge = amrex::max(
                        0.5_rt*(rho_arr(iv) + rho_arr(ivp)), rho_floor);
                    S_arr(iv) = -(Pe_arr(ivp) - Pe_arr(iv)) * inv_dx / rho_edge;
                    if (add_inertia) {
                        S_arr(iv) += Interp(ei, nodal_st, Estag, coarsen_rr,
                                            i, j, k, comp);
                    }
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

void HybridPICModel::ComputeVacuumARecovery (bool a_from_jacobian,
                                             amrex::Real a_dt_eff)
{
#if defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    // Blocked at parse time (see ReadParameters); the FD nodal operator has
    // no 1D form.
    amrex::ignore_unused(a_dt_eff);
    WARPX_ABORT_WITH_MESSAGE(
        "hybrid_pic_model.darwin_vacuum_recovery is not supported in 1D "
        "geometries.");
#else
    ABLASTR_PROFILE("HybridPICModel::ComputeVacuumARecovery()");
    // Finite response time: apply only the relaxed fraction of the
    // correction (1 at tau = 0, i.e. instant replacement).
    amrex::Real const omega = (m_darwin_vacrec_relax_time > 0.0_rt)
        ? 1.0_rt - std::exp(-a_dt_eff / m_darwin_vacrec_relax_time)
        : 1.0_rt;

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
    // Recompute the recovery inside Jacobian probe evaluations instead of
    // reusing the frozen correction (WARPX_VACREC_LIVE_PROBES, or forced
    // by circuit-in-the-residual solvers). The frozen-probe lag costs
    // ~0.99/iteration Newton contraction on violent decks; with exact
    // (mass-matrix) particle JVPs the FD-noise objection to differencing
    // through the solve reduces to the recovery MLMG's own tolerance --
    // run with a tight darwin_vacuum_recovery_relative_tolerance
    // (<= 1e-10).
    const bool live_probes = m_vacuum_recovery_live_probes;
    const bool frozen_probe = a_from_jacobian && !live_probes;

    // Mask policy on the step-entry charge density. With the frozen mask
    // (default) the masks read the per-step snapshot captured at implicit
    // step entry, so membership cannot flicker between residual
    // evaluations; the legacy path reads rho_fp component 0 live, which
    // under the implicit scheme is a pre-push deposit that follows the
    // iterate from the second evaluation on (see the
    // m_darwin_vacuum_recovery_frozen_mask member documentation).
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
    amrex::MultiFab const & rho = m_darwin_vacuum_recovery_frozen_mask
        ? *warpx.m_fields.get("hybrid_rho_vacmask_fp", lev)
        : *warpx.m_fields.get(FieldType::rho_fp, lev);

    // The field-implied current J_imp = curl(curl A)/mu0 on the E staggering.
    // Bfield_fp is curl scratch here (the caller rebuilds B = B_static +
    // curl A right after the recovery); zero-extension past non-periodic
    // walls mirrors the E_L source convention.
    // FD-Jacobian probes must not difference through the iterative solve
    // (its rtol-level noise is amplified by the 1/(theta dt) scaling of the
    // band field rows): they reuse the correction from the last
    // non-Jacobian evaluation, keeping the probed map smooth and affine.
    // (Unless WARPX_VACREC_LIVE_PROBES is set -- see above.)
    if (!frozen_probe) {
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

    // "curlcurl" iteration operator: the native edge-staggered
    // amrex::MLCurlCurl backend (Cartesian-only, parse-guarded; see the
    // m_darwin_vacuum_recovery_operator member documentation). J_imp was
    // computed above on the shared path.
    if (m_darwin_vacuum_recovery_operator == "curlcurl") {
        VacuumRecoveryCurlCurlSolve(frozen_probe, omega, rho, rho_floor,
                                    mask_mode, flux_only);
        return;
    }

    for (int dir = 0; dir < 3; ++dir) {
        if (flux_only && dir != 1) { continue; }
        amrex::MultiFab & dAd = *dA[dir];
        amrex::MultiFab const & Jd = *Jvac[dir];
        amrex::GpuArray<int, 3> const Jstag = A_stag[dir];

        if (frozen_probe) {
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
                a_arr(i, j, k) += omega
                    * Interp(dA_arr, nodal, Astag, coarsen_rr, i, j, k, 0);
            });
        }
        Ad.FillBoundary(geom.periodicity());
    }
#endif
}

void HybridPICModel::VacuumRecoveryCurlCurlSolve (bool a_frozen_probe,
                                                  amrex::Real a_omega,
                                                  amrex::MultiFab const& a_rho,
                                                  amrex::Real a_rho_floor,
                                                  int a_mask_mode,
                                                  bool a_flux_only)
{
#if defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE) || defined(WARPX_DIM_RZ)
    amrex::ignore_unused(a_frozen_probe, a_omega, a_rho, a_rho_floor,
                         a_mask_mode, a_flux_only);
    WARPX_ABORT_WITH_MESSAGE(
        "darwin_vacuum_recovery_operator = curlcurl requires Cartesian "
        "geometry (parse-guarded; this call is unreachable).");
#else
    ABLASTR_PROFILE("HybridPICModel::VacuumRecoveryCurlCurlSolve()");
    using namespace amrex;
    using namespace ablastr::coarsen::sample;

    auto & warpx = WarpX::GetInstance();
    constexpr int lev = 0;
    Geometry const & geom = warpx.Geom(lev);
    const bool trace = (std::getenv("WARPX_DEBUG_VACREC") != nullptr);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!EB::enabled(),
        "darwin_vacuum_recovery_operator = curlcurl does not support "
        "embedded boundaries (amrex::MLCurlCurl has no EB form).");

    ablastr::fields::VectorField A =
        warpx.m_fields.get_alldirs("hybrid_A_fp", lev);
    ablastr::fields::VectorField Jvac =
        warpx.m_fields.get_alldirs("hybrid_J_vac_fp", lev);
    ablastr::fields::VectorField dA =
        warpx.m_fields.get_alldirs("hybrid_A_vac_dA_edge", lev);

    amrex::GpuArray<int, 3> const coarsen_rr = {1, 1, 1};
    amrex::GpuArray<int, 3> const nodal = {1, 1, 1};
    const amrex::GpuArray<int, 3> A_stag[3] =
        {Ex_IndexType, Ey_IndexType, Ez_IndexType};

    // Domain extents/periodicity for the boundary-edge guard (the
    // homogeneous-Dirichlet correction rows carry no source, matching the
    // Poisson path's boundary-node convention).
    const amrex::Box dom_cc = geom.Domain();
    amrex::GpuArray<int, 3> per{{1, 1, 1}};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        per[d] = geom.isPeriodic(d) ? 1 : 0;
    }

    const amrex::Real rho_floor = a_rho_floor;
    const int mask_mode = a_mask_mode;

    if (!a_frozen_probe) {
        // rhs = -mu0 J_imp on the edge staggering, source-masked, zero on
        // non-periodic boundary edges; at the fixed point of the recovery
        // iteration the masked J_imp (composed curl(curl A)) vanishes
        // identically, exactly as on the Poisson path.
        std::array<std::unique_ptr<amrex::MultiFab>, 3> rhs_v;
        amrex::Real rhs_max = 0.0_rt;
        amrex::Real dA_max = 0.0_rt;
        for (int dir = 0; dir < 3; ++dir) {
            rhs_v[dir] = std::make_unique<amrex::MultiFab>(
                dA[dir]->boxArray(), dA[dir]->DistributionMap(), 1,
                dA[dir]->nGrowVect());
            rhs_v[dir]->setVal(0.0_rt);
            if (a_flux_only && dir != 1) { continue; }
            amrex::MultiFab const & Jd = *Jvac[dir];
            const amrex::Box dom_e = amrex::convert(
                dom_cc, dA[dir]->ixType().toIntVect());
            amrex::GpuArray<int, 3> dlo{{0, 0, 0}};
            amrex::GpuArray<int, 3> dhi{{0, 0, 0}};
            amrex::GpuArray<int, 3> is_nd{{0, 0, 0}};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                dlo[d] = dom_e.smallEnd(d);
                dhi[d] = dom_e.bigEnd(d);
                is_nd[d] = dA[dir]->ixType().nodeCentered(d) ? 1 : 0;
            }
            amrex::GpuArray<int, 3> const stag = A_stag[dir];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(*rhs_v[dir], TilingIfNotGPU());
                 mfi.isValid(); ++mfi) {
                auto const & rhs_arr = rhs_v[dir]->array(mfi);
                auto const & J_arr   = Jd.const_array(mfi);
                auto const & rho_arr = a_rho.const_array(mfi);
                amrex::ParallelFor(mfi.tilebox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    const int idx[3] = {i, j, k};
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        // Dirichlet rows: edge points lying in a
                        // non-periodic domain-boundary face (nodal index
                        // at either end).
                        if (!per[d] && is_nd[d]
                            && (idx[d] <= dlo[d] || idx[d] >= dhi[d])) {
                            rhs_arr(i, j, k) = 0.0_rt;
                            return;
                        }
                    }
                    amrex::Real const rho_val =
                        Interp(rho_arr, nodal, stag, coarsen_rr, i, j, k, 0);
                    const bool in_mask = (mask_mode == 2)
                        || (mask_mode == 0 && rho_val < rho_floor)
                        || (mask_mode == 1 && rho_val > 0.0_rt
                            && rho_val < rho_floor);
                    rhs_arr(i, j, k) = in_mask
                        ? -PhysConst::mu0 * J_arr(i, j, k)
                        : 0.0_rt;
                });
            }
            rhs_max = amrex::max(rhs_max, rhs_v[dir]->norminf(0));
            dA_max = amrex::max(dA_max, dA[dir]->norminf(0));
        }

        // Empty mask and no stored correction: exact identity, skip.
        if (rhs_max == 0.0_rt && dA_max == 0.0_rt) {
            if (trace) {
                amrex::Print() << "[vac-recover] (curlcurl) empty mask, "
                    "skipped\n";
            }
            return;
        }

        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lobc, hibc;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            lobc[d] = geom.isPeriodic(d) ? amrex::LinOpBCType::Periodic
                                         : amrex::LinOpBCType::Dirichlet;
            hibc[d] = lobc[d];
        }

        amrex::LPInfo const info;
        amrex::MLCurlCurl linop({geom}, {warpx.boxArray(lev)},
                                {warpx.DistributionMap(lev)}, info);
        linop.setDomainBC(lobc, hibc);
        // Tikhonov closure of the curl-curl null space (gradient family):
        // beta relative to the grid-scale curl-curl diagonal. Shapes the
        // iteration only -- the fixed point (masked J_imp = 0) does not
        // move, and gradient-family dA content never reaches B = curl A.
        amrex::Real kg2 = 0.0_rt;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            kg2 += 4.0_rt / (geom.CellSize(d) * geom.CellSize(d));
        }
        linop.setScalars(1.0_rt, m_darwin_vacrec_cc_beta_rel * kg2);

        // Warm start: the persistent edge correction (Dirichlet rows stay
        // zero by construction).
        for (int dir = 0; dir < 3; ++dir) {
            dA[dir]->setBndry(0.0_rt);
            dA[dir]->FillBoundary(geom.periodicity());
        }

        // WarpX -> AMReX component slots (see CurlCurlMLMGPC):
        // 2D: missing dim is y in WarpX, z in AMReX -> {0, 2, 1}.
#if defined(WARPX_DIM_3D)
        const int slot[3] = {0, 1, 2};
#else
        const int slot[3] = {0, 2, 1};
#endif
        amrex::Array<amrex::MultiFab, 3> sol
            {amrex::MultiFab(*dA[slot[0]], amrex::make_alias, 0, 1),
             amrex::MultiFab(*dA[slot[1]], amrex::make_alias, 0, 1),
             amrex::MultiFab(*dA[slot[2]], amrex::make_alias, 0, 1)};
        amrex::Array<amrex::MultiFab, 3> rhs
            {amrex::MultiFab(*rhs_v[slot[0]], amrex::make_alias, 0, 1),
             amrex::MultiFab(*rhs_v[slot[1]], amrex::make_alias, 0, 1),
             amrex::MultiFab(*rhs_v[slot[2]], amrex::make_alias, 0, 1)};

        amrex::MLMGT<amrex::Array<amrex::MultiFab, 3>> mlmg(linop);
        mlmg.setVerbose(trace ? std::max(m_darwin_vacrec_verbosity, 2)
                              : m_darwin_vacrec_verbosity);
        mlmg.setMaxIter(m_darwin_vacrec_max_iters);
        mlmg.setThrowException(true);
        linop.prepareRHS({&rhs});
        amrex::Real resid = 0.0_rt;
        bool solve_ok = true;
        std::string fail_msg;
        try {
            resid = mlmg.solve({&sol}, {&rhs},
                               m_darwin_vacrec_rtol, m_darwin_vacrec_atol);
        } catch (std::exception const & e) {
            // Degrade to the identity for this evaluation (as on the
            // Poisson path) instead of aborting the nonlinear iteration.
            for (int dir = 0; dir < 3; ++dir) { dA[dir]->setVal(0.0_rt); }
            solve_ok = false;
            fail_msg = e.what();
        }
        if (trace) {
            amrex::Print() << "[vac-recover] (curlcurl) |rhs|max = "
                << rhs_max << " |dA|max = ("
                << dA[0]->norminf(0) << ", " << dA[1]->norminf(0) << ", "
                << dA[2]->norminf(0) << ") resid = " << resid
                << (solve_ok ? "" : (" (SOLVE FAILED: " + fail_msg + ")"))
                << "\n";
        }
        if (!solve_ok) { return; }
        for (int dir = 0; dir < 3; ++dir) {
            dA[dir]->FillBoundary(geom.periodicity());
        }
    } else {
        // Frozen probe: nothing to apply if the stored correction is zero
        // (preserves the exact-identity property bit-for-bit).
        amrex::Real dA_max = 0.0_rt;
        for (int dir = 0; dir < 3; ++dir) {
            dA_max = amrex::max(dA_max, dA[dir]->norminf(0));
        }
        if (dA_max == 0.0_rt) { return; }
    }

    // A += omega dA at the component's edge points, replace-masked (same
    // staggering: no interpolation). The caller re-applies the A boundary
    // values afterwards.
    for (int dir = 0; dir < 3; ++dir) {
        if (a_flux_only && dir != 1) { continue; }
        amrex::MultiFab & Ad = *A[dir];
        amrex::MultiFab const & dAd = *dA[dir];
        amrex::GpuArray<int, 3> const stag = A_stag[dir];
        const amrex::Real omega = a_omega;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Ad, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const & a_arr   = Ad.array(mfi);
            auto const & dA_arr  = dAd.const_array(mfi);
            auto const & rho_arr = a_rho.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real const rho_val =
                    Interp(rho_arr, nodal, stag, coarsen_rr, i, j, k, 0);
                const bool in_mask = (mask_mode == 2)
                    || (mask_mode == 0 && rho_val < rho_floor)
                    || (mask_mode == 1 && rho_val > 0.0_rt
                        && rho_val < rho_floor);
                if (!in_mask) { return; }
                a_arr(i, j, k) += omega * dA_arr(i, j, k);
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

    // Same frozen-vs-live mask density selection as ComputeVacuumARecovery
    // (the two consumers straddle the per-evaluation deposit: a live read
    // hands them INCONSISTENT masks within one evaluation).
    amrex::MultiFab const & rho = m_darwin_vacuum_recovery_frozen_mask
        ? *warpx.m_fields.get("hybrid_rho_vacmask_fp", lev)
        : *warpx.m_fields.get(FieldType::rho_fp, lev);
    amrex::GpuArray<int, 3> const coarsen_rr = {1, 1, 1};
    amrex::GpuArray<int, 3> const nodal = {1, 1, 1};
    const amrex::GpuArray<int, 3> A_stag[3] =
        {Ex_IndexType, Ey_IndexType, Ez_IndexType};
    // Live-probe mode (see ComputeVacuumARecovery): probes evaluate the
    // band field from the live recovered A instead of the stored target.
    const bool live_probes = m_vacuum_recovery_live_probes;

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
            const bool from_jac = a_from_jacobian && !live_probes;
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


void HybridPICModel::ResolveElectronInertiaMass ()
{
    // Effective electron mass: the lightest ion species divided by the
    // reduced mass ratio (physical m_e when the ratio is unset). The ratio
    // moves the effective electron skin depth d_e = c/omega_pe(m_e_eff)
    // relative to the grid. Must run before any consumer of
    // m_electron_inertia_mass (the curlcurl_form elliptic solves read it
    // through beta_fac and divide by it).
    if (m_electron_inertia_mass != 0.0_rt) { return; }
    if (m_reduced_electron_mass_ratio > 0.0_rt) {
        amrex::Real m_ion_min = std::numeric_limits<amrex::Real>::max();
        auto & mypc = WarpX::GetInstance().GetPartContainer();
        for (int isp = 0; isp < mypc.nSpecies(); ++isp) {
            auto & pc = mypc.GetParticleContainer(isp);
            if (pc.getCharge() != 0.0_rt) {
                m_ion_min = std::min(m_ion_min, pc.getMass());
            }
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_ion_min < std::numeric_limits<amrex::Real>::max(),
            "reduced_electron_mass_ratio requires at least one charged "
            "ion species");
        m_electron_inertia_mass = m_ion_min / m_reduced_electron_mass_ratio;
    } else {
        m_electron_inertia_mass = PhysConst::m_e;
    }
    amrex::Print() << "[HybridPICModel] electron inertia: m_e_eff = "
        << m_electron_inertia_mass << " kg (mass ratio "
        << m_reduced_electron_mass_ratio << ")\n";
}

void HybridPICModel::EnsureCurlCurlScratch ()
{
    if (!m_esolve_curlcurl || m_jp_old_nodal) { return; }
    ResolveElectronInertiaMass();
    using ablastr::fields::Direction;
    auto & warpx = WarpX::GetInstance();
    constexpr int lev = 0;
    ablastr::fields::VectorField const Jp =
        warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    amrex::MultiFab const & Ei =
        *warpx.m_fields.get("hybrid_E_inertial_nodal", lev);
    for (int c = 0; c < 3; ++c) {
        m_jp_old_stag[c] = std::make_unique<amrex::MultiFab>(
            Jp[c]->boxArray(), Jp[c]->DistributionMap(), 1,
            Jp[c]->nGrowVect());
        m_jp_old_stag[c]->setVal(0.0);
    }
    m_jp_old_nodal = std::make_unique<amrex::MultiFab>(
        Ei.boxArray(), Ei.DistributionMap(), 3, amrex::IntVect(1));
    m_jp_old_nodal->setVal(0.0);
    // Zero measured-inertia numerator = the exact quiescent pre-history:
    // solves that run before the first inertia assembly (the resistive
    // push-field correction pair) read it.
    m_ei_curlcurl_theta = std::make_unique<amrex::MultiFab>(
        Ei.boxArray(), Ei.DistributionMap(), 3, amrex::IntVect(1));
    m_ei_curlcurl_theta->setVal(0.0);
    m_num_theta = std::make_unique<amrex::MultiFab>(
        Ei.boxArray(), Ei.DistributionMap(), 1, amrex::IntVect(1));
    m_num_theta->setVal(0.0);
    amrex::MultiFab const & Er_mf = *warpx.m_fields.get(
        FieldType::Efield_fp, Direction{0}, lev);
    amrex::MultiFab const & Ez_mf = *warpx.m_fields.get(
        FieldType::Efield_fp, Direction{2}, lev);
    m_num_pol[0] = std::make_unique<amrex::MultiFab>(
        Er_mf.boxArray(), Er_mf.DistributionMap(), 1, amrex::IntVect(1));
    m_num_pol[0]->setVal(0.0);
    m_num_pol[1] = std::make_unique<amrex::MultiFab>(
        Ez_mf.boxArray(), Ez_mf.DistributionMap(), 1, amrex::IntVect(1));
    m_num_pol[1]->setVal(0.0);
}

void HybridPICModel::EnsureTensorScratch () const
{
    if (!m_esolve_tensor || m_tensor_jp_old[0]) { return; }
    const_cast<HybridPICModel*>(this)->ResolveElectronInertiaMass();
    using ablastr::fields::Direction;
    auto & warpx = WarpX::GetInstance();
    constexpr int lev = 0;
    for (int c = 0; c < 3; ++c) {
        amrex::MultiFab const & E = *warpx.m_fields.get(
            FieldType::Efield_fp, Direction{c}, lev);
        m_tensor_jp_old[c] = std::make_unique<amrex::MultiFab>(
            E.boxArray(), E.DistributionMap(), 1, E.nGrowVect());
        // Zero = the exact quiescent pre-history for any solve that runs
        // before the first step-entry capture.
        m_tensor_jp_old[c]->setVal(0.0);
    }
    // Per-step-frozen density snapshot for the RZ tensor assembly (see
    // m_tensor_rho_old): every rho-derived coefficient AND binary gate of
    // the RZ solve reads this one bit-frozen field, so gate membership and
    // screening rows cannot flip between residual evaluations (the item-1
    // iron-rule discipline; the 1D path keeps the legacy live comp-0 read).
    {
        amrex::MultiFab const & rho_mf = *warpx.m_fields.get(
            FieldType::rho_fp, lev);
        m_tensor_rho_old = std::make_unique<amrex::MultiFab>(
            rho_mf.boxArray(), rho_mf.DistributionMap(), 1,
            rho_mf.nGrowVect());
        m_tensor_rho_old->setVal(0.0);
    }
}

void HybridPICModel::CaptureTensorStepStart ()
{
    if (!m_esolve_tensor) { return; }
    ABLASTR_PROFILE("HybridPICModel::CaptureTensorStepStart()");
    EnsureTensorScratch();
    using ablastr::fields::Direction;
    auto & warpx = WarpX::GetInstance();
    constexpr int lev = 0;
    // Bfield_fp holds the committed B^n at step entry; refresh the plasma
    // current register from it (J_ext-aware) and snapshot it. The register
    // is recomputed at the theta stage by every residual evaluation, so
    // the copy is the only per-step-frozen carrier.
    CalculatePlasmaCurrent(
        warpx.m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp,
                                             warpx.finestLevel()),
        warpx.GetEBUpdateEFlag());
    for (int c = 0; c < 3; ++c) {
        amrex::MultiFab const & Jp = *warpx.m_fields.get(
            FieldType::hybrid_current_fp_plasma, Direction{c}, lev);
        amrex::MultiFab::Copy(*m_tensor_jp_old[c], Jp, 0, 0, 1,
                              amrex::min(m_tensor_jp_old[c]->nGrowVect(),
                                         Jp.nGrowVect()));
    }
    // Frozen density snapshot: component 0 of rho_fp holds the committed
    // entry deposit at step entry (the darwin vacmask / A-recovery mask
    // family). Bit-frozen through every residual evaluation of the step.
    {
        amrex::MultiFab const & rho_mf = *warpx.m_fields.get(
            FieldType::rho_fp, lev);
        amrex::MultiFab::Copy(*m_tensor_rho_old, rho_mf, 0, 0, 1,
                              amrex::min(m_tensor_rho_old->nGrowVect(),
                                         rho_mf.nGrowVect()));
        m_tensor_rho_captured = true;
    }
#if defined(WARPX_DIM_RZ)
    // Exchange + refresh the frozen carriers' ghosts once per step: the RZ
    // assembly Interps them across staggers (the E_z axis row reads the
    // r < 0 ghost of Jp_r, which is odd across the axis for m = 0).
    {
        amrex::Geometry const & geom = warpx.Geom(lev);
        for (int c = 0; c < 3; ++c) {
            m_tensor_jp_old[c]->FillBoundary(geom.periodicity());
        }
        m_tensor_rho_old->FillBoundary(geom.periodicity());
        if (geom.ProbLo(0) == 0.0) {
            amrex::MultiFab & jr = *m_tensor_jp_old[0];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(jr); mfi.isValid(); ++mfi) {
                amrex::Box gbx = mfi.growntilebox();
                if (gbx.smallEnd(0) >= 0) { continue; }
                gbx.setBig(0, -1);
                auto const & arr = jr.array(mfi);
                amrex::ParallelFor(gbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    // J_r is cc in r: the mirror of ghost cell i is cell
                    // -1 - i (odd parity across the axis at m = 0)
                    arr(i, j, k) = -arr(-1 - i, j, k);
                });
            }
        }
    }
#endif
}

namespace
{
    /** Closed-form inverse application of M = a I - [beta]x:
     *  M^{-1} v = (a^2 v + a beta x v + (beta.v) beta) / (a (a^2 + b2)).
     *  a >= 1 by construction, so the division is regular everywhere. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void TensorMinv (amrex::Real a,
                     amrex::Real bx, amrex::Real by, amrex::Real bz,
                     amrex::Real vx, amrex::Real vy, amrex::Real vz,
                     amrex::Real& ox, amrex::Real& oy, amrex::Real& oz)
    {
        amrex::Real const b2 = bx*bx + by*by + bz*bz;
        amrex::Real const inv = 1.0_rt / (a * (a*a + b2));
        amrex::Real const bdv = bx*vx + by*vy + bz*vz;
        ox = (a*a*vx + a*(by*vz - bz*vy) + bdv*bx) * inv;
        oy = (a*a*vy + a*(bz*vx - bx*vz) + bdv*by) * inv;
        oz = (a*a*vz + a*(bx*vy - by*vx) + bdv*bz) * inv;
    }
}

void HybridPICModel::SolveETensor1D (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rho,
    amrex::MultiFab const& Pe,
    int const lev, bool const solve_for_Faraday,
    bool const include_resistivity) const
{
#if !defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(Efield, Jifield, Bfield, rho, Pe, lev,
                         solve_for_Faraday, include_resistivity);
    WARPX_ABORT_WITH_MESSAGE(
        "SolveETensor1D: the tensor_form E-solve is 1D-only in v1");
#else
    ABLASTR_PROFILE("HybridPICModel::SolveETensor1D()");
    using namespace amrex;
    using namespace ablastr::coarsen::sample;

    auto & warpx = WarpX::GetInstance();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(lev == 0,
        "hybrid_pic_model.esolve = tensor_form supports a single level");
    Geometry const & geom = warpx.Geom(lev);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(geom.isPeriodic(0),
        "hybrid_pic_model.esolve = tensor_form (1D) requires periodic z");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_add_external_fields,
        "hybrid_pic_model.esolve = tensor_form does not support split "
        "external fields yet");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_resistivity_has_J_dependence,
        "hybrid_pic_model.esolve = tensor_form does not support "
        "J-dependent resistivity yet");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_include_hyper_resistivity_term,
        "hybrid_pic_model.esolve = tensor_form does not support "
        "hyper-resistivity yet");
    // Transverse components must share the z-nodal stagger (Yee); a
    // collocated grid composes its curls to a different (wide) stencil
    // than the 3-point form assembled here.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        Efield[0]->ixType() == Efield[1]->ixType()
            && Efield[0]->ixType() != Efield[2]->ixType(),
        "hybrid_pic_model.esolve = tensor_form (1D) requires the staggered "
        "(Yee) grid");

    EnsureTensorScratch();

    Real const dz = geom.CellSizeArray()[0];
    Real const inv_dz2 = 1.0_rt/(dz*dz);
    // dt_eff = theta dt, published per step by the theta-implicit solver;
    // solves that run before the first step entry (initialization) fall
    // back to the theta = 1/2 value -- constant within any given step
    // either way.
    Real const dt_eff = (m_tensor_dt_eff > 0.0_rt)
        ? m_tensor_dt_eff : 0.5_rt*warpx.getdt(lev);
    Real const rho_1c = PhysConst::q_e
        * ((m_tensor_n_min > 0.0_rt) ? m_tensor_n_min : m_n_floor);
    Real const me_eff = m_electron_inertia_mass;
    Real const alpha = m_tensor_alpha;
    Real const qe = PhysConst::q_e;
    Real const mu0 = PhysConst::mu0;
    Real const inv_mu0 = 1.0_rt/mu0;
    const auto eta = m_eta;
    Real const t_new = warpx.gett_new(lev);
    const bool use_eta = include_resistivity;
    const bool use_gpe = !solve_for_Faraday
        && m_include_electron_pressure_term;
    // Component 0 = the pre-push deposit: the SAME density the e_form Ohm
    // kernel this path replaces consumes (always valid -- the midpoint
    // component nComp()/2 is empty until the step's first PreRHSOp
    // deposit, but the resistive push-field correction pair solves E
    // inside UpdateWarpXFields BEFORE it; measured: the empty component
    // gives kE = 0 everywhere and the pure-curl-curl periodic transverse
    // system is singular).
    const int rho_mid = 0;

    GpuArray<int, 3> const nodal = {1, 1, 1};
    GpuArray<int, 3> const coarsen_rr = {1, 1, 1};
    const GpuArray<int, 3> E_stag[3] =
        {Ex_IndexType, Ey_IndexType, Ez_IndexType};
    const GpuArray<int, 3> B_stag[3] =
        {Bx_IndexType, By_IndexType, Bz_IndexType};
    const GpuArray<int, 3> J_stag[3] =
        {Jx_IndexType, Jy_IndexType, Jz_IndexType};

    // Per-component pointwise coefficients on the component's stagger:
    // comp 0 = a, 1..3 = beta, 4 = kE; and the RHS.
    std::array<std::unique_ptr<MultiFab>, 3> coef, rhs;
    for (int c = 0; c < 3; ++c) {
        coef[c] = std::make_unique<MultiFab>(
            Efield[c]->boxArray(), Efield[c]->DistributionMap(), 5,
            IntVect(0));
        rhs[c] = std::make_unique<MultiFab>(
            Efield[c]->boxArray(), Efield[c]->DistributionMap(), 1,
            IntVect(0));
    }

    for (int c = 0; c < 3; ++c) {
        GpuArray<int, 3> const Sc = E_stag[c];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*coef[c], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const & cf  = coef[c]->array(mfi);
            auto const & rh  = rhs[c]->array(mfi);
            auto const & rho_arr = rho.const_array(mfi);
            auto const & pe_arr  = Pe.const_array(mfi);
            auto const & bx_arr = Bfield[0]->const_array(mfi);
            auto const & by_arr = Bfield[1]->const_array(mfi);
            auto const & bz_arr = Bfield[2]->const_array(mfi);
            auto const & jix_arr = Jifield[0]->const_array(mfi);
            auto const & jiy_arr = Jifield[1]->const_array(mfi);
            auto const & jiz_arr = Jifield[2]->const_array(mfi);
            auto const & jpx_arr = m_tensor_jp_old[0]->const_array(mfi);
            auto const & jpy_arr = m_tensor_jp_old[1]->const_array(mfi);
            auto const & jpz_arr = m_tensor_jp_old[2]->const_array(mfi);
            const GpuArray<int, 3> Sx = E_stag[0], Sy = E_stag[1],
                Sz = E_stag[2];
            const GpuArray<int, 3> Bx_s = B_stag[0], By_s = B_stag[1],
                Bz_s = B_stag[2];
            const GpuArray<int, 3> Jx_s = J_stag[0], Jy_s = J_stag[1],
                Jz_s = J_stag[2];
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // local density (midpoint deposit) and total B at Sc
                Real const rho_v = amrex::max(0.0_rt,
                    Interp(rho_arr, nodal, Sc, coarsen_rr, i, j, k, rho_mid));
                Real const bx = Interp(bx_arr, Bx_s, Sc, coarsen_rr, i, j, k, 0);
                Real const by = Interp(by_arr, By_s, Sc, coarsen_rr, i, j, k, 0);
                Real const bz = Interp(bz_arr, Bz_s, Sc, coarsen_rr, i, j, k, 0);
                // Eq-(27)-class effective mass: vacuum/statistics
                // regularization of qm (inert wherever resolved)
                Real const B2 = bx*bx + by*by + bz*bz;
                Real const s = dt_eff / (2.0_rt*alpha*dz);
                Real const me_v = amrex::max(me_eff,
                    B2*qe/(mu0*amrex::max(rho_v, rho_1c)) * s*s);
                Real const qm = qe/me_v;
                Real const kE = qm*rho_v;
                Real const eta_v = use_eta
                    ? eta(rho_v, 0.0_rt, t_new) : 0.0_rt;
                Real const a = 1.0_rt + dt_eff*kE*eta_v;
                Real const bex = dt_eff*qm*bx;
                Real const bey = dt_eff*qm*by;
                Real const bez = dt_eff*qm*bz;
                cf(i, j, k, 0) = a;
                cf(i, j, k, 1) = bex;
                cf(i, j, k, 2) = bey;
                cf(i, j, k, 3) = bez;
                cf(i, j, k, 4) = kE;
                // measured currents at Sc: step-start plasma current and
                // the midpoint ion deposit
                Real const jpx = Interp(jpx_arr, Sx, Sc, coarsen_rr, i, j, k, 0);
                Real const jpy = Interp(jpy_arr, Sy, Sc, coarsen_rr, i, j, k, 0);
                Real const jpz = Interp(jpz_arr, Sz, Sc, coarsen_rr, i, j, k, 0);
                Real const jix = Interp(jix_arr, Jx_s, Sc, coarsen_rr, i, j, k, 0);
                Real const jiy = Interp(jiy_arr, Jy_s, Sc, coarsen_rr, i, j, k, 0);
                Real const jiz = Interp(jiz_arr, Jz_s, Sc, coarsen_rr, i, j, k, 0);
                Real const jex = jpx - jix;
                Real const jey = jpy - jiy;
                Real const jez = jpz - jiz;
                // grad Pe on the component's stagger: only the z component
                // carries a gradient in 1D (Pe is z-nodal, E_z z-cell-
                // centered: the two-point difference is the native
                // staggered gradient)
                Real gpx = 0.0_rt, gpy = 0.0_rt, gpz = 0.0_rt;
                if (use_gpe && c == 2) {
                    gpz = (pe_arr(i+1, j, k) - pe_arr(i, j, k))/dz;
                }
                // w = kE eta Jp^n - qm (B x Je_meas) - qm grad Pe
                Real const wx = kE*eta_v*jpx - qm*(by*jez - bz*jey) - qm*gpx;
                Real const wy = kE*eta_v*jpy - qm*(bz*jex - bx*jez) - qm*gpy;
                Real const wz = kE*eta_v*jpz - qm*(bx*jey - by*jex) - qm*gpz;
                Real ox, oy, oz;
                TensorMinv(a, bex, bey, bez, wx, wy, wz, ox, oy, oz);
                // The 1D E_z row has no elliptic part: at kE = 0 (true
                // vacuum) it degenerates -- make it an identity row, E_z = 0
                // (no longitudinal field is minted in vacuum).
                if (c == 2 && !(kE > 0.0_rt)) {
                    rh(i, j, k) = 0.0_rt;
                } else {
                    rh(i, j, k) = (c == 0) ? ox : ((c == 1) ? oy : oz);
                }
            });
        }
    }

    // Matrix-free operator: L E |_c = (1/mu0) curlcurl E |_c
    //                                 + kE [M^{-1} E]_c  at stagger Sc,
    // with curlcurl = -d2/dz2 on the transverse pair (the discrete
    // staggered curl composed with itself) and identically zero on E_z.
    auto apply_op = [&](std::array<MultiFab, 3>& out,
                        std::array<MultiFab, 3>& in)
    {
        for (int c = 0; c < 3; ++c) {
            in[c].FillBoundary(geom.periodicity());
        }
        for (int c = 0; c < 3; ++c) {
            GpuArray<int, 3> const Sc = E_stag[c];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(out[c], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                auto const & o  = out[c].array(mfi);
                auto const & px = in[0].const_array(mfi);
                auto const & py = in[1].const_array(mfi);
                auto const & pz = in[2].const_array(mfi);
                auto const & cf = coef[c]->const_array(mfi);
                const GpuArray<int, 3> Sx = E_stag[0], Sy = E_stag[1],
                    Sz = E_stag[2];
                const int cc = c;
                amrex::ParallelFor(mfi.tilebox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    auto const & pself = (cc == 0) ? px : ((cc == 1) ? py : pz);
                    Real ell = 0.0_rt;
                    if (cc < 2) {
                        ell = -inv_mu0*(pself(i+1, j, k)
                            - 2.0_rt*pself(i, j, k)
                            + pself(i-1, j, k))*inv_dz2;
                    }
                    // degenerate vacuum E_z row: identity (see the RHS
                    // assembly)
                    if (cc == 2 && !(cf(i,j,k,4) > 0.0_rt)) {
                        o(i, j, k) = pself(i, j, k);
                        return;
                    }
                    Real const vx = Interp(px, Sx, Sc, coarsen_rr, i, j, k, 0);
                    Real const vy = Interp(py, Sy, Sc, coarsen_rr, i, j, k, 0);
                    Real const vz = Interp(pz, Sz, Sc, coarsen_rr, i, j, k, 0);
                    Real ox, oy, oz;
                    TensorMinv(cf(i,j,k,0), cf(i,j,k,1), cf(i,j,k,2),
                               cf(i,j,k,3), vx, vy, vz, ox, oy, oz);
                    Real const tv = (cc == 0) ? ox : ((cc == 1) ? oy : oz);
                    o(i, j, k) = ell + cf(i,j,k,4)*tv;
                });
            }
        }
    };

    // Jacobi diagonal: elliptic row diagonal plus the tensor's own
    // diagonal entry kE (a^2 + beta_c^2)/(a (a^2 + b2)).
    std::array<MultiFab, 3> diag;
    for (int c = 0; c < 3; ++c) {
        diag[c].define(Efield[c]->boxArray(),
                       Efield[c]->DistributionMap(), 1, IntVect(0));
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(diag[c], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const & d  = diag[c].array(mfi);
            auto const & cf = coef[c]->const_array(mfi);
            const int cc = c;
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Real const a = cf(i,j,k,0);
                Real const bc = cf(i,j,k,1+cc);
                Real const b2 = cf(i,j,k,1)*cf(i,j,k,1)
                    + cf(i,j,k,2)*cf(i,j,k,2) + cf(i,j,k,3)*cf(i,j,k,3);
                Real const tdiag = cf(i,j,k,4)*(a*a + bc*bc)
                    /(a*(a*a + b2));
                Real const ell = (cc < 2)
                    ? 2.0_rt*inv_mu0*inv_dz2 : 0.0_rt;
                // identity diagonal for the degenerate vacuum E_z row
                d(i,j,k) = (cc == 2 && !(cf(i,j,k,4) > 0.0_rt))
                    ? 1.0_rt : (ell + tdiag);
            });
        }
    }

    // Preconditioned BiCGStab on the 3-component composite (the gyration
    // part of the tensor is antisymmetric: the operator is non-symmetric).
    // Warm start from the incoming E (previous evaluation); tight
    // tolerance -- the solve runs inside FD-Jacobian probe evaluations.
    auto dot3 = [&](std::array<MultiFab, 3>& u, std::array<MultiFab, 3>& v)
    {
        Real s = 0.0_rt;
        for (int c = 0; c < 3; ++c) {
            s += MultiFab::Dot(u[c], 0, v[c], 0, 1, 0);
        }
        return s;
    };
    auto make3 = [&](std::array<MultiFab, 3>& v, int ng)
    {
        for (int c = 0; c < 3; ++c) {
            v[c].define(Efield[c]->boxArray(),
                        Efield[c]->DistributionMap(), 1, IntVect(ng));
            v[c].setVal(0.0_rt);
        }
    };
    auto prec3 = [&](std::array<MultiFab, 3>& z, std::array<MultiFab, 3>& r)
    {
        for (int c = 0; c < 3; ++c) {
            MultiFab::Copy(z[c], r[c], 0, 0, 1, 0);
            MultiFab::Divide(z[c], diag[c], 0, 0, 1, 0);
        }
    };

    // One-shot discrete verification (hybrid_pic_model.tensor_verify):
    // apply_op on a
    // deterministic multi-mode field vs the independently composed
    // (1/mu0) curl(curl F) (two staggered curls through a stored
    // cell-centered G) + kE M^{-1} F with M inverted by a general 3x3
    // adjugate. Checks staggering, signs, and factors of the fused rows.
    if (m_esolve_tensor_verify && !m_esolve_tensor_verified) {
        m_esolve_tensor_verified = true;
        // one-shot debug block: temporaries live in pinned host memory so
        // the host comparison loop below can read them on GPU builds
        // (device-arena fabs would segfault under amrex::LoopOnCpu)
        std::array<MultiFab, 3> F, AF;
        for (int c = 0; c < 3; ++c) {
            F[c].define(Efield[c]->boxArray(),
                        Efield[c]->DistributionMap(), 1, IntVect(1),
                        MFInfo().SetArena(The_Pinned_Arena()));
            AF[c].define(Efield[c]->boxArray(),
                         Efield[c]->DistributionMap(), 1, IntVect(0),
                         MFInfo().SetArena(The_Pinned_Arena()));
        }
        Real const Lz = geom.ProbLength(0);
        Real const zlo0 = geom.ProbLo(0);
        for (int c = 0; c < 3; ++c) {
            GpuArray<int, 3> const Sc = E_stag[c];
            Real const off = (Sc[0] == 1) ? 0.0_rt : 0.5_rt;
            Real const ph = 0.7_rt*(c + 1);
            for (MFIter mfi(F[c], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                auto const & f = F[c].array(mfi);
                amrex::ParallelFor(mfi.growntilebox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    Real const z = zlo0 + (i + off)*dz;
                    Real const k1 = 2.0_rt*MathConst::pi*3.0_rt/Lz;
                    Real const k2 = 2.0_rt*MathConst::pi*17.0_rt/Lz;
                    f(i, j, k) = std::sin(k1*z + ph)
                        + 0.37_rt*std::cos(k2*z + 2.0_rt*ph);
                });
            }
        }
        apply_op(AF, F);  // fills F's ghosts
        // stored intermediate G = curl F on the cell-centered stagger
        // (rho's BoxArray is nodal: convert, or the nodal seam cell would
        // be a never-filled VALID row)
        MultiFab G(amrex::convert(rho.boxArray(),
                                  amrex::IntVect::TheZeroVector()),
                   rho.DistributionMap(), 2, IntVect(1),
                   MFInfo().SetArena(The_Pinned_Arena()));
        // G comp 0 = (curl F)_x = -dFy/dz, comp 1 = (curl F)_y = +dFx/dz,
        // both cell-centered from the nodal transverse pair
        for (MFIter mfi(G, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const & g = G.array(mfi);
            auto const & fx = F[0].const_array(mfi);
            auto const & fy = F[1].const_array(mfi);
            amrex::ParallelFor(amrex::convert(mfi.validbox(),
                                              amrex::IntVect(0)),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                g(i, j, k, 0) = -(fy(i+1, j, k) - fy(i, j, k))/dz;
                g(i, j, k, 1) =  (fx(i+1, j, k) - fx(i, j, k))/dz;
            });
        }
        G.FillBoundary(geom.periodicity());
        // pinned copy of the (device) coefficient fields for the host
        // loop, and a stream sync so every device write above is visible
        std::array<MultiFab, 3> coef_h;
        for (int c = 0; c < 3; ++c) {
            coef_h[c].define(coef[c]->boxArray(),
                             coef[c]->DistributionMap(), coef[c]->nComp(),
                             IntVect(0), MFInfo().SetArena(The_Pinned_Arena()));
            MultiFab::Copy(coef_h[c], *coef[c], 0, 0, coef[c]->nComp(), 0);
        }
        Gpu::streamSynchronize();
        Real errmax[3] = {0.0_rt, 0.0_rt, 0.0_rt};
        for (int c = 0; c < 3; ++c) {
            GpuArray<int, 3> const Sc = E_stag[c];
            const GpuArray<int, 3> Sx = E_stag[0], Sy = E_stag[1],
                Sz = E_stag[2];
            Real emax = 0.0_rt;
            // normalize by the global row magnitude (a pointwise relative
            // error inflates at the field's zero crossings)
            Real const afmax = AF[c].norminf(0);
            for (MFIter mfi(AF[c]); mfi.isValid(); ++mfi) {
                auto const & af = AF[c].const_array(mfi);
                auto const & fx = F[0].const_array(mfi);
                auto const & fy = F[1].const_array(mfi);
                auto const & fz = F[2].const_array(mfi);
                auto const & g  = G.const_array(mfi);
                auto const & cf = coef_h[c].const_array(mfi);
                const int cc = c;
                auto const & bx0 = mfi.validbox();
                amrex::LoopOnCpu(bx0, [&] (int i, int j, int k)
                {
                    // reference (1/mu0) curl(curl F): back-curl of the
                    // stored G ((curl G)_x = -dG_y/dz, (curl G)_y =
                    // +dG_x/dz; both nodal from the cc pair)
                    Real ell_ref = 0.0_rt;
                    if (cc == 0) {
                        ell_ref = inv_mu0
                            *(-(g(i, j, k, 1) - g(i-1, j, k, 1))/dz);
                    } else if (cc == 1) {
                        ell_ref = inv_mu0
                            *( (g(i, j, k, 0) - g(i-1, j, k, 0))/dz);
                    }
                    // reference tensor part with an adjugate 3x3 inverse
                    Real const a  = cf(i, j, k, 0);
                    Real const b1 = cf(i, j, k, 1);
                    Real const b2c = cf(i, j, k, 2);
                    Real const b3 = cf(i, j, k, 3);
                    Real const kE = cf(i, j, k, 4);
                    Real const vx = Interp(fx, Sx, Sc, coarsen_rr,
                                           i, j, k, 0);
                    Real const vy = Interp(fy, Sy, Sc, coarsen_rr,
                                           i, j, k, 0);
                    Real const vz = Interp(fz, Sz, Sc, coarsen_rr,
                                           i, j, k, 0);
                    // M = a I - [b]x, rows: [a, b3, -b2; -b3, a, b1;
                    //                        b2, -b1, a]
                    Real const m11 = a, m12 = b3, m13 = -b2c;
                    Real const m21 = -b3, m22 = a, m23 = b1;
                    Real const m31 = b2c, m32 = -b1, m33 = a;
                    Real const det = m11*(m22*m33 - m23*m32)
                        - m12*(m21*m33 - m23*m31)
                        + m13*(m21*m32 - m22*m31);
                    Real const i11 = (m22*m33 - m23*m32)/det;
                    Real const i12 = (m13*m32 - m12*m33)/det;
                    Real const i13 = (m12*m23 - m13*m22)/det;
                    Real const i21 = (m23*m31 - m21*m33)/det;
                    Real const i22 = (m11*m33 - m13*m31)/det;
                    Real const i23 = (m13*m21 - m11*m23)/det;
                    Real const i31 = (m21*m32 - m22*m31)/det;
                    Real const i32 = (m12*m31 - m11*m32)/det;
                    Real const i33 = (m11*m22 - m12*m21)/det;
                    Real minv_v[3] = {
                        i11*vx + i12*vy + i13*vz,
                        i21*vx + i22*vy + i23*vz,
                        i31*vx + i32*vy + i33*vz };
                    Real const ref = ell_ref + kE*minv_v[cc];
                    Real const scale = amrex::max(afmax, 1.0e-30_rt);
                    Real const e = std::abs(af(i, j, k) - ref)/scale;
                    if (false) {
                        amrex::AllPrint() << "[tv-seam] c=" << cc
                            << " i=" << i << " e=" << e
                            << " af=" << af(i,j,k) << " ref=" << ref
                            << " ell_ref=" << ell_ref
                            << " kEminv=" << kE*minv_v[cc]
                            << " kE=" << kE << " a=" << a
                            << " v=(" << vx << "," << vy << "," << vz
                            << ")\n";
                    }
                    if (false) {
                        amrex::AllPrint() << "[tv-dbg] c=" << cc
                            << " i=" << i << " af=" << af(i,j,k)
                            << " ref=" << ref << " ell_ref=" << ell_ref
                            << " kE=" << kE << " minv=" << minv_v[cc]
                            << " a=" << a << " b=(" << b1 << "," << b2c
                            << "," << b3 << ") v=(" << vx << "," << vy
                            << "," << vz << ")\n";
                    }
                    emax = amrex::max(emax, e);
                });
            }
            amrex::ParallelDescriptor::ReduceRealMax(emax);
            errmax[c] = emax;
        }
        amrex::Print() << "[tensor-verify] max rel row error: x = "
            << errmax[0] << " y = " << errmax[1] << " z = " << errmax[2]
            << "\n";
    }

    std::array<MultiFab, 3> x, r0, rr, pv, vv, sv, tv, zv, yv;
    make3(x, 1); make3(r0, 0); make3(rr, 0); make3(pv, 1);
    make3(vv, 0); make3(sv, 1); make3(tv, 0); make3(zv, 1); make3(yv, 1);
    for (int c = 0; c < 3; ++c) {
        MultiFab::Copy(x[c], *Efield[c], 0, 0, 1, 0);
    }
    apply_op(vv, x);   // vv = A x0 (scratch use)
    Real rhs_nrm2 = 0.0_rt;
    for (int c = 0; c < 3; ++c) {
        MultiFab::LinComb(r0[c], 1.0_rt, *rhs[c], 0, -1.0_rt, vv[c], 0,
                          0, 1, IntVect(0));
        MultiFab::Copy(rr[c], r0[c], 0, 0, 1, 0);
        rhs_nrm2 += MultiFab::Dot(*rhs[c], 0, *rhs[c], 0, 1, 0);
    }
    Real const tol = 1.0e-11_rt
        * amrex::max(std::sqrt(rhs_nrm2), 1.0e-300_rt);
    Real rho_o = 1.0_rt, alpha_b = 1.0_rt, omega_b = 1.0_rt;
    for (int c = 0; c < 3; ++c) { vv[c].setVal(0.0_rt); }
    int const max_it = 2000;
    int it = 0;
    const bool dbg = (std::getenv("WARPX_TENSOR_DEBUG") != nullptr);
    if (dbg) {
        amrex::Print() << "[tensor] |rhs| = " << std::sqrt(rhs_nrm2)
            << " tol = " << tol << " |r0| = " << std::sqrt(dot3(rr, rr))
            << "\n";
    }
    for (; it < max_it; ++it) {
        Real const rnrm = std::sqrt(dot3(rr, rr));
        if (dbg && (it % 100 == 0)) {
            amrex::Print() << "[tensor] it " << it << " |r| = " << rnrm
                << " alpha = " << alpha_b << " omega = " << omega_b
                << "\n";
        }
        if (rnrm <= tol) { break; }
        Real const rho_n = dot3(r0, rr);
        Real const beta_b = (rho_n/rho_o)*(alpha_b/omega_b);
        rho_o = rho_n;
        for (int c = 0; c < 3; ++c) {
            // p = r + beta (p - omega v)
            MultiFab::Saxpy(pv[c], -omega_b, vv[c], 0, 0, 1, IntVect(0));
            MultiFab::LinComb(pv[c], 1.0_rt, rr[c], 0, beta_b, pv[c], 0,
                              0, 1, IntVect(0));
        }
        prec3(yv, pv);
        apply_op(vv, yv);
        alpha_b = rho_n/dot3(r0, vv);
        for (int c = 0; c < 3; ++c) {
            MultiFab::LinComb(sv[c], 1.0_rt, rr[c], 0, -alpha_b, vv[c], 0,
                              0, 1, IntVect(0));
            MultiFab::Saxpy(x[c], alpha_b, yv[c], 0, 0, 1, IntVect(0));
        }
        if (std::sqrt(dot3(sv, sv)) <= tol) { break; }
        prec3(zv, sv);
        apply_op(tv, zv);
        omega_b = dot3(tv, sv)/dot3(tv, tv);
        for (int c = 0; c < 3; ++c) {
            MultiFab::Saxpy(x[c], omega_b, zv[c], 0, 0, 1, IntVect(0));
            MultiFab::LinComb(rr[c], 1.0_rt, sv[c], 0, -omega_b, tv[c], 0,
                              0, 1, IntVect(0));
        }
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(it < max_it,
        "SolveETensor1D: BiCGStab failed to converge");

    for (int c = 0; c < 3; ++c) {
        MultiFab::Copy(*Efield[c], x[c], 0, 0, 1, 0);
        Efield[c]->FillBoundary(geom.periodicity());
    }
#endif
}

void HybridPICModel::SolveETensorRZ (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rho,
    amrex::MultiFab const& Pe,
    int const lev, bool const solve_for_Faraday,
    bool const include_resistivity) const
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(Efield, Jifield, Bfield, rho, Pe, lev,
                         solve_for_Faraday, include_resistivity);
    WARPX_ABORT_WITH_MESSAGE(
        "SolveETensorRZ: the RZ tensor_form E-solve is RZ-only");
#else
    ABLASTR_PROFILE("HybridPICModel::SolveETensorRZ()");
    using namespace amrex;
    using namespace ablastr::coarsen::sample;

    auto & warpx = WarpX::GetInstance();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(lev == 0,
        "hybrid_pic_model.esolve = tensor_form supports a single level");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::ncomps == 1,
        "hybrid_pic_model.esolve = tensor_form supports only the m = 0 "
        "azimuthal mode in RZ");
    Geometry const & geom = warpx.Geom(lev);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(geom.isPeriodic(1),
        "hybrid_pic_model.esolve = tensor_form (RZ) requires periodic z "
        "(v1)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(geom.ProbLo(0) == 0.0,
        "hybrid_pic_model.esolve = tensor_form (RZ) requires the domain "
        "to include the axis (r_min = 0)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_darwin,
        "hybrid_pic_model.esolve = tensor_form (RZ) does not support the "
        "darwin split yet (curlcurl E_L != 0 discretely in RZ -- the "
        "transverse-state sector caveat)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_add_external_fields,
        "hybrid_pic_model.esolve = tensor_form does not support split "
        "external fields yet");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_resistivity_has_J_dependence,
        "hybrid_pic_model.esolve = tensor_form does not support "
        "J-dependent resistivity yet");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_include_hyper_resistivity_term,
        "hybrid_pic_model.esolve = tensor_form does not support "
        "hyper-resistivity yet");

    EnsureTensorScratch();

    auto const dx_arr = geom.CellSizeArray();
    Real const dr = dx_arr[0];
    Real const dz = dx_arr[1];
    Real const rmin = geom.ProbLo(0);
    Real const inv_dr = 1.0_rt/dr;
    Real const inv_dz = 1.0_rt/dz;
    Real const inv_dz2 = inv_dz*inv_dz;
    Real const dx_min = amrex::min(dr, dz);
    Real const dt_eff = (m_tensor_dt_eff > 0.0_rt)
        ? m_tensor_dt_eff : 0.5_rt*warpx.getdt(lev);
    Real const rho_1c = PhysConst::q_e
        * ((m_tensor_n_min > 0.0_rt) ? m_tensor_n_min : m_n_floor);
    Real const me_eff = m_electron_inertia_mass;
    Real const alpha = m_tensor_alpha;
    Real const qe = PhysConst::q_e;
    Real const mu0 = PhysConst::mu0;
    Real const inv_mu0 = 1.0_rt/mu0;
    const auto eta = m_eta;
    Real const t_new = warpx.gett_new(lev);
    const bool use_eta = include_resistivity;
    const bool use_gpe = !solve_for_Faraday
        && m_include_electron_pressure_term;

    // Per-step-frozen density: EVERY rho-derived coefficient and binary
    // gate below reads this one bit-frozen snapshot (see m_tensor_rho_old).
    // Solves before the first step-entry capture (initialization) fall
    // back to the live component-0 deposit.
    amrex::MultiFab const & rho_g = m_tensor_rho_captured
        ? *m_tensor_rho_old : rho;

    GpuArray<int, 3> const nodal = {1, 1, 1};
    GpuArray<int, 3> const coarsen_rr = {1, 1, 1};
    const GpuArray<int, 3> E_stag[3] =
        {Ex_IndexType, Ey_IndexType, Ez_IndexType};
    const GpuArray<int, 3> B_stag[3] =
        {Bx_IndexType, By_IndexType, Bz_IndexType};
    const GpuArray<int, 3> J_stag[3] =
        {Jx_IndexType, Jy_IndexType, Jz_IndexType};

    // Staggered domain bounds: E_r (cc r, nodal z), E_t (nodal),
    // E_z (nodal r, cc z); plus the nodal box for the closure gate.
    Box const dom_r = amrex::convert(geom.Domain(),
                                     Efield[0]->ixType().toIntVect());
    Box const dom_t = amrex::convert(geom.Domain(),
                                     Efield[1]->ixType().toIntVect());
    Box const dom_z = amrex::convert(geom.Domain(),
                                     Efield[2]->ixType().toIntVect());
    auto const tlo = amrex::lbound(dom_t);
    auto const thi = amrex::ubound(dom_t);
    auto const zlo = amrex::lbound(dom_z);
    auto const zhi = amrex::ubound(dom_z);
    Box const dom_nd = amrex::convert(geom.Domain(),
                                      IntVect::TheNodeVector());
    auto const nlo = amrex::lbound(dom_nd);
    auto const nhi = amrex::ubound(dom_nd);
    int const ihi_nd = nhi.x;
    auto const & dm = Efield[0]->DistributionMap();

    // Vacuum-closure gate on sub-one-count NODES from the FROZEN nodal
    // density (direct nodal read -- rho is nodal in RZ). The threshold is
    // the one-count anchor rho_1c (tensor_n_min / n_floor): a STATISTICS
    // guard in the anchor taxonomy, not a physics floor -- no density is
    // ever modified. A strictly-zero gate leaves epsilon-density deposit-
    // tail rows with near-null poloidal gradient directions (eigenvalue
    // ~ V kE ~ rho_tail) driven by a NON-rho-scaled RHS (qm B x Je_meas
    // bleed): the solve then regenerates the unfloored 1/rho division as
    // an intractable conditioning floor (measured: point-Jacobi BiCGStab
    // flooring 5 orders above tolerance with restart churn). Gating at
    // one count hands those rows to the div E = 0 continuation instead;
    // their (tiny) screening rows stay in the operator, which keeps every
    // row definite regardless of the gate (closure + screening overlap is
    // additive in the energy form -- membership mismatch is safe in this
    // direction, unlike the item-1 null-family direction).
    MultiFab gam(amrex::convert(warpx.boxArray(lev),
                                IntVect::TheNodeVector()), dm, 1, 1);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(gam, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const & g = gam.array(mfi);
        auto const & rho_arr = rho_g.const_array(mfi);
        const bool zper = geom.isPeriodic(1);
        Real const rho_gate = rho_1c;
        amrex::ParallelFor(mfi.tilebox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (!zper && (j <= nlo.y || j >= nhi.y)) {
                g(i,j,k) = 0.0_rt; return;
            }
            g(i,j,k) = (rho_arr(i, j, k, 0) > rho_gate) ? 0.0_rt : 1.0_rt;
        });
    }
    gam.setBndry(0.0_rt);
    gam.FillBoundary(geom.periodicity());

    // Pointwise coefficients on each component's stagger (comp 0 = a,
    // 1..3 = beta, 4 = kE) and the V-scaled RHS. Every rho read is the
    // frozen snapshot through a proper nodal-source Interp.
    std::array<std::unique_ptr<MultiFab>, 3> coef, rhs;
    for (int c = 0; c < 3; ++c) {
        coef[c] = std::make_unique<MultiFab>(
            Efield[c]->boxArray(), dm, 5, IntVect(0));
        rhs[c] = std::make_unique<MultiFab>(
            Efield[c]->boxArray(), dm, 1, IntVect(0));
    }
    for (int c = 0; c < 3; ++c) {
        GpuArray<int, 3> const Sc = E_stag[c];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*coef[c], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const & cf  = coef[c]->array(mfi);
            auto const & rh  = rhs[c]->array(mfi);
            auto const & e_arr = Efield[c]->array(mfi);
            auto const & rho_arr = rho_g.const_array(mfi);
            auto const & pe_arr  = Pe.const_array(mfi);
            auto const & br_arr = Bfield[0]->const_array(mfi);
            auto const & bt_arr = Bfield[1]->const_array(mfi);
            auto const & bz_arr = Bfield[2]->const_array(mfi);
            auto const & jir_arr = Jifield[0]->const_array(mfi);
            auto const & jit_arr = Jifield[1]->const_array(mfi);
            auto const & jiz_arr = Jifield[2]->const_array(mfi);
            auto const & jpr_arr = m_tensor_jp_old[0]->const_array(mfi);
            auto const & jpt_arr = m_tensor_jp_old[1]->const_array(mfi);
            auto const & jpz_arr = m_tensor_jp_old[2]->const_array(mfi);
            const GpuArray<int, 3> Sr = E_stag[0], St = E_stag[1],
                Sz = E_stag[2];
            const GpuArray<int, 3> Br_s = B_stag[0], Bt_s = B_stag[1],
                Bz_s = B_stag[2];
            const GpuArray<int, 3> Jr_s = J_stag[0], Jt_s = J_stag[1],
                Jz_s = J_stag[2];
            const int cc = c;
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // Dirichlet identity rows: E_theta axis (regularity) and
                // wall (PEC), E_z wall (PEC). Zero RHS and warm start.
                const bool dirichlet =
                    (cc == 1 && (i <= tlo.x || i >= thi.x))
                    || (cc == 2 && i >= zhi.x);
                Real const rho_v = amrex::max(0.0_rt,
                    Interp(rho_arr, nodal, Sc, coarsen_rr, i, j, k, 0));
                Real const br = Interp(br_arr, Br_s, Sc, coarsen_rr, i, j, k, 0);
                Real const bt = Interp(bt_arr, Bt_s, Sc, coarsen_rr, i, j, k, 0);
                Real const bz = Interp(bz_arr, Bz_s, Sc, coarsen_rr, i, j, k, 0);
                Real const B2 = br*br + bt*bt + bz*bz;
                Real const s = dt_eff / (2.0_rt*alpha*dx_min);
                Real const me_v = amrex::max(me_eff,
                    B2*qe/(mu0*amrex::max(rho_v, rho_1c)) * s*s);
                Real const qm = qe/me_v;
                Real const kE = qm*rho_v;
                Real const eta_v = use_eta
                    ? eta(rho_v, 0.0_rt, t_new) : 0.0_rt;
                Real const a = 1.0_rt + dt_eff*kE*eta_v;
                Real const ber = dt_eff*qm*br;
                Real const bet = dt_eff*qm*bt;
                Real const bez = dt_eff*qm*bz;
                cf(i, j, k, 0) = a;
                cf(i, j, k, 1) = ber;
                cf(i, j, k, 2) = bet;
                cf(i, j, k, 3) = bez;
                cf(i, j, k, 4) = kE;
                if (dirichlet) {
                    rh(i, j, k) = 0.0_rt;
                    e_arr(i, j, k) = 0.0_rt;
                    return;
                }
                // Sub-one-count rows: the numerator there is stencil-
                // support bleed of the wider interpolations (the identity
                // form's RHS-membership argument) driving near-null
                // gradient directions -- zero it so the closed operator's
                // continuation owns those rows (see the gamma gate).
                if (!(rho_v > rho_1c)) {
                    rh(i, j, k) = 0.0_rt;
                    return;
                }
                Real const jpr = Interp(jpr_arr, Sr, Sc, coarsen_rr, i, j, k, 0);
                Real const jpt = Interp(jpt_arr, St, Sc, coarsen_rr, i, j, k, 0);
                Real const jpz = Interp(jpz_arr, Sz, Sc, coarsen_rr, i, j, k, 0);
                Real const jir = Interp(jir_arr, Jr_s, Sc, coarsen_rr, i, j, k, 0);
                Real const jit = Interp(jit_arr, Jt_s, Sc, coarsen_rr, i, j, k, 0);
                Real const jiz = Interp(jiz_arr, Jz_s, Sc, coarsen_rr, i, j, k, 0);
                Real const jer = jpr - jir;
                Real const jet = jpt - jit;
                Real const jez = jpz - jiz;
                // grad Pe on the component's stagger: nodal Pe makes the
                // poloidal two-point differences land exactly on E_r/E_z
                // (no averaging); no theta gradient at m = 0.
                Real gpr = 0.0_rt, gpz = 0.0_rt;
                if (use_gpe && cc == 0) {
                    gpr = (pe_arr(i+1, j, k) - pe_arr(i, j, k))*inv_dr;
                } else if (use_gpe && cc == 2) {
                    gpz = (pe_arr(i, j+1, k) - pe_arr(i, j, k))*inv_dz;
                }
                // w = kE eta Jp^n - qm (B x Je_meas) - qm grad Pe
                Real const wr = kE*eta_v*jpr - qm*(bt*jez - bz*jet) - qm*gpr;
                Real const wt = kE*eta_v*jpt - qm*(bz*jer - br*jez);
                Real const wz = kE*eta_v*jpz - qm*(br*jet - bt*jer) - qm*gpz;
                Real or_, ot_, oz_;
                TensorMinv(a, ber, bet, bez, wr, wt, wz, or_, ot_, oz_);
                // Control-volume row scaling (the identity form's): V_r =
                // r_c dr, V_t = r_n dr, V_z = r_n dr (axis: dr^2/8).
                Real V;
                if (cc == 0) {
                    V = (rmin + (i + 0.5_rt)*dr)*dr;
                } else if (cc == 1) {
                    V = (rmin + i*dr)*dr;
                } else {
                    V = (i == zlo.x) ? 0.125_rt*dr*dr : (rmin + i*dr)*dr;
                }
                rh(i, j, k) = V*((cc == 0) ? or_ : ((cc == 1) ? ot_ : oz_));
            });
        }
    }

    // Axis parity for the scratch r-component ghosts: the E_z axis row's
    // tensor coupling Interps E_r across the axis (cc r: ghost i maps to
    // cell -1 - i, odd for m = 0). Applied after each ghost exchange.
    auto axis_mirror_r = [&](MultiFab& fr)
    {
        if (rmin != 0.0_rt) { return; }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(fr); mfi.isValid(); ++mfi) {
            Box gbx = mfi.growntilebox();
            if (gbx.smallEnd(0) >= 0) { continue; }
            gbx.setBig(0, -1);
            auto const & arr = fr.array(mfi);
            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                arr(i, j, k) = -arr(-1 - i, j, k);
            });
        }
    };

    // Fused operator: out_c = inv_mu0 * [V-scaled (curlcurl in)_c
    // - V grad(gamma div in)|_pol] + V_c kE [M^{-1} in]_c.
    // Poloidal rows are the composed identity-form rows (W through cell
    // centers); the toroidal row composes w_r = -d_z E_t (B_r stagger) and
    // w_z = (1/r_c) d_r (r E_t) (B_z stagger), whose energy gradient
    // carries the axis regularity through r_0 = 0 (no hand-built branch).
    auto apply_op = [&](std::array<MultiFab, 3>& out,
                        std::array<MultiFab, 3>& in)
    {
        for (int c = 0; c < 3; ++c) {
            in[c].FillBoundary(geom.periodicity());
        }
        axis_mirror_r(in[0]);
        for (int c = 0; c < 3; ++c) {
            GpuArray<int, 3> const Sc = E_stag[c];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(out[c], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                auto const & o  = out[c].array(mfi);
                auto const & pr = in[0].const_array(mfi);
                auto const & pt = in[1].const_array(mfi);
                auto const & pz = in[2].const_array(mfi);
                auto const & cf = coef[c]->const_array(mfi);
                auto const & g  = gam.const_array(mfi);
                const GpuArray<int, 3> Sr = E_stag[0], St = E_stag[1],
                    Sz = E_stag[2];
                const int cc = c;
                amrex::ParallelFor(mfi.tilebox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    // Dirichlet identity rows (see the RHS assembly)
                    if ((cc == 1 && (i <= tlo.x || i >= thi.x))
                        || (cc == 2 && i >= zhi.x)) {
                        auto const & pself =
                            (cc == 1) ? pt : pz;
                        o(i,j,k) = pself(i,j,k);
                        return;
                    }
                    Real ell = 0.0_rt;
                    if (cc == 0) {
                        Real const rc = rmin + (i + 0.5_rt)*dr;
                        Real const wp = (pr(i,j+1,k) - pr(i,j,k))*inv_dz
                                      - (pz(i+1,j,k) - pz(i,j,k))*inv_dr;
                        Real const wm = (pr(i,j,k) - pr(i,j-1,k))*inv_dz
                                      - (pz(i+1,j-1,k) - pz(i,j-1,k))*inv_dr;
                        Real dd = 0.0_rt;
                        if (g(i,j,k) > 0.0_rt) {
                            Real f = rc*pr(i,j,k);
                            if (i > 0) {
                                f -= (rmin + (i - 0.5_rt)*dr)*pr(i-1,j,k);
                            }
                            dd += f/PolNodeVol(i, rmin, dr, ihi_nd)
                                + (pz(i,j,k) - pz(i,j-1,k))*inv_dz;
                        }
                        if (g(i+1,j,k) > 0.0_rt) {
                            Real f = -rc*pr(i,j,k);
                            if (i + 1 <= ihi_nd - 1) {
                                f += (rmin + (i + 1.5_rt)*dr)*pr(i+1,j,k);
                            }
                            dd -= f/PolNodeVol(i+1, rmin, dr, ihi_nd)
                                + (pz(i+1,j,k) - pz(i+1,j-1,k))*inv_dz;
                        }
                        ell = -rc*dr*(wp - wm)*inv_dz + rc*dd;
                    } else if (cc == 2) {
                        Real const rcp = rmin + (i + 0.5_rt)*dr;
                        Real const rcm = rmin + (i - 0.5_rt)*dr;
                        Real const wp = (pr(i,j+1,k) - pr(i,j,k))*inv_dz
                                      - (pz(i+1,j,k) - pz(i,j,k))*inv_dr;
                        Real ccv = rcp*wp;
                        if (i > zlo.x) {
                            Real const wm =
                                (pr(i-1,j+1,k) - pr(i-1,j,k))*inv_dz
                                - (pz(i,j,k) - pz(i-1,j,k))*inv_dr;
                            ccv -= rcm*wm;
                        }
                        if (g(i,j,k) > 0.0_rt || g(i,j+1,k) > 0.0_rt) {
                            Real const Vn = PolNodeVol(i, rmin, dr, ihi_nd);
                            Real dd = 0.0_rt;
                            if (g(i,j,k) > 0.0_rt) {
                                Real f = rcp*pr(i,j,k);
                                if (i > 0) { f -= rcm*pr(i-1,j,k); }
                                dd += f/Vn
                                    + (pz(i,j,k) - pz(i,j-1,k))*inv_dz;
                            }
                            if (g(i,j+1,k) > 0.0_rt) {
                                Real f = rcp*pr(i,j+1,k);
                                if (i > 0) { f -= rcm*pr(i-1,j+1,k); }
                                dd -= f/Vn
                                    + (pz(i,j+1,k) - pz(i,j,k))*inv_dz;
                            }
                            ccv += Vn*dd*inv_dz;
                        }
                        ell = ccv;
                    } else {
                        // toroidal: V_t (d_z w_r - d_r w_z), w_r/w_z
                        // recomputed inline from the nodal in_t
                        Real const rn = rmin + i*dr;
                        Real const rp = rmin + (i + 1)*dr;
                        Real const rm = rmin + (i - 1)*dr;
                        Real const rcp = rmin + (i + 0.5_rt)*dr;
                        Real const rcm = rmin + (i - 0.5_rt)*dr;
                        Real const wrp = -(pt(i,j+1,k) - pt(i,j,k))*inv_dz;
                        Real const wrm = -(pt(i,j,k) - pt(i,j-1,k))*inv_dz;
                        Real const wzp =
                            (rp*pt(i+1,j,k) - rn*pt(i,j,k))/(rcp*dr);
                        Real const wzm =
                            (rn*pt(i,j,k) - rm*pt(i-1,j,k))/(rcm*dr);
                        ell = rn*dr*((wrp - wrm)*inv_dz
                                     - (wzp - wzm)*inv_dr);
                    }
                    Real const vr = Interp(pr, Sr, Sc, coarsen_rr, i, j, k, 0);
                    Real const vt = Interp(pt, St, Sc, coarsen_rr, i, j, k, 0);
                    Real const vz = Interp(pz, Sz, Sc, coarsen_rr, i, j, k, 0);
                    Real or_, ot_, oz_;
                    TensorMinv(cf(i,j,k,0), cf(i,j,k,1), cf(i,j,k,2),
                               cf(i,j,k,3), vr, vt, vz, or_, ot_, oz_);
                    Real const tv = (cc == 0) ? or_ : ((cc == 1) ? ot_ : oz_);
                    Real V;
                    if (cc == 0) {
                        V = (rmin + (i + 0.5_rt)*dr)*dr;
                    } else if (cc == 1) {
                        V = (rmin + i*dr)*dr;
                    } else {
                        V = (i == zlo.x) ? 0.125_rt*dr*dr
                                         : (rmin + i*dr)*dr;
                    }
                    o(i,j,k) = inv_mu0*ell + V*cf(i,j,k,4)*tv;
                });
            }
        }
    };

    // Jacobi diagonals: the elliptic self-coefficients of the fused rows
    // (inv_mu0-scaled, closure-gated) plus the tensor's own V-scaled
    // diagonal; Dirichlet rows get 1.
    std::array<MultiFab, 3> diag;
    for (int c = 0; c < 3; ++c) {
        diag[c].define(Efield[c]->boxArray(), dm, 1, IntVect(0));
        GpuArray<int, 3> const Sc = E_stag[c];
        amrex::ignore_unused(Sc);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(diag[c], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const & d  = diag[c].array(mfi);
            auto const & cf = coef[c]->const_array(mfi);
            auto const & g  = gam.const_array(mfi);
            const int cc = c;
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if ((cc == 1 && (i <= tlo.x || i >= thi.x))
                    || (cc == 2 && i >= zhi.x)) {
                    d(i,j,k) = 1.0_rt; return;
                }
                Real ell;
                Real V;
                if (cc == 0) {
                    Real const rc = rmin + (i + 0.5_rt)*dr;
                    Real dv = 0.0_rt;
                    if (g(i,j,k) > 0.0_rt) {
                        dv += rc*rc/PolNodeVol(i, rmin, dr, ihi_nd);
                    }
                    if (g(i+1,j,k) > 0.0_rt) {
                        dv += rc*rc/PolNodeVol(i+1, rmin, dr, ihi_nd);
                    }
                    ell = 2.0_rt*rc*dr*inv_dz2 + dv;
                    V = rc*dr;
                } else if (cc == 2) {
                    Real rdiag = rmin + (i + 0.5_rt)*dr;
                    if (i > zlo.x) { rdiag += rmin + (i - 0.5_rt)*dr; }
                    Real const gsum =
                        ((g(i,j,k) > 0.0_rt) ? 1.0_rt : 0.0_rt)
                        + ((g(i,j+1,k) > 0.0_rt) ? 1.0_rt : 0.0_rt);
                    Real const Vn = PolNodeVol(i, rmin, dr, ihi_nd);
                    ell = rdiag*inv_dr + gsum*Vn*inv_dz2;
                    V = (i == zlo.x) ? 0.125_rt*dr*dr : (rmin + i*dr)*dr;
                } else {
                    Real const rn = rmin + i*dr;
                    Real const rcp = rmin + (i + 0.5_rt)*dr;
                    Real const rcm = rmin + (i - 0.5_rt)*dr;
                    ell = 2.0_rt*rn*dr*inv_dz2
                        + rn*rn*(1.0_rt/rcp + 1.0_rt/rcm)*inv_dr;
                    V = rn*dr;
                }
                Real const a = cf(i,j,k,0);
                Real const bc = cf(i,j,k,1+cc);
                Real const b2 = cf(i,j,k,1)*cf(i,j,k,1)
                    + cf(i,j,k,2)*cf(i,j,k,2) + cf(i,j,k,3)*cf(i,j,k,3);
                Real const tdiag = cf(i,j,k,4)*(a*a + bc*bc)
                    /(a*(a*a + b2));
                d(i,j,k) = inv_mu0*ell + V*tdiag;
            });
        }
    }

    auto dot3 = [&](std::array<MultiFab, 3>& u, std::array<MultiFab, 3>& v)
    {
        Real s = 0.0_rt;
        for (int c = 0; c < 3; ++c) {
            s += MultiFab::Dot(u[c], 0, v[c], 0, 1, 0);
        }
        return s;
    };
    auto make3 = [&](std::array<MultiFab, 3>& v, int ng)
    {
        for (int c = 0; c < 3; ++c) {
            v[c].define(Efield[c]->boxArray(), dm, 1, IntVect(ng));
            v[c].setVal(0.0_rt);
        }
    };
    auto prec3 = [&](std::array<MultiFab, 3>& z, std::array<MultiFab, 3>& r)
    {
        for (int c = 0; c < 3; ++c) {
            MultiFab::Copy(z[c], r[c], 0, 0, 1, 0);
            MultiFab::Divide(z[c], diag[c], 0, 0, 1, 0);
        }
    };

    // One-shot discrete verification (hybrid_pic_model.tensor_verify):
    // the fused rows on a deterministic pseudo-random field vs the
    // independently composed references through STORED, ghost-exchanged
    // intermediates -- cc W and gated nodal D for the poloidal pair
    // (the esolve_pol_verify pattern), the (B_r, B_z)-staggered curl pair
    // for the toroidal row -- plus the pointwise tensor through a general
    // 3x3 adjugate inverse (the 1D tensor_verify pattern).
    if (m_esolve_tensor_verify && !m_esolve_tensor_verified) {
        m_esolve_tensor_verified = true;
        // one-shot debug block: temporaries live in pinned host memory so
        // the host comparison loop below can read them on GPU builds
        // (device-arena fabs would segfault under amrex::LoopOnCpu)
        std::array<MultiFab, 3> F, AF;
        for (int c = 0; c < 3; ++c) {
            F[c].define(Efield[c]->boxArray(), dm, 1, IntVect(1),
                        MFInfo().SetArena(The_Pinned_Arena()));
            AF[c].define(Efield[c]->boxArray(), dm, 1, IntVect(0),
                         MFInfo().SetArena(The_Pinned_Arena()));
            F[c].setVal(0.0_rt);
        }
        for (int c = 0; c < 3; ++c) {
            const int cc = c;
            const bool znodal = (E_stag[c][1] == 1);
            const int nzc = nhi.y - nlo.y;
            const int jlo_nd = nlo.y;
            for (MFIter mfi(F[c], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                auto const & f = F[c].array(mfi);
                amrex::ParallelFor(mfi.tilebox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if ((cc == 1 && (i <= tlo.x || i >= thi.x))
                        || (cc == 2 && i >= zhi.x)) {
                        f(i,j,k) = 0.0_rt; return;
                    }
                    // periodic z identifies duplicated nodal DOFs: hash
                    // the wrapped index (single-valued through the ghost
                    // exchange)
                    int jw = j;
                    if (znodal) {
                        jw = jlo_nd
                            + (((j - jlo_nd) % nzc) + nzc) % nzc;
                    }
                    auto h = static_cast<unsigned int>(i*73856093)
                           ^ static_cast<unsigned int>(jw*19349663)
                           ^ (0x9e3779b9u + 0x1234567u*cc);
                    h = h*1103515245u + 12345u;
                    f(i,j,k) = Real((h >> 8) & 0xFFFFu)/32768.0_rt
                        - 1.0_rt;
                });
            }
        }
        apply_op(AF, F);  // fills F's ghosts (+ axis mirror on F_r)
        // stored cc W = d_z F_r - d_r F_z
        MultiFab Wc(amrex::convert(warpx.boxArray(lev),
                                   amrex::IntVect::TheZeroVector()),
                    dm, 1, IntVect(1),
                    MFInfo().SetArena(The_Pinned_Arena()));
        for (MFIter mfi(Wc, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const & w = Wc.array(mfi);
            auto const & fr = F[0].const_array(mfi);
            auto const & fz = F[2].const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                w(i,j,k) = (fr(i,j+1,k) - fr(i,j,k))*inv_dz
                         - (fz(i+1,j,k) - fz(i,j,k))*inv_dr;
            });
        }
        Wc.setBndry(0.0_rt);
        Wc.FillBoundary(geom.periodicity());
        // stored gated nodal D = gamma div_pol F
        MultiFab D(gam.boxArray(), dm, 1, 1,
                   MFInfo().SetArena(The_Pinned_Arena()));
        for (MFIter mfi(D, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const & dv = D.array(mfi);
            auto const & fr = F[0].const_array(mfi);
            auto const & fz = F[2].const_array(mfi);
            auto const & g  = gam.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (!(g(i,j,k) > 0.0_rt)) { dv(i,j,k) = 0.0_rt; return; }
                Real f = 0.0_rt;
                if (i <= ihi_nd - 1) {
                    f += (rmin + (i + 0.5_rt)*dr)*fr(i,j,k);
                }
                if (i > 0) {
                    f -= (rmin + (i - 0.5_rt)*dr)*fr(i-1,j,k);
                }
                dv(i,j,k) = f/PolNodeVol(i, rmin, dr, ihi_nd)
                    + (fz(i,j,k) - fz(i,j-1,k))*inv_dz;
            });
        }
        D.setBndry(0.0_rt);
        D.FillBoundary(geom.periodicity());
        // stored toroidal curl pair: wr on B_r stagger, wz on B_z stagger
        MultiFab Wr(amrex::convert(warpx.boxArray(lev),
                                   Bfield[0]->ixType().toIntVect()),
                    dm, 1, IntVect(1),
                    MFInfo().SetArena(The_Pinned_Arena()));
        MultiFab Wz(amrex::convert(warpx.boxArray(lev),
                                   Bfield[2]->ixType().toIntVect()),
                    dm, 1, IntVect(1),
                    MFInfo().SetArena(The_Pinned_Arena()));
        for (MFIter mfi(Wr, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const & w = Wr.array(mfi);
            auto const & ft = F[1].const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                w(i,j,k) = -(ft(i,j+1,k) - ft(i,j,k))*inv_dz;
            });
        }
        for (MFIter mfi(Wz, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const & w = Wz.array(mfi);
            auto const & ft = F[1].const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Real const rc = rmin + (i + 0.5_rt)*dr;
                w(i,j,k) = ((rmin + (i+1)*dr)*ft(i+1,j,k)
                            - (rmin + i*dr)*ft(i,j,k))/(rc*dr);
            });
        }
        Wr.setBndry(0.0_rt);
        Wz.setBndry(0.0_rt);
        Wr.FillBoundary(geom.periodicity());
        Wz.FillBoundary(geom.periodicity());
        // pinned copy of the (device) coefficient fields for the host
        // loop, and a stream sync so every device write above is visible
        std::array<MultiFab, 3> coef_h;
        for (int c = 0; c < 3; ++c) {
            coef_h[c].define(coef[c]->boxArray(),
                             coef[c]->DistributionMap(), coef[c]->nComp(),
                             IntVect(0), MFInfo().SetArena(The_Pinned_Arena()));
            MultiFab::Copy(coef_h[c], *coef[c], 0, 0, coef[c]->nComp(), 0);
        }
        Gpu::streamSynchronize();
        Real errmax[3] = {0.0_rt, 0.0_rt, 0.0_rt};
        for (int c = 0; c < 3; ++c) {
            GpuArray<int, 3> const Sc = E_stag[c];
            const GpuArray<int, 3> Sr = E_stag[0], St = E_stag[1],
                Sz = E_stag[2];
            Real emax = 0.0_rt;
            Real const afmax = AF[c].norminf(0);
            for (MFIter mfi(AF[c]); mfi.isValid(); ++mfi) {
                auto const & af = AF[c].const_array(mfi);
                auto const & fr = F[0].const_array(mfi);
                auto const & ft = F[1].const_array(mfi);
                auto const & fz = F[2].const_array(mfi);
                auto const & w  = Wc.const_array(mfi);
                auto const & dv = D.const_array(mfi);
                auto const & wr = Wr.const_array(mfi);
                auto const & wz = Wz.const_array(mfi);
                auto const & cf = coef_h[c].const_array(mfi);
                const int cc = c;
                auto const & bx0 = mfi.validbox();
                amrex::LoopOnCpu(bx0, [&] (int i, int j, int k)
                {
                    if ((cc == 1 && (i <= tlo.x || i >= thi.x))
                        || (cc == 2 && i >= zhi.x)) {
                        return;  // identity rows by construction
                    }
                    Real ell_ref;
                    Real V;
                    if (cc == 0) {
                        Real const rc = rmin + (i + 0.5_rt)*dr;
                        ell_ref = -rc*dr*(w(i,j,k) - w(i,j-1,k))*inv_dz
                            + rc*(dv(i,j,k) - dv(i+1,j,k));
                        V = rc*dr;
                    } else if (cc == 2) {
                        Real const rcp = rmin + (i + 0.5_rt)*dr;
                        Real ccv = rcp*w(i,j,k);
                        if (i > zlo.x) {
                            ccv -= (rmin + (i - 0.5_rt)*dr)*w(i-1,j,k);
                        }
                        Real const Vn = PolNodeVol(i, rmin, dr, ihi_nd);
                        ccv += Vn*(dv(i,j,k) - dv(i,j+1,k))*inv_dz;
                        ell_ref = ccv;
                        V = (i == zlo.x) ? 0.125_rt*dr*dr
                                         : (rmin + i*dr)*dr;
                    } else {
                        Real const rn = rmin + i*dr;
                        ell_ref = rn*dr*((wr(i,j,k) - wr(i,j-1,k))*inv_dz
                            - (wz(i,j,k) - wz(i-1,j,k))*inv_dr);
                        V = rn*dr;
                    }
                    Real const a  = cf(i, j, k, 0);
                    Real const b1 = cf(i, j, k, 1);
                    Real const b2c = cf(i, j, k, 2);
                    Real const b3 = cf(i, j, k, 3);
                    Real const kE = cf(i, j, k, 4);
                    Real const vr = Interp(fr, Sr, Sc, coarsen_rr,
                                           i, j, k, 0);
                    Real const vt = Interp(ft, St, Sc, coarsen_rr,
                                           i, j, k, 0);
                    Real const vz = Interp(fz, Sz, Sc, coarsen_rr,
                                           i, j, k, 0);
                    Real const m11 = a, m12 = b3, m13 = -b2c;
                    Real const m21 = -b3, m22 = a, m23 = b1;
                    Real const m31 = b2c, m32 = -b1, m33 = a;
                    Real const det = m11*(m22*m33 - m23*m32)
                        - m12*(m21*m33 - m23*m31)
                        + m13*(m21*m32 - m22*m31);
                    Real const minv_v0 = ((m22*m33 - m23*m32)*vr
                        + (m13*m32 - m12*m33)*vt
                        + (m12*m23 - m13*m22)*vz)/det;
                    Real const minv_v1 = ((m23*m31 - m21*m33)*vr
                        + (m11*m33 - m13*m31)*vt
                        + (m13*m21 - m11*m23)*vz)/det;
                    Real const minv_v2 = ((m21*m32 - m22*m31)*vr
                        + (m12*m31 - m11*m32)*vt
                        + (m11*m22 - m12*m21)*vz)/det;
                    Real const tvv = (cc == 0) ? minv_v0
                        : ((cc == 1) ? minv_v1 : minv_v2);
                    Real const ref = inv_mu0*ell_ref + V*kE*tvv;
                    Real const scale = amrex::max(afmax, 1.0e-30_rt);
                    emax = amrex::max(emax,
                                      std::abs(af(i, j, k) - ref)/scale);
                });
            }
            amrex::ParallelDescriptor::ReduceRealMax(emax);
            errmax[c] = emax;
        }
        amrex::Print() << "[tensor-verify-rz] max rel row error: r = "
            << errmax[0] << " t = " << errmax[1] << " z = " << errmax[2]
            << "\n";
    }

    // Point-Jacobi preconditioned BiCGStab on the coupled 3-component
    // system (gyration makes it non-symmetric). Warm start from the
    // incoming E; tight tolerance -- the solve runs inside FD-Jacobian
    // probe evaluations. The vacuum-annulus iteration economics are the
    // identity form's (Jacobi does not see the long-wavelength vacuum
    // modes; the factored backend is the production follow-up).
    std::array<MultiFab, 3> x, r0, rr, pv, vv, sv, tv2, zv, yv;
    make3(x, 1); make3(r0, 0); make3(rr, 0); make3(pv, 1);
    make3(vv, 0); make3(sv, 1); make3(tv2, 0); make3(zv, 1); make3(yv, 1);
    for (int c = 0; c < 3; ++c) {
        MultiFab::Copy(x[c], *Efield[c], 0, 0, 1, 0);
    }
    apply_op(vv, x);
    Real rhs_nrm2 = 0.0_rt;
    for (int c = 0; c < 3; ++c) {
        MultiFab::LinComb(r0[c], 1.0_rt, *rhs[c], 0, -1.0_rt, vv[c], 0,
                          0, 1, IntVect(0));
        MultiFab::Copy(rr[c], r0[c], 0, 0, 1, 0);
        rhs_nrm2 += MultiFab::Dot(*rhs[c], 0, *rhs[c], 0, 1, 0);
    }
    Real const tol = 1.0e-11_rt
        * amrex::max(std::sqrt(rhs_nrm2), 1.0e-300_rt);
    Real rho_o = 1.0_rt, alpha_b = 1.0_rt, omega_b = 1.0_rt;
    for (int c = 0; c < 3; ++c) { vv[c].setVal(0.0_rt); }
    int const max_it = 25000;
    int it = 0;
    int n_restart = 0;
    const bool dbg = (std::getenv("WARPX_TENSOR_DEBUG") != nullptr);
    // Breakdown/drift guard: point-Jacobi BiCGStab on the ill-conditioned
    // vacuum-annulus composite is prone to (near-)breakdown (rho or
    // omega denominators collapsing) and to recursion drift of the
    // recursive residual. Restart from the current (finite) iterate:
    // recompute the true residual, re-seed the shadow vector, and clear
    // the directions. Detected BEFORE the update, so x never absorbs a
    // non-finite step.
    auto restart = [&]()
    {
        apply_op(vv, x);
        for (int c = 0; c < 3; ++c) {
            MultiFab::LinComb(rr[c], 1.0_rt, *rhs[c], 0, -1.0_rt, vv[c],
                              0, 0, 1, IntVect(0));
            MultiFab::Copy(r0[c], rr[c], 0, 0, 1, 0);
            pv[c].setVal(0.0_rt);
            vv[c].setVal(0.0_rt);
        }
        rho_o = 1.0_rt; alpha_b = 1.0_rt; omega_b = 1.0_rt;
        ++n_restart;
    };
    auto bad = [](Real v) { return !std::isfinite(v); };
    for (; it < max_it; ++it) {
        Real const rnrm = std::sqrt(dot3(rr, rr));
        if (dbg && (it % 200 == 0)) {
            amrex::Print() << "[tensor-rz] it " << it << " |r| = " << rnrm
                << " tol = " << tol << " restarts = " << n_restart
                << "\n";
        }
        if (rnrm <= tol) { break; }
        if (bad(rnrm)) {
            // recursive residual corrupted but x is still finite (the
            // guards below block non-finite updates): resync
            restart();
            continue;
        }
        // periodic true-residual resync against recursion drift
        if (it > 0 && (it % 1000 == 0)) { restart(); continue; }
        Real const rho_n = dot3(r0, rr);
        Real const beta_b = (rho_n/rho_o)*(alpha_b/omega_b);
        if (bad(beta_b) || rho_n == 0.0_rt) { restart(); continue; }
        rho_o = rho_n;
        for (int c = 0; c < 3; ++c) {
            MultiFab::Saxpy(pv[c], -omega_b, vv[c], 0, 0, 1, IntVect(0));
            MultiFab::LinComb(pv[c], 1.0_rt, rr[c], 0, beta_b, pv[c], 0,
                              0, 1, IntVect(0));
        }
        prec3(yv, pv);
        apply_op(vv, yv);
        Real const den_a = dot3(r0, vv);
        alpha_b = rho_n/den_a;
        if (bad(alpha_b)) { restart(); continue; }
        for (int c = 0; c < 3; ++c) {
            MultiFab::LinComb(sv[c], 1.0_rt, rr[c], 0, -alpha_b, vv[c], 0,
                              0, 1, IntVect(0));
            MultiFab::Saxpy(x[c], alpha_b, yv[c], 0, 0, 1, IntVect(0));
        }
        if (std::sqrt(dot3(sv, sv)) <= tol) { break; }
        prec3(zv, sv);
        apply_op(tv2, zv);
        Real const den_o = dot3(tv2, tv2);
        omega_b = dot3(tv2, sv)/den_o;
        if (bad(omega_b) || omega_b == 0.0_rt) {
            // accept the alpha half-step (already in x), then restart
            for (int c = 0; c < 3; ++c) {
                MultiFab::Copy(rr[c], sv[c], 0, 0, 1, 0);
            }
            restart();
            continue;
        }
        for (int c = 0; c < 3; ++c) {
            MultiFab::Saxpy(x[c], omega_b, zv[c], 0, 0, 1, IntVect(0));
            MultiFab::LinComb(rr[c], 1.0_rt, sv[c], 0, -omega_b, tv2[c], 0,
                              0, 1, IntVect(0));
        }
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(it < max_it,
        "SolveETensorRZ: BiCGStab failed to converge");

    for (int c = 0; c < 3; ++c) {
        MultiFab::Copy(*Efield[c], x[c], 0, 0, 1, 0);
        Efield[c]->FillBoundary(geom.periodicity());
    }
#endif
}

void HybridPICModel::ComputeElectronInertiaNodal (amrex::Real a_theta,
                                                  amrex::Real a_dt,
                                                  bool a_from_jacobian)
{
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    amrex::ignore_unused(a_theta, a_dt, a_from_jacobian);
    WARPX_ABORT_WITH_MESSAGE(
        "hybrid_pic_model.include_electron_inertia is not supported in 1D "
        "radial geometries.");
#else
    ABLASTR_PROFILE("HybridPICModel::ComputeElectronInertiaNodal()");

#if defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::ncomps == 1,
        "hybrid_pic_model.include_electron_inertia supports only the m = 0 "
        "azimuthal mode in RZ");
#endif

    using ablastr::fields::Direction;
    using namespace ablastr::coarsen::sample;

    auto & warpx = WarpX::GetInstance();
    constexpr int lev = 0;
    amrex::Geometry const & geom = warpx.Geom(lev);
    auto const dx_arr = geom.CellSizeArray();

    // Effective electron mass, resolved on first use (see
    // ResolveElectronInertiaMass).
    ResolveElectronInertiaMass();
    amrex::Real const me_over_e = m_electron_inertia_mass / PhysConst::q_e;

    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp, lev);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rho.nComp() >= 2,
        "hybrid_pic_model.include_electron_inertia requires the "
        "theta-implicit hybrid evolve scheme (two charge-density time "
        "levels)");
    const int rho_mid_comp = rho.nComp() / 2;
    amrex::Real const rho_floor = PhysConst::q_e * m_n_floor;
    amrex::Real const floor_w = m_n_floor_smooth_width * rho_floor;
    // Electron-inertia fluid-validity taper: the inertia term is the
    // electron-fluid momentum response, which does not exist at deposit
    // granularity -- in a density-floored cell holding O(1) particles,
    // (m_e/e) dJe/dt / rho_lim couples a particle's own current back
    // onto itself with gain ~ m_e_eff/rho_lim and runs away on a
    // physical (dt-independent) clock. Taper the term to zero through
    // the floor band (the sub-floor halo reverts to the massless-
    // electron model); width in units of rho_floor, 0 disables the
    // taper (legacy behavior).
    amrex::Real const taper_w = m_electron_inertia_floor_taper * rho_floor;

    // Freeze rho(x^n) for the drho/dt leg on the first non-Jacobian
    // evaluation of the step: rho_fp component 0 is a PRE-push deposit, so
    // only that evaluation sees the true step-start density there -- from
    // the second evaluation on it holds the previous evaluation's midpoint
    // positions, and differencing the midpoint component against it drives
    // drho/dt toward zero at convergence (and makes it probe-order noise
    // in finite-difference Jacobian evaluations). Same per-step freeze
    // discipline as the Je histories; same deposit family as the midpoint
    // component it is differenced against. Cleared by
    // RotateElectronInertiaHistory.
    amrex::MultiFab & rho_n_frozen =
        *warpx.m_fields.get("hybrid_rho_n_frozen", lev);
    if (!a_from_jacobian && !m_inertia_rho_n_captured) {
        amrex::MultiFab::Copy(rho_n_frozen, rho, 0, 0, 1,
                              amrex::min(rho_n_frozen.nGrowVect(),
                                         rho.nGrowVect()));
        m_inertia_rho_n_captured = true;
    }

    ablastr::fields::VectorField Jp =
        warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    ablastr::fields::VectorField Ji =
        warpx.m_fields.get_alldirs(FieldType::current_fp, lev);
    amrex::MultiFab const & Je_n =
        *warpx.m_fields.get("hybrid_Je_n_nodal", lev);
    amrex::MultiFab const & Je_nm1 =
        *warpx.m_fields.get("hybrid_Je_nm1_nodal", lev);
    amrex::MultiFab & Ei = *warpx.m_fields.get("hybrid_E_inertial_nodal", lev);

    amrex::GpuArray<int, 3> const coarsen_rr = {1, 1, 1};
    amrex::GpuArray<int, 3> const nodal = {1, 1, 1};
    const amrex::GpuArray<int, 3> J_stag[3] =
        {Jx_IndexType, Jy_IndexType, Jz_IndexType};

    // Scratch: the theta-stage nodal electron current (with one ghost for
    // the advection differences).
    amrex::MultiFab Je_th(Ei.boxArray(), Ei.DistributionMap(), 3,
                          amrex::IntVect(1));
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Je_th, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const & je   = Je_th.array(mfi);
        auto const & jpx  = Jp[0]->const_array(mfi);
        auto const & jpy  = Jp[1]->const_array(mfi);
        auto const & jpz  = Jp[2]->const_array(mfi);
        auto const & jix  = Ji[0]->const_array(mfi);
        auto const & jiy  = Ji[1]->const_array(mfi);
        auto const & jiz  = Ji[2]->const_array(mfi);
        amrex::GpuArray<int, 3> const sx = J_stag[0];
        amrex::GpuArray<int, 3> const sy = J_stag[1];
        amrex::GpuArray<int, 3> const sz = J_stag[2];
        amrex::ParallelFor(mfi.tilebox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            je(i,j,k,0) = Interp(jpx, sx, nodal, coarsen_rr, i, j, k, 0)
                        - Interp(jix, sx, nodal, coarsen_rr, i, j, k, 0);
            je(i,j,k,1) = Interp(jpy, sy, nodal, coarsen_rr, i, j, k, 0)
                        - Interp(jiy, sy, nodal, coarsen_rr, i, j, k, 0);
            je(i,j,k,2) = Interp(jpz, sz, nodal, coarsen_rr, i, j, k, 0)
                        - Interp(jiz, sz, nodal, coarsen_rr, i, j, k, 0);
        });
    }
    // Zero-extend past physical boundaries (the E_L source convention);
    // interior/periodic ghosts from the exchange.
    Je_th.setBndry(0.0_rt);
    Je_th.FillBoundary(geom.periodicity());

    // Save the theta-stage assembly for the end-of-step history rotation
    // (non-Jacobian evaluations only: probe states are perturbed). The last
    // write before FinishFieldUpdate is the converged iterate's assembly.
    if (!a_from_jacobian) {
        amrex::MultiFab & Je_sv =
            *warpx.m_fields.get("hybrid_Je_theta_nodal", lev);
        amrex::MultiFab::Copy(Je_sv, Je_th, 0, 0, 3,
                              amrex::min(Je_sv.nGrowVect(),
                                         Je_th.nGrowVect()));
    }

    // First evaluation ever: seed the history with the initial electron
    // current (see m_inertia_history_initialized) so the quiescent start
    // carries no fake dJe/dt impulse. Never seed from a perturbed
    // Jacobian-probe state.
    if (!m_inertia_history_initialized && !a_from_jacobian) {
        amrex::MultiFab & Je_n_mut =
            *warpx.m_fields.get("hybrid_Je_n_nodal", lev);
        amrex::MultiFab & Je_nm1_mut =
            *warpx.m_fields.get("hybrid_Je_nm1_nodal", lev);
        amrex::MultiFab::Copy(Je_n_mut, Je_th, 0, 0, 3,
                              amrex::min(Je_n_mut.nGrowVect(),
                                         Je_th.nGrowVect()));
        amrex::MultiFab::Copy(Je_nm1_mut, Je_th, 0, 0, 3,
                              amrex::min(Je_nm1_mut.nGrowVect(),
                                         Je_th.nGrowVect()));
        m_inertia_history_initialized = true;
        m_inertia_history_levels = 1;
    }

    // curlcurl_form: capture the step-start plasma current (curl B_old) once
    // per step and keep its nodal theta component as the measured
    // reference of the dJe/dt leg. Differencing against B_old (rather
    // than the iterate's B(theta)) is what leaves the iterate's
    // curl-curl-E content on the operator side of the toroidal solve.
    if (m_esolve_curlcurl) {
        EnsureCurlCurlScratch();
        if (!a_from_jacobian && !m_inertia_jpold_captured) {
            ablastr::fields::VectorField B_old =
                warpx.m_fields.get_alldirs(FieldType::B_old, lev);
            ablastr::fields::VectorField jp_old =
                {m_jp_old_stag[0].get(), m_jp_old_stag[1].get(),
                 m_jp_old_stag[2].get()};
            warpx.get_pointer_fdtd_solver_fp(lev)->CalculateCurrentAmpere(
                jp_old, B_old, warpx.GetEBUpdateEFlag()[lev], lev);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(*m_jp_old_nodal, TilingIfNotGPU());
                 mfi.isValid(); ++mfi) {
                auto const & jo  = m_jp_old_nodal->array(mfi);
                auto const & jor = m_jp_old_stag[0]->const_array(mfi);
                auto const & jot = m_jp_old_stag[1]->const_array(mfi);
                auto const & joz = m_jp_old_stag[2]->const_array(mfi);
                amrex::GpuArray<int, 3> const sx = J_stag[0];
                amrex::GpuArray<int, 3> const sy = J_stag[1];
                amrex::GpuArray<int, 3> const sz = J_stag[2];
                amrex::ParallelFor(mfi.tilebox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    jo(i,j,k,0) =
                        Interp(jor, sx, nodal, coarsen_rr, i, j, k, 0);
                    jo(i,j,k,1) =
                        Interp(jot, sy, nodal, coarsen_rr, i, j, k, 0);
                    jo(i,j,k,2) =
                        Interp(joz, sz, nodal, coarsen_rr, i, j, k, 0);
                });
            }
            m_jp_old_nodal->setBndry(0.0_rt);
            m_jp_old_nodal->FillBoundary(geom.periodicity());
            m_inertia_jpold_captured = true;
        }
    }

    // Assemble E_inertial = +(m_e_eff/(e rho)) [ dJe/dt - (Je/rho) drho/dt
    //                                           - (Je.grad)(Je/rho) ],
    // the Je-form of -(m_e_eff/e) D u_e/Dt with u_e = -Je/rho. The sign
    // and the 1/rho are fixed by the cold-electron limit of the electron
    // momentum equation, E = (m_e/(e^2 n_e)) dJe/dt (the collisionless
    // inertial "resistivity"). dJe/dt uses the theta-extrapolated iterate
    // Je^{n+1} = (Je^theta - (1-theta) Je^n)/theta and the per-step-frozen
    // history {Je^n, Je^{n-1}}; the first step after seeding uses the
    // two-point form.
    const bool bdf2 = m_electron_inertia_bdf2
        && (m_inertia_history_levels >= 2);
    const bool djedt_only = m_electron_inertia_djedt_only;
    amrex::Real const inv_th = 1.0_rt / a_theta;
    amrex::Real const inv_dt = 1.0_rt / a_dt;
    // Three-point (second-order) time-derivative stencil for dJe/dt,
    // centered at the theta stage where Ohm's law is imposed: the
    // quadratic through {Je^{n-1}, Je^n, Je^{n+1}} differentiated at
    // t^{n+theta}. At theta = 1 this is classic BDF2; at theta = 1/2 the
    // Je^{n-1} weight vanishes exactly and it reduces to the two-point
    // midpoint form. An endpoint-BDF2 stencil here (centered at t^{n+1})
    // would be mis-centered by (1-theta) dt relative to the stage and
    // pumps reactive (whistler) modes: for dJ/dt = i w J it amplifies by
    // |zeta| ~ 1 + (w dt)^2 / 2 per step at theta = 1/2.
    amrex::Real const c_p1 = (2.0_rt * a_theta + 1.0_rt) * 0.5_rt * inv_dt;
    amrex::Real const c_0  = -2.0_rt * a_theta * inv_dt;
    amrex::Real const c_m1 = (2.0_rt * a_theta - 1.0_rt) * 0.5_rt * inv_dt;

    // Nodal index bounds for one-sided differences at non-periodic edges
    // (periodic dimensions keep the central stencil through the ghosts --
    // clamping there would make the seam images inconsistent).
    const amrex::Box dom_nd = amrex::convert(geom.Domain(),
                                             amrex::IntVect::TheNodeVector());
    amrex::GpuArray<int, 3> is_per{{1, 1, 1}};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        is_per[d] = geom.isPeriodic(d) ? 1 : 0;
    }
#if defined(WARPX_DIM_RZ)
    amrex::Real const dr = dx_arr[0];
    amrex::Real const dz = dx_arr[1];
    amrex::Real const rmin = geom.ProbLo(0);
#endif

    const bool curlcurl = m_esolve_curlcurl;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Ei, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const & ei     = Ei.array(mfi);
        auto const & je     = Je_th.const_array(mfi);
        auto const & jen    = Je_n.const_array(mfi);
        auto const & jenm1  = Je_nm1.const_array(mfi);
        auto const & rhoa   = rho.const_array(mfi);
        auto const & rhona  = rho_n_frozen.const_array(mfi);
        amrex::Array4<amrex::Real> eia;
        amrex::Array4<amrex::Real const> jpr, jpth, jpz, jold;
        amrex::GpuArray<int, 3> const sx_th = J_stag[0];
        amrex::GpuArray<int, 3> const sy_th = J_stag[1];
        amrex::GpuArray<int, 3> const sz_th = J_stag[2];
        if (curlcurl) {
            eia  = m_ei_curlcurl_theta->array(mfi);
            jpr  = Jp[0]->const_array(mfi);
            jpth = Jp[1]->const_array(mfi);
            jpz  = Jp[2]->const_array(mfi);
            jold = m_jp_old_nodal->const_array(mfi);
        }
        const int mid = rho_mid_comp;
        auto const dlo = amrex::lbound(dom_nd);
        auto const dhi = amrex::ubound(dom_nd);
        amrex::ParallelFor(mfi.tilebox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real const rho_n   = rhona(i, j, k, 0);
            amrex::Real const rho_mid = rhoa(i, j, k, mid);
            // curlcurl_form: the division-free measured inertia numerator for
            // the toroidal solve, (m_e/e) dJe_meas/dt, with the measured
            // electron current differenced against the STEP-START plasma
            // current (curl B_old) so the iterate's curl-curl-E content
            // stays on the operator side. The SOURCE is tapered by
            // electron presence: the Ampere-closure Je = curl B/mu0 - J_i
            // is a grid fiction in vacuum (discrete fringing curl with no
            // electrons to carry it -- measured as a wall-corner Newton
            // stall at the 700 V/m scale on the driven Lenz deck), and the
            // Hewett-Nielson source is the REAL electron current, zero in
            // vacuum by construction. The taper touches only this
            // fictional source; the operator's screening beta stays
            // unfloored (no pedestal in the dynamics) and nothing divides.
            if (curlcurl) {
                amrex::Real const w_src = (taper_w > 0.0_rt)
                    ? 0.5_rt*(1.0_rt + std::tanh(
                        (amrex::max(rho_mid, 0.0_rt) - rho_floor)/taper_w))
                    : ((rho_mid > 0.0_rt) ? 1.0_rt : 0.0_rt);
                amrex::Real djop[3];
                djop[0] = Interp(jpr, sx_th, nodal, coarsen_rr, i, j, k, 0)
                        - jold(i, j, k, 0);
                djop[1] = Interp(jpth, sy_th, nodal, coarsen_rr, i, j, k, 0)
                        - jold(i, j, k, 1);
                djop[2] = Interp(jpz, sz_th, nodal, coarsen_rr, i, j, k, 0)
                        - jold(i, j, k, 2);
                for (int c = 0; c < 3; ++c) {
                    amrex::Real const je_meas = je(i,j,k,c) - djop[c];
                    amrex::Real const je1m =
                        (je_meas - (1.0_rt - a_theta) * jen(i,j,k,c)) * inv_th;
                    amrex::Real const djedt_m = bdf2
                        ? (c_p1 * je1m + c_0 * jen(i,j,k,c)
                           + c_m1 * jenm1(i,j,k,c))
                        : (je1m - jen(i,j,k,c)) * inv_dt;
                    eia(i, j, k, c) = w_src * me_over_e * djedt_m;
                }
            }
            // True vacuum carries no electron fluid: zero the term there
            // (density-floored cells keep their inertia).
            if (rho_mid <= 0.0_rt) {
                ei(i,j,k,0) = 0.0_rt;
                ei(i,j,k,1) = 0.0_rt;
                ei(i,j,k,2) = 0.0_rt;
                return;
            }
            amrex::Real const rho_lim = HybridSmoothFloor(rho_mid, rho_floor, floor_w);
            // rho_mid is the MIDPOINT deposit for every theta (the implicit
            // particle push advances positions by dt/2), so the rate is the
            // half-step difference quotient -- dividing by theta*dt instead
            // is correct only at theta = 1/2.
            amrex::Real const drhodt = (rho_mid - rho_n) * (2.0_rt * inv_dt);

            // Central differences of u = Je/rho at nodes (one-sided at
            // non-periodic edges); rho ghosts follow the deposit exchange.
            amrex::Real u[3], dudx[3], dudz[3];
#if defined(WARPX_DIM_3D)
            amrex::Real dudy[3];
#endif
            for (int c = 0; c < 3; ++c) {
                u[c] = je(i,j,k,c) / rho_lim;
            }
            // Plain nested lambda: implicitly a device lambda in this
            // context (an annotated nested extended lambda is illegal
            // under nvcc).
            auto ucomp = [&] (int ii, int jj, int kk,
                              int c) -> amrex::Real
            {
                amrex::Real const rl = HybridSmoothFloor(
                    amrex::max(rhoa(ii,jj,kk,mid), 0.0_rt), rho_floor, floor_w);
                return je(ii,jj,kk,c) / rl;
            };
#if defined(WARPX_DIM_3D)
            for (int c = 0; c < 3; ++c) {
                const int im = (is_per[0] || i > dlo.x) ? i-1 : i;
                const int ip = (is_per[0] || i < dhi.x) ? i+1 : i;
                dudx[c] = (ucomp(ip,j,k,c) - ucomp(im,j,k,c))
                          / ((ip - im) * dx_arr[0]);
                const int jm = (is_per[1] || j > dlo.y) ? j-1 : j;
                const int jp = (is_per[1] || j < dhi.y) ? j+1 : j;
                dudy[c] = (ucomp(i,jp,k,c) - ucomp(i,jm,k,c))
                          / ((jp - jm) * dx_arr[1]);
                const int km = (is_per[2] || k > dlo.z) ? k-1 : k;
                const int kp = (is_per[2] || k < dhi.z) ? k+1 : k;
                dudz[c] = (ucomp(i,j,kp,c) - ucomp(i,j,km,c))
                          / ((kp - km) * dx_arr[2]);
            }
#elif defined(WARPX_DIM_1D_Z)
            for (int c = 0; c < 3; ++c) {
                dudx[c] = 0.0_rt;
                const int im = (is_per[0] || i > dlo.x) ? i-1 : i;
                const int ip = (is_per[0] || i < dhi.x) ? i+1 : i;
                dudz[c] = (ucomp(ip,j,k,c) - ucomp(im,j,k,c))
                          / ((ip - im) * dx_arr[0]);
            }
#else
            for (int c = 0; c < 3; ++c) {
                const int im = (is_per[0] || i > dlo.x) ? i-1 : i;
                const int ip = (is_per[0] || i < dhi.x) ? i+1 : i;
                dudx[c] = (ucomp(ip,j,k,c) - ucomp(im,j,k,c))
                          / ((ip - im) * dx_arr[0]);
                const int jm = (is_per[1] || j > dlo.y) ? j-1 : j;
                const int jp = (is_per[1] || j < dhi.y) ? j+1 : j;
                dudz[c] = (ucomp(i,jp,k,c) - ucomp(i,jm,k,c))
                          / ((jp - jm) * dx_arr[1]);
            }
#endif
            // (Je.grad)u, with the cylindrical metric terms in RZ (m = 0):
            //   (A.grad B)_r     += -A_t B_t / r
            //   (A.grad B)_theta += +A_t B_r / r
            // both zero on the axis by m = 0 regularity (A_t, B_t ~ r).
            amrex::Real adv[3];
#if defined(WARPX_DIM_3D)
            for (int c = 0; c < 3; ++c) {
                adv[c] = je(i,j,k,0)*dudx[c] + je(i,j,k,1)*dudy[c]
                       + je(i,j,k,2)*dudz[c];
            }
#elif defined(WARPX_DIM_RZ)
            for (int c = 0; c < 3; ++c) {
                adv[c] = je(i,j,k,0)*dudx[c] + je(i,j,k,2)*dudz[c];
            }
            {
                amrex::Real const r = rmin + i * dr;
                if (r > 0.0_rt) {
                    adv[0] -= je(i,j,k,1) * u[1] / r;
                    adv[1] += je(i,j,k,1) * u[0] / r;
                }
            }
            amrex::ignore_unused(dz);
#else // WARPX_DIM_XZ
            for (int c = 0; c < 3; ++c) {
                adv[c] = je(i,j,k,0)*dudx[c] + je(i,j,k,2)*dudz[c];
            }
#endif

            amrex::Real const w_taper = (taper_w > 0.0_rt)
                ? 0.5_rt*(1.0_rt + std::tanh((rho_mid - rho_floor)/taper_w))
                : 1.0_rt;
            for (int c = 0; c < 3; ++c) {
                // Je^{n+1} from the theta-stage iterate:
                // Je1 = (Je^theta - (1-theta) Je^n)/theta.
                // dJe/dt: the stage-centered three-point stencil (BDF2 at
                // theta = 1, two-point midpoint form at theta = 1/2), or
                // the two-point form when electron_inertia_bdf2 = 0.
                amrex::Real const je1 =
                    (je(i,j,k,c) - (1.0_rt - a_theta) * jen(i,j,k,c)) * inv_th;
                amrex::Real const djedt = bdf2
                    ? (c_p1 * je1 + c_0 * jen(i,j,k,c)
                       + c_m1 * jenm1(i,j,k,c))
                    : (je1 - jen(i,j,k,c)) * inv_dt;
                ei(i,j,k,c) = w_taper * me_over_e
                    * (djedt_only ? djedt
                                  : (djedt - u[c] * drhodt - adv[c]))
                    / rho_lim;
            }
        });
    }
    Ei.setBndry(0.0_rt);
    Ei.FillBoundary(geom.periodicity());
    if (m_esolve_curlcurl) {
        m_ei_curlcurl_theta->setBndry(0.0_rt);
        m_ei_curlcurl_theta->FillBoundary(geom.periodicity());
    }
#endif
}

void HybridPICModel::RotateElectronInertiaHistory (amrex::Real a_theta)
{
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    amrex::ignore_unused(a_theta);
    WARPX_ABORT_WITH_MESSAGE(
        "hybrid_pic_model.include_electron_inertia is not supported in 1D "
        "radial geometries.");
#else
    auto & warpx = WarpX::GetInstance();
    constexpr int lev = 0;

    amrex::MultiFab & Je_n = *warpx.m_fields.get("hybrid_Je_n_nodal", lev);
    amrex::MultiFab & Je_nm1 =
        *warpx.m_fields.get("hybrid_Je_nm1_nodal", lev);

    amrex::MultiFab::Copy(Je_nm1, Je_n, 0, 0, 3, Je_nm1.nGrowVect());
    if (m_inertia_history_levels < 2) { ++m_inertia_history_levels; }

    if (m_electron_inertia_extrapolated_history) {
        // Legacy store: Je^{n+1} = (Je^theta - (1-theta) Je^n)/theta.
        // DEFECTIVE at theta = 1/2: the stored value is a function of the
        // previously stored value with eigenvalue -(1-theta)/theta = -1
        // (marginal), and the E_inertial feedback destabilizes it into a
        // period-2 ringing of the near-null-region field that breaks the
        // step on a physical (dt-independent) clock. Kept only as an
        // opt-in comparison knob.
        amrex::MultiFab const & Je_th =
            *warpx.m_fields.get("hybrid_Je_theta_nodal", lev);
        amrex::MultiFab::LinComb(Je_n,
            1.0_rt / a_theta, Je_th, 0,
            1.0_rt - 1.0_rt / a_theta, Je_nm1, 0,
            0, 3, Je_n.nGrowVect());
    } else {
        // Measured store (default): assemble Je^{n+1} from the DELIVERED
        // end-of-step state -- the caller runs this after the delivered
        // plasma-current refresh, so hybrid_current_fp_plasma holds
        // J_plasma^{n+1} = curl(B^{n+1})/mu0 (- J_ext, - the Darwin
        // displacement piece), and current_fp holds the same ion-deposit
        // family every theta-stage assembly used. The stored history is
        // never a function of previously stored values, so no recursion
        // exists to ring. The theta extrapolation survives only INSIDE
        // the residual evaluation (Je^{n+1} from Je^theta), recomputed
        // fresh each evaluation and never fed back.
        amrex::ignore_unused(a_theta);
        ablastr::fields::VectorField Jp =
            warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
        ablastr::fields::VectorField Ji =
            warpx.m_fields.get_alldirs(FieldType::current_fp, lev);
        auto const & geom = warpx.Geom(lev);
        amrex::GpuArray<int, 3> const coarsen_rr = {1, 1, 1};
        amrex::GpuArray<int, 3> const nodal = {1, 1, 1};
        const amrex::GpuArray<int, 3> J_stag[3] =
            {Jx_IndexType, Jy_IndexType, Jz_IndexType};
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Je_n, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const & je   = Je_n.array(mfi);
            auto const & jpx  = Jp[0]->const_array(mfi);
            auto const & jpy  = Jp[1]->const_array(mfi);
            auto const & jpz  = Jp[2]->const_array(mfi);
            auto const & jix  = Ji[0]->const_array(mfi);
            auto const & jiy  = Ji[1]->const_array(mfi);
            auto const & jiz  = Ji[2]->const_array(mfi);
            amrex::GpuArray<int, 3> const sx = J_stag[0];
            amrex::GpuArray<int, 3> const sy = J_stag[1];
            amrex::GpuArray<int, 3> const sz = J_stag[2];
            using ablastr::coarsen::sample::Interp;
            amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                je(i,j,k,0) = Interp(jpx, sx, nodal, coarsen_rr, i, j, k, 0)
                            - Interp(jix, sx, nodal, coarsen_rr, i, j, k, 0);
                je(i,j,k,1) = Interp(jpy, sy, nodal, coarsen_rr, i, j, k, 0)
                            - Interp(jiy, sy, nodal, coarsen_rr, i, j, k, 0);
                je(i,j,k,2) = Interp(jpz, sz, nodal, coarsen_rr, i, j, k, 0)
                            - Interp(jiz, sz, nodal, coarsen_rr, i, j, k, 0);
            });
        }
        // Same boundary conventions as the theta-stage assembly.
        Je_n.setBndry(0.0_rt);
        Je_n.FillBoundary(geom.periodicity());
    }

    // Re-arm the step-start density capture for the next step's first
    // evaluation (see ComputeElectronInertiaNodal), and the curlcurl_form
    // step-start plasma-current capture with it.
    m_inertia_rho_n_captured = false;
    m_inertia_jpold_captured = false;
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
    // Heating-side resistivity and gate (may differ from the E-solve
    // eta and the Ohm floor; see the m_eta_heating member doc).
    auto const eta           = m_eta_heating;
    auto const rho_heat_gate = PhysConst::q_e *
        (m_joule_heating_n_min >= 0.0_rt ? m_joule_heating_n_min
                                         : m_n_floor);
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
                if (rho_val <= rho_heat_gate) { return; }
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
            // Density: the raw midpoint deposit (second time-level
            // component), NOT a time extrapolation toward n+theta.
            // Extrapolating between the step-start and midpoint DEPOSITS,
            //   (1-2 theta) rho^n + 2 theta rho^{n+1/2},
            // amplifies the step-alternating (deposit-noise) component of
            // their difference by (2 theta - 1) and rectifies it through
            // Pe into the slow dynamics: a dt-INDEPENDENT spurious response
            // ~ 13% x (theta - 1/2) of the compression amplitude on the
            // adiabat deck (6.5e-3 at theta = 1; the 2026-07-16 order-study
            // offset). Ratio/geometric extrapolation does not help (the
            // alternating component still extrapolates). Deposits are
            // consumed raw -- the same rule the RZ deposit conventions and
            // the QDSMC recovery already follow; the O((theta-1/2) dt)
            // label error this leaves in Pe is subdominant to the
            // theta > 1/2 path's own first-order time biasing.
            amrex::Real const rho_val =
                std::max(rho_arr(i,j,k,c_half), rho_floor);
            amrex::ignore_unused(rho_old_arr);
            amrex::Real const ne      = rho_val / PhysConst::q_e;
            amrex::Real const Te_th   = (1.0_rt - th) * Te_old_arr(i,j,k)
                                      + th * Te_arr(i,j,k);
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
            // Step 3a: markers to the temporal midpoint of the characteristic.
            // (The half-push samples V_e at the home position; a quarter-point
            // V_e resample -- one implicit-midpoint iteration for the
            // half-characteristic -- was tested in the 2026-07-30 order study
            // and changed the solution by only ~3e-9 relative: the
            // displacement-sampling error is NOT the coherent O(dt) term on
            // the adiabat deck. See NOTES-theta-offset.md.)
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

    // Adjust the number of substeps for the next RKF45/RK4 half-step.
    // Jump up immediately when this half needed more attempts; otherwise
    // slowly relax toward target = 2*n_attempts.
    // Blend on the half-step counts M = m_substeps/2 and N = n_attempts via
    // integer arithmetic: relaxed = 2*((19*M + N)/20). That is exactly the
    // 95/5 blend, stays even, holds when N == M, and actually decays when
    // N < M (e.g. m=40, n_attempts=10 → 38 → … → 20). Floating-point
    // 0.95*M+0.05*N can undershoot M slightly so floor would leak even at
    // equilibrium.
    {
        const int target = 2 * n_attempts;
        if (m_substeps < target) {
            m_substeps = target;
        } else {
            const int M = m_substeps / 2;
            const int N = n_attempts;
            const int relaxed = 2 * ((19 * M + N) / 20);
            m_substeps = std::max(relaxed, 2);
        }
        // Stay within the abort budget so the controller cannot request more
        // substeps than max_substep_attempts allows.
        if (m_substeps > m_max_substep_attempts) {
            m_substeps = m_max_substep_attempts - (m_max_substep_attempts % 2);
            m_substeps = std::max(m_substeps, 2);
        }
    }

    if (WarpX::GetInstance().Verbose()) {
        amrex::Print() << "B-field update "
            << (subcycling_half == SubcyclingHalf::FirstHalf ? "1st" : "2nd") << " half"
            << ": " << n_accepted << " accepted, "
            << (n_attempts - n_accepted) << " rejected substeps"
            << " (dt_sub_final/dt_half = " << dt_sub / dt_half
            << ", m_substeps = " << m_substeps << ")\n";
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
    // Refresh E ghosts before the Faraday push reads them: always on a
    // collocated grid (the nodal curl reads ghost E), and on the conformal
    // wall path, whose ECT circulation reads cross-box ghost E edges.
    if (Bz_IndexType[0] == Ez_IndexType[0] || m_use_conformal_eb) {
        warpx.FillBoundaryE(ng, nodal_sync);
    }

#ifdef AMREX_USE_EB
    // Conformal wall: recompute the per-face EMF circulations (ECTRhofield)
    // from the new Ohm E so the Faraday push is consistent with Ohm's law.
    if (m_use_conformal_eb) {
        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            warpx.get_pointer_fdtd_solver_fp(lev)->EvolveECTRho(
                Efield[lev],
                warpx.m_fields.get_alldirs(FieldType::edge_lengths, lev),
                warpx.m_fields.get_alldirs(FieldType::face_areas, lev),
                warpx.m_fields.get_alldirs(FieldType::ECTRhofield, lev),
                lev);
        }
    }
#endif

    // Push forward the B-field using Faraday's law
    warpx.EvolveB(dt, subcycling_half, t_old);
    warpx.FillBoundaryB(ng, nodal_sync);
}
