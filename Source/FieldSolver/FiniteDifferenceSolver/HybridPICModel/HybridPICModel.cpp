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

#include "QdsmcRKIntegrator.H"

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

    // Time-advance scheme for the QDSMC energy step (only consulted when
    // solve_electron_energy_equation is on). Default "pc" (selected
    // 2026-08-04): second-order predictor-corrector, transport between the
    // two B half-pushes. "euler" reproduces the #6982 scheme bit-for-bit
    // (first-order control); "leapfrog" is the half-staggered alternative.
    {
        std::string advance_str = "pc";
        pp_hybrid.query("qdsmc_time_advance", advance_str);
        if (advance_str == "euler") {
            m_qdsmc_time_advance = QdsmcTimeAdvance::Euler;
        } else if (advance_str == "leapfrog") {
            m_qdsmc_time_advance = QdsmcTimeAdvance::Leapfrog;
        } else if (advance_str == "pc") {
            m_qdsmc_time_advance = QdsmcTimeAdvance::PC;
        } else {
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.qdsmc_time_advance must be one of "
                "'euler', 'leapfrog', 'pc' (got '" + advance_str + "')");
        }
    }

    // Half-gradient-corrected deposit (2nd-order spatial remap; see the
    // member doc). Default on; disable together with qdsmc_time_advance =
    // euler to recover the #6982 scheme bit-for-bit.
    pp_hybrid.query("qdsmc_gradient_deposit", m_qdsmc_gradient_deposit);

    // Ito tensor thermal conduction (split substep on u = 3/2 n_e k_B T_e;
    // see QdsmcConductionOnce). Enabled by specifying the parallel
    // conductivity parser; kappa_perp is optional (default 0 -- the full
    // tensor always ships, kappa_perp = 0 is just the trivial setting).
    m_include_thermal_conduction =
        pp_hybrid.query("qdsmc_kappa_par(n,Te,t)", m_kappa_par_expression);
    pp_hybrid.query("qdsmc_kappa_perp(n,Te,t)", m_kappa_perp_expression);
    pp_hybrid.query("qdsmc_conduction_isotropic", m_cond_isotropic);
    utils::parser::queryWithParser(pp_hybrid, "qdsmc_conduction_iso_B",
                                   m_cond_iso_B);
    {
        std::string op = "sde";
        pp_hybrid.query("qdsmc_conduction_operator", op);
        if (op == "sde") { m_cond_operator = 0; }
        else if (op == "fd") { m_cond_operator = 1; }
        else {
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.qdsmc_conduction_operator must be 'sde' "
                "or 'fd'");
        }
        pp_hybrid.query("qdsmc_conduction_fd_order", m_cond_fd_order);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_cond_fd_order == 2 || m_cond_fd_order == 4,
            "hybrid_pic_model.qdsmc_conduction_fd_order must be 2 or 4");
        std::string fdlim = "smart";
        pp_hybrid.query("qdsmc_conduction_fd_limiter", fdlim);
        if (fdlim == "none") { m_cond_fd_limiter = 0; }
        else if (fdlim == "upwind1") { m_cond_fd_limiter = 1; }
        else if (fdlim == "smart") { m_cond_fd_limiter = 2; }
        else {
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.qdsmc_conduction_fd_limiter must be "
                "'none', 'upwind1' or 'smart'");
        }
        utils::parser::queryWithParser(pp_hybrid,
            "qdsmc_conduction_fd_cfl", m_cond_fd_cfl);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_cond_fd_cfl > 0._rt && m_cond_fd_cfl <= 1._rt,
            "hybrid_pic_model.qdsmc_conduction_fd_cfl must be in (0, 1]");
        pp_hybrid.query("qdsmc_conduction_fd_max_subcycles",
                        m_cond_fd_max_subcycles);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_cond_fd_max_subcycles > 0,
            "hybrid_pic_model.qdsmc_conduction_fd_max_subcycles must be "
            "positive");
        std::string fdtime = "ssprk2";
        pp_hybrid.query("qdsmc_conduction_fd_time", fdtime);
        if (fdtime == "ssprk2") { m_cond_fd_time = 0; }
        else if (fdtime == "rkf45") { m_cond_fd_time = 1; }
        else {
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.qdsmc_conduction_fd_time must be "
                "'ssprk2' or 'rkf45'");
        }
        utils::parser::queryWithParser(pp_hybrid,
            "qdsmc_conduction_fd_rtol", m_cond_fd_rtol);
        utils::parser::queryWithParser(pp_hybrid,
            "qdsmc_conduction_fd_atol", m_cond_fd_atol);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_cond_fd_rtol > 0._rt && m_cond_fd_atol > 0._rt,
            "hybrid_pic_model.qdsmc_conduction_fd_rtol/_atol must be "
            "positive");
        std::string top = "markers";
        pp_hybrid.query("qdsmc_transport_operator", top);
        if (top == "markers") { m_qdsmc_transport_operator = 0; }
        else if (top == "grid") { m_qdsmc_transport_operator = 1; }
        else {
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.qdsmc_transport_operator must be "
                "'markers' or 'grid'");
        }
    }
    {
        std::vector<int> npts;
        pp_hybrid.queryarr("qdsmc_conduction_quadrature_points", npts);
        if (npts.size() == 1) {
            m_cond_npts_par = m_cond_npts_perp = npts[0];
        } else if (npts.size() >= 2) {
            m_cond_npts_par  = npts[0];
            m_cond_npts_perp = npts[1];
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_cond_npts_par >= 2 && m_cond_npts_par <= 7 &&
            m_cond_npts_perp >= 2 && m_cond_npts_perp <= 7,
            "hybrid_pic_model.qdsmc_conduction_quadrature_points entries "
            "must be in [2, 7]");
    }
    utils::parser::queryWithParser(pp_hybrid,
        "qdsmc_conduction_flux_limit_factor", m_cond_flux_limit_factor);
    utils::parser::queryWithParser(pp_hybrid,
        "qdsmc_conduction_max_hop", m_cond_max_hop);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_cond_max_hop > 0._rt,
        "hybrid_pic_model.qdsmc_conduction_max_hop must be positive");
    pp_hybrid.query("qdsmc_conduction_vacuum_fast_front",
                    m_cond_vacuum_fast_front);
    {
        // Default = the production/gate-passed form: fluxform split sweeps
        // with the closed floor faces and the per-line EB machinery.
        // scatter/layer are the historical control arms -- they keep pre-EB
        // behavior (no floor/EB masks), which next to a maintained wall
        // band acts as a perpetual bath donor (measured on the liftoff
        // slab: +12.8% open-set energy per 400 steps with the insulating
        // wall's Te fill). Conduction itself is off unless the kappa_par
        // parser is given, so decks without conduction are unaffected.
        std::string form = "fluxform";
        pp_hybrid.query("qdsmc_conduction_form", form);
        if (form == "scatter") { m_cond_form = 0; }
        else if (form == "layer") { m_cond_form = 1; }
        else if (form == "fluxform") { m_cond_form = 2; }
        else {
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.qdsmc_conduction_form must be 'scatter', "
                "'layer' or 'fluxform'");
        }
        std::string interp = "monocubic";
        pp_hybrid.query("qdsmc_conduction_interp", interp);
        if (interp == "linear") { m_cond_interp = 0; }
        else if (interp == "monocubic") { m_cond_interp = 1; }
        else if (interp == "keys") { m_cond_interp = 2; }
        else {
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.qdsmc_conduction_interp must be 'linear', "
                "'monocubic' or 'keys'");
        }
        pp_hybrid.query("qdsmc_conduction_curved_feet", m_cond_curved_feet);
        // Mehrstellen-style isotropized perpendicular launch: split
        // chi_perp equally between the (e1, e2) quadrature lattice and its
        // 45-degree-rotated pair (sigma/sqrt(2) each, half weight each).
        // The equal variance split cancels the leading cos(4 theta)
        // fourth-moment anisotropy of the axis-pair launch (the
        // star-shaped diffusion pattern of point sources), and gives
        // diagonal hops first-class per-axis clamping. Doubles the
        // perpendicular branch count when on. Default off (bit-identical
        // baselines).
        pp_hybrid.query("qdsmc_conduction_isotropic_launch",
                        m_cond_iso_launch);
        pp_hybrid.query("qdsmc_conduction_conserve_fixup",
                        m_cond_conserve_fixup);
        std::string depk = "hat";
        pp_hybrid.query("qdsmc_conduction_deposit_kernel", depk);
        if (depk == "hat") { m_cond_deposit_kernel = 0; }
        else if (depk == "keys") { m_cond_deposit_kernel = 1; }
        else {
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.qdsmc_conduction_deposit_kernel must be "
                "'hat' or 'keys'");
        }
        pp_hybrid.query("qdsmc_conduction_compensate", m_cond_compensate);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_cond_compensate ||
            (m_cond_deposit_kernel == 0 && !m_qdsmc_gradient_deposit),
            "hybrid_pic_model.qdsmc_conduction_compensate requires the hat "
            "deposit kernel and qdsmc_gradient_deposit = 0 (the FCT pass "
            "replaces the B1 correction)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !(m_cond_form == 2 && m_cond_compensate),
            "hybrid_pic_model.qdsmc_conduction_compensate is a scatter-form "
            "option; the fluxform remap is conservative and monotone by "
            "construction");
        std::string slim = "mc";
        pp_hybrid.query("qdsmc_conduction_slope_limiter", slim);
        if (slim == "mc") { m_cond_slope_limiter = 0; }
        else if (slim == "none") { m_cond_slope_limiter = 1; }
        else {
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.qdsmc_conduction_slope_limiter must be "
                "'mc' or 'none'");
        }
        pp_hybrid.query("qdsmc_conduction_fluxform_unsplit",
                        m_cond_ff_unsplit);
        std::string recon = "ppm";
        bool const recon_given =
            pp_hybrid.query("qdsmc_conduction_reconstruction", recon);
        if (recon == "plm") { m_cond_reconstruction = 0; }
        else if (recon == "ppm") { m_cond_reconstruction = 1; }
        else {
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.qdsmc_conduction_reconstruction must be "
                "'plm' or 'ppm'");
        }
        if (m_cond_ff_unsplit && !recon_given) {
            // the unsplit control arm's piece bookkeeping is PLM-exact;
            // the ppm DEFAULT quietly steps aside rather than aborting
            m_cond_reconstruction = 0;
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !(m_cond_reconstruction == 1 && m_cond_ff_unsplit),
            "hybrid_pic_model.qdsmc_conduction_reconstruction = ppm is "
            "implemented for the split fluxform sweeps only (the unsplit "
            "donor piece bookkeeping is PLM-exact); set it to 'plm' when "
            "using qdsmc_conduction_fluxform_unsplit");
        pp_hybrid.query("qdsmc_conduction_closed_floor_faces",
                        m_cond_closed_floor_faces);
        pp_hybrid.query("qdsmc_eb_marker_reflect", m_qdsmc_eb_marker_reflect);
        std::string ebbc = "adiabatic";
        pp_hybrid.query("qdsmc_conduction_eb_bc", ebbc);
        pp_hybrid.query("qdsmc_conduction_eb_ring", m_cond_eb_ring);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_cond_eb_ring >= 1 && m_cond_eb_ring <= 3,
            "hybrid_pic_model.qdsmc_conduction_eb_ring must be in [1, 3]");
        if (ebbc == "adiabatic") { m_cond_eb_bc = 0; }
        else if (ebbc == "isothermal") {
            m_cond_eb_bc = 1;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                pp_hybrid.query("qdsmc_conduction_eb_Te(x,y,z)",
                                m_cond_eb_Te_expression),
                "hybrid_pic_model.qdsmc_conduction_eb_bc = isothermal "
                "requires qdsmc_conduction_eb_Te(x,y,z) [eV]");
        }
        else {
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.qdsmc_conduction_eb_bc must be "
                "'adiabatic' or 'isothermal'");
        }
        // Thrust-D domain-face BCs (per grid dim, lo/hi)
        for (int side = 0; side < 2; ++side) {
            std::string const sfx = (side == 0) ? "_lo" : "_hi";
            std::vector<std::string> types;
            pp_hybrid.queryarr(("qdsmc_conduction_bc" + sfx).c_str(),
                               types);
            if (types.empty()) { continue; }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                types.size() == AMREX_SPACEDIM,
                "hybrid_pic_model.qdsmc_conduction_bc_lo/_hi needs one "
                "entry per grid dimension");
            std::vector<amrex::Real> vte, vq;
            utils::parser::queryArrWithParser(pp_hybrid,
                ("qdsmc_conduction_bc_Te" + sfx).c_str(), vte);
            utils::parser::queryArrWithParser(pp_hybrid,
                ("qdsmc_conduction_bc_q" + sfx).c_str(), vq);
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (types[d] == "adiabatic") { m_cond_bc[d][side] = 0; }
                else if (types[d] == "isothermal") {
                    m_cond_bc[d][side] = 1;
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        vte.size() == AMREX_SPACEDIM,
                        "isothermal qdsmc conduction BC needs "
                        "qdsmc_conduction_bc_Te_lo/_hi (eV, one entry "
                        "per grid dimension)");
                    m_cond_bc_Te[d][side] = vte[d];
                }
                else if (types[d] == "flux") {
                    m_cond_bc[d][side] = 2;
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        vq.size() == AMREX_SPACEDIM,
                        "flux qdsmc conduction BC needs "
                        "qdsmc_conduction_bc_q_lo/_hi (W/m^2, one "
                        "entry per grid dimension)");
                    m_cond_bc_q[d][side] = vq[d];
                }
                else {
                    WARPX_ABORT_WITH_MESSAGE(
                        "hybrid_pic_model.qdsmc_conduction_bc entries "
                        "must be 'adiabatic', 'isothermal' or 'flux'");
                }
            }
        }
        std::string fctl = "bb";
        pp_hybrid.query("qdsmc_conduction_fct_limiter", fctl);
        if (fctl == "bb") { m_cond_fct_limiter = 0; }
        else if (fctl == "zalesak") { m_cond_fct_limiter = 1; }
        else if (fctl == "none") { m_cond_fct_limiter = 2; }
        else {
            WARPX_ABORT_WITH_MESSAGE(
                "hybrid_pic_model.qdsmc_conduction_fct_limiter must be "
                "'bb', 'zalesak' or 'none'");
        }
    }
#if defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_include_thermal_conduction,
        "hybrid_pic_model.qdsmc_kappa_par: QDSMC thermal conduction is not "
        "supported in RZ geometry yet (the daughter deposit is Cartesian).");
#endif
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

    // Biermann battery: keep -grad Pe/(e n) in the Faraday-solve E. The
    // closure-era drop relies on curl(grad Pe/(e n)) = 0, i.e. grad Pe
    // parallel to grad n -- guaranteed by a barotropic Pe(n) closure,
    // broken by the electron energy equation (conduction/Joule/Q_ei
    // decouple Te from n). The dropped physics is the battery term
    // (grad Pe x grad n)/(e n^2). Default off (recorded baselines).
    pp_hybrid.query("include_biermann_battery", m_include_biermann_battery);

    // Te-threshold Joule redirection: heat electrons where Te < threshold,
    // deposit the Joule energy to ions where Te >= threshold. Armed by
    // specifying a threshold >= 0 (in eV).
    {
        bool dep_redirect = false;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_hybrid.query("redirect_joule_to_ions", dep_redirect),
            "hybrid_pic_model.redirect_joule_to_ions is retired: the "
            "redirect is armed by setting "
            "hybrid_pic_model.joule_redirect_Te_threshold >= 0 (in eV).");
    }
    utils::parser::queryWithParser(pp_hybrid, "joule_redirect_Te_threshold", m_joule_redirect_Te_eV);
    m_joule_redirect_to_ions = (m_joule_redirect_Te_eV >= 0._rt);

    // Redirect hardening knobs (2026-08 liftoff runaway forensics; see the
    // member docs): escape hatch for the undamped-redirect guard rail, the
    // per-particle kick cap, and the redirect density gate.
    pp_hybrid.query("joule_redirect_allow_undamped", m_joule_redirect_allow_undamped);
    utils::parser::queryWithParser(pp_hybrid, "joule_redirect_kick_cap_vth_frac",
                                   m_joule_redirect_kick_cap_vth_frac);
    utils::parser::queryWithParser(pp_hybrid, "joule_redirect_n_min_factor",
                                   m_joule_redirect_n_min_factor);

    // Physical-eta Joule source: separate resistivity for the heating (the
    // E-solve keeps plasma_resistivity with its numerical vacuum ramp), plus
    // an independent density gate for where heating is permitted at all.
    m_has_heating_resistivity =
        pp_hybrid.query("joule_heating_resistivity(rho,J,Te,t)", m_eta_heating_expression);
    utils::parser::queryWithParser(pp_hybrid, "joule_heating_n_min", m_joule_heating_n_min);

    // General Te limiter with ion shunt (see member doc): any-channel
    // excess above the threshold goes to the ions through the redirect
    // machinery (kick cap / density gate / guard rail all apply).
    utils::parser::queryWithParser(pp_hybrid, "Te_shunt_threshold", m_te_shunt_eV);

    // Graceful Te-runaway abort ceiling [eV] and the dropped-energy tally
    // print cadence (the print only fires when a decline channel is armed).
    utils::parser::queryWithParser(pp_hybrid, "Te_abort_threshold", m_te_abort_threshold_eV);
    pp_hybrid.query("joule_dropped_energy_print_interval", m_joule_dropped_print_interval);

    // Quarantine/contamination tally class boundary [m^-3] (see member doc;
    // instrument arms only, off by default).
    utils::parser::queryWithParser(pp_hybrid, "qdsmc_contamination_n_boundary",
                                   m_contam_n_boundary);

    // Per-stage open-set energy budget (instrument; pc scheme only).
    pp_hybrid.query("qdsmc_energy_budget", m_energy_budget);

    // Cliff-limited entropy deposit (option-b K-diffusion heat-pump fix).
    pp_hybrid.query("qdsmc_cliff_limited_deposit", m_cliff_limited_deposit);
    utils::parser::queryWithParser(pp_hybrid, "qdsmc_cliff_deposit_r1",
                                   m_cliff_deposit_r1);
    utils::parser::queryWithParser(pp_hybrid, "qdsmc_cliff_deposit_r2",
                                   m_cliff_deposit_r2);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_cliff_limited_deposit ||
        (m_qdsmc_gradient_deposit &&
         m_cliff_deposit_r2 > m_cliff_deposit_r1 &&
         m_cliff_deposit_r1 >= 0.0),
        "hybrid_pic_model.qdsmc_cliff_limited_deposit requires "
        "qdsmc_gradient_deposit = 1 and 0 <= r1 < r2");

    // Electron-ion thermal equilibration (Q_ei) on T_e:
    //   Q_ei = 3 n_e k_B nu_ei (T_e - T_i),  applied per ion species weighted by
    //   n_s/n_e, cooling T_e toward T_i. nu_ei[1/s] comes from the
    //   electron_ion_relaxation_rate(rho,Te,Ti,t) parser (rho [C/m^3], Te,Ti [eV]).
    //   The matching ion heating is deposited conservatively, so the exchange
    //   conserves energy. Enabled by specifying the rate expression (only
    //   consulted when solve_electron_energy_equation is on).
    {
        bool dep_relax = false;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp_hybrid.query("include_temperature_relaxation", dep_relax),
            "hybrid_pic_model.include_temperature_relaxation is retired: the "
            "Q_ei exchange is enabled by specifying "
            "hybrid_pic_model.electron_ion_relaxation_rate(rho,Te,Ti,t).");
    }
    m_include_temperature_relaxation =
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
    // With the energy equation on, T_e is flagged into checkpoint/restart:
    // it is evolved state there and cannot be re-derived after a restart
    // (the adiabat seed would discard the evolved thermal structure).
    fields.alloc_init(FieldType::hybrid_electron_temperature_fp,
        lev, amrex::convert(ba, rho_nodal_flag),
        dm, ncomps, ngRho, 0.0_rt,
        true, true, m_solve_electron_energy_equation);

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

        // leapfrog time advance: previous half-level electron pressure
        // (Pe^{n-1/2}) and the extrapolated integer-time pressure staged for
        // the final E-solve (see ApplyQdsmcPeExtrapolation).
        if (m_qdsmc_time_advance == QdsmcTimeAdvance::Leapfrog) {
            fields.alloc_init("hybrid_qdsmc_pe_prev_fp",
                lev, amrex::convert(ba, rho_nodal_flag),
                dm, ncomps, ngRho, 0.0_rt);
            fields.alloc_init("hybrid_qdsmc_pe_ext_fp",
                lev, amrex::convert(ba, rho_nodal_flag),
                dm, ncomps, ngRho, 0.0_rt);
        }
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
    // as the curl of B minus any external current. Under the QDSMC energy
    // equation it is carried state across steps (m_qdsmc_J_plasma_valid):
    // checkpoint it so a restart continues with the carried value instead
    // of a recompute (bit-consistent restart continuation).
    fields.alloc_init(FieldType::hybrid_current_fp_plasma, Direction{0},
        lev, amrex::convert(ba, jx_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt,
        true, true, m_solve_electron_energy_equation);
    fields.alloc_init(FieldType::hybrid_current_fp_plasma, Direction{1},
        lev, amrex::convert(ba, jy_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt,
        true, true, m_solve_electron_energy_equation);
    fields.alloc_init(FieldType::hybrid_current_fp_plasma, Direction{2},
        lev, amrex::convert(ba, jz_nodal_flag),
        dm, ncomps, ngJ, 0.0_rt,
        true, true, m_solve_electron_energy_equation);

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

    const std::set<std::string> resistivity_symbols = m_resistivity_parser->symbols();
    m_resistivity_has_J_dependence += resistivity_symbols.count("J");

    // Electron-ion energy-equilibration rate nu_ei(rho,Te,Ti,t) for the Q_ei term.
    m_nu_ei_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_nu_ei_expression, {"rho","Te","Ti","t"}));
    m_nu_ei = m_nu_ei_parser->compile<4>();

    // Thermal conductivities kappa(n [m^-3], Te [eV], t [s]) in W/(m K) for
    // the Ito conduction substep (chi = kappa / (3/2 n_e k_B)).
    m_kappa_par_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_kappa_par_expression, {"n","Te","t"}));
    m_kappa_par = m_kappa_par_parser->compile<3>();
    m_kappa_perp_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_kappa_perp_expression, {"n","Te","t"}));
    m_kappa_perp = m_kappa_perp_parser->compile<3>();

    // Isothermal-EB bath temperature Te(x,y,z) [eV] for the conduction
    // EB BC (evaluated at covered node positions).
    m_cond_eb_Te_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_cond_eb_Te_expression, {"x","y","z"}));
    m_cond_eb_Te = m_cond_eb_Te_parser->compile<3>();


    // Separate Joule-heating resistivity (physical eta; the E-solve keeps the
    // numerical vacuum-regularizer ramp). Compiled unconditionally ("0.0"
    // when unset) but only consulted when m_has_heating_resistivity.
    m_heating_resistivity_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_eta_heating_expression, {"rho","J","Te","t"}));
    m_eta_heating = m_heating_resistivity_parser->compile<4>();

    // Guard rail (2026-08 liftoff runaway forensics): without a relaxation
    // channel the redirect's OU ion-heating operator has no drag leg and
    // runs as pure undamped diffusion -- measured anti-stabilizing (arm SR
    // ran 10x hotter than its bit-identical control within 600 steps of the
    // redirect engaging, via the ion-kick current-noise feedback). Refuse to
    // run this configuration unless the user explicitly opts back in.
    if ((m_joule_redirect_to_ions || m_te_shunt_eV > 0.0) &&
        !m_include_temperature_relaxation &&
        !m_joule_redirect_allow_undamped) {
        WARPX_ABORT_WITH_MESSAGE(
            "hybrid_pic_model.joule_redirect_Te_threshold / "
            "Te_shunt_threshold is set but no electron-ion relaxation "
            "channel is active: without "
            "hybrid_pic_model.electron_ion_relaxation_rate(rho,Te,Ti,t) the "
            "redirected energy lands on the ions as pure undamped "
            "stochastic diffusion (no drag leg), which is anti-stabilizing "
            "(runaway ion-kick feedback). Set the relaxation rate, or set "
            "hybrid_pic_model.joule_redirect_allow_undamped = 1 to force the "
            "legacy undamped behavior (control arms only).");
    }
    if (m_te_shunt_eV > 0.0 && m_te_abort_threshold_eV > 0.0) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_te_abort_threshold_eV > m_te_shunt_eV,
            "hybrid_pic_model.Te_abort_threshold must sit above "
            "Te_shunt_threshold (the abort is the shunt's backstop)");
    }

    // The contamination tally taps the split-fluxform conduction sweeps
    // only; warn when it is armed on an untapped control form.
    if (m_contam_n_boundary > 0.0 && m_include_thermal_conduction &&
        (m_cond_form != 2 || m_cond_ff_unsplit)) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPICModel",
            "hybrid_pic_model.qdsmc_contamination_n_boundary is set, but the "
            "conduction form is not the split fluxform path (the scatter/"
            "layer/unsplit control arms are not tapped): the conduction "
            "contamination channels will read zero. The ion-kick channel "
            "still tallies.",
            ablastr::warn_manager::WarnPriority::medium);
    }

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
    // particle per grid node.
    if (m_solve_electron_energy_equation) {
        m_qdsmc_pc = std::make_unique<QdsmcParticleContainer>(&warpx);
        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            m_qdsmc_pc->InitParticles(lev);
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

    // Effective electron mass, resolved on first use: the lightest ion
    // species divided by the reduced mass ratio (physical m_e when the
    // ratio is unset). The ratio moves the effective electron skin depth
    // d_e = c/omega_pe(m_e_eff) relative to the grid.
    if (m_electron_inertia_mass == 0.0_rt) {
        if (m_reduced_electron_mass_ratio > 0.0_rt) {
            amrex::Real m_ion_min = std::numeric_limits<amrex::Real>::max();
            auto & mypc = warpx.GetPartContainer();
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
    amrex::Real const me_over_e = m_electron_inertia_mass / PhysConst::q_e;

    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp, lev);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rho.nComp() >= 2,
        "hybrid_pic_model.include_electron_inertia requires the "
        "theta-implicit hybrid evolve scheme (two charge-density time "
        "levels)");
    const int rho_mid_comp = rho.nComp() / 2;
    amrex::Real const rho_floor = PhysConst::q_e * m_n_floor;

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
        const int mid = rho_mid_comp;
        auto const dlo = amrex::lbound(dom_nd);
        auto const dhi = amrex::ubound(dom_nd);
        amrex::ParallelFor(mfi.tilebox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real const rho_n   = rhona(i, j, k, 0);
            amrex::Real const rho_mid = rhoa(i, j, k, mid);
            // True vacuum carries no electron fluid: zero the term there
            // (density-floored cells keep their inertia).
            if (rho_mid <= 0.0_rt) {
                ei(i,j,k,0) = 0.0_rt;
                ei(i,j,k,1) = 0.0_rt;
                ei(i,j,k,2) = 0.0_rt;
                return;
            }
            amrex::Real const rho_lim = amrex::max(rho_mid, rho_floor);
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
                amrex::Real const rl = amrex::max(
                    amrex::max(rhoa(ii,jj,kk,mid), 0.0_rt), rho_floor);
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
                ei(i,j,k,c) = me_over_e
                    * (djedt - u[c] * drhodt - adv[c]) / rho_lim;
            }
        });
    }
    Ei.setBndry(0.0_rt);
    Ei.FillBoundary(geom.periodicity());
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
    amrex::MultiFab const & Je_th =
        *warpx.m_fields.get("hybrid_Je_theta_nodal", lev);

    amrex::MultiFab::Copy(Je_nm1, Je_n, 0, 0, 3, Je_nm1.nGrowVect());
    if (m_inertia_history_levels < 2) { ++m_inertia_history_levels; }
    // Je^{n+1} = (Je^theta - (1-theta) Je^n)/theta -- the same
    // extrapolation family as the other end-of-step fields.
    amrex::MultiFab::LinComb(Je_n,
        1.0_rt / a_theta, Je_th, 0,
        1.0_rt - 1.0_rt / a_theta, Je_nm1, 0,
        0, 3, Je_n.nGrowVect());
    // Re-arm the step-start density capture for the next step's first
    // evaluation (see ComputeElectronInertiaNodal).
    m_inertia_rho_n_captured = false;
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

void HybridPICModel::QDSMCInitializeUe (int const lev, QdsmcUeMode const mode) const
{
    auto & warpx = WarpX::GetInstance();
    // Time levels per mode (see QdsmcUeMode): the advance call sites leave
    // hybrid_current_fp_temp = J_i^{n-1/2} (J_i^n for the pc corrector),
    // current_fp = J_i^{n+1/2}, hybrid_rho_fp_temp = rho^n and
    // rho_fp = rho^{n+1}; the plasma current is at its current (single)
    // state, so the Jp pair degenerates to a no-op interpolation.
    ablastr::fields::VectorField const J_plasma =
        warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    amrex::MultiFab const & rho_temp =
        *warpx.m_fields.get(FieldType::hybrid_rho_fp_temp, lev);
    amrex::MultiFab const & rho_new =
        *warpx.m_fields.get(FieldType::rho_fp, lev);
    ablastr::fields::VectorField const J_i_temp =
        warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_temp, lev);
    ablastr::fields::VectorField const J_i_new =
        warpx.m_fields.get_alldirs(FieldType::current_fp, lev);

    if (mode == QdsmcUeMode::JiNewRhoHalf) {
        // pc corrector: J_i^{n+1/2} with rho^{n+1/2} = avg(rho^n, rho^{n+1})
        QDSMCInitializeUe(lev, rho_temp, J_i_new, J_plasma, J_plasma, 1.0_rt,
                          &rho_new, nullptr);
    } else if (mode == QdsmcUeMode::JiAvg) {
        // leapfrog: J_i^n = avg of the two bracketing half-integer deposits
        QDSMCInitializeUe(lev, rho_temp, J_i_temp, J_plasma, J_plasma, 1.0_rt,
                          nullptr, &J_i_new);
    } else {
        // euler (#6982): J_i^{n-1/2} with rho^n
        QDSMCInitializeUe(lev, rho_temp, J_i_temp, J_plasma, J_plasma, 1.0_rt);
    }
}

void HybridPICModel::QDSMCInitializeUe (int const lev,
    amrex::MultiFab const & rho_in,
    ablastr::fields::VectorField const & J_i,
    ablastr::fields::VectorField const & Jp_new,
    ablastr::fields::VectorField const & Jp_old,
    amrex::Real const jp_interp,
    amrex::MultiFab const * rho_avg_with,
    ablastr::fields::VectorField const * J_i_avg_with) const
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
    bool const avg_rho = (rho_avg_with != nullptr);
    bool const avg_ji  = (J_i_avg_with != nullptr);

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
        amrex::Array4<amrex::Real const> const rho1_arr = avg_rho
            ? rho_avg_with->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
        amrex::Array4<amrex::Real const> const & Jpx     = Jp_new[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jpy     = Jp_new[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jpz     = Jp_new[2]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jpx0    = Jp_old[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jpy0    = Jp_old[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jpz0    = Jp_old[2]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jix     = J_i[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jiy     = J_i[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const & Jiz     = J_i[2]->const_array(mfi);
        amrex::Array4<amrex::Real const> const Jbx = avg_ji
            ? (*J_i_avg_with)[0]->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
        amrex::Array4<amrex::Real const> const Jby = avg_ji
            ? (*J_i_avg_with)[1]->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
        amrex::Array4<amrex::Real const> const Jbz = avg_ji
            ? (*J_i_avg_with)[2]->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
        amrex::Array4<amrex::Real>       const & Vex_arr = Vex.array(mfi);
        amrex::Array4<amrex::Real>       const & Vey_arr = Vey.array(mfi);
        amrex::Array4<amrex::Real>       const & Vez_arr = Vez.array(mfi);

        amrex::Box const & tbox = mfi.tilebox();

        amrex::Real const c_interp = jp_interp;

        amrex::ParallelFor(tbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real const rho_val = avg_rho
                ? 0.5_rt * (rho_arr(i,j,k) + rho1_arr(i,j,k))
                : rho_arr(i,j,k);
            if (rho_val <= rho_floor) { return; }

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
            auto jix = ablastr::coarsen::sample::Interp(Jix, Jx_stag, nodal, coarsen, i, j, k, 0);
            auto jiy = ablastr::coarsen::sample::Interp(Jiy, Jy_stag, nodal, coarsen, i, j, k, 0);
            auto jiz = ablastr::coarsen::sample::Interp(Jiz, Jz_stag, nodal, coarsen, i, j, k, 0);
            if (avg_ji) {
                jix = 0.5_rt * (jix + ablastr::coarsen::sample::Interp(Jbx, Jx_stag, nodal, coarsen, i, j, k, 0));
                jiy = 0.5_rt * (jiy + ablastr::coarsen::sample::Interp(Jby, Jy_stag, nodal, coarsen, i, j, k, 0));
                jiz = 0.5_rt * (jiz + ablastr::coarsen::sample::Interp(Jbz, Jz_stag, nodal, coarsen, i, j, k, 0));
            }

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
            // Floor the density instead of skipping low-density cells:
            // leaving K_e = 0 in the floored halo turns it into an absorbing
            // boundary that drains the plasma's electron thermal energy via
            // the remap diffusion (global exponential T_e collapse). With the
            // floor, the halo keeps whatever T_e it holds and is insulating.
            amrex::Real const ne =
                amrex::max(rho_arr(i,j,k), rho_floor) / PhysConst::q_e;
            Ke_arr(i,j,k) = Te_arr(i,j,k) * std::pow(ne, 1.0_rt - gamma) * kb_over_qe;
        });
    }

    Ke.FillBoundary(Ke.nGrowVect(), period);
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

    // Note: T_e is NOT zeroed here. Cells that received no QDSMC weight
    // keep their previous T_e -- zeroing them would erase valid state (and
    // seed a wrong K_e into neighbors on the next step) whenever a cell
    // momentarily receives no deposit.

    auto const gamma      = m_gamma;
    // Conversion floor: must match the floor QDSMCInitializeKe applies, so
    // the K -> T_e round-trip is exact for a marker that stayed home.
    auto const n_floor    = m_n_floor;
    // Deposited-weight guard: cells no QDSMC marker reached keep their T_e.
    auto const w_floor    = m_qdsmc_n_floor;
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
            if (weights_arr(i,j,k) <= w_floor) { return; }
            amrex::Real const w = weights_arr(i,j,k) * cell_volume;
            // Floored density, mirroring QDSMCInitializeKe: below-floor
            // cells are updated too (insulating halo), and the K <-> T_e
            // conversion uses the same n_e^(gamma-1) factor on both sides of
            // the step, so a cell whose marker did not move keeps its T_e
            // exactly.
            amrex::Real const ne =
                amrex::max(rho_arr(i,j,k) / PhysConst::q_e, n_floor);
            Te_arr(i,j,k) = Ke_arr(i,j,k)
                          / std::pow(ne, 1.0_rt - gamma)
                          / w
                          / kb_over_qe;
        });
    }

    Te.FillBoundary(Te.nGrowVect(), period);
}


void HybridPICModel::QDSMCFoldInsulatingDeposits (int const lev) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCFoldInsulatingDeposits()");

    auto & warpx = WarpX::GetInstance();
    amrex::Geometry const & geom = warpx.Geom(lev);
    amrex::Periodicity const & period = geom.periodicity();

    amrex::MultiFab       & K     = *warpx.m_fields.get(FieldType::hybrid_entropy_fp,              lev);
    amrex::MultiFab const & Te    = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho_n = *warpx.m_fields.get(FieldType::hybrid_rho_fp_temp,             lev);
    amrex::MultiFab const & phi   = *warpx.m_fields.get(FieldType::distance_to_eb,                 lev);

    // SPILL-ONLY fold (this branch): a below-floor node's own marker is
    // static (the QDSMCInitializeUe floor guard leaves V_e = 0) and
    // node-homed, so it deposits exactly its own content back on its home
    // node -- bit-for-bit, including the half-gradient correction, whose
    // (x_j - y_i) factor vanishes for an unmoved marker. Everything ABOVE
    // that own content is spill from moving open markers whose deposit
    // split across the floor boundary; without the fold it is dropped by
    // the QDSMCUpdateTe weights guard and never re-enters the open set.
    // The own content itself must NOT be folded: on this branch the #7128
    // insulating floor seeds K_e in the floored halo (QDSMCInitializeKe's
    // floored-density conversion regenerates it from the stale T_e every
    // step), so folding it would pump halo entropy into the open set at
    // every density ramp. Recomputing own = Ke(m) * N_own(m) with exactly
    // the QDSMCInitializeKe / SetK formulas keeps the subtraction exact
    // (spill = 0 to round-off at rest). Only the entropy K*N is folded;
    // the deposited weight N is a marker-density estimate the recovery
    // discards below the floor anyway (and the dev-side instrument
    // measured N-folding to dilute the edge temperature).
    //
    // Node classes: 0 = open (live plasma), 1 = foldable (below the
    // floor), 2 = masked with unknown own content (covered but above the
    // floor; only reachable through a pathological initial load -- neither
    // folded from nor counted as a destination). Built once with two ghost
    // nodes so the gather below never reads raw-field ghosts beyond their
    // filled range; domain-exterior ghosts stay at class 2.
    amrex::iMultiFab imask(K.boxArray(), K.DistributionMap(), 1, 2);
    imask.setVal(2);

    auto const rho_floor = PhysConst::q_e * m_n_floor;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(K, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<int>               const & m_arr   = imask.array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho_n.const_array(mfi);
        amrex::Array4<amrex::Real const> const & phi_arr = phi.const_array(mfi);

        amrex::Box const box = amrex::convert(mfi.tilebox(), K.ixType().toIntVect());

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            bool const below_floor = (rho_arr(i,j,k) <= rho_floor);
            bool const covered     = (phi_arr(i,j,k) <= 0.0_rt);
            m_arr(i,j,k) = below_floor ? 1 : (covered ? 2 : 0);
        });
    }
    imask.FillBoundary(period);

    // Pure gather formulation: every node computes its own folded value from
    // the (consistent) deposited data, so no atomics and no post-pass
    // SumBoundary of increments are needed; box-seam copies of a node
    // compute identical results from identical ghost data.
    amrex::MultiFab K_new(K.boxArray(), K.DistributionMap(), 1, 0);

    auto const gamma      = m_gamma;
    auto const kb_over_qe = PhysConst::kb / PhysConst::q_e;
    auto const dx_arr     = geom.CellSizeArray();
    amrex::Real cell_volume = 1.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { cell_volume *= dx_arr[d]; }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(K, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real>       const & Kn_arr  = K_new.array(mfi);
        amrex::Array4<amrex::Real const> const & K_arr   = K.const_array(mfi);
        amrex::Array4<amrex::Real const> const & Te_arr  = Te.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho_n.const_array(mfi);
        amrex::Array4<int const>         const & m_arr   = imask.const_array(mfi);

        amrex::Box const box = amrex::convert(mfi.tilebox(), K.ixType().toIntVect());

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Loop bounds of the 3^D-1 neighborhood (collapsed dims excluded)
#if defined(WARPX_DIM_3D)
            constexpr int jlo = -1; constexpr int jhi = 1;
            constexpr int klo = -1; constexpr int khi = 1;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            constexpr int jlo = -1; constexpr int jhi = 1;
            constexpr int klo =  0; constexpr int khi = 0;
#else
            constexpr int jlo =  0; constexpr int jhi = 0;
            constexpr int klo =  0; constexpr int khi = 0;
#endif

            // Spill above the own-marker content of a foldable node,
            // recomputed with the exact QDSMCInitializeKe (floored-density
            // K_e) and SetK (N = rho^n V / q_e) formulas so the
            // subtraction is bit-exact for a marker that stayed home.
            auto const spill = [=] (int ii, int jj, int kk) {
                amrex::Real const ne_fl =
                    amrex::max(rho_arr(ii,jj,kk), rho_floor) / PhysConst::q_e;
                amrex::Real const ke_own =
                    Te_arr(ii,jj,kk) * std::pow(ne_fl, 1.0_rt - gamma) * kb_over_qe;
                amrex::Real const n_own =
                    rho_arr(ii,jj,kk) * cell_volume / PhysConst::q_e;
                return amrex::max(K_arr(ii,jj,kk) - ke_own * n_own, 0.0_rt);
            };

            if (m_arr(i,j,k) == 1) {
                // Keep the own-marker (halo) content; only the spill moves
                // to the open neighbors below. (The recovery's weights
                // guard skips this node and the next QDSMCInitializeKe
                // rewrites it, so this is bookkeeping hygiene, not state.)
                Kn_arr(i,j,k) = K_arr(i,j,k) - spill(i,j,k);
                return;
            }
            if (m_arr(i,j,k) == 2) {
                Kn_arr(i,j,k) = K_arr(i,j,k);
                return;
            }

            // Open node: reclaim the spill of every foldable neighbor,
            // split equally over that neighbor's open nodes.
            amrex::Real k_gain = 0.0_rt;
            for         (int kk = k+klo; kk <= k+khi; ++kk) {
                for     (int jj = j+jlo; jj <= j+jhi; ++jj) {
                    for (int ii = i-1;   ii <= i+1;   ++ii) {
                        if (ii == i && jj == j && kk == k) { continue; }
                        if (m_arr(ii,jj,kk) != 1) { continue; }

                        int n_open = 0;
                        for         (int k2 = kk+klo; k2 <= kk+khi; ++k2) {
                            for     (int j2 = jj+jlo; j2 <= jj+jhi; ++j2) {
                                for (int i2 = ii-1;   i2 <= ii+1;   ++i2) {
                                    if (i2 == ii && j2 == jj && k2 == kk) { continue; }
                                    if (m_arr(i2,j2,k2) == 0) { ++n_open; }
                                }
                            }
                        }
                        // n_open >= 1: this node is one of them
                        k_gain += spill(ii,jj,kk) / n_open;
                    }
                }
            }
            Kn_arr(i,j,k) = K_arr(i,j,k) + k_gain;
        });
    }

    amrex::MultiFab::Copy(K, K_new, 0, 0, 1, 0);
    K.FillBoundary(K.nGrowVect(), period);
}


void HybridPICModel::QDSMCApplyInsulatingEBFill (int const lev) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCApplyInsulatingEBFill()");

    auto & warpx = WarpX::GetInstance();
    amrex::Geometry const & geom = warpx.Geom(lev);
    amrex::Periodicity const & period = geom.periodicity();

    amrex::MultiFab       & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp,                         lev);
    amrex::MultiFab const & phi = *warpx.m_fields.get(FieldType::distance_to_eb,                 lev);

    amrex::Real dx_max = geom.CellSize(0);
    for (int d = 1; d < AMREX_SPACEDIM; ++d) {
        dx_max = amrex::max(dx_max, geom.CellSize(d));
    }
    // Fill reach: the standoff band plus one ring of ramp spillover. Below-
    // floor nodes farther from the wall keep the QDSMCUpdateTe stale-T_e
    // semantics unchanged.
    amrex::Real const band_bound = (WarpX::eb_standoff_cells + 1.0_rt) * dx_max;
    int const n_sweeps =
        static_cast<int>(std::ceil(WarpX::eb_standoff_cells)) + 3;

    auto const n_floor = m_qdsmc_n_floor;

    // status = 1: valid fill source (live plasma, or filled in an earlier
    // sweep); 0: to fill. Ghosts outside the domain stay 0 and never donate.
    amrex::iMultiFab status(amrex::convert(Te.boxArray(), Te.ixType().toIntVect()),
                            Te.DistributionMap(), 1, 1);
    status.setVal(0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<int>               const & st_arr  = status.array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);
        amrex::Array4<amrex::Real const> const & phi_arr = phi.const_array(mfi);

        amrex::Box const box = amrex::convert(mfi.tilebox(), Te.ixType().toIntVect());

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            bool const boundary_node =
                (rho_arr(i,j,k) / PhysConst::q_e <= n_floor) ||
                (phi_arr(i,j,k) <= 0.0_rt);
            bool const to_fill = boundary_node && (phi_arr(i,j,k) <= band_bound);
            st_arr(i,j,k) = to_fill ? 0 : 1;
        });
    }
    status.FillBoundary(period);

    amrex::MultiFab  Te_new(Te.boxArray(), Te.DistributionMap(), 1, 0);
    amrex::iMultiFab st_new(status.boxArray(), status.DistributionMap(), 1, 0);

    for (int sweep = 0; sweep < n_sweeps; ++sweep)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Array4<amrex::Real>       const & Tn_arr = Te_new.array(mfi);
            amrex::Array4<int>               const & sn_arr = st_new.array(mfi);
            amrex::Array4<amrex::Real const> const & T_arr  = Te.const_array(mfi);
            amrex::Array4<int const>         const & s_arr  = status.const_array(mfi);

            amrex::Box const box = amrex::convert(mfi.tilebox(), Te.ixType().toIntVect());

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (s_arr(i,j,k) == 1) {
                    Tn_arr(i,j,k) = T_arr(i,j,k);
                    sn_arr(i,j,k) = 1;
                    return;
                }
#if defined(WARPX_DIM_3D)
                constexpr int jlo = -1; constexpr int jhi = 1;
                constexpr int klo = -1; constexpr int khi = 1;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                constexpr int jlo = -1; constexpr int jhi = 1;
                constexpr int klo =  0; constexpr int khi = 0;
#else
                constexpr int jlo =  0; constexpr int jhi = 0;
                constexpr int klo =  0; constexpr int khi = 0;
#endif
                amrex::Real t_sum = 0.0_rt;
                int n_src = 0;
                for         (int kk = k+klo; kk <= k+khi; ++kk) {
                    for     (int jj = j+jlo; jj <= j+jhi; ++jj) {
                        for (int ii = i-1;   ii <= i+1;   ++ii) {
                            if (ii == i && jj == j && kk == k) { continue; }
                            if (s_arr(ii,jj,kk) == 1) {
                                t_sum += T_arr(ii,jj,kk);
                                ++n_src;
                            }
                        }
                    }
                }
                if (n_src > 0) {
                    Tn_arr(i,j,k) = t_sum / n_src;
                    sn_arr(i,j,k) = 1;
                } else {
                    Tn_arr(i,j,k) = T_arr(i,j,k);
                    sn_arr(i,j,k) = 0;
                }
            });
        }

        amrex::MultiFab::Copy(Te, Te_new, 0, 0, 1, 0);
        amrex::iMultiFab::Copy(status, st_new, 0, 0, 1, 0);
        Te.FillBoundary(Te.nGrowVect(), period);
        status.FillBoundary(period);
    }
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
    // Physical-eta heating: when joule_heating_resistivity(rho,J,Te,t) is
    // given, the heating source uses it instead of the E-solve eta (whose
    // numerical vacuum-regularizer ramp was the liftoff runaway's ignition
    // source). Different arity (Te in eV as an argument, so Spitzer eta(Te)
    // is expressible), hence the in-kernel branch rather than a ternary.
    bool const has_heat_eta  = m_has_heating_resistivity;
    auto const eta_heat      = m_eta_heating;
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

    // Decline gates (both default inactive): the decoupled heating gate
    // joule_heating_n_min (heating permitted only above it, independent of
    // the solver floor) and the redirect density gate
    // joule_redirect_n_min_factor * n_floor (redirected energy only staged
    // above it -- below, per-macro-ion kicks in nearly empty cells are the
    // measured runaway feedback vector).
    amrex::Real const rho_heat_gate =
        PhysConst::q_e * std::max(m_joule_heating_n_min, m_n_floor);
    bool const heat_gate_armed  = (rho_heat_gate > rho_floor);
    amrex::Real const rho_redir_gate =
        PhysConst::q_e * (m_joule_redirect_n_min_factor * m_n_floor);
    bool const redir_gate_armed =
        do_redirect && (m_joule_redirect_n_min_factor > 0._rt);

    // Per-cell declined source-energy densities [J/m^3] for the audit tally
    // (component 0 = heating gate, 1 = redirect gate), accumulated over
    // species. Allocated only when a decline channel is armed. Nodal like
    // Te: box-seam nodes carry redundant consistent copies, deduplicated by
    // sum_unique below.
    bool const any_drop_tally = heat_gate_armed || redir_gate_armed;
    amrex::MultiFab dropped_mf;
    if (any_drop_tally) {
        dropped_mf.define(Te.boxArray(), Te.DistributionMap(), 2, 0);
        dropped_mf.setVal(0.0_rt);
    }

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

            // Dropped-energy tally (default Array4 when no decline channel is
            // armed -> never indexed, the gate conditions cannot fire).
            amrex::Array4<amrex::Real> dropped_arr;
            if (any_drop_tally) { dropped_arr = dropped_mf.array(mfi); }

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

                // eta: the Ohm's-law E-solve parser by default (per-cell
                // heat reduces to eta J^2 exactly in single species), or the
                // separate physical heating resistivity when specified. The
                // per-species overlay below adds on top of either choice.
                amrex::Real eta_s_eff = has_heat_eta
                    ? eta_heat(rho_val, Jmag, Te_arr(i,j,k) / K_per_eV, t_new)
                    : eta(rho_val, Jmag, t_new);

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
                // Source energy density this species would deposit [J/m^3]
                // (= n_e kB dTe_s/(gamma-1); the redirect's eventual ion
                // delivery (3/2) n_s E_s is the same quantity). Used only by
                // the decline-tally branches below.
                amrex::Real const du_s = dt * Z_s
                               * PhysConst::q_e * PhysConst::q_e
                               * eta_s_eff * ns * ne * dv2;
                // Decoupled heating gate: between the solver floor and
                // joule_heating_n_min no heat is created at all; the declined
                // source goes to the audit tally (only reachable when armed).
                if (rho_val <= rho_heat_gate) {
                    dropped_arr(i,j,k,0) += du_s;
                    return;
                }
                // Te-threshold redirection: below threshold heat electrons (the
                // usual Joule deposit); at/above it write this species'
                // m_i-independent redirected energy E_s = (2/3) n_e Z_s e^2 eta
                // |dV|^2 dt [J] into its component for the ion-heating step --
                // unless the cell is below the redirect density gate, in which
                // case the source is dropped to the tally instead.
                if (do_redirect && Te_arr(i,j,k) >= Te_thresh_K) {
                    if (redir_gate_armed && rho_val <= rho_redir_gate) {
                        dropped_arr(i,j,k,1) += du_s;
                    } else {
                        redirect_arr(i,j,k,ion_comp) = (2.0_rt/3.0_rt) * ne
                            * Z_s * PhysConst::q_e * PhysConst::q_e * eta_s_eff * dv2 * dt;
                    }
                } else {
                    Te_arr(i,j,k) += dTe_s;
                }
            });
        }
    }

    // Fold the declined per-cell energy densities into the cumulative audit
    // tallies [J] (unique-node sum, periodic images deduplicated; the
    // nodal control volume dV is uniform up to boundary halves, which the
    // gates never straddle in practice).
    if (any_drop_tally) {
        auto const dx = warpx.Geom(lev).CellSizeArray();
        amrex::Real dV = 1.0_rt;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { dV *= dx[d]; }
        m_joule_dropped_heat_gate_J     += dropped_mf.sum_unique(0, false, period) * dV;
        m_joule_dropped_redirect_gate_J += dropped_mf.sum_unique(1, false, period) * dV;
    }

    Te.FillBoundary(Te.nGrowVect(), period);
}


void HybridPICModel::QDSMCShuntTeExcess (int const lev,
                                         amrex::MultiFab * const redirect_E) const
{
    ABLASTR_PROFILE("HybridPICModel::QDSMCShuntTeExcess()");

    using warpx::fields::FieldType;

    // General Te limiter with ion shunt (see the member doc): cap open-set
    // Te at the threshold and stage the excess as per-species per-ion
    // energy E_s = Z_s kB (Te - Tcap) into redirect_E (+=: the Joule
    // redirect may already have staged). The density-fraction split
    // Sum_s 3/2 n_s E_s = 3/2 n_e kB (Te - Tcap) delivers the capped
    // excess exactly. Below the redirect density gate the excess is
    // dropped to the tally instead (quarantine: per-macro-ion kicks in
    // nearly empty cells are the measured feedback vector); sub-floor
    // cells are never touched.
    auto & warpx = WarpX::GetInstance();
    amrex::Periodicity const & period = warpx.Geom(lev).periodicity();

    amrex::MultiFab       & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp, lev);

    auto const rho_floor  = PhysConst::q_e * m_n_floor;
    auto const K_per_eV   = PhysConst::q_e / PhysConst::kb;
    amrex::Real const cap_K = m_te_shunt_eV * K_per_eV;

    amrex::Real const rho_redir_gate =
        PhysConst::q_e * (m_joule_redirect_n_min_factor * m_n_floor);
    bool const redir_gate_armed = (m_joule_redirect_n_min_factor > 0._rt);

    // Z per charged-species redirect component (same ordering as
    // QDSMCAddJouleHeating / QDSMCApplyIonHeating).
    amrex::GpuArray<amrex::Real, 8> Zs{};
    int ncomp_ion = 0;
    auto & mypc = warpx.GetPartContainer();
    for (auto const & nm : mypc.GetSpeciesNames()) {
        auto & pc = mypc.GetParticleContainerFromName(nm);
        if (pc.getCharge() == 0._prt) { continue; }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ncomp_ion < 8,
            "Te_shunt_threshold supports at most 8 charged species");
        Zs[ncomp_ion++] = pc.getCharge() / PhysConst::q_e;
    }
    int const ncomp = ncomp_ion;

    // Staged (channel 0) and gate-dropped (channel 1) energy densities for
    // the audit tallies [J/m^3]; nodal like Te, deduplicated by sum_unique.
    amrex::MultiFab tally_mf(Te.boxArray(), Te.DistributionMap(), 2, 0);
    tally_mf.setVal(0.0_rt);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real>       const & Te_arr  = Te.array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);
        amrex::Array4<amrex::Real>       const & red_arr = redirect_E->array(mfi);
        amrex::Array4<amrex::Real>       const & tly_arr = tally_mf.array(mfi);

        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real const rho_val = rho_arr(i,j,k);
            if (rho_val <= rho_floor) { return; }         // quarantined
            amrex::Real const Te_K = Te_arr(i,j,k);
            if (Te_K <= cap_K) { return; }
            amrex::Real const ne  = rho_val / PhysConst::q_e;
            amrex::Real const dT  = Te_K - cap_K;
            amrex::Real const du  = 1.5_rt * ne * PhysConst::kb * dT;
            if (redir_gate_armed && rho_val <= rho_redir_gate) {
                tly_arr(i,j,k,1) += du;                    // dropped
            } else {
                for (int c = 0; c < ncomp; ++c) {
                    red_arr(i,j,k,c) += Zs[c] * PhysConst::kb * dT;
                }
                tly_arr(i,j,k,0) += du;                    // staged
            }
            Te_arr(i,j,k) = cap_K;
        });
    }

    Te.FillBoundary(Te.nGrowVect(), period);

    auto const dxc = warpx.Geom(lev).CellSizeArray();
    amrex::Real dV = 1.0_rt;
    for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) { dV *= dxc[dd]; }
    m_te_shunt_J += tally_mf.sum_unique(0, false, period) * dV;
    m_joule_dropped_redirect_gate_J +=
        tally_mf.sum_unique(1, false, period) * dV;
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

    // Per-particle kick cap (redirect hardening): clamp the redirect's OU
    // variance so sigma_redir <= f v_th, i.e. E_s <= f^2 kB T (the ion mass
    // cancels). Armed only when the redirect is active and
    // joule_redirect_kick_cap_vth_frac > 0. The clipped remainder is dropped
    // (not carried over) and accumulated in the audit tally: re-depositing it
    // later in a still-hot cell would only defer the same unbounded kicks.
    bool const kick_cap_armed =
        do_redir && (m_joule_redirect_kick_cap_vth_frac > 0._rt);
    amrex::Real const kick_cap_f2 =
        m_joule_redirect_kick_cap_vth_frac * m_joule_redirect_kick_cap_vth_frac;
    amrex::Real const Te_floor_K = Te_floor_eV * K_per_eV;
    amrex::MultiFab dropped_cc;   // clipped energy density [J/m^3], cc grid
    if (kick_cap_armed) {
        dropped_cc.define(cc_ba, Te.DistributionMap(), 1, 0);
        dropped_cc.setVal(0.0_rt);
    }

    // Contamination-tally kicks channel: redirected energy staged for OU
    // kicks inside QUAR cells (n <= qdsmc_contamination_n_boundary) -- the
    // grid-side proxy for ion-carried energy injected into quarantined
    // cells (QUARANTINE_INSTRUMENT_PLAN.md).
    bool const contam_kicks_on =
        do_redir && (m_contam_n_boundary > 0._rt);
    amrex::Real const rho_contam_bnd = PhysConst::q_e * m_contam_n_boundary;
    amrex::MultiFab contam_cc;    // staged kick energy density [J/m^3]
    if (contam_kicks_on) {
        contam_cc.define(cc_ba, Te.DistributionMap(), 1, 0);
        contam_cc.setVal(0.0_rt);
    }

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

        // Per-species charge densities, needed only for the kick-cap and
        // contamination tallies' n_s = f_s n_e / Z_s.
        amrex::MultiFab const * rho_s_mf    = nullptr;
        amrex::MultiFab const * rhos_sum_mf = nullptr;
        if (kick_cap_armed || contam_kicks_on) {
            rho_s_mf    = warpx.m_fields.get("rho_fp_" + spec_name, lev);
            rhos_sum_mf = warpx.m_fields.get("hybrid_rho_species_sum_fp", lev);
        }
        amrex::Real const Z_s = pc.getCharge() / PhysConst::q_e;

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

            // Kick-cap / contamination inputs and outputs (default Array4
            // when the feature is off -> never indexed, the armed flags
            // gate every access).
            amrex::Array4<amrex::Real>       dropped_arr;
            amrex::Array4<amrex::Real>       contam_arr;
            amrex::Array4<amrex::Real const> rhos_arr;
            amrex::Array4<amrex::Real const> rhosum_arr;
            if (kick_cap_armed) { dropped_arr = dropped_cc.array(mfi); }
            if (contam_kicks_on) { contam_arr = contam_cc.array(mfi); }
            if (kick_cap_armed || contam_kicks_on) {
                rhos_arr    = rho_s_mf->const_array(mfi);
                rhosum_arr  = rhos_sum_mf->const_array(mfi);
            }

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
                    amrex::Real E_s = ablastr::coarsen::sample::Interp(
                        redirect_arr, nodal_src, cc_dst, coarsen, i, j, k, ion_comp);
                    // Species density n_s = f_s n_e / Z_s, shared by the
                    // kick-cap drop tally and the contamination channel
                    // (rhos arrays are staged whenever either is armed).
                    auto n_s_at = [&] () {
                        amrex::Real const rhos_v = ablastr::coarsen::sample::Interp(
                            rhos_arr, nodal_src, cc_dst, coarsen, i, j, k, 0);
                        amrex::Real const rhosum_v = amrex::max(
                            ablastr::coarsen::sample::Interp(
                                rhosum_arr, nodal_src, cc_dst, coarsen, i, j, k, 0),
                            rho_floor);
                        return (rhos_v / rhosum_v) * (rho_val / PhysConst::q_e) / Z_s;
                    };
                    if (kick_cap_armed && E_s > 0._rt) {
                        // sigma_redir <= f v_th  <=>  E_s <= f^2 kB T, with
                        // T the deposited T_i when relaxation is on (else the
                        // local T_e sets the velocity scale). In evaporated
                        // cells T_i ~ 0 -> the cap declines essentially the
                        // whole kick, which is the quarantine intent.
                        amrex::Real const T_cap_K = do_relax
                            ? amrex::max(Ti_arr(i,j,k) * K_per_eV, Te_floor_K)
                            : amrex::max(Te_K, Te_floor_K);
                        amrex::Real const E_s_max = kick_cap_f2 * PhysConst::kb * T_cap_K;
                        if (E_s > E_s_max) {
                            // Clipped remainder -> dropped tally: the ions
                            // will not receive (3/2) n_s (E_s - E_s_max).
                            dropped_arr(i,j,k) += 1.5_rt * n_s_at() * (E_s - E_s_max);
                            E_s = E_s_max;
                        }
                    }
                    // Contamination kicks channel: energy the OU operator
                    // will deliver to ions inside QUAR cells (post-cap).
                    if (contam_kicks_on && E_s > 0._rt &&
                        rho_val <= rho_contam_bnd) {
                        contam_arr(i,j,k) += 1.5_rt * n_s_at() * E_s;
                    }
                    coef_arr(i,j,k,5) = E_s;
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

    // Fold the clipped kick energy and the QUAR-staged kick energy into
    // their cumulative audit tallies [J] (cell-centered boxes are
    // disjoint, a plain sum suffices).
    if (kick_cap_armed || contam_kicks_on) {
        auto const dx = warpx.Geom(lev).CellSizeArray();
        amrex::Real dV = 1.0_rt;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { dV *= dx[d]; }
        if (kick_cap_armed) {
            m_joule_dropped_kick_cap_J += dropped_cc.sum(0, false) * dV;
        }
        if (contam_kicks_on) {
            m_contam_kicks_J += contam_cc.sum(0, false) * dV;
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
    ablastr::utils::communication::FillBoundary(
        Pe, WarpX::do_single_precision_comms,
        warpx.Geom(lev).periodicity(), true);
}


namespace
{
    /** SMART-limited (Gaskell--Lau 1988) face value for the advective
     *  cross-derivative fluxes of the FD conduction operator, in the
     *  normalized-variable frame of the local upwind triple (tUU = far
     *  upwind, tU = upwind, tD = downwind): QUICK where smooth, blended
     *  to bounded legs near extrema, 1st-order upwind outside [0, 1] --
     *  monotone by construction. limiter = 0 returns unlimited QUICK
     *  (attribution control, can overshoot), 1 returns 1st-order upwind. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real qdsmc_fd_face_value (amrex::Real const tUU,
                                     amrex::Real const tU,
                                     amrex::Real const tD, int const limiter)
    {
        if (limiter == 1) { return tU; }
        if (limiter == 0) {
            return 0.375_rt*tD + 0.75_rt*tU - 0.125_rt*tUU;   // QUICK
        }
        amrex::Real const den = tD - tUU;
        if (den == 0.0_rt) { return tU; }
        amrex::Real const ph = (tU - tUU) / den;
        amrex::Real phf;
        if (ph <= 0.0_rt || ph >= 1.0_rt) { phf = ph; }
        else if (ph < 1.0_rt/6.0_rt)      { phf = 3.0_rt*ph; }
        else if (ph <= 5.0_rt/6.0_rt)     { phf = 0.375_rt + 0.75_rt*ph; }
        else                              { phf = 1.0_rt; }
        return tUU + phf*den;
    }
}

void HybridPICModel::QdsmcTransportOnceGrid (int const lev,
                                             amrex::Real const dt_adv) const
{
    ABLASTR_PROFILE("HybridPICModel::QdsmcTransportOnceGrid()");

#ifdef WARPX_DIM_RZ
    WARPX_ABORT_WITH_MESSAGE(
        "qdsmc_transport_operator = grid: RZ metric not implemented");
#endif
    auto & warpx = WarpX::GetInstance();
    using ablastr::fields::Direction;

    amrex::Geometry const & geom = warpx.Geom(lev);
    amrex::Periodicity const & period = geom.periodicity();
    auto const dxi_arr = geom.InvCellSizeArray();

    // EB + floor: covered (distance_to_eb <= 0) and sub-floor nodes close
    // their faces and freeze -- the flux-form analog of the marker path's
    // EB reflection + floored-node hold, matching the FD conduction
    // operator's mask semantics.
    bool const has_eb = EB::enabled();
    amrex::MultiFab const * eb_dist = has_eb
        ? warpx.m_fields.get(FieldType::distance_to_eb, lev) : nullptr;
    amrex::Real const n_floor_t = m_qdsmc_n_floor;

    // Same entry state as the marker path: K at nodes from T_e and rho^n.
    QDSMCInitializeKe(lev);

    amrex::MultiFab const & Vex = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{0}, lev);
    amrex::MultiFab const & Vey = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{1}, lev);
    amrex::MultiFab const & Vez = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{2}, lev);
    amrex::MultiFab       & Ke  = *warpx.m_fields.get(FieldType::hybrid_entropy_fp, lev);
    amrex::MultiFab       & wts = *warpx.m_fields.get(FieldType::hybrid_qdsmc_weights_fp, lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::hybrid_rho_fp_temp, lev);

    amrex::Real const qe = PhysConst::q_e;
    int const fd_limiter = m_cond_fd_limiter;
    amrex::Real const fd_cfl = m_cond_fd_cfl;
    bool const use_rkf45 = (m_cond_fd_time == 1);

#if defined(WARPX_DIM_3D)
    amrex::GpuArray<int, AMREX_SPACEDIM> const gd2ax = {0, 1, 2};
#elif (AMREX_SPACEDIM == 2)
    amrex::GpuArray<int, AMREX_SPACEDIM> const gd2ax = {0, 2};
#else
    amrex::GpuArray<int, AMREX_SPACEDIM> const gd2ax = {2};
#endif
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxi_g;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { dxi_g[d] = dxi_arr[d]; }

    // Nodal velocity components per grid dim, gathered once (V_e frozen
    // over the transport slot, like the marker push's SetV).
    amrex::MultiFab vel(Ke.boxArray(), Ke.DistributionMap(),
                        AMREX_SPACEDIM, 2);
    amrex::MultiFab const * vsrc[3] = {&Vex, &Vey, &Vez};
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(vel, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Box const box = mfi.tilebox();
        amrex::Array4<amrex::Real> const & v_arr = vel.array(mfi);
        for (int g = 0; g < AMREX_SPACEDIM; ++g) {
            amrex::Array4<amrex::Real const> const & s_arr =
                vsrc[gd2ax[g]]->const_array(mfi);
            amrex::ParallelFor(box,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                v_arr(i,j,k,g) = s_arr(i,j,k);
            });
        }
    }
    vel.FillBoundary(period);

    amrex::MultiFab opn(Ke.boxArray(), Ke.DistributionMap(), 1, 2);
    opn.setVal(1.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(opn, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Box const box = mfi.tilebox();
        amrex::Array4<amrex::Real>       const & o_arr   = opn.array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);
        amrex::Array4<amrex::Real const> const & phi_arr = has_eb
            ? eb_dist->const_array(mfi) : amrex::Array4<amrex::Real const>{};
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            bool const covered = has_eb && (phi_arr(i,j,k) <= 0.0_rt);
            o_arr(i,j,k) = (!covered &&
                            rho_arr(i,j,k)/PhysConst::q_e > n_floor_t)
                           ? 1.0_rt : 0.0_rt;
        });
    }
    opn.FillBoundary(period);

    // State y = {n_e, K n_e}: conservative pair under the common flow.
    int constexpr ngT = 2;
    amrex::MultiFab y(Ke.boxArray(), Ke.DistributionMap(), 2, ngT);
    y.setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(y, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Box const box = mfi.tilebox();
        amrex::Array4<amrex::Real>       const & y_arr = y.array(mfi);
        amrex::Array4<amrex::Real const> const & K_arr = Ke.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real const ne = amrex::max(rho_arr(i,j,k)/qe, 0.0_rt);
            y_arr(i,j,k,0) = ne;
            y_arr(i,j,k,1) = K_arr(i,j,k)*ne;
        });
    }

    // RHS: K = -div(y v) per component, SMART-limited upwind face values
    // on the face-averaged velocity. Same inline face-window discipline
    // as the FD conduction operator (identical fluxes on both sides of
    // every face -> Sigma(y) telescopes to round-off).
    auto eval_rhs = [&] (amrex::MultiFab & Yin, amrex::MultiFab & Kout)
    {
        ablastr::utils::communication::FillBoundary(
            Yin, WarpX::do_single_precision_comms, period, true);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Kout, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box const box = mfi.tilebox();
            amrex::Array4<amrex::Real>       const & ko = Kout.array(mfi);
            amrex::Array4<amrex::Real const> const & yc = Yin.const_array(mfi);
            amrex::Array4<amrex::Real const> const & v_arr = vel.const_array(mfi);
            amrex::Array4<amrex::Real const> const & op = opn.const_array(mfi);

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // flux of comp c through the face m0 -> m0 + e_g
                auto face_flux = [&] (int const m0[3], int const g,
                                      int const c)
                {
                    int m1[3] = {m0[0], m0[1], m0[2]};
                    m1[g] += 1;
                    if (op(m0[0],m0[1],m0[2]) <= 0.5_rt ||
                        op(m1[0],m1[1],m1[2]) <= 0.5_rt) {
                        return 0.0_rt;   // closed floor/EB faces
                    }
                    amrex::Real const vf = 0.5_rt*(
                        v_arr(m0[0],m0[1],m0[2],g) +
                        v_arr(m1[0],m1[1],m1[2],g));
                    if (vf == 0.0_rt) { return 0.0_rt; }
                    int uu[3];
                    amrex::Real tU, tD;
                    if (vf > 0.0_rt) {
                        tU = yc(m0[0],m0[1],m0[2],c);
                        tD = yc(m1[0],m1[1],m1[2],c);
                        uu[0] = m0[0]; uu[1] = m0[1]; uu[2] = m0[2];
                        uu[g] -= 1;
                    } else {
                        tU = yc(m1[0],m1[1],m1[2],c);
                        tD = yc(m0[0],m0[1],m0[2],c);
                        uu[0] = m1[0]; uu[1] = m1[1]; uu[2] = m1[2];
                        uu[g] += 1;
                    }
                    amrex::Real const tUU = yc(uu[0],uu[1],uu[2],c);
                    return vf*qdsmc_fd_face_value(tUU, tU, tD, fd_limiter);
                };
                if (op(i,j,k) <= 0.5_rt) {
                    ko(i,j,k,0) = 0.0_rt;   // frozen: closed nodes hold state
                    ko(i,j,k,1) = 0.0_rt;
                    return;
                }
                int const node[3] = {i, j, k};
                for (int c = 0; c < 2; ++c) {
                    amrex::Real div = 0.0_rt;
                    for (int g = 0; g < AMREX_SPACEDIM; ++g) {
                        int lo[3] = {i, j, k};
                        lo[g] -= 1;
                        div += (face_flux(node, g, c) - face_flux(lo, g, c))
                               * dxi_g[g];
                    }
                    ko(i,j,k,c) = -div;
                }
            });
        }
    };

    // Advective stability ceiling: dt <= cfl / max_node sum_g |v| dxi
    // (upwind CFL; velocity is frozen, so this is computed once).
    amrex::Real s_adv = 0.0_rt;
    {
        amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (MFIter mfi(vel, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box const box = mfi.tilebox();
            amrex::Array4<amrex::Real const> const & v_arr = vel.const_array(mfi);
            reduce_op.eval(box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                amrex::Real s = 0.0_rt;
                for (int g = 0; g < AMREX_SPACEDIM; ++g) {
                    s += std::abs(v_arr(i,j,k,g))*dxi_g[g];
                }
                return {s};
            });
        }
        auto tup = reduce_data.value(reduce_op);
        s_adv = amrex::get<0>(tup);
        amrex::ParallelDescriptor::ReduceRealMax(s_adv);
    }
    auto cap = [&] () -> amrex::Real
    {
        return (s_adv > 0.0_rt) ? fd_cfl/s_adv
                                : std::numeric_limits<amrex::Real>::max();
    };

    QdsmcRKIntegrator const integ(
        use_rkf45 ? QdsmcRKIntegrator::Scheme::RKF45
                  : QdsmcRKIntegrator::Scheme::SSPRK2,
        eval_rhs, cap, m_cond_fd_rtol, m_cond_fd_atol,
        m_substep_safety, m_substep_max_growth, m_cond_fd_max_subcycles);
    QdsmcRKStats const st = integ.Advance(y, dt_adv);

    if (st.t_done < dt_adv*(1.0_rt - 1.0e-12_rt)) {
        amrex::Warning(
            "[qdsmc] QdsmcTransportOnceGrid: attempts budget hit; dropped "
            + std::to_string(1.0 - st.t_done/dt_adv)
            + " of the transport step");
    }

    auto const dx_cv = geom.CellSizeArray();
    amrex::Real cell_volume = 1.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { cell_volume *= dx_cv[d]; }

    // Marker-deposit convention (QDSMCUpdateTe recovers K_e =
    // entropy_fp / (weights_fp * V_cell)): weights carries the density
    // n_e and entropy carries K times the electron COUNT, so the
    // transported K n_e picks up one cell volume. Getting this wrong by
    // 1/V_cell inflates T_e -> grad Pe -> E -> multi-cell ion crossings
    // -> the documented Esirkepov segfault (measured).
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Ke, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Box const box = mfi.tilebox();
        amrex::Array4<amrex::Real>       const & K_arr = Ke.array(mfi);
        amrex::Array4<amrex::Real>       const & w_arr = wts.array(mfi);
        amrex::Array4<amrex::Real const> const & y_arr = y.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            w_arr(i,j,k) = y_arr(i,j,k,0);
            K_arr(i,j,k) = y_arr(i,j,k,1)*cell_volume;
        });
    }

    QDSMCUpdateTe(lev);
}

void HybridPICModel::QdsmcTransportOnce (int const lev, amrex::Real const dt_adv,
                                         bool const midpoint) const
{
    if (m_qdsmc_transport_operator == 1) {
        amrex::ignore_unused(midpoint);   // adaptive RK supersedes it
        QdsmcTransportOnceGrid(lev, dt_adv);
        return;
    }

    ABLASTR_PROFILE("HybridPICModel::QdsmcTransportOnce()");

    auto & warpx = WarpX::GetInstance();
    using ablastr::fields::Direction;

    // Grid-side K_e initialization from the current T_e and rho^n. V_e must
    // already be filled (QDSMCInitializeUe) by the scheme driver.
    QDSMCInitializeKe(lev);

    amrex::MultiFab const & Vex = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{0}, lev);
    amrex::MultiFab const & Vey = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{1}, lev);
    amrex::MultiFab const & Vez = *warpx.m_fields.get(FieldType::hybrid_electron_velocity_fp, Direction{2}, lev);
    amrex::MultiFab const & Ke  = *warpx.m_fields.get(FieldType::hybrid_entropy_fp,           lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::hybrid_rho_fp_temp,          lev);
    amrex::MultiFab       & Karr_out    = *warpx.m_fields.get(FieldType::hybrid_entropy_fp,        lev);
    amrex::MultiFab       & weights_out = *warpx.m_fields.get(FieldType::hybrid_qdsmc_weights_fp, lev);

    // Load each QDSMC particle with V_e and (K_e * N_e, N_e) from its home
    // node.
    m_qdsmc_pc->SetV(lev, Vex, Vey, Vez);
    m_qdsmc_pc->SetK(lev, Ke, rho);

    // EB marker handling (adiabatic E7-replacement, matching the
    // conduction sweeps' fold-back): covered-home markers freeze,
    // markers pushed into the conductor mirror back across the level
    // set, and the midpoint velocity sample is mirror-guarded too.
    amrex::MultiFab const * eb_dist =
        (EB::enabled() && m_qdsmc_eb_marker_reflect)
            ? warpx.m_fields.get(FieldType::distance_to_eb, lev)
            : nullptr;

    // Two-stage midpoint (RK2) evaluation: replace the home-gathered
    // velocity by V_e sampled at the trajectory midpoint
    // x_mid = home + (dt/2) v(home). With a time-centered V_e this makes
    // the advection globally second order in dt; conservation is untouched
    // (the deposit still moves the full carried content).
    if (midpoint) {
        m_qdsmc_pc->GatherVAtMidpoint(lev, dt_adv, eb_dist, Vex, Vey, Vez);
    }

    // Push by dt_adv; redistribute so particles end up in their new tile.
    m_qdsmc_pc->PushX(lev, dt_adv, 1.0_rt, eb_dist);

    // Scatter the carried entropy and weight onto the grid (each call
    // zeroes its target field, then deposits, then SumBoundary), with the
    // half-gradient correction when enabled. The cliff-limited option
    // rescales each destination node's ENTROPY share toward the isothermal
    // spill across unresolved density jumps (the K-diffusion heat-pump fix;
    // the weight deposit is untouched -- N is a marker-count estimate).
    QdsmcParticleContainer::CliffDeposit cliff{};
    if (m_cliff_limited_deposit) {
        cliff.enabled = true;
        cliff.rho_new = warpx.m_fields.get(FieldType::rho_fp, lev);
        cliff.n_floor = m_n_floor;
        cliff.gamma   = m_gamma;
        cliff.r1      = m_cliff_deposit_r1;
        cliff.r2      = m_cliff_deposit_r2;
    }
    m_qdsmc_pc->DepositK(lev, Karr_out, m_qdsmc_gradient_deposit, cliff);
    m_qdsmc_pc->DepositField(lev, weights_out, m_qdsmc_gradient_deposit);

    // Insulating EB wall (PR #7138, branch-adapted): SPILL-ONLY fold --
    // the entropy that moving open markers spilled onto below-floor nodes
    // is reclaimed for the open set before the recovery's weights guard
    // drops it, while the halo's own (per-step-regenerated) content stays
    // put. See the QDSMCFoldInsulatingDeposits body: the dev-form full
    // fold of the standalone PR would pump halo entropy here.
    bool const insulating_eb = EB::enabled() &&
        (WarpX::eb_boundary_type == EmbeddedBoundaryType::Insulating);
    if (insulating_eb) {
        QDSMCFoldInsulatingDeposits(lev);
    }

    // Recover the new T_e from (deposited K*N) / (deposited N) and the
    // updated n_e (from rho_fp = rho^{n+1}).
    QDSMCUpdateTe(lev);

    // Insulating EB wall (PR #7138): zero-normal-gradient T_e into the
    // standoff band and the covered region -- the C++ form of the validated
    // annulus band-copy callback (the conduction sweeps and recovery skip
    // masked nodes, so the fill persists through the Strang bracket to the
    // Pe emission).
    if (insulating_eb) {
        QDSMCApplyInsulatingEBFill(lev);
    }
}


void HybridPICModel::ApplyQdsmcEnergySources (int const lev, amrex::Real const dt_src,
                                              bool const fill_te_ghosts) const
{
    ABLASTR_PROFILE("HybridPICModel::ApplyQdsmcEnergySources()");

    auto & warpx = WarpX::GetInstance();

    // Step 6: Joule-heating source on T_e (Phys. Plasmas 31, 012902 (2024), Eq. 12), per-cell from
    // rho_fp(_s), the plasma current, and the Ohm's-law eta parser. With the
    // Te-threshold redirect on, the above-threshold heat is staged in
    // ion_redirect_E (per-charged-species energy, J) for the ion-heating step.
    bool redirect_active = m_include_joule_heating && m_joule_redirect_to_ions;
    bool shunt_active = (m_te_shunt_eV > 0._rt);
    int n_ion_species = 0;
    if (redirect_active || shunt_active) {
        auto & mpc = warpx.GetPartContainer();
        for (auto const & nm : mpc.GetSpeciesNames()) {
            if (mpc.GetParticleContainerFromName(nm).getCharge() != 0._prt) { ++n_ion_species; }
        }
        if (n_ion_species == 0) { redirect_active = false; shunt_active = false; }
    }
    amrex::MultiFab ion_redirect_E;
    if (redirect_active || shunt_active) {
        amrex::MultiFab const & Te_mf =
            *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
        ion_redirect_E.define(Te_mf.boxArray(), Te_mf.DistributionMap(), n_ion_species, 0);
        ion_redirect_E.setVal(0.0_rt);
    }
    if (m_include_joule_heating) {
        QDSMCAddJouleHeating(lev, dt_src, redirect_active ? &ion_redirect_E : nullptr);
    }
    // General Te limiter with ion shunt (any-channel excess -> ions; runs
    // after the Joule source so the staging merges into one OU kick
    // application).
    if (shunt_active) {
        QDSMCShuntTeExcess(lev, &ion_redirect_E);
    }

    // Steps 6b/6c both need each charged species' T_i when Q_ei relaxation is
    // on. Deposit it ONCE here (the expensive per-particle NGP temperature
    // reduction) and share it: the electron sink (6b) and the ion-heating
    // operator (6c) run back-to-back with no intervening ion motion, so the
    // deposited T_i is identical for both.
    std::map<std::string, amrex::MultiFab*> Ti_dep_by_species;
    // Owns the per-species cell-centered scalar T_i built from the shape-aware
    // deposition below; must outlive the QDSMCAddTemperatureRelaxation /
    // QDSMCApplyIonHeating calls that read it through Ti_dep_by_species.
    std::map<std::string, std::unique_ptr<amrex::MultiFab>> Ti_scalar_owned;
    if (m_include_temperature_relaxation) {
        using ablastr::fields::Direction;
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
                "The Q_ei temperature relaxation requires do_temperature_deposition "
                "on every charged ion species; it is enabled automatically at species "
                "construction, so hitting this indicates the species was created "
                "before the hybrid_pic_model relaxation parameters were readable.");

            // Shape-aware ion temperature (particle-shape order, consistent with
            // charge/current) in the Yee-staggered 3-component T_<nm> vector field
            // (Tr,Tt,Tz), deposited in Kelvin by HybridPICDepositRhoAndJ ->
            // mypc->DepositTemperatures earlier this step and read here. Fill guard
            // cells so the cell-centered interpolation below reads finite
            // neighbours at box/domain edges.
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
            Ti_dep_by_species[nm] = Ti_s.get();
            Ti_scalar_owned[nm] = std::move(Ti_s);
        }
    }

    // Step 6b: electron-ion thermal-equilibration (Q_ei) sink on T_e
    // (cools T_e toward each ion species' T_i).
    if (m_include_temperature_relaxation) {
        QDSMCAddTemperatureRelaxation(lev, dt_src, Ti_dep_by_species);
    }

    // Step 6c: stochastic drag-diffusion ion-heating operator -- delivers the Q_ei
    // conjugate (when relaxation is on) and/or the redirected Joule energy
    // (when the redirect is on), so the ions are heated by one mechanism.
    if (m_include_temperature_relaxation || redirect_active || shunt_active) {
        QDSMCApplyIonHeating(lev, dt_src,
                             (redirect_active || shunt_active) ? &ion_redirect_E : nullptr,
                             m_include_temperature_relaxation ? &Ti_dep_by_species : nullptr);
    }

    // The source kernels write valid cells only; the Strang pre-half runs
    // right before the K_e initialization, which reads T_e ghosts, so
    // refresh them here (the euler control path skips this to stay
    // bit-identical to #6982).
    if (fill_te_ghosts &&
        (m_include_joule_heating || m_include_temperature_relaxation)) {
        amrex::MultiFab & Te =
            *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
        Te.FillBoundary(warpx.Geom(lev).periodicity());
    }

    // Dropped-energy tally print (rank 0), for the deck-side energy audit:
    // fires only when a decline channel is armed and the cadence input is on.
    bool const any_decline_armed =
        (m_joule_heating_n_min > m_n_floor) ||
        (m_te_shunt_eV > 0._rt) ||
        (m_cond_eb_bc == 1) ||
        ((m_joule_redirect_to_ions || m_te_shunt_eV > 0._rt) &&
         (m_joule_redirect_n_min_factor > 0._rt ||
          m_joule_redirect_kick_cap_vth_frac > 0._rt));
    if (any_decline_armed && m_joule_dropped_print_interval > 0 &&
        warpx.getistep(0) % m_joule_dropped_print_interval == 0) {
        amrex::Print() << "[qdsmc] step " << warpx.getistep(0)
            << " joule_dropped_J: heat_gate=" << m_joule_dropped_heat_gate_J
            << " redirect_gate=" << m_joule_dropped_redirect_gate_J
            << " kick_cap=" << m_joule_dropped_kick_cap_J
            << " te_shunt=" << m_te_shunt_J
            << " wall_bath=" << m_cond_eb_tally
            << " (cumulative; wall_bath > 0 = into plasma)\n";
    }

    // Contamination tally print (same cadence; the conduction channels are
    // accumulated in QdsmcConductionOnce, the kicks channel above -- this
    // site runs every step whether or not conduction is enabled).
    if (m_contam_n_boundary > 0._rt && m_joule_dropped_print_interval > 0 &&
        warpx.getistep(0) % m_joule_dropped_print_interval == 0) {
        amrex::Print() << "[qdsmc] step " << warpx.getistep(0)
            << " contamination_J: ax0=" << m_contam_axis_J[0]
            << " ax1=" << m_contam_axis_J[1]
            << " ax2=" << m_contam_axis_J[2]
            << " fast_front=" << m_contam_fast_front_J
            << " kicks=" << m_contam_kicks_J
            << " (cumulative)\n";
    }

    // Graceful Te-runaway abort (>700 keV liftoff runaways otherwise ended
    // in SEGV deep in the transport): check the post-source OPEN-SET
    // (n > n_floor) Te maximum against the ceiling and stop with a clear
    // message instead. The open-set gate is load-bearing: quarantined
    // sub-floor (marooned) cells legitimately carry keV after an S0-style
    // self-quench -- evaporate, strand, decouple -- and a full-field max
    // would abort exactly the runs the quarantine design counts as
    // successes (measured: arm SH tripped at 1.75 keV open-set because a
    // quarantined cell crossed the ceiling).
    if (m_te_abort_threshold_eV > 0._rt) {
        amrex::MultiFab const & Te =
            *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
        amrex::MultiFab const & rho_ab =
            *warpx.m_fields.get(FieldType::rho_fp, lev);
        amrex::Real const rho_floor_ab = PhysConst::q_e * m_n_floor;
        amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Array4<amrex::Real const> const & te_arr  = Te.const_array(mfi);
            amrex::Array4<amrex::Real const> const & rho_arr = rho_ab.const_array(mfi);
            reduce_op.eval(mfi.tilebox(), reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                return {(rho_arr(i,j,k) > rho_floor_ab)
                            ? te_arr(i,j,k) : 0.0_rt};
            });
        }
        amrex::Real te_max_K = amrex::get<0>(reduce_data.value(reduce_op));
        amrex::ParallelAllReduce::Max(
            te_max_K, amrex::ParallelDescriptor::Communicator());
        amrex::Real const K_per_eV = PhysConst::q_e / PhysConst::kb;
        amrex::Real const te_max_eV = te_max_K / K_per_eV;
        if (te_max_eV > m_te_abort_threshold_eV) {
            WARPX_ABORT_WITH_MESSAGE(
                "QDSMC electron energy equation: open-set (n > n_floor) "
                "max(Te) = " + std::to_string(te_max_eV)
                + " eV exceeds hybrid_pic_model.Te_abort_threshold = "
                + std::to_string(m_te_abort_threshold_eV)
                + " eV at step " + std::to_string(warpx.getistep(0))
                + " (t = " + std::to_string(warpx.gett_new(0))
                + " s): electron-temperature runaway.");
        }
    }
}


namespace
{
    /** 1D interpolation kernel for the layer (gather) conduction form on a
     *  4-point stencil {fm, f0, f1, f2} at fractional t in [0, 1) between
     *  f0 and f1. kind: 0 = linear (fm/f2 ignored; the exact adjoint of
     *  the uncorrected hat deposit), 1 = monotone cubic Hermite with
     *  MC-limited node slopes (values stay in [min(f0,f1), max(f0,f1)]:
     *  positive and overshoot-free; the MC limiter keeps the slopes inside
     *  the Fritsch--Carlson monotonicity box), 2 = Keys cubic convolution
     *  (a = -1/2; higher order, NOT monotone). All kinds are interpolating
     *  (exact at t = 0), so a zero-displacement foot returns the node
     *  value bit-for-bit (machine-exact at-rest identity). */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real qdsmc_interp1d (int const kind,
                                amrex::Real const fm, amrex::Real const f0,
                                amrex::Real const f1, amrex::Real const f2,
                                amrex::Real const t)
    {
        if (kind == 0) { return f0 + (f1 - f0)*t; }
        amrex::Real const t2 = t*t;
        amrex::Real const t3 = t2*t;
        if (kind == 2) {
            return 0.5_rt*( fm*(-t3 + 2.0_rt*t2 - t)
                          + f0*( 3.0_rt*t3 - 5.0_rt*t2 + 2.0_rt)
                          + f1*(-3.0_rt*t3 + 4.0_rt*t2 + t)
                          + f2*( t3 - t2) );
        }
        amrex::Real const d0 = f0 - fm;
        amrex::Real const d1 = f1 - f0;
        amrex::Real const d2 = f2 - f1;
        amrex::Real m0 = 0.0_rt, m1 = 0.0_rt;
        if (d0*d1 > 0.0_rt) {
            amrex::Real const s = (d1 > 0.0_rt) ? 1.0_rt : -1.0_rt;
            m0 = s*amrex::min(2.0_rt*std::abs(d0), 2.0_rt*std::abs(d1),
                              0.5_rt*std::abs(d0 + d1));
        }
        if (d1*d2 > 0.0_rt) {
            amrex::Real const s = (d1 > 0.0_rt) ? 1.0_rt : -1.0_rt;
            m1 = s*amrex::min(2.0_rt*std::abs(d1), 2.0_rt*std::abs(d2),
                              0.5_rt*std::abs(d1 + d2));
        }
        return f0*( 2.0_rt*t3 - 3.0_rt*t2 + 1.0_rt) + m0*(t3 - 2.0_rt*t2 + t)
             + f1*(-2.0_rt*t3 + 3.0_rt*t2)          + m1*(t3 - t2);
    }

    /** 1D Keys cubic-convolution (a = -1/2) weights on the stencil
     *  offsets {-1, 0, 1, 2} for fractional t in [0, 1). Sum(w) = 1
     *  identically (exact partition of unity: a conservative scatter),
     *  and linears AND quadratics are reproduced, so the deposit's first
     *  and second moments vanish. w[0] and w[3] are negative lobes. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void qdsmc_keys_w (amrex::Real const t, amrex::Real w[4])
    {
        amrex::Real const t2 = t*t;
        amrex::Real const t3 = t2*t;
        w[0] = 0.5_rt*(-t3 + 2.0_rt*t2 - t);
        w[1] = 0.5_rt*( 3.0_rt*t3 - 5.0_rt*t2 + 2.0_rt);
        w[2] = 0.5_rt*(-3.0_rt*t3 + 4.0_rt*t2 + t);
        w[3] = 0.5_rt*( t3 - t2);
    }

    /** Tensor-product interpolation of a nodal Array4 at the fractional
     *  index position (i0 + fr0, j0 + fr1, k0 + fr2), nesting
     *  qdsmc_interp1d per grid dim (the monotone kind is nonlinear in the
     *  data, so per-dim nesting -- not a weight product -- is required).
     *  Stencil indices are clamped into [lo, hi] on non-periodic dims
     *  (degenerate end stencils at walls, E7-spirit placeholder until the
     *  Thrust-D reflection BCs); on periodic dims they read the filled
     *  ghosts. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real qdsmc_interp_nd (int const kind,
                                 amrex::Array4<amrex::Real const> const& f,
                                 int const i0, int const j0, int const k0,
                                 amrex::Real const fr0, amrex::Real const fr1,
                                 amrex::Real const fr2,
                                 amrex::GpuArray<int, 3> const& is_per,
                                 amrex::GpuArray<int, 3> const& lo,
                                 amrex::GpuArray<int, 3> const& hi)
    {
        auto clampi = [&] (int const d, int idx) {
            if (!is_per[d]) {
                idx = amrex::min(amrex::max(idx, lo[d]), hi[d]);
            }
            return idx;
        };
        amrex::ignore_unused(j0, k0, fr1, fr2, clampi);
#if defined(WARPX_DIM_3D)
        amrex::Real h[4];
        for (int kk = -1; kk <= 2; ++kk) {
            int const kc = clampi(2, k0 + kk);
            amrex::Real g[4];
            for (int jj = -1; jj <= 2; ++jj) {
                int const jc = clampi(1, j0 + jj);
                amrex::Real const v = qdsmc_interp1d(kind,
                    f(clampi(0, i0 - 1), jc, kc), f(clampi(0, i0), jc, kc),
                    f(clampi(0, i0 + 1), jc, kc), f(clampi(0, i0 + 2), jc, kc),
                    fr0);
                g[jj + 1] = v;
            }
            h[kk + 1] = qdsmc_interp1d(kind, g[0], g[1], g[2], g[3], fr1);
        }
        return qdsmc_interp1d(kind, h[0], h[1], h[2], h[3], fr2);
#elif (AMREX_SPACEDIM == 2)
        amrex::Real g[4];
        for (int jj = -1; jj <= 2; ++jj) {
            int const jc = clampi(1, j0 + jj);
            g[jj + 1] = qdsmc_interp1d(kind,
                f(clampi(0, i0 - 1), jc, k0), f(clampi(0, i0), jc, k0),
                f(clampi(0, i0 + 1), jc, k0), f(clampi(0, i0 + 2), jc, k0),
                fr0);
        }
        return qdsmc_interp1d(kind, g[0], g[1], g[2], g[3], fr1);
#else
        return qdsmc_interp1d(kind,
            f(clampi(0, i0 - 1), j0, k0), f(clampi(0, i0), j0, k0),
            f(clampi(0, i0 + 1), j0, k0), f(clampi(0, i0 + 2), j0, k0),
            fr0);
#endif
    }

    /** Probabilists' Gauss--Hermite abscissae/weights for the unit-variance
     *  Gaussian (x = sqrt(2) * physicists' roots, w = physicists' weights /
     *  sqrt(pi); sum w = 1, sum w x^2 = 1), npts in [2, 7]. All weights are
     *  positive, so positivity of the deposited energy is automatic at
     *  every npts. npts = 2 is variance-exact (weak order 1 per step from
     *  the 4th-moment defect); npts = 3 matches through the 5th moment
     *  (weak order 2); npts >= 4 buys Gaussian tails out to larger
     *  multiples of sigma. */
    void qdsmc_gh_table (int const npts,
                         amrex::GpuArray<amrex::Real, 8> & x,
                         amrex::GpuArray<amrex::Real, 8> & w)
    {
        for (int q = 0; q < 8; ++q) { x[q] = 0.0_rt; w[q] = 0.0_rt; }
        switch (npts) {
        case 2:
            x[0] = -1.0_rt; x[1] = 1.0_rt;
            w[0] = 0.5_rt;  w[1] = 0.5_rt;
            break;
        case 3:
            x[0] = -1.7320508075688772_rt; x[1] = 0.0_rt;
            x[2] =  1.7320508075688772_rt;
            w[0] = 1.0_rt/6.0_rt; w[1] = 2.0_rt/3.0_rt; w[2] = 1.0_rt/6.0_rt;
            break;
        case 4:
            x[0] = -2.3344142183389773_rt; x[1] = -0.7419637843027258_rt;
            x[2] =  0.7419637843027258_rt; x[3] =  2.3344142183389773_rt;
            w[0] = 0.045875854768068503_rt; w[1] = 0.45412414523193150_rt;
            w[2] = w[1]; w[3] = w[0];
            break;
        case 5:
            x[0] = -2.8569700138728056_rt; x[1] = -1.3556261799742659_rt;
            x[2] =  0.0_rt;
            x[3] =  1.3556261799742659_rt; x[4] =  2.8569700138728056_rt;
            w[0] = 0.011257411327720693_rt; w[1] = 0.22207592200561263_rt;
            w[2] = 8.0_rt/15.0_rt; w[3] = w[1]; w[4] = w[0];
            break;
        case 6:
            x[0] = -3.3242574335521193_rt; x[1] = -1.8891758777537107_rt;
            x[2] = -0.6167065901925941_rt; x[3] =  0.6167065901925941_rt;
            x[4] =  1.8891758777537107_rt; x[5] =  3.3242574335521193_rt;
            w[0] = 0.0025557844020562464_rt; w[1] = 0.088615746041914523_rt;
            w[2] = 0.40882846955602925_rt;
            w[3] = w[2]; w[4] = w[1]; w[5] = w[0];
            break;
        case 7:
            x[0] = -3.7504397177257425_rt; x[1] = -2.3667594107345415_rt;
            x[2] = -1.1544053947399682_rt; x[3] =  0.0_rt;
            x[4] =  1.1544053947399682_rt; x[5] =  2.3667594107345415_rt;
            x[6] =  3.7504397177257425_rt;
            w[0] = 0.00054826885597221730_rt; w[1] = 0.030757124976191933_rt;
            w[2] = 0.24012317860501250_rt;    w[3] = 16.0_rt/35.0_rt;
            w[4] = w[2]; w[5] = w[1]; w[6] = w[0];
            break;
        default:
            WARPX_ABORT_WITH_MESSAGE(
                "qdsmc_gh_table: quadrature point count must be in [2, 7]");
        }
    }

    /** Field-aligned perpendicular frame (e1, e2) for a unit b, matching
     *  the scatter/layer construction: e1 is the perpendicular direction
     *  nearest yhat (falls back to xhat when b ~ yhat), e2 = b x e1. The
     *  same deterministic formula everywhere keeps the per-branch
     *  quadrature displacement field single-valued across kernels. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void qdsmc_perp_frame (amrex::Real const bx, amrex::Real const by,
                           amrex::Real const bz,
                           amrex::Real & e1x, amrex::Real & e1y,
                           amrex::Real & e1z,
                           amrex::Real & e2x, amrex::Real & e2y,
                           amrex::Real & e2z)
    {
        e1x = -by*bx; e1y = 1.0_rt - by*by; e1z = -by*bz;
        amrex::Real e1n = e1x*e1x + e1y*e1y + e1z*e1z;
        if (e1n < 1.0e-12_rt) {   // b ~ yhat: build from xhat instead
            e1x = 1.0_rt - bx*bx; e1y = -bx*by; e1z = -bx*bz;
            e1n = e1x*e1x + e1y*e1y + e1z*e1z;
        }
        amrex::Real const e1i = 1.0_rt/std::sqrt(e1n);
        e1x *= e1i; e1y *= e1i; e1z *= e1i;
        e2x = by*e1z - bz*e1y;
        e2y = bz*e1x - bx*e1z;
        e2z = bx*e1y - by*e1x;
    }

}


void HybridPICModel::ApplyQdsmcConductionWallBCs (
    int const lev, amrex::Real const dt_c,
    amrex::MultiFab & Te, amrex::MultiFab const & rho) const
{
    bool any = false;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        any = any || (m_cond_bc[d][0] != 0) || (m_cond_bc[d][1] != 0);
    }
    if (!any) { return; }

    auto & warpx = WarpX::GetInstance();
    amrex::Geometry const & geom = warpx.Geom(lev);
    amrex::Box const dom_nodes = amrex::surroundingNodes(geom.Domain());
    auto const dx_arr = geom.CellSizeArray();
    amrex::Real const kb = PhysConst::kb;
    amrex::Real const qe = PhysConst::q_e;

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    for (int s = 0; s < 2; ++s) {
        int const bc = m_cond_bc[d][s];
        if (bc == 0) { continue; }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!geom.isPeriodic(d),
            "qdsmc_conduction_bc_lo/_hi: non-adiabatic conduction BCs "
            "require a non-periodic domain dimension");
        int const wall = (s == 0) ? dom_nodes.smallEnd(d)
                                  : dom_nodes.bigEnd(d);
        amrex::Real const Te_wall_K = m_cond_bc_Te[d][s] * qe / kb;
        // prescribed flux: u injected per substep into the boundary
        // row's dual cells, q A dt / V = q dt / dx  [J/m^3]
        amrex::Real const du_flux = m_cond_bc_q[d][s] * dt_c / dx_arr[d];

        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box tile_box = mfi.tilebox();
            {   // unique node ownership (fixup-loop seam trim)
                amrex::Box const box_nodes =
                    amrex::surroundingNodes(mfi.validbox());
                for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                    if (tile_box.bigEnd(dd) == box_nodes.bigEnd(dd) &&
                        (box_nodes.bigEnd(dd) != dom_nodes.bigEnd(dd) ||
                         geom.isPeriodic(dd))) {
                        tile_box.growHi(dd, -1);
                    }
                }
            }
            if (wall < tile_box.smallEnd(d) ||
                wall > tile_box.bigEnd(d)) { continue; }
            amrex::Box wall_plane = tile_box;
            wall_plane.setSmall(d, wall);
            wall_plane.setBig(d, wall);

            amrex::Array4<amrex::Real>       const & Te_arr  = Te.array(mfi);
            amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);

            reduce_op.eval(wall_plane, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                amrex::Real const ne = rho_arr(i,j,k) / qe;
                if (ne <= 0.0_rt) { return {0.0_rt}; }
                if (bc == 1) {
                    // thermal bath: pin the row, tally the exchange
                    amrex::Real const du =
                        1.5_rt * kb * ne * (Te_wall_K - Te_arr(i,j,k));
                    Te_arr(i,j,k) = Te_wall_K;
                    return {du};
                }
                // prescribed flux: inject, clamp cooling at Te = 0,
                // tally what was actually applied
                amrex::Real const dTe =
                    amrex::max(du_flux / (1.5_rt * kb * ne),
                               -Te_arr(i,j,k));
                Te_arr(i,j,k) += dTe;
                return {1.5_rt * kb * ne * dTe};
            });
        }
        auto tup = reduce_data.value(reduce_op);
        amrex::Real tally = amrex::get<0>(tup);
        amrex::ParallelDescriptor::ReduceRealSum(tally);
        m_cond_wall_tally[d][s] += tally;
    }}
}

void HybridPICModel::QdsmcConductionOnceFD (int const lev, amrex::Real const dt_c,
                                            bool const use_rho_new) const
{
    ABLASTR_PROFILE("HybridPICModel::QdsmcConductionOnceFD()");

#ifdef WARPX_DIM_RZ
    WARPX_ABORT_WITH_MESSAGE(
        "qdsmc_conduction_operator = fd: RZ requires the curvilinear "
        "J Xi^ij metric form and is not implemented");
#endif
    auto & warpx = WarpX::GetInstance();
    using ablastr::fields::Direction;

    amrex::Geometry const & geom = warpx.Geom(lev);
    amrex::Periodicity const & period = geom.periodicity();
    auto const dxi_arr = geom.InvCellSizeArray();

    // EB: covered nodes (distance_to_eb <= 0) close their faces -- the
    // flux-form staircase-adiabatic wall -- and keep their T_e frozen;
    // the isothermal option pins the wall-adjacent fluid ring after the
    // integrate (same ring-2 semantics and tally as the SDE path).
    bool const has_eb = EB::enabled();
    amrex::MultiFab const * eb_dist = has_eb
        ? warpx.m_fields.get(FieldType::distance_to_eb, lev) : nullptr;
    bool const eb_iso = has_eb && (m_cond_eb_bc == 1);

    amrex::MultiFab & Te =
        *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    // Same time-level pairing as the SDE path: rho^n for the pre-transport
    // Strang half, rho^{n+1} for the post-transport half.
    amrex::MultiFab const & rho = use_rho_new
        ? *warpx.m_fields.get(FieldType::rho_fp, lev)
        : *warpx.m_fields.get(FieldType::hybrid_rho_fp_temp, lev);
    amrex::MultiFab const & Bx = *warpx.m_fields.get(FieldType::Bfield_fp, Direction{0}, lev);
    amrex::MultiFab const & By = *warpx.m_fields.get(FieldType::Bfield_fp, Direction{1}, lev);
    amrex::MultiFab const & Bz = *warpx.m_fields.get(FieldType::Bfield_fp, Direction{2}, lev);

    amrex::Real const t_now = warpx.gett_new(lev);
    amrex::Real const kb = PhysConst::kb;
    amrex::Real const qe = PhysConst::q_e;
    amrex::Real const me = PhysConst::m_e;
    auto const kappa_par_ex  = m_kappa_par;
    auto const kappa_perp_ex = m_kappa_perp;
    amrex::Real const n_floor = m_qdsmc_n_floor;
    amrex::Real const f_lim   = m_cond_flux_limit_factor;
    bool const iso_full       = m_cond_isotropic;
    amrex::Real const iso_B   = m_cond_iso_B;
    bool const iso_any        = iso_full || (iso_B > 0.0_rt);
    bool const fd4            = (m_cond_fd_order == 4);
    int const fd_limiter      = m_cond_fd_limiter;
    amrex::Real const fd_cfl  = m_cond_fd_cfl;
    int const max_sub         = m_cond_fd_max_subcycles;
    bool const use_rkf45      = (m_cond_fd_time == 1);

    // Grid-dim bookkeeping (same conventions as the SDE kernels).
#if defined(WARPX_DIM_3D)
    amrex::GpuArray<int, AMREX_SPACEDIM> const gd2ax = {0, 1, 2};
#elif (AMREX_SPACEDIM == 2)
    amrex::GpuArray<int, AMREX_SPACEDIM> const gd2ax = {0, 2};
#else
    amrex::GpuArray<int, AMREX_SPACEDIM> const gd2ax = {2};
#endif
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxi_g;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { dxi_g[d] = dxi_arr[d]; }
    amrex::Box const dom_nodes = amrex::surroundingNodes(geom.Domain());
    amrex::GpuArray<int, AMREX_SPACEDIM> is_per, dom_lo, dom_hi;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        is_per[d] = geom.isPeriodic(d) ? 1 : 0;
        dom_lo[d] = dom_nodes.smallEnd(d);
        dom_hi[d] = dom_nodes.bigEnd(d);
    }

    amrex::GpuArray<int, 3> const Bx_stag = Bx_IndexType;
    amrex::GpuArray<int, 3> const By_stag = By_IndexType;
    amrex::GpuArray<int, 3> const Bz_stag = Bz_IndexType;
    amrex::GpuArray<int, 3> nd_x = {1, 1, 1};
    amrex::GpuArray<int, 3> nd_y = {1, 1, 1};
    amrex::GpuArray<int, 3> nd_z = {1, 1, 1};
    for (int d = AMREX_SPACEDIM; d < 3; ++d) {
        nd_x[d] = Bx_stag[d]; nd_y[d] = By_stag[d]; nd_z[d] = Bz_stag[d];
    }
    amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};

    // --- Pass A (once per call): unit b, |B|^2, floored n_e, open mask,
    // EB coverage --- b and n_e do not change across subcycles; only the
    // chi tensor does. Ghosts = 3 for the isothermal ring scan (eb_ring
    // <= 3); mask comps default OPEN so unset non-periodic domain ghosts
    // never read as walls (the SDE kin convention).
    enum BNE : int { b_bx = 0, b_by, b_bz, b_B2, b_ne, b_open, b_ebm,
                     b_ncomp };
    amrex::MultiFab bne(Te.boxArray(), Te.DistributionMap(), BNE::b_ncomp, 3);
    bne.setVal(0.0_rt);
    bne.setVal(1.0_rt, BNE::b_open, 1, bne.nGrow());
    bne.setVal(1.0_rt, BNE::b_ebm, 1, bne.nGrow());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(bne, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Box const box = mfi.tilebox();
        amrex::Array4<amrex::Real>       const & b_arr   = bne.array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);
        amrex::Array4<amrex::Real const> const & Bx_arr  = Bx.const_array(mfi);
        amrex::Array4<amrex::Real const> const & By_arr  = By.const_array(mfi);
        amrex::Array4<amrex::Real const> const & Bz_arr  = Bz.const_array(mfi);
        amrex::Array4<amrex::Real const> const & phi_arr = has_eb
            ? eb_dist->const_array(mfi) : amrex::Array4<amrex::Real const>{};

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            bool const covered = has_eb && (phi_arr(i,j,k) <= 0.0_rt);
            amrex::Real const ne_raw = rho_arr(i,j,k) / qe;
            amrex::Real const bxv = ablastr::coarsen::sample::Interp(
                Bx_arr, Bx_stag, nd_x, coarsen, i, j, k, 0);
            amrex::Real const byv = ablastr::coarsen::sample::Interp(
                By_arr, By_stag, nd_y, coarsen, i, j, k, 0);
            amrex::Real const bzv = ablastr::coarsen::sample::Interp(
                Bz_arr, Bz_stag, nd_z, coarsen, i, j, k, 0);
            amrex::Real const B2 = bxv*bxv + byv*byv + bzv*bzv;
            bool const unmag = (B2 <= 0.0_rt);
            amrex::Real const Binv = unmag ? 0.0_rt : 1.0_rt/std::sqrt(B2);
            b_arr(i,j,k,BNE::b_bx)   = unmag ? 0.0_rt : bxv*Binv;
            b_arr(i,j,k,BNE::b_by)   = unmag ? 0.0_rt : byv*Binv;
            b_arr(i,j,k,BNE::b_bz)   = unmag ? 1.0_rt : bzv*Binv;
            b_arr(i,j,k,BNE::b_B2)   = B2;
            b_arr(i,j,k,BNE::b_ne)   = amrex::max(ne_raw, n_floor);
            b_arr(i,j,k,BNE::b_open) =
                (ne_raw > n_floor && !covered) ? 1.0_rt : 0.0_rt;
            b_arr(i,j,k,BNE::b_ebm)  = covered ? 0.0_rt : 1.0_rt;
        });
    }
    bne.FillBoundary(period);

    // --- scratch: grid-projected chi tensor + T ping-pong ---------------
    // Symmetric tensor comps, sym(g,h) = g*SPACEDIM - g(g-1)/2 + (h-g) for
    // g <= h (2D: xx, xz, zz). ngT = 3 covers the widest face stencil (4th
    // order window +-3) plus the tensor build on grown(2) boxes reading
    // the flux-limiter gradient at +-1.
    int constexpr NXI = AMREX_SPACEDIM*(AMREX_SPACEDIM + 1)/2;
    int constexpr ngT = 3;
    amrex::MultiFab xi(Te.boxArray(), Te.DistributionMap(), NXI, 2);
    amrex::MultiFab T_cur(Te.boxArray(), Te.DistributionMap(), 1, ngT);
    T_cur.setVal(0.0_rt);
    amrex::MultiFab::Copy(T_cur, Te, 0, 0, 1, 0);

    // 4th-order coefficient tables (Chacon et al. Appendix A): face
    // interpolation C0 over the middle-4 nodes of the 6-point window, and
    // the A-matrix derivative rows at window positions 1..4.
    amrex::GpuArray<amrex::Real, 4> const c0 =
        {-1.0_rt/12.0_rt, 7.0_rt/12.0_rt, 7.0_rt/12.0_rt, -1.0_rt/12.0_rt};
    amrex::GpuArray<amrex::Real, 24> A6{};   // rows 1..4 of A_6x6, flattened
    {
        amrex::Real const a[4][6] = {
            {-12.0_rt, -65.0_rt, 120.0_rt,  -60.0_rt,  20.0_rt,  -3.0_rt},
            {  3.0_rt, -30.0_rt, -20.0_rt,   60.0_rt, -15.0_rt,   2.0_rt},
            { -2.0_rt,  15.0_rt, -60.0_rt,   20.0_rt,  30.0_rt,  -3.0_rt},
            {  3.0_rt, -20.0_rt,  60.0_rt, -120.0_rt,  65.0_rt,  12.0_rt}};
        for (int r = 0; r < 4; ++r) {
            for (int m = 0; m < 6; ++m) { A6[6*r + m] = a[r][m]/60.0_rt; }
        }
    }
    // D0 5-point 4th-order central first derivative (transverse cross term)
    amrex::GpuArray<amrex::Real, 5> const d0 =
        {1.0_rt/12.0_rt, -8.0_rt/12.0_rt, 0.0_rt, 8.0_rt/12.0_rt,
         -1.0_rt/12.0_rt};

    // --- chi tensor build on grown(2) boxes (per stage): the kappa
    // parsers and the free-streaming limiter depend on the evolving T_e,
    // so this is rebuilt for every SSP-RK2 stage. Computed straight into
    // the ghost ring (no comm) from Tin/bne ghosts.
    auto build_xi = [&] (amrex::MultiFab const & Tin)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(xi, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box const box = mfi.growntilebox(2);
            amrex::Array4<amrex::Real>       const & x_arr = xi.array(mfi);
            amrex::Array4<amrex::Real const> const & b_arr = bne.const_array(mfi);
            amrex::Array4<amrex::Real const> const & T_arr = Tin.const_array(mfi);

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real const ne  = b_arr(i,j,k,BNE::b_ne);
                amrex::Real const TeK = amrex::max(T_arr(i,j,k), 0.0_rt);
                amrex::Real const Te_eV = TeK * kb / qe;
                amrex::Real const B2 = b_arr(i,j,k,BNE::b_B2);
                bool const unmag = (B2 <= 0.0_rt);
                amrex::Real const ubx = b_arr(i,j,k,BNE::b_bx);
                amrex::Real const uby = b_arr(i,j,k,BNE::b_by);
                amrex::Real const ubz = b_arr(i,j,k,BNE::b_bz);

                amrex::Real chi_par =
                    kappa_par_ex(ne, Te_eV, t_now) / (1.5_rt * ne * kb);
                amrex::Real chi_perp = (unmag && !iso_any) ? chi_par :
                    kappa_perp_ex(ne, Te_eV, t_now) / (1.5_rt * ne * kb);
                chi_par  = amrex::max(chi_par,  0.0_rt);
                chi_perp = amrex::max(chi_perp, 0.0_rt);

                // iso options: same semantics as the SDE path
                if (iso_full) {
                    chi_par = chi_perp;
                } else if (iso_B > 0.0_rt) {
                    amrex::Real const s = B2 / (B2 + iso_B*iso_B);
                    chi_par = chi_perp + (chi_par - chi_perp) * s;
                }

                // Free-streaming limiter (longitudinal), same form as the
                // SDE path but on the subcycle-current T_e.
                bool const open = (b_arr(i,j,k,BNE::b_open) > 0.5_rt);
                if (f_lim > 0.0_rt && TeK > 0.0_rt && open &&
                    chi_par > 0.0_rt)
                {
                    amrex::Real gT[3] = {0.0_rt, 0.0_rt, 0.0_rt};
                    int const node[3] = {i, j, k};
                    for (int g = 0; g < AMREX_SPACEDIM; ++g) {
                        int cp = node[g] + 1, cm = node[g] - 1;
                        if (!is_per[g]) {
                            cp = amrex::min(cp, dom_hi[g]);
                            cm = amrex::max(cm, dom_lo[g]);
                        }
                        if (cp == cm) { continue; }
                        int p[3] = {i, j, k}, m[3] = {i, j, k};
                        p[g] = cp; m[g] = cm;
                        gT[gd2ax[g]] =
                            (T_arr(p[0],p[1],p[2]) - T_arr(m[0],m[1],m[2]))
                            * dxi_g[g] / amrex::Real(cp - cm);
                    }
                    amrex::Real const gparT =
                        std::abs(ubx*gT[0] + uby*gT[1] + ubz*gT[2]);
                    amrex::Real const q_sp = 1.5_rt*ne*kb*chi_par*gparT;
                    amrex::Real const q_fs = ne*kb*TeK*std::sqrt(kb*TeK/me);
                    chi_par /= (1.0_rt + q_sp / (f_lim * q_fs));
                }
                // No hop cap and no vacuum fast front here: subcycling
                // handles stability and closed floor faces handle vacuum.

                amrex::Real const dchi = chi_par - chi_perp;
                amrex::Real const ub3[3] = {ubx, uby, ubz};
                for (int g = 0; g < AMREX_SPACEDIM; ++g) {
                    for (int h = g; h < AMREX_SPACEDIM; ++h) {
                        int const sidx = g*AMREX_SPACEDIM - g*(g-1)/2 + (h-g);
                        amrex::Real const del =
                            (gd2ax[g] == gd2ax[h]) ? 1.0_rt : 0.0_rt;
                        x_arr(i,j,k,sidx) = chi_perp*del
                            + dchi*ub3[gd2ax[g]]*ub3[gd2ax[h]];
                    }
                }
            });
        }

    };

    // --- stability: Gershgorin bound, neighbor-max chi, n-ratio ----------
    auto stable_rate = [&] () -> amrex::Real
    {
        amrex::Real s_max = 0.0_rt;
        {
            amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
            amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;
            for (MFIter mfi(xi, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Box const box = mfi.tilebox();
                amrex::Array4<amrex::Real const> const & x_arr = xi.const_array(mfi);
                amrex::Array4<amrex::Real const> const & b_arr = bne.const_array(mfi);
                reduce_op.eval(box, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    if (b_arr(i,j,k,BNE::b_open) <= 0.5_rt) { return {0.0_rt}; }
                    amrex::Real const ne0 = b_arr(i,j,k,BNE::b_ne);
                    int const node[3] = {i, j, k};
                    amrex::Real nrat = 1.0_rt;
                    amrex::Real s = 0.0_rt;
                    for (int g = 0; g < AMREX_SPACEDIM; ++g) {
                        int p[3] = {i, j, k}, m[3] = {i, j, k};
                        p[g] = node[g] + 1; m[g] = node[g] - 1;
                        nrat = amrex::max(nrat,
                            0.5_rt*(1.0_rt + b_arr(p[0],p[1],p[2],BNE::b_ne)/ne0),
                            0.5_rt*(1.0_rt + b_arr(m[0],m[1],m[2],BNE::b_ne)/ne0));
                        for (int h = 0; h < AMREX_SPACEDIM; ++h) {
                            int const gg = amrex::min(g, h);
                            int const hh = amrex::max(g, h);
                            int const sidx =
                                gg*AMREX_SPACEDIM - gg*(gg-1)/2 + (hh-gg);
                            amrex::Real const xin = amrex::max(
                                std::abs(x_arr(i,j,k,sidx)),
                                std::abs(x_arr(p[0],p[1],p[2],sidx)),
                                std::abs(x_arr(m[0],m[1],m[2],sidx)));
                            s += 2.0_rt*xin*dxi_g[g]*dxi_g[h];
                        }
                    }
                    return {s*nrat};
                });
            }
            auto tup = reduce_data.value(reduce_op);
            s_max = amrex::get<0>(tup);
            amrex::ParallelDescriptor::ReduceRealMax(s_max);
        }
        return s_max;
    };

    // --- RHS for the adaptive integrator: K = dTe/dt = du/dt / (1.5 kB
    // ne), the flux divergence of the FD operator; ghost refresh of the
    // state is this functor's job (QdsmcRKIntegrator contract). Face
    // fluxes are evaluated inline per node (each interior face twice,
    // once per neighbor, from identical inputs -> bitwise identical
    // values, so Sigma(u) telescopes to round-off with no face storage
    // and no seam sync).
    auto eval_rhs = [&] (amrex::MultiFab & Tin, amrex::MultiFab & Kout)
    {
        ablastr::utils::communication::FillBoundary(
            Tin, WarpX::do_single_precision_comms, period, true);
        build_xi(Tin);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Kout, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box const box = mfi.tilebox();
            amrex::Array4<amrex::Real>       const & tn    = Kout.array(mfi);
            amrex::Array4<amrex::Real const> const & tc    = Tin.const_array(mfi);
            amrex::Array4<amrex::Real const> const & x_arr = xi.const_array(mfi);
            amrex::Array4<amrex::Real const> const & b_arr = bne.const_array(mfi);

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (b_arr(i,j,k,BNE::b_open) <= 0.5_rt) {
                    tn(i,j,k) = 0.0_rt;   // floored nodes keep their T_e
                    return;
                }

                auto Tat = [&] (int const p[3]) {
                    return tc(p[0], p[1], p[2]);
                };
                // clamped shift along grid dim g (non-periodic walls)
                auto shift = [&] (int const p[3], int const g, int const off,
                                  int q[3]) {
                    q[0] = p[0]; q[1] = p[1]; q[2] = p[2];
                    int idx = p[g] + off;
                    if (!is_per[g]) {
                        idx = amrex::min(amrex::max(idx, dom_lo[g]),
                                         dom_hi[g]);
                    }
                    q[g] = idx;
                };
                // central transverse derivative of T at node p along h
                auto dTh2 = [&] (int const p[3], int const h) {
                    int qp[3], qm[3];
                    shift(p, h, +1, qp);
                    shift(p, h, -1, qm);
                    if (qp[h] == qm[h]) { return 0.0_rt; }
                    return (Tat(qp) - Tat(qm)) * dxi_g[h]
                           / amrex::Real(qp[h] - qm[h]);
                };
                // 4th-order D0 transverse derivative (interior only)
                auto dTh4 = [&] (int const p[3], int const h) {
                    amrex::Real der = 0.0_rt;
                    for (int m = 0; m < 5; ++m) {
                        if (m == 2) { continue; }
                        int q[3] = {p[0], p[1], p[2]};
                        q[h] += (m - 2);
                        der += d0[m]*tc(q[0], q[1], q[2]);
                    }
                    return der*dxi_g[h];
                };

                // flux through the face between node m0 and m0 + e_g,
                // positive along +g, in u units [W/m^2]
                auto face_flux = [&] (int const m0[3], int const g) {
                    // outside a non-periodic wall: no face (adiabatic)
                    if (!is_per[g] &&
                        (m0[g] < dom_lo[g] || m0[g] + 1 > dom_hi[g])) {
                        return 0.0_rt;
                    }
                    int m1[3] = {m0[0], m0[1], m0[2]};
                    m1[g] += 1;
                    if (b_arr(m0[0],m0[1],m0[2],BNE::b_open) <= 0.5_rt ||
                        b_arr(m1[0],m1[1],m1[2],BNE::b_open) <= 0.5_rt) {
                        return 0.0_rt;   // closed floor faces
                    }
                    amrex::Real const ne_f = 0.5_rt*(
                        b_arr(m0[0],m0[1],m0[2],BNE::b_ne) +
                        b_arr(m1[0],m1[1],m1[2],BNE::b_ne));
                    amrex::Real const kfac = 1.5_rt*kb*ne_f;
                    int const sgg = g*AMREX_SPACEDIM - g*(g-1)/2;

                    // 4th-order stencils need the full interior window;
                    // near non-periodic walls fall back to 2nd order.
                    bool use4 = fd4;
                    if (use4 && !is_per[g]) {
                        use4 = (m0[g] - 2 >= dom_lo[g]) &&
                               (m0[g] + 3 <= dom_hi[g]);
                    }
                    if (use4) {
                        for (int h = 0; h < AMREX_SPACEDIM; ++h) {
                            if (h == g || is_per[h]) { continue; }
                            use4 = use4 &&
                                   (m0[h] - 2 >= dom_lo[h]) &&
                                   (m0[h] + 2 <= dom_hi[h]);
                        }
                    }

                    amrex::Real F = 0.0_rt;    // chi-units flux [K m/s]
                    if (use4) {
                        // co-derivative: C0 over A-row node fluxes
                        for (int l = 0; l < 4; ++l) {
                            int q[3] = {m0[0], m0[1], m0[2]};
                            q[g] += (l - 1);
                            amrex::Real der = 0.0_rt;
                            for (int m = 0; m < 6; ++m) {
                                int w[3] = {m0[0], m0[1], m0[2]};
                                w[g] += (m - 2);
                                der += A6[6*(l) + m]*tc(w[0], w[1], w[2]);
                            }
                            F += c0[l]*x_arr(q[0],q[1],q[2],sgg)
                                 *der*dxi_g[g];
                        }
                    } else {
                        F = 0.5_rt*(x_arr(m0[0],m0[1],m0[2],sgg) +
                                    x_arr(m1[0],m1[1],m1[2],sgg))
                            * (Tat(m1) - Tat(m0)) * dxi_g[g];
                    }

                    // cross-derivative fluxes as SMART-limited advection
                    for (int h = 0; h < AMREX_SPACEDIM; ++h) {
                        if (h == g) { continue; }
                        int const gg = amrex::min(g, h);
                        int const hh = amrex::max(g, h);
                        int const sgh =
                            gg*AMREX_SPACEDIM - gg*(gg-1)/2 + (hh-gg);
                        amrex::Real raw;
                        if (use4) {
                            raw = 0.0_rt;
                            for (int l = 0; l < 4; ++l) {
                                int q[3] = {m0[0], m0[1], m0[2]};
                                q[g] += (l - 1);
                                raw += c0[l]*x_arr(q[0],q[1],q[2],sgh)
                                       *dTh4(q, h);
                            }
                        } else {
                            raw = 0.5_rt*(
                                x_arr(m0[0],m0[1],m0[2],sgh)*dTh2(m0, h) +
                                x_arr(m1[0],m1[1],m1[2],sgh)*dTh2(m1, h));
                        }
                        if (raw == 0.0_rt) { continue; }
                        amrex::Real Tf = 0.5_rt*(Tat(m0) + Tat(m1));
                        if (Tf <= 0.0_rt) { continue; }
                        Tf = amrex::max(Tf,
                            1.0e-4_rt*amrex::max(Tat(m0), Tat(m1)));
                        amrex::Real const v = raw / Tf;
                        int uu[3];
                        amrex::Real tU, tD;
                        if (v > 0.0_rt) {
                            tU = Tat(m0); tD = Tat(m1);
                            shift(m0, g, -1, uu);
                        } else {
                            tU = Tat(m1); tD = Tat(m0);
                            shift(m1, g, +1, uu);
                        }
                        F += v*qdsmc_fd_face_value(Tat(uu), tU, tD,
                                                   fd_limiter);
                    }
                    return kfac*F;
                };

                int const node[3] = {i, j, k};
                amrex::Real div = 0.0_rt;
                for (int g = 0; g < AMREX_SPACEDIM; ++g) {
                    int lo[3] = {i, j, k};
                    lo[g] -= 1;
                    div += (face_flux(node, g) - face_flux(lo, g))
                           * dxi_g[g];
                }
                amrex::Real const ne0 = b_arr(i,j,k,BNE::b_ne);
                tn(i,j,k) = div/(1.5_rt*kb*ne0);
            });
        }
    };

    // Stability ceiling for the adaptive integrator (valid after each
    // RHS evaluation, which rebuilds xi). The Gershgorin normalization
    // puts the forward-Euler/SSP-RK2 real-axis edge at dt = 1/s (z = -2);
    // RKF45's edge sits at |z| ~ 3, hence the 3/2 scale. fd_cfl is the
    // user fraction of the edge (default 0.4, Nyquist-damping margin).
    amrex::Real const edge_scale = use_rkf45 ? 1.5_rt : 1.0_rt;
    auto cap = [&] () -> amrex::Real
    {
        amrex::Real const s_max = stable_rate();
        return (s_max > 0.0_rt) ? fd_cfl*edge_scale/s_max
                                : std::numeric_limits<amrex::Real>::max();
    };

    // Isothermal EB ring pin (Chebyshev distance
    // <= eb_ring of a covered node) to the parser T_wall and tally the
    // exchange -- identical semantics to the SDE path's ring pin (the
    // ring-2 default covers the deposition density ramp; the FD operator
    // has no deposit ramp, but the shared default keeps the arms
    // comparable; positive tally = energy into the plasma).
    auto pin_eb_ring = [&] (amrex::MultiFab & Tf)
    {
        auto const ebTe = m_cond_eb_Te;
        auto const plo_arr = geom.ProbLoArray();
        auto const dx_arr  = geom.CellSizeArray();
        int const eb_ring = m_cond_eb_ring;
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (MFIter mfi(Tf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box tile_box = mfi.tilebox();
            {   // unique node ownership (fixup-loop seam trim)
                amrex::Box const box_nodes =
                    amrex::surroundingNodes(mfi.validbox());
                for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                    if (tile_box.bigEnd(dd) == box_nodes.bigEnd(dd) &&
                        (box_nodes.bigEnd(dd) != dom_nodes.bigEnd(dd) ||
                         geom.isPeriodic(dd))) {
                        tile_box.growHi(dd, -1);
                    }
                }
            }
            amrex::Array4<amrex::Real>       const & Te_arr  = Tf.array(mfi);
            amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);
            amrex::Array4<amrex::Real const> const & b_arr   = bne.const_array(mfi);
            reduce_op.eval(tile_box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                if (b_arr(i,j,k,BNE::b_ebm) == 0.0_rt) { return {0.0_rt}; }
                amrex::Real const ne = rho_arr(i,j,k) / qe;
                if (ne <= 0.0_rt) { return {0.0_rt}; }
                bool ring = false;
                int const node[3] = {i, j, k};
                int const rr = eb_ring;
#if defined(WARPX_DIM_3D)
                for (int ok = -rr; ok <= rr; ++ok) {
#else
                int const ok = 0;
                {
#endif
                for (int oj = -rr; oj <= rr; ++oj) {
                for (int oi = -rr; oi <= rr; ++oi) {
                    if (oi == 0 && oj == 0 && ok == 0) { continue; }
                    ring = ring ||
                        (b_arr(node[0]+oi, node[1]+oj, node[2]+ok,
                               BNE::b_ebm) == 0.0_rt);
                }}}
                if (!ring) { return {0.0_rt}; }
                amrex::Real cx[3] = {0.0_rt, 0.0_rt, 0.0_rt};
                for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                    cx[dd] = plo_arr[dd] + amrex::Real(node[dd])*dx_arr[dd];
                }
#if defined(WARPX_DIM_3D)
                amrex::Real const Te_eV = ebTe(cx[0], cx[1], cx[2]);
#else
                amrex::Real const Te_eV = ebTe(cx[0], 0.0_rt, cx[1]);
#endif
                amrex::Real const TwK = Te_eV * qe / kb;
                amrex::Real const du =
                    1.5_rt * kb * ne * (TwK - Te_arr(i,j,k));
                Te_arr(i,j,k) = TwK;
                return {du};
            });
        }
        auto tup = reduce_data.value(reduce_op);
        amrex::Real tly = amrex::get<0>(tup);
        amrex::ParallelDescriptor::ReduceRealSum(tly);
        m_cond_eb_tally += tly;
        };

    // Bath pins must track the SUBCYCLE cadence: applied once per call
    // they are a contact resistance whose magnitude GROWS under
    // refinement (bath-row heat capacity ~ dx; measured anti-convergent
    // slab wall flux 0.986 -> 0.967 from N=64 -> 128). Re-pinning after
    // every accepted subcycle makes the deficit ~ dt_sub ~ dx^2 -- the
    // FD analog of the SDE fold-back's continuous bath sampling. The
    // domain helper also handles flux-injection BCs, scaled by the
    // accepted dt so the total injection sums to dt_c exactly.
    auto post_step = [&] (amrex::MultiFab & yy, amrex::Real const dts)
    {
        ApplyQdsmcConductionWallBCs(lev, dts, yy, rho);
        if (eb_iso) { pin_eb_ring(yy); }
    };

    QdsmcRKIntegrator const integ(
        use_rkf45 ? QdsmcRKIntegrator::Scheme::RKF45
                  : QdsmcRKIntegrator::Scheme::SSPRK2,
        eval_rhs, cap, m_cond_fd_rtol, m_cond_fd_atol,
        m_substep_safety, m_substep_max_growth, max_sub, post_step);
    QdsmcRKStats const st = integ.Advance(T_cur, dt_c);

    if (st.t_done < dt_c*(1.0_rt - 1.0e-12_rt)) {
        amrex::Warning(
            "[qdsmc] QdsmcConductionOnceFD: attempts budget ("
            + std::to_string(max_sub) + ") hit; dropped "
            + std::to_string(1.0 - st.t_done/dt_c)
            + " of the conduction substep ("
            + std::to_string(st.n_accepted) + "/"
            + std::to_string(st.n_attempts)
            + " accepted) -- raise qdsmc_conduction_fd_max_subcycles or "
            "loosen qdsmc_conduction_fd_rtol");
    }

    amrex::MultiFab::Copy(Te, T_cur, 0, 0, 1, 0);

    ablastr::utils::communication::FillBoundary(
        Te, WarpX::do_single_precision_comms, period, true);
}

void HybridPICModel::QdsmcConductionOnce (int const lev, amrex::Real const dt_c,
                                          bool const use_rho_new) const
{
    if (!m_include_thermal_conduction) { return; }

    if (m_cond_operator == 1) {
        QdsmcConductionOnceFD(lev, dt_c, use_rho_new);
        return;
    }

    ABLASTR_PROFILE("HybridPICModel::QdsmcConductionOnce()");

    auto & warpx = WarpX::GetInstance();
    using ablastr::fields::Direction;

    amrex::Geometry const & geom = warpx.Geom(lev);
    amrex::Periodicity const & period = geom.periodicity();
    auto const dx_arr  = geom.CellSizeArray();
    auto const dxi_arr = geom.InvCellSizeArray();

    amrex::MultiFab & Te =
        *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    // The rho paired with T_e at this substep's time level: rho^n
    // (hybrid_rho_fp_temp) for the pre-transport half, rho^{n+1} (rho_fp)
    // for the post-transport half. Used consistently for the u build, the
    // grad ln n drift AND the recovery, so pure-T_e dynamics are exact.
    amrex::MultiFab const & rho = use_rho_new
        ? *warpx.m_fields.get(FieldType::rho_fp, lev)
        : *warpx.m_fields.get(FieldType::hybrid_rho_fp_temp, lev);
    amrex::MultiFab const & Bx = *warpx.m_fields.get(FieldType::Bfield_fp, Direction{0}, lev);
    amrex::MultiFab const & By = *warpx.m_fields.get(FieldType::Bfield_fp, Direction{1}, lev);
    amrex::MultiFab const & Bz = *warpx.m_fields.get(FieldType::Bfield_fp, Direction{2}, lev);
    // TODO(add_external_fields): Bfield_fp holds the internal field only;
    // with hybrid external vector-potential fields on, b should be built
    // from B + B_ext. Not needed by the current research decks.

    // T_e ghosts feed the flux-limiter gradient and the source-node u
    // slopes below.
    ablastr::utils::communication::FillBoundary(
        Te, WarpX::do_single_precision_comms, period, true);

    amrex::Real const t_now = warpx.gett_new(lev);
    amrex::Real const kb = PhysConst::kb;
    amrex::Real const qe = PhysConst::q_e;
    amrex::Real const me = PhysConst::m_e;
    auto const kappa_par_ex  = m_kappa_par;
    bool const iso_launch    = m_cond_iso_launch;
    auto const kappa_perp_ex = m_kappa_perp;
    amrex::Real const n_floor  = m_qdsmc_n_floor;
    amrex::Real const f_lim    = m_cond_flux_limit_factor;
    bool const vac_fast        = m_cond_vacuum_fast_front;
    bool const grad_dep        = m_qdsmc_gradient_deposit;
    bool const iso_full        = m_cond_isotropic;
    amrex::Real const iso_B    = m_cond_iso_B;
    bool const iso_any         = iso_full || (iso_B > 0.0_rt);

    amrex::GpuArray<amrex::Real, 8> xq_par, wq_par, xq_perp, wq_perp;
    qdsmc_gh_table(m_cond_npts_par,  xq_par,  wq_par);
    qdsmc_gh_table(m_cond_npts_perp, xq_perp, wq_perp);
    int const nq_par  = m_cond_npts_par;
    int const nq_perp = m_cond_npts_perp;
    amrex::Real const xmax_par  = std::abs(xq_par[0]);
    amrex::Real const xmax_perp = std::abs(xq_perp[0]);

    // Hop-cap chi ceilings: largest quadrature offset x_max sqrt(2 chi dt_c)
    // limited to m_cond_max_hop * dx_min (per direction class, since the
    // largest abscissa differs with npts).
    amrex::Real dx_min = dx_arr[0];
    for (int d = 1; d < AMREX_SPACEDIM; ++d) { dx_min = std::min(dx_min, dx_arr[d]); }
    amrex::Real const hop = m_cond_max_hop * dx_min;
    amrex::Real const chi_cap_par  = hop*hop / (2.0_rt * dt_c * xmax_par*xmax_par);
    amrex::Real const chi_cap_perp = hop*hop / (2.0_rt * dt_c * xmax_perp*xmax_perp);

    // Grid-dim bookkeeping for the device kernels: physical axis -> grid
    // dim (-1 where collapsed), physical-axis cell sizes, periodicity and
    // the nodal domain extents per grid dim.
#if defined(WARPX_DIM_3D)
    amrex::GpuArray<int, 3> const ax2gd = {0, 1, 2};
#elif (AMREX_SPACEDIM == 2)
    amrex::GpuArray<int, 3> const ax2gd = {0, -1, 1};
#else
    amrex::GpuArray<int, 3> const ax2gd = {-1, -1, 0};
#endif
    amrex::GpuArray<amrex::Real, 3> dxi3 = {0.0_rt, 0.0_rt, 0.0_rt};
    for (int ax = 0; ax < 3; ++ax) {
        if (ax2gd[ax] >= 0) { dxi3[ax] = dxi_arr[ax2gd[ax]]; }
    }
    amrex::Box const dom_nodes = amrex::surroundingNodes(geom.Domain());
    amrex::GpuArray<int, 3> is_per = {0, 0, 0};
    amrex::GpuArray<int, 3> dom_lo = {0, 0, 0};
    amrex::GpuArray<int, 3> dom_hi = {0, 0, 0};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        is_per[d] = geom.isPeriodic(d) ? 1 : 0;
        dom_lo[d] = dom_nodes.smallEnd(d);
        dom_hi[d] = dom_nodes.bigEnd(d);
    }

    // B staggering for the node interpolation (collapsed dims match the
    // source staggering so Interp does not read the out-of-bounds
    // neighbour there, as in the T_i collapse above).
    amrex::GpuArray<int, 3> const Bx_stag = Bx_IndexType;
    amrex::GpuArray<int, 3> const By_stag = By_IndexType;
    amrex::GpuArray<int, 3> const Bz_stag = Bz_IndexType;
    amrex::GpuArray<int, 3> nd_x = {1, 1, 1};
    amrex::GpuArray<int, 3> nd_y = {1, 1, 1};
    amrex::GpuArray<int, 3> nd_z = {1, 1, 1};
    for (int d = AMREX_SPACEDIM; d < 3; ++d) {
        nd_x[d] = Bx_stag[d]; nd_y[d] = By_stag[d]; nd_z[d] = Bz_stag[d];
    }
    amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};

    // Scratch: per-node capped diffusivities, unit b, and the full tensor
    // D = chi_perp I + (chi_par - chi_perp) b b (whose divergence feeds the
    // Ito drift). One ghost ring for the drift's central differences.
    enum Cond : int { c_chip = 0, c_chiq, c_bx, c_by, c_bz,
                      c_dxx, c_dxy, c_dxz, c_dyy, c_dyz, c_dzz, c_ncomp };
    // The layer form's curved-foot trace samples b at the half-displacement
    // point, up to ~max_hop cells away; the scatter form only reads +-1.
    int const ng_cond = (m_cond_form == 1 && m_cond_curved_feet)
        ? static_cast<int>(std::ceil(m_cond_max_hop)) + 2 : 1;
    amrex::MultiFab cond(Te.boxArray(), Te.DistributionMap(), Cond::c_ncomp,
                         ng_cond);

    // Deposit target: guard reach must cover the clamped hop.
    bool const dep_keys = (m_cond_deposit_kernel == 1);
    bool const compensate = m_cond_compensate;
    int const ng_dep = static_cast<int>(std::ceil(m_cond_max_hop))
                       + (dep_keys ? 3 : 2);
    // comps 1..SPACEDIM (compensated deposit only): the bookkept per-axis
    // deposit covariance, u-weighted fr(1-fr)dx^2, hat-distributed -- the
    // exact antidiffusion coefficient for the FCT pass below.
    int const nc_dep = compensate ? 1 + AMREX_SPACEDIM : 1;
    amrex::MultiFab u_dep(Te.boxArray(), Te.DistributionMap(), nc_dep,
                          ng_dep);

    // --- Pass 1: capped/limited diffusivities and the D tensor per node ---
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(cond, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Box const box = mfi.tilebox();
        amrex::Array4<amrex::Real>       const & c_arr   = cond.array(mfi);
        amrex::Array4<amrex::Real const> const & Te_arr  = Te.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);
        amrex::Array4<amrex::Real const> const & Bx_arr  = Bx.const_array(mfi);
        amrex::Array4<amrex::Real const> const & By_arr  = By.const_array(mfi);
        amrex::Array4<amrex::Real const> const & Bz_arr  = Bz.const_array(mfi);

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real const ne_raw = rho_arr(i,j,k) / qe;
            bool const vac = (ne_raw <= n_floor);
            amrex::Real const ne  = amrex::max(ne_raw, n_floor);
            amrex::Real const TeK = amrex::max(Te_arr(i,j,k), 0.0_rt);
            amrex::Real const Te_eV = TeK * kb / qe;

            // Unit b from B interpolated to the node; unmagnetized nodes
            // conduct isotropically at chi_par.
            amrex::Real const bxv = ablastr::coarsen::sample::Interp(
                Bx_arr, Bx_stag, nd_x, coarsen, i, j, k, 0);
            amrex::Real const byv = ablastr::coarsen::sample::Interp(
                By_arr, By_stag, nd_y, coarsen, i, j, k, 0);
            amrex::Real const bzv = ablastr::coarsen::sample::Interp(
                Bz_arr, Bz_stag, nd_z, coarsen, i, j, k, 0);
            amrex::Real const B2 = bxv*bxv + byv*byv + bzv*bzv;
            bool const unmag = (B2 <= 0.0_rt);
            amrex::Real const Binv = unmag ? 0.0_rt : 1.0_rt/std::sqrt(B2);
            amrex::Real const ubx = unmag ? 0.0_rt : bxv*Binv;
            amrex::Real const uby = unmag ? 0.0_rt : byv*Binv;
            amrex::Real const ubz = unmag ? 1.0_rt : bzv*Binv;

            amrex::Real chi_par =
                kappa_par_ex(ne, Te_eV, t_now) / (1.5_rt * ne * kb);
            amrex::Real chi_perp = (unmag && !iso_any) ? chi_par :
                kappa_perp_ex(ne, Te_eV, t_now) / (1.5_rt * ne * kb);
            chi_par  = amrex::max(chi_par,  0.0_rt);
            chi_perp = amrex::max(chi_perp, 0.0_rt);

            // Isotropic-conduction options: where |B| is small, b-hat is
            // noise and the field-aligned tensor points the (huge) chi_par
            // in per-node random directions -- at the reconnection null
            // this is spurious violent mixing exactly where the event
            // lives. Full mode conducts at the cross-field rate everywhere
            // (chi_par == chi_perp collapses D to chi_perp I: the frame
            // drops out exactly). The |B|-threshold blend pulls chi_par
            // smoothly to chi_perp below B_iso: s = B^2/(B^2 + B_iso^2).
            if (iso_full) {
                chi_par = chi_perp;
            } else if (iso_B > 0.0_rt) {
                amrex::Real const s = B2 / (B2 + iso_B*iso_B);
                chi_par = chi_perp + (chi_par - chi_perp) * s;
            }

            // Physical free-streaming limiter (longitudinal only):
            // kappa_eff = kappa / (1 + |q_Sp| / (f q_fs)), with
            // q_Sp = kappa |grad_par T_e| and q_fs = n_e k_B T_e v_te.
            if (f_lim > 0.0_rt && TeK > 0.0_rt && !vac && chi_par > 0.0_rt) {
                amrex::Real gT[3] = {0.0_rt, 0.0_rt, 0.0_rt};
                for (int ax = 0; ax < 3; ++ax) {
                    int const g = ax2gd[ax];
                    if (g < 0) { continue; }
                    int c[3] = {i, j, k};
                    int cp = c[g] + 1, cm = c[g] - 1;
                    if (!is_per[g]) {
                        cp = amrex::min(cp, dom_hi[g]);
                        cm = amrex::max(cm, dom_lo[g]);
                    }
                    if (cp == cm) { continue; }
                    int p[3] = {i, j, k}, m[3] = {i, j, k};
                    p[g] = cp; m[g] = cm;
                    gT[ax] = (Te_arr(p[0],p[1],p[2]) - Te_arr(m[0],m[1],m[2]))
                             * dxi3[ax] / amrex::Real(cp - cm);
                }
                amrex::Real const gparT =
                    std::abs(ubx*gT[0] + uby*gT[1] + ubz*gT[2]);
                amrex::Real const q_sp = 1.5_rt * ne * kb * chi_par * gparT;
                amrex::Real const q_fs = ne * kb * TeK * std::sqrt(kb*TeK/me);
                chi_par /= (1.0_rt + q_sp / (f_lim * q_fs));
            }

            // Numerical hop cap: smooth p = 4 soft-min
            // chi / (1 + (chi/cap)^4)^(1/4), sharp enough that chi <= cap/2
            // is biased < 0.4% (a harmonic soft-min would bite ~20% already
            // at cap/4 and pollute convergence), yet kink-free where the
            // cap engages so div D stays smooth for the drift term.
            if (chi_par > 0.0_rt) {
                amrex::Real const r2 = (chi_par/chi_cap_par)*(chi_par/chi_cap_par);
                chi_par /= std::sqrt(std::sqrt(1.0_rt + r2*r2));
            }
            if (chi_perp > 0.0_rt) {
                amrex::Real const r2 = (chi_perp/chi_cap_perp)*(chi_perp/chi_cap_perp);
                chi_perp /= std::sqrt(std::sqrt(1.0_rt + r2*r2));
            }
            // Vacuum policy: floored cells conduct isotropically at the
            // capped ceiling (fast-but-finite front, not a stall).
            if (vac && vac_fast) {
                chi_par  = chi_cap_par;
                chi_perp = chi_cap_perp;
            }

            amrex::Real const dchi = chi_par - chi_perp;
            c_arr(i,j,k,Cond::c_chip) = chi_par;
            c_arr(i,j,k,Cond::c_chiq) = chi_perp;
            c_arr(i,j,k,Cond::c_bx)   = ubx;
            c_arr(i,j,k,Cond::c_by)   = uby;
            c_arr(i,j,k,Cond::c_bz)   = ubz;
            c_arr(i,j,k,Cond::c_dxx)  = chi_perp + dchi*ubx*ubx;
            c_arr(i,j,k,Cond::c_dxy)  = dchi*ubx*uby;
            c_arr(i,j,k,Cond::c_dxz)  = dchi*ubx*ubz;
            c_arr(i,j,k,Cond::c_dyy)  = chi_perp + dchi*uby*uby;
            c_arr(i,j,k,Cond::c_dyz)  = dchi*uby*ubz;
            c_arr(i,j,k,Cond::c_dzz)  = chi_perp + dchi*ubz*ubz;
        });
    }
    cond.FillBoundary(period);

    // --- Layer (gather) form: T^{n+1}(node) = sum_q w_q T^n(foot_q) ------
    // The Milstein layer/adjoint transfer: T satisfies the backward
    // equation dT/dt = D:grad grad T + [div D + D grad ln n].grad T (the
    // same Ito drift as the scatter form), so each node gathers the
    // interpolated previous layer at its forward-SDE quadrature feet.
    // Pointwise interpolation has no deposit-moment frame, so the
    // curvature-activated remap leak of the scatter deposit is absent by
    // construction; the cost is exact conservation (optional global
    // fixup below). Pure gather = race-free: OMP-threadable, unlike the
    // scatter deposit. TODO(Thrust D): reflected feet at walls replace the
    // E7-style clamp; vacuum_fast_front semantics differ here (floored
    // nodes keep their Te and are only read, never updated).
    if (m_cond_form == 1)
    {
        int const ng_gather =
            static_cast<int>(std::ceil(m_cond_max_hop)) + 3;
        amrex::MultiFab T_old(Te.boxArray(), Te.DistributionMap(), 1,
                              ng_gather);
        T_old.setVal(0.0_rt);
        amrex::MultiFab::Copy(T_old, Te, 0, 0, 1, 0);
        ablastr::utils::communication::FillBoundary(
            T_old, WarpX::do_single_precision_comms, period, true);
        amrex::MultiFab T_new(Te.boxArray(), Te.DistributionMap(), 1, 0);

        int const interp_kind = m_cond_interp;
        bool const curved = m_cond_curved_feet;
        // stencil reach: foot floor +- (1, 2) for the cubic kinds
        amrex::Real const rmax_g = amrex::Real(ng_gather - 3) - 1.0e-6_rt;
        amrex::Real const rmax_c = amrex::Real(ng_cond - 1) - 1.0e-6_rt;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(T_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box const gbox = mfi.tilebox();
            amrex::Array4<amrex::Real>       const & tn      = T_new.array(mfi);
            amrex::Array4<amrex::Real const> const & to      = T_old.const_array(mfi);
            amrex::Array4<amrex::Real const> const & c_arr   = cond.const_array(mfi);
            amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);

            amrex::ParallelFor(gbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // floored nodes keep their previous T_e (the scatter
                // recovery's semantics)
                amrex::Real const ne0 = rho_arr(i,j,k) / qe;
                if (ne0 <= n_floor) { tn(i,j,k) = to(i,j,k); return; }

                int const node[3] = {i, j, k};

                // Ito drift a = div D + D . grad ln n_e -- same clamped
                // central differences as the scatter form.
                amrex::Real dD[3][3];
                amrex::Real gln[3];
                {
                    int const drow[3][3] = {
                        {Cond::c_dxx, Cond::c_dxy, Cond::c_dxz},
                        {Cond::c_dxy, Cond::c_dyy, Cond::c_dyz},
                        {Cond::c_dxz, Cond::c_dyz, Cond::c_dzz}};
                    amrex::Real const rho_fl =
                        amrex::max(rho_arr(i,j,k), n_floor*qe);
                    for (int ax = 0; ax < 3; ++ax) {
                        int const g = ax2gd[ax];
                        if (g < 0) {
                            for (int r = 0; r < 3; ++r) { dD[r][ax] = 0.0_rt; }
                            gln[ax] = 0.0_rt;
                            continue;
                        }
                        int cp = node[g] + 1, cm = node[g] - 1;
                        if (!is_per[g]) {
                            cp = amrex::min(cp, dom_hi[g]);
                            cm = amrex::max(cm, dom_lo[g]);
                        }
                        if (cp == cm) {
                            for (int r = 0; r < 3; ++r) { dD[r][ax] = 0.0_rt; }
                            gln[ax] = 0.0_rt;
                            continue;
                        }
                        int p[3] = {i, j, k}, m[3] = {i, j, k};
                        p[g] = cp; m[g] = cm;
                        amrex::Real const fac = dxi3[ax] / amrex::Real(cp - cm);
                        for (int r = 0; r < 3; ++r) {
                            dD[r][ax] = (c_arr(p[0],p[1],p[2],drow[r][ax])
                                       - c_arr(m[0],m[1],m[2],drow[r][ax])) * fac;
                        }
                        gln[ax] = (rho_arr(p[0],p[1],p[2])
                                 - rho_arr(m[0],m[1],m[2])) * fac / rho_fl;
                    }
                }
                amrex::Real const Dxx = c_arr(i,j,k,Cond::c_dxx);
                amrex::Real const Dxy = c_arr(i,j,k,Cond::c_dxy);
                amrex::Real const Dxz = c_arr(i,j,k,Cond::c_dxz);
                amrex::Real const Dyy = c_arr(i,j,k,Cond::c_dyy);
                amrex::Real const Dyz = c_arr(i,j,k,Cond::c_dyz);
                amrex::Real const Dzz = c_arr(i,j,k,Cond::c_dzz);
                amrex::Real drift[3];
                drift[0] = (dD[0][0] + dD[0][1] + dD[0][2]
                            + Dxx*gln[0] + Dxy*gln[1] + Dxz*gln[2]) * dt_c;
                drift[1] = (dD[1][0] + dD[1][1] + dD[1][2]
                            + Dxy*gln[0] + Dyy*gln[1] + Dyz*gln[2]) * dt_c;
                drift[2] = (dD[2][0] + dD[2][1] + dD[2][2]
                            + Dxz*gln[0] + Dyz*gln[1] + Dzz*gln[2]) * dt_c;
                for (int ax = 0; ax < 3; ++ax) {
                    drift[ax] = amrex::min(amrex::max(drift[ax], -hop), hop);
                }

                // node quadrature frame (b, e1, e2), as in the scatter form
                amrex::Real const bxv = c_arr(i,j,k,Cond::c_bx);
                amrex::Real const byv = c_arr(i,j,k,Cond::c_by);
                amrex::Real const bzv = c_arr(i,j,k,Cond::c_bz);
                amrex::Real e1x = -byv*bxv, e1y = 1.0_rt - byv*byv, e1z = -byv*bzv;
                amrex::Real e1n = e1x*e1x + e1y*e1y + e1z*e1z;
                if (e1n < 1.0e-12_rt) {
                    e1x = 1.0_rt - bxv*bxv; e1y = -bxv*byv; e1z = -bxv*bzv;
                    e1n = e1x*e1x + e1y*e1y + e1z*e1z;
                }
                amrex::Real const e1i = 1.0_rt/std::sqrt(e1n);
                e1x *= e1i; e1y *= e1i; e1z *= e1i;
                amrex::Real const e2x = byv*e1z - bzv*e1y;
                amrex::Real const e2y = bzv*e1x - bxv*e1z;
                amrex::Real const e2z = bxv*e1y - byv*e1x;

                amrex::Real const chi_par  = c_arr(i,j,k,Cond::c_chip);
                amrex::Real const chi_perp = c_arr(i,j,k,Cond::c_chiq);
                amrex::Real const sig_par  = std::sqrt(2.0_rt*chi_par *dt_c);
                amrex::Real const sig_perp = std::sqrt(2.0_rt*chi_perp*dt_c);

                auto proj_sq = [&] (amrex::Real vx, amrex::Real vy, amrex::Real vz)
                {
                    amrex::Real s2 = 0.0_rt;
                    for (int ax = 0; ax < 3; ++ax) {
                        amrex::Real const v = (ax == 0) ? vx : ((ax == 1) ? vy : vz);
                        if (ax2gd[ax] >= 0) { s2 += v*v; }
                    }
                    return s2;
                };
                int const n0 = (sig_par  > 0.0_rt &&
                                proj_sq(bxv, byv, bzv) > 1.0e-24_rt) ? nq_par  : 1;

                // Isotropized launch: second, 45-degree-rotated perp
                // lattice at half variance (see the scatter form).
                int const n_lat = (iso_launch && sig_perp > 0.0_rt) ? 2 : 1;
                // full sigma per lattice; see the scatter-form note
                amrex::Real const sig_q = sig_perp;
                amrex::Real const w_lat = (n_lat == 2) ? 0.5_rt : 1.0_rt;

                // multilinear sample of a cond component at fractional
                // index coords (curved-foot midpoint b), clamped into the
                // valid region on non-periodic dims
                auto csample = [&] (int const comp, amrex::Real const * sidx)
                {
                    int ib[3] = {i, j, k};
                    amrex::Real fb[3] = {0.0_rt, 0.0_rt, 0.0_rt};
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        auto const fl = std::floor(sidx[d]);
                        ib[d] = static_cast<int>(fl);
                        fb[d] = sidx[d] - fl;
                    }
                    amrex::Real acc = 0.0_rt;
                    for (int corner = 0; corner < (1 << AMREX_SPACEDIM); ++corner) {
                        int idx[3] = {ib[0], ib[1], ib[2]};
                        amrex::Real w = 1.0_rt;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            if (corner & (1 << d)) { idx[d] += 1; w *= fb[d]; }
                            else                   { w *= 1.0_rt - fb[d]; }
                            if (!is_per[d]) {
                                idx[d] = amrex::min(
                                    amrex::max(idx[d], dom_lo[d]), dom_hi[d]);
                            }
                        }
                        acc += w * c_arr(idx[0], idx[1], idx[2], comp);
                    }
                    return acc;
                };

                amrex::Real acc_T = 0.0_rt;
                for (int lat = 0; lat < n_lat; ++lat) {
                amrex::Real p1x = e1x, p1y = e1y, p1z = e1z;
                amrex::Real p2x = e2x, p2y = e2y, p2z = e2z;
                if (lat == 1) {
                    amrex::Real const sr = 0.7071067811865475_rt;
                    p1x = (e1x + e2x)*sr; p1y = (e1y + e2y)*sr;
                    p1z = (e1z + e2z)*sr;
                    p2x = (e2x - e1x)*sr; p2y = (e2y - e1y)*sr;
                    p2z = (e2z - e1z)*sr;
                }
                int const n1 = (sig_q > 0.0_rt &&
                                proj_sq(p1x, p1y, p1z) > 1.0e-24_rt) ? nq_perp : 1;
                int const n2 = (sig_q > 0.0_rt &&
                                proj_sq(p2x, p2y, p2z) > 1.0e-24_rt) ? nq_perp : 1;
                for (int q0 = 0; q0 < n0; ++q0) {
                for (int q1 = 0; q1 < n1; ++q1) {
                for (int q2 = 0; q2 < n2; ++q2) {
                    amrex::Real const o0 = (n0 > 1) ? xq_par[q0]*sig_par : 0.0_rt;
                    amrex::Real const o1 = (n1 > 1) ? xq_perp[q1]*sig_q  : 0.0_rt;
                    amrex::Real const o2 = (n2 > 1) ? xq_perp[q2]*sig_q  : 0.0_rt;
                    amrex::Real const wq = w_lat
                                         * ((n0 > 1) ? wq_par[q0]  : 1.0_rt)
                                         * ((n1 > 1) ? wq_perp[q1] : 1.0_rt)
                                         * ((n2 > 1) ? wq_perp[q2] : 1.0_rt);
                    amrex::Real disp[3] = {
                        drift[0] + o0*bxv + o1*p1x + o2*p2x,
                        drift[1] + o0*byv + o1*p1y + o2*p2y,
                        drift[2] + o0*bzv + o1*p1z + o2*p2z};

                    // curved foot: midpoint/RK2 rotation -- re-sample b at
                    // the half-displacement point, rebuild the frame there,
                    // and retake the quadrature offsets in the rotated
                    // frame (the drift stays node-evaluated: it is already
                    // the mean-curvature correction; midpoint evaluation
                    // of it would be a higher-order refinement).
                    if (curved &&
                        (o0 != 0.0_rt || o1 != 0.0_rt || o2 != 0.0_rt)) {
                        amrex::Real smid[AMREX_SPACEDIM];
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            int ax = 2;
#if defined(WARPX_DIM_3D)
                            ax = d;
#elif (AMREX_SPACEDIM == 2)
                            ax = (d == 0) ? 0 : 2;
#endif
                            amrex::Real dm = 0.5_rt*disp[ax]*dxi_arr[d];
                            dm = amrex::min(amrex::max(dm, -rmax_c), rmax_c);
                            smid[d] = amrex::Real(node[d]) + dm;
                            if (!is_per[d]) {
                                smid[d] = amrex::min(
                                    amrex::max(smid[d], amrex::Real(dom_lo[d])),
                                    amrex::Real(dom_hi[d]) - 1.0e-6_rt);
                            }
                        }
                        amrex::Real const bmx = csample(Cond::c_bx, smid);
                        amrex::Real const bmy = csample(Cond::c_by, smid);
                        amrex::Real const bmz = csample(Cond::c_bz, smid);
                        amrex::Real const bm2 = bmx*bmx + bmy*bmy + bmz*bmz;
                        if (bm2 > 1.0e-12_rt) {
                            amrex::Real const bmi = 1.0_rt/std::sqrt(bm2);
                            amrex::Real const ubx = bmx*bmi;
                            amrex::Real const uby = bmy*bmi;
                            amrex::Real const ubz = bmz*bmi;
                            amrex::Real f1x = -uby*ubx, f1y = 1.0_rt - uby*uby,
                                        f1z = -uby*ubz;
                            amrex::Real f1n = f1x*f1x + f1y*f1y + f1z*f1z;
                            if (f1n < 1.0e-12_rt) {
                                f1x = 1.0_rt - ubx*ubx; f1y = -ubx*uby;
                                f1z = -ubx*ubz;
                                f1n = f1x*f1x + f1y*f1y + f1z*f1z;
                            }
                            amrex::Real const f1i = 1.0_rt/std::sqrt(f1n);
                            f1x *= f1i; f1y *= f1i; f1z *= f1i;
                            amrex::Real f2x = uby*f1z - ubz*f1y;
                            amrex::Real f2y = ubz*f1x - ubx*f1z;
                            amrex::Real f2z = ubx*f1y - uby*f1x;
                            if (lat == 1) {
                                // rotated lattice: rotate the rebuilt
                                // midpoint frame the same 45 degrees
                                amrex::Real const sr = 0.7071067811865475_rt;
                                amrex::Real const g1x = (f1x + f2x)*sr;
                                amrex::Real const g1y = (f1y + f2y)*sr;
                                amrex::Real const g1z = (f1z + f2z)*sr;
                                f2x = (f2x - f1x)*sr;
                                f2y = (f2y - f1y)*sr;
                                f2z = (f2z - f1z)*sr;
                                f1x = g1x; f1y = g1y; f1z = g1z;
                            }
                            disp[0] = drift[0] + o0*ubx + o1*f1x + o2*f2x;
                            disp[1] = drift[1] + o0*uby + o1*f1y + o2*f2y;
                            disp[2] = drift[2] + o0*ubz + o1*f1z + o2*f2z;
                        }
                    }

                    // foot in index space, clamped to the interpolation
                    // reach and (E7-style, pending Thrust D) into the
                    // domain on non-periodic edges
                    int i0[3] = {i, j, k};
                    amrex::Real fr[3] = {0.0_rt, 0.0_rt, 0.0_rt};
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        int ax = 2;
#if defined(WARPX_DIM_3D)
                        ax = d;
#elif (AMREX_SPACEDIM == 2)
                        ax = (d == 0) ? 0 : 2;
#endif
                        amrex::Real ds = disp[ax]*dxi_arr[d];
                        ds = amrex::min(amrex::max(ds, -rmax_g), rmax_g);
                        amrex::Real s = amrex::Real(node[d]) + ds;
                        if (!is_per[d]) {
                            s = amrex::min(
                                amrex::max(s, amrex::Real(dom_lo[d])),
                                amrex::Real(dom_hi[d]) - 1.0e-6_rt);
                        }
                        auto const fl = std::floor(s);
                        i0[d] = static_cast<int>(fl);
                        fr[d] = s - fl;
                    }
                    acc_T += wq*qdsmc_interp_nd(interp_kind, to,
                        i0[0], i0[1], i0[2], fr[0], fr[1], fr[2],
                        is_per, dom_lo, dom_hi);
                }}}
                }   // lat
                tn(i,j,k) = acc_T;
            });
        }

        // Optional conservation fixup: restore Sigma(rho T) ~ Sigma(u)
        // over uniquely-owned nodes with a global proportional rescale.
        // (The floored nodes kept their old T_e but are rescaled too --
        // acceptable for the prototype; keep OFF while measuring.)
        if (m_cond_conserve_fixup) {
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
            amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;
            for (MFIter mfi(T_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Box tile_box = mfi.tilebox();
                {
                    amrex::Box const box_nodes =
                        amrex::surroundingNodes(mfi.validbox());
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        if (tile_box.bigEnd(d) == box_nodes.bigEnd(d) &&
                            (box_nodes.bigEnd(d) != dom_nodes.bigEnd(d) ||
                             geom.isPeriodic(d))) {
                            tile_box.growHi(d, -1);
                        }
                    }
                }
                amrex::Array4<amrex::Real const> const & tn = T_new.const_array(mfi);
                amrex::Array4<amrex::Real const> const & to = T_old.const_array(mfi);
                amrex::Array4<amrex::Real const> const & rr = rho.const_array(mfi);
                reduce_op.eval(tile_box, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                    {
                        amrex::Real const r = amrex::max(rr(i,j,k), 0.0_rt);
                        return {r*to(i,j,k), r*tn(i,j,k)};
                    });
            }
            auto tup = reduce_data.value(reduce_op);
            amrex::Real s_old = amrex::get<0>(tup);
            amrex::Real s_new = amrex::get<1>(tup);
            amrex::ParallelDescriptor::ReduceRealSum(s_old);
            amrex::ParallelDescriptor::ReduceRealSum(s_new);
            if (s_new > 0.0_rt) {
                T_new.mult(s_old/s_new, 0, 1, 0);
            }
        }

        amrex::MultiFab::Copy(Te, T_new, 0, 0, 1, 0);
        ApplyQdsmcConductionWallBCs(lev, dt_c, Te, rho);
        ablastr::utils::communication::FillBoundary(
            Te, WarpX::do_single_precision_comms, period, true);
        return;
    }

    // --- Flux-form (Esirkepov) conservative semi-Lagrangian remap --------
    // Production path (plan doc C.7d). Per GH branch the update is a
    // dimensionally split sequence of 1D conservative remaps on the node
    // dual cells: each face flux integrates the MC-limited PLM
    // reconstruction of u^n over the interval swept through the face by
    // the backward-traced branch displacement (two Picard iterations on
    // the linearly-interpolated node displacement field). The update
    // u_i -= (F_hi - F_lo) telescopes, so conservation is exact by
    // construction no matter what F is; the limited reconstruction makes
    // every branch remap never-add monotone, and the convex GH
    // combination of branches preserves the bound. The displacement field
    // interpolates continuously between nodes, so the source-evaluated
    // pushforward error of the deposit kernels (C.7c) is absent. Both
    // face fluxes of a node are recomputed locally from the same data
    // (deterministic, so the neighbor gets bit-identical values): no face
    // scratch, no atomics, race-free -> OMP-threaded. Wall faces carry
    // zero flux = exact adiabatic default (replaces the E7 clamp;
    // Thrust D adds tallies/isothermal variants on the same faces).
    if (m_cond_form == 2)
    {
        // The sweeps clamp the per-axis displacement at
        // min(max_hop, ff_rmax) cells -- a fixed safety independent of the
        // qdsmc_conduction_max_hop knob, so decks that set max_hop large to
        // keep the soft-min chi cap out of the physics (e.g. the wiggle
        // instrument) still get bounded stencils. Ghost reach: the
        // fold-guard window looks back 2*ceil(rhop) faces, each with a
        // backward trace of up to rhop cells plus the interp stencil --
        // 3*ff_rmax+4 covers every read below.
        int constexpr ff_rmax = 6;
        int const ng_f = 3*ff_rmax + 4;
        bool const slim_off = (m_cond_slope_limiter == 1);
        bool const ff_unsplit = m_cond_ff_unsplit;
        bool const recon_ppm = (m_cond_reconstruction == 1);
        bool const closed_ff = m_cond_closed_floor_faces;

        // Staircase EB conduction BC (per-sweep-line form, see the
        // m_cond_eb_bc member doc): covered nodes (nodal level set
        // <= 0) join the sweep mask.
        amrex::MultiFab const * eb_dist = EB::enabled()
            ? warpx.m_fields.get(FieldType::distance_to_eb, lev)
            : nullptr;
        bool const has_eb  = (eb_dist != nullptr);
        bool const eb_adia = has_eb && (m_cond_eb_bc == 0);
        bool const eb_iso  = has_eb && (m_cond_eb_bc == 1);
        auto const plo_arr = geom.ProbLoArray();
        auto const ebTe = m_cond_eb_Te;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!(ff_unsplit && has_eb),
            "qdsmc_conduction_fluxform_unsplit is a periodic-instrument "
            "control arm and has no EB wall handling; use the split "
            "sweeps with embedded boundaries");

        // Per-node branch kinematics: hop-clamped Ito drift (physical
        // axes), unit b, and the two quadrature sigmas. e1/e2 are rebuilt
        // from b wherever needed (deterministic), keeping the branch
        // displacement field single-valued across kernels.
        enum Kin : int { k_ax = 0, k_ay, k_az, k_bx, k_by, k_bz,
                         k_sp, k_sq, k_msk, k_ebm, k_ncomp };
        amrex::MultiFab kin(Te.boxArray(), Te.DistributionMap(),
                            Kin::k_ncomp, ng_f);
        kin.setVal(0.0_rt);
        // masks default OPEN so untouched domain ghosts never read as
        // walls (FillBoundary does not fill non-periodic domain ghosts)
        kin.setVal(1.0_rt, Kin::k_msk, 1, ng_f);
        kin.setVal(1.0_rt, Kin::k_ebm, 1, ng_f);

        // Quarantine/contamination tally (instrument; see the member doc):
        // class mask (1 = OPEN: n > boundary) with ghosts, and the per-face
        // crossing accumulator (components 0..SPACEDIM-1 = QUAR->OPEN flux
        // by sweep axis, 3 = the floored fast-front-donor subset). Ghosts
        // default OPEN; wall faces carry zero flux so unset physical ghosts
        // never contribute.
        bool const contam_on = (m_contam_n_boundary > 0.0_rt) && !ff_unsplit;
        amrex::MultiFab cls_mf, contam_mf;
        if (contam_on) {
            cls_mf.define(Te.boxArray(), Te.DistributionMap(), 1, ng_f);
            cls_mf.setVal(1.0_rt);
            contam_mf.define(Te.boxArray(), Te.DistributionMap(), 4, 0);
            contam_mf.setVal(0.0_rt);
            amrex::Real const n_bnd = m_contam_n_boundary;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(cls_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Box const box = mfi.tilebox();
                amrex::Array4<amrex::Real>       const & cl_arr  = cls_mf.array(mfi);
                amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);
                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    cl_arr(i,j,k) = (rho_arr(i,j,k)/qe > n_bnd) ? 1.0_rt : 0.0_rt;
                });
            }
            cls_mf.FillBoundary(period);
        }

        // Mask pass first (floored + EB-covered), then a ghost fill, so
        // the drift pass below can clamp its stencils one-sided at mask
        // boundaries (the D/grad-ln-n cliff at floors and covered bands
        // is the measured drift-kick class of the ring-test record).
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(kin, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box const box = mfi.tilebox();
            amrex::Array4<amrex::Real>       const & kn      = kin.array(mfi);
            amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);
            if (has_eb) {
                amrex::Array4<amrex::Real const> const & phi =
                    eb_dist->const_array(mfi);
                amrex::ParallelFor(box,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    kn(i,j,k,Kin::k_msk) =
                        (rho_arr(i,j,k)/qe > n_floor) ? 1.0_rt : 0.0_rt;
                    kn(i,j,k,Kin::k_ebm) =
                        (phi(i,j,k) > 0.0_rt) ? 1.0_rt : 0.0_rt;
                });
            } else {
                amrex::ParallelFor(box,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    kn(i,j,k,Kin::k_msk) =
                        (rho_arr(i,j,k)/qe > n_floor) ? 1.0_rt : 0.0_rt;
                });
            }
        }
        ablastr::utils::communication::FillBoundary(
            kin, WarpX::do_single_precision_comms, period, true);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(kin, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box const box = mfi.tilebox();
            amrex::Array4<amrex::Real>       const & kn      = kin.array(mfi);
            amrex::Array4<amrex::Real const> const & c_arr   = cond.const_array(mfi);
            amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                int const node[3] = {i, j, k};
                // Ito drift a = div D + D . grad ln n_e -- the same
                // clamped central differences and node-evaluated pairing
                // as the scatter/layer forms (the drift already carries
                // the mean curvature; do not re-rotate, trap 13).
                amrex::Real dD[3][3];
                amrex::Real gln[3];
                {
                    int const drow[3][3] = {
                        {Cond::c_dxx, Cond::c_dxy, Cond::c_dxz},
                        {Cond::c_dxy, Cond::c_dyy, Cond::c_dyz},
                        {Cond::c_dxz, Cond::c_dyz, Cond::c_dzz}};
                    amrex::Real const rho_fl =
                        amrex::max(rho_arr(i,j,k), n_floor*qe);
                    for (int ax = 0; ax < 3; ++ax) {
                        int const g = ax2gd[ax];
                        if (g < 0) {
                            for (int r = 0; r < 3; ++r) { dD[r][ax] = 0.0_rt; }
                            gln[ax] = 0.0_rt;
                            continue;
                        }
                        int cp = node[g] + 1, cm = node[g] - 1;
                        if (!is_per[g]) {
                            cp = amrex::min(cp, dom_hi[g]);
                            cm = amrex::max(cm, dom_lo[g]);
                        }
                        // one-sided at mask boundaries: the parser chi
                        // at floored/covered nodes is not physical, and
                        // the cliff drift its central difference implies
                        // is the measured wall-kick class (ring-test
                        // record). Masks are stable pre-pass data.
                        {
                            int pmc[3] = {i, j, k};
                            pmc[g] = cp;
                            if (kn(pmc[0],pmc[1],pmc[2],Kin::k_msk) == 0.0_rt ||
                                kn(pmc[0],pmc[1],pmc[2],Kin::k_ebm) == 0.0_rt) {
                                cp = node[g];
                            }
                            pmc[g] = cm;
                            if (kn(pmc[0],pmc[1],pmc[2],Kin::k_msk) == 0.0_rt ||
                                kn(pmc[0],pmc[1],pmc[2],Kin::k_ebm) == 0.0_rt) {
                                cm = node[g];
                            }
                        }
                        if (cp == cm) {
                            for (int r = 0; r < 3; ++r) { dD[r][ax] = 0.0_rt; }
                            gln[ax] = 0.0_rt;
                            continue;
                        }
                        int p[3] = {i, j, k}, m[3] = {i, j, k};
                        p[g] = cp; m[g] = cm;
                        amrex::Real const fac = dxi3[ax] / amrex::Real(cp - cm);
                        for (int r = 0; r < 3; ++r) {
                            dD[r][ax] = (c_arr(p[0],p[1],p[2],drow[r][ax])
                                       - c_arr(m[0],m[1],m[2],drow[r][ax])) * fac;
                        }
                        gln[ax] = (rho_arr(p[0],p[1],p[2])
                                 - rho_arr(m[0],m[1],m[2])) * fac / rho_fl;
                    }
                }
                amrex::Real const Dxx = c_arr(i,j,k,Cond::c_dxx);
                amrex::Real const Dxy = c_arr(i,j,k,Cond::c_dxy);
                amrex::Real const Dxz = c_arr(i,j,k,Cond::c_dxz);
                amrex::Real const Dyy = c_arr(i,j,k,Cond::c_dyy);
                amrex::Real const Dyz = c_arr(i,j,k,Cond::c_dyz);
                amrex::Real const Dzz = c_arr(i,j,k,Cond::c_dzz);
                amrex::Real drift[3];
                drift[0] = (dD[0][0] + dD[0][1] + dD[0][2]
                            + Dxx*gln[0] + Dxy*gln[1] + Dxz*gln[2]) * dt_c;
                drift[1] = (dD[1][0] + dD[1][1] + dD[1][2]
                            + Dxy*gln[0] + Dyy*gln[1] + Dyz*gln[2]) * dt_c;
                drift[2] = (dD[2][0] + dD[2][1] + dD[2][2]
                            + Dxz*gln[0] + Dyz*gln[1] + Dzz*gln[2]) * dt_c;
                for (int ax = 0; ax < 3; ++ax) {
                    drift[ax] = amrex::min(amrex::max(drift[ax], -hop), hop);
                }
                kn(i,j,k,Kin::k_ax) = drift[0];
                kn(i,j,k,Kin::k_ay) = drift[1];
                kn(i,j,k,Kin::k_az) = drift[2];
                kn(i,j,k,Kin::k_bx) = c_arr(i,j,k,Cond::c_bx);
                kn(i,j,k,Kin::k_by) = c_arr(i,j,k,Cond::c_by);
                kn(i,j,k,Kin::k_bz) = c_arr(i,j,k,Cond::c_bz);
                kn(i,j,k,Kin::k_sp) =
                    std::sqrt(2.0_rt*c_arr(i,j,k,Cond::c_chip)*dt_c);
                kn(i,j,k,Kin::k_sq) =
                    std::sqrt(2.0_rt*c_arr(i,j,k,Cond::c_chiq)*dt_c);
            });
        }
        ablastr::utils::communication::FillBoundary(
            kin, WarpX::do_single_precision_comms, period, true);

        // u^n = 3/2 n_e k_B T_e (zero where n_e <= 0, as in the scatter
        // spawn skip); every branch remap restarts from this frozen copy.
        amrex::MultiFab u0(Te.boxArray(), Te.DistributionMap(), 1, ng_f);
        u0.setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(u0, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box const box = mfi.tilebox();
            amrex::Array4<amrex::Real>       const & ua      = u0.array(mfi);
            amrex::Array4<amrex::Real const> const & Te_arr  = Te.const_array(mfi);
            amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real const ne = amrex::max(rho_arr(i,j,k)/qe, 0.0_rt);
                ua(i,j,k) = 1.5_rt * kb * ne * Te_arr(i,j,k);
            });
        }
        ablastr::utils::communication::FillBoundary(
            u0, WarpX::do_single_precision_comms, period, true);

        amrex::MultiFab u_a(Te.boxArray(), Te.DistributionMap(), 1, ng_f);
        amrex::MultiFab u_b(Te.boxArray(), Te.DistributionMap(), 1, ng_f);
        u_b.setVal(0.0_rt);
        amrex::MultiFab u_acc(Te.boxArray(), Te.DistributionMap(), 1, 0);
        u_acc.setVal(0.0_rt);

        // Alternate the split order per substep call (the two Strang
        // halves get opposite orders; euler's single Lie call alternates
        // by step) so the O(dt) splitting cross term does not bias one
        // diagonal.
        int const parity = (warpx.getistep(lev) + (use_rho_new ? 1 : 0)) % 2;
        amrex::GpuArray<int, 3> order_arr = {0, 0, 0};
        for (int pass = 0; pass < AMREX_SPACEDIM; ++pass) {
            order_arr[pass] = (parity == 0)
                ? pass : (AMREX_SPACEDIM - 1 - pass);
        }
        // per-grid-dim hop clamp in index units, bounded by ff_rmax
        amrex::GpuArray<amrex::Real, 3> rhop3 = {0.0_rt, 0.0_rt, 0.0_rt};
        for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
            rhop3[dd] = amrex::min(hop * dxi_arr[dd],
                                   amrex::Real(ff_rmax));
        }

        // Per-branch displacement field (grid-dim components, index
        // units, hop-clamped per axis) -- rebuilt per branch below.
        amrex::MultiFab dsp(Te.boxArray(), Te.DistributionMap(),
                            AMREX_SPACEDIM, ng_f);

        // Branch-lattice collapse: a quadrature direction whose offsets
        // cannot move any sweep -- zero sigma everywhere, or (2D XZ with
        // b in-plane everywhere, so e1 = yhat exactly) no grid projection
        // -- produces the same remap at every abscissa; run it once at
        // weight 1 instead (the GH weights sum to 1). This is the global
        // analogue of the scatter form's per-node loop collapse and is
        // what keeps the aligned/in-plane production decks at a few
        // branches instead of npts^3.
        amrex::Real const spmax = kin.max(Kin::k_sp, 0);
        amrex::Real const sqmax = kin.max(Kin::k_sq, 0);
        int const n0_eff = (spmax > 0.0_rt) ? nq_par : 1;

        // Isotropized launch: second, 45-degree-rotated perp lattice at
        // half variance (see the scatter form). The rotated lattice never
        // takes the 2D in-plane-b collapse: its directions project into
        // the plane by construction.
        int const n_lat = (iso_launch && sqmax > 0.0_rt) ? 2 : 1;
        // full sigma per lattice; see the scatter-form note
        amrex::Real const sq_fac = 1.0_rt;
        amrex::Real const w_lat_ff = (n_lat == 2) ? 0.5_rt : 1.0_rt;

        for (int lat = 0; lat < n_lat; ++lat) {
        int n1_eff = (sqmax > 0.0_rt) ? nq_perp : 1;
        int const n2_eff = (sqmax > 0.0_rt) ? nq_perp : 1;
#if !defined(WARPX_DIM_3D) && (AMREX_SPACEDIM == 2)
        if (lat == 0 && kin.norm0(Kin::k_by, 0, 0) == 0.0_rt) { n1_eff = 1; }
#endif
        bool const lat_rot = (lat == 1);

        for (int q0 = 0; q0 < n0_eff; ++q0) {
        for (int q1 = 0; q1 < n1_eff; ++q1) {
        for (int q2 = 0; q2 < n2_eff; ++q2) {
            // Offsets of non-collapsed directions with no grid projection
            // at some node drop out of the sweeps there automatically.
            amrex::Real const xq0 = (n0_eff > 1) ? xq_par[q0]  : 0.0_rt;
            amrex::Real const xq1 = (n1_eff > 1) ? xq_perp[q1] : 0.0_rt;
            amrex::Real const xq2 = (n2_eff > 1) ? xq_perp[q2] : 0.0_rt;
            amrex::Real const w_branch = w_lat_ff
                * ((n0_eff > 1) ? wq_par[q0]  : 1.0_rt)
                * ((n1_eff > 1) ? wq_perp[q1] : 1.0_rt)
                * ((n2_eff > 1) ? wq_perp[q2] : 1.0_rt);

            // Build the branch displacement field at every node including
            // ghosts, straight from kin's filled ghosts (deterministic per
            // node, so no communication is needed; ghost values at
            // physical boundaries reduce to zero displacement and are
            // never read thanks to the wall clamps below).
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(dsp, false); mfi.isValid(); ++mfi)
            {
                amrex::Box const gbox = amrex::grow(mfi.validbox(), ng_f);
                amrex::Array4<amrex::Real>       const & dv = dsp.array(mfi);
                amrex::Array4<amrex::Real const> const & kn = kin.const_array(mfi);
                amrex::ParallelFor(gbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    // masked nodes (floored or EB-covered) carry no
                    // displacement: their cap-sized fast-front hops
                    // must not pollute the edge interpolation of
                    // adjacent fluid donors
                    if (kn(i,j,k,Kin::k_msk) == 0.0_rt ||
                        kn(i,j,k,Kin::k_ebm) == 0.0_rt) {
                        for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                            dv(i,j,k,dd) = 0.0_rt;
                        }
                        return;
                    }
                    amrex::Real const bxv = kn(i,j,k,Kin::k_bx);
                    amrex::Real const byv = kn(i,j,k,Kin::k_by);
                    amrex::Real const bzv = kn(i,j,k,Kin::k_bz);
                    amrex::Real e1x, e1y, e1z, e2x, e2y, e2z;
                    qdsmc_perp_frame(bxv, byv, bzv,
                                     e1x, e1y, e1z, e2x, e2y, e2z);
                    if (lat_rot) {
                        // rotated perp lattice (isotropized launch)
                        amrex::Real const sr = 0.7071067811865475_rt;
                        amrex::Real const g1x = (e1x + e2x)*sr;
                        amrex::Real const g1y = (e1y + e2y)*sr;
                        amrex::Real const g1z = (e1z + e2z)*sr;
                        e2x = (e2x - e1x)*sr;
                        e2y = (e2y - e1y)*sr;
                        e2z = (e2z - e1z)*sr;
                        e1x = g1x; e1y = g1y; e1z = g1z;
                    }
                    amrex::Real const sp = kn(i,j,k,Kin::k_sp);
                    amrex::Real const sq = kn(i,j,k,Kin::k_sq) * sq_fac;
                    amrex::Real const dphys[3] = {
                        kn(i,j,k,Kin::k_ax) + xq0*sp*bxv + xq1*sq*e1x + xq2*sq*e2x,
                        kn(i,j,k,Kin::k_ay) + xq0*sp*byv + xq1*sq*e1y + xq2*sq*e2y,
                        kn(i,j,k,Kin::k_az) + xq0*sp*bzv + xq1*sq*e1z + xq2*sq*e2z};
                    for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                        int ax = 2;
#if defined(WARPX_DIM_3D)
                        ax = dd;
#elif (AMREX_SPACEDIM == 2)
                        ax = (dd == 0) ? 0 : 2;
#endif
                        amrex::Real const del = dphys[ax] * dxi3[ax];
                        dv(i,j,k,dd) = amrex::min(
                            amrex::max(del, -rhop3[dd]), rhop3[dd]);
                    }
                });
            }

            // --- Unsplit donor transport (C.7d fork (b), CSLAM-style) ---
            // The split sweeps below remap the INTERMEDIATE field, losing
            // donor identity between sweeps -- the residual splitting
            // error measured as the curvature leak (GF3). Here each donor
            // dual cell is instead carried through ALL sweep legs with
            // its OWN displacement, edge-evaluated at the donor's
            // material centers of the earlier legs (no pull-back, no
            // intermediate reconstruction): the composed per-donor map is
            // the bilinear corner image, so cross/corner transport
            // happens within a single branch remap. Every leg keeps the
            // forward-donor construction (fold = point mass, positivity
            // unconditional, F = 0 exact at zero displacement), fluxes
            // stay face-local and deterministic from both neighbors
            // (telescoping-exact conservation, race-free, OMP-threaded),
            // and the per-leg piece bookkeeping is exact for the PLM
            // reconstruction (linear integrands, midpoint-exact), so the
            // flux update reproduces the per-donor composed deposit to
            // machine. All dims' fluxes act on the frozen u0 in one shot:
            // u_branch = u0 - sum_p div_p F^p (no sweep ping-pong).
            if (ff_unsplit)
            {
                // Donor windows scale with the measured branch
                // displacement, not the rhop clamp (decks park the chi
                // cap with a large max_hop; the actual hops are what
                // bound the reach). norm0 is a global reduction, so the
                // windows are identical on every rank (determinism).
                amrex::GpuArray<int, 3> W3 = {0, 0, 0};
                for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                    amrex::Real const wmax =
                        amrex::min(rhop3[dd], dsp.norm0(dd, 0));
                    W3[dd] = static_cast<int>(std::ceil(wmax)) + 1;
                }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(u_b, TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    amrex::Box const box = mfi.tilebox();
                    amrex::Array4<amrex::Real>       const & ud = u_b.array(mfi);
                    amrex::Array4<amrex::Real const> const & us = u0.const_array(mfi);
                    amrex::Array4<amrex::Real const> const & dv = dsp.const_array(mfi);
                    amrex::Array4<amrex::Real const> const & kmk = kin.const_array(mfi);

                    amrex::ParallelFor(box,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        int const node[3] = {i, j, k};

                        // multilinear read of a dsp component at the
                        // fractional index position q (grid dims)
                        auto dinterp = [&] (amrex::Real const * q, int const dc)
                        {
                            int ib[3] = {i, j, k};
                            amrex::Real fb[3] = {0.0_rt, 0.0_rt, 0.0_rt};
                            for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                                amrex::Real s = q[dd];
                                if (!is_per[dd]) {
                                    s = amrex::min(amrex::max(s,
                                        amrex::Real(dom_lo[dd])),
                                        amrex::Real(dom_hi[dd]));
                                }
                                auto const fl = std::floor(s);
                                ib[dd] = static_cast<int>(fl);
                                fb[dd] = s - fl;
                            }
                            amrex::Real acc = 0.0_rt;
                            for (int cn = 0; cn < (1 << AMREX_SPACEDIM); ++cn) {
                                int idx[3] = {ib[0], ib[1], ib[2]};
                                amrex::Real w = 1.0_rt;
                                for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                                    if (cn & (1 << dd)) { idx[dd] += 1; w *= fb[dd]; }
                                    else                { w *= 1.0_rt - fb[dd]; }
                                    if (!is_per[dd]) {
                                        idx[dd] = amrex::min(
                                            amrex::max(idx[dd], dom_lo[dd]),
                                            dom_hi[dd]);
                                    }
                                }
                                acc += w * dv(idx[0], idx[1], idx[2], dc);
                            }
                            return acc;
                        };

                        auto u_at = [&] (int const * c) {
                            int p[3] = {c[0], c[1], c[2]};
                            for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                                if (!is_per[dd]) {
                                    p[dd] = amrex::min(
                                        amrex::max(p[dd], dom_lo[dd]),
                                        dom_hi[dd]);
                                }
                            }
                            return us(p[0], p[1], p[2]);
                        };
                        // MC-limited PLM slope of u0 at donor c along
                        // grid dim dd (per index unit); one-sided wall
                        // stencils degenerate to zero via the clamp.
                        auto slope_at = [&] (int const * c, int const dd) {
                            int cp[3] = {c[0], c[1], c[2]};
                            int cm[3] = {c[0], c[1], c[2]};
                            cp[dd] += 1; cm[dd] -= 1;
                            amrex::Real const uc  = u_at(c);
                            amrex::Real const dfp = u_at(cp) - uc;
                            amrex::Real const dfm = uc - u_at(cm);
                            if (slim_off) {
                                return 0.5_rt*(dfp + dfm);
                            }
                            amrex::Real g = 0.0_rt;
                            if (dfp*dfm > 0.0_rt) {
                                amrex::Real const sgn =
                                    (dfp > 0.0_rt) ? 1.0_rt : -1.0_rt;
                                g = sgn*amrex::min(2.0_rt*std::abs(dfp),
                                                   2.0_rt*std::abs(dfm),
                                                   0.5_rt*std::abs(dfp + dfm));
                            }
                            return g;
                        };

                        // 1D affine leg map of donor D along grid dim
                        // dd, edges displaced at the material position
                        // mpos (earlier legs' piece centers): the SAME
                        // computation whether the leg is being fluxed
                        // (final) or walked through (prior), which is
                        // what makes the composed telescoping exact.
                        auto leg_map = [&] (int const * D,
                                            amrex::Real const * mpos,
                                            int const dd,
                                            amrex::Real & L, amrex::Real & R)
                        {
                            amrex::Real qe_[3] = {mpos[0], mpos[1], mpos[2]};
                            qe_[dd] = amrex::Real(D[dd]) - 0.5_rt;
                            L = qe_[dd] + dinterp(qe_, dd);
                            qe_[dd] = amrex::Real(D[dd]) + 0.5_rt;
                            R = qe_[dd] + dinterp(qe_, dd);
                            if (!is_per[dd]) {
                                // adiabatic: images stay inside
                                amrex::Real const blo =
                                    amrex::Real(dom_lo[dd]) - 0.5_rt
                                    + 1.0e-6_rt;
                                amrex::Real const bhi =
                                    amrex::Real(dom_hi[dd]) + 0.5_rt
                                    - 1.0e-6_rt;
                                L = amrex::min(amrex::max(L, blo), bhi);
                                R = amrex::min(amrex::max(R, blo), bhi);
                            }
                        };

                        // Composed forward-donor contribution of donor D
                        // to the flux through face fn+1/2 of pass p (dim
                        // e): walk the prior legs to find the donor's
                        // piece in the face's transverse columns (the
                        // piece selection reproduces those legs' flux
                        // telescoping exactly, fold rule included), then
                        // apply the 1D forward-donor mass-beyond rule on
                        // the final leg with the piece's marginal PLM
                        // line density.
                        auto donor_contrib = [&] (int const * D, int const p,
                                                  int const e, int const fn)
                        {
                            amrex::Real mpos[3] = {amrex::Real(D[0]),
                                                   amrex::Real(D[1]),
                                                   amrex::Real(D[2])};
                            amrex::Real fprod  = 1.0_rt;
                            amrex::Real du_off = 0.0_rt;
                            for (int qq = 0; qq < p; ++qq) {
                                int const eq = order_arr[qq];
                                amrex::Real L, R;
                                leg_map(D, mpos, eq, L, R);
                                amrex::Real const wid = R - L;
                                amrex::Real const clo =
                                    amrex::Real(node[eq]) - 0.5_rt;
                                amrex::Real const chi_c =
                                    amrex::Real(node[eq]) + 0.5_rt;
                                if (wid <= 1.0e-10_rt) {
                                    // folded donor: point mass; the
                                    // (lo, hi] membership matches the
                                    // telescoped (mid > sf) flux rule
                                    amrex::Real const mid = 0.5_rt*(L + R);
                                    if (!(mid > clo && mid <= chi_c)) {
                                        return 0.0_rt;
                                    }
                                    // piece = whole donor, center kept
                                } else {
                                    amrex::Real const a = amrex::max(L, clo);
                                    amrex::Real const b = amrex::min(R, chi_c);
                                    if (b <= a) { return 0.0_rt; }
                                    amrex::Real const xi1 =
                                        amrex::Real(D[eq]) - 0.5_rt
                                        + (a - L)/wid;
                                    amrex::Real const xi2 =
                                        amrex::Real(D[eq]) - 0.5_rt
                                        + (b - L)/wid;
                                    fprod *= (xi2 - xi1);
                                    amrex::Real const cq =
                                        0.5_rt*(xi1 + xi2)
                                        - amrex::Real(D[eq]);
                                    du_off += slope_at(D, eq) * cq;
                                    mpos[eq] = amrex::Real(D[eq]) + cq;
                                }
                            }
                            // final leg along e on the piece's marginal
                            // line density (mean mbar, slope c1)
                            amrex::Real L, R;
                            leg_map(D, mpos, e, L, R);
                            amrex::Real const mbar =
                                fprod * (u_at(D) + du_off);
                            amrex::Real const c1 = fprod * slope_at(D, e);
                            amrex::Real const sf = amrex::Real(fn) + 0.5_rt;
                            amrex::Real const wid = R - L;
                            amrex::Real mright;
                            if (wid <= 1.0e-10_rt) {
                                mright = (0.5_rt*(L + R) > sf)
                                    ? mbar : 0.0_rt;
                            } else if (sf <= L) {
                                mright = mbar;
                            } else if (sf >= R) {
                                mright = 0.0_rt;
                            } else {
                                amrex::Real const xs = amrex::Real(D[e])
                                    - 0.5_rt + (sf - L)/wid;
                                amrex::Real const dx_hi = 0.5_rt;
                                amrex::Real const dx_lo =
                                    xs - amrex::Real(D[e]);
                                mright = (dx_hi - dx_lo)*mbar
                                    + 0.5_rt*c1
                                    *(dx_hi*dx_hi - dx_lo*dx_lo);
                            }
                            if (D[e] >= fn + 1) { mright -= mbar; }
                            return mright;
                        };

                        // Flux through face fn+1/2 of pass p (dim e).
                        // Windows are face/column-centered (never node-
                        // centered), so both neighbors of a face run the
                        // identical donor loop and get bit-identical F.
                        auto face_flux = [&] (int const p, int const e,
                                              int const fn)
                        {
                            if (!is_per[e] &&
                                (fn < dom_lo[e] || fn >= dom_hi[e])) {
                                return 0.0_rt;   // wall face: adiabatic
                            }
                            if (closed_ff) {
                                // faces touching a floored node carry no
                                // flux (vacuum deletion class closed)
                                int plo[3] = {i, j, k};
                                int phi[3] = {i, j, k};
                                plo[e] = fn; phi[e] = fn + 1;
                                if (kmk(plo[0],plo[1],plo[2],Kin::k_msk)
                                        == 0.0_rt ||
                                    kmk(phi[0],phi[1],phi[2],Kin::k_msk)
                                        == 0.0_rt) {
                                    return 0.0_rt;
                                }
                            }
                            int wlo[3] = {node[0], node[1], node[2]};
                            int whi[3] = {node[0], node[1], node[2]};
                            wlo[e] = fn - W3[e];
                            whi[e] = fn + W3[e] + 1;
                            for (int qq = 0; qq < p; ++qq) {
                                int const eq = order_arr[qq];
                                wlo[eq] = node[eq] - W3[eq];
                                whi[eq] = node[eq] + W3[eq];
                            }
                            amrex::Real F = 0.0_rt;
                            int D[3];
                            for (D[2] = wlo[2]; D[2] <= whi[2]; ++D[2]) {
                            for (D[1] = wlo[1]; D[1] <= whi[1]; ++D[1]) {
                            for (D[0] = wlo[0]; D[0] <= whi[0]; ++D[0]) {
                                bool skip = false;
                                for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                                    if (!is_per[dd] &&
                                        (D[dd] < dom_lo[dd] ||
                                         D[dd] > dom_hi[dd])) {
                                        skip = true;
                                    }
                                }
                                if (!skip && closed_ff) {
                                    // floored donors carry no heat
                                    skip = (kmk(D[0],D[1],D[2],
                                                Kin::k_msk) == 0.0_rt);
                                }
                                if (!skip) {
                                    F += donor_contrib(D, p, e, fn);
                                }
                            }}}
                            return F;
                        };

                        amrex::Real du = 0.0_rt;
                        for (int p = 0; p < AMREX_SPACEDIM; ++p) {
                            int const e = order_arr[p];
                            du += face_flux(p, e, node[e])
                                - face_flux(p, e, node[e] - 1);
                        }
                        ud(i,j,k) = us(i,j,k) - du;
                    });
                }
                amrex::MultiFab::Saxpy(u_acc, w_branch, u_b, 0, 0, 1, 0);
                continue;   // next branch (skip the split sweeps below)
            }

            amrex::MultiFab::Copy(u_a, u0, 0, 0, 1, ng_f);
            amrex::MultiFab * src = &u_a;
            amrex::MultiFab * dst = &u_b;

            for (int pass = 0; pass < AMREX_SPACEDIM; ++pass)
            {
                int const d = order_arr[pass];
                amrex::Real const rhop = rhop3[d];

                if (pass > 0) {
                    // the previous sweep wrote valid nodes only
                    ablastr::utils::communication::FillBoundary(
                        *src, WarpX::do_single_precision_comms, period, true);
                }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(*dst, TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    amrex::Box const box = mfi.tilebox();
                    amrex::Array4<amrex::Real>       const & ud = dst->array(mfi);
                    amrex::Array4<amrex::Real const> const & us = src->const_array(mfi);
                    amrex::Array4<amrex::Real const> const & dv = dsp.const_array(mfi);
                    amrex::Array4<amrex::Real const> const & kmk = kin.const_array(mfi);

                    // Contamination tally I/O (default Array4 when off ->
                    // never indexed, contam_on gates every access).
                    amrex::Array4<amrex::Real const> cls_arr;
                    amrex::Array4<amrex::Real>       ct_arr;
                    if (contam_on) {
                        cls_arr = cls_mf.const_array(mfi);
                        ct_arr  = contam_mf.array(mfi);
                    }

                    amrex::ParallelFor(box,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        int const node[3] = {i, j, k};

                        auto clampd = [&] (int idx) {
                            if (!is_per[d]) {
                                idx = amrex::min(
                                    amrex::max(idx, dom_lo[d]), dom_hi[d]);
                            }
                            return idx;
                        };

                        // multilinear read of a dsp component at the
                        // fractional index position q (grid dims)
                        auto dinterp = [&] (amrex::Real const * q, int const dc)
                        {
                            int ib[3] = {i, j, k};
                            amrex::Real fb[3] = {0.0_rt, 0.0_rt, 0.0_rt};
                            for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                                amrex::Real s = q[dd];
                                if (!is_per[dd]) {
                                    s = amrex::min(amrex::max(s,
                                        amrex::Real(dom_lo[dd])),
                                        amrex::Real(dom_hi[dd]));
                                }
                                auto const fl = std::floor(s);
                                ib[dd] = static_cast<int>(fl);
                                fb[dd] = s - fl;
                            }
                            amrex::Real acc = 0.0_rt;
                            for (int cn = 0; cn < (1 << AMREX_SPACEDIM); ++cn) {
                                int idx[3] = {ib[0], ib[1], ib[2]};
                                amrex::Real w = 1.0_rt;
                                for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                                    if (cn & (1 << dd)) { idx[dd] += 1; w *= fb[dd]; }
                                    else                { w *= 1.0_rt - fb[dd]; }
                                    if (!is_per[dd]) {
                                        idx[dd] = amrex::min(
                                            amrex::max(idx[dd], dom_lo[dd]),
                                            dom_hi[dd]);
                                    }
                                }
                                acc += w * dv(idx[0], idx[1], idx[2], dc);
                            }
                            return acc;
                        };

                        // Pulled-back sweep-axis displacement at line
                        // coordinate s: sweeps after the first evaluate
                        // their component at the SOURCE point of the
                        // earlier sweeps (Picard backward trace per prior
                        // axis, most recent first), which restores the
                        // exact composed branch map through second order.
                        // Without the pull-back the second sweep
                        // re-introduces the source-vs-destination cross
                        // term -- measured as a dt-flat eps=0.2 wiggle
                        // leak of 5.7e-2 with a systematic 1e-2 adrift.
                        auto dsp_line = [&] (amrex::Real const s)
                        {
                            amrex::Real q[3] = {amrex::Real(node[0]),
                                                amrex::Real(node[1]),
                                                amrex::Real(node[2])};
                            q[d] = s;
                            for (int pp = pass - 1; pp >= 0; --pp) {
                                int const e = order_arr[pp];
                                amrex::Real xi = dinterp(q, e);
                                amrex::Real qp[3] = {q[0], q[1], q[2]};
                                qp[e] -= xi;
                                xi = dinterp(qp, e);
                                q[e] -= xi;
                            }
                            return dinterp(q, d);
                        };

                        auto u_at = [&] (int const c) {
                            int p[3] = {i, j, k};
                            p[d] = clampd(c);
                            return us(p[0], p[1], p[2]);
                        };
                        // EB staircase masks along the sweep line (see
                        // the m_cond_eb_bc member doc); short-circuit to
                        // the pre-EB path when no EB is present.
                        auto covered = [&] (int const c) {
                            if (!has_eb) { return false; }
                            int p[3] = {i, j, k};
                            p[d] = c;
                            return kmk(p[0],p[1],p[2],Kin::k_ebm) == 0.0_rt;
                        };
                        auto floored_ln = [&] (int const c) {
                            int p[3] = {i, j, k};
                            p[d] = c;
                            return kmk(p[0],p[1],p[2],Kin::k_msk) == 0.0_rt;
                        };
                        // faces closed by the floor rule: floored AND
                        // not EB-covered (covered nodes are the wall's
                        // business — the fold — not the floor's)
                        auto blocked = [&] (int const c) {
                            return floored_ln(c) && !covered(c);
                        };
                        // mask-aware stencil read from center cc:
                        // covered reads become a zero-gradient ghost.
                        // (A bath-DONOR formulation was measured broken:
                        // masked donors carry zero displacement, so a
                        // bath that never hops only absorbs — one-way
                        // drain toward Te = 0. The isothermal EB instead
                        // folds like the adiabatic wall and pins the
                        // wall-adjacent fluid ring after recovery, the
                        // slab-validated bath semantics.)
                        auto u_rd = [&] (int const c, int const cc) {
                            if (covered(c)) { return u_at(cc); }
                            return u_at(c);
                        };
                        // MC-limited PLM slope of u along the sweep axis
                        // (per index unit); one-sided wall stencils
                        // degenerate to zero slope via the index clamp.
                        // slim_off = unlimited central slopes (diagnostic
                        // control arm, see the knob doc).
                        auto slope_at = [&] (int const c) {
                            if (covered(c)) { return 0.0_rt; }
                            amrex::Real const uc  = u_at(c);
                            amrex::Real const dfp = u_rd(c + 1, c) - uc;
                            amrex::Real const dfm = uc - u_rd(c - 1, c);
                            if (slim_off) {
                                return 0.5_rt*(dfp + dfm);
                            }
                            amrex::Real g = 0.0_rt;
                            if (dfp*dfm > 0.0_rt) {
                                amrex::Real const sgn =
                                    (dfp > 0.0_rt) ? 1.0_rt : -1.0_rt;
                                g = sgn*amrex::min(2.0_rt*std::abs(dfp),
                                                   2.0_rt*std::abs(dfm),
                                                   0.5_rt*std::abs(dfp + dfm));
                            }
                            return g;
                        };

                        // PPM (Colella--Woodward 1984) parabola of donor
                        // c: face values from the limited cell
                        // differences (slope_at doubles as the CW
                        // delta_m; unlimited central when slim_off),
                        // then extremum collapse + overshoot pull-back
                        // (skipped when slim_off). Mean is uc exactly,
                        // so total-mass bookkeeping is untouched.
                        auto ppm_coefs = [&] (int const c,
                                              amrex::Real & uLf,
                                              amrex::Real & du6,
                                              amrex::Real & u6)
                        {
                            amrex::Real const um1 = u_rd(c - 1, c);
                            amrex::Real const uc  = u_at(c);
                            amrex::Real const up1 = u_rd(c + 1, c);
                            amrex::Real uR = uc + 0.5_rt*(up1 - uc)
                                - (slope_at(c + 1) - slope_at(c))
                                  /6.0_rt;
                            amrex::Real uL = um1 + 0.5_rt*(uc - um1)
                                - (slope_at(c) - slope_at(c - 1))
                                  /6.0_rt;
                            if (!slim_off) {
                                if ((uR - uc)*(uc - uL) <= 0.0_rt) {
                                    uL = uc; uR = uc;
                                } else {
                                    amrex::Real const du = uR - uL;
                                    amrex::Real const s6 = 6.0_rt
                                        *(uc - 0.5_rt*(uL + uR));
                                    if (du*s6 > du*du) {
                                        uL = 3.0_rt*uc - 2.0_rt*uR;
                                    } else if (-du*du > du*s6) {
                                        uR = 3.0_rt*uc - 2.0_rt*uL;
                                    }
                                }
                            }
                            uLf = uL;
                            u6  = 6.0_rt*(uc - 0.5_rt*(uL + uR));
                            du6 = (uR - uL) + u6;
                        };

                        // Flux through the face between nodes fn and fn+1
                        // (positive = mass toward +d), by FORWARD donor
                        // decomposition (the Esirkepov construction): each
                        // dual cell maps affinely by the face-interpolated
                        // displacement; its contribution is the mass of
                        // its image beyond the face minus its pre-move
                        // mass beyond the face. Both neighbors of a face
                        // sum the same donor window (deterministic), so
                        // fluxes are consistent and conservation
                        // telescopes exactly. A fold (map compression at
                        // a chi cliff, e.g. a kappa ~ Te^2.5 front) just
                        // degenerates a donor's image to a point: the
                        // mass lands where the map says and positivity is
                        // unconditional. A backward face trace fails
                        // exactly there -- its Picard iteration converges
                        // to the nearest root of the folded map and
                        // misses fast donors hopping past the face, which
                        // starves the front (measured: ZK front_exp -0.5
                        // and a hole dug below ambient at the front foot).
                        auto face_flux = [&] (int const fn) {
                            if (!is_per[d] &&
                                (fn < dom_lo[d] || fn >= dom_hi[d])) {
                                return 0.0_rt;   // wall face: adiabatic
                            }
                            if (has_eb &&
                                (covered(fn) || covered(fn + 1))) {
                                // EB wall face (adiabatic): the fold
                                // below returns everything, nothing
                                // crosses
                                return 0.0_rt;
                            }
                            if (closed_ff &&
                                (blocked(fn) || blocked(fn + 1))) {
                                // faces touching a floored (non-EB) node
                                // carry no flux (vacuum deletion class)
                                return 0.0_rt;
                            }
                            amrex::Real const sf = amrex::Real(fn) + 0.5_rt;
                            int const W =
                                static_cast<int>(std::ceil(rhop)) + 1;
                            // mass of donor c's affine image [L, R]
                            // beyond position y (PLM or PPM sub-cell
                            // distribution; degenerate image = point)
                            auto mass_beyond = [&] (int const c,
                                                    amrex::Real const L,
                                                    amrex::Real const R,
                                                    amrex::Real const y)
                            {
                                amrex::Real const wid = R - L;
                                if (wid <= 1.0e-10_rt) {
                                    // folded donor: point mass
                                    return (0.5_rt*(L + R) > y)
                                        ? u_at(c) : 0.0_rt;
                                }
                                if (y <= L) { return u_at(c); }
                                if (y >= R) { return 0.0_rt; }
                                // affine preimage of y in the donor;
                                // mass beyond = reconstruction integral
                                // over [xs, c+1/2]
                                amrex::Real const xs = amrex::Real(c)
                                    - 0.5_rt + (y - L)/wid;
                                if (recon_ppm) {
                                    amrex::Real uLf, du6, u6;
                                    ppm_coefs(c, uLf, du6, u6);
                                    amrex::Real const zs = xs
                                        - (amrex::Real(c) - 0.5_rt);
                                    return u_at(c)
                                        - (uLf*zs
                                           + 0.5_rt*du6*zs*zs
                                           - u6*zs*zs*zs/3.0_rt);
                                }
                                amrex::Real const dx_hi = 0.5_rt;
                                amrex::Real const dx_lo =
                                    xs - amrex::Real(c);
                                return (dx_hi - dx_lo)*u_at(c)
                                    + 0.5_rt*slope_at(c)
                                    *(dx_hi*dx_hi - dx_lo*dx_lo);
                            };
                            amrex::Real F = 0.0_rt;
                            amrex::Real dlo =
                                dsp_line(amrex::Real(fn - W) - 0.5_rt);
                            for (int c = fn - W; c <= fn + W + 1; ++c) {
                                amrex::Real const dhi =
                                    dsp_line(amrex::Real(c) + 0.5_rt);
                                bool skip = (!is_per[d] &&
                                    (c < dom_lo[d] || c > dom_hi[d]));
                                if (!skip && closed_ff) {
                                    // floored donors carry no heat (their
                                    // cap-sized fast-front hops must not
                                    // flux sub-floor u into the interior)
                                    skip = blocked(c);
                                }
                                if (!skip && has_eb) {
                                    // covered donors are the fold's
                                    // mirror image, not mass
                                    skip = covered(c);
                                }
                                if (!skip) {
                                    amrex::Real const L =
                                        amrex::Real(c) - 0.5_rt + dlo;
                                    amrex::Real const R =
                                        amrex::Real(c) + 0.5_rt + dhi;
                                    // Specular fold-back walls (method
                                    // of images): domain walls on
                                    // non-periodic dims, plus per-line
                                    // EB walls at the covered-mask
                                    // boundary (adiabatic EB only; the
                                    // isothermal bath keeps its faces
                                    // open). Sentinel walls at +-1e30
                                    // reduce the 3-term formula to the
                                    // plain mass-beyond bit-exactly.
                                    // The former hard clamp measured as
                                    // a wall contact resistance (slab
                                    // interior saw ~9.7/4.3 eV instead
                                    // of the pinned 15/5).
                                    amrex::Real wlo = -1.0e30_rt;
                                    amrex::Real whi =  1.0e30_rt;
                                    if (!is_per[d]) {
                                        wlo = amrex::Real(dom_lo[d])
                                              - 0.5_rt;
                                        whi = amrex::Real(dom_hi[d])
                                              + 0.5_rt;
                                    }
                                    if (has_eb) {
                                        for (int cw = fn; cw >= fn - W;
                                             --cw) {
                                            if (covered(cw)) {
                                                wlo = amrex::max(wlo,
                                                    amrex::Real(cw)
                                                    + 0.5_rt);
                                                break;
                                            }
                                        }
                                        for (int cw = fn + 1;
                                             cw <= fn + W + 1; ++cw) {
                                            if (covered(cw)) {
                                                whi = amrex::min(whi,
                                                    amrex::Real(cw)
                                                    - 0.5_rt);
                                                break;
                                            }
                                        }
                                    }
                                    amrex::Real mright =
                                        (mass_beyond(c, L, R, sf)
                                         - mass_beyond(c, L, R,
                                               2.0_rt*whi - sf))
                                        + (u_at(c)
                                           - mass_beyond(c, L, R,
                                                 2.0_rt*wlo - sf));
                                    if (c >= fn + 1) { mright -= u_at(c); }
                                    F += mright;
                                }
                                dlo = dhi;
                            }
                            return F;
                        };

                        amrex::Real const Fhi = face_flux(node[d]);
                        amrex::Real const Flo = face_flux(node[d] - 1);
                        ud(i,j,k) = us(i,j,k) - (Fhi - Flo);

                        // Contamination tally: the RIGHT face only (every
                        // interior face is exactly one node's right face,
                        // so each face is written once; box-seam copies
                        // are redundant and consistent -- the deterministic
                        // face recomputation guarantees bit-identical F --
                        // and sum_unique dedups them). Positive tally =
                        // energy crossing QUAR -> OPEN this pass.
                        if (contam_on) {
                            int pr[3] = {i, j, k};
                            pr[d] += 1;
                            amrex::Real const cl = cls_arr(i,j,k);
                            amrex::Real const cr = cls_arr(pr[0],pr[1],pr[2]);
                            if (cl != cr) {
                                amrex::Real const Fq =
                                    (cl == 0.0_rt) ? Fhi : -Fhi;
                                if (Fq > 0.0_rt) {
                                    ct_arr(i,j,k,d) += w_branch * Fq;
                                    // fast-front subset: the QUAR-side
                                    // node is floored while the vacuum
                                    // fast front is boosting its chi
                                    // (identically zero under
                                    // closed_floor_faces -- that zero IS
                                    // the S0->SK conduit verification)
                                    int const * pq =
                                        (cl == 0.0_rt) ? node : pr;
                                    if (vac_fast &&
                                        kmk(pq[0],pq[1],pq[2],Kin::k_msk)
                                            == 0.0_rt) {
                                        ct_arr(i,j,k,3) += w_branch * Fq;
                                    }
                                }
                            }
                        }
                    });
                }
                std::swap(src, dst);
            }
            amrex::MultiFab::Saxpy(u_acc, w_branch, *src, 0, 0, 1, 0);
        }}}
        }   // lat

        // Fold the per-face class crossings into the cumulative
        // contamination tallies [J] (unique-node sum; u is an energy
        // density, so face flux x nodal dual-cell volume = energy moved).
        if (contam_on) {
            auto const dxc = geom.CellSizeArray();
            amrex::Real dV = 1.0_rt;
            for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) { dV *= dxc[dd]; }
            for (int c = 0; c < AMREX_SPACEDIM; ++c) {
                m_contam_axis_J[c] +=
                    contam_mf.sum_unique(c, false, period) * dV;
            }
            m_contam_fast_front_J +=
                contam_mf.sum_unique(3, false, period) * dV;
            // (printed from ApplyQdsmcEnergySources, which runs every step
            // whether or not conduction is enabled)
        }

        // Recovery (scatter Pass-3 semantics): floored and EB-covered
        // nodes keep their previous T_e; dividing by the same n_e that
        // built u keeps the zero-displacement identity exact.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box const box = mfi.tilebox();
            amrex::Array4<amrex::Real>       const & Te_arr  = Te.array(mfi);
            amrex::Array4<amrex::Real const> const & u_arr   = u_acc.const_array(mfi);
            amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);
            amrex::Array4<amrex::Real const> const & kn      = kin.const_array(mfi);

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (has_eb && kn(i,j,k,Kin::k_ebm) == 0.0_rt) { return; }
                if (rho_arr(i,j,k) <= 0.0_rt) { return; }
                amrex::Real const ne = rho_arr(i,j,k) / qe;
                if (ne <= n_floor) { return; }
                Te_arr(i,j,k) = u_arr(i,j,k) / (1.5_rt * kb * ne);
            });
        }

        // Isothermal EB: pin the wall-adjacent fluid ring (every fluid
        // node with a covered face neighbor) to the parser T_wall at
        // its position — the per-line form of thermal-bath re-emission,
        // the same slab-validated semantics as the domain pins — and
        // tally the exchange (positive = energy into the plasma).
        if (eb_iso) {
            int const eb_ring = m_cond_eb_ring;
            amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
            amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;
            for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Box tile_box = mfi.tilebox();
                {   // unique node ownership (fixup-loop seam trim)
                    amrex::Box const box_nodes =
                        amrex::surroundingNodes(mfi.validbox());
                    for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                        if (tile_box.bigEnd(dd) == box_nodes.bigEnd(dd) &&
                            (box_nodes.bigEnd(dd) != dom_nodes.bigEnd(dd) ||
                             geom.isPeriodic(dd))) {
                            tile_box.growHi(dd, -1);
                        }
                    }
                }
                amrex::Array4<amrex::Real>       const & Te_arr = Te.array(mfi);
                amrex::Array4<amrex::Real const> const & rho_arr =
                    rho.const_array(mfi);
                amrex::Array4<amrex::Real const> const & kn =
                    kin.const_array(mfi);
                reduce_op.eval(tile_box, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    if (kn(i,j,k,Kin::k_ebm) == 0.0_rt) { return {0.0_rt}; }
                    amrex::Real const ne = rho_arr(i,j,k) / qe;
                    if (ne <= 0.0_rt) { return {0.0_rt}; }
                    // wall ring = any covered node within Chebyshev
                    // distance eb_ring. Depth 1 with face adjacency
                    // measured a big contact drop (staircase jags
                    // block tangent lines); diagonals halved it; the
                    // remaining non-convergent drain was the DEPOSITION
                    // DENSITY RAMP at the wall — the flux form moves
                    // u = n T, and across an unresolved n cliff the
                    // discrete Ito drift under-corrects, so u leaks
                    // toward the low-density band where the pins eat
                    // it (interior measured BELOW both bath
                    // temperatures). Default depth 2 puts the whole
                    // ramp inside the bath; the O(dx)-convergent
                    // geometric radius shift is the price.
                    bool ring = false;
                    int const node[3] = {i, j, k};
                    int const rr = eb_ring;
#if defined(WARPX_DIM_3D)
                    for (int ok = -rr; ok <= rr; ++ok) {
#else
                    int const ok = 0;
                    {
#endif
                    for (int oj = -rr; oj <= rr; ++oj) {
                    for (int oi = -rr; oi <= rr; ++oi) {
                        if (oi == 0 && oj == 0 && ok == 0) { continue; }
                        ring = ring ||
                            (kn(node[0]+oi, node[1]+oj, node[2]+ok,
                                Kin::k_ebm) == 0.0_rt);
                    }}}
                    if (!ring) { return {0.0_rt}; }
                    amrex::Real cx[3] = {0.0_rt, 0.0_rt, 0.0_rt};
                    for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                        cx[dd] = plo_arr[dd]
                            + amrex::Real(node[dd])*dx_arr[dd];
                    }
#if defined(WARPX_DIM_3D)
                    amrex::Real const Te_eV = ebTe(cx[0], cx[1], cx[2]);
#else
                    amrex::Real const Te_eV = ebTe(cx[0], 0.0_rt, cx[1]);
#endif
                    amrex::Real const TwK = Te_eV * qe / kb;
                    amrex::Real const du =
                        1.5_rt * kb * ne * (TwK - Te_arr(i,j,k));
                    Te_arr(i,j,k) = TwK;
                    return {du};
                });
            }
            auto tup = reduce_data.value(reduce_op);
            amrex::Real tly = amrex::get<0>(tup);
            amrex::ParallelDescriptor::ReduceRealSum(tly);
            m_cond_eb_tally += tly;
        }
        ApplyQdsmcConductionWallBCs(lev, dt_c, Te, rho);
        ablastr::utils::communication::FillBoundary(
            Te, WarpX::do_single_precision_comms, period, true);
        return;
    }

    // --- Pass 2: spawn quiet daughters per owned node and deposit u ---
    // NO OpenMP here: Gpu::Atomic::AddNoRet is a plain += on the host, and
    // neighboring tiles' deposit reaches overlap (same fab), so a threaded
    // loop would race (same reason DepositScalar is unthreaded).
    u_dep.setVal(0.0_rt);

    for (MFIter mfi(u_dep, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // Unique node ownership: same seam/periodic trim as the advection
        // markers (InitParticles), so every physical node spawns exactly
        // one daughter family and Sigma(u) is not double-counted.
        amrex::Box tile_box = mfi.tilebox();
        {
            amrex::Box const box_nodes = amrex::surroundingNodes(mfi.validbox());
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (tile_box.bigEnd(d) == box_nodes.bigEnd(d) &&
                    (box_nodes.bigEnd(d) != dom_nodes.bigEnd(d) ||
                     geom.isPeriodic(d))) {
                    tile_box.growHi(d, -1);
                }
            }
        }

        amrex::Array4<amrex::Real>       const & out     = u_dep.array(mfi);
        amrex::Array4<amrex::Real const> const & c_arr   = cond.const_array(mfi);
        amrex::Array4<amrex::Real const> const & Te_arr  = Te.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);

        amrex::ParallelFor(tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real const ne0 = rho_arr(i,j,k) / qe;
            if (ne0 <= 0.0_rt) { return; }   // nothing to carry; recovery skips too

            // u and its MC-limited slopes (per grid dim) at the source
            // node -- the B1 half-gradient correction for the daughter
            // deposit. One-sided clamping at non-periodic domain edges
            // degrades to a zero slope, as in SetK.
            amrex::Real const u0 = 1.5_rt * kb * ne0 * Te_arr(i,j,k);
            amrex::Real gu[AMREX_SPACEDIM];
            int const node[3] = {i, j, k};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                int cp = node[d] + 1, cm = node[d] - 1;
                if (!is_per[d]) {
                    cp = amrex::min(cp, dom_hi[d]);
                    cm = amrex::max(cm, dom_lo[d]);
                }
                int p[3] = {i, j, k}, m[3] = {i, j, k};
                p[d] = cp; m[d] = cm;
                amrex::Real const up = 1.5_rt * kb
                    * (rho_arr(p[0],p[1],p[2])/qe) * Te_arr(p[0],p[1],p[2]);
                amrex::Real const um = 1.5_rt * kb
                    * (rho_arr(m[0],m[1],m[2])/qe) * Te_arr(m[0],m[1],m[2]);
                amrex::Real const dfp = (cp > node[d]) ? (up - u0) : 0.0_rt;
                amrex::Real const dfm = (cm < node[d]) ? (u0 - um) : 0.0_rt;
                amrex::Real g = 0.0_rt;
                if (dfp*dfm > 0.0_rt) {
                    amrex::Real const s = (dfp > 0.0_rt) ? 1.0_rt : -1.0_rt;
                    g = s * amrex::min(2.0_rt*std::abs(dfp),
                                       2.0_rt*std::abs(dfm),
                                       0.5_rt*std::abs(dfp + dfm));
                }
                gu[d] = g * dxi_arr[d];
            }

            // Ito drift a = div D + D . grad ln n_e (physical components),
            // from clamped central differences of the D tensor and rho.
            amrex::Real dD[3][3];   // dD[row][axis] = d D_{row,ax} / d x_ax
            amrex::Real gln[3];     // grad ln n_e
            {
                int const drow[3][3] = {
                    {Cond::c_dxx, Cond::c_dxy, Cond::c_dxz},
                    {Cond::c_dxy, Cond::c_dyy, Cond::c_dyz},
                    {Cond::c_dxz, Cond::c_dyz, Cond::c_dzz}};
                amrex::Real const rho_fl =
                    amrex::max(rho_arr(i,j,k), n_floor*qe);
                for (int ax = 0; ax < 3; ++ax) {
                    int const g = ax2gd[ax];
                    if (g < 0) {
                        for (int r = 0; r < 3; ++r) { dD[r][ax] = 0.0_rt; }
                        gln[ax] = 0.0_rt;
                        continue;
                    }
                    int cp = node[g] + 1, cm = node[g] - 1;
                    if (!is_per[g]) {
                        cp = amrex::min(cp, dom_hi[g]);
                        cm = amrex::max(cm, dom_lo[g]);
                    }
                    if (cp == cm) {
                        for (int r = 0; r < 3; ++r) { dD[r][ax] = 0.0_rt; }
                        gln[ax] = 0.0_rt;
                        continue;
                    }
                    int p[3] = {i, j, k}, m[3] = {i, j, k};
                    p[g] = cp; m[g] = cm;
                    amrex::Real const fac = dxi3[ax] / amrex::Real(cp - cm);
                    for (int r = 0; r < 3; ++r) {
                        dD[r][ax] = (c_arr(p[0],p[1],p[2],drow[r][ax])
                                   - c_arr(m[0],m[1],m[2],drow[r][ax])) * fac;
                    }
                    gln[ax] = (rho_arr(p[0],p[1],p[2])
                             - rho_arr(m[0],m[1],m[2])) * fac / rho_fl;
                }
            }
            amrex::Real const Dxx = c_arr(i,j,k,Cond::c_dxx);
            amrex::Real const Dxy = c_arr(i,j,k,Cond::c_dxy);
            amrex::Real const Dxz = c_arr(i,j,k,Cond::c_dxz);
            amrex::Real const Dyy = c_arr(i,j,k,Cond::c_dyy);
            amrex::Real const Dyz = c_arr(i,j,k,Cond::c_dyz);
            amrex::Real const Dzz = c_arr(i,j,k,Cond::c_dzz);
            amrex::Real drift[3];
            drift[0] = (dD[0][0] + dD[0][1] + dD[0][2]
                        + Dxx*gln[0] + Dxy*gln[1] + Dxz*gln[2]) * dt_c;
            drift[1] = (dD[1][0] + dD[1][1] + dD[1][2]
                        + Dxy*gln[0] + Dyy*gln[1] + Dyz*gln[2]) * dt_c;
            drift[2] = (dD[2][0] + dD[2][1] + dD[2][2]
                        + Dxz*gln[0] + Dyz*gln[1] + Dzz*gln[2]) * dt_c;
            for (int ax = 0; ax < 3; ++ax) {
                drift[ax] = amrex::min(amrex::max(drift[ax], -hop), hop);
            }

            // Field-aligned quadrature frame: e1 is the perpendicular
            // direction nearest yhat (so it projects out of the plane, and
            // its daughter loop collapses, in the common 2D in-plane-b
            // case); e2 = b x e1.
            amrex::Real const bxv = c_arr(i,j,k,Cond::c_bx);
            amrex::Real const byv = c_arr(i,j,k,Cond::c_by);
            amrex::Real const bzv = c_arr(i,j,k,Cond::c_bz);
            amrex::Real e1x = -byv*bxv, e1y = 1.0_rt - byv*byv, e1z = -byv*bzv;
            amrex::Real e1n = e1x*e1x + e1y*e1y + e1z*e1z;
            if (e1n < 1.0e-12_rt) {   // b ~ yhat: build from xhat instead
                e1x = 1.0_rt - bxv*bxv; e1y = -bxv*byv; e1z = -bxv*bzv;
                e1n = e1x*e1x + e1y*e1y + e1z*e1z;
            }
            amrex::Real const e1i = 1.0_rt/std::sqrt(e1n);
            e1x *= e1i; e1y *= e1i; e1z *= e1i;
            amrex::Real const e2x = byv*e1z - bzv*e1y;
            amrex::Real const e2y = bzv*e1x - bxv*e1z;
            amrex::Real const e2z = bxv*e1y - byv*e1x;

            amrex::Real const chi_par  = c_arr(i,j,k,Cond::c_chip);
            amrex::Real const chi_perp = c_arr(i,j,k,Cond::c_chiq);
            amrex::Real const sig_par  = std::sqrt(2.0_rt*chi_par *dt_c);
            amrex::Real const sig_perp = std::sqrt(2.0_rt*chi_perp*dt_c);

            // Collapse quadrature loops whose offsets cannot move the
            // deposit: zero sigma, or a direction with no projection onto
            // the simulated axes (weights then sum to 1 at zero offset).
            auto proj_sq = [&] (amrex::Real vx, amrex::Real vy, amrex::Real vz)
            {
                amrex::Real s2 = 0.0_rt;
                for (int ax = 0; ax < 3; ++ax) {
                    amrex::Real const v = (ax == 0) ? vx : ((ax == 1) ? vy : vz);
                    if (ax2gd[ax] >= 0) { s2 += v*v; }
                }
                return s2;
            };
            int const n0 = (sig_par  > 0.0_rt &&
                            proj_sq(bxv, byv, bzv) > 1.0e-24_rt) ? nq_par  : 1;

            // Isotropized launch: a second, 45-degree-rotated perp lattice
            // at half variance (equal split cancels the cos(4 theta)
            // fourth-moment anisotropy of the axis-pair launch).
            int const n_lat = (iso_launch && sig_perp > 0.0_rt) ? 2 : 1;
            // Each lattice carries the FULL sigma: the lattices are
            // half-weight alternatives, so the weighted variance sums to
            // sigma^2 and the equal fourth-moment weights cancel the
            // cos(4 theta) anisotropy.
            amrex::Real const sig_q = sig_perp;
            amrex::Real const w_lat = (n_lat == 2) ? 0.5_rt : 1.0_rt;

            for (int lat = 0; lat < n_lat; ++lat) {
            amrex::Real p1x = e1x, p1y = e1y, p1z = e1z;
            amrex::Real p2x = e2x, p2y = e2y, p2z = e2z;
            if (lat == 1) {
                amrex::Real const s = 0.7071067811865475_rt;
                p1x = (e1x + e2x)*s; p1y = (e1y + e2y)*s; p1z = (e1z + e2z)*s;
                p2x = (e2x - e1x)*s; p2y = (e2y - e1y)*s; p2z = (e2z - e1z)*s;
            }
            int const n1 = (sig_q > 0.0_rt &&
                            proj_sq(p1x, p1y, p1z) > 1.0e-24_rt) ? nq_perp : 1;
            int const n2 = (sig_q > 0.0_rt &&
                            proj_sq(p2x, p2y, p2z) > 1.0e-24_rt) ? nq_perp : 1;

            for (int q0 = 0; q0 < n0; ++q0) {
            for (int q1 = 0; q1 < n1; ++q1) {
            for (int q2 = 0; q2 < n2; ++q2) {
                amrex::Real const o0 = (n0 > 1) ? xq_par[q0]*sig_par : 0.0_rt;
                amrex::Real const o1 = (n1 > 1) ? xq_perp[q1]*sig_q  : 0.0_rt;
                amrex::Real const o2 = (n2 > 1) ? xq_perp[q2]*sig_q  : 0.0_rt;
                amrex::Real const wq = w_lat
                                     * ((n0 > 1) ? wq_par[q0]  : 1.0_rt)
                                     * ((n1 > 1) ? wq_perp[q1] : 1.0_rt)
                                     * ((n2 > 1) ? wq_perp[q2] : 1.0_rt);
                amrex::Real const disp[3] = {
                    drift[0] + o0*bxv + o1*p1x + o2*p2x,
                    drift[1] + o0*byv + o1*p1y + o2*p2y,
                    drift[2] + o0*bzv + o1*p1z + o2*p2z};

                // Continuous destination in grid-index space, hard-clamped
                // to the deposit's guard reach and (E7-style, pending the
                // Thrust-D reflection BCs) just inside non-periodic domain
                // edges.
                amrex::Real s[AMREX_SPACEDIM];
                int i0[AMREX_SPACEDIM];
                amrex::Real fr[AMREX_SPACEDIM];
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    int ax = 2;   // 1D: grid dim 0 = z
#if defined(WARPX_DIM_3D)
                    ax = d;
#elif (AMREX_SPACEDIM == 2)
                    ax = (d == 0) ? 0 : 2;
#endif
                    amrex::Real ds = disp[ax] * dxi_arr[d];
                    amrex::Real const rmax =
                        amrex::Real(ng_dep - (dep_keys ? 2 : 1)) - 1.0e-6_rt;
                    ds = amrex::min(amrex::max(ds, -rmax), rmax);
                    s[d] = amrex::Real(node[d]) + ds;
                    if (!is_per[d]) {
                        s[d] = amrex::min(
                            amrex::max(s[d], amrex::Real(dom_lo[d])),
                            amrex::Real(dom_hi[d]) - 1.0e-6_rt);
                    }
                    auto const fl = std::floor(s[d]);
                    i0[d] = static_cast<int>(fl);
                    fr[d] = s[d] - fl;
                }

                // Keys deposit (4^d): partition of unity exact, first and
                // second moments vanish (quadratic reproduction) -- no
                // gradient correction needed or applied. Negative lobes:
                // not monotone. Non-periodic writes fold onto the wall
                // nodes (index clamp), which preserves conservation.
                if (dep_keys) {
                    amrex::Real wk[AMREX_SPACEDIM][4];
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        qdsmc_keys_w(fr[d], wk[d]);
                    }
                    auto widx = [&] (int const d, int idx) {
                        if (!is_per[d]) {
                            idx = amrex::min(amrex::max(idx, dom_lo[d]),
                                             dom_hi[d]);
                        }
                        return idx;
                    };
#if defined(WARPX_DIM_3D)
                    for (int kk = 0; kk < 4; ++kk) {
                    for (int jj = 0; jj < 4; ++jj) {
                    for (int ii = 0; ii < 4; ++ii) {
                        amrex::Real const w =
                            wk[0][ii]*wk[1][jj]*wk[2][kk];
                        amrex::Gpu::Atomic::AddNoRet(
                            &out(widx(0, i0[0]+ii-1), widx(1, i0[1]+jj-1),
                                 widx(2, i0[2]+kk-1)), wq*w*u0);
                    }}}
#elif (AMREX_SPACEDIM == 2)
                    for (int jj = 0; jj < 4; ++jj) {
                    for (int ii = 0; ii < 4; ++ii) {
                        amrex::Real const w = wk[0][ii]*wk[1][jj];
                        amrex::Gpu::Atomic::AddNoRet(
                            &out(widx(0, i0[0]+ii-1), widx(1, i0[1]+jj-1),
                                 k), wq*w*u0);
                    }}
#else
                    for (int ii = 0; ii < 4; ++ii) {
                        amrex::Gpu::Atomic::AddNoRet(
                            &out(widx(0, i0[0]+ii-1), j, k),
                            wq*wk[0][ii]*u0);
                    }
#endif
                    continue;
                }

                // Hat deposit with the half-gradient correction
                // w * (u + 1/2 gu . (x_dest - x_daughter)): the correction
                // sums to zero (conservation exact) and cancels the hat's
                // second moment (remap exact through quadratics).
#if defined(WARPX_DIM_3D)
                for (int kk = 0; kk < 2; ++kk) {
                for (int jj = 0; jj < 2; ++jj) {
                for (int ii = 0; ii < 2; ++ii) {
                    amrex::Real const w = (ii ? fr[0] : 1.0_rt - fr[0])
                                        * (jj ? fr[1] : 1.0_rt - fr[1])
                                        * (kk ? fr[2] : 1.0_rt - fr[2]);
                    amrex::Real val = u0;
                    if (grad_dep) {
                        val += 0.5_rt * (
                            gu[0]*(amrex::Real(i0[0]+ii) - s[0])*dx_arr[0] +
                            gu[1]*(amrex::Real(i0[1]+jj) - s[1])*dx_arr[1] +
                            gu[2]*(amrex::Real(i0[2]+kk) - s[2])*dx_arr[2]);
                    }
                    amrex::Gpu::Atomic::AddNoRet(
                        &out(i0[0]+ii, i0[1]+jj, i0[2]+kk), wq*w*val);
                    if (compensate) {
                        for (int d = 0; d < 3; ++d) {
                            amrex::Real const frv =
                                fr[d]*(1.0_rt - fr[d])*dx_arr[d]*dx_arr[d];
                            amrex::Gpu::Atomic::AddNoRet(
                                &out(i0[0]+ii, i0[1]+jj, i0[2]+kk, 1 + d),
                                wq*w*u0*frv);
                        }
                    }
                }}}
#elif (AMREX_SPACEDIM == 2)
                for (int jj = 0; jj < 2; ++jj) {
                for (int ii = 0; ii < 2; ++ii) {
                    amrex::Real const w = (ii ? fr[0] : 1.0_rt - fr[0])
                                        * (jj ? fr[1] : 1.0_rt - fr[1]);
                    amrex::Real val = u0;
                    if (grad_dep) {
                        val += 0.5_rt * (
                            gu[0]*(amrex::Real(i0[0]+ii) - s[0])*dx_arr[0] +
                            gu[1]*(amrex::Real(i0[1]+jj) - s[1])*dx_arr[1]);
                    }
                    amrex::Gpu::Atomic::AddNoRet(
                        &out(i0[0]+ii, i0[1]+jj, k), wq*w*val);
                    if (compensate) {
                        for (int d = 0; d < 2; ++d) {
                            amrex::Real const frv =
                                fr[d]*(1.0_rt - fr[d])*dx_arr[d]*dx_arr[d];
                            amrex::Gpu::Atomic::AddNoRet(
                                &out(i0[0]+ii, i0[1]+jj, k, 1 + d),
                                wq*w*u0*frv);
                        }
                    }
                }}
#else
                for (int ii = 0; ii < 2; ++ii) {
                    amrex::Real const w = (ii ? fr[0] : 1.0_rt - fr[0]);
                    amrex::Real val = u0;
                    if (grad_dep) {
                        val += 0.5_rt *
                            gu[0]*(amrex::Real(i0[0]+ii) - s[0])*dx_arr[0];
                    }
                    amrex::Gpu::Atomic::AddNoRet(
                        &out(i0[0]+ii, j, k), wq*w*val);
                    if (compensate) {
                        amrex::Real const frv =
                            fr[0]*(1.0_rt - fr[0])*dx_arr[0]*dx_arr[0];
                        amrex::Gpu::Atomic::AddNoRet(
                            &out(i0[0]+ii, j, k, 1), wq*w*u0*frv);
                    }
                }
#endif
            }}}
            }   // lat
        });
    }

    amrex::Gpu::synchronize();
    ablastr::utils::communication::SumBoundary(
        u_dep, 0, nc_dep, u_dep.nGrowVect(), u_dep.nGrowVect(),
        WarpX::do_single_precision_comms, period);

    // --- Pass 2.5 (compensated deposit): Boris--Book FCT antidiffusion --
    // One limited antidiffusive flux sweep per axis. The raw flux between
    // nodes i and i+1 is Phi = vbar/(2 dx^2) * (u_{i+1} - u_i) with vbar =
    // (C_i + C_{i+1})/(u_i + u_{i+1}) the bookkept per-unit-u deposit
    // variance (<= dx^2/4, so the sweep is stable); the Boris--Book
    // limiter Phi~ = S max(0, min(|Phi|, S(u_{i+2}-u_{i+1}),
    // S(u_i-u_{i-1}))), S = sign(u_{i+1}-u_i), forbids new extrema
    // (positivity and monotonicity). Flux form => conservation exact.
    // In the unlimited smooth limit this composes with the hat to cancel
    // the kernel's second moment (Keys-equivalent through quadratics),
    // with destination-local gradients. Dimensionally split; each axis
    // sweep reads a frozen copy.
    if (compensate)
    {
        int const fct_lim = m_cond_fct_limiter;
        ablastr::utils::communication::FillBoundary(
            u_dep, WarpX::do_single_precision_comms, period, false);
        amrex::MultiFab u_fct(Te.boxArray(), Te.DistributionMap(), 1, 2);
        // Pre-deposit u field: the zalesak bounds include it, so
        // restoring a crest toward its pre-substep height is allowed
        // (it is not a new extremum).
        amrex::MultiFab u_pre(Te.boxArray(), Te.DistributionMap(), 1, 2);
        if (fct_lim == 1) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(u_pre, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Box const pbox = mfi.tilebox();
                amrex::Array4<amrex::Real>       const & up  = u_pre.array(mfi);
                amrex::Array4<amrex::Real const> const & Te_arr  = Te.const_array(mfi);
                amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);
                amrex::ParallelFor(pbox,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    up(i,j,k) = 1.5_rt*kb*(rho_arr(i,j,k)/qe)*Te_arr(i,j,k);
                });
            }
            ablastr::utils::communication::FillBoundary(
                u_pre, WarpX::do_single_precision_comms, period, false);
        }
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            // frozen per-axis copy; refresh ghosts each sweep (the
            // previous sweep updated valid values only)
            amrex::MultiFab::Copy(u_fct, u_dep, 0, 0, 1, 0);
            ablastr::utils::communication::FillBoundary(
                u_fct, WarpX::do_single_precision_comms, period, false);
            amrex::Real const dxi2 = dxi_arr[d]*dxi_arr[d];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(u_dep, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Box const fbox = mfi.tilebox();
                amrex::Array4<amrex::Real>       const & ua = u_dep.array(mfi);
                amrex::Array4<amrex::Real const> const & uf = u_fct.const_array(mfi);
                amrex::Array4<amrex::Real const> const & up = u_pre.const_array(mfi);

                amrex::ParallelFor(fbox,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    int const node[3] = {i, j, k};
                    auto pidx = [&] (int const off) {
                        int idx = node[d] + off;
                        if (!is_per[d]) {
                            idx = amrex::min(amrex::max(idx, dom_lo[d]),
                                             dom_hi[d]);
                        }
                        return idx;
                    };
                    auto uat = [&] (int const off) {
                        int p[3] = {i, j, k};
                        p[d] = pidx(off);
                        return uf(p[0], p[1], p[2]);
                    };
                    auto cat = [&] (int const off) {
                        int p[3] = {i, j, k};
                        p[d] = pidx(off);
                        return ua(p[0], p[1], p[2], 1 + d);
                    };
                    auto upre = [&] (int const off) {
                        int p[3] = {i, j, k};
                        p[d] = pidx(off);
                        return up(p[0], p[1], p[2]);
                    };
                    auto rawphi = [&] (int const lo) {
                        amrex::Real const u0v = uat(lo);
                        amrex::Real const u1v = uat(lo + 1);
                        amrex::Real const du = u1v - u0v;
                        if (du == 0.0_rt) { return 0.0_rt; }
                        amrex::Real const usum = u0v + u1v;
                        if (usum <= 0.0_rt) { return 0.0_rt; }
                        amrex::Real const vbar =
                            (cat(lo) + cat(lo + 1)) / usum;
                        return 0.5_rt*vbar*dxi2*du;
                    };
                    amrex::Real phiL = 0.0_rt, phiH = 0.0_rt;
                    if (fct_lim == 2) {          // unlimited (control)
                        phiH = rawphi(0);
                        phiL = rawphi(-1);
                    } else if (fct_lim == 0) {   // Boris--Book
                        auto bb = [&] (int const lo) {
                            amrex::Real const raw = rawphi(lo);
                            if (raw == 0.0_rt) { return 0.0_rt; }
                            amrex::Real const S =
                                (raw > 0.0_rt) ? 1.0_rt : -1.0_rt;
                            return S*amrex::max(0.0_rt,
                                amrex::min(std::abs(raw),
                                           S*(uat(lo + 2) - uat(lo + 1)),
                                           S*(uat(lo) - uat(lo - 1))));
                        };
                        phiH = bb(0);
                        phiL = bb(-1);
                    } else {                     // Zalesak + pre-deposit
                        auto rfac = [&] (int const off,
                                         amrex::Real & Rp, amrex::Real & Rm) {
                            amrex::Real const fl = rawphi(off - 1);
                            amrex::Real const fh = rawphi(off);
                            // update is u -= (phi_hi - phi_lo): inflow =
                            // max(fl,0) - min(fh,0), outflow the mirror
                            amrex::Real const Pp =
                                amrex::max(fl, 0.0_rt) - amrex::min(fh, 0.0_rt);
                            amrex::Real const Pm =
                                amrex::max(fh, 0.0_rt) - amrex::min(fl, 0.0_rt);
                            amrex::Real const utd = uat(off);
                            amrex::Real umax = utd, umin = utd;
                            for (int o = -1; o <= 1; ++o) {
                                umax = amrex::max(umax, uat(off + o),
                                                  upre(off + o));
                                umin = amrex::min(umin, uat(off + o),
                                                  upre(off + o));
                            }
                            Rp = (Pp > 0.0_rt)
                               ? amrex::min(1.0_rt, (umax - utd)/Pp) : 0.0_rt;
                            Rm = (Pm > 0.0_rt)
                               ? amrex::min(1.0_rt, (utd - umin)/Pm) : 0.0_rt;
                        };
                        auto zal = [&] (int const lo) {
                            amrex::Real const raw = rawphi(lo);
                            if (raw == 0.0_rt) { return 0.0_rt; }
                            amrex::Real Rp0, Rm0, Rp1, Rm1;
                            rfac(lo,     Rp0, Rm0);
                            rfac(lo + 1, Rp1, Rm1);
                            // raw > 0 removes from node lo (donor) and
                            // adds to node lo+1 (receiver)
                            amrex::Real const C = (raw > 0.0_rt)
                                ? amrex::min(Rp1, Rm0)
                                : amrex::min(Rp0, Rm1);
                            return C*raw;
                        };
                        phiH = zal(0);
                        phiL = zal(-1);
                    }
                    ua(i,j,k) = uf(i,j,k) - (phiH - phiL);
                });
            }
        }
    }

    // --- Pass 3: recover T_e = u / (3/2 k_B n_e) -----------------------
    // Floored cells keep their previous T_e (mirroring QDSMCUpdateTe);
    // dividing by the same n_e that built u makes chi = 0 the exact
    // identity.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Box const box = mfi.tilebox();
        amrex::Array4<amrex::Real>       const & Te_arr  = Te.array(mfi);
        amrex::Array4<amrex::Real const> const & u_arr   = u_dep.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (rho_arr(i,j,k) <= 0.0_rt) { return; }
            amrex::Real const ne = rho_arr(i,j,k) / qe;
            if (ne <= n_floor) { return; }
            Te_arr(i,j,k) = u_arr(i,j,k) / (1.5_rt * kb * ne);
        });
    }
    ApplyQdsmcConductionWallBCs(lev, dt_c, Te, rho);
    // The subsequent K_e init (and the E-solve's grad Pe) read T_e ghosts.
    ablastr::utils::communication::FillBoundary(
        Te, WarpX::do_single_precision_comms, period, true);
}


void HybridPICModel::SeedTeAdiabat (int const lev) const
{
    ABLASTR_PROFILE("HybridPICModel::SeedTeAdiabat()");

    auto & warpx = WarpX::GetInstance();

    amrex::MultiFab       & Te  = *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp, lev);

    auto const n0_ref    = m_n0_ref;
    auto const gamma     = m_gamma;
    auto const rho_floor = PhysConst::q_e * m_n_floor;
    // m_elec_temp is k_B T_e0 [J]; the T_e field is in K.
    auto const Te0_K     = m_elec_temp / PhysConst::kb;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Te, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real>       const & Te_arr  = Te.array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);

        amrex::Box const tbox = amrex::convert(mfi.tilebox(), Te.ixType().toIntVect());
        amrex::Box       box  = tbox;
        box.grow(Te.nGrowVect());

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real const ne =
                amrex::max(rho_arr(i,j,k), rho_floor) / PhysConst::q_e;
            Te_arr(i,j,k) = Te0_K * std::pow(ne / n0_ref, gamma - 1.0_rt);
        });
    }
}


void HybridPICModel::AdvanceElectronEnergyQDSMC (amrex::Real const dt) const
{
    ABLASTR_PROFILE("HybridPICModel::AdvanceElectronEnergyQDSMC()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_qdsmc_pc != nullptr,
        "AdvanceElectronEnergyQDSMC called with "
        "solve_electron_energy_equation=true but the "
        "QDSMC particle container was not constructed (InitData not run?)");

    // pc mode: the transport runs between the two B half-pushes
    // (AdvanceElectronEnergyQDSMC_PC); nothing happens at this call site.
    if (m_qdsmc_time_advance == QdsmcTimeAdvance::PC) { return; }

    auto & warpx = WarpX::GetInstance();

    // J_plasma (at B^n) is needed for V_e. On all but the first step it is
    // already valid: the previous step's final E-solve computed it from B^n
    // (the external-field subtract at the top of this step exactly cancels
    // the re-add at the end of the previous one), and B has not changed
    // since. Only the first step of a run or restart arrives here with an
    // unfilled J_plasma.
    if (!m_qdsmc_J_plasma_valid) {
        CalculatePlasmaCurrent(
            warpx.m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, warpx.finestLevel()),
            warpx.GetEBUpdateEFlag());
        m_qdsmc_J_plasma_valid = true;
    }

    if (m_qdsmc_time_advance == QdsmcTimeAdvance::Euler) {
        // #6982 scheme, kept bit-identical as the bake-off control:
        // forward-Euler push with V_e(J_i^{n-1/2}, B^n, rho^n) and Lie
        // (post-transport, full-dt) sources.
        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            QDSMCInitializeUe(lev, QdsmcUeMode::JiOld);
            QdsmcTransportOnce(lev, dt, /*midpoint=*/false);
            ApplyQdsmcEnergySources(lev, dt, /*fill_te_ghosts=*/false);
            // Lie conduction, matching euler's Lie sources (no-op unless
            // the kappa_par parser is set, keeping the control bit-identical
            // to #6982).
            QdsmcConductionOnce(lev, dt, /*use_rho_new=*/true);
            QDSMCFillElectronPressureFromTe(lev);
            // same Pe boundary treatment as the algebraic closure
            // (PR #7128), mirrored in every advance path
            warpx.ApplyElectronPressureBoundary(lev, PatchType::fine);
            ablastr::utils::communication::FillBoundary(
                *warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev),
                WarpX::do_single_precision_comms,
                warpx.Geom(lev).periodicity(),
                true);
            m_qdsmc_pc->ResetParticles(lev);
        }
        return;
    }

    // leapfrog: T_e lives at half-integer times; advance
    // T_e^{n-1/2} -> T_e^{n+1/2} with the time-centered
    // V_e^n(J_i^n, B^n, rho^n) (J_i^n = average of the two bracketing
    // half-integer deposits -- the main loop's own LinComb runs after this
    // call site), a midpoint push, and Strang-split sources (which are
    // themselves midpoint-valued here: J_plasma^n and the t^n ion
    // temperature deposit). The emitted Pe^{n+1/2} is exactly time-centered
    // for the full [n, n+1] B push. The first call self-starts with a dt/2
    // advance (T_e^0 -> T_e^{1/2}), mirroring the PIC velocity half-kick.
    amrex::Real const dt_adv = m_qdsmc_leapfrog_started ? dt : 0.5_rt * dt;

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        QDSMCInitializeUe(lev, QdsmcUeMode::JiAvg);

        // Strang bracket as in the pc driver (the rho pairing is
        // half-integer-staggered here, accepted for the non-default scheme).
        QdsmcConductionOnce(lev, 0.5_rt * dt_adv, /*use_rho_new=*/false);
        ApplyQdsmcEnergySources(lev, 0.5_rt * dt_adv, /*fill_te_ghosts=*/true);
        QdsmcTransportOnce(lev, dt_adv, /*midpoint=*/true);
        ApplyQdsmcEnergySources(lev, 0.5_rt * dt_adv, /*fill_te_ghosts=*/true);
        QdsmcConductionOnce(lev, 0.5_rt * dt_adv, /*use_rho_new=*/true);

        // Pe bookkeeping for the integer-time extrapolation (consumed by
        // ApplyQdsmcPeExtrapolation right before the final E-solve). The
        // previous Pe level must be captured before the register is
        // overwritten: on the first step the register still holds the
        // closure Pe^0 (HybridPICInitializeRhoJandB); afterwards pe_prev
        // already holds Pe^{n-1/2} from the previous advance.
        amrex::MultiFab & Pe      = *warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
        amrex::MultiFab & Pe_prev = *warpx.m_fields.get("hybrid_qdsmc_pe_prev_fp", lev);
        amrex::MultiFab & Pe_ext  = *warpx.m_fields.get("hybrid_qdsmc_pe_ext_fp",  lev);
        if (!m_qdsmc_pe_prev_valid) {
            amrex::MultiFab::Copy(Pe_prev, Pe, 0, 0, 1, Pe.nGrowVect());
        }

        QDSMCFillElectronPressureFromTe(lev);   // register <- Pe^{n+1/2}
        // same Pe boundary treatment as the algebraic closure (PR
        // #7128); the LinComb below is linear, so Pe_ext inherits it
        warpx.ApplyElectronPressureBoundary(lev, PatchType::fine);
        ablastr::utils::communication::FillBoundary(
            *warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev),
            WarpX::do_single_precision_comms,
            warpx.Geom(lev).periodicity(),
            true);

        // Linear extrapolation to t^{n+1}, mirroring the J_i^{n+1}
        // extrapolation: from (t^{n-1/2}, t^{n+1/2}) the coefficients are
        // (3/2, -1/2); on the first step the previous level is Pe^0 at t^0,
        // giving (2, -1).
        amrex::Real const c_new  = m_qdsmc_pe_prev_valid ? 1.5_rt  : 2.0_rt;
        amrex::Real const c_prev = m_qdsmc_pe_prev_valid ? -0.5_rt : -1.0_rt;
        amrex::MultiFab::LinComb(Pe_ext, c_new, Pe, 0, c_prev, Pe_prev, 0,
                                 0, 1, Pe.nGrowVect());
        amrex::MultiFab::Copy(Pe_prev, Pe, 0, 0, 1, Pe.nGrowVect());

        m_qdsmc_pc->ResetParticles(lev);
    }
    m_qdsmc_leapfrog_started = true;
    m_qdsmc_pe_prev_valid    = true;
}


std::array<amrex::Real, 2> HybridPICModel::QDSMCClassEnergy (int const lev) const
{
    // Class-summed thermal energy {bulk, band} [J] for the per-stage energy
    // budget: u = 3/2 n_e k_B T_e on the nodal grid, class-masked into a
    // temp MultiFab and reduced with sum_unique (box-seam nodes carry
    // redundant consistent copies). Fixed rho_fp weights: within one Strang
    // bracket rho does not change, so each stage's dU attributes that
    // stage's T_e change alone.
    auto & warpx = WarpX::GetInstance();
    amrex::MultiFab const & Te =
        *warpx.m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
    amrex::MultiFab const & rho = *warpx.m_fields.get(FieldType::rho_fp, lev);

    amrex::Real const n_flr = m_n_floor;
    amrex::Real const n_bnd = amrex::max(m_contam_n_boundary, m_n_floor);

    amrex::MultiFab u_cls(Te.boxArray(), Te.DistributionMap(), 2, 0);
    u_cls.setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(u_cls, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Box const box = mfi.tilebox();
        amrex::Array4<amrex::Real>       const & u_arr   = u_cls.array(mfi);
        amrex::Array4<amrex::Real const> const & te_arr  = Te.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rho_arr = rho.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real const ne = rho_arr(i,j,k) / PhysConst::q_e;
            if (ne <= n_flr) { return; }
            amrex::Real const u =
                1.5_rt * ne * PhysConst::kb * te_arr(i,j,k);
            u_arr(i,j,k, (ne > n_bnd) ? 0 : 1) = u;
        });
    }
    amrex::Periodicity const & period = warpx.Geom(lev).periodicity();
    auto const dxc = warpx.Geom(lev).CellSizeArray();
    amrex::Real dV = 1.0_rt;
    for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) { dV *= dxc[dd]; }
    return {u_cls.sum_unique(0, false, period) * dV,
            u_cls.sum_unique(1, false, period) * dV};
}


std::array<amrex::Real, 2> HybridPICModel::QDSMCCompressionEnergy (
    int const lev, amrex::MultiFab const & te_pre) const
{
    // Exact polytropic compression term of the transport-stage dU:
    //   du = 3/2 kB n^{n+1} Te_pre [ (ne_{n+1}/ne_n)^{gamma-1} - 1 ],
    // with both densities floored exactly as the K <-> Te conversions floor
    // them (QDSMCInitializeKe / QDSMCUpdateTe), so a marker that stays home
    // reproduces the recovery's Te change bit-consistently. Classes and
    // weights match QDSMCClassEnergy (fixed rho_fp).
    auto & warpx = WarpX::GetInstance();
    amrex::MultiFab const & rho_new =
        *warpx.m_fields.get(FieldType::rho_fp, lev);
    amrex::MultiFab const & rho_old =
        *warpx.m_fields.get(FieldType::hybrid_rho_fp_temp, lev);

    amrex::Real const n_flr = m_n_floor;
    amrex::Real const n_bnd = amrex::max(m_contam_n_boundary, m_n_floor);
    amrex::Real const gm1   = m_gamma - 1.0_rt;

    amrex::MultiFab u_cls(te_pre.boxArray(), te_pre.DistributionMap(), 2, 0);
    u_cls.setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(u_cls, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Box const box = mfi.tilebox();
        amrex::Array4<amrex::Real>       const & u_arr  = u_cls.array(mfi);
        amrex::Array4<amrex::Real const> const & te_arr = te_pre.const_array(mfi);
        amrex::Array4<amrex::Real const> const & rn_arr = rho_new.const_array(mfi);
        amrex::Array4<amrex::Real const> const & ro_arr = rho_old.const_array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real const ne = rn_arr(i,j,k) / PhysConst::q_e;
            if (ne <= n_flr) { return; }
            amrex::Real const ne_new_f = amrex::max(ne, n_flr);
            amrex::Real const ne_old_f =
                amrex::max(ro_arr(i,j,k) / PhysConst::q_e, n_flr);
            amrex::Real const du = 1.5_rt * ne * PhysConst::kb * te_arr(i,j,k)
                * (std::pow(ne_new_f / ne_old_f, gm1) - 1.0_rt);
            u_arr(i,j,k, (ne > n_bnd) ? 0 : 1) = du;
        });
    }
    amrex::Periodicity const & period = warpx.Geom(lev).periodicity();
    auto const dxc = warpx.Geom(lev).CellSizeArray();
    amrex::Real dV = 1.0_rt;
    for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) { dV *= dxc[dd]; }
    return {u_cls.sum_unique(0, false, period) * dV,
            u_cls.sum_unique(1, false, period) * dV};
}


void HybridPICModel::AdvanceElectronEnergyQDSMC_PC (amrex::Real const dt) const
{
    ABLASTR_PROFILE("HybridPICModel::AdvanceElectronEnergyQDSMC_PC()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_qdsmc_pc != nullptr,
        "AdvanceElectronEnergyQDSMC_PC called with "
        "solve_electron_energy_equation=true but the "
        "QDSMC particle container was not constructed (InitData not run?)");

    auto & warpx = WarpX::GetInstance();

    // Called between the two B half-pushes: the register holds B^{n+1/2},
    // so this yields J_plasma^{n+1/2} for both V_e and the Joule source.
    // (The final E-solve recomputes J_plasma from B^{n+1} afterwards.)
    CalculatePlasmaCurrent(
        warpx.m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, warpx.finestLevel()),
        warpx.GetEBUpdateEFlag());

    // Single transport T_e^n -> T_e^{n+1} with the mid-step velocity
    // V_e^{n+1/2}(J_i^{n+1/2}, B^{n+1/2}, rho^{n+1/2}), midpoint push and
    // Strang sources. Runs BEFORE the rho^{n+1/2} LinComb so
    // hybrid_rho_fp_temp still holds rho^n for the K_e / N_e load. The
    // first B half-push ran with the previous step's Pe^n (already in the
    // register); the Pe^{n+1} emitted here serves the second half-push and
    // the final E-solve.
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        QDSMCInitializeUe(lev, QdsmcUeMode::JiNewRhoHalf);

        // Strang: C(dt/2) . [S(dt/2) A(dt) S(dt/2)] . C(dt/2). The
        // conduction halves pair T_e with the rho of their time level
        // (rho^n before the transport, rho^{n+1} after); both no-op unless
        // the kappa_par parser is set. When the energy budget is armed,
        // bracket every stage with the class-summed thermal energy and
        // attribute each stage's dU (transport carries advection AND the
        // polytropic compression signal).
        bool const ebud = m_energy_budget;
        std::array<amrex::Real, 2> ub0{}, ub1{}, ub2{}, ub3{}, ub4{}, ub5{};
        if (ebud) { ub0 = QDSMCClassEnergy(lev); }
        QdsmcConductionOnce(lev, 0.5_rt * dt, /*use_rho_new=*/false);
        if (ebud) { ub1 = QDSMCClassEnergy(lev); }
        ApplyQdsmcEnergySources(lev, 0.5_rt * dt, /*fill_te_ghosts=*/true);
        if (ebud) { ub2 = QDSMCClassEnergy(lev); }
        // Pre-transport Te snapshot for the compression split of the
        // transport-stage dU (instrument only).
        amrex::MultiFab te_pre;
        if (ebud) {
            amrex::MultiFab const & Te_now = *warpx.m_fields.get(
                FieldType::hybrid_electron_temperature_fp, lev);
            te_pre.define(Te_now.boxArray(), Te_now.DistributionMap(), 1, 0);
            amrex::MultiFab::Copy(te_pre, Te_now, 0, 0, 1, 0);
        }
        QdsmcTransportOnce(lev, dt, /*midpoint=*/true);
        if (ebud) {
            ub3 = QDSMCClassEnergy(lev);
            auto const comp = QDSMCCompressionEnergy(lev, te_pre);
            m_ebud_comp_bulk += comp[0];
            m_ebud_comp_band += comp[1];
        }
        ApplyQdsmcEnergySources(lev, 0.5_rt * dt, /*fill_te_ghosts=*/true);
        if (ebud) { ub4 = QDSMCClassEnergy(lev); }
        QdsmcConductionOnce(lev, 0.5_rt * dt, /*use_rho_new=*/true);
        if (ebud) {
            ub5 = QDSMCClassEnergy(lev);
            m_ebud_cond_bulk += (ub1[0] - ub0[0]) + (ub5[0] - ub4[0]);
            m_ebud_cond_band += (ub1[1] - ub0[1]) + (ub5[1] - ub4[1]);
            m_ebud_src_bulk  += (ub2[0] - ub1[0]) + (ub4[0] - ub3[0]);
            m_ebud_src_band  += (ub2[1] - ub1[1]) + (ub4[1] - ub3[1]);
            m_ebud_adv_bulk  += ub3[0] - ub2[0];
            m_ebud_adv_band  += ub3[1] - ub2[1];
            if (m_joule_dropped_print_interval > 0 &&
                warpx.getistep(0) % m_joule_dropped_print_interval == 0) {
                amrex::Print() << "[qdsmc] step " << warpx.getistep(0)
                    << " energy_budget_J:"
                    << " adv_bulk="  << m_ebud_adv_bulk
                    << " comp_bulk=" << m_ebud_comp_bulk
                    << " cond_bulk=" << m_ebud_cond_bulk
                    << " src_bulk="  << m_ebud_src_bulk
                    << " adv_band="  << m_ebud_adv_band
                    << " comp_band=" << m_ebud_comp_band
                    << " cond_band=" << m_ebud_cond_band
                    << " src_band="  << m_ebud_src_band
                    << " U_bulk=" << ub5[0] << " U_band=" << ub5[1]
                    << " (dU cumulative; adv includes comp)\n";
            }
        }

        // Emit P_e = n_e * k_B * T_e for the downstream Ohm's-law
        // solve, with the same boundary treatment the algebraic closure gets
        // in CalculateElectronPressure (grad P_e reads the ghost cells;
        // PR #7128).
        QDSMCFillElectronPressureFromTe(lev);
        warpx.ApplyElectronPressureBoundary(lev, PatchType::fine);
        ablastr::utils::communication::FillBoundary(
            *warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev),
            WarpX::do_single_precision_comms,
            warpx.Geom(lev).periodicity(),
            true);

        // Reset particles to home positions (and zero velocity / weight /
        // entropy) so the next step starts with a clean grid.
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
            m_qdsmc_pc->DepositK(lev, Karr_out, /*gradient_corrected=*/false);
            m_qdsmc_pc->DepositField(lev, weights_out, /*gradient_corrected=*/false);
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
            m_qdsmc_pc->DepositK(lev, Karr_out, /*gradient_corrected=*/false);
            m_qdsmc_pc->DepositField(lev, weights_out, /*gradient_corrected=*/false);
            QDSMCUpdateTe(lev, rho_np1);

            // Step 5: full-step deterministic sources with midpoint
            // coefficient states.
            if (m_include_joule_heating) {
                QDSMCAddJouleHeating(lev, dt, rho_half, nullptr);
            }
            if (m_include_temperature_relaxation) {
                QDSMCAddTemperatureRelaxation(lev, dt, rho_half, m_qdsmc_Ti_by_name[lev]);
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
            m_qdsmc_pc->DepositK(lev, Karr_out, /*gradient_corrected=*/false);
            m_qdsmc_pc->DepositField(lev, weights_out, /*gradient_corrected=*/false);
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


void HybridPICModel::ApplyQdsmcPeExtrapolation () const
{
    if (m_qdsmc_time_advance != QdsmcTimeAdvance::Leapfrog) { return; }

    ABLASTR_PROFILE("HybridPICModel::ApplyQdsmcPeExtrapolation()");

    auto & warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        amrex::MultiFab       & Pe     = *warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
        amrex::MultiFab const & Pe_ext = *warpx.m_fields.get("hybrid_qdsmc_pe_ext_fp", lev);
        amrex::MultiFab::Copy(Pe, Pe_ext, 0, 0, 1, Pe.nGrowVect());
    }
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
    // Call FillBoundary if a collocated grid is used
    if (Bz_IndexType[0] == Ez_IndexType[0]) {
        warpx.FillBoundaryE(ng, nodal_sync);
    }

    // Push forward the B-field using Faraday's law
    warpx.EvolveB(dt, subcycling_half, t_old);
    warpx.FillBoundaryB(ng, nodal_sync);
}
