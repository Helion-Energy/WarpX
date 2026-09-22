/* Copyright 2026 Prabhat Kumar
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Fields.H"
#include "Circuit/CircuitCoupling.H"
#include "ThetaImplicitHybrid.H"
#include "DarwinVacuumERecovery.H"
#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Particles/MultiParticleContainer.H"
#include "Python/callbacks.H"
#include "WarpX.H"
#include <ablastr/utils/Communication.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

using warpx::fields::FieldType;
using namespace amrex::literals;

namespace {
// TEMPORARY diagnosis instrument (WARPX_EXT_LEDGER): valid-region sums of the
// Bz state and the stored external Bz at each strip/add choreography point.
void ExtLedgerPrint (WarpX* a_wx, const char* a_tag)
{
    if (std::getenv("WARPX_EXT_LEDGER") == nullptr) { return; }
    using ablastr::fields::Direction;
    const amrex::Real bsum =
        a_wx->m_fields.get(FieldType::Bfield_fp, Direction{2}, 0)->sum(0);
    amrex::Real esum = 0.0_rt;
    if (a_wx->m_fields.has(FieldType::hybrid_B_fp_external, Direction{2}, 0)) {
        esum = a_wx->m_fields.get(
            FieldType::hybrid_B_fp_external, Direction{2}, 0)->sum(0);
    }
    amrex::Print() << "EXT_LEDGER[" << a_tag << "] sumBz=" << bsum
                   << " sumBextz=" << esum << "\n";
}
}

void ThetaImplicitHybrid::Define ( WarpX* const a_WarpX, const bool a_from_restart )
{
    BL_PROFILE("ThetaImplicitHybrid::Define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_is_defined,
        "ThetaImplicitHybrid object is already defined!");

    m_WarpX = a_WarpX;
    m_num_amr_levels = 1;

    m_hybrid_pic_model = m_WarpX->get_pointer_HybridPICModel();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_pic_model != nullptr,
        "ThetaImplicitHybrid solver requires hybrid PIC model to be defined");

    m_darwin = m_hybrid_pic_model->m_darwin;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(m_darwin && a_from_restart),
        "hybrid_pic_model.darwin does not support restarts yet (the vector "
        "potential and static magnetic field are not checkpointed)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(m_hybrid_pic_model->m_include_electron_inertia && a_from_restart),
        "hybrid_pic_model.include_electron_inertia does not support "
        "restarts yet (the electron-current history is not checkpointed)");

    // Vacuum vector-potential recovery cadence (see HybridPICModel.H):
    // "half" applies inside every residual evaluation at the theta-stage
    // field and once at the end-of-step state; "full" end-of-step only.
    m_vacuum_recovery = m_darwin && m_hybrid_pic_model->m_darwin_vacuum_recovery;
    m_vacuum_recovery_half = m_vacuum_recovery
        && (m_hybrid_pic_model->m_darwin_vacuum_recovery_cadence == "half");

    // External vector-potential fields use the split-field convention here,
    // like the explicit scheme: the solver state carries the plasma fields
    // (OneStep strips B_ext/E_ext at entry; Bfield_fp holds totals between
    // steps for diagnostics), the external flux advances analytically from
    // A(t) so wall boundary conditions cannot exclude the programmed coil
    // flux, and the Ohm kernels subtract the inductive E_ext from plasma
    // cells (inside the plasma the generalized Ohm's law IS the electric
    // field). Clearing m_external_split only tells the kernels that the
    // Hall-term field they receive is already the total.
    if (m_hybrid_pic_model->m_add_external_fields) {
        if (m_darwin) {
            // Unified drive: the external vector potential enters through
            // the boundary values of the evolved A (DarwinApplyABoundary);
            // the kernels and the split-field machinery stay out of it.
            m_hybrid_pic_model->m_external_unified = true;
        } else {
            m_hybrid_pic_model->m_external_split = false;
        }
    }

#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
    if (m_vacuum_recovery) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_hybrid_pic_model->m_darwin_vacrec_relax_time == 0._rt,
            "Spatial vacuum electric recovery requires instantaneous magnetic recovery");
    }
#endif

    m_E.Define( m_WarpX, "Efield_fp" );
    m_Eold.Define( m_E );
    m_Eprev.Define( m_E );

    // Set initial values for E and Eold vectors
    m_E.Copy(FieldType::Efield_fp);
    m_Eold.Copy(a_from_restart ? FieldType::E_old : FieldType::Efield_fp, FieldType::None, true);

    // Define B_old MultiFabs
    using ablastr::fields::Direction;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const auto& Bfp_x = m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, lev);
        const auto& dm = Bfp_x->DistributionMap();
        const amrex::IntVect ngb = Bfp_x->nGrowVect();

        for (int dir = 0; dir < 3; ++dir) {
            const auto& ba = m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{dir}, lev)->boxArray();
            m_WarpX->m_fields.alloc_init(FieldType::B_old, Direction{dir}, lev, ba, dm, 1, ngb, 0.0_rt);
        }
    }

    // Scratch for the resistive push-field correction assembled in every
    // residual evaluation (see ComputeRHS). Only allocated when the opt-in
    // momentum-consistent push field is enabled and a resistive term is
    // configured.
    m_use_resistive_push_correction = m_hybrid_pic_model->HasResistivity()
        && m_hybrid_pic_model->m_implicit_push_excludes_resistive_field;
    if (m_use_resistive_push_correction) {
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                const auto& Efp = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
                m_WarpX->m_fields.alloc_init("hybrid_E_resistive_fp", Direction{dir}, lev,
                    Efp->boxArray(), Efp->DistributionMap(), Efp->nComp(),
                    Efp->nGrowVect(), 0.0_rt);
            }
        }
    }

    // curlcurl_form: the correction's per-evaluation Ohm-solve pair runs
    // BEFORE the inertia assembly that used to allocate the elliptic
    // scratch lazily, so allocate it up front (no-op on e_form). Stale
    // scratch content cancels exactly in the correction's two-pass
    // difference; unallocated arrays do not.
    m_hybrid_pic_model->EnsureCurlCurlScratch();

    const amrex::ParmParse pp("implicit_evolve");
    pp.query("theta", m_theta);
    pp.query("extrapolate_initial_guess", m_extrapolate_initial_guess);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_theta >= 0.5 && m_theta <= 1.0,
        "theta parameter must be between 0.5 and 1.0");

    {
        // Default: re-evaluate the generalized Ohm's law at the delivered
        // end-of-step state. The theta extrapolation of the ALGEBRAIC E is
        // a -(1-theta)/theta recursion on the stored field (marginal at
        // theta = 1/2) whose error grows linearly under a steady drift.
        std::string e_finisher = "reevaluate";
        const bool user_set = pp.query("hybrid_e_finisher", e_finisher);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            e_finisher == "extrapolate" || e_finisher == "reevaluate",
            "implicit_evolve.hybrid_e_finisher must be 'extrapolate' or "
            "'reevaluate'");
        m_e_finisher_reevaluate = (e_finisher == "reevaluate");
        if (m_darwin && m_e_finisher_reevaluate) {
            // Not implemented for the Darwin field split (E_L comes from
            // the ambipolar constraint, E_T from the vector potential);
            // fall back to the legacy extrapolation unless the user asked
            // for the re-evaluated finisher explicitly.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!user_set,
                "implicit_evolve.hybrid_e_finisher = reevaluate is not "
                "implemented for the Darwin field split");
            m_e_finisher_reevaluate = false;
        }
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_hybrid_pic_model->m_include_thermal_conduction || m_darwin ||
            !m_hybrid_pic_model->m_add_external_fields,
        "Implicit thermal conduction with external fields requires unified Darwin fields");

    // Segregated midpoint-iterated solve for the QDSMC electron-energy
    // stage (see the member documentation in the header).
    // Temperature-dependent push and Ohm coefficients must share a frozen
    // thermal stage during every inner residual and Jacobian evaluation.
    m_qdsmc_segregated_solve = m_hybrid_pic_model->m_resistivity_has_Te_dependence;
    pp.query("qdsmc_segregated_solve", m_qdsmc_segregated_solve);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_hybrid_pic_model->m_resistivity_has_Te_dependence || m_qdsmc_segregated_solve,
        "Temperature-dependent implicit resistivity requires qdsmc_segregated_solve");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(m_hybrid_pic_model->m_resistivity_has_Te_dependence ||
          m_hybrid_pic_model->m_include_thermal_conduction) ||
            m_theta == 0.5_rt,
        "Implicit live-temperature resistivity and split conduction require theta=0.5");
    if (m_qdsmc_segregated_solve) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_hybrid_pic_model->m_solve_electron_energy_equation,
            "implicit_evolve.qdsmc_segregated_solve requires "
            "hybrid_pic_model.solve_electron_energy_equation = true");
        pp.query("qdsmc_outer_max_iterations", m_qdsmc_outer_max_iterations);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_qdsmc_outer_max_iterations >= 1,
            "implicit_evolve.qdsmc_outer_max_iterations must be >= 1");
        pp.query("qdsmc_outer_relative_tolerance", m_qdsmc_outer_relative_tolerance);
        pp.query("qdsmc_outer_require_convergence", m_qdsmc_outer_require_convergence);
        pp.query("qdsmc_outer_verbose", m_qdsmc_outer_verbose);
    }

    pp.query("darwin_segregated_solve", m_darwin_segregated_solve);
    if (m_darwin_segregated_solve) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_darwin, "darwin_segregated_solve requires Darwin");
        pp.query("darwin_outer_max_iterations", m_darwin_outer_max_iterations);
        pp.query("darwin_outer_relative_tolerance", m_darwin_outer_rtol);
        pp.query("darwin_outer_absolute_tolerance", m_darwin_outer_atol);
        pp.query("darwin_outer_verbose", m_darwin_outer_verbose);
        pp.query("darwin_outer_relaxation", m_darwin_outer_relaxation);
        pp.query("darwin_outer_relaxation_start", m_darwin_outer_relaxation_start);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_darwin_outer_relaxation > 0.0_rt
            && m_darwin_outer_relaxation <= 1.0_rt
            && m_darwin_outer_relaxation_start >= 0,
            "darwin_outer_relaxation must be in (0,1] and its start iteration nonnegative");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_darwin_outer_max_iterations > 0
            && m_darwin_outer_rtol >= 0 && m_darwin_outer_atol >= 0
            && (m_darwin_outer_rtol > 0 || m_darwin_outer_atol > 0),
            "Darwin outer iterations and tolerances must be positive");
    }

    pp.query("darwin_vacuum_pc_regularization", m_darwin_vacuum_pc_regularization);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_darwin_vacuum_pc_regularization >= 0.0_rt,
        "darwin_vacuum_pc_regularization must be nonnegative (0 disables it)");
    if (m_darwin_vacuum_pc_regularization > 0.0_rt) {
        bool preserve_rows = false;
        amrex::ParmParse("pc_curl_curl_mlmg").query("preserve_dirichlet_rows", preserve_rows);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_hybrid_pic_model->m_include_electron_inertia && preserve_rows,
            "Darwin vacuum PC requires electron inertia and pc_curl_curl_mlmg.preserve_dirichlet_rows=1");
    }

    pp.query("darwin_vacuum_gauge_projection", m_darwin_vacuum_gauge_projection);
    pp.query("darwin_vacuum_gauge_pc_relative_tolerance", m_darwin_vacuum_gauge_pc_rtol);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_darwin_vacuum_gauge_pc_rtol > 0.0_rt
        && m_darwin_vacuum_gauge_pc_rtol < 1.0_rt,
        "darwin_vacuum_gauge_pc_relative_tolerance must lie strictly between 0 and 1");
    if (m_darwin_vacuum_gauge_projection) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(AMREX_SPACEDIM == 3 && m_darwin_segregated_solve
            && m_darwin_vacuum_pc_regularization > 0.0_rt,
            "darwin_vacuum_gauge_projection requires 3D split and native vacuum PC");
    }

    // Circuit-in-the-residual coupling (see the member documentation in
    // the header).
    pp.query("external_field_iteration", m_external_field_iteration);
    m_darwin_circuit_consistent_stage =
        AMREX_SPACEDIM == 3 && m_WarpX->maxLevel() == 0
        && m_darwin && m_darwin_segregated_solve && m_external_field_iteration;
    pp.query("darwin_circuit_consistent_stage", m_darwin_circuit_consistent_stage);
    pp.query("darwin_circuit_max_iterations", m_darwin_circuit_max_iterations);
    pp.query("darwin_circuit_scale_tolerance", m_darwin_circuit_scale_tolerance);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_darwin_circuit_max_iterations > 0
        && std::isfinite(m_darwin_circuit_scale_tolerance)
        && m_darwin_circuit_scale_tolerance > 0._rt
        && m_darwin_circuit_scale_tolerance < 1._rt,
        "Darwin circuit stage requires positive iteration limit and scale tolerance in (0,1)");
    if (m_darwin_circuit_consistent_stage) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(AMREX_SPACEDIM == 3 && m_WarpX->maxLevel() == 0
            && m_darwin && m_darwin_segregated_solve && m_external_field_iteration,
            "darwin_circuit_consistent_stage requires single-level 3D segregated Darwin coupling");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_vacuum_recovery
            || (m_hybrid_pic_model->m_darwin_vacrec_relax_time == 0._rt
                && m_hybrid_pic_model->m_darwin_vacuum_recovery_frozen_mask),
            "Darwin circuit stage requires instantaneous recovery with a frozen mask");
    }


    // Redistribute ahead of the end-of-step deposits (see the member
    // documentation in the header).
    pp.query("redistribute_before_end_deposits",
             m_redistribute_before_end_deposits);
    if (m_external_field_iteration) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_hybrid_pic_model->m_add_external_fields,
            "implicit_evolve.external_field_iteration requires "
            "hybrid_pic_model.add_external_fields");
        if (m_vacuum_recovery) {
            // The recovery's frozen-probe shortcut reuses the correction
            // from the last non-Jacobian evaluation; with the coil scales
            // changing inside every evaluation that would make the probed
            // map inconsistent with the iterate map (the frozen-probe-lag
            // failure class). Force the live-probe recovery; run with a
            // tight darwin_vacuum_recovery_relative_tolerance.
            m_hybrid_pic_model->m_vacuum_recovery_live_probes = true;
        }
    }

    {
        std::string driver = m_darwin_circuit_consistent_stage ? "native" : "python";
        pp.query("circuit_driver", driver);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(driver == "native" || driver == "python",
            "implicit_evolve.circuit_driver must be native or python");
        m_circuit_native = driver == "native";
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_circuit_native
            || (AMREX_SPACEDIM == 3 && m_WarpX->maxLevel() == 0 && m_darwin
                && m_darwin_segregated_solve && m_external_field_iteration
                && m_darwin_circuit_consistent_stage && sizeof(amrex::Real) == sizeof(double)),
            "Native circuit driver requires single-level 3D double-precision segregated Darwin with consistent circuit stages");
    }

    parseNonlinearSolverParams( pp );
    m_nlsolver->Define(m_E, this);

    if (m_use_mass_matrices) { InitializeMassMatrices(); }

    m_is_defined = true;
}

void ThetaImplicitHybrid::PrintParameters () const
{
    BL_PROFILE("ThetaImplicitHybrid::PrintParameters()");

    if (!m_WarpX->Verbose()) { return; }
    amrex::Print() << "\n";
    amrex::Print() << "-----------------------------------------------------------\n";
    amrex::Print() << "-------- THETA IMPLICIT HYBRID PIC SOLVER PARAMETERS ------\n";
    amrex::Print() << "-----------------------------------------------------------\n";
    amrex::Print() << "Time-bias parameter theta:           " << m_theta << "\n";
    if (m_qdsmc_segregated_solve) {
        amrex::Print() << "QDSMC segregated solve:              on\n";
        amrex::Print() << "  outer max iterations:              " << m_qdsmc_outer_max_iterations << "\n";
        amrex::Print() << "  outer relative tolerance:          " << m_qdsmc_outer_relative_tolerance << "\n";
        amrex::Print() << "  outer require convergence:         " << (m_qdsmc_outer_require_convergence?"true":"false") << "\n";
    }
    PrintBaseImplicitSolverParameters();
    m_nlsolver->PrintParams();
    amrex::Print() << "-----------------------------------------------------------\n\n";
}

int ThetaImplicitHybrid::OneStep ( const amrex::Real  start_time,
                                   const amrex::Real  a_dt,
                                   const int          a_step )
{
    BL_PROFILE("ThetaImplicitHybrid::OneStep()");

    m_dt = a_dt;

    if (m_darwin_circuit_consistent_stage) {
        auto const& external = *m_hybrid_pic_model->m_external_vector_potential;
        m_darwin_circuit_accepted_scales.resize(external.nFields());
        for (int coil = 0; coil < external.nFields(); ++coil) {
            auto const value = external.TimeScale(coil, start_time);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(value),
                "Nonfinite accepted circuit scale");
            m_darwin_circuit_accepted_scales[coil] = value;
        }
    }

    // tensor_form: publish the theta interval and snapshot the step-start
    // plasma current Jp^n = curl(B^n)/mu0 - J_ext (Bfield_fp holds the
    // committed B^n here). Both are per-step-frozen inputs of the
    // stateless Je elimination, constant through every residual
    // evaluation of the step.
    if (m_hybrid_pic_model->m_esolve_tensor) {
        m_hybrid_pic_model->m_tensor_dt_eff = m_theta * m_dt;
        m_hybrid_pic_model->CaptureTensorStepStart();
    }

    // curlcurl_form + frozen gates: capture the per-step-frozen rho^n
    // snapshot HERE, from the committed entry deposit in component 0 of
    // rho_fp (the tensor/vacmask capture family), BEFORE any residual
    // evaluation of the step. The resistive push-field correction runs
    // its Ohm passes before the first midpoint deposit and before the
    // inertia assembly's lazy capture, so the first elliptic solves of
    // the run otherwise read an EMPTY midpoint density -- beta = 0
    // everywhere and every anchored RHS row zeroed against a nonzero
    // resistive warm start, an unreachable tolerance (measured: seeded
    // vacuum-column decks cap the toroidal CG at the very first solve).
    // The inertia assembly's own capture becomes a per-step no-op (latch
    // already set); the drho/dt leg, the gates, and the toroidal fold all
    // read this same entry snapshot.
    if (m_hybrid_pic_model->m_esolve_curlcurl
        && m_hybrid_pic_model->m_curlcurl_pol_frozen_rho) {
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            amrex::MultiFab const & rho_fp =
                *m_WarpX->m_fields.get(FieldType::rho_fp, lev);
            amrex::MultiFab & rho_n_frozen =
                *m_WarpX->m_fields.get("hybrid_rho_n_frozen", lev);
            amrex::MultiFab::Copy(rho_n_frozen, rho_fp, 0, 0, 1,
                                  amrex::min(rho_n_frozen.nGrowVect(),
                                             rho_fp.nGrowVect()));
        }
        m_hybrid_pic_model->m_inertia_rho_n_captured = true;
    }

    // External vector-potential drive, split-field form: the solver state
    // carries the PLASMA fields only, so the field boundary conditions act
    // on the plasma response while the imposed external field rides
    // through the wall unchanged (a conducting boundary must not exclude
    // the programmed coil flux; advancing the external flux through the
    // discrete Faraday/vector-potential update would pin it to the wall
    // value of E or A). Strip the external field at t^n here; the field
    // assembly in UpdateWarpXFields re-adds B_ext at the theta-time and
    // the step-averaged inductive E_ext on top of the plasma fields, and
    // FinishFieldUpdate restores end-of-step totals.
    if (m_hybrid_pic_model->m_add_external_fields && !m_darwin) {
        using ablastr::fields::Direction;
        auto & ext = *m_hybrid_pic_model->m_external_vector_potential;
        ext.UpdateHybridExternalFields(start_time, a_dt);
        ExtLedgerPrint(m_WarpX, "entry pre-strip");
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab & B = *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{dir}, lev);
                amrex::MultiFab const & B_ext = *m_WarpX->m_fields.get(FieldType::hybrid_B_fp_external, Direction{dir}, lev);
                amrex::MultiFab::Subtract(B, B_ext, 0, 0, B.nComp(), B.nGrowVect());
                amrex::MultiFab & E = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
                amrex::MultiFab const & E_ext = *m_WarpX->m_fields.get(FieldType::hybrid_E_fp_external, Direction{dir}, lev);
                amrex::MultiFab::Subtract(E, E_ext, 0, 0, E.nComp(), E.nGrowVect());
            }
        }
        ExtLedgerPrint(m_WarpX, "entry post-strip");
        // Mid-step values used throughout the nonlinear solve: B_ext at
        // t^{n+theta}, and E_ext = -[f(t^{n+theta}+dt/2) -
        // f(t^{n+theta}-dt/2)]/dt * A for the push field (at theta = 1/2
        // this is the exact step mean of the inductive field).
        ext.UpdateHybridExternalFields(start_time + m_theta*a_dt, a_dt);
        ExtLedgerPrint(m_WarpX, "entry post-theta-eval");
    }

    // Save particle state at t^n
    m_WarpX->SaveParticlesAtImplicitStepStart();

    // Save E^n
    SaveEoldMultifab();
    if (m_darwin) {
        using ablastr::fields::Direction;
        if (!m_darwin_initialized) {
            // Gauge-free initialization: A(0) = 0 and B_static = B(t=0), so
            // B(t) = B_static + curl A(t) holds exactly for all later times
            // (Faraday integrates the change of B into A). No curl inversion
            // is ever required.
            for (int lev = 0; lev < m_num_amr_levels; ++lev) {
                for (int dir = 0; dir < 3; ++dir) {
                    amrex::MultiFab const & B = *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{dir}, lev);
                    amrex::MultiFab & Bs = *m_WarpX->m_fields.get("hybrid_B_static_fp", Direction{dir}, lev);
                    amrex::MultiFab::Copy(Bs, B, 0, 0, Bs.nComp(), Bs.nGrowVect());
                }
            }
            // A consistent electron pressure for the first E_L solve (the
            // energy-equation path fills Pe during InitData; the closure
            // path needs one evaluation from the entry-deposit density).
            if (!m_hybrid_pic_model->m_solve_electron_energy_equation) {
                m_hybrid_pic_model->CalculateElectronPressure();
            }
            m_darwin_initialized = true;
        }

        // E_L^n from the entry state (Pe^n and the Evolve-entry deposit
        // rho^n in component 0 of rho_fp); save it and A^n for the theta
        // reconstructions, then strip E_L from Efield_fp so the solver
        // state (seeded from Efield_fp below) is the transverse field E_T.
        ablastr::fields::MultiLevelScalarField rho_n_alias;
        amrex::Vector<std::unique_ptr<amrex::MultiFab>> rho_n_store(m_num_amr_levels);
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            amrex::MultiFab & rho_fp = *m_WarpX->m_fields.get(FieldType::rho_fp, lev);
            rho_n_store[lev] = std::make_unique<amrex::MultiFab>(rho_fp, amrex::make_alias, 0, 1);
            rho_n_alias.push_back(rho_n_store[lev].get());
        }
        // Frozen vacuum-recovery mask density: snapshot the same committed
        // entry rho the E_L^n solve below consumes, so the recovery/
        // Faraday-overwrite partition is constant through every residual
        // evaluation of the step (rho_fp component 0 is a pre-push deposit
        // that follows the iterate from the second evaluation on -- see
        // m_darwin_vacuum_recovery_frozen_mask).
        if (m_vacuum_recovery
            && m_hybrid_pic_model->m_darwin_vacuum_recovery_frozen_mask) {
            for (int lev = 0; lev < m_num_amr_levels; ++lev) {
                amrex::MultiFab const & rho_fp =
                    *m_WarpX->m_fields.get(FieldType::rho_fp, lev);
                amrex::MultiFab & rho_mask =
                    *m_WarpX->m_fields.get("hybrid_rho_vacmask_fp", lev);
                amrex::MultiFab::Copy(rho_mask, rho_fp, 0, 0, 1,
                                      amrex::min(rho_mask.nGrowVect(),
                                                 rho_fp.nGrowVect()));
            }
        }
        // With electron inertia, the E_L source reads the inertial field
        // at its last converged assembly (t^{n-1+theta}) here -- a
        // half-step staleness of the same order as the ion half-step
        // offset in the Je history; a t^n reassembly from the histories
        // is a possible refinement.
        m_hybrid_pic_model->ComputeDarwinELong(rho_n_alias, start_time);
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab const & EL = *m_WarpX->m_fields.get("hybrid_E_long_fp", Direction{dir}, lev);
                amrex::MultiFab & EL_old = *m_WarpX->m_fields.get("hybrid_E_long_old_fp", Direction{dir}, lev);
                amrex::MultiFab::Copy(EL_old, EL, 0, 0, EL.nComp(), EL.nGrowVect());
                amrex::MultiFab const & A = *m_WarpX->m_fields.get("hybrid_A_fp", Direction{dir}, lev);
                amrex::MultiFab & A_old = *m_WarpX->m_fields.get("hybrid_A_old_fp", Direction{dir}, lev);
                amrex::MultiFab::Copy(A_old, A, 0, 0, A.nComp(), A.nGrowVect());
                amrex::MultiFab & E = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
                amrex::MultiFab::Subtract(E, EL, 0, 0, E.nComp(), E.nGrowVect());
            }
        }
        // Boundary-driven external flux: pin A^n (idempotent re-pin of the
        // end-of-last-step values, and the gauge reference on step one).
        DarwinApplyABoundary(start_time);

        // The transverse state at t^n (Efield_fp = E^n - E_L^n here).
        // Keep the previous step's E^n as E^{n-1} for the extrapolated guess.
        if (m_extrapolate_initial_guess && m_have_Eold) {
            m_Eprev.Copy(m_Eold);
            m_have_Eprev = true;
        }
        m_Eold.Copy(FieldType::Efield_fp);
        m_have_Eold = true;
    } else {
        // Non-Darwin path: E^n sits in the E_old register. Same E^{n-1} bookkeeping
        // (a restart starts with m_have_Eold false, so its first step uses E^n).
        if (m_extrapolate_initial_guess && m_have_Eold) {
            m_Eprev.Copy(m_Eold);
            m_have_Eprev = true;
        }
        m_Eold.Copy(FieldType::E_old, FieldType::None, true);
        m_have_Eold = true;
    }

    // Save B^n
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const ablastr::fields::VectorField Bfp = m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, lev);
        ablastr::fields::VectorField B_old = m_WarpX->m_fields.get_alldirs(FieldType::B_old, lev);
        for (int n = 0; n < 3; ++n) {
            amrex::MultiFab::Copy(*B_old[n], *Bfp[n], 0, 0,
                                  B_old[n]->nComp(), B_old[n]->nGrowVect());
        }
    }

    // Save the electron-energy start-of-step state (T_e^n, J_plasma(B^n),
    // rho^n, frozen T_i^n deposits). B currently holds B^n, so refresh the
    // plasma current from it first.
    if (m_hybrid_pic_model->m_solve_electron_energy_equation) {
        m_hybrid_pic_model->CalculatePlasmaCurrent(
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1),
            m_WarpX->GetEBUpdateEFlag());
        m_hybrid_pic_model->QDSMCSaveImplicitStepStart(m_dt, start_time);
        if (m_qdsmc_segregated_solve)
        {
            m_qdsmc_rho_frozen.resize(m_num_amr_levels);
            for (int lev = 0; lev < m_num_amr_levels; ++lev)
            {
                auto const& rho = *m_WarpX->m_fields.get(FieldType::hybrid_rho_fp_temp, lev);
                auto& frozen = m_qdsmc_rho_frozen[lev];
                if (!frozen || frozen->boxArray() != rho.boxArray() ||
                    frozen->DistributionMap() != rho.DistributionMap())
                {
                    frozen = std::make_unique<amrex::MultiFab>(
                        rho.boxArray(), rho.DistributionMap(), 1, rho.nGrowVect());
                }
                amrex::MultiFab::Copy(*frozen, rho, 0, 0, 1, rho.nGrowVect());
            }
        }
    }

    if (m_circuit_native) {
        auto& coupler = NativeCircuitCoupler();
        if (!m_native_circuit_configured) {
            coupler.ConfigureDarwinMagneticResponse();
            m_native_circuit_configured = true;
        }
        // These are the actual accepted B^n registers, before any trial update.
        coupler.MeasureDarwinLinkages(start_time);
        coupler.BeginStepMeasured(start_time, m_dt);
        if (coupler.DeviceTrials()) {
            if (m_fext_init.empty()) {
                auto const& ext = *m_hybrid_pic_model->m_external_vector_potential;
                for (int i = 0; i < ext.nFields(); ++i) {
                    m_fext_init.push_back(ext.TimeScale(i,start_time));
                }
            }
            coupler.PrepareDarwinDeviceStep(start_time,m_dt,m_theta,m_darwin_circuit_scale_tolerance);
        }
        m_native_circuit_step_open = true;
    }

    // Initial guess: E^{n+theta} = E^n, or the linear extrapolation of the
    // field history (1 + theta) E^n - theta E^{n-1} when opted in and E^{n-1}
    // exists (saves the Newton iteration that otherwise rebuilds the step's
    // change from scratch; the converged state does not depend on the guess).
    if (m_extrapolate_initial_guess && m_have_Eprev) {
        m_E.linComb(1.0_rt + m_theta, m_Eold, -m_theta, m_Eprev);
    } else {
        m_E.Copy(m_Eold);
    }

    // Solve nonlinear system for E^{n+theta} (and eventually Pe^{n+theta})
    int exit_status = 0;
    if (m_darwin_segregated_solve) {
        exit_status = SolveDarwinSegregated(start_time, a_step);
    } else if (m_qdsmc_segregated_solve) {
        exit_status = SolveSegregated( start_time, a_step );
    } else {
        m_nlsolver->Solve( m_E, m_Eold, start_time, m_dt, a_step );
        exit_status = m_nlsolver->GetExitStatus();
    }
    if (exit_status < 0) { return exit_status; }

    // Update WarpX fields to t^{n+theta}
    UpdateWarpXFields( m_E, false, start_time );
    m_WarpX->reduced_diags->ComputeDiagsMidStep(a_step);

    const amrex::Real new_time = start_time + m_dt;

    // Advance particles from t^{n+1/2} to t^{n+1}
    m_WarpX->FinishImplicitParticleUpdate(new_time);

    if (m_circuit_native) { CommitNativeCircuitStage(start_time); }

    // Advance fields from t^{n+theta} to t^{n+1}
    FinishFieldUpdate( new_time );

    // Opt-in: apply the particle boundary conditions and re-bin before the
    // end-of-step deposits below (see the member documentation). The rho
    // deposition guard range covers displacements of up to
    // (nox + particles.max_grid_crossings - nox/2 - 1) cells from the home
    // tile (GuardCellManager: rho band nox + max_grid_crossings, J band one
    // less; default max_grid_crossings 1), and the full-dt extrapolation
    // above can exceed it for warm boundary populations. The midpoint
    // deposits inside every nonlinear iteration are bound by the same band
    // and are NOT covered by this knob: widen particles.max_grid_crossings
    // for large time steps.
    if (m_redistribute_before_end_deposits) {
        m_WarpX->GetPartContainer().Redistribute();
    }

    // Complete the electron-energy step: apply the stochastic ion-heating
    // realization once with converged states, refresh Pe^{n+1}, and reset
    // the QDSMC markers.
    if (m_hybrid_pic_model->m_solve_electron_energy_equation) {
        m_hybrid_pic_model->QDSMCFinishImplicitStep(m_dt, m_theta, new_time);
    } else if (!m_darwin) {
        // Closure path: re-evaluate Pe^{n+1} (and the diagnostic T_e
        // mirror) from a true end-of-step density deposit. The in-solve
        // closure evaluations consumed midpoint-position deposits, so
        // without this the dumped Pe/Te lag the ion state by half a step
        // -- a first-order error in Te-based convergence metrics on a
        // scheme whose dynamics are second order. The energy-equation
        // branch above already ends with an equivalent end-of-step
        // recovery inside QDSMCFinishImplicitStep. Skipped under darwin:
        // the entry E_L^n solve consumes the as-left Pe together with the
        // as-left rho_fp, and re-labeling only one of that pair (or both)
        // changes validated darwin evolution -- the entry-state
        // time-labeling there is a separate, jointly-decided item.
        m_hybrid_pic_model->CalculateElectronPressureAtStepEnd();
    }

    // Refresh the per-species temperature deposits from the end-of-step
    // particle state (after the ion-heating realization above). The
    // explicit scheme deposits these every step; without this call the
    // T_<species> diagnostics would hold their initialization values for
    // the whole run. Species without do_temperature_deposition are
    // skipped inside.
    m_WarpX->GetPartContainer().DepositTemperatures(m_WarpX->m_fields, 0.0_rt);

    // Leave hybrid_current_fp_plasma holding the delivered end-of-step
    // Ampere closure, J_plasma^{n+1} = curl(B^{n+1})/mu0 - J_ext. The
    // residual evaluations left the theta-stage value; everything that
    // reads the register between steps must observe the same t^{n+1}
    // state the explicit loop ends on: the coherent afterEpush /
    // afterEsolve python callbacks (e.g. a segregated circuit coupler
    // measuring the plasma flux linkage), the particle-level resistive
    // drag (collisions run after OneStep under the implicit schemes),
    // and the displacement-current diagnostic. Runs after
    // QDSMCFinishImplicitStep, which consumes the theta-stage value.
    // Split-field externals: Bfield_fp holds end-of-step TOTALS here, so
    // strip the external field around the curl -- the plasma-current
    // register must stay response-only (the explicit loop's final refresh
    // runs before its external add-back; a totals-frame curl would leak
    // the coil field's O(h^2) discrete curl into the circuit flux-linkage
    // probes and the resistive drag).
    const bool strip_ext =
        m_hybrid_pic_model->m_add_external_fields && !m_darwin;
    if (strip_ext) { AddSplitExternalFields(-1.0_rt); }
    m_hybrid_pic_model->CalculatePlasmaCurrent(
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1),
        m_WarpX->GetEBUpdateEFlag());
    if (strip_ext) { AddSplitExternalFields(1.0_rt); }
    if (m_darwin)
    {
        ApplyDarwinDisplacementCurrent(m_dt);
    }

    // Re-evaluated E finisher: overwrite the extrapolated E^{n+1} with the
    // generalized Ohm's law evaluated at the DELIVERED end-of-step state
    // (total B^{n+1}, the delivered plasma current refreshed above, the
    // same ion-deposit family the theta-stage used), as the explicit
    // hybrid loop finishes its step. The extrapolated finisher is a
    // -(1-theta)/theta recursion on the stored algebraic field.
    if (m_e_finisher_reevaluate) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_darwin,
            "implicit_evolve.hybrid_e_finisher = reevaluate is not "
            "implemented for the Darwin field split");
        if (!m_hybrid_pic_model->m_solve_electron_energy_equation) {
            m_hybrid_pic_model->CalculateElectronPressure();
        }
        // per-level variant: the multi-level HybridPICSolveE fires the
        // afterEpush python callback, which the implicit step already
        // fires exactly once at the delivered state (WarpX::OneStep) --
        // the finisher solve must not add a second firing per step
        {
            ablastr::fields::MultiLevelVectorField E_fp =
                m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, m_num_amr_levels - 1);
            ablastr::fields::MultiLevelVectorField J_fp =
                m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, m_num_amr_levels - 1);
            ablastr::fields::MultiLevelVectorField B_fp =
                m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1);
            ablastr::fields::MultiLevelScalarField r_fp =
                m_WarpX->m_fields.get_mr_levels(FieldType::rho_fp, m_num_amr_levels - 1);
            for (int lev = 0; lev < m_num_amr_levels; ++lev) {
                m_hybrid_pic_model->HybridPICSolveE(
                    E_fp[lev], J_fp[lev], B_fp[lev], *r_fp[lev],
                    m_WarpX->GetEBUpdateEFlag()[lev], lev,
                    false /* solve_for_Faraday */,
                    true /* include_resistivity */);
            }
        }
        {
            using ablastr::fields::Direction;
            amrex::IntVect const ngE = m_WarpX->m_fields.get(
                FieldType::Efield_fp, Direction{0}, 0)->nGrowVect();
            m_WarpX->FillBoundaryE(ngE, true /* sync nodal points */);
        }
        // keep the solver vector consistent with the delivered field
        m_E.Copy(FieldType::Efield_fp);
    }

    // Density-band statistics feeding (DSMC-style split in depleted
    // cells): runs at step boundaries only, never inside the residual.
    // Band limits are configured per species in units of the hybrid
    // n_floor; the merge relief valve is the stock velocity-coincidence
    // resampler.
    {
        auto& mpc = m_WarpX->GetPartContainer();
        const amrex::Real rho_floor =
            m_hybrid_pic_model->m_n_floor * PhysConst::q_e;
        for (int isp = 0; isp < mpc.nSpecies(); ++isp) {
            auto* pc = dynamic_cast<PhysicalParticleContainer*>(
                &mpc.GetParticleContainer(isp));
            if (pc == nullptr) { continue; }
            const int interval = pc->HybridSplitInterval();
            if (interval <= 0 || ((a_step + 1) % interval != 0)) { continue; }
            const amrex::MultiFab& rho0 =
                *m_WarpX->m_fields.get(FieldType::rho_fp, 0);
            pc->SplitDepletedBand(rho0,
                pc->HybridSplitBandLo()*rho_floor,
                pc->HybridSplitBandHi()*rho_floor, 0);
            pc->Redistribute();
        }
    }

    // Electron inertia: rotate the per-step nodal Je history from the
    // MEASURED delivered state -- hybrid_current_fp_plasma now holds
    // J_plasma^{n+1} (including the Darwin displacement piece above), and
    // current_fp still holds the same ion-deposit family the theta-stage
    // assemblies used. Runs here (not in FinishFieldUpdate) so the stored
    // value is a measurement, not an extrapolation of a stored value.
    if (m_hybrid_pic_model->m_include_electron_inertia) {
        m_hybrid_pic_model->RotateElectronInertiaHistory(m_theta);
    }

    return exit_status;
}

void
ThetaImplicitHybrid::ApplyDarwinDisplacementCurrent (amrex::Real interval)
{
    // Called only after Ampere rebuilds Jp. Both push correction and Ohm
    // consume J = curl(B)/mu0 - epsilon0*dEL/dt at the same stage.
    using ablastr::fields::Direction;
    amrex::Real const inv_thetadt = 1.0_rt / interval;
    for (int lev = 0; lev < m_num_amr_levels; ++lev)
    {
        for (int dir = 0; dir < 3; ++dir)
        {
            amrex::MultiFab& Jp =
                *m_WarpX->m_fields.get(FieldType::hybrid_current_fp_plasma, Direction{dir}, lev);
            amrex::MultiFab const& EL =
                *m_WarpX->m_fields.get("hybrid_E_long_fp", Direction{dir}, lev);
            amrex::MultiFab const& EL_old =
                *m_WarpX->m_fields.get("hybrid_E_long_old_fp", Direction{dir}, lev);
            amrex::MultiFab::Saxpy(Jp, -PhysConst::epsilon_0 * inv_thetadt, EL, 0, 0, Jp.nComp(),
                                   Jp.nGrowVect());
            amrex::MultiFab::Saxpy(Jp, PhysConst::epsilon_0 * inv_thetadt, EL_old, 0, 0, Jp.nComp(),
                                   Jp.nGrowVect());
        }
    }
    if (EB::enabled())
    {
        for (int lev = 0; lev < m_num_amr_levels; ++lev)
        {
            auto current = m_WarpX->m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
            auto& flags = m_WarpX->GetEBUpdateEFlag()[lev];
            if (m_hybrid_pic_model->m_use_conformal_eb &&
                m_hybrid_pic_model->m_conformal_wall_conductor)
            {
                m_hybrid_pic_model->ZeroConductorEdges(current, flags, lev);
            }
            else
            {
                for (int d = 0; d < 3; ++d)
                {
                    for (amrex::MFIter mfi(*current[d], amrex::TilingIfNotGPU()); mfi.isValid();
                         ++mfi)
                    {
                        auto const jp = current[d]->array(mfi);
                        auto const open = flags[d]->const_array(mfi);
                        amrex::ParallelFor(mfi.tilebox(),
                                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                           {
                                               if (!open(i, j, k))
                                               {
                                                   jp(i, j, k) = 0.0_rt;
                                               }
                                           });
                    }
                }
            }
            for (int d = 0; d < 3; ++d)
            {
                current[d]->FillBoundary(m_WarpX->Geom(lev).periodicity());
            }
        }
    }
}

void ThetaImplicitHybrid::RefreshDarwinELong (amrex::Real theta_time)
{
    ablastr::fields::MultiLevelScalarField rho_half_alias;
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> rho_half_store(m_num_amr_levels);
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        auto& rho = *m_WarpX->m_fields.get(FieldType::rho_fp, lev);
        rho_half_store[lev] = std::make_unique<amrex::MultiFab>(
            rho, amrex::make_alias, rho.nComp()/2, 1);
        rho_half_alias.push_back(rho_half_store[lev].get());
    }
    m_hybrid_pic_model->ComputeDarwinELong(
        rho_half_alias, theta_time, m_darwin_segregated_solve);
}

int ThetaImplicitHybrid::SolveDarwinSegregated (amrex::Real start_time, int a_step)
{
    BL_PROFILE("ThetaImplicitHybrid::SolveDarwinSegregated()");
    // The density mask is frozen only within this step. Rebuild its scalar
    // correction space before projecting the first guess and Krylov updates.
    m_vacuum_gauge_solver.reset();
    m_vacuum_gauge_op.reset();
    ProjectDarwinVacuumGauge(m_E);
    using ablastr::fields::Direction;
    // Keep ghosts too: a rejected candidate must not alter the field gathered
    // by the converged particle/energy stage. Scratch is local to this step.
    amrex::Vector<amrex::Array<std::unique_ptr<amrex::MultiFab>, 3>> previous(
        m_num_amr_levels);
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        for (int dir = 0; dir < 3; ++dir) {
            auto const& field = *m_WarpX->m_fields.get("hybrid_E_long_fp", Direction{dir}, lev);
            previous[lev][dir] = std::make_unique<amrex::MultiFab>(
                field.boxArray(), field.DistributionMap(), field.nComp(), field.nGrowVect());
        }
    }
    amrex::Real field_rtol, field_atol;
    int field_maxits;
    m_nlsolver->GetSolverParams(field_rtol, field_atol, field_maxits);
    amrex::ignore_unused(field_maxits);
    WarpXSolverVec residual;
    residual.Define(m_E);
    amrex::Real field_reference = 0.0_rt;
    for (int outer = 0; outer < m_darwin_outer_max_iterations; ++outer) {
        // Grade the coupled solve against this STEP's initial field residual.
        // Starting a fresh relative Newton solve after every tiny E_L update
        // would tighten the physical tolerance repeatedly, down to roundoff.
        ComputeRHS(residual, m_E, start_time, 0, false);
        residual.increment(m_Eold, 1.0_rt);
        residual.increment(m_E, -1.0_rt);
        amrex::Real const field_norm = residual.norm2();
        if (!std::isfinite(field_norm)) { return -8; }
        if (outer == 0 || field_reference == 0.0_rt) { field_reference = field_norm; }
        amrex::Real const field_target = std::max(field_atol, field_rtol * field_reference);
        if (m_darwin_outer_verbose) {
            amrex::Print() << "Darwin field: outer=" << outer
                << " residual=" << field_norm << " target=" << field_target << "\n";
        }
        int status = 3;
        if (m_qdsmc_segregated_solve)
        {
            // The thermal update is required even when the initial field
            // residual already passes. E_L stays fixed throughout this solve.
            status = SolveSegregated(start_time, a_step, field_reference);
            if (status < 0)
            {
                return status;
            }
        }
        else if (!(field_norm == 0.0_rt || field_norm < field_target))
        {
            // E_L stays fixed for every Jv and line search in this solve.
            m_nlsolver->SetConvergenceReferenceNorm(field_reference);
            m_nlsolver->Solve(m_E, m_Eold, start_time, m_dt, a_step);
            m_nlsolver->SetConvergenceReferenceNorm(0.0_rt);
            status = m_nlsolver->GetExitStatus();
            if (status < 0) { return status; }
            // A permissive/fixed-iteration inner solve is not evidence that
            // the field equation converged. The split requires both gates.
            if (status != 2 && status != 3) { return -8; }
        }
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                auto const& field = *m_WarpX->m_fields.get(
                    "hybrid_E_long_fp", Direction{dir}, lev);
                amrex::MultiFab::Copy(*previous[lev][dir], field, 0, 0,
                                      field.nComp(), field.nGrowVect());
            }
        }
        // The successful solver leaves the particle deposits, pressure and
        // inertial field at its accepted state. Test the constraint there.
        RefreshDarwinELong(start_time + m_theta*m_dt);
        amrex::Real change = 0.0_rt;
        amrex::Real scale = 0.0_rt;
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                auto const& field = *m_WarpX->m_fields.get(
                    "hybrid_E_long_fp", Direction{dir}, lev);
                auto const& old = *previous[lev][dir];
                if (!field.is_finite()) { return -8; }
                amrex::MultiFab diff(field.boxArray(), field.DistributionMap(),
                                    field.nComp(), 0);
                amrex::MultiFab::Copy(diff, field, 0, 0, field.nComp(), 0);
                amrex::MultiFab::Subtract(diff, old, 0, 0, field.nComp(), 0);
                change = std::max(change, diff.norminf());
                scale = std::max(scale, std::max(field.norminf(), old.norminf()));
            }
        }
        amrex::Real const target = m_darwin_outer_atol + m_darwin_outer_rtol * scale;
        if (m_darwin_outer_verbose) {
            amrex::Print() << "Darwin segregated: outer=" << outer
                << " constraint_change=" << change << " target=" << target << "\n";
        }
        if (change <= target) {
            // Accept the pair actually solved by Newton. Its independently
            // measured longitudinal defect is bounded by the outer tolerance.
            for (int lev = 0; lev < m_num_amr_levels; ++lev) {
                for (int dir = 0; dir < 3; ++dir) {
                    auto& field = *m_WarpX->m_fields.get(
                        "hybrid_E_long_fp", Direction{dir}, lev);
                    amrex::MultiFab::Copy(field, *previous[lev][dir], 0, 0,
                                          field.nComp(), field.nGrowVect());
                }
            }
            return status;
        }
        if (m_darwin_outer_relaxation != 1.0_rt
            && outer >= m_darwin_outer_relaxation_start) {
            // Grade the unrelaxed constraint defect above. Damping controls
            // the iteration only; it must never reduce the acceptance test.
            for (int lev = 0; lev < m_num_amr_levels; ++lev) {
                for (int dir = 0; dir < 3; ++dir) {
                    auto& field = *m_WarpX->m_fields.get(
                        "hybrid_E_long_fp", Direction{dir}, lev);
                    amrex::MultiFab::LinComb(field,
                        m_darwin_outer_relaxation, field, 0,
                        1.0_rt - m_darwin_outer_relaxation, *previous[lev][dir], 0,
                        0, field.nComp(), field.nGrowVect());
                }
            }
        }
        // Warm-start at unchanged E_total: E_T(new) = E_T(old) - delta E_L.
        // The next Newton solve handles any BC-induced change of the A/B map.
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                auto& transverse = *m_E.getArrayVec()[lev][dir];
                auto const& field = *m_WarpX->m_fields.get(
                    "hybrid_E_long_fp", Direction{dir}, lev);
                amrex::MultiFab::Add(transverse, *previous[lev][dir], 0, 0,
                                     transverse.nComp(), 0);
                amrex::MultiFab::Subtract(transverse, field, 0, 0,
                                          transverse.nComp(), 0);
            }
        }
        // E_L changes can inject a vacuum gradient through the total-E
        // warm start. Select the same gauge used by all Krylov corrections.
        ProjectDarwinVacuumGauge(m_E);
    }
    ablastr::warn_manager::WMRecordWarning("ThetaImplicitHybrid",
        "Darwin segregated longitudinal constraint failed to converge");
    return -8;
}

int
ThetaImplicitHybrid::SolveSegregated (const amrex::Real start_time, const int a_step,
                                      amrex::Real field_reference)
{
    BL_PROFILE("ThetaImplicitHybrid::SolveSegregated()");

    // Previous-iterate pressure scratch for the outer convergence test
    // (re-allocated if a load balance moved the field distribution).
    if (m_qdsmc_Pe_prev.empty()) { m_qdsmc_Pe_prev.resize(m_num_amr_levels); }
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        amrex::MultiFab const & Pe =
            *m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
        if (!m_qdsmc_Pe_prev[lev]
            || m_qdsmc_Pe_prev[lev]->boxArray() != Pe.boxArray()
            || m_qdsmc_Pe_prev[lev]->DistributionMap() != Pe.DistributionMap()) {
            m_qdsmc_Pe_prev[lev] = std::make_unique<amrex::MultiFab>(
                Pe.boxArray(), Pe.DistributionMap(), Pe.nComp(), amrex::IntVect(0));
        }
    }

    // Alternate {inner nonlinear solve at frozen Pe, one re-entrant stage
    // pass} until the emitted pressure stops changing between outer
    // iterations. The stage pass runs on the grid state as left by the
    // inner solver's last (non-probe) residual evaluation -- the plasma
    // current at B^{n+theta}, the midpoint particle deposits, and the
    // frozen step-start states are all mutually consistent there, and
    // nothing may be re-pushed or re-deposited (the as-left discipline of
    // CalculateElectronPressureAtStepEnd). Each outer iteration ENDS on the
    // stage pass, so the markers and the electron velocity are consistent
    // with the accepted field state and QDSMCFinishImplicitStep completes
    // the characteristic unchanged; the O(tolerance) pressure-field
    // mismatch of the final pair is the explicit segregation error.
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> previous_temperature(m_num_amr_levels);
    for (int lev = 0; lev < m_num_amr_levels; ++lev)
    {
        auto const& Te = *m_WarpX->m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
        previous_temperature[lev] =
            std::make_unique<amrex::MultiFab>(Te.boxArray(), Te.DistributionMap(), Te.nComp(), 0);
    }
    WarpXSolverVec residual;
    residual.Define(m_E);
    amrex::Real field_rtol, field_atol;
    int field_maxits;
    m_nlsolver->GetSolverParams(field_rtol, field_atol, field_maxits);
    amrex::ignore_unused(field_maxits);
    auto residual_norm = [&] ()
    {
        ComputeRHS(residual, m_E, start_time, 0, false);
        residual.increment(m_Eold, 1.0_rt);
        residual.increment(m_E, -1.0_rt);
        return residual.norm2();
    };
    auto capture_thermal = [&] ()
    {
        for (int lev = 0; lev < m_num_amr_levels; ++lev)
        {
            amrex::MultiFab const & Pe =
                *m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
            amrex::MultiFab::Copy(*m_qdsmc_Pe_prev[lev], Pe, 0, 0, Pe.nComp(),
                                  amrex::IntVect(0));
            auto const& Te = *m_WarpX->m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
            amrex::MultiFab::Copy(*previous_temperature[lev], Te, 0, 0, Te.nComp(), 0);
        }
    };
    auto thermal_defect = [&] ()
    {
        amrex::Real dPe_rel = 0.0_rt;
        amrex::Real dTe_rel = 0.0_rt;
        amrex::Real drho_rel = 0.0_rt;
        for (int lev = 0; lev < m_num_amr_levels; ++lev)
        {
            amrex::MultiFab const & Pe =
                *m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
            amrex::MultiFab & dPe = *m_qdsmc_Pe_prev[lev];
            amrex::MultiFab::Subtract(dPe, Pe, 0, 0, Pe.nComp(), amrex::IntVect(0));
            amrex::Real const norm_Pe = Pe.norm2(0);
            amrex::Real const norm_dPe = dPe.norm2(0);
            dPe_rel = std::max(dPe_rel,
                norm_dPe / std::max(norm_Pe, std::numeric_limits<amrex::Real>::min()));
            auto const& Te = *m_WarpX->m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
            auto& diff = *previous_temperature[lev];
            for (amrex::MFIter mfi(diff, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                auto const delta = diff.array(mfi);
                auto const now = Te.const_array(mfi);
                amrex::ParallelFor(mfi.tilebox(),
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
                                       amrex::Real const old = delta(i, j, k);
                                       delta(i, j, k) = std::abs(now(i, j, k) - old) /
                                                        amrex::max(std::abs(now(i, j, k)),
                                                                   std::abs(old), 1.0_rt);
                                   });
            }
            dTe_rel = std::max(dTe_rel, diff.norminf());
            auto const& rho = *m_WarpX->m_fields.get(FieldType::rho_fp, lev);
            auto const& frozen = *m_qdsmc_rho_frozen[lev];
            int const midpoint = rho.nComp() / 2;
            amrex::Real const floor = PhysConst::q_e * m_hybrid_pic_model->m_n_floor;
            for (amrex::MFIter mfi(diff, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                auto const delta = diff.array(mfi);
                auto const now = rho.const_array(mfi);
                auto const old = frozen.const_array(mfi);
                amrex::ParallelFor(mfi.tilebox(),
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
                                       delta(i, j, k) =
                                           std::abs(now(i, j, k, midpoint) - old(i, j, k)) /
                                           amrex::max(std::abs(now(i, j, k, midpoint)),
                                                      std::abs(old(i, j, k)), floor, 1.e-100_rt);
                                   });
            }
            drho_rel = std::max(drho_rel, diff.norminf());
        }
        return std::array<amrex::Real, 3>{dPe_rel, dTe_rel, drho_rel};
    };
    auto refresh_density = [&] ()
    {
        for (int lev = 0; lev < m_num_amr_levels; ++lev)
        {
            auto const& rho = *m_WarpX->m_fields.get(FieldType::rho_fp, lev);
            auto& frozen = *m_qdsmc_rho_frozen[lev];
            amrex::MultiFab::Copy(frozen, rho, rho.nComp() / 2, 0, 1, frozen.nGrowVect());
        }
    };
    int exit_status = 3;
    amrex::Real dPe_rel = std::numeric_limits<amrex::Real>::max();
    int outer = 0;
    for (; outer < m_qdsmc_outer_max_iterations; ++outer)
    {

        // Inner solve: E^{n+theta} at frozen electron pressure. Warm-started
        // from the previous outer iterate (m_E carries through).
        amrex::Real const initial_norm = residual_norm();
        if (!std::isfinite(initial_norm))
        {
            return -7;
        }
        if (field_reference == 0.0_rt)
        {
            field_reference = initial_norm;
        }
        amrex::Real const field_target = std::max(field_atol, field_rtol * field_reference);
        if (!(initial_norm == 0.0_rt || initial_norm < field_target))
        {
            m_nlsolver->SetConvergenceReferenceNorm(field_reference);
            m_nlsolver->Solve(m_E, m_Eold, start_time, m_dt, a_step);
            m_nlsolver->SetConvergenceReferenceNorm(0.0_rt);
            exit_status = m_nlsolver->GetExitStatus();
            if (exit_status < 0)
            {
                return exit_status;
            }
            if (exit_status != 2 && exit_status != 3)
            {
                return -7;
            }
        }

        capture_thermal();

        // One stage pass against the converged fields (re-entrant: restarts
        // from the saved t^n state and re-solves the midpoint entropy
        // transport; runs non-probe by construction, so the per-species
        // source deposits refresh once per outer iteration).
        m_hybrid_pic_model->AdvanceElectronEnergyQDSMCTheta(m_dt, m_theta, true);

        auto const defect = thermal_defect();
        refresh_density();
        dPe_rel = defect[0];
        amrex::Real const dTe_rel = defect[1];

        if (m_qdsmc_outer_verbose)
        {
            amrex::Print() << "QDSMC segregated: outer iteration = " << outer
                           << ", dPe/Pe = " << std::scientific << dPe_rel
                           << ", max relative dTe = " << dTe_rel
                           << ", max relative drho = " << defect[2] << "\n";
        }
        dPe_rel = std::max({dPe_rel, dTe_rel, defect[2]});
        if (dPe_rel < m_qdsmc_outer_relative_tolerance)
        {
            // Grade the field equation with the newly emitted thermal stage.
            // This residual skips the energy advance; reconstruct it once
            // afterwards so the finisher owns markers from these deposits.
            capture_thermal();
            amrex::Real const final_norm = residual_norm();
            if (!std::isfinite(final_norm))
            {
                return -7;
            }
            m_hybrid_pic_model->AdvanceElectronEnergyQDSMCTheta(m_dt, m_theta, true);
            auto const final_defect = thermal_defect();
            dPe_rel = std::max({final_defect[0], final_defect[1], final_defect[2]});
            refresh_density();
            if ((final_norm == 0.0_rt || final_norm < field_target) &&
                dPe_rel < m_qdsmc_outer_relative_tolerance)
            {
                break;
            }
        }
    }

    if (outer == m_qdsmc_outer_max_iterations || dPe_rel >= m_qdsmc_outer_relative_tolerance)
    {
        std::stringstream convergenceMsg;
        convergenceMsg << "QDSMC segregated outer loop failed to converge after "
                       << outer << " iterations. Relative pressure change is "
                       << dPe_rel << " and the outer relative tolerance is "
                       << m_qdsmc_outer_relative_tolerance;
        ablastr::warn_manager::WMRecordWarning(
            "ThetaImplicitHybrid", convergenceMsg.str());
        if (m_qdsmc_outer_require_convergence || m_darwin_segregated_solve)
        {
            return -7;
        }
    }

    return exit_status;
}

void ThetaImplicitHybrid::ComputeRHS ( WarpXSolverVec&        a_RHS,
                                       const WarpXSolverVec&  a_E,
                                       amrex::Real            start_time,
                                       int                    a_nl_iter,
                                       bool                   a_from_jacobian )
{
    BL_PROFILE("ThetaImplicitHybrid::ComputeRHS()");

    // The circuit and field stage must agree before particles gather B/E.
    // Replaying only after the push leaves both the moments and Jp stale.
    if (m_darwin_circuit_consistent_stage) {
        ConvergeDarwinCircuitStage(a_E, start_time, a_from_jacobian);
    } else {
        UpdateWarpXFields(a_E, a_from_jacobian, start_time);
    }

    // Split-field circuit-in-the-residual coupling, run BEFORE the
    // particle stage: python measures the flux linkage of THIS iterate's
    // plasma response (disk probes read the theta-stage response B just
    // assembled above), re-advances the coupled external circuit against
    // it, and pushes updated coil scale segments (SetScale); the external
    // fields are then refreshed at the new scales so the particle push
    // AND the Ohm's law below see circuit-consistent fields. Running the
    // coupling after the push (as the darwin branch below does for its
    // boundary pin) would make the gathered E_ext lag the iterate by one
    // evaluation -- hidden history that shows up as an
    // epsilon-independent noise floor in Jacobian secants once the drive
    // is active. Reciprocity (J-based) probes see THIS evaluation's
    // plasma current (refreshed in UpdateWarpXFields from the response
    // field); the circuit ports in use are disk-flux based.
    if (m_external_field_iteration && !m_darwin) {
        AddSplitExternalFields(-1.0_rt);
        ExecutePythonCallback("externalcoiltheta");
        m_hybrid_pic_model->m_external_vector_potential
            ->UpdateHybridExternalFields(start_time + m_theta * m_dt, m_dt);
        AddSplitExternalFields(1.0_rt);
    }

    // Momentum-consistent particle push field: the ions gather Efield_fp,
    // and the solver state deliberately includes the resistive eta*J term
    // (Faraday's law needs it), but the resistive friction must not
    // accelerate the ions through E -- the explicit scheme pushes ions with
    // the no-resistivity Ohm field, and the resistive electron-ion friction
    // is a separate (optional) collision operator. B above already used the
    // full E.
    //
    // The correction is assembled HERE, in every residual evaluation, from
    // THIS iterate: E_push = E_iterate - (E_full - E_nores), with both Ohm
    // solves running INTO Efield_fp so the solve wrapper's boundary stack
    // (PEC image, conformal-EB conductor edges, resistive-shell wall rows)
    // lands identically on both passes -- every term except eta*J (and the
    // hyper-resistive term) then cancels exactly in the difference,
    // including all boundary-controlled rows. Solving the no-resistivity
    // pass into a scratch field instead would leave those rows untreated
    // (the wrapper applies the field boundary to the registered Efield_fp,
    // not to the passed output) and the difference would carry O(1)
    // boundary-row garbage at any eta. A correction lagged from the
    // previous residual evaluation is hidden state that shifts the residual
    // between the base and probe evaluations of the difference Jacobian and
    // stalls Newton at an O(1) relative norm. The pre-push moments feeding
    // the two passes enter the difference only through the eta(rho, |J|)
    // and eta_h(rho, |B|) parametrizations (|J| is the Ampere current, a
    // pure function of the iterate); constant coefficients make the
    // correction a pure function of the iterate.
    if (m_use_resistive_push_correction) {
        using ablastr::fields::Direction;

        // The Darwin branch computes the plasma current after the particle
        // stage; the correction needs it at the iterate's B (already
        // rebuilt above). The later recomputation sees the same B.
        if (m_darwin) {
            m_hybrid_pic_model->CalculatePlasmaCurrent(
                m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1),
                m_WarpX->GetEBUpdateEFlag());
            ApplyDarwinDisplacementCurrent(m_theta * m_dt);
        }

        ablastr::fields::MultiLevelVectorField E_fp =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, m_num_amr_levels - 1);
        ablastr::fields::MultiLevelVectorField J_fp =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, m_num_amr_levels - 1);
        ablastr::fields::MultiLevelVectorField B_fp =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1);
        ablastr::fields::MultiLevelScalarField rho_pre =
            m_WarpX->m_fields.get_mr_levels(FieldType::rho_fp, m_num_amr_levels - 1);

        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            // Park the assembled push field (iterate + BCs + E_ext).
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab & E_res = *m_WarpX->m_fields.get(
                    "hybrid_E_resistive_fp", Direction{dir}, lev);
                amrex::MultiFab const& E = *E_fp[lev][dir];
                amrex::MultiFab::Copy(E_res, E, 0, 0, E.nComp(), E.nGrowVect());
            }
            // Per-level solves: no callback fires inside residual
            // evaluations (see the RHS Ohm solve below).
            m_hybrid_pic_model->HybridPICSolveE(E_fp[lev], J_fp[lev], B_fp[lev],
                                                m_qdsmc_segregated_solve ? *m_qdsmc_rho_frozen[lev]
                                                                         : *rho_pre[lev],
                                                m_WarpX->GetEBUpdateEFlag()[lev], lev,
                                                false, // solve_for_Faraday (retain grad(Pe))
                                                true   // include_resistivity
            );
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab & E_res = *m_WarpX->m_fields.get(
                    "hybrid_E_resistive_fp", Direction{dir}, lev);
                amrex::MultiFab const& E = *E_fp[lev][dir];
                amrex::MultiFab::Subtract(E_res, E, 0, 0, E.nComp(), E.nGrowVect());
            }
            m_hybrid_pic_model->HybridPICSolveE(
                E_fp[lev], J_fp[lev], B_fp[lev],
                m_qdsmc_segregated_solve ? *m_qdsmc_rho_frozen[lev] : *rho_pre[lev],
                m_WarpX->GetEBUpdateEFlag()[lev], lev,
                false, // solve_for_Faraday (retain grad(Pe))
                false  // include_resistivity: no-resistivity push field
            );
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab & E_push = *E_fp[lev][dir];
                amrex::MultiFab const& E_res = *m_WarpX->m_fields.get(
                    "hybrid_E_resistive_fp", Direction{dir}, lev);
                amrex::MultiFab::Add(E_push, E_res, 0, 0, E_push.nComp(), E_push.nGrowVect());
            }
        }
        // The Ohm kernels write valid cells only, so box-boundary ghosts
        // still hold the uncorrected iterate; the gather reads those ghosts.
        amrex::IntVect const ngE = E_fp[0][0]->nGrowVect();
        m_WarpX->FillBoundaryE(ngE, true /* sync nodal points */);
    }

    if (m_darwin && m_hybrid_pic_model->m_darwin_poisson_verbosity > 1) {
        using ablastr::fields::Direction;
        auto ni = [&](const char* nm, int dir) {
            return m_WarpX->m_fields.get(nm, Direction{dir}, 0)->norminf();
        };
        amrex::Print() << "[darwin-eval] iter " << a_nl_iter
            << (a_from_jacobian ? " (jac)" : "")
            << " max|E_push| = " << m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, 0)->norminf()
            << "/" << m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{2}, 0)->norminf()
            << " max|B| = " << m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{1}, 0)->norminf()
            << " max|A| = " << ni("hybrid_A_fp", 0) << "/" << ni("hybrid_A_fp", 2)
            << "\n";
    }

    // Advance particles and deposit J^{n+1/2}, rho^{n+1/2}
    const amrex::Real theta_time = start_time + m_theta * m_dt;
    PreRHSOp( theta_time, a_nl_iter, a_from_jacobian );

    // Get field arrays at all levels
    ablastr::fields::MultiLevelVectorField Efield_fp =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, m_num_amr_levels - 1);
    ablastr::fields::MultiLevelVectorField Bfield_fp =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1);
    ablastr::fields::MultiLevelVectorField current_fp =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, m_num_amr_levels - 1);
    ablastr::fields::MultiLevelScalarField rho_fp =
        m_WarpX->m_fields.get_mr_levels(FieldType::rho_fp, m_num_amr_levels - 1);

    // The split residual must consume THIS evaluation's midpoint density.
    // Component zero is deposited before the particle push and, after the
    // first evaluation, contains the previous evaluation's midpoint state.
    // Ohm's law and the algebraic pressure closure read component zero of
    // their argument, so expose the new midpoint component through an alias.
    // Leave the registered two-time-level density intact for inertia/history.
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> rho_half_store(m_num_amr_levels);
    if (m_darwin_segregated_solve) {
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            auto& rho = *rho_fp[lev];
            rho_half_store[lev] = std::make_unique<amrex::MultiFab>(
                rho, amrex::make_alias, rho.nComp()/2, 1);
            rho_fp[lev] = rho_half_store[lev].get();
        }
    }

    // Compute J_plasma = curl(B^{n+theta})/mu_0. The split-field (non-
    // darwin) branch computed it in UpdateWarpXFields from the plasma-
    // response field, BEFORE the external assembly -- recomputing it here
    // from the total field would re-introduce the spurious O(h^2) external
    // curl. The darwin branch computes it here from its derived
    // B = B_static + curl A, whose curl is discretely consistent (the
    // external flux enters through the evolved A, not a sampled field).
    if (m_darwin) {
        m_hybrid_pic_model->CalculatePlasmaCurrent(Bfield_fp, m_WarpX->GetEBUpdateEFlag());
    }

    if (m_darwin)
    {
        ApplyDarwinDisplacementCurrent(m_theta * m_dt);
    }

    // Circuit-in-the-residual coupling, darwin (unified-drive) branch:
    // python measures the flux linkage of THIS iterate's plasma response,
    // re-advances the coupled external circuit against it, and pushes
    // updated coil scale segments (SetScale). The scales enter through
    // the boundary pin and the vacuum band (interior A is state-only, so
    // the particle stage above needs no re-run). Re-impose the pin at
    // the updated scales and re-derive B. The split-field branch of this
    // coupling runs BEFORE the particle stage (top of this function): its
    // scales enter the gathered push field, which must not lag the
    // iterate.
    if (m_external_field_iteration && m_darwin && !m_darwin_circuit_consistent_stage) {
        if (m_circuit_native) {
            auto& coupler = NativeCircuitCoupler();
            coupler.MeasureDarwinLinkages(theta_time);
            coupler.EvaluateInterval(start_time, start_time + m_dt, false, m_theta*m_dt);
        } else {
            ExecutePythonCallback("externalcoiltheta");
        }
        DarwinApplyABoundary(theta_time);
        if (m_vacuum_recovery_half) {
            // Recompute the recovery against the re-imposed boundary
            // values (live in Jacobian probes too -- Define forced the
            // live-probe mode), then restore the exact pin. NOTE: this
            // is the second recovery application of the evaluation, so
            // a finite darwin_vacuum_recovery_relaxation_time would be
            // double-applied -- circuit decks should run the default
            // (instant) recovery.
            m_hybrid_pic_model->ComputeVacuumARecovery(
                a_from_jacobian, m_theta * m_dt);
            DarwinApplyABoundary(theta_time);
        }
        DarwinDeriveB();
    }

    // Electron inertia: assemble the nodal inertial field from the
    // theta-stage state, ahead of both the E_L constraint solve (which
    // takes its longitudinal part into the source) and the Ohm E-solve
    // (which adds it per component). Refreshed in every evaluation
    // including Jacobian probes -- a smooth function of the state; the Je
    // histories are frozen per step.
    if (m_hybrid_pic_model->m_include_electron_inertia) {
        m_hybrid_pic_model->ComputeElectronInertiaNodal(m_theta, m_dt,
                                                        a_from_jacobian);
    }

    // Electron pressure at t^{n+theta}: either the theta-centered QDSMC
    // electron-energy stage (re-entrant; re-run from the saved t^n state in
    // every residual evaluation, so the nonlinear solver converges the
    // coupled E/T_e system by elimination) or the algebraic adiabatic
    // closure. Per-species deposits feeding the multi-species sources are
    // refreshed once per Newton iteration and frozen during Jacobian
    // evaluations. Under the segregated solve the stage runs once per OUTER
    // iteration from SolveSegregated instead, and the residual consumes the
    // pressure left on the grid unchanged (Pe enters the Ohm solve purely
    // as a field argument, so freezing it is simply not overwriting it).
    if (m_hybrid_pic_model->m_solve_electron_energy_equation) {
        if (!m_qdsmc_segregated_solve) {
            m_hybrid_pic_model->AdvanceElectronEnergyQDSMCTheta(m_dt, m_theta, !a_from_jacobian);
        }
    } else {
        if (m_darwin_segregated_solve) {
            for (int lev = 0; lev < m_num_amr_levels; ++lev) {
                m_hybrid_pic_model->CalculateElectronPressure(lev, *rho_fp[lev]);
            }
        } else {
            m_hybrid_pic_model->CalculateElectronPressure();
        }
    }

    // Darwin: refresh the longitudinal constraint field from this
    // evaluation's electron pressure and midpoint density. The refresh runs
    // in EVERY residual evaluation (including finite-difference Jacobian
    // probes): E_L is a smooth function of the state, so folding it into
    // the probed residual keeps the FD Jacobian consistent with the actual
    // iterate-to-iterate map -- freezing it (as is done for the noisy
    // per-species particle deposits) makes Newton chase a moving target
    // and stall near 50% residuals. The particle push of this evaluation
    // used the previous evaluation's E_L; both agree at convergence. The
    // opt-in segregated solve instead updates E_L only between complete
    // nonlinear solves, so no residual or Jv consumes hidden E_L history.
    if (m_darwin && !m_darwin_segregated_solve) {
        RefreshDarwinELong(theta_time);
    }

    // Solve Ohm's law: E_ohm = f(B^{n+theta}, J_ion^{n+1/2}, rho^{n+1/2}, Pe)
    // Result stored in Efield_fp.
    //
    // The converged E^{n+theta} serves BOTH roles in the implicit scheme: it
    // drives Faraday's law (so the resistive terms must be included or the
    // magnetic field never decays resistively) and it is gathered by the
    // particles (so grad(Pe) must be included for the pressure coupling).
    // The explicit scheme separates these into two E-solves gated by
    // solve_for_Faraday; here the full Ohm E is assembled by overriding the
    // resistive gate, and the resistive part is subtracted again from the
    // push field at the top of this function.
    //
    // Use the per-level HybridPICSolveE: this runs inside every nonlinear
    // residual evaluation (including Jacobian probes), so the multi-level
    // variant's afterEpush python callback must not fire here. A single
    // afterEpush is fired at the converged state after
    // ImplicitSolver::OneStep completes (see WarpX::OneStep).
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        m_hybrid_pic_model->HybridPICSolveE(
            Efield_fp[lev], current_fp[lev], Bfield_fp[lev],
            m_qdsmc_segregated_solve ? *m_qdsmc_rho_frozen[lev] : *rho_fp[lev],
            m_WarpX->GetEBUpdateEFlag()[lev], lev,
            false, // solve_for_Faraday (retain grad(Pe))
            true   // include_resistivity (retain eta*J for the B-update)
        );
    }

    // The Ohm kernels above returned the stored (plasma) field convention:
    // E_ohm computed from the total B, with the inductive E_ext subtracted
    // in plasma cells (inside the plasma the generalized Ohm's law IS the
    // electric field; the external drive reaches it through B). The push
    // field assembly in UpdateWarpXFields re-adds E_ext on top.

    // Darwin: the state is the transverse field, so the fixed point is
    // E_T = E_ohm - E_L. Subtracting E_L from the assembled Ohm field
    // groups the (nearly cancelling) pressure and longitudinal-constraint
    // terms into one small object, and the converged full field
    // E = E_T + E_L satisfies the complete generalized Ohm's law
    // independently of the Helmholtz-projection quality of E_L.
    if (m_darwin) {
        using ablastr::fields::Direction;
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab & E = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
                amrex::MultiFab const & EL = *m_WarpX->m_fields.get("hybrid_E_long_fp", Direction{dir}, lev);
                amrex::MultiFab::Subtract(E, EL, 0, 0, E.nComp(), E.nGrowVect());
            }
        }
        // In the vacuum band the field is defined by the recovered vector
        // potential, E_T = -(A_rec - A^n)/(theta dt), not by the (invalid
        // there) generalized Ohm's law. The FD-Jacobian must not
        // difference through the iterative recovery solve, whose
        // rtol-level noise enters these rows amplified by 1/(theta dt):
        // probe evaluations reuse the correction and the stored target
        // (identity Jacobian rows), and the outer Newton iteration lags
        // the recovery (contraction at the recovery's small leak factor).
        if (m_vacuum_recovery_half) {
            m_hybrid_pic_model->ApplyVacuumFaradayE(
                m_theta * m_dt, false, a_from_jacobian, false);
        }
    }

    // Return RHS = E_ohm - E_old
    // Framework computes residual = E - E_old - RHS = E - E_ohm
    // Convergence: E = E_ohm
    if (std::getenv("WARPX_DEBUG_RESID") != nullptr) {
        using ablastr::fields::Direction;
        for (int dir = 0; dir < 3; ++dir) {
            amrex::MultiFab diff(m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, 0)->boxArray(),
                                 m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, 0)->DistributionMap(), 1, 0);
            amrex::MultiFab::Copy(diff, *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, 0), 0, 0, 1, 0);
            amrex::MultiFab::Subtract(diff, *a_E.getArrayVec()[0][dir], 0, 0, 1, 0);
            auto imax = diff.maxIndex(0);
            amrex::AllPrintToFile("resid_probe") << "iter " << a_nl_iter
                << " dir " << dir << " |dE|max " << diff.norminf(0)
                << " at " << imax << "\n";
        }
    }
    a_RHS.Copy(FieldType::Efield_fp);         // a_RHS = E_ohm
    a_RHS.linComb(1.0, a_RHS, -1.0, m_Eold);  // a_RHS = E_ohm - E_old
}

CircuitCoupler& ThetaImplicitHybrid::NativeCircuitCoupler () const
{
    auto* coupling = m_WarpX->get_pointer_CircuitCoupling();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(coupling != nullptr && coupling->Coupler() != nullptr
        && coupling->Coupler()->Plugin() != nullptr,
        "Native Darwin circuit driver requires circuit.coils, engine=external and plugin_library");
    return *coupling->Coupler();
}

void ThetaImplicitHybrid::CommitNativeCircuitStage (amrex::Real start_time)
{
    BL_PROFILE("ThetaImplicitHybrid::CommitNativeCircuitStage()");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_native_circuit_step_open,
        "Native circuit step is not open");
    if (NativeCircuitCoupler().DeviceTrials()) {
        NativeCircuitCoupler().CommitDarwinDeviceStep(start_time,m_dt,m_theta,
                                                     m_darwin_circuit_scale_tolerance);
        m_native_circuit_step_open = false;
        return;
    }
    auto& external = *m_hybrid_pic_model->m_external_vector_potential;
    auto const time = start_time + m_theta*m_dt;
    amrex::Vector<amrex::Real> candidate(external.nFields());
    for (int field = 0; field < external.nFields(); ++field) {
        candidate[field] = external.TimeScale(field, time);
    }
    auto& coupler = NativeCircuitCoupler();
    coupler.MeasureDarwinLinkages(time);
    // The same stage EMF drives a full-step candidate and its exact acceptance.
    // Do not reinterpret the theta linkage as an endpoint measurement.
    coupler.EvaluateInterval(start_time, start_time + m_dt, true, m_theta*m_dt);
    for (int field = 0; field < external.nFields(); ++field) {
        auto const accepted = external.TimeScale(field, time);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(accepted)
            && std::abs(accepted - candidate[field]) <= m_darwin_circuit_scale_tolerance
                *std::max({1._rt, std::abs(accepted), std::abs(candidate[field])}),
            "Exact circuit acceptance changed the converged stage; rejecting inconsistent commit");
    }
    coupler.FinishStep();
    m_native_circuit_step_open = false;
}

void ThetaImplicitHybrid::RefreshDarwinCircuitCurrent ()
{
    BL_PROFILE("ThetaImplicitHybrid::RefreshDarwinCircuitCurrent()");
    using ablastr::fields::Direction;
    m_hybrid_pic_model->CalculatePlasmaCurrent(
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1),
        m_WarpX->GetEBUpdateEFlag());
    amrex::Real const factor = PhysConst::epsilon_0 / (m_theta * m_dt);
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        for (int dir = 0; dir < 3; ++dir) {
            auto& current = *m_WarpX->m_fields.get(
                FieldType::hybrid_current_fp_plasma, Direction{dir}, lev);
            auto const& longitudinal = *m_WarpX->m_fields.get(
                "hybrid_E_long_fp", Direction{dir}, lev);
            auto const& old = *m_WarpX->m_fields.get(
                "hybrid_E_long_old_fp", Direction{dir}, lev);
            amrex::MultiFab::Saxpy(current, -factor, longitudinal,
                                 0, 0, current.nComp(), current.nGrowVect());
            amrex::MultiFab::Saxpy(current, factor, old,
                                 0, 0, current.nComp(), current.nGrowVect());
        }
    }
}

void ThetaImplicitHybrid::ConvergeDarwinCircuitStage (
    WarpXSolverVec const& electric_field, amrex::Real start_time, bool from_jacobian)
{
    BL_PROFILE("ThetaImplicitHybrid::ConvergeDarwinCircuitStage()");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_circuit_native
        || IsPythonCallbackInstalled("externalcoiltheta"),
        "Darwin circuit stage requires its native plugin or compatibility callback");
    auto& external = *m_hybrid_pic_model->m_external_vector_potential;
    amrex::Real const stage_time = start_time + m_theta * m_dt;
    if (m_circuit_native && NativeCircuitCoupler().DeviceTrials()) {
        auto& coupler = NativeCircuitCoupler();
        coupler.ResetDarwinDeviceTrial();
        // Fixed launch count: convergence and map validity are checked on
        // device, with no circuit vector/status fetch inside the residual.
        for (int iteration = 0; iteration < coupler.DeviceIterations(); ++iteration) {
            UpdateWarpXFields(electric_field,from_jacobian,start_time);
            coupler.AdvanceDarwinDeviceTrial();
        }
        coupler.RequireDarwinDeviceConvergence();
        UpdateWarpXFields(electric_field,from_jacobian,start_time);
        return;
    }

    auto const& accepted = m_darwin_circuit_accepted_scales;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(accepted.size() == external.nFields(),
        "Circuit scales must be captured at the start of the step");
    // A warm start from the preceding residual makes finite fixed-point error
    // depend on probe order. Rebuild each native trial from accepted scales.
    if (m_circuit_native) { NativeCircuitCoupler().ResetDarwinTrialScales(start_time, m_dt); }
    amrex::Vector<amrex::Real> previous(external.nFields());
    for (int iteration = 0; iteration < m_darwin_circuit_max_iterations; ++iteration) {
        // Always rebuild from the same trial E and accepted A, including Jv probes.
        // No particles or circuit accepted-state commits occur inside this loop.
        UpdateWarpXFields(electric_field, from_jacobian, start_time);
        if (!m_circuit_native) { RefreshDarwinCircuitCurrent(); }
        for (int coil = 0; coil < external.nFields(); ++coil) {
            previous[coil] = external.TimeScale(coil, stage_time);
        }
        if (m_circuit_native) {
            auto& coupler = NativeCircuitCoupler();
            coupler.MeasureDarwinLinkages(stage_time);
            coupler.EvaluateInterval(start_time, start_time + m_dt, false, m_theta*m_dt);
        } else {
            ExecutePythonCallback("externalcoiltheta");
        }
        amrex::Real defect = 0._rt;
        for (int coil = 0; coil < external.nFields(); ++coil) {
            amrex::Real const current = external.TimeScale(coil, stage_time);
            amrex::Real const start = external.TimeScale(coil, start_time);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(current) && std::isfinite(start)
                && std::isfinite(previous[coil]), "Nonfinite trial circuit scale");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(start - accepted[coil])
                <= m_darwin_circuit_scale_tolerance * std::max(1._rt, std::abs(accepted[coil])),
                "externalcoiltheta changed the accepted start-of-step coil scale");
            defect = std::max(defect, std::abs(current - previous[coil])
                / std::max({1._rt, std::abs(current), std::abs(previous[coil])}));
        }
        if (defect <= m_darwin_circuit_scale_tolerance) {
            // Use the final published segment, not the preceding fixed-point iterate.
            UpdateWarpXFields(electric_field, from_jacobian, start_time);
            if (!m_circuit_native) { RefreshDarwinCircuitCurrent(); }
            return;
        }
    }
    WARPX_ABORT_WITH_MESSAGE("Darwin circuit stage failed to converge its coil scales");
}

void ThetaImplicitHybrid::UpdateWarpXFields ( const WarpXSolverVec&  a_E,
                                                bool a_from_jacobian,
                                                amrex::Real start_time )
{
    BL_PROFILE("ThetaImplicitHybrid::UpdateWarpXFields()");

    const amrex::Real theta_time = start_time + m_theta * m_dt;

    // Set E^{n+theta} in WarpX (the transverse part E_T on the Darwin path)
    m_WarpX->SetElectricFieldAndApplyBCs( a_E, theta_time );

    // Assemble the external contributions on top of the plasma fields after
    // the respective plasma-field updates below: B_ext at t^{n+theta} for
    // the Ohm kernels and the particle push, and the step-averaged
    // inductive E_ext for the push field. Deliberately applied AFTER the
    // boundary treatments -- the imposed external field must not be
    // altered by the wall conditions.
    auto add_external = [&](warpx::fields::FieldType ftype,
                            warpx::fields::FieldType ext_type) {
        using ablastr::fields::Direction;
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab & F = *m_WarpX->m_fields.get(ftype, Direction{dir}, lev);
                amrex::MultiFab const & F_ext = *m_WarpX->m_fields.get(ext_type, Direction{dir}, lev);
                amrex::MultiFab::Add(F, F_ext, 0, 0, F.nComp(), F.nGrowVect());
            }
        }
    };
    const bool has_external = m_hybrid_pic_model->m_add_external_fields;

    if (m_darwin) {
        // Darwin: advance the vector potential with the transverse field and
        // rebuild B = B_static + curl A (Faraday integrated through A, so B
        // stays solenoidal by construction), then assemble the full
        // E = E_T + E_L for the particle push. E_L is the constraint field
        // refreshed once per nonlinear iteration in ComputeRHS.
        using ablastr::fields::Direction;
        DarwinUpdateA_B( m_theta * m_dt, start_time, a_from_jacobian );
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab & E = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
                amrex::MultiFab const & EL = *m_WarpX->m_fields.get("hybrid_E_long_fp", Direction{dir}, lev);
                amrex::MultiFab::Add(E, EL, 0, 0, E.nComp(), E.nGrowVect());
            }
        }
        // Re-apply the field boundary treatment to the ASSEMBLED field: the
        // boundary/ghost values applied to the transverse state above do not
        // survive the E_L addition (E_L physical ghosts are zero-extended),
        // and wall-adjacent particles gather from those ghost layers. With
        // stale ghosts the wall layer picks up O(E_L) spurious kicks whose
        // density response feeds back through grad(Pe) into E_L -- a
        // divergent per-iteration wall loop in non-periodic directions.
        amrex::IntVect const ngE =
            m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, 0)->nGrowVect();
        m_WarpX->FillBoundaryE(ngE, true /* sync nodal points */);
        m_WarpX->ApplyEfieldBoundary(0, PatchType::fine, theta_time);
    } else {
        // Compute B^{n+theta} = B^n - theta*dt*curl(E^{n+theta}) via Faraday's law
        ablastr::fields::MultiLevelVectorField const& B_old =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, m_num_amr_levels - 1);
        m_WarpX->UpdateMagneticFieldAndApplyBCs( B_old, m_theta * m_dt, start_time );

        // Plasma current from the RESPONSE field, before the external
        // assembly below: J_plasma = curl(B_plasma)/mu0, matching the
        // explicit advance (which computes it from the stripped field).
        // The discrete curl of the stored external field is only O(h^2)
        // zero for a spatially varying coil field, so computing the plasma
        // current from the total field deposits that truncation artifact
        // as a spurious near-boundary current, proportional to the coil
        // scale, whose Hall/resistive Ohm response integrates secularly
        // through Faraday (measured as a linear ride-through drift on the
        // coil-pair vacuum deck; a uniform external A is blind to the
        // defect since its discrete curl vanishes identically).
        m_hybrid_pic_model->CalculatePlasmaCurrent(
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1),
            m_WarpX->GetEBUpdateEFlag());
    }

    if (has_external && !m_darwin) {
        add_external(FieldType::Bfield_fp, FieldType::hybrid_B_fp_external);
        add_external(FieldType::Efield_fp, FieldType::hybrid_E_fp_external);
    }
}

void ThetaImplicitHybrid::DarwinUpdateA_B ( amrex::Real a_thetadt, amrex::Real a_time,
                                            bool a_from_jacobian )
{
    const amrex::Real pin_time = a_time + a_thetadt;
    BL_PROFILE("ThetaImplicitHybrid::DarwinUpdateA_B()");

    using ablastr::fields::Direction;

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        // A^{n+theta} = A_old - theta*dt * E_T (Efield_fp holds E_T here)
        ablastr::fields::VectorField A = m_WarpX->m_fields.get_alldirs("hybrid_A_fp", lev);
        ablastr::fields::VectorField B = m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, lev);
        for (int dir = 0; dir < 3; ++dir) {
            amrex::MultiFab const & A_old = *m_WarpX->m_fields.get("hybrid_A_old_fp", Direction{dir}, lev);
            amrex::MultiFab const & E = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
            amrex::MultiFab::LinComb(*A[dir], 1.0_rt, A_old, 0, -a_thetadt, E, 0,
                                     0, A[dir]->nComp(), A[dir]->nGrowVect());
        }
        // Boundary-driven external flux and embedded conductors act on A
        // itself (see DarwinApplyABoundary).
        DarwinApplyABoundary(pin_time);
        // In-residual vacuum recovery: replace A in masked (vacuum) cells
        // with the magnetostatic solution before deriving B, then restore
        // the exact boundary pin. Runs in every residual evaluation
        // including FD-Jacobian probes -- like E_L, the recovery is a
        // smooth function of the state and freezing it per iteration would
        // make the Jacobian inconsistent with the iterate map.
        if (m_vacuum_recovery_half) {
            m_hybrid_pic_model->ComputeVacuumARecovery(
                a_from_jacobian, a_thetadt);
            DarwinApplyABoundary(pin_time);
        }
        // B = B_static + curl A
        m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(
            B, A, m_WarpX->GetEBUpdateBFlag()[lev], lev);
        for (int dir = 0; dir < 3; ++dir) {
            amrex::MultiFab const & Bs = *m_WarpX->m_fields.get("hybrid_B_static_fp", Direction{dir}, lev);
            amrex::MultiFab::Add(*B[dir], Bs, 0, 0, B[dir]->nComp(), B[dir]->nGrowVect());
        }
    }
    // B is DERIVED here (B = B_static + curl A): boundary conditions act on
    // A (DarwinApplyABoundary) and must not re-condition the curl, or the
    // wall ring picks up values inconsistent with the enclosed-flux pin.
    amrex::IntVect const ngB =
        m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, 0)->nGrowVect();
    m_WarpX->FillBoundaryB(ngB, true /* sync nodal points */);
}

amrex::Array<const amrex::MultiFab*, 3>
ThetaImplicitHybrid::GetBfieldThetaForPC ( const int lev ) const
{
    // During the nonlinear solve, UpdateWarpXFields (called from every
    // residual evaluation) leaves the Bfield_fp registry holding the TOTAL
    // theta-midpoint field B^{n+theta} of the current iterate: the
    // Faraday-advanced plasma field with B_ext^{n+theta} assembled on top
    // (split-field external drive), or B_static + curl A^{n+theta} on the
    // Darwin path. Valid only after the first residual evaluation of the
    // current Newton iterate; before that (and between steps) the registry
    // holds the end-of-step totals B^{n+1} (= B^n at the next entry).
    using ablastr::fields::Direction;
    return { m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, lev),
             m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{1}, lev),
             m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{2}, lev) };
}

amrex::Array<const amrex::MultiFab*, 3>
ThetaImplicitHybrid::GetIonCurrentForPC ( const int lev ) const
{
    // The Ohm solve consumes current_fp as the ion (particle) current
    // (see the HybridPICSolveE call in ComputeRHS). PreRHSOp deposits it
    // each residual evaluation and freezes it during Jacobian probes, so
    // between updatePreCondMat and the GMRES solve it holds exactly the
    // frozen drift-leg coefficient (J - J_i) x delta_B needs.
    using ablastr::fields::Direction;
    return { m_WarpX->m_fields.get(FieldType::current_fp, Direction{0}, lev),
             m_WarpX->m_fields.get(FieldType::current_fp, Direction{1}, lev),
             m_WarpX->m_fields.get(FieldType::current_fp, Direction{2}, lev) };
}

const amrex::MultiFab*
ThetaImplicitHybrid::GetRhoMidForPC ( const int lev ) const
{
    // The rho_fp registry carries two time slots of WarpX::ncomps components
    // each: component 0 holds the pre-push deposit (rho(x^n) only in the
    // first evaluation of a step; the previous evaluation's midpoint
    // positions afterwards), and component nComp()/2 holds the
    // midpoint-position deposit rho^{n+1/2} of the current iterate, written
    // by PreRHSOp's PushParticlesandDeposit in every residual evaluation.
    // Consumers should read component nComp()/2 (the same convention as
    // rho_mid_comp in HybridPICModel and the Darwin rho_half alias in
    // ComputeRHS). Valid only after the first residual evaluation of the
    // current Newton iterate has deposited it.
    return m_WarpX->m_fields.get(FieldType::rho_fp, lev);
}

const amrex::MultiFab*
ThetaImplicitHybrid::GetRhoPolFrozenForPC ( const int lev ) const
{
    // Mirror of the poloidal stage's own density-source selection
    // (HybridPICModel::HybridPICSolveE): the per-step-frozen rho^n
    // snapshot once captured this step, otherwise nullptr so PC callers
    // fall back to the midpoint density exactly as the solve does.
    const HybridPICModel* hybrid = m_hybrid_pic_model;
    if (hybrid == nullptr
        || !hybrid->m_esolve_curlcurl
        || !hybrid->m_curlcurl_pol_frozen_rho
        || !hybrid->m_inertia_rho_n_captured) {
        return nullptr;
    }
    return m_WarpX->m_fields.get("hybrid_rho_n_frozen", lev);
}

namespace
{
    /** Per-node beta = 1/d_e^2 for the divided electron-inertia curl-curl
     *  operator, mirroring the E-solve kernel's floor and taper. Rows where
     *  the kernel zeroes the inertia term (no deposit, or fully tapered)
     *  are identity rows, approximated with the finite ceiling a_beta_id. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real InertiaBetaNode (amrex::Real a_rho, amrex::Real a_rho_floor,
                                 amrex::Real a_floor_w, amrex::Real a_taper_w,
                                 amrex::Real a_beta_fac, amrex::Real a_beta_id)
    {
        using amrex::Real;
        if (a_rho <= Real(0.0)) { return a_beta_id; }
        const Real rho_lim = HybridSmoothFloor(a_rho, a_rho_floor, a_floor_w);
        Real b = a_beta_fac*rho_lim;
        if (a_taper_w > Real(0.0)) {
            const Real tp = Real(0.5)*(Real(1.0)
                + std::tanh((a_rho - a_rho_floor)/a_taper_w));
            b /= amrex::max(tp, Real(1.0e-4));
        }
        // the physical branch needs no ceiling (finite rho -> finite beta;
        // the taper amplification is already capped at 1e4x local); the
        // identity ceiling a_beta_id applies only to the rho <= 0 rows
        return b;
    }
}

const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>*
ThetaImplicitHybrid::FillInertiaBetaCoeff ()
{
    using namespace amrex;

    const HybridPICModel* hybrid = m_hybrid_pic_model;
    if (hybrid == nullptr || !hybrid->m_include_electron_inertia) {
        return nullptr;
    }

#if defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE(
        "jacobian.pc_type = pc_curl_curl_mlmg is not available for the "
        "hybrid solver in RZ (the curl-curl operator carries no cylindrical "
        "metric) - use jacobian.pc_type = pc_block_banded instead.");
    return nullptr; // unreachable
#else
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        hybrid->m_electron_inertia_djedt_only
            && !hybrid->m_electron_inertia_bdf2,
        "pc_curl_curl_mlmg with the hybrid solver models the operator form "
        "of the electron-inertia term: set "
        "hybrid_pic_model.electron_inertia_djedt_only = 1 and "
        "hybrid_pic_model.electron_inertia_bdf2 = 0 (the form of Amano et "
        "al., J. Comput. Phys. 275, 197 (2014)), or use another "
        "preconditioner.");

    const bool vacuum_pc = m_darwin_vacuum_pc_regularization > 0.0_rt;
    if (vacuum_pc) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(AMREX_SPACEDIM == 3 && m_darwin_segregated_solve
            && m_vacuum_recovery_half && !m_external_field_iteration
            && hybrid->m_darwin_vacuum_recovery_operator == "edge_relaxation"
            && hybrid->m_darwin_vacuum_recovery_frozen_mask,
            "Darwin vacuum PC requires 3D split, native half-cadence recovery, frozen mask and no circuit iteration");
    }
    const Real rho_floor = static_cast<Real>(hybrid->m_n_floor)*PhysConst::q_e;
    const Real floor_w =
        static_cast<Real>(hybrid->m_n_floor_smooth_width)*rho_floor;
    const Real taper_w =
        static_cast<Real>(hybrid->m_electron_inertia_floor_taper)*rho_floor;
    const Real me_eff = static_cast<Real>(hybrid->m_electron_inertia_mass);
    // divided operator: beta(x) E + curl curl E = beta(x) b with
    // beta = 1/d_e^2 = mu0 q_e rho_lim / m_e_eff
    const Real beta_fac = PhysConst::mu0*PhysConst::q_e/me_eff;

    if (m_inertia_beta.empty()) {
        m_inertia_beta_owned.resize(m_num_amr_levels);
        m_inertia_beta.resize(m_num_amr_levels);
        if (vacuum_pc) {
            m_inertia_rhs_scale_owned.resize(m_num_amr_levels);
            m_inertia_rhs_scale.resize(m_num_amr_levels);
        }
        const auto& e_mfarrvec = m_E.getArrayVec();
        for (int lev = 0; lev < m_num_amr_levels; lev++) {
            for (int c = 0; c < 3; c++) {
                const MultiFab& emf = *e_mfarrvec[lev][c];
                m_inertia_beta_owned[lev][c] = std::make_unique<MultiFab>(
                    emf.boxArray(), emf.DistributionMap(), 1, 0);
                m_inertia_beta[lev][c] = m_inertia_beta_owned[lev][c].get();
                if (vacuum_pc) {
                    m_inertia_rhs_scale_owned[lev][c] = std::make_unique<MultiFab>(
                        emf.boxArray(), emf.DistributionMap(), 1, 0);
                    m_inertia_rhs_scale[lev][c] = m_inertia_rhs_scale_owned[lev][c].get();
                }
            }
        }
    }

    for (int lev = 0; lev < m_num_amr_levels; lev++) {
        Real vacuum_scale = 0.0_rt;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            const Real dx = m_WarpX->Geom(lev).CellSize(d);
            vacuum_scale += 4.0_rt / (dx*dx);
        }
        const Real vacuum_beta = m_darwin_vacuum_pc_regularization * vacuum_scale;
        const Real mask_floor = rho_floor * hybrid->m_darwin_vacuum_recovery_density_fraction;
        const int mask_mode = hybrid->m_darwin_vacuum_recovery_mask == "global" ? 2
            : (hybrid->m_darwin_vacuum_recovery_mask == "transition" ? 1 : 0);
        const MultiFab* mask_rho = vacuum_pc
            ? m_WarpX->m_fields.get("hybrid_rho_vacmask_fp", lev) : nullptr;
        const MultiFab* rho_mf = GetRhoMidForPC(lev);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rho_mf != nullptr,
            "FillInertiaBetaCoeff: no midpoint density available");
        const int rho_comp = rho_mf->nComp()/2;
        // identity-row ceiling: anchored on the LARGEST density present
        // (anchoring on the floor breaks on decks whose n_floor is tiny
        // relative to the plasma - the ceiling must sit above every
        // physical beta on the level)
        const Real rho_max = rho_mf->max(rho_comp);
        const Real beta_id =
            beta_fac*amrex::max(rho_max, rho_floor)*Real(1.0e4);

        // conformal-wall mirror: the residual zeroes E on covered and cut
        // edges (ZeroConductorEdges inside every Ohm solve), making those
        // Jacobian rows identity -- mirror them with the identity-row beta
        const bool eb_mirror = EB::enabled()
            && hybrid->m_use_conformal_eb
            && hybrid->m_conformal_wall_conductor;
        const auto& eb_update_E = m_WarpX->GetEBUpdateEFlag();

        for (int c = 0; c < 3; c++) {
            MultiFab& bmf = *m_inertia_beta[lev][c];
            // this E component is cell-centered in at most one direction;
            // there the edge value is the harmonic mean of the two
            // neighboring density nodes (harmonic: the depleted node
            // dominates at the density-contrast edge)
            const IntVect etype = bmf.ixType().toIntVect();
            const int ox = (etype[0] == 0) ? 1 : 0;
            const int oy = (AMREX_SPACEDIM >= 2 && etype[1] == 0) ? 1 : 0;
            const int oz = (AMREX_SPACEDIM == 3 && etype[2] == 0) ? 1 : 0;
            const bool has_cc = (ox + oy + oz > 0);
            const Box pc_domain = amrex::convert(m_WarpX->Geom(lev).Domain(), etype);
            const auto pc_lo = pc_domain.smallEnd();
            const auto pc_hi = pc_domain.bigEnd();
            GpuArray<int,AMREX_SPACEDIM> pin_normal{};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                pin_normal[d] = !etype[d] && !m_WarpX->Geom(lev).isPeriodic(d)
                    && (hybrid->m_add_external_fields || EB::enabled());
            }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(bmf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box bx = mfi.tilebox();
                const auto beta_arr = bmf.array(mfi);
                const auto rhs_scale = vacuum_pc ? m_inertia_rhs_scale[lev][c]->array(mfi)
                    : amrex::Array4<Real>{};
                const auto mask_arr = vacuum_pc ? mask_rho->const_array(mfi)
                    : amrex::Array4<Real const>{};
                const auto rho_arr = rho_mf->const_array(mfi, rho_comp);
                const auto eb_arr = eb_mirror
                    ? eb_update_E[lev][c]->const_array(mfi)
                    : amrex::Array4<int const>{};
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (eb_mirror && eb_arr(i,j,k) == 0) {
                        beta_arr(i,j,k) = beta_id;
                        if (vacuum_pc) { rhs_scale(i,j,k) = beta_id; }
                        return;
                    }
                    const Real ba = InertiaBetaNode(rho_arr(i,j,k),
                        rho_floor, floor_w, taper_w, beta_fac, beta_id);
                    Real bv = ba;
                    if (has_cc) {
                        const Real bb = InertiaBetaNode(
                            rho_arr(i+ox, j+oy, k+oz),
                            rho_floor, floor_w, taper_w, beta_fac, beta_id);
                        bv = Real(2.0)*ba*bb/(ba + bb);
                    }
                    beta_arr(i,j,k) = bv;
                    if (vacuum_pc) {
                        // Match the arithmetic nodal-to-edge mask interpolation
                        // used by native recovery and the Faraday overwrite.
                        const Real rho_edge = has_cc ? Real(0.5)*(mask_arr(i,j,k)
                            + mask_arr(i+ox,j+oy,k+oz)) : mask_arr(i,j,k);
                        const bool in_mask = mask_mode == 2
                            || (mask_mode == 0 && rho_edge < mask_floor)
                            || (mask_mode == 1 && rho_edge > 0.0_rt && rho_edge < mask_floor);
                        // The native vacuum Jacobian is K/Lambda. Regularize
                        // only its PC: (K + eps Lambda I)x = Lambda b.
                        beta_arr(i,j,k) = in_mask ? vacuum_beta : bv;
                        rhs_scale(i,j,k) = in_mask ? vacuum_scale : bv;
                        if (in_mask) {
                            // Darwin A pins include the first/last normal,
                            // cell-centered DOFs; MLCurlCurl PEC does not.
                            const IntVect iv(AMREX_D_DECL(i,j,k));
                            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                                if (pin_normal[d] && (iv[d] == pc_lo[d] || iv[d] == pc_hi[d])) {
                                    const Real pin_beta = amrex::max(beta_id, Real(1.0e4)*vacuum_scale);
                                    beta_arr(i,j,k) = pin_beta;
                                    rhs_scale(i,j,k) = pin_beta;
                                }
                            }
                        }
                    }
                });
            }
        }
    }
    return &m_inertia_beta;
#endif
}

void ThetaImplicitHybrid::ProjectDarwinVacuumGauge (WarpXSolverVec& field, bool preconditioner)
{
    if (!m_darwin_vacuum_gauge_projection) { return; }
#if defined(WARPX_DIM_3D)
    BL_PROFILE("ThetaImplicitHybrid::ProjectDarwinVacuumGauge()");
    using namespace amrex;
    constexpr int lev = 0;
    auto const& geom = m_WarpX->Geom(lev);
    auto const& period = geom.periodicity();
    auto const dx = geom.CellSizeArray();
    auto& rho = *m_WarpX->m_fields.get("hybrid_rho_vacmask_fp", lev);
    auto const& shape = *m_WarpX->m_fields.get("hybrid_phi_darwin_fp", lev);
    auto const& vectors = field.getArrayVec()[lev];
    if (!m_vacuum_gauge_solver) {
        rho.OverrideSync(period);
        rho.FillBoundary(period);
        m_vacuum_gauge_mask = std::make_unique<iMultiFab>(
            shape.boxArray(), shape.DistributionMap(), 1, 0);
        m_vacuum_gauge_phi = std::make_unique<MultiFab>(
            shape.boxArray(), shape.DistributionMap(), 1, 1);
        m_vacuum_gauge_rhs = std::make_unique<MultiFab>(
            shape.boxArray(), shape.DistributionMap(), 1, 0);
        Real const floor = PhysConst::q_e * m_hybrid_pic_model->m_n_floor
            * m_hybrid_pic_model->m_darwin_vacuum_recovery_density_fraction;
        int const mode = m_hybrid_pic_model->m_darwin_vacuum_recovery_mask == "global" ? 2
            : (m_hybrid_pic_model->m_darwin_vacuum_recovery_mask == "transition" ? 1 : 0);
        auto const dom = surroundingNodes(geom.Domain());
        auto const lo = dom.smallEnd();
        auto const hi = dom.bigEnd();
        GpuArray<int,3> per{geom.isPeriodic(0),geom.isPeriodic(1),geom.isPeriodic(2)};
        bool const use_eb = EB::enabled();
        auto const& flags = m_WarpX->GetEBUpdateEFlag();
        for (MFIter mfi(*m_vacuum_gauge_mask); mfi.isValid(); ++mfi) {
            auto const mask = m_vacuum_gauge_mask->array(mfi);
            auto const rr = rho.const_array(mfi);
            GpuArray<Array4<int const>,3> eb{};
            if (use_eb) {
                for (int d = 0; d < 3; ++d) { eb[d] = flags[lev][d]->const_array(mfi); }
            }
            ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                IntVect const iv(i,j,k);
                // phi=0 on the wall AND its neighboring nodal layer keeps
                // grad(phi)=0 on all pinned tangential and normal A rows.
                for (int d = 0; d < 3; ++d) {
                    if (!per[d] && (iv[d] <= lo[d]+1 || iv[d] >= hi[d]-1)) {
                        mask(iv) = 0; return;
                    }
                }
                // Permit a potential only if all incident edges belong to
                // the unconstrained vacuum. Its gradient is then zero on
                // plasma/covered rows and lies in the native curl nullspace.
                for (int d = 0; d < 3; ++d) {
                    IntVect off(0); off[d] = 1;
                    for (int side = 0; side < 2; ++side) {
                        IntVect const edge = side ? iv-off : iv;
                        Real const re = Real(0.5)*(rr(edge)+rr(edge+off));
                        bool const vac = mode == 2 || (mode == 0 && re < floor)
                            || (mode == 1 && re > 0.0_rt && re < floor);
                        if (!vac || (use_eb && eb[d](edge) == 0)) {
                            mask(iv) = 0; return;
                        }
                    }
                }
                mask(iv) = 1;
            });
        }
        m_vacuum_gauge_mask->OverrideSync(period);
        LPInfo const info;
        m_vacuum_gauge_op = std::make_unique<MLNodeTensorLaplacian>(
            Vector<Geometry>{geom}, Vector<BoxArray>{m_WarpX->boxArray(lev)},
            Vector<DistributionMapping>{m_WarpX->DistributionMap(lev)}, info);
        m_vacuum_gauge_op->setSigma({1._rt,0._rt,0._rt,1._rt,0._rt,1._rt});
        Array<LinOpBCType,3> lo_bc, hi_bc;
        for (int d = 0; d < 3; ++d) {
            lo_bc[d] = hi_bc[d] = per[d] ? LinOpBCType::Periodic : LinOpBCType::Dirichlet;
        }
        m_vacuum_gauge_op->setDomainBC(lo_bc,hi_bc);
        // With no known nodes in a fully periodic vacuum, leave AMReX's
        // singular/nullspace handling active rather than attaching an all-1 mask.
        if (m_vacuum_gauge_mask->min(0) == 0) {
            m_vacuum_gauge_op->setOversetMask(lev,*m_vacuum_gauge_mask);
        }
        m_vacuum_gauge_solver = std::make_unique<MLMG>(*m_vacuum_gauge_op);
        m_vacuum_gauge_solver->setVerbose(0);
        m_vacuum_gauge_solver->setMaxIter(200);
    }
    // Solver vectors have no ghosts. Divergence needs neighboring edges,
    // including those across box/rank boundaries, so use owned ghosted scratch.
    for (int d = 0; d < 3; ++d) {
        auto& edge = m_vacuum_gauge_edges[d];
        if (!edge.isDefined()) {
            edge.define(vectors[d]->boxArray(), vectors[d]->DistributionMap(), 1, 1);
        }
        edge.setVal(0.0_rt);
        MultiFab::Copy(edge,*vectors[d],0,0,1,0);
        edge.OverrideSync(period);
        edge.FillBoundary(period);
    }
    ablastr::fields::VectorField vf{
        &m_vacuum_gauge_edges[0],&m_vacuum_gauge_edges[1],&m_vacuum_gauge_edges[2]};
    auto& rhs = *m_vacuum_gauge_rhs;
    auto& phi = *m_vacuum_gauge_phi;
    m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeDivE(vf,rhs);
    for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
        auto const rr = rhs.array(mfi);
        auto const mask = m_vacuum_gauge_mask->const_array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
            if (mask(i,j,k) == 0) { rr(i,j,k) = 0.0_rt; }
        });
    }
    rhs.OverrideSync(period);
    phi.setVal(0.0_rt);
    // Only preconditioner corrections may use a cheaper projection. The split
    // solution and warm start retain the strict gauge used by the field gates.
    Real const rtol = preconditioner ? m_darwin_vacuum_gauge_pc_rtol : 1.e-12_rt;
    m_vacuum_gauge_solver->solve({&phi},{&rhs},rtol,0.0_rt);
    phi.OverrideSync(period);
    phi.FillBoundary(period);
    for (int d = 0; d < 3; ++d) {
        auto& v = *vectors[d];
        IntVect off(0); off[d] = 1;
        Real const idx = 1.0_rt/dx[d];
        for (MFIter mfi(v); mfi.isValid(); ++mfi) {
            auto const vv = v.array(mfi);
            auto const pp = phi.const_array(mfi);
            ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                IntVect const iv(i,j,k);
                vv(iv) -= (pp(iv+off)-pp(iv))*idx;
            });
        }
        v.OverrideSync(period);
        v.FillBoundary(period);
    }
#else
    amrex::ignore_unused(field, preconditioner);
    WARPX_ABORT_WITH_MESSAGE("darwin_vacuum_gauge_projection requires 3D");
#endif
}

void ThetaImplicitHybrid::DarwinDeriveB ()
{
    using ablastr::fields::Direction;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        ablastr::fields::VectorField A =
            m_WarpX->m_fields.get_alldirs("hybrid_A_fp", lev);
        ablastr::fields::VectorField B =
            m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, lev);
        m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(
            B, A, m_WarpX->GetEBUpdateBFlag()[lev], lev);
        for (int dir = 0; dir < 3; ++dir) {
            amrex::MultiFab const & Bs = *m_WarpX->m_fields.get(
                "hybrid_B_static_fp", Direction{dir}, lev);
            amrex::MultiFab::Add(*B[dir], Bs, 0, 0,
                                 B[dir]->nComp(), B[dir]->nGrowVect());
        }
    }
    // Derived B: boundary conditions act on A (DarwinApplyABoundary), so
    // only exchange/sync the ghosts of the derived field.
    amrex::IntVect const ngB =
        m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, 0)->nGrowVect();
    m_WarpX->FillBoundaryB(ngB, true /* sync nodal points */);
}

void ThetaImplicitHybrid::DarwinApplyABoundary ( amrex::Real a_time )
{
    using ablastr::fields::Direction;
    constexpr int NODE = amrex::IndexType::NODE;

    const bool has_external = m_hybrid_pic_model->m_add_external_fields;
    const bool has_eb = !m_WarpX->GetEBUpdateEFlag().empty()
        && m_WarpX->GetEBUpdateEFlag()[0][0] != nullptr;
    if (!has_external && !has_eb) { return; }

    // Gauge: A was zeroed at initialization, so boundary values impose the
    // CHANGE of the external vector potential since then.
    amrex::Vector<amrex::Real> scales;
    if (has_external) {
        auto & ext = *m_hybrid_pic_model->m_external_vector_potential;
        if (m_fext_init.empty()) {
            for (int i = 0; i < ext.nFields(); ++i) {
                m_fext_init.push_back(ext.TimeScale(i, a_time));
            }
        }
        for (int i = 0; i < ext.nFields(); ++i) {
            scales.push_back(ext.DeviceDriven(i) ? 0. : ext.TimeScale(i, a_time) - m_fext_init[i]);
        }
    }

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const amrex::Box& domain = m_WarpX->Geom(lev).Domain();
        const amrex::Periodicity& period = m_WarpX->Geom(lev).periodicity();

        for (int dir = 0; dir < 3; ++dir) {
            amrex::MultiFab & A = *m_WarpX->m_fields.get("hybrid_A_fp", Direction{dir}, lev);

            // Sum of the (gauge-shifted) external vector potentials on this
            // component's staggering.
            amrex::MultiFab A_bc(A.boxArray(), A.DistributionMap(), 1, A.nGrowVect());
            A_bc.setVal(0.0_rt);
            if (has_external) {
                auto & ext = *m_hybrid_pic_model->m_external_vector_potential;
                for (int i = 0; i < ext.nFields(); ++i) {
                    amrex::MultiFab const & Aext = *m_WarpX->m_fields.get(
                        ext.FieldName(i) + "_Aext", Direction{dir}, lev);
                    auto const ng = amrex::min(A.nGrowVect(), Aext.nGrowVect());
                    if (ext.DeviceDriven(i)) {
                        auto const device = ext.DeviceScales();
                        double const initial = m_fext_init[i];
                        int const field = i;
                        for (amrex::MFIter mfi(A_bc,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                            auto const dst = A_bc.array(mfi); auto const unit = Aext.const_array(mfi);
                            amrex::ParallelFor(mfi.growntilebox(ng),
                                [=] AMREX_GPU_DEVICE(int ii,int jj,int kk) noexcept {
                                    dst(ii,jj,kk) += (device.Value(field,a_time)-initial)*unit(ii,jj,kk);
                                });
                        }
                    } else {
                        amrex::MultiFab::Saxpy(A_bc,scales[i],Aext,0,0,1,ng);
                    }
                }
            }

            if (std::getenv("WARPX_DEBUG_ABC") != nullptr) {
                amrex::Print() << "[A-bc] t=" << a_time << " dir " << dir
                    << " |A_bc|max = " << A_bc.norminf(0)
                    << " scale0 = " << (scales.empty() ? 0.0 : scales[0])
                    << " |A|max = " << A.norminf(0) << "\n";
            }

            const amrex::iMultiFab* eb_flag = has_eb
                ? m_WarpX->GetEBUpdateEFlag()[lev][dir].get() : nullptr;

            for (amrex::MFIter mfi(A, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                amrex::Box tb = mfi.tilebox();
                tb.grow(A.nGrowVect());
                const amrex::Box domain_t = amrex::convert(domain, A.ixType().toIntVect());

                amrex::Array4<amrex::Real> const& a = A.array(mfi);
                amrex::Array4<amrex::Real const> const& abc = A_bc.const_array(mfi);
                amrex::Array4<int const> eb;
                if (eb_flag) { eb = eb_flag->const_array(mfi); }
                const bool use_eb = (eb_flag != nullptr);

                amrex::GpuArray<int, 3> dlo{{0, 0, 0}};
                amrex::GpuArray<int, 3> dhi{{0, 0, 0}};
                amrex::GpuArray<int, 3> per{{1, 1, 1}};
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    dlo[d] = domain_t.smallEnd(d);
                    dhi[d] = domain_t.bigEnd(d);
                    per[d] = period.isPeriodic(d) ? 1 : 0;
                }

#if defined(WARPX_DIM_RZ)
                bool const on_axis = m_WarpX->Geom(lev).ProbLo(0) == 0.;
#endif
                amrex::ParallelFor(tb,
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
#if defined(WARPX_DIM_RZ)
                                       if (on_axis && dir == 1 && i == 0)
                                       {
                                           a(i, j, k) = 0.;
                                           return;
                                       }
#endif
                                       // Embedded conductors: hold A at the gauge zero inside
                                       // masked cells (frozen enclosed flux; the interior field
                                       // stays at B_static).
                                       if (use_eb && eb(i, j, k) == 0)
                                       {
                                           a(i, j, k) = 0.0_rt;
                                           return;
                                       }
                                       // Non-periodic domain boundaries: impose the external
                                       // vector potential on the boundary point and everything
                                       // beyond it.
                                       const int idx[3] = {i, j, k};
                                       bool on_boundary = false;
                                       for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                       {
                                           if (per[d])
                                           {
                                               continue;
                                           }
                                           bool lower = idx[d] <= dlo[d];
#if defined(WARPX_DIM_RZ)
                                           if (d == 0 && on_axis)
                                           {
                                               lower = false;
                                           }
#endif
                                           if (lower || idx[d] >= dhi[d])
                                           {
                                               on_boundary = true;
                                           }
                                       }
                                       if (on_boundary)
                                       {
                                           // Clamp the imposed value to the domain edge:
                                           // ghosts continue the wall value rather than the
                                           // (growing) exterior vector potential, so the wall
                                           // ring carries no spurious curl sheet.
                                           int ic[3] = {i, j, k};
                                           for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                           {
                                               if (per[d])
                                               {
                                                   continue;
                                               }
#if defined(WARPX_DIM_RZ)
                                               if (d == 0 && on_axis && ic[d] < dlo[d])
                                               {
                                                   continue;
                                               }
#endif
                                               ic[d] = amrex::Clamp(ic[d], dlo[d], dhi[d]);
                                           }
                                           a(i, j, k) = abc(ic[0], ic[1], ic[2]);
                                       }
                                   });
            }
            A.FillBoundary(m_WarpX->Geom(lev).periodicity());

        }
#if defined(WARPX_DIM_RZ)
        if (m_WarpX->Geom(lev).ProbLo(0) == 0.)
        {
            auto A = m_WarpX->m_fields.get_alldirs("hybrid_A_fp", lev);
            m_WarpX->ApplyFieldBoundaryOnAxis(A[0], A[1], A[2], lev);
        }
#endif
    }
    amrex::ignore_unused(NODE);
}

void ThetaImplicitHybrid::FinishFieldUpdate( amrex::Real end_time )
{
    BL_PROFILE("ThetaImplicitHybrid::FinishFieldUpdate()");

    // Extrapolate from t^{n+theta} to t^{n+1}:
    // F^{n+1} = (1/theta)*F^{n+theta} + (1 - 1/theta)*F^n
    const amrex::Real c0 = 1.0_rt / m_theta;
    const amrex::Real c1 = 1.0_rt - c0;

    // E^{n+1} (the transverse part on the Darwin path)
    m_E.linComb( c0, m_E, c1, m_Eold );
    m_WarpX->SetElectricFieldAndApplyBCs( m_E, end_time );

    if (m_darwin) {
        using ablastr::fields::Direction;
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                // A^{n+1} = A_old + (A^{n+theta} - A_old)/theta
                //         = A_old - dt E_T^{n+theta}
                amrex::MultiFab & A = *m_WarpX->m_fields.get("hybrid_A_fp", Direction{dir}, lev);
                amrex::MultiFab const & A_old = *m_WarpX->m_fields.get("hybrid_A_old_fp", Direction{dir}, lev);
                amrex::MultiFab::LinComb(A, c0, A, 0, c1, A_old, 0,
                                         0, A.nComp(), A.nGrowVect());
                // E_L^{n+1} = (E_L^{n+theta} - (1-theta) E_L^n)/theta
                amrex::MultiFab & EL = *m_WarpX->m_fields.get("hybrid_E_long_fp", Direction{dir}, lev);
                amrex::MultiFab const & EL_old = *m_WarpX->m_fields.get("hybrid_E_long_old_fp", Direction{dir}, lev);
                amrex::MultiFab::LinComb(EL, c0, EL, 0, c1, EL_old, 0,
                                         0, EL.nComp(), EL.nGrowVect());
                // Full E^{n+1} = E_T^{n+1} + E_L^{n+1} (SetElectricFieldAndApplyBCs
                // above wrote the transverse part into Efield_fp)
                amrex::MultiFab & E = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
                amrex::MultiFab::Add(E, EL, 0, 0, E.nComp(), E.nGrowVect());
            }
        }
        if (m_external_field_iteration && !m_circuit_native) {
            // Final circuit pass against the converged theta-stage plasma
            // current (hybrid_current_fp_plasma as left by the last
            // residual evaluation), leaving the circuit state -- and the
            // coil scale segments the pin below reads -- at t^{n+1}.
            ExecutePythonCallback("externalcoilfinish");
        }
        DarwinApplyABoundary(end_time);
        // Vacuum recovery at the full-step state (both cadences: in "half"
        // mode the theta-stage was recovered inside the solve, and the
        // extrapolated end state gets the same treatment so the delivered
        // field is exactly recovered; in "full" mode this is the only
        // application). Restore the exact boundary pin afterwards.
        if (m_vacuum_recovery) {
            // Half cadence already relaxed over theta*dt inside the
            // solve; the end application covers the remaining
            // (1 - theta)*dt so the configured tau is the effective
            // response time in both cadences.
            m_hybrid_pic_model->ComputeVacuumARecovery(false,
                m_vacuum_recovery_half ? (1.0_rt - m_theta) * m_dt
                                       : m_dt);
            DarwinApplyABoundary(end_time);
            // Recover the endpoint transverse electric field spatially in 3D/RZ.
            // The implicit current and theta-stage updates are unchanged.
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
            auto const& endpoint = m_E.getArrayVec()[0];
            auto const& accepted = m_Eold.getArrayVec()[0];
            RecoverDarwinVacuumE(*m_WarpX, *m_hybrid_pic_model,
                {endpoint[0], endpoint[1], endpoint[2]},
                {accepted[0], accepted[1], accepted[2]}, end_time, m_dt);
#else
            m_hybrid_pic_model->ApplyVacuumFaradayE(m_dt, true, false,
                                                    true /* BDF2 */);
            // Rotate the A history for the next step's BDF2: A^n becomes
            // A^{n-1} (A_old still holds A^n here; it is rewritten from the
            // end state at the next OneStep entry).
            for (int lev = 0; lev < m_num_amr_levels; ++lev) {
                for (int dir = 0; dir < 3; ++dir) {
                    amrex::MultiFab & Anm1 = *m_WarpX->m_fields.get(
                        "hybrid_A_vac_nm1_fp", Direction{dir}, lev);
                    amrex::MultiFab const & Aold = *m_WarpX->m_fields.get(
                        "hybrid_A_old_fp", Direction{dir}, lev);
                    amrex::MultiFab::Copy(Anm1, Aold, 0, 0, Anm1.nComp(),
                                          Anm1.nGrowVect());
                }
            }
#endif
        }
        // B^{n+1} = B_static + curl A^{n+1}
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            ablastr::fields::VectorField A = m_WarpX->m_fields.get_alldirs("hybrid_A_fp", lev);
            ablastr::fields::VectorField B = m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, lev);
            m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(
                B, A, m_WarpX->GetEBUpdateBFlag()[lev], lev);
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab const & Bs = *m_WarpX->m_fields.get("hybrid_B_static_fp", Direction{dir}, lev);
                amrex::MultiFab::Add(*B[dir], Bs, 0, 0, B[dir]->nComp(), B[dir]->nGrowVect());
            }
        }
        // Derived B: no independent boundary conditioning (see
        // DarwinUpdateA_B).
        amrex::IntVect const ngB =
            m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, 0)->nGrowVect();
        m_WarpX->FillBoundaryB(ngB, true /* sync nodal points */);
    } else {
        // The residual evaluations (and the post-solve UpdateWarpXFields)
        // leave Bfield_fp holding the TOTAL theta-time field, with
        // B_ext^{n+theta} assembled on top of the Faraday-advanced plasma
        // field, while B_old holds the plasma response alone. Strip the
        // still-stored theta-time external before the theta-extrapolation:
        // extrapolating the total against the plasma B_old amplifies
        // B_ext^{n+theta} by 1/theta, and with B_ext^{n+1} restored below
        // the carried plasma field would gain a spurious
        // (1/theta)*B_ext^{n+theta} every step (the vacuum-ramp deck
        // integrates the programmed ramp to a ~300x overshoot).
        ExtLedgerPrint(m_WarpX, "finish pre-theta-strip");
        if (m_hybrid_pic_model->m_add_external_fields) {
            using ablastr::fields::Direction;
            for (int lev = 0; lev < m_num_amr_levels; ++lev) {
                for (int dir = 0; dir < 3; ++dir) {
                    amrex::MultiFab & B = *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{dir}, lev);
                    amrex::MultiFab const & B_ext = *m_WarpX->m_fields.get(FieldType::hybrid_B_fp_external, Direction{dir}, lev);
                    amrex::MultiFab::Subtract(B, B_ext, 0, 0, B.nComp(), B.nGrowVect());
                }
            }
        }
        ExtLedgerPrint(m_WarpX, "finish post-theta-strip");
        // B^{n+1}
        ablastr::fields::MultiLevelVectorField const& B_old =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, 0);
        m_WarpX->FinishMagneticFieldAndApplyBCs( B_old, m_theta, end_time );
        ExtLedgerPrint(m_WarpX, "finish post-extrap");
    }

    // (The electron-inertia Je history rotates in Advance, AFTER the
    // delivered-state plasma-current refresh: the stored history values
    // must be MEASURED end-of-step assemblies, never extrapolations --
    // storing the extrapolation (Je^theta - (1-theta) Je^n)/theta feeds
    // the stored value back into itself with eigenvalue -(1-theta)/theta,
    // which is marginal (-1) at theta = 1/2 and rings at period 2 where
    // the inertia term dominates the Ohm law.)

    // Restore end-of-step totals: the analytic external flux advance means
    // Bfield_fp = B_plasma^{n+1} + f(t^{n+1}) curl A_ext exactly, for any
    // ramp shape (OneStep strips the same values at the next entry).
    // Split-field form only: under the Darwin unified drive the external
    // flux already lives inside A through its boundary values, and adding
    // E_ext here would poison the saved E^n and re-inject the drive
    // volumetrically through the A rebuild (doubling the programmed flux).
    if (m_hybrid_pic_model->m_add_external_fields && !m_darwin) {
        using ablastr::fields::Direction;
        if (m_external_field_iteration) {
            // Final circuit pass against the accepted end-of-step plasma
            // response (Bfield_fp still holds the response-only field
            // here), leaving the circuit state -- and the coil segments
            // the refresh below reads -- at t^{n+1}.
            ExecutePythonCallback("externalcoilfinish");
        }
        m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
            end_time, m_dt);
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab & B = *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{dir}, lev);
                amrex::MultiFab const & B_ext = *m_WarpX->m_fields.get(FieldType::hybrid_B_fp_external, Direction{dir}, lev);
                amrex::MultiFab::Add(B, B_ext, 0, 0, B.nComp(), B.nGrowVect());
                amrex::MultiFab & E = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
                amrex::MultiFab const & E_ext = *m_WarpX->m_fields.get(FieldType::hybrid_E_fp_external, Direction{dir}, lev);
                amrex::MultiFab::Add(E, E_ext, 0, 0, E.nComp(), E.nGrowVect());
            }
        }
        ExtLedgerPrint(m_WarpX, "finish post-restore");
    }
}

void ThetaImplicitHybrid::AddSplitExternalFields ( amrex::Real a_sign )
{
    using ablastr::fields::Direction;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        for (int dir = 0; dir < 3; ++dir) {
            amrex::MultiFab & B = *m_WarpX->m_fields.get(
                FieldType::Bfield_fp, Direction{dir}, lev);
            amrex::MultiFab const & B_ext = *m_WarpX->m_fields.get(
                FieldType::hybrid_B_fp_external, Direction{dir}, lev);
            amrex::MultiFab::Saxpy(B, a_sign, B_ext, 0, 0,
                                   B.nComp(), B.nGrowVect());
            amrex::MultiFab & E = *m_WarpX->m_fields.get(
                FieldType::Efield_fp, Direction{dir}, lev);
            amrex::MultiFab const & E_ext = *m_WarpX->m_fields.get(
                FieldType::hybrid_E_fp_external, Direction{dir}, lev);
            amrex::MultiFab::Saxpy(E, a_sign, E_ext, 0, 0,
                                   E.nComp(), E.nGrowVect());
        }
    }
}
