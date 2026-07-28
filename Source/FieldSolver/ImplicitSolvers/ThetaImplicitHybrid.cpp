/* Copyright 2026 Prabhat Kumar
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Fields.H"
#include "ThetaImplicitHybrid.H"
#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"
#include <ablastr/utils/Communication.H>

using warpx::fields::FieldType;
using namespace amrex::literals;

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

    m_E.Define( m_WarpX, "Efield_fp" );
    m_Eold.Define( m_E );

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

    // Resistive part of the Ohm's-law field, subtracted from the
    // particle-push field (see ComputeRHS). Only allocated when the opt-in
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

    const amrex::ParmParse pp("implicit_evolve");
    pp.query("theta", m_theta);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_theta >= 0.5 && m_theta <= 1.0,
        "theta parameter must be between 0.5 and 1.0");

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
        // Mid-step values used throughout the nonlinear solve: B_ext at
        // t^{n+theta}, and E_ext = -[f(t^{n+theta}+dt/2) -
        // f(t^{n+theta}-dt/2)]/dt * A for the push field (at theta = 1/2
        // this is the exact step mean of the inductive field).
        ext.UpdateHybridExternalFields(start_time + m_theta*a_dt, a_dt);
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
        m_Eold.Copy(FieldType::Efield_fp);
    } else {
        m_Eold.Copy(FieldType::E_old, FieldType::None, true);
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
        m_hybrid_pic_model->QDSMCSaveImplicitStepStart();
    }

    // Initial guess: E^{n+theta} = E^n
    m_E.Copy(m_Eold);

    // Solve nonlinear system for E^{n+theta} (and eventually Pe^{n+theta})
    m_nlsolver->Solve( m_E, m_Eold, start_time, m_dt, a_step );

    const int exit_status = m_nlsolver->GetExitStatus();
    if (exit_status < 0) { return exit_status; }

    // Update WarpX fields to t^{n+theta}
    UpdateWarpXFields( m_E, start_time );
    m_WarpX->reduced_diags->ComputeDiagsMidStep(a_step);

    const amrex::Real new_time = start_time + m_dt;

    // Advance particles from t^{n+1/2} to t^{n+1}
    m_WarpX->FinishImplicitParticleUpdate(new_time);

    // Advance fields from t^{n+theta} to t^{n+1}
    FinishFieldUpdate( new_time );

    // Complete the electron-energy step: apply the stochastic ion-heating
    // realization once with converged states, refresh Pe^{n+1}, and reset
    // the QDSMC markers.
    if (m_hybrid_pic_model->m_solve_electron_energy_equation) {
        m_hybrid_pic_model->QDSMCFinishImplicitStep(m_dt, m_theta);
    }

    // Refresh the per-species temperature deposits from the end-of-step
    // particle state (after the ion-heating realization above). The
    // explicit scheme deposits these every step; without this call the
    // T_<species> diagnostics would hold their initialization values for
    // the whole run. Species without do_temperature_deposition are
    // skipped inside.
    m_WarpX->GetPartContainer().DepositTemperatures(m_WarpX->m_fields, 0.0_rt);

    return exit_status;
}

void ThetaImplicitHybrid::ComputeRHS ( WarpXSolverVec&        a_RHS,
                                       const WarpXSolverVec&  a_E,
                                       amrex::Real            start_time,
                                       int                    a_nl_iter,
                                       bool                   a_from_jacobian )
{
    BL_PROFILE("ThetaImplicitHybrid::ComputeRHS()");

    // Update B^{n+theta} from current E estimate via Faraday's law
    UpdateWarpXFields( a_E, start_time );

    // Momentum-consistent particle push field: the ions gather Efield_fp,
    // and the solver state deliberately includes the resistive eta*J term
    // (Faraday's law needs it), but the resistive friction must not
    // accelerate the ions through E -- the explicit scheme pushes ions with
    // the no-resistivity Ohm field, and the resistive electron-ion friction
    // is a separate (optional) collision operator. Subtract the resistive
    // part (refreshed from the previous residual evaluation at the bottom
    // of this function; exact at convergence). B above already used the
    // full E.
    if (m_use_resistive_push_correction) {
        using ablastr::fields::Direction;
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab& E_push = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
                amrex::MultiFab const& E_res = *m_WarpX->m_fields.get("hybrid_E_resistive_fp", Direction{dir}, lev);
                amrex::MultiFab::Subtract(E_push, E_res, 0, 0, E_push.nComp(), E_push.nGrowVect());
            }
        }
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

    // Compute J_plasma = curl(B^{n+theta})/mu_0
    m_hybrid_pic_model->CalculatePlasmaCurrent(Bfield_fp, m_WarpX->GetEBUpdateEFlag());

    // Darwin: the total current retains the longitudinal displacement
    // current, J = curl(B)/mu_0 - eps0 dE_L/dt, which preserves charge
    // continuity of J_e (Hewett & Nielson 1978). The transverse
    // displacement current is dropped -- that is the Darwin approximation.
    if (m_darwin) {
        using ablastr::fields::Direction;
        amrex::Real const inv_thetadt = 1.0_rt / (m_theta * m_dt);
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab & Jp = *m_WarpX->m_fields.get(FieldType::hybrid_current_fp_plasma, Direction{dir}, lev);
                amrex::MultiFab const & EL = *m_WarpX->m_fields.get("hybrid_E_long_fp", Direction{dir}, lev);
                amrex::MultiFab const & EL_old = *m_WarpX->m_fields.get("hybrid_E_long_old_fp", Direction{dir}, lev);
                amrex::MultiFab::Saxpy(Jp, -PhysConst::epsilon_0 * inv_thetadt, EL, 0, 0, Jp.nComp(), Jp.nGrowVect());
                amrex::MultiFab::Saxpy(Jp,  PhysConst::epsilon_0 * inv_thetadt, EL_old, 0, 0, Jp.nComp(), Jp.nGrowVect());
            }
        }
    }

    // Electron pressure at t^{n+theta}: either the theta-centered QDSMC
    // electron-energy stage (re-entrant; re-run from the saved t^n state in
    // every residual evaluation, so the nonlinear solver converges the
    // coupled E/T_e system by elimination) or the algebraic adiabatic
    // closure. Per-species deposits feeding the multi-species sources are
    // refreshed once per Newton iteration and frozen during Jacobian
    // evaluations.
    if (m_hybrid_pic_model->m_solve_electron_energy_equation) {
        m_hybrid_pic_model->AdvanceElectronEnergyQDSMCTheta(m_dt, m_theta, !a_from_jacobian);
    } else {
        m_hybrid_pic_model->CalculateElectronPressure();
    }

    // Darwin: refresh the longitudinal constraint field from this
    // evaluation's electron pressure and midpoint density. The refresh runs
    // in EVERY residual evaluation (including finite-difference Jacobian
    // probes): E_L is a smooth function of the state, so folding it into
    // the probed residual keeps the FD Jacobian consistent with the actual
    // iterate-to-iterate map -- freezing it (as is done for the noisy
    // per-species particle deposits) makes Newton chase a moving target
    // and stall near 50% residuals. The particle push of this evaluation
    // used the previous evaluation's E_L; both agree at convergence.
    if (m_darwin) {
        ablastr::fields::MultiLevelScalarField rho_half_alias;
        amrex::Vector<std::unique_ptr<amrex::MultiFab>> rho_half_store(m_num_amr_levels);
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            amrex::MultiFab & rho_mf = *rho_fp[lev];
            rho_half_store[lev] = std::make_unique<amrex::MultiFab>(
                rho_mf, amrex::make_alias, rho_mf.nComp()/2, 1);
            rho_half_alias.push_back(rho_half_store[lev].get());
        }
        m_hybrid_pic_model->ComputeDarwinELong(rho_half_alias, start_time + m_theta*m_dt);
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
    m_hybrid_pic_model->HybridPICSolveE(
        Efield_fp, current_fp, Bfield_fp, rho_fp,
        m_WarpX->GetEBUpdateEFlag(),
        false,  // solve_for_Faraday (retain grad(Pe))
        true    // include_resistivity (retain eta*J for the B-update)
    );

    // Refresh the resistive push-field correction from this evaluation's
    // fields: E_res = E_ohm(with resistivity) - E_ohm(without). The
    // no-resistivity solve writes directly into the E_res fabs, so
    // Efield_fp (holding the full Ohm field for the RHS below) is never
    // clobbered. Like the Darwin E_L refresh above, this runs in every
    // residual evaluation (Jacobian probes included): the correction is a
    // smooth function of the state and freezing it degrades the Newton
    // linearization.
    if (m_use_resistive_push_correction) {
        using ablastr::fields::Direction;
        ablastr::fields::MultiLevelVectorField E_res_fp =
            m_WarpX->m_fields.get_mr_levels_alldirs("hybrid_E_resistive_fp", m_num_amr_levels - 1);
        m_hybrid_pic_model->HybridPICSolveE(
            E_res_fp, current_fp, Bfield_fp, rho_fp,
            m_WarpX->GetEBUpdateEFlag(),
            false,  // solve_for_Faraday (retain grad(Pe))
            false   // include_resistivity: no-resistivity push field
        );
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab & E_res = *m_WarpX->m_fields.get("hybrid_E_resistive_fp", Direction{dir}, lev);
                amrex::MultiFab const & E_full = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
                amrex::MultiFab::LinComb(E_res, 1.0_rt, E_full, 0, -1.0_rt, E_res, 0,
                                         0, E_res.nComp(), E_res.nGrowVect());
                // The push-field subtraction at the top of this function
                // includes the ghosts the particle gather reads.
                E_res.FillBoundary(m_WarpX->Geom(lev).periodicity());
            }
        }
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
        // NOTE: overwriting the band E_ohm with the Faraday value of the
        // recovered A inside the residual is ill-conditioned: the band
        // Jacobian degenerates to the recovery's small leak factor and
        // Newton steps blow up as (mismatch / leak). The Faraday-consistent
        // band field is imposed once per step in FinishFieldUpdate instead
        // (operator-split vacuum field stage).
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

void ThetaImplicitHybrid::UpdateWarpXFields ( const WarpXSolverVec&  a_E,
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
        DarwinUpdateA_B( m_theta * m_dt, start_time );
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
    }

    if (has_external && !m_darwin) {
        add_external(FieldType::Bfield_fp, FieldType::hybrid_B_fp_external);
        add_external(FieldType::Efield_fp, FieldType::hybrid_E_fp_external);
    }
}

void ThetaImplicitHybrid::DarwinUpdateA_B ( amrex::Real a_thetadt, amrex::Real a_time )
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
            m_hybrid_pic_model->ComputeVacuumARecovery();
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
            scales.push_back(ext.TimeScale(i, a_time) - m_fext_init[i]);
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
                    amrex::MultiFab::Saxpy(A_bc, scales[i], Aext, 0, 0, 1,
                                           amrex::min(A.nGrowVect(), Aext.nGrowVect()));
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

                amrex::ParallelFor(tb,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    // Embedded conductors: hold A at the gauge zero inside
                    // masked cells (frozen enclosed flux; the interior field
                    // stays at B_static).
                    if (use_eb && eb(i,j,k) == 0) {
                        a(i,j,k) = 0.0_rt;
                        return;
                    }
                    // Non-periodic domain boundaries: impose the external
                    // vector potential on the boundary point and everything
                    // beyond it.
                    const int idx[3] = {i, j, k};
                    bool on_boundary = false;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        if (per[d]) { continue; }
                        if (idx[d] <= dlo[d] || idx[d] >= dhi[d]) { on_boundary = true; }
                    }
                    if (on_boundary) {
                        // Clamp the imposed value to the domain edge:
                        // ghosts continue the wall value rather than the
                        // (growing) exterior vector potential, so the wall
                        // ring carries no spurious curl sheet.
                        int ic[3] = {i, j, k};
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            if (per[d]) { continue; }
                            ic[d] = amrex::Clamp(ic[d], dlo[d], dhi[d]);
                        }
                        a(i,j,k) = abc(ic[0], ic[1], ic[2]);
                    }
                });
            }
            A.FillBoundary(m_WarpX->Geom(lev).periodicity());

        }
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
        DarwinApplyABoundary(end_time);
        // Vacuum recovery at the full-step state (both cadences: in "half"
        // mode the theta-stage was recovered inside the solve, and the
        // extrapolated end state gets the same treatment so the delivered
        // field is exactly recovered; in "full" mode this is the only
        // application). Restore the exact boundary pin afterwards.
        if (m_vacuum_recovery) {
            m_hybrid_pic_model->ComputeVacuumARecovery();
            DarwinApplyABoundary(end_time);
            // The delivered end-of-step field in the band is the Faraday
            // value of the recovered A across the step (Efield_fp holds
            // the full field here, so E_L is added back on top).
            m_hybrid_pic_model->ApplyVacuumFaradayE(m_dt, true);
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
        // B^{n+1}
        ablastr::fields::MultiLevelVectorField const& B_old =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, 0);
        m_WarpX->FinishMagneticFieldAndApplyBCs( B_old, m_theta, end_time );
    }

    // Restore end-of-step totals: the analytic external flux advance means
    // Bfield_fp = B_plasma^{n+1} + f(t^{n+1}) curl A_ext exactly, for any
    // ramp shape (OneStep strips the same values at the next entry).
    // Split-field form only: under the Darwin unified drive the external
    // flux already lives inside A through its boundary values, and adding
    // E_ext here would poison the saved E^n and re-inject the drive
    // volumetrically through the A rebuild (doubling the programmed flux).
    if (m_hybrid_pic_model->m_add_external_fields && !m_darwin) {
        using ablastr::fields::Direction;
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
    }
}
