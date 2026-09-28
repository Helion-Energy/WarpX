/* Copyright 2026 Prabhat Kumar
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef THETA_IMPLICIT_HYBRID_H_
#define THETA_IMPLICIT_HYBRID_H_

#include "NativePairedDarwinFields.H"
#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"
#include "ImplicitSolver.H"
#include "NativeAcceptedStepCandidate.H"
#include "MassMatrixDensityProjection.H"
#include "NativeVacuumMassMatrixCapabilities.H"

#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_MLNodeTensorLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_Parser.H>
#include <array>
#include <string>

class HybridPICModel;
namespace warpx::thermal { class DarwinThermalAdvance; class DarwinVacuumJointSolve; }
namespace warpx::darwin { class NativeLongitudinalProducerCertificate; class NativeCoilField; }

/** @file
 *  Theta-implicit hybrid-PIC time solver class. This is a fully implicit
 *  algorithm where both the fields and particles are treated implicitly.
 *
 *  In the hybrid-PIC model, ions are kinetic particles while electrons are
 *  treated as a massless, charge-neutralizing fluid. The electric field is
 *  determined algebraically from the generalized Ohm's law.
 *
 *  The magnetic field evolves via Faraday's law:
 *  Bg^{n+1} = Bg^n - dt*curl(Eg^{n+theta})
 *
 *  Particles are advanced as:
 *  xp^{n+1} = xp^n + dt*up^{n+1/2}/(0.5*(gammap^n + gammap^{n+1}))
 *  up^{n+1} = up^n + dt*qp/mp*(Ep^{n+theta} + up^{n+1/2}/gammap^{n+1/2} x Bp^{n+theta})
 *
 *  where f^{n+theta} = (1.0-theta)*f^{n} + theta*f^{n+1} with 0.5 <= theta <= 1.0
 *
 *  The key difference from fully electromagnetic implicit solvers is that E is computed
 *  from the generalized Ohm's law at each nonlinear iteration. The nonlinear solver finds
 *  the self-consistent E^{n+theta} such that:
 *
 *     E = (J_e x B) / (n_e * e)          (Hall term)
 *       - grad(P_e) / (n_e * e)          (Electron pressure)
 *       + eta * J                         (Resistivity)
 *       - eta_H * laplacian(J_e)          (Hyper-resistivity)
 *
 *  where the electron current is:
 *     J_e = J_plasma - J_ion
 *
 *  and the plasma current is computed from Ampere's law:
 *     J_plasma = curl(B) / mu_0
 *
 *  The nonlinear solver iterates to find the self-consistent solution
 *  E^{n+theta} = Ohm(B^{n+theta}, J_i^{n+1/2}, rho^{n+1/2}).
 *
 *  The user-specified time-biasing parameter theta is bounded between 0.5 and 1.0.
 *  Significant damping of high-k modes will occur as theta approaches 1.0. Since
 *  there are no light waves in the hybrid model (the displacement current is neglected),
 *  the CFL condition for light waves does not apply. The time step is limited by:
 *  1. Resolving the relevant ion physics (cyclotron motion, etc.)
 *  2. Particles not traveling outside the ghost region (MPI limitation)
 *  3. The need to resolve whistler waves if hyper-resistivity is insufficient
 *
 */

class CircuitCoupler;

class ThetaImplicitHybrid : public ImplicitSolver
{
public:

    ThetaImplicitHybrid();

    ~ThetaImplicitHybrid() override;

    // Prohibit Move and Copy operations
    ThetaImplicitHybrid(const ThetaImplicitHybrid&) = delete;
    ThetaImplicitHybrid& operator=(const ThetaImplicitHybrid&) = delete;
    ThetaImplicitHybrid(ThetaImplicitHybrid&&) = delete;
    ThetaImplicitHybrid& operator=(ThetaImplicitHybrid&&) = delete;

    void Define ( WarpX*  a_WarpX, bool  a_from_restart ) override;

    void PrintParameters () const override;

    // Accepted external A/B split reference, preserved across restart.
    void WriteDarwinDriveReference (std::string const& directory) const;
    void ReadDarwinDriveReference (std::string const& directory);
    bool NativeEndpointCheckpointReady() const;
    void WriteAcceptedEndpointCheckpoint(std::string const& directory) const;
    void ReadAcceptedEndpointCheckpoint(std::string const& directory);

    void InitializeFields (amrex::Real time, amrex::Real dt) override;

    /**
     * \brief Advances the simulation one time step
     *
     * \param[in] start_time The time at the start of the time step
     * \param[in] a_dt The time step size
     * \param[in] a_step The time step number
     *
     * \return The nonlinear solver exit status (negative on failure)
     */
    int OneStep ( amrex::Real  start_time,
                  amrex::Real  a_dt,
                  int          a_step ) override;

    void ComputeRHS ( WarpXSolverVec&  a_RHS,
                const WarpXSolverVec&  a_E,
                      amrex::Real      start_time,
                      int              a_nl_iter,
                      bool             a_from_jacobian ) override;

    // Hybrid model has no light waves, so return 0 for the PC theta parameter
    [[nodiscard]] amrex::Real GetThetaForPC () const override { return amrex::Real(0.0); }

    /** The hybrid-PIC model (owned by WarpX, cached in Define): n_floor,
     *  the resistivity and hyper-resistivity parsers, and the Hall flags
     *  a Hall-aware preconditioner needs. */
    const HybridPICModel* GetHybridPICModel () const override { return m_hybrid_pic_model; }

    /** The theta-midpoint magnetic field B^{n+theta} (Bfield_fp registry;
     *  see the implementation for the time-level bookkeeping). Valid only
     *  after the first residual evaluation of the current Newton iterate. */
    amrex::Array<const amrex::MultiFab*, 3> GetBfieldThetaForPC (int lev) const override;

    /** The charge density holding the midpoint deposit rho^{n+1/2} in
     *  component nComp()/2 (rho_fp registry; see the implementation for the
     *  component convention). Valid only after the first residual evaluation
     *  of the current Newton iterate. */
    const amrex::MultiFab* GetRhoMidForPC (int lev) const override;
    std::pair<const amrex::MultiFab*, int> GetOhmDensityForPC (int lev) const override;

    /** The per-step-frozen rho^n snapshot consumed by the curlcurl_form
     *  poloidal stage (hybrid_rho_n_frozen, component 0); nullptr whenever
     *  the solve itself would fall back to the midpoint density. */
    const amrex::MultiFab* GetRhoPolFrozenForPC (int lev) const override;

    /** E-staggered beta = 1/d_e^2 coefficients for the electron-inertia
     *  curl-curl preconditioner (the divided curl-curl operator form of
     *  Amano et al., J. Comput. Phys. 275, 197 (2014)), assembled
     *  from the midpoint density with the Ohm floor and taper mirrored
     *  from the E-solve kernel. Non-null only with electron inertia in the
     *  two-point dJe/dt-only (operator) form; aborts in RZ. */
    const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* FillInertiaBetaCoeff () override;
    const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* GetInertiaPCRHSScale () const override
    {
        return m_inertia_rhs_scale.empty() ? nullptr : &m_inertia_rhs_scale;
    }

    void ProjectDarwinVacuumGauge (WarpXSolverVec& field, bool preconditioner = false) override;

    /** The ion current deposit consumed by the Ohm solve (current_fp
     *  registry) -- frozen during Jacobian probes, so it is the exact
     *  drift-leg coefficient of the linear solve. */
    amrex::Array<const amrex::MultiFab*, 3> GetIonCurrentForPC (int lev) const override;

private:

    friend ThetaImplicitHybrid const* NativePrivatePMCMassMatrixOwner(WarpX&);
    friend ThetaImplicitHybrid const* NativePrivatePMCMassMatrixOwnerLocal(WarpX&);
    friend class warpx::thermal::DarwinThermalAdvance;
    friend class warpx::thermal::DarwinVacuumJointSolve;
    enum class JointStagePurpose { Standard, Residual, VerifiedPublication, PublishedCurrent };
    JointStagePurpose m_joint_stage_purpose = JointStagePurpose::Standard;
    warpx::thermal::DarwinVacuumJointSolve* m_joint_stage_owner = nullptr;
    friend warpx::darwin::NativePairedDarwinFields* warpx::darwin::NativeActivePairedFields(WarpX&) noexcept;
    friend warpx::darwin::NativePairedDarwinFields* warpx::darwin::NativeEndpointPairedFields(WarpX&) noexcept;
    friend bool warpx::darwin::NativeEndpointPairSelected(WarpX&) noexcept;
    friend bool warpx::darwin::NativeEndpointPairRestartContext(WarpX&) noexcept;
    friend bool warpx::darwin::NativeEndpointPMCQualificationSelected(WarpX&);
    friend bool warpx::darwin::NativeEndpointCudaQualificationSelected(WarpX&) noexcept;
    friend bool warpx::darwin::BindNativeAcceptedEndpointBoundary(WarpX&);
    friend class warpx::implicit::NativeAcceptedStepCandidate;
    friend warpx::implicit::NativeStepCandidateResult
        warpx::implicit::AdvanceNativeStepCandidate(WarpX&, amrex::Real, amrex::Real, int,
            warpx::implicit::NativeSymmetricStoppingOptions const*,
        warpx::thermal::NativeStoppingCarryRequest const*);
    int OneStepImpl(amrex::Real, amrex::Real, int,
        std::unique_ptr<warpx::implicit::NativeAcceptedStepCandidate>* deferred,
        warpx::implicit::NativeSymmetricStoppingOptions const* symmetric_source=nullptr,
        warpx::thermal::NativeStoppingCarryRequest const* stopping_carry=nullptr);
    bool NativeRetainedScopeSupported() const;
    // Private source-free PMC qualification. Kept separate from the shared
    // static-vacuum predicate, which also admits the existing periodic MM path.
    bool NativePMCJointScopeSupported() const;
    bool NativePMCJointConfigurationValid(bool check_drive) const;
    bool NativePMCJointRequestSelectorsCurrent() const;
    bool NativePMCJointMassMatrixScopeSupported() const;
    // Current+density tensors keep the original private scope. The separate
    // finite-conduction selector requires the owned thermal companion as well.
    bool m_native_pmc_mm_requested = false;
    bool m_native_pmc_finite_conduction_requested = false;
    bool m_native_pmc_joint_requested = false;
    // Captured once at Define after rank-uniform agreement; no live ParmParse selection.
    bool m_native_owned_endpoint_ampere_requested = false;
    // Private thermal capability: immutable material/BC identity, not accepted
    // physical/checkpoint state. Parser copies retain compiled shared Data.
    bool m_native_pmc_thermal_requested = false;
    std::string m_native_pmc_thermal_descriptor;
    std::array<std::unique_ptr<amrex::Parser>,2> m_native_pmc_thermal_parsers;
    std::array<char const*,4> m_native_pmc_thermal_executors{};
    std::string NativePMCJointThermalDescriptor() const;
    bool NativePMCJointThermalScopeSupported() const;
    bool NativePMCJointThermalExecutorsValid() const;
    void BindNativePMCJointThermalExecutors();
    // Ownership only, never a physical/checkpoint state or rollback generation.
    bool m_native_candidate_active = false;
    bool m_native_stopping_certificate = false;
    bool NativeStoppingCertificateScopeSupported() const;
    bool NativePairedFieldScopeSupported() const;
    bool NativeEndpointPairScopeSupported() const;
    bool NativeEndpointPairMassMatrixScopeSupported() const;
    friend ThetaImplicitHybrid const* NativeEndpointPairMassMatrixOwnerLocal(WarpX&) noexcept;
    bool m_native_paired_requested=false;
    std::unique_ptr<warpx::darwin::NativePairedDarwinFields> m_native_paired_fields;
    bool m_native_endpoint_pair_requested=false;
    bool m_native_endpoint_pair_accept_requested=false;
    bool m_native_endpoint_restart_reconstructing=false;
    bool m_native_endpoint_restart_validated=false;
    // Default-OFF CPU qualification, captured collectively before initialization.
    bool m_native_endpoint_pair_mm_requested=false;
    std::unique_ptr<warpx::darwin::NativePairedDarwinFields> m_native_endpoint_pair;
    // The accepted proof is immutable throughout a step. A candidate owns a
    // private producer trace; cancel discards it and finalize alone publishes it.
    std::shared_ptr<warpx::darwin::NativeLongitudinalProducerCertificate> m_native_longitudinal_accepted;
    std::shared_ptr<warpx::darwin::NativeLongitudinalProducerCertificate> m_native_longitudinal_pending;

    std::unique_ptr<warpx::thermal::DarwinThermalAdvance> m_eulerian_energy;

    /**
     * \brief Pointer to the hybrid-PIC model that computes E from Ohm's law.
     * This is owned by WarpX, not by this class.
     */
    HybridPICModel* m_hybrid_pic_model = nullptr;

    /** Continuity density response during hybrid MM Jacobian evaluations.
     * Each full particle residual rebases the predictor; every linear probe
     * starts from that base. With the thermal split, density and pressure
     * respond inside the field solve while temperature stays outer-iterated.
     */
    bool m_mass_matrices_density_projection = false;
    bool m_continuity_density = false;
    bool m_live_ion_density = false;
    amrex::Vector<std::unique_ptr<MassMatrixDensityProjection>> m_mass_matrix_density;
    void UpdateMassMatrixDensity (bool from_jacobian);
    [[nodiscard]] bool UseMassMatrixCurrentIncrement () const override;
    void ComposeMassMatrixCurrentIncrement () override;
    void InvalidateMassMatrixCurrentIncrement () override;

    /** Owned storage and pointer view for the electron-inertia beta
     *  coefficients served by FillInertiaBetaCoeff (E-staggered, one
     *  component each, no ghosts; allocated lazily on first fill). */
    amrex::Vector<amrex::Array<std::unique_ptr<amrex::MultiFab>,3>> m_inertia_beta_owned;
    amrex::Vector<amrex::Array<amrex::MultiFab*,3>> m_inertia_beta;
    amrex::Vector<amrex::Array<std::unique_ptr<amrex::MultiFab>,3>> m_inertia_rhs_scale_owned;
    amrex::Vector<amrex::Array<amrex::MultiFab*,3>> m_inertia_rhs_scale;
    amrex::Real m_darwin_vacuum_pc_regularization = 0.0;
    bool m_darwin_vacuum_gauge_projection = false;
    amrex::Real m_darwin_vacuum_gauge_pc_rtol = 1.e-12;
    amrex::Array<amrex::MultiFab,3> m_vacuum_gauge_edges;
    std::unique_ptr<amrex::iMultiFab> m_vacuum_gauge_mask;
    std::unique_ptr<amrex::MultiFab> m_vacuum_gauge_phi;
    std::unique_ptr<amrex::MultiFab> m_vacuum_gauge_rhs;
    std::unique_ptr<amrex::MLNodeTensorLaplacian> m_vacuum_gauge_op;
    std::unique_ptr<amrex::MLMG> m_vacuum_gauge_solver;

    /**
     * \brief Solver vectors for the electric field.
     * m_E: E field at time n+theta (during solve), then n+1 (after FinishFieldUpdate)
     * m_Eold: E field at time n (saved at start of timestep)
     *
     * m_Eold is needed for:
     * - Initial guess for the nonlinear solver
     * - Reference value for residual computation in ComputeRHS
     * - Extrapolation from t^{n+theta} to t^{n+1} in FinishFieldUpdate
     */
    WarpXSolverVec m_E;
    WarpXSolverVec m_Eold;
    /** E^{n-1} (the previous step's E^n) for the extrapolated initial guess. */
    WarpXSolverVec m_Eprev;
    /** Whether m_Eold holds a completed step's E^n (false until the first step
     *  has stored it), so m_Eprev is only formed from real data. */
    bool m_have_Eold = false;
    /** Whether m_Eprev holds E^{n-1}. */
    bool m_have_Eprev = false;
    /** Opt-in (implicit_evolve.extrapolate_initial_guess): start Newton from
     *  E^{n+theta} = (1 + theta) E^n - theta E^{n-1} (linear extrapolation of
     *  the field history) instead of E^n. Default off: bit-identical. The first
     *  step of a run and the first step after a restart use E^n. */
    bool m_extrapolate_initial_guess = false;

    /**
     * \brief Update the E and B fields owned by WarpX. Updates B^{n+theta}
     * from E via Faraday's law, then computes E^{n+theta} from Ohm's law.
     */
    void UpdateWarpXFields ( const WarpXSolverVec&  a_E,
                             bool a_from_jacobian,
                             amrex::Real            start_time );

    /**
     * \brief Extrapolate fields from t^{n+theta} to t^{n+1} after convergence.
     * Uses: F^{n+1} = (1/theta)*F^{n+theta} + (1 - 1/theta)*F^n
     * This is applied to both E and B fields.
     */
    void FinishFieldUpdate ( amrex::Real end_time, bool owned_ampere = false );

    /** Darwin split (hybrid_pic_model.darwin): when true, the solver state
     *  is the transverse field E_T, the magnetic field is advanced through
     *  the vector potential (B = B_static + curl A,
     *  A^{n+1} = A^n - dt E_T^{n+theta}) instead of Faraday's law, and the
     *  longitudinal field E_L = grad(phi) comes from the instantaneous
     *  ambipolar constraint (HybridPICModel::ComputeDarwinELong), refreshed
     *  once per nonlinear iteration. */
    bool m_darwin = false;
    /** Solve E_T at fixed E_L, then update the longitudinal constraint.
     *  The default keeps the historical residual-evaluation ordering. */
    bool m_darwin_segregated_solve = false;
    int m_darwin_outer_max_iterations = 20;
    amrex::Real m_darwin_outer_rtol = 1.0e-8;
    amrex::Real m_darwin_outer_atol = 1.0e-12;
    bool m_darwin_outer_verbose = false;
    amrex::Real m_darwin_outer_relaxation = 1.0;
    int m_darwin_outer_relaxation_start = 0;
    int SolveDarwinSegregated (amrex::Real start_time, int a_step);
    void RefreshDarwinELong (amrex::Real theta_time);

    /** End-of-step E finisher: false (default) extrapolates the algebraic
     *  field, E^{n+1} = (E^theta - (1-theta) E^n)/theta -- a -(1-theta)/
     *  theta recursion on the stored E (marginal at theta = 1/2; its error
     *  grows linearly under a steady drift and pollutes the delivered
     *  field). True re-evaluates the generalized Ohm's law at the
     *  delivered end-of-step state instead, as the explicit hybrid loop
     *  does (implicit_evolve.hybrid_e_finisher = reevaluate). */
    bool m_e_finisher_reevaluate = false;

    /** Initial moments, E/EL, Je history and magnetic reference are ready.
     *  Set after InitializeFields, before diagnostics or the first trial push. */
    bool m_darwin_initialized = false;
    std::unique_ptr<warpx::darwin::NativeCoilField> m_native_coil_field;
    amrex::Array<amrex::MultiFab,3> m_native_longitudinal_gather;
    warpx::darwin::NativeCoilField* NativeCoilCurrentContext();
    void CalculateNativePlasmaCurrent(ablastr::fields::MultiLevelVectorField const&, amrex::Real time);
    void DiagnosticConstrainEndpoint(amrex::Real time,bool initial);

    /** Cached in Define: vacuum vector-potential recovery enabled
     *  (hybrid_pic_model.darwin_vacuum_recovery), and its "half" cadence
     *  (apply inside every residual evaluation at the theta-stage field, in
     *  addition to the end-of-step application both cadences share). */
    bool m_vacuum_recovery = false;
    bool m_vacuum_recovery_half = false;

    /** Execute the stage/finish circuit callbacks. Unified Darwin exposes total
     * B; split-field hybrid exposes plasma-response B during the stage callback.
     * Callbacks replay accepted circuit state and publish trial scale segments.
     * Segregated Darwin converges these scales before particles by default;
     * the legacy branch instead updates A/B after particles and current.
     * Finish must commit the accepted candidate, not reinterpret the as-left
     * theta B as an endpoint observation. */
    bool m_external_field_iteration = false;

    /** Eliminate circuit scales before particles, using a replayable B/Jp-only
     * callback. Enabled by default for single-level 3D segregated Darwin circuit coupling. */
    bool m_darwin_circuit_consistent_stage = false;
    bool m_circuit_native = false;
    bool m_native_circuit_configured = false;
    bool m_native_circuit_step_open = false;
    CircuitCoupler& NativeCircuitCoupler () const;
    void CommitNativeCircuitStage (amrex::Real start_time);
    int m_darwin_circuit_max_iterations = 20;
    amrex::Vector<amrex::Real> m_darwin_circuit_accepted_scales;
    amrex::Real m_darwin_circuit_scale_tolerance = amrex::Real(1.e-14);

    void ConvergeDarwinCircuitStage (WarpXSolverVec const& electric_field,
                                    amrex::Real start_time, bool from_jacobian);
    void RefreshDarwinCircuitCurrent ();

    /** Apply the particle boundary conditions and re-bin (Redistribute)
     *  BEFORE the end-of-step deposits
     *  (implicit_evolve.redistribute_before_end_deposits, default false).
     *  The full-dt extrapolated positions can exceed the deposit guard
     *  range (v dt > range * dx) -- a documented hazard of the
     *  deposit-then-redistribute posture that CI decks dodge with inert
     *  ions but driven production decks with warm boundary populations
     *  cannot (the crosser rate scales with particle count). Opt-in: the
     *  redistribution reorders particles ahead of the next step's
     *  deposits, so enabling it changes bit-pinned CI trajectories. */
    bool m_redistribute_before_end_deposits = false;

    /** Cached in Define: subtract the resistive Ohm-field part from the
     *  particle push field (hybrid_pic_model.implicit_push_excludes_resistive_field
     *  and a resistive term configured). */
    bool m_use_resistive_push_correction = false;

    /** Segregated midpoint-iterated solve for the QDSMC electron-energy
     *  stage (implicit_evolve.qdsmc_segregated_solve): the stage is lifted
     *  out of the residual evaluations -- the inner nonlinear solver
     *  converges the field system at frozen electron pressure, then one
     *  re-entrant stage pass re-solves the midpoint entropy transport
     *  against the converged fields, and the outer loop in OneStep iterates
     *  the pair to joint self-consistency (convergence measured on the
     *  relative change of the emitted pressure). Keeps the marker
     *  gather/push/deposit chain out of Jacobian-vector products and
     *  line-search trials; the coupled fixed point is unchanged. */
    bool m_qdsmc_segregated_solve = false;
    int m_qdsmc_outer_max_iterations = 20;
    amrex::Real m_qdsmc_outer_relative_tolerance = 1.0e-6;
    bool m_qdsmc_outer_require_convergence = true;
    bool m_qdsmc_outer_verbose = false;
    bool m_qdsmc_outer_acceleration = false;

    /** Previous outer iterate of the emitted electron pressure, for the
     *  outer convergence test (allocated on first use). */
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_qdsmc_Pe_prev;
    /** Frozen pre-push coefficient density, updated by the thermal outer loop.
     * Also the Ohm density unless the MM density response is enabled. */
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_qdsmc_rho_frozen;

    /** Darwin magnetic-field update: A^{n+theta} = A_old - theta dt E_T
     *  (from Efield_fp, which holds the transverse state at entry),
     *  then B = B_static + curl A with boundary treatment. */
    void DarwinUpdateA_B (amrex::Real a_thetadt, amrex::Real a_time, bool a_from_jacobian);

  public:
    // Public for nvcc: these functions enclose extended device lambdas.
    void ApplyDarwinDisplacementCurrent (amrex::Real interval);

    /** Outer loop of the segregated solve: alternate {inner nonlinear solve
     *  at frozen Pe, one stage pass on the as-left converged grid state}
     *  until the emitted pressure stops changing. Ends on the stage pass so
     *  the markers and V_e are consistent with the accepted field state
     *  (QDSMCFinishImplicitStep completes the characteristic unchanged).
     *
     *  \return the inner solver's exit status, or -7 if the outer loop
     *          reached qdsmc_outer_max_iterations with
     *          qdsmc_outer_require_convergence set. */
    int SolveSegregated (amrex::Real start_time, int a_step, amrex::Real field_reference = 0.0);

    /** Unified external drive on the Darwin path: impose the (gauge-shifted)
     * external vector potential as the boundary value of the evolved A on
     * non-periodic domain boundaries, and hold A inside embedded-boundary
     * bodies (frozen enclosed flux). Applied after every A update.
     *
     * Public only for CUDA: the body defines extended __device__ lambdas
     * (ParallelFor), and nvcc requires the enclosing member function of an
     * extended lambda to have public access (hard error since CUDA 13). */
    void DarwinApplyABoundary ( amrex::Real a_time );

private:

    /** Derive B = B_static + curl A (with the derived-field boundary
     * treatment) from the current A. Used by the circuit-in-the-residual
     * coupling to re-derive B after the boundary values of A were
     * re-imposed at updated coil scales. */
    void DarwinDeriveB ();

    /** Split-field path: add a_sign times the stored external B/E arrays
     * to the total fields. The circuit-in-the-residual coupling uses the
     * (-1, refresh, +1) sequence to expose the plasma-response frame to
     * the python flux-linkage probes and re-assemble circuit-consistent
     * totals within the evaluation. */
    void AddSplitExternalFields ( amrex::Real a_sign );

    /** f_i(t_init) for each external A field: the Darwin gauge zeroes A at
     * initialization, so boundary values impose the CHANGE of the external
     * vector potential since then. */
    amrex::Vector<amrex::Real> m_fext_init;

};

#endif
