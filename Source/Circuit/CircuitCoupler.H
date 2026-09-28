/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CIRCUIT_CIRCUITCOUPLER_H_
#define WARPX_CIRCUIT_CIRCUITCOUPLER_H_

#include "Coils/CoilSet.H"
#include "Coils/FluxProbes.H"
#include "ExternalCircuit.H"
#include "DeviceCircuit.H"
#include "DeviceCircuitRate.H"
#include "NativeCircuitRejection.H"
#include "NativeCircuitTransaction.H"

#include <AMReX_REAL.H>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

class ExternalVectorPotential;

/**
 * \brief Per-substep predictor-corrector coupling of an external circuit
 * engine to the hybrid solver's split external fields.
 *
 * The coupler is driven from inside the hybrid B-field substep loop
 * (HybridPICModel::BfieldEvolve), where the evolved fields hold the PLASMA
 * response (split-field convention) so the flux-linkage probes read
 * plasma-frame fields directly, and where the loop's own attempt/rollback
 * machinery provides the field snapshots. Per attempted substep [t0, t1]:
 *
 *  P  PredictSubstep: measure the plasma linkages lambda(t0) and let the
 *     engine advance the circuit over [t0, t1] with its held (last
 *     accepted) EMF estimates, pushing linear scale segments; the
 *     circuit-driven external fields are refreshed (CircuitOnly, B at the
 *     interval midpoint, E with the exact segment slope). The engine
 *     always advances from its interval-entry state, so a rejected or
 *     resized substep simply predicts again.
 *  E  the caller runs the substep integration;
 *  C  CorrectSubstep: measure lambda(t1) of the current iterate and hand
 *     the engine the corrected EMFs; if the re-advanced scales moved by
 *     less than corrector_rtol the substep is converged (no field re-run
 *     needed), otherwise the caller resets B to the substep entry and
 *     re-integrates with the refreshed segments;
 *  A  AcceptSubstep: final linkages of the accepted state (the engine's
 *     next predictor and the step-close bookkeeping read these).
 *
 * corrector_iterations = 0 is the classical lagged segregated exchange,
 * unstable for strong coil-plasma coupling (k^2 > 1/2); the per-substep
 * exchange divides the lag by the substep count and each corrector pass
 * multiplies the residual lag error by the same small factor again.
 *
 * The engine is either the Python-callback contract (circuitbeginstep /
 * circuitpredict / circuitcorrect / circuitfinish, reading the coupling
 * interval and linkage registers through the pywarpx bindings and pushing
 * segments through set_external_vector_potential_scale) or a compiled
 * ExternalCircuit plugin.
 */
class CircuitCoupler
{
public:
    struct Params
    {
        bool device_affine = false;
        int device_iterations = 20;
        int corrector_iterations = 1;
        amrex::Real corrector_rtol = 1.0e-6;
        amrex::Real scale_floor = 1.0e-3;   //!< floor of the convergence denominator
        /** Cross-check every batched linkage measurement against the
         * single-coil reference probes and abort beyond
         * crosscheck_rtol (a validation knob; circuit.probe_crosscheck). */
        bool probe_crosscheck = false;
        amrex::Real crosscheck_rtol = 1.0e-12;
        /** One-pole low-pass (exponential moving average) on every
         * measured coil's port EMF before it is handed to a compiled
         * engine, time constant [s] (circuit.eps_lowpass_tau; 0 = off =
         * the raw interval-averaged EMF, bit-identical to a coupler
         * without the knob):
         *     e = sigma * e_raw + (1 - sigma) * e_mem,
         *     sigma = dt_step / (dt_step + tau),
         * with dt_step the coupling STEP dt (the weight is a per-step
         * quantity: a theta-stage or corrector sub-interval filters with
         * the same sigma as the accepting evaluation). The memory e_mem
         * is committed ONLY by the accepting (accept = true) evaluation
         * of a step -- every non-accepted evaluation of that step sees
         * the same frozen memory, so the eps -> scales map stays a smooth
         * function of the iterate. This is the python coupling
         * reference's per-step-frozen filter contract (its pending filter
         * value is committed by the finish hook), reproduced here so
         * both drivers apply the same model. */
        amrex::Real eps_lowpass_tau = 0.0;
        /** Which linkage the step's EMF differences are taken against
         * (circuit.linkage_reference; Newton-scope driving via
         * EvaluateInterval only):
         *   false = "first_iterate": the linkage measured at the first
         *           residual evaluation of the step (the Newton initial
         *           iterate = the committed t^n state; a full-grid curl
         *           is saved because the residual just computed J);
         *   true  = "accepted": the linkage the previous step's ACCEPTING
         *           evaluation measured on its accepted state. Before the
         *           first accepted step there is none, and the step runs
         *           OPEN LOOP (eps = 0 for every evaluation, including the
         *           accepting one) -- the python coupling reference's
         *           convention (its lambda^n cache is None before the
         *           first finish hook; zero-seeding would misread a
         *           pre-existing plasma's whole equilibrium linkage as a
         *           one-step change). The accepted linkage is
         *           checkpointed so a restart continues closed-loop. */
        bool linkage_reference_accepted = false;
        /** In-residual advance target for Newton-scope consumers
         * (circuit.residual_advance):
         *   false = "theta_stage": advance the engine to the theta-stage
         *           time and push the segment [t0, t0 + theta dt], so the
         *           residual's B_ext(theta) is s(theta) exactly and its
         *           E_ext the half-interval slope;
         *   true  = "full_step": advance over the whole step [t0, t0 + dt]
         *           with the EMF still differenced over the theta interval
         *           the iterate's linkage lives on, and let the segment's
         *           linear interpolation supply B_ext(theta) = (1 - theta)
         *           s0 + theta s1 and E_ext = -(s1 - s0)/dt -- the python
         *           coupling reference's hook semantics (its theta hook
         *           sets eps from (lambda_theta - lambda^n)/(theta dt),
         *           sub-cycles the full step and pushes [t^n, t^{n+1}]).
         *           Identical to theta_stage at theta = 1. */
        bool residual_advance_full_step = false;
    };

    /** \param probes           per-coil probe kind
     *  \param probe_exclusion  per-coil mask radius [m] of the loop probe
     *                          (ignored by the other kinds) */
    CircuitCoupler (warpx::circuit::CoilSet const& coils,
                    std::vector<warpx::circuit::ProbeKind> probes,
                    std::vector<double> probe_exclusion,
                    Params params,
                    std::unique_ptr<ExternalCircuit> plugin,
                    WarpxCircuitAffineApiV1 const* affine_api = nullptr);

    /** Optional plugin capability; attach once after construction, before any
     * reversible step. Legacy ABI2 stepping is unchanged when absent. */
    bool SetRejectionCapability (WarpxCircuitRejectionApiV1 const* api);
    [[nodiscard]] bool SupportsNativeRejection () const;
    /** BEFORE any step-entry measurement or BeginStep: preserve coupler,
     * external scale segments, and plugin state. external must outlive this
     * attempt. No field arrays are copied. False leaves no retry guarantee. */
    bool SnapshotNativeStep (ExternalVectorPotential& external);
    /** Before accept=true only. On success clears borrowed device scale views,
     * device trial readiness and restores exact pre-BeginStep host metadata.
     * Parent separately restores field/particle/thermal arrays. Failure is
     * terminal if NativeRejectionFailed(); never use FinishStep as cancellation. */
    bool CancelNativeStep ();
    [[nodiscard]] bool NativeStepCancelable () const { return m_rejection.Ready(); }
    [[nodiscard]] bool NativeRejectionFailed () const { return m_rejection.Failed(); }
    [[nodiscard]] bool DarwinDeviceStepReady () const { return m_device != nullptr; }

    /** Separate optional post-accept transaction. Snapshot before BeginStep;
     * true+Finish publishes only an internal candidate until Finalize. These
     * operations never use or weaken PRE_ACCEPT V1 cancellation. */
    bool SetTransactionCapability(WarpxCircuitTransactionApiV1 const* api);
    [[nodiscard]] bool SupportsNativeTransaction() const;
    bool SnapshotRetainedNativeStep(ExternalVectorPotential& external);
    [[nodiscard]] bool RetainedNativeStepCancelable() const;
    [[nodiscard]] bool RetainedNativeStepFinalizable() const;
    [[nodiscard]] bool NativeStepRetained() const { return m_transaction.Active(); }
    [[nodiscard]] bool RetainedNativeEndpointReady() const;
    bool CancelRetainedNativeStep();
    bool FinalizeRetainedNativeStep();
    /** Same device/native accepting map, but errors after native true remain
     * recoverable and the accepted receipt is deferred. Endpoint setup may use
     * its closed native n+1 rate. Only the retained OneStep owner calls this. */
    bool TryCommitRetainedDarwinDeviceStep(amrex::Real t0,amrex::Real dt,
        amrex::Real theta,amrex::Real tolerance,std::string& error);

    enum class SourceImpulsePhase : std::uint32_t {
        PreField = WARPX_IMPULSE_PRE_FIELD, PostField = WARPX_IMPULSE_POST_FIELD
    };
    /** Optional source impulse; setup/publication run only at source boundaries.
     * The original retained transaction owns all rollback. */
    bool SetSourceImpulseCapability(WarpxCircuitImpulseApiV1 const* api);
    [[nodiscard]] bool SupportsSourceImpulse() const;
    [[nodiscard]] bool RetainedSourcePhaseReady(SourceImpulsePhase phase) const;
    bool PrepareNativeSourceImpulse(amrex::Real time, SourceImpulsePhase phase,
                                    std::string& error);
    void ApplyNativeSourceImpulse(amrex::MultiFab const& homogeneous_delta_bz);
    [[nodiscard]] amrex::Gpu::DeviceVector<double> const& SourceFieldIncrements() const;
    [[nodiscard]] amrex::Gpu::DeviceVector<double> const& SourcePortImpulses() const;
    /** Setup-only accepted port values for the exact prepared source epoch.
     * No provider call or device transfer; callers retain their own copy. */
    [[nodiscard]] std::vector<double> const& SourceEntryCurrents() const
    { AMREX_ALWAYS_ASSERT(m_source_impulse_ready); return m_source_entry_current; }

    /** Reads only the small final port vectors, checks exact native acceptance,
     * and publishes provisional provider/scales. Caller must cancel the whole
     * retained step on failure, including a post-publication failure. */
    bool CommitNativeSourceImpulse(WarpxCircuitImpulseReceiptV1& receipt,
                                    std::string& error);
    /** Call after publishing the matching source field increments; clears the
     * required rebase fence before endpoint closure or finalization. */
    bool RebaseNativeSourceLinkage(std::string& error);
    [[nodiscard]] bool RetainedSourceEndpointReady(SourceImpulsePhase phase,
                                                   amrex::Real time) const;
    void InvalidateNativeSourceImpulse();
    [[nodiscard]] bool NativeSourceImpulseReady() const { return m_source_impulse_ready; }
    [[nodiscard]] std::uint64_t NativeSourceImpulseGeneration() const
    { return m_source_impulse_generation; }
    /** Exact last device application within this preparation generation.
     * Zero means no live applied output. Reading it performs no device copy. */
    [[nodiscard]] std::uint64_t NativeSourceImpulseApplication() const noexcept
    { return m_source_impulse_ready && m_source_impulse_applied ? m_source_impulse_application : 0; }

    /** Optional differential-rate and clock capabilities. Attach once after
     * construction. Missing APIs preserve legacy ABI2 interval behavior. */
    bool SetNativeEndpointRateCapabilities (WarpxCircuitRateApiV1 const* rate,
                                            WarpxCircuitRateClockApiV1 const* clock);
    [[nodiscard]] bool SupportsNativeEndpointRate () const;
    /** Fixed-grid coil-by-external-field H, in the exact cached Q convention.
     * ConfigureDarwinMagneticResponse clears H; time/restart invalidation does
     * not. Geometry, unit fields and coil metadata must remain immutable. */
    bool SetNativeEndpointBoundaryResponse (std::vector<double> const& response,
                                            std::string& error);
    [[nodiscard]] std::vector<double> const& NativeEndpointBoundaryResponse () const
    { return m_endpoint_boundary_response; }
    /** Collective accepted-boundary preparation. Prescribed rates are ordered
     * like ExternalVectorPotential fields and zero at every coupled port.
     * Copies and releases both plugin views before return; no time advance.
     * Nonzero lowpass and active interval device scales are unsupported. */
    bool PrepareNativeEndpointRate (amrex::Real time,
        std::vector<double> const& prescribed_field_rates, std::string& error);
    /** Optional response for a frozen, eliminated field block. Copied for this
     * rate generation only; the standard geometry H cache is unchanged. */
    bool PrepareNativeEndpointRate (amrex::Real time,
        std::vector<double> const& prescribed_field_rates, std::string& error,
        std::vector<double> const* boundary_response_override);
    /** Fixed prepared state, native disk/none measurement then device action.
     * No plugin calls or host coil-vector readbacks. Input layout must match
     * the registered Bz layout; all ranks participate. affine=false suppresses
     * both circuit offset and prescribed-field rates for homogeneous Jv. */
    void ApplyNativeEndpointRate (amrex::MultiFab const& homogeneous_Bdot_z, bool affine);
    [[nodiscard]] amrex::Gpu::DeviceVector<double> const& EndpointFieldRates () const;
    [[nodiscard]] amrex::Gpu::DeviceVector<double> const& EndpointEmf () const;
    /** Call before ANY out-of-band plugin/external-field mutation. Standard
     * coupler begin/trial/accept/cancel/restart paths invalidate automatically.
     * Device output references expire at invalidation or the next Apply. */
    void InvalidateNativeEndpointRate ();
    [[nodiscard]] bool NativeEndpointRateReady () const { return m_endpoint_rate_ready; }
    /** Pin after Prepare; a mismatch invalidates any enclosing field context. */
    [[nodiscard]] std::uint64_t NativeEndpointRateGeneration () const
    { return m_endpoint_rate_generation; }

    /** Once per step, after the split-field subtraction (the evolved
     * fields hold the plasma response): seed the linkage registers at t^n
     * and open the engine's step. */
    void BeginStep (amrex::Real t0, amrex::Real dt);

    /** Predictor of the (possibly re-attempted) substep [t0, t1]. */
    void PredictSubstep (amrex::Real t0, amrex::Real t1);

    /** One corrector pass on the current field iterate; returns true when
     * the re-advanced scales already match the applied segments (converged:
     * the caller must NOT re-run the substep), false when the caller must
     * reset B to the substep entry and re-integrate. */
    [[nodiscard]] bool CorrectSubstep (amrex::Real t0, amrex::Real t1);

    /** Number of corrector passes to run per substep. */
    [[nodiscard]] int CorrectorIterations () const { return m_params.corrector_iterations; }

    /** Close an accepted substep on its final field state. */
    void AcceptSubstep (amrex::Real t0, amrex::Real t1);

    /** Once per step, before the external fields are added back to the
     * totals. */
    void FinishStep ();

    /** Measure every coil's plasma flux linkage from the current
     * plasma-frame field registers into the linkage registers, batched:
     * one device pass, one stream synchronization, one MPI all-reduce
     * over the coil vector (LinkageBatch). Pass refresh_plasma_current =
     * false when the caller's own loop has just computed
     * hybrid_current_fp_plasma for this exact field state (implicit
     * consumers: the residual evaluation already holds it), skipping the
     * full-grid curl recomputation the reciprocity probe otherwise pays.
     */
    void MeasureLinkages (bool refresh_plasma_current);

    /** Magnetic-response disk probes for a unified Darwin field. Subtract
     * every prescribed unit field through the same discrete probe functional.
     * The cached matrix is valid for fixed geometry and immutable unit fields. */
    void ConfigureDarwinMagneticResponse ();
    void MeasureDarwinLinkages (amrex::Real time);
    /** Deterministic trial seed from the fixed accepted interval entry. */
    void ResetDarwinTrialScales (amrex::Real t0, amrex::Real dt);
    [[nodiscard]] bool DeviceTrials () const { return m_params.device_affine; }
    [[nodiscard]] int DeviceIterations () const { return m_params.device_iterations; }
    void PrepareDarwinDeviceStep (amrex::Real t0, amrex::Real dt,
                                 amrex::Real theta, amrex::Real tolerance);
    // Preparation is outside residual/Jv. A false return requires the caller's
    // original whole-step rollback; the provider may have opened its interval.
    bool TryPrepareDarwinDeviceStep (amrex::Real t0, amrex::Real dt,
        amrex::Real theta, amrex::Real tolerance, std::string& error);
    void ResetDarwinDeviceTrial ();
    void AdvanceDarwinDeviceTrial (bool force = false);
    void RequireDarwinDeviceConvergence ();
    void CommitDarwinDeviceStep (amrex::Real t0, amrex::Real dt,
                                amrex::Real theta, amrex::Real tolerance);



    /** Newton/iterate-scope driving for implicit consumers, replacing the
     * substep protocol above: open the coupling step [t0, t0 + dt],
     * seeding the engine's interval-entry linkages from the LATEST
     * measurement (call MeasureLinkages first, on the committed t^n
     * state). The engine snapshots its accepted state; discontinuous
     * transitions (switches, stage swaps) latch on committed time only.
     */
    void BeginStepMeasured (amrex::Real t0, amrex::Real dt);

    /** One engine advance over [t0, t1] against the latest measured
     * linkages (call MeasureLinkages on the iterate first). accept =
     * false evaluations replay from the interval entry state -- a pure
     * function of the measurement, legal inside every residual
     * evaluation (exact in-residual coupling) or once per Newton iterate
     * (lagged quasi-Newton); the interval may be re-targeted freely
     * (theta-stage time in-residual, the full step on the accepting
     * call). Pushes the realized scales onto the field registers as
     * linear segments over [t0, t1] but does NOT refresh the external
     * fields -- the caller owns its refresh points. Exactly one
     * accept = true call per step, on the accepted final state, followed
     * by FinishStep().
     *
     * eps_interval (optional): the time span the EMF is differenced over,
     * when it differs from t1 - t0 -- the theta-stage interval the
     * iterate's linkage lives on while the engine advances the full step
     * (Params::residual_advance_full_step); <= 0 means t1 - t0. */
    void EvaluateInterval (amrex::Real t0, amrex::Real t1, bool accept,
                           amrex::Real eps_interval = -1.0);

    /** Whether Newton-scope consumers advance the engine over the full
     * step inside the residual (Params::residual_advance_full_step). */
    [[nodiscard]] bool ResidualAdvanceFullStep () const
    {
        return m_params.residual_advance_full_step;
    }

    /** The current coupling interval, for the Python engine. */
    struct Interval
    {
        amrex::Real t0 = 0.0;
        amrex::Real t1 = 0.0;
        int substep = -1;
        int iteration = 0;   //!< 0 = predictor, k >= 1 = corrector pass k
    };
    [[nodiscard]] Interval const& CurrentInterval () const { return m_interval; }

    /** Latest measured plasma flux linkage of a coil, reported as
     * lambda_phys * I_ref * n_turns (the port EMF is then
     * d lambda / dt / (I_ref * n_turns) = d lambda_phys / dt in volts),
     * for the Python engine. */
    [[nodiscard]] amrex::Real CoilLinkage (std::string const& coil_name) const;

    /** As CoilLinkage, but returning fallback for unmeasured coils
     * (probe = none) instead of aborting -- for diagnostics. */
    [[nodiscard]] amrex::Real CoilLinkageOr (std::string const& coil_name,
                                             amrex::Real fallback) const;

    /** The compiled engine, if one is attached (engine = external), for
     * configuration and checkpoint delegation. */
    [[nodiscard]] ExternalCircuit* Plugin () const { return m_plugin.get(); }

    /** Persist / restore the coupler's per-step measurement memory (the
     * committed EMF low-pass state per coil and the accepted-state
     * linkage reference of the last accepted step) into / from
     * `<dir>/circuit_coupler_memory.dat`. Without it a restart re-seeds
     * the memory from zero and the drive carries a filter warm-up
     * transient (and, with linkage_reference = accepted, an open-loop
     * first step) the uninterrupted run never had. Write on the I/O rank
     * only; read on every rank (rank-replicated state); tolerant of
     * checkpoints written before the file existed. */
    void WriteMemoryCheckpoint (std::string const& dir) const;
    void ReadMemoryCheckpoint (std::string const& dir);

private:
    WarpxCircuitAffineApiV1 const* m_affine_api = nullptr;
    std::unique_ptr<warpx::circuit::DeviceCircuit> m_device;
    std::vector<amrex::Real> m_darwin_unit_response;
    int m_darwin_external_count = -1;
    WarpxCircuitImpulseApiV1 const* m_source_impulse_api = nullptr;
    bool m_source_capability_attached = false;
    std::uint64_t m_source_impulse_token = 0, m_source_impulse_generation = 0;
    bool m_source_impulse_ready = false, m_source_impulse_applied = false;
    std::uint64_t m_source_impulse_application = 0;
    bool m_source_rebase_pending = false;
    std::uint32_t m_source_endpoint_phase = 0;
    amrex::Real m_source_time = 0.;
    SourceImpulsePhase m_source_phase = SourceImpulsePhase::PreField;
    std::unique_ptr<warpx::circuit::DeviceCircuitRate> m_source_impulse;
    std::vector<double> m_source_entry_current;
    WarpxCircuitRateApiV1 const* m_rate_api = nullptr;
    WarpxCircuitRateClockApiV1 const* m_rate_clock_api = nullptr;
    std::uint64_t m_endpoint_rate_generation = 0;
    bool m_endpoint_capabilities_attached = false;
    bool m_endpoint_rate_ready = false, m_endpoint_rate_applied = false;
    std::unique_ptr<warpx::circuit::DeviceCircuitRate> m_endpoint_rate;
    std::vector<double> m_endpoint_boundary_response;
    amrex::Box m_endpoint_domain;
    amrex::BoxArray m_endpoint_ba;
    amrex::DistributionMapping m_endpoint_dm;
    std::array<amrex::Real, AMREX_SPACEDIM> m_endpoint_prob_lo{}, m_endpoint_prob_hi{};
    std::array<int, AMREX_SPACEDIM> m_endpoint_periodic{};
    int m_endpoint_coord = -1;
    std::vector<std::string> m_endpoint_field_names;
    [[nodiscard]] bool NativeEndpointGeometryMatches (amrex::MultiFab const& bz) const;
    void RefreshCircuitFields (amrex::Real t0, amrex::Real t1);
    void FireEngine (char const* hook, bool accept);
    [[nodiscard]] std::vector<amrex::Real> CoupledScales (amrex::Real t) const;

    warpx::circuit::CoilSet const& m_coils;
    std::vector<warpx::circuit::ProbeKind> m_probes;
    std::vector<double> m_probe_exclusion;   //!< loop-probe mask radii [m]
    Params m_params;
    std::unique_ptr<ExternalCircuit> m_plugin;

    Interval m_interval;
    int m_substep_count = 0;
    amrex::Real m_step_dt = 0.0;                        //!< the coupling step's dt
    std::map<std::string, amrex::Real> m_lambda;        //!< latest measurement
    std::map<std::string, amrex::Real> m_lambda_start;  //!< at interval entry
    /** Committed EMF low-pass memory [V] per coil (Params::eps_lowpass_tau);
     * moves only on accept = true. */
    std::map<std::string, amrex::Real> m_eps_filt;
    /** The linkages measured for the last ACCEPTING evaluation (the
     * accepted state); the step's EMF reference under
     * Params::linkage_reference_accepted. */
    std::map<std::string, amrex::Real> m_lambda_accepted;
    bool m_have_lambda_accepted = false;
    /** The current step has no linkage reference (accepted mode before
     * the first accepted step): every evaluation hands eps = 0. */
    bool m_open_loop_step = false;
    /** EMF differencing span of the current evaluation when it differs
     * from the interval length (EvaluateInterval's eps_interval); <= 0 =
     * the interval length. */
    amrex::Real m_eps_interval = -1.0;

    warpx::circuit::LinkageBatch m_batch;               //!< batched probes
    std::vector<amrex::Real> m_lambda_scratch;
    //! per-coil unit-field components (RZ reads the theta entry; 3D all three)
    std::vector<warpx::circuit::VectorFieldPtrs> m_a_ext_scratch;

    struct RejectionSnapshot {
        struct Segment {
            std::string name;
            bool python = false;
            amrex::Real old_scale = 0., new_scale = 0., old_time = 0., new_time = 0.;
        };
        Interval interval;
        int substep_count = 0;
        amrex::Real step_dt = 0., eps_interval = -1.;
        std::map<std::string, amrex::Real> lambda, lambda_start, eps_filt, lambda_accepted;
        bool have_lambda_accepted = false, open_loop_step = false;
        std::vector<Segment> segments;
    };
    // These members follow m_plugin: callback cleanup must precede destruction
    // of the engine instance. The borrowed external object lives through Finish
    // or successful Cancel and never changes within an attempt.
    WarpxCircuitRejectionApiV1 const* m_rejection_api = nullptr;
    warpx::circuit::NativeCircuitRejection m_rejection;
    std::unique_ptr<RejectionSnapshot> m_rejection_snapshot;
    ExternalVectorPotential* m_rejection_external = nullptr;
    WarpxCircuitTransactionApiV1 const* m_transaction_api = nullptr;
    warpx::circuit::NativeCircuitTransaction m_transaction;
    std::unique_ptr<RejectionSnapshot> m_transaction_snapshot;
    ExternalVectorPotential* m_transaction_external = nullptr;
    int m_transaction_device_trials = -1;
    [[nodiscard]] bool RetainedExternalMatches() const;
    void BeforeNativeAccept ();
    ExternalVectorPotential& StepExternal ();

};

#endif // WARPX_CIRCUIT_CIRCUITCOUPLER_H_
