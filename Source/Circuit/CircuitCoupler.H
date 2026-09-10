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

#include <AMReX_REAL.H>

#include <map>
#include <memory>
#include <string>
#include <vector>

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
    /** Region of the nodal J_theta mesh the J-based linkage probes
     * (reciprocity, loop) integrate over (circuit.probe_region). */
    enum class ProbeRegion
    {
        domain,        //!< the whole domain (default; today's behaviour, bit-identical)
        wall_interior  //!< nodes with r < r_wall(z) of the wall polyline only (warpx::circuit::ProbeRegionTable)
    };
    /** Per-node weight of the J-based probe integrand (circuit.probe_weight). */
    enum class ProbeWeight
    {
        none,           //!< unit weight (default; bit-identical)
        physical_share  //!< w = eta_phys / eta_field from the "circuit_linkage_weight" register
    };

    struct Params
    {
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
         * both drivers apply the same model. Defined for the measured-step
         * protocol (BeginStepMeasured) only: BeginStep (the explicit
         * substep protocol, one accept per substep) refuses a nonzero
         * value. */
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
        /** Which nodes of the J_theta mesh the J-based probes (reciprocity
         * and loop; the disk probe is a B_z surface integral and is not
         * affected) integrate over (circuit.probe_region): the whole
         * domain (default, bit-identical to a coupler without the knob),
         * or the wall interior only -- node (r_i, z_j) counts iff
         * r_i < r_wall(z_j) with r_wall the piecewise-linear, end-clamped
         * interpolation of the wall polyline (the rule is documented on
         * warpx::circuit::ProbeRegionTable). Excludes the resistive
         * exterior between the dielectric wall and the outer boundary,
         * the two-cell wall band and any paint leakage from every coil's
         * back-EMF. The mask is folded into the batched weight tables
         * (a masked node's weight is exactly 0) and applied identically
         * by the single-coil reference probes (circuit.probe_crosscheck
         * validates the two under the mask). */
        ProbeRegion probe_region = ProbeRegion::domain;
        /** Per-node weight of the J-based probe integrand
         * (circuit.probe_weight): none (default, bit-identical), or the
         * PHYSICAL SHARE of the current, w = eta_phys / eta_field at the
         * J_theta node, read from the nodal "circuit_linkage_weight"
         * register that the theta-implicit MHD solver fills right before
         * every measurement (ThetaImplicitMHD::FillCircuitLinkageWeight;
         * that solver has no displacement current: its vacuum is mocked
         * by the density-keyed boosted field resistivity eta_field =
         * sqrt(eta_phys^2 + eta_vac^2), and the current flowing where the
         * boost dominates is the numerical stand-in for the displacement
         * current, which must not induce eddy currents in the circuit).
         * The linkage becomes Int w mask A_theta J_theta dV: the weight
         * is applied by pre-multiplying J_theta (a scratch copy), so the
         * batched tables and the reference probes see the same weighted
         * current. w -> 1 in the plasma (exactly 1 with the boost off),
         * ~eta_phys/eta_vac -> 0 in the deep vacuum, eta_phys/eta_override
         * (~0) in the wall band under implicit_mhd.wall_band_eta_override
         * -- the band override IS the field eta there. Requires the
         * theta-implicit MHD solver (the register does not exist
         * otherwise). The Green's-function open-boundary source integral
         * keeps the TOTAL current (the ghost field must support the
         * interior curl B as it is); only the back-EMF excludes the mock
         * current. */
        ProbeWeight probe_weight = ProbeWeight::none;
        /** Optional region report file (circuit.probe_region_report; empty
         * = off): at every ACCEPTING evaluation and at the first residual
         * evaluation of every step, one row per J-based coil splitting
         * the UNWEIGHTED, UNMASKED linkage by radial region
         * (interior / band / exterior) and by weight class
         * (plasma w > 0.9 / mixed / boost-dominated w < 0.1), plus the
         * weighted totals, the value the coupler actually used and the
         * RMS of J_theta over the interior, band and exterior nodes
         * (columns: kind step t coil lambda_total lambda_interior
         * lambda_band lambda_exterior lambda_w_plasma lambda_w_mixed
         * lambda_w_boost lambda_weighted lambda_used jtheta_rms_interior
         * jtheta_rms_band jtheta_rms_exterior lambda_weighted_inside;
         * the python coupling reference's report carries the same
         * columns). A measurement of the split,
         * independent of probe_region / probe_weight (works with the
         * defaults too). Needs a wall polyline. */
        std::string probe_region_report;
    };

    /** Name of the nodal (node r, node z) linkage-weight register the
     * theta-implicit MHD solver fills for probe_weight = physical_share
     * and for the report's weight-class split; python reads the identical
     * field through
     * pywarpx.fields.MultiFabWrapper(mf_name="circuit_linkage_weight",
     * level=0)[...] (shape (nr + 1, nz + 1), node r x node z). */
    inline static constexpr char LinkageWeightName[] = "circuit_linkage_weight";

    /** \param probes           per-coil probe kind
     *  \param probe_exclusion  per-coil mask radius [m] of the loop probe
     *                          (ignored by the other kinds)
     *  \param region           the wall-polyline region table (owned by
     *                          the caller, must outlive the coupler);
     *                          required when params.probe_region is
     *                          wall_interior or a report is requested,
     *                          may be null otherwise */
    CircuitCoupler (warpx::circuit::CoilSet const& coils,
                    std::vector<warpx::circuit::ProbeKind> probes,
                    std::vector<double> probe_exclusion,
                    Params params,
                    std::unique_ptr<ExternalCircuit> plugin,
                    const warpx::circuit::ProbeRegionTable* region = nullptr);

    /** Whether the measurement needs the solver-filled linkage-weight
     * register (probe_weight = physical_share, or the report's
     * weight-class split): the theta-implicit MHD solver refreshes
     * LinkageWeightName right before every MeasureLinkages when true. */
    [[nodiscard]] bool NeedsLinkageWeight () const
    {
        return m_params.probe_weight == ProbeWeight::physical_share ||
               !m_params.probe_region_report.empty();
    }

    /** Open the region report file (I/O rank): truncate and write the
     * header, or append on a restart. Called once from
     * CircuitCoupling::InitData. */
    void InitRegionReport (bool restarting);

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
    void RefreshCircuitFields (amrex::Real t0, amrex::Real t1);
    void FireEngine (char const* hook, bool accept);
    [[nodiscard]] std::vector<amrex::Real> CoupledScales (amrex::Real t) const;
    /** The solver-filled nodal linkage weight, or nullptr when the
     * register does not exist (no theta-implicit MHD solver). */
    [[nodiscard]] const amrex::MultiFab* LinkageWeightField () const;
    /** Refresh the scratch J_theta * w on the J layout. */
    void RefreshWeightedCurrent (const amrex::MultiFab& j_theta,
                                 const amrex::MultiFab& weight);
    /** The node filter of the coupler's own measurement (the
     * probe_region mask; no weight class). */
    [[nodiscard]] warpx::circuit::ProbeNodeFilter MeasurementFilter () const;
    /** Append the report rows of the current field registers
     * (Params::probe_region_report); kind = "first" | "accept".
     * Collective. */
    void WriteRegionReport (const char* kind, amrex::Real time);

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
    std::vector<const amrex::MultiFab*> m_a_theta_scratch;
    /** Wall-polyline region table of Params::probe_region and the report
     * (caller-owned; null when neither is requested). */
    const warpx::circuit::ProbeRegionTable* m_region = nullptr;
    /** Scratch J_theta * w on the J_theta layout (Params::probe_weight and
     * the report's weighted columns). */
    amrex::MultiFab m_j_weighted;
    bool m_report_header_written = false;
};

#endif // WARPX_CIRCUIT_CIRCUITCOUPLER_H_
