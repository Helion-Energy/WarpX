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
    struct Params
    {
        int corrector_iterations = 1;
        amrex::Real corrector_rtol = 1.0e-6;
        amrex::Real scale_floor = 1.0e-3;   //!< floor of the convergence denominator
    };

    CircuitCoupler (warpx::circuit::CoilSet const& coils,
                    std::vector<warpx::circuit::ProbeKind> probes,
                    Params params,
                    std::unique_ptr<ExternalCircuit> plugin);

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

private:
    void MeasureLinkages ();
    void RefreshCircuitFields (amrex::Real t0, amrex::Real t1);
    void FireEngine (char const* hook, bool accept);
    [[nodiscard]] std::vector<amrex::Real> CoupledScales (amrex::Real t) const;

    warpx::circuit::CoilSet const& m_coils;
    std::vector<warpx::circuit::ProbeKind> m_probes;
    Params m_params;
    std::unique_ptr<ExternalCircuit> m_plugin;

    Interval m_interval;
    int m_substep_count = 0;
    std::map<std::string, amrex::Real> m_lambda;        //!< latest measurement
    std::map<std::string, amrex::Real> m_lambda_start;  //!< at interval entry
};

#endif // WARPX_CIRCUIT_CIRCUITCOUPLER_H_
