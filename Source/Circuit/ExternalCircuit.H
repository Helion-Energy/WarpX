/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CIRCUIT_EXTERNALCIRCUIT_H_
#define WARPX_CIRCUIT_EXTERNALCIRCUIT_H_

#include <AMReX_REAL.H>

#include <string>
#include <vector>

/** ABI version of the plugin contract in this header. A plugin library must
 * export `extern "C" int warpx_external_circuit_abi_version ()` returning
 * the value it was compiled against; the loader refuses a mismatch. Bump on
 * ANY change to this header's virtual interface or factory signature. */
#define WARPX_EXTERNAL_CIRCUIT_ABI_VERSION 2

/**
 * \brief Abstract interface an external circuit engine implements to drive
 * the coil scales of the circuit-coupling subsystem from a compiled
 * plugin (circuit.engine = external, circuit.plugin_library = <path>).
 *
 * The engine owns the circuit model entirely (topology, elements, state,
 * its own sub-stepping); WarpX owns the coils, the field registers, the
 * flux-linkage probes and the predictor-corrector orchestration.
 *
 * The ABI is deliberately self-contained: a plugin implementation calls NO
 * WarpX or AMReX symbols (this header brings in only the amrex::Real
 * typedef), so it loads into any host -- the standalone executables or the
 * Python module (whose own symbols are not globally visible) -- without
 * link-time coupling. All data flows through the call arguments: the
 * coupler measures the flux linkages and hands the engine per-coil EMF
 * estimates; the engine returns the realized coil scales s = I / I_ref and
 * the COUPLER pushes them onto the field registers as linear segments over
 * [t0, t1].
 *
 * Call sequence per coupling step, mirroring the Python hook contract
 * (circuit.engine = callbacks):
 *   BeginStep                          once, after the split-field entry
 *   per substep [t0, t1] (possibly re-sized by the adaptive integrator):
 *     AdvanceInterval(accept = false)  the predictor evaluation, then once
 *                                      more per corrector pass with an
 *                                      improved eps
 *     AdvanceInterval(accept = true)   closes the accepted substep
 *   FinishStep                         once, before the split-field exit
 *
 * Every accept = false call advances from the same interval-entry state
 * (restore-and-re-advance): repeated evaluations must be smooth, so any
 * discontinuous internal state machines stay frozen at their last accepted
 * configuration and commit only on accept = true. A new interval's entry
 * state is the previous interval's accepted state.
 *
 * The eps entries follow the Define coil order. On the first (predictor)
 * evaluation of an interval the coupler has no fresh measurement and
 * passes its best available difference (zero at the very first call):
 * engines wanting a hold or extrapolated predictor keep their own held
 * EMF internally, exactly like the Python reference engines. Smoothing
 * (EMA etc.) of the raw per-interval EMF is engine-side policy.
 */
class ExternalCircuit
{
public:
    virtual ~ExternalCircuit () = default;

    /** One-time configuration after loading, before any stepping: the
     * coupled coil ports in coupler order (the eps and scale vectors of
     * AdvanceInterval use this order), their reference currents [A]
     * (s = I / I_ref), and the free-form circuit.plugin_config string
     * (typically a path to the engine's own configuration file; its
     * format is entirely the engine's business). */
    virtual void Define (std::vector<std::string> const& coil_names,
                         std::vector<amrex::Real> const& i_ref,
                         std::string const& config) = 0;

    /** A new coupling step begins: snapshot the accepted state. */
    virtual void BeginStep (amrex::Real t0, amrex::Real dt) = 0;

    /** Advance the interval [t0, t1] from its entry state with the given
     * per-coil EMF estimates [V] (dt_sub-averaged d(lambda_phys)/dt of the
     * measured plasma linkage), and fill `scales` (resized by the callee)
     * with the realized end-of-interval coil scales s(t1) = I(t1) / I_ref
     * in Define order. The coupler realizes them on the fields as linear
     * segments over [t0, t1]. Commit discontinuous internal state only
     * when accept is true. */
    virtual void AdvanceInterval (amrex::Real t0, amrex::Real t1,
                                  std::vector<amrex::Real> const& eps,
                                  bool accept,
                                  std::vector<amrex::Real>& scales) = 0;

    /** Close the step (histories, per-step bookkeeping). */
    virtual void FinishStep () = 0;

    /** Serialize / restore the engine state for checkpoint restart into /
     * from the given checkpoint directory (I/O rank only on write; the
     * scale segments themselves are checkpointed by WarpX). */
    virtual void WriteCheckpoint (std::string const& dir) const = 0;
    virtual void ReadCheckpoint (std::string const& dir) = 0;
};

extern "C"
{
    /** The factory symbol a plugin library exports:
     *     ExternalCircuit* warpx_create_external_circuit ();
     * together with the ABI stamp:
     *     int warpx_external_circuit_abi_version ();   */
    using warpx_create_external_circuit_t = ExternalCircuit* (*) ();
    using warpx_external_circuit_abi_version_t = int (*) ();
}

#endif // WARPX_CIRCUIT_EXTERNALCIRCUIT_H_
