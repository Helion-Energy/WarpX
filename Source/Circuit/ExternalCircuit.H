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

/**
 * \brief Abstract interface an external circuit engine implements to drive
 * the coil scales of the circuit-coupling subsystem from a compiled
 * plugin (circuit.engine = external, circuit.plugin_library = <path>).
 *
 * The engine owns the circuit model entirely (topology, elements, state,
 * its own sub-stepping); WarpX owns the coils, the field registers, the
 * flux-linkage probes and the predictor-corrector orchestration. Per
 * coupling interval [t0, t1] the coupler calls AdvanceInterval -- possibly
 * several times with improving EMF estimates (accept = false: restore the
 * interval-entry state, re-advance, hold any discontinuous switch state
 * machines at their last accepted configuration so repeated evaluations
 * stay smooth) and exactly once to close the interval (accept = true:
 * commit state machines and histories). After every call the realized coil
 * scales s = I / I_ref must be pushed by the engine through
 * ExternalVectorPotential::SetScale (linear segments over [t0, t1]).
 *
 * The Python-callback engine (circuit.engine = callbacks) implements the
 * identical contract through the circuitbeginstep / circuitpredict /
 * circuitcorrect / circuitfinish callbacks instead of this class.
 */
class ExternalCircuit
{
public:
    virtual ~ExternalCircuit () = default;

    /** A new coupling step begins: snapshot the accepted state. eps
     * estimates for the first interval come from the engine's own held
     * values (never zero-seed a loaded equilibrium's standing linkage). */
    virtual void BeginStep (amrex::Real t0, amrex::Real dt) = 0;

    /** Advance the interval [t0, t1] from its entry state with the given
     * per-coupled-coil EMF estimates [V], pushing the realized scale
     * segments; commit discontinuous internal state only when accept. */
    virtual void AdvanceInterval (amrex::Real t0, amrex::Real t1,
                                  std::vector<amrex::Real> const& eps,
                                  bool accept) = 0;

    /** Close the step (histories, per-step bookkeeping). */
    virtual void FinishStep () = 0;

    /** Serialize / restore the engine state for checkpoint restart. */
    virtual void WriteCheckpoint (std::string const& dir) const = 0;
    virtual void ReadCheckpoint (std::string const& dir) = 0;
};

extern "C"
{
    /** The factory symbol a plugin library exports:
     *     ExternalCircuit* warpx_create_external_circuit (); */
    using warpx_create_external_circuit_t = ExternalCircuit* (*) ();
}

#endif // WARPX_CIRCUIT_EXTERNALCIRCUIT_H_
