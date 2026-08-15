/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CIRCUIT_CIRCUITCOUPLING_H_
#define WARPX_CIRCUIT_CIRCUITCOUPLING_H_

#include "CircuitCoupling_fwd.H"

#include "CircuitCoupler.H"
#include "Coils/CoilSet.H"
#include "Coils/FluxProbes.H"

#include <memory>
#include <string>
#include <vector>

/**
 * \brief Coil / circuit-coupling subsystem of the hybrid-PIC solver.
 *
 * Owns the coil geometry (`circuit.coils`), fills the coils' unit external
 * vector potential fields from the axisymmetric ring kernel at their
 * reference currents, and computes the DISCRETE (mesh-realized) self- and
 * mutual-inductance table of the coil set on the run mesh -- the values a
 * coupled circuit model's ports must use ("discrete-L" rule; the discrete
 * self-inductance is resolution dependent by construction and is printed
 * loudly at initialization).
 *
 * The coils pair with entries of external_vector_potential.fields (the
 * hybrid solver's split external fields), whose runtime scale segments
 * (<name>.python_scale) are the drive interface: a circuit engine sets
 * s = I/I_ref per coupling interval and the field registers follow with the
 * exact discrete-Faraday E partner.
 */
class CircuitCoupling
{
public:
    /** Whether the coil subsystem is configured (circuit.coils non-empty).
     * Cheap; reads ParmParse only. */
    [[nodiscard]] static bool IsConfigured ();

    CircuitCoupling ();

    /** Validate the coil/external-field pairing, fill the unit fields,
     * refresh the curls and split-field registers, compute + print the
     * discrete inductance table (RZ), resolve the per-coil linkage probes
     * and construct the coupling engine (circuit.engine). Must run after
     * the hybrid model's external-field initialization. */
    void InitData ();

    [[nodiscard]] const warpx::circuit::CoilSet& GetCoilSet () const { return m_coils; }

    /** The coupling engine's coupler, or nullptr when no engine is
     * configured (circuit.engine = none). */
    [[nodiscard]] CircuitCoupler* Coupler () { return m_coupler.get(); }
    [[nodiscard]] const CircuitCoupler* Coupler () const { return m_coupler.get(); }

    /** Persist the live scale segments of the scale-driven external fields
     * into the checkpoint (IO rank; the drive state would otherwise reset
     * to initial_scale on restart and silently jump). A compiled engine's
     * own state is delegated to ExternalCircuit::WriteCheckpoint; the
     * Python engine re-seeds its own state (its linkage registers are
     * re-measured from the restored fields). */
    void WriteCheckpointData (std::string const& dir) const;

    /** Restore the scale segments on restart (all ranks read; the segments
     * are rank-replicated state). Tolerant of checkpoints from before the
     * circuit subsystem (no file: keeps initial_scale). */
    void ReadCheckpointData (std::string const& dir);

private:
    warpx::circuit::CoilSet m_coils;

    /** Per-coil linkage probe kind (parallel to the coil set); resolved at
     * InitData ("default" -> reciprocity with the Green's-function open
     * boundary active, disk otherwise). */
    std::vector<warpx::circuit::ProbeKind> m_probes;

    std::string m_engine = "none";          //!< none | callbacks | external
    std::string m_plugin_library;           //!< engine = external only
    CircuitCoupler::Params m_coupler_params;
    std::unique_ptr<CircuitCoupler> m_coupler;
};

#endif // WARPX_CIRCUIT_CIRCUITCOUPLING_H_
