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

#include "Coils/CoilSet.H"

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
     * refresh the curls and split-field registers, and compute + print the
     * discrete inductance table (RZ). Must run after the hybrid model's
     * external-field initialization. */
    void InitData ();

    [[nodiscard]] const warpx::circuit::CoilSet& GetCoilSet () const { return m_coils; }

private:
    warpx::circuit::CoilSet m_coils;
};

#endif // WARPX_CIRCUIT_CIRCUITCOUPLING_H_
