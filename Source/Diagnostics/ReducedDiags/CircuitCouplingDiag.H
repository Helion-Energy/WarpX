/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_CIRCUITCOUPLINGDIAG_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_CIRCUITCOUPLINGDIAG_H_

#include "ReducedDiags.H"

#include <string>

/**
 * \brief Per-step ledger of the coil / circuit coupling: for every coil the
 * realized drive scale s = I/I_ref, its segment slope ds/dt, and the
 * measured plasma flux linkage lambda (lambda_phys * I_ref * n_turns; zero
 * for unmeasured coils or without a coupling engine), followed by the two
 * sides of the coupling-power double entry:
 *
 *     P_circuit = -sum_k (ds_k/dt) lambda_k
 *     P_field   = Int J_plasma . E_ext dV
 *
 * which are identical (to roundoff) whenever the drive slopes are constant
 * over the step -- E_ext of a scale-driven coil is exactly -(ds/dt) A_ref,
 * so the field-side integral telescopes into the circuit-side sum through
 * the same reciprocity dot the linkage probes use. This is the acceptance
 * identity of the coupling ("no power is created or lost in the exchange"),
 * and in coupled runs the imbalance measures the within-step slope
 * variation.
 */
class CircuitCouplingDiag : public ReducedDiags
{
public:
    CircuitCouplingDiag (const std::string& rd_name);

    void ComputeDiags (int step) final;
};

#endif // WARPX_DIAGNOSTICS_REDUCEDDIAGS_CIRCUITCOUPLINGDIAG_H_
