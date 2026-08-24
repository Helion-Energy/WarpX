/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_HYBRIDDISSIPATION_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_HYBRIDDISSIPATION_H_

#include "ReducedDiags.H"

#include <string>

/**
 * \brief Volume-integrated Ohmic and hyper-resistive dissipation power of
 * the hybrid (Ohm's law) solver:
 *
 *   P_eta  = Int eta(rho, |J|, t) |J|^2 dV
 *   P_etaH = -Int eta_H(rho, |B|) J . lap(J) dV
 *
 * evaluated on the same staggering, interpolations, and vector-Laplacian
 * stencils as the Ohm's-law kernels, with J the plasma current of the
 * step's final E-solve. Under the polytropic electron closure these terms
 * are a TRUE energy sink (no Ohmic-heating channel returns them to the
 * plasma); under the electron-energy-equation closure the eta term is a
 * transfer into the electron internal energy (its Joule source) and this
 * diagnostic then reports the transfer rate rather than a system loss.
 *
 * Booking note for the electron energy equation on this branch: the Joule
 * source (QDSMCAddJouleHeating) uses the dedicated
 * joule_heating_resistivity(rho,J,Te,t) parser when it is armed, which may
 * deliberately differ from the E-solve resistivity integrated here. P_eta
 * is the FIELD-side extraction; the electron-side receipt is the energy
 * budget's Joule column; their difference is the dissipation the heating
 * resistivity deliberately declines (e.g. the vacuum-regularizer ramp) and
 * must be booked as a true sink. The hyper-resistive term never enters the
 * heating source and is always a true sink.
 *
 * Columns: P_eta [W], P_etaH [W], P_diss = P_eta + P_etaH [W].
 *
 * The hyper-resistive integrand needs one-cell neighbor stencils and is
 * skipped on the outermost cell ring of the domain (boundary-condition
 * dependent there); the eta term covers the full domain. Nodal-staggered
 * components use owner masks so shared points count once.
 */
class HybridDissipation : public ReducedDiags
{
public:

    /**
     * \brief Constructor.
     *
     * \param[in] rd_name reduced diags names
     */
    HybridDissipation (const std::string& rd_name);

    /**
     * \brief Compute the dissipation powers.
     *
     * \param[in] step current time step
     */
    void ComputeDiags (int step) final;
};

#endif  // WARPX_DIAGNOSTICS_REDUCEDDIAGS_HYBRIDDISSIPATION_H_
