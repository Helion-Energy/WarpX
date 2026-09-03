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
 * \brief Volume-integrated dissipation power of the hybrid (Ohm's law)
 * solver, one column per channel:
 *
 *   P_eta  = Int eta(rho, |J|, t) |J|^2 dV
 *   P_etaH = -Int eta_H(rho, |B|) J . lap(J) dV
 *   P_nu   = Int Q_nu dV
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
 * must be booked as a true sink.
 *
 * THE THREE CHANNELS ARE BOOKED DIFFERENTLY ON PURPOSE:
 *
 *  - P_eta  (resistive) is a transfer into electron internal energy under
 *    the energy-equation closure, a true sink under the polytropic one.
 *  - P_etaH (hyper-resistive) is NEVER booked as heating. eta_H is a
 *    NUMERICAL stabiliser for unphysical modes; its magnitude is chosen for
 *    stability, not from a transport coefficient, so the energy it removes
 *    has no claim on T_e. This column REPORTS it; nothing receives it. Do
 *    not "fix" that by routing it into the electron energy equation -- see
 *    the m_eta_h declaration in HybridPICModel.H.
 *  - P_nu   (viscous) IS booked as heating. It is the PHYSICAL Braginskii
 *    electron stress (BraginskiiViscosity.H), and unlike the two terms
 *    above it is not re-derived here: the column integrates the solver's
 *    own nodal heating field hybrid_qdsmc_visc_heating_fp, so it cannot
 *    disagree with what was actually applied. It reads exactly zero when
 *    qdsmc_viscosity_model = none.
 *
 * P_etaH and P_nu have the same grad^2-on-the-current FORM and are kept in
 * separate columns precisely so the numerical and the physical dissipation
 * can never be confused for one another.
 *
 * Columns: P_eta [W], P_etaH [W], P_diss = P_eta + P_etaH + P_nu [W],
 *          P_nu [W].
 *
 * P_nu sits AFTER the total rather than beside the channels it belongs with,
 * deliberately: this file is read positionally across repositories, so columns
 * 0-4 keep their historical meaning and new channels are appended. Add future
 * channels the same way -- fold them into P_diss and append the column, never
 * insert one. Read by name from the header line, not by position.
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
