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
 * Two dimensionless GYROVISCOSITY REQUIREMENT GATES are appended after the
 * power channels. They are not dissipation and take no part in P_diss; they
 * measure whether the run is anywhere near a regime in which a gyroviscous
 * (finite-Larmor-radius) stress would matter, which is the measurement taken
 * INSTEAD of implementing a Braginskii eta_3/eta_4 electron gyroviscosity:
 * the electron gyroviscous term is suppressed in the generalized Ohm's law by
 * at least the mass ratio relative to the other terms (King & Kruger,
 * arXiv:1407.3864, Sec. II), so the ION gates below bound the electron
 * requirement from above by m_e/m_i, and where a sheet narrows to the
 * electron gyroradius the FLR ordering a Braginskii closure rests on has
 * already failed anyway.
 *
 *   beta_i_over_4 = beta_i/4 = rho_i^2/(2 d_i^2) = mu0 <p_i> / (2 <|B|^2>)
 *   F_gyro        = (beta_i/4) (omega_dyn / Omega_ci)
 *
 * the wavenumber-independent gyroviscous-to-Hall stress ratio (Ferraro,
 * ApJ 662, 512 (2007), Sec. 3, stated there as A/H = beta_i/4) and the
 * gyroviscous force measured against magnetic tension (same paper, Eq. (8b),
 * whose omega_dyn is its disk rotation rate); see also Schnack et al., Phys.
 * Plasmas 13, 058103 (2006), Eqs. (55ab)/(56ab).
 * omega_dyn = |u_i|/L is the soft part of the criterion and the choices are
 * declared: |u_i| is the ION BULK flow speed from the same NGP moment deposit
 * that supplies T_i, and L = |B|/(mu0 |J_plasma|) is the resolved gradient
 * scale of the magnetic field, i.e. the local current-sheet width. Each
 * factor is a domain-volume average in its own right and the gates are
 * assembled from those averages, NOT averaged pointwise -- the pointwise
 * forms diverge at a field null and a volume average of them would report the
 * null rather than the plasma. Read F_gyro in decades, not as a calibrated
 * number.
 *
 * Columns: P_eta [W], P_etaH [W], P_diss = P_eta + P_etaH + P_nu [W],
 *          P_nu [W], beta_i_over_4 [-], F_gyro [-].
 *
 * P_nu sits AFTER the total rather than beside the channels it belongs with,
 * deliberately: this file is read positionally across repositories, so columns
 * 0-4 keep their historical meaning and new channels are appended. Add future
 * channels the same way -- fold them into P_diss and append the column, never
 * insert one. The two gates were appended under the same rule. Read by name
 * from the header line, not by position.
 *
 * The hyper-resistive integrand needs one-cell neighbor stencils and is
 * skipped on the outermost cell ring of the domain (boundary-condition
 * dependent there); the eta term covers the full domain. Nodal-staggered
 * components use owner masks so shared points count once.
 *
 * KNOWN LIMITATION -- P_etaH under hyper_resistivity_curl_curl. The P_etaH
 * integrand is the component-wise vector Laplacian form -eta_H J.lap(J),
 * unconditionally. When hybrid_pic_model.hyper_resistivity_curl_curl is on
 * the solver instead applies E_H = +curl(eta_H curl J), whose dissipation is
 * +Int eta_H |curl J|^2; the two differ by the eta_H grad(div J) terms, which
 * is the whole reason that option exists. The curl-curl integrand is not
 * implemented here, so with that flag set this column reports a quantity the
 * solver never produced. ComputeDiags raises a high-priority warning in that
 * case rather than returning the mismatched number silently. P_eta and P_nu
 * are unaffected. Fixing this means adding the native-Yee curl-curl integrand
 * (cylindrical curls, adjoint-paired as in the solver) behind the same flag.
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
