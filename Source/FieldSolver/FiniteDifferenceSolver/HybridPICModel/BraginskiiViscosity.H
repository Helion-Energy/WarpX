/* Copyright 2024-2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_BRAGINSKII_VISCOSITY_H_
#define WARPX_BRAGINSKII_VISCOSITY_H_

#include "Utils/WarpXConst.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>

/** \file BraginskiiViscosity.H
 *
 * PHYSICAL anisotropic electron viscosity: the Braginskii (1965) electron
 * stress tensor, its transport coefficients, and the viscous heating rate
 * it delivers to the electron energy equation.
 *
 * READ THIS BEFORE TOUCHING EITHER TERM
 * ------------------------------------
 * The hybrid model carries two operators that both look like grad^2 acting
 * on the current (because u_e ~ -J/(e n_e)), and they are NOT the same
 * thing. They are kept apart on purpose:
 *
 *   hyper-resistivity eta_H   NUMERICAL stabiliser for unphysical modes.
 *   (m_eta_h, Ohm's law)      Its magnitude is chosen for stability, not
 *                             from a transport coefficient. Its dissipation
 *                             is DELIBERATELY NOT booked as heating -- see
 *                             the m_eta_h declaration in HybridPICModel.H.
 *
 *   electron viscosity        PHYSICAL Braginskii stress. Its coefficients
 *   (this file)               come from the collision time and the
 *                             magnetisation, and its dissipation IS booked
 *                             into the electron energy equation, with its
 *                             own reduced-diagnostic column (P_nu).
 *
 * The two share a FORM and differ entirely in the COEFFICIENT. A reader who
 * sees only the form will conflate them; do not merge, do not fold one into
 * the other, and do not "fix" eta_H by booking it.
 *
 * WHY ANISOTROPIC, AND WHY THE PARALLEL CHANNEL IS FLUX-LIMITED
 * ------------------------------------------------------------
 * The dissipative electron viscosity is anisotropic by (Omega_ce tau_e)^2,
 * which in a magnetised hybrid target is O(1e12). An ISOTROPIC electron
 * viscosity is therefore physically meaningless in this regime: the
 * cross-field coefficient is ~10 orders below the numerical stabiliser
 * already present, while the field-aligned coefficient is ~3 orders ABOVE
 * it. Only the anisotropic form is worth carrying.
 *
 * Unbounded, the parallel coefficient is also unphysically large -- the same
 * pathology as unbounded Spitzer parallel thermal conduction, and it takes
 * the same remedy: a free-streaming flux limit (see
 * braginskii_visc_flux_limit below, the momentum-flux analogue of
 * qdsmc_conduction_flux_limit_factor).
 *
 * GYROVISCOSITY IS ABSENT BY CONSTRUCTION, NOT BY APPROXIMATION
 * ------------------------------------------------------------
 * Braginskii's eta_3 / eta_4 (gyroviscous) stress is REACTIVE: it transports
 * momentum without dissipating energy. Its contraction with the rate-of-
 * strain tensor vanishes identically, Pi^gyro : grad u == 0, so it
 * contributes EXACTLY ZERO to the heating computed here. Omitting it from
 * the energy booking is exact. It is not "small and dropped"; it is zero. Do
 * not add it to Q on the grounds that it exceeds the perpendicular
 * dissipative coefficient (it does, by ~6 orders -- in the MOMENTUM
 * equation, which this file does not touch).
 *
 * Reference: S. I. Braginskii, "Transport Processes in a Plasma", Reviews of
 * Plasma Physics Vol. 1 (Consultants Bureau, New York, 1965), pp. 205-311;
 * electron viscosity coefficients eta_0^e = 0.73 n_e T_e tau_e and
 * eta_1^e = 0.51 n_e T_e tau_e / (Omega_ce tau_e)^2, collision time tau_e
 * from Eq. (2.5e). The tensor bookkeeping below follows the standard
 * projector decomposition (see e.g. J. J. Ramos, Phys. Plasmas 12, 052102
 * (2005)); the anisotropic-transport discretisation conventions mirror the
 * Chacon-style tensor conduction operator already in this directory.
 *
 * This header is deliberately self-contained (AMReX scalar types and
 * PhysConst only, no solver state) so that the reduced diagnostic and the
 * solver evaluate the SAME function -- a re-derived diagnostic can drift
 * away from what the solver actually did.
 */

/** Braginskii electron collision time tau_e [s].
 *
 * \f[
 *   \tau_e = \frac{3\sqrt{m_e}\,(k_B T_e)^{3/2}\,(4\pi\epsilon_0)^2}
 *                 {4\sqrt{2\pi}\; n_e Z\, e^4 \ln\Lambda}
 * \f]
 *
 * (Braginskii 1965, Eq. 2.5e, written with n_i Z^2 = n_e Z for a
 * quasineutral single effective ion charge.)
 *
 * @param n_e        electron number density [m^-3]
 * @param Te_J       electron temperature [J]
 * @param Z_eff      effective ion charge state [-]
 * @param ln_lambda  Coulomb logarithm [-]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real braginskii_tau_e (amrex::Real const n_e, amrex::Real const Te_J,
                              amrex::Real const Z_eff,
                              amrex::Real const ln_lambda)
{
    using namespace amrex::literals;
    constexpr amrex::Real four_pi_eps0 =
        4.0_rt * MathConst::pi * PhysConst::epsilon_0;
    constexpr amrex::Real e2 = PhysConst::q_e * PhysConst::q_e;
    constexpr amrex::Real num_c =
        3.0_rt * four_pi_eps0 * four_pi_eps0;
    constexpr amrex::Real den_c =
        4.0_rt * 2.5066282746310002_rt * e2 * e2;   // 4 sqrt(2 pi) e^4
    return num_c * std::sqrt(PhysConst::m_e) * Te_J * std::sqrt(Te_J)
        / (den_c * n_e * Z_eff * ln_lambda);
}

/** Field-aligned kinematic electron viscosity nu_par [m^2/s], from
 *  Braginskii's eta_0^e = 0.73 n_e k_B T_e tau_e divided by the electron
 *  mass density m_e n_e. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real braginskii_nu_par (amrex::Real const Te_J, amrex::Real const tau_e)
{
    using namespace amrex::literals;
    return 0.73_rt * Te_J * tau_e / PhysConst::m_e;
}

/** Cross-field kinematic electron viscosity nu_perp [m^2/s], from
 *  Braginskii's eta_1^e = 0.51 n_e k_B T_e tau_e / (Omega_ce tau_e)^2.
 *
 *  The asymptotic Braginskii expression diverges as |B| -> 0; it is
 *  regularised here as
 *      nu_perp = (0.51/0.73) nu_par / (1 + (Omega_ce tau_e)^2),
 *  which is the Braginskii value to one part in 1e12 at the magnetisations
 *  this term is meant for, and smoothly returns the (nearly isotropic)
 *  unmagnetised value at a field null instead of infinity. Without this the
 *  b-hat direction -- which is pure noise at a null -- would be multiplied
 *  by an unbounded coefficient. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real braginskii_nu_perp (amrex::Real const nu_par,
                                amrex::Real const omega_ce_tau_e)
{
    using namespace amrex::literals;
    return (0.51_rt / 0.73_rt) * nu_par
        / (1.0_rt + omega_ce_tau_e * omega_ce_tau_e);
}

/** Free-streaming flux limit on the PARALLEL momentum flux (the momentum
 *  analogue of the conduction free-streaming limiter):
 *
 *      mu_eff = mu_par / (1 + |Pi_Brag| / (f Pi_fs)),
 *      |Pi_Brag| = mu_par |S|,   Pi_fs = n_e m_e v_te^2 = 2 n_e k_B T_e,
 *
 *  with v_te = sqrt(2 k_B T_e / m_e) and S the field-aligned rate-of-strain
 *  scalar (see braginskii_viscous_heating). Values f <= 0 disable it.
 *
 *  Beyond bounding the coefficient, the limiter bounds the HEATING RATE: it
 *  caps mu_eff |S| at f Pi_fs, so the temperature response is at most
 *  dTe/dt <= f |S| Te. A finite resolved shear therefore cannot drive Te to
 *  infinity in finite time, which unbounded Braginskii can.
 *
 * @param mu_par  unlimited parallel dynamic viscosity [Pa s]
 * @param S_mag   |S|, the field-aligned rate-of-strain scalar [1/s]
 * @param p_e     electron pressure n_e k_B T_e [Pa]
 * @param f       free-streaming factor (<= 0 disables) */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real braginskii_visc_flux_limit (amrex::Real const mu_par,
                                        amrex::Real const S_mag,
                                        amrex::Real const p_e,
                                        amrex::Real const f)
{
    using namespace amrex::literals;
    if (f <= 0.0_rt || p_e <= 0.0_rt) { return mu_par; }
    amrex::Real const pi_fs = 2.0_rt * p_e;
    return mu_par / (1.0_rt + mu_par * S_mag / (f * pi_fs));
}

/** Field-aligned rate-of-strain scalar
 *      S = b.W.b = 2 b_i b_j (grad u)_{ij} - (2/3) div u   [1/s],
 *  the only strain component the Braginskii parallel stress responds to.
 *
 * @param G  velocity gradient tensor G[i][j] = d u_j / d x_i [1/s]
 * @param b  unit vector along B (b[0..2]) */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real braginskii_strain_par (amrex::Real const G[3][3],
                                   amrex::Real const b[3])
{
    using namespace amrex::literals;
    amrex::Real bGb = 0.0_rt;
    amrex::Real div_u = 0.0_rt;
    for (int i = 0; i < 3; ++i) {
        div_u += G[i][i];
        for (int j = 0; j < 3; ++j) { bGb += b[i] * b[j] * G[i][j]; }
    }
    return 2.0_rt * bGb - (2.0_rt / 3.0_rt) * div_u;
}

/** Braginskii electron viscous heating rate Q_nu [W/m^3].
 *
 * With the traceless rate-of-strain tensor
 *     W_ij = G_ij + G_ji - (2/3) delta_ij tr(G),
 * the stress is split on the field-aligned projector P = bb - I/3,
 *     W^par = (3/2) S P,   S = b.W.b,      W^perp = W - W^par,
 * which are orthogonal under the Frobenius product (W^par : W^perp = 0), so
 * the dissipation separates exactly:
 * \f[
 *   Q_\nu = -\Pi : \nabla u
 *         = \tfrac{3}{4}\,\mu_\parallel S^2
 *         + \tfrac{1}{2}\,\mu_\perp \left( W\!:\!W - \tfrac{3}{2} S^2 \right).
 * \f]
 * The first term is Braginskii's eta_0 (parallel) dissipation exactly; the
 * second collects the remaining dissipative strain at the cross-field
 * coefficient. Both are separately non-negative, so the heating is
 * pointwise positive-definite -- this term can never cool a cell, and needs
 * no temperature floor of its own. Setting mu_perp = mu_par recovers the
 * isotropic result Q = (1/2) mu W:W.
 *
 * The gyroviscous stress contributes exactly zero here; see the file
 * comment.
 *
 * @param mu_par   parallel dynamic viscosity m_e n_e nu_par [Pa s]
 * @param mu_perp  cross-field dynamic viscosity m_e n_e nu_perp [Pa s]
 * @param G        velocity gradient tensor G[i][j] = d u_j / d x_i [1/s]
 * @param S        field-aligned strain from braginskii_strain_par(G, b);
 *                 this is the only route by which b-hat enters Q */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real braginskii_viscous_heating (amrex::Real const mu_par,
                                        amrex::Real const mu_perp,
                                        amrex::Real const G[3][3],
                                        amrex::Real const S)
{
    using namespace amrex::literals;

    amrex::Real div_u = 0.0_rt;
    for (int i = 0; i < 3; ++i) { div_u += G[i][i]; }
    amrex::Real const two_thirds_div = (2.0_rt / 3.0_rt) * div_u;

    amrex::Real WddW = 0.0_rt;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            amrex::Real const W_ij =
                G[i][j] + G[j][i] - ((i == j) ? two_thirds_div : 0.0_rt);
            WddW += W_ij * W_ij;
        }
    }

    // W^perp : W^perp = W:W - (3/2) S^2, non-negative by orthogonality;
    // the clamp only absorbs round-off when the strain is purely parallel.
    amrex::Real const W_perp2 =
        std::max(WddW - 1.5_rt * S * S, 0.0_rt);

    return 0.75_rt * mu_par * S * S + 0.5_rt * mu_perp * W_perp2;
}

#endif // WARPX_BRAGINSKII_VISCOSITY_H_
