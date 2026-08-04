#!/usr/bin/env python3
"""C.7 regime survey: parallel-conduction stiffness numbers, no runs needed.

For each (deck, T_e) point, tabulates:
  chi_par      Spitzer/Braginskii parallel thermal diffusivity [m^2/s]
  S            chi_par * dt / dx^2 (grid-form explicit stiffness number;
               N_sub ~ S for an RKF45-embedded grid RHS)
  l_hop        x_max * sqrt(2 chi_par dt): outermost quiet-daughter hop for
               the given GH quadrature count [m]
  l_hop/dx     Redistribute-reach / hop-cap engagement indicator
  l_hop/R_c    curved-field-line leak indicator (spurious chi_perp/chi_par
               ~ (l_hop/R_c)^2 for straight hops)

Deck parameters below are REPRESENTATIVE PLACEHOLDERS — replace dx, dt, n_e,
B, R_c with the real liftoff / annulus / compression deck values before
drawing conclusions (marked TODO).

Physics:
  tau_e  = 3.44e5 * Te_eV^1.5 / (n_cc * lnL)   [s]   (NRL formulary)
  kappa_par = 3.16 * n_e * kB * Te * tau_e / m_e
  chi_par   = kappa_par / (1.5 * n_e * kB) = (3.16/1.5) * (kB Te/m_e) * tau_e
  chi_perp  ~ chi_par / (omega_ce * tau_e)^2   (Braginskii, for reference)
"""

import numpy as np

q_e = 1.602176634e-19
m_e = 9.1093837015e-31
kB = 1.380649e-23
lnL = 10.0

# Largest probabilists' Gauss-Hermite abscissa (units of sigma) per npts.
X_MAX = {2: 1.0, 3: np.sqrt(3.0), 4: 2.3344, 5: 2.8570, 7: 3.7504}

# TODO(Eric): replace with real deck numbers (dx, dt, n_e, B, R_c).
DECKS = [
    # name,        n_e [m^-3], B [T],  dx [m],  dt [s],   R_c [m]
    ("liftoff", 1.0e19, 1.0, 1.0e-3, 1.0e-9, 0.05),
    ("annulus", 5.0e18, 2.0, 5.0e-4, 5.0e-10, 0.03),
    ("compression", 1.0e20, 10.0, 5.0e-4, 2.0e-10, 0.10),
]

TE_EV = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
NPTS = [2, 3, 5]


def tau_e(te_ev, n_e):
    return 3.44e5 * te_ev**1.5 / ((n_e * 1e-6) * lnL)


def chi_par(te_ev, n_e):
    return (3.16 / 1.5) * (kB * te_ev * 11604.5 / m_e) * tau_e(te_ev, n_e)


def chi_perp_over_par(te_ev, n_e, b_t):
    om_ce = q_e * b_t / m_e
    x = om_ce * tau_e(te_ev, n_e)
    return 1.0 / x**2


F_LIMIT = 0.1  # free-streaming flux-limit factor
LT_OVER_DX = 10.0  # assumed temperature gradient scale L_T in units of dx


def main():
    for name, n_e, b_t, dx, dt, r_c in DECKS:
        print(
            f"\n=== {name}: n_e={n_e:.1e} m^-3, B={b_t} T, dx={dx:.1e} m, "
            f"dt={dt:.1e} s, R_c={r_c} m ==="
        )
        print(
            f"    (flux limiter: f={F_LIMIT}, L_T={LT_OVER_DX}*dx; "
            f"chi_eff = min(chi_Sp, f*v_te*L_T))"
        )
        hdr = (
            "Te[eV]   chi_Sp[m2/s]     S_Sp        chi_eff[m2/s]    S_eff    chi_pp/par"
        )
        for np_ in NPTS:
            hdr += f"  lhop_eff/dx(n={np_})  lhop_eff/Rc(n={np_})"
        print(hdr)
        for te in TE_EV:
            chi = chi_par(te, n_e)
            v_te = np.sqrt(kB * te * 11604.5 / m_e)
            chi_fs = F_LIMIT * v_te * LT_OVER_DX * dx
            chi_eff = min(chi, chi_fs)
            S = chi * dt / dx**2
            S_eff = chi_eff * dt / dx**2
            row = (
                f"{te:7.0f}  {chi:12.3e}  {S:10.2e}   {chi_eff:12.3e}  "
                f"{S_eff:8.2f}  {chi_perp_over_par(te, n_e, b_t):10.2e}"
            )
            for np_ in NPTS:
                lhop = X_MAX[np_] * np.sqrt(2.0 * chi_eff * dt)
                row += f"      {lhop / dx:8.2f}       {lhop / r_c:10.4f}"
            print(row)
    print(
        "\nReading: S is the grid-form explicit substep count scale "
        "(RKF45-embedded RHS); l_hop/dx > m_hop means the hop cap engages "
        "(capped fast-front transport); (l_hop/R_c)^2 is the straight-hop "
        "cross-field leak fraction (drives field-line-following hops)."
    )


if __name__ == "__main__":
    main()
