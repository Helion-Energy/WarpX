#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""Gates for the coil-set drive against the Green's-function open boundary.

The twin baseline is the same discrete field the split-field machinery
realizes: the analytic loop A_theta of the (out-of-domain) coil pair,
sampled at the Yee E staggering (nodal r, nodal z for A_theta), curled
with the discrete RZ Yee curl, scaled by the Hann engagement s(t), and
averaged to cell centers exactly as the plotfile output is. The parser
strings in the deck and this scipy twin are the established
continuum-twin pair (modulus vs parameter conventions included), so any
disagreement beyond roundoff is a plumbing defect, not discretization.

Modes:
  vacuum <dump...> [rtol]
      Speciesless ride-through: every dump must satisfy
      B == s(t) * B_unit to rtol (default 1e-8) relative to max |B_unit|.
      This gates the analytic external advance, the Faraday pairing of
      (scale_B, scale_E), and -- the new contract -- that the Green's
      ghost fill at the open face never touches the ride-through field.
  lenz <mid> <end> [ref_end rtol]
      Plasma column: the response B - s(t) B_unit at mid-ramp must OPPOSE
      the applied field on axis (Lenz sign) with a sane magnitude. If a
      reference end dump is given (the explicit twin run), the end states
      must agree to rtol relative to max |B| (truncation-level match of
      the two integrators, transient-guard pattern).

Deck constants are duplicated from inputs_test_rz_open_bc_greens_coil_picmi.py
and must match.
"""

import sys

import numpy as np
import scipy.constants as spc
import scipy.special
import yt

yt.funcs.mylog.setLevel(0)

# --- deck constants (must match inputs_test_rz_open_bc_greens_coil_picmi.py)
RC = 1.3
ZC = 0.45
BC = 0.03
LR = 1.0
LZ = 2.0
NR = 32
NZ = 64
T_RAMP_CI = 1.0

B_REF = 2.0 * BC * RC**3 / (RC**2 + ZC**2) ** 1.5
T_CI = 2.0 * np.pi * spc.m_p / (spc.e * B_REF)
T_RAMP = T_RAMP_CI * T_CI

DR = LR / NR
DZ = LZ / NZ


def s_of_t(t):
    """Hann engagement, twin of the deck's A_time_external_function."""
    t = float(t)
    return 0.5 - 0.5 * np.cos(np.pi * t / T_RAMP) if t < T_RAMP else 1.0


def loop_A_theta(rr, zz, Rc, zc, current):
    """A_theta of a circular filament loop (continuum twin of the deck's
    parser strings; scipy elliptic integrals take the PARAMETER m = k^2,
    the parser functions take the modulus k)."""
    b2 = (rr + Rc) ** 2 + (zz - zc) ** 2
    a2 = (rr - Rc) ** 2 + (zz - zc) ** 2
    k2 = np.clip(1.0 - a2 / b2, 1e-15, 1.0 - 1e-15)
    term = ((2.0 - k2) * scipy.special.ellipk(k2) - 2.0 * scipy.special.ellipe(k2)) / k2
    psi = spc.mu_0 / (4.0 * np.pi) * 4.0 * rr * Rc / np.sqrt(b2) * term
    return current * np.where(rr > 0.0, psi / np.maximum(rr, 1e-300), 0.0)


def unit_B_cell_centered():
    """(Br, Bz) of the unit-scale coil pair: analytic A_theta sampled at
    the Yee staggering (nodal r, nodal z), discrete RZ Yee curl, then the
    same staggered-to-cell-center average the plotfile output applies."""
    r_nodes = np.arange(NR + 1) * DR
    z_nodes = -LZ / 2.0 + np.arange(NZ + 1) * DZ
    R, Z = np.meshgrid(r_nodes, z_nodes, indexing="ij")

    current = 2.0 * RC * BC / spc.mu_0  # on-axis loop-plane field = BC
    At = loop_A_theta(R, Z, RC, ZC, current) + loop_A_theta(R, Z, RC, -ZC, current)

    # Br(i, j+1/2) = -dz A_theta at nodal r; Bz(i+1/2, j) = (1/r) dr (r A)
    Br = -(At[:, 1:] - At[:, :-1]) / DZ
    r_cell = (np.arange(NR) + 0.5) * DR
    Bz = (r_nodes[1:, None] * At[1:, :] - r_nodes[:-1, None] * At[:-1, :]) / (
        r_cell[:, None] * DR
    )

    # staggered -> cell centers (the plotfile convention)
    Br_cc = 0.5 * (Br[1:, :] + Br[:-1, :])
    Bz_cc = 0.5 * (Bz[:, 1:] + Bz[:, :-1])
    return Br_cc, Bz_cc


def load_dump(fn):
    ds = yt.load(fn)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    B = {c: grid["boxlib", c].v.squeeze() for c in ("Br", "Bt", "Bz")}
    return B, float(ds.current_time)


def response(B, t, Br_u, Bz_u):
    s = s_of_t(t)
    return B["Br"] - s * Br_u, B["Bz"] - s * Bz_u


def main():
    mode = sys.argv[1]
    Br_u, Bz_u = unit_B_cell_centered()
    b_unit_max = max(np.abs(Br_u).max(), np.abs(Bz_u).max())
    print(f"unit coil field: max |B_unit| = {b_unit_max:.4e} T")

    if mode == "vacuum":
        args = sys.argv[2:]
        rtol = 1.0e-8
        if args and not args[-1].startswith("diag") and "/" not in args[-1]:
            rtol = float(args.pop())
        assert args, "vacuum mode needs at least one dump"
        for fn in args:
            B, t = load_dump(fn)
            dBr, dBz = response(B, t, Br_u, Bz_u)
            err = max(np.abs(dBr).max(), np.abs(dBz).max()) / b_unit_max
            err_t = np.abs(B["Bt"]).max() / b_unit_max
            print(
                f"t = {t:.4e} (s = {s_of_t(t):.4f}): ride-through residual "
                f"= {err:.3e}, |Bt| = {err_t:.3e}"
            )
            assert err < rtol, (
                f"vacuum ride-through violated at t = {t:.4e}: {err:.3e} "
                "(is the split external field being touched by the "
                "boundary operators or the ghost fill?)"
            )
            assert err_t < rtol, f"spurious toroidal field: {err_t:.3e}"
        print("Coil ride-through test PASSED")

    elif mode == "lenz":
        fn_mid, fn_end = sys.argv[2], sys.argv[3]
        fn_ref = sys.argv[4] if len(sys.argv) > 4 else None
        rtol = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0e-2

        B_mid, t_mid = load_dump(fn_mid)
        _, dBz = response(B_mid, t_mid, Br_u, Bz_u)
        s = s_of_t(t_mid)
        # axis-adjacent cells inside the column core
        core = dBz[:4, NZ // 2 - 4 : NZ // 2 + 4]
        applied = s * Bz_u[:4, NZ // 2 - 4 : NZ // 2 + 4]
        frac = -core.mean() / applied.mean()
        print(
            f"mid-ramp t = {t_mid:.4e} (s = {s:.4f}): core response / applied "
            f"= {-frac:.4f} (Lenz => negative)"
        )
        assert core.mean() * applied.mean() < 0.0, (
            "response does not oppose the applied flux (Lenz violation)"
        )
        # The deck's skin time mu0 R_col^2 / eta is ~24x the ramp, so the
        # column must be a near-perfect flux excluder at mid-ramp (measured
        # 0.984); leakage-dominated (<0.5) or over-cancelling (>1.02)
        # responses both indicate a broken drive coupling.
        assert 0.5 < frac < 1.02, f"implausible exclusion fraction {frac:.3f}"

        if fn_ref is not None:
            B_end, t_end = load_dump(fn_end)
            B_ref, t_ref = load_dump(fn_ref)
            assert abs(t_end - t_ref) < 1e-12 * max(t_end, 1e-30)
            scale = max(np.abs(B_end[c]).max() for c in ("Br", "Bz"))
            diff = max(np.abs(B_end[c] - B_ref[c]).max() for c in ("Br", "Bt", "Bz"))
            print(f"implicit vs explicit end state: {diff / scale:.3e}")
            assert diff / scale < rtol, (
                f"integrator cross-check failed: {diff / scale:.3e} >= {rtol:g}"
            )
        print("Coil Lenz test PASSED")

    else:
        raise SystemExit(f"unknown mode '{mode}'")


if __name__ == "__main__":
    main()
