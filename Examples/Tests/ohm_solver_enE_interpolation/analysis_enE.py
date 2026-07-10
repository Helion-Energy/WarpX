#!/usr/bin/env python3
"""Analysis for the enE interpolation-order test.

Modes:
  analysis_enE.py <plt> [<plt2> ...]   error of raw E vs analytic Ohm's law
                                       E = (curl B x B)/(mu0 e n), plus the
                                       discrete divergence of the raw B field;
                                       with several plotfiles, prints the
                                       error ratio relative to the first.
  analysis_enE.py --same <pltA> <pltB> assert all raw E/B fields are bitwise
                                       identical between the two plotfiles.
"""

import sys

import numpy as np
import yt
from scipy.constants import e as q_e
from scipy.constants import mu_0

yt.set_log_level(50)

B0 = 0.1
BAMP = 0.02
DENS = 1e18
K = 2.0 * np.pi
NCELL = 32
H = 1.0 / NCELL

STAG = {
    "Ex": (0.5, 0.0, 0.0),
    "Ey": (0.0, 0.5, 0.0),
    "Ez": (0.0, 0.0, 0.5),
    "Bx": (0.0, 0.5, 0.5),
    "By": (0.5, 0.0, 0.5),
    "Bz": (0.5, 0.5, 0.0),
}


def bfield(x, y, z):
    return (BAMP * np.sin(K * y), BAMP * np.sin(K * z), B0 + BAMP * np.sin(K * x))


def analytic_E(x, y, z):
    """Ohm's law E with J = the *discrete* Yee curl of the analytic B.

    For sinusoidal B the discrete curl equals the analytic curl times the
    exact factor sin(kh/2)/(kh/2), so this reference isolates the error of
    the staggering interpolation from the curl truncation error.
    """
    bx, by, bz = bfield(x, y, z)
    sinc = np.sin(K * H / 2) / (K * H / 2)
    fac = -BAMP * K / mu_0 * sinc
    jx, jy, jz = fac * np.cos(K * z), fac * np.cos(K * x), fac * np.cos(K * y)
    rho = q_e * DENS
    return (
        (jy * bz - jz * by) / rho,
        (jz * bx - jx * bz) / rho,
        (jx * by - jy * bx) / rho,
    )


def load_raw(plotfile):
    ds = yt.load(plotfile)
    out = {}
    for c in STAG:
        pieces = []
        for g in ds.index.grids:
            raw = g["raw", f"{c}_fp"].v
            i0 = np.rint(g.LeftEdge.v / H).astype(int)
            pieces.append((i0, raw))
        nx = max(i0[0] + r.shape[0] for i0, r in pieces)
        ny = max(i0[1] + r.shape[1] for i0, r in pieces)
        nz = max(i0[2] + r.shape[2] for i0, r in pieces)
        full = np.full((nx, ny, nz, pieces[0][1].shape[3]), np.nan)
        for i0, r in pieces:
            full[
                i0[0] : i0[0] + r.shape[0],
                i0[1] : i0[1] + r.shape[1],
                i0[2] : i0[2] + r.shape[2],
                :,
            ] = r
        assert not np.isnan(full).any()
        out[c] = full
    return out


def positions(shape, frac):
    x = (np.arange(shape[0]) + frac[0]) * H
    y = (np.arange(shape[1]) + frac[1]) * H
    z = (np.arange(shape[2]) + frac[2]) * H
    return np.meshgrid(x, y, z, indexing="ij")


def analyze(plotfile):
    raw = load_raw(plotfile)
    print(f"\n=== {plotfile} ===")

    errs = {}
    for idx, c in enumerate(("Ex", "Ey", "Ez")):
        num = raw[c][..., 0]
        X, Y, Z = positions(num.shape, STAG[c])
        ref = analytic_E(X, Y, Z)[idx]
        scale = max(np.abs(r).max() for r in analytic_E(X, Y, Z))
        err = np.abs(num - ref)
        errs[c] = err.max() / scale
        print(
            f"  {c}: max rel err {errs[c]:10.3e}   rms rel err {np.sqrt((err**2).mean()) / scale:10.3e}"
        )

    divb = sum((raw[c][..., 1] - raw[c][..., 0]) / H for c in ("Bx", "By", "Bz"))
    divb_norm = np.abs(divb).max() / (BAMP * K / H * H)  # normalized by b*k
    print(f"  max |div B| / (b k): {divb_norm:10.3e}")
    return errs


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--same":
        a, b = load_raw(args[1]), load_raw(args[2])
        worst = 0.0
        for c in a:
            d = np.abs(a[c] - b[c]).max()
            worst = max(worst, d)
            print(f"  {c}: max |diff| = {d:.3e}")
        assert worst == 0.0, "plotfiles differ!"
        print("PASS: all raw fields bitwise identical")
    else:
        results = [analyze(p) for p in args]
        if len(results) > 1:
            base = results[0]
            for p, r in zip(args[1:], results[1:]):
                print(f"\n=== error ratio {p} / {args[0]} ===")
                for c in ("Ex", "Ey", "Ez"):
                    print(f"  {c}: {r[c] / base[c]:8.4f}")
