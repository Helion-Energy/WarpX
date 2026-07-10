#!/usr/bin/env python3
"""Analysis for the external A -> B curl test.

The raw B components are compared against the analytic (closed-form) curl
of the vector potential, and the error is ring-decomposed into azimuthal
harmonics: m=0 is the isotropic part of the curl truncation error, m=4 is
the axis-aligned (C4v) anisotropy of the per-axis Yee derivatives. The
discrete divergence of the raw B is also checked: since B is computed as
a discrete curl (with or without the isotropizing preconditioner), it
must vanish to round-off.

Usage:
  analysis_acurl.py label=plotfile [label=plotfile ...]
      print the decomposition (and cross-case m=4 ratios)

  analysis_acurl.py --case {acurl,acurl_iso} plotfile
      CI mode with assertions (measured at 64x64x8, h = a/10):
      - acurl: plain Yee curl, m=4 of B_z ~ 1.4e-4 of B_z(0)
      - acurl_iso: preconditioned curl, m=4 of B_z suppressed ~ 165x
      - both: discrete div(B) at round-off

Note on B_theta: it derives from A_z, which is read by BOTH in-plane
derivatives (Dx for By, Dy for Bx). No single per-component
preconditioner can isotropize both uses (the requirements conflict), so
the A_z-driven m=4 (~1.2e-5 here) is identical with and without the
option -- for this z-invariant A_z the correction terms vanish exactly.
Axisymmetric coil drives carry A_theta only, where every affected B
component is isotropized.
"""

import sys

import numpy as np
import yt
from analysis_m4 import harmonics, ring_sample
from equilibrium import C1, CURL_A_FUNCS, PROB_LO, YEE_POSITION, A, H

yt.set_log_level(50)

RADII = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
BCOMPS = ["Bx", "By", "Bz"]
BZ0 = 2.0 * C1 / A


def load_raw(plotfile):
    """Assemble {comp: full 4d raw array} from the (possibly split) boxes."""
    ds = yt.load(plotfile)
    out = {}
    for c in BCOMPS:
        pieces = []
        for g in ds.index.grids:
            raw = g["raw", f"{c}_fp"].v
            i0 = np.rint((g.LeftEdge.v - PROB_LO) / H).astype(int)
            pieces.append((i0, raw))
        shape = tuple(max(i0[d] + r.shape[d] for i0, r in pieces) for d in range(3)) + (
            pieces[0][1].shape[3],
        )
        full = np.full(shape, np.nan)
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


def analyze(plotfile, label):
    raw = load_raw(plotfile)

    print(f"\n=== case '{label}'  ({plotfile}) ===")

    # discrete divergence of the raw B (lo/hi face pairs per cell)
    divb = sum((raw[c][..., 1] - raw[c][..., 0]) / H[d] for d, c in enumerate(BCOMPS))
    divb_norm = np.abs(divb).max() * H[0] / BZ0
    print(f"  max |div B| * h / Bz(0): {divb_norm:10.3e}")

    # z-averaged error vs the analytic curl at each component's positions
    err2d = {}
    for c in BCOMPS:
        num3d = raw[c][..., 0]
        frac = np.array(YEE_POSITION[c])
        nx, ny, nz = num3d.shape
        x = PROB_LO[0] + (np.arange(nx) + frac[0]) * H[0]
        y = PROB_LO[1] + (np.arange(ny) + frac[1]) * H[1]
        z = PROB_LO[2] + (np.arange(nz) + frac[2]) * H[2]
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        err2d[c] = (num3d - CURL_A_FUNCS[c](X, Y, Z)).mean(axis=2)

    print("  ring decomposition of the error (normalized by Bz(0)):")
    print(f"   {'r':>5} | {'Bth m0':>9} {'Bth m4':>9} | {'Bz m0':>9} {'Bz m4':>9}")
    rows = {}
    for r0 in RADII:
        th, bx = ring_sample(err2d["Bx"], YEE_POSITION["Bx"], r0)
        _, by = ring_sample(err2d["By"], YEE_POSITION["By"], r0)
        _, bz = ring_sample(err2d["Bz"], YEE_POSITION["Bz"], r0)
        bth = -bx * np.sin(th) + by * np.cos(th)

        row = {"divb": divb_norm}
        for name, sig in (("Bth", bth), ("Bz", bz)):
            m0, amps = harmonics(sig)
            row[f"{name}_m0"] = m0 / BZ0
            row[f"{name}_m4"] = amps[4] / BZ0
        rows[r0] = row
        print(
            f"   {r0:5.2f} | {row['Bth_m0']:9.2e} {row['Bth_m4']:9.2e}"
            f" | {row['Bz_m0']:9.2e} {row['Bz_m4']:9.2e}"
        )
    return rows


def assert_case(case, plotfile):
    rows = analyze(plotfile, case)
    row = rows[0.30]
    # B is an exact discrete curl in both cases
    assert row["divb"] <= 1.0e-12, row["divb"]
    if case == "acurl":
        # plain Yee curl: m=4 from the per-axis derivative truncation
        assert 3.0e-5 <= row["Bz_m4"] <= 1.0e-3, row["Bz_m4"]
    elif case == "acurl_iso":
        # isotropized curl: m=4 of the A_theta-driven B_z cancelled at
        # leading order; the A_z-driven B_theta m=4 is unchanged (see the
        # module docstring)
        assert row["Bz_m4"] <= 5.0e-6, row["Bz_m4"]
        assert row["Bth_m4"] <= 3.0e-5, row["Bth_m4"]
        assert row["Bz_m0"] <= 1.0e-3, row["Bz_m0"]
    else:
        raise ValueError(f"unknown case '{case}'")
    print(f"PASS: case '{case}'")


if __name__ == "__main__":
    if sys.argv[1] == "--case":
        assert_case(sys.argv[2], sys.argv[3])
        sys.exit(0)

    cases = {}
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        cases[label] = analyze(path, label)

    labels = list(cases)
    if len(labels) > 1:
        base = labels[0]
        for other in labels[1:]:
            print(f"\n=== m=4 ratio: {other} / {base} ===")
            for r0 in RADII:
                a, b = cases[base][r0], cases[other][r0]
                print(
                    f"   {r0:5.2f} | Bth {b['Bth_m4'] / max(a['Bth_m4'], 1e-30):9.4f}"
                    f" | Bz {b['Bz_m4'] / max(a['Bz_m4'], 1e-30):9.4f}"
                )
