#!/usr/bin/env python3
"""
Analysis for the RTZ m=1 theta-mode Ohm's-law hybrid test.

This is an in-situ check that the new RTZ theta-derivative finite-difference
operators carry an m=1 azimuthal perturbation through the Ampere/Faraday curls
of the hybrid solver (the axisymmetric smoke test leaves all theta terms zero).

It asserts:
  1. all field components are finite (no NaN/Inf from a broken theta operator),
  2. |Bz| stays bounded (no numerical blow-up),
  3. the m=1 azimuthal component of Bz is present and bounded relative to the
     m=0 (theta-averaged) part -- i.e. the theta terms neither zeroed the
     perturbation (m1/m0 -> 0) nor amplified it unphysically (m1/m0 -> large).

Usage: analysis_rtz_theta_mode.py [--path diags/diagXXXXXXX]
(if --path is omitted, the highest-step plotfile in ./diags is used)
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import yt

yt.set_log_level(50)


def latest_plotfile():
    cands = [p for p in glob.glob("diags/*") if os.path.exists(os.path.join(p, "Header"))]
    if not cands:
        raise FileNotFoundError("no plotfiles found under ./diags")

    def step(p):
        m = re.search(r"(\d+)\s*$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    return sorted(cands, key=step)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None, help="plotfile directory")
    args, _ = ap.parse_known_args()

    path = args.path or latest_plotfile()
    print(f"[analysis_rtz_theta_mode] reading {path}")

    ds = yt.load(path)
    cg = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )

    # RTZ plotfile axes are (r, theta, z); cylindrical B components are Br/Bt/Bz.
    fields = {f[1] for f in ds.field_list}
    comps = [c for c in ("Br", "Bt", "Bz", "Er", "Et", "Ez", "rho") if c in fields]
    print(f"[analysis_rtz_theta_mode] checking finiteness of: {comps}")

    # (1) finiteness of all available field components
    for c in comps:
        arr = np.asarray(cg[("boxlib", c)])
        assert np.isfinite(arr).all(), f"non-finite values found in {c}"

    Bz = np.asarray(cg[("boxlib", "Bz")])  # (nr, ntheta, nz)
    nr, nt, nz = Bz.shape
    assert nt >= 2, "need at least 2 theta cells to assess m=1 structure"

    # (2) boundedness (B0 ~ 0.028 T here; allow generous head-room for compression)
    bzmax = float(np.nanmax(np.abs(Bz)))
    print(f"[analysis_rtz_theta_mode] max|Bz| = {bzmax:.4e} T")
    assert bzmax < 1.0, f"|Bz| unexpectedly large ({bzmax:.3e} T) -- possible blow-up"

    # (3) azimuthal Fourier content: m=0 (mean) and m=1 (first harmonic amplitude)
    F = np.fft.rfft(Bz, axis=1)
    m0 = np.abs(F[:, 0, :]) / nt              # theta-averaged magnitude
    m1 = 2.0 * np.abs(F[:, 1, :]) / nt        # cos(theta) amplitude
    # only look where the (m=0) field is significant
    mask = m0 > 0.1 * np.nanmax(m0)
    assert mask.any(), "no significant Bz region found"
    ratio = np.median(m1[mask] / m0[mask])
    print(f"[analysis_rtz_theta_mode] median m1/m0 of Bz = {ratio:.4f} "
          f"(eps_theta=0.2 at t=0)")

    # The perturbation must survive the solver (theta terms active) but stay bounded.
    assert ratio > 0.02, (
        f"m=1 component vanished (m1/m0={ratio:.3e}); theta-curl terms may be inactive"
    )
    assert ratio < 1.0, (
        f"m=1 component unphysically large (m1/m0={ratio:.3e})"
    )

    print("[analysis_rtz_theta_mode] PASS")


if __name__ == "__main__":
    main()
    sys.exit(0)
