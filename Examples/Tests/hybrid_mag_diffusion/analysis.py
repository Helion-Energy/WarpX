#!/usr/bin/env python3
"""
Analytic check for constant-η magnetic diffusion of a Fourier By mode.

Expected (SI):
  By(x,t) ≈ B_peak0 * exp(-χ kx^2 t)
  kx = 2π / Lx,  χ = η / μ0

Reads AMReX plotfile VisMF min/max for By from Cell_H (no yt/openPMD required).
Pass if relative amplitude error vs exp-decay of the measured t=0 peak is < 1%.
"""
import glob
import os
import re
import sys

import numpy as np

mu0 = 1.2566370612685e-06
eta = 1.0e-2
Lx = 1.0
kx = 2.0 * np.pi / Lx
chi = eta / mu0
dt = 1.0e-9
nsteps = 20
t_end = nsteps * dt
decay = np.exp(-chi * kx * kx * t_end)


def by_minmax_from_diag(diag_dir):
    """Return (min, max) of By (component 0) from Level_0/Cell_H."""
    cell_h = os.path.join(diag_dir, "Level_0", "Cell_H")
    if not os.path.isfile(cell_h):
        raise FileNotFoundError(cell_h)
    text = open(cell_h).read()
    # VisMF trailing lines: "1,ncomp" then mins, blank, "1,ncomp" then maxs
    # Find last two float-list lines with 3 comma-separated values
    float_lines = []
    for line in text.splitlines():
        if re.match(r"^-?[0-9]", line.strip()):
            parts = [p for p in line.strip().rstrip(",").split(",") if p]
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if len(vals) >= 3:
                float_lines.append(vals)
    if len(float_lines) < 2:
        raise RuntimeError(f"Could not parse By min/max from {cell_h}:\n{text}")
    # mins then maxs
    mins, maxs = float_lines[-2], float_lines[-1]
    return mins[0], maxs[0]


def main():
    d0 = glob.glob("diags/diag1*")
    # Prefer numbered plotfiles
    candidates = sorted(glob.glob("diags/diag1*"))
    if not candidates:
        print("FAILED: no diags/diag1* plotfiles")
        sys.exit(1)

    # First and last dumps
    first = candidates[0]
    last = candidates[-1]
    print(f"first={first} last={last}")

    bmin0, bmax0 = by_minmax_from_diag(first)
    bmin1, bmax1 = by_minmax_from_diag(last)
    amp0 = max(abs(bmin0), abs(bmax0))
    amp1 = max(abs(bmin1), abs(bmax1))

    expected = amp0 * decay
    rel_err = abs(amp1 - expected) / max(amp0, 1e-30)

    print(f"chi = {chi:.6e} m^2/s  kx = {kx:.6e} 1/m  t = {t_end:.6e} s")
    print(f"decay factor = {decay:.8f}")
    print(f"By amp t=0   = {amp0:.8e} T")
    print(f"By amp t_end = {amp1:.8e} T")
    print(f"expected     = {expected:.8e} T")
    print(f"rel error    = {rel_err:.4e}")

    if amp0 < 1e-8:
        print("FAILED: initial By amplitude too small (B init missing?)")
        sys.exit(1)

    if amp1 > 0.999 * amp0 and decay < 0.999:
        print("FAILED: amplitude did not decay")
        sys.exit(1)

    tol = 0.01  # 1%
    if rel_err > tol:
        print(f"FAILED: relative error {rel_err} > {tol}")
        sys.exit(1)

    print("PASSED: hybrid magnetic diffusion analytic amplitude")
    sys.exit(0)


if __name__ == "__main__":
    main()
