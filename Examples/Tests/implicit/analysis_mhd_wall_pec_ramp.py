#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""Stair-step PEC shaped-wall test (RZ MHD): programmed-drive exclusion,
metal flux freeze, and stair-surface flux pinning.

A straight-cylinder wall polyline at r_w = 0.3 m (implicit_mhd.wall_model
= pec) sits in a resistive vacuum with two programmed split-field drives
ramping linearly from zero: a uniform-Bz ramp (amplitude B0ext) and an
in-domain coil blob INSIDE the wall (A_theta = A0 exp(-((r-r0)^2+z^2)/a0^2)).
The plotfiles hold TOTAL fields, so the perfect-conductor statements are
direct:

1. Metal flux freeze to SOLVER precision: at every masked location the
   wall projection makes the total tangential E vanish, so every fully
   masked cell retains its initial (zero) total field: both programmed
   drives are cancelled exactly and the interior's image response has
   zero exterior field. The masked rows are linear; the surviving field
   is the accumulated Newton/GMRES tolerance.

2. Stair-surface flux pinning (the defining property of the analytic
   cylinder-shell image solution, discretized): the total poloidal flux
   through the disk bounded by the FIRST MASKED corner ring r_eff is
   pinned at zero for EVERY z row -- the discrete Faraday telescoping
   makes this exactly the invariant the masked E_theta ring enforces,
   for the z-dependent coil drive as well as the uniform ramp. A mask
   that leaks or sits at the wrong radius breaks the per-row sum by the
   local vacuum field.

3. Coil presence + ramp exclusion: the relaxed interior carries the
   prescribed analytic coil field (Bz_coil(r0, 0) = A0/r0 at full scale)
   plus the wall-image correction. An over-covering mask makes the
   interior track -B_ext pointwise (no coil field survives: measured
   ratio 0); a missing wall adds the penetrated uniform ramp
   (B0ext = 5 A0/r0 here). Both fail the bracket.
"""

import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(0)

# Must match the inputs file
B0EXT = 1.0e-6  # [T] uniform-ramp amplitude at t_end (scale = 1)
A0 = 2.0e-7  # [T m] coil-blob vector-potential amplitude
R0 = 0.2  # [m] coil-blob radius
A_BLOB = 0.08  # [m] coil-blob width
R_WALL = 0.3  # [m] wall polyline radius


def load(fn):
    ds = yt.load(fn)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, (
        grid["boxlib", "Br"].v.squeeze(),
        grid["boxlib", "Bz"].v.squeeze(),
    )


ds0, (Br0, Bz0) = load(sys.argv[1])
ds1, (Br1, Bz1) = load(sys.argv[2])

nr, nz = int(ds0.domain_dimensions[0]), int(ds0.domain_dimensions[1])
rmin = float(ds0.domain_left_edge[0])
rmax = float(ds0.domain_right_edge[0])
dr = (rmax - rmin) / nr
dz = (float(ds0.domain_right_edge[1]) - float(ds0.domain_left_edge[1])) / nz
r_cc = rmin + dr * (np.arange(nr) + 0.5)
z_cc = float(ds0.domain_left_edge[1]) + dz * (np.arange(nz) + 0.5)
RCC, ZCC = np.meshgrid(r_cc, z_cc, indexing="ij")

# Stair-surface radius: first nodal radius on or outside the polyline
# (the same on-or-outside rule as the mask, tol = 1e-3 dr)
i_eff = int(np.ceil((R_WALL - 1.0e-3 * dr - rmin) / dr - 1.0e-12))
r_eff = rmin + i_eff * dr
print(f"stair-surface radius r_eff = {r_eff:.6f} (polyline {R_WALL})")

# Field scale of the relaxed state (coil + ramp class)
b_scale = max(B0EXT, A0 / R0)

# --- 1. Metal flux freeze to solver precision -------------------------------
# Cells whose complete Faraday edge sets are masked (one cell beyond the
# stair covers every staggering). The outermost cell ring is excluded:
# it averages the r_max boundary FACE Br, whose normal component the
# domain r_hi boundary owns (PEC pins the plasma B_n there, so the total
# carries the prescribed external B_r(r_max) ~ 1e-13 T in this deck).
metal = (RCC >= r_eff + 1.5 * dr) & (RCC < rmax - dr)
err_metal_bz = np.abs(Bz1[metal] - Bz0[metal]).max() / b_scale
err_metal_br = np.abs(Br1[metal] - Br0[metal]).max() / b_scale
print(f"metal    max|dBz_total|/scale = {err_metal_bz:.3e}")
print(f"metal    max|dBr_total|/scale = {err_metal_br:.3e}")
assert err_metal_bz < 1.0e-6, (
    f"metal total Bz not frozen to solver precision: {err_metal_bz:.3e}"
)
assert err_metal_br < 1.0e-6, (
    f"metal total Br not frozen to solver precision: {err_metal_br:.3e}"
)

# --- 2. Stair-surface flux pinned at zero, per z row -------------------------
# The cell-centered plotfile Bz is the z-average of the two staggered
# planes, each of which carries the exact conserved disk sum, so the
# cell-centered sum is exactly conserved as well.
psi_rows = (Bz1[:i_eff, :] * r_cc[:i_eff, None] * dr).sum(axis=0)
psi_scale = np.abs(np.cumsum(Bz1[:, nz // 2] * r_cc * dr)).max()
err_psi = np.abs(psi_rows).max() / psi_scale
print(
    f"stair-ring |psi_total|max/psi_scale = {err_psi:.3e} (psi_scale {psi_scale:.3e})"
)
assert err_psi < 1.0e-6, f"stair-surface flux not pinned: {err_psi:.3e}"

# --- 3. Coil present, ramp excluded ------------------------------------------
# Probe the analytic coil field at its peak-Bz location (r0, 0):
# Bz_coil = A_theta (1/r - 2(r-r0)/a0^2); at (r0, 0) exactly A0/r0. The
# wall-image correction is negative and modest (|psi_coil(r_eff)| ~ 20%
# of the coil flux); an over-covering mask gives ratio ~ 0 and a missing
# wall adds the full ramp (+5x A0/r0 here).
bz_coil_peak = A0 / R0
near = (np.abs(RCC - R0) <= 1.5 * dr) & (np.abs(ZCC) <= 1.5 * dz)
ratio = Bz1[near].mean() / bz_coil_peak
print(f"coil-peak Bz ratio (measured/prescribed) = {ratio:.4f}")
# Measured 1.07 (blob discretization ~ (dr/a0)^2 plus the image
# correction); over-covering mask -> 0, no wall -> +B0EXT/(A0/R0) = +1.
assert 0.8 < ratio < 1.3, f"coil-peak field ratio out of bracket: {ratio:.4f}"

# ... and the off-coil interior pocket must not carry the ramp: at
# (r < 0.1, |z| > 0.35) the coil field is < 1e-3 A0/r0, so anything
# there at the B0EXT scale is ramp penetration.
pocket = (RCC <= 0.1) & (np.abs(ZCC) >= 0.35)
err_pocket = np.abs(Bz1[pocket]).max() / B0EXT
print(f"off-coil pocket max|Bz|/B0ext = {err_pocket:.3e}")
assert err_pocket < 0.2, f"uniform ramp not excluded: {err_pocket:.3e}"

print("PEC shaped-wall ramp-exclusion test PASSED")
