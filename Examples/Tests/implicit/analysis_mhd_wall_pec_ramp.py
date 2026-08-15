#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""Stair-step conducting shaped-wall test (RZ MHD): programmed-drive
handling, metal flux freeze, and stair-surface flux pinning, for BOTH
conductor contracts (third argument, default ``pec``).

A straight-cylinder wall polyline at r_w = 0.3 m sits in a resistive
vacuum with two programmed split-field drives ramping linearly from
zero: a uniform-Bz ramp (amplitude B0ext) and an in-domain coil blob
INSIDE the wall (A_theta = A0 exp(-((r-r0)^2+z^2)/a0^2)). The plotfiles
hold TOTAL fields.

wall_model = pec (total-field contract):

1. Metal flux freeze to SOLVER precision: at every masked location the
   wall projection makes the total tangential E vanish, so every fully
   masked cell retains its initial (zero) total field: both programmed
   drives are cancelled exactly and the interior's image response has
   zero exterior field.

2. Stair-surface flux pinning (the defining property of the analytic
   cylinder-shell image solution, discretized): the total poloidal flux
   through the disk bounded by the FIRST MASKED corner ring r_eff is
   pinned at zero for EVERY z row -- the discrete Faraday telescoping
   makes this exactly the invariant the masked E_theta ring enforces.

3. Coil presence + ramp exclusion: the relaxed interior carries the
   prescribed analytic coil field plus the wall-image correction; the
   uniform ramp is excluded. An over-covering mask (interior tracks
   -B_ext: no coil field) and a missing wall (ramp penetration) both
   fail.

wall_model = pec_response (plasma-response contract, EB parity): the
same mask pins the PLASMA-RESPONSE field only and the prescribed drive
is transparent. In this vacuum deck the relaxed plasma response is zero
(the pinned rings hold psi_plasma = 0 and the interior decays onto it),
so:

1. Drive transparency: the interior carries the FULL uniform ramp
   (pocket ratio 1, vs < 0.2 under pec) and the coil field.

2. Plasma-flux pinning to SOLVER precision: at every masked corner ring
   r_m the per-z-row disk flux satisfies psi_total(r_m, z) =
   psi_external(r_m, z) EXACTLY, because psi_plasma is pinned at its
   initial zero (masked E_plasma = 0, discrete Faraday telescoping) and
   the discrete disk sum of the external Bz telescopes to
   r_m * A_theta(r_m, z) exactly (the same discrete curl builds the
   external registers). The reference is therefore the ANALYTIC r * A
   of the two drives, checked at the stair ring and at a mid-metal
   ring. Under pec this quantity is 0 instead, so contract mixups fail
   loudly.
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
mode = sys.argv[3] if len(sys.argv) > 3 else "pec"
assert mode in ("pec", "pec_response"), f"unknown mode {mode}"

nr, nz = int(ds0.domain_dimensions[0]), int(ds0.domain_dimensions[1])
rmin = float(ds0.domain_left_edge[0])
rmax = float(ds0.domain_right_edge[0])
zmin = float(ds0.domain_left_edge[1])
dr = (rmax - rmin) / nr
dz = (float(ds0.domain_right_edge[1]) - zmin) / nz
r_cc = rmin + dr * (np.arange(nr) + 0.5)
z_cc = zmin + dz * (np.arange(nz) + 0.5)
z_nodes = zmin + dz * np.arange(nz + 1)
RCC, ZCC = np.meshgrid(r_cc, z_cc, indexing="ij")

# Stair-surface radius: first nodal radius on or outside the polyline
# (the same on-or-outside rule as the mask, tol = 1e-3 dr)
i_eff = int(np.ceil((R_WALL - 1.0e-3 * dr - rmin) / dr - 1.0e-12))
r_eff = rmin + i_eff * dr
print(f"mode = {mode}; stair-surface radius r_eff = {r_eff:.6f} (polyline {R_WALL})")

# Field scale of the relaxed state (coil + ramp class)
b_scale = max(B0EXT, A0 / R0)


def a_theta_ext(r, z):
    """Analytic external A_theta of both drives at full scale."""
    return 0.5 * B0EXT * r + A0 * np.exp(-((r - R0) ** 2 + z**2) / A_BLOB**2)


def psi_rows_at(bz, i_ring):
    """Per-z-row disk flux through the corner ring i_ring (cell-centered
    plotfile Bz = z-average of the two staggered planes, each carrying
    the exact conserved staggered disk sum)."""
    return (bz[:i_ring, :] * r_cc[:i_ring, None] * dr).sum(axis=0)


psi_scale = np.abs(np.cumsum(Bz1[:, nz // 2] * r_cc * dr)).max()

if mode == "pec":
    # --- 1. Metal flux freeze to solver precision ---------------------------
    # Cells whose complete Faraday edge sets are masked (one cell beyond
    # the stair covers every staggering). The outermost cell ring is
    # excluded: it averages the r_max boundary FACE Br, whose normal
    # component the domain r_hi boundary owns (PEC pins the plasma B_n
    # there, so the total carries the prescribed external B_r(r_max)
    # ~ 1e-13 T in this deck).
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

    # --- 2. Stair-surface TOTAL flux pinned at zero, per z row --------------
    err_psi = np.abs(psi_rows_at(Bz1, i_eff)).max() / psi_scale
    print(f"stair-ring |psi_total|max/psi_scale = {err_psi:.3e}")
    assert err_psi < 1.0e-6, f"stair-surface flux not pinned: {err_psi:.3e}"

    # --- 3. Coil present, ramp excluded --------------------------------------
    # Probe the analytic coil field at its peak-Bz location (r0, 0):
    # Bz_coil = A_theta (1/r - 2(r-r0)/a0^2); at (r0, 0) exactly A0/r0.
    bz_coil_peak = A0 / R0
    near = (np.abs(RCC - R0) <= 1.5 * dr) & (np.abs(ZCC) <= 1.5 * dz)
    ratio = Bz1[near].mean() / bz_coil_peak
    print(f"coil-peak Bz ratio (measured/prescribed) = {ratio:.4f}")
    # Measured 1.07 (blob discretization ~ (dr/a0)^2 plus the image
    # correction); over-covering mask -> 0, no wall -> +1 on top.
    assert 0.8 < ratio < 1.3, f"coil-peak field ratio out of bracket: {ratio:.4f}"

    # ... and the off-coil interior pocket must not carry the ramp
    pocket = (RCC <= 0.1) & (np.abs(ZCC) >= 0.35)
    err_pocket = np.abs(Bz1[pocket]).max() / B0EXT
    print(f"off-coil pocket max|Bz|/B0ext = {err_pocket:.3e}")
    assert err_pocket < 0.2, f"uniform ramp not excluded: {err_pocket:.3e}"

else:  # pec_response
    # --- 1. Plasma-flux pinning to solver precision --------------------------
    # psi_total(r_m, z) must equal the analytic external r_m * A_theta
    # (z-averaged over the two adjacent nodal planes, matching the
    # cell-centered plotfile) at the stair ring AND at a mid-metal ring;
    # the pec contract gives 0 here instead.
    for i_ring, label in ((i_eff, "stair"), (min(i_eff + 6, nr - 2), "metal")):
        r_m = rmin + i_ring * dr
        psi_ref = (
            r_m * 0.5 * (a_theta_ext(r_m, z_nodes[:-1]) + a_theta_ext(r_m, z_nodes[1:]))
        )
        err = np.abs(psi_rows_at(Bz1, i_ring) - psi_ref).max() / psi_scale
        print(f"{label}-ring |psi_total - r A_ext|max/psi_scale = {err:.3e}")
        assert err < 1.0e-6, f"{label}-ring plasma flux not pinned: {err:.3e}"

    # --- 2. Drive transparency ------------------------------------------------
    # The uniform ramp penetrates in full (discretely exact for the
    # linear-in-r A) up to the ~1% quasi-steady resistive lag of the
    # decaying tracking response; under pec this is < 0.2 instead.
    pocket = (RCC <= 0.1) & (np.abs(ZCC) >= 0.35)
    ramp_ratio = Bz1[pocket].mean() / B0EXT
    print(f"off-coil pocket mean Bz/B0ext = {ramp_ratio:.4f}")
    assert 0.95 < ramp_ratio < 1.05, f"drive not transparent: {ramp_ratio:.4f}"

    # ... and the coil field is present at its prescribed value (the
    # penetrated uniform ramp is subtracted: unlike pec, it is NOT
    # excluded here)
    bz_coil_peak = A0 / R0
    near = (np.abs(RCC - R0) <= 1.5 * dr) & (np.abs(ZCC) <= 1.5 * dz)
    ratio = (Bz1[near].mean() - B0EXT) / bz_coil_peak
    print(f"coil-peak Bz ratio (measured/prescribed) = {ratio:.4f}")
    assert 0.8 < ratio < 1.3, f"coil-peak field ratio out of bracket: {ratio:.4f}"

print(f"conducting shaped-wall ramp test ({mode}) PASSED")
