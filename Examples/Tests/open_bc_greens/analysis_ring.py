#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""Test 1 of the RZ Green's-function open boundary (see README.rst).

A Gaussian ring of azimuthal current is held fixed in a resistive vacuum and
the explicit hybrid solver relaxes B to the magnetostatic steady state,
curl B = mu0 J_ext, with boundary values from the field BC. This script:

1. reconstructs the analytic free-space (z-periodic) field of the discrete
   ring source by superposing exact current-loop psi values (the same
   ring-current Green's function used by the BC, evaluated per source node
   at full resolution -- no coarse-graining) and Yee-differencing psi
   exactly as the code does;
2. verifies the open-BC run matches this analytic field to discretization +
   coarse-graining order, globally and in a near-wall band;
3. measures the image-suppression factor relative to the PEC baseline run
   (which is distorted near the wall by its image currents).
"""

import sys

import numpy as np
import yt
from scipy.special import ellipe, ellipk

yt.funcs.mylog.setLevel(0)

mu0 = 4.0e-7 * np.pi

# ---------------------------------------------------------------------------
# Load the open-BC run (argv[1]) and the PEC baseline (argv[2])
# ---------------------------------------------------------------------------
fn_open = sys.argv[1]
fn_pec = sys.argv[2]


def load_B(fn):
    ds = yt.load(fn)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    Br = grid["boxlib", "Br"].v.squeeze()
    Bz = grid["boxlib", "Bz"].v.squeeze()
    extent = [
        ds.domain_left_edge[0].v,
        ds.domain_right_edge[0].v,
        ds.domain_left_edge[1].v,
        ds.domain_right_edge[1].v,
    ]
    return Br, Bz, extent, ds.domain_dimensions


Br_o, Bz_o, extent, dims = load_B(fn_open)
Br_p, Bz_p, _, _ = load_B(fn_pec)

nr, nz = int(dims[0]), int(dims[1])
rmin, rmax, zmin, zmax = extent
dr = (rmax - rmin) / nr
dz = (zmax - zmin) / nz
Lz = zmax - zmin

# ---------------------------------------------------------------------------
# Discrete source: J_theta sampled at its (nodal r, nodal z) staggering,
# exactly as hybrid_pic_model.Jy_external_grid_function is evaluated.
# ---------------------------------------------------------------------------
J0, r0, z0, a = 1.0e3, 0.4, 0.0, 0.1
r_nodes = rmin + dr * np.arange(nr + 1)
z_nodes = zmin + dz * np.arange(nz)  # periodic: node nz == node 0
RN, ZN = np.meshgrid(r_nodes, z_nodes, indexing="ij")
w = J0 * np.exp(-(((RN - r0) ** 2) + (ZN - z0) ** 2) / a**2) * dr * dz  # ring amps


def psi_loop(rb, zb, rs, zs):
    """psi = r A_theta at (rb, zb) per unit ring current at (rs, zs)."""
    d2 = (zb - zs) ** 2 + (rb + rs) ** 2
    m = 4.0 * rb * rs / d2
    return mu0 / (4.0 * np.pi) * np.sqrt(d2) * ((2.0 - m) * ellipk(m) - 2.0 * ellipe(m))


# psi kernel T[ie, is, djz] for all (eval r-node, source r-node, z-node offset),
# with the periodic image sum (per-image tail ~ 1/n^3). The filament
# self-term (eval node == source node, where the loop psi diverges) is set
# to zero; the comparison regions below lie outside the source support, so
# the dropped self-terms carry negligible current.
n_img = 80
djz = dz * np.arange(nz)
RB = r_nodes[:, None, None]
RS = r_nodes[None, :, None]
DZ = djz[None, None, :]
T = np.zeros((nr + 1, nr + 1, nz))
for n in range(-n_img, n_img + 1):
    with np.errstate(invalid="ignore", divide="ignore"):
        T += np.nan_to_num(
            psi_loop(RB, DZ + n * Lz, RS, 0.0), nan=0.0, posinf=0.0, neginf=0.0
        )

# psi at all nodal points, by circular convolution in z
psi = np.zeros((nr + 1, nz + 1))
je = np.arange(nz)
js = np.arange(nz)
D = (je[:, None] - js[None, :]) % nz  # (je, js) -> z-offset index
for i_s in range(nr + 1):
    if not np.any(w[i_s]):
        continue
    # T[:, i_s, D] has shape (nr+1, nz, nz); contract over js
    psi[:, :nz] += np.tensordot(T[:, i_s, :][:, D], w[i_s], axes=([2], [0]))
psi[:, nz] = psi[:, 0]  # periodic top node

# ---------------------------------------------------------------------------
# Yee-difference psi exactly as the code does, then average to cell centers
# exactly as the plotfile does.
# ---------------------------------------------------------------------------
r_cc = rmin + dr * (np.arange(nr) + 0.5)

# Br at (nodal r, cc z); Br = 0 on axis
Br_stag = np.zeros((nr + 1, nz))
Br_stag[1:, :] = -(psi[1:, 1:] - psi[1:, :-1]) / (r_nodes[1:, None] * dz)
# Bz at (cc r, nodal z)
Bz_stag = (psi[1:, :] - psi[:-1, :]) / (r_cc[:, None] * dr)

Br_a = 0.5 * (Br_stag[:-1, :] + Br_stag[1:, :])  # -> cell centers
Bz_a = 0.5 * (Bz_stag[:, :-1] + Bz_stag[:, 1:])

# ---------------------------------------------------------------------------
# Error metrics, normalized by the analytic field scale in each region
# ---------------------------------------------------------------------------
Bmag_a = np.sqrt(Br_a**2 + Bz_a**2)


def err(Br_g, Bz_g, mask):
    dB = np.sqrt((Br_g - Br_a) ** 2 + (Bz_g - Bz_a) ** 2)
    return dB[mask].max() / Bmag_a[mask].max()


# Probe regions: outside the source support (Gaussian ring at r0 = 0.4,
# a = 0.1; J/J0 < 1.3e-4 for r >= 0.7), where the analytic filament
# superposition (with self-terms dropped) is exact to discretization order.
outer = np.broadcast_to((r_cc >= 0.7)[:, None], (nr, nz))
wall_band = np.broadcast_to((r_cc > rmax - 4.5 * dr)[:, None], (nr, nz))

err_open_out = err(Br_o, Bz_o, outer)
err_open_wall = err(Br_o, Bz_o, wall_band)
err_pec_out = err(Br_p, Bz_p, outer)
err_pec_wall = err(Br_p, Bz_p, wall_band)
suppression = err_pec_wall / err_open_wall

print(f"outer     error: open = {err_open_out:.3e}   pec = {err_pec_out:.3e}")
print(f"wall-band error: open = {err_open_wall:.3e}   pec = {err_pec_wall:.3e}")
print(f"image-suppression factor (wall band, pec/open) = {suppression:.1f}")

# The open-BC solution must match free space to discretization +
# coarse-graining order (measured ~5e-3 on this grid) ...
assert err_open_out < 0.02, f"open-BC outer error too large: {err_open_out:.3e}"
assert err_open_wall < 0.02, f"open-BC wall-band error too large: {err_open_wall:.3e}"
# ... and the PEC image must be suppressed by at least the design acceptance
# factor of 100x (measured ~400x on this grid)
assert suppression > 100.0, f"image suppression too weak: {suppression:.1f}x"
# sanity: the PEC baseline really is image-distorted near the wall (i.e. the
# comparison is meaningful)
assert err_pec_wall > 0.10, (
    f"PEC baseline suspiciously free-space-like: {err_pec_wall:.3e}"
)

print("Ring test PASSED")
