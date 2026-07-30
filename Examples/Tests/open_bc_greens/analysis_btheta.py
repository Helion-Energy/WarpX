#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""Test 3 of the RZ Green's-function open boundary (see README.rst).

A Gaussian axial current column J_z(r) = J0 exp(-r^2/a^2) is held fixed in a
resistive vacuum. The relaxed toroidal field must satisfy Ampere's law,

    B_theta(r) = mu0 I_enc(r) / (2 pi r),
    I_enc(r) = pi a^2 J0 (1 - exp(-r^2/a^2)),

everywhere -- including at the outermost cells next to the open r_hi face,
where the ghost B_theta is the free-space continuation r B_theta = const
(no image/surface currents at the boundary). The solution must also stay
z-uniform.
"""

import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(0)

mu0 = 4.0e-7 * np.pi

ds = yt.load(sys.argv[1])
grid = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
Bt = grid["boxlib", "Bt"].v.squeeze()

nr, nz = int(ds.domain_dimensions[0]), int(ds.domain_dimensions[1])
rmin = float(ds.domain_left_edge[0].v)
rmax = float(ds.domain_right_edge[0].v)
dr = (rmax - rmin) / nr
r_cc = rmin + dr * (np.arange(nr) + 0.5)

# Analytic Ampere profile of the Gaussian column
J0, a = 1.0e3, 0.15
I_enc = np.pi * a**2 * J0 * (1.0 - np.exp(-(r_cc**2) / a**2))
Bt_a = mu0 * I_enc / (2.0 * np.pi * r_cc)

# z-uniformity of the relaxed solution
z_var = np.max(np.std(Bt, axis=1)) / np.max(np.abs(Bt_a))
print(f"max z-variation of B_theta: {z_var:.3e}")
assert z_var < 1.0e-2, f"B_theta not z-uniform: {z_var:.3e}"

Bt_profile = Bt.mean(axis=1)

# Full-profile agreement with Ampere's law (r >= 0.25, outside the axis
# region where the relative error is dominated by the small local value)
sel = r_cc >= 0.25
rel_err = np.abs(Bt_profile[sel] - Bt_a[sel]) / Bt_a[sel]
print(f"max relative Ampere error (r >= 0.25): {rel_err.max():.3e}")
assert rel_err.max() < 0.01, f"Ampere profile error too large: {rel_err.max():.3e}"

# The outermost cells, adjacent to the open face, must carry the exact
# mu0 I / (2 pi r) vacuum field (no image/surface current at the wall)
edge = np.abs(Bt_profile[-2:] - Bt_a[-2:]) / Bt_a[-2:]
print(f"open-face edge cells relative error: {edge.max():.3e}")
assert edge.max() < 0.01, f"open-face B_theta error too large: {edge.max():.3e}"

print("B_theta test PASSED")
