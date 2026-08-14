#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""Consistency gate of the open z caps: axial current column crossing the
caps (see README.rst).

A Gaussian axial current column J_z(r) = J0 exp(-r^2/a^2) crosses both
open z faces (r_hi, z_lo, z_hi all open; non-periodic z). The relaxed
toroidal field must satisfy Ampere's law, B_theta = mu0 I_enc(r)/(2 pi r).
The cap closure is d(r B_theta)/dz = mu0 r J_r = 0 (no radial current
beyond the cap, i.e. the enclosed axial-current profile I_z(r) freezes at
its boundary-plane value), so the cap-ghost B_theta must

1. equal the last valid plane at fixed radius EXACTLY (bitwise -- the
   continuation is a copy),
2. reproduce mu0 I_enc(r)/(2 pi r) to the interior Ampere tolerance,

while the corner ghosts (beyond the open r_hi face) carry the exact
r_c B_theta = const radial continuation of the wall cell. The poloidal
system is decoupled: with a purely axial source, Br/Bz and the psi table
contraction vanish identically -- the cap-ghost Br/Bz must be exactly 0.
"""

import numpy as np

from pywarpx import fields, picmi

constants = picmi.constants
mu0 = constants.mu0

NR, NZ = 32, 32
RMIN, RMAX = 0.0, 1.0
ZMIN, ZMAX = -0.5, 0.5
J0, A0 = 1.0e3, 0.15
MAX_STEPS = 170

grid = picmi.CylindricalGrid(
    number_of_cells=[NR, NZ],
    n_azimuthal_modes=1,
    lower_bound=[RMIN, ZMIN],
    upper_bound=[RMAX, ZMAX],
    lower_boundary_conditions=["none", "open"],
    upper_boundary_conditions=["open", "open"],
    lower_boundary_conditions_particles=["none", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
)

solver = picmi.HybridPICSolver(
    grid=grid,
    Te=1.0,
    n0=1.0e12,
    gamma=1.0,
    n_floor=1.0e12,
    # vacuum domain: rho = 0 < n_floor everywhere, so this drops the Hall and
    # electron-pressure terms in every cell (exactly linear resistive vacuum)
    holmstrom_vacuum_region=True,
    plasma_resistivity=mu0,
    substeps=20,
    Jz_external_function=f"{J0}*exp(-(x/{A0})^2)",
)

simulation = picmi.Simulation(
    solver=solver,
    time_step_size=6.0e-3,
    max_steps=MAX_STEPS,
    verbose=False,
)

simulation.step(MAX_STEPS)

Br_w = fields.BxFPWrapper()
Bt_w = fields.ByFPWrapper()
Bz_w = fields.BzFPWrapper()
ngr = int(Bt_w.mf.n_grow_vect[0])
ngz = int(Bt_w.mf.n_grow_vect[1])
Br = np.squeeze(Br_w[()])
Bt = np.squeeze(Bt_w[()])
Bz = np.squeeze(Bz_w[()])

dr = (RMAX - RMIN) / NR
r_cc = dr * (np.arange(-ngr, NR + ngr) + 0.5)  # cc radii incl. ghosts
rsel_v = slice(ngr, ngr + NR)  # valid radial cells in the gathered arrays
zsel_v = slice(ngz, ngz + NZ)  # valid axial cells (cc)

# Analytic Ampere profile of the Gaussian column at the valid cc radii
r_valid = r_cc[rsel_v]
I_enc = np.pi * A0**2 * J0 * (1.0 - np.exp(-(r_valid**2) / A0**2))
Bt_a = mu0 * I_enc / (2.0 * np.pi * r_valid)

Bt_valid = Bt[rsel_v, zsel_v]

# interior Ampere agreement (r >= 0.25; the axis cells have tiny local
# values) and z-uniformity, as in analysis_btheta.py
z_var = np.max(np.std(Bt_valid, axis=1)) / np.max(np.abs(Bt_a))
print(f"max z-variation of B_theta: {z_var:.3e}")
assert z_var < 1.0e-2, f"B_theta not z-uniform: {z_var:.3e}"
prof = Bt_valid.mean(axis=1)
sel = r_valid >= 0.25
rel_err = np.abs(prof[sel] - Bt_a[sel]) / Bt_a[sel]
print(f"max relative Ampere error (r >= 0.25): {rel_err.max():.3e}")
assert rel_err.max() < 0.01, f"Ampere profile error: {rel_err.max():.3e}"

# 1. cap ghosts must be the EXACT copy of the last valid plane (closure is
#    a z-invariant continuation at fixed radius)
for cap, ghost_js, valid_j in (
    ("z_lo", range(-ngz, 0), 0),
    ("z_hi", range(NZ, NZ + ngz), NZ - 1),
):
    for j in ghost_js:
        g = Bt[rsel_v, ngz + j]
        v = Bt[rsel_v, ngz + valid_j]
        assert np.array_equal(g, v), f"{cap} ghost row {j} is not the exact continuation"

    # 2. ...and therefore carry the Ampere value to the interior tolerance
    g = Bt[rsel_v, ngz + list(ghost_js)[0]]
    ghost_err = np.abs(g[sel] - Bt_a[sel]) / Bt_a[sel]
    print(f"{cap} ghost B_theta Ampere error (r >= 0.25): {ghost_err.max():.3e}")
    assert ghost_err.max() < 0.01, f"{cap} ghost Ampere error: {ghost_err.max():.3e}"

# corner ghosts: exact r_c B_theta = const continuation of the wall cell
rc_last = (NR - 0.5) * dr
for j in list(range(-ngz, 0)) + list(range(NZ, NZ + ngz)):
    jj = min(max(j, 0), NZ - 1)
    for gi in range(NR, NR + ngr):
        expected = Bt[ngr + NR - 1, ngz + jj] * rc_last / ((gi + 0.5) * dr)
        got = Bt[ngr + gi, ngz + j]
        assert np.isclose(got, expected, rtol=1e-13, atol=0.0), (
            f"corner ghost Bt({gi},{j}) = {got} != {expected}"
        )

# poloidal system must stay exactly zero (decoupled; psi contraction of a
# zero azimuthal current)
assert np.abs(Br).max() == 0.0, f"Br contaminated: {np.abs(Br).max():.3e}"
assert np.abs(Bz).max() == 0.0, f"Bz contaminated: {np.abs(Bz).max():.3e}"

print("open z-cap B_theta closure gate PASSED")
