#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""implicit_mhd.wall_friction_heating = drop on the RZ no-slip deck.

The drop run is the dual-energy no-slip ledger deck
(test_rz_theta_implicit_mhd_viscous_heating_no_slip: sheared column
u_z = A r against a cylinder wall at r_w = 15 dr, ion_closure =
dual_energy, re-sync off, theta = 0.5, dt = 1e-8, 3 steps) with
wall_friction_heating = drop and every step written; the book twin is
that test's own output (plotfiles 0 and 3).

Row 14 is the wall-adjacent live row; its no-slip face at r_f = 15 dr
carries the wall shear tau_w = mu 2 u_t/dn with mu = rho nu (the deck has
no wall band), so under `book` the row heats at w 2 mu |u_t|^2/dn^2 with
w = r_f/r_c (the register's metric weight) -- 430x the interior shear
heating rho nu A^2 on this deck. Under `drop` that friction work leaves
E_i through the face instead, and the register skips it.

Checks:
  A  drop: the cell-wise ledger identity dU_i = d(E_i - KE) still holds
     in the wall row (both registers lose the friction) -- 3e-3 (the
     remaining heating there is ~1e-3 of book, so the O(dt^2) PdV term is
     relatively 430x larger than in the book gate).
  B  the friction heat is gone from U_i: dU_i(book) - dU_i(drop) in row
     14 equals the discrete friction deposit sum_n dt w 2 mu |u_t|^2/dn^2
     built from the drop run's own midpoint velocities (1e-3), and what
     is left in row 14 under drop is below 1 % of the book heating.
  C  the friction still removes the kinetic energy: the momentum fields
     of the two runs agree to 1e-6 of the shear momentum.
  D  the ledger closes with the dropped term: the fourth column of
     wall_ledger_file equals the volume integral of the row-14 friction
     deposit (1e-6) and equals the E_i the drop run lost relative to the
     book run (3e-4; the two flows differ at the level of the pressure the
     friction heat builds under book).

Usage:
    analysis_mhd_wall_friction_drop.py <drop diags dir> <book diags dir>
"""

import sys

import numpy as np
import yt

drop_dir, book_dir = sys.argv[1], sys.argv[2]

# Deck constants (inputs_test_rz_theta_implicit_mhd_wall_no_slip with the
# ctest overrides); m_p is the WarpX parser constant (CODATA 2022).
m_p = 1.67262192595e-27
n0 = 1.0e20
rho0 = n0 * m_p
A = 1.0e3
nu = 20.0
Lr, Lz = 0.48, 0.32
nr, nz = 24, 8
dr, dz = Lr / nr, Lz / nz
dt = 1.0e-8
n_steps = 3
wall_row = 14  # last live row; cells i >= 15 are masked (r_w = 0.30 m)
r_center = (np.arange(nr) + 0.5) * dr
r_face_wall = (wall_row + 1) * dr
weight_wall = r_face_wall / r_center[wall_row]
cell_volume = 2.0 * np.pi * r_center * dr * dz  # per z column

FIELDS = [
    "implicit_mhd_mass_density",
    "implicit_mhd_ion_energy",
    "implicit_mhd_ion_internal_energy",
    "implicit_mhd_momentum_r",
    "implicit_mhd_momentum_t",
    "implicit_mhd_momentum_z",
]


def load(plotfile):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return {name: grid["boxlib", name].value[:, :, 0] for name in FIELDS}


def kinetic(f):
    return (
        0.5
        * (
            f["implicit_mhd_momentum_r"] ** 2
            + f["implicit_mhd_momentum_t"] ** 2
            + f["implicit_mhd_momentum_z"] ** 2
        )
        / f["implicit_mhd_mass_density"]
    )


def internal(f):
    return f["implicit_mhd_ion_energy"] - kinetic(f)


drop = [load(f"{drop_dir}/diag{step:06d}") for step in range(n_steps + 1)]
book0, book1 = load(f"{book_dir}/diag000000"), load(f"{book_dir}/diag000003")
drop0, drop1 = drop[0], drop[-1]

# A. cell-wise identity under drop in the wall row (z-mean).
dU_drop = (drop1["implicit_mhd_ion_internal_energy"] - drop0["implicit_mhd_ion_internal_energy"])
dI_drop = internal(drop1) - internal(drop0)
ratio = dU_drop[wall_row].mean() / dI_drop[wall_row].mean()
# Under drop the wall row's remaining heating is ~1e-3 of the book value
# (the interior shear only), so the O(dt^2) PdV work of the developing
# radial flow -- 1.7e-7 of the book heating in this row -- is ~1e-3 of
# what is left (measured 8.9e-4); gate at 3e-3.
assert abs(ratio - 1.0) < 3.0e-3, f"drop: wall-row dU_i/d(E_i-KE) = {ratio:.6f}"
print(f"[drop] wall row dU_i/d(E_i-KE) = {ratio:.7f} (identity holds without the friction)")

# B. the friction deposit, from the drop run's own midpoint velocities:
# per step, w 2 mu |u_t|^2/dn^2 dt with u_t the tangential (theta, z)
# velocity of the wall row at the theta = 1/2 stage.
friction = np.zeros(nz)
for n in range(n_steps):
    f_a, f_b = drop[n], drop[n + 1]
    rho = 0.5 * (f_a["implicit_mhd_mass_density"][wall_row] + f_b["implicit_mhd_mass_density"][wall_row])
    u_t2 = 0.0
    for name in ("implicit_mhd_momentum_t", "implicit_mhd_momentum_z"):
        u_mid = 0.5 * (
            f_a[name][wall_row] / f_a["implicit_mhd_mass_density"][wall_row]
            + f_b[name][wall_row] / f_b["implicit_mhd_mass_density"][wall_row]
        )
        u_t2 = u_t2 + u_mid**2
    friction += dt * weight_wall * 2.0 * rho * nu * u_t2 / dr**2
dU_book = book1["implicit_mhd_ion_internal_energy"] - book0["implicit_mhd_ion_internal_energy"]
removed = dU_book[wall_row] - dU_drop[wall_row]
worst = np.max(np.abs(removed / friction - 1.0))
assert worst < 1.0e-3, (
    f"the heat removed from U_i in the wall row is not the friction deposit: worst {worst:.3e}"
)
residual = dU_drop[wall_row].mean() / dU_book[wall_row].mean()
assert residual < 1.0e-2, f"drop still heats the wall row: {residual:.3e} of book"
print(
    f"[drop] wall-row friction deposit removed from U_i: {removed.mean():.6e} J/m^3 vs "
    f"analytic {friction.mean():.6e} (worst {worst:.2e}); residual heating {residual:.2e} of book"
)

# C. the kinetic-energy removal is unchanged: same momentum fields.
scale = rho0 * A * Lr
for name in ("implicit_mhd_momentum_r", "implicit_mhd_momentum_t", "implicit_mhd_momentum_z"):
    diff = np.max(np.abs(drop1[name] - book1[name])) / scale
    assert diff < 1.0e-6, f"{name} differs between drop and book by {diff:.3e} of rho0 A Lr"
print("[drop] momentum fields agree with the book twin to 1e-6: the friction still removes the KE")

# D. the ledger closes with the dropped term.
ledger = np.loadtxt(f"{drop_dir}/wall_ledger.txt", ndmin=2)
assert ledger.shape[1] == 4, f"wall ledger rows have {ledger.shape[1]} columns, expected 4 under drop"
dropped = ledger[-1, 3]
friction_integral = np.sum(cell_volume[wall_row] * friction)
assert abs(dropped / friction_integral - 1.0) < 1.0e-6, (
    f"ledger friction {dropped:.9e} J vs the row-14 deposit integral {friction_integral:.9e} J"
)
delta_E_drop = np.sum(cell_volume[:, None] * (drop1["implicit_mhd_ion_energy"] - drop0["implicit_mhd_ion_energy"]))
delta_E_book = np.sum(cell_volume[:, None] * (book1["implicit_mhd_ion_energy"] - book0["implicit_mhd_ion_energy"]))
lost = delta_E_book - delta_E_drop
# Two different runs: their flows differ at the level of the pressure the
# friction heat builds under book (measured 6e-5 relative); gate at 3e-4.
assert abs(lost / dropped - 1.0) < 3.0e-4, (
    f"E_i lost relative to book {lost:.9e} J vs ledger {dropped:.9e} J"
)
print(
    f"[drop] ledger: friction energy dropped {dropped:.9e} J = row-14 deposit integral "
    f"{friction_integral:.9e} J = E_i lost vs book {lost:.9e} J"
)
print("PASS")
