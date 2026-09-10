#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""RZ gates of the change-of-variables pedestal on the dynamic column deck
(inputs_test_rz_theta_implicit_mhd_pedestal_cov; also run under the production
solver stack): the halo (loaded at 1e-5 rho0 below the background 1e-3 rho0)
is lifted once by the load-time sanitize; afterwards
  1. the ledger books no raise (0 cells, zero totals every step) and the
     injected fields are identically zero;
  2. the domain mass is conserved (periodic z, reflecting outer wall, nothing
     injected) to the solver tolerance;
  3. the background is never depleted (min density >= rho_ped within the
     sanitize slack) and never cooled below its floor (T_e >= T_ped,e, T_i >=
     T_ped,i everywhere, the background being the admissibility floor);
  4. the halo band rides on the background: median T_e / T_i of the cells at
     the background density within 25 % of the images (conduction from the
     column edge heats the adjacent cells).

Usage: analysis_mhd_pedestal_cov_rz.py <initial_plotfile> <final_plotfile> <ledger>
"""

import sys

import numpy as np
import warpx_constants as constants
import yt

yt.set_log_level(50)

initial_plotfile, final_plotfile, ledger_file = sys.argv[1:4]
n0 = 1.0e19
f_ped = 1.0e-3
T_ped_e, T_ped_i = 2.0, 2.0
gamma = 5.0 / 3.0
steps = 8
rho_ped = f_ped * n0 * constants.proton_mass
q_over_m = constants.elementary_charge / constants.proton_mass
FIELDS = ("implicit_mhd_mass_density", "implicit_mhd_electron_energy",
          "implicit_mhd_ion_internal_energy", "implicit_mhd_pedestal_injected_mass",
          "implicit_mhd_pedestal_injected_electron_energy",
          "implicit_mhd_pedestal_injected_ion_energy")


def load(plotfile):
    ds = yt.load(plotfile)
    dims = ds.domain_dimensions
    data = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)
    fields = {name: np.array(data["boxlib", name].value)[:, :, 0] for name in FIELDS}
    r_lo = float(ds.domain_left_edge[0].value); r_hi = float(ds.domain_right_edge[0].value)
    z_lo = float(ds.domain_left_edge[1].value); z_hi = float(ds.domain_right_edge[1].value)
    nr, nz = int(dims[0]), int(dims[1])
    dr = (r_hi - r_lo) / nr; dz = (z_hi - z_lo) / nz
    r = r_lo + (np.arange(nr) + 0.5) * dr
    volume = (2.0 * np.pi * r * dr * dz)[:, None] * np.ones((1, nz))
    return fields, volume


initial, volume = load(initial_plotfile)
final, _ = load(final_plotfile)
rho_i = initial["implicit_mhd_mass_density"]; rho_f = final["implicit_mhd_mass_density"]
ue_f = final["implicit_mhd_electron_energy"]; ui_f = final["implicit_mhd_ion_internal_energy"]

ledger = np.loadtxt(ledger_file, ndmin=2)
print("ledger:\n", ledger)
assert ledger.shape == (steps, 5), ledger.shape
assert np.all(ledger[:, 1] == 0) and np.all(ledger[:, 2:] == 0.0), ledger
for name in FIELDS[3:]:
    assert np.all(final[name] == 0.0), name

# 2. mass: the loaded halo is below the background, the sanitize lifted it
# (banner-booked) BEFORE the first step; the step-0 plotfile is the load, so
# compare against the lifted load: max(rho_load, rho_ped).
lifted = np.maximum(rho_i, rho_ped)
mass_lifted = np.sum(lifted * volume); mass_final = np.sum(rho_f * volume)
rel = (mass_final - mass_lifted) / mass_lifted
print(f"mass: lifted load {mass_lifted:.9e} kg, final {mass_final:.9e} kg, relative change {rel:.3e}")
assert abs(rel) < 1.0e-5, rel

# 3. never depleted, never below the floor
print(f"min rho / rho_ped = {rho_f.min() / rho_ped:.9f}")
assert rho_f.min() >= rho_ped * (1.0 - 1.0e-9)
te = (gamma - 1.0) * ue_f / (rho_f * q_over_m); ti = (gamma - 1.0) * ui_f / (rho_f * q_over_m)
print(f"min Te {te.min():.6f} eV (image {T_ped_e}), min Ti {ti.min():.6f} eV (image {T_ped_i})")
assert te.min() >= T_ped_e * (1.0 - 1.0e-6) and ti.min() >= T_ped_i * (1.0 - 1.0e-6)

# 4. the band on the background
band = rho_f <= 1.1 * rho_ped
print(f"band cells {band.sum()} of {band.size}; median Te {np.median(te[band]):.4f}, median Ti {np.median(ti[band]):.4f}")
assert band.sum() >= 0.4 * band.size
assert abs(np.median(te[band]) / T_ped_e - 1.0) < 0.25
assert abs(np.median(ti[band]) / T_ped_i - 1.0) < 0.25
print("change of variables (RZ): no injection, mass conserved, background intact")
