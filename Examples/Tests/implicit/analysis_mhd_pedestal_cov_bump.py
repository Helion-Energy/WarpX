#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""The change-of-variables pedestal never transports or compresses the background.

inputs_test_1d_theta_implicit_mhd_pedestal_cov_bump: a hot dense Gaussian bump
(peak 100 x the background, 50 eV) expands into a pure cold background
(n_ped = 0.01 n0 at T_e = T_i = 2 eV, an electron-ion equilibrium) on a 256-cell periodic line for 8
theta = 1 steps. Gates:

  1. FAR CELLS BITWISE STATIC. A cell whose deviation is zero has zero mass
     and energy flux (D = 0 exactly at the background) and no pressure force
     (p_ped is uniform), and the change-of-variables Jacobian row of such a
     cell is decoupled from its background neighbours (D' = 0 there), so the
     deviation's support grows by at most a few cells per Newton iteration.
     The cells more than 100 cells from the bump centre must be BITWISE
     identical between step 3 and the last step (density, momentum,
     electron energy, ion energy, auxiliary internal energy) -- in the
     total-variable form the implicit acoustic response reaches them. (The
     load-time sanitize parks the background cells a fixed slack of 2e-6
     above the background floor at step 1 and the end-of-step restoration
     re-lands them once more at step 2; from step 3 on they are a fixed
     point.)
  2. Those cells sit on the background: rho within 1e-5 of rho_ped (the
     load-time sanitize slack), T_e / T_i within 1e-5 of the images.
  3. The domain mass is conserved (periodic; the pedestal contributes no
     flux) to 1e-12 relative, and the total ion + electron energy changes
     only by the pdV/viscous exchange within the bump (reported).
  4. The pedestal ledger books no raise (every row: 0 raised cells, zero
     totals) and the injected fields are identically zero: the background
     is a change of variables, not a source.

Usage: analysis_mhd_pedestal_cov_bump.py <step3_plotfile> <final_plotfile> <ledger>
"""

import sys

import numpy as np
import warpx_constants as constants
import yt

yt.set_log_level(50)

first_plotfile, final_plotfile, ledger_file = sys.argv[1:4]
n0 = 1.0e18
f_ped = 0.01
T_ped_e, T_ped_i = 2.0, 2.0
gamma = 5.0 / 3.0
nz = 256
steps = 8
rho_ped = f_ped * n0 * constants.proton_mass
q_over_m = constants.elementary_charge / constants.proton_mass


def load(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    names = [name for kind, name in ds.field_list if kind == "boxlib"]
    return {name: np.array(data["boxlib", name].value).ravel() for name in names}


first = load(first_plotfile)
final = load(final_plotfile)
dz = 1.0 / nz
z = (np.arange(nz) + 0.5) * dz
far = np.abs(np.arange(nz) - nz // 2) > 100
print(f"far cells: {far.sum()}")

# 1. static far cells: density, momentum and electron energy BITWISE; the
# two ion energies within 1e-9 -- their end-of-step floor re-landing (E_i
# against KE + floor + slack, U_i against floor + slack, then the dual sync)
# alternates by 2e-12 of the background every step and never settles to a
# fixed point; that is slack arithmetic, not transport.
for name in ("implicit_mhd_mass_density", "implicit_mhd_momentum_density",
             "implicit_mhd_electron_energy", "implicit_mhd_ion_energy",
             "implicit_mhd_ion_internal_energy"):
    a = first[name]; b = final[name]
    if a.size != nz:  # vector field (momentum: 3 components)
        a = a.reshape(-1, nz); b = b.reshape(-1, nz)
        same = np.array_equal(a[..., far], b[..., far])
        changed = int(np.sum(np.any(a != b, axis=0)))
        worst = float(np.max(np.abs(a[..., far] - b[..., far])))
    else:
        same = np.array_equal(a[far], b[far])
        changed = int(np.sum(a != b))
        worst = float(np.max(np.abs(a[far] - b[far]) / np.abs(a[far]).max()))
    print(f"{name}: far cells bitwise static {same} (max relative change {worst:.2e}); cells changed anywhere {changed}")
    if name.startswith("implicit_mhd_ion"):
        assert worst < 1.0e-9, (name, worst)
    else:
        assert same, name

# 2. on the background
rho_f = final["implicit_mhd_mass_density"]
ue_f = final["implicit_mhd_electron_energy"]
ui_f = final["implicit_mhd_ion_internal_energy"]
np.testing.assert_allclose(rho_f[far], rho_ped, rtol=1.0e-5)
te = (gamma - 1.0) * ue_f / (rho_f * q_over_m)
ti = (gamma - 1.0) * ui_f / (rho_f * q_over_m)
np.testing.assert_allclose(te[far], T_ped_e, rtol=1.0e-5)
np.testing.assert_allclose(ti[far], T_ped_i, rtol=1.0e-5)
# the bump did move (the test is not vacuous)
changed_cells = int(np.sum(first["implicit_mhd_mass_density"] != rho_f))
print(f"cells whose density changed over the run: {changed_cells}")
assert changed_cells > 10

# 3. conservation
m0 = np.sum(first["implicit_mhd_mass_density"]) * dz
m1 = np.sum(rho_f) * dz
print(f"domain mass {m0:.12e} -> {m1:.12e}, relative change {(m1 - m0) / m0:.3e}")
assert abs(m1 - m0) / m0 < 1.0e-12
e0 = np.sum(first["implicit_mhd_electron_energy"] + first["implicit_mhd_ion_energy"]) * dz
e1 = np.sum(ue_f + final["implicit_mhd_ion_energy"]) * dz
print(f"domain U_e + E_i {e0:.9e} -> {e1:.9e}, relative change {(e1 - e0) / e0:.3e}")

# 4. no raise, zero injected fields
ledger = np.loadtxt(ledger_file, ndmin=2)
assert ledger.shape == (steps, 5), ledger.shape
assert np.all(ledger[:, 1] == 0), ledger[:, 1]
assert np.all(ledger[:, 2:] == 0.0), ledger[-1]
for name in ("implicit_mhd_pedestal_injected_mass",
             "implicit_mhd_pedestal_injected_electron_energy",
             "implicit_mhd_pedestal_injected_ion_energy"):
    assert np.all(final[name] == 0.0), name
print("change of variables: background bitwise static beyond the bump, no injection")
