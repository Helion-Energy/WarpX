#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""The vacuum dust drag must not read as ion heat when its kinetic drain is on.

Uniform dust (below implicit_mhd.vacuum_mass_density) slides along x at u0 in a
uniform B along x; every flux divergence vanishes and the vacuum drag nu_eff =
nu (1 - w), w the Holmstrom weight of the dust density, is the only dynamics:
|m| decays as ((1 - nu_eff dt/2)/(1 + nu_eff dt/2))^N under theta = 1/2.

mode "off" (the default): the ion total energy E_i is untouched by the drag, so
the lost kinetic energy reappears as ion INTERNAL energy, exactly (E_i constant
to 1e-10, Delta internal = -Delta KE).
mode "on" (implicit_mhd.vacuum_drag_kinetic_drain = 1): E_i loses exactly the
kinetic decay, so the internal energy is constant (1e-8 relative; the pairing is
discretely exact at theta = 1/2 up to the Newton tolerance) while KE decays as
predicted (2 %).
mode "zero" (nu = 0): nothing moves at all.

Usage: analysis_mhd_vacuum_drag_drain.py <off|on|zero> <initial_plotfile> <final_plotfile>
"""

import re
import sys
from pathlib import Path

import numpy as np
import yt

yt.set_log_level(50)

mode, initial_plotfile, final_plotfile = sys.argv[1], sys.argv[2], sys.argv[3]
assert mode in ("off", "on", "zero"), mode
used_inputs = Path("warpx_used_inputs").read_text()


def input_value(name, default=None):
    match = re.search(rf"^{re.escape(name)}\s*=\s*([^#\n]+)", used_inputs, re.MULTILINE)
    if match is None:
        assert default is not None, name
        return default
    return match.group(1).strip().strip('"')


m_p = 1.67262192369e-27
n0 = float(input_value("my_constants.n0"))
n_vac = float(input_value("my_constants.n_vac"))
u0 = float(input_value("my_constants.u0"))
nu = float(input_value("my_constants.nu"))
dt = float(input_value("my_constants.dt"))
max_step = int(input_value("max_step"))
drain = input_value("implicit_mhd.vacuum_drag_kinetic_drain", "0")
assert (drain in ("1", "true", "True")) == (mode == "on"), (drain, mode)
if mode == "zero":
    assert nu == 0.0, nu
rho0 = n0 * m_p
rho_vac = n_vac * m_p
weight = 0.5 * (1.0 + np.tanh((rho0 - rho_vac) / (0.3 * rho_vac)))  # ratio of the two parser densities: m_p cancels
nu_eff = nu * (1.0 - weight)
# theta = 1/2 momentum decay per step; the kinetic energy decays as its square
decay = ((1.0 - 0.5 * nu_eff * dt) / (1.0 + 0.5 * nu_eff * dt)) ** (2 * max_step)


def load(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    return {name: np.array(data["boxlib", name].value).ravel() for name in
            ("implicit_mhd_mass_density", "implicit_mhd_momentum_x", "implicit_mhd_momentum_y",
             "implicit_mhd_momentum_z", "implicit_mhd_ion_energy", "implicit_mhd_electron_energy")}


def kinetic(f):
    return 0.5 * (f["implicit_mhd_momentum_x"] ** 2 + f["implicit_mhd_momentum_y"] ** 2 +
                  f["implicit_mhd_momentum_z"] ** 2) / f["implicit_mhd_mass_density"]


f0, f1 = load(initial_plotfile), load(final_plotfile)
ke0, ke1 = kinetic(f0), kinetic(f1)
int0, int1 = f0["implicit_mhd_ion_energy"] - ke0, f1["implicit_mhd_ion_energy"] - ke1
ei_change = float(np.max(np.abs(f1["implicit_mhd_ion_energy"] - f0["implicit_mhd_ion_energy"])) / np.mean(f0["implicit_mhd_ion_energy"]))
int_change = float(np.max(np.abs(int1 - int0)) / np.mean(int0))
ke_ratio = float(np.mean(ke1) / np.mean(ke0))
pairing = float(np.max(np.abs((int1 - int0) + (ke1 - ke0))) / np.mean(ke0))
print(f"mode {mode}: w {weight:.4e} nu_eff {nu_eff:.4e} predicted KE ratio {decay:.5f}; measured KE ratio {ke_ratio:.5f}; "
      f"KE0/internal0 {np.mean(ke0)/np.mean(int0):.4f}; |dE_i| {ei_change:.3e}; |d internal| {int_change:.3e}; "
      f"|d internal + d KE|/KE0 {pairing:.3e}")
rho_init = f0["implicit_mhd_mass_density"]   # the code's own m_p (the parser constant), not this script's
assert float(np.max(np.abs(f1["implicit_mhd_mass_density"] - rho_init)) / np.mean(rho_init)) < 1.0e-12
assert float(np.max(np.abs(f1["implicit_mhd_electron_energy"] - f0["implicit_mhd_electron_energy"])) / np.mean(f0["implicit_mhd_electron_energy"])) < 1.0e-10
if mode == "zero":
    assert ke_ratio > 1.0 - 1.0e-12 and ke_ratio < 1.0 + 1.0e-12, ke_ratio
    assert ei_change < 1.0e-12 and int_change < 1.0e-12, (ei_change, int_change)
elif mode == "off":
    assert abs(ke_ratio - decay) < 0.02 * decay, (ke_ratio, decay)
    assert ei_change < 1.0e-10, ei_change
    assert pairing < 1.0e-8, pairing            # the lost kinetic energy is all internal now
    assert int_change > 0.05, int_change         # and it is a sizeable heating (KE0 ~ 14 % of internal0)
else:
    assert abs(ke_ratio - decay) < 0.02 * decay, (ke_ratio, decay)
    assert int_change < 1.0e-8, int_change       # the drain removed exactly the kinetic decay
    assert ei_change > 0.05, ei_change
print(f"vacuum drag kinetic drain ({mode}): PASS")
