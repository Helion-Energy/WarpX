#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import re
import sys
from pathlib import Path

import numpy as np
import scipy.constants as constants
import yt


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])

number_density = 1.0e20
mass_density_reference = number_density * constants.proton_mass
electron_pressure_reference = number_density * constants.elementary_charge
gamma = 5.0 / 3.0
mass_density_floor = 1.0e-5 * mass_density_reference
electron_energy_floor = 1.0e-5 * electron_pressure_reference / (gamma - 1.0)

initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
initial_current = initial["boxlib", "jz"].value.ravel()
final_current = final["boxlib", "jz"].value.ravel()
initial_energy = initial["boxlib", "implicit_mhd_electron_energy"].value.ravel()
final_energy = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()

# The deliberately severe wave spans almost two orders of magnitude initially,
# then undergoes a large-CFL compression and rarefaction during one
# Crank-Nicolson step. Both constrained state variables must remain finite and
# strictly above their configured floors after endpoint extrapolation.
assert np.ptp(initial_density) > 100.0 * np.min(initial_density)
assert np.all(np.isfinite(final_density))
assert np.all(np.isfinite(final_energy))
assert np.min(final_density) > mass_density_floor
assert np.min(final_energy) > electron_energy_floor
assert np.max(np.abs(final_density - initial_density)) > mass_density_reference
assert np.max(np.abs(final_energy - initial_energy)) > electron_pressure_reference

# The periodic face-flux divergence telescopes, so total ion mass and momentum
# remain conserved even during the strong nonlinear solve. Ion current is
# charge-to-mass ratio times momentum density.
np.testing.assert_allclose(
    np.sum(final_density),
    np.sum(initial_density),
    rtol=2.0e-12,
    atol=0.0,
)
np.testing.assert_allclose(
    np.sum(final_current),
    np.sum(initial_current),
    rtol=2.0e-12,
    atol=0.0,
)

# Guard the intended time centering, verify that damping was actually active,
# and check nonlinear convergence.
used_inputs = Path("warpx_used_inputs").read_text()
theta_match = re.search(r"^implicit_evolve\.theta\s*=\s*([^#\s]+)", used_inputs, re.M)
assert theta_match is not None
np.testing.assert_allclose(float(theta_match.group(1)), 0.5, rtol=0.0, atol=0.0)
np.testing.assert_allclose(
    float(final_ds.current_time),
    2.0e-6,
    rtol=5.0e-14,
    atol=0.0,
)

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
last_solve = newton_history[-1]
assert 1 <= last_solve[2] <= 12
assert (last_solve[4] <= 1.1e-11) or (last_solve[5] <= 1.1e-9)
