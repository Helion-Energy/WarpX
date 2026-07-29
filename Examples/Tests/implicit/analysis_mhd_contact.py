#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import sys

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

number_of_cells = 64
cell_centers = (np.arange(number_of_cells) + 0.5) / number_of_cells
number_density = 1.0e20
density_low = number_density * constants.proton_mass
density_high = 4.0 * density_low
electron_pressure = number_density * constants.elementary_charge
gamma = 5.0 / 3.0
electron_energy = electron_pressure / (gamma - 1.0)

expected_density = np.where(
    (cell_centers >= 0.25) & (cell_centers < 0.75),
    density_high,
    density_low,
)

initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
initial_current = initial["boxlib", "jz"].value.ravel()
final_current = final["boxlib", "jz"].value.ravel()
initial_energy = initial["boxlib", "implicit_mhd_electron_energy"].value.ravel()
final_energy = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()

np.testing.assert_allclose(
    initial_density,
    expected_density,
    rtol=5.0e-14,
    atol=0.0,
)

# The face-flux divergence must telescope on the periodic mesh. The ion
# current is (q_i/m_i) times ion momentum density, so its global sum provides
# a diagnostic for total momentum conservation without exposing the internal
# three-component momentum MultiFab.
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

# Backward-Euler Rusanov advection should preserve the invariant interval of
# this positive contact profile. Allow only roundoff-sized excursions.
density_tolerance = 2.0e-12 * density_high
assert np.min(final_density) >= density_low - density_tolerance
assert np.max(final_density) <= density_high + density_tolerance
assert np.all(final_density > 0.0)
assert np.max(np.abs(final_density - initial_density)) > 1.0e-3 * density_low

# Constant pressure and velocity form an exact contact solution. Electron
# energy must remain positive and constant while the density discontinuities
# advect and diffuse.
np.testing.assert_allclose(
    initial_energy,
    electron_energy,
    rtol=5.0e-14,
    atol=0.0,
)
np.testing.assert_allclose(
    final_energy,
    electron_energy,
    rtol=2.0e-11,
    atol=0.0,
)
assert np.all(final_energy > 0.0)

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
last_solve = newton_history[-1]
assert 1 <= last_solve[2] <= 12
assert (last_solve[4] <= 1.1e-11) or (last_solve[5] <= 1.1e-9)
