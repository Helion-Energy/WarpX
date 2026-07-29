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

n0 = 1.0e20
rho0 = n0 * constants.proton_mass
b0 = 0.1
delta_b = 1.0e-4
number_of_cells = 64
cell_size = 1.0 / number_of_cells
wavenumber = 2.0 * np.pi
hall_diffusivity = b0 / (constants.mu_0 * constants.elementary_charge * n0)
continuum_frequency = hall_diffusivity * wavenumber**2
dt = 0.01 / continuum_frequency

# The Yee curl-curl pair has k_eff = 2 sin(k*dz/2)/dz. Crank-Nicolson
# rotates the two transverse magnetic components through this discrete phase.
effective_wavenumber = 2.0 * np.sin(0.5 * wavenumber * cell_size) / cell_size
discrete_frequency = hall_diffusivity * effective_wavenumber**2
phase = 2.0 * np.arctan(0.5 * discrete_frequency * dt)
z = (np.arange(number_of_cells) + 0.5) * cell_size
initial_mode = delta_b * np.sin(wavenumber * z)

initial_bx = initial["boxlib", "Bx"].value.ravel()
initial_by = initial["boxlib", "By"].value.ravel()
final_bx = final["boxlib", "Bx"].value.ravel()
final_by = final["boxlib", "By"].value.ravel()

np.testing.assert_allclose(initial_bx, initial_mode, rtol=2.0e-12, atol=2.0e-15)
np.testing.assert_allclose(initial_by, 0.0, rtol=0.0, atol=1.0e-15)
np.testing.assert_allclose(
    final_bx,
    np.cos(phase) * initial_mode,
    rtol=2.0e-6,
    atol=2.0e-12,
)
np.testing.assert_allclose(
    final_by,
    np.sin(phase) * initial_mode,
    rtol=2.0e-6,
    atol=2.0e-12,
)

initial_density = initial["boxlib", "implicit_mhd_mass_density"].value
final_density = final["boxlib", "implicit_mhd_mass_density"].value
np.testing.assert_array_equal(final_density, initial_density)
np.testing.assert_allclose(final_density, rho0, rtol=5.0e-14, atol=0.0)

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history[-1, 2] >= 1
assert newton_history[-1, 4] < 1.0e-9
