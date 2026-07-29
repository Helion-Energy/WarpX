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
alfven_speed = b0 / np.sqrt(constants.mu_0 * rho0)
dt = 0.1 * cell_size / alfven_speed
z = (np.arange(number_of_cells) + 0.5) * cell_size

initial_bx = initial["boxlib", "Bx"].value.ravel()
final_bx = final["boxlib", "Bx"].value.ravel()
expected_bx = delta_b * np.sin(2.0 * np.pi * (z - alfven_speed * dt))

np.testing.assert_allclose(
    initial_bx,
    delta_b * np.sin(2.0 * np.pi * z),
    rtol=2.0e-12,
    atol=2.0e-15,
)
np.testing.assert_allclose(
    final_bx,
    expected_bx,
    rtol=5.0e-5,
    # The centered spatial derivative gives a phase error of
    # O((k*dz)^2 * k*v_A*dt), or about 1.6e-9 T for this setup.
    atol=2.0e-9,
)
assert np.linalg.norm(final_bx - initial_bx) > 1.0e-7

final_density = final["boxlib", "implicit_mhd_mass_density"].value
np.testing.assert_allclose(
    np.mean(final_density),
    rho0,
    rtol=5.0e-14,
    atol=0.0,
)
assert np.all(final_density > 0.0)

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history[-1, 2] >= 1
assert newton_history[-1, 4] < 1.0e-9
