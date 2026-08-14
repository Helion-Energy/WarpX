#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import sys

import numpy as np
import warpx_constants as constants
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
pressure0 = n0 * 10.0 * constants.elementary_charge
gamma = 5.0 / 3.0
relative_amplitude = 1.0e-4
number_of_cells = 64
cell_size = 1.0 / number_of_cells
wavenumber = 2.0 * np.pi
sound_speed = np.sqrt(gamma * (pressure0 + pressure0) / rho0)
dt = 0.1 * cell_size / sound_speed

# The fluid residual uses a centered cell-to-cell derivative. Its acoustic
# eigenvalue is therefore k_eff = sin(k*dz)/dz. Crank-Nicolson advances the
# right-going mode through the corresponding discrete phase.
effective_wavenumber = np.sin(wavenumber * cell_size) / cell_size
phase = 2.0 * np.arctan(0.5 * sound_speed * effective_wavenumber * dt)
z = (np.arange(number_of_cells) + 0.5) * cell_size
initial_mode = relative_amplitude * np.sin(wavenumber * z)
final_mode = relative_amplitude * np.sin(wavenumber * z - phase)
# The nodal Pe diagnostic is TEMPERATURE-primary: Te = p/(n kB) is formed
# natively on the cell grid, node-interpolated, remultiplied by the
# node-interpolated density (floors inactive here), and cell-averaged
# again for plotfile output. To first order in the amplitude this is the
# cos^2(k dz/2) smoothing of the old product interpolation (kept below
# for the looser final-state assert), but the ratio's O(amplitude^2)
# harmonics survive the tight initial assert, so the initial expected
# value is built through the exact discrete pipeline.
pressure_diagnostic_factor = np.cos(0.5 * wavenumber * cell_size) ** 2

n_cell = 1.0 + relative_amplitude * np.sin(wavenumber * z)
p_cell = 1.0 + gamma * relative_amplitude * np.sin(wavenumber * z)
te_cell = p_cell / n_cell
# cell -> node (periodic; node j sits between cells j-1 and j), the
# co-located nodal product, then node -> cell for the plotfile
te_nodal = 0.5 * (te_cell + np.roll(te_cell, 1))
n_nodal = 0.5 * (n_cell + np.roll(n_cell, 1))
pe_nodal = n_nodal * te_nodal
pe_diagnostic = 0.5 * (pe_nodal + np.roll(pe_nodal, -1))

initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
initial_pressure = initial["boxlib", "Pe"].value.ravel()
final_pressure = final["boxlib", "Pe"].value.ravel()

np.testing.assert_allclose(
    initial_density / rho0 - 1.0,
    initial_mode,
    rtol=2.0e-9,
    atol=2.0e-13,
)
np.testing.assert_allclose(
    initial_pressure / pressure0 - 1.0,
    pe_diagnostic - 1.0,
    rtol=2.0e-9,
    atol=2.0e-13,
)

# Terms quadratic in the perturbation are retained by the nonlinear residual,
# so the linear discrete wave is accurate through O(amplitude^2 * phase).
np.testing.assert_allclose(
    final_density / rho0 - 1.0,
    final_mode,
    rtol=4.0e-6,
    atol=2.0e-10,
)
np.testing.assert_allclose(
    final_pressure / pressure0 - 1.0,
    pressure_diagnostic_factor * gamma * final_mode,
    rtol=4.0e-6,
    atol=4.0e-10,
)

np.testing.assert_allclose(np.mean(final_density), rho0, rtol=5.0e-14, atol=0.0)
assert np.linalg.norm(final_density - initial_density) > 1.0e-13
assert np.all(final_density > 0.0)

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history[-1, 2] >= 1
assert newton_history[-1, 4] < 1.0e-9
