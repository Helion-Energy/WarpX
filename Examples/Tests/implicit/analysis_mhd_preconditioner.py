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


def input_value(name):
    match = re.search(rf"^{re.escape(name)}\s*=\s*([^#\s]+)", used_inputs, re.MULTILINE)
    assert match is not None
    return match.group(1)


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])
used_inputs = Path("warpx_used_inputs").read_text()

number_density = 1.0e20
mass_density_reference = number_density * constants.proton_mass
pressure_reference = number_density * 10.0 * constants.elementary_charge
gamma = 5.0 / 3.0
relative_amplitude = 1.0e-4
magnetic_amplitude = 1.0e-8
theta = 0.5
resistive_number = 0.4

initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
initial_pressure = initial["boxlib", "Pe"].value.ravel()
final_pressure = final["boxlib", "Pe"].value.ravel()
initial_energy = initial["boxlib", "implicit_mhd_electron_energy"].value.ravel()
final_energy = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()
initial_bx = initial["boxlib", "Bx"].value.ravel()
final_bx = final["boxlib", "Bx"].value.ravel()

number_of_cells = initial_density.size
domain_length = float((initial_ds.domain_right_edge - initial_ds.domain_left_edge)[0])
cell_size = domain_length / number_of_cells
wavenumber = 2.0 * np.pi / domain_length
sound_speed = np.sqrt(
    gamma * (pressure_reference + pressure_reference) / mass_density_reference
)
dt = float(final_ds.current_time - initial_ds.current_time)
acoustic_cfl = sound_speed * dt / cell_size
np.testing.assert_allclose(acoustic_cfl, 10.0, rtol=2.0e-14, atol=0.0)

# The centered fluid residual has acoustic eigenvalue
# i*c_s*sin(k*dz)/dz. Crank-Nicolson therefore rotates the right-going
# Fourier mode by this discrete phase, even though the step spans ten cells
# at the sound speed.
effective_wavenumber = np.sin(wavenumber * cell_size) / cell_size
phase = 2.0 * np.arctan(0.5 * sound_speed * effective_wavenumber * dt)
z = (
    float(initial_ds.domain_left_edge[0])
    + (np.arange(number_of_cells) + 0.5) * cell_size
)
initial_mode = relative_amplitude * np.sin(wavenumber * z)
final_mode = relative_amplitude * np.sin(wavenumber * z - phase)

# Pe is interpolated from cell centers to the hybrid pressure grid and back
# to cell centers for plotfile output.
pressure_diagnostic_factor = np.cos(0.5 * wavenumber * cell_size) ** 2

np.testing.assert_allclose(
    initial_density / mass_density_reference - 1.0,
    initial_mode,
    rtol=2.0e-9,
    atol=2.0e-13,
)
np.testing.assert_allclose(
    initial_pressure / pressure_reference - 1.0,
    pressure_diagnostic_factor * gamma * initial_mode,
    rtol=2.0e-9,
    atol=2.0e-13,
)

# The nonlinear residual retains terms of order amplitude squared. At
# acoustic CFL 10 those terms are resolved but are not part of the linear
# sound-wave prediction tested here.
nonlinear_scale = 8.0 * relative_amplitude**2
np.testing.assert_allclose(
    final_density / mass_density_reference - 1.0,
    final_mode,
    rtol=5.0e-4,
    atol=nonlinear_scale,
)
np.testing.assert_allclose(
    final_pressure / pressure_reference - 1.0,
    pressure_diagnostic_factor * gamma * final_mode,
    rtol=5.0e-4,
    atol=2.0 * nonlinear_scale,
)

# The same solve also exercises the nonzero resistive curl--curl block. For
# the staggered Yee curl, k_eff=2*sin(k*dz/2)/dz, and the time-dependent eta
# was selected so that eta(t_n+theta*dt)*k_eff^2*dt/mu0=resistive_number. The
# O(drho*dB) ideal coupling to the acoustic wave is intentionally small
# compared with this theta-method decay.
magnetic_amplification = (1.0 - (1.0 - theta) * resistive_number) / (
    1.0 + theta * resistive_number
)
magnetic_mode = magnetic_amplitude * np.sin(wavenumber * z)
np.testing.assert_allclose(initial_bx, magnetic_mode, rtol=2.0e-10, atol=2.0e-18)
measured_magnetic_amplification = np.dot(final_bx, magnetic_mode) / np.dot(
    magnetic_mode, magnetic_mode
)
np.testing.assert_allclose(
    measured_magnetic_amplification,
    magnetic_amplification,
    rtol=2.0e-4,
    atol=2.0e-7,
)

# The periodic flux divergence telescopes, and both positive state variables
# must remain safely above zero during this large-CFL solve.
np.testing.assert_allclose(
    np.sum(final_density),
    np.sum(initial_density),
    rtol=5.0e-14,
    atol=0.0,
)
assert np.all(np.isfinite(final_density))
assert np.all(np.isfinite(final_energy))
assert np.min(final_density) > 0.999 * mass_density_reference
assert np.min(final_energy) > 0.999 * pressure_reference / (gamma - 1.0)

# Make the preconditioner selection and its stationary fixed-cycle settings
# part of the regression contract.
assert input_value("jacobian.pc_type") == "pc_mhd_block"
assert int(input_value("pc_mhd_block.field_iterations")) == 2
assert int(input_value("pc_mhd_block.fluid_iterations")) == 4
assert input_value("pc_mhd_block.agglomeration") == "true"
assert input_value("pc_mhd_block.consolidation") == "true"

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
last_solve = newton_history[-1]
assert 1 <= last_solve[2] <= 8
assert (last_solve[4] <= 1.1e-12) or (last_solve[5] <= 1.1e-10)
# The compact pressure Helmholtz plus resistive curl--curl approximation cuts
# this combined case from 333 unpreconditioned Krylov iterations to about 41.
# Keep margin for backend-dependent matrix-free roundoff without allowing a
# major loss of that reduction.
assert 0 < last_solve[7] <= 50
