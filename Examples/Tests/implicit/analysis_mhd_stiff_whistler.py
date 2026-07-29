#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Validate one large-CFL circularly polarized electron-MHD whistler step."""

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
background_field = 0.1
perturbation_field = 1.0e-4
mode_numbers = np.array([1, 2, 4, 7, 11, 16, 21, 27])
mode_weights = np.array([1.0, 0.7, -0.5, 0.35, -0.25, 0.18, 0.12, -0.08])

initial_bx = initial["boxlib", "Bx"].value.ravel()
initial_by = initial["boxlib", "By"].value.ravel()
final_bx = final["boxlib", "Bx"].value.ravel()
final_by = final["boxlib", "By"].value.ravel()
initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
initial_energy = initial["boxlib", "implicit_mhd_electron_energy"].value.ravel()
final_energy = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()

number_of_cells = initial_bx.size
domain_length = float((initial_ds.domain_right_edge - initial_ds.domain_left_edge)[0])
cell_size = domain_length / number_of_cells
dt = float(final_ds.current_time - initial_ds.current_time)
hall_diffusivity = background_field / (
    constants.mu_0 * constants.elementary_charge * number_density
)
whistler_cfl = hall_diffusivity * dt / cell_size**2
np.testing.assert_allclose(whistler_cfl, 20.0, rtol=2.0e-12, atol=0.0)

z = (
    float(initial_ds.domain_left_edge[0])
    + (np.arange(number_of_cells) + 0.5) * cell_size
)
phases = 2.0 * np.pi * np.outer(mode_numbers, z) / domain_length
initial_bx_expected = perturbation_field * np.sum(
    mode_weights[:, np.newaxis] * np.cos(phases), axis=0
)
initial_by_expected = perturbation_field * np.sum(
    mode_weights[:, np.newaxis] * np.sin(phases), axis=0
)
np.testing.assert_allclose(initial_bx, initial_bx_expected, rtol=3.0e-12, atol=2.0e-15)
np.testing.assert_allclose(initial_by, initial_by_expected, rtol=3.0e-12, atol=2.0e-15)

# For B+ = Bx+i*By, the homogeneous electron-MHD Hall operator gives
#
#     d B+ / dt = i*omega_h B+.
#
# The staggered Yee curl pair has
# k_eff=2*sin(k*dz/2)/dz and omega_h=D_H*k_eff^2. Crank-Nicolson therefore
# advances each circular eigenmode by the unit-modulus amplification
#
#     G_k=(1+i*a_k)/(1-i*a_k),  a_k=omega_h,k*dt/2.
#
# The chosen grid whistler CFL is 20. The broadband same-helicity state
# includes modal omega_h*dt from about 0.19 through 75.3, far outside an
# explicit Hall stability limit while remaining an exact linear solution in
# this 1D homogeneous equilibrium.
wavenumbers = 2.0 * np.pi * np.fft.fftfreq(number_of_cells, d=cell_size)
effective_wavenumbers = 2.0 * np.sin(0.5 * wavenumbers * cell_size) / cell_size
discrete_frequencies = hall_diffusivity * effective_wavenumbers**2
theta_number = 0.5 * discrete_frequencies * dt
amplification = (1.0 + 1.0j * theta_number) / (1.0 - 1.0j * theta_number)

initial_polarization = initial_bx + 1.0j * initial_by
final_polarization = final_bx + 1.0j * final_by
expected_polarization = np.fft.ifft(amplification * np.fft.fft(initial_polarization))
np.testing.assert_allclose(
    final_polarization,
    expected_polarization,
    rtol=2.0e-8,
    atol=2.0e-12,
)
np.testing.assert_allclose(
    np.abs(final_polarization),
    np.abs(expected_polarization),
    rtol=2.0e-8,
    atol=3.0e-12,
)

# Frozen ions and a homogeneous electron pressure leave rho and Ue exactly
# unchanged. This also checks that enabling the pressure term consistently
# with Hall introduces no spurious force in the homogeneous equilibrium.
np.testing.assert_array_equal(final_density, initial_density)
np.testing.assert_array_equal(final_energy, initial_energy)
np.testing.assert_allclose(
    final_density,
    mass_density_reference,
    rtol=5.0e-14,
    atol=0.0,
)
np.testing.assert_allclose(
    final_energy,
    pressure_reference / (gamma - 1.0),
    rtol=5.0e-14,
    atol=0.0,
)

assert input_value("jacobian.pc_type") == "pc_mhd_block"
assert input_value("pc_mhd_block.include_hall_mhd_coupling") == "true"
assert int(input_value("pc_mhd_block.field_iterations")) == 2
assert input_value("implicit_mhd.evolve_ion_fluid") == "false"
assert input_value("hybrid_pic_model.include_hall_term") == "true"
assert input_value("hybrid_pic_model.include_electron_pressure_term") == "true"

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
last_solve = newton_history[-1]
assert 1 <= last_solve[2] <= 8
assert (last_solve[4] <= 1.1e-12) or (last_solve[5] <= 1.1e-10)
# The same broadband state requires 435 total GMRES iterations with no
# preconditioner. Keep enough platform margin while requiring the Hall block
# to remove the dispersive spectral stiffness.
assert 0 < last_solve[7] <= 50

relative_wave_error = np.linalg.norm(
    final_polarization - expected_polarization
) / np.linalg.norm(initial_polarization)
excited_frequencies = (
    hall_diffusivity
    * (2.0 * np.sin(np.pi * mode_numbers / number_of_cells) / cell_size) ** 2
)
excited_frequency_steps = excited_frequencies * dt
print(
    "modal omega*dt range="
    f"{np.min(excited_frequency_steps):.12g}.."
    f"{np.max(excited_frequency_steps):.12g}"
)
print(f"grid whistler CFL={whistler_cfl:.12g}")
print(f"theta-method relative B+ error={relative_wave_error:.12e}")
print(f"Newton iterations={int(last_solve[2])}")
print(f"GMRES iterations={int(last_solve[7])}")
