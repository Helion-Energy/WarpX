#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Validate a broadband, large-CFL theta-method Alfvén step."""

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
alfven_speed = background_field / np.sqrt(constants.mu_0 * mass_density_reference)

initial_bx = initial["boxlib", "Bx"].value.ravel()
final_bx = final["boxlib", "Bx"].value.ravel()
initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_energy = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()

number_of_cells = initial_bx.size
domain_length = float((initial_ds.domain_right_edge - initial_ds.domain_left_edge)[0])
cell_size = domain_length / number_of_cells
dt = float(final_ds.current_time - initial_ds.current_time)
alfven_cfl = alfven_speed * dt / cell_size
np.testing.assert_allclose(alfven_cfl, 20.0, rtol=1.0e-12, atol=0.0)

# For the right-going linear Alfvén eigenmode, the centered spatial operator
# reduces the coupled induction/momentum system to
#
#     d Bx / dt = -v_A D_0 Bx.
#
# On Fourier mode k, D_0 -> i*sin(k*dz)/dz. Crank-Nicolson therefore has
# amplification G_k=(1-i*a_k)/(1+i*a_k), with
# a_k=0.5*v_A*dt*sin(k*dz)/dz. Applying this exact discrete theta-method
# prediction to the measured initial field also avoids assumptions about the
# diagnostic sampling of the parser-defined profile.
wavenumbers = 2.0 * np.pi * np.fft.fftfreq(number_of_cells, d=cell_size)
effective_wavenumbers = np.sin(wavenumbers * cell_size) / cell_size
theta_number = 0.5 * alfven_speed * dt * effective_wavenumbers
amplification = (1.0 - 1.0j * theta_number) / (1.0 + 1.0j * theta_number)
expected_bx = np.fft.ifft(np.fft.fft(initial_bx) * amplification).real

# The finite-amplitude linearly polarized wave drives O((dB/B0)^2)
# compressive corrections. They are intentionally retained by the nonlinear
# solve but excluded from the linear Alfvén prediction.
perturbation_ratio = np.max(np.abs(initial_bx)) / background_field
nonlinear_field_scale = 8.0 * background_field * perturbation_ratio**2
np.testing.assert_allclose(
    final_bx,
    expected_bx,
    rtol=2.0e-3,
    atol=nonlinear_field_scale,
)
assert np.linalg.norm(final_bx - initial_bx) > 0.25 * np.linalg.norm(initial_bx)

# Periodicity conserves ion mass, while the weak nonlinear compressive
# response must preserve positivity of density and electron internal energy.
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

assert input_value("hybrid_pic_model.include_hall_term") == "false"
assert input_value("hybrid_pic_model.include_electron_pressure_term") == "false"
assert input_value("implicit_mhd.fluid_flux") == "centered"
preconditioner = input_value("jacobian.pc_type")
assert preconditioner == "pc_mhd_block"
assert input_value("pc_mhd_block.include_ideal_mhd_coupling") == "true"
np.testing.assert_allclose(
    float(input_value("pc_mhd_block.wave_relaxation")),
    0.5,
    rtol=0.0,
    atol=0.0,
)
assert int(input_value("pc_mhd_block.fluid_iterations")) == 1

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
last_solve = newton_history[-1]
assert 1 <= last_solve[2] <= 8
assert (last_solve[4] <= 1.1e-12) or (last_solve[5] <= 1.1e-10)
assert 0 < last_solve[7] <= 130

relative_wave_error = np.linalg.norm(final_bx - expected_bx) / np.linalg.norm(
    initial_bx
)
print(f"preconditioner={preconditioner}")
print(f"Alfven CFL={alfven_cfl:.12g}")
print(f"theta-method relative Bx error={relative_wave_error:.12e}")
print(f"Newton iterations={int(last_solve[2])}")
print(f"GMRES iterations={int(last_solve[7])}")
