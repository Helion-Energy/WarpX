#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check one theta-method step of the discrete resistive MHD model.

This test intentionally checks the implemented staggered Yee/Ohm operator and
cell-centered Joule source.  It does not assert equality between magnetic
energy loss and electron heating, because those two discrete expressions are
not presently constructed as a total-energy-conserving pair.
"""

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
domain_length = 1.0
cell_size = domain_length / number_of_cells
wavenumber = 2.0 * np.pi / domain_length
theta = 0.5
eta0 = 1.0e-6
eta_time_slope = 6.0
delta_b = 1.0e-4
damping_number = 0.4
gamma_e = 5.0 / 3.0
number_density = 1.0e18
electron_temperature_ev = 0.01

effective_wavenumber = 2.0 * np.sin(0.5 * wavenumber * cell_size) / cell_size
eta_theta = eta0 * (1.0 + eta_time_slope * theta)
eta_end = eta0 * (1.0 + eta_time_slope)
dt = damping_number * constants.mu_0 / (eta_theta * effective_wavenumber**2)

# For dB/dt = -(eta_theta/mu0) k_eff^2 B, the theta method has
# g = [1-(1-theta)*alpha]/[1+theta*alpha], alpha=lambda_theta*dt.
amplification = (1.0 - (1.0 - theta) * damping_number) / (1.0 + theta * damping_number)
z = (np.arange(number_of_cells) + 0.5) * cell_size
initial_bx_expected = delta_b * np.sin(wavenumber * z)
final_bx_expected = amplification * initial_bx_expected

initial_bx = initial["boxlib", "Bx"].value.ravel()
final_bx = final["boxlib", "Bx"].value.ravel()
np.testing.assert_allclose(initial_bx, initial_bx_expected, rtol=2.0e-12, atol=2.0e-16)
np.testing.assert_allclose(final_bx, final_bx_expected, rtol=2.0e-8, atol=2.0e-13)


# Make the time-centering check unambiguous: evaluating eta at t^n or t^{n+1}
# gives amplification factors far from the measured t^{n+theta} result.
def theta_amplification(eta):
    alpha = damping_number * eta / eta_theta
    return (1.0 - (1.0 - theta) * alpha) / (1.0 + theta * alpha)


mode_norm = np.dot(initial_bx_expected, initial_bx_expected)
measured_amplification = np.dot(final_bx, initial_bx_expected) / mode_norm
np.testing.assert_allclose(
    measured_amplification, amplification, rtol=2.0e-8, atol=2.0e-11
)
assert abs(measured_amplification - theta_amplification(eta0)) > 0.1
assert abs(measured_amplification - theta_amplification(eta_end)) > 0.1

# Jy is nodal before diagnostics average it to cell centers.  Therefore the
# cell-centered current for Bx=A sin(kz) has k_cc=sin(k*dz)/dz, not k_eff.
cell_centered_wavenumber = np.sin(wavenumber * cell_size) / cell_size
theta_amplitude = delta_b * ((1.0 - theta) + theta * amplification)
theta_current = (
    theta_amplitude * cell_centered_wavenumber / constants.mu_0 * np.cos(wavenumber * z)
)
initial_pressure = (
    number_density * electron_temperature_ev * constants.elementary_charge
)
initial_energy_expected = initial_pressure / (gamma_e - 1.0)
final_energy_expected = initial_energy_expected + dt * eta_theta * theta_current**2

initial_energy = initial["boxlib", "implicit_mhd_electron_energy"].value.ravel()
final_energy = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()
np.testing.assert_allclose(
    initial_energy, initial_energy_expected, rtol=5.0e-14, atol=0.0
)
np.testing.assert_allclose(
    final_energy, final_energy_expected, rtol=5.0e-8, atol=2.0e-12
)
assert np.min(final_energy - initial_energy) >= -2.0e-12
assert np.max(final_energy - initial_energy) > 1.0e-4

# FinishStateUpdate recomputes algebraic E at t^{n+1}.  With uniform eta and
# zero ion current, diagnostic cell-centering commutes with Ey=eta*Jy.
final_total_jy = final["boxlib", "jy_displacement"].value.ravel()
final_ey = final["boxlib", "Ey"].value.ravel()
final_current_expected = (
    amplification
    * delta_b
    * cell_centered_wavenumber
    / constants.mu_0
    * np.cos(wavenumber * z)
)
np.testing.assert_allclose(
    final_total_jy, final_current_expected, rtol=2.0e-8, atol=2.0e-8
)
np.testing.assert_allclose(
    final_ey, eta_end * final_total_jy, rtol=2.0e-10, atol=2.0e-13
)

# Frozen ions retain rho and M.  The raw momentum diagnostic exposes Mx;
# all three ion-current components independently remain zero and therefore
# also check the corresponding momentum-to-current mapping.
rho0 = number_density * constants.proton_mass
initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
np.testing.assert_array_equal(final_density, initial_density)
np.testing.assert_allclose(final_density, rho0, rtol=5.0e-14, atol=0.0)

for data in (initial, final):
    np.testing.assert_allclose(
        data["boxlib", "implicit_mhd_momentum_density"].value,
        0.0,
        rtol=0.0,
        atol=0.0,
    )
    for field_name in ("jx", "jy", "jz"):
        np.testing.assert_allclose(
            data["boxlib", field_name].value,
            0.0,
            rtol=0.0,
            atol=1.0e-24,
        )

for field_name in ("Ex", "Ez", "By", "Bz"):
    np.testing.assert_allclose(
        final["boxlib", field_name].value,
        0.0,
        rtol=0.0,
        atol=2.0e-13,
    )

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history[-1, 2] >= 1
assert newton_history[-1, 4] < 1.0e-10
