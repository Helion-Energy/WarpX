#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Calibrated one-step check of the density-keyed vacuum resistivity and
the resistive backward-Euler centering (hlld path).

Setup: static uniform plasma (frozen ions), zero user resistivity, and a
transverse Bx = dB sin(kz) mode.  The hybrid Ohm guard (n_floor) sits a
factor of 4 above the plasma density, so the field advance sees the
uncapped density-keyed boost

    eta_field = mu0 * D_vac * (n_floor / n_guarded)^2,

with n_guarded the smooth positivity-floor division guard (inert here to
~1e-25 relative).  With the user eta exactly zero the quadrature smooth
max reduces to eta_field = eta_vac exactly, so the mode obeys the
discrete diffusion dBx/dt = (eta_vac/mu0) d^2Bx/dz^2 with the staggered
k_eff = 2 sin(k dz/2)/dz.

Centering: implicit_evolve.theta = 0.5 but implicit_mhd.resistive_theta
= 1.  The dissipative Ohm terms are then evaluated at the extrapolated
end-of-step current J^{n+1} = 2 J^{n+theta} - J^n, which for this linear
mode gives the exact backward-Euler amplification

    g = (1 - (1 - theta_r) z) / (1 + theta_r z) = 1/(1 + z),

z = dt (eta_vac/mu0) k_eff^2 = 1, i.e. g = 1/2 -- well separated from
the trapezoidal (theta_r = 1/2) value (1 - z/2)/(1 + z/2) = 1/3, which
the discrimination assert below rules out.

The electron energy must NOT follow the boosted eta: Joule heating uses
the un-boosted user resistivity (zero here), so U_e stays constant to
solver tolerance.  If the boost leaked into the heating source, U_e
would rise by eta_vac J^2 dt / U_e ~ 3e-2 relative -- the calibrated
bound below is 1e-5.
"""

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

number_of_cells = 64
domain_length = 1.0
cell_size = domain_length / number_of_cells
wavenumber = 2.0 * np.pi / domain_length
theta = 0.5
theta_resistive = 1.0
delta_b = 1.0e-4
number_density = 1.0e18
ohm_floor_density = 4.0e18
vacuum_diffusivity = 1.0e6
damping_number = 1.0

# Mirror the implemented operator: eta_vac = mu0 D (n_ref/n_guarded)^2
# with the smooth positivity-floor division guard
# n_guarded = f + (e + sqrt(e^2 + f^2))/2, e = n - f, f the positivity
# floor's charge-density equivalent (default 1e-12 of the reference).
positivity_guard = 1.0e-12 * number_density
excess = number_density - positivity_guard
guarded_density = positivity_guard + 0.5 * (
    excess + np.sqrt(excess**2 + positivity_guard**2)
)
eta_vacuum = (
    constants.mu_0 * vacuum_diffusivity * (ohm_floor_density / guarded_density) ** 2
)

effective_wavenumber = 2.0 * np.sin(0.5 * wavenumber * cell_size) / cell_size
dt = damping_number * constants.mu_0 / (eta_vacuum * effective_wavenumber**2)
z_stiffness = dt * (eta_vacuum / constants.mu_0) * effective_wavenumber**2
np.testing.assert_allclose(z_stiffness, damping_number, rtol=1e-13)


def theta_amplification(theta_r):
    return (1.0 - (1.0 - theta_r) * z_stiffness) / (1.0 + theta_r * z_stiffness)


amplification = theta_amplification(theta_resistive)

z = (np.arange(number_of_cells) + 0.5) * cell_size
initial_bx_expected = delta_b * np.sin(wavenumber * z)
final_bx_expected = amplification * initial_bx_expected

initial_bx = initial["boxlib", "Bx"].value.ravel()
final_bx = final["boxlib", "Bx"].value.ravel()
np.testing.assert_allclose(initial_bx, initial_bx_expected, rtol=2.0e-12, atol=2.0e-16)
np.testing.assert_allclose(final_bx, final_bx_expected, rtol=2.0e-6, atol=2.0e-11)

mode_norm = np.dot(initial_bx_expected, initial_bx_expected)
measured_amplification = np.dot(final_bx, initial_bx_expected) / mode_norm
np.testing.assert_allclose(measured_amplification, amplification, rtol=2.0e-6)

# Discrimination: the measured decay matches neither the trapezoidal
# resistive centering (theta_r = theta = 1/2) nor an un-boosted field
# advance (user eta = 0 => no decay at all).
assert abs(measured_amplification - theta_amplification(theta)) > 0.1
assert abs(measured_amplification - 1.0) > 0.4

# Joule heating uses the UN-boosted user eta (zero here): the electron
# energy stays constant to solver tolerance while the field decays.
initial_energy = initial["boxlib", "implicit_mhd_electron_energy"].value.ravel()
final_energy = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()
np.testing.assert_allclose(final_energy, initial_energy, rtol=1.0e-5, atol=0.0)

# Frozen ions: rho and M are untouched, and the ion current stays zero.
initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
np.testing.assert_array_equal(final_density, initial_density)
for data in (initial, final):
    np.testing.assert_allclose(
        data["boxlib", "implicit_mhd_momentum_density"].value, 0.0, rtol=0.0, atol=0.0
    )
    for field_name in ("jx", "jy", "jz"):
        np.testing.assert_allclose(
            data["boxlib", field_name].value, 0.0, rtol=0.0, atol=1.0e-24
        )

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history[-1, 2] >= 1
