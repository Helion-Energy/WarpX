#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Ion thermal conduction on the central-flux recast face registers.

A small sinusoidal ion-temperature perturbation (1% in p_i) rides a uniform,
static, magnetized background with implicit_mhd.thermal_diffusivity_ion
chosen stiff against the acoustic response (chi k >> c_s), so the ion
internal-energy mode decays diffusively before sound can convert it: the
measured decay rate must match the analytic q = -kappa grad T rate chi k^2.
The discretization corrections are small and known -- the face-difference
Laplacian eigenvalue k_d = 2 sin(k dz/2)/dz (-0.08%) and the Crank-Nicolson
rate ln[(1+x/2)/(1-x/2)]/dt at x = chi k_d^2 dt ~ 0.2 (+0.33%) -- so the
5% gate on the CONTINUOUS rate leaves an order of magnitude of margin while
still rejecting any wrong prefactor (a gamma factor is a 67% error; a
missing rho_face weighting or a wrong spacing power changes the rate
outright). Conduction is a conservative face flux, so sum(E_i) + sum(U_e)
must still close to roundoff, and with chi_e = 0 the electron energy and
the density must stay uniform (acoustic contamination O((c_s/(chi k))^2)).

Usage: analysis_mhd_central_conduction.py <initial_plotfile> <final_plotfile>
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

# constants from inputs_test_1d_theta_implicit_mhd_central_conduction
n0 = 1.0e20
rho0 = n0 * constants.proton_mass
ion_pressure0 = n0 * 50.0 * constants.elementary_charge
gamma = 5.0 / 3.0
relative_amplitude = 0.01
number_of_cells = 64
cell_size = 1.0 / number_of_cells
wavenumber = 2.0 * np.pi
thermal_diffusivity = 1.0e6

elapsed_time = float(final_ds.current_time - initial_ds.current_time)
assert elapsed_time > 0.0

z = (np.arange(number_of_cells) + 0.5) * cell_size
mode = np.sin(wavenumber * z)


def mode_amplitude(values):
    return 2.0 * np.mean(values * mode)


initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
initial_ion_energy = initial["boxlib", "implicit_mhd_ion_energy"].value.ravel()
final_ion_energy = final["boxlib", "implicit_mhd_ion_energy"].value.ravel()
initial_electron_energy = initial[
    "boxlib", "implicit_mhd_electron_energy"
].value.ravel()
final_electron_energy = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()

# the initial condition must carry exactly the intended temperature mode on a
# uniform static background (E_i = p_i/(gamma - 1) at u = 0)
initial_amplitude = mode_amplitude(initial_ion_energy)
np.testing.assert_allclose(
    initial_amplitude,
    relative_amplitude * ion_pressure0 / (gamma - 1.0),
    rtol=1.0e-9,
)
np.testing.assert_allclose(initial_density, rho0, rtol=1.0e-12, atol=0.0)

final_amplitude = mode_amplitude(final_ion_energy)
assert 0.05 < final_amplitude / initial_amplitude < 0.25, (
    f"decay left the resolvable window: {final_amplitude / initial_amplitude:.4f}"
)

measured_rate = np.log(initial_amplitude / final_amplitude) / elapsed_time
analytic_rate = thermal_diffusivity * wavenumber**2
print(f"measured decay rate  = {measured_rate:.6e} 1/s")
print(f"analytic chi k^2     = {analytic_rate:.6e} 1/s")
print(f"ratio                = {measured_rate / analytic_rate:.6f}")
assert abs(measured_rate / analytic_rate - 1.0) < 0.05, (
    f"conductive decay rate off by {measured_rate / analytic_rate - 1.0:+.3%}"
)

# conduction is a conservative face flux: the fluid energy ledger closes
initial_total = np.sum(initial_ion_energy) + np.sum(initial_electron_energy)
final_total = np.sum(final_ion_energy) + np.sum(final_electron_energy)
energy_drift = abs(final_total - initial_total) / initial_total
print(f"relative fluid-energy drift = {energy_drift:.3e}")
assert energy_drift < 1.0e-11

# chi_e = 0 and the acoustic response is suppressed by the stiff diffusion:
# density and electron energy stay uniform far below the ion-mode scale
assert np.max(np.abs(final_density / rho0 - 1.0)) < 1.0e-4
assert (
    np.max(np.abs(final_electron_energy / initial_electron_energy - 1.0)) < 1.0e-4
)

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history[-1, 2] >= 1
assert newton_history[-1, 4] < 1.0e-9
