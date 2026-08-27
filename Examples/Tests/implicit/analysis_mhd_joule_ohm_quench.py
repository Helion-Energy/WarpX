#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Calibrated one-step check of the reference-code-style Ohm-current Joule quench
(implicit_mhd.joule_ohm_current).

Setup: static two-zone plasma (frozen ions) with a transverse
Bx = dB sin(kz) mode, constant user resistivity eta0 = mu0 D_user, and
the density-keyed vacuum boost active in the low-density halo half
(z >= Lz/2) only.  The reference code's diffusion-dominance criterion
dt eta_field / mu0 > dz^2 is met in the halo (field diffusivity 1e8
m^2/s, 4 decades above the dz^2/dt threshold) and NOT met in the bulk
(D_user = 100 m^2/s, a factor 24 below it).

Bulk (criterion not met -- deposit UNCHANGED relative to OFF): the
electron-energy Joule deposit integrates the curl-B current at the theta
stage,

    dU_e = dt * eta0 * |J^theta|^2,

with J^theta = (J^n + J^{n+1})/2 exactly (theta = 1/2 extrapolation
identity; J is linear in B).  Both endpoint currents are measured from
the plotfile Bx, so the OFF-knob deposit is known exactly and the
equality is asserted tightly: eta0 is the un-boosted user eta and the
knob must not touch non-dominated cells.

Halo (criterion met -- deposit QUENCHED): with resistive_theta = 1 the
solved stage E is eta_field J^{n+1}, so the Ohm-current deposit is
eta0 |J^{n+1}|^2 while OFF would have integrated eta0 |J^theta|^2.  The
one-step backward-Euler stiffness z = dt (eta_field/mu0) k_eff^2 ~ 395
collapses the halo current, J^{n+1} = J^n/(1+z), so the quench ratio is
(J^{n+1}/J^theta)^2 ~ (2/z)^2 ~ 2.6e-5: the assert below demands more
than THREE decades of quench, with the expected value a further factor
of ~40 inside the bound.
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
delta_b = 1.0e-4
bulk_density = 1.0e18
halo_density = 1.0e14
ohm_floor_density = 1.0e15
user_diffusivity = 1.0e2
vacuum_diffusivity = 1.0e6
dt = 1.0e-7
eta_user = constants.mu_0 * user_diffusivity

# --- Calibration guard: recompute the reference code's criterion margins the
# implementation evaluates (dt * eta_field / mu0 > dz^2).  The halo must
# be diffusion dominated by a wide margin and the bulk must not be.
threshold_diffusivity = cell_size**2 / dt
halo_field_diffusivity = np.hypot(
    user_diffusivity, vacuum_diffusivity * (ohm_floor_density / halo_density) ** 2
)
bulk_field_diffusivity = np.hypot(
    user_diffusivity, vacuum_diffusivity * (ohm_floor_density / bulk_density) ** 2
)
assert halo_field_diffusivity / threshold_diffusivity > 1.0e1
assert bulk_field_diffusivity / threshold_diffusivity < 0.5

z = (np.arange(number_of_cells) + 0.5) * cell_size

initial_bx = initial["boxlib", "Bx"].value.ravel()
final_bx = final["boxlib", "Bx"].value.ravel()
np.testing.assert_allclose(
    initial_bx, delta_b * np.sin(wavenumber * z), rtol=2.0e-12, atol=2.0e-16
)


def cell_centered_current(bx):
    """The solver's cell-centered J_y: the average of the two adjacent
    nodal curl-B values, (Bx(i+1) - Bx(i-1)) / (2 dz mu0), periodic."""
    return (np.roll(bx, -1) - np.roll(bx, 1)) / (2.0 * cell_size * constants.mu_0)


current_initial = cell_centered_current(initial_bx)
current_final = cell_centered_current(final_bx)
# theta = 1/2 extrapolation identity: the theta-stage (deposit) current
# is exactly the endpoint average.
current_theta = 0.5 * (current_initial + current_final)

energy_initial = initial["boxlib", "implicit_mhd_electron_energy"].value.ravel()
energy_final = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()
energy_change = energy_final - energy_initial

# The OFF-knob deposit, exactly: dt * eta0 * |J^theta|^2 per cell.
deposit_off = dt * eta_user * current_theta**2

# Probe cells: at least 3 cells from the density interfaces (z = 0 and
# z = Lz/2, periodic) and away from the current nulls of the mode
# (|cos(kz)| > 0.55).
mode_amplitude = np.abs(np.cos(wavenumber * z))
away_from_interfaces = (
    (z > 3.5 * cell_size)
    & (np.abs(z - 0.5) > 3.5 * cell_size)
    & (z < domain_length - 3.5 * cell_size)
)
bulk_probe = (z < 0.5) & away_from_interfaces & (mode_amplitude > 0.55)
halo_probe = (z > 0.5) & away_from_interfaces & (mode_amplitude > 0.55)
assert np.count_nonzero(bulk_probe) >= 8
assert np.count_nonzero(halo_probe) >= 8

# Signal guard: the halo current at the theta stage is still a
# significant fraction of the initial mode current (the quench is
# measured against a live deposit, not an already-decayed one).
initial_current_scale = delta_b * wavenumber / constants.mu_0
assert (
    np.max(np.abs(current_theta[halo_probe])) > 0.3 * initial_current_scale
)

# --- Bulk: criterion not met, the deposit is UNCHANGED relative to OFF
# (the identical eta0 |J^theta|^2 integral).
np.testing.assert_allclose(
    energy_change[bulk_probe], deposit_off[bulk_probe], rtol=1.0e-3
)

# --- Halo: the Ohm-current deposit is quenched by ORDERS OF MAGNITUDE
# relative to the OFF deposit of the SAME run (the field trajectory is
# independent of U_e here, so deposit_off is exactly what the knob-OFF
# run would have integrated).
quench_ratio = np.abs(energy_change[halo_probe]) / deposit_off[halo_probe]
print(f"halo quench ratio: min {quench_ratio.min():.3e} max {quench_ratio.max():.3e}")
# Expected (J^{n+1}/J^theta)^2 for reference.
predicted = (current_final[halo_probe] / current_theta[halo_probe]) ** 2
print(f"predicted (J_final/J_theta)^2: {predicted.min():.3e} .. {predicted.max():.3e}")
assert quench_ratio.max() < 1.0e-3

# Frozen ions: rho and M are untouched.
np.testing.assert_array_equal(
    final["boxlib", "implicit_mhd_mass_density"].value,
    initial["boxlib", "implicit_mhd_mass_density"].value,
)
np.testing.assert_allclose(
    final["boxlib", "implicit_mhd_momentum_density"].value, 0.0, rtol=0.0, atol=0.0
)

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history[-1, 2] >= 1
