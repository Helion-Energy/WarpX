#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Calibrated one-step check of the reference-code-style DYNAMIC reference density
(implicit_mhd.vacuum_reference_peak_fraction), through the Ohm-current
Joule quench it re-keys.

Setup (the joule_ohm_quench sibling with a MID-DENSITY zone): static
two-zone plasma (frozen ions) with a transverse Bx = dB sin(kz) mode,
constant user resistivity eta0 = mu0 D_user, the density-keyed vacuum
boost D_vac, and vacuum_reference_peak_fraction = 0.1 -- the effective
reference density is max(n_floor, 0.1 n_peak) = 1e17, far ABOVE the
static Ohm guard (1e15).

Mid zone (n_mid = 0.02 n_peak, z >= Lz/2): with the STATIC reference the
vacuum boost is D_vac (n_floor/n_mid)^2 = 250 m^2/s and the reference code's
dominance criterion dt eta_field / mu0 > dz^2 is NOT met (factor ~9
margin) -- the knob-OFF run integrates the plain eta0 |J^theta|^2
deposit there.  With the DYNAMIC reference the boost is
D_vac (n_ref/n_mid)^2 = 2.5e6 m^2/s, dominated by three decades: the
deposit is QUENCHED to eta0 |J^{n+1}|^2 (resistive_theta = 1), a
measured factor (J^{n+1}/J^theta)^2 ~ 3e-2 below the OFF integral.
This zone is quenched ONLY because the dynamic reference re-keyed the
criterion -- the calibration guards below re-derive both margins.

Bulk (n_peak): the dynamic boost D_vac (n_ref/n0)^2 = 1e3 still fails
the criterion (factor ~2.4 margin), so the bulk deposit must match the
exact OFF integral eta0 |J^theta|^2 dt tightly: the dynamic reference
must not touch non-dominated cells.
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
mid_density = 2.0e16
ohm_floor_density = 1.0e15
peak_fraction = 0.1
user_diffusivity = 1.0e2
vacuum_diffusivity = 1.0e5
dt = 1.0e-7
eta_user = constants.mu_0 * user_diffusivity

# --- Calibration guards: recompute the reference code's criterion margins the
# implementation evaluates (dt * eta_field / mu0 > dz^2) under BOTH
# references.  The effective dynamic reference is the implementation's
# max(static Ohm guard, fraction * n_peak); n_peak is the frozen
# step-old (initial) density peak = the bulk density.
threshold_diffusivity = cell_size**2 / dt
reference_static = ohm_floor_density
reference_dynamic = max(ohm_floor_density, peak_fraction * bulk_density)
assert reference_dynamic == 1.0e17


def field_diffusivity(reference, density):
    return np.hypot(user_diffusivity, vacuum_diffusivity * (reference / density) ** 2)


# The mid zone is quenched ONLY when the dynamic reference is on: with
# the static reference the criterion fails there (the knob-OFF twin of
# this run would integrate the plain OFF deposit).
assert field_diffusivity(reference_static, mid_density) / threshold_diffusivity < 0.5
assert field_diffusivity(reference_dynamic, mid_density) / threshold_diffusivity > 1.0e2
# The bulk must stay non-dominated under the dynamic reference.
assert field_diffusivity(reference_dynamic, bulk_density) / threshold_diffusivity < 0.5

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

# The OFF-knob deposit, exactly: dt * eta0 * |J^theta|^2 per cell (the
# field trajectory is independent of U_e, so this is what the same run
# with the static reference would have integrated in the mid zone).
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
mid_probe = (z > 0.5) & away_from_interfaces & (mode_amplitude > 0.55)
assert np.count_nonzero(bulk_probe) >= 8
assert np.count_nonzero(mid_probe) >= 8

# Signal guard: the mid-zone current at the theta stage is still a
# significant fraction of the initial mode current (the quench is
# measured against a live deposit, not an already-decayed one).
initial_current_scale = delta_b * wavenumber / constants.mu_0
assert np.max(np.abs(current_theta[mid_probe])) > 0.3 * initial_current_scale

# --- Bulk: criterion not met under the dynamic reference, the deposit
# is UNCHANGED relative to OFF (the identical eta0 |J^theta|^2 integral).
np.testing.assert_allclose(
    energy_change[bulk_probe], deposit_off[bulk_probe], rtol=1.0e-3
)

# --- Mid zone: quenched ONLY because the dynamic reference re-keyed the
# criterion.  With resistive_theta = 1 the solved stage E is
# eta_field J^{n+1}, so the Ohm-current deposit is eta0 |J^{n+1}|^2 and
# the quench ratio against the OFF integral is (J^{n+1}/J^theta)^2,
# measured from the plotfile field trajectory itself.
quench_ratio = np.abs(energy_change[mid_probe]) / deposit_off[mid_probe]
predicted = (current_final[mid_probe] / current_theta[mid_probe]) ** 2
print(
    f"mid-zone quench ratio: min {quench_ratio.min():.3e} max {quench_ratio.max():.3e}"
)
print(f"predicted (J_final/J_theta)^2: {predicted.min():.3e} .. {predicted.max():.3e}")
assert quench_ratio.max() < 0.1
np.testing.assert_allclose(quench_ratio, predicted, rtol=0.5)

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

print("dynamic reference: all gates passed")
