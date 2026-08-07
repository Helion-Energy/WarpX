#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Temperature floors under a supersonic rarefaction (HLLC, total_energy).

The sinusoidal velocity profile drains half the periodic domain; the halo
cools adiabatically, T = T0 (rho/rho0)^(gamma-1), and would drop far below
the configured 7500 K floor well before reaching the density floor (the
absolute pressure floors are set orders of magnitude lower still, so they
never engage). The assertions check that the density-dependent
n kB T_floor admissibility bound held BOTH species at or above the floor,
that the run genuinely reached densities whose adiabat lies below the
floor (so the test is discriminating), and that holding the temperature
did not break mass conservation.

The ion temperature needs the kinetic energy: in this 1D deck the plotted
ion current jz is an EXACT cell-centered image of the z momentum density
(jz = (e/m_p) M_z; the only flow is axial, and the transverse components
are asserted to vanish), so M_z = jz m_p / e reconstructs it without
interpolation error.

Usage: analysis_mhd_temperature_floor.py <initial_plotfile> <final_plotfile>
"""

import sys

import numpy as np
import warpx_constants as constants
import yt

GAMMA = 5.0 / 3.0
T_FLOOR = 7500.0  # K, must match the deck


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


def temperatures(data):
    """Per-cell ion and electron temperatures [K]."""
    rho = data["boxlib", "implicit_mhd_mass_density"].value.ravel()
    # the flow is purely axial: transverse ion currents vanish and the
    # (single-component) momentum diagnostic exposes M_x = 0
    np.testing.assert_allclose(
        data["boxlib", "implicit_mhd_momentum_density"].value, 0.0, atol=0.0
    )
    for field_name in ("jx", "jy"):
        np.testing.assert_allclose(data["boxlib", field_name].value, 0.0, atol=1.0e-24)
    momentum_z = data["boxlib", "jz"].value.ravel() * constants.m_p / constants.e
    kinetic = 0.5 * momentum_z**2 / rho
    ion_energy = data["boxlib", "implicit_mhd_ion_energy"].value.ravel()
    electron_energy = data["boxlib", "implicit_mhd_electron_energy"].value.ravel()
    number_density = rho / constants.m_p
    ion_temperature = (
        (GAMMA - 1.0) * (ion_energy - kinetic) / (number_density * constants.k)
    )
    electron_temperature = (
        (GAMMA - 1.0) * electron_energy / (number_density * constants.k)
    )
    return rho, ion_temperature, electron_temperature


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])

rho_initial, t_i_initial, t_e_initial = temperatures(initial)
rho_final, t_i_final, t_e_final = temperatures(final)

rho0 = np.mean(rho_initial)
t_i0 = np.mean(t_i_initial)
t_e0 = np.mean(t_e_initial)
print(f"initial temperatures: T_i {t_i0:.1f} K, T_e {t_e0:.1f} K")
print(
    f"final density min/max: {rho_final.min() / rho0:.4f} "
    f"{rho_final.max() / rho0:.4f} (of rho0)"
)
print(
    f"final min T_i: {t_i_final.min():.1f} K, "
    f"min T_e: {t_e_final.min():.1f} K (floor {T_FLOOR:.0f} K)"
)

# the initial state is comfortably ABOVE the floors (the floor acts
# through the dynamics, not through initialization)
assert t_i_initial.min() > 1.4 * T_FLOOR
assert t_e_initial.min() > 1.4 * T_FLOOR

# mass conservation: the bounds only touch the energy blocks and the
# HLLC mass flux is face-based, so the divergence still telescopes
np.testing.assert_allclose(
    np.sum(rho_final), np.sum(rho_initial), rtol=1.0e-11, atol=0.0
)

# THE floor assertions: neither species ends below the temperature floor
# (1e-6 headroom for reconstruction roundoff)
assert t_i_final.min() >= T_FLOOR * (1.0 - 1.0e-6), (
    f"min ion temperature {t_i_final.min():.1f} K fell below the {T_FLOOR:.0f} K floor"
)
assert t_e_final.min() >= T_FLOOR * (1.0 - 1.0e-6), (
    f"min electron temperature {t_e_final.min():.1f} K fell below the "
    f"{T_FLOOR:.0f} K floor"
)

# ... and the floor was genuinely engaged: the halo drained to densities
# whose ADIABAT lies well below the floor, so without the bound the
# temperatures would have kept falling
rho_adiabat = rho0 * (T_FLOOR / t_i0) ** (1.0 / (GAMMA - 1.0))
assert rho_final.min() < 0.8 * rho_adiabat, (
    "rarefaction never reached the temperature-floor regime; the test is "
    f"not discriminating (min density {rho_final.min() / rho0:.4f} rho0, "
    f"adiabat crossing {rho_adiabat / rho0:.4f} rho0)"
)
# the coldest cells sit near the floor rather than far above it
assert t_i_final.min() < 1.2 * T_FLOOR
assert t_e_final.min() < 1.2 * T_FLOOR
# nothing reached the (much lower) density floor, so the min-temperature
# result is attributable to the temperature bound alone
assert rho_final.min() > 1.5e-2 * rho0
# the compression side piled up: a shock formed and was handled
assert rho_final.max() > 1.5 * rho0

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert 1 <= newton_history[-1][2] <= 20

print("PASS")
