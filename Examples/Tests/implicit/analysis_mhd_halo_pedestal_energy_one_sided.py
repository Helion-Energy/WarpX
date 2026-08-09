#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""One-sidedness (rectifier) check of the pedestal-band energy drain.

Companion to analysis_mhd_halo_pedestal_energy.py: the same static
tangential-slip contact (drag OFF), but the band's ion pressure is
loaded at 0.2 * f * Pi, i.e. its internal energy sits at 0.2 e_ped --
INSIDE the dead zone of the one-sided rectifier gate -- and, with the
slip kinetic energy, the band's E_i is ~99.9% kinetic, mirroring the
kinetically dominated wall-band population of the FRC benchmark
ladder. The relaxation must do NOTHING here:

  * the deep band's internal energy e = E_i - |m|^2/(2 rho) stays at
    0.2 e_ped (a two-sided form would PUMP it toward e_ped at the
    exact theta-discretized rate, nearly tripling it within these 8
    steps; at scale this pumping was measured to drive a secular
    runaway that NaN-blew an FRC benchmark run) -- asserted at 2% of
    the would-be pump, ~5 orders of magnitude above the measured
    noise floor;
  * momentum, density, and electron energy are preserved away from
    the contacts.

Tolerances: the sub-pedestal band pressure is a STATIC contact jump
of ~Pi from t = 0 (unlike the drain test, where the jump only
develops), so the two cells on each side of each contact carry a
genuine, confined acoustic inflow (measured after 8 steps: 3.2e-3
relative on band-edge density, 6.3e-3 on its momentum, and the
NEXT-to-edge band cells gain ~6 e_int of z-kinetic energy inside E_i
-- invisible in the plotted x momentum -- i.e. 5.7e-3 of E_i).
Beyond two cells the acoustic leakage is suppressed by the tiny
acoustic CFL (~1e-4 per cell-hop): the deep-band E_i is invariant to
5.3e-11 relative, and the remaining interior deviations (<= 9e-6
relative on the band's momentum and U_e) are per-block
solver-tolerance noise in reference-scaled units, largest relative
to the smallest-magnitude block values. Asserted bounds leave
roughly an order of magnitude over the measured deviations.
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
number_density = 1.0e18
rho0 = number_density * constants.proton_mass
pedestal_fraction = 1.0e-3
rho_halo = pedestal_fraction * rho0
electron_temperature = 0.01
ion_pressure = number_density * electron_temperature * constants.elementary_charge
gamma_i = 5.0 / 3.0
internal_energy = ion_pressure / (gamma_i - 1.0)
internal_pedestal = pedestal_fraction * internal_energy
# Band internal energy: 0.2 e_ped, in the rectifier's dead zone.
internal_halo = 0.2 * internal_pedestal
transverse_velocity = 2.0e4
energy_rate = 5.0e7
dt = 1.0e-9
theta = 0.5
steps = 8

z = (np.arange(number_of_cells) + 0.5) * cell_size
bulk = (z > 0.25) & (z < 0.75)
# Contact-adjacent rings: edge = touching the contact, edge2 = one
# further in; both carry the acoustic response of the static jump.
edge = np.zeros_like(bulk)
edge[1:] |= bulk[:-1] != bulk[1:]
edge[:-1] |= bulk[:-1] != bulk[1:]
edge2 = np.zeros_like(bulk)
edge2[2:] |= bulk[:-2] != bulk[2:]
edge2[:-2] |= bulk[:-2] != bulk[2:]
edge2 &= ~edge
near = edge | edge2
bulk_edge = bulk & edge
band_edge = ~bulk & edge
band_deep = ~bulk & ~near
bulk_deep = bulk & ~near

# What a two-sided implementation would do: pump the deviation at the
# theta-discretized rate (kept for the discrimination scale).
x = energy_rate * dt
decay = (1.0 - (1.0 - theta) * x) / (1.0 + theta * x)
pump = (internal_pedestal - internal_halo) * (1.0 - decay**steps)

initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
np.testing.assert_allclose(initial_density[bulk], rho0, rtol=1.0e-14)
np.testing.assert_allclose(initial_density[~bulk], rho_halo, rtol=1.0e-12)
np.testing.assert_allclose(final_density[~edge], initial_density[~edge], rtol=1.0e-6)
np.testing.assert_allclose(
    final_density[bulk_edge], initial_density[bulk_edge], rtol=4.0e-5
)
np.testing.assert_allclose(
    final_density[band_edge], initial_density[band_edge], rtol=4.0e-2
)

momentum_initial = initial["boxlib", "implicit_mhd_momentum_density"].value.ravel()
momentum_final = final["boxlib", "implicit_mhd_momentum_density"].value.ravel()
np.testing.assert_allclose(
    momentum_initial, initial_density * transverse_velocity, rtol=1.0e-13
)
np.testing.assert_allclose(momentum_final[~edge], momentum_initial[~edge], rtol=1.0e-4)
np.testing.assert_allclose(
    momentum_final[band_edge], momentum_initial[band_edge], rtol=6.0e-2
)
np.testing.assert_allclose(
    momentum_final[bulk_edge], momentum_initial[bulk_edge], rtol=1.0e-4
)

initial_electron = initial["boxlib", "implicit_mhd_electron_energy"].value.ravel()
final_electron = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()
np.testing.assert_allclose(final_electron[~edge], initial_electron[~edge], rtol=1.0e-4)
np.testing.assert_allclose(final_electron[edge], initial_electron[edge], rtol=2.0e-4)

# The rectifier check, on the DEEP band (two or more cells from any
# contact, where the acoustic z-kinetic contamination of E_i is below
# 1e-10 relative): the internal energy must stay at 0.2 e_ped -- the
# one-sided gate is closed, the term contributes exactly zero. A
# two-sided implementation pumps it by pump = 0.8 e_ped (1 - A^8),
# nearly tripling it; asserted at 2% of that, ~5 orders above the
# measured noise floor.
initial_ion = initial["boxlib", "implicit_mhd_ion_energy"].value.ravel()
final_ion = final["boxlib", "implicit_mhd_ion_energy"].value.ravel()
kinetic_initial = 0.5 * momentum_initial**2 / initial_density
internal_initial = initial_ion - kinetic_initial
np.testing.assert_allclose(internal_initial[~bulk], internal_halo, rtol=1.0e-10)
np.testing.assert_allclose(internal_initial[bulk], internal_energy, rtol=1.0e-11)
internal_final = final_ion - kinetic_initial
assert np.all(np.abs(internal_final[band_deep] - internal_halo) < 0.02 * pump), (
    "pedestal-band energy relaxation acted on a below-pedestal band "
    "(one-sided rectifier violated)"
)
# Bulk E_i untouched; the contact rings carry the acoustic response
# of the static jump (in E_i mostly through the kinetic part).
np.testing.assert_allclose(final_ion[bulk_deep], initial_ion[bulk_deep], rtol=1.0e-6)
np.testing.assert_allclose(
    final_ion[bulk & near], initial_ion[bulk & near], rtol=4.0e-5
)
np.testing.assert_allclose(
    final_ion[~bulk & near], initial_ion[~bulk & near], rtol=8.0e-2
)

# Nothing electromagnetic happens: B stays zero, and E = -u_e x B = 0.
for data in (initial, final):
    for field_name in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"):
        np.testing.assert_allclose(
            data["boxlib", field_name].value, 0.0, rtol=0.0, atol=1.0e-20
        )
