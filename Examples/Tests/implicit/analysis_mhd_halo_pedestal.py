#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Calibrated check of the offset-density halo pedestal (hlld path).

Setup: a static contact -- uniform electron pressure, zero velocity and
field, zero ion pressure -- whose density is rho0 in the central half of
the domain and 1e-6 rho0 (three decades BELOW the pedestal) outside.
With halo_pedestal_fraction = 1e-3 and reference_mass_density = rho0/2,
the dynamic pedestal is

    rho_ped = f * max(rho_peak, rho_ref) = 1e-3 * rho0,

keyed to the instantaneous PEAK: an implementation keying to the
reference would land at 5e-4 rho0 instead, which the exact-equality
assert below rules out.

Expected behavior over five steps:
  * the sub-pedestal halo band is raised exactly ONTO rho_ped at the
    first step and rides there (an interior point of the admissible
    set -- the positivity floor is nine decades lower);
  * the bulk is untouched: the raise is confined to sub-pedestal cells,
    and the pedestal machinery (drain-gate anchors, source taper) is
    exactly inert on a static contact, whose HLLD fluxes vanish
    identically (S_M = 0, u = 0);
  * momentum and electron energy stay exactly at their initial values
    (the residual is identically zero, so Newton accepts the state
    without an update every step).
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
rho_halo = 1.0e-6 * rho0
pedestal_fraction = 1.0e-3
reference_density = 0.5 * rho0

z = (np.arange(number_of_cells) + 0.5) * cell_size
bulk = (z > 0.25) & (z < 0.75)

# The dynamic pedestal keys to the instantaneous peak, not the (lower)
# reference density.
pedestal = pedestal_fraction * max(rho0, reference_density)
assert pedestal == pedestal_fraction * rho0
assert pedestal != pedestal_fraction * reference_density

initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()

# Step 0 (written before the first OneStep): the loaded profile, halo
# three decades below the pedestal.
np.testing.assert_allclose(initial_density[bulk], rho0, rtol=1.0e-14)
np.testing.assert_allclose(initial_density[~bulk], rho_halo, rtol=1.0e-14)

# Step 5: the halo band rides exactly ON the pedestal; the bulk is
# untouched.  Both are exact-equality checks: the raise is
# rho -> max(rho, rho_ped) and the static-contact residual is
# identically zero, so no solver noise enters.
np.testing.assert_allclose(final_density[bulk], rho0, rtol=1.0e-14)
np.testing.assert_allclose(final_density[~bulk], pedestal, rtol=1.0e-14)

# Discrimination: the halo moved (the raise really happened) and landed
# at the peak-keyed pedestal, not the reference-keyed value.
assert np.all(final_density[~bulk] > 100.0 * rho_halo)
assert np.all(
    np.abs(final_density[~bulk] - pedestal_fraction * reference_density)
    > 0.4 * pedestal
)

# Static contact: momentum and electron energy are exactly preserved.
initial_energy = initial["boxlib", "implicit_mhd_electron_energy"].value.ravel()
final_energy = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()
np.testing.assert_allclose(final_energy, initial_energy, rtol=1.0e-13, atol=0.0)
for data in (initial, final):
    np.testing.assert_allclose(
        data["boxlib", "implicit_mhd_momentum_density"].value, 0.0, rtol=0.0, atol=0.0
    )
    for field_name in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"):
        np.testing.assert_allclose(
            data["boxlib", field_name].value, 0.0, rtol=0.0, atol=1.0e-20
        )
