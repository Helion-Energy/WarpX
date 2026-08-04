#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Fluid energy conservation for the implicit-MHD ion total-energy closure.

A finite-amplitude periodic acoustic perturbation exchanges energy between
ion kinetic, ion internal, and electron internal channels for 100 steps with
eta = 0 and a quiescent uniform Bz. The conservative (enthalpy-form) E_i and
U_e face fluxes telescope exactly, and the electron pdV work (-p_e div u_e)
and its ion counterpart (+p_e div u_e) are evaluated on the same faces with
the same clamped pressure, cancelling identically cell-by-cell away from the
floors. sum(E_i) + sum(U_e) is therefore conserved to roundoff, not merely
truncation order, and the assert demands 1e-12 relative; mass and momentum,
whose transport is purely conservative, hold to similar precision. All
references are sums of the initial data (no physical constants).

Usage: analysis_mhd_energy_conservation.py <initial_plotfile> <final_plotfile>
"""

import sys

import numpy as np
import yt


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])

initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
initial_ion_energy = initial["boxlib", "implicit_mhd_ion_energy"].value.ravel()
final_ion_energy = final["boxlib", "implicit_mhd_ion_energy"].value.ravel()
initial_electron_energy = initial[
    "boxlib", "implicit_mhd_electron_energy"
].value.ravel()
final_electron_energy = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()
initial_jz = initial["boxlib", "jz"].value.ravel()
final_jz = final["boxlib", "jz"].value.ravel()

# the run must be nontrivial: the acoustic perturbation redistributes ion
# energy by an appreciable fraction before conservation is asserted
ion_energy_change = np.max(np.abs(final_ion_energy - initial_ion_energy))
assert ion_energy_change > 1.0e-3 * np.mean(initial_ion_energy), (
    f"perturbation did not evolve: max |dE_i| = {ion_energy_change:.3e}"
)

initial_total = np.sum(initial_ion_energy) + np.sum(initial_electron_energy)
final_total = np.sum(final_ion_energy) + np.sum(final_electron_energy)
energy_drift = abs(final_total - initial_total) / initial_total
print(f"relative fluid-energy drift = {energy_drift:.3e}")
assert energy_drift < 1.0e-12, (
    f"sum(E_i) + sum(U_e) drifted by {energy_drift:.3e} relative"
)

np.testing.assert_allclose(
    np.sum(final_density), np.sum(initial_density), rtol=1.0e-11, atol=0.0
)

# total momentum via the deposited ion current; sum(jz) of the sinusoidal
# initial condition vanishes, so normalize the drift by sum(|jz|)
momentum_drift = abs(np.sum(final_jz) - np.sum(initial_jz))
momentum_scale = np.sum(np.abs(initial_jz))
print(f"relative momentum drift = {momentum_drift / momentum_scale:.3e}")
assert momentum_drift <= 1.0e-11 * momentum_scale, (
    f"sum(jz) drifted by {momentum_drift / momentum_scale:.3e} relative"
)

print("PASS")
