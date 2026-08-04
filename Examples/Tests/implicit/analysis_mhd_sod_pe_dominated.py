#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Electron-pressure-dominated shock tube for the ion total-energy closure.

Same periodic double Riemann problem as analysis_mhd_sod.py, but with the
pressure roles swapped: the electron pressure carries the jump and the ion
pressure is an order of magnitude below it (p_e/p_i = 9, the FRC-halo
regime). This exercises the electron-pressure coupling of the ion
total-energy flux at full strength; a discretization whose E_i flux
pressure is inconsistent with the total pressure of the HLLC wave
structure locks the Newton line search here at the first steps.

No exact-Riemann comparison is made: the electron pressure follows a
non-conservative (isentropic) advection equation with no shock entropy
production, so the electron-dominated two-pressure system does not reduce
to the single-gas exact solution. The asserts are structural instead --
completion, conservation, an invariant density interval, and monotone
density through the right-moving wave train. All references come from the
simulation data (no physical constants).

Usage: analysis_mhd_sod_pe_dominated.py <initial_plotfile> <final_plotfile>
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
final_jz = final["boxlib", "jz"].value.ravel()

number_of_cells = initial_density.size
z_lo = float(initial_ds.domain_left_edge[0])
z_hi = float(initial_ds.domain_right_edge[0])
cell_size = (z_hi - z_lo) / number_of_cells
z = z_lo + (np.arange(number_of_cells) + 0.5) * cell_size

rho_l = np.max(initial_density)
rho_r = np.min(initial_density)
high = initial_density > 0.5 * (rho_l + rho_r)
high_indices = np.where(high)[0]
assert 0 < high_indices[0] and high_indices[-1] < number_of_cells - 1
z_diaphragm = z[high_indices[-1]] + 0.5 * cell_size
z_center = 0.5 * (z[high_indices[0]] + z[high_indices[-1]])

# the tube must actually have evolved: the upper diaphragm cell drains
# toward the star region
diaphragm_cell = high_indices[-1]
assert final_density[diaphragm_cell] < 0.9 * rho_l, "the shock tube did not evolve"

# conservative transport: total mass to solver tolerance; total momentum
# stays zero by the mirror symmetry of the double Riemann problem
np.testing.assert_allclose(
    np.sum(final_density), np.sum(initial_density), rtol=1.0e-10, atol=0.0
)
momentum_scale = np.sum(np.abs(final_jz))
assert momentum_scale > 0.0
momentum_drift = abs(np.sum(final_jz)) / momentum_scale
print(f"relative momentum drift = {momentum_drift:.3e}")
assert momentum_drift < 1.0e-10

# invariant interval: no over/undershoots
density_tolerance = 1.0e-6 * rho_l
assert np.min(final_density) >= rho_r - density_tolerance
assert np.max(final_density) <= rho_l + density_tolerance

# the right-moving wave train (rarefaction, contact, shock) has a
# monotonically non-increasing density profile at fixed time; measure from
# the undisturbed center of the high region to the last cell before the
# periodic seam, with a small slack for plateau-level wiggles
window = (z > z_center) & (z < z_hi - 2.0 * cell_size)
wiggle_tolerance = 2.0e-3 * rho_l
increments = np.diff(final_density[window])
print(
    f"max density increment in wave train = {np.max(increments):.3e} "
    f"({np.max(increments) / rho_l:.3e} rho_l)"
)
assert np.all(increments <= wiggle_tolerance), (
    "density is not monotone through the right-moving wave train"
)

print("PASS")
