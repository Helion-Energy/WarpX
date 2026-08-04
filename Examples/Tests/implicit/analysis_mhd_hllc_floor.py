#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Floor-outflow limiter under a supersonic rarefaction (HLLC flux).

The sinusoidal velocity profile drains half the periodic domain toward the
density floor while a shock forms on the compression side. The run
completing at all exercises the limiter (unlimited HLLC locks the Newton
line search here); the assertions check the drained cells actually engaged
the floor, that nothing crossed it, and that the face-based limiter kept
the scheme exactly conservative.

Usage: analysis_mhd_hllc_floor.py <initial_plotfile> <final_plotfile>
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
initial_energy = initial["boxlib", "implicit_mhd_electron_energy"].value.ravel()
final_energy = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()

# reference values from the (uniform) initial data, immune to CODATA
# revisions of the physical constants between code and analysis
density0 = np.mean(initial_density)
density_floor = 1.0e-1 * density0
energy_floor = 1.0e-2 * np.mean(initial_energy)

print(f"min/max final density: {final_density.min():.3e} "
      f"{final_density.max():.3e} (floor {density_floor:.3e})")

# mass conservation: the limiter is face-based, so the divergence still
# telescopes on the periodic mesh
np.testing.assert_allclose(
    np.sum(final_density), np.sum(initial_density), rtol=1.0e-11, atol=0.0
)

# nothing below the floors (beyond roundoff) ...
assert np.min(final_density) >= density_floor * (1.0 - 1.0e-9)
assert np.min(final_energy) >= energy_floor * (1.0 - 1.0e-9)
# ... and the rarefaction actually reached the limiter's action band
# (within 2x the floor), so this test genuinely exercises it
assert np.min(final_density) < 2.0 * density_floor, (
    "rarefaction never engaged the floor limiter; the test is not "
    f"discriminating (min density {np.min(final_density):.3e})"
)
# the compression side piled up: a shock formed and was handled
assert np.max(final_density) > 1.5 * density0

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert 1 <= newton_history[-1][2] <= 20

print("PASS")
