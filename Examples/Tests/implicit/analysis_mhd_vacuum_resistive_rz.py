#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Stiff RZ vacuum-resistive diffusion (hlld path, frozen ions).

A radial Bz profile decays through the corner E_theta = eta J_theta with
the density-keyed vacuum eta (grid diffusion number z = 150 at the halo
minimum), and the z-modulated eta bends the decay into Br. Structural
invariants of the residual's corner-EMF form are asserted to roundoff:

- B_theta stays exactly zero: with u = 0 (frozen ions) and no initial
  B_theta, E_r and E_z are pure eta J with J_r = J_z = 0.
- The r-weighted axial flux sum_cells r_c Bz is conserved: the Bz update
  is the r-weighted divergence of the corner E_theta, which telescopes
  radially with a zero axis corner (parity) and a zero wall corner (PEC
  tangential E).
- The axial sum of Br vanishes on every radius: the Br update is a
  periodic z difference of E_theta, and Br starts at zero.
- Br itself is excited (the z-modulated eta gradient does real work),
  and the Bz profile relaxes toward its r-weighted mean.
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

number_of_radial_cells = 16
radial_extent = 0.1
cell_size = radial_extent / number_of_radial_cells
radius = (np.arange(number_of_radial_cells) + 0.5) * cell_size
magnetic_field = 1.0e-3
perturbation = 1.0e-4

initial_bz = initial["boxlib", "Bz"].value[:, :, 0]
final_bz = final["boxlib", "Bz"].value[:, :, 0]
initial_br = initial["boxlib", "Br"].value[:, :, 0]
final_br = final["boxlib", "Br"].value[:, :, 0]

# B_theta is exactly zero at all times.
for data in (initial, final):
    np.testing.assert_array_equal(data["boxlib", "Bt"].value, 0.0)

# Frozen ions: the density is untouched and the momentum stays zero.
np.testing.assert_array_equal(
    final["boxlib", "implicit_mhd_mass_density"].value,
    initial["boxlib", "implicit_mhd_mass_density"].value,
)
np.testing.assert_array_equal(
    final["boxlib", "implicit_mhd_momentum_density"].value, 0.0
)

# r-weighted axial flux conservation (radially telescoping corner-EMF
# divergence with zero axis and wall corners).
initial_flux = np.sum(radius[:, None] * initial_bz)
final_flux = np.sum(radius[:, None] * final_bz)
np.testing.assert_allclose(final_flux, initial_flux, rtol=1.0e-12)

# The axial sum of Br vanishes on every radius (periodic z telescoping).
np.testing.assert_allclose(np.sum(final_br, axis=1), 0.0, atol=1.0e-9 * perturbation)

# The decay is real: the initial radial profile relaxes strongly toward
# its r-weighted mean at z = 150 per step (two steps taken), and the
# z-modulated eta excites a nonzero Br.
initial_variation = np.max(initial_bz) - np.min(initial_bz)
final_variation = np.max(final_bz) - np.min(final_bz)
expected_variation = perturbation * (
    np.cos(0.5 * np.pi * radius[0] / radial_extent)
    - np.cos(0.5 * np.pi * radius[-1] / radial_extent)
)
np.testing.assert_allclose(initial_variation, expected_variation, rtol=1.0e-6)
# The halo band (grid diffusion number 150 per step) flattens immediately;
# the core (Ohm-guard boost only, diffusion number ~3 per step) sets the
# measured 0.455 remaining variation after the two steps.
assert 0.35 < final_variation / initial_variation < 0.55
assert np.max(np.abs(final_br)) > 1.0e-2 * perturbation
assert np.all(np.isfinite(final_bz)) and np.all(np.isfinite(final_br))
assert np.min(final_bz) > 0.9 * magnetic_field

# The solve converged (the diagnostic file appends across reruns of the
# same work directory, so only the final row is examined).
newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert int(newton_history[-1, 0]) == 2
assert (newton_history[-1, 4] <= 1.1e-14) or (newton_history[-1, 5] <= 1.1e-11)
