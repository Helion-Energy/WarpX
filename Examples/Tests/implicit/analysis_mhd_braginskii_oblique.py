#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Braginskii conduction with a live cross term in a tilted field (RZ):
the gate of the cross-term tangential limiter options
(implicit_mhd.braginskii_tangential_limiter).

A 2x-hot Gaussian electron spot sits off axis in a strongly magnetized
poloidal field tilted ~25 degrees at the spot, so the tangential corner
stencil of the tensor flux carries an O(1) fraction of the parallel flux
through both face families, at a face conduction number ~50-100 in the
background and several hundred at the spot peak. Exact anisotropic
diffusion obeys the maximum principle: no cell may leave the initial
[e_min, e_max]. The centered corner stencil ("none") is non-monotone and
the minmod / SMART variants are the remedies; this test measures what
each does.

Gates (on the arm it is run for):
 1. every step's Newton solve converged (newton.txt), and the total
    Newton and GMRES work is printed for the convergence table. The
    symmetric SMART pair ("smart") is the exception: its derivative
    kinks (slope ratios 1/5 and 5, right on the steep spot front) make
    the Newton contraction LINEAR (GMRES fully converged, residual down
    by a factor 2-3 per iteration), so it runs to the iteration cap and
    stalls at a relative norm of 6e-8 (D ~ 14), 9e-8 (D ~ 28) and 8e-6
    (D ~ 70) where minmod and smart_upwind meet the solver tolerances;
    that arm runs with newton.require_convergence = 0 and is gated at
    1e-4, i.e. the test pins the measured stall level (with a 1000x
    margin over the CTest dt) and the fact that the stalled solve still
    conserves energy and stays inside the limited-arm monotonicity
    ceiling;
 2. the r-weighted electron energy ledger closes to solver tolerance
    (the tensor flux, tangential term included, is a conservative face
    flux);
 3. maximum principle: the largest excursion of the specific electron
    energy outside the initial [min, max], as a fraction of the initial
    contrast, is printed and gated at 1% for the limited arms (minmod,
    smart, smart_upwind); the centered "none" arm is only printed;
 4. the spot spread more along z than along r (the field is mostly
    axial): a flipped projection or an isotropic tensor fails this.

Usage: analysis_mhd_braginskii_oblique.py <initial> <final> <limiter>
"""

import sys

import numpy as np
import yt

yt.set_log_level(50)

# constants from inputs_test_rz_theta_implicit_mhd_braginskii_oblique
number_of_cells_r = 24
number_of_cells_z = 48
number_of_steps = 10  # max_step; newton.txt is opened in append mode, so
# a re-run in an uncleaned test directory carries the previous run's rows
# and only the LAST number_of_steps rows belong to this run
radial_extent = 0.6
axial_extent = 1.0
# monotonicity ceiling for the limited arms, as a fraction of the initial
# contrast e_max - e_min (measured at dt = 5e-9: minmod 0 exactly, smart
# 2.0e-3, smart_upwind 2.7e-3, the centered stencil 4.6e-3; at the CTest
# dt = 2e-9 the limited arms are lower still)
LIMITED_CEILING = 1.0e-2
# Newton exit criteria of the deck (newton.absolute_tolerance 1e-12,
# newton.relative_tolerance 1e-10): a converged step met EITHER, with a
# 2x margin on the printed norms. The symmetric SMART pair is gated on
# its measured stall level instead (see the docstring).
ABSOLUTE_TOLERANCE = 1.0e-12
RELATIVE_TOLERANCE = 1.0e-10
STALL_CEILING = {"smart": 1.0e-4}


def get_data(plotfile):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    energy = np.asarray(grid["boxlib", "implicit_mhd_electron_energy"])[:, :, 0]
    density = np.asarray(grid["boxlib", "implicit_mhd_mass_density"])[:, :, 0]
    return ds, energy, density


limiter = sys.argv[3]
initial_ds, initial_energy, initial_density = get_data(sys.argv[1])
final_ds, final_energy, final_density = get_data(sys.argv[2])
elapsed_time = float(final_ds.current_time - initial_ds.current_time)
assert elapsed_time > 0.0

cell_size_r = radial_extent / number_of_cells_r
cell_size_z = axial_extent / number_of_cells_z
r = (np.arange(number_of_cells_r) + 0.5) * cell_size_r
z = (np.arange(number_of_cells_z) + 0.5) * cell_size_z - 0.5 * axial_extent
r_grid = r[:, np.newaxis]
z_grid = z[np.newaxis, :]

# the density is uniform and frozen, so the specific energy is e/rho
initial_specific = initial_energy / initial_density
final_specific = final_energy / final_density
np.testing.assert_allclose(final_density, initial_density, rtol=1.0e-12)

# 1. Newton convergence, every step.
newton = np.atleast_2d(np.loadtxt("diags/newton.txt"))[-number_of_steps:]
steps = newton.shape[0]
assert steps == number_of_steps and int(newton[-1, 0]) == number_of_steps, (
    newton[:, 0].astype(int).tolist()
)
iterations = newton[:, 2]
gmres = newton[:, 6]
absolute_norms = newton[:, 4]
relative_norms = newton[:, 5]
print(
    f"limiter = {limiter}: {steps} steps, Newton iterations per step "
    f"{iterations.astype(int).tolist()}, GMRES per step {gmres.astype(int).tolist()}"
)
print(
    f"  total Newton = {int(iterations.sum())}, total GMRES = {int(gmres.sum())}, "
    f"worst final relative norm = {relative_norms.max():.3e}"
)
assert steps >= 1
if limiter in STALL_CEILING:
    assert np.all(relative_norms < STALL_CEILING[limiter]), (
        limiter,
        relative_norms.max(),
        STALL_CEILING[limiter],
    )
else:
    converged = (absolute_norms < 2.0 * ABSOLUTE_TOLERANCE) | (
        relative_norms < 2.0 * RELATIVE_TOLERANCE
    )
    assert np.all(converged), (
        limiter,
        "steps not converged:",
        np.flatnonzero(~converged) + 1,
        absolute_norms[~converged],
        relative_norms[~converged],
    )

# 2. Conservation.
initial_total = np.sum(initial_energy * r_grid)
final_total = np.sum(final_energy * r_grid)
energy_drift = abs(final_total - initial_total) / initial_total
print(f"  relative fluid-energy drift = {energy_drift:.3e}")
assert energy_drift < 1.0e-10

# 3. Maximum principle.
e_min = float(np.min(initial_specific))
e_max = float(np.max(initial_specific))
contrast = e_max - e_min
overshoot = max(0.0, float(np.max(final_specific)) - e_max) / contrast
undershoot = max(0.0, e_min - float(np.min(final_specific))) / contrast
print(
    f"  new extrema: overshoot {overshoot:.3e}, undershoot {undershoot:.3e} "
    f"of the initial contrast (limited-arm ceiling {LIMITED_CEILING:.1e})"
)
if limiter != "none":
    assert max(overshoot, undershoot) < LIMITED_CEILING, (
        limiter,
        overshoot,
        undershoot,
    )

# 4. Anisotropy: the perturbation's second moments must grow more along z
# than along r (r-weighted, about the spot centre).
background = initial_specific[-1, 0]


def moments(specific):
    perturbation = np.maximum(specific - background, 0.0)
    weight = np.sum(perturbation * r_grid)
    r_mean = np.sum(perturbation * r_grid * r_grid) / weight
    z_mean = np.sum(perturbation * r_grid * z_grid) / weight
    sigma_r2 = np.sum(perturbation * r_grid * (r_grid - r_mean) ** 2) / weight
    sigma_z2 = np.sum(perturbation * r_grid * (z_grid - z_mean) ** 2) / weight
    return sigma_r2, sigma_z2


initial_r2, initial_z2 = moments(initial_specific)
final_r2, final_z2 = moments(final_specific)
growth_r = final_r2 - initial_r2
growth_z = final_z2 - initial_z2
print(f"  sigma_r^2 growth {growth_r:.4e} m^2, sigma_z^2 growth {growth_z:.4e} m^2")
assert growth_z > 0.0
assert growth_z > 2.0 * abs(growth_r), (growth_r, growth_z)

print("PASS")
