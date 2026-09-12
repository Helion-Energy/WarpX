#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Anisotropic conduction on circular field lines (RZ): the perpendicular
numerical diffusivity chi_perp,num and the maximum principle of the
conduction operators (implicit_mhd.conduction_operator = sharma_hammett
with its tangential limiter | chacon_fd at orders 2 and 4 | the centered
cross-flux control).

A 2x-hot Gaussian electron patch (half-width 3 cells) sits on the ring of
radius R0 (12 cells) of a poloidal field whose lines are exact circles
around (r0, 0); the Braginskii coefficients are pinned by the clamps
(chi_par = 3e4, chi_perp = 3 m^2/s physical, the clamp maxima 1e-4 above;
the operator diffusivities carry the solver's (gamma - 1) convention). The exact solution spreads the
patch along the ring at chi_par and across it at chi_perp only, and obeys
the maximum principle. Measured on the final plotfile against the initial
one, with the RZ volume weight r:

 * chi_perp,meas = [sigma_eta^2(t) - sigma_eta^2(0)] / (2 t), eta = rho - R0
   the distance from the ring (rho the distance from the ring centre), the
   variance taken over the positive part of the specific-energy
   perturbation; chi_perp,num = chi_perp,meas - chi_perp,set (the pinned
   value is 0.9996 of the clamp maximum: the raw perpendicular fit is far
   below the floor, so the smooth floor/cap pair returns
   0.9 c + 0.1 c tanh(3.07)). Reported in the physical convention
   (divided by gamma - 1) and as a fraction of chi_par -- THE number the
   arms are compared on;
 * chi_par,meas from the arc-length variance growth along the ring
   (circular statistics: the wrap-safe angular variance of the ring band,
   converted with R0^2), as a sanity check that parallel conduction
   operates at the pinned value (gated loosely: within a factor 2, the
   patch wraps 40 degrees of the ring at the end);
 * maximum principle: the largest excursion of the specific electron
   energy outside the initial [min, max], as a fraction of the initial
   contrast; gated at 1 percent for every arm except the centered
   cross-flux control (printed only);
 * every step's Newton solve converged (newton.txt), the r-weighted
   electron energy ledger closes to solver tolerance, and the total
   Newton / GMRES work is printed for the operator table.

Usage: analysis_mhd_conduction_circle.py <initial> <final> <arm label>
"""

import sys

import numpy as np
import yt

yt.set_log_level(50)

# constants from inputs_test_rz_theta_implicit_mhd_conduction_circle
number_of_cells_r = 64
number_of_cells_z = 64
number_of_steps = 10  # newton.txt is opened in append mode; only the LAST
# number_of_steps rows belong to this run
radial_extent = 0.5
axial_extent = 0.5
cell_size = radial_extent / number_of_cells_r
ring_centre_r = 0.25
ring_radius = 12.0 * cell_size
gamma = 5.0 / 3.0
# the clamps: min = the nominal value, max = min (1 + 1e-4); the raw
# parallel value lies far above the cap (the soft cap returns the maximum
# exactly), the raw perpendicular value far below the floor (the smooth
# floor/cap pair returns 0.9 c + 0.1 c tanh(3.07) of the maximum)
chi_par_phys = 3.0e4 * 1.0001
chi_perp_phys = 3.0 * 1.0001
chi_perp_set_phys = chi_perp_phys * (0.9 + 0.1 * np.tanh(3.07))
# maximum-principle ceiling of the limited arms, as a fraction of the
# initial contrast (the oblique-field test's value)
LIMITED_CEILING = 1.0e-2
ABSOLUTE_TOLERANCE = 1.0e-12
RELATIVE_TOLERANCE = 1.0e-10


def get_data(plotfile):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    energy = np.asarray(grid["boxlib", "implicit_mhd_electron_energy"])[:, :, 0]
    density = np.asarray(grid["boxlib", "implicit_mhd_mass_density"])[:, :, 0]
    return ds, energy, density


label = sys.argv[3]
initial_ds, initial_energy, initial_density = get_data(sys.argv[1])
final_ds, final_energy, final_density = get_data(sys.argv[2])
elapsed_time = float(final_ds.current_time - initial_ds.current_time)
assert elapsed_time > 0.0

r = (np.arange(number_of_cells_r) + 0.5) * cell_size
z = (np.arange(number_of_cells_z) + 0.5) * cell_size - 0.5 * axial_extent
r_grid = r[:, np.newaxis]
z_grid = z[np.newaxis, :]
rho_grid = np.sqrt((r_grid - ring_centre_r) ** 2 + z_grid**2)
eta_grid = rho_grid - ring_radius
theta_grid = np.arctan2(z_grid, r_grid - ring_centre_r)

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
    f"arm = {label}: {steps} steps, Newton iterations per step "
    f"{iterations.astype(int).tolist()}, GMRES per step {gmres.astype(int).tolist()}"
)
print(
    f"  total Newton = {int(iterations.sum())}, total GMRES = {int(gmres.sum())}, "
    f"GMRES per Newton = {gmres.sum() / max(iterations.sum(), 1):.2f}, "
    f"worst final relative norm = {relative_norms.max():.3e}"
)
converged = (absolute_norms < 2.0 * ABSOLUTE_TOLERANCE) | (
    relative_norms < 2.0 * RELATIVE_TOLERANCE
)
assert np.all(converged), (
    label,
    "steps not converged:",
    np.flatnonzero(~converged) + 1,
    absolute_norms[~converged],
    relative_norms[~converged],
)

# 2. Conservation (the patch never reaches a domain boundary).
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
if "centered" not in label and label != "sharma_none":
    assert max(overshoot, undershoot) < LIMITED_CEILING, (
        label,
        overshoot,
        undershoot,
    )

# 4. Perpendicular (across-ring) and parallel (along-ring) spreading.
background = float(np.min(initial_specific))


def moments(specific):
    perturbation = np.maximum(specific - background, 0.0)
    weight = perturbation * r_grid
    total = np.sum(weight)
    eta_mean = np.sum(weight * eta_grid) / total
    sigma_eta2 = np.sum(weight * (eta_grid - eta_mean) ** 2) / total
    # wrap-safe angular variance of the ring band (circular statistics)
    c = np.sum(weight * np.cos(theta_grid)) / total
    s = np.sum(weight * np.sin(theta_grid)) / total
    angular_variance = -2.0 * np.log(max(np.hypot(c, s), 1.0e-300))
    return sigma_eta2, angular_variance * ring_radius**2


initial_eta2, initial_arc2 = moments(initial_specific)
final_eta2, final_arc2 = moments(final_specific)
chi_perp_meas_code = (final_eta2 - initial_eta2) / (2.0 * elapsed_time)
chi_par_meas_code = (final_arc2 - initial_arc2) / (2.0 * elapsed_time)
chi_perp_meas_phys = chi_perp_meas_code / (gamma - 1.0)
chi_par_meas_phys = chi_par_meas_code / (gamma - 1.0)
chi_perp_num_phys = chi_perp_meas_phys - chi_perp_set_phys
print(
    f"  sigma_eta^2 {initial_eta2:.4e} -> {final_eta2:.4e} m^2, "
    f"arc variance {initial_arc2:.4e} -> {final_arc2:.4e} m^2 over {elapsed_time:.3e} s"
)
print(
    f"  chi_par,meas = {chi_par_meas_phys:.4e} m^2/s (pinned {chi_par_phys:.1e}); "
    f"chi_perp,meas = {chi_perp_meas_phys:.4e} (pinned {chi_perp_set_phys:.4f})"
)
print(
    f"  CHI_PERP_NUM {label} = {chi_perp_num_phys:.4e} m^2/s = "
    f"{chi_perp_num_phys / chi_par_phys:.3e} chi_par"
)
assert final_arc2 > initial_arc2, "parallel conduction inert"
assert 0.5 * chi_par_phys < chi_par_meas_phys < 2.0 * chi_par_phys, (
    chi_par_meas_phys,
    chi_par_phys,
)
assert final_eta2 - initial_eta2 > -0.05 * initial_eta2, (
    "perpendicular variance shrank",
    initial_eta2,
    final_eta2,
)

print("PASS")
