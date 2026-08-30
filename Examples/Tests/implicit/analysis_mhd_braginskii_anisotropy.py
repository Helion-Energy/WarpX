#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Anisotropic Braginskii spreading of a hot spot in a uniform Bz (RZ).

A Gaussian electron-temperature spot on the axis of a strongly magnetized
uniform Bz (Omega_e tau_e ~ 9.6e3, chi_perp/chi_par ~ 1.6e-8) must spread
ALONG the field at the Braginskii parallel rate while staying frozen
across it. Measured on the second moments of the electron-energy
perturbation: the z-variance grows at d sigma_z^2/dt = 2 chi_par
(chi_par_e(Te0, n0) from first principles, the same formula gated to 5%
by the isolimit test; the band here is wider because the spot's +2% Te
modulation and the 5-cells-per-sigma resolution shift the moment growth
at the few-percent level), while the r-variance must grow at the TINY
Braginskii perpendicular rate 4 chi_perp t itself (a 3x band around a
signal 3e7 times below the parallel one) -- an accidentally isotropic
tensor (bhat projection dropped: r and z spread alike), a flipped
projection sign, or swapped par/perp coefficients all fail one of the
gates by orders of magnitude. Total fluid energy
(r-weighted) must close to solver tolerance: the tensor flux, tangential
corner-stencil term included, is a conservative face flux.

Usage: analysis_mhd_braginskii_anisotropy.py <initial_plotfile> <final_plotfile>
"""

import sys

import numpy as np
import warpx_constants as constants
import yt

yt.set_log_level(50)

# constants from inputs_test_rz_theta_implicit_mhd_braginskii_spreading
number_density = 1.0e18
te0_eV = 10.0
coulomb_log = 10.0
magnetic_field = 0.05
number_of_cells_r = 24
number_of_cells_z = 48
radial_extent = 0.6
axial_extent = 1.0
spot_width = 0.1

kbt = te0_eV * constants.elementary_charge
tau_electron = (
    6.0
    * np.sqrt(2.0)
    * np.pi**1.5
    * constants.epsilon_0**2
    * np.sqrt(constants.electron_mass)
    * kbt
    * np.sqrt(kbt)
    / (number_density * constants.elementary_charge**4 * coulomb_log)
)
gamma = 5.0 / 3.0  # the deck's gamma_e
# (gamma - 1): kappa-convention -> operator convention (see the
# isolimit analysis; the solver applies the same factor).
chi_parallel = (gamma - 1.0) * 3.16 * kbt * tau_electron / constants.electron_mass
x_magnetization = (
    constants.elementary_charge
    * magnetic_field
    / constants.electron_mass
    * tau_electron
) ** 2
chi_perpendicular = (
    chi_parallel
    * (4.664 / 11.92 * x_magnetization + 1.0)
    / ((x_magnetization / 3.7703 + 14.79 / 3.7703) * x_magnetization + 1.0)
)
print(f"chi_par_e  = {chi_parallel:.6e} m^2/s")
print(
    f"chi_perp_e = {chi_perpendicular:.6e} m^2/s "
    f"(x = (Omega tau)^2 = {x_magnetization:.3e})"
)


def get_data(plotfile):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, np.asarray(grid["boxlib", "implicit_mhd_electron_energy"])[:, :, 0]


initial_ds, initial_energy = get_data(sys.argv[1])
final_ds, final_energy = get_data(sys.argv[2])
elapsed_time = float(final_ds.current_time - initial_ds.current_time)
assert elapsed_time > 0.0

cell_size_r = radial_extent / number_of_cells_r
cell_size_z = axial_extent / number_of_cells_z
r = (np.arange(number_of_cells_r) + 0.5) * cell_size_r
z = (np.arange(number_of_cells_z) + 0.5) * cell_size_z - 0.5 * axial_extent
r_grid = r[:, np.newaxis]
z_grid = z[np.newaxis, :]

# uniform background energy from the far corner (the spot is 5 sigma away)
background = initial_energy[-1, 0]


def moments(energy):
    """r-weighted second moments of the (positive) energy perturbation."""
    perturbation = energy - background
    weight = np.sum(perturbation * r_grid)
    sigma_z2 = np.sum(perturbation * r_grid * z_grid**2) / weight
    r2 = np.sum(perturbation * r_grid * r_grid**2) / weight
    return weight, sigma_z2, r2


initial_weight, initial_sigma_z2, initial_r2 = moments(initial_energy)
final_weight, final_sigma_z2, final_r2 = moments(final_energy)

# the initial spot must carry the intended Gaussian moments
# (sigma_z^2 = w^2/2, <r^2> = w^2, both to grid accuracy)
np.testing.assert_allclose(initial_sigma_z2, spot_width**2 / 2.0, rtol=0.02)
np.testing.assert_allclose(initial_r2, spot_width**2, rtol=0.02)

growth_z = final_sigma_z2 - initial_sigma_z2
growth_r = final_r2 - initial_r2
analytic_growth_z = 2.0 * chi_parallel * elapsed_time
print(
    f"sigma_z^2 growth: measured = {growth_z:.6e} m^2, "
    f"analytic 2 chi_par t = {analytic_growth_z:.6e} m^2, "
    f"ratio = {growth_z / analytic_growth_z:.4f}"
)
print(
    f"<r^2>   growth: measured = {growth_r:.6e} m^2 "
    f"(analytic 4 chi_perp t = {4.0 * chi_perpendicular * elapsed_time:.3e})"
)

# parallel spreading at the Braginskii rate (measured 1.0074 on the
# calibration run; the band absorbs the +2% Te chi-modulation and the
# 5-cells-per-sigma moment bias); an accidentally
# perpendicular-projected tensor (ratio ~ chi_perp/chi_par ~ 1.6e-8) or
# a flipped projection sign (anti-diffusion) leaves the band by orders
# of magnitude
assert 0.9 < growth_z / analytic_growth_z < 1.15, (
    f"parallel variance growth off: {growth_z / analytic_growth_z:.4f}"
)

# perpendicular spreading must MATCH the tiny Braginskii perp rate, not
# merely stay small (measured/analytic = 0.999 on the calibration run;
# the moment noise floor sits ~4 decades below the signal: the Newton
# tolerance bounds the state noise at ~1e-10 relative, i.e. ~1e-12 m^2
# moments): an accidentally isotropic tensor overshoots by 3e7, a wrong
# perp-fit coefficient leaves the 3x band
analytic_growth_r = 4.0 * chi_perpendicular * elapsed_time
assert 0.3 < growth_r / analytic_growth_r < 3.0, (
    f"perpendicular variance growth off: {growth_r / analytic_growth_r:.4f}"
)

# anisotropy ratio floor (redundant with the two gates above, stated as
# the headline number)
anisotropy_ratio = growth_z / max(abs(growth_r), 1.0e-30)
print(f"measured spreading anisotropy ratio = {anisotropy_ratio:.3e}")
assert anisotropy_ratio > 100.0

# the tensor flux (tangential corner term included) is a conservative
# face flux: the r-weighted electron energy ledger closes to solver
# tolerance
initial_total = np.sum(initial_energy * r_grid)
final_total = np.sum(final_energy * r_grid)
energy_drift = abs(final_total - initial_total) / initial_total
print(f"relative fluid-energy drift = {energy_drift:.3e}")
assert energy_drift < 1.0e-10

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history[-1, 2] >= 1
assert newton_history[-1, 4] < 1.0e-9

print("PASS")
