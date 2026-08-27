#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Wall-row viscosity mask (implicit_mhd.wall_viscosity_mask).

Uniform plasma with a linear axial shear u_z = A r inside a cylinder
conducting wall (r_w = 0.30 m -> first masked cell i = 15 at
dr = 0.02), viscosity nu, everything else quiescent (B = 0, J = 0,
eta = 0, no conduction, no Joule).  The viscous r-face stress
tau = -rho nu A is constant, so its paired stress work deposits the
DISCRETELY EXACT uniform shear heating rate

    dE_i/dt = 2 rho nu A^2

in every cell whose two radial faces carry live viscosity (in RZ,
r W = -rho nu A^2 r^2 gives the r-weighted divergence
(r_hi^2 - r_lo^2)/(r_c dr) = 2 exactly).  With the slip mask on
(width 2) every r-face at i >= first_masked - width = 13 carries zero
viscous stress AND zero stress work:

  - INTERIOR gate (cells 4 <= i <= 10): the ion total energy gains the
    full analytic deposit (the mask must not touch live rows);
  - SLIP-BAND gate (cells i >= 13, the masked-adjacent rows and the
    masked band): the viscous deposit is ZERO -- E_i changes only by
    the percent-scale acoustic transient the differential heating
    launches, bounded at 10% of the interior deposit;
  - the transition row i = 12 (one live, one masked face) is excluded.

Usage: analysis_mhd_wall_viscosity_mask.py <initial_plotfile> <final_plotfile>
"""

import sys

import numpy as np
import warpx_constants as constants
import yt

yt.set_log_level(50)


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])

# constants from inputs_test_rz_theta_implicit_mhd_wall_viscosity_mask
number_density = 1.0e20
rho0 = number_density * constants.proton_mass
shear_rate = 1.0e3
viscosity = 20.0
wall_radius = 0.30
mask_width = 2
number_of_cells_r = 24
radial_extent = 0.48
cell_size = radial_extent / number_of_cells_r

elapsed_time = float(final_ds.current_time - initial_ds.current_time)
assert elapsed_time > 0.0

# Mask geometry (the implementation's cell-centered convention: first
# cell whose center (i + 1/2) dr sits on/outside the polyline radius).
first_masked = int(np.ceil(wall_radius / cell_size - 0.5 - 1.0e-3))
assert first_masked == 15
first_slip = first_masked - mask_width  # = 13: first zero-viscosity row


def ion_energy(data):
    # (nr, nz) ion total energy
    return data["boxlib", "implicit_mhd_ion_energy"].value[:, :, 0]


energy_change = ion_energy(final) - ion_energy(initial)

# The analytic shear-heating deposit (discretely exact for the linear
# profile; theta = 1 momentum drift and the acoustic transient
# contribute percent-scale corrections).
deposit = 2.0 * rho0 * viscosity * shear_rate**2 * elapsed_time
print(f"analytic viscous deposit = {deposit:.6e} J/m^3 over {elapsed_time:.3e} s")

# --- INTERIOR gate: full analytic heating, uniform, at every z ---
interior = energy_change[4:11, :]
print(
    f"interior deposit/analytic: min {interior.min() / deposit:.4f} "
    f"max {interior.max() / deposit:.4f}"
)
np.testing.assert_allclose(interior, deposit, rtol=0.1)

# --- SLIP-BAND gate: zero viscous deposit at masked-adjacent cells ---
band = energy_change[first_slip:, :]
band_max = np.max(np.abs(band))
print(f"slip-band |dE_i| max / analytic deposit = {band_max / deposit:.3e}")
assert band_max < 0.1 * deposit

# The z-shear-free setup deposits nothing through z faces: the momentum
# stays r-dependent only, and the azimuthal momentum stays exactly zero.
momentum_theta = final["boxlib", "implicit_mhd_momentum_t"].value
np.testing.assert_allclose(momentum_theta, 0.0, rtol=0.0, atol=1.0e-20)

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history[-1, 2] >= 1

print("wall viscosity mask: all gates passed")
