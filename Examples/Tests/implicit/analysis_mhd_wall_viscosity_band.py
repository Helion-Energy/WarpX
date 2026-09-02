#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Wall viscosity band coefficient (implicit_mhd.wall_viscosity_band_value).

Uniform plasma with a linear axial shear u_z = A r inside a cylinder
conducting wall (r_w = 0.30 m -> first masked cell i = 15 at
dr = 0.02), everything else quiescent (B = 0, J = 0, eta = 0, no
conduction, no Joule).  For a constant viscous face coefficient mu the
r-face stress tau = -mu A is constant, so its paired stress work
deposits the DISCRETELY EXACT uniform shear heating

    dE_i/dt = 2 mu A^2

in every cell whose two radial faces carry the SAME mu (in RZ,
r W = -mu A^2 r^2 gives the r-weighted divergence
(r_hi^2 - r_lo^2)/(r_c dr) = 2 exactly).

Three coefficient regimes of the same deck, each asserted against its
OWN analytic mu -- this is what makes the band a coefficient
SUBSTITUTION rather than a zeroing:

  unmasked  wall_viscosity_mask = 0        -> mu = rho nu everywhere
  zero      mask on, band value 0          -> mu = 0 on band faces
  band      mask on, band value 1e-4 Pa s  -> mu = 1e-4 on band faces
                                              (the reference code's small_vis)

Band faces are i >= first_masked - width = 12, and the band is
deliberately one cell wider than needed so the two gated rows (13, 14)
sit strictly INSIDE it: the coefficient jump at the band edge relaxes
the local velocity jump there at the (30x faster) band rate, so the
first band row (12) and the last interior rows (10, 11) are transition
rows whose stress is no longer the constant tau = -mu A the analytic
deposit assumes.  They are excluded, as is the last cell (open r face).
The gated interior (cells 4..9) must take the full rho nu deposit in
all three regimes: the band must never touch live rows.

Note on units: implicit_mhd.viscosity is the kinematic-style knob nu
[m^2/s] that the face assembly multiplies by the face mass density,
whereas the band value is the absolute DYNAMIC viscosity [Pa s],
deliberately not density-scaled.  Here rho nu = 3.345e-6 Pa s, so the
the reference code's pedestal 1e-4 Pa s is a 29.9x INCREASE -- the halo-side face of
the same pedestal that CAPS the coefficient 3.3-40x below rho nu where
compressed plasma touches the wall.

Usage:
  analysis_mhd_wall_viscosity_band.py <initial> <final> unmasked
  analysis_mhd_wall_viscosity_band.py <initial> <final> zero
  analysis_mhd_wall_viscosity_band.py <initial> <final> band
"""

import sys

import numpy as np
import warpx_constants as constants
import yt

yt.set_log_level(50)

# constants from inputs_test_rz_theta_implicit_mhd_wall_viscosity_band
NUMBER_DENSITY = 1.0e20
RHO0 = NUMBER_DENSITY * constants.proton_mass
SHEAR_RATE = 1.0e3
VISCOSITY = 20.0
BAND_VALUE = 1.0e-4  # Pa s (dynamic), the reference code's small_vis
WALL_RADIUS = 0.30
MASK_WIDTH = 3
NUMBER_OF_CELLS_R = 24
RADIAL_EXTENT = 0.48
CELL_SIZE = RADIAL_EXTENT / NUMBER_OF_CELLS_R

INTERIOR_VISCOSITY = RHO0 * VISCOSITY  # Pa s

# Mask geometry (the implementation's cell-centered convention: first
# cell whose center (i + 1/2) dr sits on/outside the polyline radius).
FIRST_MASKED = int(np.ceil(WALL_RADIUS / CELL_SIZE - 0.5 - 1.0e-3))
assert FIRST_MASKED == 15
FIRST_BAND = FIRST_MASKED - MASK_WIDTH  # = 12: first band-coefficient row
# Cells 13, 14: both radial faces AND both radial neighbours carry the
# band coefficient, so the constant-tau analytic deposit holds there.
# Row 12 (and 10, 11 on the live side) straddle the coefficient jump.
PURE_BAND = slice(FIRST_BAND + 1, FIRST_MASKED)
INTERIOR = slice(4, FIRST_BAND - 2)


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])
mode = sys.argv[3]
assert mode in ("unmasked", "zero", "band"), f"unknown mode {mode}"

elapsed_time = float(final_ds.current_time - initial_ds.current_time)
assert elapsed_time > 0.0


def ion_energy(data):
    # (nr, nz) ion total energy
    return data["boxlib", "implicit_mhd_ion_energy"].value[:, :, 0]


energy_change = ion_energy(final) - ion_energy(initial)


def deposit_of(dynamic_viscosity):
    return 2.0 * dynamic_viscosity * SHEAR_RATE**2 * elapsed_time


interior_deposit = deposit_of(INTERIOR_VISCOSITY)
band_viscosity = {
    "unmasked": INTERIOR_VISCOSITY,
    "zero": 0.0,
    "band": BAND_VALUE,
}[mode]
band_deposit = deposit_of(band_viscosity)

print(f"mode = {mode}, elapsed {elapsed_time:.6e} s")
print(
    f"interior mu = {INTERIOR_VISCOSITY:.6e} Pa s -> analytic deposit "
    f"{interior_deposit:.6e} J/m^3"
)
print(
    f"band     mu = {band_viscosity:.6e} Pa s -> analytic deposit "
    f"{band_deposit:.6e} J/m^3"
)

# --- INTERIOR gate: the full rho nu deposit, in every regime ---
interior = energy_change[INTERIOR, :]
print(
    f"interior deposit/analytic: min {interior.min() / interior_deposit:.4f} "
    f"max {interior.max() / interior_deposit:.4f}"
)
np.testing.assert_allclose(interior, interior_deposit, rtol=0.1)

# --- BAND gate: each regime matches ITS OWN analytic coefficient ---
band = energy_change[PURE_BAND, :]
if band_deposit > 0.0:
    print(
        f"band deposit/analytic: min {band.min() / band_deposit:.4f} "
        f"max {band.max() / band_deposit:.4f}"
    )
    np.testing.assert_allclose(band, band_deposit, rtol=0.1)
else:
    band_max = float(np.max(np.abs(band)))
    print(f"band |dE_i| max / interior deposit = {band_max / interior_deposit:.3e}")
    assert band_max < 0.1 * interior_deposit, (
        "band value 0 must reproduce the legacy exact-zero band"
    )

# The z-shear-free setup deposits nothing through z faces: the azimuthal
# momentum stays exactly zero.
momentum_theta = final["boxlib", "implicit_mhd_momentum_t"].value
np.testing.assert_allclose(momentum_theta, 0.0, rtol=0.0, atol=1.0e-20)

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history[-1, 2] >= 1

print(f"wall viscosity band ({mode}): all gates passed")
