#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Reference-parity shaped-wall no-slip FACE condition
(implicit_mhd.wall_no_slip).

A linear axial shear u_z = A r inside a cylinder conducting wall
(r_w = 0.30 m -> first masked cell i = 15 at dr = 0.02) with the
explicit viscosity live everywhere (no wall_viscosity_mask), B = 0,
J = 0, eta = 0, no conduction, no Joule.  wall_thermal_bc = zero_flux
makes the masked band a rigid, insulating conductor, so the tangential
wall shear is the ONLY wall-fluid coupling.

A no-slip wall constrains the TANGENTIAL slip AT THE CONTOUR.  The
masked side of a stair interface face presents the ANTISYMMETRIC
tangential image -u_t of the interior state, so the face-centered
tangential velocity is exactly zero and the viscous difference quotient
(u_t^image - u_t^interior)/dr = -2 u_t/dr is the half-cell one-sided
wall gradient (0 - u_t)/(dr/2): the textbook wall shear
tau_w = mu u_t/(dr/2) = 2 rho nu u_t/dr.  The NORMAL component is
untouched and every wall-adjacent LIVE cell keeps all three momentum
rows as ordinary unknowns.

Gates, against the wall_no_slip = 0 twin of the identical deck:

  1. LIVE ROWS: the wall-adjacent live rows are NOT pinned -- they keep
     a finite tangential momentum.  A volumetric momentum pin would seal
     those control volumes off from the domain entirely (both of their
     face velocities become zero), which is what
     analysis_mhd_wall_no_slip_lift.py gates.
  2. WALL SHEAR: the momentum the wall removes from the wall-adjacent
     row matches the analytic first-order estimate
     dm_z = 2 rho nu u_t t / dr^2, i.e. it really is the half-cell wall
     shear and not an arbitrary sink.
  3. FREE SLIP (off mode): with the knob OFF those rows slide freely --
     the stair-face image copies the tangential momentum verbatim --
     staying within a few percent of rho A r.
  4. LOCALITY: the far interior (i <= 8) is unchanged from the OFF twin
     to solver precision; the wall layer must not reach past itself.

Usage:
  analysis_mhd_wall_no_slip.py <final_plotfile> off
  analysis_mhd_wall_no_slip.py <final_plotfile> on <off_final_plotfile>
"""

import sys

import numpy as np
import warpx_constants as constants
import yt

yt.set_log_level(50)

# constants from inputs_test_rz_theta_implicit_mhd_wall_no_slip
NUMBER_DENSITY = 1.0e20
RHO0 = NUMBER_DENSITY * constants.proton_mass
SHEAR_RATE = 1.0e3
VISCOSITY = 20.0
FINAL_TIME = 3 * 5.0e-8
WALL_RADIUS = 0.30
NUMBER_OF_CELLS_R = 24
RADIAL_EXTENT = 0.48
CELL_SIZE = RADIAL_EXTENT / NUMBER_OF_CELLS_R

# Mask geometry (the implementation's cell-centered convention: first
# cell whose center (i + 1/2) dr sits on/outside the polyline radius).
FIRST_MASKED = int(np.ceil(WALL_RADIUS / CELL_SIZE - 0.5 - 1.0e-3))
assert FIRST_MASKED == 15
WALL_CELL = FIRST_MASKED - 1  # = 14, the live cell touching the contour
FAR_INTERIOR = slice(2, 9)  # rows 2..8

COMPONENTS = ("r", "t", "z")


def load(plotfile):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return {
        name: grid["boxlib", f"implicit_mhd_momentum_{name}"].value[:, :, 0]
        for name in COMPONENTS
    }


state = load(sys.argv[1])
mode = sys.argv[2]
assert mode in ("on", "off"), f"unknown mode {mode}"

radius = (np.arange(NUMBER_OF_CELLS_R) + 0.5) * CELL_SIZE
free_slip_momentum = RHO0 * SHEAR_RATE * radius
momentum_scale = float(np.max(free_slip_momentum))

for name, field in state.items():
    assert np.isfinite(field).all(), f"momentum_{name} has non-finite values"

wall_row = {name: float(np.mean(field[WALL_CELL, :])) for name, field in state.items()}
for name in COMPONENTS:
    print(
        f"wall-adjacent live row i = {WALL_CELL}: momentum_{name} = "
        f"{wall_row[name]:.6e} "
        f"({wall_row[name] / free_slip_momentum[WALL_CELL]:.6f} of rho A r)"
    )

if mode == "on":
    assert len(sys.argv) > 3, "on mode needs the wall_no_slip = 0 twin"
    reference = load(sys.argv[3])
    reference_row = float(np.mean(reference["z"][WALL_CELL, :]))

    # --- Gate 1: the wall-adjacent rows stay LIVE ---
    # The no-slip condition is a face constraint on the tangential
    # velocity, not a volumetric pin: the cell keeps moving.
    slide = wall_row["z"] / free_slip_momentum[WALL_CELL]
    print(f"gate 1 live row: momentum_z / (rho A r) = {slide:.6f}")
    assert slide > 0.5, (
        f"the wall-adjacent live row kept only {slide:.3e} of its "
        "tangential momentum: a no-slip WALL condition constrains the "
        "slip at the face, it does not pin the interior cell"
    )

    # --- Gate 2: the removed momentum is the half-cell wall shear ---
    # First-order: dm_z = tau_w t / dr with tau_w = rho nu u_t/(dr/2).
    removed = reference_row - wall_row["z"]
    analytic = (
        2.0
        * RHO0
        * VISCOSITY
        * SHEAR_RATE
        * radius[WALL_CELL]
        * FINAL_TIME
        / CELL_SIZE**2
    )
    print(
        f"gate 2 wall shear: removed {removed:.6e} vs analytic "
        f"{analytic:.6e} (ratio {removed / analytic:.4f})"
    )
    assert 0.5 < removed / analytic < 1.5, (
        f"the wall removed {removed:.3e} of tangential momentum against "
        f"the analytic half-cell wall shear {analytic:.3e}: this is not a "
        "no-slip wall boundary layer"
    )

    # --- Gate 4: the far interior is untouched ---
    # Normalized by the GLOBAL momentum scale rho A r_max: the radial and
    # azimuthal components are identically zero in this setup, so their
    # own scale is round-off and would make a relative test meaningless.
    for name in COMPONENTS:
        far_error = (
            float(
                np.max(
                    np.abs(
                        state[name][FAR_INTERIOR, :] - reference[name][FAR_INTERIOR, :]
                    )
                )
            )
            / momentum_scale
        )
        print(f"far interior |d momentum_{name}|max / (rho A r_max) = {far_error:.3e}")
        assert far_error < 1.0e-6, (
            f"the wall condition changed the far interior momentum_{name} by "
            f"{far_error:.3e} of the momentum scale: the shear layer is "
            "leaking past the wall row"
        )
else:
    # --- Gate 3: today's free-slip wall, reproduced by the OFF twin ---
    band = slice(FIRST_MASKED - 2, FIRST_MASKED)
    free_slip_band = free_slip_momentum[band, np.newaxis]
    slide = state["z"][band, :] / free_slip_band
    print(
        f"free-slip rows momentum_z / (rho A r): min {slide.min():.6f} "
        f"max {slide.max():.6f}"
    )
    np.testing.assert_allclose(slide, 1.0, rtol=0.05)
    # The setup drives no radial or azimuthal flow at all: those stay at
    # round-off.
    for name in ("r", "t"):
        np.testing.assert_allclose(
            state[name], 0.0, rtol=0.0, atol=1.0e-6 * momentum_scale
        )

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history[-1, 2] >= 1

print(f"wall no-slip face condition ({mode}): all gates passed")
