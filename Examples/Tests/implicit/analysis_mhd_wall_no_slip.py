#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Reference-parity shaped-wall no-slip pin (implicit_mhd.wall_no_slip).

A linear axial shear u_z = A r inside a cylinder conducting wall
(r_w = 0.30 m -> first masked cell i = 15 at dr = 0.02) with the
explicit viscosity live everywhere (no wall_viscosity_mask), B = 0,
J = 0, eta = 0, no conduction, no Joule.

the reference code omits the vr/vz rows of the wall-contour vertices AND of the
adjacent cut-cell skin layer from its implicit momentum matrices and
starts every velocity at exactly zero, so those two vertex layers are
pinned at u = 0 for the whole run (measured bit-exact zero on all 762
wall and 762 skin vertices of the reference shot's earlier dump).  Our port is the
structural twin: momentum identity rows on the first
wall_no_slip_width LIVE cell rows adjacent to the contour, composed
with a load-time zeroing of the same rows.

Gates, against the wall_no_slip = 0 twin of the identical deck:

  1. PIN: with the knob ON, cells i = 13, 14 (width 2 = the reference code's wall +
     skin depth) carry EXACTLY zero momentum in all three components.
  2. FREE SLIP: with the knob OFF, those same rows slide essentially
     freely -- our stair-face image copies the tangential momentum
     verbatim -- staying within a few percent of rho A r.  This is
     today's behavior, and the OFF twin reproduces it.
  3. WALL DRAG: the pin deposits drag into the first UNPINNED row
     (i = 12), which is the momentum transport channel the free-slip
     wall does not have.
  4. LOCALITY: the far interior (i <= 8) is unchanged from the OFF
     twin to solver precision -- the pin must not reach past its own
     shear layer.

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
WALL_RADIUS = 0.30
NUMBER_OF_CELLS_R = 24
RADIAL_EXTENT = 0.48
CELL_SIZE = RADIAL_EXTENT / NUMBER_OF_CELLS_R
NO_SLIP_WIDTH = 2

# Mask geometry (the implementation's cell-centered convention: first
# cell whose center (i + 1/2) dr sits on/outside the polyline radius).
FIRST_MASKED = int(np.ceil(WALL_RADIUS / CELL_SIZE - 0.5 - 1.0e-3))
assert FIRST_MASKED == 15
FIRST_PINNED = FIRST_MASKED - NO_SLIP_WIDTH  # = 13
PINNED = slice(FIRST_PINNED, FIRST_MASKED)  # live rows 13, 14
DRAG_ROW = FIRST_PINNED - 1  # = 12, first unpinned row
FAR_INTERIOR = slice(2, FIRST_PINNED - 4)  # rows 2..8

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

band_absmax = {
    name: float(np.max(np.abs(field[PINNED, :]))) for name, field in state.items()
}
for name in COMPONENTS:
    print(
        f"pin band (i = {FIRST_PINNED}..{FIRST_MASKED - 1}) "
        f"|momentum_{name}|max = {band_absmax[name]:.6e} "
        f"({band_absmax[name] / momentum_scale:.3e} of rho A r_max)"
    )

if mode == "on":
    # --- Gate 1: the pin is exact ---
    for name in COMPONENTS:
        assert band_absmax[name] == 0.0, (
            f"wall_no_slip = 1 left momentum_{name} = {band_absmax[name]:.6e} "
            f"on the pinned rows (must be exactly zero: the residual's "
            f"momentum identity rows carry the load-time zeroing forward)"
        )

    assert len(sys.argv) > 3, "on mode needs the wall_no_slip = 0 twin"
    reference = load(sys.argv[3])

    # --- Gate 3: wall drag reaches the first unpinned row ---
    pinned_drag = float(np.mean(state["z"][DRAG_ROW, :]))
    free_drag = float(np.mean(reference["z"][DRAG_ROW, :]))
    relative_drag = (free_drag - pinned_drag) / free_drag
    print(
        f"first unpinned row i = {DRAG_ROW}: momentum_z pinned "
        f"{pinned_drag:.6e} vs free-slip {free_drag:.6e} "
        f"(drag = {relative_drag:.3e} of the free-slip value)"
    )
    assert relative_drag > 1.0e-4, (
        "the pin deposited no measurable wall drag into the first "
        "unpinned row: the no-slip shear layer is missing"
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
        assert far_error < 1.0e-8, (
            f"the pin changed the far interior momentum_{name} by "
            f"{far_error:.3e} of the momentum scale: the shear layer is "
            "leaking past its own band"
        )
else:
    # --- Gate 2: today's free-slip wall, reproduced by the OFF twin ---
    free_slip_band = free_slip_momentum[PINNED, np.newaxis]
    slide = state["z"][PINNED, :] / free_slip_band
    print(
        f"free-slip band momentum_z / (rho A r): min {slide.min():.6f} "
        f"max {slide.max():.6f}"
    )
    np.testing.assert_allclose(slide, 1.0, rtol=0.05)
    assert band_absmax["z"] > 0.5 * momentum_scale * (
        radius[FIRST_PINNED] / radius[-1]
    ), "the wall_no_slip = 0 twin is not free-slip"
    # The setup drives no radial or azimuthal flow at all: those stay at
    # round-off (measured 5.5e-9 of the momentum scale, in the wall band).
    for name in ("r", "t"):
        np.testing.assert_allclose(
            state[name], 0.0, rtol=0.0, atol=1.0e-6 * momentum_scale
        )

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history[-1, 2] >= 1

print(f"wall no-slip pin ({mode}): all gates passed")
