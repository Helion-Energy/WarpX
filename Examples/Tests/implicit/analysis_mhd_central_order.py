#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Order of accuracy of the CENTRAL flux, with and without the limited
Rusanov dissipation (implicit_mhd.central_dissipation).

WHY THIS GATE EXISTS. The other central-flux gates protect monotonicity
(analysis_mhd_central_dissipation.py) and bit-identity of the default
path. Neither would notice if the scheme silently dropped to first
order -- which is the entire product: a monotone first-order scheme is
easy and useless, because at production resolution its numerical
diffusivity is ~4x the physical viscosity the deck sets.

A sine density perturbation is advected as an exact CONTACT (uniform
velocity, uniform ion and electron pressure, uniform field along the
flow) on 32 and 64 cells with the production flux. dt is FIXED across
the pair (keyed to the deck's finest reference grid), so the theta = 0.5
temporal error is a common constant of both arms and the measured order
is purely spatial.

THREE arms, because the two bracketing ones are what make the middle
number mean anything:

  baseline (none,   c=0)  the flux as it ships with the knob off. A pure
                          centered scheme: second order, and dissipating
                          NOTHING (measured D_eff = +1.2e-12 m^2/s, i.e.
                          zero to twelve figures).
  lo       (none,   c=1)  first-order Rusanov. The floor.
  MUSCL    (median, c=1)  the deliverable: the limiter sets the
                          dissipation, so second order survives.

Measured, 30 steps, 32->64, against the 100-step three-level study the
port was characterised with:

    arm        this gate    100-step 32->64 / 64->128 / overall
    baseline     1.994        1.994 / 1.983 / 1.988
    lo           0.971        0.907 / 0.954 / 0.930
    MUSCL        1.843        1.734 / 1.926 / 1.830

The cheap estimate reproduces the expensive one to within the expensive
one's own grid-to-grid spread, which is why this gate is allowed to be
cheap: all seven tests run in 19.9 s on an idle host.

TOLERANCES. The MUSCL floor is 1.5. The lowest MUSCL order observed
anywhere in the characterisation is 1.734 (the coarsest 100-step pair),
so 1.5 sits 0.23 below the worst measurement and 0.5 above the lo arm --
it discriminates second order from first decisively without tracking the
grid-to-grid noise. The bracket tolerances (baseline >= 1.7, lo <= 1.3)
are set the same way from their own spreads.

Usage: analysis_mhd_central_order.py (arm paths are derived, not passed)
       (the other five arms are read from
        ../test_1d_theta_implicit_mhd_central_order_*/diags)
"""

import os

import numpy as np
import warpx_constants as constants
import yt

yt.set_log_level(50)

# Deck constants (inputs_base_1d_theta_implicit_mhd_reconstruction_order).
NUMBER_DENSITY = 1.0e20
DENSITY = NUMBER_DENSITY * constants.proton_mass
DENSITY_AMPLITUDE = 0.2 * DENSITY
ELECTRON_PRESSURE = NUMBER_DENSITY * 10.0 * constants.elementary_charge
ION_PRESSURE = ELECTRON_PRESSURE
GAMMA = 5.0 / 3.0
VELOCITY = 3.0e4
MAGNETIC_FIELD = 1.0e-3
DOMAIN_LENGTH = 1.0
WAVENUMBER = 2.0 * np.pi / DOMAIN_LENGTH

STEM = "test_1d_theta_implicit_mhd_central_order"
PLOTFILE = "diags/diag000030"

SOUND_SPEED = np.sqrt(GAMMA * (ELECTRON_PRESSURE + ION_PRESSURE) / DENSITY)
ALFVEN_SPEED = MAGNETIC_FIELD / np.sqrt(constants.mu_0 * DENSITY)
FAST_SPEED = max(SOUND_SPEED, ALFVEN_SPEED)
# The Rusanov alpha this flux actually uses: the Davis bound max|u +- c_f|.
ALPHA = VELOCITY + FAST_SPEED


def arm_path(tag, cells):
    if tag == "musc" and cells == 64:
        return PLOTFILE
    suffix = "" if tag == "musc" else f"_{tag}"
    return os.path.join("..", f"{STEM}{suffix}_n{cells}", PLOTFILE)


def measure(tag, cells):
    ds = yt.load(arm_path(tag, cells))
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    density = data["boxlib", "implicit_mhd_mass_density"].value.ravel()
    time = float(ds.current_time)
    centers = (np.arange(cells) + 0.5) * DOMAIN_LENGTH / cells
    exact = DENSITY + DENSITY_AMPLITUDE * np.sin(
        WAVENUMBER * (centers - VELOCITY * time)
    )
    error = np.mean(np.abs(density - exact)) / DENSITY_AMPLITUDE
    # A pure single mode sampled at cell centres has an exact DFT
    # amplitude, so the decay needs no windowing correction.
    amplitude = 2.0 * np.abs(np.fft.rfft(density - np.mean(density))[1]) / cells
    diffusivity = -np.log(amplitude / DENSITY_AMPLITUDE) / (WAVENUMBER**2 * time)
    # Total mass must telescope on the periodic mesh.
    np.testing.assert_allclose(np.mean(density), DENSITY, rtol=5.0e-12, atol=0.0)
    assert np.all(density > 0.0), (tag, cells)
    return error, diffusivity


results = {}
for tag in ("base", "lo", "musc"):
    errors, diffusivities = {}, {}
    for cells in (32, 64):
        errors[cells], diffusivities[cells] = measure(tag, cells)
    order = np.log(errors[32] / errors[64]) / np.log(2.0)
    results[tag] = {"order": order, "error": errors, "diffusivity": diffusivities}
    ratio = diffusivities[64] / (0.5 * ALPHA * DOMAIN_LENGTH / 64)
    print(
        f"{tag:5s} order(32->64) = {order:6.3f}"
        f"   L1/A = {errors[32]:.4e} -> {errors[64]:.4e}"
        f"   D_eff(64) = {diffusivities[64]:+.4e} m^2/s"
        f"   D/(alpha dx/2) = {ratio:+.4f}"
    )

# --- ORDER: the product.
assert results["musc"]["order"] >= 1.5, (
    "MUSCL-Rusanov lost second order",
    results["musc"]["order"],
)
# --- BRACKETS: without these the MUSCL number is unfalsifiable.
assert results["base"]["order"] >= 1.7, results["base"]["order"]
assert results["lo"]["order"] <= 1.3, results["lo"]["order"]
assert results["musc"]["order"] >= results["lo"]["order"] + 0.5, (
    results["musc"]["order"],
    results["lo"]["order"],
)

# --- DISSIPATION. Two independent statements about the implementation.
#
# (a) The knob off means OFF: the centered flux dissipates nothing. The
#     bound is loose in absolute terms only because the measurement floor
#     is round-off; the measured value is ~1e-11, twelve orders below the
#     first-order arm.
assert abs(results["base"]["diffusivity"][64]) < 1.0e-3, results["base"]["diffusivity"][
    64
]
# (b) The first-order arm reproduces the Rusanov closed form D = alpha dx/2.
#     This is the check that the term matches its CONTRACT rather than
#     merely running: measured 1.067 against the analytic 1.0, the excess
#     being the C-infinity smoothing that widens alpha slightly.
lo_ratio = results["lo"]["diffusivity"][64] / (0.5 * ALPHA * DOMAIN_LENGTH / 64)
assert 0.95 < lo_ratio < 1.20, ("Rusanov coefficient off its closed form", lo_ratio)
# (c) And the limiter must actually buy something: at the same resolution
#     MUSCL-Rusanov is an order of magnitude less dissipative.
assert results["musc"]["diffusivity"][64] < 0.2 * results["lo"]["diffusivity"][64], (
    results["musc"]["diffusivity"][64],
    results["lo"]["diffusivity"][64],
)
