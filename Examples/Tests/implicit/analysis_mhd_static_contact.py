#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Static-contact preservation for the implicit-MHD fluid fluxes.

A zero-velocity, uniform-pressure density step is an exact steady solution.
Mode "hllc" asserts the HLLC flux preserves it to solver roundoff over 100
steps. Mode "rusanov" runs the SAME input with the Lax--Friedrichs flux and
asserts the steps DO diffuse -- calibrating that the hllc tolerance is
meaningful rather than trivially satisfied.

Modes "entropy_flow" / "central_fast" run the SAME input on the central
flux with the limited Rusanov penalty (implicit_mhd.central_dissipation =
1): with implicit_mhd.central_dissipation_entropy_speed = flow the
entropy channels are penalised at the smoothed normal flow speed, which is
exactly zero at u = 0, so the contact must be preserved like hllc's
("entropy_flow", the hllc gates); with the default fast speed the same
penalty diffuses the density steps at the acoustic speed ("central_fast",
the rusanov gates).

Usage: analysis_mhd_static_contact.py <initial_plotfile> <final_plotfile> <mode>
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
mode = sys.argv[3]
assert mode in ("hllc", "hlld", "rusanov", "entropy_flow", "central_fast")
# the exact-preservation modes and the calibration (diffusing) modes
exact_modes = ("hllc", "hlld", "entropy_flow")

initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
initial_energy = initial["boxlib", "implicit_mhd_electron_energy"].value.ravel()
final_energy = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()

# reference values from the initial data (immune to CODATA revisions of
# the physical constants between the code and the analysis environment)
density_low = np.min(initial_density)
density_high = np.max(initial_density)
np.testing.assert_allclose(density_high, 4.0 * density_low, rtol=1.0e-6)
electron_energy = np.mean(initial_energy)

# conservation holds for either flux (face-based divergence telescopes)
np.testing.assert_allclose(
    np.sum(final_density), np.sum(initial_density), rtol=2.0e-12, atol=0.0
)

drift = np.max(np.abs(final_density - initial_density))
print(f"mode = {mode}: max |rho_final - rho_initial| = {drift:.3e} "
      f"({drift / density_low:.3e} rho_low)")

if mode in exact_modes:
    # exact preservation to solver roundoff (100 steps at newton
    # atol 1e-11 in scaled units)
    assert drift < 1.0e-9 * density_low, (
        f"{mode} diffused a static contact: drift = {drift:.3e}"
    )
    np.testing.assert_allclose(
        final_energy, electron_energy, rtol=1.0e-9, atol=0.0
    )
else:
    # Rusanov diffuses the step at ~c_s dz / 2: order-unity smearing of
    # the edge cells over this run. This bound failing would mean the
    # hllc assertion above is not actually discriminating.
    assert drift > 0.1 * density_low, (
        f"Rusanov calibration did not diffuse: drift = {drift:.3e}"
    )

# invariant interval: no over/undershoots for the exact and first-order
# fluxes; the MUSCL-Rusanov calibration twin (central_fast) is graded to a
# per-cent bound instead -- its limited reconstruction is TVD only up to
# the limiter's documented new-extremum allowance, and the point of that
# mode is the diffusion, not the bound
density_tolerance = (1.0e-9 if mode != "central_fast" else 2.0e-2) * density_high
assert np.min(final_density) >= density_low - density_tolerance
assert np.max(final_density) <= density_high + density_tolerance

# hllc: the preserved state has an identically zero residual, so Newton
# legitimately exits with 0 iterations; rusanov must be doing real work
newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
minimum_iterations = 0 if mode in exact_modes else 1
assert minimum_iterations <= newton_history[-1][2] <= 12

print(f"mode = {mode}: PASS")
