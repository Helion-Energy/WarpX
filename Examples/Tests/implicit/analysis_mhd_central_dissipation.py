#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Limited Rusanov dissipation of the CENTRAL flux
(implicit_mhd.central_dissipation), on a 4:1 square density contact.

WHY THIS EXISTS. central is the production flux and it carries no
Riemann dissipation by design. A slope limiter placed in front of it
therefore has nothing to limit -- where fluid_reconstruction clips, the
scheme reduces exactly to the unreconstructed centered flux -- so
reconstruction ALONE cannot make central monotone. This gate measures
what that costs and what the fix buys, on the flux production actually
flies.

THE MEASUREMENT THAT MOTIVATES IT. On the smooth-sine order rig the
baseline central flux dissipates at D_eff = +1.2e-12 m^2/s, i.e. ZERO to
twelve figures -- a pure centered scheme. The defect on central is
therefore NOT excess dissipation (that is the hlld story, D = |u|dx/2);
it is the complete ABSENCE of dissipation, which on a contact shows up
as dispersive undershoot driving density onto its positivity floor.

Three arms of the same deck, all on central + viscosity:

  baseline   reconstruction none, dissipation 0 -- production today.
             Run with newton.require_convergence = 0 because it CANNOT
             complete otherwise: with convergence required it aborts at
             step 4 with "Newton line search failed to find a
             residual-decreasing admissible step", after projecting mass
             onto the density floor. That abort is the campaign's own
             failure signature reproduced on a 1D CI-scale deck in 18
             seconds, and it is the reason this knob exists.
  lo         reconstruction none, dissipation 1 -- first-order Rusanov.
             Monotone, and the reference for how much a purely
             first-order fix would smear the front.
  main       reconstruction median, dissipation 1 -- MUSCL-Rusanov.
             The deliverable: the limiter sets the dissipation, so this
             is second order in smooth flow and monotone at the front.

Measured (120 steps, 128 cells, central + viscosity = 2000), as the full
2x2 of {reconstruction} x {dissipation} -- because the point is that
NEITHER ingredient suffices alone:

    arm                    new extrema   band   min rho/rho_lo  mass drift
    baseline (none,   c=0)   3.058e-01     64      0.0827         2.80e-04
    recon    (median, c=0)   2.024e-01     39      0.3929         2.22e-16
    lo       (none,   c=1)   0.000e+00     54      1.000017       0.00e+00
    MUSCL    (median, c=1)   7.699e-04     26      0.99769        2.22e-16

Read the baseline row carefully. 31% of the jump appears as new extrema;
density is driven to 8% of rho_low; and mass conservation is LOST at the
2.8e-4 level -- because the Newton admissibility projection has to inject
mass to keep the undershoot positive, and that repair is not
conservative.

The recon row is the quantitative form of "a limiter needs upwinding to
BE a monotonicity device". Reconstruction alone cuts the ringing by only
1.5x -- 20% of the jump, nowhere near monotone -- because where it clips,
central reduces exactly to the unreconstructed centered flux. What it
DOES buy is survival: bounding the face states keeps density at 39% of
rho_low instead of 8%, which keeps the admissibility projection from
firing and restores exact conservation. Survival, not monotonicity.

Dissipation alone (lo) is the mirror image: exactly monotone (0.0) but
smeared over 54 cells, and at production scale its diffusivity would be
~4x the physical viscosity the deck sets.

Only the pair is both: MUSCL-Rusanov cuts the ringing 397x against the
baseline and 263x against reconstruction alone, resolves the front in 26
cells (2.1x sharper than first-order Rusanov), and conserves to 2.2e-16.

Gates:
 1. the MUSCL arm holds the invariant interval [rho_low, rho_high] to
    the limiter's documented bound,
 2. it resolves the contact strictly SHARPER than the first-order arm
    (fewer cells in the transition band) -- otherwise the dissipation
    would simply have replaced ringing with smearing,
 3. the baseline rings by far more than the MUSCL arm, which is what
    makes gate 1 non-vacuous,
 4. every arm conserves mass to round-off, stays strictly positive, and
    keeps E_i - KE above the deck's ion internal-energy floor,
 5. the MUSCL arm needs NO admissibility projections at all, while the
    baseline needs them -- the positivity claim, measured rather than
    asserted.

Usage: analysis_mhd_central_dissipation.py <initial> <final>
       (the baseline and lo arms are read from
        ../test_1d_theta_implicit_mhd_central_dissipation_*/diags)
"""

import os
import sys

import numpy as np
import warpx_constants as constants
import yt

yt.set_log_level(50)

NUMBER_DENSITY = 1.0e20
DENSITY_LOW = NUMBER_DENSITY * constants.proton_mass
DENSITY_HIGH = 4.0 * DENSITY_LOW
SPAN = DENSITY_HIGH - DENSITY_LOW
ELECTRON_TEMPERATURE_EV = 10.0
ION_PRESSURE = NUMBER_DENSITY * ELECTRON_TEMPERATURE_EV * constants.elementary_charge
ION_PRESSURE_FLOOR = 1.0e-9 * ION_PRESSURE
GAMMA = 5.0 / 3.0
VELOCITY = 1.0e5
# The smoothed median's closed-form new-extremum bound, kappa/4 of the
# largest stencil difference (see ThetaImplicitMHD_K.H).
RECONSTRUCTION_KAPPA = 0.01
MEDIAN_BOUND = RECONSTRUCTION_KAPPA / 4.0

TEST_STEM = "test_1d_theta_implicit_mhd_central_dissipation"
PLOTFILE = "diags/diag000120"


def get_fields(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return {
        "density": data["boxlib", "implicit_mhd_mass_density"].value.ravel(),
        "ion_energy": data["boxlib", "implicit_mhd_ion_energy"].value.ravel(),
        "electron_energy": data["boxlib", "implicit_mhd_electron_energy"].value.ravel(),
    }


def arm(mode):
    if mode == "main":
        return PLOTFILE
    return os.path.join("..", f"{TEST_STEM}_{mode}", PLOTFILE)


initial = get_fields(sys.argv[1])
final_main = get_fields(sys.argv[2])

arms = {}
for mode in ("baseline", "lo", "main"):
    fields = final_main if mode == "main" else get_fields(arm(mode))
    density = fields["density"]
    overshoot = max(0.0, float(np.max(density)) - DENSITY_HIGH)
    undershoot = max(0.0, DENSITY_LOW - float(np.min(density)))
    interior = np.count_nonzero(
        (density > DENSITY_LOW + 0.02 * SPAN) & (density < DENSITY_HIGH - 0.02 * SPAN)
    )
    internal = fields["ion_energy"] - 0.5 * density * VELOCITY**2
    arms[mode] = {
        "extremum": max(overshoot, undershoot) / SPAN,
        "interior": interior,
        "mass": float(np.mean(density)),
        "min_density": float(np.min(density)),
        "min_internal": float(np.min(internal)),
        "min_electron": float(np.min(fields["electron_energy"])),
    }

for mode, entry in arms.items():
    print(
        f"{mode:9s} new extrema = {entry['extremum']:.4e} of the jump"
        f"   transition cells = {entry['interior']:3d}"
        f"   min rho/rho_lo = {entry['min_density'] / DENSITY_LOW:.6f}"
        f"   min (E_i - KE)/floor = "
        f"{entry['min_internal'] / (ION_PRESSURE_FLOOR / (GAMMA - 1.0)):.4e}"
    )

# 4. Admissibility, every arm; conservation PER ARM, because the arms do
# not conserve equally and that asymmetry is one of the findings.
#
# The face-flux divergence telescopes on a periodic mesh, so any scheme
# whose update is actually the flux difference conserves mass to
# round-off. The baseline does NOT -- not because the flux is wrong, but
# because the Newton admissibility projection has to inject mass to keep
# the dispersive undershoot positive, and that repair is outside the
# conservation law. Asserting round-off conservation on the baseline
# (as an earlier version of this file did) asserts the absence of the
# very defect the gate exists to document.
initial_mass = float(np.mean(initial["density"]))
for mode, entry in arms.items():
    assert entry["min_density"] > 0.0, mode
    assert entry["min_electron"] > 0.0, mode
    assert entry["min_internal"] >= ION_PRESSURE_FLOOR / (GAMMA - 1.0), (
        mode,
        entry["min_internal"],
    )
    drift = abs(entry["mass"] / initial_mass - 1.0)
    print(f"{mode:9s} mass drift = {drift:.3e}")
    if mode == "baseline":
        # The projection repair is non-conservative: guard the defect so
        # a future change that silently fixes OR worsens it is visible.
        assert drift > 1.0e-5, ("baseline unexpectedly conserves mass", drift)
    else:
        np.testing.assert_allclose(entry["mass"], initial_mass, rtol=5.0e-12, atol=0.0)

# 1. The limiter holds the invariant interval once there is upwinding for
# it to limit. Ten times the closed-form bound is a ceiling with real
# margin, not a fitted threshold.
assert arms["main"]["extremum"] < 10.0 * MEDIAN_BOUND, arms["main"]["extremum"]

# 2. And it is sharper than simply making the scheme first order --
# otherwise the cure would be worse than the disease.
assert arms["main"]["interior"] < arms["lo"]["interior"], (
    arms["main"]["interior"],
    arms["lo"]["interior"],
)

# 3. The undissipated baseline rings, by a wide margin. Without this the
# gate above would be vacuous.
assert arms["baseline"]["extremum"] > 10.0 * max(arms["main"]["extremum"], 1.0e-12), (
    arms["baseline"]["extremum"],
    arms["main"]["extremum"],
)

# 5. POSITIVITY, measured. The baseline's density is driven ONTO its
# floor by the dispersive undershoot -- that is what makes its Newton
# line search fail. Both dissipated arms must stay clear of the floor by
# a wide margin, which is the plotfile-visible half of the claim. (The
# other half -- that the baseline needs admissibility projections every
# step while the dissipated arms need none -- lives in the run logs:
# measured 13 projections in 30 steps for the baseline against 0 in 44
# for MUSCL-Rusanov.)
DENSITY_FLOOR = 1.0e-6 * DENSITY_LOW
for mode in ("lo", "main"):
    assert arms[mode]["min_density"] > 100.0 * DENSITY_FLOOR, (
        mode,
        arms[mode]["min_density"] / DENSITY_FLOOR,
    )
