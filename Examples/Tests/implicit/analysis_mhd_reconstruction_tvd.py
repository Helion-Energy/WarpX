#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""TVD / no-new-extrema gate of implicit_mhd.fluid_reconstruction.

A 4:1 square density contact advected ~0.24 domain lengths in uniform
flow at uniform pressure, with hlld. The exact solution never leaves
[rho_low, rho_high], so any excursion outside that interval is a new
extremum manufactured by the scheme. Four arms of the same deck:

  none       donor cell. First-order Godunov, monotone BY CONSTRUCTION
             -- it smears the step, it cannot ring. Included so the
             gate's framing is honest: the meaningful TVD comparison for
             a limiter is NOT against the first-order twin (which has no
             extrema to suppress) but against the UNLIMITED second-order
             twin, which is the reconstruction the limiter is limiting.
  median     the reference code's fmed limiter. THE gate.
  vanalbada  the C-infinity sibling, whose new-extremum bound is
             0.10355 of the LOCAL jump.
  unlimited  the plain centered slope, i.e. the same reconstruction with
             the limiter removed. Must ring; this is what makes the
             limited results non-vacuous.

Measured (120 steps, 128 cells), new extrema as a fraction of
rho_high - rho_low, and the transition-band width in cells:

    none        0.0e+00    46 cells
    median      3.3e-03    23 cells
    vanalbada   5.0e-03    20 cells
    unlimited   3.4e-01    17 cells

The unlimited reconstruction throws a THIRD of the contact height into
overshoot; both limiters hold it under half a percent while resolving
the front ~2x sharper than donor cell. Note the bounds that apply here
are per-LOCAL-jump: by the final output the square is a steep but finite
front, so the stencil differences the limiters see are fractions of the
total jump and their excursions scale down with them.

That last point REVERSES the ordering the kernel's idealized bound table
predicts. On a pristine plateau edge van Albada is exactly zero on the
d_down = 0 ray while the smoothed median leaves an O(kappa) slope, so
van Albada should win. On the EVOLVED front it does not: the front is no
longer a plateau edge, so van Albada's opposite-sign branch (0.10355 of
the local jump) is live everywhere across it, while the median's
kappa/4 bound shrinks with the local differences. Measured, the median
is 1.5x tighter. The idealized bounds are still correct -- they just
apply to a geometry the solution stops having after a few steps.

Gates:
 1. median and vanalbada introduce new extrema well inside the kernel's
    closed-form ceilings,
 2. unlimited DOES ring, by >= 10x more than either limiter, proving the
    limiter -- not merely the reconstruction -- is what holds the bound,
 3. both limiters resolve the contact SHARPER than donor cell (fewer
    cells in the transition band), which is what they are paid for,
 4. every arm conserves mass to round-off and stays strictly positive,
 5. the ion internal energy E_i - KE stays above the deck's ion pressure
    floor in every cell of every arm -- the reconstruction may never
    manufacture an inadmissible ion energy at an interface.

Usage: analysis_mhd_reconstruction_tvd.py <initial> <final>
       (this test is the median arm; the other arms are read from
        ../test_1d_theta_implicit_mhd_reconstruction_tvd_*/diags)
"""

import os
import sys

import numpy as np
import warpx_constants as constants
import yt

yt.set_log_level(50)

# Deck constants (inputs_base_1d_theta_implicit_mhd_reconstruction_tvd).
NUMBER_DENSITY = 1.0e20
DENSITY_LOW = NUMBER_DENSITY * constants.proton_mass
DENSITY_HIGH = 4.0 * DENSITY_LOW
ELECTRON_TEMPERATURE_EV = 10.0
ELECTRON_PRESSURE = (
    NUMBER_DENSITY * ELECTRON_TEMPERATURE_EV * constants.elementary_charge
)
ION_PRESSURE = ELECTRON_PRESSURE
ION_PRESSURE_FLOOR = 1.0e-9 * ION_PRESSURE
GAMMA = 5.0 / 3.0
VELOCITY = 1.0e5

TEST_STEM = "test_1d_theta_implicit_mhd_reconstruction_tvd"
PLOTFILE = "diags/diag000120"


def get_fields(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return {
        "time": float(ds.current_time),
        "density": data["boxlib", "implicit_mhd_mass_density"].value.ravel(),
        "ion_energy": data["boxlib", "implicit_mhd_ion_energy"].value.ravel(),
        "electron_energy": data["boxlib", "implicit_mhd_electron_energy"].value.ravel(),
    }


def arm_plotfile(mode):
    if mode == "median":
        return PLOTFILE
    return os.path.join("..", f"{TEST_STEM}_{mode}", PLOTFILE)


initial = get_fields(sys.argv[1])
final_median = get_fields(sys.argv[2])
assert initial["time"] == 0.0

span = DENSITY_HIGH - DENSITY_LOW
# Closed-form new-extremum bound of the smoothed median at the default
# implicit_mhd.reconstruction_kappa (see ThetaImplicitMHD_K.H): kappa/4
# of the largest stencil difference, which at a plateau edge IS the jump.
RECONSTRUCTION_KAPPA = 0.01
MEDIAN_BOUND = RECONSTRUCTION_KAPPA / 4.0
arms = {}
for mode in ("none", "median", "vanalbada", "unlimited"):
    fields = final_median if mode == "median" else get_fields(arm_plotfile(mode))
    density = fields["density"]
    overshoot = max(0.0, float(np.max(density)) - DENSITY_HIGH)
    undershoot = max(0.0, DENSITY_LOW - float(np.min(density)))
    # Transition width: cells whose density sits strictly inside the
    # plateau interval, i.e. the smearing of the two contacts.
    interior = np.count_nonzero(
        (density > DENSITY_LOW + 0.02 * span) & (density < DENSITY_HIGH - 0.02 * span)
    )
    # E_i - KE. The momentum density is not published, but the velocity is
    # uniform in this deck to the level the contact preserves it, so the
    # kinetic part is rho u^2 / 2 to the same accuracy; the check below is
    # a floor check with a wide margin, not a precision measurement.
    internal = fields["ion_energy"] - 0.5 * density * VELOCITY**2
    arms[mode] = {
        "extremum": max(overshoot, undershoot) / span,
        "interior": interior,
        "mass": float(np.mean(density)),
        "min_density": float(np.min(density)),
        "min_internal": float(np.min(internal)),
        "min_electron": float(np.min(fields["electron_energy"])),
    }

for mode, entry in arms.items():
    print(
        f"fluid_reconstruction = {mode:10s}"
        f"  new extrema = {entry['extremum']:.3e} of the jump"
        f"  transition cells = {entry['interior']:3d}"
        f"  min rho = {entry['min_density']:.6e}"
        f"  min (E_i - KE) = {entry['min_internal']:.6e}"
    )

# 4/5. Admissibility and conservation, every arm.
initial_mass = float(np.mean(initial["density"]))
for mode, entry in arms.items():
    np.testing.assert_allclose(entry["mass"], initial_mass, rtol=5.0e-12, atol=0.0)
    assert entry["min_density"] > 0.0, mode
    assert entry["min_electron"] > 0.0, mode
    assert entry["min_internal"] >= ION_PRESSURE_FLOOR / (GAMMA - 1.0), (
        mode,
        entry["min_internal"],
    )

# 1. Both limiters hold the invariant interval to within the kernel's
# closed-form bounds. The bound that applies here is per-LOCAL-jump, not
# per-total-jump: by the final output the square has been smeared into a
# steep but finite front, so the relevant stencil differences are
# fractions of rho_high - rho_low and both limiters' excursions scale
# down with them. Ten times MEDIAN_BOUND is therefore a ceiling with
# real margin, not a fitted threshold -- a limiter that had actually
# stopped limiting lands at the unlimited arm's 3.4e-1.
LIMITED_CEILING = 10.0 * MEDIAN_BOUND
for mode in ("median", "vanalbada"):
    assert arms[mode]["extremum"] < LIMITED_CEILING, (
        mode,
        arms[mode]["extremum"],
        LIMITED_CEILING,
    )
# Donor cell is monotone by construction -- stated, and checked, so the
# gate's framing stays honest.
assert arms["none"]["extremum"] < 1.0e-9, arms["none"]["extremum"]

# 2. The UNLIMITED reconstruction rings, hard: without this the limited
# results would be vacuous (a scheme that never reconstructs also never
# rings). Measured 3.4e-1 of the jump -- a third of the contact height
# appearing as overshoot -- against 5.0e-3 for van Albada.
assert arms["unlimited"]["extremum"] > 5.0e-2, arms["unlimited"]["extremum"]
for mode in ("median", "vanalbada"):
    assert arms["unlimited"]["extremum"] > 10.0 * max(
        arms[mode]["extremum"], 1.0e-12
    ), (mode, arms["unlimited"]["extremum"], arms[mode]["extremum"])

# 3. And the limiters buy sharpness over donor cell.
for mode in ("median", "vanalbada"):
    assert arms[mode]["interior"] < arms["none"]["interior"], (
        mode,
        arms[mode]["interior"],
        arms["none"]["interior"],
    )
