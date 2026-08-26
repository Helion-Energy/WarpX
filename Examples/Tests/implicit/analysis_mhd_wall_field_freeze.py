#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""Shaped-wall FIELD freeze (implicit_mhd.wall_field_freeze) for the
theta-implicit RZ MHD recast -- work item B, T-B1.

The freeze turns every evolved magnetic-field face whose complete
Faraday stencil lies in the masked band into an exact identity row
(F = B - B^n): the exterior plasma-response B stays bit-frozen at its
boot value (zero), so the discrete curl of the evolved field -- the
band eddy current that screened the coil drive on the formation ladder
-- is exactly zero outside the wall from step 1 onward, with NO
band-eta knob armed (wall_band_eta_override = 0 here: the freeze
supersedes the override's field-advance job).

This run is the conductive contrast twin
(test_rz_theta_implicit_mhd_wall_band_eta_conductive: same deck, same
zeroed override, freeze OFF -- deep-band |j_theta| ~3.2e4 A/m^2,
DOMINATING the interior) plus wall_field_freeze = 1. Asserted here, at
every post-boot snapshot (steps 2, 4, 6):

1. deep-band plotted Br and Bt are EXACTLY 0.0 (bitwise): the external
   drive has no r/theta component (A_theta is z-independent, so its
   discrete curl gives exactly zero Br), so the plotted totals ARE the
   evolved response there -- the eddy components stay bit-frozen at
   the zero boot response across all steps;
2. deep-band |j_theta| (numerical curl of the plotted totals; the
   uniform external Bz ramp is curl-free so it drops out) is zero to
   ROUND-OFF -- bounded 10 orders below the conductive twin's band
   current, and the suppression ratio vs the twin is < 1e-8;
3. the deep-band Bz carries no eddy structure (spatial spread at the
   round-off scale of the uniform external ramp value);
4. the LIVE interior response RECOVERS vs the screened twin (> 1.5x:
   with the band frozen, the drive reaches the plasma -- the same
   coil-coupling recovery bar as the band-eta override test);
5. the driven window stays Newton-grind-free, and the run itself
   validates the frozen-row PC assembly against the matrix-free
   operator at every update (pc_mhd_block.resistive_validate_assembly).

Usage:
  analysis_mhd_wall_field_freeze.py <diags_dir> <reference_diag_end>
"""

import os
import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(0)

FIELDS = ("Br", "Bt", "Bz")
MU0 = 4.0e-7 * np.pi
# Round-off bar for the band current [A/m^2]: the only nonzero band
# signal is the ULP wobble of the discrete-curl uniform external ramp
# (~1e-12 measured); the conductive twin books ~3.2e4.
BAND_JT_ROUNDOFF = 1.0e-6
BAND_SUPPRESSION = 1.0e-8
INTERIOR_RECOVERY = 1.5
MAX_NEWTON_ITERS = 4


def load(plotfile):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, {f: np.squeeze(grid["boxlib", f].value) for f in FIELDS}


diags_dir = sys.argv[1]
snapshots = [os.path.join(diags_dir, f"diag{step:06d}") for step in (2, 4, 6)]
ds, _ = load(snapshots[0])

nr, nz = int(ds.domain_dimensions[0]), int(ds.domain_dimensions[1])
dr = (float(ds.domain_right_edge[0]) - float(ds.domain_left_edge[0])) / nr
dz = (float(ds.domain_right_edge[1]) - float(ds.domain_left_edge[1])) / nz
r_centers = float(ds.domain_left_edge[0]) + (np.arange(nr) + 0.5) * dr
z_centers = float(ds.domain_left_edge[1]) + (np.arange(nz) + 0.5) * dz

# The mask, replicated exactly as ImplicitMHDWallMask builds it, then
# eroded twice (edge-padded): the probed deep-band cells are strictly
# exterior -- every magnetic-field face feeding their cell-centered
# diagnostic values is frozen (the freeze tables sit within one cell of
# the masked contour), clear of the live/frozen straddle shell.
wall_radius = np.where(z_centers < 0.0, 0.30, 0.22)
masked = (r_centers[:, None] + 1.0e-3 * dr) >= wall_radius[None, :]


def erode(region):
    padded = np.pad(region, 1, mode="edge")
    return (
        region
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )


deep_band = erode(erode(masked))
interior = ~masked
assert deep_band.any() and interior.any(), "degenerate mask layout"


def j_theta(fields):
    """Response |J_theta| = |dBr/dz - dBz/dr| / mu0 from the plotted
    (total) cell-centered fields: the uniform external Bz ramp is
    curl-free, so the numerical curl isolates the eddy response."""
    dbr_dz = np.gradient(fields["Br"], dz, axis=1)
    dbz_dr = np.gradient(fields["Bz"], dr, axis=0)
    return np.abs(dbr_dz - dbz_dr) / MU0


# Conductive twin (freeze off, override off): the screening teeth.
_, reference = load(sys.argv[2])
reference_jt = j_theta(reference)
reference_band_jt = float(np.max(reference_jt, where=deep_band, initial=0.0))
reference_interior_jt = float(
    np.max(reference_jt, where=interior, initial=0.0)
)
assert reference_band_jt > 1.0, (
    f"conductive twin band current is tiny ({reference_band_jt:.3e}): "
    "the freeze assertions below would be vacuous"
)

interior_jt_end = 0.0
for plotfile in snapshots:
    _, state = load(plotfile)
    for name, field in state.items():
        assert np.isfinite(field).all(), f"{name} has non-finite values"

    # 1. Bit-frozen eddy components: the plotted band totals ARE the
    # evolved response for Br/Bt (zero external there), and the
    # response boots at exactly zero -- any evolved band face would
    # break bitwise equality with 0.0.
    for name in ("Br", "Bt"):
        band_values = state[name][deep_band]
        nonzero = int(np.count_nonzero(band_values))
        print(f"{plotfile}: deep-band {name} nonzero count = {nonzero}")
        assert nonzero == 0, (
            f"{plotfile}: {name} has {nonzero} nonzero deep-band values "
            f"(max |{name}| = {np.max(np.abs(band_values)):.3e}): the "
            "exterior evolved field is not frozen"
        )

    # 2. Band current zero to round-off, and collapsed vs the twin.
    jt = j_theta(state)
    band_jt = float(np.max(jt, where=deep_band, initial=0.0))
    interior_jt = float(np.max(jt, where=interior, initial=0.0))
    suppression = band_jt / reference_band_jt
    print(
        f"{plotfile}: deep-band max |j_theta| = {band_jt:.3e} A/m^2 "
        f"(twin {reference_band_jt:.3e}, suppression {suppression:.3e}); "
        f"interior max = {interior_jt:.3e}"
    )
    assert band_jt < BAND_JT_ROUNDOFF, (
        f"{plotfile}: band current {band_jt:.3e} A/m^2 above the "
        f"round-off bar {BAND_JT_ROUNDOFF:.1e}"
    )
    assert suppression < BAND_SUPPRESSION, (
        f"{plotfile}: band current only suppressed to {suppression:.3e} "
        f"of the conductive twin (bar {BAND_SUPPRESSION:.1e})"
    )

    # 3. No eddy structure in the band Bz: the plotted values are the
    # uniform external ramp alone (spread at its discrete-curl ULP
    # wobble).
    band_bz = state["Bz"][deep_band]
    bz_spread = float(np.max(band_bz) - np.min(band_bz))
    bz_scale = max(float(np.max(np.abs(band_bz))), 1.0e-300)
    print(f"{plotfile}: deep-band Bz spread = {bz_spread:.3e} "
          f"(scale {bz_scale:.3e})")
    assert bz_spread < 1.0e-12 * bz_scale, (
        f"{plotfile}: deep-band Bz carries structure "
        f"({bz_spread:.3e} of {bz_scale:.3e}): a band response leaked in"
    )

    # The interior must be shielding (the drive is alive).
    assert interior_jt > 0.0, "no interior response current: dead drive"
    interior_jt_end = interior_jt

# 4. Coil-coupling recovery: with the band frozen (unable to screen),
# the interior carries the shielding itself.
interior_match = interior_jt_end / max(reference_interior_jt, 1.0e-300)
print(f"interior response vs screened twin: {interior_match:.3f}x")
assert interior_match > INTERIOR_RECOVERY, (
    f"interior response did not recover ({interior_match:.3f}x): the "
    "freeze leaked into live rows"
)

# 5. Newton health over the driven window. WarpX APPENDS to the newton
# diagnostic file and ctest does not clean the test directory, so a
# re-run accumulates rows: judge the LAST full window.
newton_rows = np.loadtxt(os.path.join(diags_dir, "newton.txt"), ndmin=2)
assert len(newton_rows) >= 6, "the freeze run did not complete all steps"
newton_iters = newton_rows[-6:, 2]
print(f"newton iters/step: {newton_iters.astype(int).tolist()}")
assert np.max(newton_iters) <= MAX_NEWTON_ITERS, (
    f"drive-era Newton grind ({int(np.max(newton_iters))} iters in a step)"
)

print("shaped-wall field freeze test PASSED")
