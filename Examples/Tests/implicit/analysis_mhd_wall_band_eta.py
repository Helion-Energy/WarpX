#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""Wall-band resistivity override (implicit_mhd.wall_band_eta_override)
for the theta-implicit RZ MHD recast -- the T5 conductive-frozen-band
repro in miniature.

A uniform external Bz ramp drives a shielding response through a
dielectric (EM-transparent) stepped wall whose plasma resistivity is a
density-INDEPENDENT constant eta = mu0*D: even after the exterior clamp
scrapes the band to the mass floor, nothing keys the band's eta up, so
without the knob the frozen band carries a finite eddy J_theta -- the
measured formation-ladder screening channel (band |J_theta| 20x the
interior median).

mode = "reference" (CLI wall_band_eta_override=0, the T5 repro):
asserts the DEEP-band |j_theta| DOMINATES the interior response -- the
screening teeth of the override run's collapse check.

mode = "override" (deck default, eta_band = mu0*Dwall with Dwall = 1e4
m^2/s = 500x the plasma D): the band eta is a state-independent
CONSTANT -- bit-equal across the whole deep band, asserted from the
plotted PC eta register (a mid-band ramp would leave a resistive shell
whose mu0 L^2/eta time can sit on the drive scale, and a state-keyed
band eta is the Jacobian stiffness that froze the production arm) --
band-interior response currents decay on L^2/Dwall ~ 4e-6 s << dt, so
the deep-band |j_theta| must COLLAPSE below 1e-2 of the reference
run's (the production bar from the run_v8_wall frozen-step abort;
expected ~D/Dwall, measured 3.6e-3), the interior response must
RECOVER (the un-screened drive reaches the plasma), and the driven
window must stay Newton-grind-free (the freeze signature of the
aborted arm). The run itself validates the overridden-eta PC assembly
against the matrix-free operator at every update
(pc_mhd_block.resistive_validate_assembly).

Usage:
  analysis_mhd_wall_band_eta.py <diag_end> reference
  analysis_mhd_wall_band_eta.py <diag_end> override <reference_diag_end>
"""

import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(0)

FIELDS = ("Br", "Bz")


def load(plotfile):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, {f: np.squeeze(grid["boxlib", f].value) for f in FIELDS}


ds, state = load(sys.argv[1])
mode = sys.argv[2]
assert mode in ("reference", "override"), f"unknown mode {mode}"

nr, nz = int(ds.domain_dimensions[0]), int(ds.domain_dimensions[1])
dr = (float(ds.domain_right_edge[0]) - float(ds.domain_left_edge[0])) / nr
dz = (float(ds.domain_right_edge[1]) - float(ds.domain_left_edge[1])) / nz
r_centers = float(ds.domain_left_edge[0]) + (np.arange(nr) + 0.5) * dr
z_centers = float(ds.domain_left_edge[1]) + (np.arange(nz) + 0.5) * dz

# The mask, replicated exactly as ImplicitMHDWallMask builds it, then
# eroded twice (edge-padded: constant continuation past the domain) so
# the probed DEEP-band cells are clear of the un-overridden interface
# monolayer and of the cell-centered diagnostic's corner averaging.
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

for name, field in state.items():
    assert np.isfinite(field).all(), f"{name} has non-finite values"

MU0 = 4.0e-7 * np.pi
MU0_WARPX = 1.2566370612685e-06  # WarpX parser mu0 (ablastr::constant)


def j_theta(fields):
    """Response |J_theta| = |dBr/dz - dBz/dr| / mu0 from the plotted
    (total) cell-centered fields: the uniform external Bz ramp is
    curl-free, so the numerical curl isolates the eddy response."""
    dbr_dz = np.gradient(fields["Br"], dz, axis=1)
    dbz_dr = np.gradient(fields["Bz"], dr, axis=0)
    return np.abs(dbr_dz - dbz_dr) / MU0


jt = j_theta(state)
band_jt = float(np.max(jt, where=deep_band, initial=0.0))
interior_jt = float(np.max(jt, where=interior, initial=0.0))
print(f"deep-band max |j_theta|: {band_jt:.6e} A/m^2")
print(f"interior  max |j_theta|: {interior_jt:.6e} A/m^2")

# The interior must be shielding in BOTH runs (the drive is on).
assert interior_jt > 0.0, "no interior response current: the drive is dead"

if mode == "reference":
    # The T5 repro teeth: the conductive frozen band DOMINATES the
    # response -- its eddy current exceeds the whole live interior's
    # (measured on this deck: band 3.2e4 vs interior 5.7e2 A/m^2, the
    # flying run's "band |J_theta| 20x the interior median" screening
    # channel).
    band_fraction = band_jt / interior_jt
    print(f"reference band/interior |j_theta| fraction: {band_fraction:.3e}")
    assert band_fraction > 1.0, (
        f"reference deep-band current does not dominate "
        f"({band_fraction:.3e}): the suppression check would be vacuous"
    )
else:
    _, reference = load(sys.argv[3])
    reference_jt = j_theta(reference)
    reference_band_jt = float(
        np.max(reference_jt, where=deep_band, initial=0.0)
    )
    reference_interior_jt = float(
        np.max(reference_jt, where=interior, initial=0.0)
    )
    # The band eta is a state-independent CONSTANT: the plotted PC eta
    # register (the corner E_theta staggering, cell-centered by the
    # diagnostic as the exact average of four identical corner values
    # deep in the band) must be BIT-EQUAL across the whole deep band at
    # exactly the deck's mu0*Dwall -- a mid-band value would be the
    # intermediate-eta resistive shell, and any spread would betray a
    # state-keyed contribution leaking into the band Jacobian.
    ds_eta = yt.load(sys.argv[1])
    grid_eta = ds_eta.covering_grid(
        level=0, left_edge=ds_eta.domain_left_edge,
        dims=ds_eta.domain_dimensions,
    )
    eta_plot = np.squeeze(
        grid_eta["boxlib", "implicit_mhd_field_resistivity_e1"].value
    )
    band_eta_values = np.unique(eta_plot[deep_band])
    eta_expected = MU0_WARPX * 1.0e4  # deck: mu0*Dwall, Dwall = 1e4
    print(f"deep-band eta values: {band_eta_values.tolist()} "
          f"(expected {eta_expected:.16e})")
    assert band_eta_values.size == 1, (
        f"band eta not constant ({band_eta_values.size} distinct values): "
        "a state-dependent contribution leaked into the override"
    )
    assert abs(band_eta_values[0] - eta_expected) < 1.0e-13 * eta_expected, (
        f"band eta {band_eta_values[0]:.16e} is not the override value"
    )

    suppression = band_jt / max(reference_band_jt, 1.0e-300)
    print(f"deep-band suppression vs reference: {suppression:.3e} "
          f"(reference band {reference_band_jt:.6e})")
    # eta override = 500x the plasma eta: the deep-band current must
    # COLLAPSE, not merely improve (expected ~1/500, measured 3.6e-3;
    # the production bar from the run_v8_wall frozen-step abort is
    # <= 1e-2 of the no-knob case at matched drive).
    assert suppression < 1.0e-2, (
        f"wall_band_eta_override did not collapse the band current "
        f"({suppression:.3e} of the no-knob run; bar 1e-2)"
    )
    # ...and the band must be subdominant to the LIVE interior inside
    # its own run (the inverse of the reference's screening signature).
    band_fraction = band_jt / interior_jt
    print(f"override-run band/interior |j_theta| fraction: "
          f"{band_fraction:.3e}")
    assert band_fraction < 0.1, (
        f"override-run band current not subdominant ({band_fraction:.3e})"
    )
    # ...while the LIVE interior response GROWS: with the band unable
    # to screen, the drive reaches the plasma and the interior carries
    # the shielding itself (measured 18x the screened reference run) --
    # the exact coil-coupling recovery the knob exists for. An override
    # that leaked into live rows would SUPPRESS the interior instead.
    interior_match = interior_jt / max(reference_interior_jt, 1.0e-300)
    print(f"interior response vs screened reference: {interior_match:.3f}x")
    assert interior_match > 1.5, (
        f"interior response did not recover ({interior_match:.3f}x): the "
        "band eta override leaked into the live region"
    )

    # Newton health over the driven window (the run_v8_wall freeze
    # signature was a drive-era grind: line-search collapse, then 150
    # consecutive zero-progress solves): every step of the overridden run
    # must CONVERGE in a few iterations. The admissibility-pinned
    # component counts themselves are stdout-only (not ctest-readable);
    # a grind-free converged trace is the available proxy -- pinning at
    # the band/interface rows manifests as exactly this iteration
    # blow-up before the frozen-step guard trips.
    newton_rows = np.loadtxt("diags/newton.txt", ndmin=2)
    newton_iters = newton_rows[:, 2]
    print(f"override-run newton iters/step: {newton_iters.astype(int).tolist()}")
    assert len(newton_rows) == 6, "override run did not complete all steps"
    assert np.max(newton_iters) <= 4, (
        f"drive-era Newton grind ({int(np.max(newton_iters))} iters in a "
        "step): the band/interface freeze signature"
    )

print(f"wall-band eta override test ({mode}) PASSED")
