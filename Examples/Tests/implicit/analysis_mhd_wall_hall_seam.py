#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""Wall-seam guard test for the Hall/inertia/hyper Ohm terms (RZ MHD).

A stepped pec wall (r_w = 0.30 m for z < 0, 0.22 m for z > 0) is driven
by a ramped external A_theta coil blob just inside the step corner over
a uniform embedded B_z, with the Hall term active. The pec image
current is a SHEET along the stair surface: at seam-adjacent live rows
the Hall stencils (half-sum/four-point corner currents, multi-cell b_cc
averages) read straight across the seam, where curl B / mu0 is the
conductor's drive-scale SURFACE current -- without the wall-seam guard
this pumps a spurious electric field at the seam (the measured rr14f
formation-deck E_theta blow-up at the exit step corner, in miniature).

Checks (bounds calibrated against this deck, 10 steps, dt = 1e-7 s;
UNFIXED = the pre-guard solver, quoted from a scratch build of the
parent commit):

1. Seam quiet: max |E| over the seam band (cells whose corners are all
   seam-guarded) stays at the eta J / ideal-dissipation scale.
   Measured at step 10: |E_theta| 2.24 V/m guarded vs 33.0 V/m
   UNFIXED (grew 6.4 -> 33.0 over steps 1..10 and was the GLOBAL
   E_theta max throughout -- the production signature); |E_r| 0.091 vs
   69.3; |E_z| 0.34 vs 9.0. The eta J scale of the seam band (from the
   plotted-B curl) is ~0.94 V/m; the guarded seam sits at 2.4x it (the
   remainder is the untouched ideal UCT dissipation on the seam field
   jumps), the unfixed pump at ~30x.

2. Production signature: the seam band must NOT be the global
   |E_theta| maximum (guarded ratio 0.23; unfixed 1.0). The interior
   Hall physics stays alive (global max 9.8 V/m, whistler response of
   the coil ramp) -- an over-covering guard that zeroes the Hall term
   everywhere fails this check from the other side.

3. Masked-band freeze: every fully-masked cell keeps its initial total
   field to solver precision (the wall projection idiom of the
   wall_pec_ramp test; measured drift ~1e-15 relative).

The run itself asserts residual/PC seam consistency:
pc_mhd_block.resistive_validate_assembly checks the banded/direct
sparse rows -- including the seam-guard row drops -- against the
matrix-free operator to roundoff at every preconditioner update.
"""

import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(50)

# Must match the inputs file / wall_step_r30_r22.csv
MU0 = 4.0e-7 * np.pi
ETA = MU0 * 50.0  # plasma_resistivity = mu0 * D
B0Z = 2.0e-3  # embedded uniform B_z
R_LO = 0.30  # wall radius, z < 0
R_HI = 0.22  # wall radius, z > 0
GUARD_RADIAL_REACH = 2  # ImplicitMHDWallMask seam-guard footprint
GUARD_WINDOW = (-2, 1)  # axial cell window around a z-node row

FIELDS = ["Br", "Bt", "Bz", "Er", "Et", "Ez"]


def load(fn):
    ds = yt.load(fn)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    data = {f: grid["boxlib", f].v.squeeze() for f in FIELDS}
    nr, nz = int(ds.domain_dimensions[0]), int(ds.domain_dimensions[1])
    rmin = float(ds.domain_left_edge[0])
    zmin = float(ds.domain_left_edge[1])
    dr = (float(ds.domain_right_edge[0]) - rmin) / nr
    dz = (float(ds.domain_right_edge[1]) - zmin) / nz
    return data, (nr, nz, dr, dz, rmin, zmin)


def band_masks(nr, nz, dr, dz, rmin, zmin):
    """Seam band (cells whose four corners are all seam-guarded) and
    deep-metal band, replicating ImplicitMHDWallMask's tables."""
    tol = 1.0e-3 * dr
    z_cells = zmin + (np.arange(nz) + 0.5) * dz
    z_nodes = zmin + np.arange(nz + 1) * dz
    rw_cell = np.where(z_cells < 0.0, R_LO, R_HI)
    # the z = 0 node belongs to the (-2, 0.30) -> (0, 0.30) polyline
    # segment (piecewise-linear interpolation, endpoint inclusive)
    rw_node = np.where(z_nodes <= 0.0, R_LO, R_HI)
    cc = np.ceil((rw_cell - tol) / dr - 0.5).astype(int)  # first masked cell
    et = np.ceil((rw_node - tol) / dr).astype(int)  # first masked corner
    # first seam-guarded corner: min over the axial cell window of the
    # cell-centered masked table, minus the radial reach; clipped by the
    # family's own masked table
    guard = np.empty(nz + 1, dtype=int)
    for jn in range(nz + 1):
        lo = max(jn + GUARD_WINDOW[0], 0)
        hi = min(jn + GUARD_WINDOW[1], nz - 1)
        guard[jn] = min(et[jn], cc[lo : hi + 1].min() - GUARD_RADIAL_REACH)
    icell = np.arange(nr)
    # cell (i, j) holds corners (i..i+1) x (z-nodes j..j+1): all four are
    # guarded when i >= max(guard[j], guard[j+1])
    seam = icell[:, None] >= np.maximum(guard[:-1], guard[1:])[None, :]
    # deep metal: one full cell past the largest wall radius over the
    # cell's z extent (freezes every staggering of the plotted cell
    # average); outermost ring excluded (r_max boundary faces)
    r_cc = rmin + (icell + 0.5) * dr
    rw_max = np.maximum(np.maximum(rw_node[:-1], rw_node[1:]), rw_cell)
    metal = (r_cc[:, None] >= rw_max[None, :] + 1.5 * dr) & (
        icell[:, None] < nr - 1
    )
    assert seam.any() and metal.any(), "degenerate band layout"
    return seam, metal


data0, geom = load(sys.argv[1])
data1, geom1 = load(sys.argv[2])
assert geom == geom1, "snapshot geometry mismatch"
nr, nz, dr, dz, rmin, zmin = geom
seam, metal = band_masks(*geom)

# --- 1. Seam band quiet: the Hall/inertia/hyper pump is gone -----------
seam_et = np.abs(data1["Et"])[seam].max()
seam_er = np.abs(data1["Er"])[seam].max()
seam_ez = np.abs(data1["Ez"])[seam].max()

# eta J scale of the seam band from the plotted-B curl (crude cell-
# centered gradients; the right scale). The guarded seam additionally
# carries the untouched ideal UCT dissipation on the seam field jumps,
# a few x this scale.
r = rmin + (np.arange(nr) + 0.5) * dr
j_theta = (
    np.gradient(data1["Br"], dz, axis=1) - np.gradient(data1["Bz"], dr, axis=0)
) / MU0
j_r = -np.gradient(data1["Bt"], dz, axis=1) / MU0
j_z = np.gradient(r[:, None] * data1["Bt"], dr, axis=0) / r[:, None] / MU0
eta_j_seam = ETA * max(
    np.abs(j_theta)[seam].max(),
    np.abs(j_r)[seam].max(),
    np.abs(j_z)[seam].max(),
)
print(f"seam band: max|Et| = {seam_et:.3e} V/m, max|Er| = {seam_er:.3e} V/m,")
print(f"           max|Ez| = {seam_ez:.3e} V/m, eta*J scale = {eta_j_seam:.3e} V/m")

# Calibrated: guarded 2.24 / 0.091 / 0.34 V/m; UNFIXED 33.0 / 69.3 /
# 9.0 V/m (see the module docstring). The Et bound also stays tied to
# the measured eta*J scale so a drive recalibration cannot silently
# hollow the check out.
bound_et = max(8.0 * eta_j_seam, 4.0)
assert seam_et < bound_et, (
    f"seam E_theta above the eta*J scale: {seam_et:.3e} V/m "
    f"(bound {bound_et:.3e}) -- unguarded Hall/inertia/hyper seam terms?"
)
assert seam_er < 2.0, f"seam E_r pumped: {seam_er:.3e} V/m (bound 2.0)"
assert seam_ez < 3.0, f"seam E_z pumped: {seam_ez:.3e} V/m (bound 3.0)"

# --- 2. Production signature + interior Hall alive ----------------------
glob_et = np.abs(data1["Et"]).max()
ratio = seam_et / glob_et
print(f"global max|Et| = {glob_et:.3e} V/m; seam/global = {ratio:.3f}")
assert ratio < 0.5, (
    f"the seam band is the global E_theta maximum (ratio {ratio:.3f}) -- "
    "the rr14f drive-powered seam pump signature"
)
assert glob_et > 3.0, (
    f"interior Hall response missing (global max|Et| = {glob_et:.3e} V/m) "
    "-- is the seam guard over-covering?"
)

# --- 3. Masked-band total-field freeze ----------------------------------
for f in ("Br", "Bt", "Bz"):
    drift = np.abs(data1[f] - data0[f])[metal].max() / B0Z
    print(f"metal-band relative drift {f}: {drift:.3e}")
    assert drift < 1.0e-9, f"masked band changed ({f}: {drift:.3e})"

print("wall Hall-seam guard test PASSED")
