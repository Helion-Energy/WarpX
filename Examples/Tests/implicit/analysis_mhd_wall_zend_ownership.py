#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Ownership of the z_hi domain end face where the shaped wall runs
into it (theta-implicit RZ MHD recast).

The contour steps inward EXACTLY at the z_hi plane (r_w = 0.30 m for
z < 0.125, 0.20 m beyond) -- the production formation geometry in
miniature, where the contour is still cutting inward as it crosses the
end plane.  The wall mask's per-z first-masked table is built over the
mask's GHOST rows as well, and every face classifier reads it at the
ghost cell just outside a z domain face.  Without a precedence rule the
boundary condition of a DOMAIN face is therefore decided by wall
geometry OUTSIDE the simulated volume, and the last live z row splits
mid-row: the cells "under the overhang" (whose z_hi ghost neighbour is
masked) take the shaped-wall interface contract -- absorb image,
wall_temperature bath, wall_conduction_scale chi clamp, wall
cap/rectifier rules -- while their neighbours on the SAME flat face
take the z-end contract (z_wall_temperature, tensor chi, uncapped).

CONTRACT.  The mask's authority stops at the domain: a cell outside the
z domain range takes the class of the nearest IN-DOMAIN cell, so a z
domain face is classified by its INTERIOR cell alone.  The mask still
wins for every cell it claims in-domain (a masked interior cell is
interior metal and takes no domain-end exchange) and the z boundary
condition applies exactly to the cells the mask does not claim.

GATE.  The two reservoirs sit on OPPOSITE sides of the column
temperature (T_wall = 2 eV, z_wall_temperature = 200 eV, T0 = 100 eV),
so a split face is unmistakable: every live cell of the last z row that
the mask does not claim must HEAT toward the z-end reservoir, and the
cells that carry no radial wall face must all heat by the SAME amount
(one contract, one flat boundary).  On the unguarded classification the
overhang cells cool instead -- a ~14 eV sign-flipped step between
neighbouring cells of one domain boundary.

Usage: analysis_mhd_wall_zend_ownership.py <diag_dir>
"""

import glob
import sys

import numpy as np
import yt

yt.set_log_level(50)

diag_dir = sys.argv[1]

proton_mass = 1.67262192595e-27  # WarpX PhysConst::m_p (ablastr/constant.H)
qe = 1.602176634e-19
T0_ev = 100.0
Twall_ev = 2.0
Tzend_ev = 200.0
gamma = 5.0 / 3.0
nr = 32
nz = 16
dr = 0.5 / nr
dz = 0.25 / nz
z_lo = -0.125
tol_r = 1.0e-3 * dr

e0 = (qe / proton_mass) * T0_ev / (gamma - 1.0)


def first_masked(j):
    """ImplicitMHDWallMask: first masked cell-centered radial index at the
    cell-centre axial position, with the mask's authority clamped to the
    domain (the contract under test)."""
    z = z_lo + (min(max(j, 0), nz - 1) + 0.5) * dz
    rw = (0.30 if z < 0.125 else 0.20) - tol_r
    return int(np.ceil(rw / dr - 0.5 - 1.0e-12))


wall_fm = first_masked(nz - 1)
# the geometry must actually contend: the polyline's OUT-of-domain value
# has to disagree with the last in-domain row, or the test proves nothing
z_ghost = z_lo + (nz + 0.5) * dz
ghost_fm = int(
    np.ceil(((0.30 if z_ghost < 0.125 else 0.20) - tol_r) / dr - 0.5 - 1.0e-12)
)
assert ghost_fm < wall_fm, (
    f"no contention in this geometry: ghost row first-masked {ghost_fm} vs "
    f"in-domain {wall_fm}"
)
print(
    f"z_hi end row: mask claims i >= {wall_fm}; the polyline continued "
    f"past the domain claims i >= {ghost_fm} -- contended band "
    f"i = {ghost_fm}..{wall_fm - 1}"
)


def load(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    rho = np.squeeze(data["boxlib", "implicit_mhd_mass_density"].value)
    ue = np.squeeze(data["boxlib", "implicit_mhd_electron_energy"].value)
    return ue / rho


plotfiles = sorted(glob.glob(f"{diag_dir}/diag??????"))
assert len(plotfiles) >= 2, f"need initial+final snapshots, got {len(plotfiles)}"
row_initial = load(plotfiles[0])[:, nz - 1]
row = load(plotfiles[-1])[:, nz - 1]

# the column boots uniform on every live cell of the row
assert np.all(np.abs(row_initial[:wall_fm] / e0 - 1.0) < 1.0e-12)

print(
    "last z row T [eV], i = 0..%d: %s"
    % (
        wall_fm,
        np.array2string(
            row[: wall_fm + 1] / e0 * T0_ev, precision=3, max_line_width=200
        ),
    )
)

# ------------------------------------------------------------- gate 1
# every live cell of the domain end row exchanges with the z-end
# reservoir, so every one of them HEATS (T_zend = 200 eV > T0 = 100 eV).
# The wall-adjacent cell i = wall_fm - 1 is excluded: it legitimately
# carries the cold RADIAL wall face as well (it is the corner, and its
# net drift is the sum of the two contracts).
for i in range(wall_fm - 1):
    assert row[i] > e0, (
        f"cell ({i}, {nz - 1}) on the z_hi domain face did not heat toward "
        f"the {Tzend_ev:.0f} eV z-end reservoir: "
        f"{row[i] / e0 * T0_ev:.3f} eV -- its boundary condition was taken "
        f"over by the shaped-wall interface contract"
    )

# ------------------------------------------------------------- gate 2
# ONE contract on one flat face: the cells with no radial wall face and
# no radial-conduction reach to it must all heat by the same amount
interior = [i for i in range(wall_fm - 3)]
rise = np.array([row[i] - e0 for i in interior])
spread = (rise.max() - rise.min()) / rise.max()
print(
    f"z_hi end row heating: {rise.max() / e0 * T0_ev:.4f} eV, "
    f"spread across i = 0..{interior[-1]} = {spread:.3e}"
)
assert spread < 1.0e-3, (
    f"the z_hi domain face does not carry one boundary contract: the "
    f"heating varies by {spread:.3e} across the row"
)

# ------------------------------------------------------------- gate 3
# the mask still wins for the cells it claims in-domain: the masked band
# of the domain end row sits on its rigid exterior clamp image (T_wall),
# untouched by the z-end reservoir it shares a face with
for i in range(wall_fm, nr):
    masked_ev = row[i] / e0 * T0_ev
    assert abs(masked_ev - Twall_ev) < 1.0e-3 * Twall_ev, (
        f"masked cell ({i}, {nz - 1}) left its rigid clamp image: "
        f"{masked_ev:.4f} eV vs T_wall {Twall_ev:.2f} eV -- the domain end "
        f"exchange must not reach cells the mask claims"
    )

# ------------------------------------------------------------- gate 4
# the corner cell keeps BOTH contracts: heated through the z_hi face,
# cooled through the radial wall face, so it must sit between the two
corner = row[wall_fm - 1]
print(
    f"z_hi/r-wall corner cell ({wall_fm - 1}, {nz - 1}): {corner / e0 * T0_ev:.3f} eV"
)
assert row[wall_fm - 2] > corner, (
    "the corner cell is not cooled by its radial wall face"
)
assert corner / e0 * T0_ev > Twall_ev, (
    "the corner cell collapsed onto the wall bath: its z_hi face lost its "
    "domain-end contract"
)

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert 1 <= newton_history[-1][2] <= 20

print("z_hi domain-face ownership: all gates passed")
