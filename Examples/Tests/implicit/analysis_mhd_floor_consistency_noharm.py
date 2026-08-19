#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Do-no-harm gate of the floor-consistency relaxation source.

The quiet static-contact deck (an exact steady solution the HLLC flux
preserves to solver round-off over 100 steps) runs with
implicit_mhd.floor_consistency_rate armed at the internal 1/(theta dt)
cap and a ledger file. Every cell of this deck sits orders of magnitude
above its admissibility bound, far outside the rectifier's exact-zero
gate boundary at 1.2x the bound (theta = 1 here, so the bound IS the
cell floor, at 1e-5 of the state) -- by the construction guarantee of
floor_consistency_deficit the source is IDENTICALLY zero, not merely
small:

  1. every output field of the final plotfile is BIT-IDENTICAL
     (np.array_equal) to the rate-off twin
     (test_1d_theta_implicit_mhd_static_contact_hllc, the dependency);
  2. the Newton iteration history (newton.txt) matches the twin's
     bit for bit -- identical residuals converge identically;
  3. the supply ledger books EXACTLY zero mass and energy in every one
     of its 100 rows (the file's existence itself proves the source was
     armed: implicit_mhd.floor_ledger_file requires a positive rate).

Under the adversarially FLIPPED rectifier (source drains above the bound
instead of supplying below), every healthy cell feels an O(state) drain:
gates 1 and 2 fail catastrophically and the ledger books a nonzero
column -- this is the tripwire that pins the one-sidedness.

Usage: analysis_mhd_floor_consistency_noharm.py <initial> <final plotfile>
"""

import sys

import numpy as np
import yt

baseline_directory = "../test_1d_theta_implicit_mhd_static_contact_hllc"

FIELDS = [
    "Bz",
    "jz",
    "Pe",
    "implicit_mhd_mass_density",
    "implicit_mhd_electron_energy",
]


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


# Keep the dataset handles alive: yt covering grids hold only a weak
# reference to their dataset.
final_ds, final = get_data(sys.argv[2])
baseline_ds, baseline_final = get_data(f"{baseline_directory}/diags/diag000100")

# 1. Bit-identity of every output field against the rate-off twin.
for field in FIELDS:
    mine = final["boxlib", field].value
    twin = baseline_final["boxlib", field].value
    assert np.array_equal(mine, twin), (
        f"field {field} is not bit-identical to the rate-off twin: "
        f"max |diff| = {np.max(np.abs(mine - twin)):.3e} "
        f"(max |twin| = {np.max(np.abs(twin)):.3e})"
    )
print(f"bit-identity: {len(FIELDS)} fields match the rate-off twin exactly")

# 2. Bit-identical Newton history: identical residuals converge
# identically, so every logged column must match. newton.txt APPENDS
# across reruns of a test directory (long-lived dev trees accumulate
# many sessions), so compare only the most recent session on each side
# (rows after the last step-number reset).
def last_session(rows, step_col=0):
    resets = np.nonzero(np.diff(rows[:, step_col]) < 0)[0]
    return rows[(resets[-1] + 1) if len(resets) else 0 :]


history = last_session(np.atleast_2d(np.loadtxt("diags/newton.txt")))
baseline_history = last_session(
    np.atleast_2d(np.loadtxt(f"{baseline_directory}/diags/newton.txt"))
)
assert np.array_equal(history, baseline_history), (
    "newton.txt differs from the rate-off twin: the armed source "
    "perturbed the solve"
)
print(f"newton.txt: {history.shape[0]} solves match the twin bit for bit")

# 3. The ledger books EXACTLY zero (and its existence proves the source
# was armed: floor_ledger_file requires floor_consistency_rate > 0).
ledger = np.atleast_2d(np.loadtxt("diags/floor_ledger.txt"))
assert ledger.shape[0] == history.shape[0], (
    f"ledger rows ({ledger.shape[0]}) != solves ({history.shape[0]})"
)
assert np.all(ledger[:, 1] == 0.0) and np.all(ledger[:, 2] == 0.0), (
    "the supply ledger booked a nonzero amount on a healthy deck: "
    f"final row mass = {ledger[-1, 1]:.3e}, energy = {ledger[-1, 2]:.3e}"
)
print(f"ledger: {ledger.shape[0]} rows, all exactly zero")

print("PASS")
