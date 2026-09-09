#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""The debt-bounded active-set hold on the engineered wall-row churn deck.

newton.active_set_hold_defect_bound puts a bound on the physics debt the
release hysteresis (newton.active_set_hysteresis) may carry: at every
identification after iteration 0 the residual norm over the HELD members is
compared with max(active_set_tolerance x free_norm0, bound x norm0) and,
above it, every held member is released for that identification (the plain
release rule decides) before the linear solve and the convergence test. The
bound has two exactness limits, and this script gates both against the
sibling arms of the churn deck (same binary, same deck):

  release  -- bound 0: the release pass runs at EVERY identification that
              holds something, so the masks the linear solve, the line
              search and the exit see are the plain rule's. The solve must
              therefore reproduce the CONTROL arm (plain rule +
              globalization_diagnostics) exactly: the first 16 newton.txt
              columns (Newton and GMRES counts, norms, pinned counts, exit
              statuses, self-checks) identical token by token on every row,
              the final plotfile fields identical to the last bit, and the
              release pass reported on every solve that held a member
              (hold_releases > 0 with held == 0 after the pass, entrants and
              releases equal to the control's).
  hold     -- bound 1e9 (never reached on this deck: the held defect is of
              order 1e-6 of the iteration-0 norm): the whole-solve hold must
              be untouched -- the first 16 columns and the counters identical
              to the STICKY arm (hysteresis 30), the plotfile identical, and
              hold_releases == 0 on every row.

A sabotage that inverts the bound test (release below the bound) fails the
release arm (nothing released: identical to sticky, not to the control) and
the hold arm (released: identical to the control, not to sticky); one that
skips the recount after the release leaves stale pinned counts in column 11.

Usage: analysis_mhd_active_set_debt_bound.py <release|hold> <final plotfile>
"""

import sys

import numpy as np
import yt

yt.set_log_level(50)

REFERENCE = {
    "release": "../test_rz_theta_implicit_mhd_wall_active_set_churn",
    "hold": "../test_rz_theta_implicit_mhd_wall_active_set_sticky",
}
NUM_COLUMNS = 24
NUM_SOLVER_COLUMNS = 16  # step .. max_bound_excess: the arithmetic of the solve
COL_ENTRANTS, COL_RELEASED, COL_HELD, COL_RESOLVES, COL_DAMPED, COL_REJECTED, COL_MOVED, COL_HOLD_RELEASES = range(16, 24)
MAX_STEP = 16
FLUID_FIELDS = [
    "implicit_mhd_mass_density",
    "implicit_mhd_electron_energy",
    "implicit_mhd_ion_energy",
    "implicit_mhd_ion_internal_energy",
    "implicit_mhd_momentum_z",
]


def load_newton_tokens(directory):
    # newton.txt APPENDS across reruns of a test directory; keep the most
    # recent session (the rows after the last step reset), as string tokens
    # so the comparison is exact (no float parsing).
    sessions, current = [], []
    for line in open(f"{directory}/diags/newton.txt"):
        if line.startswith("#") or not line.strip():
            continue
        tokens = line.split()
        if current and int(tokens[0]) <= int(current[-1][0]):
            sessions.append(current)
            current = []
        current.append(tokens)
    sessions.append(current)
    rows = sessions[-1]
    widths = {len(r) for r in rows}
    assert widths == {NUM_COLUMNS}, (
        f"{directory}/diags/newton.txt (last session) has columns {sorted(widths)}, expected {NUM_COLUMNS}"
    )
    assert int(rows[-1][0]) == MAX_STEP, f"{directory}: run stopped at step {rows[-1][0]} of {MAX_STEP}"
    return rows


def get_fields(plotfile):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    return {f: np.array(grid["boxlib", f]).ravel() for f in FLUID_FIELDS}


mode, final_plotfile = sys.argv[1], sys.argv[2]
assert mode in REFERENCE, mode
reference_directory = REFERENCE[mode]

mine = load_newton_tokens(".")
reference = load_newton_tokens(reference_directory)
assert len(mine) == len(reference), (len(mine), len(reference))

# Exactness of the solve: the first 16 columns token-identical on every row.
for row_mine, row_reference in zip(mine, reference):
    assert row_mine[:NUM_SOLVER_COLUMNS] == row_reference[:NUM_SOLVER_COLUMNS], (
        f"step {row_mine[0]}: solver columns differ from {reference_directory}:\n"
        f"  mine {row_mine[:NUM_SOLVER_COLUMNS]}\n  ref  {row_reference[:NUM_SOLVER_COLUMNS]}"
    )
print(f"{mode}: the first {NUM_SOLVER_COLUMNS} newton.txt columns are identical to {reference_directory} on all {len(mine)} rows")

counters = np.array([[float(v) for v in r[NUM_SOLVER_COLUMNS:]] for r in mine])
reference_counters = np.array([[float(v) for v in r[NUM_SOLVER_COLUMNS:]] for r in reference])
hold_releases = int(counters[:, COL_HOLD_RELEASES - NUM_SOLVER_COLUMNS].sum())
held = int(counters[:, COL_HELD - NUM_SOLVER_COLUMNS].sum())
entrants = int(counters[:, COL_ENTRANTS - NUM_SOLVER_COLUMNS].sum())
released = int(counters[:, COL_RELEASED - NUM_SOLVER_COLUMNS].sum())
print(
    f"{mode}: entrants {entrants}, released {released}, held {held}, hold releases {hold_releases}; "
    f"reference entrants {int(reference_counters[:, 0].sum())}, released {int(reference_counters[:, 1].sum())}, "
    f"held {int(reference_counters[:, 2].sum())}"
)
if mode == "release":
    # Every held member is released again before it can act: nothing stays
    # held, and the plain rule's entrants/releases are reproduced. Measured:
    # the whole-solve hold holds members in 7 of the 16 solves (the churn's
    # releases at iterations 1-2), so the release pass fires 7 times.
    assert hold_releases >= 5, f"the release pass fired in only {hold_releases} solves"
    assert held == 0, f"{held} members stayed held after the release pass"
    assert entrants == int(reference_counters[:, 0].sum()), (entrants, reference_counters[:, 0].sum())
    assert released == int(reference_counters[:, 1].sum()), (released, reference_counters[:, 1].sum())
else:
    assert hold_releases == 0, f"{hold_releases} release passes with the bound out of reach"
    # The whole-solve hold's own counters are unchanged.
    for column in range(0, COL_HOLD_RELEASES - NUM_SOLVER_COLUMNS):
        assert np.array_equal(counters[:, column], reference_counters[:, column]), (
            f"counter column {NUM_SOLVER_COLUMNS + column} differs from the sticky arm"
        )

mine_fields = get_fields(final_plotfile)
reference_fields = get_fields(f"{reference_directory}/{final_plotfile}")
for name in FLUID_FIELDS:
    identical = np.array_equal(mine_fields[name], reference_fields[name])
    print(f"{name:40s} identical to {reference_directory}: {identical}")
    assert identical, name
print(f"debt-bounded hold ({mode}): exact PASS")
