#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Polynomial (cubic) Newton line search on the oblique-edge Braginskii deck.

The oblique-edge deck (inputs_test_rz_theta_implicit_mhd_braginskii_oblique_edge)
is the suite's deck with genuine free-subspace damping: 13 of its 69 Newton
iterations (10 steps, plain projected Newton, no active set, nothing pinned)
accept a damped step after one rejected trial, one solve stagnates. It
therefore exercises the step-length rule without any active-set effect.

Two modes:
  control -- newton.line_search = backtrack (halving) with
             newton.globalization_diagnostics = 1: records the counters
             (damped steps, rejected trials) of the historical ladder without
             changing any arithmetic. Asserts the deck damps (damped >= 1).
  cubic   -- newton.line_search = cubic: the safeguarded quadratic/cubic
             model of the merit function 0.5||F||^2 through the rejected
             trials (Dennis & Schnabel 1983 sec. 6.3.2; Kelley 1995 sec.
             8.3.1), clipped to [0.1, 0.5] of the rejected step. Asserts the
             rule engaged and differs from halving (the per-step Newton or
             rejected-trial counts differ from the control in at least one
             step; a rule that degenerates to 0.5 x lambda, e.g. through a
             sign error in the slope that trips the fallback, reproduces the
             control exactly), spends no more rejected trials than the ladder,
             does not blow up the Newton or linear work (<= 1.3x the control:
             measured 77 vs 69 Newton iterations -- the polynomial rule
             accepts shorter steps than the ladder's 0.5 on this deck and
             needs more iterations; it is a mechanics option, not a speed
             lever, which is why it stays off by default), and reproduces the
             control's physics to 1e-4 relative L2 (measured 5e-6 on the
             electron energy, identical mass density).

newton.txt (plain layout + the seven globalization counters):
  [2] iters [6] gmres [12] entrants [13] released [14] held [15] resolves
  [16] damped_steps [17] rejected_trials [18] moved

Usage: analysis_mhd_line_search_polynomial.py <control|cubic> <final plotfile>
"""

import sys

import numpy as np
import yt

control_directory = "../test_rz_theta_implicit_mhd_braginskii_oblique_edge_line_search_control"

MAX_STEP = 10
NUM_COLUMNS = 19
COL_ITERS, COL_GMRES = 2, 6
COL_DAMPED, COL_REJECTED = 16, 17
WORK_RATIO = 1.3
PHYSICS_TOLERANCE = 1.0e-4
FIELDS = ["implicit_mhd_mass_density", "implicit_mhd_electron_energy", "Bz"]


def load_newton(directory):
    # newton.txt APPENDS across reruns of a test directory, and earlier
    # sessions may have a different column count: parse line by line and
    # keep the most recent session (the rows after the last step reset).
    sessions, current = [], []
    for line in open(f"{directory}/diags/newton.txt"):
        if line.startswith("#") or not line.strip():
            continue
        values = [float(v) for v in line.split()]
        if current and values[0] <= current[-1][0]:
            sessions.append(current)
            current = []
        current.append(values)
    sessions.append(current)
    rows = np.array(sessions[-1])
    assert rows.shape[1] == NUM_COLUMNS, (
        f"{directory}/diags/newton.txt (last session) has {rows.shape[1]} columns, expected {NUM_COLUMNS}"
    )
    assert int(rows[-1, 0]) == MAX_STEP, f"{directory}: run stopped at step {int(rows[-1, 0])}"
    return rows


def get_fields(plotfile):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    return {f: np.array(grid["boxlib", f]).ravel() for f in FIELDS}


mode, final_plotfile = sys.argv[1], sys.argv[2]
assert mode in ("control", "cubic"), mode
rows = load_newton(".")
damped, rejected = int(rows[:, COL_DAMPED].sum()), int(rows[:, COL_REJECTED].sum())
print(
    f"{mode}: Newton {int(rows[:, COL_ITERS].sum())}, GMRES {int(rows[:, COL_GMRES].sum())}, "
    f"damped steps {damped}, rejected trials {rejected}"
)
assert damped >= 1, f"{mode}: the deck did not damp; the line search is not exercised"
if mode == "cubic":
    control = load_newton(control_directory)
    differs = np.any(rows[:, COL_ITERS] != control[:, COL_ITERS]) or np.any(
        rows[:, COL_REJECTED] != control[:, COL_REJECTED]
    )
    assert differs, "the cubic rule reproduced the halving ladder exactly (degenerate step rule)"
    assert rejected <= int(control[:, COL_REJECTED].sum()), (
        f"more rejected trials than the halving ladder ({rejected} vs {int(control[:, COL_REJECTED].sum())})"
    )
    for col, name in ((COL_ITERS, "Newton"), (COL_GMRES, "linear")):
        assert rows[:, col].sum() <= WORK_RATIO * control[:, col].sum(), (
            f"{name} iterations {int(rows[:, col].sum())} exceed {WORK_RATIO} x the control "
            f"{int(control[:, col].sum())}"
        )
    mine = get_fields(final_plotfile)
    theirs = get_fields(f"{control_directory}/diags/diag{MAX_STEP:06d}")
    for f in FIELDS:
        scale = np.linalg.norm(theirs[f])
        diff = np.linalg.norm(mine[f] - theirs[f]) / (scale if scale > 0 else 1.0)
        print(f"cubic: {f} relative L2 difference vs control {diff:.3e}")
        assert diff <= PHYSICS_TOLERANCE, f"{f} differs from the control by {diff:.3e}"
print("PASS")
