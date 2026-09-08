#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Temperature floors under the reduced-space (active-set) Newton.

The temperature-floor deck (a supersonic rarefaction cools the halo
below the 7500 K floor of both species; the density-dependent n kB
T_floor admissibility bound must hold them) run with newton.active_set =
1. Two gates:

  1. PHYSICS: every assertion of analysis_mhd_temperature_floor.py holds
     unchanged (both species at or above the floor, the run genuinely
     reached sub-floor adiabats, mass conserved) -- the pinned components
     rest on their bounds and the free solve does not disturb them;
  2. SOLVER: read against the default run (a declared dependency,
     ../test_1d_theta_implicit_mhd_temperature_floor/diags/newton.txt):
     the active-set run pins components (the floor engages: measured 164
     of 200 solves, 2-48 components), EVERY one of its solves exits
     converged (status 2, 3 or 4; measured 36 by the relative tolerance,
     164 on the free subspace) where the plain projected Newton returns
     unconverged stagnation exits (measured 164 of 200, mean exit rel.
     norm 4.7e-4 against the 1e-8 tolerance), within 4 Newton iterations
     per solve (measured 2-3), and its linear work per solve is not
     larger than 1.25x the plain solver's (measured 30.0 vs 32.3 GMRES
     iterations per solve; the plain solver's iteration counts are not a
     cost reference, since it gives up early).

Usage: analysis_mhd_temperature_floor_active_set.py <initial> <final plotfile>
"""

import os
import runpy
import sys

import numpy as np

baseline_directory = "../test_1d_theta_implicit_mhd_temperature_floor"
MAX_STEP = 200
RELATIVE_TOLERANCE = 1.0e-8  # newton.relative_tolerance of the deck
COL_STEP, COL_ITERS, COL_NORM_REL, COL_GMRES, COL_PINNED, COL_STATUS = 0, 2, 5, 6, 11, 13

# 1. Physics: the default test's assertions on this run's plotfiles.
runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "analysis_mhd_temperature_floor.py"),
    run_name="__main__",
)


def last_session(rows, step_col=0):
    resets = np.nonzero(np.diff(rows[:, step_col]) < 0)[0]
    return rows[(resets[-1] + 1) if len(resets) else 0 :]


# 2. Solver statistics against the plain projected Newton.
history = last_session(np.atleast_2d(np.loadtxt("diags/newton.txt")))
baseline = last_session(
    np.atleast_2d(np.loadtxt(f"{baseline_directory}/diags/newton.txt"))
)
assert history.shape[1] == 16, "newton.txt lacks the active-set columns"
assert np.all(history[:, 14] == 0.0), (
    f"the Newton direction leaks onto the active set (max {history[:, 14].max():.3e})"
)
pinned_rows = history[:, COL_PINNED] > 0
assert np.all(history[pinned_rows, 15] <= 0.5), (
    f"a pinned component sat {history[pinned_rows, 15].max():.2f} margins above its bound after the move"
)
assert history[-1, COL_STEP] == MAX_STEP and baseline[-1, COL_STEP] == MAX_STEP
pinned_solves = int(np.sum(history[:, COL_PINNED] > 0))
converged = np.isin(history[:, COL_STATUS], (2, 3, 4))
baseline_unconverged = int(np.sum(baseline[:, COL_NORM_REL] >= RELATIVE_TOLERANCE))
print(
    f"active set: {pinned_solves} of {history.shape[0]} solves pinned, "
    f"{converged.sum()} converged exits (statuses "
    f"{sorted(set(history[:, COL_STATUS].astype(int)))}), Newton "
    f"{history[:, COL_ITERS].mean():.2f}/solve, GMRES {history[:, COL_GMRES].mean():.1f}/solve"
)
print(
    f"baseline: {baseline_unconverged} unconverged exits, Newton "
    f"{baseline[:, COL_ITERS].mean():.2f}/solve, GMRES {baseline[:, COL_GMRES].mean():.1f}/solve"
)
assert pinned_solves > 0, "the temperature floor never pinned a component"
assert baseline_unconverged > 0, (
    "the plain projected Newton converged every solve: no discrimination"
)
assert np.all(converged), (
    "an active-set solve exited unconverged (statuses "
    f"{sorted(set(history[:, COL_STATUS].astype(int)))})"
)
assert history[:, COL_ITERS].max() <= 4, (
    f"an active-set solve needed {history[:, COL_ITERS].max():.0f} Newton iterations"
)
gmres_ratio = history[:, COL_GMRES].mean() / baseline[:, COL_GMRES].mean()
print(f"linear work ratio (active set / plain): {gmres_ratio:.3f}")
assert gmres_ratio <= 1.25, (
    "the active-set solve needs more than 1.25x the plain solver's GMRES "
    f"iterations per solve (ratio {gmres_ratio:.3f})"
)

print("PASS")
