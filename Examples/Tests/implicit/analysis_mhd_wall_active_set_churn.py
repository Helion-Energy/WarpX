#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Active-set release hysteresis on the engineered wall-row churn deck.

Deck: inputs_test_rz_theta_implicit_mhd_wall_active_set_churn -- a 100 eV
plasma against a 2 eV conducting z-end wall with an ion temperature floor
(50 eV) between the two, Braginskii conduction at a diffusion number of 10
per step, the dual-energy ion closure, and the reduced-space (active-set)
Newton. The wall drain pins the wall-adjacent ion internal energy every
step; with the plain release rule the identification drops the pinned wall
row at iterations 0-2 of most solves (its residual points inward at the
new time level and after the neighbours' update -- the rows are not
diagonally dominant) and the same iteration's free direction drives it
back through the bound, where the projection re-clamps it: the set is
re-formed while the Newton direction is being computed, which is the
production formation deck's 14-17 us mechanism (170 releases and 169
re-entrants per step there).

Two modes, one script:
  control     -- newton.active_set = 1 with the plain release rule and
                 newton.globalization_diagnostics = 1 (the six counter
                 columns recorded, no arithmetic change). Asserts the churn
                 is present: entrants and releases every step, releases
                 tracking the entrants (the released components are the ones
                 re-clamped), no held members, no re-solves, every solve
                 converged, the reduced-space self-checks clean.
  hysteresis  -- newton.active_set_hysteresis = 1 and
                 newton.line_search_resolve = 1 (the production candidate).
                 Asserts the rule engaged (held > 0), the churn dropped
                 (fewer entrants and releases than the control by at least
                 the hold count of one wall row per churning solve), the
                 Newton and linear work did not grow, every solve converged,
                 the self-checks stay clean, the entrant re-solve was never
                 needed on this deck (no damped step: resolves == 0 -- the
                 re-solve is measured on the production twin), and the
                 converged physics agrees with the control to the solver
                 tolerance class (relative L2 of every fluid block <= 1e-7;
                 measured 4e-10).

newton.txt columns (active-set layout + the six globalization counters):
  [2] iters [5] norm_rel [6] gmres [11] num_pinned [12] free_norm_rel
  [13] exit_status [14] max_pinned_direction [15] max_bound_excess
  [16] entrants [17] released [18] held [19] resolves [20] damped_steps
  [21] rejected_trials

Usage: analysis_mhd_wall_active_set_churn.py <control|hysteresis> <final plotfile>
"""

import sys

import numpy as np
import yt

control_directory = "../test_rz_theta_implicit_mhd_wall_active_set_churn"

MAX_STEP = 16
NUM_COLUMNS = 22
COL_ITERS, COL_NORM_REL, COL_GMRES, COL_PINNED = 2, 5, 6, 11
COL_FREE_REL, COL_STATUS, COL_LEAK, COL_EXCESS = 12, 13, 14, 15
COL_ENTRANTS, COL_RELEASED, COL_HELD, COL_RESOLVES, COL_DAMPED, COL_REJECTED = range(16, 22)
CONVERGED_STATUSES = {2, 3, 4}
# Measured on the control (16 steps, 2 ranks): 16 or 32 components released
# and re-clamped at iterations 0-2 of the churning solves; totals 384
# entrants / 368 released over the run, 15 of 16 solves pinned, Newton 52,
# GMRES 11451. The gates leave a ~30% margin on the totals.
MIN_CONTROL_ENTRANTS = 260
MIN_CONTROL_RELEASED = 250
MIN_PINNED_SOLVES = 12
MIN_PINNED_COMPONENTS = 16
# The hysteresis run holds one wall row (16 components) per churning
# identification: measured held 96, entrants 304 (-80), released 288 (-80),
# Newton 49, GMRES 11139.
MIN_HELD = 16
MIN_CHURN_CUT = 32
PHYSICS_TOLERANCE = 1.0e-7
FLUID_FIELDS = [
    "implicit_mhd_mass_density",
    "implicit_mhd_electron_energy",
    "implicit_mhd_ion_energy",
    "implicit_mhd_ion_internal_energy",
    "implicit_mhd_momentum_z",
]


def last_session(rows, step_col=0):
    # newton.txt APPENDS across reruns of a test directory; keep the most
    # recent session.
    resets = np.nonzero(np.diff(rows[:, step_col]) < 0)[0]
    return rows[(resets[-1] + 1) if len(resets) else 0 :]


def load_newton(directory):
    rows = np.loadtxt(f"{directory}/diags/newton.txt", comments="#", ndmin=2)
    rows = last_session(rows)
    assert rows.shape[1] == NUM_COLUMNS, (
        f"{directory}/diags/newton.txt has {rows.shape[1]} columns, expected "
        f"{NUM_COLUMNS} (active-set layout + globalization counters)"
    )
    return rows


def get_fields(plotfile):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return {f: np.array(grid["boxlib", f]).ravel() for f in FLUID_FIELDS}


def common_checks(rows, label):
    steps = rows[:, 0].astype(int)
    assert steps[-1] == MAX_STEP, f"{label}: run stopped at step {steps[-1]} of {MAX_STEP}"
    statuses = set(rows[:, COL_STATUS].astype(int))
    assert statuses <= CONVERGED_STATUSES, (
        f"{label}: non-converged exits present, statuses {sorted(statuses)}"
    )
    assert np.all(rows[:, COL_LEAK] == 0.0), (
        f"{label}: the Newton direction leaks onto the active set "
        f"(max {rows[:, COL_LEAK].max():.3e})"
    )
    assert np.all(rows[:, COL_EXCESS] <= 0.5), (
        f"{label}: a pinned component sat {rows[:, COL_EXCESS].max():.2f} margins "
        "above its bound after the move"
    )
    pinned_solves = int(np.sum(rows[:, COL_PINNED] > 0))
    assert pinned_solves >= MIN_PINNED_SOLVES, (
        f"{label}: the wall row pinned in only {pinned_solves} solves"
    )
    assert rows[:, COL_PINNED].max() >= MIN_PINNED_COMPONENTS, (
        f"{label}: at most {int(rows[:, COL_PINNED].max())} pinned components"
    )
    print(
        f"{label}: {len(rows)} steps, Newton {int(rows[:, COL_ITERS].sum())}, "
        f"GMRES {int(rows[:, COL_GMRES].sum())}, pinned solves {pinned_solves} "
        f"(max {int(rows[:, COL_PINNED].max())}), entrants {int(rows[:, COL_ENTRANTS].sum())}, "
        f"released {int(rows[:, COL_RELEASED].sum())}, held {int(rows[:, COL_HELD].sum())}, "
        f"re-solves {int(rows[:, COL_RESOLVES].sum())}, damped {int(rows[:, COL_DAMPED].sum())}, "
        f"rejected trials {int(rows[:, COL_REJECTED].sum())}, statuses {sorted(statuses)}"
    )


mode, final_plotfile = sys.argv[1], sys.argv[2]
assert mode in ("control", "hysteresis"), mode

rows = load_newton(".")
common_checks(rows, mode)
entrants = int(rows[:, COL_ENTRANTS].sum())
released = int(rows[:, COL_RELEASED].sum())
held = int(rows[:, COL_HELD].sum())
resolves = int(rows[:, COL_RESOLVES].sum())

if mode == "control":
    # The churn: the projection clamps the wall row into the set and the
    # identification releases it again, solve after solve.
    assert entrants >= MIN_CONTROL_ENTRANTS, f"too few entrants ({entrants}): the deck does not churn"
    assert released >= MIN_CONTROL_RELEASED, f"too few releases ({released}): the deck does not churn"
    assert released >= 0.9 * entrants, (
        f"releases ({released}) do not track the entrants ({entrants}): the "
        "re-clamped components are not the released ones"
    )
    assert held == 0, f"the plain rule must hold nothing (held {held})"
    assert resolves == 0, f"the plain rule must not re-solve (resolves {resolves})"
    print("control: churn present, plain rule inert as expected")
else:
    control = load_newton(control_directory)
    control_entrants = int(control[:, COL_ENTRANTS].sum())
    control_released = int(control[:, COL_RELEASED].sum())
    assert held >= MIN_HELD, f"the hysteresis never engaged (held {held})"
    assert entrants <= control_entrants - MIN_CHURN_CUT, (
        f"entrants {entrants} vs control {control_entrants}: the re-clamps were not cut"
    )
    assert released <= control_released - MIN_CHURN_CUT, (
        f"releases {released} vs control {control_released}: the releases were not cut"
    )
    assert rows[:, COL_ITERS].sum() <= control[:, COL_ITERS].sum(), (
        f"more Newton iterations than the control ({int(rows[:, COL_ITERS].sum())} vs "
        f"{int(control[:, COL_ITERS].sum())})"
    )
    assert rows[:, COL_GMRES].sum() <= 1.02 * control[:, COL_GMRES].sum(), (
        f"more linear iterations than the control ({int(rows[:, COL_GMRES].sum())} vs "
        f"{int(control[:, COL_GMRES].sum())})"
    )
    assert resolves == 0, (
        f"the entrant re-solve fired {resolves} times: this deck has no damped "
        "step; the re-solve is measured on the production twin"
    )
    # Converged physics: the hold changes the path, not the converged state.
    mine = get_fields(final_plotfile)
    theirs = get_fields(f"{control_directory}/diags/diag{MAX_STEP:06d}")
    for f in FLUID_FIELDS:
        scale = np.linalg.norm(theirs[f])
        diff = np.linalg.norm(mine[f] - theirs[f]) / (scale if scale > 0 else 1.0)
        print(f"hysteresis: {f} relative L2 difference vs control {diff:.3e}")
        assert diff <= PHYSICS_TOLERANCE, f"{f} differs from the control by {diff:.3e}"
    print(
        f"hysteresis: held {held}, entrants {entrants} (control {control_entrants}), "
        f"released {released} (control {control_released}), Newton "
        f"{int(rows[:, COL_ITERS].sum())} (control {int(control[:, COL_ITERS].sum())}), "
        f"GMRES {int(rows[:, COL_GMRES].sum())} (control {int(control[:, COL_GMRES].sum())})"
    )
print("PASS")
