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

Three modes, one script (the control is a dependency of the other two):
  control  -- newton.active_set = 1 with the plain release rule and
              newton.globalization_diagnostics = 1 (the seven counter
              columns recorded, no arithmetic change). Asserts the churn is
              present: entrants and releases every step, releases tracking
              the entrants (the released components are the ones re-clamped),
              nothing held, no re-solve, no member moved onto its bound after
              iteration 0, every solve converged, the reduced-space
              self-checks clean.
  sticky   -- newton.active_set_hysteresis = 30 (>= newton.max_iterations:
              a member is held for the rest of the solve; THE PRODUCTION
              CANDIDATE, measured on the formation deck at 17 us). Asserts
              the hold engaged and did NOT outlive the solve: held within
              [100, 224] (measured 160; the hold applied at iteration 0 --
              the one invariant the design relies on, iteration 0 rebuilds
              the set from the state -- gives 288), entrants and releases cut
              by at least 32 but by at most 30% (measured 384 -> 304 and
              368 -> 288; the same sabotage gives 176 / 160), nothing moved
              onto a bound after iteration 0 (a member held while off its
              bound would be moved down here), no re-solve, every solve
              converged, the self-checks clean, Newton and linear work not
              above the control, and the converged physics within 1e-5
              relative L2 of the control on every fluid block (measured
              1.3e-6 .. 3.1e-6: the hold keeps members whose residual has
              turned inward on their bound for the rest of the solve, which
              changes the converged state at the deck's tolerance class).
  resolve  -- newton.active_set_hysteresis = 1 and newton.line_search_resolve
              = 1 (the mechanics arm of the re-solve knob, NOT the production
              candidate). Asserts the one-identification hold engaged (held
              within [32, 128]; measured 96), the churn cut, no re-solve fired
              and no status-5 exit occurred (this deck has no damped step:
              the re-solve and its exit are exercised only by the production
              twin), and the physics equal to the control to 1e-7 (measured
              4e-10: a one-identification hold does not change the converged
              state).

newton.txt columns (active-set layout + the eight globalization counters):
  [2] iters [5] norm_rel [6] gmres [11] num_pinned [12] free_norm_rel
  [13] exit_status [14] max_pinned_direction [15] max_bound_excess
  [16] entrants [17] released [18] held [19] resolves [20] damped_steps
  [21] rejected_trials [22] moved [23] hold_releases

Usage: analysis_mhd_wall_active_set_churn.py <control|sticky|resolve> <final plotfile>
"""

import sys

import numpy as np
import yt

control_directory = "../test_rz_theta_implicit_mhd_wall_active_set_churn"

MAX_STEP = 16
NUM_COLUMNS = 24
COL_ITERS, COL_NORM_REL, COL_GMRES, COL_PINNED = 2, 5, 6, 11
COL_FREE_REL, COL_STATUS, COL_LEAK, COL_EXCESS = 12, 13, 14, 15
COL_ENTRANTS, COL_RELEASED, COL_HELD, COL_RESOLVES, COL_DAMPED, COL_REJECTED, COL_MOVED = range(16, 23)
# The debt-bounded hold's release passes (newton.active_set_hold_defect_bound,
# unset in every arm here): must stay 0.
COL_HOLD_RELEASES = 23
CONVERGED_STATUSES = {2, 3, 4}
STATUS_UNION_EXIT = 5
# Measured on the control (16 steps, 2 ranks): 16 or 32 components released
# and re-clamped at iterations 0-2 of the churning solves; totals 384
# entrants / 368 released over the run, 15 of 16 solves pinned, Newton 52,
# GMRES 11451. The gates leave a ~30% margin on the totals.
MIN_CONTROL_ENTRANTS = 260
MIN_CONTROL_RELEASED = 250
MIN_PINNED_SOLVES = 12
MIN_PINNED_COMPONENTS = 16
# Two-sided gates. The one-identification hold (resolve arm) holds one wall
# row per churning identification: measured held 96, entrants 304 (-80),
# released 288 (-80), Newton 49, GMRES 11139. The whole-solve hold (sticky
# arm): held 160, entrants 304, released 288, Newton 48, GMRES 10989. A hold
# that outlives the solve (the iteration-0 reset removed) gives held 224 /
# 288 and entrants 176 / released 160 on the two arms: outside every bound
# below.
MIN_CHURN_CUT = 32
MAX_CHURN_CUT_FRACTION = 0.30
HELD_BOUNDS = {"sticky": (100, 224), "resolve": (32, 128)}
PHYSICS_TOLERANCE = {"sticky": 1.0e-5, "resolve": 1.0e-7}
FLUID_FIELDS = [
    "implicit_mhd_mass_density",
    "implicit_mhd_electron_energy",
    "implicit_mhd_ion_energy",
    "implicit_mhd_ion_internal_energy",
    "implicit_mhd_momentum_z",
]


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
    rows = sessions[-1]
    widths = {len(r) for r in rows}
    assert widths == {NUM_COLUMNS}, (
        f"{directory}/diags/newton.txt (last session) has columns {sorted(widths)}, expected "
        f"{NUM_COLUMNS} (active-set layout + globalization counters)"
    )
    return np.array(rows)


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
        f"{label}: non-converged exits present, statuses {sorted(int(x) for x in statuses)}"
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
        f"rejected trials {int(rows[:, COL_REJECTED].sum())}, moved {int(rows[:, COL_MOVED].sum())}, "
        f"statuses {sorted(int(x) for x in statuses)}"
    )


mode, final_plotfile = sys.argv[1], sys.argv[2]
assert mode in ("control", "sticky", "resolve"), mode

rows = load_newton(".")
common_checks(rows, mode)
entrants = int(rows[:, COL_ENTRANTS].sum())
released = int(rows[:, COL_RELEASED].sum())
held = int(rows[:, COL_HELD].sum())
resolves = int(rows[:, COL_RESOLVES].sum())
moved = int(rows[:, COL_MOVED].sum())
# Every arm: nothing is moved onto its bound after iteration 0 on this deck
# (the projection lands entrants exactly at the landing point and no step is
# damped). A member held while OFF its bound would be moved down by the
# identification and appear here: the witness of the hold's bound-resident
# guard (max_bound_excess is measured after the move and cannot see it).
assert moved == 0, f"{moved} components were moved onto their bounds: a member was held off its bound"
# The debt-bounded hold is unset in every arm here: its release pass must
# never run (the arms with the bound have their own gate,
# analysis_mhd_active_set_debt_bound.py).
hold_releases = int(rows[:, COL_HOLD_RELEASES].sum())
assert hold_releases == 0, f"{hold_releases} hold release passes with newton.active_set_hold_defect_bound unset"

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
    held_lo, held_hi = HELD_BOUNDS[mode]
    assert held >= held_lo, f"the hysteresis never engaged (held {held} < {held_lo})"
    assert held <= held_hi, (
        f"held {held} > {held_hi}: the hold outlived the solve (iteration 0 must rebuild the set "
        "from the state and reset the hold counters)"
    )
    assert entrants <= control_entrants - MIN_CHURN_CUT, (
        f"entrants {entrants} vs control {control_entrants}: the re-clamps were not cut"
    )
    assert released <= control_released - MIN_CHURN_CUT, (
        f"releases {released} vs control {control_released}: the releases were not cut"
    )
    assert entrants >= (1.0 - MAX_CHURN_CUT_FRACTION) * control_entrants, (
        f"entrants {entrants} vs control {control_entrants}: cut by more than "
        f"{MAX_CHURN_CUT_FRACTION:.0%} -- members are being held across solves"
    )
    assert released >= (1.0 - MAX_CHURN_CUT_FRACTION) * control_released, (
        f"releases {released} vs control {control_released}: cut by more than "
        f"{MAX_CHURN_CUT_FRACTION:.0%} -- members are being held across solves"
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
    union_exits = int(np.sum(rows[:, COL_STATUS].astype(int) == STATUS_UNION_EXIT))
    assert union_exits == 0, (
        f"{union_exits} status-5 exits: the union-set exit needs a failed ladder, "
        "which this deck does not produce"
    )
    # Converged physics: a one-identification hold changes the path only;
    # the whole-solve hold changes the converged state at the deck's
    # tolerance class (members held on their bound while their residual
    # points inward).
    mine = get_fields(final_plotfile)
    theirs = get_fields(f"{control_directory}/diags/diag{MAX_STEP:06d}")
    for f in FLUID_FIELDS:
        scale = np.linalg.norm(theirs[f])
        diff = np.linalg.norm(mine[f] - theirs[f]) / (scale if scale > 0 else 1.0)
        print(f"{mode}: {f} relative L2 difference vs control {diff:.3e}")
        assert diff <= PHYSICS_TOLERANCE[mode], (
            f"{f} differs from the control by {diff:.3e} (> {PHYSICS_TOLERANCE[mode]:.0e})"
        )
    print(
        f"{mode}: held {held}, entrants {entrants} (control {control_entrants}), "
        f"released {released} (control {control_released}), Newton "
        f"{int(rows[:, COL_ITERS].sum())} (control {int(control[:, COL_ITERS].sum())}), "
        f"GMRES {int(rows[:, COL_GMRES].sum())} (control {int(control[:, COL_GMRES].sum())})"
    )
print("PASS")
