#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Reduced-space (active-set) Newton on the engineered active-set deadlock.

The SAME deck as the active-set rescue test -- a pedestal-resident halo
band whose partnerless ion-energy sink is pinned at an ion temperature-
floor admissibility bound ~500x above its relaxation target -- run with
newton.active_set = 1 (plus a floor ledger file). The plain projected
Newton of the rescue baseline (a declared dependency; its newton.txt is
read from ../test_1d_theta_implicit_mhd_active_set_rescue) computes the
unconstrained direction, clamps the band afterwards and can only accept
free-subspace rescue steps: NO pinned solve ever converges (exit rel.
norm O(1) through the whole pinned window). The active-set solve
identifies the band at iteration 0 of every solve (on its bound, residual
demanding the sub-bound drain), holds it there, solves the Newton system
in the free subspace only (pinned rows/columns replaced by the identity,
right-hand side and preconditioner masked), converges on the free
residual (exit status 4) and books the drain the floor refused -- the
pinned defect divided by theta -- to the floor ledger.

Assertions (newton.txt, floor_ledger.txt, plotfiles, the baseline's
newton.txt and final plotfile):
  1. the run COMPLETED all its steps;
  2. the engineered active set formed: a window of >= 5 solves with
     pinned components (measured 11 here, 8 in the baseline, where the
     rescue steps let the band density drift and release the ratchet a
     little earlier), the band size (32) matching the baseline's;
  3. CONVERGENCE WHERE THE PLAIN SOLVE STAGNATES: every pinned solve
     (after the entry solve, in which the band is first clamped) exited
     with status 4 and free_norm_rel <= newton.active_set_tolerance (or
     the free norm below the absolute tolerance), in <= 3 Newton
     iterations (measured: 1, with 4-5 GMRES) -- while in the baseline no
     pinned solve came within 1e-3 of its initial residual (measured exit
     rel. norm 0.12 for the entry solve, 0.99999 afterwards);
  4. LEDGER CLOSURE: the supply columns are identically zero (no floor-
     consistency source), the pinned-energy column is non-decreasing,
     and its per-step increment while the whole band is pinned equals
     the analytic sink demand at the bound,
         dt * SINK_RATE * n_band * (E_bound - target) * dz,
     to 1% (measured 2.998e-5 J/m^2 per step against 2.998e-5 analytic;
     the theta stage sits one projection margin, 1e-6, above the bound
     and the band density drifts ~1e-4 with the sound wave);
  5. the pinned MASS column is exactly zero (no density bound engages);
  6. physics continuity with the rescue baseline: the band drains THROUGH
     the former bound once the temperature ratchet releases, the final
     solves converge to round-off, and the state keeps advancing.

Usage: analysis_mhd_active_set_newton.py <initial> <mid> <final plotfile>
"""

import sys

import numpy as np
import yt

baseline_directory = "../test_1d_theta_implicit_mhd_active_set_rescue"

MAX_STEP = 120
# Deck constants (inputs_test_1d_theta_implicit_mhd_active_set_rescue).
K_B = 1.380649e-23
M_P = 1.67262192369e-27
Q_E = 1.602176634e-19
N0 = 1.0e18
PEDESTAL_FRACTION = 1.0e-3
RHO_HALO = PEDESTAL_FRACTION * N0 * M_P
GAMMA = 5.0 / 3.0
T_FLOOR = 58022.0
SINK_RATE = 5.0e7
DT = 1.0e-9
NZ = 64
DZ = 1.0 / NZ
TE_EV = 0.01
# The core's internal energy P/(gamma - 1) (uniform pressures Pe = Pi =
# n0 Te) sets the pedestal relaxation target, PEDESTAL_FRACTION x peak.
E_PEAK = N0 * TE_EV * Q_E / (GAMMA - 1.0)
# The band's temperature-floor admissibility image (internal energy):
# n_halo kB T_floor / (gamma - 1) = 0.5 E_PEAK.
E_BOUND = (RHO_HALO / M_P) * K_B * T_FLOOR / (GAMMA - 1.0)
# Test-time knobs (the CMake inputs string).
ACTIVE_SET_TOLERANCE = 1.0e-5
ABSOLUTE_TOLERANCE = 1.0e-12
# newton.txt columns
COL_STEP, COL_ITERS, COL_NORM_ABS, COL_NORM_REL = 0, 2, 4, 5
COL_FREE_ABS, COL_DEFECT, COL_PINNED, COL_FREE_REL, COL_STATUS = 9, 10, 11, 12, 13
EXIT_FREE_SUBSPACE = 4


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


def last_session(rows, step_col=0):
    # newton.txt / floor_ledger.txt APPEND across reruns of a test
    # directory; keep only the most recent session.
    resets = np.nonzero(np.diff(rows[:, step_col]) < 0)[0]
    return rows[(resets[-1] + 1) if len(resets) else 0 :]


def band_internal(data):
    rho = data["boxlib", "implicit_mhd_mass_density"].value.ravel()
    ion_e = data["boxlib", "implicit_mhd_ion_energy"].value.ravel()
    mom = data["boxlib", "implicit_mhd_momentum_density"].value.ravel()
    band = rho < 0.5 * rho.max()
    return ion_e[band] - 0.5 * mom[band] ** 2 / rho[band]


initial_ds, initial = get_data(sys.argv[1])
mid_ds, mid = get_data(sys.argv[2])
final_ds, final = get_data(sys.argv[3])

history = last_session(np.atleast_2d(np.loadtxt("diags/newton.txt")))
assert history.shape[1] == 14, (
    "newton.txt must carry the active-set columns free_norm_rel and "
    f"exit_status (14 columns), found {history.shape[1]}"
)
steps = history[:, COL_STEP]
iters = history[:, COL_ITERS]
norm_abs = history[:, COL_NORM_ABS]
free_abs = history[:, COL_FREE_ABS]
num_pinned = history[:, COL_PINNED]
free_rel = history[:, COL_FREE_REL]
status = history[:, COL_STATUS]

ledger = last_session(np.atleast_2d(np.loadtxt("diags/floor_ledger.txt")))
assert ledger.shape[1] == 5, (
    f"floor_ledger.txt must carry 5 columns in the active-set mode, found {ledger.shape[1]}"
)

# 1. Completion.
assert steps[-1] == MAX_STEP, (
    f"run did not complete: newton.txt ends at step {steps[-1]:.0f} of {MAX_STEP}"
)
assert ledger.shape[0] == history.shape[0], (
    f"ledger rows ({ledger.shape[0]}) != solves ({history.shape[0]})"
)

# 2. The engineered active set formed, with the baseline's band size.
pinned = num_pinned > 0
band_size = int(num_pinned.max())
print(
    f"pinned window: {pinned.sum()} solves (steps {steps[pinned].min():.0f}-"
    f"{steps[pinned].max():.0f}), band size {band_size}"
)
assert pinned.sum() >= 5, "the engineered active set never formed"
baseline = last_session(
    np.atleast_2d(np.loadtxt(f"{baseline_directory}/diags/newton.txt"))
)
baseline_pinned = baseline[:, COL_PINNED] > 0
assert baseline_pinned.sum() >= 5, (
    "the rescue baseline lost its pinned window: this test discriminates nothing"
)
assert int(baseline[:, COL_PINNED].max()) == band_size, (
    f"band size {band_size} differs from the baseline's "
    f"{int(baseline[:, COL_PINNED].max())}"
)

# 3. Convergence where the plain solve stagnates. The entry solve (the
# first pinned one) starts with the band ABOVE its bound: the projection
# clamps it during the free solve and the set is only complete from the
# next iteration on, so it is graded separately (it must still exit
# converged on the free subspace within the iteration cap).
first_pinned = np.nonzero(pinned)[0][0]
settled = pinned.copy()
settled[first_pinned] = False
converged_free = (status == EXIT_FREE_SUBSPACE) & (
    (free_rel <= ACTIVE_SET_TOLERANCE) | (free_abs <= ABSOLUTE_TOLERANCE)
)
print(
    f"entry solve (step {steps[first_pinned]:.0f}): {iters[first_pinned]:.0f} "
    f"iterations, exit status {status[first_pinned]:.0f}, free rel. norm "
    f"{free_rel[first_pinned]:.3e}"
)
assert converged_free[first_pinned], (
    "the entry solve did not converge on the free subspace "
    f"(status {status[first_pinned]:.0f}, free rel {free_rel[first_pinned]:.3e})"
)
print(
    f"settled pinned solves: {settled.sum()}, iterations "
    f"{iters[settled].min():.0f}-{iters[settled].max():.0f}, exit statuses "
    f"{sorted(set(status[settled].astype(int)))}, max free rel. norm "
    f"{free_rel[settled].max():.3e}"
)
assert np.all(converged_free[settled]), (
    "a settled pinned solve did not converge on the free subspace: statuses "
    f"{sorted(set(status[settled].astype(int)))}, max free rel "
    f"{free_rel[settled].max():.3e}"
)
assert iters[settled].max() <= 3, (
    f"a settled pinned solve needed {iters[settled].max():.0f} Newton "
    "iterations (the reduced solve should converge in a few)"
)
baseline_rel = baseline[:, COL_NORM_REL][baseline_pinned]
print(
    f"baseline pinned solves: {baseline_pinned.sum()}, min exit rel. norm "
    f"{baseline_rel.min():.3e} (never within 1e-3)"
)
assert baseline_rel.min() > 1.0e-3, (
    "the plain projected Newton converged a pinned solve "
    f"(min rel {baseline_rel.min():.3e}): this test discriminates nothing"
)

# 4./5. Ledger closure.
supplied_mass, supplied_energy = ledger[:, 1], ledger[:, 2]
pinned_mass, pinned_energy = ledger[:, 3], ledger[:, 4]
assert np.all(supplied_mass == 0.0) and np.all(supplied_energy == 0.0), (
    "no floor-consistency source is armed: the supply columns must be zero"
)
assert np.all(pinned_mass == 0.0), (
    f"the pinned MASS column must be exactly zero (max {pinned_mass.max():.3e})"
)
increments = np.diff(pinned_energy, prepend=0.0)
assert np.all(increments >= 0.0), (
    f"the booked pinned energy decreased (min increment {increments.min():.3e})"
)
assert np.all(increments[~pinned] == 0.0), (
    "energy was booked in a solve without pinned components"
)
target = PEDESTAL_FRACTION * E_PEAK
analytic_step = DT * SINK_RATE * band_size * (E_BOUND - target) * DZ
full_band = num_pinned == band_size
full_band[first_pinned] = False
closure = np.abs(increments[full_band] - analytic_step) / analytic_step
print(
    f"ledger closure over {full_band.sum()} fully pinned solves: booked "
    f"{increments[full_band].mean():.6e} vs analytic {analytic_step:.6e} "
    f"J/m^2 per step (max relative gap {closure.max():.3e}); cumulative "
    f"booked energy {pinned_energy[-1]:.6e}"
)
assert full_band.sum() >= 5, "too few fully pinned solves for the closure"
assert closure.max() < 0.01, (
    "the booked pinned defect does not close against the analytic sink "
    f"demand at the bound (max relative gap {closure.max():.3e})"
)

# 6. Physics continuity with the rescue baseline.
assert norm_abs[-1] < 1.0e-6, (
    f"the pinned defect never resolved (final norm {norm_abs[-1]:.3e})"
)
band_initial = band_internal(initial)
band_final = band_internal(final)
print(
    f"band ion internal energy: initial {band_initial.mean():.3e} -> final "
    f"{band_final.mean():.3e} (bound image {E_BOUND:.3e})"
)
assert band_final.max() < 0.1 * band_initial.min(), (
    "the halo band never drained through the former temperature bound"
)
assert band_final.min() > 0.0, "band internal energy went negative"
mid_density = mid["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
drift = np.max(np.abs(final_density - mid_density)) / np.max(mid_density)
print(f"density drift between mid and final dumps: {drift:.3e} (relative)")
assert drift > 1.0e-7, "the state stopped advancing after the pinned window"

print("PASS")
