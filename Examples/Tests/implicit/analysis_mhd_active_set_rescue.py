#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Active-set-aware Newton line-search rescue under an engineered deadlock.

The deck engineers the production (rr13d_C1) freeze mechanism: a
pedestal-resident halo band whose ion-energy relaxation -- a pure sink with
no conservation partner in the residual -- is pinned at an ion
TEMPERATURE-floor admissibility bound sitting ~500x above the relaxation
target. From the step the band hits the bound (~15), the projection clamps
the same 32 E_i direction components every solve, the constant drain demand
(pinned defect ~2.1e-2 in scaled residual units) dominates the residual
norm, and the small free subspace (a gentle sound wave, ~1e-6) cannot buy
the full-norm Armijo decrease: the line search stagnates at iteration 0.

WITHOUT the free-subspace rescue the state freezes bit-identically and the
run ABORTS after newton.max_frozen_steps = 20 consecutive frozen solves
(measured: "Newton made zero progress for newton.max_frozen_steps
consecutive solves" at step 33). WITH it, every pinned-regime solve still
accepts a free-subspace step (newton.txt logs iters >= 1 instead of the
frozen 0), the band density drifts with the wave, the one-way temperature
ratchet -- re-evaluated per solve from step-old values -- disengages, the
drain completes, and the solver returns to machine-precision convergence.

Assertions (newton.txt + plotfiles):
  1. the run COMPLETED all its steps (the unfixed solver aborts at ~33);
  2. the deadlock genuinely formed (defect plateau above 5e-3);
  3. every pinned-regime solve accepted an update (iters >= 1: an iter-0
     stagnation logs 0);
  4. the pinned defect stayed BOUNDED (max <= 1.5x the plateau onset);
  5. the defect RESOLVED once the ratchet released (final norm < 1e-6);
  6. the band's ion internal energy punched through the former bound;
  7. the state kept advancing after the pinned window (mid != final).

Usage: analysis_mhd_active_set_rescue.py <initial> <mid> <final plotfile>
"""

import sys

import numpy as np
import yt

MAX_STEP = 120
PLATEAU_THRESHOLD = 5.0e-3


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


initial_ds, initial = get_data(sys.argv[1])
mid_ds, mid = get_data(sys.argv[2])
final_ds, final = get_data(sys.argv[3])

history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
steps = history[:, 0]
iters = history[:, 2]
norm_abs = history[:, 4]

# 1. Completion: the unfixed solver aborts at ~step 33 with
# "Newton made zero progress for newton.max_frozen_steps consecutive
# solves"; the diagnostic file then never reaches MAX_STEP.
assert steps[-1] == MAX_STEP, (
    f"run did not complete: newton.txt ends at step {steps[-1]:.0f} of {MAX_STEP}"
)

# 2. The engineered active set genuinely formed: the pinned drain demand
# plateaus near 2.1e-2 (scaled). Without it this test discriminates
# nothing.
pinned = norm_abs > PLATEAU_THRESHOLD
assert np.any(pinned), (
    "the engineered deadlock never formed: no solve ended with residual "
    f"norm above {PLATEAU_THRESHOLD} (max {norm_abs.max():.3e})"
)
plateau_onset = norm_abs[pinned][0]
print(
    f"pinned window: {pinned.sum()} solves, onset defect "
    f"{plateau_onset:.6e}, max {norm_abs.max():.6e}, "
    f"final norm {norm_abs[-1]:.3e}"
)

# 3. Every pinned-regime solve still accepted an update: an iter-0
# stagnation exits with 0 iterations (the frozen-step path), a rescued
# solve with >= 1.
assert np.all(iters[pinned] >= 1), (
    "a pinned-regime solve made zero progress (iters == 0): the "
    "free-subspace rescue did not accept a step"
)

# 4. Bounded defect: the drain demand is constant (frozen per-solve
# pedestal target), so the defect must not grow. Calibrated: the plateau
# is flat to 5 digits; 1.5x is generous headroom for platform round-off.
assert norm_abs.max() <= 1.5 * plateau_onset, (
    f"pinned defect grew: max {norm_abs.max():.3e} vs onset "
    f"{plateau_onset:.3e}. If this is a genuine unbounded defect, the "
    "documented escalation is a floor-consistency relaxation source "
    "(not implemented)."
)

# 5. Resolution: the rescued free subspace lets the band density drift,
# the per-solve temperature ratchet disengages, and the drain completes
# (measured final norm ~8e-13).
assert norm_abs[-1] < 1.0e-6, (
    f"the pinned defect never resolved (final norm {norm_abs[-1]:.3e})"
)


# 6. The band drained THROUGH the former temperature bound (0.5x its
# initial internal energy): the deadlocked (unfixed) solver can never
# take it below the bound.
def band_internal(data):
    rho = data["boxlib", "implicit_mhd_mass_density"].value.ravel()
    ion_e = data["boxlib", "implicit_mhd_ion_energy"].value.ravel()
    mom = data["boxlib", "implicit_mhd_momentum_density"].value.ravel()
    band = rho < 0.5 * rho.max()
    kinetic = 0.5 * mom[band] ** 2 / rho[band]
    return ion_e[band] - kinetic


band_internal_initial = band_internal(initial)
band_internal_final = band_internal(final)
print(
    f"band ion internal energy: initial {band_internal_initial.mean():.3e}"
    f" -> final {band_internal_final.mean():.3e}"
)
assert band_internal_final.max() < 0.1 * band_internal_initial.min(), (
    "the halo band never drained through the former temperature bound"
)
assert band_internal_final.min() > 0.0, "band internal energy went negative"

# 7. The state kept advancing after the pinned window: the wave's density
# drift between the mid and final dumps (measured ~4e-5 relative) would
# be identically zero for a frozen state.
mid_density = mid["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
drift = np.max(np.abs(final_density - mid_density)) / np.max(mid_density)
print(f"density drift between mid and final dumps: {drift:.3e} (relative)")
assert drift > 1.0e-7, "the state stopped advancing after the pinned window"

print("PASS")
