#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Reduced-space (active-set) Newton on the engineered deadlock under the
DUAL-ENERGY ion closure.

The rescue deck (a pedestal-resident halo band whose partnerless ion-energy
sink is pinned at an ion temperature-floor bound) with
implicit_mhd.ion_closure = dual_energy and newton.active_set = 1. The ion
energy is now carried by two blocks -- the conserved total E_i (block 2)
and the auxiliary internal U_i (block 3), each with its own theta-image
bound -- so the 4-block masks, the E_i/U_i pair under the move and the
sync-weighted booking (E_i := KE + p_blend/(gamma-1) at the step end, so a
floor-held U_i reaches the conserved energy with weight 1 - f_k and an E_i
pin survives with weight f_k) are exercised here, on CPU, the way the
production population uses them.

Under dual_energy the band does NOT release. Entry phase (steps 14-~24):
both carriers pin (64 components = the 32 band cells in E_i and in U_i)
and the sync-weighted booking equals the full sink demand. Sustained
regime (to step 120): the auxiliary U_i stays pinned at its
temperature-floor image in all 32 cells while the CONSERVED E_i, whose
own ratchet disengages once a free step takes its internal content below
the temperature bound, drains through the bound toward the pedestal
target (band E_i - KE falls to 0.1 percent of the bound by step 120); only
~2 E_i components stay pinned. The weighted booking then books ~6 percent
of the raw U_i defect (f_k ~ 0.94 in the static band: the sync trusts the
freely draining E_i), which is the physically right answer -- little
energy is created because the conserved carrier is not held up -- and is
exactly the E_i/U_i distinction the booking rule exists for. The plain
projected Newton on the same deck (the declared dependency
test_1d_theta_implicit_mhd_active_set_dual_plain) never converges any of
its 107 pinned solves (stagnation exits at rel. norm 0.98-1.0 after 1
iteration).

Assertions (newton.txt, floor_ledger.txt, plotfiles, the plain run's
newton.txt from ../test_1d_theta_implicit_mhd_active_set_dual_plain):
  1. completion; the pinned regime is SUSTAINED (>= 100 pinned solves), the
     entry phase pins both carriers (64 components = 32 cells x {E_i, U_i})
     and the U_i band (>= 30 components) stays pinned in >= 95 percent of
     the pinned solves;
  2. every pinned solve exits converged on the free subspace (status 4,
     free_norm_rel <= tolerance) within 2 iterations, where the plain run
     converges none (every pinned solve above 1e-3 relative);
  3. self-checks: the Newton direction vanishes identically on the active
     set; no pinned component is left above its landing point;
  4. the ledger has 6 columns; the supply columns and the pinned mass are
     zero; in the entry phase the booked (sync-weighted) energy per fully
     pinned step matches the analytic sink demand at the bound to 2
     percent COUNTING EACH CELL ONCE although two components pin per cell
     (measured gap 1e-6) and the raw U_i column carries the same demand
     within 10 percent; in the sustained regime the raw U_i column still
     carries the full demand (within 10 percent) while the weighted
     booking is below 15 percent of it (measured 6 percent);
  5. the state keeps advancing (the free sound wave); at the end the band's
     U_i rides its bound image (within 2e-3) while the median band E_i
     internal content has drained below half the bound.

Usage: analysis_mhd_active_set_dual.py <initial> <mid> <final plotfile>
"""

import sys

import numpy as np
import yt

MAX_STEP = 120
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
E_PEAK = N0 * TE_EV * Q_E / (GAMMA - 1.0)
E_BOUND = (RHO_HALO / M_P) * K_B * T_FLOOR / (GAMMA - 1.0)
ACTIVE_SET_TOLERANCE = 1.0e-5
ABSOLUTE_TOLERANCE = 1.0e-12
COL_STEP, COL_ITERS, COL_NORM_ABS, COL_NORM_REL = 0, 2, 4, 5
COL_FREE_ABS, COL_DEFECT, COL_PINNED, COL_FREE_REL, COL_STATUS = 9, 10, 11, 12, 13
COL_LEAK, COL_EXCESS = 14, 15
EXIT_FREE_SUBSPACE = 4


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


def last_session(rows, step_col=0):
    resets = np.nonzero(np.diff(rows[:, step_col]) < 0)[0]
    return rows[(resets[-1] + 1) if len(resets) else 0 :]


def band_internal(data):
    rho = data["boxlib", "implicit_mhd_mass_density"].value.ravel()
    u_i = data["boxlib", "implicit_mhd_ion_internal_energy"].value.ravel()
    band = rho < 0.5 * rho.max()
    return u_i[band]


initial_ds, initial = get_data(sys.argv[1])
mid_ds, mid = get_data(sys.argv[2])
final_ds, final = get_data(sys.argv[3])

history = last_session(np.atleast_2d(np.loadtxt("diags/newton.txt")))
assert history.shape[1] == 16, f"newton.txt has {history.shape[1]} columns, expected 16"
steps = history[:, COL_STEP]
iters = history[:, COL_ITERS]
norm_abs = history[:, COL_NORM_ABS]
free_abs = history[:, COL_FREE_ABS]
num_pinned = history[:, COL_PINNED]
free_rel = history[:, COL_FREE_REL]
status = history[:, COL_STATUS]
leak = history[:, COL_LEAK]
excess = history[:, COL_EXCESS]
ledger = last_session(np.atleast_2d(np.loadtxt("diags/floor_ledger.txt")))
assert ledger.shape[1] == 6, f"floor_ledger.txt has {ledger.shape[1]} columns, expected 6"

# 1.
assert steps[-1] == MAX_STEP, f"run ended at step {steps[-1]:.0f} of {MAX_STEP}"
assert ledger.shape[0] == history.shape[0]
pinned = num_pinned > 0
band_size = int(num_pinned.max())
print(
    f"pinned window: {pinned.sum()} solves (steps {steps[pinned].min():.0f}-"
    f"{steps[pinned].max():.0f}), max pinned components {band_size}"
)
assert pinned.sum() >= 100, (
    f"the pinned regime is not sustained ({pinned.sum()} solves; measured 107)"
)
assert band_size == 64, (
    f"expected both carriers of the 32 band cells pinned (64 components), got {band_size}"
)
sustained = num_pinned[pinned] >= 30
assert np.mean(sustained) >= 0.95, (
    f"the U_i band left its bound in {100*(1-np.mean(sustained)):.1f}% of the pinned solves"
)
plain = last_session(
    np.atleast_2d(np.loadtxt("../test_1d_theta_implicit_mhd_active_set_dual_plain/diags/newton.txt"))
)
plain_pinned = plain[:, COL_PINNED] > 0
print(
    f"plain projected Newton: {plain_pinned.sum()} pinned solves, min exit rel. norm "
    f"{plain[plain_pinned, COL_NORM_REL].min():.3e}, iterations {plain[plain_pinned, COL_ITERS].min():.0f}-"
    f"{plain[plain_pinned, COL_ITERS].max():.0f}"
)
assert plain_pinned.sum() >= 100, "the plain run lost its pinned regime: no discrimination"
assert plain[plain_pinned, COL_NORM_REL].min() > 1.0e-3, (
    "the plain projected Newton converged a pinned solve on the dual-energy deck"
)

# 2.
first_pinned = np.nonzero(pinned)[0][0]
settled = pinned.copy()
settled[first_pinned] = False
converged_free = (status == EXIT_FREE_SUBSPACE) & (
    (free_rel <= ACTIVE_SET_TOLERANCE) | (free_abs <= ABSOLUTE_TOLERANCE)
)
print(
    f"entry solve (step {steps[first_pinned]:.0f}): {iters[first_pinned]:.0f} "
    f"iterations, status {status[first_pinned]:.0f}, free rel {free_rel[first_pinned]:.3e}; "
    f"settled: iterations {iters[settled].min():.0f}-{iters[settled].max():.0f}, statuses "
    f"{sorted(set(status[settled].astype(int)))}, max free rel {free_rel[settled].max():.3e}"
)
assert converged_free[first_pinned], "the entry solve did not converge on the free subspace"
assert np.all(converged_free[settled]), "a settled pinned solve did not converge on the free subspace"
assert iters[settled].max() <= 2, f"a settled pinned solve needed {iters[settled].max():.0f} iterations (measured 1-2)"

# 3.
assert np.all(leak == 0.0), f"direction leak onto the active set: max {leak.max():.3e}"
assert np.all(excess[pinned] <= 0.5), f"pinned component {excess[pinned].max():.2f} margins above its bound"

# 4.
supplied_mass, supplied_energy = ledger[:, 1], ledger[:, 2]
pinned_mass, pinned_energy, internal_raw = ledger[:, 3], ledger[:, 4], ledger[:, 5]
assert np.all(supplied_mass == 0.0) and np.all(supplied_energy == 0.0)
assert np.all(pinned_mass == 0.0), f"pinned mass booked ({pinned_mass.max():.3e})"
inc_energy = np.diff(pinned_energy, prepend=0.0)
inc_internal = np.diff(internal_raw, prepend=0.0)
assert np.all(inc_energy[~pinned] == 0.0) and np.all(inc_internal[~pinned] == 0.0)
assert np.all(inc_energy >= 0.0), f"booked energy decreased ({inc_energy.min():.3e})"
target = PEDESTAL_FRACTION * E_PEAK
per_cell_step = DT * SINK_RATE * (E_BOUND - target) * DZ
full_band = num_pinned == band_size
full_band[first_pinned] = False
assert full_band.sum() >= 3, "too few fully pinned solves for the closure"
# The band pins BOTH carriers of the same ion energy (E_i at its internal
# image and U_i at its own bound): the weighted booking must count the
# refused drain ONCE per cell -- the per-cell demand times the number of
# CELLS (half the pinned-component count when both blocks pin).
cells_per_solve = band_size / 2.0 if band_size % 2 == 0 else band_size
analytic_step = per_cell_step * cells_per_solve
closure = np.abs(inc_energy[full_band] - analytic_step) / analytic_step
print(
    f"ledger closure over {full_band.sum()} fully pinned solves ({band_size} pinned "
    f"components = {cells_per_solve:.0f} cells): booked {inc_energy[full_band].mean():.6e} vs "
    f"analytic {analytic_step:.6e} J/m^2 per step (max gap {closure.max():.3e}); raw U_i "
    f"increment {inc_internal[full_band].mean():.6e}; cumulative energy {pinned_energy[-1]:.6e}, "
    f"raw U_i {internal_raw[-1]:.6e}"
)
assert closure.max() < 0.02, f"weighted booking does not close against the sink demand (gap {closure.max():.3e})"
raw_ratio = inc_internal[full_band].mean() / inc_energy[full_band].mean()
assert 0.9 < raw_ratio < 1.1, (
    f"the raw U_i defect ({raw_ratio:.3f} of the weighted booking) should carry the same demand in the entry phase"
)
# Sustained regime: U_i alone holds the band; the conserved E_i drains.
sustained_rows = pinned & (num_pinned >= 30) & (num_pinned <= 40)
sustained_rows[first_pinned] = False
assert sustained_rows.sum() >= 50, f"too few sustained-regime solves ({sustained_rows.sum()})"
raw_sustained = inc_internal[sustained_rows].mean() / (per_cell_step * 32.0)
weighted_share = inc_energy[sustained_rows].mean() / inc_internal[sustained_rows].mean()
print(
    f"sustained regime ({sustained_rows.sum()} solves): raw U_i defect {raw_sustained:.3f} of the 32-cell demand, "
    f"weighted booking {weighted_share:.3f} of the raw U_i defect"
)
assert 0.9 < raw_sustained < 1.1, "the pinned U_i band should carry the full sink demand"
assert weighted_share < 0.15, (
    f"the weighted booking ({weighted_share:.3f} of raw U_i) should be small while the conserved E_i drains freely"
)

# 5.
band_initial = band_internal(initial)
band_final = band_internal(final)
rho_f = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
ei_f = final["boxlib", "implicit_mhd_ion_energy"].value.ravel()
mom_f = final["boxlib", "implicit_mhd_momentum_density"].value.ravel()
band_f = rho_f < 0.5 * rho_f.max()
ei_internal_final = (ei_f - 0.5 * mom_f**2 / rho_f)[band_f]
print(
    f"band U_i: initial {band_initial.mean():.3e} -> final {band_final.mean():.3e} (bound image {E_BOUND:.3e}); "
    f"band E_i - KE final median {np.median(ei_internal_final):.3e}"
)
assert np.all(band_final >= E_BOUND * (1.0 - 2.0e-3)) and np.all(band_final <= E_BOUND * (1.0 + 2.0e-3)), (
    "the band's U_i does not ride its temperature-floor bound under dual_energy"
)
assert np.median(ei_internal_final) < 0.5 * E_BOUND, (
    "the conserved E_i should drain through the bound while U_i is held (the sink acts on the free carrier)"
)
mid_density = mid["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
drift = np.max(np.abs(final_density - mid_density)) / np.max(mid_density)
print(f"density drift between mid and final dumps: {drift:.3e}")
assert drift > 1.0e-7, "the state stopped advancing"

print("PASS")
