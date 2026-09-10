#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# Half-window twin of the dual-energy guard test (guard = P0): most cells sit
# inside the C^1 window, so fk_eff = w x fk_kin is a PARTIAL blend. With the
# step-31 dump as the step-old input of the step-32 rewrite: (i) the rewrite
# identity E_i = KE + p_blend(w)/(gamma - 1) holds to round-off in every cell
# with the python window replica; (ii) the ledger's guarded-cell count equals
# the number of cells with w(U_i(31)) < 1, strictly between 0 and nz; (iii)
# the domain energy budget closes on the ledger; (iv) the adiabat is held.
#
# Usage: analysis_mhd_dual_energy_guard_half.py <diag000032> <diag000031> <diag000000> <ledger>

import sys

import numpy as np
import warpx_constants as constants
from analysis_mhd_dual_energy_common import (
    blended_pressure_guarded,
    load_fluid_state,
    window,
)

gamma_i = 5.0 / 3.0
Ti0_eV = 10.0
n0 = 1.0e20
nz = 128
max_step = 32
rho0 = n0 * constants.proton_mass
P0 = n0 * Ti0_eV * constants.elementary_charge
pressure_floor = 1.0e-6 * P0
guard = 1.0 * P0

final = load_fluid_state(sys.argv[1])
previous = load_fluid_state(sys.argv[2])
initial = load_fluid_state(sys.argv[3])
ledger = np.loadtxt(sys.argv[4], ndmin=2)

E, K, U = final["ion_energy"], final["kinetic_energy"], final["ion_internal_energy"]
U_old = previous["ion_internal_energy"]
w = window(U_old, guard)
print(f"guard half: window at step 31: min {w.min():.4f} max {w.max():.4f}, cells with w < 1: {int((w < 1).sum())}, w = 0: {int((w <= 0).sum())}")
assert 0 < int((w < 1).sum()) < nz, "the half window must split the domain"
assert (w > 0).any() and ((w > 0) & (w < 1)).sum() > nz // 4, "most cells must be PARTIALLY guarded"

# (i) rewrite identity with the partial blend. The kinetic fraction inside the
# rewrite is evaluated with the PRE-rewrite E_i, which is not dumped, so the
# identity is checked as a per-cell fixed point: recover E_pre from
# E_post = KE + p_blend(E_pre, KE, U, U_old)/(gamma - 1) (vectorised Newton),
# then require (a) the map to be solvable to round-off in every cell and
# (b) the ledger's last-step increment to equal sum_{w<1} (E_pre - E_post) dz
# -- the guarded discard the solver booked at step 32. A hard-step window, a
# window-blind Enzo overwrite, a sign flip or a doubled ledger scale all
# break (b) at the 1e-3..1e0 level against a 1e-9 gate.
def rewrite_map(E_pre):
    return K + blended_pressure_guarded(E_pre, K, U, U_old, gamma_i, pressure_floor, guard) / (gamma_i - 1.0) - E
E_pre = E.copy()
for _ in range(60):
    f = rewrite_map(E_pre)
    h = 1.0e-7 * np.maximum(np.abs(E_pre), 1.0)
    df = (rewrite_map(E_pre + h) - rewrite_map(E_pre - h)) / (2.0 * h)
    step = np.where(np.abs(df) > 1e-300, f / df, 0.0)
    E_pre = E_pre - step
    if np.max(np.abs(step) / np.maximum(np.abs(E_pre), 1e-300)) < 1.0e-15:
        break
residual = np.abs(rewrite_map(E_pre)) / np.maximum(np.abs(E), 1e-300)
print(f"guard half: fixed-point residual of the rewrite map max {residual.max():.3e}; E_pre - E_post: min {np.min(E_pre - E):.3e} max {np.max(E_pre - E):.3e}")
assert residual.max() < 1.0e-12, "rewrite map not solvable to round-off"
dz = 1.0 / nz
guarded_mask = w < 1.0
step_discard = float(np.sum((E_pre - E)[guarded_mask]) * dz)
ledger_increment = float(ledger[-1, 2] - ledger[-2, 2])
print(f"guard half: step-32 discard from the dumps {step_discard:.9e} vs ledger increment {ledger_increment:.9e} J/m^2 "
      f"(rel {abs(step_discard - ledger_increment) / abs(ledger_increment):.3e}); unguarded cells' E_pre - E_post max {np.abs((E_pre - E)[~guarded_mask]).max() if (~guarded_mask).any() else 0.0:.3e}")
np.testing.assert_allclose(step_discard, ledger_increment, rtol=1.0e-9)

# (ii) ledger cell count = cells with w < 1 at the last step
assert ledger.shape == (max_step, 3), ledger.shape
print(f"guard half: ledger last row {ledger[-1]}; cells with w < 1: {int((w < 1).sum())}")
assert int(ledger[-1, 1]) == int((w < 1).sum()), "ledger guarded-cell count must equal the window's"

# (iii) value gate, single step: in this periodic box sum(E_i + U_e) changes ONLY
# through the mixmaster rewrite (the E_i fluxes telescope, the electron pdV
# pair cancels cell by cell, the viscous stress work is a face flux), so the
# step-32 budget closes on the fixed-point discard of ALL cells: the guarded
# part is the ledger increment (checked above), the rest is the base code's
# rewrite drain in unguarded cells (not the guard's, not booked by design).
budget_step = (np.sum(E + final["electron_energy"]) - np.sum(previous["ion_energy"] + previous["electron_energy"])) * dz
total_discard = float(np.sum(E_pre - E) * dz)
print(f"guard half: step-32 energy change {budget_step:.9e} vs -total rewrite discard {-total_discard:.9e} J/m^2 "
      f"(rel {abs(budget_step + total_discard) / abs(total_discard):.3e}); guarded share {step_discard / total_discard:.4f}, "
      f"unguarded (base-code drain) {1.0 - step_discard / total_discard:.4f}")
np.testing.assert_allclose(budget_step, -total_discard, rtol=1.0e-9)
# and the multi-step ledger is a strict lower bound of the whole rewrite drain
budget = (np.sum(E + final["electron_energy"]) - np.sum(initial["ion_energy"] + initial["electron_energy"])) * dz
print(f"guard half: 0-32 energy change {budget:.9e} vs -ledger {-ledger[-1, 2]:.9e} J/m^2 (ledger = guarded part only)")
assert -budget > ledger[-1, 2] > 0.0

# (iv) adiabat of the pure-internal part (U_i) within the dual run's tolerance
compression = final["rho"].max() / rho0
assert compression > 1.2, "stagnation flow failed to compress"
adiabat_error = np.abs((gamma_i - 1.0) * U / final["rho"] ** gamma_i / (P0 / rho0**gamma_i) - 1.0)
print(f"guard half: max adiabat error (U_i) = {adiabat_error.max():.4e}")
assert adiabat_error.max() < 0.08

print("dual_energy guard half: PASS")
