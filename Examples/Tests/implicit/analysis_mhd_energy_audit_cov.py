#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Global energy audit with the change-of-variables pedestal on (periodic).

Usage: analysis_mhd_energy_audit_cov.py <energy_audit.txt> <halo_pedestal_ledger.txt>

The pure-background velocity-pulse deck (every cell on the background,
dual-energy ions, the sync on): nothing enters or leaves, the pedestal
ledger books nothing, and the change of variables drops the background's
work terms by construction. Checks:
1. Every column finite; boundary inflows, wall, eater and injection
   columns exactly zero; the pedestal ledger's injected energies are zero
   and the audit's ledger_pedestal_* columns agree with it.
2. The 'full' closure residual is at round-off on every step: the terms
   the change of variables removes are either species transfers (the
   electron pressure-work shift and its E_i counterpart cancel in the
   total, i.e. pw_pair is the only footprint) or enter through columns the
   audit measures (the U_i pdV shift through the dual-energy sync's
   rewrite of E_i), so nothing is left unbooked.
3. The background shares U_e_bg and U_i_bg are the constant background
   energies (nonzero, equal to the deck's whole electron/ion thermal
   content since every cell sits on the background) -- the audit knows
   what part of the totals is the static pedestal.
4. The dropped-work columns are recorded: cov_shift_pw_e and
   cov_shift_pdv_ui are finite and, with every cell on the background
   (D = 0, shift = -X_ped), they are the domain sums of the FULL pressure
   works (gamma - 1) U_ped div u the change of variables removes. For a
   UNIFORM background on a periodic line that sum is -p_ped x (net
   outflow) = 0: the background's work is a pure redistribution, so the
   columns must sit at the round-off of that cancellation (|.| < 1e-12 of
   the background energy) while the gated deposited works pw_e/pdv_ui
   vanish on the pure background. In an open or walled domain the same
   columns carry the net work the change of variables drops.
"""

import sys

import numpy as np


def read_audit(path):
    columns = None
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                if line.startswith("# columns:"):
                    columns = line[len("# columns:"):].split()
                continue
            values = line.split()
            if values:
                rows.append([float(v) for v in values])
    data = np.array(rows)
    return {name: data[:, i] for i, name in enumerate(columns)}


a = read_audit(sys.argv[1])
ledger = np.loadtxt(sys.argv[2], ndmin=2)
n = len(a["step"])
print(f"{sys.argv[1]}: {n} rows")

for name, values in a.items():
    assert np.all(np.isfinite(values)), f"non-finite column {name}"
for name in [
    "poynt_in", "fluid_in_e_rhi", "fluid_in_e_zlo", "fluid_in_e_zhi",
    "fluid_in_i_zlo", "fluid_in_i_zhi", "fluid_in_mass_zlo", "fluid_in_mass_zhi",
    "wall_e", "wall_i", "wall_mass", "eater_e", "eater_i", "inject_e", "inject_i", "inject_mass",
]:
    assert np.all(a[name] == 0.0), f"{name} must be exactly zero here: {a[name]}"

# 1. pedestal ledger: nothing injected (columns: step raised_cells mass energy_e energy_i)
assert np.all(ledger[:, 3] == 0.0) and np.all(ledger[:, 4] == 0.0), "the change of variables injected energy"
assert np.all(a["ledger_pedestal_e_cum"] == 0.0) and np.all(a["ledger_pedestal_i_cum"] == 0.0)

w_total = a["W_B_tot"] + a["U_e"] + a["E_i"]
scale = np.max(w_total)
tol = 1.0e-11 * scale

# 2. full closure at round-off
print(f"  max |resid_full| = {np.max(np.abs(a['resid_full'])):.3e} J (bound {tol:.3e}); "
      f"sync_Ei_cum = {a['sync_Ei_cum'][-1]:.3e} J; pw_pair_cum = {a['pw_pair_cum'][-1]:.3e} J")
assert np.all(np.abs(a["resid_full"]) < tol), "full closure residual above round-off with the CoV on"
for name in ["fluxw_mass", "fluxw_e", "fluxw_i", "fluxw_ui", "faraday_defect"]:
    assert np.all(np.abs(a[name]) < tol), f"{name} above round-off: {np.max(np.abs(a[name]))}"

# 3. background shares
print(f"  U_e = {a['U_e'][-1]:.6e} J, U_e_bg = {a['U_e_bg'][-1]:.6e} J; "
      f"U_i = {a['U_i'][-1]:.6e} J, U_i_bg = {a['U_i_bg'][-1]:.6e} J")
assert np.all(a["U_e_bg"] > 0.0) and np.all(a["U_i_bg"] > 0.0)
assert np.all(a["U_e_bg"] == a["U_e_bg"][0]) and np.all(a["U_i_bg"] == a["U_i_bg"][0]), "background shares must be static"
# every cell sits on the background: the totals ARE the background (to the
# sanitize slack and the pulse's viscous/kinetic bookkeeping, ~1e-6)
assert np.all(np.abs(a["U_e"] / a["U_e_bg"] - 1.0) < 1.0e-4), "electron total is not the background"
assert np.all(np.abs(a["U_i"] / a["U_i_bg"] - 1.0) < 1.0e-3), "ion internal total is not the background"

# 4. dropped works recorded
print(f"  cov_shift_pw_e: max |.| = {np.max(np.abs(a['cov_shift_pw_e'])):.3e} J; "
      f"cov_shift_pdv_ui: max |.| = {np.max(np.abs(a['cov_shift_pdv_ui'])):.3e} J; "
      f"pw_e max |.| = {np.max(np.abs(a['pw_e'])):.3e}, pdv_ui max |.| = {np.max(np.abs(a['pdv_ui'])):.3e}")
# a uniform background does no net work on a periodic line: the recorded
# domain sums are the round-off of that cancellation
assert np.max(np.abs(a["cov_shift_pw_e"])) < 1.0e-12 * a["U_e_bg"][0], "net electron background work on a periodic line"
assert np.max(np.abs(a["cov_shift_pdv_ui"])) < 1.0e-12 * a["U_i_bg"][0], "net ion background work on a periodic line"
# and the deposited (gated) electron pressure work vanishes on the pure background
assert np.max(np.abs(a["pw_e"])) < 1.0e-12 * a["U_e_bg"][0], \
    "the change of variables should leave no electron pressure work on a pure background"
print("PASS")
