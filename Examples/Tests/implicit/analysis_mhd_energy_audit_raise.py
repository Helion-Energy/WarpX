#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Global energy audit on a raise-mode halo pedestal deck (rev30 F2).

Usage: analysis_mhd_energy_audit_raise.py <energy_audit.txt> <halo_pedestal_ledger.txt>

The legacy pedestal raise (RefreshHaloPedestal) injects mass and energy
BETWEEN steps, before the step-old state the audit differences against.
The audit reports the injection as inject_* (this row's step-old totals
minus the previous row's final totals) and books it in neither residual,
because dW_total spans the step only. Checks:
1. resid_full at round-off on every step (|.| < 1e-11 W_total) although
   inject_* is nonzero on at least one step (the deck's raise fires);
2. inject_e / inject_i equal the pedestal ledger's per-step energy deltas
   (columns "step raised_cells mass energy_e energy_i", cumulative) to
   1e-6 relative, and the audit's own ledger_pedestal_* columns agree;
3. the first row's injection is zero by definition (no previous row).
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
for name, values in a.items():
    assert np.all(np.isfinite(values)), f"non-finite column {name}"
w_total = a["W_B_tot"] + a["U_e"] + a["E_i"]
tol = 1.0e-11 * np.max(w_total)
inject = a["inject_e"] + a["inject_i"]
print(f"{sys.argv[1]}: {n} rows; max |inject_e + inject_i| = {np.max(np.abs(inject)):.3e} J; "
      f"max |resid_full| = {np.max(np.abs(a['resid_full'])):.3e} J (bound {tol:.3e}); "
      f"cumulative injection {np.sum(inject):.4e} J vs ledger {ledger[-1, 3] + ledger[-1, 4]:.4e} J")
assert np.max(np.abs(inject)) > 0.0, "the raise never fired: this deck does not exercise inject_*"
assert np.all(np.abs(a["resid_full"]) < tol), "full closure residual above round-off with the raise active"
assert a["inject_e"][0] == 0.0 and a["inject_i"][0] == 0.0, "the first row has no previous row"
# ledger cumulative energies per step -> deltas; the ledger row k books the raise applied
# before step k, the audit's inject at row k sees the same raise
assert len(ledger) == n, "the pedestal ledger must have one row per step"
d_e = np.diff(np.concatenate([[0.0], ledger[:, 3]]))
d_i = np.diff(np.concatenate([[0.0], ledger[:, 4]]))
scale = max(np.max(np.abs(ledger[:, 3])), np.max(np.abs(ledger[:, 4])), 1.0e-300)
for k in range(1, n):
    assert abs(a["inject_e"][k] - d_e[k]) < 1.0e-6 * scale + 1.0e-14 * np.max(w_total), f"inject_e differs from the ledger at row {k + 1}"
    assert abs(a["inject_i"][k] - d_i[k]) < 1.0e-6 * scale + 1.0e-14 * np.max(w_total), f"inject_i differs from the ledger at row {k + 1}"
    assert abs(a["ledger_pedestal_e"][k] - d_e[k]) < 1.0e-12 * scale + 1.0e-30, "ledger_pedestal_e column"
    assert abs(a["ledger_pedestal_i"][k] - d_i[k]) < 1.0e-12 * scale + 1.0e-30, "ledger_pedestal_i column"
print("PASS")
