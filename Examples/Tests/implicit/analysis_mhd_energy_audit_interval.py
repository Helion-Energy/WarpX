#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""energy_audit_interval > 1 must write exact single-step rows (rev30 F3).

Usage: analysis_mhd_energy_audit_interval.py <energy_audit_interval_k.txt> <energy_audit_interval_1.txt>

Every audited step of the coarse run must reproduce the interval-1 run's
row for the same step in every per-step column (the same deck, the same
binary: the solution is bit-identical, and the audit's per-step terms --
including the code's ledger deltas, which are snapshotted every step --
are functions of that step alone). Excluded by definition: the *_cum
columns (they sum the audited steps only), inject_* (needs consecutive
audited steps; they are zero at interval k > 1) and the frac_* columns
(built from the cumulatives).
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


coarse = read_audit(sys.argv[1])
fine = read_audit(sys.argv[2])
steps_c = coarse["step"].astype(int)
steps_f = fine["step"].astype(int)
assert len(steps_c) > 1 and len(steps_c) < len(steps_f), "expected a coarse and a fine run"
k = steps_c[1] - steps_c[0]
print(f"coarse rows {len(steps_c)} (interval {k}), fine rows {len(steps_f)}")
excluded = [c for c in coarse if c.endswith("_cum") or c.startswith("inject_") or c.startswith("frac_")]
worst = 0.0
worst_name = ""
for name in coarse:
    if name in excluded:
        continue
    for i, s in enumerate(steps_c):
        j = int(np.where(steps_f == s)[0][0])
        vc, vf = coarse[name][i], fine[name][j]
        scale = max(abs(vf), 1.0e-300)
        rel = abs(vc - vf) / scale
        if rel > worst and abs(vc - vf) > 1.0e-14 * max(abs(vf), 1.0):
            worst, worst_name = rel, f"{name} at step {s} ({vc:.17g} vs {vf:.17g})"
        assert abs(vc - vf) <= 1.0e-10 * max(abs(vf), 1.0e-30) + 1.0e-14 * max(1.0, abs(vf)), \
            f"column {name} differs between interval {k} and 1 at step {s}: {vc:.17g} vs {vf:.17g}"
assert np.all(coarse["inject_e"] == 0.0) and np.all(coarse["inject_i"] == 0.0), "inject_* must be zero at a coarse interval"
print(f"  every per-step column matches the interval-1 run at the audited steps (worst relative {worst:.1e}: {worst_name or 'none'})")
# the cumulative columns sum the audited steps only: check one of them
cum_c = coarse["poynt_in_cum"][-1]
cum_expected = np.sum([fine["poynt_in"][int(np.where(steps_f == s)[0][0])] for s in steps_c])
assert abs(cum_c - cum_expected) <= 1.0e-10 * max(abs(cum_expected), 1.0e-30) + 1.0e-14, "cumulative over the audited steps"
print("PASS")
