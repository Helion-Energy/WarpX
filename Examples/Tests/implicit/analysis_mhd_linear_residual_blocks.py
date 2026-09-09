#!/usr/bin/env python3
"""Gate of the linear-residual block report (pc_mhd_block.residual_block_norms).

usage: analysis_mhd_linear_residual_blocks.py <residual_blocks.txt> <newton.txt>

The report file holds, per linear solve that exceeded the iteration threshold
and per state block, the scaled squared norms of the TRUE linear residual
r = b - J dU and of the right-hand side b, split into regions. This gate checks
the report against quantities the solver computes independently:

1. the explicit residual norm sqrt(sum_blocks r2_total) reproduces the Krylov
   solver's own final residual norm (its Arnoldi recurrence estimate): the
   MEDIAN mismatch over the reported solves is below 1e-3 of |b| and every
   report's explicit/reported ratio lies in [0.5, 4]. The recurrence assumes a
   linear operator; the finite-difference Jacobian is not exactly linear over a
   Krylov basis that crosses limiter kinks, so single solves can show a true
   residual up to ~2x the estimate (measured: the first solve of the ions-on
   deck, 6.9e-4 vs 3.5e-4 at |b| 1.5e-2; the other 35 agree to 1e-6..7e-4 of
   |b|). A sign error in r = b - J dU gives |r| ~ 2 |b| against an estimate
   below 0.05 |b| (ratio >= 40) and fails; a wrong block scale fails the median
   (the sabotage checks of the diagnostic);
2. the three density classes partition every block exactly (core + edge + halo
   = total, for r2 and b2), every overlay is bounded by the total, the shares
   are finite;
3. the pinned columns of the floored blocks vanish in the active-set mode (the
   free solve has an exactly zero residual on the active set by construction:
   masked right-hand side, masked operator); rows without a floor mask carry -1;
4. every block of the dual-energy RZ state is reported (mass_density,
   momentum_density r/theta/z, electron_energy, ion_energy, ion_internal_energy,
   B_r/theta/z), and the number of reports equals the number of Newton iterations
   of newton.txt (threshold 0).
"""
import sys
from collections import defaultdict

import numpy as np

report_file = sys.argv[1]
newton_file = sys.argv[2]

columns = ["core", "edge", "halo", "closed", "wall", "zend", "pinned", "total"]
reports = defaultdict(dict)  # (step, newton_iter) -> row label -> dict
reported_norm = {}
gmres_iters = {}
with open(report_file) as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        step, time, newton_iter, iters, norm, label = parts[:6]
        key = (int(step), int(newton_iter))
        values = np.array([float(v) for v in parts[6:]])
        assert values.size == 2 * len(columns), (len(values), line)
        reports[key][label] = {
            "r2": {c: values[2 * n] for n, c in enumerate(columns)},
            "b2": {c: values[2 * n + 1] for n, c in enumerate(columns)},
        }
        reported_norm[key] = float(norm)
        gmres_iters[key] = int(iters)

assert len(reports) >= 3, f"only {len(reports)} reports found in {report_file}"

# Row labels are the solver-vector block names without their "implicit_mhd_"
# prefix, multi-component blocks suffixed per component.
expected_rows = {"mass_density", "momentum_density_r", "momentum_density_theta",
                 "momentum_density_z", "electron_energy", "ion_energy",
                 "ion_internal_energy", "B_r", "B_theta", "B_z"}
floored_rows = {"mass_density", "electron_energy", "ion_energy", "ion_internal_energy"}
worst_mismatch = 0.0
mismatches = []
worst_partition = 0.0
worst_pinned = 0.0
pinned_rows_seen = 0
for key in sorted(reports):
    rows = reports[key]
    assert set(rows) == expected_rows, (key, sorted(rows))
    total_r2 = sum(row["r2"]["total"] for row in rows.values())
    total_b2 = sum(row["b2"]["total"] for row in rows.values())
    assert total_b2 > 0.0, key
    explicit = np.sqrt(total_r2)
    reported = reported_norm[key]
    mismatch = abs(explicit - reported) / np.sqrt(total_b2)
    worst_mismatch = max(worst_mismatch, mismatch)
    mismatches.append(mismatch)
    ratio = explicit / reported if reported > 0.0 else np.inf
    assert 0.5 <= ratio <= 4.0, (
        f"step {key[0]} newton_iter {key[1]}: explicit |r| {explicit:.6e} vs "
        f"reported {reported:.6e} (|b| {np.sqrt(total_b2):.6e}, gmres {gmres_iters[key]})")
    for label, row in rows.items():
        for kind in ("r2", "b2"):
            v = row[kind]
            total = v["total"]
            partition = v["core"] + v["edge"] + v["halo"]
            scale = max(total, 1e-300)
            worst_partition = max(worst_partition, abs(partition - total) / scale)
            assert abs(partition - total) <= 1.0e-8 * scale + 1e-300, (key, label, kind, partition, total)
            for c in ("closed", "wall", "zend"):
                assert v[c] <= total * (1.0 + 1.0e-10) + 1e-300, (key, label, kind, c, v[c], total)
            assert np.isfinite(total)
        if row["r2"]["pinned"] >= 0.0:
            pinned_rows_seen += 1
            assert label in floored_rows, label
            rel = row["r2"]["pinned"] / max(total_r2, 1e-300)
            worst_pinned = max(worst_pinned, rel)
            assert rel <= 1.0e-12, (key, label, row["r2"]["pinned"], total_r2)
        else:
            assert row["r2"]["pinned"] == -1.0 and row["b2"]["pinned"] == -1.0, (key, label)

median_mismatch = float(np.median(mismatches))
assert median_mismatch <= 1.0e-3, ("median explicit-vs-reported mismatch", median_mismatch)

# One report per Newton iteration (threshold 0): newton.txt column [2] = iters per step.
newton = np.loadtxt(newton_file, comments="#", ndmin=2)
steps_reported = defaultdict(int)
for step, _ in reports:
    steps_reported[step] += 1
for row in newton:
    step = int(row[0])
    iters = int(row[2])
    # every Newton iteration solves once; a solve that converged in zero
    # Krylov iterations (never seen; kept tolerant) would not be reported
    assert iters - 1 <= steps_reported[step] <= iters, (step, steps_reported[step], iters)

# Human-readable summary of the last report (block shares, region shares).
key = max(reports)
rows = reports[key]
total_r2 = sum(row["r2"]["total"] for row in rows.values())
print(f"last report: step {key[0]} newton_iter {key[1]} gmres_iters {gmres_iters[key]} "
      f"|r| {np.sqrt(total_r2):.4e} (reported {reported_norm[key]:.4e})")
print("  block shares of |r|^2:")
for label in sorted(rows, key=lambda l: -rows[l]["r2"]["total"]):
    r2 = rows[label]["r2"]["total"]
    b2 = rows[label]["b2"]["total"]
    print(f"    {label:20s} share {r2 / total_r2:8.4f}  r/b {np.sqrt(r2 / b2) if b2 > 0 else 0.0:9.3e}")
print("  region shares of |r|^2:")
for c in columns[:-1]:
    r2 = sum(row["r2"][c] for row in rows.values() if row["r2"][c] >= 0.0)
    print(f"    {c:8s} {r2 / total_r2:8.4f}")
print(f"gates: {len(reports)} reports, explicit-vs-reported mismatch median {median_mismatch:.3e} (<= 1e-3) "
      f"worst {worst_mismatch:.3e} of |b| (ratio in [0.5, 4]), worst partition defect {worst_partition:.3e} (<= 1e-8), "
      f"pinned rows {pinned_rows_seen}, worst pinned share {worst_pinned:.3e} (<= 1e-12)")
print("PASS")
