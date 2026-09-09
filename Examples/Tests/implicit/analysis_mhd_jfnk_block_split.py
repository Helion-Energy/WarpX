#!/usr/bin/env python3
"""Gate of the block-split JFNK probe (newton.jfnk_epsilon_mode = block_split).

usage: analysis_mhd_jfnk_block_split.py <residual_blocks.txt> <newton.txt> <median_max> <p90_max> <any_max>

The residual-block report (pc_mhd_block.residual_block_norms, every solve) gives per linear solve the TRUE residual
r = b - J dU of the returned direction next to the Krylov solver's own recurrence estimate. For a consistent (linear)
finite-difference operator the two agree; when one epsilon must serve components whose magnitudes differ by orders of
magnitude within one Krylov vector, they do not. Gates: the true/estimate ratio over all reported solves has a median
<= median_max, a 90th percentile <= p90_max and no solve above any_max; the report's row partition holds. The same
deck with the global probe fails these gates (the sabotage; documented in the test's CMake comment).
"""
import sys
from collections import defaultdict

import numpy as np

report_file, newton_file = sys.argv[1], sys.argv[2]
median_max, p90_max, any_max = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
cols = ["core", "edge", "halo", "closed", "wall", "zend", "pinned", "total"]
T = cols.index("total")
rep = defaultdict(dict); norm = {}; its = {}
for line in open(report_file):
    if line.startswith("#") or not line.strip():
        continue
    p = line.rstrip("\n").split("\t"); key = (int(p[0]), int(p[2])); v = np.array([float(x) for x in p[6:]])
    rep[key][p[5]] = v; norm[key] = float(p[4]); its[key] = int(p[3])
assert len(rep) >= 5, len(rep)
ratios = []; worst_row = {}
for key in sorted(rep):
    R = rep[key]
    r2 = sum(v[2 * T] for v in R.values()); b2 = sum(v[2 * T + 1] for v in R.values())
    assert b2 > 0.0 and norm[key] > 0.0, key
    for label, v in R.items():
        part = v[0] + v[2] + v[4]  # core + edge + halo r2
        assert abs(part - v[2 * T]) <= 1.0e-8 * max(v[2 * T], 1e-300) + 1e-300, (key, label)
    ratio = np.sqrt(r2) / norm[key]
    ratios.append(ratio)
    dom = max(R, key=lambda l: R[l][2 * T])
    worst_row[dom] = worst_row.get(dom, 0) + 1
ratios = np.array(ratios)
med, p90, mx = np.median(ratios), np.percentile(ratios, 90), ratios.max()
newton = np.loadtxt(newton_file, comments="#", ndmin=2)
print(f"{len(ratios)} solves over {len(newton)} steps: true/estimate ratio median {med:.3f} p90 {p90:.3f} max {mx:.3f}; "
      f"mean GMRES {np.mean(list(its.values())):.1f}; Newton/step {newton[:, 2].mean():.2f}; dominant residual block per solve: "
      + ", ".join(f"{k} x{n}" for k, n in sorted(worst_row.items(), key=lambda kv: -kv[1])))
assert med <= median_max, ("median", med, median_max)
assert p90 <= p90_max, ("p90", p90, p90_max)
assert mx <= any_max, ("max", mx, any_max)
print("PASS")
