#!/usr/bin/env python3
#
# --- Analysis for the GOL vacuum-gap gate (gol_vacuum_gap.py).
# --- (a) finiteness: final E/B/rho dumps contain no NaN/Inf and the
# ---     vacuum E stays at a physical scale;
# --- (b) per-sweep Jacobi decay: geometric factor of the [gol] sweep
# ---     increment lines, global and vacuum-restricted;
# --- (c) sweep-ladder under-convergence: relL2 between the sweeps-4 and
# ---     sweeps-16 final fields = the production-solve error.
# ---
# --- usage: gol_gap_analysis.py <s4_dir> <s16_dir>

import glob
import json
import re
import sys

import h5py
import numpy as np


def last_dump(d):
    files = glob.glob(f"{d}/diags/field_diags/openpmd_*.h5")
    assert files, f"no dumps under {d}"
    return max(files, key=lambda p: int(p.split("_")[-1].split(".")[0]))


def arrs(p):
    out = {}
    with h5py.File(p) as f:
        it = list(f["data"].keys())[0]

        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = np.squeeze(obj[()])

        f[f"data/{it}/fields"].visititems(walk)
    return out


def sweep_decay(log, n_sweeps):
    """Per-call increment sequences from the [gol] sweep lines; returns
    the median per-sweep decay factor (global, vacuum) over the last
    quarter of the run and the median terminal vacuum increment."""
    pat = re.compile(r"\[gol\] sweep (\d+) dE_l2 (\S+) dE_l2_vac (\S+) dE_max (\S+)")
    calls, cur = [], []
    with open(log) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            s = int(m.group(1))
            if s == 0 and cur:
                calls.append(cur)
                cur = []
            cur.append((float(m.group(2)), float(m.group(3))))
    if cur:
        calls.append(cur)
    calls = [c for c in calls if len(c) == n_sweeps]
    tail = calls[-max(1, len(calls) // 4) :]
    qg, qv, term = [], [], []
    for c in tail:
        g = np.array([x[0] for x in c])
        v = np.array([x[1] for x in c])
        if np.all(g[:-1] > 0):
            qg.extend(g[1:] / g[:-1])
        if np.all(v[:-1] > 0):
            qv.extend(v[1:] / v[:-1])
        term.append(v[-1])
    return (len(calls), np.median(qg), np.median(qv), np.median(term))


s4_dir, s16_dir = sys.argv[1:3]
with open(f"{s4_dir}/gate_params.json") as f:
    p = json.load(f)

a4 = arrs(last_dump(s4_dir))
a16 = arrs(last_dump(s16_dir))

# (a) finiteness + vacuum scale
bad = [k for k, v in a4.items() if not np.all(np.isfinite(v))]
assert not bad, f"non-finite fields in s4 final dump: {bad}"
z = np.arange(len(a4["rho"])) * p["dz"]
vac = (z < p["z_lo"] - 2 * p["dz"]) | (z > p["z_hi"] + 2 * p["dz"])
e_vac = max(np.abs(a4[f"E/{c}"][vac]).max() for c in "xyz")
e_all = max(np.abs(a4[f"E/{c}"]).max() for c in "xyz")
print(f"(a) finite: yes | max|E| vacuum {e_vac:.3e} vs global {e_all:.3e} V/m")

# (b) per-sweep decay
n4, qg4, qv4, t4 = sweep_decay(f"{s4_dir}/run.log", p["gol_sweeps"])
n16, qg16, qv16, t16 = sweep_decay(f"{s16_dir}/run.log", 16)
print(
    f"(b) s4 : {n4} solves | decay/sweep global {qg4:.3f}, vacuum {qv4:.3f}"
    f" | terminal vac increment {t4:.3e} V/m"
)
print(
    f"    s16: {n16} solves | decay/sweep global {qg16:.3f}, "
    f"vacuum {qv16:.3f} | terminal vac increment {t16:.3e} V/m"
)

# (c) ladder under-convergence
rel = {}
for c in "xyz":
    d = a4[f"E/{c}"] - a16[f"E/{c}"]
    n = np.sqrt(np.sum(a16[f"E/{c}"] ** 2))
    rel[f"E{c}"] = np.sqrt(np.sum(d**2)) / max(n, 1e-300)
for c in "xyz":
    d = a4[f"B/{c}"] - a16[f"B/{c}"]
    n = np.sqrt(np.sum(a16[f"B/{c}"] ** 2))
    rel[f"B{c}"] = np.sqrt(np.sum(d**2)) / max(n, 1e-300)
print("(c) s4-vs-s16 final relL2: " + " ".join(f"{k}={v:.3e}" for k, v in rel.items()))

worst_b = max(rel[f"B{c}"] for c in "xyz")
verdict = "PASS" if worst_b < 1e-3 else "MARGINAL" if worst_b < 1e-2 else "FAIL"
print(
    f"{verdict}: sweeps-4 B-field solve error {worst_b:.3e} relL2 "
    f"vs sweeps-16 across a {p['gap_cells']}-cell vacuum gap"
)
