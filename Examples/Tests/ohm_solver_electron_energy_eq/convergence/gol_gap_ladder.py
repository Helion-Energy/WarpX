#!/usr/bin/env python3
#
# --- Sweep-ladder convergence curve for the GOL vacuum-gap gate:
# --- pairwise relL2 between consecutive rungs, global and
# --- gap-restricted, to separate true gap-solve error from
# --- plasma-noise chaos amplification (identical ICs, so a converged
# --- pair must collapse toward zero).
# --- usage: gol_gap_ladder.py <dir_s4> <dir_s16> <dir_s64> [...]

import glob
import json
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


dirs = sys.argv[1:]
runs = []
for d in dirs:
    with open(f"{d}/gate_params.json") as f:
        p = json.load(f)
    runs.append((p["gol_sweeps"], p, arrs(last_dump(d))))
runs.sort(key=lambda r: (r[0], r[1].get("substeps", 0)))

p0 = runs[0][1]
z = np.arange(len(runs[0][2]["rho"])) * p0["dz"]
gap = (z < p0["z_lo"] - 2 * p0["dz"]) | (z > p0["z_hi"] + 2 * p0["dz"])


def rel(a, b, k, mask=None):
    x, y = a[k], b[k]
    if mask is not None:
        x, y = x[mask], y[mask]
    n = np.sqrt(np.sum(y**2))
    return np.sqrt(np.sum((x - y) ** 2)) / max(n, 1e-300)


print(f"gap cells: {p0['gap_cells']}")
for (s1, _, a), (s2, _, b) in zip(runs[:-1], runs[1:]):
    row = [f"s{s1} vs s{s2}:"]
    for k in ("B/x", "B/y", "E/x", "E/y"):
        row.append(f"{k} {rel(a, b, k):.3e} (gap {rel(a, b, k, gap):.3e})")
    print("  " + "  ".join(row))
