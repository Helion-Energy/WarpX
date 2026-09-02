#!/usr/bin/env python3
"""Identity-gate comparison.

Compares a remapped continuation against the uninterrupted reference over
the same interval.  At refinement ratio 1 the prolongation is the
identity, so whatever difference appears here is a round-trip loss.
"""

import argparse
import glob
import sys

import h5py
import numpy as np

sys.path.insert(0, "..")
from grid_remap import CylindricalMesh, divergence  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--ref", default="run_ref")
parser.add_argument("--cont", default="run_cont")
parser.add_argument("--nr", type=int, default=64)
parser.add_argument("--nz", type=int, default=128)
parser.add_argument("--split", type=int, default=200)
parser.add_argument("--final", type=int, default=400)
args = parser.parse_args()

QE = 1.602176634e-19


def relerr(a, b):
    scale = max(np.abs(a).max(), 1e-300)
    d = np.abs(a - b)
    return d.max() / scale, np.sqrt((d**2).mean()) / scale


print("=== B components, continuation vs uninterrupted reference ===")
ref = np.load(f"{args.ref}/B_final.npz")
con = np.load(f"{args.cont}/B_final.npz")
worst = 0.0
for k, label in (("br", "B_r"), ("bt", "B_theta"), ("bz", "B_z")):
    m, l2 = relerr(ref[k], con[k])
    worst = max(worst, m)
    ident = "BIT-IDENTICAL" if m == 0.0 else ""
    print(f"  {label:8s} max rel {m:.6e}   L2 rel {l2:.6e}  {ident}")

print("\n=== divergence cleanliness of the continued state ===")
mesh = CylindricalMesh(args.nr, args.nz, 0.0, 0.10, -0.25, 0.25)
for tag, dat in (("reference   ", ref), ("continuation", con)):
    div = divergence(mesh, dat["br"], dat["bz"])
    norm = max(np.abs(dat["bz"]).max(), 1e-300) / mesh.dr
    print(f"  {tag}: max|div B| {np.abs(div).max():.4e} T/m"
          f"   L2 {np.sqrt((div**2).mean()):.4e}"
          f"   relative {np.abs(div).max()/norm:.3e}")


def read_field(d, step, name):
    fns = sorted(glob.glob(f"{d}/fdump/openpmd_*.h5"))
    for fn in fns:
        with h5py.File(fn, "r") as f:
            it = list(f["data"].keys())[0]
            if int(it) != step:
                continue
            g = f[f"data/{it}"]
            ds = g[f"fields/{name}"]
            return np.asarray(ds)[0] * float(ds.attrs.get("unitSI", 1.0))
    return None


print("\n=== rho, continuation vs reference (same physical time) ===")
r_ref = read_field(args.ref, args.final, "rho")
r_con = read_field(args.cont, args.final - args.split, "rho")
if r_ref is None or r_con is None:
    print("  (rho dumps not found at the matched steps)")
else:
    m, l2 = relerr(r_ref, r_con)
    worst = max(worst, m)
    print(f"  rho      max rel {m:.6e}   L2 rel {l2:.6e}")
    w_ref = np.abs(r_ref).sum()
    w_con = np.abs(r_con).sum()
    print(f"  total |rho|: ref {w_ref:.10e}  cont {w_con:.10e}"
          f"   rel diff {abs(w_con-w_ref)/max(w_ref,1e-300):.3e}")

print("\n=== field energy proxy (sum B^2) ===")
e_ref = sum(float((ref[k] ** 2).sum()) for k in ("br", "bt", "bz"))
e_con = sum(float((con[k] ** 2).sum()) for k in ("br", "bt", "bz"))
print(f"  ref {e_ref:.10e}   cont {e_con:.10e}"
      f"   rel drift {abs(e_con-e_ref)/max(e_ref,1e-300):.3e}")

print(f"\nWORST relative difference across all compared fields: {worst:.6e}")
if worst == 0.0:
    print("VERDICT: BIT-IDENTICAL")
elif worst < 1e-10:
    print("VERDICT: PASS (roundoff)")
elif worst < 1e-3:
    print(f"VERDICT: TRACKS to {worst:.2e} -- state which losses account for it")
else:
    print(f"VERDICT: FAIL ({worst:.2e}) -- diagnose the round-trip loss")
