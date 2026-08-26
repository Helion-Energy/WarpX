#!/usr/bin/env python3
#
# --- Stability-contract analysis for the vacuum-annulus normal-modes
# --- test: the plasma-column edge carries a sub-grid diamagnetic
# --- current sheet whose surface mode historically ran away (through
# --- the qn-pin exit channel, or through the artificial fast branch
# --- at the default je_c_frac). The test contract is boundedness:
# --- every field finite at the final dump and the edge-mode amplitude
# --- max|B_theta| held well below the runaway trajectory (which
# --- crosses 10% of B0 within the first 200 steps at test scale;
# --- healthy arms of either esolve sit at or below ~1%).

import glob
import sys

import dill
import h5py
import numpy as np

with open("sim_parameters.dpkl", "rb") as f:
    sim = dill.load(f)

files = sorted(glob.glob("diags/field_diags/openpmd_*.h5"))
assert files, "no field dumps found"

with h5py.File(files[-1], "r") as f:
    it = list(f["data"].keys())[0]
    fields = f[f"data/{it}/fields"]
    bt_max = float(np.abs(np.array(fields["B"]["t"])).max())
    finite = all(
        np.isfinite(np.array(fields[q][c])).all()
        for q in ("B", "E")
        for c in ("r", "t", "z")
    )

frac = bt_max / abs(sim.B0)
print(f"final dump (step {it}): max|B_theta| = {bt_max:.4e} T "
      f"= {100 * frac:.3f}% of B0, finite = {finite}")

assert finite, "non-finite B or E in the final dump"
assert frac < 0.05, (
    f"edge-mode amplitude {100 * frac:.2f}% of B0 exceeds the 5% "
    "stability contract (the runaway class crosses 10% within 200 "
    "steps at this scale)"
)
sys.exit(0)
