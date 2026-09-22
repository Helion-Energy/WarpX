#!/usr/bin/env python3
"""Energy closure of the booked hyper-resistive dissipation.

Reads, from the run directory:
  * the stdout log (path given as the first argument) for the
    ``energy_budget_J`` lines -> final cumulative hyp_bulk + hyp_band and
    the clamp tally hyp_clamp;
  * diags/reducedfiles/fe.txt (FieldEnergy) -> magnetic energy U_B(t);
  * diags/reducedfiles/diss.txt (HybridDissipation) -> P_etaH_booked(t).

Checks
  1. field closure:   |dU_B + hyp| / |dU_B| < tol      (dU_B = U_B(end) - U_B(0) < 0)
  2. diag closure:    |sum_steps dt P_etaH_booked - hyp| / hyp < tol
  3. hyp > 0 and hyp_clamp == 0.

Only the last two channels are first order in time (the booking samples J . E_H
once per step while the B substeps integrate the drag exactly), so tol is set
at 1 percent and the measured numbers are printed for the report.
"""

import re
import sys

import numpy as np

log_path = sys.argv[1] if len(sys.argv) > 1 else "run.log"
tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01

# --- ledger -----------------------------------------------------------------
hyp_bulk = hyp_band = hyp_clamp = None
with open(log_path) as fh:
    for line in fh:
        if "energy_budget_J:" not in line:
            continue
        kv = dict(re.findall(r"(\w+)=([-+0-9.eE]+)", line))
        hyp_bulk = float(kv["hyp_bulk"])
        hyp_band = float(kv["hyp_band"])
        hyp_clamp = float(kv["hyp_clamp"])
assert hyp_bulk is not None, "no energy_budget_J line found in " + log_path
hyp = hyp_bulk + hyp_band

# --- reduced diags ----------------------------------------------------------
fe = np.loadtxt("diags/reducedfiles/fe.txt")
diss = np.loadtxt("diags/reducedfiles/diss.txt")
# FieldEnergy columns: step, time, total, E, B  (level 0)
t = fe[:, 1]
U_B = fe[:, 4]
dU_B = U_B[-1] - U_B[0]
dt = np.diff(t)
# HybridDissipation: header names the columns; P_etaH_booked is the last one.
with open("diags/reducedfiles/diss.txt") as fh:
    header = fh.readline()
cols = re.findall(r"\[(\d+)\](\w+)", header)
names = [c[1] for c in cols]
ib = names.index("P_etaH_booked")
P_booked = diss[:, ib]
# The booking of step n happens with the state the diag samples at the end of
# step n; the ledger is cumulative over steps 1..N, so integrate the sampled
# power over those steps.
E_booked = (
    float(np.sum(P_booked[1:] * dt))
    if len(dt) == len(P_booked) - 1
    else float(np.sum(P_booked[1:] * dt[-1]))
)

print(f"[hyper_res_heating] steps            = {int(fe[-1, 0])}")
print(f"[hyper_res_heating] U_B(0)           = {U_B[0]:.10e} J")
print(f"[hyper_res_heating] U_B(end)         = {U_B[-1]:.10e} J")
print(f"[hyper_res_heating] dU_B             = {dU_B:.6e} J")
print(f"[hyper_res_heating] hyp_bulk         = {hyp_bulk:.6e} J")
print(f"[hyper_res_heating] hyp_band         = {hyp_band:.6e} J")
print(f"[hyper_res_heating] hyp_clamp        = {hyp_clamp:.6e} J")
print(f"[hyper_res_heating] int P_booked dt  = {E_booked:.6e} J")
r_field = abs(dU_B + hyp) / abs(dU_B)
r_diag = abs(E_booked - hyp) / abs(hyp)
print(f"[hyper_res_heating] field closure    |dU_B + hyp|/|dU_B| = {r_field:.3e}")
print(f"[hyper_res_heating] diag  closure    |int P dt - hyp|/hyp = {r_diag:.3e}")

ok = True
if not (hyp > 0.0):
    print("FAIL: booked hyper-resistive heat is not positive")
    ok = False
if hyp_clamp != 0.0:
    print("FAIL: Te-floor clamp tally is nonzero")
    ok = False
if not (dU_B < 0.0):
    print("FAIL: magnetic energy did not decrease")
    ok = False
if r_field > tol:
    print(f"FAIL: field closure {r_field:.3e} exceeds {tol}")
    ok = False
if r_diag > tol:
    print(f"FAIL: diag closure {r_diag:.3e} exceeds {tol}")
    ok = False
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
