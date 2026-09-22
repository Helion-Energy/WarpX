#!/usr/bin/env python3
"""Viscous drag in Ohm's law: decay rate (T1/T3) and energy closure (T2).

usage: analysis_viscous_drag.py run.log [--eta ETA_EFF] [--k K] [--dx DX]
                                        [--form visc|lap] [--aniso] [--tol-rate R] [--tol-close C]

Reads the stdout log for the energy_budget_J lines (visc_bulk/visc_band =
booked heating, visc_work_bulk/visc_work_band = drag work J . E_visc,
hyp_* for the eta_H arm), diags/reducedfiles/fe.txt for U_B(t), and
diags/reducedfiles/diss.txt for P_nu / P_visc_work.

Decay rate: the mode energy U_B(t) - U_B0 (U_B0 = B0^2 V/(2 mu0), the exactly
conserved uniform part) decays as exp(-2 gamma t). Predicted discrete rates:
    form = visc : gamma = eta kc^2 ky^2 / mu0,  kc = sin(k dx)/dx,  ky = 2 sin(k dx/2)/dx
    form = lap  : gamma = eta ky^4 / mu0
Closure: the booked heat (strain form: visc = Int Q_nu; work form / eta_H arm:
the work) must equal -dU_B; the strain and work ledgers must agree.
"""

import argparse
import re
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--eta", type=float, default=1.0e-9)
ap.add_argument("--k", type=float, default=41.887902047863905)
ap.add_argument("--dx", type=float, default=0.3 / 32)
ap.add_argument("--form", default="visc")
ap.add_argument("--aniso", action="store_true")
ap.add_argument("--B0", type=float, default=0.1)
ap.add_argument("--vol", type=float, default=0.3 * 0.075 * 0.075)
ap.add_argument("--tol-rate", type=float, default=2.0e-2)
ap.add_argument("--tol-close", type=float, default=1.0e-2)
a = ap.parse_args()

mu0 = 4.0e-7 * np.pi

# --- ledger -----------------------------------------------------------------
kv = None
with open(a.log) as fh:
    for line in fh:
        if "energy_budget_J:" in line:
            kv = dict(re.findall(r"(\w+)=([-+0-9.eE]+)", line))
assert kv is not None, "no energy_budget_J line in " + a.log


def g(key):
    return float(kv.get(key, "nan"))


visc = g("visc_bulk") + g("visc_band")
work = g("visc_work_bulk") + g("visc_work_band")
hyp = g("hyp_bulk") + g("hyp_band")
booked = visc if np.isfinite(visc) and visc != 0.0 else hyp

# --- field energy -----------------------------------------------------------
fe = np.loadtxt("diags/reducedfiles/fe.txt")
t = fe[:, 1]
U_B = fe[:, 4]
dU_B = U_B[-1] - U_B[0]
U_B0 = a.B0**2 * a.vol / (2.0 * mu0)
U_mode = U_B - U_B0

print(f"[viscous_drag] steps        = {int(fe[-1, 0])}")
print(f"[viscous_drag] U_B(0)       = {U_B[0]:.10e} J   (uniform part {U_B0:.10e} J)")
print(
    f"[viscous_drag] U_mode(0)    = {U_mode[0]:.6e} J,  U_mode(end) = {U_mode[-1]:.6e} J"
)
print(f"[viscous_drag] dU_B         = {dU_B:.6e} J")
print(f"[viscous_drag] visc (Q_nu)  = {visc:.6e} J")
print(f"[viscous_drag] visc_work    = {work:.6e} J")
print(f"[viscous_drag] hyp          = {hyp:.6e} J")

ok = True
# closure of the booked channel against the field
r_field = abs(dU_B + booked) / abs(dU_B)
print(f"[viscous_drag] field closure |dU_B + booked|/|dU_B| = {r_field:.3e}")
if r_field > a.tol_close:
    print(f"FAIL: field closure {r_field:.3e} > {a.tol_close}")
    ok = False
# strain vs work
if np.isfinite(work) and np.isfinite(visc) and visc != 0.0:
    r_sw = abs(work - visc) / abs(visc)
    print(f"[viscous_drag] strain/work  |work - visc|/visc = {r_sw:.3e}")
    if r_sw > a.tol_close:
        print(f"FAIL: strain vs work {r_sw:.3e} > {a.tol_close}")
        ok = False

# decay rate (isotropic arms only)
if not a.aniso:
    sel = slice(len(t) // 10, None)
    tt = t[sel]
    yy = np.log(U_mode[sel])
    slope, _ = np.polyfit(tt, yy, 1)
    gamma_fit = -0.5 * slope
    kdx = a.k * a.dx
    kc = np.sin(kdx) / a.dx
    ky = 2.0 * np.sin(0.5 * kdx) / a.dx
    if a.form == "visc":
        gamma_pred = a.eta * kc**2 * ky**2 / mu0
    else:
        gamma_pred = a.eta * ky**4 / mu0
    gamma_cont = a.eta * a.k**4 / mu0
    r_rate = abs(gamma_fit - gamma_pred) / gamma_pred
    print(f"[viscous_drag] gamma fit    = {gamma_fit:.6e} 1/s")
    print(
        f"[viscous_drag] gamma pred   = {gamma_pred:.6e} 1/s (discrete symbol, form={a.form})"
    )
    print(f"[viscous_drag] gamma cont.  = {gamma_cont:.6e} 1/s (eta k^4/mu0)")
    print(f"[viscous_drag] rate error    |fit - pred|/pred = {r_rate:.3e}")
    if r_rate > a.tol_rate:
        print(f"FAIL: rate error {r_rate:.3e} > {a.tol_rate}")
        ok = False

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
