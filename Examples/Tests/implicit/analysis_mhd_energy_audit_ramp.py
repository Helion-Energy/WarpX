#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Independent expected values for the circuit power and the Poynting faces
(rev30 F1): a prescribed linear ramp s(t) = 1 + t/tau of the external coil
field on the Green's-open magnetized column.

Usage: analysis_mhd_energy_audit_ramp.py <energy_audit.txt> <tau_seconds>

The external field B_ext = s(t) curl_h A has its own exact discrete
Faraday identity, independent of the plasma:
    dW_B_ext = circuit_in_ext - poynt_out_ext - theta_diss_ext,
with dW_B_ext measured from the stored fields, circuit_in_ext = -dt sum
E_ext . J_coil (J_coil = curl B_ext/mu0 by the solver's Ampere operator),
poynt_out_ext the face formula applied to (E_ext, B_ext) and theta_diss_ext
exact algebra. Checks:
1. |ext_defect| < 1e-12 W_B_ext on every step: this gates J_coil, the
   E.J dot and the face formula (sign, staggering, area) against a number
   that none of them enters (dW_B_ext);
2. W_B_ext(t) / W_B_ext(0) = s(t)^2 to 1e-10 (analytic: the stored field
   is s(t) times a fixed field);
3. |faraday_defect| < 1e-12 W_total on every step (the total field's
   identity with the coil work named);
4. circuit_in_ext, poynt_in_ext and theta_diss_ext are nonzero (the ramp
   drives all three), resid_full at round-off.
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
tau = float(sys.argv[2])
n = len(a["step"])
for name, values in a.items():
    assert np.all(np.isfinite(values)), f"non-finite column {name}"
w_total = a["W_B_tot"] + a["U_e"] + a["E_i"]
w_ext = a["W_B_ext"]
t = a["time"]
dt = a["dt"]
print(f"{sys.argv[1]}: {n} rows; W_B_ext {w_ext[0]:.6e} -> {w_ext[-1]:.6e} J; circuit_in_ext_cum "
      f"{np.sum(a['circuit_in_ext']):.6e} J; poynt_in_ext_cum {a['poynt_in_ext_cum'][-1]:.6e} J; "
      f"theta_diss_ext_cum {a['theta_diss_ext_cum'][-1]:.6e} J; max |ext_defect| {np.max(np.abs(a['ext_defect'])):.3e} J; "
      f"max |faraday_defect| {np.max(np.abs(a['faraday_defect'])):.3e} J")
# 1. the external identity
assert np.all(np.abs(a["ext_defect"]) < 1.0e-12 * np.max(w_ext)), "external-field identity broken (J_coil, E.J or the Poynting faces)"
# 2. analytic scaling of the stored external energy: W(t) = s(t)^2 W(0) with s = 1 + t/tau
#    (the time column is the end-of-step time, W_B_ext the end-of-step value)
s0 = 1.0 + t[0] / tau
w0 = w_ext[0] / s0**2
assert np.all(dt > 0.0)
for k in range(n):
    s = 1.0 + t[k] / tau
    assert abs(w_ext[k] - s * s * w0) < 1.0e-10 * w0 * s * s, f"W_B_ext does not follow s(t)^2 at row {k + 1}"
print(f"  W_B_ext(t) = s(t)^2 W_B_ext(0) to 1e-10 (s_end = {1.0 + t[-1] / tau:.4f})")
# 3. the total field's identity
assert np.all(np.abs(a["faraday_defect"]) < 1.0e-12 * np.max(w_total)), "Faraday defect above round-off"
# 4. live terms and closure
assert np.all(a["circuit_in_ext"] != 0.0), "circuit_in_ext must be nonzero on every step of the ramp"
assert np.max(np.abs(a["poynt_in_ext"])) > 0.0 and np.max(a["theta_diss_ext"]) > 0.0
assert np.all(np.abs(a["resid_full"]) < 1.0e-11 * np.max(w_total)), "full closure residual above round-off"
print("PASS")
