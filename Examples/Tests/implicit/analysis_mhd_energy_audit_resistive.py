#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Gates for the Joule and resistive audit columns (rev30 F4).

Usage: analysis_mhd_energy_audit_resistive.py <energy_audit.txt> uniform|boosted

uniform: the periodic circularly polarized Alfven wave with a uniform user
resistivity eta = 1e-4 Ohm m and the Joule deposit on, no vacuum boost.
  - joule_e > 0 on every step (the wave's current heats the electrons);
  - res_boost == 0 exactly (no boost) and res_field == res_user;
  - |res_user - (joule_e + joule_i)| < 1 % of the Joule heat (the field's
    resistive dissipation on the edges vs the cell-centered deposit: the
    edge-vs-cell sampling of J, measured 0.24 % by the reviewer's probe);
  - |exchange_rest| < 1e-12 W_total (the ideal exchange is discretely exact
    in 1D, and the hyper / Holmstrom columns are zero here);
  - res_hyper == 0, lorentz_withheld == 0; resid_full at round-off.
boosted: the same wave on a density step (0.02 rho0 in the outer half) with
the density-keyed vacuum boost D_vac = 1 m^2/s, eta_user = 1e-5 and the
Ohm-current Joule quench (the production mechanism).
  - res_boost > 0 and res_boost > 0.9 exchange (the boost is the exchange
    defect: the reviewer's probe gave 98 %);
  - |exchange_rest2| < 2 % exchange; resid_full at round-off;
  - res_field > res_user > 0.
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
mode = sys.argv[2]
n = len(a["step"])
for name, values in a.items():
    assert np.all(np.isfinite(values)), f"non-finite column {name}"
w_total = a["W_B_tot"] + a["U_e"] + a["E_i"]
tol = 1.0e-11 * np.max(w_total)
joule = np.sum(a["joule_e"] + a["joule_i"])
res_field, res_user, res_boost = np.sum(a["res_field"]), np.sum(a["res_user"]), np.sum(a["res_boost"])
exchange, rest, rest2 = np.sum(a["exchange"]), np.sum(a["exchange_rest"]), np.sum(a["exchange_rest2"])
print(f"{sys.argv[1]} ({mode}): {n} rows; EJ {np.sum(a['EJ']):.4e} J; joule {joule:.4e} J; res_field {res_field:.4e}; "
      f"res_user {res_user:.4e}; res_boost {res_boost:.4e}; exchange {exchange:.4e}; exchange_rest {rest:.3e}; "
      f"exchange_rest2 {rest2:.3e}; res_hyper {np.sum(a['res_hyper']):.3e}; lorentz_withheld {np.sum(a['lorentz_withheld']):.3e}; "
      f"max |resid_full| {np.max(np.abs(a['resid_full'])):.3e} (bound {tol:.3e})")
assert np.all(np.abs(a["resid_full"]) < tol), "full closure residual above round-off"
# the Ohm's-law component split (v5) on the line: the ideal pairing exact, no
# projections, Hall/inertia/corners off
for name in ["ohm_rest", "exchange_rest3"]:
    assert np.all(np.abs(a[name]) < tol), f"{name} above round-off on the line: {np.max(np.abs(a[name])):.3e}"
for name in ["hall_work", "inertia_work", "corner_diss_work"]:
    assert np.all(a[name] == 0.0), f"{name} must be exactly zero here"
ideal_mismatch = np.sum(a["ideal_mismatch"])
if mode == "uniform":
    # uniform density, plain central flux: the face induction flux is exactly
    # paired with the face stress work
    assert np.all(np.abs(a["ideal_mismatch"]) < tol), f"ideal pairing defect above round-off: {np.max(np.abs(a['ideal_mismatch'])):.3e}"
else:
    # the density step breaks the exact pairing at the 1e-5 level of sum |EJ|
    assert abs(ideal_mismatch) < 1.0e-4 * np.sum(np.abs(a["EJ"])), f"the density step's pairing defect {ideal_mismatch:.3e} J is not small"
    print(f"  ideal pairing defect of the density step: {ideal_mismatch:.3e} J = {ideal_mismatch / np.sum(np.abs(a['EJ'])):.2e} of sum |EJ|")
if mode == "uniform":
    assert np.all(a["joule_e"] > 0.0), "the Joule deposit must be live on every step"
    assert np.all(a["res_boost"] == 0.0), "no vacuum boost: res_boost must be exactly zero"
    assert np.all(a["res_field"] == a["res_user"]), "no boost: res_field must equal res_user exactly"
    assert res_user > 0.0 and abs(res_user - joule) < 0.01 * joule, "edge resistive dissipation vs cell Joule deposit differ by more than 1 %"
    assert np.all(np.abs(a["exchange_rest"]) < tol), "non-resistive exchange mismatch above round-off in 1D"
    assert np.all(a["res_hyper"] == 0.0) and np.all(a["lorentz_withheld"] == 0.0)
    print(f"  res_user vs joule: {100 * (res_user - joule) / joule:+.3f} %")
elif mode == "boosted":
    assert res_boost > 0.0 and exchange > 0.0, "the boost must dissipate"
    assert res_boost > 0.9 * exchange, f"res_boost is only {100 * res_boost / exchange:.1f} % of the exchange defect"
    assert abs(rest2) < 0.02 * exchange, f"unnamed remainder {100 * rest2 / exchange:.2f} % of the exchange defect"
    assert res_field > res_user > 0.0
    print(f"  res_boost / exchange = {100 * res_boost / exchange:.2f} %; exchange_rest2 / exchange = {100 * rest2 / exchange:.3f} %; "
          f"joule / res_user = {joule / res_user:.4f}")
else:
    raise SystemExit(f"unknown mode {mode}")
print("PASS")
