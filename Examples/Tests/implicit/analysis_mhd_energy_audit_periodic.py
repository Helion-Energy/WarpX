#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Global energy audit on a closed periodic system (no walls, no drive).

Usage: analysis_mhd_energy_audit_periodic.py <energy_audit.txt> <theta> [<half_dt_energy_audit.txt>]

Checks, on every row:
1. Every column is finite; boundary inflows (Poynting, fluid) are exactly
   zero on the periodic line; the wall, eater, injection and pedestal
   columns are exactly zero (none of those mechanisms is active).
2. The 'full' closure residual -- the step's total-energy change minus
   every named term -- is at round-off: |resid_full| < 1e-11 x W_total
   per step and cumulatively (a missing or mis-signed term shows up at the
   size of that term, orders of magnitude above this bound).
3. The Faraday defect (sum B^theta . dB/mu0 + dt(poynt_out + E.J)) is at
   round-off: on a periodic Yee mesh the curl pair is exactly adjoint
   under the dual-volume weights, so the field's energy change IS -dt E.J
   up to the theta term.
4. The fluid self-checks fluxw_* (sum rhs dV/theta vs the export + the
   registered sources) are at round-off: every source the kernel deposits
   is registered.
5. theta = 1/2: theta_diss is identically zero. theta = 1: theta_diss > 0
   on every step (backward Euler dissipates), and when the half-dt twin
   is given its cumulative theta_diss over the SAME time interval is
   0.5 x this run's (the per-step term is O(dt^2), the number of steps
   O(1/dt)), within 25 % (the wave's damping changes the amplitude the
   twins see at second order).
6. The 'booked' residual (no scheme terms subtracted) is NOT at round-off
   on the closed system whenever the scheme terms are: it equals
   -(theta_diss) + faraday_defect - exchange + pw_pair - drain_ei +
   newton_e + newton_i + floors + sync, i.e. the two residuals differ by
   exactly the named scheme terms (an arithmetic identity of the row).
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
    assert columns is not None, f"{path}: no '# columns:' header"
    data = np.array(rows)
    assert data.ndim == 2 and data.shape[1] == len(columns), (
        f"{path}: {data.shape} vs {len(columns)} columns"
    )
    return {name: data[:, i] for i, name in enumerate(columns)}


path = sys.argv[1]
theta = float(sys.argv[2])
twin = sys.argv[3] if len(sys.argv) > 3 else None

a = read_audit(path)
n = len(a["step"])
print(f"{path}: {n} rows, theta = {theta}")

# 1. finite; boundary and inactive columns exactly zero
for name, values in a.items():
    assert np.all(np.isfinite(values)), f"non-finite column {name}"
for name in [
    "poynt_in", "poynt_in_rhi", "poynt_in_zlo", "poynt_in_zhi",
    "fluid_in_e_rhi", "fluid_in_e_zlo", "fluid_in_e_zhi",
    "fluid_in_i_rhi", "fluid_in_i_zlo", "fluid_in_i_zhi",
    "fluid_in_mass_rhi", "fluid_in_mass_zlo", "fluid_in_mass_zhi",
    "wall_e", "wall_i", "wall_mass", "eater_e", "eater_i", "eater_mass",
    "inject_e", "inject_i", "inject_mass", "U_e_bg", "U_i_bg",
    "cov_shift_pw_e", "cov_shift_pdv_ui", "drain_ei", "drain_ui",
    "ledger_wall_energy_cum", "ledger_eater_energy_cum",
]:
    assert np.all(a[name] == 0.0), f"{name} must be exactly zero here: {a[name]}"

w_total = a["W_B_tot"] + a["U_e"] + a["E_i"]
scale = np.max(w_total)
print(f"  W_total = {w_total[-1]:.6e} J; dW_total per step (max |.|) = {np.max(np.abs(a['dW_total'])):.3e} J")

# 2. full closure at round-off
tol = 1.0e-11 * scale
print(f"  max |resid_full| = {np.max(np.abs(a['resid_full'])):.3e} J (bound {tol:.3e}); "
      f"cumulative {a['resid_full_cum'][-1]:.3e} J")
assert np.all(np.abs(a["resid_full"]) < tol), "full closure residual above round-off"
assert abs(a["resid_full_cum"][-1]) < n * tol, "cumulative full residual above round-off"

# 3. Faraday defect at round-off (periodic adjoint curl pair)
print(f"  max |faraday_defect| = {np.max(np.abs(a['faraday_defect'])):.3e} J")
assert np.all(np.abs(a["faraday_defect"]) < tol), "Faraday defect above round-off on a periodic line"

# 4. fluid self-checks at round-off
for name in ["fluxw_mass", "fluxw_e", "fluxw_i", "fluxw_ui"]:
    print(f"  max |{name}| = {np.max(np.abs(a[name])):.3e}")
    assert np.all(np.abs(a[name]) < tol), f"{name} above round-off: an unregistered source"

# 5. theta term
if abs(theta - 0.5) < 1.0e-12:
    assert np.all(a["theta_diss"] == 0.0), "theta_diss must vanish identically at theta = 1/2"
    print("  theta = 1/2: theta_diss identically zero")
else:
    assert np.all(a["theta_diss"] > 0.0), "backward Euler must dissipate on every step"
    print(f"  theta = {theta}: cumulative theta_diss = {a['theta_diss_cum'][-1]:.6e} J "
          f"({a['theta_diss_cum'][-1] / scale:.3e} of W_total)")
    if twin is not None:
        b = read_audit(twin)
        assert abs(b["time"][-1] - a["time"][-1]) < 1.0e-12 * a["time"][-1], "twins must end at the same time"
        ratio = b["theta_diss_cum"][-1] / a["theta_diss_cum"][-1]
        print(f"  half-dt twin: {len(b['step'])} rows, cumulative theta_diss ratio = {ratio:.4f} (expected 0.5)")
        assert 0.375 < ratio < 0.625, "theta dissipation does not scale linearly with dt"
        assert np.all(np.abs(b["resid_full"]) < 1.0e-11 * np.max(b["W_B_tot"] + b["U_e"] + b["E_i"])), \
            "twin full closure residual above round-off"

# 6. the two residuals differ by exactly the named scheme terms
named = (-a["theta_diss"] + a["faraday_defect"] - a["exchange"] + a["pw_pair"] -
         a["drain_ei"] + a["newton_e"] + a["newton_i"] + a["floor_e"] + a["floor_Ei"] +
         a["sync_Ei"] + a["post_e"] + a["post_i"] + a["fluxw_e"] + a["fluxw_i"])
identity = a["resid_booked"] - a["resid_full"] - named
assert np.all(np.abs(identity) < 1.0e-12 * scale), "row identity between the two residuals broken"
print(f"  booked residual (scheme terms unsubtracted): max |.| = {np.max(np.abs(a['resid_booked'])):.3e} J, "
      f"cumulative {a['resid_booked_cum'][-1]:.3e} J; exchange_cum = {a['exchange_cum'][-1]:.3e}, "
      f"newton_cum = {a['newton_cum'][-1]:.3e}")
print("PASS")
