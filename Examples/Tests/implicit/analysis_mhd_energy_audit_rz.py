#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Global energy audit on an RZ shaped-wall column (the production stack
shape: external field, dielectric stair wall with a thermal BC, dual-energy
ions, two ranks), plus the null-twin identity.

Usage: analysis_mhd_energy_audit_rz.py <energy_audit.txt> <ledger> <theta>
           [<final_plotfile> <twin_final_plotfile> <twin_newton_txt>]

<ledger> is the shaped-wall ledger file (rows every step), or
"absorb:<absorb_ledger.txt>" for a Green's-open column with the absorbing
r_hi fluid boundary: the audit's cumulative fluid export through r_hi
(mass and electron + ion energy, positive outward) must then match the
absorb ledger's cumulative counters at the ledger's rows.

Checks:
1. Every column finite on every row.
2. The audit's own shaped-wall deposition (per block, positive into the
   wall) reproduces the solver's wall ledger: the per-step difference of
   the ledger's cumulative energy column equals wall_e + wall_i to
   round-off, and the ledger's mass column equals the cumulative
   wall_mass. Same reduction on the same accepted theta-state fluxes,
   booked independently -- the sign convention gate.
3. The fluid self-checks fluxw_* are at round-off: every deposited source
   is registered, and the domain + stair export accounts for the whole
   flux divergence (the wall band's frozen cells included).
4. The 'full' closure residual is at round-off relative to the total
   energy on every step, so the row's named terms are complete on the
   production stack shape (external field split, PEC + open + mirror
   boundaries, the wall freeze, floors, the sync).
5. theta = 1/2: theta_diss identically zero; theta = 1: non-negative,
   and positive somewhere when the column carries a magnetic field.
6. Null twin (optional): the run WITHOUT the audit and the run WITH it
   must be byte-identical -- newton.txt identical, every plotfile field
   identical to the last bit (the audit reads the state and restores the
   scratch it touches; it must never change the solution).
"""

import filecmp
import sys
from pathlib import Path

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
absorb_mode = sys.argv[2].startswith("absorb:")
ledger_path = sys.argv[2].split(":", 1)[1] if absorb_mode else sys.argv[2]
ledger_rows = [line.split() for line in open(ledger_path) if not line.startswith("#") and line.strip()]
ledger = np.array([[float(v) for v in row[:3]] for row in ledger_rows])
theta = float(sys.argv[3])
n = len(a["step"])
print(f"{sys.argv[1]}: {n} rows; {'absorb' if absorb_mode else 'wall'} ledger rows: {len(ledger)}")

# 1. finite
for name, values in a.items():
    assert np.all(np.isfinite(values)), f"non-finite column {name}"

w_total = a["W_B_tot"] + a["U_e"] + a["E_i"]
scale = np.max(w_total)
tol = 1.0e-11 * scale
print(f"  W_B_tot = {a['W_B_tot'][-1]:.6e} (resp {a['W_B_resp'][-1]:.6e}, ext {a['W_B_ext'][-1]:.6e}) J; "
      f"U_e = {a['U_e'][-1]:.6e}, E_i = {a['E_i'][-1]:.6e}, KE = {a['KE'][-1]:.6e} J")

if absorb_mode:
    # 2'. r_hi fluid export (positive outward) vs the absorbing-wall ledger's
    #     cumulative counters (rows "step mass energy" at its cadence)
    steps = a["step"].astype(int)
    out_mass = np.cumsum(-a["fluid_in_mass_rhi"])
    out_energy = np.cumsum(-(a["fluid_in_e_rhi"] + a["fluid_in_i_rhi"]))
    for row in ledger:
        idx = np.where(steps == int(row[0]))[0]
        assert len(idx) == 1, f"absorb ledger step {int(row[0])} not audited"
        i = idx[0]
        print(f"  step {int(row[0])}: absorb ledger mass {row[1]:.6e} kg / energy {row[2]:.6e} J; "
              f"audit r_hi export {out_mass[i]:.6e} kg / {out_energy[i]:.6e} J")
        assert abs(out_mass[i] - row[1]) < 1.0e-9 * max(abs(row[1]), 1.0e-300) + 1.0e-30, "r_hi mass export differs from the absorb ledger"
        assert abs(out_energy[i] - row[2]) < 1.0e-9 * max(abs(row[2]), 1.0e-300) + 1.0e-12 * scale, "r_hi energy export differs from the absorb ledger"
    assert np.all(a["wall_e"] == 0.0) and np.all(a["wall_i"] == 0.0), "no stair wall on this deck"
    print(f"  Green's open r_hi: poynt_in_cum {a['poynt_in_cum'][-1]:.6e} J, circuit_in_cum "
          f"{a['circuit_in_cum'][-1]:.6e} J (ext {np.sum(a['circuit_in_ext']):.6e}, plasma {np.sum(a['circuit_in_plasma']):.6e}), "
          f"faraday_defect_cum {a['faraday_defect_cum'][-1]:.6e} J, W_B_tot {a['W_B_tot'][-1]:.6e} J (ext {a['W_B_ext'][-1]:.6e})")
    assert np.max(a["W_B_tot"]) > 0.0 and np.any(a["EJ"] != 0.0), "a magnetized column must exchange energy"
else:
    # 2. wall deposition vs the solver's ledger (rows every step here)
    assert len(ledger) == n, "the wall ledger must have one row per step for this gate"
    ledger_energy_step = np.diff(np.concatenate([[0.0], ledger[:, 2]]))
    audit_wall = a["wall_e"] + a["wall_i"]
    wall_scale = max(np.max(np.abs(ledger[:, 2])), 1.0e-300)
    print(f"  wall energy: ledger cumulative {ledger[-1, 2]:.6e} J, audit cumulative {a['wall_cum'][-1]:.6e} J; "
          f"max per-step mismatch {np.max(np.abs(audit_wall - ledger_energy_step)):.3e} J")
    assert np.all(np.abs(audit_wall - ledger_energy_step) < 1.0e-10 * wall_scale + 1.0e-12 * scale), \
        "audit wall deposition differs from the wall ledger"
    assert abs(np.sum(a["wall_mass"]) - ledger[-1, 1]) < 1.0e-10 * max(abs(ledger[-1, 1]), 1.0e-300) + 1.0e-30, \
        "audit wall mass differs from the wall ledger"

# 3. fluid self-checks
for name in ["fluxw_mass", "fluxw_e", "fluxw_i", "fluxw_ui"]:
    print(f"  max |{name}| = {np.max(np.abs(a[name])):.3e}")
    assert np.all(np.abs(a[name]) < tol), f"{name} above round-off: an unregistered source or a missed face"

# 4. full closure
print(f"  max |resid_full| = {np.max(np.abs(a['resid_full'])):.3e} J (bound {tol:.3e}); "
      f"cumulative {a['resid_full_cum'][-1]:.3e} J")
print(f"  booked residual cumulative {a['resid_booked_cum'][-1]:.3e} J; poynt_in_cum {a['poynt_in_cum'][-1]:.3e}; "
      f"EJ_cum {a['EJ_cum'][-1]:.3e}; exchange_cum {a['exchange_cum'][-1]:.3e}; faraday_defect_cum "
      f"{a['faraday_defect_cum'][-1]:.3e}; sync_Ei_cum {a['sync_Ei_cum'][-1]:.3e}; floor_cum {a['floor_cum'][-1]:.3e}; "
      f"newton_cum {a['newton_cum'][-1]:.3e}; theta_diss_cum {a['theta_diss_cum'][-1]:.3e}")
assert np.all(np.abs(a["resid_full"]) < tol), "full closure residual above round-off"

# 5. theta term (identically zero at theta = 1/2 or without a magnetic field;
#    non-negative and somewhere positive under backward Euler with a field)
if abs(theta - 0.5) < 1.0e-12:
    assert np.all(a["theta_diss"] == 0.0), "theta_diss must vanish identically at theta = 1/2"
else:
    assert np.all(a["theta_diss"] >= 0.0), "the theta term is a non-negative quadrature term at theta = 1"
    if np.max(a["W_B_tot"]) > 0.0:
        assert np.any(a["theta_diss"] > 0.0), "backward Euler must dissipate a magnetized column"
    else:
        assert np.all(a["theta_diss"] == 0.0), "no field, no theta term"

# 6. null twin identity
if len(sys.argv) > 6:
    import yt

    yt.set_log_level(50)
    final_plotfile, twin_plotfile, twin_newton = sys.argv[4], sys.argv[5], sys.argv[6]
    own_newton = str(Path(final_plotfile).parents[1] / "diags/newton.txt")
    assert filecmp.cmp(own_newton, twin_newton, shallow=False), "newton.txt differs from the null twin"
    ds = yt.load(final_plotfile)
    ds_twin = yt.load(twin_plotfile)
    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    grid_twin = ds_twin.covering_grid(0, ds_twin.domain_left_edge, ds_twin.domain_dimensions)
    fields = [f for _, f in ds.field_list]
    for field in fields:
        own = np.asarray(grid["boxlib", field])
        other = np.asarray(grid_twin["boxlib", field])
        assert own.shape == other.shape and np.array_equal(own, other), f"field {field} differs from the null twin"
    print(f"  null twin: newton.txt byte-identical, {len(fields)} plotfile fields identical")
print("PASS")
