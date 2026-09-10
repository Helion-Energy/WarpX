#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Ohm's-law component split on a magnetized, flowing RZ column with the
production flux stack (central, median, Rusanov, backward Euler) -- the
case the collocated 1D line cannot test (v5; the rev30b asks).

Usage: analysis_mhd_energy_audit_central.py <energy_audit.txt> [<energy_audit_fine.txt>]

1. closure (resid_full), the Faraday defect and the split's own
   arithmetic (exchange_rest3) at round-off; E.J live;
2. under the central flux the UCT corner dissipation is exactly zero,
   Hall and inertia are off (exactly zero columns), and the work of the
   post-assembly PEC projection (ohm_rest) is negligible (the current
   vanishes at the wall like exp(-16));
3. the reduction weights (header): the FieldEnergy dual volumes of the
   cell-centered-in-r staggering (E_r) sum to the domain volume exactly;
   the nodal-in-r ones (E_theta, E_z) agree with each other and exceed it
   by exactly pi dr^2 L_z / 4 (one axis disk), i.e. V/(4 (sum - V)) is the
   square of the integer radial cell count;
4. with a twin at twice the resolution (same dt, same steps): the RZ
   pairing defect |ideal_mismatch| relative to |EJ| must fall (first order
   at least; the order is printed), and so must its plain-mean part
   (stagger_mismatch), which is where an axis/r-weight inconsistency of
   the edge sums against the cell-centered stress work would show.
"""

import sys

import numpy as np


def read_audit(path):
    columns = None
    header = {}
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                if line.startswith("# columns:"):
                    columns = line[len("# columns:"):].split()
                elif "domain volume = " in line:
                    header["domain_volume"] = float(line.split("domain volume = ")[1].split()[0])
                elif line.startswith("# Weights:"):
                    for key in ["E_r", "E_theta", "E_z"]:
                        header[key] = float(line.split(key + " ")[1].split()[0].rstrip(","))
                continue
            values = line.split()
            if values:
                rows.append([float(v) for v in values])
    data = np.array(rows)
    return {name: data[:, i] for i, name in enumerate(columns)}, header


def report(path):
    a, h = read_audit(path)
    n = len(a["step"])
    for name, values in a.items():
        assert np.all(np.isfinite(values)), f"non-finite column {name}"
    scale = np.max(a["W_B_tot"] + a["U_e"] + a["E_i"])
    tol = 1.0e-11 * scale
    ej = np.sum(np.abs(a["EJ"]))
    print(f"{path}: {n} rows; W_total {scale:.4e} J; sum |EJ| {ej:.4e} J; EJ_cum {a['EJ_cum'][-1]:.4e}; "
          f"lorentz {np.sum(a['lorentz']):.4e}; ideal_edge {np.sum(a['ideal_edge_work']):.4e}; "
          f"ideal_cc {np.sum(a['ideal_cc_work']):.4e}; ideal_mismatch {a['ideal_mismatch_cum'][-1]:.4e}; "
          f"stagger {a['stagger_mismatch_cum'][-1]:.4e}; recon {a['recon_work_cum'][-1]:.4e}; "
          f"corner_diss {a['corner_diss_work_cum'][-1]:.3e}; ohm_rest {a['ohm_rest_cum'][-1]:.3e}; "
          f"max |resid_full| {np.max(np.abs(a['resid_full'])):.3e} (bound {tol:.3e}); "
          f"max |faraday_defect| {np.max(np.abs(a['faraday_defect'])):.3e}; max |exchange_rest3| {np.max(np.abs(a['exchange_rest3'])):.3e}")
    # 1.
    assert np.all(np.abs(a["resid_full"]) < tol), "full closure residual above round-off"
    assert np.all(np.abs(a["faraday_defect"]) < 1.0e-12 * scale), "Faraday defect above round-off"
    assert np.all(np.abs(a["exchange_rest3"]) < tol), "the component split's arithmetic identity broken"
    assert ej > 1.0e6 * tol, "E.J is not live on this deck"
    # 2.
    assert np.all(a["corner_diss_work"] == 0.0), "the UCT corner dissipation must vanish exactly under the central flux"
    assert np.all(a["hall_work"] == 0.0) and np.all(a["inertia_work"] == 0.0), "Hall and inertia are off here"
    assert abs(a["ohm_rest_cum"][-1]) < 1.0e-4 * ej, f"post-assembly projection work {a['ohm_rest_cum'][-1]:.3e} J is not negligible against sum |EJ| {ej:.3e} J"
    # 3.
    v = h["domain_volume"]
    assert abs(h["E_r"] - v) < 1.0e-12 * v, "the cell-centered-in-r dual volumes must sum to the domain volume"
    assert abs(h["E_theta"] - h["E_z"]) < 1.0e-12 * v, "the two nodal-in-r staggerings must carry the same total weight"
    excess = h["E_theta"] - v
    assert excess > 0.0, "a nodal-in-r staggering must exceed the domain volume by one axis disk"
    nr_implied = np.sqrt(v / (4.0 * excess))
    print(f"  weights: E_r = V = {v:.10e}; E_theta = E_z = V + {excess:.4e} (= pi dr^2 L_z/4: implied nr {nr_implied:.8f})")
    assert abs(nr_implied - round(nr_implied)) < 1.0e-6, "the nodal-in-r excess is not exactly one axis disk"
    rel_mismatch = abs(a["ideal_mismatch_cum"][-1]) / ej
    rel_stagger = abs(a["stagger_mismatch_cum"][-1]) / ej
    rel_recon = abs(a["recon_work_cum"][-1]) / ej
    print(f"  ideal_mismatch / sum|EJ| = {rel_mismatch:.4e}; stagger part {rel_stagger:.4e}; reconstruction part {rel_recon:.4e}")
    return a, rel_mismatch, rel_stagger, rel_recon


a_c, m_c, s_c, r_c = report(sys.argv[1])
if len(sys.argv) > 2:
    a_f, m_f, s_f, r_f = report(sys.argv[2])
    assert abs(a_f["time"][-1] - a_c["time"][-1]) < 1.0e-12 * a_c["time"][-1], "the twins must end at the same time"
    ratio_m = m_c / max(m_f, 1.0e-300)
    ratio_s = s_c / max(s_f, 1.0e-300)
    print(f"  convergence coarse/fine: ideal_mismatch ratio {ratio_m:.3f} (order {np.log2(max(ratio_m, 1e-300)):.2f}); "
          f"stagger_mismatch ratio {ratio_s:.3f} (order {np.log2(max(ratio_s, 1e-300)):.2f}); "
          f"recon ratio {r_c / max(r_f, 1e-300):.3f}")
    assert ratio_m > 1.5, "the RZ pairing defect does not converge with resolution"
    assert ratio_s > 1.5, "the plain-mean (staggering / r-weight / axis) part does not converge with resolution"
print("PASS")
