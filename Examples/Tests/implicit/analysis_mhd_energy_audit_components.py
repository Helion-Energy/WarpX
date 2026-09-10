#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Ohm's-law component split of the exchange remainder on the 1D line (v5).

Usage: analysis_mhd_energy_audit_components.py <energy_audit.txt> <mode>

The audit decomposes the field-to-fluid exchange remainder exactly,
    exchange_rest2 = ideal_mismatch + hall_work + inertia_work + ohm_rest,
    ideal_mismatch = stagger_mismatch + recon_work + corner_diss_work,
with every term a register of the Ohm assembly dotted with J^theta. On
the periodic 1D line there are no corners and no post-assembly
projections, J is transverse to the gradient, and for the smooth
uniform-density wave with the plain central flux the face induction flux
is EXACTLY paired with the face Maxwell-stress work (ideal_mismatch at
round-off). Each mode makes one component carry the exchange:

ideal      the periodic wave, eta = 0, no Hall, no reconstruction: every
           component at round-off, the exchange itself at round-off.
recon      + median reconstruction and the Rusanov channel dissipation:
           the pairing is NOT exact any more -- the whole exchange defect
           is the ideal pairing defect (exchange = ideal_mismatch to
           round-off), most of it the reconstruction part (recon_work)
           and the rest the plain-mean part (stagger_mismatch, the
           reconstructed states' velocity vs the cell mean).
hall       + include_hall_term: on the line J_z = 0 and J_x, J_y share
           the node, so the Hall EMF's work (J_h x B) . J cancels
           (identically in exact arithmetic; round-off here); the
           closure and every other component are unchanged.
hyper      + plasma_hyper_resistivity: res_hyper alone carries the
           exchange (exchange_rest = res_hyper to round-off, res_hyper >
           0: -eta_H laplacian(J) . J sums to +eta_H |grad J|^2), the v4
           and v5 remainders at round-off.
holmstrom  the density-step wave with the Holmstrom vacuum switch on:
           lorentz_withheld carries the exchange (the switch withholds
           part of the deposited J x B work) up to the small pairing
           defect the density step itself introduces (ideal_mismatch,
           < 1e-4 of sum |EJ|, equal to exchange_rest2 to round-off).
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
scale = np.max(w_total)
tol = 1.0e-11 * scale
S = {k: np.sum(a[k]) for k in ["EJ", "lorentz", "lorentz_unweighted", "lorentz_withheld", "joule_e", "joule_i",
                               "exchange", "exchange_rest", "exchange_rest2", "exchange_rest3", "res_field",
                               "res_hyper", "ideal_edge_work", "ideal_cc_work", "ideal_mismatch", "stagger_mismatch",
                               "recon_work", "corner_diss_work", "hall_work", "inertia_work", "ohm_rest",
                               "poynt_in_zlo", "poynt_in_zhi", "fluid_in_e_zlo", "fluid_in_e_zhi"]}
print(f"{sys.argv[1]} ({mode}): {n} rows; W_total {scale:.4e} J; cumulative: EJ {S['EJ']:.4e}, lorentz {S['lorentz']:.4e}, "
      f"exchange {S['exchange']:.3e}, ideal_edge {S['ideal_edge_work']:.4e}, ideal_cc {S['ideal_cc_work']:.4e}, "
      f"ideal_mismatch {S['ideal_mismatch']:.3e}, recon {S['recon_work']:.3e}, stagger {S['stagger_mismatch']:.3e}, "
      f"hall {S['hall_work']:.3e}, inertia {S['inertia_work']:.3e}, res_hyper {S['res_hyper']:.3e}, "
      f"lorentz_withheld {S['lorentz_withheld']:.3e}, ohm_rest {S['ohm_rest']:.3e}, rest2 {S['exchange_rest2']:.3e}, "
      f"rest3 {S['exchange_rest3']:.3e}; max |resid_full| {np.max(np.abs(a['resid_full'])):.3e} (bound {tol:.3e})")


def roundoff(name):
    assert np.all(np.abs(a[name]) < tol), f"{name} above round-off: max {np.max(np.abs(a[name])):.3e} J (bound {tol:.3e})"


def exactly_zero(name):
    assert np.all(a[name] == 0.0), f"{name} must be exactly zero here: max |.| {np.max(np.abs(a[name])):.3e}"


# common: closure, the Faraday identity, the split's own arithmetic; the
# post-assembly projections touch no current-carrying edge on the line;
# no corners, no inertia
roundoff("resid_full")
roundoff("faraday_defect")
roundoff("exchange_rest3")
roundoff("ohm_rest")
exactly_zero("corner_diss_work")
exactly_zero("inertia_work")
ej_abs = np.sum(np.abs(a["EJ"]))
if mode == "ideal":
    exactly_zero("hall_work")
    for name in ["ideal_mismatch", "recon_work", "stagger_mismatch", "exchange"]:
        roundoff(name)
elif mode == "recon":
    exactly_zero("hall_work")
    assert np.max(np.abs(a["recon_work"])) > 1.0e3 * tol, "reconstruction must move the face EMF off the plain cell mean here"
    assert np.all(np.abs(a["exchange"] - a["ideal_mismatch"]) < tol), "with reconstruction the exchange defect must be the ideal pairing defect alone"
    assert abs(S["recon_work"]) > abs(S["stagger_mismatch"]), "the reconstruction part should dominate the pairing defect"
    print(f"  reconstruction breaks the 1D pairing: exchange = ideal_mismatch = {S['ideal_mismatch']:.4e} J "
          f"({100 * S['ideal_mismatch'] / S['EJ']:.2f} % of EJ), recon_work {S['recon_work']:.4e}, stagger_mismatch {S['stagger_mismatch']:.4e}")
elif mode == "hall":
    roundoff("hall_work")
    for name in ["ideal_mismatch", "recon_work", "stagger_mismatch", "exchange"]:
        roundoff(name)
    print("  Hall on: the Hall EMF's work cancels on the line (J_z = 0, J_x and J_y collocated)")
elif mode == "hyper":
    exactly_zero("hall_work")
    roundoff("ideal_mismatch")
    assert S["res_hyper"] > 1.0e3 * tol and np.all(a["res_hyper"] > 0.0), "the hyper-resistive dissipation must be live and positive"
    assert np.all(np.abs(a["exchange_rest"] - a["res_hyper"]) < tol), "res_hyper alone must carry the exchange"
    roundoff("exchange_rest2")
    print(f"  res_hyper carries the exchange: {100 * S['res_hyper'] / S['exchange']:.4f} % of exchange")
elif mode == "holmstrom":
    exactly_zero("hall_work")
    assert S["lorentz_withheld"] != 0.0 and abs(S["lorentz_withheld"]) > 1.0e3 * tol, "the vacuum switch must withhold work here"
    assert np.all(np.abs(a["exchange_rest2"] - a["ideal_mismatch"]) < tol), "after lorentz_withheld only the pairing defect may remain"
    assert abs(S["ideal_mismatch"]) < 1.0e-4 * ej_abs, "the density step's pairing defect must stay small against sum |EJ|"
    assert abs(S["lorentz_withheld"]) > 100.0 * abs(S["ideal_mismatch"]), "lorentz_withheld must dominate the exchange"
    print(f"  lorentz_withheld carries the exchange: {100 * S['lorentz_withheld'] / S['exchange']:.4f} % of exchange; "
          f"the density step's pairing defect {S['ideal_mismatch']:.3e} J = {S['ideal_mismatch'] / ej_abs:.2e} of sum |EJ|")
else:
    raise SystemExit(f"unknown mode {mode}")
print("PASS")
