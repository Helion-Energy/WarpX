#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Row-wise viscous-heating ledger of the dual-energy closure on the RZ
wall decks (the reviewer-proposed gate of the WP5 booking; author
mhd-rev5-viscous, adopted verbatim in its gates).

Run the existing RZ band deck or the RZ no-slip deck with
    implicit_mhd.ion_closure=dual_energy implicit_mhd.dual_energy_sync=0
    implicit_mhd.allow_dual_energy_sync_off=1 implicit_evolve.theta=0.5
    diag1.fields_to_plot=... implicit_mhd_ion_internal_energy ...
and compare, per radial row (z-mean over the run), the internal register's
gain dU_i with the conservative register's implied internal gain
d(E_i - KE). With the E_i work flux and the dissipation register built
from the same face stress and the same cell velocities (viscous_theta =
theta) this is a CELL-WISE identity at theta = 0.5 (exact to solver
tolerance); it is what the RZ metric weights of the deposit and the wall
pairing are visible in (a domain sum cannot see the weights on a uniform
grid: r_face = (r_L + r_R)/2 there).

Modes and gates (measured on the fixed booking):
  band     all rows 0..23 within 1e-3 (fixed booking 8e-5; with the RZ
           weights dropped row 0 = 0.500 and row 11 = 0.961; legacy
           booking: band rows 0.033)
  no_slip  (dt = 1e-8) row 14 -- the wall-adjacent live row -- within
           1e-4 (fixed 1.000000; legacy 0.120); rows 0..13 within 1e-3
           (the PdV work of the developing radial flow is O(dt^2): 6.9e-3
           at dt 5e-8, 4.5e-4 at 1e-8)
implicit_evolve.theta = 0.5 is REQUIRED: at the decks' own theta = 1 the
identity misses the |dm|^2/(2 rho) term (row 11 reads 0.973 for the
correct booking).

Usage:
    analysis_mhd_viscous_heating_rz.py <initial plotfile> <final plotfile> band|no_slip
"""

import sys

import numpy as np
import yt

initial, final, mode = sys.argv[1], sys.argv[2], sys.argv[3]
FIELDS = [
    "implicit_mhd_mass_density",
    "implicit_mhd_ion_energy",
    "implicit_mhd_ion_internal_energy",
    "implicit_mhd_momentum_r",
    "implicit_mhd_momentum_t",
    "implicit_mhd_momentum_z",
]


def load(plotfile):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return {name: grid["boxlib", name].value[:, :, 0] for name in FIELDS}


def kinetic(fields):
    return (
        0.5
        * (
            fields["implicit_mhd_momentum_r"] ** 2
            + fields["implicit_mhd_momentum_t"] ** 2
            + fields["implicit_mhd_momentum_z"] ** 2
        )
        / fields["implicit_mhd_mass_density"]
    )


f0, f1 = load(initial), load(final)
dU = (
    f1["implicit_mhd_ion_internal_energy"] - f0["implicit_mhd_ion_internal_energy"]
).mean(axis=1)
dI = (
    (f1["implicit_mhd_ion_energy"] - kinetic(f1))
    - (f0["implicit_mhd_ion_energy"] - kinetic(f0))
).mean(axis=1)
ratio = dU / dI
if mode == "band":
    checks = [(list(range(24)), 1.0e-3)]
elif mode == "no_slip":
    checks = [([14], 1.0e-4), (list(range(14)), 1.0e-3)]
else:
    raise SystemExit(f"unknown mode {mode}")
for rows, tol in checks:
    worst = max(abs(ratio[i] - 1.0) for i in rows)
    print(
        f"[{mode}] rows {rows[0]}..{rows[-1]}: worst |dU_i/d(E_i-KE) - 1| = "
        f"{worst:.2e} (tol {tol:.0e})"
    )
    assert worst < tol, "row-wise viscous booking off: " + ", ".join(
        f"r{i}={ratio[i]:.5f}" for i in rows
    )
print("PASS")
