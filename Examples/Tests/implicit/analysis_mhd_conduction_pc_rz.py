#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Conduction Helmholtz block of the MHD block preconditioner, RZ twin.

The braginskii_spreading problem (hot electron spot in a strongly
magnetized uniform Bz, frozen ions, quiescent electromagnetics) run
backward-Euler in the conduction stage at an axial conduction number of
about 19 with chi_perp/chi_par ~ 1.6e-8, on two ranks. Gates against the
_off twin (pc_mhd_block.conduction_block = 0, otherwise identical):

1. The preconditioner must not change the answer: final electron energy
   fields agree to the Newton tolerance, and the r-weighted electron
   energy is conserved to round-off in both runs (the tensor flux is a
   conservative face flux; periodic z, no-flux r walls).

2. The block must earn its keep on an anisotropic, cylindrical, two-rank
   problem: total GMRES iterations at most 0.6 of the unpreconditioned
   twin's and at most 17 per Newton iteration on average (measured 234 vs
   918 over 18 Newton iterations at landing). The bound is looser than the
   1D pair's because full coarsening with a point smoother is known to
   degrade at per-direction anisotropy above 1e2 (the offline gate
   measured 13 vs 74 on the production analog; semicoarsening, which
   restores 3, is a follow-on).

Usage: analysis_mhd_conduction_pc_rz.py <final_plotfile>
       <baseline_final_plotfile> <initial_plotfile>
"""

import sys
from pathlib import Path

import numpy as np
import yt

yt.set_log_level(50)

number_of_cells_r = 24
number_of_cells_z = 48
radial_extent = 0.6


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


def field(data, name):
    values = np.squeeze(data["boxlib", name].value)
    assert values.shape == (number_of_cells_r, number_of_cells_z), values.shape
    return values


final_path = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])
initial_path = Path(sys.argv[3])

# keep the datasets alive: the covering grids hold weak references to them
final_ds, final = get_data(str(final_path))
baseline_ds, baseline = get_data(str(baseline_path))
initial_ds, initial = get_data(str(initial_path))

energy_initial = field(initial, "implicit_mhd_electron_energy")
energy_final = field(final, "implicit_mhd_electron_energy")
energy_baseline = field(baseline, "implicit_mhd_electron_energy")

# --- gate 1a: same converged state as the unpreconditioned twin ---
scale = np.max(np.abs(energy_baseline))
field_deviation = np.max(np.abs(energy_final - energy_baseline)) / scale
print(f"max |U_e(pc) - U_e(off)| / max|U_e| = {field_deviation:.3e}")
assert field_deviation < 1.0e-7

# --- gate 1b: r-weighted electron energy conserved in both runs ---
cell_size_r = radial_extent / number_of_cells_r
radius = (np.arange(number_of_cells_r) + 0.5) * cell_size_r
weights = radius[:, None]
total_initial = np.sum(weights * energy_initial)
for name, values in (("pc", energy_final), ("off", energy_baseline)):
    drift = abs(np.sum(weights * values) - total_initial) / total_initial
    print(f"{name}: r-weighted sum(U_e) relative drift = {drift:.3e}")
    assert drift < 1.0e-9

# --- gate 2: Krylov work ---
# newton.txt columns (one row per step): [0]step [1]time [2]iters
# [3]total_iters (cumulative) [4]norm_abs [5]norm_rel [6]gmres_iters
# [7]gmres_total_iters (cumulative) [8]gmres_last_res
# newton.txt APPENDS across reruns in the same test directory (ctest
# re-executions): read the LAST 4 rows, i.e. the most recent run, whose
# cumulative columns restart at its first step.
number_of_steps = 4
history = np.atleast_2d(np.loadtxt("diags/newton.txt"))[-number_of_steps:]
baseline_history = np.atleast_2d(
    np.loadtxt(baseline_path.parent.parent / "diags" / "newton.txt")
)[-number_of_steps:]
assert history.shape[0] == baseline_history.shape[0] == number_of_steps
assert history[0, 0] == baseline_history[0, 0] == 1  # first step of the run

newton_iterations = history[-1, 3]
gmres_pc = history[-1, 7]
gmres_off = baseline_history[-1, 7]
print(
    f"Newton iterations {int(newton_iterations)}; GMRES iterations with the "
    f"conduction block {int(gmres_pc)} vs {int(gmres_off)} without "
    f"(ratio {gmres_pc / gmres_off:.3f})"
)
for name, h in (("pc", history), ("off", baseline_history)):
    converged = (h[:, 4] <= 1.1e-12) | (h[:, 5] <= 1.1e-10)
    assert np.all(converged), f"{name}: an unconverged Newton solve"
# Measured at landing: 234 vs 918 over 18 Newton iterations (0.25 and 13
# per Newton iteration at gmres.relative_tolerance 1e-10, per-direction
# anisotropy 6e7, two ranks). Gates at 2.4x and 1.3x the measured values.
assert gmres_pc <= 0.6 * gmres_off
assert gmres_pc <= 17.0 * newton_iterations
