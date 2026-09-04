#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Conduction Helmholtz block of the MHD block preconditioner
(pc_mhd_block.conduction_block).

Same deck as the conduction_theta test: a hot Gaussian electron-temperature
spike over a low-density halo band with a seeded grid-Nyquist ripple, ion
fluid frozen, backward-Euler conduction at the stiff grid diffusion number
conduction_theta dt chi/dx^2 = 2e3. Every Newton solve is then a linear
Helmholtz problem [I + conduction_theta dt (-d/dz chi rho_f d/dz)/rho] with
condition number ~ 4 * 2e3, so unpreconditioned GMRES needs of order a
hundred Krylov iterations per solve while the conduction block, a fixed
two-V-cycle MLMG inverse of the same operator at frozen coefficients,
should need a handful.

Two gates against the _off twin (pc_mhd_block.conduction_block = 0,
otherwise identical):

1. The preconditioner must not change the answer: the final electron
   energy fields of the two runs agree to the Newton tolerance
   (relative_tolerance 1e-8 on a solve that converges quadratically, so
   1e-7 of the field scale is generous), and both runs land on the
   analytic fully-diffused profile with sum(U_e) conserved to round-off
   (the physics gates of the conduction_theta test, repeated here so the
   block cannot pass while breaking the operator it preconditions).

2. The block must earn its keep: total GMRES iterations over the run at
   most 0.35 of the unpreconditioned twin's, and at most 20 per Newton
   iteration on average (measured 185 vs 762 over 12 Newton iterations at
   landing; the offline gate's 2-3 iterations were at a 1e-4 Krylov
   tolerance, this deck asks for 1e-12).

Usage: analysis_mhd_conduction_pc.py <final_plotfile>
       <baseline_final_plotfile> <initial_plotfile>
The baseline is the _off twin's final plotfile
(../test_1d_theta_implicit_mhd_conduction_pc_off/diags/diag000010); its
newton.txt is read from the same directory.
"""

import sys
from pathlib import Path

import numpy as np
import yt

yt.set_log_level(50)


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


def field(data, name):
    return data["boxlib", name].value.ravel()


final_path = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])
initial_path = Path(sys.argv[3])

# keep the datasets alive: the covering grids hold weak references to them
final_ds, final = get_data(str(final_path))
baseline_ds, baseline = get_data(str(baseline_path))
initial_ds, initial = get_data(str(initial_path))

rho_initial = field(initial, "implicit_mhd_mass_density")
rho_final = field(final, "implicit_mhd_mass_density")
energy_initial = field(initial, "implicit_mhd_electron_energy")
energy_final = field(final, "implicit_mhd_electron_energy")
energy_baseline = field(baseline, "implicit_mhd_electron_energy")

# the ion fluid is frozen (evolve_ion_fluid = false)
assert np.array_equal(rho_initial, rho_final)

# --- gate 1a: same converged state as the unpreconditioned twin ---
scale = np.max(np.abs(energy_baseline))
field_deviation = np.max(np.abs(energy_final - energy_baseline)) / scale
print(f"max |U_e(pc) - U_e(off)| / max|U_e| = {field_deviation:.3e}")
assert field_deviation < 1.0e-7

# --- gate 1b: the physics of the conduction_theta backward-Euler variant ---
# conduction is a conservative face flux: sum(U_e) closes to round-off
energy_drift = abs(np.sum(energy_final) - np.sum(energy_initial)) / np.sum(
    energy_initial
)
print(f"sum(U_e) relative drift = {energy_drift:.3e}")
assert energy_drift < 1.0e-9
# backward-Euler conduction at z ~ 8e3 lands on the fully-diffused limit:
# uniform specific internal energy e_inf = sum(U_e)/sum(rho)
e_spec_final = energy_final / rho_final
e_infinity = np.sum(energy_initial) / np.sum(rho_initial)
profile_deviation = np.max(np.abs(e_spec_final - e_infinity)) / e_infinity
print(f"max |e_spec - e_inf| / e_inf = {profile_deviation:.3e}")
assert profile_deviation < 1.0e-9

# --- gate 2: Krylov work ---
# newton.txt columns (one row per step): [0]step [1]time [2]iters
# [3]total_iters (cumulative) [4]norm_abs [5]norm_rel [6]gmres_iters
# [7]gmres_total_iters (cumulative) [8]gmres_last_res
# newton.txt APPENDS across reruns in the same test directory (ctest
# re-executions): read the LAST 10 rows, i.e. the most recent run, whose
# cumulative columns restart at its first step.
number_of_steps = 10
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
# both runs must have converged every solve
for name, h in (("pc", history), ("off", baseline_history)):
    converged = (h[:, 4] <= 1.1e-11) | (h[:, 5] <= 1.1e-8)
    assert np.all(converged), f"{name}: an unconverged Newton solve"
# Measured at landing (K = 2 GSRB V-cycles, gmres.relative_tolerance
# 1e-12): 185 vs 762 over 12 Newton iterations, i.e. 0.24 and 15.4 per
# Newton iteration -- the 12-decade Krylov tolerance, not the block, sets
# the per-solve count (contraction ~0.16 per iteration). Gates sit at
# 1.4x and 1.3x the measured values.
assert gmres_pc <= 0.35 * gmres_off
assert gmres_pc <= 20.0 * newton_iterations
