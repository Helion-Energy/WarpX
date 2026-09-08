#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Matrix-free Jacobian probe at a plasma-vacuum edge
(newton.jfnk_epsilon, newton.jfnk_epsilon_mode).

Deck inputs_test_1d_theta_implicit_mhd_probe_edge: a hot core next to a
tenuous, cold edge (300x density step, 30x temperature step) under
Braginskii parallel electron conduction (chi ~ Te^{5/2}/n), ion fluid
frozen, backward Euler. The edge state is ~1e-4 of the core in the solver
norm while the Newton corrections concentrate at the heat front in the
edge, so the GLOBAL probe eps = jfnk_epsilon ||U||/||dU|| is a large
relative perturbation exactly where the nonlinearity is strongest and
the finite-difference Jacobian-vector product degrades to a secant.

Four runs of the same deck, compared here on their Newton work
(newton.txt columns: [2] Newton iterations of the step, [3] cumulative,
[5] exit relative residual, [6] GMRES iterations of the step, [7]
cumulative, [8] last linear residual):

  global6    newton.jfnk_epsilon = 1e-6, mode global (historical default)
  global8    newton.jfnk_epsilon = 1e-8, mode global
  component  newton.jfnk_epsilon = 1e-6, mode component (D = |U| + floor)
  flat       mode component with newton.jfnk_component_floor = 1e12: the
             floor swamps |U_i|, D is a constant multiple of the block
             reference scales, and the component probe reduces to the
             global one EXACTLY (the constant cancels in
             ||D^-1 U||/||D^-1 dU||) -- an implementation check.

Gates:
1. Physics: all four runs land on the same state (electron energy fields
   agree to 1e-6 of the field scale: the SAME nonlinear problem is solved
   to newton.relative_tolerance = 1e-8 each step) and conduction conserves
   sum(U_e) to round-off in every run (periodic domain, conservative
   fluxes).
2. Consistency: flat reproduces global6 step by step -- identical Newton
   and GMRES iteration counts every step and the same final state to
   round-off (1e-12 of the field scale; measured 2e-16).
3. The probe matters: every step of the component run converges within
   newton.max_iterations, its total Newton work is below 0.9 of the
   historical global 1e-6 probe's (measured 0.78: 94 vs 120 over 16
   steps, 5-7 vs 7-8 per step), and it is no worse than the global 1e-8
   probe's within one iteration per step.

Usage: analysis_mhd_probe_edge.py <plotfile of the component run>
         <global6 test dir> <global8 test dir> <flat test dir>
(the test directories are the CTest working directories of the twins,
../test_1d_theta_implicit_mhd_probe_edge etc.; newton.txt APPENDS across
reruns, so only the last max_step rows of each file are read.)
"""

import sys
from pathlib import Path

import numpy as np
import yt

yt.set_log_level(50)

NUMBER_OF_STEPS = 16
FINAL_PLOTFILE = "diags/diag000016"


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


def field(data, name):
    return data["boxlib", name].value.ravel()


def history(test_dir):
    rows = np.atleast_2d(np.loadtxt(Path(test_dir) / "diags" / "newton.txt"))
    rows = rows[-NUMBER_OF_STEPS:]
    assert rows.shape[0] == NUMBER_OF_STEPS, f"{test_dir}: incomplete newton.txt"
    assert rows[0, 0] == 1, f"{test_dir}: newton.txt does not start at step 1"
    return rows


component_plotfile = Path(sys.argv[1])
component_dir = component_plotfile.parent.parent
runs = {
    "global6": Path(sys.argv[2]),
    "global8": Path(sys.argv[3]),
    "component": component_dir,
    "flat": Path(sys.argv[4]),
}

# --- gate 1: same physics in every run ---
datasets = {}
energies = {}
densities = {}
for name, test_dir in runs.items():
    ds_final, final = get_data(str(test_dir / FINAL_PLOTFILE))
    ds_initial, initial = get_data(str(test_dir / "diags/diag000000"))
    datasets[name] = (ds_final, ds_initial)  # keep the datasets alive
    energies[name] = (
        field(initial, "implicit_mhd_electron_energy"),
        field(final, "implicit_mhd_electron_energy"),
    )
    densities[name] = (
        field(initial, "implicit_mhd_mass_density"),
        field(final, "implicit_mhd_mass_density"),
    )

for name in runs:
    rho_initial, rho_final = densities[name]
    energy_initial, energy_final = energies[name]
    # the ion fluid is frozen
    assert np.array_equal(rho_initial, rho_final), f"{name}: density changed"
    drift = abs(np.sum(energy_final) - np.sum(energy_initial)) / np.sum(energy_initial)
    print(f"{name:10s} sum(U_e) relative drift = {drift:.3e}")
    assert drift < 1.0e-9, f"{name}: conduction did not conserve energy"
    # the heat front crossed most of the edge: at least half of the edge
    # cells (rho < 0.01 rho_core) warmed by more than 10x in specific
    # energy (measured: 58 of 70, peak 23.5x)
    e_spec_initial = energy_initial / rho_initial
    e_spec_final = energy_final / rho_final
    edge = rho_initial < 0.01 * np.max(rho_initial)
    heating = e_spec_final[edge] / e_spec_initial[edge]
    heated = int(np.sum(heating > 10.0))
    print(
        f"{name:10s} edge cells heated >10x: {heated} of {int(edge.sum())} "
        f"(peak {np.max(heating):.1f}x)"
    )
    assert heated >= edge.sum() // 2, f"{name}: the heat front did not cross the edge"

reference = energies["global6"][1]
scale = np.max(np.abs(reference))
for name in ("global8", "component", "flat"):
    deviation = np.max(np.abs(energies[name][1] - reference)) / scale
    print(f"max |U_e({name}) - U_e(global6)| / max|U_e| = {deviation:.3e}")
    assert deviation < 1.0e-6, f"{name}: different converged state"

# --- Newton work table ---
histories = {name: history(test_dir) for name, test_dir in runs.items()}
print(
    f"{'run':10s} {'Newton':>7s} {'GMRES':>7s} {'max/step':>9s} "
    f"{'unconverged':>12s} {'GMRES/Newton':>13s}"
)
totals = {}
for name, h in histories.items():
    newton_total = int(h[-1, 3])
    gmres_total = int(h[-1, 7])
    max_per_step = int(np.max(h[:, 2]))
    unconverged = int(np.sum((h[:, 4] > 1.0e-13) & (h[:, 5] > 1.0e-8)))
    totals[name] = (newton_total, gmres_total, unconverged)
    print(
        f"{name:10s} {newton_total:7d} {gmres_total:7d} {max_per_step:9d} "
        f"{unconverged:12d} {gmres_total / newton_total:13.2f}"
    )
print("per-step Newton iterations:")
for name, h in histories.items():
    print(f"  {name:10s} " + " ".join(f"{int(x):2d}" for x in h[:, 2]))

# --- gate 2: the flat-floor component probe IS the global probe ---
# Identical Newton and GMRES counts every step, and the same final state
# to round-off (measured 2e-16 of the field scale, against 1e-14 for the
# genuinely different probes). The exit residuals themselves are
# converged noise (~1e-12 of the step's initial residual) and differ at
# the 1e-4 level between the two arithmetic paths of eps; they are
# printed, not gated.
flat, global6 = histories["flat"], histories["global6"]
assert np.array_equal(flat[:, 2], global6[:, 2]), "flat: Newton counts differ from global6"
assert np.array_equal(flat[:, 6], global6[:, 6]), "flat: GMRES counts differ from global6"
exit_residual_deviation = np.max(
    np.abs(flat[:, 4] - global6[:, 4]) / np.maximum(global6[:, 4], 1.0e-300)
)
print(f"flat vs global6 exit residual deviation = {exit_residual_deviation:.3e}")
flat_state_deviation = np.max(np.abs(energies["flat"][1] - reference)) / scale
print(f"flat vs global6 final state deviation = {flat_state_deviation:.3e}")
assert flat_state_deviation < 1.0e-12

# --- gate 3: the component probe restores Newton's rate ---
component_newton, component_gmres, component_unconverged = totals["component"]
global6_newton = totals["global6"][0]
global8_newton = totals["global8"][0]
assert component_unconverged == 0, "component: an unconverged Newton solve"
print(
    f"Newton total: component {component_newton} vs global 1e-6 "
    f"{global6_newton} (ratio {component_newton / global6_newton:.3f}) vs "
    f"global 1e-8 {global8_newton}"
)
# LANDING MEASUREMENT (16 steps, rtol 1e-10): global 1e-6 120 Newton /
# 2103 GMRES (7-8 per step), global 1e-8 95 / 1664, component 94 / 1647
# (5-7 per step), flat 120 / 2103 = global6 exactly. Gates: the component
# probe within 0.9 of the global 1e-6 count (measured 0.78) and within one
# iteration per step of the global 1e-8 count.
assert component_newton <= 0.9 * global6_newton
assert component_newton <= global8_newton + NUMBER_OF_STEPS
