#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Wall-corner conduction stiffness of the production formation deck (RZ):
the baseline count of the point-smoothed conduction block and the gate a
better block must beat.

The deck (inputs_test_rz_theta_implicit_mhd_braginskii_oblique_edge) is
the production wall corner isolated to the tensor conduction channel: an
electron column whose axial edge (density 5e19 -> 3.3e17 over a few
cells, temperature 10 -> 0.5 eV over two to three cells) sits next to a
cold conducting wall threaded by an oblique field (|B_r/B_z| = 0.5 at the
wall row), with the production Braginskii clamps (chi_par/chi_perp = 1e4
at the caps, face conduction number 57 at the parallel clamp), caps,
floors, time step and solver settings (theta 1, forcing cap 0.05, probe
1e-8, Newton cap 30, exit 1e-4). newton.txt carries, per step, the Newton
iterations, the GMRES iterations, the exit residual and the free/pinned
split of the admissibility projection.

mode = "baseline" (the point-smoothed MLMG conduction block with the
plain chi-as-diffusivity coefficients, the production preconditioner
before the density weight). Measured at landing (2 ranks): Newton 69,
GMRES 3363 (49 per Newton solve), one step exiting at 1e-2 (a
line-search stagnation), the rest within 1e-4 -- and the block OFF costs
2079 GMRES, i.e. this block is worse than none at a tenfold density
step, which is the finding the deck records. Gates:
 1. at most two steps miss the 1e-4 exit and none exits above 2e-2 (the
    stagnation exits are the production failure mode this deck exists to
    show; the count is recorded, not hidden);
 2. the cumulative Newton and GMRES iterations stay within ceilings of
    90 and 4400 (the measured counts with a 30% margin), so a regression
    of the block or of the Braginskii kernel is caught; the per-step
    GMRES-per-Newton-solve and the pinned counts are printed for the
    convergence table;
 3. the frozen ions leave the density bit-unchanged, the wall and plate
    drains remove electron energy (the r-weighted total decreases), and
    the specific electron energy obeys the maximum principle to within
    a small fraction of the initial contrast (the Dirichlet baths sit at
    the halo temperature, so no cell may leave [e_min, e_max]).

mode = "improved" (a conduction-block upgrade on the same deck; the
registered arm is pc_mhd_block.conduction_density_weight = 1, measured
at landing Newton 35, GMRES 233, 7 per solve, every step within 1e-4;
a direct/assembled block or the cross-term consumer run the same way):
 the physics gates, every step within the 1e-4 exit, and against the
 baseline run: cumulative GMRES at most half the baseline's, Newton not
 above the baseline's, and the same final state to the accuracy the
 baseline itself reached (a preconditioner may not move the answer: the
 tolerance is ten times the baseline's worst exit residual, never below
 1e-6, because the baseline's stagnated steps are only accurate to that).

Usage:
  analysis_mhd_braginskii_oblique_edge.py <initial> <final> baseline
  analysis_mhd_braginskii_oblique_edge.py <initial> <final> improved \
      <baseline_final_plotfile> <baseline_newton.txt>
"""

import sys

import numpy as np
import yt

yt.set_log_level(50)

# constants from inputs_test_rz_theta_implicit_mhd_braginskii_oblique_edge
number_of_cells_r = 32
number_of_cells_z = 56
number_of_steps = 10
cell_size = 1.0 / 128.0
radial_extent = number_of_cells_r * cell_size
live_rows = 24  # rows inside the wall polyline (r < 24 h)
# Newton exit criteria of the deck (relative 1e-4 or absolute 1e-12),
# with a 10% margin on the printed norms.
RELATIVE_TOLERANCE = 1.0e-4
ABSOLUTE_TOLERANCE = 1.0e-12
# Ceilings for the baseline block, from the counts measured at landing
# (Newton 69, GMRES 3363; see the deck header and the campaign report):
# cumulative Newton and GMRES over the 10 steps with a 30% margin, and
# the stagnation exits the baseline is allowed (measured: one at 1e-2).
BASELINE_NEWTON_CEILING = 90
BASELINE_GMRES_CEILING = 4400
BASELINE_MAX_MISSED_STEPS = 2
BASELINE_WORST_EXIT = 2.0e-2
# Maximum-principle excursion as a fraction of the initial contrast.
EXCURSION_CEILING = 2.0e-2
# "improved" mode: GMRES fraction of the baseline, Newton drift and the
# state agreement floor (see the header).
IMPROVED_GMRES_FRACTION = 0.5
IMPROVED_NEWTON_DRIFT = 0
STATE_TOLERANCE_FLOOR = 1.0e-6


def get_data(plotfile):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    energy = np.asarray(grid["boxlib", "implicit_mhd_electron_energy"])[:, :, 0]
    density = np.asarray(grid["boxlib", "implicit_mhd_mass_density"])[:, :, 0]
    assert energy.shape == (number_of_cells_r, number_of_cells_z), energy.shape
    return ds, energy, density


def read_newton(path):
    """The last number_of_steps rows: newton.txt is opened in append mode,
    so a re-run in an uncleaned directory carries the previous rows."""
    rows = np.atleast_2d(np.loadtxt(path))[-number_of_steps:]
    assert rows.shape[0] == number_of_steps and int(rows[-1, 0]) == number_of_steps, (
        rows[:, 0].astype(int).tolist()
    )
    assert int(rows[0, 0]) == 1
    return rows


def report(label, rows):
    iterations = rows[:, 2].astype(int)
    gmres = rows[:, 6].astype(int)
    per_solve = gmres / np.maximum(iterations, 1)
    print(f"{label}: Newton per step      {iterations.tolist()}")
    print(f"{label}: GMRES per step       {gmres.tolist()}")
    print(f"{label}: GMRES per Newton     {np.round(per_solve, 1).tolist()}")
    print(f"{label}: exit relative norm   {[f'{v:.1e}' for v in rows[:, 5]]}")
    print(f"{label}: pinned components    {rows[:, 11].astype(int).tolist()}")
    print(f"{label}: pinned defect        {[f'{v:.1e}' for v in rows[:, 10]]}")
    print(
        f"{label}: totals Newton = {int(iterations.sum())}, GMRES = {int(gmres.sum())}, "
        f"GMRES per Newton solve = {gmres.sum() / iterations.sum():.1f}, "
        f"worst exit relative norm = {rows[:, 5].max():.2e}"
    )
    return int(iterations.sum()), int(gmres.sum())


def missed_steps(rows):
    converged = (rows[:, 4] <= 1.1 * ABSOLUTE_TOLERANCE) | (
        rows[:, 5] <= 1.1 * RELATIVE_TOLERANCE
    )
    return np.flatnonzero(~converged) + 1


mode = sys.argv[3]
assert mode in ("baseline", "improved"), mode
initial_ds, initial_energy, initial_density = get_data(sys.argv[1])
final_ds, final_energy, final_density = get_data(sys.argv[2])
assert float(final_ds.current_time - initial_ds.current_time) > 0.0

# --- the solver record ---
rows = read_newton("diags/newton.txt")
newton_total, gmres_total = report(mode, rows)
missed = missed_steps(rows)
print(f"{mode}: steps missing the 1e-4 exit: {missed.tolist()}")

# --- physics sanity: frozen density, draining wall, maximum principle ---
# Live rows only: the shaped wall's exterior clamp scrapes the 8 masked
# rows to the rigid vacuum image on the first step, after the initial
# plotfile is written, so only the fluid inside the wall is compared.
r = ((np.arange(number_of_cells_r) + 0.5) * cell_size)[:, np.newaxis]
live = slice(0, live_rows)
np.testing.assert_allclose(final_density[live], initial_density[live], rtol=1.0e-12)
initial_total = np.sum(initial_energy[live] * r[live])
final_total = np.sum(final_energy[live] * r[live])
print(
    f"{mode}: r-weighted live electron energy {initial_total:.6e} -> {final_total:.6e} "
    f"(relative change {(final_total - initial_total) / initial_total:.3e})"
)
assert final_total < initial_total, "the wall and plate drains must remove energy"
initial_specific = initial_energy[live] / initial_density[live]
final_specific = final_energy[live] / final_density[live]
e_min = float(np.min(initial_specific))
e_max = float(np.max(initial_specific))
contrast = e_max - e_min
overshoot = max(0.0, float(np.max(final_specific)) - e_max) / contrast
undershoot = max(0.0, e_min - float(np.min(final_specific))) / contrast
print(
    f"{mode}: maximum-principle excursion: overshoot {overshoot:.3e}, "
    f"undershoot {undershoot:.3e} of the initial contrast (ceiling {EXCURSION_CEILING:.0e})"
)
assert max(overshoot, undershoot) < EXCURSION_CEILING, (overshoot, undershoot)

if mode == "baseline":
    # --- gates 1 and 2: the recorded behaviour of the point-smoothed block ---
    assert len(missed) <= BASELINE_MAX_MISSED_STEPS, missed.tolist()
    assert rows[:, 5].max() <= BASELINE_WORST_EXIT, rows[:, 5].max()
    assert newton_total <= BASELINE_NEWTON_CEILING, (newton_total, BASELINE_NEWTON_CEILING)
    assert gmres_total <= BASELINE_GMRES_CEILING, (gmres_total, BASELINE_GMRES_CEILING)
else:
    # --- the upgrade must beat the baseline on the same deck ---
    assert len(missed) == 0, missed.tolist()
    assert len(sys.argv) > 5, "improved mode needs the baseline plotfile and newton.txt"
    _, baseline_energy, _ = get_data(sys.argv[4])
    baseline_rows = read_newton(sys.argv[5])
    baseline_newton, baseline_gmres = report("baseline", baseline_rows)
    scale = float(np.max(np.abs(baseline_energy[live])))
    deviation = float(np.max(np.abs(final_energy[live] - baseline_energy[live]))) / scale
    state_tolerance = max(STATE_TOLERANCE_FLOOR, 10.0 * float(baseline_rows[:, 5].max()))
    print(
        f"improved: max |U_e - U_e(baseline)| / max|U_e| = {deviation:.3e} "
        f"(tolerance {state_tolerance:.1e} from the baseline's worst exit)"
    )
    assert deviation < state_tolerance, (deviation, state_tolerance)
    assert newton_total <= baseline_newton + IMPROVED_NEWTON_DRIFT, (newton_total, baseline_newton)
    print(
        f"improved: cumulative GMRES {gmres_total} vs baseline {baseline_gmres} "
        f"(ratio {gmres_total / baseline_gmres:.3f}, required <= {IMPROVED_GMRES_FRACTION})"
    )
    assert gmres_total <= IMPROVED_GMRES_FRACTION * baseline_gmres, (gmres_total, baseline_gmres)

print("PASS")
