#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Exact inverse of the conduction block of the MHD block preconditioner
(pc_mhd_block.conduction_solver = direct | banded), against the
two-V-cycle MLMG block run of the identical deck.

The exact solvers factorize the energy channels of the frozen stacked
Helmholtz -- rows emitted from the same cell and face coefficients the
MLMG operator applies -- so a run with them differs from the MLMG run
only through the preconditioner. Three gates:

1. Same converged state: every listed field of the final plotfile agrees
   with the reference run's to the Newton tolerance (1e-7 of the field
   scale; the decks converge Newton to 1e-8 .. 1e-10 relative).
2. Both runs converged every Newton solve (newton.txt against the
   tolerances read from each run's warpx_used_inputs), and the exact run
   took no more Newton iterations than the reference plus one.
3. Krylov work: the exact run's total GMRES iterations are at most
   <gmres_ratio_max> times the reference's. An exact inverse of the
   conduction rows makes every energy row of the preconditioned Jacobian
   the identity at frozen coefficients, so the count must not grow; the
   gate value is set per deck from the measured drop.

The runs with pc_mhd_block.conduction_validate_assembly = 1 additionally
assert INSIDE the solver, at every preconditioner update, that the
assembled rows reproduce MLMG::apply of the stacked operator to roundoff
(1e-12 of the operator scale) and, on single-rank runs, that the device-
and host-assembled values agree bitwise (<= 4 ULP); a mismatch aborts the
run before this script sees it.

Usage: analysis_mhd_conduction_pc_exact.py <final_plotfile>
       <reference_final_plotfile> <number_of_steps> <gmres_ratio_max>
       <field> [<field> ...]
The reference run's newton.txt and warpx_used_inputs are read from the
directory two levels above its plotfile (../<reference test>/diags/...).
"""

import re
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


def newton_tolerances(run_directory):
    """(absolute, relative) Newton tolerances of a run from its
    warpx_used_inputs (plain numbers in the conduction decks)."""
    text = (run_directory / "warpx_used_inputs").read_text()
    values = {}
    for name in ("absolute_tolerance", "relative_tolerance"):
        match = re.search(
            rf"^newton\.{name}\s*=\s*([-+0-9.eE]+)", text, flags=re.MULTILINE
        )
        assert match is not None, f"newton.{name} missing from {run_directory}"
        values[name] = float(match.group(1))
    return values["absolute_tolerance"], values["relative_tolerance"]


def history(run_directory, number_of_steps):
    # newton.txt APPENDS across reruns in the same test directory: read the
    # LAST number_of_steps rows, i.e. the most recent run, whose cumulative
    # columns restart at its first step.
    rows = np.atleast_2d(np.loadtxt(run_directory / "diags" / "newton.txt"))
    rows = rows[-number_of_steps:]
    assert rows.shape[0] == number_of_steps, rows.shape
    assert int(rows[0, 0]) == 1, "the last rows do not start at step 1"
    return rows


final_path = Path(sys.argv[1])
reference_path = Path(sys.argv[2])
number_of_steps = int(sys.argv[3])
gmres_ratio_max = float(sys.argv[4])
fields = sys.argv[5:]
assert fields, "list at least one field to compare"

# keep the datasets alive: the covering grids hold weak references to them
final_ds, final = get_data(str(final_path))
reference_ds, reference = get_data(str(reference_path))

# --- gate 1: same converged state as the MLMG-block run ---
for name in fields:
    a = final["boxlib", name].value.ravel()
    b = reference["boxlib", name].value.ravel()
    scale = np.max(np.abs(b))
    deviation = np.max(np.abs(a - b)) / scale
    print(f"{name}: max |exact - mlmg| / max|mlmg| = {deviation:.3e}")
    assert deviation < 1.0e-7, name

# --- gate 2: convergence and Newton work ---
run_directory = Path("diags").resolve().parent
reference_directory = reference_path.resolve().parent.parent
# newton.txt columns (one row per step): [0]step [1]time [2]iters
# [3]total_iters (cumulative) [4]norm_abs [5]norm_rel [6]gmres_iters
# [7]gmres_total_iters (cumulative) [8]gmres_last_res
exact = history(run_directory, number_of_steps)
mlmg = history(reference_directory, number_of_steps)
for label, rows, directory in (
    ("exact", exact, run_directory),
    ("mlmg", mlmg, reference_directory),
):
    absolute, relative = newton_tolerances(directory)
    converged = (rows[:, 4] <= 1.1 * absolute) | (rows[:, 5] <= 1.1 * relative)
    assert np.all(converged), f"{label}: an unconverged Newton solve"

newton_exact = int(exact[-1, 3])
newton_mlmg = int(mlmg[-1, 3])
gmres_exact = int(exact[-1, 7])
gmres_mlmg = int(mlmg[-1, 7])
print(
    f"Newton iterations {newton_exact} (exact) vs {newton_mlmg} (mlmg); GMRES "
    f"iterations {gmres_exact} vs {gmres_mlmg} (ratio {gmres_exact / gmres_mlmg:.3f}, "
    f"gate {gmres_ratio_max}); GMRES per Newton iteration "
    f"{gmres_exact / newton_exact:.2f} vs {gmres_mlmg / newton_mlmg:.2f}"
)
assert newton_exact <= newton_mlmg + 1, (newton_exact, newton_mlmg)

# --- gate 3: Krylov work ---
assert gmres_exact <= gmres_ratio_max * gmres_mlmg, (gmres_exact, gmres_mlmg)
