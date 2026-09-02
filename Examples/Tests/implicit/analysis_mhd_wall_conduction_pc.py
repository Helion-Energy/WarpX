#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Wall thermal / z-end conduction preconditioner-row test (RZ MHD).

Two residual terms drain the wall-adjacent energy rows -- the shaped-wall
interface drain (implicit_mhd.wall_thermal_bc reservoir modes) and the
z-end conductive exchange (implicit_mhd.z_wall_conduction) -- and both
are, at frozen coefficients, a plain POSITIVE diagonal theta dt D on the
interior cell's energy row.  pc_mhd_block carried no row for either until
implicit_mhd.wall_conduction_pc_rows, so those rows were preconditioned
as identities against a residual diagonal measured at 57 on the
production formation deck under wall_conduction_scale = parallel.

mode = "off" (wall_conduction_pc_rows = 0): records the row-less
baseline's Newton and cumulative GMRES cost.

mode = "on" (the headline): asserts, against the OFF twin,
 1. the converged state is IDENTICAL to solver tolerance -- the rows are
    a preconditioner and the residual is bit-unchanged by switching them
    on, so the physics may not move;
 2. Newton takes the same number of iterations (the comparison is a
    LINEAR-solver comparison, not a nonlinear one);
 3. the cumulative GMRES iteration count is STRICTLY below the twin's,
    with margin.

The in-run round-off gates on the emitted rows themselves live in the
solver (implicit_mhd.wall_conduction_validate_rows and
..._validate_rows_reference, which abort); this script grades what the
rows are FOR.

Usage:
  analysis_mhd_wall_conduction_pc.py <diag> <newton.txt> off
  analysis_mhd_wall_conduction_pc.py <diag> <newton.txt> on \
      <off_diag> <off_newton.txt>
"""

import sys

import numpy as np
import yt

yt.set_log_level(50)

FIELDS = (
    "implicit_mhd_mass_density",
    "implicit_mhd_electron_energy",
    "implicit_mhd_ion_energy",
)

# The rows must buy at least this fraction of the baseline's linear
# iterations (measured on this deck: 497 vs 602, a 0.826 ratio).
MAX_GMRES_RATIO = 0.95
# Newton must be unchanged: same residual, only a different
# preconditioner.  A one-iteration drift would still be honest, but on
# this deck the counts match exactly.
MAX_NEWTON_DRIFT = 1
# The converged states agree to the nonlinear solve's own tolerance, not
# to round-off: the two runs take different Krylov paths.
STATE_TOLERANCE = 1.0e-8


def read_newton(path):
    rows = np.loadtxt(path, ndmin=2)
    # newton.diagnostic_file is APPENDED across runs in the same working
    # directory (a local re-run, not CI): keep only the last contiguous
    # block of increasing step numbers.
    starts = np.nonzero(np.diff(rows[:, 0]) <= 0)[0]
    if starts.size:
        rows = rows[starts[-1] + 1 :]
    assert len(rows) >= 3, f"expected at least 3 recorded steps, got {len(rows)}"
    newton_total = int(rows[-1, 3])
    gmres_total = int(rows[-1, 7])
    assert newton_total > 0 and gmres_total > 0
    return newton_total, gmres_total


def read_state(plotfile):
    dataset = yt.load(plotfile)
    data = dataset.covering_grid(
        level=0,
        left_edge=dataset.domain_left_edge,
        dims=dataset.domain_dimensions,
    )
    return {f: np.squeeze(data["boxlib", f].value) for f in FIELDS}


diag = sys.argv[1]
newton_file = sys.argv[2]
mode = sys.argv[3]
assert mode in ("on", "off"), f"unknown mode {mode}"

newton_total, gmres_total = read_newton(newton_file)
print(
    f"mode = {mode}: cumulative Newton = {newton_total}, "
    f"cumulative GMRES = {gmres_total}"
)

if mode == "on":
    assert len(sys.argv) > 5, "on mode needs the wall_conduction_pc_rows = 0 twin"
    reference_newton, reference_gmres = read_newton(sys.argv[5])
    print(
        f"rows ON {gmres_total} vs OFF {reference_gmres} cumulative GMRES "
        f"-> ratio {gmres_total / reference_gmres:.3f}"
    )

    # 1. the physics must not move
    mine = read_state(diag)
    theirs = read_state(sys.argv[4])
    for name, values in mine.items():
        scale = max(float(np.max(np.abs(theirs[name]))), 1.0e-300)
        deviation = float(np.max(np.abs(values - theirs[name]))) / scale
        print(f"  {name}: max relative difference vs the OFF twin = {deviation:.3e}")
        assert deviation < STATE_TOLERANCE, (
            f"{name} moved by {deviation:.3e} when the preconditioner rows "
            "were switched on: the rows must not change the converged state"
        )

    # 2. same nonlinear work
    assert abs(newton_total - reference_newton) <= MAX_NEWTON_DRIFT, (
        f"Newton took {newton_total} iterations against the twin's "
        f"{reference_newton}: this is meant to be a linear-solver comparison"
    )

    # 3. strictly fewer linear iterations, with margin
    assert gmres_total < reference_gmres, (
        f"the wall conduction preconditioner rows did not reduce the "
        f"cumulative GMRES count ({gmres_total} vs {reference_gmres})"
    )
    assert gmres_total <= MAX_GMRES_RATIO * reference_gmres, (
        f"the wall conduction preconditioner rows bought only "
        f"{1.0 - gmres_total / reference_gmres:.1%} of the baseline's linear "
        f"iterations (required at least {1.0 - MAX_GMRES_RATIO:.0%})"
    )

print(f"wall conduction preconditioner-row test ({mode}) PASSED")
