#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Preconditioned twin of the stiff RZ vacuum-resistive test.

Physics: reruns the baseline's structural assertions unchanged, and
additionally requires the final fields to match the unpreconditioned
baseline run (the test dependency) to solver roundoff: right
preconditioning changes the Krylov space, never the converged answer
(measured field differences are ~1e-19 T against Newton tolerances of
1e-11 relative).

Solver: the cumulative GMRES iteration count must be STRICTLY below the
baseline run's, with margin: the Chebyshev resistive block was measured
to cut this deck from ~1570 to ~28 cumulative iterations.
"""

import os
import re
import runpy
from pathlib import Path

import numpy as np
import yt

# Same structural physics assertions as the unpreconditioned baseline.
runpy.run_path(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "analysis_mhd_vacuum_resistive_rz.py",
    ),
    run_name="__main__",
)

# The preconditioner selection is part of the regression contract.
used_inputs = Path("warpx_used_inputs").read_text()
match = re.search(r"^jacobian\.pc_type\s*=\s*(\S+)", used_inputs, re.MULTILINE)
assert match is not None and match.group(1) == "pc_mhd_block"

baseline_directory = "../test_rz_theta_implicit_mhd_vacuum_resistive"


def get_fields(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return {name: data["boxlib", name].value for name in ("Br", "Bt", "Bz")}


mine = get_fields("diags/diag000002")
baseline = get_fields(os.path.join(baseline_directory, "diags/diag000002"))
for name in ("Br", "Bt", "Bz"):
    np.testing.assert_allclose(mine[name], baseline[name], rtol=0.0, atol=1.0e-14)

# Cumulative GMRES iterations, this run vs the unpreconditioned baseline
# (column 7 of the Newton diagnostic file is the running total).
mine_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
baseline_history = np.atleast_2d(
    np.loadtxt(os.path.join(baseline_directory, "diags/newton.txt"))
)
pc_iterations = int(mine_history[-1, 7])
baseline_iterations = int(baseline_history[-1, 7])
print(
    f"cumulative GMRES iterations: pc-on = {pc_iterations}, "
    f"pc-off = {baseline_iterations}"
)
assert pc_iterations < baseline_iterations
# Preserve most of the measured ~1570 -> ~28 reduction, with margin for
# backend-dependent matrix-free roundoff.
assert 10 * pc_iterations <= baseline_iterations
