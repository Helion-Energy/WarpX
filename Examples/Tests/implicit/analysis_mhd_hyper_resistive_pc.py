#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Preconditioned banded twin of the calibrated hyper-resistive decay.

Physics: reruns the baseline's calibrated per-mode decay assertions
unchanged, so the eta_H content of the pc_mhd_block resistive block is
shown to leave the converged physics identical.

Solver: requires the cumulative GMRES iteration count to be STRICTLY
below the unpreconditioned baseline run's, with margin. This is the
load-bearing eta_H check: the preconditioner can only cut the iteration
count of this grid-stiff hyper-resistive deck if the frozen chain it
inverts matches the residual's operator (the in-run
resistive_validate_assembly abort separately pins the assembled rows to
the matrix-free application).
"""

import os
import re
import runpy
from pathlib import Path

import numpy as np

# Same calibrated physics assertions as the unpreconditioned baseline.
runpy.run_path(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "analysis_mhd_hyper_resistive.py"
    ),
    run_name="__main__",
)

# The preconditioner and solver selections are part of the contract.
used_inputs = Path("warpx_used_inputs").read_text()
match = re.search(r"^jacobian\.pc_type\s*=\s*(\S+)", used_inputs, re.MULTILINE)
assert match is not None and match.group(1) == "pc_mhd_block"
match = re.search(
    r"^pc_mhd_block\.resistive_solver\s*=\s*(\S+)", used_inputs, re.MULTILINE
)
assert match is not None and match.group(1) == "banded"

# Cumulative GMRES iterations, this run vs the unpreconditioned baseline
# (column 7 of the Newton diagnostic file is the running total).
mine = np.atleast_2d(np.loadtxt("diags/newton.txt"))
baseline = np.atleast_2d(
    np.loadtxt("../test_1d_theta_implicit_mhd_hyper_resistive/diags/newton.txt")
)
pc_iterations = int(mine[-1, 7])
baseline_iterations = int(baseline[-1, 7])
print(
    f"cumulative GMRES iterations: banded pc = {pc_iterations}, "
    f"pc-off = {baseline_iterations}"
)
assert pc_iterations < baseline_iterations
# Preserve most of the measured reduction, with margin for
# backend-dependent matrix-free roundoff.
assert 3 * pc_iterations <= baseline_iterations
