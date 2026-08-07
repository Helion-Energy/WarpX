#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Preconditioned twin of the calibrated vacuum-resistive decay test.

Physics: reruns the baseline's calibrated one-step assertions unchanged
(the theta_r-discrete transverse-mode damping, the un-boosted Joule
heating, the frozen ions), so the pc_mhd_block resistive block is shown
to leave the converged physics identical.

Solver: additionally requires the cumulative GMRES iteration count to be
STRICTLY below the unpreconditioned baseline run's (read from the
dependency's diagnostic file), with margin: the Chebyshev resistive
block was measured to cut this calibrated case from 90 to 8 iterations.
"""

import os
import re
import runpy
from pathlib import Path

import numpy as np

# Same calibrated physics assertions as the unpreconditioned baseline.
runpy.run_path(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "analysis_mhd_vacuum_resistive.py"
    ),
    run_name="__main__",
)

# The preconditioner selection is part of the regression contract.
used_inputs = Path("warpx_used_inputs").read_text()
match = re.search(r"^jacobian\.pc_type\s*=\s*(\S+)", used_inputs, re.MULTILINE)
assert match is not None and match.group(1) == "pc_mhd_block"

# Cumulative GMRES iterations, this run vs the unpreconditioned baseline
# (column 7 of the Newton diagnostic file is the running total).
mine = np.atleast_2d(np.loadtxt("diags/newton.txt"))
baseline = np.atleast_2d(
    np.loadtxt("../test_1d_theta_implicit_mhd_vacuum_resistive/diags/newton.txt")
)
pc_iterations = int(mine[-1, 7])
baseline_iterations = int(baseline[-1, 7])
print(
    f"cumulative GMRES iterations: pc-on = {pc_iterations}, "
    f"pc-off = {baseline_iterations}"
)
assert pc_iterations < baseline_iterations
# Preserve most of the measured 90 -> 8 reduction, with margin for
# backend-dependent matrix-free roundoff.
assert 3 * pc_iterations <= baseline_iterations
