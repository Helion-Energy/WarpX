#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Banded twin of the preconditioned stiff RZ vacuum-resistive test.

Physics: reruns the baseline's structural assertions unchanged, and
requires the final fields to match the unpreconditioned baseline run to
solver tolerance. This twin runs on ONE rank (the banded solver asserts
single-rank) against the two-rank baseline, so the comparison tolerance
is the Newton relative tolerance (1e-11 of the 1e-3 T field scale)
rather than the same-decomposition roundoff of the Chebyshev twin.

Solver: the same 10x cumulative-GMRES margin as the Chebyshev twin (the
banded LU is an exact inverse up to its factorization precision; the
periodic z is eliminated in folded ring order).
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

# The preconditioner and solver selections are part of the contract.
used_inputs = Path("warpx_used_inputs").read_text()
match = re.search(r"^jacobian\.pc_type\s*=\s*(\S+)", used_inputs, re.MULTILINE)
assert match is not None and match.group(1) == "pc_mhd_block"
match = re.search(
    r"^pc_mhd_block\.resistive_solver\s*=\s*(\S+)", used_inputs, re.MULTILINE
)
assert match is not None and match.group(1) == "banded"

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
    np.testing.assert_allclose(mine[name], baseline[name], rtol=0.0, atol=1.0e-13)

# Cumulative GMRES iterations, this run vs the unpreconditioned baseline
# (column 7 of the Newton diagnostic file is the running total).
mine_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
baseline_history = np.atleast_2d(
    np.loadtxt(os.path.join(baseline_directory, "diags/newton.txt"))
)
pc_iterations = int(mine_history[-1, 7])
baseline_iterations = int(baseline_history[-1, 7])
print(
    f"cumulative GMRES iterations: banded pc = {pc_iterations}, "
    f"pc-off = {baseline_iterations}"
)
assert pc_iterations < baseline_iterations
assert 10 * pc_iterations <= baseline_iterations
