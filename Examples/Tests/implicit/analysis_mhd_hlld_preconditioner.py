#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Paired pc/no-pc gate for the hlld block preconditioner.

This runs on the SAME Sod shock tube as test_1d_theta_implicit_mhd_sod_hlld
with jacobian.pc_type = pc_mhd_block. The identical-physics contract is the
full exact-Riemann plateau analysis of the pc-off twin, executed unchanged at
its tolerances. On top of that, the preconditioner engagement and its
stationary bound on the linear work become part of the regression contract:
the Stage-2 1D form (identity B block plus the triangular Faraday corrector,
which engages here because the reference Alfven CFL is below one; the
per-direction signal diffusion is opt-in and off by default) must keep plain
GMRES convergent without inflating the iteration count of the pc-off twin
(about 3.9e3 cumulative) by more than the allowed margin.

Usage: analysis_mhd_hlld_preconditioner.py <initial_plotfile> <final_plotfile> total_energy
"""

import re
import runpy
from pathlib import Path

import numpy as np

# Identical physics within the pc-off pattern's tolerances: execute the Sod
# analysis unchanged (it reads the same sys.argv).
runpy.run_path(
    str(Path(__file__).resolve().parent / "analysis_mhd_sod.py"), run_name="__main__"
)

used_inputs = Path("warpx_used_inputs").read_text()


def input_value(name):
    match = re.search(rf"^{re.escape(name)}\s*=\s*([^#\s]+)", used_inputs, re.MULTILINE)
    assert match is not None
    return match.group(1)


# Make the preconditioner selection part of the regression contract.
assert input_value("jacobian.pc_type") == "pc_mhd_block"
assert input_value("implicit_mhd.fluid_flux") == "hlld"

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
# Every step converged in a few Newton iterations (the run would have aborted
# on non-convergence; the bound guards against silent degradation).
assert np.max(newton_history[:, 2]) <= 8
# Cumulative GMRES iterations: the pc-off twin needs about 3.9e3; the Stage-1
# preconditioner must stay within a modest margin of that. A singular or
# non-stationary preconditioner would blow far past this bound.
total_gmres_iterations = newton_history[-1, 7]
assert 0 < total_gmres_iterations <= 4.7e3, (
    f"cumulative GMRES iterations {total_gmres_iterations:.0f} exceed the "
    f"pc-on budget; the Stage-1 hlld preconditioner has regressed"
)

print("hlld preconditioner pair: PASS")
