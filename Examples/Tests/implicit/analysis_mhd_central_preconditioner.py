#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Paired pc/no-pc gate for the block preconditioner's fluid upwind rows.

This runs the SAME central shock tube as test_1d_theta_implicit_mhd_sod_
central_fast (MUSCL-Rusanov: median reconstruction, central_dissipation = 1)
with jacobian.pc_type = pc_mhd_block and pc_mhd_block.fluid_upwind = rusanov,
i.e. the block's cell-centered fluid rows carry the first-order Rusanov
diffusion theta dt alpha dx/2 at the residual's own face fast bound. The
identical-physics contract is the full exact-Riemann plateau analysis of the
pc-off twin, executed unchanged at its tolerances (a preconditioner changes
the linear iteration, never the converged solution). On top of that the
engagement of the rows and a stationary bound on the linear work become part
of the regression contract: every step converges within the Newton budget
and the cumulative GMRES count stays within the measured pc-on envelope.

Usage: analysis_mhd_central_preconditioner.py <initial_plotfile> <final_plotfile> total_energy
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
    assert match is not None, name
    return match.group(1).strip('"')


# Make the preconditioner selection part of the regression contract.
assert input_value("jacobian.pc_type") == "pc_mhd_block"
assert input_value("implicit_mhd.fluid_flux") == "central"
assert input_value("pc_mhd_block.fluid_upwind") == "rusanov"

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
# Every step converged in a few Newton iterations (the run would have aborted
# on non-convergence; the bound guards against silent degradation).
max_newton = np.max(newton_history[:, 2])
total_gmres_iterations = newton_history[-1, 7]
print(
    f"central pc upwind: max Newton per step {max_newton:.0f}, "
    f"cumulative GMRES {total_gmres_iterations:.0f}"
)
assert max_newton <= 8
# Cumulative GMRES iterations: measured 6.2e3 with the upwind rows against
# 6.8e3 for the pc-off twin (test_1d_theta_implicit_mhd_sod_central_fast) --
# the rows buy 8 percent here; the bound holds the block to "no worse than
# pc-off". A singular or non-stationary preconditioner would blow far past it.
assert 0 < total_gmres_iterations <= 7.0e3, (
    f"cumulative GMRES iterations {total_gmres_iterations:.0f} exceed the "
    f"pc-on budget; the fluid upwind rows have regressed"
)

print("central preconditioner pair: PASS")
