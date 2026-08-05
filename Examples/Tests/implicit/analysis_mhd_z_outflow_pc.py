#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Paired pc/no-pc gate for the Stage-2 hlld block preconditioner in RZ.

This runs on the SAME advected-pulse outflow deck as
test_rz_theta_implicit_mhd_z_outflow_hlld with
jacobian.pc_type = pc_mhd_block. The identical-physics contract is the full
z-outflow analysis of the pc-off twin, executed unchanged at its
tolerances. On top of that, the linear work becomes part of the regression
contract: this deck is fully resolved (signal, fast, and Alfven CFL all
below one), so the stiffness-gated Stage-2 preconditioner must reduce to
the identity plus the (physically inert here) triangular Faraday corrector
and keep plain GMRES at the pc-off twin's cumulative iteration count
(about 1.7e3). The Stage-1 isotropic signal diffusion sat at 2.8x that
count; a regression of the gating (or a non-stationary preconditioner)
would blow past this bound.

Usage: analysis_mhd_z_outflow_pc.py <initial_plotfile> <final_plotfile> pec
"""

import re
import runpy
from pathlib import Path

import numpy as np

# Identical physics within the pc-off pattern's tolerances: execute the
# z-outflow analysis unchanged (it reads the same sys.argv).
runpy.run_path(
    str(Path(__file__).resolve().parent / "analysis_mhd_z_outflow.py"),
    run_name="__main__",
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
# Every step converged in a few Newton iterations (the run would have
# aborted on non-convergence; the bound guards against silent degradation).
assert np.max(newton_history[:, 2]) <= 8
# Cumulative GMRES iterations: the pc-off twin needs about 1.7e3 and the
# gated preconditioner must stay iteration-neutral on this resolved deck.
total_gmres_iterations = newton_history[-1, 7]
assert 0 < total_gmres_iterations <= 2.1e3, (
    f"cumulative GMRES iterations {total_gmres_iterations:.0f} exceed the "
    f"pc-on budget; the Stage-2 hlld preconditioner gating has regressed"
)

print("hlld RZ preconditioner pair: PASS")
