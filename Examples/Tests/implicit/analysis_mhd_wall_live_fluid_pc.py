#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""Shaped-wall live-fluid preconditioner-parity test (RZ MHD).

The rigid-conductor fluid freeze (implicit_mhd.wall_thermal_bc) makes
every masked wall-band fluid row an exact identity in the residual, in
every wall_model including dielectric. pc_mhd_block's cell-centered
fluid sweeps must mirror that structure; a PC that feeds the band into
the signal/wave inverses collapses the preconditioned spectrum on the
band's identity rows and GMRES stalls hard (measured on this deck:
every GMRES call pegged at its 1000-iteration cap with the final linear
residual at the scale of its own RHS, against ~300 converged iterations
per call when healthy).

mode = "reference" (jacobian.pc_type = none twin): asserts the
unpreconditioned solves are healthy (few Newton iterations per step) --
the teeth for the headline run's pointwise match.

mode = "pc" (the headline, pc_mhd_block): asserts
 1. Newton converged in a few iterations per step (the run itself
    aborts otherwise: newton.require_convergence),
 2. every step's GMRES calls CONVERGED well inside the iteration cap
    (mean iterations per Newton iteration < the cap; final linear
    residual far below the nonlinear residual scale),
 3. the end state matches the reference twin pointwise to solver
    precision: the preconditioner must not change the answer at the
    wall (measured relative mismatch ~8e-14).

Usage:
  analysis_mhd_wall_live_fluid_pc.py <newton.txt> <diag_end> reference
  analysis_mhd_wall_live_fluid_pc.py <newton.txt> <diag_end> pc <reference_diag_end>
"""

import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(0)

MAX_NEWTON_ITERS_PER_STEP = 6
# The broken PC pegs the 1000-iteration GMRES cap on every call; healthy
# solves land near ~300 iterations per call on this deck's 2-rank layout.
MAX_GMRES_ITERS_PER_NEWTON = 800.0
# Healthy final linear residuals sit ~4 orders below the nonlinear norm
# (measured 1e-18 vs 2e-14); the broken PC leaves them AT the RHS scale.
MAX_GMRES_RESIDUAL_FRACTION = 1.0e-2
TOL_MATCH = 1.0e-6  # of each field's own scale; measured ~8e-14

FIELDS = ("Br", "Bz", "Et", "implicit_mhd_mass_density", "implicit_mhd_momentum_r")


def load(fn):
    ds = yt.load(fn)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return {name: grid["boxlib", name].v.squeeze() for name in FIELDS}


newton_rows = np.loadtxt(sys.argv[1], ndmin=2)
state = load(sys.argv[2])
mode = sys.argv[3]
assert mode in ("reference", "pc"), f"unknown mode {mode}"

for name, field in state.items():
    assert np.isfinite(field).all(), f"{name} has non-finite values"

steps = newton_rows[:, 0]
newton_iters = newton_rows[:, 2]
gmres_iters = newton_rows[:, 6]
norm_abs = newton_rows[:, 4]
gmres_last_res = newton_rows[:, 8]
assert len(steps) >= 5, f"expected at least 5 recorded steps, got {len(steps)}"

for step, iters in zip(steps, newton_iters):
    print(f"step {int(step)}: newton iters = {int(iters)}")
    assert iters <= MAX_NEWTON_ITERS_PER_STEP, (
        f"step {int(step)}: Newton needed {int(iters)} iterations "
        f"(> {MAX_NEWTON_ITERS_PER_STEP}): the wall solve is degraded"
    )

if mode == "pc":
    for step, iters, gmres, res, norm in zip(
        steps, newton_iters, gmres_iters, gmres_last_res, norm_abs
    ):
        per_call = gmres / max(iters, 1.0)
        print(
            f"step {int(step)}: gmres/newton-iter = {per_call:.1f}, "
            f"last linear residual = {res:.3e} (nonlinear norm {norm:.3e})"
        )
        # The broken PC pegs the cap exactly (per_call == 1000).
        assert per_call < MAX_GMRES_ITERS_PER_NEWTON, (
            f"step {int(step)}: GMRES needed {per_call:.0f} iterations per "
            "Newton iteration -- the preconditioned wall solve is stalling "
            "(pc_mhd_block is not mirroring the wall-band fluid freeze)"
        )
        # ... and leaves the final linear residual at the RHS scale.
        assert res < MAX_GMRES_RESIDUAL_FRACTION * max(norm, 1.0e-300), (
            f"step {int(step)}: final GMRES residual {res:.3e} did not "
            f"converge below the nonlinear norm {norm:.3e}"
        )

    # Pointwise preconditioner parity against the unpreconditioned twin.
    assert len(sys.argv) > 4, "pc mode needs the reference twin's plotfile"
    reference = load(sys.argv[4])
    for name in FIELDS:
        scale = max(np.max(np.abs(reference[name])), 1.0e-300)
        err = float(np.max(np.abs(state[name] - reference[name]))) / scale
        print(f"pointwise |{name}_pc - {name}_nopc|max/scale = {err:.3e}")
        assert err < TOL_MATCH, (
            f"{name}: pc_mhd_block changed the converged wall answer "
            f"({err:.3e} of scale)"
        )

print(f"shaped-wall live-fluid preconditioner-parity test ({mode}) PASSED")
