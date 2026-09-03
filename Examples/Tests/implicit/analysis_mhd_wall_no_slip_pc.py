#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""No-slip wall preconditioner-parity test (RZ MHD).

implicit_mhd.wall_no_slip is a FACE condition on the tangential velocity
at the shaped-wall contour: the masked side of a stair interface face
presents the antisymmetric tangential image -u_t of the interior state,
so the face-centered tangential velocity is exactly zero and the viscous
face stress carries the half-cell wall shear.  It adds NO identity rows
anywhere -- every wall-adjacent LIVE cell keeps ordinary momentum
unknowns -- so it is a plain residual-only face flux, exactly like the
interior viscous stress, and pc_mhd_block covers it through the same
wave/diffusion Schur.

This test is the guard on that claim.  A residual-only wall term the
preconditioner does not cover is a measured 7-21x GMRES stagnation (the
z_wall_conduction lesson), and the wall face sits on the LIVE rows where
the momentum wave Schur is strongest -- the rows the masked-band rigid
freeze does not cover.

mode = "off" (the wall_no_slip = 0 twin): asserts the preconditioned
baseline is healthy and records its GMRES cost per Newton iteration.

mode = "on" (the headline): asserts
 1. Newton converged in a few iterations per step,
 2. every step's GMRES calls converged well inside the iteration cap,
 3. the mean GMRES iterations per Newton iteration do not exceed the
    OFF twin's by more than MAX_GMRES_MARGIN.

Usage:
  analysis_mhd_wall_no_slip_pc.py <newton.txt> off
  analysis_mhd_wall_no_slip_pc.py <newton.txt> on <off_newton.txt>
"""

import sys

import numpy as np

MAX_NEWTON_ITERS_PER_STEP = 6
# The deck's GMRES cap; a PC that does not cover the wall term pegs it.
GMRES_CAP = 400.0
MAX_GMRES_ITERS_PER_NEWTON = 300.0
# The final linear residual is measured against the step's FINAL
# nonlinear norm, which is smaller than the RHS that GMRES call
# actually saw, so this is a sanity bound (a stalled PC leaves it
# AT the RHS scale, orders above the converged norm), not a
# restatement of gmres.relative_tolerance.
MAX_GMRES_RESIDUAL_FRACTION = 1.0
# "within a small factor": the wall face term must not cost the
# preconditioned solve anything structural.
MAX_GMRES_MARGIN = 2.0


def gmres_per_newton(path):
    rows = np.loadtxt(path, ndmin=2)
    # newton.diagnostic_file is APPENDED across runs in the same working
    # directory (a local re-run, not CI): keep only the last contiguous
    # block of increasing step numbers.
    starts = np.nonzero(np.diff(rows[:, 0]) <= 0)[0]
    if starts.size:
        rows = rows[starts[-1] + 1 :]
    steps = rows[:, 0]
    newton_iters = rows[:, 2]
    gmres_iters = rows[:, 6]
    norm_abs = rows[:, 4]
    gmres_last_res = rows[:, 8]
    assert len(steps) >= 3, f"expected at least 3 recorded steps, got {len(steps)}"

    per_call = []
    for step, iters, gmres, res, norm in zip(
        steps, newton_iters, gmres_iters, gmres_last_res, norm_abs
    ):
        assert iters <= MAX_NEWTON_ITERS_PER_STEP, (
            f"step {int(step)}: Newton needed {int(iters)} iterations "
            f"(> {MAX_NEWTON_ITERS_PER_STEP}): the wall solve is degraded"
        )
        ratio = gmres / max(iters, 1.0)
        per_call.append(ratio)
        print(
            f"step {int(step)}: newton {int(iters)}, "
            f"gmres/newton-iter = {ratio:.1f}, "
            f"last linear residual = {res:.3e} (nonlinear norm {norm:.3e})"
        )
        assert ratio < MAX_GMRES_ITERS_PER_NEWTON, (
            f"step {int(step)}: GMRES needed {ratio:.0f} iterations per "
            "Newton iteration -- the preconditioned wall solve is stalling"
        )
        assert res < MAX_GMRES_RESIDUAL_FRACTION * max(norm, 1.0e-300), (
            f"step {int(step)}: final GMRES residual {res:.3e} did not "
            f"converge below the nonlinear norm {norm:.3e}"
        )
    return float(np.mean(per_call))


mode = sys.argv[2]
assert mode in ("on", "off"), f"unknown mode {mode}"

mean_per_newton = gmres_per_newton(sys.argv[1])
print(f"mode = {mode}: mean gmres/newton-iter = {mean_per_newton:.2f}")
assert mean_per_newton < GMRES_CAP

if mode == "on":
    assert len(sys.argv) > 3, "on mode needs the wall_no_slip = 0 twin"
    reference_per_newton = gmres_per_newton(sys.argv[3])
    margin = mean_per_newton / max(reference_per_newton, 1.0e-300)
    print(
        f"no-slip ON {mean_per_newton:.2f} vs OFF "
        f"{reference_per_newton:.2f} gmres/newton-iter -> margin "
        f"{margin:.3f}x"
    )
    assert margin < MAX_GMRES_MARGIN, (
        f"the no-slip wall face costs {margin:.2f}x the OFF twin's GMRES "
        "iterations: pc_mhd_block no longer covers the wall shear term"
    )

print(f"wall no-slip preconditioner-parity test ({mode}) PASSED")
