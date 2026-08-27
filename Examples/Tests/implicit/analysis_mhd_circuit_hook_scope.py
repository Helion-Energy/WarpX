#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Newton-scope circuit hook vs the residual-scope reference.

Runs in test_rz_theta_implicit_mhd_circuit_hook_newton and reads the
circuit_hook_history.csv written by both this run (scope = newton) and
the dependency run (scope = residual). Asserts the two contracts of
implicit_mhd.circuit_hook_scope = newton:

(1) ANSWER UNCHANGED: the committed per-step coil scales (and the
    measured flux linkages) match the residual-scope run to solver
    tolerance. Only the Jacobian and the line-search grades see the
    lagged circuit; convergence is still tested on the live-coupled
    iterate residual, so the accepted states -- and the circuit
    committed from them -- agree to the Newton tolerance (1e-10 here),
    NOT bitwise (the Newton paths differ).

(2) SYNC TAX REMOVED: the per-step "externalcoiltheta" firing count
    collapses from every-residual-evaluation (Newton iterates + GMRES
    matrix-free probes + line-search trials) to iterates only.
"""

import os

import numpy as np

BASELINE_DIRECTORY = "../test_rz_theta_implicit_mhd_circuit_hook_residual"
HISTORY_FILE = "circuit_hook_history.csv"

# Committed-scale parity budget: both runs converge Newton to 1e-10
# relative; the committed circuit state is a smooth functional of the
# accepted states with a 5-step accumulation. 1e-6 leaves four decades
# of headroom while staying far below any quasi-Newton pathology.
SCALE_RTOL = 1.0e-6
# The count contrast: every Newton iteration adds GMRES probes (and any
# backtracks) to the residual-scope count, so a factor >= 2 per step is
# a conservative floor for "count << count" (measured contrast is far
# larger; see the printed ratio).
COUNT_CONTRAST = 2.0


def load_history(path):
    rows = np.loadtxt(path)
    rows = np.atleast_2d(rows)
    assert rows.shape[1] == 6, f"malformed history file {path}"
    return rows


def main():
    newton = load_history(HISTORY_FILE)
    residual = load_history(os.path.join(BASELINE_DIRECTORY, HISTORY_FILE))
    assert newton.shape == residual.shape, "step-count mismatch between scopes"

    steps = newton[:, 0].astype(int)
    newton_counts = newton[:, 2]
    residual_counts = residual[:, 2]

    # (1) converged answer: per-step committed scales and linkages.
    scale_reference = np.max(np.abs(residual[:, 3]))
    lambda_reference = np.max(np.abs(residual[:, 4]))
    assert scale_reference > 0.0 and lambda_reference > 0.0
    print("--- newton vs residual scope ---")
    for k, step in enumerate(steps):
        scale_error = abs(newton[k, 3] - residual[k, 3]) / scale_reference
        lambda_error = abs(newton[k, 4] - residual[k, 4]) / lambda_reference
        print(
            f"step {step}: hook calls {int(newton_counts[k])} vs "
            f"{int(residual_counts[k])}, scale rel. err = {scale_error:.3e}, "
            f"lambda rel. err = {lambda_error:.3e}"
        )
        assert scale_error < SCALE_RTOL, (
            f"step {step}: committed coil scale differs beyond solver "
            f"tolerance ({scale_error:.3e} >= {SCALE_RTOL})"
        )
        assert lambda_error < SCALE_RTOL, (
            f"step {step}: committed flux linkage differs beyond solver "
            f"tolerance ({lambda_error:.3e} >= {SCALE_RTOL})"
        )

    # (2) the sync tax: newton-scope counts must be a small fraction of
    # the residual-scope counts, per step and in total.
    total_newton = float(np.sum(newton_counts))
    total_residual = float(np.sum(residual_counts))
    print(
        f"total externalcoiltheta calls: newton = {int(total_newton)}, "
        f"residual = {int(total_residual)} "
        f"(ratio = {total_residual / total_newton:.1f}x)"
    )
    assert np.all(newton_counts * COUNT_CONTRAST <= residual_counts), (
        "newton-scope hook counts are not well below the residual-scope "
        "counts on every step"
    )
    assert total_newton * COUNT_CONTRAST <= total_residual

    print("PASS")


if __name__ == "__main__":
    main()
