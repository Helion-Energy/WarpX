#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Native circuit driver vs the python-hook reference.

Runs in test_rz_theta_implicit_mhd_circuit_native and reads the
circuit_hook_history.csv written by both this run
(implicit_mhd.circuit_driver = native: the compiled RL plugin behind
the C++ circuit-coupling engine, batched device disk probes, in-process
AdvanceInterval) and the dependency run
(test_rz_theta_implicit_mhd_circuit_hook_residual: the IDENTICAL
backward-Euler RL map and the IDENTICAL DiskFluxLinkage functional
advanced in python through the externalcoiltheta hooks). Both fire at
residual scope, so the native driver's contract is EXACT
circuit-in-residual coupling with the python round-trip removed.

Asserts the committed per-step coil scales (and linkages) agree to
solver tolerance. The two drivers integrate the same coupled map with
the same discretization; they differ only in floating-point summation
order of the flux integral (batched fixed-tree vs numpy) and in the
Newton paths that follow, so the committed states agree to the Newton
tolerance class (1e-10 here), NOT bitwise. The 1e-8 budget matches the
offline python-vs-C++ stepper record (1e-12-class per interval) plus
the in-loop feedback accumulation headroom.
"""

import os
import sys

import numpy as np

# The python-driver reference run directory: the residual-scope hook test
# by default, or the twin named on the command line (the coupling-model
# parity arms pass their own reference).
BASELINE_DIRECTORY = (
    sys.argv[1] if len(sys.argv) > 1 else "../test_rz_theta_implicit_mhd_circuit_hook_residual"
)
HISTORY_FILE = "circuit_hook_history.csv"
# Written by the analytic-loop drive-probe arms (--drive-probe loop): per
# step the weighted and unweighted loop integrands of the drive coil and the
# extrema of the solver's circuit_linkage_weight register. Compared when
# both runs wrote it.
WEIGHT_CHECK_FILE = "circuit_weight_check.csv"

SCALE_RTOL = 1.0e-8


def load_history(path):
    rows = np.loadtxt(path)
    rows = np.atleast_2d(rows)
    assert rows.shape[1] == 6, f"malformed history file {path}"
    return rows


def main():
    native = load_history(HISTORY_FILE)
    reference = load_history(os.path.join(BASELINE_DIRECTORY, HISTORY_FILE))
    assert native.shape == reference.shape, "step-count mismatch between drivers"

    scale_native = native[:, 3]
    scale_reference = reference[:, 3]
    lam_native = native[:, 4]
    lam_reference = reference[:, 4]

    scale_norm = np.max(np.abs(scale_reference))
    lam_norm = np.max(np.abs(lam_reference))
    assert scale_norm > 0.0 and lam_norm > 0.0, "dead reference run"

    scale_err = np.max(np.abs(scale_native - scale_reference)) / scale_norm
    lam_err = np.max(np.abs(lam_native - lam_reference)) / lam_norm

    print("--- native vs python circuit driver, committed per-step state ---")
    for k in range(native.shape[0]):
        print(
            f"step {int(native[k, 0])}: scale native = {scale_native[k]:.12e} "
            f"python = {scale_reference[k]:.12e}  lambda native = "
            f"{lam_native[k]:.6e} python = {lam_reference[k]:.6e}"
        )
    print(f"max relative committed-scale mismatch  = {scale_err:.3e}")
    print(f"max relative committed-lambda mismatch = {lam_err:.3e}")

    assert scale_err <= SCALE_RTOL, (
        f"native-driver committed scales deviate from the python reference: "
        f"{scale_err:.3e} > {SCALE_RTOL:.1e}"
    )
    assert lam_err <= SCALE_RTOL, (
        f"native-driver committed linkages deviate from the python reference: "
        f"{lam_err:.3e} > {SCALE_RTOL:.1e}"
    )

    if os.path.exists(WEIGHT_CHECK_FILE) and os.path.exists(
        os.path.join(BASELINE_DIRECTORY, WEIGHT_CHECK_FILE)
    ):
        native_w = np.atleast_2d(np.loadtxt(WEIGHT_CHECK_FILE))
        reference_w = np.atleast_2d(np.loadtxt(os.path.join(BASELINE_DIRECTORY, WEIGHT_CHECK_FILE)))
        assert native_w.shape == reference_w.shape == (native.shape[0], 6), "malformed weight check"
        print("--- drive-coil loop integrand and linkage-weight register, per step ---")
        for k in range(native_w.shape[0]):
            print(
                f"step {int(native_w[k, 0])}: lambda_w native = {native_w[k, 1]:.6e} "
                f"python = {reference_w[k, 1]:.6e}  lambda_u native = {native_w[k, 2]:.6e} "
                f"python = {reference_w[k, 2]:.6e}  w_min = {native_w[k, 3]:.6e} / "
                f"{reference_w[k, 3]:.6e}  w_max = {native_w[k, 4]:.10f} / {reference_w[k, 4]:.10f}"
            )
        for col, name in ((1, "weighted loop linkage"), (2, "unweighted loop linkage")):
            norm = np.max(np.abs(reference_w[:, col]))
            assert norm > 0.0, f"dead {name} column"
            err = np.max(np.abs(native_w[:, col] - reference_w[:, col])) / norm
            print(f"max relative {name} mismatch = {err:.3e}")
            assert err <= SCALE_RTOL, f"{name}: {err:.3e} > {SCALE_RTOL:.1e}"
        for col, name in ((3, "w_min"), (4, "w_max")):
            err = np.max(np.abs(native_w[:, col] - reference_w[:, col]))
            print(f"max absolute register {name} mismatch = {err:.3e}")
            assert err <= SCALE_RTOL, f"register {name}: {err:.3e} > {SCALE_RTOL:.1e}"
        assert np.array_equal(native_w[:, 5], reference_w[:, 5]), "boost-class node counts differ"
    print("PASS")


if __name__ == "__main__":
    main()
