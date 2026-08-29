#!/usr/bin/env python3
#
# --- Analysis for the 3D vacuum inductive ramp with an embedded conducting
# --- cylinder: the flux threading the (masked) conductor stays frozen at
# --- its initial value while the far exterior tracks B(0) + f(t) dB.

import glob
import sys

import numpy as np
import yt

# Deck constants (keep in sync with the inputs file)
B0 = 0.1
DB = 0.1
R_COND = 0.08
M_P = 1.67262192369e-27
Q_E = 1.602176634e-19
W_CI = Q_E * B0 / M_P
T_CI = 2.0 * np.pi / W_CI
T0_RAMP = 1.0 * T_CI
TAU_RAMP = 4.0 * T_CI


def f_ramp(t):
    return 1.0 / (1.0 + np.exp(5.0 * (1.0 - (t - T0_RAMP) * np.sqrt(2.0) / TAU_RAMP)))


def main():
    dirs = sorted(glob.glob("diags/diag1" + "[0-9]" * 6))
    if not dirs:
        raise RuntimeError("no plotfiles found under diags/")
    yt.set_log_level(50)

    print(f"{'t/t_ci':>7s} {'f(t)':>7s} {'Bz_in':>9s} {'Bz_out':>9s} "
          f"{'expected':>9s} {'in drift':>9s} {'out err':>9s}")
    max_in_drift = 0.0
    max_out_err = 0.0
    b_in_0 = None
    for d in dirs:
        ds = yt.load(d)
        t = float(ds.current_time)
        ad = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        bz = np.array(ad["boxlib", "Bz"])
        nx, ny, nz = bz.shape
        x = np.linspace(float(ds.domain_left_edge[0]),
                        float(ds.domain_right_edge[0]), nx)
        y = np.linspace(float(ds.domain_left_edge[1]),
                        float(ds.domain_right_edge[1]), ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        r2 = X**2 + Y**2
        inner = r2 < (0.7 * R_COND) ** 2
        outer = (r2 > (2.2 * R_COND) ** 2) & (np.abs(X) < 0.2) & (np.abs(Y) < 0.2)

        bz_mid = bz.mean(axis=2)  # z-average (periodic, uniform)
        b_in = bz_mid[inner].mean()
        b_out = bz_mid[outer].mean()
        if b_in_0 is None:
            b_in_0 = b_in
        expected = B0 + (f_ramp(t) - f_ramp(0.0)) * DB

        in_drift = abs(b_in - b_in_0) / B0
        out_err = abs(b_out - expected) / B0
        max_in_drift = max(max_in_drift, in_drift)
        max_out_err = max(max_out_err, out_err)
        print(f"{t / T_CI:7.3f} {f_ramp(t):7.4f} {b_in:9.5f} {b_out:9.5f} "
              f"{expected:9.5f} {in_drift:9.2e} {out_err:9.2e}")

    # The conductor's flux is frozen by the masked update; the exterior
    # tracks the programmed ramp with a modest offset from the conductor's
    # excluded-flux redistribution inside a finite (dirichlet-wall) box.
    assert max_in_drift < 2.0e-2, f"conductor flux drifted: {max_in_drift:.2e}"
    assert max_out_err < 8.0e-2, f"exterior flux tracking error {max_out_err:.2e}"
    print("\nAll 3D EB vacuum-ramp checks passed.")


if __name__ == "__main__":
    sys.exit(main())
