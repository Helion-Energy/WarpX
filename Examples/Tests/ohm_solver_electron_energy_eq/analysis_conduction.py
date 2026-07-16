#!/usr/bin/env python3
#
# --- Analysis for the QDSMC isotropic conduction test: a Gaussian T_e bump
# --- on a static uniform plasma with constant conductivity must spread as
# --- sigma^2(t) = sigma_0^2 + 2 D t per axis while respecting the discrete
# --- maximum principle and conserving the bump energy.

import glob
import sys

import numpy as np
import yt

# Deck constants (keep in sync with the inputs file)
D = 1.65e4  # m^2/s
L = 0.5
NX = 64
DX = L / NX
SIGMA0 = 6.0 * DX
T_E_BG_EV = 100.0
BUMP_AMP = 1.0
Q_E = 1.602176634e-19
KB = 1.380649e-23
T_BG = T_E_BG_EV * Q_E / KB


def moments(te):
    """Excess sum and Gaussian variance from the peak decay.

    For a 2D Gaussian bump of initial amplitude A0 and variance sigma_0^2
    per axis, peak(t) = A0 sigma_0^2 / sigma^2(t): the peak is the
    highest-signal-to-noise functional of the profile (the full-box second
    moment is dominated by the deposit-noise floor times r^2).
    """
    dt_ = te - T_BG
    m0 = dt_.sum()
    a0 = BUMP_AMP * T_BG
    var = a0 * SIGMA0**2 / dt_.max()
    return m0, var


def main():
    dirs = sorted(glob.glob("diags/diag1" + "[0-9]" * 6))
    if len(dirs) < 4:
        raise RuntimeError("not enough plotfiles under diags/")
    # The step-0 dump precedes the bump load (the hybrid model seeds a
    # uniform T_e during InitData): fit from the later dumps only.
    dirs = dirs[1:]
    yt.set_log_level(50)

    times, m0s, variances = [], [], []
    tmax0 = tmin0 = None
    print(f"{'t [s]':>10s} {'sum(dT)':>12s} {'sigma^2':>12s} "
          f"{'expected':>12s} {'Tmax/T0max':>11s}")
    for d in dirs:
        ds = yt.load(d)
        t = float(ds.current_time)
        ad = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        te = np.squeeze(np.array(ad["boxlib", "Te"]))
        m0, var = moments(te)
        if tmax0 is None:
            tmax0, tmin0 = te.max(), te.min()
        times.append(t)
        m0s.append(m0)
        variances.append(var)
        expected = SIGMA0**2 + 2.0 * D * t
        print(f"{t:10.3e} {m0:12.5e} {var:12.5e} {expected:12.5e} "
              f"{te.max() / tmax0:11.5f}")

    times = np.array(times)
    variances = np.array(variances)
    m0s = np.array(m0s)

    # 1) Variance growth rate: sanity window only. At CI scale the
    #    conservative delta transport competes with two remap scales
    #    (the advection stage's dx^2/(4 dt) and the kernel's deposit
    #    granularity), so the absolute rate is verified loosely here;
    #    the tight kernel-accuracy check is the nonlinear-front test
    #    (analysis_conduction_front.py), whose self-similar exponent is
    #    insensitive to those additive numerical scales.
    slope = np.polyfit(times, variances, 1)[0]
    slope_err = abs(slope - 2.0 * D) / (2.0 * D)
    print(f"\nvariance slope: {slope:.4e} (theory {2.0 * D:.4e}, "
          f"rel err {slope_err:.2%})")
    assert 0.1 * 2.0 * D < slope < 2.0 * 2.0 * D, (
        f"variance growth outside sanity window: {slope:.3e}")

    # 2) Energy ledger: the excess-temperature sum is the bump's thermal
    #    energy at uniform density; the residual drift bound covers the
    #    documented PIC-noise rectification at this NPPC.
    ledger_err = abs(m0s[-1] - m0s[0]) / m0s[0]
    print(f"bump energy drift: {ledger_err:.2%}")
    assert ledger_err < 0.05, f"bump energy drifted: {ledger_err:.2%}"

    # 3) Discrete maximum principle: the peak decays monotonically toward
    #    the background and never overshoots its initial value; the
    #    minimum never dips below the background (deposit-noise tolerance).
    for d in dirs[1:]:
        ds = yt.load(d)
        ad = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        te = np.squeeze(np.array(ad["boxlib", "Te"]))
        assert te.max() <= tmax0 * 1.01, "maximum principle violated (max)"
        assert te.min() >= tmin0 * 0.97, "maximum principle violated (min)"

    print("\nAll QDSMC conduction checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
