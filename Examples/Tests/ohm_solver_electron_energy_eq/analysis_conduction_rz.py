#!/usr/bin/env python3
#
# --- Analysis for the RZ grid-FD conduction test (curvilinear J Xi^ij
# --- form): a radial Gaussian T_e bump centered ON THE AXIS of a static
# --- uniform plasma with constant conductivity spreads as planar 2D
# --- diffusion, peak(t) = A0 sigma_0^2 / sigma^2(t) with sigma^2(t) =
# --- sigma_0^2 + 2 D t, while conserving the r-weighted bump energy
# --- (the operator's dual-cell measure) and respecting the discrete
# --- maximum principle. The solution must also stay uniform in z: any
# --- systematic z-structure flags metric leakage into the z fluxes.

import glob
import sys

import numpy as np
import yt

# Deck constants (keep in sync with the inputs file, --rz mode)
D = 5.6e3  # m^2/s
L = 0.5  # r extent
NX = 32  # radial cells
DX = L / NX
SIGMA0 = 6.0 * DX
T_E_BG_EV = 100.0
BUMP_AMP = 1.0
Q_E = 1.602176634e-19
KB = 1.380649e-23
T_BG = T_E_BG_EV * Q_E / KB


def moments(te):
    """r-weighted excess sum and Gaussian variance from the peak decay.

    peak(t) = A0 sigma_0^2 / sigma^2(t) for the axis-centered radial
    Gaussian (planar 2D diffusion of an axisymmetric profile); the peak
    is the highest-signal-to-noise functional, as in the 2D analysis.
    The ledger sum is weighted by the cell-center radius (the plotfile
    is cell-centered, so every weight is r > 0).
    """
    dt_ = te - T_BG
    r_c = (np.arange(te.shape[0]) + 0.5) * DX
    m0 = (dt_ * r_c[:, None]).sum()
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
    zvar_max = 0.0
    print(
        f"{'t [s]':>10s} {'sum(r dT)':>12s} {'sigma^2':>12s} "
        f"{'expected':>12s} {'Tmax/T0max':>11s} {'z-var':>10s}"
    )
    for d in dirs:
        ds = yt.load(d)
        t = float(ds.current_time)
        ad = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        te = np.squeeze(np.array(ad["boxlib", "Te"]))
        m0, var = moments(te)
        # z-uniformity of the axisymmetric, z-independent solution:
        # per-radial-row spread over z, relative to the bump amplitude
        zvar = (te.max(axis=1) - te.min(axis=1)).max() / (BUMP_AMP * T_BG)
        zvar_max = max(zvar_max, zvar)
        if tmax0 is None:
            tmax0, tmin0 = te.max(), te.min()
        times.append(t)
        m0s.append(m0)
        variances.append(var)
        expected = SIGMA0**2 + 2.0 * D * t
        print(
            f"{t:10.3e} {m0:12.5e} {var:12.5e} {expected:12.5e} "
            f"{te.max() / tmax0:11.5f} {zvar:10.2e}"
        )

    times = np.array(times)
    variances = np.array(variances)
    m0s = np.array(m0s)

    # 1) Variance growth rate. The FD operator is deterministic and
    #    subcycled into the resolved regime, so the gate is much tighter
    #    than the SDE test's sanity window; the residual error budget is
    #    peak-functional noise from the deposited-density wobble in the
    #    face coefficients.
    slope = np.polyfit(times, variances, 1)[0]
    slope_err = abs(slope - 2.0 * D) / (2.0 * D)
    print(
        f"\nvariance slope: {slope:.4e} (theory {2.0 * D:.4e}, rel err {slope_err:.2%})"
    )
    assert slope_err < 0.10, f"variance growth off theory: {slope:.3e} vs {2.0 * D:.3e}"

    # 2) Energy ledger (r-weighted -- the curvilinear operator conserves
    #    the Vr-measure sum; drift bound covers PIC-noise rectification).
    ledger_err = abs(m0s[-1] - m0s[0]) / m0s[0]
    print(f"bump energy drift (r-weighted): {ledger_err:.2%}")
    assert ledger_err < 0.05, f"bump energy drifted: {ledger_err:.2%}"

    # 3) Discrete maximum principle (axis node included).
    for d in dirs[1:]:
        ds = yt.load(d)
        ad = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        te = np.squeeze(np.array(ad["boxlib", "Te"]))
        assert te.max() <= tmax0 * 1.01, "maximum principle violated (max)"
        assert te.min() >= tmin0 * 0.97, "maximum principle violated (min)"

    # 4) z-uniformity: metric leakage into the z fluxes (or an axis-column
    #    pathology) would print as systematic z-structure.
    print(f"max z-variation: {zvar_max:.2e}")
    assert zvar_max < 0.02, f"z-structure in the radial solution: {zvar_max:.2e}"

    print("\nAll RZ FD conduction checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
