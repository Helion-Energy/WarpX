#!/usr/bin/env python3
#
# --- Analysis for the parallel QDSMC conduction test on circular field
# --- lines: a hot patch on a ring spreads along the line as 1D arc-length
# --- diffusion, sigma_theta^2(t) = sigma_0^2 + 2 D t / r0^2, while its
# --- RADIAL width must stay put -- the field-line kernel transports no
# --- energy across lines by construction (del-Castillo-Negrete & Chacon,
# --- PRL 106, 195004 (2011)).

import glob
import sys

import numpy as np
import yt

# Deck constants (keep in sync with the inputs file)
L = 0.5
NX = 64
DX = L / NX
SIGMA0 = 3.0 * DX
R0 = 16.0 * DX
D = DX**2 / 2.0e-8 * 2.0
T_E_BG_EV = 100.0
Q_E = 1.602176634e-19
KB = 1.380649e-23
T_BG = T_E_BG_EV * Q_E / KB


def ring_moments(te):
    """Angular spread (wrap-safe) and radial width of the ring patch."""
    n = te.shape[0]
    x = (np.arange(n) + 0.5) * DX - L / 2.0
    X, Z = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt(X**2 + Z**2)
    theta = np.arctan2(Z, X)
    dt_ = np.clip(te - T_BG, 0.0, None)
    # Narrow band about the ring: the angular rate scales as 1/r^2, so
    # inner-radius tails would otherwise dominate the weighted variance.
    band = np.abs(r - R0) < 1.5 * SIGMA0
    w = dt_ * band
    m0 = w.sum()
    # circular statistics: sigma_theta^2 = -2 ln |<e^{i theta}>|
    c = (w * np.cos(theta)).sum() / m0
    s = (w * np.sin(theta)).sum() / m0
    var_theta = -2.0 * np.log(np.hypot(c, s))
    # radial width about the ring
    mr = (w * r).sum() / m0
    var_r = (w * (r - mr) ** 2).sum() / m0
    return m0, var_theta, var_r


def main():
    dirs = sorted(glob.glob("diags/diag1" + "[0-9]" * 6))
    if len(dirs) < 5:
        raise RuntimeError("not enough plotfiles under diags/")
    yt.set_log_level(50)

    times, vth, vr, sums = [], [], [], []
    print(f"{'t [s]':>10s} {'var_theta':>10s} {'expected':>10s} "
          f"{'sigma_r/dx':>10s} {'sum(dT)':>12s}")
    for d in dirs[1:]:  # step-0 dump predates the bump load
        ds = yt.load(d)
        t = float(ds.current_time)
        ad = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        te = np.squeeze(np.array(ad["boxlib", "Te"]))
        m0, var_theta, var_r = ring_moments(te)
        times.append(t)
        vth.append(var_theta)
        vr.append(var_r)
        sums.append(m0)
        expected = (SIGMA0 / R0) ** 2 + 2.0 * D * t / R0**2
        print(f"{t:10.3e} {var_theta:10.4f} {expected:10.4f} "
              f"{np.sqrt(var_r) / DX:10.3f} {m0:12.5e}")

    times = np.array(times)
    vth = np.array(vth)
    vr = np.array(vr)

    # 1) Arc-length diffusion rate (angular variance growth).
    slope = np.polyfit(times, vth, 1)[0]
    slope_err = abs(slope - 2.0 * D / R0**2) / (2.0 * D / R0**2)
    print(f"\nangular variance slope: {slope:.4e} "
          f"(theory {2.0 * D / R0**2:.4e}, rel err {slope_err:.2%})")
    assert slope_err < 0.25, f"parallel diffusion rate off: {slope_err:.2%}"

    # 2) Cross-field confinement: the radial width must not grow by more
    #    than a small fraction while the patch spreads freely along the
    #    line. The implied kappa_perp/kappa_par pollution follows from the
    #    variance growth ratio.
    r_growth = vr[-1] - vr[0]
    theta_growth = (vth[-1] - vth[0]) * R0**2
    pollution = max(r_growth, 0.0) / theta_growth
    print(f"radial width: {np.sqrt(vr[0])/DX:.2f} -> {np.sqrt(vr[-1])/DX:.2f} dx"
          f" (implied kappa_perp/kappa_par ~ {pollution:.2e})")
    assert np.sqrt(vr[-1]) < np.sqrt(vr[0]) * 1.25 + 0.5 * DX, (
        "cross-field leakage: radial width grew "
        f"{np.sqrt(vr[-1])/np.sqrt(vr[0]):.2f}x")

    # 3) Ledger on the band energy (loose: the band window clips tails).
    drift = abs(sums[-1] - sums[0]) / sums[0]
    print(f"band energy drift: {drift:.2%}")
    assert drift < 0.25, f"band energy drifted: {drift:.2%}"

    print("\nAll QDSMC parallel-conduction checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
