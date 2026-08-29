#!/usr/bin/env python3
#
# --- Analysis for the grid-FD conduction operator on circular field
# --- lines with the Sharma-Hammett cross-flux limiter: the anisotropic
# --- cross-term stress case (b-hat rotates against the grid everywhere).
# --- Gates the flux-level extrema bound: the global temperature maximum
# --- must never grow and the global minimum must never undershoot the
# --- uniform background -- the cross terms of the naive/NVD-limited
# --- operator can violate both at sharp field-line rotation. A loose
# --- angular-spread check confirms parallel conduction still operates.

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
BUMP_AMP = 1.0
T_E_BG_EV = 100.0
Q_E = 1.602176634e-19
KB = 1.380649e-23
T_BG = T_E_BG_EV * Q_E / KB

# The bound is a per-substep round-off property. Measured on this deck
# (40 steps): the sh arm accumulates ~1e-10 relative round-off in the
# background minimum while the smart arm MINTS new minima at the 3e-3
# level (the cross-term extrema-creation channel this limiter closes).
# 1e-6 sits four decades above the sh class and four below the smart
# violation.
BOUND_RTOL = 1e-6


def angular_spread(te):
    """Wrap-safe angular spread of the ring patch (circular statistics)."""
    n = te.shape[0]
    x = (np.arange(n) + 0.5) * DX - L / 2.0
    X, Z = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt(X**2 + Z**2)
    theta = np.arctan2(Z, X)
    dt_ = np.clip(te - T_BG, 0.0, None)
    band = np.abs(r - R0) < 1.5 * SIGMA0
    w = dt_ * band
    m0 = w.sum()
    if m0 <= 0.0:
        return 0.0
    c = (w * np.cos(theta)).sum() / m0
    s = (w * np.sin(theta)).sum() / m0
    return -2.0 * np.log(max(np.hypot(c, s), 1e-300))


def load_te(path):
    ds = yt.load(path)
    ad = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return np.squeeze(ad["boxlib", "Te"].value)


def main():
    dirs = sorted(glob.glob("diags/diag1" + "[0-9]" * 6))
    if len(dirs) < 4:
        raise RuntimeError("not enough plotfiles under diags/")
    yt.set_log_level(50)

    # Dump 0 shows the uniform seed (the bump is installed after step 1),
    # so the reference extrema come from the first post-bump dump.
    te1 = load_te(dirs[1])
    max_ref = te1.max()
    min_ref = te1.min()
    t_max_expected = T_BG * (1.0 + BUMP_AMP)
    if not (1.1 * T_BG <= max_ref <= 1.05 * t_max_expected):
        raise RuntimeError(
            f"first post-bump dump peak {max_ref:.6e} K is outside the "
            f"decaying-bump window (installed {t_max_expected:.6e} K): "
            "dump/bump ordering changed, analysis reference invalid"
        )

    print(f"{'dump':>28s} {'max(Te)/max0 - 1':>18s} "
          f"{'(T_bg - min)/T_bg':>18s} {'var_theta':>10s}")
    sig0 = angular_spread(te1)
    sig_last = sig0
    for d in dirs[1:]:
        te = load_te(d)
        over = te.max() / max_ref - 1.0
        under = (min(min_ref, T_BG) - te.min()) / T_BG
        sig_last = angular_spread(te)
        print(f"{d:>28s} {over:>18.3e} {under:>18.3e} {sig_last:>10.4f}")
        assert over <= BOUND_RTOL, (
            f"{d}: global max grew by {over:.3e} rel -- the cross-flux "
            "extrema bound is violated"
        )
        assert under <= BOUND_RTOL, (
            f"{d}: global min undershot the background by {under:.3e} "
            "rel -- the cross-flux extrema bound is violated"
        )

    # Parallel conduction must still spread the patch along the ring.
    assert sig_last > 1.5 * sig0, (
        f"angular spread did not grow (sig0 {sig0:.4f} -> {sig_last:.4f}): "
        "parallel conduction inert"
    )
    print("extrema bound held on every dump; parallel spreading confirmed")


if __name__ == "__main__":
    sys.exit(main())
