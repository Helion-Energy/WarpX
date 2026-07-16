#!/usr/bin/env python3
#
# --- Analysis for the QDSMC nonlinear-conduction front test: a hot slab
# --- released into a cold background with kappa ~ T^{5/2} develops the
# --- Zel'dovich-Barenblatt self-similar front x_f ~ t^{1/(m+1)} = t^{2/9}
# --- (porous-medium exponent m = 7/2 in one dimension). The front position
# --- is the outermost crossing of a fixed threshold above the background.

import argparse
import glob
import sys

import numpy as np
import yt

parser = argparse.ArgumentParser()
parser.add_argument("--rz", action="store_true",
                    help="RZ variant: the front runs along z")
args, _ = parser.parse_known_args()

# Deck constants (keep in sync with the inputs file)
L = 0.5
NX = 128
DX = L / NX
SIGMA0 = 6.0 * DX
T_BG_EV = 1.0
Q_E = 1.602176634e-19
KB = 1.380649e-23
T_BG = T_BG_EV * Q_E / KB
THRESH = 4.0 * T_BG  # front marker: well above background, deep in the tail

EXPONENT = 2.0 / 9.0


def front_position(te):
    # slab: average over the uniform direction (r in RZ, z in Cartesian)
    prof = te.mean(axis=0) if args.rz else te.mean(axis=1)
    x = (np.arange(prof.size) + 0.5) * DX - L / 2.0
    above = np.where(prof > THRESH)[0]
    if above.size == 0:
        return np.nan
    # outermost right-hand crossing, linearly interpolated
    i = above[-1]
    if i + 1 >= prof.size:
        return np.nan
    f = (prof[i] - THRESH) / (prof[i] - prof[i + 1])
    return x[i] + f * DX


def main():
    dirs = sorted(glob.glob("diags/diag1" + "[0-9]" * 6))
    if len(dirs) < 6:
        raise RuntimeError("not enough plotfiles under diags/")
    yt.set_log_level(50)

    times, fronts, sums = [], [], []
    print(f"{'t [s]':>10s} {'x_f [m]':>10s} {'x_f/dx':>8s} {'sum(dT)':>12s}")
    for d in dirs[1:]:  # step-0 dump predates the bump load
        ds = yt.load(d)
        t = float(ds.current_time)
        ad = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        te = np.squeeze(np.array(ad["boxlib", "Te"]))
        xf = front_position(te)
        m0 = (te - T_BG).sum()
        times.append(t)
        fronts.append(xf)
        sums.append(m0)
        print(f"{t:10.3e} {xf:10.4e} {xf / DX:8.2f} {m0:12.5e}")

    times = np.array(times)
    fronts = np.array(fronts)
    sums = np.array(sums)

    # 1) Self-similar exponent: fit log x_f vs log t once the front has
    #    forgotten the initial profile (x_f > 2 sigma_0).
    sel = fronts > 2.5 * SIGMA0
    assert sel.sum() >= 4, "front never reached the self-similar regime"
    slope = np.polyfit(np.log(times[sel]), np.log(fronts[sel]), 1)[0]
    print(f"\nfront exponent: {slope:.4f} (theory {EXPONENT:.4f})")
    assert abs(slope - EXPONENT) < 0.03, f"front exponent off: {slope:.4f}"

    # 2) Energy ledger: the bound covers the documented delta-form
    #    residual (~0.03%/step at this NPPC) over the 300-step run.
    ledger_err = abs(sums[-1] - sums[0]) / sums[0]
    print(f"slab energy drift: {ledger_err:.2%}")
    assert ledger_err < 0.15, f"slab energy drifted: {ledger_err:.2%}"

    # 3) Positivity / cold background intact: the minimum never dips below
    #    the background by more than deposit noise.
    ds = yt.load(dirs[-1])
    ad = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    te = np.squeeze(np.array(ad["boxlib", "Te"]))
    assert te.min() >= 0.5 * T_BG, "background cooled unphysically"

    # 4) RZ only: the cylindrically uniform slab must STAY radially
    #    uniform -- the off-plane kick fold, the cylindrical deposit
    #    volumes and the axis/wall handling all cancel in the r-profile.
    if args.rz:
        hot = te[:, np.abs(te.mean(axis=0) - te.mean()).argmax()]
        r_spread = (hot.max() - hot.min()) / hot.mean()
        print(f"radial spread at the hottest z: {r_spread:.2%}")
        assert r_spread < 0.10, f"radial nonuniformity: {r_spread:.2%}"

    print("\nAll QDSMC conduction-front checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
