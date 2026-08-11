#!/usr/bin/env python
"""RZ-first calibration deliverable: radial density/Bz profiles vs time from
the RZ liftoff arm — front shape and speed through the would-be 3D patch-seam
radius. Produces a spacetime map of rho and Bz (azimuthal-mean = the full RZ
m=0 field), the front trajectory, front speed, and the crossing timings at
r = 0.5 m and the realized 3D patch face radius.

Usage: rz_front.py RUNDIR [--face 0.4583] [--out rz_front]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from m4_battery import FRONT_THR, RADII, analyze_rz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--face", type=float, default=0.4583)
    ap.add_argument("--out", default="rz_front")
    args = ap.parse_args()

    d = analyze_rz(args.rundir)
    t, rf = d["t"], d["rfront"]

    def crossing(rtgt):
        k = np.where(rf <= rtgt)[0]
        # first sustained crossing after liftoff (ignore t=0 threshold jitter)
        k = k[k > 2]
        return t[k[0]] if len(k) else np.nan

    t50, tface = crossing(0.5), crossing(args.face)
    ipk = int(np.argmax(d["rhopk"]))
    # front speed through the seam radius (centered difference around t50)
    k50 = int(np.argmin(np.abs(t - t50)))
    ks = slice(max(k50 - 2, 0), min(k50 + 3, len(t)))
    v_front = np.polyfit(t[ks], rf[ks], 1)[0]  # m/us
    print(
        f"RZ front calibration: cross r=0.5 at t={t50:.3f} us; "
        f"cross face {args.face:.4f} at t={tface:.3f} us; "
        f"front speed through seam = {v_front * 1e6:.3e} m/s; "
        f"peak compression t={d['t'][ipk]:.3f} us "
        f"(rho={d['rhopk'][ipk]:.1f} C/m^3 at r={d['rcol'][ipk]:.3f} m)"
    )

    fig, ax = plt.subplots(1, 3, figsize=(19, 6.5), constrained_layout=True)
    ext = [t[0], t[-1], RADII[0], RADII[-1]]
    im0 = ax[0].imshow(
        d["m0prof"].T, origin="lower", aspect="auto", extent=ext, cmap="viridis"
    )
    ax[0].set_title("rho m0 (C/m^3)")
    im1 = ax[1].imshow(
        d["bz0prof"].T,
        origin="lower",
        aspect="auto",
        extent=ext,
        cmap="RdBu_r",
        vmin=-np.max(np.abs(d["bz0prof"])),
        vmax=np.max(np.abs(d["bz0prof"])),
    )
    ax[1].set_title("Bz m0 (T)")
    for a, im in ((ax[0], im0), (ax[1], im1)):
        a.plot(t, rf, "w-", lw=1.6, label="front (outer half-max)")
        a.axhline(0.5, color="w", ls=":", lw=1)
        a.axhline(args.face, color="w", ls="--", lw=1)
        a.set_xlabel("t (us)")
        a.set_ylabel("r (m)")
        a.legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=a, shrink=0.85)
    ax[2].plot(t, rf, "k-", lw=2, label="front radius")
    ax[2].plot(t, d["rcol"], "C2--", lw=1.5, label="densest ring")
    ax[2].axhline(0.5, color="gray", ls=":", lw=1)
    ax[2].axhline(args.face, color="gray", ls="--", lw=1)
    for tt, nm in ((t50, "cross 0.5"), (tface, "cross face"), (d["t"][ipk], "peak")):
        ax[2].axvline(tt, color="C3", ls=":", lw=1)
        ax[2].text(tt, 0.75, f" {nm}\n {tt:.2f} us", fontsize=8, color="C3")
    ax[2].set_xlabel("t (us)")
    ax[2].set_ylabel("r (m)")
    ax[2].grid(alpha=0.3)
    ax[2].legend()
    ax[2].set_title(
        f"front speed through seam {v_front * 1e6:.2e} m/s; thr {FRONT_THR:.0f} C/m^3"
    )
    fig.suptitle("RZ liftoff reference: radial profiles vs time (m=0)")
    fig.savefig(f"{args.out}.png", dpi=280)
    print(f"saved {args.out}.png")


if __name__ == "__main__":
    main()
