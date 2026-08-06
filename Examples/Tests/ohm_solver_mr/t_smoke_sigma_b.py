#!/usr/bin/env python3
"""Hybrid-PIC MR T0.3-lite analysis: sigma(B)/B0 of the MR run vs the
single-level control (both from t_smoke_inputs_2d_uniform), plus a final
snapshot mosaic to check for seam stripes.

Usage:
    python t_smoke_sigma_b.py MR_RUN_DIR CTL_RUN_DIR [OUT_PREFIX]

Each run dir must contain diags/diag1?????? plotfiles. Writes
<OUT_PREFIX>_sigma_b.png and <OUT_PREFIX>_snapshot.png (default prefix
"t_smoke").

Note on the density panel: the plotfile "rho" is re-deposited per level at
output time (RhoFunctor), so the level-0 values under the fine patch only
contain the level-0 particles (a taper artifact of the diagnostic, not of
the solver state). The mosaic therefore samples rho on a level-1 covering
grid and the seam check should lean on the solver-state fields (B, j).
"""

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt

yt.set_log_level(50)

B0 = 0.1  # background Bz (T), matches the input deck


def sigma_b_series(run_dir):
    """Return (times, sigma(B)/B0) from the level-0 covering grid of every
    plotfile (level 0 carries the restricted fine solution under the patch,
    so this compares MR and control at equal resolution)."""
    times, sigmas = [], []
    for pf in sorted(glob.glob(os.path.join(run_dir, "diags", "diag1??????"))):
        ds = yt.load(pf)
        ad = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        bx = np.array(ad["boxlib", "Bx"])
        by = np.array(ad["boxlib", "By"])
        bz = np.array(ad["boxlib", "Bz"]) - B0
        fluct = np.sqrt(np.mean(bx**2 + by**2 + bz**2))
        times.append(float(ds.current_time))
        sigmas.append(fluct / B0)
    return np.array(times), np.array(sigmas)


def main():
    mr_dir, ctl_dir = sys.argv[1], sys.argv[2]
    prefix = sys.argv[3] if len(sys.argv) > 3 else "t_smoke"

    t_mr, s_mr = sigma_b_series(mr_dir)
    t_ctl, s_ctl = sigma_b_series(ctl_dir)

    n = min(len(s_mr), len(s_ctl))
    ratio = s_mr[:n] / np.where(s_ctl[:n] > 0, s_ctl[:n], np.nan)
    print("# time(ns)  sigmaB/B0(MR)  sigmaB/B0(ctl)  ratio")
    for i in range(n):
        print(f"{t_mr[i] * 1e9:8.4f}  {s_mr[i]:.6e}  {s_ctl[i]:.6e}  {ratio[i]:.3f}")
    print(
        f"# final ratio MR/ctl = {ratio[-1]:.3f}, max ratio = {np.nanmax(ratio[1:]):.3f}"
    )

    # --- sigma(B)/B0 time series -----------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(t_mr * 1e9, s_mr, color="#3d65f5", lw=2, label="MR (max_level=1)")
    ax.plot(t_ctl * 1e9, s_ctl, color="#c95d20", lw=2, label="control (max_level=0)")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel(r"$\sigma(B)/B_0$")
    ax.set_title("Hybrid-PIC MR T0.3-lite: magnetic fluctuation level, level-0 view")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(f"{prefix}_sigma_b.png", dpi=270)

    # --- final snapshot mosaic (seam-stripe check) ------------------------
    ds = yt.load(sorted(glob.glob(os.path.join(mr_dir, "diags", "diag1??????")))[-1])
    dims = ds.domain_dimensions * ds.refine_by**1
    ad = ds.covering_grid(1, ds.domain_left_edge, dims)
    ext = [
        float(ds.domain_left_edge[1]) * 1e3,
        float(ds.domain_right_edge[1]) * 1e3,
        float(ds.domain_left_edge[0]) * 1e3,
        float(ds.domain_right_edge[0]) * 1e3,
    ]
    fields = [
        ("rho", "rho (diagnostic; per-level deposit)", "Blues", None),
        ("Bx", "Bx fluctuation (T)", "RdBu_r", "sym"),
        ("jz", "jz (A/m^2)", "RdBu_r", "sym"),
    ]
    fig, axs = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for axi, (f, title, cmap, sym) in zip(axs, fields):
        a = np.array(ad["boxlib", f])[:, :, 0]
        if sym == "sym":
            a = a - np.median(a)
            vmax = np.percentile(np.abs(a), 99.5)
            im = axi.imshow(
                a,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
                extent=ext[:2] + ext[2:],
            )
        else:
            im = axi.imshow(
                a, origin="lower", aspect="auto", cmap=cmap, extent=ext[:2] + ext[2:]
            )
        axi.set_title(title, fontsize=11)
        axi.set_ylabel("x (mm)")
        fig.colorbar(im, ax=axi, pad=0.01)
    axs[-1].set_xlabel("z (mm)")
    fig.suptitle(f"MR run, final step (t = {float(ds.current_time) * 1e9:.3f} ns)")
    fig.tight_layout()
    fig.savefig(f"{prefix}_snapshot.png", dpi=270)
    print(f"wrote {prefix}_sigma_b.png and {prefix}_snapshot.png")


if __name__ == "__main__":
    main()
