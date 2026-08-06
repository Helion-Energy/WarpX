#!/usr/bin/env python3
"""Hybrid-PIC MR seam audit: compare the SOLVER charge density (raw_fields
rho_fp, written as-is from the registry when diag1.plot_raw_fields = 1)
against the plotfile DIAGNOSTIC rho (RhoFunctor) along lineouts across the
fine-patch seam, on both levels, against the expected uniform value. Also
lines out |E| (raw Ex/Ey/Ez_fp) across the seam.

Usage:
    python t_smoke_rho_seam.py RUN_DIR PLOTFILE_NAME [OUT_PREFIX]

e.g.  python t_smoke_rho_seam.py runs/t03_mr_raw diag1000300 seam
Writes <OUT_PREFIX>_rho_z.png, <OUT_PREFIX>_rho_x.png, <OUT_PREFIX>_E_z.png.

Geometry is hard-wired to the t_smoke_inputs_2d_uniform deck: 32x256 coarse
cells, fine patch covering coarse cells x in [6,25], z in [62,194].
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt

yt.set_log_level(50)

RHO0 = 9.623617654281555e20 * 1.6021766339999999e-19  # n0 * q_e
# fine-patch coverage in coarse cells (from the run's grid summary)
PXLO, PXHI, PZLO, PZHI = 6, 26, 62, 194  # [lo, hi) cell ranges


def raw_level_array(ds, field, lev, shape):
    """Assemble corner-0 (node) values of a raw field onto a level-wide
    array indexed by cell (i, j); entry (i, j) is the node at the low
    corner of cell (i, j)."""
    out = np.full(shape, np.nan)
    dx = (ds.domain_width / ds.domain_dimensions / ds.refine_by**lev).d
    for g in ds.index.grids:
        if g.Level != lev:
            continue
        a = np.array(g[("raw", field)])[:, :, 0, 0]
        i0 = int(round(float((g.LeftEdge[0] - ds.domain_left_edge[0]).d) / dx[0]))
        j0 = int(round(float((g.LeftEdge[1] - ds.domain_left_edge[1]).d) / dx[1]))
        out[i0 : i0 + a.shape[0], j0 : j0 + a.shape[1]] = a
    return out


def diag_level_array(ds, lev, shape):
    out = np.full(shape, np.nan)
    dx = (ds.domain_width / ds.domain_dimensions / ds.refine_by**lev).d
    for g in ds.index.grids:
        if g.Level != lev:
            continue
        a = np.array(g[("boxlib", "rho")])[:, :, 0]
        i0 = int(round(float((g.LeftEdge[0] - ds.domain_left_edge[0]).d) / dx[0]))
        j0 = int(round(float((g.LeftEdge[1] - ds.domain_left_edge[1]).d) / dx[1]))
        out[i0 : i0 + a.shape[0], j0 : j0 + a.shape[1]] = a
    return out


def main():
    run_dir, pf = sys.argv[1], sys.argv[2]
    prefix = sys.argv[3] if len(sys.argv) > 3 else "seam"
    ds = yt.load(os.path.join(run_dir, "diags", pf))
    nx, nz = int(ds.domain_dimensions[0]), int(ds.domain_dimensions[1])

    rho_solver = [
        raw_level_array(ds, "rho_fp", 0, (nx, nz)),
        raw_level_array(ds, "rho_fp", 1, (2 * nx, 2 * nz)),
    ]
    rho_diag = [
        diag_level_array(ds, 0, (nx, nz)),
        diag_level_array(ds, 1, (2 * nx, 2 * nz)),
    ]
    e_solver = []
    for lev, shape in ((0, (nx, nz)), (1, (2 * nx, 2 * nz))):
        ex = raw_level_array(ds, "Ex_fp", lev, shape)
        ey = raw_level_array(ds, "Ey_fp", lev, shape)
        ez = raw_level_array(ds, "Ez_fp", lev, shape)
        e_solver.append(np.sqrt(ex**2 + ey**2 + ez**2))

    def lineout_fig(direction, fname, data_by_level, ylabel, ref=None):
        # (cut through a patch face) and (cut passing one cell from the
        # corner), for both levels
        fig, axs = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
        cuts = (
            [("face (x = patch center)", nx // 2), ("corner (x = edge+1)", PXLO + 1)]
            if direction == "z"
            else [
                ("face (z = patch center)", nz // 2),
                ("corner (z = edge+1)", PZLO + 1),
            ]
        )
        for ax, (label, c_idx) in zip(axs, cuts):
            for lev, ls in ((0, "-"), (1, "-")):
                r = 2**lev
                for name, arr, color in data_by_level(lev):
                    line = arr[c_idx * r, :] if direction == "z" else arr[:, c_idx * r]
                    coord = np.arange(line.size) / r  # coarse-cell units
                    ax.plot(
                        coord,
                        line,
                        ls,
                        lw=1.2 if lev else 1.8,
                        color=color,
                        alpha=0.9 if lev else 1.0,
                        label=f"{name} lev {lev}",
                    )
            if ref is not None:
                ax.axhline(ref, color="#555555", lw=1, ls=":", label="expected n0 e")
            lo, hi = (PZLO, PZHI) if direction == "z" else (PXLO, PXHI)
            for edge in (lo, hi):
                ax.axvline(edge, color="#999999", lw=0.8, ls="--")
            ax.set_title(f"cut through {label}", fontsize=11)
            ax.set_ylabel(ylabel)
            ax.legend(frameon=False, fontsize=8, ncol=3)
            ax.grid(alpha=0.2, lw=0.4)
        axs[-1].set_xlabel(f"{direction} (coarse cells); dashed = patch edges")
        fig.suptitle(f"{pf}: seam lineouts along {direction}")
        fig.tight_layout()
        fig.savefig(fname, dpi=270)
        print("wrote", fname)

    def rho_series(lev):
        return [
            ("solver rho_fp", rho_solver[lev], "#3d65f5" if lev == 0 else "#8fb0ff"),
            ("diag rho", rho_diag[lev], "#c95d20" if lev == 0 else "#f0a878"),
        ]

    def e_series(lev):
        return [("solver |E|", e_solver[lev], "#3d65f5" if lev == 0 else "#c95d20")]

    lineout_fig("z", f"{prefix}_rho_z.png", rho_series, "rho (C/m^3)", ref=RHO0)
    lineout_fig("x", f"{prefix}_rho_x.png", rho_series, "rho (C/m^3)", ref=RHO0)
    lineout_fig("z", f"{prefix}_E_z.png", e_series, "|E| (V/m)")

    # numeric summary: band statistics of solver vs diag on the fine level
    f = rho_solver[1]
    d = rho_diag[1]
    fx0, fx1, fz0, fz1 = 2 * PXLO, 2 * PXHI, 2 * PZLO, 2 * PZHI
    interior = np.s_[fx0 + 6 : fx1 - 6, fz0 + 6 : fz1 - 6]
    band = np.zeros_like(f, dtype=bool)
    band[fx0:fx1, fz0:fz1] = True
    band[fx0 + 4 : fx1 - 4, fz0 + 4 : fz1 - 4] = False
    print(f"expected rho = {RHO0:.3f}")
    print(
        f"solver fine: interior mean {np.nanmean(f[interior]):.2f}, "
        f"edge-band mean {np.nanmean(f[band]):.2f}, min {np.nanmin(f[band]):.2f}"
    )
    print(
        f"diag   fine: interior mean {np.nanmean(d[interior]):.2f}, "
        f"edge-band mean {np.nanmean(d[band]):.2f}, min {np.nanmin(d[band]):.2f}"
    )


if __name__ == "__main__":
    main()
