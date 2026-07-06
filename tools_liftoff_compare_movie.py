#!/usr/bin/env python3
"""Side-by-side XY (z=5.0) movie of the liftoff: collocated vs Yee (+LSQ).

2x2 compare: rows {rho, Bz} x cols {collocated, Yee}. Each column runs to its own
last available frame; when one run ends earlier (choke), its panes FREEZE on the
last frame while the other continues, with a "[stopped @ t=...]" tag. Also emits
per-case individual movies.

Run where the diags live + yt/matplotlib/ffmpeg are available:
  python tools_liftoff_compare_movie.py --coll liftoff_coll/diags --yee liftoff_yee/diags \
      --prefix field_diag --outdir .
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt
from matplotlib.animation import FFMpegWriter, FuncAnimation

yt.set_log_level(50)

ZSLICE = 5.0
RES = 320
WIDTH = 2.0  # domain [-1,1]
_keep = []


def frames(diag_dir, prefix):
    if not os.path.isdir(diag_dir):
        return []
    out = [
        x
        for x in os.listdir(diag_dir)
        if x.startswith(prefix) and x[len(prefix) :].isdigit()
    ]
    return sorted(out, key=lambda s: int(s[len(prefix) :]))


def slice_xy(diag_dir, name, field):
    ds = yt.load(os.path.join(diag_dir, name))
    _keep.append(ds)
    frb = ds.slice("z", ZSLICE).to_frb(
        width=(WIDTH, "m"), resolution=RES, height=(WIDTH, "m")
    )
    return np.array(frb["boxlib", field]), float(ds.current_time)


def cscale(diag_dir, names, field, pct=99.0):
    lo, hi = 0.0, 0.0
    for nm in names[:: max(1, len(names) // 6)]:
        a, _ = slice_xy(diag_dir, nm, field)
        lo = min(lo, np.percentile(a, 100 - pct))
        hi = max(hi, np.percentile(a, pct))
    return lo, hi


def make_compare(
    dc,
    dy,
    prefix,
    out,
    coll_label="collocated (nodal)",
    yee_label="Yee + LSQ (staggered)",
    title="collocated vs Yee",
):
    nc = [n for n in frames(dc, prefix) if int(n[len(prefix) :]) > 0]
    ny = [n for n in frames(dy, prefix) if int(n[len(prefix) :]) > 0]
    if not nc or not ny:
        print(f"[skip compare] coll={len(nc)} yee={len(ny)} frames")
        return
    nframes = max(len(nc), len(ny))
    print(f"compare: coll {len(nc)} frames, yee {len(ny)} frames -> {nframes}")
    rmax = max(cscale(dc, nc, "rho")[1], cscale(dy, ny, "rho")[1])
    bc, by = cscale(dc, nc, "Bz"), cscale(dy, ny, "Bz")
    bz_abs = max(abs(bc[0]), abs(bc[1]), abs(by[0]), abs(by[1])) or 1.0

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 11.8))
    ext = [-1, 1, -1, 1]
    ims, tags = {}, {}
    cfg = {
        (0, 0): ("coll", "rho"),
        (0, 1): ("yee", "rho"),
        (1, 0): ("coll", "Bz"),
        (1, 1): ("yee", "Bz"),
    }
    titles = {"coll": coll_label, "yee": yee_label}
    for (r, c), (case, fld) in cfg.items():
        ax = axs[r, c]
        if fld == "rho":
            im = ax.imshow(
                np.zeros((RES, RES)),
                origin="lower",
                extent=ext,
                cmap="viridis",
                vmin=0,
                vmax=rmax,
            )
        else:
            im = ax.imshow(
                np.zeros((RES, RES)),
                origin="lower",
                extent=ext,
                cmap="RdBu_r",
                vmin=-bz_abs,
                vmax=bz_abs,
            )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title(titles[case], fontsize=14)
        if c == 0:
            ax.set_ylabel(fld, fontsize=14)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ims[(case, fld)] = im
        tags[(case, fld)] = ax.text(
            0.03,
            0.03,
            "",
            transform=ax.transAxes,
            color="red",
            fontsize=11,
            weight="bold",
        )
    sup = fig.suptitle("", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    def draw(i):
        t = 0.0
        for case, dd, nn in (("coll", dc, nc), ("yee", dy, ny)):
            j = min(i, len(nn) - 1)
            stopped = i > len(nn) - 1
            rho, tt = slice_xy(dd, nn[j], "rho")
            bz, _ = slice_xy(dd, nn[j], "Bz")
            ims[(case, "rho")].set_data(rho.T)
            ims[(case, "Bz")].set_data(bz.T)
            if case == "coll" or not stopped:
                t = max(t, tt)
            msg = f"[stopped @ {tt * 1e6:.2f} us]" if stopped else ""
            tags[(case, "rho")].set_text(msg)
            tags[(case, "Bz")].set_text(msg)
        sup.set_text(f"Liftoff: {title}   XY @ z={ZSLICE} m   t = {t * 1e6:.3f} us")
        _keep.clear()
        return list(ims.values()) + list(tags.values()) + [sup]

    FuncAnimation(fig, draw, frames=nframes, blit=False).save(
        out, writer=FFMpegWriter(fps=12, bitrate=9000), dpi=130
    )
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coll", required=True, help="collocated diags dir")
    ap.add_argument("--yee", required=True, help="yee diags dir")
    ap.add_argument("--prefix", default="field_diag")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--coll-label", default="collocated (nodal)")
    ap.add_argument("--yee-label", default="Yee + LSQ (staggered)")
    ap.add_argument("--title", default="collocated vs Yee")
    ap.add_argument("--out", default="liftoff_coll_vs_yee_xy.mp4")
    a = ap.parse_args()
    make_compare(
        a.coll,
        a.yee,
        a.prefix,
        os.path.join(a.outdir, a.out),
        coll_label=a.coll_label,
        yee_label=a.yee_label,
        title=a.title,
    )
