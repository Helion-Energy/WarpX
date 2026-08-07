#!/usr/bin/env python3
"""Hybrid-PIC dynamic-MR (P1.5) gate analysis.

Subcommands (all paths are run directories containing diags/diag1??????):

  fielddiff RUN TEST_RUN STEP [STEP ...]
      L2 norm of the B-field difference between two runs at the given steps,
      reported per level present in both runs, normalized by L2(B_ref).
      Gate 3: activate-at-t~0 vs static init.

  sigma RUN [RUN ...] --labels L1,L2 --out PREFIX [--b0 B0]
      sigma(B)/B0 trace of each run from the level-0 covering grid
      (overlaid), plus the max |divB|*dx/max|B| trace. Gates 4/5/6.

  energy RUN --out PREFIX
      Field/particle energy traces from reducedfiles, flags NaN/blow-up.

  levels RUN
      Prints step -> number of AMR levels of every plotfile (activation/
      relocation/removal audit), plus the level-1 box extents when present.
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt

yt.set_log_level(50)


def plotfiles(run):
    return sorted(glob.glob(os.path.join(run, "diags", "diag1??????")))


def plotfile_at_step(run, step):
    p = os.path.join(run, "diags", f"diag1{step:06d}")
    if not os.path.isdir(p):
        raise FileNotFoundError(p)
    return p


def load_level_fields(ds, lev, fields):
    """Covering grid of one level over the union bounding box of that level."""
    if lev == 0:
        le, dims = ds.domain_left_edge, ds.domain_dimensions
    else:
        grids = [g for g in ds.index.grids if g.Level == lev]
        le_all = np.array([g.LeftEdge.to_value() for g in grids])
        re_all = np.array([g.RightEdge.to_value() for g in grids])
        le = le_all.min(axis=0)
        re = re_all.max(axis=0)
        dds = grids[0].dds.to_value()
        dims = np.rint((re - le) / dds).astype(int)
        le = ds.arr(le, "code_length")
    cg = ds.covering_grid(lev, le, dims)
    return {f: np.asarray(cg["boxlib", f]) for f in fields}


def cmd_fielddiff(args):
    fields = ["Bx", "By", "Bz"]
    for step in args.steps:
        ds_a = yt.load(plotfile_at_step(args.ref, step))
        ds_b = yt.load(plotfile_at_step(args.test, step))
        nlev = min(ds_a.index.max_level, ds_b.index.max_level) + 1
        print(
            f"step {step}: t_ref={float(ds_a.current_time):.6e} "
            f"t_test={float(ds_b.current_time):.6e}"
        )
        for lev in range(nlev):
            fa = load_level_fields(ds_a, lev, fields)
            fb = load_level_fields(ds_b, lev, fields)
            num = sum(np.sum((fa[f] - fb[f]) ** 2) for f in fields)
            den = sum(np.sum(fa[f] ** 2) for f in fields)
            rel = np.sqrt(num / den) if den > 0 else float("nan")
            print(f"  lev {lev}: L2(B_test - B_ref)/L2(B_ref) = {rel:.6e}")


def sigma_series(run, b0):
    t, s, d = [], [], []
    for pf in plotfiles(run):
        ds = yt.load(pf)
        cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        bx = np.asarray(cg["boxlib", "Bx"])
        by = np.asarray(cg["boxlib", "By"])
        bz = np.asarray(cg["boxlib", "Bz"]) - b0
        t.append(float(ds.current_time))
        s.append(np.sqrt(np.mean(bx**2 + by**2 + bz**2)) / b0)
        try:
            divb = np.asarray(cg["boxlib", "divB"])
            bmax = max(np.abs(bx).max(), np.abs(by).max(), np.abs(bz + b0).max())
            dx = float(ds.index.grids[0].dds.to_value()[0])
            d.append(np.abs(divb).max() * dx / bmax)
        except Exception:
            d.append(np.nan)
    return np.array(t), np.array(s), np.array(d)


def cmd_sigma(args):
    labels = args.labels.split(",")
    fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for run, lab in zip(args.runs, labels):
        t, s, d = sigma_series(run, args.b0)
        axs[0].plot(t * 1e9, s, label=lab, lw=1.2)
        axs[1].semilogy(t * 1e9, d, label=lab, lw=1.2)
        print(
            f"{lab}: n={len(t)} sigma[final]={s[-1]:.6e} "
            f"max divB rel={np.nanmax(d):.3e}"
        )
    axs[0].set_ylabel(r"$\sigma(B)/B_0$ (level 0)")
    axs[0].legend()
    axs[1].set_ylabel(r"max $|\nabla\!\cdot\!B|\,dx/\max|B|$ (level 0)")
    axs[1].set_xlabel("t (ns)")
    axs[1].legend()
    fig.tight_layout()
    out = f"{args.out}_sigma.png"
    fig.savefig(out, dpi=270)
    print(f"wrote {out}")


def cmd_energy(args):
    ok = True
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in ("field_energy", "part_energy"):
        f = os.path.join(args.run, "diags", "reducedfiles", f"{name}.txt")
        if not os.path.exists(f):
            print(f"missing {f}")
            continue
        data = np.loadtxt(f, skiprows=1)
        t = data[:, 1]
        total = data[:, 2]
        if not np.all(np.isfinite(total)):
            ok = False
            print(f"{name}: NON-FINITE values detected")
        ax.plot(t * 1e9, total, label=f"{name} total")
        print(
            f"{name}: first={total[0]:.6e} last={total[-1]:.6e} "
            f"max={total.max():.6e} (max/first={total.max() / total[0]:.4f})"
        )
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("energy (J)")
    ax.legend()
    fig.tight_layout()
    out = f"{args.out}_energy.png"
    fig.savefig(out, dpi=270)
    print(f"wrote {out}")
    print("energy trace:", "OK (finite)" if ok else "FAILED (non-finite)")


def cmd_levels(args):
    prev = None
    for pf in plotfiles(args.run):
        ds = yt.load(pf)
        nlev = ds.index.max_level + 1
        step = int(os.path.basename(pf)[-6:])
        if nlev != prev:
            msg = f"step {step:6d}: {nlev} level(s)"
            if nlev > 1:
                grids = [g for g in ds.index.grids if g.Level == 1]
                le = np.array([g.LeftEdge.to_value() for g in grids]).min(0)
                re = np.array([g.RightEdge.to_value() for g in grids]).max(0)
                msg += f"  lev1 bbox {le} -> {re}"
            print(msg)
            prev = nlev
    print(f"final: step {step}: {nlev} level(s)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("fielddiff")
    q.add_argument("ref")
    q.add_argument("test")
    q.add_argument("steps", type=int, nargs="+")
    q.set_defaults(func=cmd_fielddiff)

    q = sub.add_parser("sigma")
    q.add_argument("runs", nargs="+")
    q.add_argument("--labels", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--b0", type=float, default=0.1)
    q.set_defaults(func=cmd_sigma)

    q = sub.add_parser("energy")
    q.add_argument("run")
    q.add_argument("--out", required=True)
    q.set_defaults(func=cmd_energy)

    q = sub.add_parser("levels")
    q.add_argument("run")
    q.set_defaults(func=cmd_levels)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
