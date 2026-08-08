#!/usr/bin/env python3
"""Analysis for the T1.7 low-density-seam battery (see t1_7_8_results.md).

Reads t1_runs/t17_*/t17_reduced.npz + params.json, evaluates the
pre-registered criteria P1/P2/P4 and the P3 threshold-leg outcomes, prints
markdown tables, and writes figures:

  t1_7_maps.png    rho and max|E| space-time maps, floor position, MR vs ctl
  t1_7_seams.png   seam-band max|E| time series, all positions, MR vs ctl
  t1_7_ratios.png  P2 ratio summary (E and Hall proxy, both windows)
"""

import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "t1_runs")

SEAMS_LI = (48.0, 80.0)
BAND_LI = 2.0
W_EARLY = (0.2, 2.0)
W_FULL = (0.2, 8.0)
W_LATE = (4.0, 8.0)
W_BASE = (0.5, 2.0)


def load(name):
    d = dict(np.load(os.path.join(RUNS, name, "t17_reduced.npz")))
    with open(os.path.join(RUNS, name, "params.json")) as fh:
        p = json.load(fh)
    d["params"] = p
    d["tt"] = d["t"] / p["t_ci"]
    d["zli"] = d["z0"] / p["l_i"]
    d["vAB0"] = p["vA"] * p["B0"]
    marker = ".done" if os.path.exists(os.path.join(RUNS, name, ".done")) else ".failed"
    d["marker"] = marker
    return d


def last_step(name):
    """Last completed step from run.log (binary-safe)."""
    with open(os.path.join(RUNS, name, "run.log"), "rb") as fh:
        txt = fh.read().decode(errors="replace")
    steps = re.findall(r"STEP (\d+) ends", txt)
    return int(steps[-1]) if steps else 0


def band_series(d, key, seam_li):
    band = np.abs(d["zli"] - seam_li) <= BAND_LI
    return d[key + "0"][:, band].max(axis=1)


def wmean(t, y, w):
    sel = (t >= w[0]) & (t <= w[1])
    return float(y[sel].mean())


def wmax(t, y, w):
    sel = (t >= w[0]) & (t <= w[1])
    return float(y[sel].max())


def wmedian(t, y, w):
    sel = (t >= w[0]) & (t <= w[1])
    return float(np.median(y[sel]))


def p1_row(name, d):
    tt = d["tt"]
    mon = d["amax0"][:, 3] / d["params"]["B0"]
    rows = []
    for s in SEAMS_LI:
        e = band_series(d, "emax", s) / d["vAB0"]
        base = wmedian(tt, e, W_BASE)
        late = wmax(tt, e, W_LATE)
        rows.append((late, base, late / base))
    worst = max(rows, key=lambda r: r[2])
    ok = d["marker"] == ".done" and mon.max() < 0.4 and worst[2] <= 3.0
    return dict(
        name=name,
        done=d["marker"] == ".done",
        mon_max=float(mon.max()),
        late_over_base=worst[2],
        late=worst[0],
        base=worst[1],
        ok=ok,
    )


def p2_ratios(mr, ctl):
    out = {}
    for key, label in (("emax", "E"), ("hallmax", "Hall")):
        for si, s in enumerate(SEAMS_LI):
            e_mr = band_series(mr, key, s)
            e_ct = band_series(ctl, key, s)
            for w, wl in ((W_EARLY, "early"), (W_FULL, "full")):
                r = wmean(mr["tt"], e_mr, w) / wmean(ctl["tt"], e_ct, w)
                out[f"{label}_seam{'lo' if si == 0 else 'hi'}_{wl}"] = r
    return out


def main():
    positions = ["inramp", "floor", "above", "nogap"]
    arms = {}
    for pos in positions:
        for a in ("mr", "ctl"):
            arms[f"{pos}_{a}"] = load(f"t17_{pos}_e6_{a}")
    fine = load("t17_floor_e6_fine")
    h0mr = load("t17_floor_e6_holm0_mr")
    h0ct = load("t17_floor_e6_holm0_ctl")
    e6long = load("t17_floor_e6long_mr")

    # ---------------- P1 ------------------------------------------
    print("\n### P1 -- completion / no blow-up (eta = 1e-6 arms)\n")
    print("| arm | done | max monitor (B0) | late max / base median (seam E) | P1 |")
    print("|---|---|---|---|---|")
    p1_all = True
    p1_arms = {f"t17_{k.replace('_', '_e6_')}": v for k, v in arms.items()}
    p1_arms["t17_floor_e6_fine"] = fine
    p1_arms["t17_floor_e6_holm0_mr"] = h0mr
    p1_arms["t17_floor_e6_holm0_ctl"] = h0ct
    p1_arms["t17_floor_e6long_mr"] = e6long
    for name, d in p1_arms.items():
        r = p1_row(name, d)
        p1_all &= r["ok"]
        print(
            f"| {name} | {r['done']} | {r['mon_max']:.3f} | "
            f"{r['late']:.3g}/{r['base']:.3g} = {r['late_over_base']:.2f} | "
            f"{'PASS' if r['ok'] else 'FAIL'} |"
        )
    print(f"\nP1 overall: {'PASS' if p1_all else 'FAIL'}")

    # ---------------- P2 ------------------------------------------
    print("\n### P2 -- seam-band MR/ctl ratios (time-mean, level 0)\n")
    print("| position | window | E lo | E hi | Hall lo | Hall hi | P2 (<=2.0) |")
    print("|---|---|---|---|---|---|---|")
    p2_all = True
    p2_store = {}
    for pos in positions:
        r = p2_ratios(arms[f"{pos}_mr"], arms[f"{pos}_ctl"])
        p2_store[pos] = r
        for wl in ("early", "full"):
            vals = [
                r[f"E_seamlo_{wl}"],
                r[f"E_seamhi_{wl}"],
                r[f"Hall_seamlo_{wl}"],
                r[f"Hall_seamhi_{wl}"],
            ]
            ok = max(vals) <= 2.0
            p2_all &= ok
            print(
                f"| {pos} | {wl} | "
                + " | ".join(f"{v:.2f}" for v in vals)
                + f" | {'PASS' if ok else 'FAIL'} |"
            )
    print(f"\nP2 overall: {'PASS' if p2_all else 'FAIL'}")

    # ---------------- P4 (holmstrom attribution) -------------------
    print("\n### P4 -- holmstrom on/off at the stabilized point\n")
    r_h1 = p2_ratios(arms["floor_mr"], arms["floor_ctl"])
    r_h0 = p2_ratios(h0mr, h0ct)
    print("| metric | holm=1 MR/ctl | holm=0 MR/ctl | h0/h1 |")
    print("|---|---|---|---|")
    p4_all = True
    for k in sorted(r_h1):
        ratio = r_h0[k] / r_h1[k]
        ok = 0.5 <= ratio <= 2.0
        p4_all &= ok
        print(f"| {k} | {r_h1[k]:.2f} | {r_h0[k]:.2f} | {ratio:.2f} |")
    print(f"\nP4 overall: {'PASS' if p4_all else 'FAIL'}")

    # ---------------- P3 (threshold legs) --------------------------
    print("\n### P3 -- eta = 1e-7 threshold-documentation legs\n")
    print("| leg | last step | steps asked | expectation | met |")
    print("|---|---|---|---|---|")
    exp = {
        "t17_floor_e7_mr": ("dies < 100", lambda s: s < 100),
        "t17_floor_e7_fine": ("dies < 100", lambda s: s < 100),
        "t17_floor_e7_ctl": ("completes 667", lambda s: s >= 667),
        "t17_nogap_e7_mr": ("completes 667", lambda s: s >= 667),
    }
    p3_all = True
    for leg, (txt, fn) in exp.items():
        s = last_step(leg)
        ok = fn(s)
        p3_all &= ok
        print(f"| {leg} | {s} | 667 | {txt} | {'YES' if ok else 'NO'} |")
    print(f"\nP3 overall: {'PASS' if p3_all else 'FAIL'}")

    # ---------------- extra tables ---------------------------------
    print("\n### Seam-band absolute levels (E in vA*B0, time-mean, full window)\n")
    print("| arm | E seam_lo | E seam_hi | Hall seam_lo | Hall seam_hi |")
    print("|---|---|---|---|---|")
    for pos in positions:
        for a in ("mr", "ctl"):
            d = arms[f"{pos}_{a}"]
            row = []
            for key in ("emax", "hallmax"):
                for s in SEAMS_LI:
                    row.append(
                        wmean(d["tt"], band_series(d, key, s) / d["vAB0"], W_FULL)
                    )
            print(f"| t17_{pos}_e6_{a} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    d = fine
    row = [
        wmean(d["tt"], band_series(d, k, s) / d["vAB0"], W_FULL)
        for k in ("emax", "hallmax")
        for s in SEAMS_LI
    ]
    print("| t17_floor_e6_fine | " + " | ".join(f"{v:.3f}" for v in row) + " |")

    # e6long soak: first vs last quarter of the horizon
    tt = e6long["tt"]
    e = np.maximum(band_series(e6long, "emax", 48.0), band_series(e6long, "emax", 80.0))
    q1 = wmean(tt, e, (0.2, 3.9)) / e6long["vAB0"]
    q4 = wmean(tt, e, (11.7, 15.6)) / e6long["vAB0"]
    print(
        f"\ne6long soak (15.6 t_ci): seam-band E mean first quarter {q1:.3f} "
        f"-> last quarter {q4:.3f} vA*B0 (ratio {q4 / q1:.2f}); monitor max "
        f"{e6long['amax0'][:, 3].max() / 0.25:.3f} B0"
    )

    # ---------------- figures --------------------------------------
    fl_mr, fl_ct = arms["floor_mr"], arms["floor_ctl"]
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    q_e = 1.602176634e-19
    n0 = fl_mr["params"]["n0"]
    ext = [fl_mr["zli"][0], fl_mr["zli"][-1], 0, fl_mr["tt"][-1]]
    im = axs[0, 0].imshow(
        fl_mr["rhomean0"] / (q_e * n0),
        origin="lower",
        aspect="auto",
        extent=ext,
        vmin=0,
        vmax=1.2,
        cmap="viridis",
    )
    axs[0, 0].set_title("floor MR: rho / (q n0)")
    fig.colorbar(im, ax=axs[0, 0])
    for ax, d, lab in ((axs[0, 1], fl_mr, "floor MR"), (axs[1, 0], fl_ct, "floor ctl")):
        im = ax.imshow(
            d["emax0"] / d["vAB0"],
            origin="lower",
            aspect="auto",
            extent=ext,
            vmin=0,
            vmax=0.5,
            cmap="magma",
        )
        ax.set_title(f"{lab}: max$_x$|E| / (v$_A$B$_0$)")
        fig.colorbar(im, ax=ax)
    for ax in axs.flat[:3]:
        for s in SEAMS_LI:
            ax.axvline(s, color="w", ls="--", lw=0.8)
        ax.set_xlabel("z (l$_i$)")
        ax.set_ylabel("t (t$_{ci}$)")
    ax = axs[1, 1]
    for d, lab, c in (
        (fl_mr, "MR", "C0"),
        (fl_ct, "ctl", "C1"),
        (fine, "uniform fine", "C2"),
    ):
        ax.plot(d["tt"], band_series(d, "emax", 48.0) / d["vAB0"], c, label=lab, lw=1)
    ax.set_xlabel("t (t$_{ci}$)")
    ax.set_ylabel("seam-band max|E| / (v$_A$B$_0$)")
    ax.set_title("floor position, seam_lo band (z = 48 +- 2 l$_i$)")
    ax.legend()
    fig.suptitle("T1.7 floor-position maps (eta = 1e-6, holmstrom on)")
    fig.savefig(os.path.join(HERE, "t1_7_maps.png"), dpi=280)

    fig, axs = plt.subplots(
        2, 4, figsize=(18, 8), constrained_layout=True, sharey="row"
    )
    for j, pos in enumerate(positions):
        for i, s in enumerate(SEAMS_LI):
            ax = axs[i, j]
            for a, c in (("mr", "C0"), ("ctl", "C1")):
                d = arms[f"{pos}_{a}"]
                ax.plot(
                    d["tt"],
                    band_series(d, "emax", s) / d["vAB0"],
                    c,
                    lw=0.9,
                    label=a if (i == 0 and j == 0) else None,
                )
            ax.set_title(f"{pos}, seam {'lo' if i == 0 else 'hi'}")
            ax.set_xlabel("t (t$_{ci}$)")
            if j == 0:
                ax.set_ylabel("band max|E| / (v$_A$B$_0$)")
    axs[0, 0].legend()
    fig.suptitle("T1.7 seam-band max|E|, MR vs uniform-coarse control (eta = 1e-6)")
    fig.savefig(os.path.join(HERE, "t1_7_seams.png"), dpi=280)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(positions))
    for ax, wl in zip(axs, ("early", "full")):
        for k, (lab, off) in {
            "E": ("max|E|", -0.15),
            "Hall": ("Hall proxy", 0.15),
        }.items():
            vals = [
                max(p2_store[p][f"{k}_seamlo_{wl}"], p2_store[p][f"{k}_seamhi_{wl}"])
                for p in positions
            ]
            ax.bar(x + off, vals, 0.3, label=lab)
        ax.axhline(2.0, color="r", ls="--", lw=1, label="P2 bound")
        ax.set_xticks(x, positions)
        ax.set_ylabel("MR/ctl time-mean ratio (worse seam)")
        ax.set_title(f"{wl} window")
    axs[0].legend()
    fig.suptitle("T1.7 P2 seam-band ratios")
    fig.savefig(os.path.join(HERE, "t1_7_ratios.png"), dpi=280)
    print("\nfigures: t1_7_maps.png t1_7_seams.png t1_7_ratios.png")


if __name__ == "__main__":
    sys.exit(main())
