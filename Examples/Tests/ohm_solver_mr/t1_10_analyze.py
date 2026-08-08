#!/usr/bin/env python3
"""T1.10 analysis: onset / growth rate / saturation of the cold-beam
finite-grid instability, MR vs uniform-coarse and uniform-fine controls.

Reads t1_runs/t110_*/t1_lineouts.npz (from t1_910_battery.py), computes the
region fluctuation energies

    W(t) = < |b_perp(z) - <b_perp>_z|^2 >_region / B0^2,   b_perp = bx + i by

on level 0 in the coarse-region windows (>= 8 l_i from both seams), the seam
bands (+-4 l_i), and the patch-interior window; for the MR arm also on level 1.
Onset = first sustained crossing of 10x the early-time floor; amplitude growth
rate gamma = 0.5 dlnW/dt fitted between the 10x-floor crossing and 0.25x the
saturation (or peak) level; saturation = median W over the trailing plateau.

Writes t1_10_growth.png, t1_10_spectra.png and prints the verdict table.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "t1_runs")

B0 = 0.25
# Okabe-Ito (CVD-safe), fixed assignment per arm entity
COL = {
    "mr": "#0072B2",
    "ctl": "#E69F00",
    "fine": "#009E73",
    "mr_eta7": "#56B4E9",
    "ctl_eta7": "#D55E00",
}
LBL = {
    "mr": "MR (patch)",
    "ctl": "uniform coarse",
    "fine": "uniform fine",
    "mr_eta7": "MR, eta=1e-7",
    "ctl_eta7": "uniform coarse, eta=1e-7",
}


def load(name):
    d = np.load(os.path.join(RUNS, name, "t1_lineouts.npz"))
    p = json.load(open(os.path.join(RUNS, name, "params.json")))
    return d, p


def region_W(d, p, zwins, level=0):
    """W(t) over a union of z-windows (in l_i) on the given level."""
    li = p["l_i"]
    z = d[f"z{level}"] / li
    b = (d[f"bx{level}"] + 1j * d[f"by{level}"]).astype(np.complex64)
    m = np.zeros(z.shape, dtype=bool)
    for lo, hi in zwins:
        m |= (z >= lo) & (z <= hi)
    bb = b[:, m]
    bb = bb - bb.mean(axis=1, keepdims=True)
    return (np.abs(bb) ** 2).mean(axis=1) / B0**2


def onset_growth_sat(t, W, tci):
    """Return (floor, t_onset, gamma_wci, gamma_err, W_sat, flags)."""
    early = (t >= 0.2 * tci) & (t <= 1.0 * tci)
    floor = np.median(W[early])
    thr = 10.0 * floor
    above = W >= thr
    t_on = np.nan
    for i in range(1, len(t) - 3):
        if above[i] and above[i + 1] and above[i + 2] and above[i + 3]:
            t_on = t[i]
            i_on = i
            break
    flags = []
    if not np.isfinite(t_on):
        return floor, np.nan, np.nan, np.nan, np.nan, ["no-onset"]
    # saturation: trailing 15% of the horizon, if flat; else use running peak
    ntail = max(5, len(t) // 7)
    Wtail = W[-ntail:]
    lnslope_tail = np.polyfit(t[-ntail:] / tci, np.log(Wtail + 1e-30), 1)[0]
    W_peak = W.max()
    if abs(lnslope_tail) < 0.5:  # <0.5/t_ci = flat-ish plateau
        W_sat = float(np.median(Wtail))
    else:
        W_sat = np.nan
        flags.append("no-plateau")
    ref = W_sat if np.isfinite(W_sat) else W_peak
    # growth-fit window
    lo, hi = thr, 0.25 * ref
    m = (W >= lo) & (W <= hi) & (t >= t_on)
    # keep only the contiguous stretch from onset
    idx = np.where(m)[0]
    if len(idx) >= 4:
        idx = idx[idx >= i_on]
        stop = np.where(np.diff(idx) > 3)[0]
        if len(stop):
            idx = idx[: stop[0] + 1]
    if len(idx) < 4:
        return floor, t_on / tci, np.nan, np.nan, W_sat, flags + ["short-fit"]
    x = t[idx] / tci * 2 * np.pi  # w_ci * t
    y = np.log(W[idx])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    gamma = 0.5 * coef[0]
    yfit = A @ coef
    err = 0.5 * np.sqrt(
        np.sum((y - yfit) ** 2) / max(len(x) - 2, 1) / np.sum((x - x.mean()) ** 2)
    )
    return floor, t_on / tci, gamma, err, W_sat, flags


def kspec(d, p, level=0, tsel=None):
    li = p["l_i"]
    z = d[f"z{level}"] / li
    b = d[f"bx{level}"] + 1j * d[f"by{level}"]
    dz = z[1] - z[0]
    it = tsel if tsel is not None else len(b) - 1
    bb = b[it] - b[it].mean()
    F = np.abs(np.fft.fft(bb)) ** 2
    k = np.fft.fftfreq(len(z), d=dz) * 2 * np.pi  # k l_i
    n = len(k) // 2
    return k[1:n], F[1:n]


def main():
    arms = ["mr", "ctl", "fine", "mr_eta7", "ctl_eta7"]
    data = {}
    for a in arms:
        try:
            data[a] = load(f"t110_{a}")
        except Exception as e:
            print(f"[t1_10] missing arm t110_{a}: {e}")
    # regions (l_i): away from seams at 96/160
    coarse_wins = [(16, 88), (168, 240)]
    seam_wins = [(92, 100), (156, 164)]
    patch_wins = [(104, 152)]

    res = {}
    print("=" * 78)
    print(
        f"{'arm':<10} {'region':<8} {'floor':>9} {'t_on':>6} {'gamma':>8} "
        f"{'+-':>6} {'W_sat':>9}  flags"
    )
    for a, (d, p) in data.items():
        tci = p["t_ci"]
        t = d["t"]
        rows = {}
        for reg, wins in [
            ("coarse", coarse_wins),
            ("seam", seam_wins),
            ("patch0", patch_wins),
        ]:
            W = region_W(d, p, wins, level=0)
            rows[reg] = (t, W) + onset_growth_sat(t, W, tci)
        if "bx1" in d.files:
            W = region_W(d, p, patch_wins, level=1)
            rows["patch1"] = (t, W) + onset_growth_sat(t, W, tci)
        res[a] = rows
        for reg, r in rows.items():
            _, _, floor, t_on, g, ge, Ws, flags = r
            print(
                f"{a:<10} {reg:<8} {floor:9.2e} {t_on:6.2f} {g:8.4f} "
                f"{ge:6.4f} {Ws if np.isfinite(Ws) else float('nan'):9.2e}  "
                f"{','.join(flags) if flags else '-'}"
            )
    print("=" * 78)

    # ---------------- verdicts ----------------
    if "mr" in res and "ctl" in res:
        _, _, _, ton_m, g_m, ge_m, Ws_m, _ = res["mr"]["coarse"]
        _, _, _, ton_c, g_c, ge_c, Ws_c, _ = res["ctl"]["coarse"]
        print("\nPre-registered criteria:")
        c1 = ton_m >= 0.8 * ton_c
        print(
            f"C1 onset: t_on(MR)={ton_m:.2f} vs 0.8*t_on(ctl)={0.8 * ton_c:.2f} t_ci "
            f"-> {'PASS' if c1 else 'FAIL'}"
        )
        c2 = abs(g_m / g_c - 1) <= 0.25 if np.isfinite(g_m * g_c) else False
        print(
            f"C2 growth: gamma_MR={g_m:.4f}+-{ge_m:.4f}, gamma_ctl={g_c:.4f}+-{ge_c:.4f} w_ci, "
            f"ratio={g_m / g_c:.3f} -> {'PASS' if c2 else 'FAIL'}"
        )
        if np.isfinite(Ws_m) and np.isfinite(Ws_c):
            r = Ws_m / Ws_c
            c3 = 0.5 <= r <= 2.0
            print(
                f"C3 saturation: W_sat MR/ctl = {r:.3f} -> {'PASS' if c3 else 'FAIL'}"
            )
        else:
            print(
                f"C3 saturation: W_sat MR={Ws_m}, ctl={Ws_c} (no plateau) -> UNRESOLVED"
            )
    if "mr_eta7" in res and "ctl_eta7" in res:
        _, Wm = res["mr_eta7"]["coarse"][:2]
        _, Wc = res["ctl_eta7"]["coarse"][:2]
        fm, tonm = res["mr_eta7"]["coarse"][2], res["mr_eta7"]["coarse"][3]
        fc, tonc = res["ctl_eta7"]["coarse"][2], res["ctl_eta7"]["coarse"][3]
        gm, gc = res["mr_eta7"]["coarse"][4], res["ctl_eta7"]["coarse"][4]
        (not np.isfinite(tonm)) and (not np.isfinite(tonc))
        print(
            f"C4 eta-floor: onset MR_eta7={tonm}, ctl_eta7={tonc}; "
            f"gamma {gm} / {gc}; max W/floor: MR {Wm.max() / fm:.1f}, ctl {Wc.max() / fc:.1f}"
        )

    # ---------------- figures ----------------
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    ax = axs[0, 0]
    for a in ["mr", "ctl", "fine"]:
        if a not in res:
            continue
        t, W = res[a]["coarse"][:2]
        tci = data[a][1]["t_ci"]
        ax.semilogy(t / tci, W, color=COL[a], lw=2, label=LBL[a])
    ax.set_xlabel(r"$t/t_{ci}$")
    ax.set_ylabel(r"$W_{\rm coarse}$")
    ax.set_title("coarse-region fluctuation energy (level 0, away from seams)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axs[0, 1]
    if "mr" in res:
        tci = data["mr"][1]["t_ci"]
        for reg, ls in [("coarse", "-"), ("seam", "--"), ("patch0", ":")]:
            t, W = res["mr"][reg][:2]
            ax.semilogy(t / tci, W, ls, color=COL["mr"], lw=2, label=f"MR {reg}")
        if "patch1" in res["mr"]:
            t, W = res["mr"]["patch1"][:2]
            ax.semilogy(
                t / tci, W, "-", color="#CC79A7", lw=2, label="MR patch (level 1)"
            )
    ax.set_xlabel(r"$t/t_{ci}$")
    ax.set_ylabel(r"$W$")
    ax.set_title("MR arm by region")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axs[1, 0]
    for a in ["fine"]:
        if a in res:
            t, W = res[a]["patch0"][:2]
            tci = data[a][1]["t_ci"]
            ax.semilogy(
                t / tci, W, color=COL["fine"], lw=2, label="uniform fine, patch window"
            )
    if "mr" in res and "patch1" in res["mr"]:
        t, W = res["mr"]["patch1"][:2]
        tci = data["mr"][1]["t_ci"]
        ax.semilogy(t / tci, W, color="#CC79A7", lw=2, label="MR patch (level 1)")
    ax.set_xlabel(r"$t/t_{ci}$")
    ax.set_ylabel(r"$W_{\rm patch}$")
    ax.set_title("fine-grid comparison in the patch window")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axs[1, 1]
    for a in ["mr_eta7", "ctl_eta7"]:
        if a in res:
            t, W = res[a]["coarse"][:2]
            tci = data[a][1]["t_ci"]
            ax.semilogy(t / tci, W, color=COL[a], lw=2, label=LBL[a])
    for a in ["mr", "ctl"]:  # bare references, faded
        if a in res:
            t, W = res[a]["coarse"][:2]
            tci = data[a][1]["t_ci"]
            ax.semilogy(
                t / tci, W, color=COL[a], lw=1, alpha=0.35, label=LBL[a] + " (bare)"
            )
    ax.set_xlabel(r"$t/t_{ci}$")
    ax.set_ylabel(r"$W_{\rm coarse}$")
    ax.set_title("eta = 1e-7 floor-control legs")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "t1_10_growth.png"), dpi=300)
    print("wrote t1_10_growth.png")

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for a in ["mr", "ctl", "fine"]:
        if a not in res:
            continue
        d, p = data[a]
        t = d["t"] / p["t_ci"]
        # mid-growth snapshot: half-way between onset and horizon
        ton = res[a]["coarse"][3]
        if np.isfinite(ton):
            imid = np.argmin(np.abs(t - min(ton + 5, t[-1] * 0.8)))
        else:
            imid = int(len(t) * 0.6)
        k, F = kspec(d, p, level=0, tsel=imid)
        axs[0].loglog(k, F, color=COL[a], lw=1.6, label=f"{LBL[a]} (t={t[imid]:.1f})")
        k, F = kspec(d, p, level=0, tsel=len(t) - 1)
        axs[1].loglog(k, F, color=COL[a], lw=1.6, label=f"{LBL[a]} (t={t[-1]:.1f})")
    for ax, ttl in zip(axs, ["growth phase", "end of horizon"]):
        ax.axvline(np.pi / 0.5, color="gray", lw=1, ls="--", alpha=0.7)
        ax.text(np.pi / 0.5, ax.get_ylim()[0], " coarse Nyq", fontsize=8, color="gray")
        ax.set_xlabel(r"$k\,l_i$")
        ax.set_ylabel(r"$|b_\perp(k)|^2$ (arb)")
        ax.set_title(f"level-0 perpendicular-B spectra, {ttl}")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "t1_10_spectra.png"), dpi=300)
    print("wrote t1_10_spectra.png")


if __name__ == "__main__":
    sys.exit(main())
