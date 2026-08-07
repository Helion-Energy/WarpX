#!/usr/bin/env python3
"""T1.3 retest analysis: stabilized MR arm rerun with
warpx.refine_plasma_init = 1 (t13_mr_stab_e7_rpi) against the SAVED
t13_ctl_stab_e7 / t13_fine_stab_e7 comparators and the saved
t13_mr_stab_e7 (before) arm.

Reuses the t1_34_analyze machinery (load_run, wave_projection,
t13_arm_metrics, band_R, window_mean). Appends the measured tables to the
"## T1.3 retest -- per-level injection (2026-08-07)" section of
T1_RESULTS.md (the prediction was written there BEFORE this script ran)
and refreshes t1_3_amp_phase.png with the new arm added.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from t1_34_analyze import (
    C_CTL,
    C_EXTRA,
    C_FINE,
    C_GRAY,
    C_MR,
    HERE,
    band_R,
    load_run,
    t13_arm_metrics,
    wave_projection,
    window_mean,
)

ARMS = dict(
    mr_new="t13_mr_stab_e7_rpi",
    mr_old="t13_mr_stab_e7",
    ctl="t13_ctl_stab_e7",
    fine="t13_fine_stab_e7",
)
ORDER = ("mr_new", "mr_old", "ctl", "fine")


def spectral(r, p):
    """Main-line centroid, backward/main ratio, main-line power (t1_34
    convention)."""
    nt = r["nt_valid"]
    a = wave_projection(r, nt)
    dt_s = r["t"][1] - r["t"][0]
    wt = np.hanning(nt)
    F = np.fft.fftshift(np.fft.fft(wt * a))
    om = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nt, dt_s)) / p["w_ci"]
    P = np.abs(F) ** 2
    mp = (om > 0.02) & (om < 0.09)
    mn = (om < -0.02) & (om > -0.09)
    i1 = np.arange(nt)[mp][np.argmax(P[mp])]
    i2 = np.arange(nt)[mn][np.argmax(P[mn])]
    sl = slice(i1 - 3, i1 + 4)
    cen = (om[sl] * P[sl]).sum() / P[sl].sum()
    return dict(cen=cen, P1=P[i1], back=np.sqrt(P[i2] / P[i1]))


def main_phase(r, nt, p):
    a = wave_projection(r, nt)
    dt_s = r["t"][1] - r["t"][0]
    F = np.fft.fft(a)
    om = 2 * np.pi * np.fft.fftfreq(nt, dt_s) / p["w_ci"]
    af = np.fft.ifft(np.where((om > 0.02) & (om < 0.09), F, 0))
    return np.unwrap(np.angle(af))


def main():
    runs = {k: load_run(v) for k, v in ARMS.items()}
    p = runs["mr_new"]["p"]
    om_th, T_tr, t_ci, w_ci = p["om_theory"], p["T_transit"], p["t_ci"], p["w_ci"]

    print("valid windows (outputs):", {k: r["nt_valid"] for k, r in runs.items()})
    print("failed markers:", {k: r["failed"] for k, r in runs.items()})

    arms = {k: t13_arm_metrics(r, om_th) for k, r in runs.items()}

    # ---- per-transit raw-projection table --------------------------------
    n_max = int(min(arms["mr_new"]["t"][-1], arms["ctl"]["t"][-1]) / T_tr)
    half = 0.05 * T_tr
    rows = []
    for n in range(1, n_max + 1):
        tn = n * T_tr
        row = dict(n=n, t_tci=tn / t_ci)
        for k in ORDER:
            row[f"amp_{k}"] = window_mean(arms[k]["t"], arms[k]["amp"], tn, half)
            row[f"res_{k}"] = window_mean(arms[k]["t"], arms[k]["res"], tn, half)
        row["ratio"] = row["amp_mr_new"] / row["amp_ctl"]
        row["dphi_new"] = row["res_mr_new"] - row["res_ctl"]
        row["dphi_old"] = row["res_mr_old"] - row["res_ctl"]
        rows.append(row)
        print(
            f"transit {n:2d}: |a| new/old/ctl/fine = "
            f"{row['amp_mr_new']:.3f}/{row['amp_mr_old']:.3f}/"
            f"{row['amp_ctl']:.3f}/{row['amp_fine']:.3f}  "
            f"dphi(new-ctl) {row['dphi_new']:+.4f}  "
            f"dphi(old-ctl) {row['dphi_old']:+.4f}"
        )

    # ---- frequency fits + spectral lines ---------------------------------
    print(f"omega fits (w_ci), theory {om_th / w_ci:.4f}:")
    for k in ORDER:
        print(f"  {k:6s}: {arms[k]['om_fit'] / w_ci:.5f}")
    spec = {k: spectral(runs[k], p) for k in ORDER}
    for k in ORDER:
        print(
            f"  {k:6s}: main centroid {spec[k]['cen']:.5f} w_ci, "
            f"backward/main {spec[k]['back']:.3f}, "
            f"main amp/ctl {np.sqrt(spec[k]['P1'] / spec['ctl']['P1']):.3f}"
        )

    # ---- band-passed main-line phase at the transit marks ----------------
    ntc = min(arms[k]["nt"] for k in ORDER)
    tph = runs["mr_new"]["t"][:ntc]
    ph = {k: main_phase(runs[k], ntc, p) for k in ORDER}
    dphi_main = []
    for n in range(1, n_max + 1):
        i = np.argmin(np.abs(tph - n * T_tr))
        dphi_main.append(
            (
                n,
                ph["mr_new"][i] - ph["ctl"][i],
                ph["mr_old"][i] - ph["ctl"][i],
                ph["fine"][i] - ph["ctl"][i],
            )
        )
        print(
            f"transit {n:2d}: band dphi new-ctl {dphi_main[-1][1]:+.3f}, "
            f"old-ctl {dphi_main[-1][2]:+.3f}, fine-ctl {dphi_main[-1][3]:+.3f}"
        )

    # ---- seam scattering (spectral band R, new arm vs saved ctl) ---------
    mr, ctl = runs["mr_new"], runs["ctl"]
    nt = min(mr["nt_valid"], ctl["nt_valid"])
    _, R_mr = band_R(mr, nt, mr["b0"], mr["b0"])
    _, R_ctl = band_R(ctl, nt, ctl["b0"], ctl["b0"])
    _, R_diff = band_R(mr, nt, mr["b0"][:nt] - ctl["b0"][:nt], mr["b0"])
    print(
        f"stab spectral R: R_MR(new)={R_mr:.2e} R_ctl={R_ctl:.2e} R_diff={R_diff:.2e}"
    )

    # ---- should-be-zero monitors + level-1 noise floor -------------------
    mon = {}
    for k in ORDER:
        r = runs[k]
        ntv = r["nt_valid"]
        for lev, key in (("0", "amax0"), ("1", "amax1")):
            if key not in r:
                continue
            a = r[key][:ntv]
            ez = a[:, 2].max() / max(a[:, :2].max(), 1e-30)
            bz = a[:, 3].max() / p["amp"]
            med = np.median(a[:, 3]) / p["B0"]
            mon[f"{k}_l{lev}"] = (ntv, ez, bz, med)
            print(
                f"monitor {k} lev {lev}: max|Ez|/max|Exy| {ez:.2f}, "
                f"max|Bz-B0|/A {bz:.1f}, median max|Bz-B0|/B0 {med:.4f}"
            )

    # ---- markdown --------------------------------------------------------
    md = ["\n### Measured (post-prediction)\n"]
    md.append(
        f"New arm: t1_runs/{ARMS['mr_new']} finished clean (.done, "
        f"{arms['mr_new']['nt']} outputs, no monitor cut). Per-transit "
        f"raw-projection table (saved ctl/fine comparators; dphi columns "
        f"are MR-minus-ctl phase residuals, new arm and the original "
        f"before arm):\n"
    )
    md.append(
        "| transit | t (t_ci) | A_MRnew/A | A_ctl/A | A_fine/A | "
        "A_MRnew/A_ctl | dphi new-ctl (rad) | dphi old-ctl (rad) |"
    )
    md.append("|---|---|---|---|---|---|---|---|")
    for r_ in rows:
        md.append(
            f"| {r_['n']} | {r_['t_tci']:.1f} | {r_['amp_mr_new']:.3f} | "
            f"{r_['amp_ctl']:.3f} | {r_['amp_fine']:.3f} | "
            f"{r_['ratio']:.4f} | {r_['dphi_new']:+.4f} | "
            f"{r_['dphi_old']:+.4f} |"
        )
    md.append(
        f"\nFitted omega/w_ci over the full window: MR-new "
        f"{arms['mr_new']['om_fit'] / w_ci:.4f}, MR-old "
        f"{arms['mr_old']['om_fit'] / w_ci:.4f}, ctl "
        f"{arms['ctl']['om_fit'] / w_ci:.4f}, fine "
        f"{arms['fine']['om_fit'] / w_ci:.4f} (cold theory "
        f"{om_th / w_ci:.4f}).\n"
    )
    md.append("Spectral decomposition of a(t), full horizon:\n")
    md.append(
        "| arm | main-line centroid (w_ci) | backward/main amplitude | "
        "main-line amplitude / ctl |"
    )
    md.append("|---|---|---|---|")
    for k in ORDER:
        md.append(
            f"| {k} | {spec[k]['cen']:.5f} | {spec[k]['back']:.3f} | "
            f"{np.sqrt(spec[k]['P1'] / spec['ctl']['P1']):.3f} |"
        )
    md.append(
        "\nBand-passed main-line phase difference at the transit marks "
        "(the beat-free metric the finding was stated on):\n"
    )
    md.append(
        "| transit | dphi MRnew-ctl (rad) | dphi MRold-ctl (rad) | "
        "dphi fine-ctl (rad) |"
    )
    md.append("|---|---|---|---|")
    for n, dn, do, df in dphi_main:
        md.append(f"| {n} | {dn:+.3f} | {do:+.3f} | {df:+.3f} |")
    md.append(
        f"\nSeam scattering (spectral band R, new arm vs saved ctl, "
        f"nt = {nt}): R_MR = {R_mr:.2e}, R_ctl floor = {R_ctl:.2e}, "
        f"R_diff = {R_diff:.2e}.\n"
    )
    md.append(
        "Should-be-zero monitors and level-1 noise floor (median of the "
        "per-output pointwise max|Bz - B0|/B0 over the horizon -- the "
        "quiet-floor statistic):\n"
    )
    md.append(
        "| run | level | outputs | max|Ez|/max|Exy| | max|Bz-B0|/A | "
        "median max|Bz-B0|/B0 |"
    )
    md.append("|---|---|---|---|---|---|")
    for k in ORDER:
        for lev in ("0", "1"):
            key = f"{k}_l{lev}"
            if key not in mon:
                continue
            ntv, ez, bz, med = mon[key]
            md.append(
                f"| {ARMS[k]} | {lev} | {ntv} | {ez:.2f} | {bz:.1f} | {med:.4f} |"
            )

    out = os.path.join(HERE, "T1_RESULTS.md")
    with open(out, "a") as f:
        f.write("\n".join(md) + "\n")
    print(f"appended measured tables to {out}")

    # ---- refreshed figure -------------------------------------------------
    g_eta = p["gamma_eta"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))
    for k, c, ls, lab in (
        ("mr_new", C_EXTRA, "-", "MR + refine_plasma_init (new)"),
        ("mr_old", C_MR, "-", "MR original (diluted init)"),
        ("ctl", C_CTL, "-", "control (uniform coarse)"),
        ("fine", C_FINE, "-", "uniform fine"),
    ):
        axs[0].plot(
            arms[k]["t"] / T_tr, arms[k]["amp"], color=c, ls=ls, lw=1.6, label=lab
        )
        axs[1].plot(
            arms[k]["t"] / T_tr, arms[k]["res"], color=c, ls=ls, lw=1.6, label=lab
        )
    tt = arms["ctl"]["t"]
    axs[0].plot(
        tt / T_tr,
        np.exp(-g_eta * tt),
        ls=":",
        color=C_GRAY,
        lw=1.5,
        label=r"$e^{-\gamma_\eta t}$ (resistive)",
    )
    axs[0].set_xlabel("patch transits  $t / T_{transit}$")
    axs[0].set_ylabel(r"$|a(t)| / A$")
    axs[0].set_title("T1.3 wave amplitude (global $-k$ projection)")
    axs[0].legend(frameon=False, fontsize=10)
    axs[1].set_xlabel("patch transits  $t / T_{transit}$")
    axs[1].set_ylabel(r"phase residual  $\arg a - \omega_R t$  (rad)")
    axs[1].set_title("T1.3 phase vs cold R-mode theory")
    axs[1].legend(frameon=False, fontsize=10)
    fig.savefig(os.path.join(HERE, "t1_3_amp_phase.png"), dpi=280)
    print("refreshed t1_3_amp_phase.png")


if __name__ == "__main__":
    main()
