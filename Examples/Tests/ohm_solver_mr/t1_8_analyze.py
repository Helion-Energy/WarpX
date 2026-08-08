#!/usr/bin/env python3
"""Analysis for the T1.8 late-time soak (see t1_7_8_results.md).

Reads t1_runs/t18_{mr,ctl}/t1_lineouts.npz (+ params.json, FieldEnergy
reduced diags), evaluates the pre-registered criteria S1-S4, prints
markdown tables, and writes figures:

  t1_8_seamnoise.png  N_seam(t) MR vs ctl + quarter means + monitors
  t1_8_energy.png     level-0 field-energy ratio + wave amplitude |a|(t)
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
HP_MODES = 4  # |m| <= HP_MODES removed (wave m=1 + harmonics)


def load(name):
    d = dict(np.load(os.path.join(RUNS, name, "t1_lineouts.npz")))
    with open(os.path.join(RUNS, name, "params.json")) as fh:
        p = json.load(fh)
    d["params"] = p
    d["tt"] = d["t"] / p["t_ci"]
    d["zli"] = d["z0"] / p["l_i"]
    return d


def last_step(name):
    with open(os.path.join(RUNS, name, "run.log"), "rb") as fh:
        txt = fh.read().decode(errors="replace")
    steps = re.findall(r"STEP (\d+) ends", txt)
    return int(steps[-1]) if steps else 0


def n_seam(d):
    """High-passed transverse-field rms over the seam bands, / B0."""
    b = d["bx0"].astype(np.float64) + 1j * d["by0"].astype(np.float64)
    bk = np.fft.fft(b, axis=1)
    nz = b.shape[1]
    m = np.fft.fftfreq(nz, d=1.0 / nz)  # integer mode numbers
    bk[:, np.abs(m) <= HP_MODES] = 0.0
    bhp = np.fft.ifft(bk, axis=1)
    bands = np.zeros(nz, dtype=bool)
    for s in SEAMS_LI:
        bands |= np.abs(d["zli"] - s) <= BAND_LI
    return np.sqrt(np.mean(np.abs(bhp[:, bands]) ** 2, axis=1)) / d["params"]["B0"]


def wave_amp(d):
    """|a|(t) of the m=1 wave projection on level 0."""
    b = d["bx0"].astype(np.float64) + 1j * d["by0"].astype(np.float64)
    k = 2.0 * np.pi * 1 / (d["params"]["Lz"])
    ph = np.exp(1j * k * d["z0"])
    return np.abs((b * ph[None, :]).mean(axis=1)) / d["params"]["amp"]


def wmean(t, y, w):
    sel = (t >= w[0]) & (t <= w[1])
    return float(y[sel].mean())


def wmax(t, y, w):
    sel = (t >= w[0]) & (t <= w[1])
    return float(y[sel].max())


def field_energy(name):
    """(t, total_lev0, B_lev0) from the FieldEnergy reduced diag.

    Header: [0]step [1]time(s) [2]total_lev0(J) [3]E_lev0(J) [4]B_lev0(J)
    (+ lev1 columns when MR). total_lev0 is B-dominated (B0 background).
    """
    path = os.path.join(RUNS, name, "diags", "reducedfiles", "field_energy.txt")
    data = np.loadtxt(path, skiprows=1)
    return data[:, 1], data[:, 2], data[:, 4]


def main():
    mr_name = sys.argv[1] if len(sys.argv) > 1 else "t18_mr"
    mr = load(mr_name)
    ct = load("t18_ctl")
    p = mr["params"]
    horizon = mr["tt"][-1]
    T_tr = p["T_transit_tci"]

    print(f"horizon: {horizon:.1f} t_ci = {horizon / T_tr:.1f} transits")

    # ---------------- S1 ------------------------------------------
    print("\n### S1 -- completion\n")
    print("| arm | last step | finite | max monitor lev0 (B0) | max lev1 |")
    print("|---|---|---|---|---|")
    s1 = True
    for name, d in ((mr_name, mr), ("t18_ctl", ct)):
        ls = last_step(name)
        fin = all(np.isfinite(d[k]).all() for k in ("bx0", "by0", "bz0"))
        m0 = d["amax0"][:, 3].max() / p["B0"]
        m1 = d["amax1"][:, 3].max() / p["B0"] if "amax1" in d else float("nan")
        ok = ls >= 60000 and fin and m0 < 0.4
        s1 &= ok
        print(f"| {name} | {ls} | {fin} | {m0:.3f} | {m1:.3f} |")
    print(f"\nS1: {'PASS' if s1 else 'FAIL'}")

    # ---------------- S2 / S3 --------------------------------------
    ns_mr = n_seam(mr)
    ns_ct = n_seam(ct)
    q = horizon / 4
    quarters = [(i * q, (i + 1) * q) for i in range(4)]
    print("\n### S2/S3 -- seam-band high-pass noise N_seam (rms/B0)\n")
    print("| arm | Q1 mean | Q2 mean | Q3 mean | Q4 mean | Q4 max |")
    print("|---|---|---|---|---|---|")
    for name, ns, d in ((mr_name, ns_mr, mr), ("t18_ctl", ns_ct, ct)):
        row = [wmean(d["tt"], ns, w) for w in quarters] + [
            wmax(d["tt"], ns, quarters[3])
        ]
        print(f"| {name} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
    r_q42 = wmean(mr["tt"], ns_mr, quarters[3]) / wmean(mr["tt"], ns_mr, quarters[1])
    r_max = wmax(mr["tt"], ns_mr, quarters[3]) / wmax(mr["tt"], ns_mr, quarters[1])
    s2 = (r_q42 <= 1.3) and (r_max <= 2.0)
    print(
        f"\nS2: MR Q4/Q2 mean = {r_q42:.3f} (<= 1.3), Q4/Q2 max = "
        f"{r_max:.3f} (<= 2.0) -> {'PASS' if s2 else 'FAIL'}"
    )
    half = (horizon / 2, horizon)
    r_ctl = wmean(mr["tt"], ns_mr, half) / wmean(ct["tt"], ns_ct, half)
    s3 = r_ctl <= 2.0
    print(f"S3: last-half MR/ctl = {r_ctl:.3f} (<= 2.0) -> {'PASS' if s3 else 'FAIL'}")

    # ---------------- S4 ------------------------------------------
    print("\n### S4 -- energy history + wave amplitude\n")
    t_mr, l0_mr, _ = field_energy(mr_name)
    t_ct, l0_ct, _ = field_energy("t18_ctl")
    n = min(len(t_mr), len(t_ct))
    rat = l0_mr[:n] / l0_ct[:n]
    dev = np.abs(rat - 1).max()
    a_mr = wave_amp(mr)
    a_ct = wave_amp(ct)
    w5 = (horizon - 5 * T_tr, horizon)
    amp_ratio = wmean(mr["tt"], a_mr, w5) / wmean(ct["tt"], a_ct, w5)
    s4 = dev <= 0.05 and 0.75 <= amp_ratio <= 1.25
    print(f"max |E_lev0(MR)/E_lev0(ctl) - 1| = {dev:.4f} (<= 0.05)")
    print(
        f"wave amplitude ratio MR/ctl, final 5 transits = {amp_ratio:.3f} "
        f"(in [0.75, 1.25])"
    )
    print(f"S4: {'PASS' if s4 else 'FAIL'}")

    # per-transit amplitude table (soak record)
    print("\n| transit | t (t_ci) | A_MR/A | A_ctl/A | ratio |")
    print("|---|---|---|---|---|")
    for k in range(4, int(horizon / T_tr) + 1, 4):
        tmark = k * T_tr
        wm = (tmark - 0.5, tmark + 0.5)
        am, ac = wmean(mr["tt"], a_mr, wm), wmean(ct["tt"], a_ct, wm)
        print(f"| {k} | {tmark:.0f} | {am:.3f} | {ac:.3f} | {am / ac:.3f} |")

    # ---------------- figures --------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax = axs[0, 0]
    ax.plot(mr["tt"], ns_mr, "C0", lw=0.7, label="MR")
    ax.plot(ct["tt"], ns_ct, "C1", lw=0.7, alpha=0.8, label="ctl same-z")
    for i, w in enumerate(quarters):
        v = wmean(mr["tt"], ns_mr, w)
        ax.hlines(v, w[0], w[1], color="k", lw=2)
        ax.text(0.5 * (w[0] + w[1]), v * 1.15, f"Q{i + 1}", ha="center")
    ax.set_yscale("log")
    ax.set_xlabel("t (t$_{ci}$)")
    ax.set_ylabel("N$_{seam}$ (rms/B$_0$)")
    ax.set_title("seam-band high-pass noise (|m| > 4), level 0")
    ax.legend()
    ax = axs[0, 1]
    ax.plot(mr["tt"], mr["amax0"][:, 3] / p["B0"], "C0", lw=0.7, label="MR lev0")
    if "amax1" in mr:
        ax.plot(mr["tt"], mr["amax1"][:, 3] / p["B0"], "C3", lw=0.7, label="MR lev1")
    ax.plot(ct["tt"], ct["amax0"][:, 3] / p["B0"], "C1", lw=0.7, alpha=0.8, label="ctl")
    ax.axhline(0.4, color="r", ls="--", lw=1, label="0.4 B$_0$ cut")
    ax.set_xlabel("t (t$_{ci}$)")
    ax.set_ylabel("max|B$_z$ - B$_0$| / B$_0$")
    ax.set_title("instability monitor")
    ax.legend()
    ax = axs[1, 0]
    ax.plot(t_mr[:n] / p["t_ci"], rat, "C0", lw=0.8)
    ax.axhline(1.05, color="r", ls="--", lw=1)
    ax.axhline(0.95, color="r", ls="--", lw=1)
    ax.set_xlabel("t (t$_{ci}$)")
    ax.set_ylabel("E$_{lev0}$(MR) / E$_{lev0}$(ctl)")
    ax.set_title("level-0 field-energy ratio")
    ax = axs[1, 1]
    ax.plot(mr["tt"], a_mr, "C0", lw=0.8, label="MR")
    ax.plot(ct["tt"], a_ct, "C1", lw=0.8, alpha=0.8, label="ctl")
    ax.set_xlabel("t (t$_{ci}$)")
    ax.set_ylabel("|a|(t) / A")
    ax.set_title("m = 1 wave amplitude (resistive decay is common mode)")
    ax.legend()
    fig.suptitle(
        f"T1.8 late-time soak: 60000 steps = {horizon:.0f} t$_{{ci}}$ = "
        f"{horizon / T_tr:.1f} patch transits (eta = 1e-7, rpi = 1)"
    )
    fig.savefig(os.path.join(HERE, "t1_8_seamnoise.png"), dpi=280)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    b = mr["bx0"].astype(np.float64) + 1j * mr["by0"].astype(np.float64)
    bk = np.fft.fft(b, axis=1)
    nz = b.shape[1]
    m = np.fft.fftfreq(nz, d=1.0 / nz)
    bk[:, np.abs(m) <= HP_MODES] = 0.0
    bhp = np.abs(np.fft.ifft(bk, axis=1)) / p["B0"]
    im = ax.imshow(
        bhp.T,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, mr["tt"][-1], mr["zli"][0], mr["zli"][-1]],
        vmax=0.1,
    )
    for s in SEAMS_LI:
        ax.axhline(s, color="w", ls="--", lw=0.8)
    ax.set_xlabel("t (t$_{ci}$)")
    ax.set_ylabel("z (l$_i$)")
    ax.set_title(
        "T1.8 MR arm: |b$_{hp}$|/B$_0$ (high-passed transverse field, level 0)"
    )
    fig.colorbar(im, ax=ax)
    fig.savefig(os.path.join(HERE, "t1_8_spacetime.png"), dpi=280)
    print("\nfigures: t1_8_seamnoise.png t1_8_spacetime.png")


if __name__ == "__main__":
    sys.exit(main())
