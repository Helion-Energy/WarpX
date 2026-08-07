#!/usr/bin/env python3
"""T1b analysis: graded seam-dissipation-band experiments for hybrid-PIC MR.

Consumes t1b_runs/<case>/ (field_energy.txt + t1_lineouts.npz produced by
t1b_run_battery.py) and writes

  t1b_kill.png       E1 level-1 magnetic-energy growth, all arms
  t1b_packet.png     E2 T1.2 packet budget with the band vs T1 references
  T1B_RESULTS.md     all tables

Growth-rate convention: gamma is the AMPLITUDE growth rate, fitted as
0.5 * d ln(E_B1 - E_B1(0)) / dt from the FieldEnergy reduced diagnostic
(level-1 B energy above the initial value), over the exponential window
selected automatically between noise-floor exit and blowup. A second,
independent estimate fits the level-1 seam-band spectral amplitude
(|k| in [1.02, 1.9] k_Nc of the x-averaged perpendicular field) from the
lineout npz.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "t1b_runs")
sys.path.insert(0, HERE)
import t1_analyze  # noqa: E402
import t1_whistler  # noqa: E402

t1_analyze.RUNS = RUNS  # point the T1 loaders at the t1b runs

C_MR = "#3d65f5"
C_CTL = "#c95d20"
C_FINE = "#2e9e73"
C_EXTRA = "#8d69cf"
C_GRAY = "#6e6e6e"

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "figure.constrained_layout.use": True,
    }
)

# T1 reference numbers (measured on the hybrid_mr_p1 line, T1_RESULTS.md)
T1_RDIFF = {0.3: 5.10e-06, 0.6: 1.40e-06, 0.8: 2.05e-06}
T1_RCTL = {0.3: 3.05e-03, 0.6: 3.14e-04, 0.8: 1.87e-03}
T1_PACKET_EH020 = dict(E_in=0.065, E_out=0.0000, E_abs=0.935, Ebm=0.002, Ebp=0.0030)
T1_PACKET_BARE = dict(E_in=0.379, E_out=0.0000, E_abs=0.621, Ebm=0.019, Ebp=0.0392)
T1_GAMMA_BARE = 0.09  # w_ci, dt = 0.012 t_ci (T1 dbg convention)


def load_field_energy(name):
    """Return (t, E_B_lev1) from the FieldEnergy reduced diagnostic."""
    fe = os.path.join(RUNS, name, "diags/reducedfiles/field_energy.txt")
    a = np.loadtxt(fe, skiprows=1)
    if a.ndim == 1:
        a = a[None, :]
    # columns: step, time, total0, E0, B0, total1, E1, B1
    return a[:, 1], a[:, 7]


def fit_gamma_energy(name, p):
    """Amplitude growth rate from the level-1 B-energy excess.

    Returns (gamma/w_ci, window (t0, t1) in t_ci, dE range used, ok flag).
    """
    t, eb = load_field_energy(name)
    dE = eb - eb[0]
    w_ci, t_ci = p["w_ci"], p["t_ci"]
    good = np.isfinite(dE)
    t, dE = t[good], dE[good]
    # noise floor from the first ~1.5 t_ci (seed-wave transient included:
    # use the running max afterwards to bound it)
    n0 = max(3, np.searchsorted(t, 1.5 * t_ci))
    floor = np.percentile(np.abs(dE[:n0]), 90)
    lo = 10.0 * floor
    grow = np.nonzero(dE > lo)[0]
    grow = grow[grow >= n0]
    if len(grow) < 5:
        return 0.0, (np.nan, np.nan), (floor, np.nan), False
    i0 = grow[0]
    i1 = len(t) - 1
    # end the fit ~0.5 t_ci before the last finite sample (blowup ramp)
    t_end = t[i1] - 0.5 * t_ci
    i1 = np.searchsorted(t, t_end)
    if i1 - i0 < 5:
        i1 = min(i0 + 5, len(t) - 1)
    sl = slice(i0, i1 + 1)
    y = np.log(np.maximum(dE[sl], 1e-300))
    A = np.vstack([t[sl], np.ones(i1 + 1 - i0)]).T
    slope = np.linalg.lstsq(A, y, rcond=None)[0][0]
    return (0.5 * slope / w_ci, (t[i0] / t_ci, t[i1] / t_ci), (dE[i0], dE[i1]), True)


def fit_gamma_seam_band(r):
    """Amplitude growth rate of the level-1 above-Nyquist band power.

    Uses the full finite window (not nt_valid): at dt = 0.012 the seeded
    wave's compressional response crosses the 0.4 B0 monitor long before
    the instability dominates, so nt_valid is far too conservative here.
    """
    p = r["p"]
    if "b1" not in r:
        return np.nan
    finite = np.isfinite(r["b1"]).all(axis=1)
    nt = int(np.nonzero(~finite)[0][0]) if (~finite).any() else len(r["t"])
    b = r["b1"][:nt]
    dz = r["z1"][1] - r["z1"][0]
    w = np.hanning(b.shape[1])
    F = np.fft.fft(w * b, axis=1)
    kb = np.abs(np.fft.fftfreq(b.shape[1], dz) * 2 * np.pi / p["k_nyq_coarse"])
    band = (kb > 1.02) & (kb < 1.9)
    A = np.sqrt((np.abs(F[:, band]) ** 2).sum(axis=1))
    t = r["t"][:nt]
    # fit from 10x the early floor to the end of the valid window
    n0 = max(3, np.searchsorted(t, 1.0 * p["t_ci"]))
    floor = np.median(A[:n0])
    grow = np.nonzero(A > 10 * floor)[0]
    grow = grow[grow >= n0]
    if len(grow) < 5:
        return 0.0
    sl = slice(grow[0], nt)
    y = np.log(A[sl])
    M = np.vstack([t[sl], np.ones(len(y))]).T
    slope = np.linalg.lstsq(M, y, rcond=None)[0][0]
    return slope / p["w_ci"]


# ----------------------------------------------------------------------
# E1: instability kill test
# ----------------------------------------------------------------------
E1_ARMS = [
    ("e1_bare", "bare seam (baseline)"),
    ("e1_ehband_w6", "eta_h band, W=6 (coarse-matched value)"),
    ("e1_etaband_w6", "eta band, W=6 (3e-8 Ohm m)"),
    ("e1_both_w6", "both bands, W=6"),
    ("e1_ehband_w3", "eta_h band, W=3"),
    ("e1_etaband_w3", "eta band, W=3"),
    ("e1_both_w3", "both bands, W=3"),
    ("e1_ehband_cm_w6", "Nyquist-scaled bulk + coarse-matched band, W=6"),
    ("e1_etaband_w24", "eta band, W=24 (3e-8 Ohm m)"),
    ("e1_etastrong_w6", "eta band, W=6, 1e-7 Ohm m"),
    ("e1_ehstrong_w6", "eta_h band, W=6, 1.0 eta_h*"),
    ("e1_globaleta_ctl", "global eta 3e-8 (both levels, no band)"),
]


def kill_table(results_md, arms):
    """Append one blowup/gamma table row per finished arm; return curves."""
    curves = []
    for name, label in arms:
        run_dir = os.path.join(RUNS, name)
        if not os.path.isdir(run_dir):
            continue
        try:
            r = t1_analyze.load_run(name)
        except FileNotFoundError:
            print(f"{name}: not converted yet, skipping")
            continue
        p = r["p"]
        gE, win, rng, okfit = fit_gamma_energy(name, p)
        gk = fit_gamma_seam_band(r)
        nan_run = os.path.exists(os.path.join(run_dir, ".failed"))
        # crash step = where the plotfile stream stops (the 0.4 B0 monitor
        # is seed-contaminated at dt = 0.012 and not used here)
        blow = len(r["t"]) if nan_run else -1
        horizon = p["steps"] * p["dt"] / p["t_ci"]
        t_blow = blow * p["dt"] / p["t_ci"] if blow > 0 else np.nan
        if nan_run:
            verdict = "UNSTABLE"
        elif np.isfinite(gk) and abs(gk) < 0.005:
            # gamma_k (the seam-band spectral amplitude) is the instability
            # discriminator; gamma_E can pick up slow secular level-1
            # energy drift (resistive heating / PIC noise) in stable runs
            verdict = f"STABLE >= {horizon:.0f} t_ci (seam band power flat)"
        else:
            verdict = f"growing (slow), no blowup in {horizon:.0f} t_ci"
        results_md.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                label,
                blow if blow > 0 else "-",
                f"{t_blow:.1f}" if blow > 0 else "-",
                f"{gE:.3f}" if okfit else "< noise",
                f"{gk:.3f}" if np.isfinite(gk) else "-",
                verdict,
            )
        )
        print(
            f"{name:22s} blow={blow:5d} gamma_E={gE:7.3f} "
            f"gamma_k={gk:7.3f}  {verdict}  fitwin={win}"
        )
        t, eb = load_field_energy(name)
        curves.append((label, t / p["t_ci"], np.maximum(eb - eb[0], 1e-12), name))
    return curves


def e1_analysis(results_md):
    print("\n=== E1 instability kill test (dt = 0.012 t_ci, kfac 0.1) ===")
    results_md.append("\n## E1 -- instability kill test\n")
    results_md.append(
        "T1 unstable configuration (dt = 0.012 t_ci, kfac 0.1 seed, bulk "
        "eta = 1e-10 Ohm m, bulk eta_h = 0 unless noted), horizon 20 t_ci "
        "(1667 steps). gamma_E: amplitude rate from the level-1 B-energy "
        "excess; gamma_k: amplitude rate of the level-1 |k| in [1.02, 1.9] "
        "k_Nc band. T1 bare-seam reference: gamma = 0.09 w_ci, blowup at "
        "step ~419. Blowup manifests as the RK4 NaN guard (rc 6) or a "
        "runaway-particle deposition segfault (rc 11); the eta band W=6 "
        "arm instead entered a runaway-COMPUTE stall at step 519 (all "
        "ranks spinning >8 min on one step, killed manually) -- the same "
        "terminal event with a third face, counted here as its blowup "
        "step.\n"
    )
    results_md.append(
        "| arm | blowup step | t_blow (t_ci) | gamma_E/w_ci | gamma_k/w_ci | verdict |"
    )
    results_md.append("|---|---|---|---|---|---|")
    curves = kill_table(results_md, E1_ARMS)
    # figure
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    palette = [
        C_MR,
        C_FINE,
        C_CTL,
        C_EXTRA,
        C_FINE,
        C_CTL,
        C_EXTRA,
        C_GRAY,
        "#b03060",
        "#2f7f7f",
        "#7f7f2f",
        "#111111",
    ]
    lss = ["-", "-", "-", "-", "--", "--", "--", "-", "-", "--", ":", "-"]
    for (label, tt, dE, name), c, ls in zip(curves, palette, lss):
        m = np.isfinite(dE)
        lw = 3 if name in ("e1_bare", "e1_globaleta_ctl") else 1.8
        ax.semilogy(tt[m], dE[m], color=c, ls=ls, lw=lw, label=label)
    ax.set_xlabel(r"$t / t_{ci}$")
    ax.set_ylabel(r"level-1 $\int B^2/2\mu_0$ above initial (J)")
    ax.set_title(
        "E1 seam-instability kill test: level-1 magnetic energy "
        "(dt = 0.012 $t_{ci}$, kfac 0.1 seed)"
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.savefig(os.path.join(HERE, "t1b_kill.png"), dpi=280)
    return curves


# ----------------------------------------------------------------------
# E4/E5/E6: coarse-side band, setback A/B, cadence A/B
# ----------------------------------------------------------------------
E456_ARMS = [
    ("e1_bare", "bare seam (baseline, setback 2, substep cadence)"),
    ("e4_cband_w4", "COARSE band, eta 3e-8, W_in=W_out=4"),
    ("e4_cband_w2", "COARSE band, eta 3e-8, W_in=W_out=2"),
    ("e4_cband_w4_eta1e8", "COARSE band, eta 1e-8, W_in=W_out=4"),
    ("e4_cband_w16", "COARSE band, eta 3e-8, W_in=W_out=16"),
    ("e4_bothsides", "fine band W=6 + COARSE band W=4, eta 3e-8 both"),
    ("e5_setback0", "bare, mr_restrict_setback = 0"),
    ("e5_setback4", "bare, mr_restrict_setback = 4"),
    ("e6_halfstep", "bare, mr_restrict_cadence = half_step"),
    ("e1_globaleta_ctl", "global eta 3e-8 (reference)"),
]


def e456_analysis(results_md):
    print("\n=== E4/E5/E6 coarse-side band + operator A/Bs ===")
    results_md.append("\n## E4/E5/E6 -- coarse-side band and coupling-operator A/Bs\n")
    results_md.append(
        "Same unstable configuration as E1. E4: the coarse-side "
        "counterpart band (hybrid_pic_model.mr_coarse_seam_band_width / "
        "mr_coarse_seam_eta) -- a half-cosine ramp on the COARSE level, "
        "1 on the cells adjacent to the fine-patch edge on BOTH sides "
        "(covering the seam-adjacent exterior ring and the interior "
        "restricted/sacrificial mixed-face ring), 0 at W cells away per "
        "side. E5: restriction-setback A/B (bare seam; setback 0 removes "
        "the interior mixed-face ring entirely, restriction reaches the "
        "patch edge). E6: restriction cadence A/B (bare seam, restrict "
        "once per half-step instead of every accepted substep).\n"
    )
    results_md.append(
        "| arm | blowup step | t_blow (t_ci) | gamma_E/w_ci | gamma_k/w_ci | verdict |"
    )
    results_md.append("|---|---|---|---|---|---|")
    curves = kill_table(results_md, E456_ARMS)
    # figure
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    palette = [
        C_MR,
        C_FINE,
        C_CTL,
        C_EXTRA,
        "#b03060",
        "#d02090",
        "#2f7f7f",
        "#7f7f2f",
        "#404040",
        "#111111",
    ]
    lss = ["-", "-", "--", ":", "-", "--", "-", "--", ":", "-"]
    for (label, tt, dE, name), c, ls in zip(curves, palette, lss):
        m = np.isfinite(dE)
        lw = 3 if name in ("e1_bare", "e1_globaleta_ctl", "e4_cband_w4") else 1.8
        ax.semilogy(tt[m], dE[m], color=c, ls=ls, lw=lw, label=label)
    ax.set_xlabel(r"$t / t_{ci}$")
    ax.set_ylabel(r"level-1 $\int B^2/2\mu_0$ above initial (J)")
    ax.set_title(
        "E4/E5/E6 coarse-side band and operator A/Bs "
        "(dt = 0.012 $t_{ci}$, kfac 0.1 seed)"
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.savefig(os.path.join(HERE, "t1b_coarse_kill.png"), dpi=280)


# ----------------------------------------------------------------------
# E2: reflection regression
# ----------------------------------------------------------------------
def e2_t11(results_md):
    print("\n=== E2 T1.1 reflection regression (band on) ===")
    results_md.append("\n## E2 -- reflection regression with the band on\n")
    results_md.append(
        "T1.1 protocol (dt = 0.006 t_ci, 1000 steps, identical-seed "
        "max_level=0 control): R_diff = -k power of (b_MR - b_ctl) in the "
        "coarse window, i.e. everything the patch (now including the band) "
        "causes. T1 reference columns are the bare-seam battery values.\n"
    )
    results_md.append(
        "| k/k_Nc | w_MR/w_ci | w_ctl/w_ci | R_MR | R_ctl (floor) | R_diff "
        "(band) | R_diff (T1 bare) | note |"
    )
    results_md.append("|---|---|---|---|---|---|---|---|")
    for kf in (0.3, 0.6, 0.8):
        tag = f"t11b_k{int(round(kf * 1000)):04d}"
        try:
            mr = t1_analyze.load_run(tag + "_mr")
            ctl = t1_analyze.load_run(tag + "_ctl")
        except FileNotFoundError:
            continue
        nt = min(mr["nt_valid"], ctl["nt_valid"])
        om_mr, R_mr, _, dom = t1_analyze.whistler_R_and_omega(mr, nt)
        om_ct, R_ct, _, _ = t1_analyze.whistler_R_and_omega(ctl, nt)
        _, R_diff, _, _ = t1_analyze.whistler_R_and_omega(
            mr, nt, b_diff=mr["b0"][:nt] - ctl["b0"][:nt]
        )
        p = mr["p"]
        blow = mr["blow_step"]
        note = f"seam>0.4B0 @ {blow}" if blow > 0 else "no seam noise"
        results_md.append(
            f"| {p['kfac']:.3f} | {abs(om_mr) / p['w_ci']:.2f} | "
            f"{abs(om_ct) / p['w_ci']:.2f} | {R_mr:.2e} | {R_ct:.2e} | "
            f"{R_diff:.2e} | {T1_RDIFF[kf]:.2e} | {note} |"
        )
        print(
            f"kfac {kf}: w_MR {abs(om_mr) / p['w_ci']:.2f} "
            f"w_ctl {abs(om_ct) / p['w_ci']:.2f} R_MR {R_mr:.2e} "
            f"R_ctl {R_ct:.2e} R_diff {R_diff:.2e} "
            f"(T1 bare {T1_RDIFF[kf]:.2e})  {note}"
        )


def e2_t12(results_md):
    print("\n=== E2 T1.2 packet with the band ===")
    try:
        r = t1_analyze.load_run_pair("t12b_pk127_band", "t12b_null_band")
    except FileNotFoundError:
        return
    bud = t1_analyze.packet_budget(r)
    p = r["p"]
    vg = t1_analyze.group_velocity(p)
    t_seam = (0.5 * (p["patch_zhi"] - p["patch_zlo"])) / vg
    i_s = min(np.searchsorted(bud["t"], t_seam), len(bud["t"]) - 1)
    note = str(r["blow_step"]) if r["blow_step"] > 0 else "never"
    results_md.append(
        "\nT1.2 packet (k = 1.27 k_Nc) with the eta_h band, bulk eta_h = 0, "
        f"budget at t = {t_seam / p['t_ci']:.1f} t_ci:\n"
    )
    results_md.append(
        "| arm | in patch | outside | absorbed | band -k left | band +k "
        "(reflected) | seam noise > 0.4 B0 @ step |"
    )
    results_md.append("|---|---|---|---|---|---|---|")
    results_md.append(
        "| seam band (W=6, eta_h -> 0.2 eta_h*) | "
        f"{bud['E_in'][i_s]:.3f} | {bud['E_out'][i_s]:.4f} | "
        f"{bud['E_abs'][i_s]:.3f} | {bud['Ebm'][i_s]:.3f} | "
        f"{bud['Ebp'][i_s]:.4f} | {note} |"
    )
    results_md.append(
        "| T1 bulk eta_h = 0.2 eta_h* (reference) | "
        f"{T1_PACKET_EH020['E_in']:.3f} | {T1_PACKET_EH020['E_out']:.4f} | "
        f"{T1_PACKET_EH020['E_abs']:.3f} | {T1_PACKET_EH020['Ebm']:.3f} | "
        f"{T1_PACKET_EH020['Ebp']:.4f} | never |"
    )
    results_md.append(
        "| T1 bare seam (reference) | "
        f"{T1_PACKET_BARE['E_in']:.3f} | {T1_PACKET_BARE['E_out']:.4f} | "
        f"{T1_PACKET_BARE['E_abs']:.3f} | {T1_PACKET_BARE['Ebm']:.3f} | "
        f"{T1_PACKET_BARE['Ebp']:.4f} | 704 |"
    )
    print(
        f"band: in {bud['E_in'][i_s]:.3f} out {bud['E_out'][i_s]:.4f} "
        f"abs {bud['E_abs'][i_s]:.3f} band-k {bud['Ebm'][i_s]:.3f} "
        f"band+k {bud['Ebp'][i_s]:.4f} (T1 eh020: "
        f"abs {T1_PACKET_EH020['E_abs']}, +k {T1_PACKET_EH020['Ebp']})"
    )
    results_md.append(
        "\nThe band cannot substitute for the bulk absorber on this "
        "budget, for a simple kinematic reason: the bulk absorber acts on "
        "the packet during its whole dwell in the patch (gamma_h(1.27 "
        "k_Nc) = 0.52 w_ci over the 1.4 t_ci = 8.8 w_ci^-1 approach, "
        "damping it essentially completely before it reaches the seam), "
        "while the seam band acts only during the ~0.2 w_ci^-1 transit of "
        "the W = 6 edge cells (one-pass energy attenuation ~10%). The "
        "measured budget with the band is bare-seam-like, and the "
        "reflected-band content is unchanged to within the measurement -- "
        "consistent with the +k content being generated in the seam "
        "coupling itself rather than accumulated along the band path."
    )

    # budget-vs-time figure
    fig, ax = plt.subplots(figsize=(10, 6.5))
    tm = bud["t"] / p["t_ci"]
    ax.plot(tm, bud["E_in"], color=C_MR, lw=2, label="in patch")
    ax.plot(tm, bud["E_out"], color=C_FINE, lw=2, label="outside patch")
    ax.plot(tm, bud["E_abs"], color=C_CTL, lw=2, label="absorbed")
    ax.plot(tm, bud["Ebp"], color=C_EXTRA, lw=2, label="band +k (reflected)")
    ax.axhline(
        T1_PACKET_EH020["E_abs"],
        color=C_CTL,
        ls=":",
        lw=1.2,
        label="T1 bulk eta_h=0.2 absorbed @t_seam",
    )
    ax.axvline(t_seam / p["t_ci"], color=C_GRAY, lw=1, ls="--")
    ax.set_xlim(0, 2.5)
    ax.set_xlabel(r"$t / t_{ci}$")
    ax.set_ylabel("fraction of initial packet energy")
    ax.set_title("E2 T1.2 packet budget with the seam eta_h band (bulk eta_h = 0)")
    ax.legend(frameon=False, fontsize=10)
    fig.savefig(os.path.join(HERE, "t1b_packet.png"), dpi=280)


# ----------------------------------------------------------------------
def matched_damping_note(results_md):
    ehs = t1_whistler.ETA_H_STAR
    seam = 0.2 * ehs
    g15 = 0.2 * 1.5**4
    results_md.append("\n## Matched-damping interpretation -- and its failure\n")
    results_md.append(
        f"The band was designed under the matched-damping hypothesis: with "
        f"per-level Nyquist-scaled hyper-resistivity (eta_h(lev) "
        f"proportional to k_Nc(lev)^-4) the coarse side damps a fixed "
        f"physical k 16x (ratio^4) harder than the fine side, and the "
        f"fine-only band k in (k_Nc, 2 k_Nc) is nearly undamped on the "
        f"fine side while having no coarse counterpart to carry it away; "
        f"a coarse-matched seam band makes gamma_h(k) = eta_h k^4 / mu0 "
        f"continuous across the seam for shared k and damps the "
        f"un-crossable band at coarse strength before it reaches the seam. "
        f"Quantitatively the band is strongly supercritical in that "
        f"picture: with the coarse-matched target eta_h = 0.2 eta_h* = "
        f"{seam:.3e} Ohm m^3 (eta_h* = mu0 w_ci / k_Nc^4 = {ehs:.3e}), "
        f"gamma_h(1.5 k_Nc) = 0.2 * 1.5^4 w_ci = {g15:.2f} w_ci, about "
        f"{g15 / 0.10:.0f}x the measured bare-seam growth rate gamma_inst "
        f"~ 0.10 w_ci at dt = 0.012 t_ci. The E1 kill test falsifies the "
        f"hypothesis: an instability mode whose e-folding is ~10 w_ci^-1 "
        f"slower than the band's local damping rate is barely slowed "
        f"(blowup 419 -> ~486-520 steps), so the unstable circuit does "
        f"not pass its gain through the fine-side band content that "
        f"eta/eta_h can reach."
    )


def e3_note(results_md):
    results_md.append("\n## E3 -- bulk-physics null (quiet-drift smoke)\n")
    results_md.append(
        "t_smoke_inputs_2d_uniform (500 steps, production-like dt), band "
        "off vs band on (W = 6, mr_seam_eta = 3e-7 Ohm m = 3x this deck's "
        "bulk eta, mr_seam_eta_h = 0.2 eta_h* of this deck's coarse grid "
        "= 3.02e-15 Ohm m^3), identical seeds; driver t1b_smoke_null.py, "
        "figure t1b_null_sigma.png. sigma(B)/B0 on the level-0 view stays "
        "within 0.1% of band-off at every output (final 1.4450e-01 vs "
        "1.4439e-01, ratio 0.999); the level-1 patch-interior sigma(B)/B0 "
        "(10 edge cells excluded) within 0.5% (final 2.4900e-01 vs "
        "2.4785e-01, ratio 0.995, the band-on value slightly LOWER); "
        "total particle-energy drift +6.73e-3 (off) vs +6.84e-3 (on) of "
        "the initial energy. The band is seam-local: the bulk does not "
        "feel it beyond the identical-seed decorrelation level."
    )


def mechanism_note(results_md):
    results_md.append("\n## Mechanism: what the kill pattern says\n")
    results_md.append(
        "Round 1 (fine-side bands): every fine-side dissipation "
        "configuration -- eta_h band at the coarse-matched value and at "
        "5x that value, eta band at the global stabilization threshold "
        "(3e-8 Ohm m) and at 3x that value, both together, widths W = 3, "
        "6, and 24 (24 fine cells covers the deposition-buffer band, the "
        "restriction setback and the ghost-fill reach combined) -- only "
        "shifts the blowup step by -6% to +66% (the steep W = 3 eta_h "
        "ramp is slightly WORSE than bare). Round 2 (coarse side and "
        "operators): the coarse-side counterpart band FAILS TOO -- eta "
        "3e-8 over +-4 coarse cells of the seam ring delays blowup only "
        "to step 549 (bare 419), +-16 cells (+-8 l_i) to 566, and even "
        "fine W=6 + coarse +-4 SIMULTANEOUSLY (dissipation of globally-"
        "stabilizing strength on BOTH sides of both seams) only reaches "
        "645 -- while global eta 3e-8 is stable past 1667 steps "
        "(20 t_ci). The coarse-side-gain prediction is falsified along "
        "with the fine-side one: NO localized dissipation opens the "
        "loop.\n"
    )
    results_md.append(
        "Operator A/Bs at the same configuration: mr_restrict_setback 0 "
        "(no interior mixed restricted/free face ring; restriction "
        "reaches the patch edge) is the strongest single structural "
        "lever, delaying blowup to 723 (1.7x) with gamma_E dropping "
        "~2x -- the mixed-face ring feeds the loop but is not its sole "
        "gain stage. Setback 4 lands at 534 (non-monotonic in setback), "
        "and half-step restriction cadence at 480 (weak coupling-"
        "frequency dependence at this dt).\n"
    )
    results_md.append(
        "Verdict: no minimal localized stabilizer exists in the "
        "dissipation family -- local eta of globally-stabilizing "
        "strength, applied on the fine side, the coarse side, or both "
        "at once, and up to +-8 l_i wide, only rescales the onset time. "
        "The per-step gain is therefore not a compact seam circuit that "
        "resistive damping can sever; it is the once-per-step coupling "
        "SEQUENCE itself (moment overwrite + B ghost fill + masked "
        "restriction), whose mutual inconsistency re-injects near-and-"
        "above-Nyquist power each step, fed by the domain-wide "
        "fluctuation reservoir that only global damping suppresses. For "
        "the structural fix this points at consistency, not "
        "dissipation: the setback-0 result implicates the restriction "
        "boundary's EMF mismatch as the largest single contributor, so "
        "EMF matching (advancing the coarse faces under the patch with "
        "the restricted fine EMF so restricted and freely-evolved faces "
        "share circulations) is the first structural fix to try -- but "
        "since even setback 0 remains unstable, it must be paired with "
        "a consistent ghost-fill/moment path; no single-operator fix is "
        "indicated to suffice. Practically, quiet-plasma hybrid MR "
        "keeps the T1 prescription: a global resistive floor "
        "eta >= 3e-8 Ohm m at dt = 0.012 t_ci (~1.5e-9 s), i.e. "
        "gamma_eta(k_Nc) of a few x gamma_inst, scaled with dt.\n"
    )


def main():
    results_md = ["# T1b graded seam-dissipation band -- results\n"]
    results_md.append(
        "Feature: hybrid_pic_model.mr_seam_band_width / mr_seam_eta / "
        "mr_seam_eta_h -- a half-cosine ramp of the plasma resistivity and "
        "hyper-resistivity from the coarse-fine patch edge (ramp = 1) to "
        "the fine interior (ramp = 0 at W cells), applied on fine levels "
        "only, inside the Ohm's-law E solve (solve_for_Faraday branch "
        "only, so the seam-band eta never enters the E-field used for the "
        "particle push). Physical setup identical to the T1 battery "
        "(T1_RESULTS.md of the hybrid_mr_p1 line): quasi-1D 2D hybrid-PIC, "
        "B0 = 0.25 T, beta = 1, coarse 4 x 1024 (dz = 0.5 l_i), ratio-2 "
        "patch over the middle quarter of z, 64 ppc, 24 RK4 substeps; "
        "eta_h* = mu0 w_ci / k_Nc^4 = 7.661e-18 Ohm m^3; seam targets: "
        "eta = 3e-8 Ohm m (the T1 global stabilization threshold), eta_h "
        "= 0.2 eta_h* = 1.532e-18 Ohm m^3 (the T1.2 recommended coarse "
        "absorber = the coarse-matched value). Deck/driver: "
        "t1_whistler.py + t1b_run_battery.py; runs in t1b_runs/.\n"
    )
    e1_analysis(results_md)
    e456_analysis(results_md)
    e2_t11(results_md)
    e2_t12(results_md)
    e3_note(results_md)
    matched_damping_note(results_md)
    mechanism_note(results_md)
    results_md.append(
        "\n## Figures\n\n"
        "- t1b_kill.png -- E1 level-1 magnetic-energy growth, all arms\n"
        "- t1b_coarse_kill.png -- E4/E5/E6 coarse-side band and operator "
        "A/Bs\n"
        "- t1b_packet.png -- E2 packet budget with the band\n"
        "- t1b_null_sigma.png -- E3 bulk null (written by "
        "t1b_smoke_null.py)\n"
    )
    out = os.path.join(HERE, "T1B_RESULTS.md")
    with open(out, "w") as f:
        f.write("\n".join(results_md) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
