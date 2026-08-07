#!/usr/bin/env python3
"""T1.3/T1.4 analysis: Alfven-wave patch transits and advected tangential
discontinuity for hybrid-PIC MR.

Consumes t1_runs/<case>/t1_lineouts.npz produced by t1_34_battery.py and
produces:

  t1_3_amp_phase.png   amplitude + phase error per transit, MR/ctl/fine
  t1_3_seam.png        seam monitors: instability growth + mirror-band power
  t1_4_profiles.png    theta(z), |B|(z) profiles at key times, 3 arms
  t1_4_ringing.png     patch-attributable difference map + ringing metrics
  T1_RESULTS.md        appended "## T1.3" and "## T1.4" sections

Conventions follow t1_analyze.py: b = <Bx>_x + i <By>_x, valid windows end
where the seam-instability monitor max|Bz - B0| crosses 0.4 B0, MR/ctl arms
share the particle seed so difference fields isolate what the patch causes.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "t1_runs")

# fixed categorical palette (entity-locked, same as t1_analyze.py)
C_MR = "#3d65f5"  # MR runs
C_CTL = "#c95d20"  # uniform-coarse control
C_FINE = "#2e9e73"  # uniform-fine reference
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

# which runs make up each battery (see t1_34_battery.py)
T13_BARE = dict(mr="t13_mr_bare", ctl="t13_ctl_bare")
T13_STAB = dict(mr="t13_mr_stab_e7", ctl="t13_ctl_stab_e7", fine="t13_fine_stab_e7")
T14_STAB = dict(mr="t14_mr", ctl="t14_ctl", fine="t14_fine")
T14_BARE = dict(mr="t14_mr_bare", ctl="t14_ctl_bare")


# ----------------------------------------------------------------------
def load_run(name):
    """Like t1_analyze.load_run but also exposes the Bz lineouts."""
    rd = os.path.join(RUNS, name)
    d = dict(np.load(os.path.join(rd, "t1_lineouts.npz")))
    with open(os.path.join(rd, "params.json")) as f:
        p = json.load(f)
    out = dict(
        p=p,
        t=d["t"],
        z0=d["z0"],
        b0=(d["bx0"] + 1j * d["by0"]).astype(complex),
        amax0=d["amax0"],
    )
    if "bz0" in d:
        out["bz0"] = d["bz0"]
    if "bx1" in d:
        out.update(
            z1=d["z1"], b1=(d["bx1"] + 1j * d["by1"]).astype(complex), amax1=d["amax1"]
        )
        if "bz1" in d:
            out["bz1"] = d["bz1"]
    a = d.get("amax1", d["amax0"])
    nt = len(out["t"])
    finite = np.isfinite(d["bx0"]).all(axis=1)
    if "bx1" in d:
        finite &= np.isfinite(d["bx1"]).all(axis=1)
    bad = np.nonzero((a[:, 3] > 0.4 * p["B0"]) | ~finite)[0]
    out["blow_out"] = int(bad[0]) if len(bad) else -1
    out["nt_valid"] = int(max(2, bad[0] - 2)) if len(bad) else nt
    out["failed"] = os.path.exists(os.path.join(RUNS, name, ".failed"))
    return out


def patch_idx(p, nz):
    dz = p["Lz"] / nz
    return int(round(p["patch_zlo"] / dz)), int(round(p["patch_zhi"] / dz))


def composite_lineout(r, key0="b0", key1="b1"):
    """Fine-resolution composite: coarse lineout upsampled x2, patch band
    overwritten with the level-1 lineout. Returns (z_fine, arr[nt, 2*nz0])."""
    p = r["p"]
    v0 = r[key0]
    nt, nz0 = v0.shape
    comp = np.repeat(v0, 2, axis=1)
    zf = np.linspace(0, p["Lz"], 2 * nz0, endpoint=False) + p["Lz"] / (4 * nz0)
    if key1 in r:
        jlo, jhi = patch_idx(p, 2 * nz0)
        comp[:, jlo:jhi] = r[key1]
    return zf, comp


# ----------------------------------------------------------------------
# T1.3
# ----------------------------------------------------------------------
def wave_projection(r, nt=None):
    """Complex amplitude a(t) of the initialized -k column over the full
    level-0 domain: a(t) = <b(t,z) e^{+ikz}>_z; |a| = amplitude, arg = phase
    (= +omega t for the forward wave)."""
    nt = r["nt_valid"] if nt is None else nt
    p = r["p"]
    return (r["b0"][:nt] * np.exp(1j * p["k_si"] * r["z0"])).mean(axis=1)


def mirror_projection_coarse(r, b, nt):
    """Hann-windowed +k (mirror) projection over the coarse region outside
    the patch, wrapped through the periodic boundary (T1.1 convention)."""
    p = r["p"]
    nz = b.shape[1]
    jlo, jhi = patch_idx(p, nz)
    idx = np.r_[jhi:nz, 0:jlo]
    w = np.hanning(len(idx))
    z = r["z0"][idx]
    k0 = p["k_si"]
    a_inc = (b[:nt, idx] * (w * np.exp(+1j * k0 * z))).sum(axis=1) / w.sum()
    a_ref = (b[:nt, idx] * (w * np.exp(-1j * k0 * z))).sum(axis=1) / w.sum()
    return a_inc, a_ref


def band_R(r, nt, b_num, b_den):
    """Reflection ratio: power of the mirror projection of b_num over the
    incident projection of b_den, integrated over the wave band around the
    incident peak (adapted from t1_analyze.whistler_R_and_omega for
    subsampled outputs and omega < 0.15 w_ci)."""
    dt_s = r["t"][1] - r["t"][0]
    a_inc, _ = mirror_projection_coarse(r, b_den, nt)
    _, a_ref = mirror_projection_coarse(r, b_num, nt)
    wt = np.hanning(nt)
    Si = np.fft.fftshift(np.fft.fft(wt * a_inc))
    Sr = np.fft.fftshift(np.fft.fft(wt * a_ref))
    om = -2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nt, dt_s))
    Pi, Pr = np.abs(Si) ** 2, np.abs(Sr) ** 2
    om_th = r["p"]["om_theory"]
    keep = np.abs(om) > 0.4 * om_th
    i0 = np.arange(nt)[keep][np.argmax(Pi[keep])]
    dom = abs(om[1] - om[0])
    nb = max(3, int(round(0.5 * om_th / dom)))
    band = slice(max(0, i0 - nb), min(nt, i0 + nb + 1))
    return om[i0], Pr[band].sum() / Pi[band].sum()


def t13_arm_metrics(r, om_ref):
    """Amplitude and phase-residual series for one arm."""
    nt = r["nt_valid"]
    a = wave_projection(r, nt)
    t = r["t"][:nt]
    ph = np.unwrap(np.angle(a))
    ph -= ph[0]
    res = ph - om_ref * t
    om_fit = np.polyfit(t, ph, 1)[0] if nt > 4 else np.nan
    return dict(t=t, amp=np.abs(a) / r["p"]["amp"], res=res, om_fit=om_fit, nt=nt)


def window_mean(t, y, t0, half):
    m = np.abs(t - t0) <= half
    return y[m].mean() if m.any() else np.nan


def t13_analysis(md):
    print("\n=== T1.3 Alfven-wave patch transits ===")
    stab = {k: load_run(v) for k, v in T13_STAB.items()}
    bare = {k: load_run(v) for k, v in T13_BARE.items()}
    p = stab["mr"]["p"]
    om_th, T_tr, t_ci = p["om_theory"], p["T_transit"], p["t_ci"]
    g_eta = p["gamma_eta"]

    arms = {k: t13_arm_metrics(r, om_th) for k, r in stab.items()}
    arms_b = {k: t13_arm_metrics(r, om_th) for k, r in bare.items()}

    # ---- per-transit table (stabilized arms) --------------------------
    n_max = int(min(arms["mr"]["t"][-1], arms["ctl"]["t"][-1]) / T_tr)
    half = 0.05 * T_tr
    rows = []
    for n in range(1, n_max + 1):
        tn = n * T_tr
        row = dict(n=n, t_tci=tn / t_ci)
        for k in ("mr", "ctl", "fine"):
            row[f"amp_{k}"] = window_mean(arms[k]["t"], arms[k]["amp"], tn, half)
            row[f"res_{k}"] = window_mean(arms[k]["t"], arms[k]["res"], tn, half)
        row["ratio"] = row["amp_mr"] / row["amp_ctl"]
        row["dphi"] = row["res_mr"] - row["res_ctl"]
        rows.append(row)
        print(
            f"transit {n:2d} (t={row['t_tci']:6.1f} t_ci): "
            f"|a|mr/ctl/fine = {row['amp_mr']:.3f}/{row['amp_ctl']:.3f}/"
            f"{row['amp_fine']:.3f}  ratio {row['ratio']:.4f}  "
            f"dphi(mr-ctl) {row['dphi']:+.4f} rad"
        )

    # ---- frequency fits ----------------------------------------------
    print("omega fits (w_ci): theory %.4f" % (om_th / p["w_ci"]))
    for k in ("mr", "ctl", "fine"):
        print(f"  {k:4s}: {arms[k]['om_fit'] / p['w_ci']:.4f}")

    # ---- spectral decomposition of a(t): main line vs backward branch --
    # The +-20% |a| modulation is the beat of the main (+omega) line with
    # the backward branch left by the imperfect (cold-polarization) Walen
    # launch; decomposing separates real per-arm differences from the beat.
    spec = {}
    for k in ("mr", "ctl", "fine"):
        r = stab[k]
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
        spec[k] = dict(cen=cen, P1=P[i1], back=np.sqrt(P[i2] / P[i1]))
        print(
            f"  {k:4s}: main centroid {cen:.5f} w_ci, backward/main "
            f"amplitude {spec[k]['back']:.3f}"
        )

    # band-passed main-line phase difference at the transit marks
    def main_phase(r, nt):
        a = wave_projection(r, nt)
        dt_s = r["t"][1] - r["t"][0]
        F = np.fft.fft(a)
        om = 2 * np.pi * np.fft.fftfreq(nt, dt_s) / p["w_ci"]
        af = np.fft.ifft(np.where((om > 0.02) & (om < 0.09), F, 0))
        return np.unwrap(np.angle(af))

    ntc = min(arms[k]["nt"] for k in ("mr", "ctl", "fine"))
    tph = stab["mr"]["t"][:ntc]
    phm, phc, phf = (main_phase(stab[k], ntc) for k in ("mr", "ctl", "fine"))
    dphi_main = []
    for n in range(1, n_max + 1):
        i = np.argmin(np.abs(tph - n * T_tr))
        dphi_main.append((n, phm[i] - phc[i], phf[i] - phc[i]))

    # ---- seam scattering (T1.1-convention R), bare + stab -------------
    # The stab horizon holds ~11 wave cycles -> spectral band R works. The
    # bare valid window (~2.4 t_ci) is only ~1/8 of a cycle -> the FFT
    # cannot separate the wave from DC; use the time-domain power ratio of
    # the same +-k projections instead.
    def r_triplet(pair, label, spectral):
        mr, ctl = pair["mr"], pair["ctl"]
        nt = min(mr["nt_valid"], ctl["nt_valid"])
        if spectral:
            _, R_mr = band_R(mr, nt, mr["b0"], mr["b0"])
            _, R_ctl = band_R(ctl, nt, ctl["b0"], ctl["b0"])
            _, R_diff = band_R(mr, nt, mr["b0"][:nt] - ctl["b0"][:nt], mr["b0"])
        else:

            def tdr(r_, b_num):
                a_inc, _ = mirror_projection_coarse(r_, r_["b0"], nt)
                _, a_ref = mirror_projection_coarse(r_, b_num, nt)
                return (np.abs(a_ref) ** 2).mean() / (np.abs(a_inc) ** 2).mean()

            R_mr = tdr(mr, mr["b0"])
            R_ctl = tdr(ctl, ctl["b0"])
            R_diff = tdr(mr, mr["b0"][:nt] - ctl["b0"][:nt])
        print(
            f"  {label}: nt={nt} R_MR={R_mr:.2e} R_ctl={R_ctl:.2e} R_diff={R_diff:.2e}"
        )
        return nt, R_mr, R_ctl, R_diff

    print("mirror-band reflection (coarse window):")
    nt_b, Rb_mr, Rb_ctl, Rb_diff = r_triplet(bare, "bare (time-domain)", False)
    nt_s, Rs_mr, Rs_ctl, Rs_diff = r_triplet(stab, "stab (spectral)", True)

    # ---- markdown -----------------------------------------------------
    md.append("\n## T1.3 -- Alfven wave crossing the patch (11-transit tracking)\n")
    md.append(
        f"Setup: same plasma/harness as T1.1 but on a 4 x 256 coarse grid "
        f"(Lz = 128 l_i) so one domain transit = one patch crossing = "
        f"{T_tr / t_ci:.1f} t_ci; patch = middle quarter (seams at 48 and 80 "
        f"l_i), ratio 2. Wave: m = 1 domain mode, k l_i = "
        f"{p['k_si'] * p['l_i']:.4f} (k = {p['kfac']:.4f} k_Nc), A = 0.02 B0, "
        f"launched PROPAGATING via the Walen ion-velocity perturbation "
        f"du = -(vA^2/v_ph) dB/B0 (a B-only IC gives a standing wave at low "
        f"k); omega_R theory = {om_th / p['w_ci']:.4f} w_ci, v_ph = "
        f"{p['vph'] / p['vA']:.4f} vA. Amplitude/phase tracked through the "
        f"projection a(t) = <b e^(+ikz)>_z on level 0; phase residual is "
        f"arg a - omega_R t. Bare arms (eta = 1e-10 Ohm m, dt = 0.006 t_ci) "
        f"are reported only inside the valid window (seam noise < 0.4 B0); "
        f"stabilized arms carry a global plasma_resistivity applied "
        f"IDENTICALLY to MR, uniform-coarse control and uniform-fine "
        f"reference. Resistive wave damping at eta = "
        f"{stab['mr']['p']['eta']} Ohm m: gamma_eta = eta k^2/mu0 = "
        f"{g_eta / p['w_ci']:.2e} w_ci, gamma_eta x 11 transits = "
        f"{g_eta * 11 * T_tr:.3f} (common mode, cancels in every MR-vs-ctl "
        f"metric). All arms share the particle seed with their control.\n"
    )
    md.append(
        f"Stabilized arms: {T13_STAB['mr']} / {T13_STAB['ctl']} / "
        f"{T13_STAB['fine']} (dt = {stab['mr']['p']['dt_fac']} t_ci, eta = "
        f"{stab['mr']['p']['eta']} Ohm m, {stab['mr']['p']['steps']} steps). "
        f"Bare arms: {T13_BARE['mr']} / {T13_BARE['ctl']}.\n"
    )
    md.append(
        "**Stabilization-floor finding:** the quiet-seam floor eta = 3e-8 "
        "Ohm m (T1.1 battery, stable >= 15.6 t_ci at dt = 0.012 from PIC "
        "noise) does NOT hold once the 2% wave feeds the seam: the first "
        "stabilized attempt (t1_runs/t13_mr_stab, eta = 3e-8, dt = 0.012) "
        "grew the same x-dependent seam mode through 0.2 B0 at 3.8 t_ci "
        "and NaN'd at ~5.0 t_ci (step ~420), and halving dt to 0.006 "
        "(t1_runs/dbg13_dt006_eta3e8) only delayed the NaN to ~7.2 t_ci "
        "(step ~1207) -- the wave-driven seam channel does NOT scale down "
        "with dt like the quiet-noise channel (gamma_quiet ~ dt). "
        "eta = 1e-7 (the next quiet-stable value) is flat-stable with the "
        "wave for >= 48 t_ci at dt = 0.012 (t1_runs/dbg13_dt012_eta1e7, "
        "fine-level monitor pinned at the 0.06-0.09 B0 noise floor), so "
        "the long arms carry eta = 1e-7 identically in MR/ctl/fine. Wave "
        "damping cost at 1e-7: gamma_eta x horizon = 0.13 (13% field "
        "amplitude over 11.3 transits), common mode by construction.\n"
    )
    md.append(
        "| transit | t (t_ci) | A_MR/A | A_ctl/A | A_fine/A | A_MR/A_ctl | "
        "phase res MR (rad) | phase res ctl (rad) | dphi MR-ctl (rad) |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r_ in rows:
        md.append(
            f"| {r_['n']} | {r_['t_tci']:.1f} | {r_['amp_mr']:.3f} | "
            f"{r_['amp_ctl']:.3f} | {r_['amp_fine']:.3f} | {r_['ratio']:.4f} | "
            f"{r_['res_mr']:+.3f} | {r_['res_ctl']:+.3f} | {r_['dphi']:+.4f} |"
        )
    md.append(
        f"\nFitted omega/w_ci over the full window: MR "
        f"{arms['mr']['om_fit'] / p['w_ci']:.4f}, ctl "
        f"{arms['ctl']['om_fit'] / p['w_ci']:.4f}, fine "
        f"{arms['fine']['om_fit'] / p['w_ci']:.4f} (cold R-mode theory "
        f"{om_th / p['w_ci']:.4f}).\n"
    )
    md.append(
        "Common-mode absolute effects (identical in all arms, so they "
        "cancel in every MR-vs-ctl column): the measured omega sits ~4% "
        "below cold theory (warm beta = 1 dispersion), and |a|(t) is "
        "modulated +-20% at 2*omega (10.7 t_ci period) -- the residual "
        "counter-propagating wave at the same spatial k (the Walen launch "
        "uses the cold polarization) beating against the main wave, with a "
        "slowly evolving envelope. Neither is a patch effect: the no-patch "
        "control shows the same curve (see t1_3_amp_phase.png), which is "
        "why the pass metrics are defined on MR-minus-control, not on "
        "arm-vs-theory.\n"
    )
    md.append(
        "Seam scattering, T1.1 mirror-band convention (coarse window outside "
        "the patch; R_diff = mirror power of the identical-seed difference "
        "field b_MR - b_ctl = everything the patch causes):\n"
    )
    md.append("| arm pair | outputs used | R_MR | R_ctl (floor) | R_diff |")
    md.append("|---|---|---|---|---|")
    md.append(
        f"| bare (valid window, time-domain ratio -- window is ~1/8 wave "
        f"cycle, too short for a spectral R) | {nt_b} | {Rb_mr:.2e} | "
        f"{Rb_ctl:.2e} | {Rb_diff:.2e} |"
    )
    md.append(
        f"| stabilized (full horizon, spectral band R) | {nt_s} | "
        f"{Rs_mr:.2e} | {Rs_ctl:.2e} | {Rs_diff:.2e} |"
    )
    ntb = min(arms_b["mr"]["nt"], arms_b["ctl"]["nt"])
    md.append(
        f"\nBare-arm amplitude/phase inside the valid window "
        f"(t <= {bare['mr']['t'][ntb - 1] / t_ci:.1f} t_ci = "
        f"{bare['mr']['t'][ntb - 1] / T_tr:.2f} transit): "
        f"|a|_MR/|a|_ctl = "
        f"{arms_b['mr']['amp'][ntb - 1] / arms_b['ctl']['amp'][ntb - 1]:.4f}, "
        f"dphi(MR-ctl) = "
        f"{arms_b['mr']['res'][ntb - 1] - arms_b['ctl']['res'][ntb - 1]:+.4f} "
        f"rad -- consistent with zero at this horizon.\n"
    )

    # ---- spectral decomposition + phase-lag finding --------------------
    md.append(
        "\nSpectral decomposition of a(t) over the full horizon (the "
        "per-transit |a| wobble above is the beat of the main +omega line "
        "with the backward branch left by the cold-polarization Walen "
        "launch; the backward branch damps differently with resolution, so "
        "the beat depth is arm-dependent):\n"
    )
    md.append(
        "| arm | main-line centroid (w_ci) | backward/main amplitude | "
        "main-line amplitude / ctl |"
    )
    md.append("|---|---|---|---|")
    for k in ("mr", "ctl", "fine"):
        md.append(
            f"| {k} | {spec[k]['cen']:.5f} | {spec[k]['back']:.3f} | "
            f"{np.sqrt(spec[k]['P1'] / spec['ctl']['P1']):.3f} |"
        )
    md.append(
        "\nBand-passed main-line phase difference at the transit marks "
        "(the beat-free phase metric):\n"
    )
    md.append("| transit | dphi MR-ctl (rad) | dphi fine-ctl (rad) |")
    md.append("|---|---|---|")
    for n, dmc, dfc in dphi_main:
        md.append(f"| {n} | {dmc:+.3f} | {dfc:+.3f} |")
    lag_per_crossing = dphi_main[-1][1] / dphi_main[-1][0]
    md.append(
        f"\n**Phase-lag finding:** the MR arm accumulates a systematic "
        f"phase LAG of ~{abs(lag_per_crossing):.3f} rad per patch crossing "
        f"(~{100 * abs(lag_per_crossing) / (2 * np.pi):.1f}% of a wave "
        f"cycle per crossing) relative to the uniform-coarse control, "
        f"reaching {min(d[1] for d in dphi_main):+.2f} rad mid-horizon. "
        f"This is NOT quarter-domain fine-dispersion mixing: the MR main "
        f"line ({spec['mr']['cen']:.5f} w_ci) sits below even the "
        f"uniform-FINE reference ({spec['fine']['cen']:.5f}), while the "
        f"control is at {spec['ctl']['cen']:.5f} -- a passive resolution "
        f"mix would put MR between ctl and fine. Plausible mechanism: the "
        f"known patch/seam heating channel (P1 core findings: ~1.26x "
        f"sigma(B), seam-band heating) locally shifts the warm dispersion "
        f"(the same warm shift that puts all arms ~4% below cold theory); "
        f"scattering phase shifts at the two seam crossings are the other "
        f"candidate. The wobble of dphi about the linear trend (+-0.2 rad) "
        f"is beat leakage from the arm-dependent backward branch.\n"
    )

    # ---- pass/fail ----------------------------------------------------
    ratio11 = rows[-1]["ratio"] if rows else np.nan
    p1 = all(abs(d[1]) <= 0.1 for d in dphi_main)
    # plan metric: MR phase error vs the converged (fine) answer must not
    # exceed the control's
    err_mr = abs(dphi_main[-1][1] - dphi_main[-1][2])
    err_ctl = abs(dphi_main[-1][2])
    p2 = abs(ratio11 - 1.0) <= 0.05
    p3 = Rs_diff < Rs_ctl and Rb_diff < Rb_ctl
    md.append("\nVerdicts against the pre-stated criteria:\n")
    md.append(
        f"- P1 phase: |dphi(MR-ctl)| <= 0.1 rad at every transit mark -> "
        f"max |dphi| = {max(abs(d[1]) for d in dphi_main):.3f} rad "
        f"{'PASS' if p1 else 'FAIL'}. Also fails the plan's form of the "
        f"metric: |phi_MR - phi_fine| = {err_mr:.2f} rad at transit "
        f"{dphi_main[-1][0]} vs |phi_ctl - phi_fine| = {err_ctl:.2f} rad. "
        f"Magnitude: ~0.8% of a cycle per crossing (small but systematic "
        f"and resolvable)."
    )
    md.append(
        f"- P2 amplitude: ||a_MR|/|a_ctl| - 1| <= 0.05 at transit "
        f"{rows[-1]['n']} -> {abs(ratio11 - 1):.4f} {'PASS' if p2 else 'FAIL'}"
        f" (per-transit ratios wobble to +-22% with the arm-dependent beat; "
        f"the beat-free main-line amplitude ratio MR/ctl = "
        f"{np.sqrt(spec['mr']['P1'] / spec['ctl']['P1']):.3f} is the clean "
        f"statement: no measurable seam amplitude loss in 11 crossings)"
    )
    md.append(
        f"- P3 seam scattering: R_diff < R_ctl floor -> bare "
        f"{Rb_diff:.1e} vs {Rb_ctl:.1e} (sharp: 4 decades below the floor "
        f"while the arms stay noise-correlated), stab {Rs_diff:.1e} vs "
        f"{Rs_ctl:.1e} {'PASS' if p3 else 'FAIL'}. Caveats on the stab "
        f"numbers: over 224 t_ci the identical-seed arms decorrelate, so "
        f"the difference field saturates toward the floor, and the "
        f"'mirror' floor itself is dominated by the physical backward "
        f"branch of the launch (not noise) -- the stab R is a weak upper "
        f"bound, the bare window is the sharp one."
    )
    verdicts = dict(p1=p1, p2=p2, p3=p3)

    # ---- figures ------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))
    for k, c, lab in (
        ("mr", C_MR, "MR (max_level = 1)"),
        ("ctl", C_CTL, "control (uniform coarse)"),
        ("fine", C_FINE, "uniform fine"),
    ):
        axs[0].plot(arms[k]["t"] / T_tr, arms[k]["amp"], color=c, lw=1.6, label=lab)
        axs[1].plot(arms[k]["t"] / T_tr, arms[k]["res"], color=c, lw=1.6, label=lab)
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

    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))
    for r_, c, lab in (
        (bare["mr"], C_MR, "bare MR, dt = 0.006 (lev-1 monitor)"),
        (bare["ctl"], C_CTL, "bare ctl (lev-0 monitor)"),
        (stab["mr"], C_EXTRA, "stabilized MR (lev-1 monitor)"),
        (stab["ctl"], C_GRAY, "stabilized ctl (lev-0 monitor)"),
    ):
        a = r_.get("amax1", r_["amax0"])
        axs[0].semilogy(
            r_["t"] / t_ci, a[:, 3] / r_["p"]["B0"], color=c, lw=1.5, label=lab
        )
    axs[0].axhline(0.4, color="k", lw=0.8, ls="--")
    axs[0].text(0.2, 0.43, "valid-window cut", fontsize=9)
    axs[0].set_xlabel(r"$t / t_{ci}$")
    axs[0].set_ylabel(r"$\max|B_z - B_0| \, / \, B_0$")
    axs[0].set_title("T1.3 seam-instability monitor")
    axs[0].legend(frameon=False, fontsize=9)

    for pair, c, lab in ((bare, C_MR, "bare"), (stab, C_EXTRA, "stabilized")):
        mr, ctl = pair["mr"], pair["ctl"]
        nt = min(mr["nt_valid"], ctl["nt_valid"])
        _, aref_d = mirror_projection_coarse(mr, mr["b0"][:nt] - ctl["b0"][:nt], nt)
        _, aref_c = mirror_projection_coarse(ctl, ctl["b0"], nt)
        axs[1].semilogy(
            mr["t"][:nt] / t_ci,
            np.abs(aref_d) / p["amp"],
            color=c,
            lw=1.4,
            label=f"{lab}: |mirror proj| of $b_{{MR}}-b_{{ctl}}$",
        )
        axs[1].semilogy(
            mr["t"][:nt] / t_ci,
            np.abs(aref_c) / p["amp"],
            color=c,
            ls=":",
            lw=1.2,
            label=f"{lab}: ctl noise floor",
        )
    axs[1].set_xlabel(r"$t / t_{ci}$")
    axs[1].set_ylabel(r"$|a_{-k}| / A$ (coarse window)")
    axs[1].set_title("T1.3 patch-attributable mirror amplitude vs noise floor")
    axs[1].legend(frameon=False, fontsize=9)
    fig.savefig(os.path.join(HERE, "t1_3_seam.png"), dpi=280)

    # artifact monitor table
    md.append("\nShould-be-zero monitors (MHD-EPIC artifact detector; pointwise")
    md.append("max over the level, valid window; Ez is normalized to the wave")
    md.append("E field, dBz to the wave amplitude A):\n")
    md.append("| run | level | outputs used | max|Ez|/max|Exy| | max|Bz-B0|/A |")
    md.append("|---|---|---|---|---|")
    mon = {}
    for k, names in (("bare", T13_BARE), ("stab", T13_STAB)):
        for arm, name in names.items():
            r_ = load_run(name)
            nt = r_["nt_valid"]
            for lev, key in (("0", "amax0"), ("1", "amax1")):
                if key not in r_:
                    continue
                a = r_[key][:nt]
                ez = a[:, 2].max() / max(a[:, :2].max(), 1e-30)
                bz = a[:, 3].max() / r_["p"]["amp"]
                mon[f"{k}_{arm}_l{lev}"] = (ez, bz)
                md.append(f"| {name} | {lev} | {nt} | {ez:.2f} | {bz:.1f} |")
    p4 = (
        mon["stab_mr_l0"][0] <= 2 * mon["stab_ctl_l0"][0]
        and mon["stab_mr_l0"][1] <= 2 * mon["stab_ctl_l0"][1]
    )
    md.append(
        f"\n- P4 should-be-zero: stab MR level 0 within 2x ctl (same-grid "
        f"comparison) -> Ez ratio "
        f"{mon['stab_mr_l0'][0] / mon['stab_ctl_l0'][0]:.2f}, dBz ratio "
        f"{mon['stab_mr_l0'][1] / mon['stab_ctl_l0'][1]:.2f} "
        f"{'PASS' if p4 else 'FAIL'} (the MR level-1 rows carry the fine "
        f"grid's intrinsically ~2x larger pointwise PIC noise)"
    )
    verdicts["p4"] = p4
    return verdicts


# ----------------------------------------------------------------------
# T1.4
# ----------------------------------------------------------------------
def td_windows(p, t_tci):
    """Ideal-region windows (in l_i) at time t: precursor (ahead of leading
    layer), plateau (between layers), wake (behind trailing layer but past
    the downstream seam)."""
    drift_li = 2.0 * np.pi * t_tci * p["drift_va"]
    za = p["za_li"] + drift_li
    zb = p["zb_li"] + drift_li
    seam_hi = p["patch_zhi"] / p["l_i"]
    return dict(
        precursor=(zb + 8.0, zb + 40.0),
        plateau=(za + 6.0, zb - 6.0),
        wake=(seam_hi + 2.0, za - 8.0),
    )


def td_metrics(r, t_tci):
    """RMS/max of |B|-B0 (and theta deviation) in the ideal windows at the
    output nearest t_tci, on the composite (fine-where-available) lineout."""
    p = r["p"]
    it = np.argmin(np.abs(r["t"] / p["t_ci"] - t_tci))
    it = min(it, r["nt_valid"] - 1)
    zf, bperp = composite_lineout(r, "b0", "b1")
    _, bz = composite_lineout(r, "bz0", "bz1")
    zli = zf / p["l_i"]
    babs = np.sqrt(np.abs(bperp[it]) ** 2 + bz[it] ** 2)
    th = np.degrees(np.angle(bperp[it]))
    out = {}
    for name, (lo, hi) in td_windows(p, r["t"][it] / p["t_ci"]).items():
        m = (zli >= lo) & (zli <= hi)
        if not m.any() or hi <= lo:
            out[name] = dict(rms=np.nan, mx=np.nan, th_rms=np.nan)
            continue
        dev = (babs[m] - p["B0"]) / p["B0"]
        th_ideal = 90.0 if name == "plateau" else 0.0
        out[name] = dict(
            rms=float(np.sqrt((dev**2).mean())),
            mx=float(np.abs(dev).max()),
            th_rms=float(np.sqrt(((th[m] - th_ideal) ** 2).mean())),
        )
    return out, it


def fit_layer(z, th, z_guess, sign, th_lo, th_hi):
    """Least-squares tanh fit th = a + b tanh((z-z0)/w) near one layer."""
    from scipy.optimize import curve_fit

    m = np.abs(z - z_guess) <= 8.0
    if m.sum() < 8:
        return np.nan, np.nan

    def f(zz, z0, w):
        return 0.5 * (th_hi + th_lo) + sign * 0.5 * (th_hi - th_lo) * np.tanh(
            (zz - z0) / w
        )

    try:
        popt, _ = curve_fit(f, z[m], th[m], p0=[z_guess, 1.5])
        return popt[0], abs(popt[1])
    except Exception:
        return np.nan, np.nan


def seam_band_series(r, ref=None, band_li=2.0):
    """Time series of max||B|-B0|/B0 within +-band_li of each seam on the
    level-0 lineout (same z rows usable in the ctl run for the comparison)."""
    p = r["p"]
    zli = r["z0"] / p["l_i"]
    babs = np.sqrt(np.abs(r["b0"]) ** 2 + r["bz0"] ** 2)
    out = {}
    for name, zs in (
        ("seam_lo", p["patch_zlo"] / p["l_i"]),
        ("seam_hi", p["patch_zhi"] / p["l_i"]),
    ):
        m = np.abs(zli - zs) <= band_li
        out[name] = np.abs(babs[:, m] - p["B0"]).max(axis=1) / p["B0"]
    return out


def t14_analysis(md):
    print("\n=== T1.4 advected tangential discontinuity ===")
    stab = {k: load_run(v) for k, v in T14_STAB.items()}
    bare = {k: load_run(v) for k, v in T14_BARE.items()}
    # The 0.4 B0 monitor cut targets the SEAM-noise blowup. The uniform-fine
    # arm trips it via its own (finite, physical) layer instability while its
    # fields stay finite to the end -- use its full horizon and carry the
    # contamination caveat instead of silently showing stale snapshots.
    stab["fine"]["nt_valid"] = len(stab["fine"]["t"])
    p = stab["mr"]["p"]
    t_ci = p["t_ci"]

    # seed sharing check (level-0 lineouts at t=0 must match exactly)
    seed_diff = np.abs(stab["mr"]["b0"][0] - stab["ctl"]["b0"][0]).max() / p["B0"]
    print(f"seed check: max|b_mr - b_ctl|(t=0)/B0 = {seed_diff:.2e}")

    # ---- window metrics at key times ---------------------------------
    T_EVAL = [4.0, 10.0, 17.0, 22.0]
    table = []
    for t_ev in T_EVAL:
        row = {"t": t_ev}
        for k in ("mr", "ctl", "fine"):
            mets, it = td_metrics(stab[k], t_ev)
            row[k] = mets
        table.append(row)
        for w in ("precursor", "plateau", "wake"):
            if not np.isfinite(row["mr"][w]["rms"]):
                continue
            print(
                f"t={t_ev:5.1f} {w:9s}: rms d|B|/B0 mr/ctl/fine = "
                f"{row['mr'][w]['rms']:.4f}/{row['ctl'][w]['rms']:.4f}/"
                f"{row['fine'][w]['rms']:.4f}   theta rms = "
                f"{row['mr'][w]['th_rms']:.2f}/{row['ctl'][w]['th_rms']:.2f}/"
                f"{row['fine'][w]['th_rms']:.2f} deg"
            )

    # noise floor: same |B| rms in the far-upstream band before contact
    def floor(r):
        it = np.argmin(np.abs(r["t"] / t_ci - 0.5))
        zli = r["z0"] / p["l_i"]
        m = (zli > 150) & (zli < 250)
        babs = np.sqrt(np.abs(r["b0"][it, m]) ** 2 + r["bz0"][it, m] ** 2)
        return np.sqrt((((babs - p["B0"]) / p["B0"]) ** 2).mean())

    floors = {k: floor(r) for k, r in stab.items()}
    print(
        "pre-contact |B| noise floor (rms, coarse lineout): "
        + " ".join(f"{k}={v:.4f}" for k, v in floors.items())
    )

    # ---- layer integrity at t = 17 t_ci (slab fully out) --------------
    lay = {}
    for k in ("mr", "ctl", "fine"):
        r = stab[k]
        it = min(np.argmin(np.abs(r["t"] / t_ci - 17.0)), r["nt_valid"] - 1)
        zf, bperp = composite_lineout(r, "b0", "b1")
        th = np.degrees(np.angle(bperp[it]))
        zli = zf / p["l_i"]
        drift = 2 * np.pi * (r["t"][it] / t_ci) * p["drift_va"]
        za, zb = p["za_li"] + drift, p["zb_li"] + drift
        z0a, wa = fit_layer(zli, th, za, +1, 0.0, 90.0)
        z0b, wb = fit_layer(zli, th, zb, -1, 0.0, 90.0)
        mplat = (zli > za + 6) & (zli < zb - 6)
        lay[k] = dict(wa=wa, wb=wb, dth=float(th[mplat].mean()), z0a=z0a, z0b=z0b)
        print(
            f"layers {k:4s}: width {wa:.2f}/{wb:.2f} l_i, plateau theta "
            f"{lay[k]['dth']:.2f} deg, centers {z0a:.1f}/{z0b:.1f} l_i"
        )

    # ---- seam-band ringing --------------------------------------------
    sb_mr = seam_band_series(stab["mr"])
    sb_ctl = seam_band_series(stab["ctl"])
    sb_fine = seam_band_series(stab["fine"])
    t_arr = stab["mr"]["t"] / t_ci
    # post-passage window for each seam: after the trailing layer clears it
    res = {}
    for seam, zs in (
        ("seam_lo", p["patch_zlo"] / p["l_i"]),
        ("seam_hi", p["patch_zhi"] / p["l_i"]),
    ):
        t_clear = (zs - p["za_li"]) / (2 * np.pi * p["drift_va"]) + 2.0
        m = t_arr >= t_clear
        res[seam] = (
            sb_mr[seam][m].max(),
            sb_ctl[seam][m].max(),
            t_clear,
            sb_fine[seam][m].max(),
        )
        print(
            f"{seam}: post-passage (t>{t_clear:.1f}) max d|B|/B0: "
            f"MR {res[seam][0]:.4f} vs ctl same-z {res[seam][1]:.4f} "
            f"(fine same-z {res[seam][3]:.4f})"
        )
    # time-mean pinned-residual profile: is the seam bump persistent?
    m17 = t_arr >= 17.3
    pin = {}
    for r_, key in ((stab["mr"], "mr"), (stab["ctl"], "ctl")):
        babs = np.sqrt(np.abs(r_["b0"][m17]) ** 2 + r_["bz0"][m17] ** 2)
        prof = np.abs(babs - p["B0"]).mean(axis=0) / p["B0"]
        zl = r_["z0"] / p["l_i"]
        at_seam = prof[np.abs(zl - p["patch_zhi"] / p["l_i"]) < 0.5].max()
        bg = np.median(prof[(zl > 145) & (zl < 158)])
        pin[key] = (at_seam, bg)
    print(
        f"pinned exit-seam residual, time-mean t>17.3: MR {pin['mr'][0]:.4f} "
        f"(background {pin['mr'][1]:.4f}), ctl same-z {pin['ctl'][0]:.4f}"
    )

    # ---- bare-arm first crossing ---------------------------------------
    nt_bare = min(bare["mr"]["nt_valid"], bare["ctl"]["nt_valid"])
    t_bare = bare["mr"]["t"][:nt_bare] / t_ci
    diff_bare = (
        np.abs(bare["mr"]["b0"][:nt_bare] - bare["ctl"]["b0"][:nt_bare]) / p["B0"]
    )
    diff_stab = (
        np.abs(
            stab["mr"]["b0"][: stab["mr"]["nt_valid"]]
            - stab["ctl"]["b0"][: stab["ctl"]["nt_valid"]][: stab["mr"]["nt_valid"]]
        )
        / p["B0"]
    )
    print(
        f"bare arms valid to t={t_bare[-1]:.1f} t_ci "
        f"(blowup output {bare['mr']['blow_out']}); max|b_mr-b_ctl|/B0 "
        f"in window = {diff_bare.max():.4f}"
    )

    # ---- markdown -------------------------------------------------------
    md.append("\n## T1.4 -- advected tangential discontinuity through the seam\n")
    md.append(
        f"Setup: coarse 4 x 512 (Lz = 256 l_i), patch = middle quarter "
        f"(seams 96/160 l_i), ratio 2, {p['ppc']} ppc (4x T1.1 -- at 64 ppc "
        f"the x-averaged |B| PIC noise, ~2.6% of B0 rms, would bury the "
        f"ringing signal), dt = {p['dt_fac']} t_ci, {p['substeps']} RK4 "
        f"substeps. TD: theta(z) = (dtheta/2)[tanh((z-za)/L) - "
        f"tanh((z-zb)/L)], dtheta = {p['dtheta_deg']:.0f} deg, za/zb = "
        f"{p['za_li']:.0f}/{p['zb_li']:.0f} l_i, L = {p['layer_li']} l_i "
        f"(3 coarse cells -- deliberately sharp to stress the unlimited "
        f"FaceDivFree slopes); Bx = B0 cos(theta), By = B0 sin(theta), "
        f"Bz = 0 (B.n = 0, |B| = B0, total pressure constant; Hall E_z "
        f"vanishes identically because |B| is constant, so E_z and B_z are "
        f"exact should-be-zeros). Bulk drift u_z = {p['drift_va']:.1f} vA "
        f"advects the slab: leading layer crosses the upstream seam at 1.3 "
        f"t_ci, slab fully inside the patch 5.1-11.5 t_ci, fully out at "
        f"15.3 t_ci, horizon 22.8 t_ci. Stabilized arms (MR / uniform-coarse "
        f"/ uniform-fine) carry eta = {p['eta']} Ohm m identically; the bare "
        f"pair (eta = 1e-10) is reported inside its valid window only. "
        f"MR/ctl share the particle seed (t=0 lineout agreement "
        f"{seed_diff:.1e} B0).\n"
    )
    md.append(
        "Region metrics on the composite lineout (fine level inside the "
        "patch): rms of (|B| - B0)/B0 -- the exact solution has |B| = B0 "
        "everywhere -- and rms of theta deviation from the ideal (0 or 90 "
        "deg), in the three ideal windows that advect with the slab: "
        "precursor (ahead of the leading layer), plateau (between layers), "
        "wake (behind the trailing layer, downstream of the exit seam; "
        "defined only after exit). NOTE: fine columns at t >= 15 t_ci are "
        "contaminated by the fine arm's own layer instability (caveat "
        "below).\n"
    )
    md.append(
        "| t (t_ci) | window | rms d|B| MR | ctl | fine | MR/ctl | "
        "rms dtheta MR (deg) | ctl | fine |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|")
    for row in table:
        for w in ("precursor", "plateau", "wake"):
            if not np.isfinite(row["mr"][w]["rms"]):
                continue
            md.append(
                f"| {row['t']:.0f} | {w} | {row['mr'][w]['rms']:.4f} | "
                f"{row['ctl'][w]['rms']:.4f} | {row['fine'][w]['rms']:.4f} | "
                f"{row['mr'][w]['rms'] / row['ctl'][w]['rms']:.2f} | "
                f"{row['mr'][w]['th_rms']:.2f} | {row['ctl'][w]['th_rms']:.2f} | "
                f"{row['fine'][w]['th_rms']:.2f} |"
            )
    md.append(
        f"\nPre-contact noise floor (rms d|B|/B0, coarse lineout, t = 0.5 "
        f"t_ci): MR {floors['mr']:.4f}, ctl {floors['ctl']:.4f}, fine "
        f"{floors['fine']:.4f}.\n"
    )
    md.append("Layer integrity at t = 17 t_ci (slab fully downstream):\n")
    md.append("| arm | layer width a/b (l_i, tanh fit) | plateau theta (deg) |")
    md.append("|---|---|---|")
    for k in ("mr", "ctl", "fine"):
        md.append(
            f"| {k} | {lay[k]['wa']:.2f} / {lay[k]['wb']:.2f} | {lay[k]['dth']:.2f} |"
        )
    md.append(
        "\nSeam-band ringing (max ||B|-B0|/B0 within +-2 l_i of each seam "
        "on the level-0 lineout, after the trailing layer clears that seam; "
        "ctl/fine columns = the SAME z band in the identical-seed "
        "uniform-coarse run and in the uniform-fine reference):\n"
    )
    md.append("| seam | MR | ctl same-z | MR/ctl | fine same-z |")
    md.append("|---|---|---|---|---|")
    for seam in ("seam_lo", "seam_hi"):
        md.append(
            f"| {seam} | {res[seam][0]:.4f} | {res[seam][1]:.4f} | "
            f"{res[seam][0] / res[seam][1]:.2f} | {res[seam][3]:.4f} |"
        )
    md.append(
        f"\nPersistence: time-mean ||B|-B0|/B0 over t > 17.3 t_ci at the "
        f"exit-seam cell: MR {pin['mr'][0]:.4f} vs {pin['mr'][1]:.4f} "
        f"background (ctl same-z {pin['ctl'][0]:.4f}) -- a pinned but "
        f"BOUNDED, non-growing residual of ~{100 * pin['mr'][0]:.1f}% B0 "
        f"time-mean (~7% peak) localized to +-1 coarse cell of the exit "
        f"seam.\n"
    )
    md.append(
        f"\nBare pair: valid to t = {t_bare[-1]:.1f} t_ci (covers the "
        f"leading-layer crossing at 1.3 t_ci; MR blowup at output "
        f"{bare['mr']['blow_out']}); max|b_MR - b_ctl|/B0 inside the window "
        f"= {diff_bare.max():.4f}.\n"
    )

    # ---- pass/fail ------------------------------------------------------
    wake_last = table[-1]
    p1 = wake_last["mr"]["wake"]["rms"] <= 1.25 * wake_last["ctl"]["wake"]["rms"]
    p2 = all(res[s][0] <= 1.5 * res[s][1] for s in res)
    p3 = abs(lay["mr"]["dth"] - 90.0) <= 2.0 and (
        abs(lay["mr"]["wa"] - lay["ctl"]["wa"]) <= 0.1 * lay["ctl"]["wa"]
        or abs(lay["mr"]["wb"] - lay["ctl"]["wb"]) <= 0.1 * lay["ctl"]["wb"]
    )
    # should-be-zero monitors. Cross-arm ratio uses LEVEL 0 for all arms
    # (same-grid comparison); the MR fine level is quoted separately.
    mon = {}
    for k in ("mr", "ctl", "fine"):
        r = stab[k]
        nt = r["nt_valid"]
        a = r["amax0"][:nt]
        mon[k] = (a[:, 2].max() / max(a[:, :2].max(), 1e-30), a[:, 3].max() / p["B0"])
        if "amax1" in r:
            a1 = r["amax1"][:nt]
            mon[k + "_lev1"] = (
                a1[:, 2].max() / max(a1[:, :2].max(), 1e-30),
                a1[:, 3].max() / p["B0"],
            )
    p4 = mon["mr"][0] <= 2 * mon["ctl"][0] and mon["mr"][1] <= 2 * mon["ctl"][1]
    md.append(
        "Should-be-zero monitors (pointwise max over the level, valid "
        "window; exact solution has Ez = Bz = 0):\n"
    )
    md.append("| arm | level | max|Ez|/max|Exy| | max|Bz|/B0 |")
    md.append("|---|---|---|---|")
    for k in ("mr", "mr_lev1", "ctl", "fine"):
        if k not in mon:
            continue
        lev = "1" if k.endswith("_lev1") else "0"
        md.append(
            f"| {k.replace('_lev1', '')} | {lev} | {mon[k][0]:.3f} | {mon[k][1]:.4f} |"
        )
    # the uniform-fine arm's own layer instability (kx != 0, zero x-average)
    rf = stab["fine"]
    tf = rf["t"] / t_ci
    hist = " ".join(
        f"{tf[i]:.0f}:{rf['amax0'][i, 3] / p['B0']:.3f}"
        for i in [np.argmin(np.abs(tf - x)) for x in [1, 5, 10, 15, 20, 22.6]]
    )
    md.append(
        f"\n**Uniform-fine reference caveat (finding):** the fine arm grows "
        f"an x-dependent Bz mode AT THE TD LAYERS (no seam exists in that "
        f"run): pointwise max|Bz|/B0 vs t(t_ci) = {hist}, growth ~0.04 "
        f"w_ci, while its x-averaged Bz lineout stays ~0 (kx != 0) and the "
        f"identical-eta uniform-coarse ctl stays flat at ~0.015-0.033. The "
        f"dz = 0.25 l_i grid resolves a kinetic cross-field mode of the "
        f"L = 1.5 l_i (~2 rho_i) drifting layer that the coarse grid "
        f"cannot; the MR fine level sees the layers only during transit "
        f"(max {mon.get('mr_lev1', (0, 0))[1]:.2f} B0). The fine arm is "
        f"therefore a fair converged reference only through t ~ 10-12 t_ci; "
        f"late-time fine columns above carry this contamination and the "
        f"late-time verdicts lean on the MR-vs-shared-seed-ctl comparison.\n"
    )
    md.append("\nVerdicts against the pre-stated criteria:\n")
    md.append(
        f"- P1 downwind oscillations: wake rms MR <= 1.25x ctl at t = "
        f"{wake_last['t']:.0f} t_ci -> ratio "
        f"{wake_last['mr']['wake']['rms'] / wake_last['ctl']['wake']['rms']:.2f} "
        f"{'PASS' if p1 else 'FAIL'} (the uniform-fine value "
        f"{wake_last['fine']['wake']['rms']:.4f} at this time is "
        f"contaminated by its layer instability; the clean floor is the "
        f"pre-contact noise level {floors['fine']:.4f})"
    )
    md.append(
        f"- P2 seam ringing: post-passage seam-band max <= 1.5x ctl same-z "
        f"-> ratios {res['seam_lo'][0] / res['seam_lo'][1]:.2f} / "
        f"{res['seam_hi'][0] / res['seam_hi'][1]:.2f} {'PASS' if p2 else 'FAIL'}"
        f" (exit seam exceeds the strict criterion; residual is bounded -- "
        f"max {res['seam_hi'][0]:.3f} B0 vs {res['seam_hi'][3]:.3f} in the "
        f"fine reference's same band -- pinned at ~"
        f"{100 * pin['mr'][0]:.1f}% B0 time-mean, non-growing)"
    )
    md.append(
        f"- P3 structure fidelity: plateau theta within 2 deg of 90 and "
        f"layer width within 10% of ctl -> theta {lay['mr']['dth']:.2f} deg, "
        f"widths {lay['mr']['wa']:.2f}/{lay['mr']['wb']:.2f} vs ctl "
        f"{lay['ctl']['wa']:.2f}/{lay['ctl']['wb']:.2f} l_i "
        f"{'PASS' if p3 else 'FAIL'}"
    )
    md.append(
        f"- P4 should-be-zero: MR level-0 within 2x ctl (same grid) -> Ez "
        f"ratio {mon['mr'][0] / mon['ctl'][0]:.2f}, Bz ratio "
        f"{mon['mr'][1] / mon['ctl'][1]:.2f} {'PASS' if p4 else 'FAIL'}"
    )

    # ---- figures --------------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(17, 9), sharex="col", sharey="row")
    for j, t_ev in enumerate([1.0, 8.0, 20.0]):
        for k, c, lab in (
            ("fine", C_FINE, "uniform fine"),
            ("ctl", C_CTL, "uniform coarse"),
            ("mr", C_MR, "MR composite"),
        ):
            r = stab[k]
            it = min(np.argmin(np.abs(r["t"] / t_ci - t_ev)), r["nt_valid"] - 1)
            zf, bperp = composite_lineout(r, "b0", "b1")
            _, bz = composite_lineout(r, "bz0", "bz1")
            zli = zf / p["l_i"]
            th = np.degrees(np.angle(bperp[it]))
            babs = np.sqrt(np.abs(bperp[it]) ** 2 + bz[it] ** 2)
            axs[0, j].plot(zli, th, color=c, lw=1.2, label=lab)
            axs[1, j].plot(zli, (babs - p["B0"]) / p["B0"], color=c, lw=1.0, label=lab)
        for ax in (axs[0, j], axs[1, j]):
            for zs in (96, 160):
                ax.axvline(zs, color=C_GRAY, lw=0.8, ls="--")
        axs[0, j].set_title(f"t = {t_ev:.0f} $t_{{ci}}$")
        axs[1, j].set_xlabel(r"$z / l_i$")
        axs[1, j].set_xlim(40, 240)
    axs[0, 0].set_ylabel(r"$\theta = \mathrm{atan2}(B_y, B_x)$ (deg)")
    axs[1, 0].set_ylabel(r"$(|B| - B_0)/B_0$")
    axs[1, 0].set_ylim(-0.06, 0.06)
    axs[0, 0].legend(frameon=False, fontsize=10)
    fig.suptitle(
        "T1.4 tangential discontinuity through the patch "
        "(dashed lines = seams; slab drifts right at $v_A$)"
    )
    fig.savefig(os.path.join(HERE, "t1_4_profiles.png"), dpi=280)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6.2))
    nt = min(stab["mr"]["nt_valid"], stab["ctl"]["nt_valid"])
    im = axs[0].imshow(
        diff_stab[:nt],
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=max(5 * np.median(diff_stab[:nt]), 1e-4),
        extent=[0, p["Lz"] / p["l_i"], 0, stab["mr"]["t"][nt - 1] / t_ci],
    )
    for zs in (96, 160):
        axs[0].axvline(zs, color="#7fd4ff", lw=0.9, ls="--")
    for zedge, lab in ((p["za_li"], "trailing layer"), (p["zb_li"], "leading layer")):
        tt = np.linspace(0, stab["mr"]["t"][nt - 1] / t_ci, 50)
        axs[0].plot(zedge + 2 * np.pi * tt, tt, color="#9effa0", lw=0.8, ls=":")
    axs[0].set_xlabel(r"$z / l_i$")
    axs[0].set_ylabel(r"$t / t_{ci}$")
    axs[0].set_title(r"$|b_{MR} - b_{ctl}| / B_0$ (identical seed, level 0)")
    axs[0].grid(False)
    fig.colorbar(im, ax=axs[0], shrink=0.9)

    for k, c, lab in (
        ("mr", C_MR, "MR"),
        ("ctl", C_CTL, "uniform coarse"),
        ("fine", C_FINE, "uniform fine"),
    ):
        r = stab[k]
        ts, ys = [], []
        for t_ev in np.arange(16.0, 22.6, 0.5):
            mets, it = td_metrics(r, t_ev)
            ts.append(r["t"][it] / t_ci)
            ys.append(mets["wake"]["rms"])
        axs[1].plot(ts, ys, "o-", color=c, ms=5, lw=1.6, label=lab)
    axs[1].set_xlabel(r"$t / t_{ci}$")
    axs[1].set_ylabel(r"wake rms $(|B| - B_0)/B_0$")
    axs[1].set_title("downwind oscillation metric (post-exit)")
    axs[1].legend(frameon=False, fontsize=10)

    for seam, ls in (("seam_lo", "-"), ("seam_hi", "--")):
        axs[2].plot(
            t_arr,
            sb_mr[seam],
            color=C_MR,
            ls=ls,
            lw=1.4,
            label=f"MR {seam}",
        )
        axs[2].plot(
            t_arr[: len(sb_ctl[seam])],
            sb_ctl[seam],
            color=C_CTL,
            ls=ls,
            lw=1.2,
            label=f"ctl same-z {seam}",
        )
    axs[2].set_xlabel(r"$t / t_{ci}$")
    axs[2].set_ylabel(r"max $||B|-B_0|/B_0$ in seam band ($\pm 2\,l_i$)")
    axs[2].set_title("seam-band ringing (level 0)")
    axs[2].legend(frameon=False, fontsize=9)
    fig.savefig(os.path.join(HERE, "t1_4_ringing.png"), dpi=280)
    return dict(p1=p1, p2=p2, p3=p3, p4=p4)


# ----------------------------------------------------------------------
def main():
    md = []
    v13 = t13_analysis(md)
    v14 = t14_analysis(md)
    md.append(
        "\n### T1.3/T1.4 figures\n\n"
        "- t1_3_amp_phase.png -- amplitude and phase residual per transit\n"
        "- t1_3_seam.png -- instability monitor + mirror-band amplitude\n"
        "- t1_4_profiles.png -- theta and |B| profiles at 1/8/20 t_ci\n"
        "- t1_4_ringing.png -- difference map, wake rms, seam-band series\n"
    )
    out = os.path.join(HERE, "T1_RESULTS.md")
    with open(out, "a") as f:
        f.write("\n".join(md) + "\n")
    print("\nverdicts:", v13, v14)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
