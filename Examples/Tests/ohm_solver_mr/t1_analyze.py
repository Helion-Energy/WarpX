#!/usr/bin/env python3
"""T1 analysis: whistler seam-reflection battery for hybrid-PIC MR.

Consumes the t1_runs/<case>/t1_lineouts.npz files produced by
t1_run_battery.py (see t1_whistler.py for the deck) and produces:

  t1_kw_map_mr_vs_ctl.png   representative (k, omega) power maps, MR vs ctl
  t1_Rk_curve.png           reflection ratio R(k), MR vs control noise floor
  t1_dispersion.png         measured whistler omega vs k against theory
  t1_2_spacetime.png        T1.2 packet space-time diagrams (bare vs eta_h)
  t1_2_budget_vs_etah.png   T1.2 packet energy budget vs hyper-resistivity
  t1_instability.png        seam-instability growth (diagnostic runs)
  T1_RESULTS.md             all tables

Analysis conventions
  b(t, z) = <Bx>_x + i <By>_x. The initial condition is A e^{-i k0 z}
  (a single spatial Fourier column), so any power in the mirrored column
  +k0 comes from reflection, mode conversion at the seams, or noise. The
  reflection ratio is measured on the coarse region OUTSIDE the patch
  (wrapped through the periodic boundary, Hann-windowed):
      a_s(t) = sum_z w(z) b(t,z) e^{+i s k0 z},  s = +1 incident, -1 mirror
      R = integral of |FFT_t a_-|^2 over the whistler band
        / same for a_+.
  The identically processed max_level=0 control sets the noise floor.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "t1_runs")

# fixed categorical palette (assigned by entity, fixed order)
C_MR = "#3d65f5"  # MR runs
C_CTL = "#c95d20"  # max_level=0 coarse control
C_FINE = "#2e9e73"  # uniform-fine reference
C_EXTRA = "#8d69cf"
C_GRAY = "#6e6e6e"

KFACS = [0.1, 0.2, 0.3, 0.45, 0.6, 0.7, 0.8, 0.9]
ETAH_FACS = [0.0, 0.03, 0.2, 1.0]

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


# ----------------------------------------------------------------------
def load_run(name):
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
    if "bx1" in d:
        out.update(
            z1=d["z1"], b1=(d["bx1"] + 1j * d["by1"]).astype(complex), amax1=d["amax1"]
        )
    # valid window: trim from the first sign of blowup. Ordinary PIC noise
    # holds max|Bz-B0| at the few-percent-of-B0 level; the seam instability
    # blows through 0.4 B0 on its way to NaN.
    a = d.get("amax1", d["amax0"])
    nt = len(out["t"])
    finite = np.isfinite(d["bx0"]).all(axis=1)
    if "bx1" in d:
        finite &= np.isfinite(d["bx1"]).all(axis=1)
    bad = np.nonzero((a[:, 3] > 0.4 * p["B0"]) | ~finite)[0]
    out["blow_step"] = int(bad[0]) if len(bad) else -1
    out["nt_valid"] = int(max(2, bad[0] - 20)) if len(bad) else nt
    return out


def load_run_pair(name, null_name):
    """Load a run and coherently subtract its identical-seed null (noise
    only) twin, isolating the packet signal from thermal PIC field noise."""
    r = load_run(name)
    rn = load_run(null_name)
    nt = min(r["nt_valid"], rn["nt_valid"])
    r["nt_valid"] = nt
    r["b0"] = r["b0"][:nt] - rn["b0"][:nt]
    if "b1" in r:
        r["b1"] = r["b1"][:nt] - rn["b1"][:nt]
    return r


def patch_idx(p, nz):
    dz = p["Lz"] / nz
    return int(round(p["patch_zlo"] / dz)), int(round(p["patch_zhi"] / dz))


def coarse_window_proj(r, nt):
    """Hann-windowed +-k0 projections over the coarse region outside the
    patch (contiguous through the periodic z boundary)."""
    p = r["p"]
    b = r["b0"][:nt]
    nz = b.shape[1]
    jlo, jhi = patch_idx(p, nz)
    idx = np.r_[jhi:nz, 0:jlo]  # wrapped coarse segment
    w = np.hanning(len(idx))
    z = r["z0"][idx]
    k0 = p["k_si"]
    a_inc = (b[:, idx] * (w * np.exp(+1j * k0 * z))).sum(axis=1) / w.sum()
    a_ref = (b[:, idx] * (w * np.exp(-1j * k0 * z))).sum(axis=1) / w.sum()
    return a_inc, a_ref


def whistler_R_and_omega(r, nt, b_diff=None):
    """Return (omega_peak [rad/s], R_band, R_broad, domega) from the
    coarse-window projections of run r over nt samples. With b_diff (an
    identical-seed difference field, e.g. b_MR - b_ctl), the mirror
    projection is taken from it instead, so the coherent PIC noise cancels
    and R becomes the patch-attributable reflected power."""
    p = r["p"]
    dt = p["dt"]
    a_inc, a_ref = coarse_window_proj(r, nt)
    if b_diff is not None:
        rd = dict(r, b0=b_diff)
        _, a_ref = coarse_window_proj(rd, nt)
    wt = np.hanning(nt)
    Si = np.fft.fftshift(np.fft.fft(wt * a_inc))
    Sr = np.fft.fftshift(np.fft.fft(wt * a_ref))
    om = -2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nt, dt))
    Pi, Pr = np.abs(Si) ** 2, np.abs(Sr) ** 2
    keep = np.abs(om) > 0.15 * p["w_ci"]  # exclude DC leakage
    i0 = np.arange(nt)[keep][np.argmax(Pi[keep])]
    # parabolic refinement of the peak
    if 0 < i0 < nt - 1 and Pi[i0 - 1] > 0 and Pi[i0 + 1] > 0:
        y = np.log(Pi[i0 - 1 : i0 + 2])
        sh = 0.5 * (y[0] - y[2]) / (y[0] - 2 * y[1] + y[2])
    else:
        sh = 0.0
    dom = abs(om[1] - om[0])
    om_pk = om[i0] + sh * (om[min(i0 + 1, nt - 1)] - om[i0])
    band = slice(max(0, i0 - 3), min(nt, i0 + 4))
    R_band = Pr[band].sum() / Pi[band].sum()
    R_broad = Pr[keep].sum() / Pi[keep].sum()
    return om_pk, R_band, R_broad, dom


def whistler_analytic(k, l_i, w_ci, dz=None):
    """Cold hybrid R-mode (whistler): omega in rad/s for k in 1/m. With dz,
    use the centered-difference effective wavenumber of the grid."""
    if dz is not None:
        k = 2.0 * np.sin(k * dz / 2.0) / dz
    kn = k * l_i
    return w_ci * 0.5 * (kn**2 + kn * np.sqrt(kn**2 + 4.0))


# ----------------------------------------------------------------------
# T1.1
# ----------------------------------------------------------------------
def t11_analysis(results_md):
    rows = []
    print("\n=== T1.1 dispersion + reflection below the coarse Nyquist ===")
    hdr = (
        f"{'kfac':>6s} {'k*l_i':>6s} {'T/tci':>6s} {'w_MR':>7s} {'w_ctl':>7s} "
        f"{'dw/bin':>7s} {'w_disc':>7s} {'R_MR':>9s} {'R_ctl':>9s} {'ratio':>7s} "
        f"{'R_diff':>9s}"
    )
    print(hdr)
    for kf in KFACS:
        tag = f"t11_k{int(round(kf * 1000)):04d}"
        mr, ctl = load_run(tag + "_mr"), load_run(tag + "_ctl")
        nt = min(mr["nt_valid"], ctl["nt_valid"])
        p = mr["p"]
        w_ci, l_i, dzc = p["w_ci"], p["l_i"], p["dz_coarse"]
        om_mr, R_mr, Rb_mr, dom = whistler_R_and_omega(mr, nt)
        om_ct, R_ct, Rb_ct, _ = whistler_R_and_omega(ctl, nt)
        # identical-seed difference: the patch-attributable -k power
        _, R_diff, _, _ = whistler_R_and_omega(
            mr, nt, b_diff=mr["b0"][:nt] - ctl["b0"][:nt]
        )
        om_disc = whistler_analytic(p["k_si"], l_i, w_ci, dz=dzc)
        om_cont = whistler_analytic(p["k_si"], l_i, w_ci)
        T_tci = nt * p["dt"] / p["t_ci"]
        rows.append(
            dict(
                kfac=p["kfac"],
                kli=p["k_si"] * l_i,
                nt=nt,
                T_tci=T_tci,
                om_mr=abs(om_mr) / w_ci,
                om_ctl=abs(om_ct) / w_ci,
                dbin=abs(abs(om_mr) - abs(om_ct)) / dom,
                om_disc=om_disc / w_ci,
                om_cont=om_cont / w_ci,
                R_mr=R_mr,
                R_ctl=R_ct,
                R_diff=R_diff,
                Rb_mr=Rb_mr,
                Rb_ctl=Rb_ct,
                blow=mr["blow_step"],
                nan=os.path.exists(os.path.join(RUNS, tag + "_mr", ".failed")),
            )
        )
        r = rows[-1]
        print(
            f"{r['kfac']:6.3f} {r['kli']:6.2f} {r['T_tci']:6.2f} "
            f"{r['om_mr']:7.2f} {r['om_ctl']:7.2f} {r['dbin']:7.2f} "
            f"{r['om_disc']:7.2f} {r['R_mr']:9.2e} {r['R_ctl']:9.2e} "
            f"{r['R_mr'] / r['R_ctl']:7.2f} {r['R_diff']:9.2e}"
            + ("   [seam>0.4B0 @ step %d]" % r["blow"] if r["blow"] > 0 else "")
        )

    # uniform-fine reference at kfac 0.6
    fr = load_run("t11_k0600_fine")
    ntf = fr["nt_valid"]
    om_f, R_f, Rb_f, dom_f = whistler_R_and_omega(fr, ntf)
    p = fr["p"]
    om_f_disc = whistler_analytic(
        p["k_si"], p["l_i"], p["w_ci"], dz=0.5 * p["dz_coarse"]
    )
    print(
        f"fine ref kfac 0.600: w {abs(om_f) / p['w_ci']:.2f} "
        f"(disc fine {om_f_disc / p['w_ci']:.2f}), R {R_f:.2e}"
    )

    # ---------------- results table -----------------------------------
    results_md.append(
        "\n## T1.1 -- dispersion and reflection, k below the coarse Nyquist\n"
    )
    results_md.append(
        "Both arms share the particle seed, so the control's R is the exact "
        "noise realization under the MR run's R, and R_diff -- the -k power "
        "of the difference field (b_MR - b_ctl), i.e. everything the patch "
        "causes in the coarse region (coherent reflection + seam noise) -- "
        "is the sharpest reflection bound. The analysis window ends where "
        "the seam-instability noise (see below) reaches 0.4 B0.\n"
    )
    results_md.append(
        "| k/k_Nc | k l_i | T (t_ci) | w_MR/w_ci | w_ctl/w_ci | dw (bins) | "
        "w_grid-theory | R_MR | R_ctl (floor) | R_MR/R_ctl | R_diff | note |"
    )
    results_md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        note = f"seam noise 0.4 B0 @ step {r['blow']}" if r["blow"] > 0 else ""
        if r["nan"]:
            note += "; run later hit NaN"
        results_md.append(
            f"| {r['kfac']:.3f} | {r['kli']:.2f} | {r['T_tci']:.2f} | "
            f"{r['om_mr']:.2f} | {r['om_ctl']:.2f} | {r['dbin']:.2f} | "
            f"{r['om_disc']:.2f} | {r['R_mr']:.2e} | {r['R_ctl']:.2e} | "
            f"{r['R_mr'] / r['R_ctl']:.2f} | {r['R_diff']:.2e} | {note} |"
        )
    results_md.append(
        f"\nUniform-fine reference (kfac 0.6, no patch): "
        f"w = {abs(om_f) / p['w_ci']:.2f} w_ci "
        f"(fine-grid theory {om_f_disc / p['w_ci']:.2f}), R = {R_f:.2e}. "
        "The omega pass criterion is |w_MR - w_ctl| within ~1 FFT bin; "
        "w_grid-theory is the cold R-mode evaluated with the coarse grid's "
        "effective wavenumber 2 sin(k dz/2)/dz."
    )

    # ---------------- R(k) figure --------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6.5))
    kk = [r["kfac"] for r in rows]
    ax.semilogy(
        kk,
        [r["R_mr"] for r in rows],
        "o-",
        color=C_MR,
        lw=2,
        ms=7,
        label="MR (max_level = 1)",
    )
    ax.semilogy(
        kk,
        [r["R_ctl"] for r in rows],
        "s--",
        color=C_CTL,
        lw=2,
        ms=6,
        label="control (max_level = 0), noise floor",
    )
    ax.semilogy(
        kk,
        [r["R_diff"] for r in rows],
        "^:",
        color=C_EXTRA,
        lw=2,
        ms=7,
        label="patch-attributable (identical-seed difference)",
    )
    ax.semilogy([0.6], [R_f], "D", color=C_FINE, ms=9, label="uniform-fine reference")
    ax.set_xlabel(r"$k / k_{\rm Nyq}^{\rm coarse}$")
    ax.set_ylabel(r"$R(k) = P(-k,\omega_w)\,/\,P(+k,\omega_w)$")
    ax.set_title(
        "T1.1 whistler reflection at the MR seams (coarse-region window, whistler band)"
    )
    ax.legend(frameon=False)
    fig.savefig(os.path.join(HERE, "t1_Rk_curve.png"), dpi=280)

    # ---------------- dispersion figure --------------------------------
    fig, ax = plt.subplots(figsize=(10, 6.5))
    p0 = load_run("t11_k0100_mr")["p"]
    kfine = np.linspace(0.02, 1.0, 200)
    ks = kfine * p0["k_nyq_coarse"]
    ax.plot(
        kfine,
        whistler_analytic(ks, p0["l_i"], p0["w_ci"]) / p0["w_ci"],
        color=C_GRAY,
        lw=1.5,
        ls=":",
        label="cold R-mode (continuum)",
    )
    ax.plot(
        kfine,
        whistler_analytic(ks, p0["l_i"], p0["w_ci"], dz=p0["dz_coarse"]) / p0["w_ci"],
        color=C_GRAY,
        lw=1.5,
        ls="--",
        label="cold R-mode (coarse grid)",
    )
    ax.plot(kk, [r["om_ctl"] for r in rows], "s", color=C_CTL, ms=7, label="control")
    ax.plot(
        kk,
        [r["om_mr"] for r in rows],
        "o",
        color=C_MR,
        ms=7,
        mfc="none",
        mew=2,
        label="MR",
    )
    ax.plot(
        [0.6], [abs(om_f) / p["w_ci"]], "D", color=C_FINE, ms=8, label="uniform fine"
    )
    ax.set_xlabel(r"$k / k_{\rm Nyq}^{\rm coarse}$")
    ax.set_ylabel(r"$\omega / \omega_{ci}$")
    ax.set_title("T1.1 measured whistler frequency (coarse-region window)")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(HERE, "t1_dispersion.png"), dpi=280)

    # ---------------- representative (k, omega) maps -------------------
    tag = "t11_k0700"
    mr, ctl = load_run(tag + "_mr"), load_run(tag + "_ctl")
    nt = min(mr["nt_valid"], ctl["nt_valid"])
    fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for ax, r, ttl in (
        (axs[0], mr, "MR (max_level = 1)"),
        (axs[1], ctl, "control (max_level = 0)"),
    ):
        p = r["p"]
        b = r["b0"][:nt]
        wt = np.hanning(nt)[:, None]
        F = np.fft.fftshift(np.fft.fft2(wt * b))
        kb = (
            np.fft.fftshift(np.fft.fftfreq(b.shape[1], p["Lz"] / b.shape[1]))
            * 2
            * np.pi
        )
        ob = -np.fft.fftshift(np.fft.fftfreq(nt, p["dt"])) * 2 * np.pi
        P = np.log10(np.abs(F) ** 2 + 1e-30)
        vmax = P.max()
        im = ax.imshow(
            P[::-1],
            aspect="auto",
            cmap="magma",
            vmin=vmax - 8,
            vmax=vmax,
            extent=[
                kb[0] / p["k_nyq_coarse"],
                kb[-1] / p["k_nyq_coarse"],
                ob.min() / p["w_ci"],
                ob.max() / p["w_ci"],
            ],
        )
        kk_ = np.linspace(-1, 1, 300) * p["k_nyq_coarse"]
        for sgn in (+1, -1):
            ax.plot(
                kk_ / p["k_nyq_coarse"],
                sgn
                * whistler_analytic(np.abs(kk_), p["l_i"], p["w_ci"], dz=p["dz_coarse"])
                / p["w_ci"],
                color="#7fd4ff",
                lw=1.0,
                ls="--",
                alpha=0.9,
            )
        for x, c in ((-p["kfac"], "#9effa0"), (p["kfac"], "#ff9e9e")):
            ax.axvline(x, color=c, lw=0.8, alpha=0.8)
        ax.set_ylim(-30, 30)
        ax.set_xlabel(r"$k / k_{\rm Nyq}^{\rm coarse}$")
        ax.set_title(ttl, fontsize=12)
    axs[0].set_ylabel(r"$\omega/\omega_{ci}$")
    fig.colorbar(
        im, ax=axs, label=r"$\log_{10} |B_{x}+iB_{y}|^2(k,\omega)$", shrink=0.85
    )
    fig.suptitle(
        "T1.1 space-time power maps, k/k_Nc = 0.7 (level-0 view; "
        "green line = initialized column, red = mirror)"
    )
    fig.savefig(os.path.join(HERE, "t1_kw_map_mr_vs_ctl.png"), dpi=280)
    return rows


# ----------------------------------------------------------------------
# T1.2
# ----------------------------------------------------------------------
def packet_budget(r):
    """Time series of packet-perturbation-field energy fractions."""
    p = r["p"]
    nt = r["nt_valid"]
    if "b1" in r:
        bin_, zin = r["b1"][:nt], r["z1"]
        dz_in = zin[1] - zin[0]
        nz0 = r["b0"].shape[1]
        jlo, jhi = patch_idx(p, nz0)
        out_idx = np.r_[0:jlo, jhi:nz0]
        bout = r["b0"][:nt][:, out_idx]
        dz_out = r["z0"][1] - r["z0"][0]
    else:  # uniform fine reference
        nz0 = r["b0"].shape[1]
        jlo, jhi = patch_idx(p, nz0)
        bin_ = r["b0"][:nt][:, jlo:jhi]
        zin = r["z0"][jlo:jhi]
        dz_in = r["z0"][1] - r["z0"][0]
        out_idx = np.r_[0:jlo, jhi:nz0]
        bout = r["b0"][:nt][:, out_idx]
        dz_out = dz_in
    E_in = (np.abs(bin_) ** 2).sum(axis=1) * dz_in
    E_out = (np.abs(bout) ** 2).sum(axis=1) * dz_out
    E0 = E_in[0] + E_out[0]
    # +-k split inside the patch (the initialized packet is the k<0 half),
    # restricted to the packet band |k| in [1.02, 1.7] k_Nc so that any
    # residual broadband noise does not enter the reflection bookkeeping
    w = np.hanning(bin_.shape[1])
    F = np.fft.fft(w * bin_, axis=1)
    kb = np.fft.fftfreq(bin_.shape[1], dz_in) * 2 * np.pi / p["k_nyq_coarse"]
    band = (np.abs(kb) > 1.02) & (np.abs(kb) < 1.7)
    Pm = (np.abs(F[:, band & (kb < 0)]) ** 2).sum(axis=1)
    Pp = (np.abs(F[:, band & (kb > 0)]) ** 2).sum(axis=1)
    return dict(
        t=r["t"][:nt],
        E_in=E_in / E0,
        E_out=E_out / E0,
        E_abs=1.0 - (E_in + E_out) / E0,
        Ebm=Pm / Pm[0],
        Ebp=Pp / Pm[0],
    )


def group_velocity(p):
    """Numerical group velocity of the fine-grid discrete whistler at k0."""
    dzf = 0.5 * p["dz_coarse"]
    k0 = p["k_si"]
    dk = 1.0

    def om(q):
        return whistler_analytic(q, p["l_i"], p["w_ci"], dz=dzf)

    return (om(k0 + dk) - om(k0 - dk)) / (2 * dk)


def t12_analysis(results_md):
    print("\n=== T1.2 above-coarse-Nyquist packet, hyper-resistivity scan ===")
    runs = []
    for eh in ETAH_FACS:
        tag = f"eh{int(round(eh * 100)):03d}_mr"
        runs.append((eh, load_run_pair(f"t12_pk127_{tag}", f"t12_null_{tag}")))
    fine = load_run_pair("t12_pk127_fine", "t12_null_fine")

    p = runs[0][1]["p"]
    vg = group_velocity(p)
    t_seam = (0.5 * (p["patch_zhi"] - p["patch_zlo"])) / vg
    t_ret = 2 * t_seam
    print(
        f"packet k l_i = {p['k_si'] * p['l_i']:.2f} "
        f"({p['kfac']:.3f} k_Nc), vg(fine grid) = {vg / p['vA']:.1f} vA, "
        f"center-to-seam time = {t_seam / p['t_ci']:.2f} t_ci"
    )

    results_md.append(
        "\n## T1.2 -- packet between the Nyquists: seam budget vs eta_h\n"
    )
    results_md.append(
        f"Packet: k = {p['kfac']:.3f} k_Nc (k l_i = {p['k_si'] * p['l_i']:.2f}), "
        f"Gaussian envelope sigma = 8 l_i centered in the patch; no imposed "
        f"drift -- the B-only IC projects almost entirely onto the whistler "
        f"branch, which self-propagates (fine-grid vg = {vg / p['vA']:.1f} vA, "
        f"center-to-seam {t_seam / p['t_ci']:.2f} t_ci); horizon 4 t_ci at "
        f"dt = 0.005 t_ci. Fields are coherently noise-subtracted with an "
        f"identical-seed null (zero-amplitude) twin run per arm; the "
        f"no-patch uniform-fine reference bounds the subtraction error "
        f"(its energy total should stay 1). Budgets are quoted at "
        f"t = {t_seam / p['t_ci']:.1f} t_ci, when the first seam interaction "
        f"has completed; beyond ~2 t_ci the budget is overtaken by seam-"
        f"instability growth and subtraction decorrelation. Above the "
        f"coarse Nyquist NO propagating coarse mode exists at the packet "
        f"frequency, so 'outside patch' measures what the seam leaks into "
        f"resolvable coarse wavelengths.\n"
    )
    results_md.append(
        "| eta_h/eta_h* | eta_h (Ohm m^3) | in patch | outside | absorbed | "
        "band -k left | band +k (reflected) | seam noise > 0.4 B0 @ step |"
    )
    results_md.append("|---|---|---|---|---|---|---|---|")

    curves = []
    i_s = None
    for eh, r in runs:
        bud = packet_budget(r)
        i_s = min(np.searchsorted(bud["t"], t_seam), len(bud["t"]) - 1)
        note = str(r["blow_step"]) if r["blow_step"] > 0 else "never (800)"
        row = (
            eh,
            r["p"]["eta_h_si"],
            bud["E_in"][i_s],
            bud["E_out"][i_s],
            bud["E_abs"][i_s],
            bud["Ebm"][i_s],
            bud["Ebp"][i_s],
            note,
        )
        results_md.append(
            "| {:.2f} | {:.2e} | {:.3f} | {:.4f} | {:.3f} | {:.3f} | {:.4f} | {} |".format(
                *row
            )
        )
        print(
            "eta_h {:4.2f}*: @t_seam in {:.3f} out {:.4f} abs {:.3f} "
            "band-k {:.3f} band+k {:.4f}  seam>0.4B0 {}".format(eh, *row[2:])
        )
        curves.append((eh, bud))
    budf = packet_budget(fine)
    results_md.append(
        f"\nUniform-fine free-dispersal reference (no patch, same windows) at "
        f"t = {t_seam / p['t_ci']:.1f} t_ci: in {budf['E_in'][i_s]:.3f} / "
        f"outside {budf['E_out'][i_s]:.3f} / 'absorbed' "
        f"{budf['E_abs'][i_s]:.3f} (energy-total error = subtraction "
        f"systematic, ~25%), band +k floor {budf['Ebp'][i_s]:.4f}. The "
        f"packet freely exits the window: outside reaches "
        f"{budf['E_out'][-1]:.2f} by 4 t_ci, vs ~0.002 through 2 t_ci for "
        "the bare seam -- the seam transmits essentially nothing, as "
        "expected above the coarse passband."
    )
    print(
        "fine ref @t_seam: in {:.3f} out {:.3f} abs {:.3f} band+k {:.4f}".format(
            budf["E_in"][i_s], budf["E_out"][i_s], budf["E_abs"][i_s], budf["Ebp"][i_s]
        )
    )

    # ---------------- budget figure ------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))
    x = np.arange(len(curves))
    labels = [f"{eh:g}" for eh, _ in curves]
    series = [
        ("in patch", [b["E_in"][i_s] for _, b in curves], C_MR, "o-"),
        ("outside patch", [b["E_out"][i_s] for _, b in curves], C_FINE, "s-"),
        ("absorbed", [b["E_abs"][i_s] for _, b in curves], C_CTL, "D-"),
        ("band +k (reflected)", [b["Ebp"][i_s] for _, b in curves], C_EXTRA, "^-"),
    ]
    for lab, y, c, mk in series:
        axs[0].plot(x, y, mk, color=c, lw=2, ms=7, label=lab)
    axs[0].axhline(
        budf["Ebp"][i_s],
        color=C_EXTRA,
        lw=1.2,
        ls=":",
        label="band +k floor (uniform-fine ref)",
    )
    axs[0].set_xticks(x, labels)
    axs[0].set_xlabel(
        r"$\eta_h \, / \, \eta_h^{*}$"
        "   "
        r"($\eta_h^{*} = \mu_0 \omega_{ci}/k_{\rm Nc}^4$)"
    )
    axs[0].set_ylabel("fraction of initial packet energy")
    axs[0].set_title(
        f"T1.2 packet budget at t = {t_seam / p['t_ci']:.1f} t_ci "
        "(first seam contact done)"
    )
    axs[0].legend(frameon=False, fontsize=10)
    cyc = [C_MR, C_FINE, C_CTL, C_EXTRA]
    for (eh, b), c in zip(curves, cyc):
        tm = b["t"] / p["t_ci"]
        m = tm <= 2.5
        axs[1].semilogy(
            tm[m],
            np.maximum(b["Ebp"][m], 1e-6),
            lw=2,
            color=c,
            label=rf"$\eta_h = {eh:g}\,\eta_h^*$",
        )
    tm = budf["t"] / p["t_ci"]
    m = tm <= 2.5
    axs[1].semilogy(
        tm[m],
        np.maximum(budf["Ebp"][m], 1e-6),
        lw=1.5,
        ls=":",
        color=C_GRAY,
        label="uniform-fine ref (floor)",
    )
    axs[1].axvline(t_seam / p["t_ci"], color=C_GRAY, lw=1, ls="--")
    axs[1].text(
        t_seam / p["t_ci"] + 0.04,
        2e-4,
        "first seam\ncontact done",
        fontsize=9,
        color=C_GRAY,
    )
    axs[1].set_xlabel(r"$t / t_{ci}$")
    axs[1].set_ylabel("reflected-sign band power / initial band power")
    axs[1].set_title("reflected (+k) packet-band content inside the patch")
    axs[1].legend(frameon=False, fontsize=10)
    fig.savefig(os.path.join(HERE, "t1_2_budget_vs_etah.png"), dpi=280)

    # ---------------- space-time diagrams ------------------------------
    picks = [(0.0, "bare seam"), (1.0, r"$\eta_h = 1.0\,\eta_h^*$")]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.8), sharey=True)
    for ax, (eh, ttl) in zip(axs, picks):
        r = dict(runs)[eh]
        nt = r["nt_valid"]
        nz0 = r["b0"].shape[1]
        comp = np.repeat(np.abs(r["b0"][:nt]), 2, axis=1)
        jlo, jhi = patch_idx(r["p"], 2 * nz0)
        comp[:, jlo:jhi] = np.abs(r["b1"][:nt])
        im = ax.imshow(
            comp / r["p"]["amp"],
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=1.0,
            extent=[
                0,
                r["p"]["Lz"] / r["p"]["l_i"],
                0,
                r["t"][nt - 1] / r["p"]["t_ci"],
            ],
        )
        for zz in (r["p"]["patch_zlo"], r["p"]["patch_zhi"]):
            ax.axvline(zz / r["p"]["l_i"], color="#7fd4ff", lw=1.0, ls="--")
        ax.set_xlabel(r"$z / l_i$")
        ax.set_title(ttl, fontsize=12)
        ax.grid(False)
    axs[0].set_ylabel(r"$t / t_{ci}$")
    fig.colorbar(im, ax=axs, label=r"$|B_\perp| / A$", shrink=0.85)
    fig.suptitle(
        "T1.2 packet space-time (fine view inside patch, coarse "
        "outside; dashed lines = seams)"
    )
    fig.savefig(os.path.join(HERE, "t1_2_spacetime.png"), dpi=280)
    return curves, budf, t_ret, p


# ----------------------------------------------------------------------
# artifact monitors and instability growth
# ----------------------------------------------------------------------
def artifacts_table(results_md, names):
    results_md.append("\n## Artifact monitors (valid window)\n")
    results_md.append(
        "| run | steps used | max|Ez|/max|Exy| | max|Bz-B0|/A | blowup step |"
    )
    results_md.append("|---|---|---|---|---|")
    for n in names:
        r = load_run(n)
        nt = r["nt_valid"]
        a = r.get("amax1", r["amax0"])[:nt]
        ezr = a[:, 2].max() / max(a[:, :2].max(), 1e-30)
        bzr = a[:, 3].max() / r["p"]["amp"]
        results_md.append(
            f"| {n} | {nt} | {ezr:.2f} | {bzr:.1f} | "
            f"{r['blow_step'] if r['blow_step'] > 0 else '-'} |"
        )


def instability_md(results_md):
    results_md.append("\n## Seam instability (found during this battery)\n")
    results_md.append(
        "The bare coarse-fine seam is numerically unstable in this quiet "
        "beta = 1 configuration: broadband fluctuations localized at the "
        "two z-seams grow exponentially from PIC noise until the RK4 "
        "substep NaN guard trips (or, before commit-level guards, until "
        "runaway particles corrupt deposition -- the same event surfaces "
        "as a SumBoundaryJ-adjacent segfault). Growth is decomposition-"
        "independent (np = 1 and 8 identical), independent of the MR "
        "gather/deposition buffer widths and of the RK4 substep count, and "
        "its rate scales ~linearly with dt: a fixed ~0.1-0.13 % per STEP "
        "amplification of the seam mode. The unstable content lives at "
        "and above the coarse Nyquist (strongest growth measured at "
        "k ~ 1.2-1.9 k_Nc on the fine level, peaked at the seam cells).\n"
    )
    results_md.append(
        "| configuration (kfac 0.1 seed unless noted) | dt/t_ci | blowup step "
        "| t_blow (t_ci) | growth rate gamma/w_ci |"
    )
    results_md.append("|---|---|---|---|---|")
    for row in [
        ("bare seam, np=8", 0.012, "419", "5.0", "0.09"),
        ("bare seam, np=1 (serial)", 0.012, "416", "5.0", "0.09"),
        ("bare seam, kfac 0.3 seed", 0.012, "454", "5.4", "0.08"),
        (
            "max_grid_size 32 / narrow patch / dep+gather buffers 0 or 4",
            0.012,
            "421 / 428 / 363-490",
            "~4.4-5.9",
            "--",
        ),
        ("substeps 48 (vs 24)", 0.012, "431", "5.2", "--"),
        ("eta_h = 0.2 eta_h*", 0.012, "634", "7.6", "0.030"),
        ("eta_h = 1.0 eta_h*", 0.012, "765", "9.2", "0.035"),
        ("plasma_resistivity 1e-8 Ohm m", 0.012, "478", "5.7", "--"),
        ("plasma_resistivity 3e-8 Ohm m", 0.012, "STABLE >= 1300", ">= 15.6", "~0"),
        ("plasma_resistivity 1e-7 Ohm m", 0.012, "STABLE >= 1300", ">= 15.6", "~0"),
        ("bare seam", 0.006, "1265", "7.6", "0.045"),
        ("bare seam + T1.2 packet (seeds it)", 0.005, "704", "3.5", "--"),
        ("bare seam, null (no packet)", 0.005, "none in 800 (4 t_ci)", "-", "--"),
    ]:
        results_md.append("| {} | {} | {} | {} | {} |".format(*row))
    results_md.append(
        "\nT1.1 arms at dt = 0.006 with a 2 % seeded wave: kfac 0.6 and 0.8 "
        "reached NaN at steps 693 and 977; all other k survived 1000 steps "
        "but crossed seam noise = 0.4 B0 at steps ~370-630 (see T1.1 table)."
        " Stabilization bracket: plasma resistivity eta in (1e-8, 3e-8] "
        "Ohm m at dt = 0.012 t_ci, i.e. resistive damping at the coarse "
        "Nyquist gamma_eta(k_Nc) = eta k_Nc^2/mu0 of ~(1.4-4) x the bare "
        "growth rate; eta = 3e-8 costs gamma/w_ci = 0.30 (k/k_Nc)^2 of "
        "wave damping in the resolvable band. Hyper-resistivity up to "
        "1.0 eta_h* only slows the instability (~3x); dt halving halves "
        "the growth rate. Diagnostic run dirs: t1_runs/dbg_*."
    )


def instability_figure():
    """Growth of the level-1 B-field energy in the diagnostic runs."""
    cases = [
        ("dbg_serial", "bare, dt = 0.012 t_ci", C_MR, "-"),
        ("dbg_dt006_long", "bare, dt = 0.006 t_ci", C_MR, "--"),
        ("dbg_eh020", r"$\eta_h = 0.2\,\eta_h^*$, dt = 0.012", C_EXTRA, "-"),
        ("dbg_eh100", r"$\eta_h = 1.0\,\eta_h^*$, dt = 0.012", C_EXTRA, "--"),
        ("dbg_eta1e7", r"$\eta = 10^{-7}\,\Omega$m, dt = 0.012", C_FINE, "-"),
    ]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for name, lab, c, ls in cases:
        fe = os.path.join(RUNS, name, "diags/reducedfiles/field_energy.txt")
        if not os.path.exists(fe):
            continue
        a = np.loadtxt(fe, skiprows=1)
        t_ci = 2 * np.pi / 4.3970836382e8
        dE = np.maximum(a[:, 7] - a[0, 7], 1e-12)  # lev1 B energy above IC
        ax.semilogy(a[:, 1] / t_ci, dE, color=c, ls=ls, lw=2, label=lab)
    ax.set_xlabel(r"$t / t_{ci}$")
    ax.set_ylabel(r"level-1 $\int B^2/2\mu_0$ above initial (J)")
    ax.set_title("Seam instability: level-1 magnetic energy growth (kfac = 0.1 seed)")
    ax.legend(frameon=False, fontsize=10)
    fig.savefig(os.path.join(HERE, "t1_instability.png"), dpi=280)


# ----------------------------------------------------------------------
def main():
    results_md = []
    p0 = load_run("t11_k0100_ctl")["p"]
    results_md.append("# T1 whistler seam-reflection battery -- results\n")
    results_md.append(
        f"Setup: quasi-1D 2D hybrid-PIC (kinetic ions, massless electrons), "
        f"B0 = {p0['B0']} T along z, beta = {p0['beta']}, m_i = 100 m_e, "
        f"vA/c = 1e-4; coarse grid 4 x 1024 cells (dz = 0.5 l_i, Lz = 512 "
        f"l_i), refinement ratio 2, static full-x-width patch over the middle "
        f"quarter of z (z-seams at 192 and 320 l_i); 64 ppc, particle shape "
        f"1, direct deposition, {p0['substeps']} RK4 B-substeps, plasma "
        f"resistivity 1e-10 Ohm m unless noted; eta_h* = mu0 w_ci / k_Nc^4 = "
        f"{p0['eta_h_star']:.3e} Ohm m^3. IC: right-circular B perturbation, "
        f"A = 0.02 B0. Decks/driver: t1_whistler.py, t1_run_battery.py.\n"
    )
    t11_analysis(results_md)
    t12_analysis(results_md)
    instability_md(results_md)
    artifacts_table(
        results_md,
        [f"t11_k{int(round(kf * 1000)):04d}_mr" for kf in KFACS]
        + [f"t12_pk127_eh{int(round(eh * 100)):03d}_mr" for eh in ETAH_FACS],
    )
    results_md.append(
        "\n## Figures\n\n"
        "- t1_kw_map_mr_vs_ctl.png -- (k, omega) power maps, MR vs control\n"
        "- t1_Rk_curve.png -- R(k) with control floor and difference bound\n"
        "- t1_dispersion.png -- measured whistler omega(k) vs cold theory\n"
        "- t1_2_spacetime.png -- T1.2 packet space-time, bare vs eta_h = 1\n"
        "- t1_2_budget_vs_etah.png -- T1.2 budget and reflected band power\n"
        "- t1_instability.png -- seam-instability energy growth\n"
    )
    instability_figure()
    out = os.path.join(HERE, "T1_RESULTS.md")
    mode = "a" if os.path.exists(out) else "w"
    with open(out, mode) as f:
        f.write("\n".join(results_md) + "\n")
    print(f"\nwrote {out} and figures t1_*.png")


if __name__ == "__main__":
    main()
