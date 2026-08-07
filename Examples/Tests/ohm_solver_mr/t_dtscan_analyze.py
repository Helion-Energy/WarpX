#!/usr/bin/env python3
"""dt/substeps scan for the hybrid-MR fine-patch heating (whistler vs noise).

Discriminates two hypotheses for the Landau-smoke fine-patch heating excess
(MR minus max_level=0 control, +2.81 %/300 steps at 4 ppc):

  H_whistler : marginally-resolved grid-scale whistler pumping in the fine
               patch -> heating excess collapses when dt is cut 5-10x and the
               B-spectrum shows a k-localized, growing peak near the fine
               Nyquist that shrinks with dt.
  H_noise    : stochastic moment-noise (ppc-set) heating -> excess is
               dt-insensitive and the spectrum is a flat elevated broadband
               floor.

Arms (all at 4 ppc, equal PHYSICAL horizon = 300 x dt0):
  baseline  : preserved runs landau_mr_ppc4 / landau_ctl_ppc4 (300 steps)
  dt/5      : dt0/5, 1500 steps
  dt/10     : dt0/10, 3000 steps
  sub x5    : dt0, hybrid_pic_model.substeps 20 -> 100 (field substep only)

Reported per arm:
  (a) whistler CFL bookkeeping (omega dt_sub / omega dt at the fine Nyquist
      and at k dx_fine = 1, vs the RK4 imaginary-axis limit 2*sqrt(2));
  (b) patch-interior particle KE and kT_par excess (MR - control) at the
      baseline step-300 physical time, plus the full traces;
  (c) transverse |B(k)|^2 spectra along z on the fine level inside the patch
      at physical times of baseline steps 0/50/150/300;
  (d) level-0 div(B) seam-ring rms trace.

Usage:
    python t_dtscan_analyze.py [RUNS_ROOT] [OUT_DIR]
(defaults: RUNS_ROOT=runs, OUT_DIR=Examples/Tests/ohm_solver_mr next to this
script). Writes t_dtscan_heating.png, t_dtscan_spectra.png, t_dtscan_divb.png.
"""

import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt

yt.set_log_level(50)

# ---------------------------------------------------------------- constants
M_ION = 9.1093837139e-29  # kg (deck: 100 m_e)
Q_E = 1.602176634e-19
C = 299792458.0
EV = Q_E
EPS0 = 8.8541878128e-12
MU0 = 4e-7 * np.pi
B0 = 0.1  # T, uniform background Bz
N0 = 9.623617654281555e20  # m^-3
ETA = 1e-7  # Ohm m (deck resistivity)
DT0 = 5.685630111305193e-12  # s, baseline deck dt (= 1e-3 / Omega_ci)
DX_C = 9.136055311928102e-3 / 32.0  # coarse dx = dz
DX_F = DX_C / 2.0
RING_W = 4  # interior inset in coarse cells (matches t_smoke_ppc_heating)
RK4_IMAG_LIMIT = 2.0 * np.sqrt(2.0)

OMEGA_CI = Q_E * B0 / M_ION
OMEGA_PI = np.sqrt(N0 * Q_E**2 / (EPS0 * M_ION))
D_I = C / OMEGA_PI
V_A = OMEGA_CI * D_I

# arm table: (label, mr_dir, ctl_dir, dt_factor, requested substeps, color)
ARMS = (
    ("baseline", "landau_mr_ppc4", "landau_ctl_ppc4", 1.0, 20, "#2a78d6"),
    ("dt/5", "dtscan_dt5_mr", "dtscan_dt5_ctl", 5.0, 20, "#eb6834"),
    ("dt/10", "dtscan_dt10_mr", "dtscan_dt10_ctl", 10.0, 20, "#1baf7a"),
    ("sub x5", "dtscan_sub5_mr", "dtscan_sub5_ctl", 1.0, 100, "#eda100"),
)
SNAP_STEPS0 = (0, 50, 100, 150, 200, 250, 300)  # baseline-equivalent steps
SPEC_STEPS0 = (0, 50, 150, 300)


def omega_whistler(k):
    """Cold hybrid (massless-electron) parallel R-mode / whistler branch,
    omega = Omega_ci * [x^2/2 + x*sqrt(1 + x^2/4)], x = k*d_i.
    Large-k limit omega -> Omega_ci*(k d_i)^2 (the low-beta whistler)."""
    x = k * D_I
    return OMEGA_CI * (0.5 * x**2 + x * np.sqrt(1.0 + 0.25 * x**2))


def k_eff_yee(k, dx):
    """Effective wavenumber of the centered Yee difference."""
    return 2.0 / dx * np.sin(0.5 * k * dx)


def parse_substeps(run_dir):
    """Effective substep count from the run log: mode of the '2nd half'
    accepted counts over the last half of the run (x2 halves)."""
    log = os.path.join(run_dir, "run.log")
    if not os.path.exists(log):
        return None
    counts = []
    pat = re.compile(r"B-field update 2nd half: (\d+) accepted")
    with open(log) as f:
        for line in f:
            m = pat.search(line)
            if m:
                counts.append(int(m.group(1)))
    if not counts:
        return None
    tail = counts[len(counts) // 2 :]
    vals, freq = np.unique(tail, return_counts=True)
    return 2 * int(vals[np.argmax(freq)])


# ---------------------------------------------------------------- data access
def plotfiles(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, "diags", "diag1??????")))


def patch_bounds(ds):
    l1 = [g for g in ds.index.grids if g.Level == 1]
    lo = np.min([g.LeftEdge.d for g in l1], axis=0)
    hi = np.max([g.RightEdge.d for g in l1], axis=0)
    return (lo[0], hi[0]), (lo[1], hi[1])


def particles(ds):
    ad = ds.all_data()
    out = {
        "x": ad["ions", "particle_position_x"].d,
        "z": ad["ions", "particle_position_y"].d,
        "w": ad["ions", "particle_weight"].d,
    }
    for c, name in (("ux", "x"), ("uy", "y"), ("uz", "z")):
        out[c] = ad["ions", f"particle_momentum_{name}"].d / M_ION
    return out


def interior_mask(x, z, pb):
    (xlo, xhi), (zlo, zhi) = pb
    return (
        (x >= xlo + RING_W * DX_C)
        & (x < xhi - RING_W * DX_C)
        & (z >= zlo + RING_W * DX_C)
        & (z < zhi - RING_W * DX_C)
    )


def interior_stats(ds, pb):
    """Patch-interior weighted KE (J) and kT_par/kT_perp (eV)."""
    p = particles(ds)
    m = interior_mask(p["x"], p["z"], pb)
    w = p["w"][m]
    W = w.sum()
    drift = np.array([np.sum(w * p[c][m]) / W for c in ("ux", "uy", "uz")])
    var = np.array(
        [np.sum(w * (p[c][m] - d) ** 2) / W for c, d in zip(("ux", "uy", "uz"), drift)]
    )
    kT = M_ION * var / EV
    u2 = p["ux"][m] ** 2 + p["uy"][m] ** 2 + p["uz"][m] ** 2
    gamma = np.sqrt(1.0 + u2 / C**2)
    ke = np.sum(w * M_ION * C**2 * (gamma - 1.0))
    return {
        "N": int(m.sum()),
        "KE": ke,
        "kT_par": kT[2],
        "kT_perp": 0.5 * (kT[0] + kT[1]),
    }


def level_view(ds, lev):
    """Full-domain covering grid at level lev -> dict of 2D arrays + geometry."""
    dims = ds.domain_dimensions * (ds.refine_by**lev)
    cg = ds.covering_grid(lev, ds.domain_left_edge, dims)
    return cg, dims


def transverse_spectra(ds, pb, lev):
    """Row-averaged |B_t(k_z)|^2 (Hann PSD) of Bx and By over the patch
    interior at level lev. Returns k (rad/m), P_Bx, P_By."""
    cg, dims = level_view(ds, lev)
    dxl = float(ds.domain_width.d[0] / dims[0])
    (xlo, xhi), (zlo, zhi) = pb
    x0 = float(ds.domain_left_edge.d[0])
    z0 = float(ds.domain_left_edge.d[1])
    i0 = int(round((xlo + RING_W * DX_C - x0) / dxl))
    i1 = int(round((xhi - RING_W * DX_C - x0) / dxl))
    j0 = int(round((zlo + RING_W * DX_C - z0) / dxl))
    j1 = int(round((zhi - RING_W * DX_C - z0) / dxl))
    nz = j1 - j0
    win = np.hanning(nz)
    wnorm = np.sum(win**2)
    k = 2.0 * np.pi * np.fft.rfftfreq(nz, d=dxl)
    out = []
    for f in ("Bx", "By"):
        b = np.array(cg["boxlib", f])[i0:i1, j0:j1, 0]
        b = b - b.mean(axis=1, keepdims=True)
        spec = np.abs(np.fft.rfft(b * win[None, :], axis=1)) ** 2 / wnorm
        out.append(spec.mean(axis=0))
    return k, out[0], out[1]


def divb_ring(ds, pb):
    """Level-0 divB rms in the seam-ring band d in [-3,-2) coarse cells from
    the patch edge, plus deep-interior and far-outside floors."""
    cg, dims = level_view(ds, 0)
    db = np.array(cg["boxlib", "divB"])[:, :, 0]
    (xlo, xhi), (zlo, zhi) = pb
    x0 = float(ds.domain_left_edge.d[0])
    z0 = float(ds.domain_left_edge.d[1])
    xc = x0 + (np.arange(dims[0]) + 0.5) * DX_C
    zc = z0 + (np.arange(dims[1]) + 0.5) * DX_C
    X, Z = np.meshgrid(xc, zc, indexing="ij")
    d = np.maximum.reduce(
        [(xlo - X) / DX_C, (X - xhi) / DX_C, (zlo - Z) / DX_C, (Z - zhi) / DX_C]
    )
    ring = (d >= -3.0) & (d < -2.0)
    inside = d < -6.0
    outside = d > 4.0
    rms = lambda m: float(np.sqrt(np.mean(db[m] ** 2)))  # noqa: E731
    return {"ring": rms(ring), "inside": rms(inside), "outside": rms(outside)}


def interior_u_em(ds, pb, lev):
    """Mean EM fluctuation energy density (J/m^3) in the patch interior."""
    cg, dims = level_view(ds, lev)
    dxl = float(ds.domain_width.d[0] / dims[0])
    (xlo, xhi), (zlo, zhi) = pb
    x0 = float(ds.domain_left_edge.d[0])
    z0 = float(ds.domain_left_edge.d[1])
    i0 = int(round((xlo + RING_W * DX_C - x0) / dxl))
    i1 = int(round((xhi - RING_W * DX_C - x0) / dxl))
    j0 = int(round((zlo + RING_W * DX_C - z0) / dxl))
    j1 = int(round((zhi - RING_W * DX_C - z0) / dxl))
    u = np.zeros((i1 - i0, j1 - j0))
    for f in ("Ex", "Ey", "Ez"):
        u += 0.5 * EPS0 * np.array(cg["boxlib", f])[i0:i1, j0:j1, 0] ** 2
    for f, bg in (("Bx", 0.0), ("By", 0.0), ("Bz", B0)):
        u += (np.array(cg["boxlib", f])[i0:i1, j0:j1, 0] - bg) ** 2 / (2 * MU0)
    return float(u.mean())


def part_energy_trace(run_dir):
    pe = np.loadtxt(
        os.path.join(run_dir, "diags/reducedfiles/part_energy.txt"), skiprows=1
    )
    return pe[:, 1], pe[:, 2]  # time (s), total KE (J)


# ---------------------------------------------------------------- sections
def cfl_report(runs_root):
    print("=" * 78)
    print("(a) whistler CFL bookkeeping")
    print("=" * 78)
    print(
        "dispersion: cold hybrid (massless-electron) parallel R-mode/whistler\n"
        "  omega(k) = Omega_ci [ (k d_i)^2/2 + k d_i sqrt(1+(k d_i)^2/4) ]\n"
        "  (-> Omega_ci (k d_i)^2 for k d_i >> 1; the user-quoted bounded form\n"
        "   omega = k^2 d_i^2 Omega_ci/(1+k^2 d_i^2) is the ion-cyclotron\n"
        "   L-branch, which saturates at Omega_ci and is not the whistler)\n"
        f"deck: B0={B0} T, n0={N0:.4e} m^-3, m_i=100 m_e -> "
        f"Omega_ci={OMEGA_CI:.4e} rad/s, d_i={D_I:.4e} m, v_A={V_A:.4e} m/s\n"
        f"grid: dx_c={DX_C:.5e} m (d_i/{D_I / DX_C:.1f}), dx_f={DX_F:.5e} m "
        f"(d_i/{D_I / DX_F:.1f});  dt0={DT0:.4e} s = {OMEGA_CI * DT0:.2e}/Omega_ci"
    )
    ks = {
        "fine Nyquist (k=pi/dx_f)": np.pi / DX_F,
        "k dx_f = 1 (k=1/dx_f)": 1.0 / DX_F,
        "coarse Nyquist (k=pi/dx_c)": np.pi / DX_C,
    }
    print(
        f"\n{'mode':28s} {'k [rad/m]':>10s} {'k d_i':>7s} {'omega [rad/s]':>13s} "
        f"{'omega_num':>10s} {'gam_eta':>9s}"
    )
    for name, k in ks.items():
        dx = DX_F if "dx_f" in name or "fine" in name else DX_C
        w = omega_whistler(k)
        wn = omega_whistler(k_eff_yee(k, dx))
        print(
            f"{name:28s} {k:10.1f} {k * D_I:7.2f} {w:13.4e} {wn:10.3e} "
            f"{ETA / MU0 * k**2:9.2e}"
        )
    print(
        f"\nRK4 imaginary-axis stability limit: omega*dt_sub <= 2*sqrt(2) = "
        f"{RK4_IMAG_LIMIT:.3f}"
    )
    hdr = (
        f"{'arm':9s} {'dt [s]':>10s} {'sub_req':>7s} {'sub_eff':>7s} "
        f"{'dt_sub [s]':>10s} {'w*dtsub@NyqF':>12s} {'w*dtsub@kdx1':>12s} "
        f"{'w*dt@NyqF':>10s} {'flag(sub)':>10s}"
    )
    print(hdr)
    w_nyq = omega_whistler(np.pi / DX_F)
    w_kdx1 = omega_whistler(1.0 / DX_F)
    rows = {}
    for label, mr_dir, _, fac, sub_req, _c in ARMS:
        dt = DT0 / fac
        sub_eff = parse_substeps(os.path.join(runs_root, mr_dir)) or sub_req
        dt_sub = dt / sub_eff
        a, b_, c_ = w_nyq * dt_sub, w_kdx1 * dt_sub, w_nyq * dt
        flag = (
            "UNSTABLE" if a > RK4_IMAG_LIMIT else "marginal" if a > 0.5 else "resolved"
        )
        print(
            f"{label:9s} {dt:10.3e} {sub_req:7d} {sub_eff:7d} {dt_sub:10.3e} "
            f"{a:12.4f} {b_:12.5f} {c_:10.3f} {flag:>10s}"
        )
        rows[label] = dict(
            dt=dt, sub_eff=sub_eff, dt_sub=dt_sub, wdtsub_nyq=a, wdt_nyq=c_
        )
    print(
        "\nNOTE: omega*dt_sub is the B-push (RK4 substep) resolution; omega*dt is\n"
        "the ion-push / Ohm coupling resolution (J_i, rho frozen over dt). The\n"
        "'sub x5' arm changes only the former; the dt arms change both."
    )
    return rows


def heating_report(runs_root, pb, out_dir):
    print("\n" + "=" * 78)
    print("(b) patch-interior heating, MR - control, equal physical time")
    print("=" * 78)
    t_hor = 300 * DT0
    res = {}
    for label, mr_dir, ctl_dir, fac, _s, color in ARMS:
        arm = {}
        for tag, d in (("mr", mr_dir), ("ctl", ctl_dir)):
            pfs = plotfiles(os.path.join(runs_root, d))
            if len(pfs) < len(SNAP_STEPS0):
                print(f"  [{label}/{tag}] incomplete ({len(pfs)} plotfiles) - skipped")
                arm = None
                break
            lev = 1 if tag == "mr" else 0
            snaps = []
            for pf in pfs:
                ds = yt.load(pf)
                s = interior_stats(ds, pb)
                s["u_EM"] = interior_u_em(ds, pb, lev)
                snaps.append(s)
            t, e = part_energy_trace(os.path.join(runs_root, d))
            arm[tag] = {"snaps": snaps, "t": t, "E": e}
        if arm:
            res[label] = arm
    print(f"\nphysical horizon: t = {t_hor:.4e} s (baseline step 300)")
    print(
        f"{'arm':9s} {'KE_mr%':>8s} {'KE_ctl%':>8s} {'excessKE%':>9s} "
        f"{'dTpar_mr':>9s} {'dTpar_ctl':>9s} {'exc_Tpar':>9s} "
        f"{'glob_mr%':>8s} {'glob_ctl%':>9s} {'glob_exc%':>9s}"
    )
    for label in res:
        a = res[label]
        ke = {
            t: a[t]["snaps"][-1]["KE"] / a[t]["snaps"][0]["KE"] - 1
            for t in ("mr", "ctl")
        }
        dtp = {
            t: a[t]["snaps"][-1]["kT_par"] - a[t]["snaps"][0]["kT_par"]
            for t in ("mr", "ctl")
        }
        ge = {
            t: np.interp(t_hor, a[t]["t"], a[t]["E"]) / a[t]["E"][0] - 1
            for t in ("mr", "ctl")
        }
        a["sum"] = dict(
            exc_ke=ke["mr"] - ke["ctl"],
            exc_tpar=dtp["mr"] - dtp["ctl"],
            exc_glob=ge["mr"] - ge["ctl"],
        )
        print(
            f"{label:9s} {ke['mr']:8.2%} {ke['ctl']:8.2%} "
            f"{ke['mr'] - ke['ctl']:9.2%} {dtp['mr']:9.2f} {dtp['ctl']:9.2f} "
            f"{dtp['mr'] - dtp['ctl']:9.2f} {ge['mr']:8.2%} {ge['ctl']:9.2%} "
            f"{ge['mr'] - ge['ctl']:9.2%}"
        )
    print(
        "(KE% = interior particle KE change 0->t_hor; dTpar in eV; glob = "
        "whole-domain reduced part_energy)"
    )
    print(
        "\ninterior EM fluctuation energy density u_EM (J/m^3) at t_hor "
        "(MR level 1 / ctl level 0):"
    )
    print(f"{'arm':9s} {'u_EM mr':>10s} {'u_EM ctl':>10s} {'ratio':>7s}")
    for label in res:
        a = res[label]
        um = a["mr"]["snaps"][-1]["u_EM"]
        uc = a["ctl"]["snaps"][-1]["u_EM"]
        print(f"{label:9s} {um:10.3e} {uc:10.3e} {um / uc:7.2f}")

    # figure -----------------------------------------------------------------
    fig, axg = plt.subplots(2, 2, figsize=(14.5, 10))
    axs = axg.ravel()
    tsn = np.array(SNAP_STEPS0) * DT0 / t_hor
    for label, _m, _c, _f, _s, color in ARMS:
        if label not in res:
            continue
        a = res[label]
        base_w = 3.4 if label == "baseline" else 1.8
        for tag, ls, lw in (("mr", "-", None), ("ctl", "--", 1.4)):
            lw = base_w if lw is None else lw
            ket = [s["KE"] / a[tag]["snaps"][0]["KE"] - 1 for s in a[tag]["snaps"]]
            axs[0].plot(
                tsn,
                np.array(ket) * 100,
                ls,
                color=color,
                lw=lw,
                label=f"{label} {'MR' if tag == 'mr' else 'ctl'}",
            )
            uem = [s["u_EM"] for s in a[tag]["snaps"]]
            axs[3].semilogy(tsn[1:], uem[1:], ls, color=color, lw=lw)
        exc = [
            (a["mr"]["snaps"][i]["KE"] / a["mr"]["snaps"][0]["KE"])
            - (a["ctl"]["snaps"][i]["KE"] / a["ctl"]["snaps"][0]["KE"])
            for i in range(len(tsn))
        ]
        axs[1].plot(
            tsn, np.array(exc) * 100, "-o", color=color, lw=base_w, ms=4, label=label
        )
        tg = a["mr"]["t"] / t_hor
        eg = a["mr"]["E"] / a["mr"]["E"][0] - np.interp(
            a["mr"]["t"], a["ctl"]["t"], a["ctl"]["E"] / a["ctl"]["E"][0]
        )
        axs[2].plot(tg, eg * 100, "-", color=color, lw=base_w, label=label)
    axs[0].set_ylabel("patch-interior KE change (%)")
    axs[0].set_title("interior KE, MR (solid) vs control (dashed)")
    axs[0].legend(frameon=False, fontsize=8, ncol=2)
    axs[1].set_ylabel("interior KE excess, MR $-$ ctl (%)")
    axs[1].set_title("interior heating excess")
    axs[1].legend(frameon=False, fontsize=9)
    axs[2].set_ylabel("global KE excess, MR $-$ ctl (%)")
    axs[2].set_title("whole-domain excess (reduced diag)")
    axs[2].legend(frameon=False, fontsize=9)
    axs[3].set_ylabel("interior u$_{EM}$ (J/m$^3$)")
    axs[3].set_title("interior EM fluctuation energy, MR (solid) vs ctl (dashed)")
    for ax in axs:
        ax.set_xlabel("t / t$_{300}$")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlim(0, 1.02)
    fig.tight_layout()
    out = os.path.join(out_dir, "t_dtscan_heating.png")
    fig.savefig(out, dpi=270)
    print(f"wrote {out}")
    return res


def spectra_report(runs_root, pb, out_dir):
    print("\n" + "=" * 78)
    print("(c) transverse B spectra in the patch (fine level, along z)")
    print("=" * 78)
    k_nyq_f = np.pi / DX_F
    k_nyq_c = np.pi / DX_C
    spec = {}
    for label, mr_dir, ctl_dir, fac, _s, color in ARMS:
        pfs = plotfiles(os.path.join(runs_root, mr_dir))
        if len(pfs) < len(SNAP_STEPS0):
            continue
        idx = [SNAP_STEPS0.index(s) for s in SPEC_STEPS0]
        spec[label] = {}
        for i, s0 in zip(idx, SPEC_STEPS0):
            ds = yt.load(pfs[i])
            spec[label][s0] = transverse_spectra(ds, pb, 1)
    # coarse-level control reference (baseline ctl)
    ctl_ref = {}
    pfs = plotfiles(os.path.join(runs_root, ARMS[0][2]))
    for i, s0 in zip([SNAP_STEPS0.index(s) for s in SPEC_STEPS0], SPEC_STEPS0):
        ctl_ref[s0] = transverse_spectra(yt.load(pfs[i]), pb, 0)

    # band powers ------------------------------------------------------------
    def band(k, p, klo, khi):
        m = (k >= klo) & (k <= khi)
        return float(p[m].mean())

    print(
        "\nband power (Bx+By)/2, [T^2]:  near-Nyq = k in [0.75,1]k_NyqF,"
        "  mid = k in [0.2,0.5]k_NyqF,  coarse-band = k < k_NyqC"
    )
    print(
        f"{'arm':9s} {'t/t300':>7s} {'near-Nyq':>11s} {'mid':>11s} {'coarse-band':>12s}"
    )
    bands = {}
    for label in spec:
        bands[label] = {}
        for s0 in SPEC_STEPS0:
            k, px, py = spec[label][s0]
            p = 0.5 * (px + py)
            bands[label][s0] = (
                band(k, p, 0.75 * k_nyq_f, k_nyq_f),
                band(k, p, 0.2 * k_nyq_f, 0.5 * k_nyq_f),
                band(k, p, k[1], k_nyq_c),
            )
            b = bands[label][s0]
            print(f"{label:9s} {s0 / 300:7.3f} {b[0]:11.3e} {b[1]:11.3e} {b[2]:12.3e}")
    print("\nnear-Nyquist growth t50 -> t300 and dt-sensitivity at t300:")
    for label in bands:
        g = bands[label][300][0] / max(bands[label][50][0], 1e-300)
        r = bands[label][300][0] / max(bands["baseline"][300][0], 1e-300)
        print(f"  {label:9s}: growth x{g:6.2f}   near-Nyq(t300)/baseline = {r:6.3f}")

    # figure -----------------------------------------------------------------
    fig, axs = plt.subplots(2, len(SPEC_STEPS0), figsize=(19, 8.5), sharey="row")
    for col, s0 in enumerate(SPEC_STEPS0):
        for row, comp in enumerate(("Bx", "By")):
            ax = axs[row, col]
            for label, _m, _c, _f, _s, color in ARMS:
                if label not in spec:
                    continue
                k, px, py = spec[label][s0]
                lws = 3.0 if label == "baseline" else 1.4
                ax.loglog(
                    k / k_nyq_f,
                    (px, py)[row],
                    color=color,
                    lw=lws,
                    label=label if (row, col) == (0, 0) else None,
                )
            k, px, py = ctl_ref[s0]
            ax.loglog(
                k / k_nyq_f,
                (px, py)[row],
                color="#52514e",
                lw=1.2,
                ls="--",
                label="ctl coarse (noise ref)" if (row, col) == (0, 0) else None,
            )
            ax.axvline(k_nyq_c / k_nyq_f, color="#9a9992", lw=0.8, ls=":")
            ax.axvline(1.0 / (k_nyq_f * DX_F), color="#9a9992", lw=0.8, ls="-.")
            if row == 0:
                ax.set_title(f"t = {s0}/300 t$_{{300}}$")
            if row == 1:
                ax.set_xlabel("k / k$_{Nyq}^{fine}$")
            if col == 0:
                ax.set_ylabel(f"|{comp}(k)|$^2$ per mode (T$^2$)")
            ax.grid(alpha=0.2, lw=0.5, which="both")
    axs[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(
        "patch-interior transverse B spectra along z (solid: MR fine level; "
        "dashed gray: coarse control; dotted line = coarse Nyquist, "
        "dash-dot = k dx$_f$=1)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(out_dir, "t_dtscan_spectra.png")
    fig.savefig(out, dpi=270)
    print(f"wrote {out}")
    return bands


def divb_report(runs_root, pb, out_dir):
    print("\n" + "=" * 78)
    print("(d) level-0 div(B) seam-ring rms trace (band d in [-3,-2) coarse cells)")
    print("=" * 78)
    res = {}
    for label, mr_dir, _c, fac, _s, color in ARMS:
        pfs = plotfiles(os.path.join(runs_root, mr_dir))
        if len(pfs) < len(SNAP_STEPS0):
            continue
        res[label] = [divb_ring(yt.load(pf), pb) for pf in pfs]
    print(
        f"{'arm':9s}"
        + "".join(f" {f't{s}':>10s}" for s in SNAP_STEPS0)
        + f" {'inside(t300)':>12s} {'outside(t300)':>13s}"
    )
    for label in res:
        r = res[label]
        print(
            f"{label:9s}"
            + "".join(f" {ri['ring']:10.3e}" for ri in r)
            + f" {r[-1]['inside']:12.3e} {r[-1]['outside']:13.3e}"
        )
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    tsn = np.array(SNAP_STEPS0) / 300.0
    for label, _m, _c, _f, _s, color in ARMS:
        if label not in res:
            continue
        lwd = 3.4 if label == "baseline" else 1.8
        ax.semilogy(
            tsn[1:],
            [r["ring"] for r in res[label][1:]],
            "-o",
            color=color,
            lw=lwd,
            ms=4,
            label=label,
        )
    ax.set_xlabel("t / t$_{300}$")
    ax.set_ylabel("div(B) rms in seam ring (T/m)")
    ax.set_title("level-0 div(B) seam-ring trace")
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(out_dir, "t_dtscan_divb.png")
    fig.savefig(out, dpi=270)
    print(f"wrote {out}")
    return res


def main():
    runs_root = sys.argv[1] if len(sys.argv) > 1 else "runs"
    out_dir = (
        sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
    )
    ds0 = yt.load(plotfiles(os.path.join(runs_root, ARMS[0][1]))[0])
    pb = patch_bounds(ds0)
    print(
        f"patch bounds: x in [{pb[0][0]:.5e}, {pb[0][1]:.5e}], "
        f"z in [{pb[1][0]:.5e}, {pb[1][1]:.5e}]"
    )
    cfl_report(runs_root)
    heating_report(runs_root, pb, out_dir)
    spectra_report(runs_root, pb, out_dir)
    divb_report(runs_root, pb, out_dir)


if __name__ == "__main__":
    main()
