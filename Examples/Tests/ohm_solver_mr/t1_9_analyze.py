#!/usr/bin/env python3
"""T1.9 analysis: Mach-cone half-angle, frame-flipped (moving obstacle)
design, MR vs uniform-coarse control.

Maps (t1_runs/<arm>/t1_maps.npz, from t1_910_battery.py maps2d reduction)
are shifted to the obstacle's comoving frame with an exact periodic FFT
phase shift, time-averaged over the standing window, and the two trailing
wing ridges are traced and fit in comoving coordinates zeta = z - z_obs(t)
(wing: |x| = zeta tan(theta)).  The side seams (x = +-12 l_i) are
comoving-stationary; the wing crosses them at zeta = 12/tan(theta) ~ 43 l_i,
so fine-segment (zeta in [20,40], inside patch) and coarse-segment
(zeta in [47,130], outside) fits plus the ridge offset at the crossing give
the seam-refraction test.  The seam-band artifact metric is computed in the
LAB frame (seam artifacts are lab-static; the wing sweep is common to both
arms, so the MR/ctl ratio isolates seam effects).

Windows (from params.json): t_lo = 140 l_i / v_rel (wing developed to
>= 140 l_i), t_hi = (z_start - 44 l_i)/v_rel (obstacle >= 12 l_i inside the
patch's low edge).

Writes t1_9_cone_maps.png / t1_9_ridge_fits.png and prints the verdicts.
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

Z_REF = 40.0  # obstacle position on the comoving canvas (l_i)
FULL_SEG = (20.0, 130.0)
FINE_SEG = (20.0, 40.0)
COARSE_SEG = (47.0, 130.0)
RIDGE_HW = 4.0  # ridge-trace half-window (l_i)
THETA_REF = 15.6


def load(arm):
    d = np.load(os.path.join(RUNS, arm, "t1_maps.npz"))
    p = json.load(open(os.path.join(RUNS, arm, "params.json")))
    return d, p


def coords(p, shape):
    nz, nx = shape[-2], shape[-1]
    dxy = (p["Lz"] / p["l_i"]) / nz
    z = (np.arange(nz) + 0.5) * dxy
    x = p["xlo"] / p["l_i"] + (np.arange(nx) + 0.5) * dxy
    return x, z, dxy


def windows(p):
    li, _tci = p["l_i"], p["t_ci"]
    v = p["v_flow"] / li  # l_i / s
    t_lo = 140.0 / v
    t_hi = (p["z_obs_li"] - 44.0) / v
    return t_lo, t_hi


def comoving_mean(d, p, t_lo, t_hi):
    """FFT-shift each By map so the obstacle sits at Z_REF; average."""
    t = d["t"]
    li = p["l_i"]
    v = p["v_flow"] / li
    m = np.where((t >= t_lo) & (t <= t_hi))[0]
    assert len(m) >= 8, f"only {len(m)} outputs in window"
    nz = d["by0"].shape[1]
    Lz = p["Lz"] / li
    kz = np.fft.fftfreq(nz, d=Lz / nz) * 2 * np.pi  # rad per l_i
    acc = None
    for i in m:
        z_obs = p["z_obs_li"] - v * t[i]  # ballistic obstacle position (l_i)
        s = z_obs - Z_REF  # shift so obstacle -> Z_REF
        ph = np.exp(1j * kz * s)[:, None]
        f = np.fft.ifft(np.fft.fft(d["by0"][i].astype(float), axis=0) * ph, axis=0).real
        acc = f if acc is None else acc + f
    return acc / len(m), len(m)


def lab_mean_abs(d, p, t_lo, t_hi):
    """Lab-frame time-mean |By - B0| map over the window."""
    t = d["t"]
    m = (t >= t_lo) & (t <= t_hi)
    return np.abs(d["by0"][m].astype(float) - p["B0"]).mean(axis=0)


def scan_line(dby, x, z, sgn, pol, zwin, th_grid, b_grid):
    """Find the wing extremum line.  The wings are CHIRAL (Hall handedness,
    B0 = +y): the +x wing is a rarefaction (pol = -1), the -x wing a
    compression (pol = +1); maximize pol * mean(dBy) along the line."""
    zm = (z >= zwin[0] + Z_REF) & (z <= zwin[1] + Z_REF)
    zz = z[zm]
    best = (np.nan, np.nan, -np.inf)
    for th in th_grid:
        s = np.tan(np.deg2rad(th))
        for b in b_grid:
            xx = sgn * (b + (zz - Z_REF) * s)
            ii = np.interp(xx, x, np.arange(len(x)))
            i0 = np.clip(ii.astype(int), 0, len(x) - 2)
            fr = ii - i0
            vals = dby[zm, i0] * (1 - fr) + dby[zm, i0 + 1] * fr
            sc = pol * vals.mean()
            if sc > best[2]:
                best = (th, b, sc)
    return best


def trace_ridge(dby, x, z, sgn, pol, zwin, th0, b0):
    """Per-row subpixel wing-extremum locus (crest/trough per polarity).
    A constant source-size offset moves the intercept, not the slope, so
    the fitted angle is offset-immune in the quasi-linear far field."""
    zm = np.where((z >= zwin[0] + Z_REF) & (z <= zwin[1] + Z_REF))[0]
    s = np.tan(np.deg2rad(th0))
    dxy = x[1] - x[0]
    zs, xs = [], []
    for j in zm:
        xc = sgn * (b0 + (z[j] - Z_REF) * s)
        i0 = int(round((xc - x[0]) / dxy))
        hw = int(round(RIDGE_HW / dxy))
        lo, hi = max(i0 - hw, 1), min(i0 + hw, len(x) - 2)
        if hi <= lo:
            continue
        seg = pol * dby[j, lo:hi]
        im = np.argmax(seg) + lo
        y0, y1, y2 = pol * dby[j, im - 1], pol * dby[j, im], pol * dby[j, im + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        zs.append(z[j] - Z_REF)
        xs.append(x[im] + np.clip(off, -1, 1) * dxy)
    return np.array(zs), np.array(xs)


def fit_angle(zetas, xs, sgn):
    zz, xx = zetas.copy(), sgn * xs.copy()
    for _ in range(2):
        A = np.vstack([zz, np.ones_like(zz)]).T
        coef, *_ = np.linalg.lstsq(A, xx, rcond=None)
        r = xx - A @ coef
        keep = np.abs(r) < 2.5 * r.std() if r.std() > 0 else np.ones_like(r, bool)
        if keep.sum() < 6:
            break
        zz, xx = zz[keep], xx[keep]
    A = np.vstack([zz, np.ones_like(zz)]).T
    coef, *_ = np.linalg.lstsq(A, xx, rcond=None)
    m, a = coef
    n = len(zz)
    sig = np.sqrt(np.sum((xx - A @ coef) ** 2) / max(n - 2, 1))
    m_err = sig / np.sqrt(np.sum((zz - zz.mean()) ** 2))
    return (np.rad2deg(np.arctan(m)), np.rad2deg(m_err / (1 + m**2)), m, a, n)


def analyze_arm(arm, t_lo=None, t_hi=None):
    d, p = load(arm)
    x, z, dxy = coords(p, d["by0"].shape)
    if t_lo is None:
        t_lo, t_hi = windows(p)
    tci = p["t_ci"]
    mean, nout = comoving_mean(d, p, t_lo, t_hi)
    dby = (mean - p["B0"]) / p["B0"]
    out = dict(
        arm=arm,
        nout=nout,
        window=(t_lo / tci, t_hi / tci),
        p=p,
        dby=dby,
        x=x,
        z=z,
        d=d,
        t_lo=t_lo,
        t_hi=t_hi,
    )
    th_grid = np.arange(10.0, 22.05, 0.1)
    b_grid = np.arange(-6.0, 3.01, 0.25)
    for sgn, pol, wing in [(+1, -1, "up"), (-1, +1, "dn")]:
        th0, b0, sc = scan_line(dby, x, z, sgn, pol, FULL_SEG, th_grid, b_grid)
        segs = {}
        for name, zwin in [
            ("full", FULL_SEG),
            ("fine", FINE_SEG),
            ("coarse", COARSE_SEG),
        ]:
            zs, xs = trace_ridge(dby, x, z, sgn, pol, zwin, th0, b0)
            if len(zs) < 6:
                segs[name] = None
                continue
            th, the, m, a, n = fit_angle(zs, xs, sgn)
            segs[name] = dict(theta=th, err=the, m=m, a=a, n=n, zs=zs, xs=xs)
        out[wing] = dict(scan=(th0, b0, sc), segs=segs)
    # stationarity: window halves, full-segment angle each half
    tm = 0.5 * (t_lo + t_hi)
    halves = []
    for wl, wh in [(t_lo, tm), (tm, t_hi)]:
        mh, _ = comoving_mean(d, p, wl, wh)
        dh = (mh - p["B0"]) / p["B0"]
        ths = []
        for sgn, pol in [(+1, -1), (-1, +1)]:
            th0, b0, _ = scan_line(dh, x, z, sgn, pol, COARSE_SEG, th_grid, b_grid)
            zs, xs = trace_ridge(dh, x, z, sgn, pol, COARSE_SEG, th0, b0)
            if len(zs) >= 6:
                ths.append(fit_angle(zs, xs, sgn)[0])
        halves.append(np.mean(ths) if ths else np.nan)
    out["halves"] = halves
    return out


def seam_band_metric(res):
    """LAB-frame time-mean |dBy| in the seam bands (MR/ctl-comparable)."""
    d, p = res["d"], res["p"]
    x, z, _ = coords(p, d["by0"].shape)
    amap = lab_mean_abs(d, p, res["t_lo"], res["t_hi"]) / p["B0"]
    pxl, _pzl, pzh = 12.0, 32.0, 224.0
    vals = {}
    zm = (z >= 40) & (z <= 200)
    for name, xc in [("side_up", pxl), ("side_dn", -pxl)]:
        xm = np.abs(x - xc) <= 0.5
        vals[name] = amap[np.ix_(zm, xm)].mean()
    zm = np.abs(z - pzh) <= 0.5
    xm = np.abs(x) <= pxl
    vals["down"] = amap[np.ix_(zm, xm)].mean()
    return vals


def main():
    arms = ["t19_mr", "t19_ctl"]
    if len(sys.argv) > 1:
        arms = sys.argv[1:]
    res = {}
    for a in arms:
        try:
            res[a] = analyze_arm(a)
        except Exception as e:
            print(f"[t1_9] {a}: {e}")
    print("=" * 78)
    print(
        f"{'arm':<10} {'wing':<5} {'seg':<7} {'theta(deg)':>10} {'+-':>6} {'npts':>5}"
    )
    for a, r in res.items():
        for wing in ["up", "dn"]:
            for seg in ["full", "fine", "coarse"]:
                s = r[wing]["segs"][seg]
                if s is None:
                    print(f"{a:<10} {wing:<5} {seg:<7} {'--':>10}")
                    continue
                print(
                    f"{a:<10} {wing:<5} {seg:<7} {s['theta']:10.2f} "
                    f"{s['err']:6.2f} {s['n']:5d}"
                )
        print(
            f"{a:<10} stationarity: half-window angles = "
            f"{r['halves'][0]:.2f} / {r['halves'][1]:.2f} deg "
            f"(window {r['window'][0]:.2f}-{r['window'][1]:.2f} t_ci, "
            f"{r['nout']} outputs)"
        )
    print("=" * 78)

    if "t19_mr" in res and "t19_ctl" in res:
        rm, rc = res["t19_mr"], res["t19_ctl"]

        def wingavg(r, seg):
            ths = [
                r[w]["segs"][seg]["theta"] for w in ["up", "dn"] if r[w]["segs"][seg]
            ]
            return np.mean(ths)

        th_ctl = wingavg(rc, "coarse")  # far segment = pre-registered primary
        th_mr = wingavg(rm, "coarse")
        print(
            f"full-segment wing-averaged (reported): "
            f"MR {wingavg(rm, 'full'):.2f} / ctl {wingavg(rc, 'full'):.2f} deg"
        )
        print(
            f"\nanalytic target: {THETA_REF} deg "
            f"(v_ms = {rm['p']['v_ms_va']:.4f} vA, "
            f"v_rel = {rm['p']['v_flow_va']:.4f} vA)"
        )
        print(
            f"C1 |theta_ctl - analytic| = {abs(th_ctl - THETA_REF):.2f} deg "
            f"(theta_ctl = {th_ctl:.2f}) -> "
            f"{'PASS' if abs(th_ctl - THETA_REF) <= 1.5 else 'FAIL'}"
        )
        print(
            f"C2 |theta_MR - theta_ctl| = {abs(th_mr - th_ctl):.2f} deg "
            f"(theta_MR = {th_mr:.2f}) -> "
            f"{'PASS' if abs(th_mr - th_ctl) <= 1.0 else 'FAIL'}"
        )
        for wing in ["up", "dn"]:
            sf, sc_ = rm[wing]["segs"]["fine"], rm[wing]["segs"]["coarse"]
            if not (sf and sc_):
                print(f"C3 {wing}: segment missing -> UNRESOLVED")
                continue
            dth = sf["theta"] - sc_["theta"]
            zc = 12.0 / np.tan(np.deg2rad(sc_["theta"]))
            off = (sf["a"] + sf["m"] * zc) - (sc_["a"] + sc_["m"] * zc)
            cf, cc = rc[wing]["segs"]["fine"], rc[wing]["segs"]["coarse"]
            dth_ctl = (cf["theta"] - cc["theta"]) if (cf and cc) else np.nan
            off_ctl = np.nan
            if cf and cc:
                zcc = 12.0 / np.tan(np.deg2rad(cc["theta"]))
                off_ctl = (cf["a"] + cf["m"] * zcc) - (cc["a"] + cc["m"] * zcc)
            print(
                f"C3 {wing}: theta_fine-theta_coarse = {dth:+.2f} deg "
                f"(ctl same-windows {dth_ctl:+.2f}); ridge offset at "
                f"seam crossing = {off:+.3f} l_i (ctl {off_ctl:+.3f}) -> "
                f"{'PASS' if (abs(dth) <= 2.0 and abs(off) <= 0.5) else 'FAIL'}"
            )
        sm, sc4 = seam_band_metric(rm), seam_band_metric(rc)
        for k in sm:
            ratio = sm[k] / sc4[k]
            print(
                f"C4 seam band {k}: MR {sm[k]:.4f} vs ctl {sc4[k]:.4f} "
                f"(ratio {ratio:.2f}) -> {'PASS' if ratio <= 2.0 else 'FAIL'}"
            )

    # ---------------- figures ----------------
    if res:
        n = len(res) + (1 if len(res) == 2 else 0)
        fig, axs = plt.subplots(1, n, figsize=(7.0 * n, 9), sharey=True)
        axs = np.atleast_1d(axs)
        vmax = 0.02
        for ax, (a, r) in zip(axs, res.items()):
            im = ax.pcolormesh(
                r["x"],
                r["z"] - Z_REF,
                r["dby"],
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                shading="auto",
            )
            for wing, sgn in [("up", 1), ("dn", -1)]:
                s = r[wing]["segs"]["full"]
                if s:
                    zq = np.linspace(5, 140, 40)
                    ax.plot(sgn * (s["a"] + s["m"] * zq), zq, color="k", lw=1)
            ax.axvline(+12, color="k", lw=0.8, ls=":")
            ax.axvline(-12, color="k", lw=0.8, ls=":")
            ax.set_title(
                f"{a}: comoving <dBy>/B0 "
                f"({r['nout']} outputs, t {r['window'][0]:.1f}-"
                f"{r['window'][1]:.1f} t_ci)"
            )
            ax.set_xlabel(r"$x/l_i$")
            ax.set_xlim(-50, 50)
            ax.set_ylim(-10, 150)
        if len(res) == 2:
            rm, rc = res["t19_mr"], res["t19_ctl"]
            ax = axs[-1]
            ax.pcolormesh(
                rm["x"],
                rm["z"] - Z_REF,
                rm["dby"] - rc["dby"],
                cmap="RdBu_r",
                vmin=-vmax / 2,
                vmax=vmax / 2,
                shading="auto",
            )
            ax.axvline(+12, color="k", lw=0.8, ls=":")
            ax.axvline(-12, color="k", lw=0.8, ls=":")
            ax.set_title("difference MR - ctl (half scale)")
            ax.set_xlabel(r"$x/l_i$")
            ax.set_xlim(-50, 50)
            ax.set_ylim(-10, 150)
        axs[0].set_ylabel(r"$\zeta/l_i$ (distance behind obstacle)")
        fig.colorbar(im, ax=axs, shrink=0.75, label=r"$\langle\delta B_y\rangle/B_0$")
        fig.savefig(
            os.path.join(HERE, "t1_9_cone_maps.png"), dpi=300, bbox_inches="tight"
        )
        print("wrote t1_9_cone_maps.png")

        fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
        for ax, wing, sgn in [(axs[0], "up", 1), (axs[1], "dn", -1)]:
            for a, col in [("t19_mr", "#0072B2"), ("t19_ctl", "#E69F00")]:
                if a not in res:
                    continue
                s = res[a][wing]["segs"]["full"]
                if not s:
                    continue
                ax.plot(
                    s["zs"],
                    np.abs(s["xs"]),
                    "o",
                    ms=3,
                    color=col,
                    alpha=0.5,
                    label=f"{a} ridge",
                )
                zq = np.linspace(s["zs"][0], s["zs"][-1], 20)
                ax.plot(
                    zq,
                    s["a"] + s["m"] * zq,
                    "-",
                    color=col,
                    lw=2,
                    label=f"{a} fit {s['theta']:.2f}" + r"$^\circ$",
                )
            zq = np.linspace(0, 140, 20)
            ax.plot(
                zq,
                zq * np.tan(np.deg2rad(THETA_REF)),
                "--",
                color="gray",
                lw=1.5,
                label=f"analytic {THETA_REF}" + r"$^\circ$",
            )
            ax.axhline(12, color="k", lw=0.8, ls=":")
            ax.text(2, 12.4, "side seam", fontsize=8)
            ax.set_xlabel(r"$\zeta/l_i$")
            ax.set_title(f"{wing} wing")
            ax.legend(frameon=False, fontsize=9)
            ax.grid(alpha=0.25)
        axs[0].set_ylabel(r"$|x_{\rm ridge}|/l_i$")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "t1_9_ridge_fits.png"), dpi=300)
        print("wrote t1_9_ridge_fits.png")


if __name__ == "__main__":
    sys.exit(main())
