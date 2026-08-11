#!/usr/bin/env python
"""m=4 azimuthal metrics vs time for the liftoff MR seam battery.

For each dump: on the z-midplane of the LEVEL-0 covering grid, sample rho and
Bz on circles over a range of radii, azimuthal-FFT, and report m4/m0 at the
densest ring (column-following) plus at fixed probe radii spanning the MR
patch seam. RZ runs (auto-detected) contribute the axisymmetric bulk
reference only (column radius, peak density). The external A_ext is
axisymmetric (pure m=0) so it does not enter m=4.

Usage: m4_battery.py [--out PREFIX] [--face R] [--corner R]
                     [--map-frame-us T] label rundir [label rundir ...]
Writes <rundir>/m4_metrics.npz per run and PREFIX_m4.png / PREFIX_maps.png.
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt
from scipy.interpolate import RegularGridInterpolator

yt.set_log_level(50)

RADII = np.linspace(0.06, 0.72, 45)
NTHETA = 128
THETA = np.linspace(0.0, 2.0 * np.pi, NTHETA, endpoint=False)


def complete_dumps(rundir):
    out = []
    for base in (os.path.join(rundir, "diags"), rundir):
        for pref in ("diag1", "field_diag"):
            for p in sorted(glob.glob(os.path.join(base, pref + "*"))):
                if os.path.isfile(os.path.join(p, "Level_0", "Cell_H")):
                    out.append(p)
        if out:
            break
    return sorted(set(out))


def field(cg, name):
    for ft in (("boxlib", name), ("mesh", name), name):
        try:
            return np.array(cg[ft])
        except Exception:
            continue
    raise KeyError(name)


def ring_profiles(arr_xy, xc, yc):
    """arr_xy indexed [ix,iy]; per-radius m0 (azimuthal mean) and m4 amp."""
    interp = RegularGridInterpolator(
        (xc, yc), arr_xy, bounds_error=False, fill_value=None
    )
    m0 = np.zeros(len(RADII))
    m4 = np.zeros(len(RADII))
    for i, r in enumerate(RADII):
        pts = np.column_stack([r * np.cos(THETA), r * np.sin(THETA)])
        c = np.fft.fft(interp(pts))
        m0[i] = abs(c[0]) / NTHETA
        m4[i] = 2.0 * abs(c[4]) / NTHETA
    return m0, m4


def probe_at(rvals, m0, m4, r):
    i = int(np.argmin(np.abs(rvals - r)))
    return m4[i] / (m0[i] + 1e-30)


# Front threshold: half the initial annulus charge density (fixed absolute
# threshold so the front is the OUTER edge of the imploding shell)
RHO_ANNULUS0 = 1.602176634e-19 * 1.5e20 * 0.7**2 / (0.7**2 - 0.6**2)
FRONT_THR = 0.5 * RHO_ANNULUS0


def front_radius(m0prof):
    """Largest radius where the azimuthal-mean rho exceeds FRONT_THR."""
    idx = np.where(m0prof >= FRONT_THR)[0]
    return RADII[idx[-1]] if len(idx) else np.nan


def analyze_3d(rundir, probes):
    ts, rho4, bz4, rcol, rhopk = [], [], [], [], []
    m0profs, bz0profs, rfronts = [], [], []
    probevals = {k: [] for k in probes}
    slices = []  # (t, xc, yc, rho_xy) for map frames
    for p in complete_dumps(rundir):
        try:
            ds = yt.load(p)
            dims = ds.domain_dimensions
            le, re = ds.domain_left_edge.to_value(), ds.domain_right_edge.to_value()
            dx = (re[0] - le[0]) / dims[0]
            xc = le[0] + (np.arange(dims[0]) + 0.5) * dx
            yc = le[1] + (np.arange(dims[1]) + 0.5) * dx
            cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)
            kz = int(dims[2]) // 2
            rho_xy = field(cg, "rho")[:, :, kz]
            bz_xy = field(cg, "Bz")[:, :, kz]
            if not (np.all(np.isfinite(rho_xy)) and np.all(np.isfinite(bz_xy))):
                print(f"  NON-FINITE data in {os.path.basename(p)} -- flagged")
            rm0, rm4 = ring_profiles(rho_xy, xc, yc)
            bm0, bm4 = ring_profiles(bz_xy, xc, yc)
            icol = int(np.argmax(rm0))
            ts.append(float(ds.current_time.to_value()) * 1e6)
            rho4.append(rm4[icol] / (rm0[icol] + 1e-30))
            bz4.append(bm4[icol] / (abs(bm0[icol]) + 1e-30))
            rcol.append(RADII[icol])
            rhopk.append(rm0[icol])
            m0profs.append(rm0)
            bz0profs.append(bm0)
            rfronts.append(front_radius(rm0))
            for k, r in probes.items():
                probevals[k].append(probe_at(RADII, rm0, rm4, r))
            slices.append((ts[-1], xc, yc, rho_xy))
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {e}")
    out = dict(
        t=np.array(ts),
        rho4=np.array(rho4),
        bz4=np.array(bz4),
        rcol=np.array(rcol),
        rhopk=np.array(rhopk),
        radii=RADII,
        m0prof=np.array(m0profs),
        bz0prof=np.array(bz0profs),
        rfront=np.array(rfronts),
    )
    for k in probes:
        out[f"probe_{k}"] = np.array(probevals[k])
    return out, slices


def analyze_rz(rundir):
    ts, rcol, rhopk = [], [], []
    m0profs, bz0profs, rfronts = [], [], []
    for p in complete_dumps(rundir):
        try:
            ds = yt.load(p)
            dims = ds.domain_dimensions
            le, re = ds.domain_left_edge.to_value(), ds.domain_right_edge.to_value()
            dr = (re[0] - le[0]) / dims[0]
            rc = le[0] + (np.arange(dims[0]) + 0.5) * dr
            cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)
            rho = np.squeeze(field(cg, "rho"))
            bz = np.squeeze(field(cg, "Bz"))
            prof = rho.mean(axis=1) if rho.ndim == 2 else rho
            bprof = bz.mean(axis=1) if bz.ndim == 2 else bz
            # interpolate onto the shared RADII grid for arm comparisons
            prof_i = np.interp(RADII, rc, prof)
            bprof_i = np.interp(RADII, rc, bprof)
            icol = int(np.argmax(prof))
            ts.append(float(ds.current_time.to_value()) * 1e6)
            rcol.append(rc[icol])
            rhopk.append(prof[icol])
            m0profs.append(prof_i)
            bz0profs.append(bprof_i)
            rfronts.append(front_radius(prof_i))
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {e}")
    return dict(
        t=np.array(ts),
        rcol=np.array(rcol),
        rhopk=np.array(rhopk),
        radii=RADII,
        m0prof=np.array(m0profs),
        bz0prof=np.array(bz0profs),
        rfront=np.array(rfronts),
    )


def is_rz(rundir):
    dumps = complete_dumps(rundir)
    if not dumps:
        return False
    ds = yt.load(dumps[0])
    return int(ds.domain_dimensions[2]) == 1 or str(ds.geometry).lower().startswith(
        "cyl"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="label rundir [label rundir ...]")
    ap.add_argument("--out", default="liftoff_m4")
    ap.add_argument("--face", type=float, default=0.46, help="patch face radius (m)")
    ap.add_argument(
        "--corner", type=float, default=0.65, help="patch corner radius (m)"
    )
    ap.add_argument(
        "--map-frame-us",
        type=float,
        default=None,
        help="density-map panel time (us); default = mid seam sweep",
    )
    args = ap.parse_args()
    runs = dict(zip(args.pairs[0::2], args.pairs[1::2]))
    probes = {"r035": 0.35, "face": args.face, "r055": 0.55, "corner": args.corner}

    data, rzdata, mapslices = {}, {}, {}
    for label, rundir in runs.items():
        print(f"analyzing {label}: {rundir}")
        if is_rz(rundir):
            rzdata[label] = analyze_rz(rundir)
            np.savez(os.path.join(rundir, "m4_metrics.npz"), **rzdata[label])
            n = len(rzdata[label]["t"])
            print(f"  RZ: {n} frames")
        else:
            data[label], mapslices[label] = analyze_3d(rundir, probes)
            np.savez(os.path.join(rundir, "m4_metrics.npz"), **data[label])
            d = data[label]
            if len(d["t"]):
                i = int(np.argmax(d["rho4"]))
                print(
                    f"  {len(d['t'])} frames; peak rho m4/m0={d['rho4'][i]:.3f} "
                    f"at t={d['t'][i]:.2f} us r={d['rcol'][i]:.2f}"
                )

    # seam sweep epoch from arm M if present, else first MR-ish run.
    # Front-based (outer half-max), not densest-ring: the initial densest
    # ring straddles the corner radius, so rcol-based detection fires at t=0.
    sweep = None
    for lab in ("M", "M0", "F"):
        if lab in data and len(data[lab]["t"]):
            d = data[lab]
            tin = d["t"][d["rfront"] <= args.corner]
            tout = d["t"][d["rfront"] <= args.face]
            if len(tin) and len(tout):
                sweep = (tin[0], tout[0])
            break

    colors = {"C": "C0", "F": "k", "M": "C3", "M0": "C1", "RZ": "C2"}
    fig, ax = plt.subplots(4, 1, figsize=(11, 15), sharex=True, constrained_layout=True)
    for label, d in data.items():
        c = colors.get(label)
        lw = 2.2 if label == "F" else 1.5
        ax[0].plot(d["t"], d["rho4"], "-o", ms=3, lw=lw, color=c, label=label)
        ax[1].plot(d["t"], d["bz4"], "-o", ms=3, lw=lw, color=c, label=label)
        ax[2].plot(d["t"], d["probe_face"], "-o", ms=3, lw=lw, color=c, label=label)
        ax[3].plot(d["t"], d["rcol"], "-", lw=lw, color=c, label=label)
    for label, d in rzdata.items():
        ax[3].plot(d["t"], d["rcol"], "--", lw=2.0, color=colors.get("RZ"), label=label)
    if sweep:
        for a in ax:
            a.axvspan(sweep[0], sweep[1], color="gray", alpha=0.18)
        ax[0].text(
            sweep[0],
            0.95,
            " seam sweep (corner->face crossing)",
            transform=ax[0].get_xaxis_transform(),
            fontsize=9,
            color="gray",
        )
    ax[0].set_ylabel("rho  m=4 / m=0\n(column-following)")
    ax[1].set_ylabel("Bz  m=4 / m=0")
    ax[1].set_ylim(0, 0.8)
    ax[1].text(
        0.02,
        0.9,
        "Bz m4/m0 unreliable near field reversal (m0->0); y clipped",
        transform=ax[1].transAxes,
        fontsize=8,
        color="gray",
    )
    ax[2].set_ylabel(f"rho  m=4 / m=0\nat seam face r={args.face:.3f} m")
    ax[3].set_ylabel("column radius (m)")
    ax[3].set_xlabel("t (us)")
    ax[3].axhline(args.face, color="gray", ls=":", lw=1)
    ax[3].axhline(args.corner, color="gray", ls=":", lw=1)
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    ax[0].set_title(
        "Liftoff m=4 seam battery: column-following azimuthal m=4, z-midplane"
    )
    fig.savefig(f"{args.out}_m4.png", dpi=280)
    print(f"saved {args.out}_m4.png")

    # ---- front-fidelity (m=0 channel): azimuthal-mean radial profiles ----
    # Compare each 3D arm's azimuthal-mean rho profile against RZ as the
    # front crosses the seam radius; MR arms must track RZ/uniform-fine
    # within the coarse-vs-fine spread (pre-registered criterion 5).
    allfront = {**data, **rzdata}
    if allfront:
        band = (RADII >= 0.35) & (RADII <= 0.60)
        fig3, ax3 = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
        for label, d in allfront.items():
            if not len(d["t"]):
                continue
            c = colors.get(label)
            ls = "--" if label in rzdata else "-"
            lw = 2.2 if label in ("F", "RZ") else 1.5
            ax3[0, 0].plot(d["t"], d["rfront"], ls, lw=lw, color=c, label=label)
        for rr, txt in ((0.5, "r=0.5"), (args.face, f"face {args.face:.3f}")):
            ax3[0, 0].axhline(rr, color="gray", ls=":", lw=1)
            ax3[0, 0].text(0.05, rr + 0.005, txt, fontsize=8, color="gray")
        ax3[0, 0].set_xlabel("t (us)")
        ax3[0, 0].set_ylabel("front radius (m)\n(outer half-max of rho m0)")
        ax3[0, 0].set_title("Imploding front trajectory")
        # profile snapshots at the frames nearest the RZ front crossing of
        # r=0.5 and of the patch face, plus peak compression
        ref = rzdata[list(rzdata)[0]] if rzdata else data.get("F")
        snaps = []
        if ref is not None and len(ref["t"]):
            for rtgt in (0.5, args.face):
                k = np.where(ref["rfront"] <= rtgt)[0]
                if len(k):
                    snaps.append(("cross r=%.3f" % rtgt, ref["t"][k[0]]))
            snaps.append(("peak compression", ref["t"][int(np.argmax(ref["rhopk"]))]))
        for a, (nm, tsnap) in zip((ax3[0, 1], ax3[1, 0], ax3[1, 1]), snaps):
            for label, d in allfront.items():
                if not len(d["t"]):
                    continue
                k = int(np.argmin(np.abs(d["t"] - tsnap)))
                c = colors.get(label)
                ls = "--" if label in rzdata else "-"
                lw = 2.2 if label in ("F", "RZ") else 1.5
                a.plot(RADII, d["m0prof"][k], ls, lw=lw, color=c, label=label)
            a.axvline(args.face, color="gray", ls=":", lw=1)
            a.axvline(args.corner, color="gray", ls=":", lw=1)
            a.set_xlabel("r (m)")
            a.set_ylabel("rho m0 (C/m^3)")
            a.set_title(f"{nm}: t = {tsnap:.2f} us")
        for a in ax3.flat:
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
        fig3.savefig(f"{args.out}_front.png", dpi=280)
        print(f"saved {args.out}_front.png")
        # numbers: seam-band profile deviations at the crossing snapshots
        for nm, tsnap in snaps:
            print(f"front-fidelity @ {nm} (t={tsnap:.2f} us), seam band 0.35-0.60 m:")
            for pair in (
                ("M", "RZ"),
                ("M0", "RZ"),
                ("M", "F"),
                ("M0", "F"),
                ("C", "F"),
                ("C", "RZ"),
                ("F", "RZ"),
            ):
                a_, b_ = pair
                if (
                    a_ in allfront
                    and b_ in allfront
                    and len(allfront[a_]["t"])
                    and len(allfront[b_]["t"])
                ):
                    ka = int(np.argmin(np.abs(allfront[a_]["t"] - tsnap)))
                    kb = int(np.argmin(np.abs(allfront[b_]["t"] - tsnap)))
                    dd = np.max(
                        np.abs(
                            allfront[a_]["m0prof"][ka][band]
                            - allfront[b_]["m0prof"][kb][band]
                        )
                    )
                    print(f"  max|m0prof({a_}) - m0prof({b_})| = {dd:.4g} C/m^3")

    # density maps at the sweep epoch
    if mapslices:
        tmap = args.map_frame_us
        if tmap is None:
            tmap = 0.5 * (sweep[0] + sweep[1]) if sweep else None
        if tmap is not None:
            labs = [x for x in ("C", "F", "M", "M0") if x in mapslices] or list(
                mapslices
            )
            fig2, axs = plt.subplots(
                1, len(labs), figsize=(6.5 * len(labs), 6.5), constrained_layout=True
            )
            axs = np.atleast_1d(axs)
            for a, lab in zip(axs, labs):
                frames = mapslices[lab]
                if not frames:
                    continue
                k = int(np.argmin([abs(f[0] - tmap) for f in frames]))
                t, xc, yc, rho = frames[k]
                im = a.pcolormesh(xc, yc, rho.T, cmap="viridis", shading="auto")
                a.set_title(f"{lab}  rho, t={t:.2f} us")
                a.set_aspect("equal")
                th = np.linspace(0, 2 * np.pi, 361)
                a.plot(0.8 * np.cos(th), 0.8 * np.sin(th), "w:", lw=1)
                if lab in ("M", "M0"):
                    h = args.face
                    a.plot(
                        [-h, h, h, -h, -h], [-h, -h, h, h, -h], "r-", lw=1, alpha=0.8
                    )
                fig2.colorbar(im, ax=a, shrink=0.8)
            fig2.savefig(f"{args.out}_maps.png", dpi=280)
            print(f"saved {args.out}_maps.png")


if __name__ == "__main__":
    main()
