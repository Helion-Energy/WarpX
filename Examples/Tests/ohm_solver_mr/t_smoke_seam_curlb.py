#!/usr/bin/env python3
"""Hybrid-PIC MR seam audit of B and the plasma current (Q2 of the
seam/heating investigation), on the T0.3-lite uniform-drift smoke
(t_smoke_inputs_2d_uniform) with diag1.plot_raw_fields = 1.

The solver's plasma current hybrid_current_fp_plasma = curl(B)/mu0 - J_ext
is not part of the raw-field dump, but J_ext = 0 in these decks, so this
script recomputes it exactly: J = curl(B)/mu0 per level from the raw Yee
B (Bx_fp, Bz_fp; the cell-centered By comes from the plotfile field, which
is the same data for the hybrid solver).

NORMALIZATION: J from curl(B) legitimately jumps ~2x across a resolution
boundary from the FD spacing alone, so every seam statement is made on J
normalized by the SAME level's interior rms noise floor far from the seam
(fine: > 6 coarse cells inside the patch; coarse: > 6 coarse cells
outside).

Modes:
    python t_smoke_seam_curlb.py audit RUN_DIR [OUT_PREFIX]
        maps, face/corner lineouts and ring profiles at steps 50/150/300,
        per-side coherence table, div(B)-audit correlation (run.log).
    python t_smoke_seam_curlb.py ab BASE_DIR SB4_DIR HALF_DIR [OUT_PREFIX]
        A/B of the step-150 ring profiles: mr_restrict_setback 2 vs 4 and
        mr_restrict_cadence substep vs half_step; peak localization.
"""

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt

yt.set_log_level(50)

MU0 = 4e-7 * np.pi
FLOOR_D = 6.0  # |signed distance| (coarse cells) beyond which "interior"
BINS = np.arange(-10.0, 10.01, 0.5)
BINC = 0.5 * (BINS[:-1] + BINS[1:])
STEPS = (50, 150, 300)

FINE_RE = re.compile(
    r"fine lev (\d+): valid raw (\S+) rel (\S+) \| ghost ring raw (\S+) rel (\S+)"
    r" \| ghost band raw (\S+) rel (\S+)"
)
CRSE_RE = re.compile(
    r"crse lev (\d+): interior raw (\S+) rel (\S+) \| seam ring raw (\S+) rel (\S+)"
    r" \| exterior raw (\S+) rel (\S+)"
)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def raw_stag(ds, field, lev, nodal_x, nodal_z, hi_comp):
    """Assemble a level-wide staggered array from a raw field.
    nodal_x/z flag the staggered (node-centered) directions; hi_comp is the
    yt nodal-data component holding the high-side node of a cell."""
    r = int(ds.refine_by) ** lev
    nx = int(ds.domain_dimensions[0]) * r
    nz = int(ds.domain_dimensions[1]) * r
    out = np.full((nx + int(nodal_x), nz + int(nodal_z)), np.nan)
    dx = (ds.domain_width.d / ds.domain_dimensions / r)[:2]
    for g in ds.index.grids:
        if g.Level != lev:
            continue
        a = np.array(g[("raw", field)])
        i0 = int(round((g.LeftEdge.d[0] - ds.domain_left_edge.d[0]) / dx[0]))
        k0 = int(round((g.LeftEdge.d[1] - ds.domain_left_edge.d[1]) / dx[1]))
        snx, snz = a.shape[0], a.shape[1]
        out[i0 : i0 + snx, k0 : k0 + snz] = a[:, :, 0, 0]
        if nodal_x:
            out[i0 + snx, k0 : k0 + snz] = a[-1, :, 0, hi_comp]
        if nodal_z:
            out[i0 : i0 + snx, k0 + snz] = a[:, -1, 0, hi_comp]
    return out


def cc_level(ds, field, lev):
    """Assemble a level-wide cell-centered array from the plotfile field."""
    r = int(ds.refine_by) ** lev
    nx = int(ds.domain_dimensions[0]) * r
    nz = int(ds.domain_dimensions[1]) * r
    out = np.full((nx, nz), np.nan)
    dx = (ds.domain_width.d / ds.domain_dimensions / r)[:2]
    for g in ds.index.grids:
        if g.Level != lev:
            continue
        a = np.array(g[("boxlib", field)])[:, :, 0]
        i0 = int(round((g.LeftEdge.d[0] - ds.domain_left_edge.d[0]) / dx[0]))
        k0 = int(round((g.LeftEdge.d[1] - ds.domain_left_edge.d[1]) / dx[1]))
        out[i0 : i0 + a.shape[0], k0 : k0 + a.shape[1]] = a
    return out


def load_state(pf):
    """All per-level arrays needed by the audit, plus geometry."""
    ds = yt.load(pf)
    nlev = ds.index.max_level + 1
    st = {"ds": ds, "nlev": nlev}
    l1 = [g for g in ds.index.grids if g.Level == 1]
    lo = np.min([g.LeftEdge.d for g in l1], axis=0)
    hi = np.max([g.RightEdge.d for g in l1], axis=0)
    st["pxlo"], st["pxhi"], st["pzlo"], st["pzhi"] = lo[0], hi[0], lo[1], hi[1]
    st["dxc"], st["dzc"] = (ds.domain_width.d / ds.domain_dimensions)[:2]
    st["lev"] = []
    for lev in range(nlev):
        r = int(ds.refine_by) ** lev
        dx, dz = st["dxc"] / r, st["dzc"] / r
        Bx = raw_stag(ds, "Bx_fp", lev, True, False, 1)  # (nx+1, nz)
        Bz = raw_stag(ds, "Bz_fp", lev, False, True, 1)  # (nx, nz+1)
        By = cc_level(ds, "By", lev)  # (nx, nz)
        nx, nz = By.shape
        # J = curl(B)/mu0 on the natural staggered locations
        Jx = np.full((nx, nz + 1), np.nan)
        Jx[:, 1:nz] = -(By[:, 1:] - By[:, :-1]) / dz / MU0
        Jz = np.full((nx + 1, nz), np.nan)
        Jz[1:nx, :] = (By[1:, :] - By[:-1, :]) / dx / MU0
        A = np.full((nx + 1, nz + 1), np.nan)
        A[:, 1:nz] = (Bx[:, 1:] - Bx[:, :-1]) / dz
        Bm = np.full((nx + 1, nz + 1), np.nan)
        Bm[1:nx, :] = (Bz[1:, :] - Bz[:-1, :]) / dx
        Jy = (A - Bm) / MU0
        exc = raw_stag(ds, "Ex_fp", lev, False, False, 0)
        eyc = raw_stag(ds, "Ey_fp", lev, False, False, 0)
        ezc = raw_stag(ds, "Ez_fp", lev, False, False, 0)
        st["lev"].append(
            {
                "dx": dx,
                "dz": dz,
                "Bx": Bx,
                "Bz": Bz,
                "By": By,
                "Jx": Jx,
                "Jy": Jy,
                "Jz": Jz,
                "divB": cc_level(ds, "divB", lev),
                "Emag": np.sqrt(exc**2 + eyc**2 + ezc**2),
            }
        )
    return st


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def sdf_terms(st, lev, shape, ox, oz):
    """Signed distance (coarse cells, <0 inside patch) and active side id
    for every element of an array with staggering offsets (ox, oz)."""
    L = st["lev"][lev]
    ds = st["ds"]
    X = ds.domain_left_edge.d[0] + (np.arange(shape[0])[:, None] + ox) * L["dx"]
    Z = ds.domain_left_edge.d[1] + (np.arange(shape[1])[None, :] + oz) * L["dz"]
    t = np.stack(
        [
            (st["pxlo"] - X) / st["dxc"] + 0 * Z,
            (X - st["pxhi"]) / st["dxc"] + 0 * Z,
            (st["pzlo"] - Z) / st["dzc"] + 0 * X,
            (Z - st["pzhi"]) / st["dzc"] + 0 * X,
        ]
    )
    return t.max(axis=0), t.argmax(axis=0)  # sdf, side (0:xlo 1:xhi 2:zlo 3:zhi)


OFFS = {
    "Bx": (0.0, 0.5),
    "Bz": (0.5, 0.0),
    "By": (0.5, 0.5),
    "Jx": (0.5, 0.0),
    "Jy": (0.0, 0.0),
    "Jz": (0.0, 0.5),
    "divB": (0.5, 0.5),
    "Emag": (0.5, 0.5),
}


def ring_profile(st, lev, field):
    """rms of a field binned by signed distance to the patch edge."""
    a = st["lev"][lev][field]
    sdf, _ = sdf_terms(st, lev, a.shape, *OFFS[field])
    prof = np.full(BINC.size, np.nan)
    for b in range(BINC.size):
        m = (sdf >= BINS[b]) & (sdf < BINS[b + 1]) & np.isfinite(a)
        if m.sum() > 4:
            prof[b] = np.sqrt(np.mean(a[m] ** 2))
    return prof


def floor_rms(st, lev, field):
    """Interior rms noise floor of the level: fine = deep inside the patch,
    coarse = far outside it."""
    a = st["lev"][lev][field]
    sdf, _ = sdf_terms(st, lev, a.shape, *OFFS[field])
    m = (sdf < -FLOOR_D) if lev > 0 else (sdf > FLOOR_D)
    m &= np.isfinite(a)
    return np.sqrt(np.mean(a[m] ** 2))


def j_quad_profile(st, lev, normalized=True):
    """Quadrature ring profile over the three J components; optionally each
    component is normalized by its own level floor first."""
    tot = np.zeros(BINC.size)
    any_ok = np.zeros(BINC.size, dtype=bool)
    for f in ("Jx", "Jy", "Jz"):
        p = ring_profile(st, lev, f)
        if normalized:
            p = p / floor_rms(st, lev, f)
        any_ok |= np.isfinite(p)
        tot += np.nan_to_num(p) ** 2
    return np.where(any_ok, np.sqrt(tot / 3.0), np.nan)


def coherence_table(st, lev, field, lo, hi):
    """Per-side mean/rms of a field over the ring band sdf in [lo, hi)."""
    a = st["lev"][lev][field]
    sdf, side = sdf_terms(st, lev, a.shape, *OFFS[field])
    rows = []
    for s, name in enumerate(("x-lo", "x-hi", "z-lo", "z-hi")):
        m = (sdf >= lo) & (sdf < hi) & (side == s) & np.isfinite(a)
        if m.sum() < 4:
            continue
        v = a[m]
        rows.append(
            (
                name,
                v.size,
                v.mean(),
                np.sqrt(np.mean(v**2)),
                np.abs(v.mean()) / max(np.sqrt(np.mean(v**2)), 1e-300),
            )
        )
    return rows


# --------------------------------------------------------------------------
# audit mode
# --------------------------------------------------------------------------
def audit(run_dir, prefix):
    states = {}
    for s in STEPS:
        pf = os.path.join(run_dir, "diags", f"diag1{s:06d}")
        if os.path.isdir(pf):
            states[s] = load_state(pf)
    if not states:
        raise SystemExit(f"no plotfiles for steps {STEPS} under {run_dir}")
    s_last = max(states)
    st = states[s_last]

    # ---- printed summary: floors and normalized ring peaks ---------------
    print(
        "=== J = curl(B)/mu0 noise floors and seam peaks "
        "(J normalized per level by its own interior rms floor) ==="
    )
    for s, stt in states.items():
        for lev in range(stt["nlev"]):
            floors = {f: floor_rms(stt, lev, f) for f in ("Jx", "Jy", "Jz", "Bx")}
            prof = j_quad_profile(stt, lev)
            band = np.isfinite(prof) & (np.abs(BINC) <= 5)
            pk = np.nanmax(prof[band])
            pk_d = BINC[band][np.nanargmax(prof[band])]
            print(
                f"step {s:3d} lev {lev}: floor rms Jx/Jy/Jz ="
                f" {floors['Jx']:.3e}/{floors['Jy']:.3e}/{floors['Jz']:.3e} A/m^2,"
                f" Bx floor {floors['Bx']:.3e} T |"
                f" J/floor peak {pk:.2f} at d = {pk_d:+.2f} crse cells"
            )

    # ---- maps at the last step -------------------------------------------
    ds = st["ds"]
    ext = [
        ds.domain_left_edge.d[1] * 1e3,
        ds.domain_right_edge.d[1] * 1e3,
        ds.domain_left_edge.d[0] * 1e3,
        ds.domain_right_edge.d[0] * 1e3,
    ]
    fig, axs = plt.subplots(3, 2, figsize=(16, 9), sharex=True, sharey=True)
    for col, lev in enumerate((0, 1)):
        for row, f in enumerate(("Bx", "Jy", "divB")):
            a = st["lev"][lev][f]
            v = np.nanpercentile(np.abs(a), 99.5)
            im = axs[row, col].imshow(
                a,
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                vmin=-v,
                vmax=v,
                extent=ext,
            )
            fig.colorbar(im, ax=axs[row, col], pad=0.01)
            axs[row, col].set_title(f"{f} lev {lev}", fontsize=10)
            for dcell, color in ((0, "#222222"), (-2, "#2e9e73")):
                axs[row, col].add_patch(
                    plt.Rectangle(
                        (
                            (st["pzlo"] + dcell * st["dzc"]) * 1e3,
                            (st["pxlo"] + dcell * st["dxc"]) * 1e3,
                        ),
                        (st["pzhi"] - st["pzlo"] - 2 * dcell * st["dzc"]) * 1e3,
                        (st["pxhi"] - st["pxlo"] - 2 * dcell * st["dxc"]) * 1e3,
                        fill=False,
                        ls="--",
                        lw=0.9,
                        ec=color,
                    )
                )
    for ax in axs[-1]:
        ax.set_xlabel("z (mm)")
    for ax in axs[:, 0]:
        ax.set_ylabel("x (mm)")
    fig.suptitle(
        f"{run_dir} step {s_last}: Bx, Jy = (curl B)_y/mu0, divB "
        "(black = patch edge, green = setback ring)"
    )
    fig.tight_layout()
    fig.savefig(f"{prefix}_maps.png", dpi=270)
    print(f"wrote {prefix}_maps.png")

    # ---- face / corner lineouts ------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))
    cuts_x = (
        ("face", 0.5 * (st["pzlo"] + st["pzhi"])),
        ("corner", st["pzlo"] + 1.0 * st["dzc"]),
    )
    cuts_z = (
        ("face", 0.5 * (st["pxlo"] + st["pxhi"])),
        ("corner", st["pxlo"] + 1.0 * st["dxc"]),
    )
    for row, (f, ylab) in enumerate((("Bx", "Bx (T)"), ("Jy", "Jy / floor"))):
        for lev, lw in ((0, 1.8), (1, 1.1)):
            L = st["lev"][lev]
            a = L[f] / (floor_rms(st, lev, "Jy") if f == "Jy" else 1.0)
            ox, oz = OFFS[f]
            xs = ds.domain_left_edge.d[0] + (np.arange(a.shape[0]) + ox) * L["dx"]
            zs = ds.domain_left_edge.d[1] + (np.arange(a.shape[1]) + oz) * L["dz"]
            for (lbl, zcut), color in zip(cuts_x, ("#3d65f5", "#c95d20")):
                k = int(round((zcut - ds.domain_left_edge.d[1]) / L["dz"] - oz))
                axs[row, 0].plot(
                    (xs - st["pxlo"]) / st["dxc"],
                    a[:, k],
                    lw=lw,
                    color=color,
                    alpha=0.9 if lev else 1.0,
                    label=f"{lbl} cut, lev {lev}",
                )
            for (lbl, xcut), color in zip(cuts_z, ("#3d65f5", "#c95d20")):
                i = int(round((xcut - ds.domain_left_edge.d[0]) / L["dx"] - ox))
                axs[row, 1].plot(
                    (zs - st["pzlo"]) / st["dzc"],
                    a[i, :],
                    lw=lw,
                    color=color,
                    alpha=0.9 if lev else 1.0,
                    label=f"{lbl} cut, lev {lev}",
                )
        axs[row, 0].set_ylabel(ylab)
        for c, (lo_lab, hi_edge) in enumerate(
            (
                (0, (st["pxhi"] - st["pxlo"]) / st["dxc"]),
                (0, (st["pzhi"] - st["pzlo"]) / st["dzc"]),
            )
        ):
            for e in (0, hi_edge):
                axs[row, c].axvline(e, color="#999999", lw=0.8, ls="--")
                axs[row, c].axvline(
                    e + (2 if e == 0 else -2), color="#2e9e73", lw=0.7, ls=":"
                )
            axs[row, c].grid(alpha=0.2, lw=0.4)
            axs[row, c].legend(frameon=False, fontsize=8, ncol=2)
    axs[1, 0].set_xlabel("x (coarse cells from patch x-lo edge)")
    axs[1, 1].set_xlabel("z (coarse cells from patch z-lo edge)")
    fig.suptitle(
        f"{run_dir} step {s_last}: seam lineouts "
        "(dashed grey = patch edge, dotted green = setback ring)"
    )
    fig.tight_layout()
    fig.savefig(f"{prefix}_lineouts.png", dpi=270)
    print(f"wrote {prefix}_lineouts.png")

    # ---- ring profiles for all steps --------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.5), sharex=True)
    colors = {50: "#8fb0ff", 150: "#3d65f5", 300: "#1a2f80"}
    colf = {50: "#f0a878", 150: "#c95d20", 300: "#7a3208"}
    for s, stt in states.items():
        axs[0].plot(
            BINC,
            j_quad_profile(stt, 0),
            color=colors[s],
            lw=1.8,
            label=f"lev 0, step {s}",
        )
        axs[0].plot(
            BINC,
            j_quad_profile(stt, 1),
            color=colf[s],
            lw=1.2,
            label=f"lev 1, step {s}",
        )
        for lev, cmap in ((0, colors), (1, colf)):
            axs[1].plot(
                BINC,
                ring_profile(stt, lev, "Bx") / floor_rms(stt, lev, "Bx"),
                color=cmap[s],
                lw=1.8 if lev == 0 else 1.2,
                label=f"lev {lev}, step {s}",
            )
            p = ring_profile(stt, lev, "divB")
            axs[2].semilogy(
                BINC,
                np.where(p > 0, p, np.nan),
                color=cmap[s],
                lw=1.8 if lev == 0 else 1.2,
                label=f"lev {lev}, step {s}",
            )
    for ax, t in zip(
        axs,
        (
            "J=curl(B)/mu0, quadrature over components,\n"
            "each normalized by its level floor",
            "rms Bx / level floor",
            "rms divB (T/m)",
        ),
    ):
        ax.axvline(0, color="#999999", lw=0.8, ls="--")
        ax.axvline(-2, color="#2e9e73", lw=0.8, ls=":")
        ax.set_xlabel("signed distance to patch edge (coarse cells, <0 inside)")
        ax.set_title(t, fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(f"{prefix}_ring_profiles.png", dpi=270)
    print(f"wrote {prefix}_ring_profiles.png")

    # ---- coherence: is the ring signal a signed current sheet? ------------
    print(
        "=== per-side coherence over the setback ring band d in [-3, -1.5) "
        f"(coarse level, step {s_last}) ==="
    )
    print(
        f"{'comp':5s} {'side':5s} {'N':>6s} {'mean':>11s} {'rms':>11s} "
        f"{'|mean|/rms':>10s}"
    )
    for f in ("Jx", "Jy", "Jz"):
        for row in coherence_table(st, 0, f, -3.0, -1.5):
            print(
                f"{f:5s} {row[0]:5s} {row[1]:6d} {row[2]:11.3e} "
                f"{row[3]:11.3e} {row[4]:10.3f}"
            )
    print("=== same, patch-edge band d in [-0.5, 0.5) (fine level) ===")
    for f in ("Jx", "Jy", "Jz"):
        for row in coherence_table(st, 1, f, -0.5, 0.5):
            print(
                f"{f:5s} {row[0]:5s} {row[1]:6d} {row[2]:11.3e} "
                f"{row[3]:11.3e} {row[4]:10.3f}"
            )

    # ---- along-ring lineout of Jy on the x-lo side ------------------------
    fig, ax = plt.subplots(figsize=(13, 5))
    L = st["lev"][0]
    i_ring = int(round((st["pxlo"] - ds.domain_left_edge.d[0]) / L["dx"])) - 2
    zs = ds.domain_left_edge.d[1] + np.arange(L["Jy"].shape[1]) * L["dz"]
    m = (zs > st["pzlo"]) & (zs < st["pzhi"])
    ax.plot(
        (zs[m] - st["pzlo"]) / st["dzc"],
        L["Jy"][i_ring, m],
        lw=1.2,
        color="#3d65f5",
        label="Jy along the x-lo setback ring (2 crse cells inside)",
    )
    ax.axhline(0, color="#555555", lw=0.8)
    fl = floor_rms(st, 0, "Jy")
    ax.axhline(fl, color="#c95d20", lw=0.9, ls=":", label="+/- lev-0 floor rms")
    ax.axhline(-fl, color="#c95d20", lw=0.9, ls=":")
    ax.set_xlabel("z along the ring (coarse cells from patch z-lo corner)")
    ax.set_ylabel("Jy (A/m^2)")
    ax.set_title(f"{run_dir} step {s_last}: coherence check along the ring")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(f"{prefix}_ring_lineout.png", dpi=270)
    print(f"wrote {prefix}_ring_lineout.png")

    # ---- div(B) audit correlation -----------------------------------------
    log = os.path.join(run_dir, "run.log")
    if os.path.isfile(log):
        crse = []
        with open(log) as fh:
            for line in fh:
                mm = CRSE_RE.search(line)
                if mm:
                    crse.append([float(x) for x in mm.groups()[1:]])
        if crse:
            crse = np.array(crse)
            print(
                "=== div(B) audit (coarse seam ring, rel to max|B|/dx) vs "
                "normalized J peak ==="
            )
            for s in states:
                idx = min(s - 1, len(crse) - 1)
                prof = j_quad_profile(states[s], 0)
                band = np.isfinite(prof) & (np.abs(BINC) <= 5)
                print(
                    f"  step {s:3d}: divB seam rel {crse[idx, 3]:.3e}, "
                    f"coarse J/floor peak {np.nanmax(prof[band]):.2f}"
                )


# --------------------------------------------------------------------------
# ab mode
# --------------------------------------------------------------------------
def ab(dirs, labels, prefix, step=150):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)
    colors = ("#3d65f5", "#c95d20", "#2e9e73")
    print(f"=== A/B ring profiles at step {step} (J quadrature / level floors) ===")
    for d, lab, color in zip(dirs, labels, colors):
        st = load_state(os.path.join(d, "diags", f"diag1{step:06d}"))
        for ax, lev, lw in ((axs[0], 0, 1.8), (axs[1], 1, 1.4)):
            prof = j_quad_profile(st, lev)
            ax.plot(BINC, prof, color=color, lw=lw, label=lab)
            band = np.isfinite(prof) & (np.abs(BINC) <= 6)
            pk, pd = np.nanmax(prof[band]), BINC[band][np.nanargmax(prof[band])]
            print(f"  {lab:28s} lev {lev}: peak {pk:5.2f} at d = {pd:+.2f}")
        # the localized discriminators: where is the divB kink ring, and
        # which coarse band carries the restricted-noise plateau?
        p = ring_profile(st, 0, "divB")
        hot = [
            f"{BINC[b]:+.2f}"
            for b in range(BINC.size)
            if np.isfinite(p[b]) and p[b] > 1e-6
        ]
        print(f"  {lab:28s} coarse divB kink ring at d = {hot}")
        a = st["lev"][0]["Jy"]
        sdf, _ = sdf_terms(st, 0, a.shape, *OFFS["Jy"])
        bands = (
            ("deep<-6", -99, -6),
            ("[-5,-4)", -5, -4),
            ("[-4,-3)", -4, -3),
            ("[-3,-2)", -3, -2),
            ("[-2,-1)", -2, -1),
            ("out>6", 6, 99),
        )
        txt = "  ".join(
            f"{n} {np.sqrt(np.mean(a[(sdf >= lo) & (sdf < hi) & np.isfinite(a)] ** 2)):.2e}"
            for n, lo, hi in bands
        )
        print(f"  {lab:28s} coarse Jy rms by band: {txt}")
    for ax, t in zip(axs, ("coarse level", "fine level")):
        ax.axvline(0, color="#999999", lw=0.8, ls="--")
        for d, c in ((-2, "#3d65f5"), (-4, "#c95d20")):
            ax.axvline(d, color=c, lw=0.7, ls=":")
        ax.set_xlabel("signed distance to patch edge (coarse cells, <0 inside)")
        ax.set_ylabel("J / floor (quadrature)")
        ax.set_title(t)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle(
        "A/B localization: does the seam J move with the setback ring?"
        " (dotted blue = d=-2, dotted orange = d=-4)"
    )
    fig.tight_layout()
    fig.savefig(f"{prefix}_ab_profiles.png", dpi=270)
    print(f"wrote {prefix}_ab_profiles.png")


def main():
    mode = sys.argv[1]
    if mode == "audit":
        run_dir = sys.argv[2]
        prefix = sys.argv[3] if len(sys.argv) > 3 else "t_smoke_seam"
        audit(run_dir, prefix)
    elif mode == "ab":
        dirs = sys.argv[2:5]
        prefix = sys.argv[5] if len(sys.argv) > 5 else "t_smoke_seam"
        ab(
            dirs,
            ("setback=2, substep (base)", "setback=4, substep", "setback=2, half_step"),
            prefix,
        )
    else:
        raise SystemExit("mode must be 'audit' or 'ab'")


if __name__ == "__main__":
    main()
