#!/usr/bin/env python3
"""Hybrid-PIC MR heating audit (Q1 of the seam/heating investigation).

Three parts, all driven from the Landau-damping MR smoke
(t_smoke_inputs_2d) and its single-level control:

(a) ppc census: from the step-0 plotfile, count macroparticles per cell
    per level and report the per-level weight statistics. Answers whether
    the fine level is sampled at the same ppc PER CELL as the coarse
    level or at constant per-volume sampling (per-cell dilution).
(b) region-resolved heating: from the particles at step 0 vs the final
    step, weighted mean drift, T_parallel (z, along B0), T_perp (x, y)
    for the patch interior / seam ring (RING_W coarse cells inside the
    patch edge) / outside, plus a bulk-vs-thermal split of the kinetic
    energy using per-coarse-cell mean velocities, plus the field
    fluctuation energy density per region.
(c) ppc scaling: heating rates from the reduced particle-energy
    diagnostics of the (MR, control) pairs at baseline and 4x ppc; the
    MR-minus-control excess vs ppc gives the noise-vs-systematic verdict.

Usage:
    python t_smoke_ppc_heating.py MR_DIR CTL_DIR [MR4X_DIR CTL4X_DIR] [OUT_PREFIX]

Each run dir must hold diags/diag1?????? plotfiles written with
diag1.write_species=1 (first and last used) and
diags/reducedfiles/part_energy.txt. Writes <OUT_PREFIX>_ppc_heating.png
(default prefix "t_smoke").
"""

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt

yt.set_log_level(50)

M_ION = 9.1093837139e-29  # kg, matches the deck
C = 299792458.0
EV = 1.602176634e-19
EPS0 = 8.8541878128e-12
MU0 = 4e-7 * np.pi
B0 = np.array([0.0, 0.0, 0.1])  # uniform background field of the deck
RING_W = 4  # seam-ring width in coarse cells, inside the patch edge


def plotfiles(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, "diags", "diag1??????")))


def patch_bounds(ds):
    """[xlo, xhi], [zlo, zhi] of the level-1 union (2D: yt axis 1 = z)."""
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
        out[c] = ad["ions", f"particle_momentum_{name}"].d / M_ION  # gamma*v (m/s)
    return out


def make_regions(x, z, pb, dxc, dzc):
    (xlo, xhi), (zlo, zhi) = pb
    in_patch = (x >= xlo) & (x < xhi) & (z >= zlo) & (z < zhi)
    interior = (
        (x >= xlo + RING_W * dxc)
        & (x < xhi - RING_W * dxc)
        & (z >= zlo + RING_W * dzc)
        & (z < zhi - RING_W * dzc)
    )
    ring = in_patch & ~interior
    outside = ~in_patch
    return {"interior": interior, "ring": ring, "outside": outside}


def wstats(p, m):
    """Weighted drift, temperatures and KE split for the particle subset m."""
    w = p["w"][m]
    W = w.sum()
    drift = np.array([np.sum(w * p[c][m]) / W for c in ("ux", "uy", "uz")])
    var = np.array(
        [np.sum(w * (p[c][m] - d) ** 2) / W for c, d in zip(("ux", "uy", "uz"), drift)]
    )
    kT = M_ION * var / EV  # eV per degree of freedom
    u2 = p["ux"][m] ** 2 + p["uy"][m] ** 2 + p["uz"][m] ** 2
    gamma = np.sqrt(1.0 + u2 / C**2)
    ke = np.sum(w * M_ION * C**2 * (gamma - 1.0))
    return {
        "N": int(m.sum()),
        "W": W,
        "drift": drift,
        "kT_par": kT[2],
        "kT_perp": 0.5 * (kT[0] + kT[1]),
        "KE": ke,
    }


def bulk_thermal_split(p, m, ds):
    """KE split into per-coarse-cell bulk (cell-mean drift) and thermal."""
    dxc, dzc = (ds.domain_width.d / ds.domain_dimensions)[:2]
    nx, nz = int(ds.domain_dimensions[0]), int(ds.domain_dimensions[1])
    ix = np.clip(((p["x"][m] - ds.domain_left_edge.d[0]) / dxc).astype(int), 0, nx - 1)
    iz = np.clip(((p["z"][m] - ds.domain_left_edge.d[1]) / dzc).astype(int), 0, nz - 1)
    cell = ix * nz + iz
    w = p["w"][m]
    Wc = np.bincount(cell, weights=w, minlength=nx * nz)
    bulk = 0.0
    thermal = 0.0
    for c in ("ux", "uy", "uz"):
        u = p[c][m]
        mc = np.bincount(cell, weights=w * u, minlength=nx * nz)
        mean_c = np.divide(mc, Wc, out=np.zeros_like(mc), where=Wc > 0)
        bulk += 0.5 * M_ION * np.sum(Wc * mean_c**2)
        thermal += 0.5 * M_ION * (np.sum(w * u**2) - np.sum(Wc * mean_c**2))
    return bulk, thermal


def field_energy_by_region(ds, pb):
    """Mean EM fluctuation energy density (J/m^3) per region, level-1 view."""
    dims = ds.domain_dimensions * ds.refine_by ** min(1, ds.index.max_level)
    cg = ds.covering_grid(min(1, ds.index.max_level), ds.domain_left_edge, dims)
    u = np.zeros(dims[:2])
    for i, f in enumerate(["Ex", "Ey", "Ez"]):
        u += 0.5 * EPS0 * np.array(cg["boxlib", f])[:, :, 0] ** 2
    for i, f in enumerate(["Bx", "By", "Bz"]):
        u += (np.array(cg["boxlib", f])[:, :, 0] - B0[i]) ** 2 / (2 * MU0)
    dx = (ds.domain_width.d / dims)[:2]
    xc = ds.domain_left_edge.d[0] + (np.arange(dims[0]) + 0.5) * dx[0]
    zc = ds.domain_left_edge.d[1] + (np.arange(dims[1]) + 0.5) * dx[1]
    X, Z = np.meshgrid(xc, zc, indexing="ij")
    dxc, dzc = (ds.domain_width.d / ds.domain_dimensions)[:2]
    r = make_regions(X.ravel(), Z.ravel(), pb, dxc, dzc)
    return {k: float(u.ravel()[v].mean()) for k, v in r.items()}


def census(ds, pb):
    (xlo, xhi), (zlo, zhi) = pb
    dxc, dzc = (ds.domain_width.d / ds.domain_dimensions)[:2]
    dxf, dzf = dxc / ds.refine_by, dzc / ds.refine_by
    p = particles(ds)
    in_patch = (p["x"] >= xlo) & (p["x"] < xhi) & (p["z"] >= zlo) & (p["z"] < zhi)
    print("=== (a) ppc census (step 0) ===")
    print(f"total macroparticles: {p['x'].size}")
    res = {}
    for name, m, x0, z0, dx, dz, nx, nz in (
        (
            "coarse (outside patch)",
            ~in_patch,
            ds.domain_left_edge.d[0],
            ds.domain_left_edge.d[1],
            dxc,
            dzc,
            int(ds.domain_dimensions[0]),
            int(ds.domain_dimensions[1]),
        ),
        (
            "fine (inside patch)",
            in_patch,
            xlo,
            zlo,
            dxf,
            dzf,
            int(round((xhi - xlo) / dxf)),
            int(round((zhi - zlo) / dzf)),
        ),
    ):
        ix = np.clip(((p["x"][m] - x0) / dx).astype(int), 0, nx - 1)
        iz = np.clip(((p["z"][m] - z0) / dz).astype(int), 0, nz - 1)
        cnt = np.bincount(ix * nz + iz, minlength=nx * nz)
        if "coarse" in name:  # keep only cells outside the patch
            xc = x0 + (np.arange(nx)[:, None] + 0.5) * dx
            zc = z0 + (np.arange(nz)[None, :] + 0.5) * dz
            keep = ~((xc >= xlo) & (xc < xhi) & (zc >= zlo) & (zc < zhi))
            cnt = cnt.reshape(nx, nz)[keep]
        hist = np.bincount(cnt, minlength=10)[:10]
        w = p["w"][m]
        print(
            f"{name}: N={m.sum()}, cells={cnt.size}, ppc mean={cnt.mean():.3f}"
            f" | ppc histogram 0..9: {list(hist)}"
        )
        print(f"    weights: mean={w.mean():.4e}  min={w.min():.4e}  max={w.max():.4e}")
        res[name] = (cnt, w)
    area_fine = (xhi - xlo) * (zhi - zlo)
    area_dom = float(np.prod(ds.domain_width.d[:2]))
    n_per_area_in = in_patch.sum() / area_fine
    n_per_area_out = (~in_patch).sum() / (area_dom - area_fine)
    print(
        f"per-volume sampling (macroparticles/m^2): inside patch {n_per_area_in:.4e},"
        f" outside {n_per_area_out:.4e}, ratio {n_per_area_in / n_per_area_out:.3f}"
    )
    return res


def region_report(run_dir, pb, label):
    pfs = plotfiles(run_dir)
    ds0, ds1 = yt.load(pfs[0]), yt.load(pfs[-1])
    dxc, dzc = (ds0.domain_width.d / ds0.domain_dimensions)[:2]
    print(f"=== (b) region-resolved heating: {label} (first vs last plotfile) ===")
    out = {}
    for tag, ds in (("t0", ds0), ("t1", ds1)):
        p = particles(ds)
        regions = make_regions(p["x"], p["z"], pb, dxc, dzc)
        out[tag] = {k: wstats(p, m) for k, m in regions.items()}
        out[tag]["bulk_thermal"] = {
            k: bulk_thermal_split(p, m, ds) for k, m in regions.items()
        }
        out[tag]["field"] = field_energy_by_region(ds, pb)
    hdr = (
        f"{'region':9s} {'N':>7s} {'KE1/KE0-1':>10s} {'kT_par0':>8s} {'kT_par1':>8s}"
        f" {'kT_perp0':>9s} {'kT_perp1':>9s} {'drift1_z':>9s} {'bulk1/th1':>9s}"
        f" {'u_EM0':>9s} {'u_EM1':>9s}"
    )
    print(hdr)
    for k in ("interior", "ring", "outside"):
        s0, s1 = out["t0"][k], out["t1"][k]
        b1, th1 = out["t1"]["bulk_thermal"][k]
        print(
            f"{k:9s} {s1['N']:7d} {s1['KE'] / s0['KE'] - 1:10.4%}"
            f" {s0['kT_par']:8.3f} {s1['kT_par']:8.3f}"
            f" {s0['kT_perp']:9.3f} {s1['kT_perp']:9.3f}"
            f" {s1['drift'][2]:9.2e} {b1 / th1:9.5f}"
            f" {out['t0']['field'][k]:9.3e} {out['t1']['field'][k]:9.3e}"
        )
    print(
        "(kT in eV; drift in m/s; u_EM = mean EM fluctuation energy density J/m^3;"
        " bulk1/th1 = per-coarse-cell bulk / thermal KE at final step)"
    )
    return out


def heating_rate(run_dir):
    pe = np.loadtxt(
        os.path.join(run_dir, "diags/reducedfiles/part_energy.txt"), skiprows=1
    )
    return pe[:, 0], pe[:, 2]


def main():
    args = [a for a in sys.argv[1:]]
    prefix = "t_smoke"
    if len(args) % 2 == 1:
        prefix = args.pop()
    mr_dir, ctl_dir = args[0], args[1]
    mr4_dir, ctl4_dir = (args[2], args[3]) if len(args) >= 4 else (None, None)

    ds0 = yt.load(plotfiles(mr_dir)[0])
    pb = patch_bounds(ds0)
    census_res = census(ds0, pb)

    reg_mr = region_report(mr_dir, pb, "MR")
    reg_ctl = region_report(ctl_dir, pb, "control (same regions)")

    print("=== (c) heating vs ppc (reduced part_energy) ===")
    rates = {}
    for name, d in (
        ("MR", mr_dir),
        ("CTL", ctl_dir),
        ("MR 4x ppc", mr4_dir),
        ("CTL 4x ppc", ctl4_dir),
    ):
        if d is None:
            continue
        s, e = heating_rate(d)
        rates[name] = (s, e, e[-1] / e[0] - 1)
        print(f"  {name:11s}: KE(final)/KE(0) - 1 = {rates[name][2]:+.4%}")
    if mr4_dir:
        ex1 = rates["MR"][2] - rates["CTL"][2]
        ex4 = rates["MR 4x ppc"][2] - rates["CTL 4x ppc"][2]
        print(f"  MR-CTL excess: baseline {ex1:+.4%}, 4x ppc {ex4:+.4%}")
        if ex1 > 0 and ex4 > 0:
            expo = np.log(ex1 / ex4) / np.log(4.0)
            print(
                f"  scaling exponent p (excess ~ ppc^-p): {expo:.2f}"
                "  (p ~ 1 => noise-driven, p ~ 0 => systematic)"
            )

    # ---- figure ----------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.5))
    cnt_f = census_res["fine (inside patch)"][0]
    cnt_c = census_res["coarse (outside patch)"][0]
    bins = np.arange(-0.5, 10.5)
    axs[0].hist(
        cnt_c,
        bins=bins,
        alpha=0.65,
        color="#3d65f5",
        label="coarse cells (outside)",
        density=True,
    )
    axs[0].hist(
        cnt_f,
        bins=bins,
        alpha=0.65,
        color="#c95d20",
        label="fine cells (inside)",
        density=True,
    )
    axs[0].set_xlabel("macroparticles per cell (step 0)")
    axs[0].set_ylabel("fraction of cells")
    axs[0].set_title("(a) ppc census per level")
    axs[0].legend(frameon=False)

    labels = ("interior", "ring", "outside")
    x = np.arange(3)
    for off, (reg, name, color) in enumerate(
        ((reg_mr, "MR", "#c95d20"), (reg_ctl, "control", "#3d65f5"))
    ):
        d = [reg["t1"][k]["KE"] / reg["t0"][k]["KE"] - 1 for k in labels]
        axs[1].bar(
            x + 0.4 * off - 0.2, np.array(d) * 100, 0.35, color=color, label=name
        )
    axs[1].set_xticks(x, labels)
    axs[1].set_ylabel("region KE change over the run (%)")
    axs[1].set_title("(b) heating by region")
    axs[1].legend(frameon=False)
    axs[1].grid(alpha=0.25, lw=0.5, axis="y")

    for name, color, ls in (
        ("MR", "#c95d20", "-"),
        ("CTL", "#3d65f5", "-"),
        ("MR 4x ppc", "#c95d20", "--"),
        ("CTL 4x ppc", "#3d65f5", "--"),
    ):
        if name in rates:
            s, e, _ = rates[name]
            axs[2].plot(s, e / e[0], color=color, ls=ls, lw=1.8, label=name)
    axs[2].set_xlabel("step")
    axs[2].set_ylabel("particle energy / initial")
    axs[2].set_title("(c) heating vs ppc")
    axs[2].legend(frameon=False, fontsize=9)
    axs[2].grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(f"{prefix}_ppc_heating.png", dpi=270)
    print(f"wrote {prefix}_ppc_heating.png")


if __name__ == "__main__":
    main()
