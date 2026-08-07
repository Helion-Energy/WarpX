#!/usr/bin/env python3
"""Bias-removed deposited-temperature analysis of the landau MR smoke pair.

Consumes the run pair produced from tdep_inputs_2d (MR arm + max_level=0
control, ions.do_temperature_deposition=1, plotfiles at steps 0,1,50..300
with Tx_ions/Ty_ions/Tz_ions and write_species=1) and reports

  (1) a numpy REPLICA of the in-code estimator (same CIC shape weights,
      two-pass local-mean removal, n/(n-1) correction, T=0 where n<=1),
      validated cell-by-cell against the plotfile fields;
  (2) the bias-fair mean-T table: density(rho)-weighted mean of the
      deposited T per region (patch interior / seam ring / outside),
      T_par = Tz, T_perp = (Tx+Ty)/2, at t0 (step-1 plotfile ~ t0; the
      step-0 plotfile has T=0 because the first deposit happens at the
      start of Evolve) and step 300, MR vs control;
  (3) the cell-to-cell variance of the deposited T per region per run
      (plotfile, filtered) and per node (replica, unfiltered), plus the
      undefined-cell accounting (n<=1 -> T=0);
  (4) the decomposition: region-aggregate weighted-variance temperature
      (wstats over particles) minus deposited-T mean = meso-scale
      fluctuation temperature between the shape scale and the region
      scale; the split of the interior T_par rise into sub-grid thermal
      growth vs meso-scale fluctuation growth;
  (5) seam lineouts of the deposited T and of rho across the patch edge.

Units: the registry field T_<species> holds, per direction, the
sampling-bias-corrected weighted variance of v (= u/gamma) times m/kB,
i.e. a per-degree-of-freedom temperature in KELVIN (bilinear-filtered
once when warpx.use_filter=1, the default here). Converted to eV below.

Usage:
    python t_tdep_analyze.py MR_DIR CTL_DIR [OUT_PREFIX]
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
KB = 1.380649e-23
K_TO_EV = KB / EV  # T[eV] = T[K] * K_TO_EV
T0_EV = M_ION * (0.001 * C) ** 2 / EV  # deck u_th = 0.001c -> 51.16 eV/dof
RING_W = 4  # seam-ring width in coarse cells inside the patch edge
EDGE_TRIM_FINE = 2  # fine cells trimmed off the patch edge for "safe" stats

CMR, CCTL = "#c95d20", "#3d65f5"  # entity colors (match earlier figures)

# 2D XZ Yee staggering of the deposition target (1 = NODE, 0 = CELL),
# read off the J box arrays the T field is allocated on:
STAG = {"x": (0, 1), "y": (1, 1), "z": (1, 0)}  # Tx~jx, Ty~jy, Tz~jz


# ---------------------------------------------------------------- loading
def plotfiles(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, "diags", "diag1??????")))


def load_step(run_dir, step):
    return yt.load(os.path.join(run_dir, "diags", f"diag1{step:06d}"))


def particles(ds):
    ad = ds.all_data()
    out = {
        "x": ad["ions", "particle_position_x"].d,
        "z": ad["ions", "particle_position_y"].d,
        "w": ad["ions", "particle_weight"].d,
    }
    for c, name in (("ux", "x"), ("uy", "y"), ("uz", "z")):
        out[c] = ad["ions", f"particle_momentum_{name}"].d / M_ION  # gamma*v
    return out


def patch_bounds(ds):
    l1 = [g for g in ds.index.grids if g.Level == 1]
    lo = np.min([g.LeftEdge.d for g in l1], axis=0)
    hi = np.max([g.RightEdge.d for g in l1], axis=0)
    return (lo[0], hi[0]), (lo[1], hi[1])


def level_grid(ds, lev, fields):
    """Assemble each field of one level into a single 2D array."""
    gs = [g for g in ds.index.grids if g.Level == lev]
    dds = gs[0].dds.d[:2]
    lo = np.min([g.LeftEdge.d[:2] for g in gs], axis=0)
    hi = np.max([g.RightEdge.d[:2] for g in gs], axis=0)
    dims = np.rint((hi - lo) / dds).astype(int)
    out = {f: np.full(dims, np.nan) for f in fields}
    for g in gs:
        i0, k0 = np.rint((g.LeftEdge.d[:2] - lo) / dds).astype(int)
        for f in fields:
            a = np.array(g["boxlib", f])[:, :, 0]
            out[f][i0 : i0 + a.shape[0], k0 : k0 + a.shape[1]] = a
    return out, lo, dds


# ---------------------------------------------------------------- regions
def region_masks(X, Z, pb, dxc, dzc, trim=0.0):
    """interior / ring / outside masks for point sets (cells or particles).

    trim > 0 also returns ring_safe: the ring with `trim` (in metres)
    shaved off the patch edge, excluding the partial-support edge rows.
    """
    (xlo, xhi), (zlo, zhi) = pb
    in_patch = (X >= xlo) & (X < xhi) & (Z >= zlo) & (Z < zhi)
    interior = (
        (X >= xlo + RING_W * dxc)
        & (X < xhi - RING_W * dxc)
        & (Z >= zlo + RING_W * dzc)
        & (Z < zhi - RING_W * dzc)
    )
    ring = in_patch & ~interior
    in_safe = (
        (X >= xlo + trim) & (X < xhi - trim) & (Z >= zlo + trim) & (Z < zhi - trim)
    )
    return {
        "interior": interior,
        "ring": ring,
        "ring_safe": ring & in_safe,
        "outside": ~in_patch,
    }


def wstats(p, m):
    """Region-aggregate weighted drift and temperatures (eV/dof)."""
    w = p["w"][m]
    W = w.sum()
    drift = np.array([np.sum(w * p[c][m]) / W for c in ("ux", "uy", "uz")])
    var = np.array(
        [np.sum(w * (p[c][m] - d) ** 2) / W for c, d in zip(("ux", "uy", "uz"), drift)]
    )
    kT = M_ION * var / EV
    return {"N": int(m.sum()), "kT_par": kT[2], "kT_perp": 0.5 * (kT[0] + kT[1])}


# ---------------------------------------------------------------- replica
def depose_var(x, z, v, w, lo, dx, dz, ncx, ncz, stag, periodic):
    """Numpy replica of the two-pass shape-matched variance deposition.

    Returns (T_K, n) on the staggered grid (valid points only):
    shape (ncx + sx, ncz + sz) for stag = (sx, sz), 1 = NODE, 0 = CELL.
    Matches the C++ kernel: CIC (order 1) shape factors on the staggered
    grid, integer sample count n incremented for every touched point,
    pass 1 accumulates (n, sum wS, sum wS v), pass 2 deposits
    wS (v - vbar)^2 against the pass-1 local mean, and
    T = m/kB * n/((n-1) sum wS) * sum wS (v-vbar)^2, 0 where n <= 1.
    """
    sx, sz = stag
    G = 2  # guards
    npx, npz = ncx + sx, ncz + sz
    shp = (npx + 2 * G, npz + 2 * G)
    N = np.zeros(shp, dtype=np.int64)
    W = np.zeros(shp)
    WV = np.zeros(shp)
    W2 = np.zeros(shp)

    fx = (x - lo[0]) / dx - (0.0 if sx else 0.5)
    fz = (z - lo[1]) / dz - (0.0 if sz else 0.5)
    jx = np.floor(fx).astype(np.int64)
    jz = np.floor(fz).astype(np.int64)
    ax = fx - jx
    az = fz - jz
    wx = (1.0 - ax, ax)
    wz = (1.0 - az, az)

    idx = []
    for a in (0, 1):
        for b in (0, 1):
            ii, kk = jx + a + G, jz + b + G
            ww = w * wx[a] * wz[b]
            np.add.at(N, (ii, kk), 1)
            np.add.at(W, (ii, kk), ww)
            np.add.at(WV, (ii, kk), ww * v)
            idx.append((ii, kk, ww))

    def fold_dim(A, axis, n_valid):
        """Periodic SumBoundary along one axis: fold every staggered point
        onto its canonical image (period = n_valid lattice points, so a
        nodal hi edge aliases the lo edge), then refresh all positions
        (guards + aliased edge) from the canonical values."""
        if not periodic[axis]:
            return A
        B = A if axis == 0 else A.T
        tgt = (np.arange(B.shape[0]) - G) % n_valid + G
        out = np.zeros_like(B)
        np.add.at(out, tgt, B)
        out = out[tgt]
        return out if axis == 0 else out.T

    N = fold_dim(fold_dim(N, 0, ncx), 1, ncz)
    W = fold_dim(fold_dim(W, 0, ncx), 1, ncz)
    WV = fold_dim(fold_dim(WV, 0, ncx), 1, ncz)

    vbar = np.divide(WV, W, out=np.zeros_like(WV), where=N > 0)

    for ii, kk, ww in idx:
        np.add.at(W2, (ii, kk), ww * (v - vbar[ii, kk]) ** 2)
    W2 = fold_dim(fold_dim(W2, 0, ncx), 1, ncz)

    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(N > 1, N / ((N - 1.0) * W) * W2 * (M_ION / KB), 0.0)
    sl = (slice(G, G + npx), slice(G, G + npz))
    return T[sl], N[sl]


def bilinear_filter(A, periodic):
    """One WarpX bilinear pass per dim ([1/4,1/2,1/4]); zero-extended edges."""
    for ax in (0, 1):
        if periodic[ax]:
            Am = np.roll(A, 1, axis=ax)
            Ap = np.roll(A, -1, axis=ax)
        else:
            Am = np.zeros_like(A)
            Ap = np.zeros_like(A)
            if ax == 0:
                Am[1:], Ap[:-1] = A[:-1], A[1:]
            else:
                Am[:, 1:], Ap[:, :-1] = A[:, :-1], A[:, 1:]
        A = 0.25 * Am + 0.5 * A + 0.25 * Ap
    return A


def cellcenter(A, stag):
    if stag[0] == 1:
        A = 0.5 * (A[:-1, :] + A[1:, :])
    if stag[1] == 1:
        A = 0.5 * (A[:, :-1] + A[:, 1:])
    return A


def replica_cell_T(p, sel, lo, dx, dz, ncx, ncz, periodic, filt):
    """Replica deposited T (eV), cell-centered, per velocity component,
    plus the n<=1 node fraction per component."""
    out, nfrac = {}, {}
    inv_g = 1.0 / np.sqrt(
        1.0 + (p["ux"][sel] ** 2 + p["uy"][sel] ** 2 + p["uz"][sel] ** 2) / C**2
    )
    for comp in ("x", "y", "z"):
        v = p["u" + comp][sel] * inv_g
        T, N = depose_var(
            p["x"][sel],
            p["z"][sel],
            v,
            p["w"][sel],
            lo,
            dx,
            dz,
            ncx,
            ncz,
            STAG[comp],
            periodic,
        )
        nfrac[comp] = float(np.mean(N <= 1))
        if filt:
            T = bilinear_filter(T, periodic)
        out[comp] = cellcenter(T, STAG[comp]) * K_TO_EV
    return out, nfrac


# ---------------------------------------------------------------- stats
def dep_stats(Tpar, Tperp, rho, mask):
    """rho-weighted mean and cell-to-cell scatter of the deposited T (eV),
    excluding undefined (exactly zero) cells; reports their weight share."""
    res = {}
    for tag, T in (("par", Tpar), ("perp", Tperp)):
        m = mask & np.isfinite(T) & np.isfinite(rho)
        good = m & (T > 0)
        wsum = rho[m].sum()
        w = rho[good]
        mean = np.sum(w * T[good]) / w.sum() if w.sum() > 0 else np.nan
        res[tag] = {
            "mean": mean,
            "median": float(np.median(T[good])) if good.any() else np.nan,
            "std": float(np.std(T[good])) if good.any() else np.nan,
            "zero_wfrac": 1.0 - (w.sum() / wsum if wsum > 0 else np.nan),
            "ncells": int(good.sum()),
        }
    return res


def cell_coords(lo, dds, dims):
    xc = lo[0] + (np.arange(dims[0]) + 0.5) * dds[0]
    zc = lo[1] + (np.arange(dims[1]) + 0.5) * dds[1]
    return np.meshgrid(xc, zc, indexing="ij")


# ================================================================== main
def main():
    args = sys.argv[1:]
    prefix = args[2] if len(args) > 2 else "t_tdep"
    mr_dir, ctl_dir = args[0], args[1]
    TF = ("Tx_ions", "Ty_ions", "Tz_ions", "rho")

    ds0 = load_step(mr_dir, 0)
    pb = patch_bounds(ds0)
    dxc, dzc = (ds0.domain_width.d / ds0.domain_dimensions)[:2]
    dxf, dzf = dxc / 2, dzc / 2
    trim = EDGE_TRIM_FINE * dxf

    runs = {"MR": mr_dir, "CTL": ctl_dir}
    steps = {"t0": 1, "t300": 300}

    # ---------------- (1) replica validation, step 300 -----------------
    print("=== (1) replica validation of the deposited field (step 300) ===")
    val = {}
    for run, d in runs.items():
        ds = load_step(d, 300)
        p = particles(ds)
        if run == "MR":
            lev = 1
            G, lo, dds = level_grid(ds, 1, TF)
            (xlo, xhi), (zlo, zhi) = pb
            sel = (p["x"] >= xlo) & (p["x"] < xhi) & (p["z"] >= zlo) & (p["z"] < zhi)
            periodic = (False, False)
        else:
            lev = 0
            G, lo, dds = level_grid(ds, 0, TF)
            sel = np.ones_like(p["x"], dtype=bool)
            periodic = (True, True)
        dims = G["Tz_ions"].shape
        rep, nfrac = replica_cell_T(
            p, sel, lo, dds[0], dds[1], dims[0], dims[1], periodic, filt=True
        )
        pf = G["Tz_ions"] * K_TO_EV
        # compare away from level edges (replica filter edge treatment differs)
        c = 3
        A, B = rep["z"][c:-c, c:-c], pf[c:-c, c:-c]
        both = (A > 0) & (B > 0)
        r = A[both] / B[both]
        corr = np.corrcoef(A[both], B[both])[0, 1]
        val[run] = (np.median(r), corr)
        print(
            f"  {run} lev{lev} Tz: replica/plotfile median ratio ="
            f" {np.median(r):.4f}, corr = {corr:.4f},"
            f" n<=1 node fraction = {nfrac['z']:.3%}"
        )

    # ---------------- (2)-(4) region stats at t0 and t300 ---------------
    print(
        "\n=== (2) bias-fair deposited-T means / (3) cell scatter /"
        " (4) aggregate decomposition ==="
    )
    print(
        f"(deck truth at t=0: {T0_EV:.2f} eV/dof; deposited T from the"
        " step-1 plotfile serves as t0)"
    )
    S = {}  # S[run][time][region] -> dict
    NF = {}  # replica n<=1 fractions per run/time (fine or single level)
    for run, d in runs.items():
        S[run] = {}
        NF[run] = {}
        for tname, step in steps.items():
            ds = load_step(d, step)
            p = particles(ds)
            # particle aggregates per region
            regs_p = region_masks(p["x"], p["z"], pb, dxc, dzc, trim)
            agg = {k: wstats(p, m) for k, m in regs_p.items()}
            # deposited fields, per level
            res = {}
            if run == "MR":
                G1, lo1, dds1 = level_grid(ds, 1, TF)
                X1, Z1 = cell_coords(lo1, dds1, G1["Tz_ions"].shape)
                r1 = region_masks(X1, Z1, pb, dxc, dzc, trim)
                Tpar = G1["Tz_ions"] * K_TO_EV
                Tperp = 0.5 * (G1["Tx_ions"] + G1["Ty_ions"]) * K_TO_EV
                for k in ("interior", "ring", "ring_safe"):
                    res[k] = dep_stats(Tpar, Tperp, G1["rho"], r1[k])
                G0, lo0, dds0 = level_grid(ds, 0, TF)
                X0, Z0 = cell_coords(lo0, dds0, G0["Tz_ions"].shape)
                r0 = region_masks(X0, Z0, pb, dxc, dzc, trim)
                res["outside"] = dep_stats(
                    G0["Tz_ions"] * K_TO_EV,
                    0.5 * (G0["Tx_ions"] + G0["Ty_ions"]) * K_TO_EV,
                    G0["rho"],
                    r0["outside"],
                )
                # replica n<=1 accounting on the fine level
                (xlo, xhi), (zlo, zhi) = pb
                sel = (
                    (p["x"] >= xlo) & (p["x"] < xhi) & (p["z"] >= zlo) & (p["z"] < zhi)
                )
                _, nfrac = replica_cell_T(
                    p,
                    sel,
                    lo1,
                    dds1[0],
                    dds1[1],
                    *G1["Tz_ions"].shape,
                    (False, False),
                    filt=False,
                )
            else:
                G0, lo0, dds0 = level_grid(ds, 0, TF)
                X0, Z0 = cell_coords(lo0, dds0, G0["Tz_ions"].shape)
                r0 = region_masks(X0, Z0, pb, dxc, dzc, trim)
                Tpar = G0["Tz_ions"] * K_TO_EV
                Tperp = 0.5 * (G0["Tx_ions"] + G0["Ty_ions"]) * K_TO_EV
                for k in ("interior", "ring", "ring_safe", "outside"):
                    res[k] = dep_stats(Tpar, Tperp, G0["rho"], r0[k])
                sel = np.ones_like(p["x"], dtype=bool)
                _, nfrac = replica_cell_T(
                    p,
                    sel,
                    lo0,
                    dds0[0],
                    dds0[1],
                    *G0["Tz_ions"].shape,
                    (True, True),
                    filt=False,
                )
            S[run][tname] = {"agg": agg, "dep": res}
            NF[run][tname] = nfrac["z"]

    hdr = (
        f"{'run':4s} {'time':5s} {'region':9s}"
        f" {'dep_par':>8s} {'dep_perp':>9s} {'agg_par':>8s} {'agg_perp':>9s}"
        f" {'meso_par':>9s} {'sig_par':>8s} {'CV':>6s} {'zero_w':>7s}"
    )
    print(hdr)
    for run in runs:
        for tname in steps:
            for reg in ("interior", "ring", "ring_safe", "outside"):
                dep = S[run][tname]["dep"][reg]
                agg = S[run][tname]["agg"][reg]
                meso = agg["kT_par"] - dep["par"]["mean"]
                cv = dep["par"]["std"] / dep["par"]["mean"]
                print(
                    f"{run:4s} {tname:5s} {reg:9s}"
                    f" {dep['par']['mean']:8.2f} {dep['perp']['mean']:9.2f}"
                    f" {agg['kT_par']:8.2f} {agg['kT_perp']:9.2f}"
                    f" {meso:9.2f} {dep['par']['std']:8.2f} {cv:6.3f}"
                    f" {dep['par']['zero_wfrac']:7.3%}"
                )
    print(
        "(eV/dof; dep = rho-weighted mean of deposited T excluding T=0"
        " cells; agg = particle weighted variance over the region;"
        " meso = agg - dep; sig/CV = cell-to-cell scatter of deposited T)"
    )
    for run in runs:
        print(
            f"  replica n<=1 node fraction (Tz staggering), {run}:"
            f" t0 {NF[run]['t0']:.3%}, t300 {NF[run]['t300']:.3%}"
        )

    print("\n--- t0 estimator-bias check (uniform 51.16 eV plasma) ---")
    for run in runs:
        for reg in ("interior", "outside"):
            m = S[run]["t0"]["dep"][reg]["par"]["mean"]
            print(f"  {run} {reg}: dep t0 = {m:6.2f} eV = {m / T0_EV:.3f} x truth")

    print("\n--- (4) interior heating split, t0 -> t300 ---")
    split = {}
    for run in runs:
        for tag in ("par", "perp"):
            a0 = S[run]["t0"]["agg"]["interior"]["kT_" + tag]
            a1 = S[run]["t300"]["agg"]["interior"]["kT_" + tag]
            d0 = S[run]["t0"]["dep"]["interior"][tag]["mean"]
            d1 = S[run]["t300"]["dep"]["interior"][tag]["mean"]
            dagg, ddep = a1 - a0, d1 - d0
            split[(run, tag)] = (a0, a1, d0, d1)
            if abs(dagg) > 1e-12:
                print(
                    f"  {run} kT_{tag}: aggregate {a0:6.2f} -> {a1:6.2f}"
                    f" (D={dagg:+6.2f});  deposited {d0:6.2f} -> {d1:6.2f}"
                    f" (D={ddep:+6.2f});  thermal fraction of rise ="
                    f" {ddep / dagg:6.1%}, meso fraction = {1 - ddep / dagg:6.1%}"
                )

    # ---------------- (5) seam lineouts, MR step 300 --------------------
    ds = load_step(mr_dir, 300)
    G1, lo1, dds1 = level_grid(ds, 1, TF)
    G0, lo0, dds0 = level_grid(ds, 0, TF)
    (xlo, xhi), (zlo, zhi) = pb
    zmid = 0.5 * (zlo + zhi)
    xmid = 0.5 * (xlo + xhi)

    # x-cut at z = zmid: fine inside, coarse outside
    k1 = int((zmid - lo1[1]) / dds1[1])
    x1 = lo1[0] + (np.arange(G1["Tz_ions"].shape[0]) + 0.5) * dds1[0]
    t1x = G1["Tz_ions"][:, k1] * K_TO_EV
    k0 = int((zmid - lo0[1]) / dds0[1])
    x0 = lo0[0] + (np.arange(G0["Tz_ions"].shape[0]) + 0.5) * dds0[0]
    t0x = G0["Tz_ions"][:, k0] * K_TO_EV

    # z-cut at x = xmid
    i1 = int((xmid - lo1[0]) / dds1[0])
    z1 = lo1[1] + (np.arange(G1["Tz_ions"].shape[1]) + 0.5) * dds1[1]
    t1z = G1["Tz_ions"][i1, :] * K_TO_EV
    i0 = int((xmid - lo0[0]) / dds0[0])
    z0 = lo0[1] + (np.arange(G0["Tz_ions"].shape[1]) + 0.5) * dds0[1]
    t0z = G0["Tz_ions"][i0, :] * K_TO_EV

    Tint = S["MR"]["t300"]["dep"]["interior"]["par"]["mean"]
    rho_int = np.nanmean(G1["rho"][:, abs(z1 - zmid) < 10 * dzf])

    # ring-averaged edge profiles: mean over patch columns (x-edges trimmed)
    # vs distance from the nearest z edge, in fine cells; both edges pooled.
    xin = (x1 >= xlo + 4 * dxf) & (x1 < xhi - 4 * dxf)
    dz_edge = np.minimum(z1 - zlo, zhi - z1) / dzf  # cell centers: 0.5, 1.5, ..
    prof_T, prof_rho, dbin = [], [], []
    for d in np.arange(0.5, 12.5, 1.0):
        rows = np.abs(dz_edge - d) < 0.25
        if rows.any():
            prof_T.append(np.nanmean(G1["Tz_ions"][np.ix_(xin, rows)]) * K_TO_EV)
            prof_rho.append(np.nanmean(G1["rho"][np.ix_(xin, rows)]))
            dbin.append(d)
    prof_T, prof_rho, dbin = map(np.asarray, (prof_T, prof_rho, dbin))
    T_deep = prof_T[dbin > 6].mean()
    rho_deep = prof_rho[dbin > 6].mean()
    print("\n=== (5) seam behaviour of the deposited T (MR, step 300) ===")
    print(
        "  ring-averaged profile vs distance from the patch z-edge"
        " (fine cells; both edges pooled, x-edges trimmed):"
    )
    for d, tv, rv in zip(dbin, prof_T, prof_rho):
        print(
            f"    d = {d:4.1f}: <Tz> = {tv:6.2f} eV ({tv / T_deep:5.2f} x deep)"
            f"   <rho> ratio = {rv / rho_deep:5.3f}"
        )
    print(
        f"  (deep reference: <Tz> = {T_deep:.2f} eV; interior rho-weighted"
        f" mean = {Tint:.2f} eV)"
    )

    # ------------------------------- figures ---------------------------
    # fig 1: means + scatter + decomposition
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    regs = ("interior", "ring_safe", "outside")
    xb = np.arange(len(regs))
    for off, (run, color) in enumerate((("MR", CMR), ("CTL", CCTL))):
        dep1 = [S[run]["t300"]["dep"][r]["par"]["mean"] for r in regs]
        dep0 = [S[run]["t0"]["dep"][r]["par"]["mean"] for r in regs]
        agg1 = [S[run]["t300"]["agg"][r]["kT_par"] for r in regs]
        axs[0, 0].bar(
            xb + 0.4 * off - 0.2, dep1, 0.3, color=color, label=f"{run} dep t300"
        )
        axs[0, 0].bar(
            xb + 0.4 * off - 0.2,
            dep0,
            0.3,
            fill=False,
            edgecolor="k",
            ls="--",
            lw=1.2,
            label=f"{run} dep t0",
        )
        axs[0, 0].plot(
            xb + 0.4 * off - 0.2,
            agg1,
            "k_",
            ms=18,
            mew=2,
            label="aggregate t300" if off == 0 else None,
        )
    axs[0, 0].axhline(T0_EV, color="0.5", lw=0.8, ls=":")
    axs[0, 0].text(2.35, T0_EV, " t=0 truth", va="center", fontsize=9, color="0.4")
    axs[0, 0].set_xticks(xb, regs)
    axs[0, 0].set_ylabel(r"$kT_\parallel$ (eV)")
    axs[0, 0].set_title("(a) deposited-T mean (rho-weighted) vs aggregate")
    axs[0, 0].legend(frameon=False, fontsize=8)
    axs[0, 0].grid(alpha=0.2, lw=0.5, axis="y")

    for off, (run, color) in enumerate((("MR", CMR), ("CTL", CCTL))):
        cv0 = [
            S[run]["t0"]["dep"][r]["par"]["std"] / S[run]["t0"]["dep"][r]["par"]["mean"]
            for r in regs
        ]
        cv1 = [
            S[run]["t300"]["dep"][r]["par"]["std"]
            / S[run]["t300"]["dep"][r]["par"]["mean"]
            for r in regs
        ]
        axs[0, 1].bar(xb + 0.4 * off - 0.2, cv1, 0.3, color=color, label=f"{run} t300")
        axs[0, 1].bar(
            xb + 0.4 * off - 0.2,
            cv0,
            0.3,
            fill=False,
            edgecolor=color,
            ls="--",
            label=f"{run} t0",
        )
    axs[0, 1].set_xticks(xb, regs)
    axs[0, 1].set_ylabel(
        r"cell-to-cell $\sigma(T_\parallel)/\langle T_\parallel\rangle$"
    )
    axs[0, 1].set_title("(b) per-cell estimator scatter (filtered field)")
    axs[0, 1].legend(frameon=False, fontsize=8)
    axs[0, 1].grid(alpha=0.2, lw=0.5, axis="y")

    # (c) interior heating split
    labels, thermal, meso = [], [], []
    for run in ("MR", "CTL"):
        for tag in ("par", "perp"):
            a0, a1, d0, d1 = split[(run, tag)]
            labels.append(f"{run}\n{tag}")
            thermal.append(d1 - d0)
            meso.append((a1 - a0) - (d1 - d0))
    xs = np.arange(len(labels))
    axs[1, 0].bar(xs, thermal, 0.55, color="#7a4b8f", label="sub-grid thermal")
    axs[1, 0].bar(
        xs, meso, 0.55, bottom=thermal, color="#4b8f7a", label="meso-scale fluct."
    )
    axs[1, 0].axhline(0, color="k", lw=0.8)
    axs[1, 0].set_xticks(xs, labels)
    axs[1, 0].set_ylabel(r"$\Delta kT$ interior, t0$\to$t300 (eV)")
    axs[1, 0].set_title("(c) interior heating split: thermal vs meso")
    axs[1, 0].legend(frameon=False, fontsize=9)
    axs[1, 0].grid(alpha=0.2, lw=0.5, axis="y")

    # (d) x-cut lineout
    axs[1, 1].plot(x1 * 1e3, t1x, color=CMR, lw=1.6, label="fine level Tz")
    m_out = (x0 < xlo) | (x0 >= xhi)
    axs[1, 1].plot(
        x0[m_out] * 1e3, t0x[m_out], ".", color=CCTL, ms=7, label="coarse level Tz"
    )
    for xe in (xlo, xhi):
        axs[1, 1].axvline(xe * 1e3, color="0.6", lw=0.8, ls="--")
    axs[1, 1].set_xlabel("x (mm)")
    axs[1, 1].set_ylabel(r"deposited $T_\parallel$ (eV)")
    axs[1, 1].set_title("(d) MR x-lineout across the patch (z = patch mid)")
    axs[1, 1].legend(frameon=False, fontsize=9)
    axs[1, 1].grid(alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(f"{prefix}_means.png", dpi=270)
    print(f"\nwrote {prefix}_means.png")

    # fig 2: seam lineouts, T vs rho, both cuts
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
    axs[0].plot(
        z1 * 1e3, t1z / Tint, color=CMR, lw=1.5, label="fine Tz / interior mean"
    )
    axs[0].plot(
        z1 * 1e3,
        G1["rho"][i1, :] / rho_int,
        color="0.35",
        lw=1.2,
        label="fine rho / interior mean",
    )
    m0 = (z0 < zlo) | (z0 >= zhi)
    axs[0].plot(z0[m0] * 1e3, t0z[m0] / Tint, ".", color=CCTL, ms=6, label="coarse Tz")
    for ze in (zlo, zhi):
        axs[0].axvline(ze * 1e3, color="0.6", lw=0.8, ls="--")
    axs[0].set_xlabel("z (mm)")
    axs[0].set_ylabel("normalized value")
    axs[0].set_title("z-lineout across patch (x = patch mid), step 300")
    axs[0].legend(frameon=False, fontsize=9)
    axs[0].grid(alpha=0.2, lw=0.5)

    axs[1].plot(
        dbin,
        prof_T / T_deep,
        "o-",
        color=CMR,
        lw=1.6,
        ms=6,
        label="ring-averaged fine Tz / deep value",
    )
    axs[1].plot(
        dbin,
        prof_rho / rho_deep,
        "s-",
        color="0.35",
        lw=1.4,
        ms=5,
        label="ring-averaged fine rho / deep value",
    )
    axs[1].axhline(1, color="0.7", lw=0.8)
    axs[1].set_xlabel("distance from the patch z-edge (fine cells)")
    axs[1].set_ylabel("edge profile / deep-patch value")
    axs[1].set_title("ring-averaged edge profile (partial-support rows)")
    axs[1].legend(frameon=False, fontsize=9)
    axs[1].grid(alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(f"{prefix}_seam.png", dpi=270)
    print(f"wrote {prefix}_seam.png")

    # fig 3: maps
    fig, axs = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    ds_c = load_step(ctl_dir, 300)
    Gc, loc, ddsc = level_grid(ds_c, 0, TF)
    ex1 = [
        lo1[0] * 1e3,
        (lo1[0] + dds1[0] * G1["Tz_ions"].shape[0]) * 1e3,
        lo1[1] * 1e3,
        (lo1[1] + dds1[1] * G1["Tz_ions"].shape[1]) * 1e3,
    ]
    vmax = np.nanpercentile(G1["Tz_ions"] * K_TO_EV, 99)
    im = axs[0].imshow(
        (G1["Tz_ions"] * K_TO_EV).T,
        origin="lower",
        extent=ex1,
        aspect="auto",
        cmap="Oranges",
        vmin=0,
        vmax=vmax,
    )
    axs[0].set_title("MR fine level, deposited $T_\\parallel$ (step 300)")
    Xc, Zc = cell_coords(loc, ddsc, Gc["Tz_ions"].shape)
    mwin = (Xc >= xlo) & (Xc < xhi) & (Zc >= zlo) & (Zc < zhi)
    Tc = np.where(mwin, Gc["Tz_ions"] * K_TO_EV, np.nan)
    exc = [
        loc[0] * 1e3,
        (loc[0] + ddsc[0] * Gc["Tz_ions"].shape[0]) * 1e3,
        loc[1] * 1e3,
        (loc[1] + ddsc[1] * Gc["Tz_ions"].shape[1]) * 1e3,
    ]
    axs[1].imshow(
        Tc.T,
        origin="lower",
        extent=exc,
        aspect="auto",
        cmap="Oranges",
        vmin=0,
        vmax=vmax,
    )
    axs[1].set_xlim(ex1[0], ex1[1])
    axs[1].set_ylim(ex1[2], ex1[3])
    axs[1].set_title("control (same window), deposited $T_\\parallel$")
    for a in axs:
        a.set_xlabel("x (mm)")
    axs[0].set_ylabel("z (mm)")
    cb = fig.colorbar(im, ax=axs, shrink=0.85)
    cb.set_label(r"$T_\parallel$ (eV)")
    fig.savefig(f"{prefix}_maps.png", dpi=270)
    print(f"wrote {prefix}_maps.png")


if __name__ == "__main__":
    main()
