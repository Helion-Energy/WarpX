#!/usr/bin/env python3
"""Plotfile reducer for the T1.7 low-density-seam runs.

Extends the t1_pltreader reduction with the seam-local, POINTWISE
quantities the T1.7 metrics need (an x-average would cancel the Hall/E
noise the test is about). For every output and every level it stores, per
z-column (max/min over the thin x direction):

    t          (NT,)        output times (s)
    z0         (nz0,)       level-0 cell centers (m)
    bx0,by0,bz0 (NT,nz0)    x-averaged B (float32)          [monitor legacy]
    rhomean0   (NT,nz0)     x-averaged charge density rho (C/m^3)
    rhomin0    (NT,nz0)     x-min of rho (floor-clamp detector)
    emax0      (NT,nz0)     max over x of |E| (vector magnitude)
    ezmax0     (NT,nz0)     max over x of |Ez|
    hallmax0   (NT,nz0)     max over x of |curl B||B| / (mu0 max(rho,rho_floor))
                            -- pointwise bound on the Hall term |J||B|/rho
    amax0      (NT,4)       global max |Ex|,|Ey|,|Ez|,|Bz-b0|  [pltreader conv.]
    ...and the same with suffix 1 (+ patch_ilo/patch_ihi) when level 1 exists.

rho_floor is taken from the run's params.json (n_floor * q_e). curl B is
computed with np.gradient on the cell-centered arrays (one-sided at box
edges -- adequate for a noise proxy).
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1_pltreader  # noqa: E402

Q_E = 1.602176634e-19
MU0 = 1.25663706212e-06

FIELDS = ["Bx", "By", "Bz", "Ex", "Ey", "Ez", "jx", "jy", "jz", "rho"]


def _reduce_level(arr, dx, dz, rho_floor, b0):
    """Per-level reduction; arr[field] indexed [j (z), i (x)]."""
    bx, by, bz = arr["Bx"], arr["By"], arr["Bz"]
    ex, ey, ez = arr["Ex"], arr["Ey"], arr["Ez"]
    rho = arr["rho"]

    emag = np.sqrt(ex**2 + ey**2 + ez**2)
    bmag = np.sqrt(bx**2 + by**2 + bz**2)

    # curl B (2D x-z): d/dy = 0
    dBy_dz = np.gradient(by, dz, axis=0)
    dBx_dz = np.gradient(bx, dz, axis=0)
    dBz_dx = np.gradient(bz, dx, axis=1)
    dBy_dx = np.gradient(by, dx, axis=1)
    curl_mag = np.sqrt(dBy_dz**2 + (dBx_dz - dBz_dx) ** 2 + dBy_dx**2)
    hall = curl_mag * bmag / (MU0 * np.maximum(rho, rho_floor))

    return dict(
        bx=bx.mean(axis=1),
        by=by.mean(axis=1),
        bz=bz.mean(axis=1),
        rhomean=rho.mean(axis=1),
        rhomin=rho.min(axis=1),
        emax=emag.max(axis=1),
        ezmax=np.abs(ez).max(axis=1),
        hallmax=hall.max(axis=1),
        amax=[
            np.abs(ex).max(),
            np.abs(ey).max(),
            np.abs(ez).max(),
            np.abs(bz - b0).max(),
        ],
    )


LEV_KEYS = ["bx", "by", "bz", "rhomean", "rhomin", "emax", "ezmax", "hallmax", "amax"]


def convert_run(run_dir, b0=0.25, out_name="t17_reduced.npz"):
    with open(os.path.join(run_dir, "params.json")) as fh:
        params = json.load(fh)
    rho_floor = params["n_floor"] * Q_E

    plts = sorted(glob.glob(os.path.join(run_dir, "diags", "diag1??????")))
    assert plts, f"no plotfiles under {run_dir}"

    t_list = []
    lev_data = [dict((k, []) for k in LEV_KEYS) for _ in range(2)]
    nlev = None
    patch_lo = patch_hi = None
    dz = [None, None]
    zlo_dom = None
    Lx = params["Lx"]
    for p in plts:
        time, levels = t1_pltreader.read_plotfile(p, FIELDS)
        if nlev is None:
            nlev = len(levels)
            with open(os.path.join(p, "Header")) as fh:
                lines = fh.read().splitlines()
            nf = int(lines[1])
            zlo_dom = float(lines[2 + nf + 3].split()[1])
            zhi_dom = float(lines[2 + nf + 4].split()[1])
            nz0 = levels[0]["arr"]["Bx"].shape[0]
            dz[0] = (zhi_dom - zlo_dom) / nz0
            if nlev > 1:
                dz[1] = 0.5 * dz[0]
                patch_lo = levels[1]["lo"][1]
                patch_hi = patch_lo + levels[1]["arr"]["Bx"].shape[0]
        t_list.append(time)
        for lev, ld in enumerate(levels):
            nx_lev = ld["arr"]["Bx"].shape[1]
            red = _reduce_level(ld["arr"], Lx / nx_lev, dz[lev], rho_floor, b0)
            for k in LEV_KEYS:
                lev_data[lev][k].append(red[k])

    nz0 = len(lev_data[0]["bx"][0])
    out = dict(
        t=np.array(t_list),
        z0=zlo_dom + (np.arange(nz0) + 0.5) * dz[0],
        rho_floor=rho_floor,
    )
    for k in LEV_KEYS:
        out[k + "0"] = np.array(lev_data[0][k], dtype=np.float32)
    if nlev > 1:
        nz1 = len(lev_data[1]["bx"][0])
        out["z1"] = zlo_dom + (patch_lo + np.arange(nz1) + 0.5) * dz[1]
        out["patch_ilo"] = patch_lo
        out["patch_ihi"] = patch_hi
        for k in LEV_KEYS:
            out[k + "1"] = np.array(lev_data[1][k], dtype=np.float32)
    npz_path = os.path.join(run_dir, out_name)
    np.savez_compressed(npz_path, **out)
    return npz_path


if __name__ == "__main__":
    print(convert_run(sys.argv[1]))
