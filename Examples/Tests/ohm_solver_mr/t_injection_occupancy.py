#!/usr/bin/env python3
"""Fine-level injection occupancy + weight-normalization audit for
warpx.refine_plasma_init (per-level plasma injection at init).

Usage:
    python t_injection_occupancy.py RUN_DIR [RUN_DIR2 ...]

Each RUN_DIR must contain diags/diag1000000 written with
diag1.write_species=1 from the t_smoke_inputs_2d deck. For every run this
reports, at step 0:

  * fine-patch occupancy: particles binned on the fine grid inside the
    level-1 patch -> mean/min/max ppc and % empty cells;
  * coarse occupancy outside the patch (control);
  * weight normalization: mean density from particle weights inside the
    patch and in the uncovered coarse region, each vs the analytic
    density profile of the deck (relative errors should be comparable).
"""

import os
import sys

import numpy as np
import yt

yt.set_log_level(50)

# deck parameters (t_smoke_inputs_2d)
N0 = 9.623617654281555e20
KZ = 343.86751681418855
DELTA = 0.03


def analytic_mean_density(zlo, zhi):
    """Mean of n0*(1+delta*cos(k z)) over [zlo, zhi]."""
    return N0 * (
        1.0 + DELTA * (np.sin(KZ * zhi) - np.sin(KZ * zlo)) / (KZ * (zhi - zlo))
    )


def audit(run_dir):
    ds = yt.load(os.path.join(run_dir, "diags", "diag1000000"))
    ref = int(ds.refine_by)

    # domain / grids
    dom_lo = np.array(ds.domain_left_edge)[:2]
    dom_hi = np.array(ds.domain_right_edge)[:2]
    ncell = np.array(ds.domain_dimensions)[:2]
    dx_c = (dom_hi - dom_lo) / ncell

    fine_grids = [g for g in ds.index.grids if g.Level == 1]
    if not fine_grids:
        print(f"{run_dir}: no level-1 grids (single-level run)")
        return
    patch_lo = np.min([np.array(g.LeftEdge)[:2] for g in fine_grids], axis=0)
    patch_hi = np.max([np.array(g.RightEdge)[:2] for g in fine_grids], axis=0)
    dx_f = dx_c / ref

    ad = ds.all_data()
    x = np.array(ad["ions", "particle_position_x"])
    z = np.array(ad["ions", "particle_position_y"])  # 2nd dim is z in the deck
    w = np.array(ad["ions", "particle_weight"])
    print(f"\n=== {run_dir} ===")
    print(f"total particles: {len(x)}")
    print(f"patch (x,z): lo={patch_lo}, hi={patch_hi}")

    inpatch = (
        (x >= patch_lo[0]) & (x < patch_hi[0]) & (z >= patch_lo[1]) & (z < patch_hi[1])
    )

    # --- fine-grid occupancy inside the patch -----------------------------
    nf = np.rint((patch_hi - patch_lo) / dx_f).astype(int)
    ix = np.floor((x[inpatch] - patch_lo[0]) / dx_f[0]).astype(int)
    iz = np.floor((z[inpatch] - patch_lo[1]) / dx_f[1]).astype(int)
    ix = np.clip(ix, 0, nf[0] - 1)
    iz = np.clip(iz, 0, nf[1] - 1)
    counts = np.zeros(nf, dtype=int)
    np.add.at(counts, (ix, iz), 1)
    print(
        f"fine occupancy over {nf[0]}x{nf[1]} cells: mean ppc = {counts.mean():.4f}, "
        f"min = {counts.min()}, max = {counts.max()}, "
        f"empty = {100.0 * np.mean(counts == 0):.2f}%"
    )

    # --- coarse occupancy outside the patch (control) ---------------------
    jx = np.floor((x[~inpatch] - dom_lo[0]) / dx_c[0]).astype(int)
    jz = np.floor((z[~inpatch] - dom_lo[1]) / dx_c[1]).astype(int)
    jx = np.clip(jx, 0, ncell[0] - 1)
    jz = np.clip(jz, 0, ncell[1] - 1)
    ccounts = np.zeros(ncell, dtype=int)
    np.add.at(ccounts, (jx, jz), 1)
    # mask out coarse cells covered by the patch
    plo_c = np.rint((patch_lo - dom_lo) / dx_c).astype(int)
    phi_c = np.rint((patch_hi - dom_lo) / dx_c).astype(int)
    covered = np.zeros(ncell, dtype=bool)
    covered[plo_c[0] : phi_c[0], plo_c[1] : phi_c[1]] = True
    cc = ccounts[~covered]
    print(
        f"coarse occupancy (uncovered {cc.size} cells): mean ppc = {cc.mean():.4f}, "
        f"min = {cc.min()}, max = {cc.max()}, empty = {100.0 * np.mean(cc == 0):.2f}%"
    )

    # --- weight normalization ---------------------------------------------
    # 2D weights carry density * cell area (per unit length in the 3rd dim)
    area_patch = np.prod(patch_hi - patch_lo)
    dens_patch = w[inpatch].sum() / area_patch
    ana_patch = analytic_mean_density(patch_lo[1], patch_hi[1])
    # coarse control: the full uncovered region
    area_coarse = np.prod(dom_hi - dom_lo) - area_patch
    dens_coarse = w[~inpatch].sum() / area_coarse
    # analytic mean over the uncovered region (domain mean minus patch part)
    ana_dom = analytic_mean_density(dom_lo[1], dom_hi[1])
    ana_coarse = (
        ana_dom * np.prod(dom_hi - dom_lo) - ana_patch * area_patch
    ) / area_coarse
    print(
        f"mean density  fine patch : {dens_patch:.6e}  analytic {ana_patch:.6e}  "
        f"rel.err {abs(dens_patch / ana_patch - 1):.3e}"
    )
    print(
        f"mean density  coarse rest: {dens_coarse:.6e}  analytic {ana_coarse:.6e}  "
        f"rel.err {abs(dens_coarse / ana_coarse - 1):.3e}"
    )


if __name__ == "__main__":
    for d in sys.argv[1:]:
        audit(d)
