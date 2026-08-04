#!/usr/bin/env python3
"""Adiabat check for RCYLINDER QDSMC electron-energy equation.

For sources-off uniform initial entropy:

    T_e(r,t) = T_e0 * (n(r,t)/n0)^(gamma-1)

Masks low-density cells and a small near-axis / near-wall pad (boundary
artifacts). Exits nonzero if median/max relative error exceeds tolerances.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yt

yt.set_log_level("error")

Q_E = 1.602176634e-19


def plotfiles(path: Path) -> list[Path]:
    return sorted(
        p for p in path.glob("diag*") if p.is_dir() and (p / "Header").is_file()
    )


def load_profiles(diag_dir: Path):
    files = plotfiles(diag_dir)
    if len(files) < 2:
        raise SystemExit(f"Need >= 2 plotfiles in {diag_dir}, found {len(files)}")

    times = []
    n_list = []
    te_list = []
    radius = None
    for pf in files:
        ds = yt.load(pf)
        dims = np.asarray(ds.domain_dimensions, dtype=int)
        grid = ds.covering_grid(0, ds.domain_left_edge, dims)
        left = float(ds.domain_left_edge[0].to("m"))
        right = float(ds.domain_right_edge[0].to("m"))
        r = left + (np.arange(dims[0]) + 0.5) * ((right - left) / dims[0])
        if radius is None:
            radius = r
        rho = np.asarray(grid[("boxlib", "rho")]).squeeze()
        te = np.asarray(grid[("boxlib", "Te")]).squeeze()
        if np.nanmax(te) > 500:
            te = te / 11604.51812
        n = rho / Q_E
        times.append(float(ds.current_time.to("s")))
        n_list.append(n)
        te_list.append(te)
    return radius, np.asarray(times), np.asarray(n_list), np.asarray(te_list)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diag-dir", type=Path, default=Path("diags"))
    ap.add_argument("--gamma", type=float, default=5.0 / 3.0)
    ap.add_argument("--te0", type=float, default=100.0, help="initial Te [eV]")
    ap.add_argument("--n0", type=float, default=2.0e20)
    ap.add_argument("--n-floor-frac", type=float, default=0.05)
    ap.add_argument(
        "--edge-pad-frac",
        type=float,
        default=0.08,
        help="mask this fraction of radius near axis and outer wall",
    )
    ap.add_argument("--tol-median", type=float, default=0.05)
    ap.add_argument("--tol-max", type=float, default=0.25)
    # skip t=0 (loading noise) if requested
    ap.add_argument("--skip-initial", action="store_true", default=True)
    args = ap.parse_args(argv)

    diag_dir = args.diag_dir
    r, times, n_arr, te_arr = load_profiles(diag_dir)
    g1 = args.gamma - 1.0
    n0 = args.n0
    te0 = args.te0
    floor = args.n_floor_frac * n0
    rmax = float(r[-1] + 0.5 * (r[1] - r[0]))
    pad = args.edge_pad_frac * rmax
    spatial = (r > pad) & (r < rmax - pad)

    errors = []
    start = 1 if args.skip_initial and len(times) > 1 else 0
    for it in range(start, len(times)):
        n = n_arr[it]
        te = te_arr[it]
        mask = spatial & (n > floor)
        if not np.any(mask):
            continue
        te_ad = te0 * (n[mask] / n0) ** g1
        rel = np.abs(te[mask] - te_ad) / np.maximum(te_ad, 1e-30)
        errors.append(rel)

    if not errors:
        raise SystemExit("No cells above n_floor to score")

    all_err = np.concatenate(errors)
    med = float(np.median(all_err))
    mx = float(np.max(all_err))
    print(f"RCYLINDER QDSMC adiabat: median_rel_err={med:.4e}  max_rel_err={mx:.4e}")
    print(
        f"  n_times={len(times) - start}  n_samples={all_err.size}  "
        f"tol_med={args.tol_median}  tol_max={args.tol_max}"
    )

    ok = (med <= args.tol_median) and (mx <= args.tol_max)
    if not ok:
        print("FAIL: adiabat error exceeds tolerance")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
