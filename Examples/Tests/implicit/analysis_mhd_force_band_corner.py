#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""The band-cells force mask at a stair CORNER, and the cell-centre z_max gate.

The RZ external-current wall deck with a stepped polyline
(wall_step_r065_r080.csv: r = 0.065 for z < 0, 0.08 for z > 0; 16 x 8 cells,
dr = dz = 6.25 mm): first_masked per row = [10 10 10 10 13 13 13 13]. The
mask's rule -- a cell is a band cell if i >= min(first_masked[j-1], [j],
[j+1]) - N and the row's cell centre lies at z <= z_max -- predicts, for
N = 2: rows 0-3 cells {8, 9}; row 4 (the wide row next to the corner) cells
{8, 9, 10, 11, 12} (the two cells radially inside row 3's wall plus the three
live cells under the corner, whose axial face is a wall face); rows 5-7 cells
{11, 12}. With z_max = 0 only rows 0-3 (centres below 0) carry band cells.

mode "twin" (N = 0): reports the twin and checks its band moves.
mode "band" (N = 2, twin plotfile, N, z_max, polyline as the arguments):
every predicted band cell is at rest (rms m_r below 5 % of the twin's over
the same cells), the live non-band interior follows the twin (rms of the
difference below 3 % of the twin's band m_r), and no live non-band cell has
its motion suppressed (no unpredicted masking); with z_max = 0 the rows
above the gate are unmasked exactly.

Usage: analysis_mhd_force_band_corner.py twin <final_plotfile>
       analysis_mhd_force_band_corner.py band <final_plotfile> <twin_final_plotfile> <N> <z_max> <polyline.csv>
"""

import sys

import numpy as np
import yt

yt.set_log_level(50)

mode, final_plotfile = sys.argv[1], sys.argv[2]
assert mode in ("twin", "band"), mode


def load(plotfile):
    ds = yt.load(plotfile)
    dims = ds.domain_dimensions
    data = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)
    nr, nz = int(dims[0]), int(dims[1])
    lo, hi = ds.domain_left_edge.value, ds.domain_right_edge.value
    m_r = np.array(data["boxlib", "implicit_mhd_momentum_r"].value).reshape(nr, nz)
    return nr, nz, float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), m_r


nr, nz, r_lo, r_hi, z_lo, z_hi, mine = load(final_plotfile)
dr, dz = (r_hi - r_lo) / nr, (z_hi - z_lo) / nz

if mode == "twin":
    rms = float(np.sqrt(np.mean(mine[1:, :] ** 2)))
    print(f"twin: rms m_r over the live cells {rms:.3e}")
    assert rms > 0.0
    print("band-cells force mask at a stair corner (twin): PASS")
    sys.exit(0)

twin_plotfile, n_band, z_max, polyline = sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
_, _, _, _, _, _, twin = load(twin_plotfile)
points = np.loadtxt(polyline, delimiter=",", skiprows=1)
z_points, r_points = points[:, 0], points[:, 1]

# the code's wall classification per row: the first cell whose centre is
# on or outside the (constant-continued) polyline, with the 1e-3 dr tolerance
first_masked = np.zeros(nz, dtype=int)
for j in range(nz):
    z_c = z_lo + (j + 0.5) * dz
    r_wall = np.interp(z_c, z_points, r_points) - 1.0e-3 * dr
    first_masked[j] = int(np.argmax(r_lo + (np.arange(nr + 3) + 0.5) * dr >= r_wall))
print(f"first_masked per row: {first_masked.tolist()}")
expected_first = [10, 10, 10, 10, 13, 13, 13, 13]
assert first_masked.tolist() == expected_first, first_masked.tolist()

band = np.zeros((nr, nz), dtype=bool)
live = np.zeros((nr, nz), dtype=bool)
for j in range(nz):
    first = min(first_masked[max(j - 1, 0)], first_masked[j], first_masked[min(j + 1, nz - 1)])
    z_c = z_lo + (j + 0.5) * dz
    for i in range(nr):
        live[i, j] = i < first_masked[j]
        band[i, j] = live[i, j] and i >= first - n_band and z_c <= z_max
per_row = band.sum(axis=0).tolist()
print(f"predicted band cells per row (N = {n_band}, z_max = {z_max}): {per_row}")
if z_max > 0.0:
    assert per_row == [2, 2, 2, 2, 5, 2, 2, 2], per_row
else:
    assert per_row == [2, 2, 2, 2, 0, 0, 0, 0], per_row

print("row j: z_c | predicted band i | max |m_r| masked run | rms m_r twin | ratio")
for j in range(nz):
    cells = np.flatnonzero(band[:, j])
    z_c = z_lo + (j + 0.5) * dz
    if len(cells):
        a = float(np.max(np.abs(mine[cells, j])))
        b = float(np.sqrt(np.mean(twin[cells, j] ** 2)))
        print(f"  j={j} z_c={z_c:+.5f} | {cells.tolist()} | {a:.3e} | {b:.3e} | {a / b if b else float('nan'):.3e}")
    else:
        print(f"  j={j} z_c={z_c:+.5f} | (none)")

reference = float(np.sqrt(np.mean(twin[band] ** 2)))
assert reference > 0.0, "the twin's band did not move"
band_motion = float(np.sqrt(np.mean(mine[band] ** 2))) / reference
interior = live & ~band
interior[0, :] = False   # the axis cell, as in the parent test
interior_residual = float(np.sqrt(np.mean((twin - mine)[interior] ** 2))) / reference
suppressed = interior & (np.abs(twin) > 0.2 * reference) & (np.abs(mine) < 0.05 * np.abs(twin))
print(f"band: rms(m_r masked)/rms(m_r twin) = {band_motion:.3e}; interior rms(twin - masked)/rms(twin band) = "
      f"{interior_residual:.3e}; live non-band cells with motion suppressed below 5 % of the twin: {int(suppressed.sum())}")
assert band_motion < 0.05, band_motion
assert interior_residual < 0.03, interior_residual
assert int(suppressed.sum()) == 0, int(suppressed.sum())
if z_max <= 0.0:
    # the rows above the gate carry the twin's motion in the cells the whole-wall
    # rule would have masked
    ungated = np.zeros((nr, nz), dtype=bool)
    for j in range(4, nz):
        first = min(first_masked[max(j - 1, 0)], first_masked[j], first_masked[min(j + 1, nz - 1)])
        ungated[first - n_band:first_masked[j], j] = True
    ungated_ratio = float(np.sqrt(np.mean(mine[ungated] ** 2)) / np.sqrt(np.mean(twin[ungated] ** 2)))
    print(f"z_max gate: rows 4-7's would-be band cells move like the twin: ratio {ungated_ratio:.4f}")
    assert 0.95 < ungated_ratio < 1.05, ungated_ratio
print("band-cells force mask at a stair corner (band): PASS")
