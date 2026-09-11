#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""The EMF side of the band-cells force mask (implicit_mhd.lorentz_force_band_emf).

The external-current wall column (uniform B_z = B0 + the static external
B_z,ext = 4 c r^2, the flat dielectric wall masking i >= 10 of 16) given a
linear radial flow u_r = u1 r/Lr. The corner EMF E_theta = u_r B_z,total
compresses the evolved field: dB_z/dt = -(u1/Lr)(2 B0 + 16 c r^2) in every
cell whose four corner EMFs read live velocities. The band-cells mask
(N = 2) is on in both arms, so cells 8-9 receive no magnetic force either
way while they move at u_r through the external current's force field.

mode "live" (lorentz_force_band_emf = 0): the inner interior (cells 1-5)
compresses within 25 % of the analytic rate, the inner band cell 8 too,
the wall-adjacent band cell 9 piles flux up against the rigid wall (its
wall corners average the masked cells' frozen state in, so E_theta drops
there: a nonzero change of the opposite sign, reported), the audit's
force_withheld is nonzero (the band's withheld work) and emf_withheld is
exactly zero.

mode "frozen" (lorentz_force_band_emf = 1, live plotfile as the 3rd
argument): the wall-adjacent band cell 9 -- every corner of which reads
only band or masked cells -- does NOT move (below 0.1 % of the analytic
change), the inner interior is the live arm's (max difference below 5 % of
the smallest interior signal: the frozen band edge's flux pile-up sends a
fast-magnetosonic response inward), force_withheld is exactly zero, and
emf_withheld carries the band's withheld work (nonzero, the sign of the
live arm's force_withheld; the magnitudes differ because the band fields do).

Usage: analysis_mhd_force_band_emf.py <live|frozen> <initial_plotfile> <final_plotfile> [<live_final_plotfile>]
"""

import re
import sys
from pathlib import Path

import numpy as np
import yt

yt.set_log_level(50)

mode, initial_plotfile, final_plotfile = sys.argv[1], sys.argv[2], sys.argv[3]
assert mode in ("live", "frozen"), mode
used_inputs = Path("warpx_used_inputs").read_text()


def input_value(name, default=None):
    match = re.search(rf"^{re.escape(name)}\s*=\s*([^#\n]+)", used_inputs, re.MULTILINE)
    if match is None:
        assert default is not None, name
        return default
    return match.group(1).strip().strip('"')


B0 = float(input_value("my_constants.B0"))
c_ext = float(input_value("my_constants.c_ext"))
u1 = float(input_value("my_constants.u1"))
Lr = float(input_value("my_constants.Lr"))
dt = float(input_value("my_constants.dt"))
max_step = int(input_value("max_step"))
band_cells = int(input_value("implicit_mhd.lorentz_force_band_cells"))
band_emf = int(input_value("implicit_mhd.lorentz_force_band_emf", "0"))
assert band_cells == 2, band_cells
assert band_emf == (1 if mode == "frozen" else 0), (mode, band_emf)
t_final = max_step * dt


def load(plotfile):
    ds = yt.load(plotfile)
    dims = ds.domain_dimensions
    data = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)
    nr, nz = int(dims[0]), int(dims[1])
    r_lo = float(ds.domain_left_edge[0].value)
    dr = (float(ds.domain_right_edge[0].value) - r_lo) / nr
    r = r_lo + (np.arange(nr) + 0.5) * dr
    bz = np.array(data["boxlib", "Bz"].value).reshape(nr, nz)
    return float(ds.current_time), r, dr, bz


def read_audit(path):
    columns, rows = None, []
    with open(path) as handle:
        for line in handle:
            if line.startswith("# columns:"):
                columns = line[len("# columns:"):].split()
            elif line.startswith("#") or not line.strip():
                continue
            else:
                rows.append([float(x) for x in line.split()])
    assert columns is not None, path
    data = np.array(rows)
    return {name: data[:, i] for i, name in enumerate(columns)}


t0, r, dr, bz0 = load(initial_plotfile)
t1, _, _, bz1 = load(final_plotfile)
assert abs(t1 - t_final) <= 1.0e-3 * t_final, (t1, t_final)
nr, nz = bz1.shape
# the code's wall classification (wall_flat_r065.csv): the first masked cell
r_wall = 0.065
first_masked = int(np.argmax(r >= r_wall - 1.0e-3 * dr))
assert first_masked == 10, first_masked
band = np.zeros(nr, dtype=bool)
band[first_masked - band_cells:first_masked] = True
inner = np.zeros(nr, dtype=bool)
inner[1:first_masked - band_cells - 2] = True    # cells 1-5: the axis cell excluded, two live cells from the band
wall_cell = first_masked - 1                      # cell 9: every corner reads band or masked cells

# analytic compression of the total field by the linear radial flow
expected = -(u1 * t_final / Lr) * (2.0 * B0 + 16.0 * c_ext * r ** 2)
# The plotfile's Bz carries the split external field (4 c r^2 on the discrete
# curl of A_theta = c r^3, i.e. 4 c (r^2 + dr^2/4) at a cell centre) from the
# first step on but not in the t = 0 dump, so the evolved change is the
# difference minus that static offset (checked: the outermost masked cells,
# where nothing moves, come out exactly static).
external = 4.0 * c_ext * (r ** 2 + 0.25 * dr ** 2)
change = bz1 - bz0 - external[:, None]
static_tail = np.arange(nr) >= nr - 2
static_change = float(np.max(np.abs(change[static_tail])))
print(f"outermost masked cells {np.flatnonzero(static_tail).tolist()}: max |dBz| after the external offset "
      f"{static_change:.3e} T (must be static)")
assert static_change < 1.0e-3 * float(np.min(np.abs(expected))), static_change
print(f"mode {mode}: band cells {np.flatnonzero(band).tolist()}, inner interior {np.flatnonzero(inner).tolist()}, "
      f"wall-adjacent band cell {wall_cell}, t_final {t_final:.3e} s")
print("radial profile (z-mean): i r expected_dBz measured_dBz ratio")
for i in range(nr):
    measured = float(change[i].mean())
    print(f"  {i:2d} {r[i]:.4f} {expected[i]:12.4e} {measured:12.4e} {measured / expected[i]:8.4f}"
          f"{'  (band)' if band[i] else ''}{'  (masked)' if i >= first_masked else ''}")


def ratio_to_expected(cells):
    measured = change[cells].ravel()
    reference = np.repeat(expected[cells], nz)
    return float(np.sum(measured * reference) / np.sum(reference * reference))


inner_ratio = ratio_to_expected(inner)
band_ratio = ratio_to_expected(band)
wall_ratio = ratio_to_expected(np.arange(nr) == wall_cell)
print(f"compression ratio measured/analytic: inner interior {inner_ratio:.4f}, band {band_ratio:.4f}, "
      f"wall-adjacent band cell {wall_ratio:.4f}")
assert 0.75 < inner_ratio < 1.25, inner_ratio

audit = read_audit("diags/energy_audit.txt")
lorentz_sum = float(np.sum(audit["lorentz"]))
force_withheld_sum = float(np.sum(audit["force_withheld"]))
emf_withheld_sum = float(np.sum(audit["emf_withheld"]))
print(f"audit sums: lorentz {lorentz_sum:.6e} J, force_withheld {force_withheld_sum:.6e} J, "
      f"emf_withheld {emf_withheld_sum:.6e} J")

inner_band_ratio = ratio_to_expected(np.arange(nr) == first_masked - band_cells)
print(f"inner band cell {first_masked - band_cells}: ratio {inner_band_ratio:.4f}")
if mode == "live":
    # the inner band cell compresses like the interior; the wall-adjacent
    # cell's flux piles up against the rigid wall (its wall corners average
    # the frozen masked state in, so E_theta drops there): nonzero, reported
    assert 0.75 < inner_band_ratio < 1.5, inner_band_ratio
    assert abs(wall_ratio) > 0.5, wall_ratio
    assert force_withheld_sum != 0.0, force_withheld_sum
    assert emf_withheld_sum == 0.0, emf_withheld_sum
else:
    # every corner of the wall-adjacent band cell reads a frozen state: its
    # induction EMF is identically zero and its evolved field does not move
    assert abs(wall_ratio) < 1.0e-3, wall_ratio
    live_plotfile = sys.argv[4]
    _, _, _, bz_live = load(live_plotfile)
    inner_signal = float(np.min(np.abs(expected[inner])))
    per_cell = np.max(np.abs(bz1 - bz_live), axis=1) / inner_signal
    print("frozen vs live, max |dBz| per cell over the smallest inner signal: "
          + " ".join(f"{i}:{per_cell[i]:.4f}" for i in range(nr)))
    inner_difference = float(np.max(per_cell[inner]))
    # The frozen band's edge piles flux up (cell 8 above), and that pile-up's
    # fast-magnetosonic response (0.8 cells over the run) reaches the inner
    # interior at the percent level; the interior's own induction never
    # reads a frozen state.
    assert inner_difference < 0.05, inner_difference
    assert force_withheld_sum == 0.0, force_withheld_sum
    assert emf_withheld_sum != 0.0, emf_withheld_sum
    live_audit = read_audit(str(Path(live_plotfile).parents[1] / "diags/energy_audit.txt"))
    live_force_withheld = float(np.sum(live_audit["force_withheld"]))
    # The two arms' band fields differ (static vs piled up), so the two
    # withheld works are the same sign and order but not equal: reported.
    print(f"frozen emf_withheld {emf_withheld_sum:.6e} vs live force_withheld {live_force_withheld:.6e} J "
          f"(ratio {emf_withheld_sum / live_force_withheld:.3f})")
    assert emf_withheld_sum * live_force_withheld > 0.0, (emf_withheld_sum, live_force_withheld)
print(f"band EMF freeze ({mode}): PASS")
