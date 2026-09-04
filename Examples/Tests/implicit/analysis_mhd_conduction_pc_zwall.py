#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Conduction Helmholtz block of the MHD block preconditioner with the
conductive z-end exchange (implicit_mhd.z_wall_conduction).

The z_wall_conduction deck (RZ 8x64, electron channel only: frozen
barotropic ions) run backward-Euler at 200x its resolved step, i.e. a bulk
conduction number of 100 per face and an end-face exchange diagonal of 200.
The end exchange is a half-cell Dirichlet drain against the z-wall
reservoir; the preconditioner owns it through a Dirichlet end boundary and
the emitted end-face coefficient, and composes only the shaped-wall
interface remainder of the Jacobi rows (there is none on this deck).
Before that treatment the block made this deck WORSE than no block at all
(301 vs 128 GMRES iterations at this step, 534 vs 128 at twice it), because
the z-end rows were composed after a Helmholtz that was no longer the
identity on them.

Gates against the _off twin (pc_mhd_block.conduction_block = 0):

1. Same converged state: the final electron energy fields agree to the
   Newton tolerance.
2. Krylov work: total GMRES iterations at most 0.8 of the twin's and no
   more Newton iterations than the twin.

Usage: analysis_mhd_conduction_pc_zwall.py <final_plotfile>
       <baseline_final_plotfile>
"""

import sys
from pathlib import Path

import numpy as np
import yt

yt.set_log_level(50)


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


final_path = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])
# keep the datasets alive: the covering grids hold weak references to them
final_ds, final = get_data(str(final_path))
baseline_ds, baseline = get_data(str(baseline_path))

energy_final = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()
energy_baseline = baseline["boxlib", "implicit_mhd_electron_energy"].value.ravel()
scale = np.max(np.abs(energy_baseline))
field_deviation = np.max(np.abs(energy_final - energy_baseline)) / scale
print(f"max |U_e(pc) - U_e(off)| / max|U_e| = {field_deviation:.3e}")
assert field_deviation < 1.0e-7

# newton.txt APPENDS across reruns: read the LAST 4 rows (one per step);
# columns [3] total_iters and [7] gmres_total_iters are cumulative per run.
number_of_steps = 4
history = np.atleast_2d(np.loadtxt("diags/newton.txt"))[-number_of_steps:]
baseline_history = np.atleast_2d(
    np.loadtxt(baseline_path.parent.parent / "diags" / "newton.txt")
)[-number_of_steps:]
assert history.shape[0] == baseline_history.shape[0] == number_of_steps
assert history[0, 0] == baseline_history[0, 0] == 1

newton_pc = history[-1, 3]
newton_off = baseline_history[-1, 3]
gmres_pc = history[-1, 7]
gmres_off = baseline_history[-1, 7]
print(
    f"Newton iterations {int(newton_pc)} vs {int(newton_off)}; GMRES iterations "
    f"with the conduction block {int(gmres_pc)} vs {int(gmres_off)} without "
    f"(ratio {gmres_pc / gmres_off:.3f})"
)
# the deck's Newton tolerances: absolute 1e-11, relative 1e-8 (the block
# run stops on the relative criterion with the 1e-10 Krylov tolerance; the
# twin without the block terminates GMRES exactly on this small symmetric
# problem, which is why its counts are multiples of 32)
for name, h in (("pc", history), ("off", baseline_history)):
    converged = (h[:, 4] <= 1.1e-11) | (h[:, 5] <= 1.1e-8)
    assert np.all(converged), f"{name}: an unconverged Newton solve"
assert newton_pc <= newton_off
assert gmres_pc <= 0.8 * gmres_off
