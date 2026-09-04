#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Conduction Helmholtz block of the MHD block preconditioner under the
dual_energy ion closure: the exact rank-one register-pair inverse.

The dual_energy_advection problem with a hot pressure spot and isotropic
conduction on both channels at conduction number ~50, kinetic energy 500x
the internal energy (blend weight f_k = 1 - 1/gamma). Gates against the
_off twin (pc_mhd_block.conduction_block = 0):

1. Same converged state: ion total, ion internal and electron energies
   agree to the Newton tolerance (measured 1e-13 to 1e-15 at landing).
2. Krylov work: total GMRES iterations at most 0.35 of the twin's
   (measured 368 vs 2029 over 8 steps / 24 Newton iterations; the f_k-split
   diagonal version this replaced gave 2453, i.e. worse than no block, so
   the ratio gate is the one that matters here).

Usage: analysis_mhd_conduction_pc_dual.py <final_plotfile>
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

for name in (
    "implicit_mhd_ion_energy",
    "implicit_mhd_ion_internal_energy",
    "implicit_mhd_electron_energy",
):
    a = final["boxlib", name].value.ravel()
    b = baseline["boxlib", name].value.ravel()
    deviation = np.max(np.abs(a - b)) / np.max(np.abs(b))
    print(f"{name}: max |pc - off| / max|off| = {deviation:.3e}")
    assert deviation < 1.0e-7

# newton.txt APPENDS across reruns: read the LAST 8 rows (one per step);
# columns [3] total_iters and [7] gmres_total_iters are cumulative per run.
number_of_steps = 8
history = np.atleast_2d(np.loadtxt("diags/newton.txt"))[-number_of_steps:]
baseline_history = np.atleast_2d(
    np.loadtxt(baseline_path.parent.parent / "diags" / "newton.txt")
)[-number_of_steps:]
assert history.shape[0] == baseline_history.shape[0] == number_of_steps
assert history[0, 0] == baseline_history[0, 0] == 1

newton_iterations = history[-1, 3]
gmres_pc = history[-1, 7]
gmres_off = baseline_history[-1, 7]
print(
    f"Newton iterations {int(newton_iterations)}; GMRES iterations with the "
    f"conduction block {int(gmres_pc)} vs {int(gmres_off)} without "
    f"(ratio {gmres_pc / gmres_off:.3f})"
)
for name, h in (("pc", history), ("off", baseline_history)):
    converged = (h[:, 4] <= 1.1e-12) | (h[:, 5] <= 1.1e-10)
    assert np.all(converged), f"{name}: an unconverged Newton solve"
assert gmres_pc <= 0.35 * gmres_off
