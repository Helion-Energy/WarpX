#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Zero-flux reflecting wall at an OPEN radial field boundary (RZ, hlld).

A rotating column is driven into the r_max wall (u_r > 0 there) while the
split external vector potential threads a nonzero B_r through the open
wall face. With implicit_mhd.r_open_fluid = reflect the r_max face must be
a TRUE zero-flux wall for the fluid: with periodic z, the zero-area axis
face, and the wall, no fluid channel has an exit, so the r-weighted total
mass must be conserved to round-off. Without the wall-face override the
smoothed HLLD fan at the mirror interface leaks mass outward at the
compression-jump dissipation scale (~1e-4 relative over this run) and the
open-face B_n opens the tangential Maxwell stress / Alfven energy channels
that drain the wall-ring ion energy toward its floor.

Usage: analysis_mhd_r_open_reflect_wall.py <initial_plotfile> <final_plotfile>
"""

import sys

import numpy as np
import yt


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])

initial_density = np.squeeze(initial["boxlib", "implicit_mhd_mass_density"].value)
final_density = np.squeeze(final["boxlib", "implicit_mhd_mass_density"].value)
final_ion_energy = np.squeeze(final["boxlib", "implicit_mhd_ion_energy"].value)

# r-weighted total mass (uniform grid: cell volume proportional to the
# cell-center radius; the constant 2*pi*dr*dz factor cancels in the
# relative comparison).
nr = initial_density.shape[0]
radius = np.arange(nr) + 0.5
initial_mass = np.sum(initial_density * radius[:, None])
final_mass = np.sum(final_density * radius[:, None])
mass_change = abs(final_mass - initial_mass) / initial_mass
print(f"relative total mass change: {mass_change:.3e}")
# Zero-flux wall: exact conservation up to accumulated round-off (the
# unfixed wall face leaks ~1e-4 relative over these 80 steps).
assert mass_change < 1.0e-11, (
    f"total mass not conserved by the reflecting zero-flux wall "
    f"(relative change {mass_change:.3e})"
)

# Wall-ring ion-energy health: the pre-fix open-B_n wall face drains the
# outermost ring's ion energy toward its floor (E_i -> KE + U_floor with
# U_floor = 1e-4 * U_0). The initial uniform internal energy is
# Pi/(gamma-1); require the final wall ring to retain at least half of
# it (the compression against the wall only ADDS energy there).
n0 = 1.0e20
q_e = 1.602176634e-19
Ti = 10.0
gamma = 5.0 / 3.0
internal_0 = n0 * q_e * Ti / (gamma - 1.0)
wall_ring_min = final_ion_energy[-1, :].min()
print(
    f"wall-ring minimum ion energy: {wall_ring_min:.6e} "
    f"(initial internal energy {internal_0:.6e})"
)
assert wall_ring_min > 0.5 * internal_0, (
    f"wall-ring ion energy drained ({wall_ring_min:.3e} < "
    f"{0.5 * internal_0:.3e}): the zero-flux wall is leaking"
)

# Newton converged every step: the wall-face override and the (per-term
# gated) open normal-stress work leave no irreducible residual in the
# wall band. Columns of newton.txt: step, time, iters, total_iters,
# norm_abs, norm_rel, ... (the file appends across reruns; take the
# last run's rows). Tolerances must match the deck.
newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
last_run = newton_history[-int(newton_history[-1][0]) :]
converged = (last_run[:, 4] < 1.0e-11) | (last_run[:, 5] < 1.0e-8)
n_bad = int(np.count_nonzero(~converged))
print(f"non-converged Newton solves: {n_bad} of {len(last_run)}")
assert n_bad == 0, f"{n_bad} Newton solves failed to converge at the zero-flux wall"

print("zero-flux reflecting wall checks passed")
