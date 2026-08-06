#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Wall-band ion-temperature floor at the zero-flux open-field wall (RZ, hlld).

A COLD rotating column (T_i = 2 eV, just 5.5% above the configured
temperature floor) is driven into the r_max reflecting wall while the
split external vector potential threads a nonzero B_r through the OPEN
field face. The retained open normal Maxwell-stress channel of the
wall-face override does persistent negative work on the wall band, and
the surrounding rarefaction cools the column into the floor's gate band.
The RHS drain gates must anchor every energy drain -- the stress work,
the electron-pdV pairing, and the donor-side face fluxes -- at the
PER-CELL effective bound max(absolute floor, n kB T_floor) so that

  * no cell's ion temperature ends below the floor (without the anchored
    gates the drains punch through the kinetic-energy corner of the
    total-energy admissibility bound and the end-of-step ratchet
    disengages: measured 19.7 kK against this 22 kK floor), and
  * Newton converges EVERY step while the band rides the floor
    (newton.require_convergence = true also aborts the run otherwise),
  * with the zero-flux wall still conserving the r-weighted total mass
    to round-off (the gates touch no conservative channel).

The ion temperature needs the kinetic energy: the plotted jr/jt/jz are
the ion current (q/m rho u) sampled from the cell-centered momentum, so
M = j m_p/e reconstructs it up to face-interpolation error (~0.5%; the
assert tolerances leave several times that).

Usage: analysis_mhd_r_open_reflect_wall_floor.py <initial_plotfile> <final_plotfile>
"""

import sys

import numpy as np
import yt

M_P = 1.67262192369e-27
Q_E = 1.602176634e-19
K_B = 1.380649e-23
GAMMA = 5.0 / 3.0
T_FLOOR = 22000.0  # K, must match the deck
NEWTON_ATOL = 1.0e-11  # must match the deck
NEWTON_RTOL = 1.0e-8


def fields(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    rho = np.squeeze(data["boxlib", "implicit_mhd_mass_density"].value)
    ion_energy = np.squeeze(data["boxlib", "implicit_mhd_ion_energy"].value)
    momentum_squared = (M_P / Q_E) ** 2 * (
        np.squeeze(data["boxlib", "jr"].value) ** 2
        + np.squeeze(data["boxlib", "jt"].value) ** 2
        + np.squeeze(data["boxlib", "jz"].value) ** 2
    )
    kinetic = 0.5 * momentum_squared / rho
    ion_temperature = (GAMMA - 1.0) * (ion_energy - kinetic) / (rho / M_P * K_B)
    return rho, ion_temperature


rho_initial, t_i_initial = fields(sys.argv[1])
rho_final, t_i_final = fields(sys.argv[2])

# r-weighted total mass (uniform grid: cell volume proportional to the
# cell-center radius; constant factors cancel in the relative change).
nr = rho_initial.shape[0]
radius = np.arange(nr) + 0.5
initial_mass = np.sum(rho_initial * radius[:, None])
final_mass = np.sum(rho_final * radius[:, None])
mass_change = abs(final_mass - initial_mass) / initial_mass
print(f"relative total mass change: {mass_change:.3e}")
assert mass_change < 1.0e-11, (
    f"total mass not conserved by the reflecting zero-flux wall "
    f"(relative change {mass_change:.3e})"
)

print(
    f"initial min T_i: {t_i_initial.min():.1f} K, "
    f"final min T_i: {t_i_final.min():.1f} K (floor {T_FLOOR:.0f} K)"
)

# the initial state sits ABOVE the floor's gate band (the floor engages
# through the dynamics, not through initialization) ...
assert t_i_initial.min() > 1.045 * T_FLOOR
# ... the run cools the coldest cells INTO the band (discriminating) ...
assert t_i_final.min() < 1.035 * T_FLOOR, (
    "the wall-band drain never engaged the temperature floor; the test "
    f"is not discriminating (min T_i {t_i_final.min():.1f} K)"
)
# ... and the floor HELD: no cell ends below it (2% reconstruction slack)
assert t_i_final.min() >= 0.98 * T_FLOOR, (
    f"ion temperature fell through the floor ({t_i_final.min():.1f} K < "
    f"{T_FLOOR:.0f} K): the wall-band E_i drains are not anchored at the "
    "per-cell temperature bound"
)

# Newton converged EVERY step while the band rode the floor (columns of
# newton.txt: step, time, iters, total_iters, norm_abs, norm_rel, ...;
# the file appends across reruns, so take the last run's rows).
newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
last_run = newton_history[-int(newton_history[-1][0]) :]
converged = (last_run[:, 4] < NEWTON_ATOL) | (last_run[:, 5] < NEWTON_RTOL)
n_bad = int(np.count_nonzero(~converged))
print(f"non-converged Newton solves: {n_bad} of {len(last_run)}")
assert n_bad == 0, (
    f"{n_bad} Newton solves failed to converge while the wall band rode "
    "the temperature floor (uninsulated wall-band drain)"
)

print("wall-band temperature-floor checks passed")
