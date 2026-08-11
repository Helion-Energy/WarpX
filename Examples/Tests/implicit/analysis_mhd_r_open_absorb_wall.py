#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Absorbing wall at an OPEN radial field boundary (RZ, hlld).

The same column-into-the-wall deck as the reflect zero-flux-wall test
(which asserts EXACT round-off mass conservation): a rotating column is
driven into the r_max wall (u_r > 0 there) while the split external
vector potential threads a nonzero B_r through the open wall face. With
implicit_mhd.r_open_fluid = absorb the wall is a solid conductor that
SWALLOWS incident plasma at up to the local Alfven rate: the total mass
must decrease substantially (the reflect twin conserves it to < 1e-11
relative), the wall ring must not accumulate (its density ends BELOW the
initial uniform value), and the runtime absorbed-mass ledger -- the
r-weighted wall-face flux integral -- must match the domain mass loss to
the nonlinear solver tolerance, keeping the conservation ledger honest.

Usage: analysis_mhd_r_open_absorb_wall.py <initial_plotfile> <final_plotfile> <ledger_file>
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
ledger = np.atleast_2d(np.loadtxt(sys.argv[3]))

initial_density = np.squeeze(initial["boxlib", "implicit_mhd_mass_density"].value)
final_density = np.squeeze(final["boxlib", "implicit_mhd_mass_density"].value)

# Absolute total mass [kg] (uniform grid): M = sum rho * 2*pi*r_c*dr*dz.
nr, nz = initial_density.shape
r_lo = float(initial_ds.domain_left_edge[0].value)
r_hi = float(initial_ds.domain_right_edge[0].value)
z_lo = float(initial_ds.domain_left_edge[1].value)
z_hi = float(initial_ds.domain_right_edge[1].value)
dr = (r_hi - r_lo) / nr
dz = (z_hi - z_lo) / nz
radius = r_lo + (np.arange(nr) + 0.5) * dr
cell_volume = 2.0 * np.pi * radius[:, None] * dr * dz
initial_mass = np.sum(initial_density * cell_volume)
final_mass = np.sum(final_density * cell_volume)
mass_loss = initial_mass - final_mass
relative_loss = mass_loss / initial_mass
print(f"initial mass: {initial_mass:.15e} kg")
print(f"final mass:   {final_mass:.15e} kg")
print(f"relative mass loss: {relative_loss:.6e}")

# 1) The absorbing wall must DRAIN the driven column: a substantial mass
# export (measured 5.74e-1 relative over these 80 steps; the reflect twin
# conserves to < 1e-11), so require at least half the measured drain.
assert relative_loss > 0.25, (
    f"absorbing wall did not drain the driven column "
    f"(relative mass loss {relative_loss:.3e})"
)

# 2) No wall-ring accumulation: the wall ring must end well BELOW the
# initial uniform density (the reflect wall stores the driven column
# against the wall instead; measured reflect wall-ring mean 1.04 rho0 at
# step 80 vs 0.205 rho0 under absorb).
rho0 = 1.0e20 * 1.67262192369e-27
wall_ring_mean = final_density[-1, :].mean()
print(f"wall-ring mean density: {wall_ring_mean / rho0:.6e} rho0")
assert wall_ring_mean < 0.5 * rho0, (
    f"wall ring accumulated against the absorbing wall "
    f"(mean {wall_ring_mean / rho0:.3e} rho0)"
)

# 3) Ledger honesty: the cumulative absorbed-mass counter (r-weighted
# wall-face flux integral of the accepted theta-state fluxes) must match
# the domain mass loss to the nonlinear solver tolerance. Measured
# mismatch 4.4e-13 relative to the initial mass (Newton abs/rel
# tolerances 1e-11/1e-8); assert with two decades of margin. Column
# layout of the ledger file: step, absorbed mass [kg], absorbed
# energy [J].
ledger_mass = ledger[-1, 1]
ledger_energy = ledger[-1, 2]
mismatch = abs(ledger_mass - mass_loss) / initial_mass
print(f"ledger absorbed mass:   {ledger_mass:.15e} kg")
print(f"domain mass loss:       {mass_loss:.15e} kg")
print(f"ledger mismatch (relative to initial mass): {mismatch:.6e}")
assert mismatch < 1.0e-10, (
    f"absorbed-mass ledger does not match the domain mass loss "
    f"(mismatch {mismatch:.3e} of the initial mass)"
)

# 4) The absorber is a one-way sink: the cumulative counters must be
# positive and monotone non-decreasing (inflow from the ghost is gated
# to zero by the pedestal-resident donor image).
assert ledger_mass > 0.0 and ledger_energy > 0.0, (
    f"absorbed counters not positive (mass {ledger_mass:.3e}, "
    f"energy {ledger_energy:.3e})"
)
assert np.all(np.diff(ledger[:, 1]) >= 0.0), (
    "cumulative absorbed mass is not monotone non-decreasing"
)
assert np.all(np.diff(ledger[:, 2]) >= 0.0), (
    "cumulative absorbed energy is not monotone non-decreasing"
)

print("absorbing wall checks passed")
