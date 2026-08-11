#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Scrape-off arrival onto the absorbing wall (RZ, hlld, low beta).

A dense column edge is driven into a near-pedestal wall band threaded by
a strong external field with a nonzero B_r through the open wall face --
the miniature of the FRC mirror benchmark's deterministic
scrape-off-arrival blowup. The retired forced-v_A vacuum-image absorber
is Newton-intractable on this deck (measured: every solve frozen from
step 1, frozen-guard abort at step 10, absorbed-ENERGY counter running
NEGATIVE -- the wall pumping the domain). The impedance-matched one-way
absorber must complete the window with bounded fields, a monotone
absorbed-mass/energy ledger whose mass counter matches the domain mass
loss (periodic z: the wall face is the only exit), and no wall-band
pile-up.

Usage: analysis_mhd_absorb_scrape_off.py <initial_plotfile> <final_plotfile> <ledger_file>
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
final_b = np.stack(
    [np.squeeze(final["boxlib", c].value) for c in ("Br", "Bt", "Bz")]
)

# 1) Bounded fields through the arrival window: the retired forced-v_A
# absorber explodes B at the wall corner rows (measured 5.7e3 T on the
# FRC benchmark; on this deck it freezes from step 1 and aborts before
# its first dump). Measured healthy max plotted |B| component 2.4e-2 T.
b_max = np.abs(final_b).max()
print(f"final max |B| component: {b_max:.6e} T")
assert b_max < 0.2, f"wall fields not bounded (max |B| {b_max:.3e} T)"

# Absolute total mass [kg]: M = sum rho * 2*pi*r_c*dr*dz.
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
print(f"initial mass: {initial_mass:.15e} kg")
print(f"final mass:   {final_mass:.15e} kg")
print(f"relative mass loss: {mass_loss / initial_mass:.6e}")

# 2) The arriving scrape-off must actually be absorbed (measured
# 2.91e-1 relative loss over the 120-step window; require a third of
# it). The reflect wall aborts at step 54 on this deck (frozen-guard),
# storing the arrival instead of absorbing it.
assert mass_loss > 1.0e-1 * initial_mass, (
    f"absorbing wall did not absorb the arriving scrape-off "
    f"(relative loss {mass_loss / initial_mass:.3e})"
)

# 3) Ledger honesty: cumulative absorbed mass matches the domain mass
# loss. The solver runs the production policy (non-converged solves are
# accepted during the genuinely hard arrival), so the match tolerance is
# the accumulated accepted-residual scale, measured 4.2e-8 of the
# initial mass on this deck (a wrong area/radius factor misses by the
# full loss, 2.9e-1): assert with 20x margin.
ledger_mass = ledger[-1, 1]
ledger_energy = ledger[-1, 2]
mismatch = abs(ledger_mass - mass_loss) / initial_mass
print(f"ledger absorbed mass: {ledger_mass:.15e} kg")
print(f"ledger mismatch (relative to initial mass): {mismatch:.6e}")
assert mismatch < 1.0e-6, (
    f"absorbed-mass ledger does not match the domain mass loss "
    f"(mismatch {mismatch:.3e} of the initial mass)"
)

# 4) One-way wall: both cumulative counters positive and monotone
# non-decreasing (the retired absorber's energy counter ran NEGATIVE on
# this deck before it froze).
assert ledger_mass > 0.0 and ledger_energy > 0.0, (
    f"absorbed counters not positive (mass {ledger_mass:.3e}, "
    f"energy {ledger_energy:.3e})"
)
assert np.all(np.diff(ledger[:, 1]) >= -1.0e-9 * ledger_mass), (
    "cumulative absorbed mass is not monotone non-decreasing"
)
assert np.all(np.diff(ledger[:, 2]) >= -1.0e-9 * ledger_energy), (
    "cumulative absorbed energy is not monotone non-decreasing"
)

# 5) No wall-band pile-up: the wall ring carries the TRANSITING
# scrape-off at step 120 (measured mean 7.9e-2 of the peak) but must
# not accumulate toward the column density (the reflect wall stores the
# arrival and freezes before reaching this step).
wall_ring_mean = final_density[-1, :].mean()
peak = final_density.max()
print(f"final wall-ring mean density: {wall_ring_mean / peak:.6e} of peak")
assert wall_ring_mean < 0.2 * peak, (
    f"wall ring piled up ({wall_ring_mean / peak:.3e} of peak)"
)

print("absorbing-wall scrape-off arrival checks passed")
