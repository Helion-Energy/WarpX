#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Neumann z-outflow ends for the theta-implicit RZ MHD solver.

A passive density pulse (uniform pressure and axial velocity, uniform Bz,
so E = -u x B = 0 exactly) advects through the z_hi boundary. By the final
time the pulse has fully exited: the interior must have relaxed back to
the uniform background with no reflected residue, the total mass must have
dropped by the pulse mass, and the quiescent electromagnetic state must be
unperturbed. With periodic z the pulse would have wrapped and still be
inside, so the background-residue bound is discriminating.

Usage: analysis_mhd_z_outflow.py <initial_plotfile> <final_plotfile> [mode]

mode = "pec" (default): the uniform Bz lives in the evolved field, so the
plotfiles carry B0 at both times. mode = "open": the Green's-function open
radial boundary requires the uniform Bz to enter through the SPLIT external
vector potential; the t=0 plotfile is written before the externals are
added to the totals, so it must show the plasma-response frame (Bz = 0),
while the final plotfile holds totals (Bz = B0). The frame asserts pin
this convention.
"""

import sys

import numpy as np
import warpx_constants as constants
import yt


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])
mode = sys.argv[3] if len(sys.argv) > 3 else "pec"
assert mode in ("pec", "open"), f"unknown mode {mode}"

number_density = 1.0e20
density_bg = number_density * constants.proton_mass
density_pulse = 3.0 * density_bg
B0 = 1.0e-3

initial_density = np.squeeze(
    initial["boxlib", "implicit_mhd_mass_density"].value
)
final_density = np.squeeze(final["boxlib", "implicit_mhd_mass_density"].value)
initial_bz = np.squeeze(initial["boxlib", "Bz"].value)
final_bz = np.squeeze(final["boxlib", "Bz"].value)

# the pulse must have left: every interior cell back to background within
# a small fraction of the pulse amplitude (donor-cell smearing of the exit
# leaves only roundoff-level residue once the tail clears the boundary)
residue = np.max(np.abs(final_density - density_bg)) / density_pulse
print(f"post-exit density residue = {residue:.3e} of pulse amplitude")
assert residue < 2.0e-2, f"reflected/retained residue too large: {residue:.3e}"

# and it was actually there at t=0 (the assertion above is discriminating:
# a periodic run would still hold the full pulse)
initial_excursion = np.max(initial_density) - density_bg
assert initial_excursion > 0.9 * density_pulse

# mass decreased by roughly the pulse mass (uniform background inflow at
# z_lo replaces background outflow at z_hi, so the change is the pulse)
mass_ratio = np.sum(final_density) / np.sum(initial_density)
print(f"final/initial mass = {mass_ratio:.4f}")
assert mass_ratio < 0.95

# the electromagnetic subsystem stayed exactly quiescent. In "open" mode
# Bz is supplied through the split external vector potential: between
# steps the plotfile holds totals (B0), but the t=0 plotfile is written
# before HybridPICInitializeRhoJandB adds the externals, so it shows the
# plasma-response frame (exactly zero for this quiescent state).
if mode == "open":
    np.testing.assert_allclose(final_bz, B0, rtol=1.0e-9, atol=0.0)
    np.testing.assert_allclose(initial_bz, 0.0, rtol=0.0, atol=1.0e-12 * B0)
else:
    np.testing.assert_allclose(final_bz, B0, rtol=1.0e-9, atol=0.0)
    np.testing.assert_allclose(initial_bz, B0, rtol=1.0e-9, atol=0.0)

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert 1 <= newton_history[-1][2] <= 12

print("PASS")
