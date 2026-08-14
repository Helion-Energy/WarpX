#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""z-end fluid boundary "outflow" for the theta-implicit RZ MHD recast.

A hot uniform slab with end-ward flow u_z = u_dir * u0 * tanh(z/w) runs
against smoothly RECTIFIED (outgoing-only) end ghosts.

mode = "drain" (u_dir = +1): both ends advect mass and total fluid
energy out at interior values, so the volume-integrated mass and energy
must decrease STRICTLY MONOTONICALLY between every pair of snapshots and
by a substantial fraction overall (the z-face energy flux is
outward-only: the r wall is a PEC no-flow wall, so any interval increase
of a total would be end-face injection).

mode = "noinject" (u_dir = -1, CLI override my_constants.u_dir=-1): the
interior pulls AWAY from both ends. The plain Neumann ghosts would feed
plasma in at interior values (the mass-injection channel this mode
kills); the rectified ghosts hold their axial momentum at ~0, so the
total mass must never increase (only the small rarefaction drain of the
end faces is allowed) and neither must the total energy.

Also pins the momentum-density diagnostic views: at t = 0
implicit_mhd_momentum_z must equal rho0 * u_dir * u0 * tanh(z/w).

Usage: analysis_mhd_z_end_outflow.py <diag_dir> <mode>
"""

import glob
import sys

import numpy as np
import yt

diag_dir = sys.argv[1]
mode = sys.argv[2]
assert mode in ("drain", "noinject"), f"unknown mode {mode}"

proton_mass = 1.67262192369e-27
qe = 1.602176634e-19
n0 = 1.0e20
rho0 = n0 * proton_mass
u0 = 1.0e5
w = 0.2
u_dir = 1.0 if mode == "drain" else -1.0


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


plotfiles = sorted(glob.glob(f"{diag_dir}/diag??????"))
assert len(plotfiles) >= 4, f"need at least 4 snapshots, got {len(plotfiles)}"

ds0, _ = get_data(plotfiles[0])
nr, nz = int(ds0.domain_dimensions[0]), int(ds0.domain_dimensions[1])
r_edges = np.linspace(
    float(ds0.domain_left_edge[0]), float(ds0.domain_right_edge[0]), nr + 1
)
z_centers = np.linspace(
    float(ds0.domain_left_edge[1]), float(ds0.domain_right_edge[1]), nz + 1
)
z_centers = 0.5 * (z_centers[1:] + z_centers[:-1])
# cylindrical cell volumes (up to the common 2 pi dz factor)
radial_weight = 0.5 * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)

masses = []
energies = []
for plotfile in plotfiles:
    _, data = get_data(plotfile)
    rho = np.squeeze(data["boxlib", "implicit_mhd_mass_density"].value)
    ue = np.squeeze(data["boxlib", "implicit_mhd_electron_energy"].value)
    ei = np.squeeze(data["boxlib", "implicit_mhd_ion_energy"].value)
    masses.append(np.sum(radial_weight[:, None] * rho))
    masses[-1] = float(masses[-1])
    energies.append(float(np.sum(radial_weight[:, None] * (ue + ei))))

masses = np.array(masses)
energies = np.array(energies)
print("mass ratios:  ", masses / masses[0])
print("energy ratios:", energies / energies[0])

# momentum-density diagnostic views: the rho0 u_z profile at t = 0
# (~1e-9 relative parser-vs-numpy tanh roundoff)
_, initial = get_data(plotfiles[0])
mom_z = np.squeeze(initial["boxlib", "implicit_mhd_momentum_z"].value)
mom_r = np.squeeze(initial["boxlib", "implicit_mhd_momentum_r"].value)
expected = rho0 * u_dir * u0 * np.tanh(z_centers / w)
np.testing.assert_allclose(
    mom_z, np.broadcast_to(expected, (nr, nz)), rtol=1.0e-6, atol=0.0
)
np.testing.assert_allclose(mom_r, 0.0, atol=1.0e-9 * rho0 * u0)

if mode == "drain":
    # outward-only z-face fluxes: totals strictly monotone decreasing at
    # every interval (the r wall is PEC no-flow, so any increase would
    # be end-face injection), plus a substantial overall drain (measured
    # 0.48/0.25 of the initial mass/energy remain; the plain Neumann
    # ends hold the energy cork closed)
    interval_tolerance = 1.0e-12
    assert np.all(np.diff(masses) / masses[0] < interval_tolerance), (
        f"end faces injected mass: {masses / masses[0]}"
    )
    assert np.all(np.diff(energies) / energies[0] < interval_tolerance), (
        f"end faces injected energy: {energies / energies[0]}"
    )
    assert masses[-1] / masses[0] < 0.75, "outflow ends failed to drain mass"
    assert energies[-1] / energies[0] < 0.6, (
        "outflow ends failed to drain energy (the Neumann energy cork)"
    )
else:
    # interior receding from both ends: the plain Neumann ghosts comove
    # inward and inject at interior values (the measured Neumann twin
    # gains +15.4% mass PER INTERVAL, +77% by the last snapshot); the
    # rectified ghosts are held, and only the small impedance-matched
    # reservoir expansion of the held ghost gas enters transiently
    # (measured peak +2.0%) before the midplane bounce reverses the end
    # flow and the ends drain net mass (measured final ratio 0.86)
    assert np.max(masses / masses[0]) < 1.05, (
        f"rectified ends injected like Neumann: {masses / masses[0]}"
    )
    assert np.max(energies / energies[0]) < 1.06, (
        f"rectified ends injected energy: {energies / energies[0]}"
    )
    assert masses[-1] / masses[0] < 1.0, (
        "no net drain after the bounce; ends are not outflow"
    )
    assert masses[-1] / masses[0] > 0.7, (
        "no-injection run lost far more than the rarefaction drain"
    )

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert 1 <= newton_history[-1][2] <= 12

print("PASS")
