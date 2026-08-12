#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""z-only open-face combination of the RZ Green's-function boundary (see
README.rst): PEC conducting wall at r_hi, open z_lo/z_hi caps.

The ring-current relax of the zcaps gate, but with the radial wall kept
conducting: the caps reconstruct the free-space field of the interior
currents while the wall supplies its image response. This is the plugged
flux-conserver configuration of the FRC hold decks. Checked here:

1. the run boots (the input surface accepts z-only open) and stays finite;
2. the cap-ghost fill is divergence-free to machine precision on the
   pure-psi cells at interior radii (the wall corners belong to the PEC
   mirror and are excluded);
3. the cap-ghost B_theta continuation is the exact copy of the last valid
   plane (here trivially zero);
4. the cap-ghost poloidal field is a smooth free-space continuation: the
   first ghost row agrees with the adjacent valid row to the fill + wall
   image accuracy (the analytic image-free comparison of the zcaps gate
   does not apply against a conducting wall).
"""

import numpy as np

from pywarpx import fields, picmi

constants = picmi.constants
mu0 = constants.mu0

NR, NZ = 32, 32
RMIN, RMAX = 0.0, 1.0
ZMIN, ZMAX = -0.5, 0.5
R0, Z0, A0, J0 = 0.4, 0.1, 0.1, 1.0e3
MAX_STEPS = 170

grid = picmi.CylindricalGrid(
    number_of_cells=[NR, NZ],
    n_azimuthal_modes=1,
    lower_bound=[RMIN, ZMIN],
    upper_bound=[RMAX, ZMAX],
    lower_boundary_conditions=["none", "open"],
    upper_boundary_conditions=["dirichlet", "open"],
    lower_boundary_conditions_particles=["none", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
)

solver = picmi.HybridPICSolver(
    grid=grid,
    Te=1.0,
    n0=1.0e12,
    gamma=1.0,
    n_floor=1.0e12,
    include_hall_term=False,
    include_electron_pressure_term=False,
    plasma_resistivity=mu0,
    substeps=20,
    Jy_external_function=f"{J0}*exp(-((x-{R0})^2 + (z-{Z0})^2)/{A0}^2)",
)

simulation = picmi.Simulation(
    solver=solver,
    time_step_size=6.0e-3,
    max_steps=MAX_STEPS,
    verbose=False,
)

simulation.step(MAX_STEPS)

Br_w = fields.BxFPWrapper()
Bt_w = fields.ByFPWrapper()
Bz_w = fields.BzFPWrapper()
ngr = int(Br_w.mf.n_grow_vect[0])
ngz = int(Br_w.mf.n_grow_vect[1])
Br = np.squeeze(Br_w[()])
Bt = np.squeeze(Bt_w[()])
Bz = np.squeeze(Bz_w[()])

dr = (RMAX - RMIN) / NR
dz = (ZMAX - ZMIN) / NZ

# 1. finite everywhere (including every ghost)
for name, arr in (("Br", Br), ("Bt", Bt), ("Bz", Bz)):
    assert np.all(np.isfinite(arr)), f"{name} has non-finite entries"
b_scale = max(np.abs(Br).max(), np.abs(Bz).max())
assert b_scale > 0.0, "field did not develop"

# 2. div B at machine precision on the pure-psi cap-ghost cells at interior
# radii (all faces psi-derived: ghost rows past the first). The wall
# column i = NR-1 and everything beyond it are excluded -- their outer
# radial face carries the PEC mirror, which is applied after the cap fill
# and owns the wall/corner values.
r_nodes = dr * np.arange(NR + 1)


def divB_cells(i_cc, j_cc):
    II, JJ = np.meshgrid(i_cc, j_cc, indexing="ij")
    rc = (II + 0.5) * dr
    return (
        r_nodes[II + 1] * Br[np.ix_(II[:, 0] + 1 + ngr, JJ[0] + ngz)]
        - r_nodes[II] * Br[np.ix_(II[:, 0] + ngr, JJ[0] + ngz)]
    ) / (rc * dr) + (
        Bz[np.ix_(II[:, 0] + ngr, JJ[0] + 1 + ngz)]
        - Bz[np.ix_(II[:, 0] + ngr, JJ[0] + ngz)]
    ) / dz


i_int = np.arange(0, NR - 1)
deep_j = np.array([j for j in range(-ngz, 0) if j != -1]
                  + [j for j in range(NZ, NZ + ngz) if j != NZ])
if deep_j.size > 0:
    div_deep = np.abs(divB_cells(i_int, deep_j)) * dr / b_scale
    print(f"max |div B| dr/|B| on deep cap-ghost cells: {div_deep.max():.3e}")
    assert div_deep.max() < 1.0e-11, f"cap-ghost div B: {div_deep.max():.3e}"

# 3. exact B_theta continuation at interior radii
for j, jj in [(jg, 0) for jg in range(-ngz, 0)] + [
    (jg, NZ - 1) for jg in range(NZ, NZ + ngz)
]:
    assert np.array_equal(Bt[ngr : ngr + NR, ngz + j], Bt[ngr : ngr + NR, ngz + jj]), (
        f"Bt cap ghost row {j} is not the exact continuation"
    )

# 4. the first cap-ghost rows continue the interior field smoothly
for j_ghost, j_valid in ((-1, 0), (NZ, NZ - 1)):
    jump = np.abs(
        Br[ngr : ngr + NR + 1, ngz + j_ghost] - Br[ngr : ngr + NR + 1, ngz + j_valid]
    ).max()
    print(f"cap ghost/valid Br jump at j = {j_ghost}: {jump / b_scale:.3e}")
    assert jump / b_scale < 0.05, f"cap ghost row {j_ghost} tears the field"

print("z-only open-cap combination gate PASSED")
