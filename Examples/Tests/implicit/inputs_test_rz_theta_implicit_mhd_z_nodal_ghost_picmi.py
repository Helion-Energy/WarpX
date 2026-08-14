#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Staggering-aware Neumann z-ghost fill of the theta-implicit MHD solver.

ApplyNeumannZDomainGhosts must fill z-domain ghosts with ZERO-GRADIENT
values for both z staggerings: cell-centered-in-z data clamps to the
boundary cell (zero gradient across the boundary face), while
NODAL-in-z data mirrors EVENLY across the boundary node, ghost(jb -+ d)
= f(jb +- d) -- the clamp there would leave a spurious HALF-gradient at
the end node (a centered derivative reads (f_1 - f_0)/(2 dz) instead of
0), feeding e.g. a spurious end-node grad_z of the nodal electron
pressure into the Ohm's-law evaluations.

The run carries a z-LINEAR electron-pressure and mass-density ramp (so
clamp and mirror give different ghosts and the asserts are
discriminating) with a static fluid and uniform Bz. After a step, the
ghost rows are read back directly through the MultiFab register:

* hybrid_electron_pressure_fp (fully nodal): ghosts must EXACTLY mirror
  the interior rows about the boundary node (zero centered gradient at
  the end node).
* implicit_mhd_mass_density (cell-centered): ghosts must EXACTLY clamp
  to the boundary row (the cell-centered fill is bit-identical to the
  pre-fix behavior).
"""

import numpy as np

from pywarpx import picmi

constants = picmi.constants

n0 = 1.0e20
rho0 = n0 * constants.m_p
T0_ev = 50.0
Pe0 = n0 * T0_ev * constants.q_e
gamma = 5.0 / 3.0
B0 = 0.05
# gentle z-linear ramps (z in [-0.5, 0.5] m)
pressure_slope = 0.4
density_slope = 0.2

nr = 8
nz = 32
rmax = 0.1
zmax = 0.5

sound_speed = np.sqrt(gamma * Pe0 / rho0)
alfven_speed = B0 / np.sqrt(constants.mu0 * rho0)
dt = 0.05 * (2.0 * zmax / nz) / np.sqrt(sound_speed**2 + alfven_speed**2)

grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz],
    warpx_max_grid_size=64,
    lower_bound=[0.0, -zmax],
    upper_bound=[rmax, zmax],
    lower_boundary_conditions=["none", "none"],
    upper_boundary_conditions=["dirichlet", "none"],
    lower_boundary_conditions_particles=["none", "absorbing"],
    upper_boundary_conditions_particles=["reflecting", "absorbing"],
)

solver = picmi.HybridPICSolver(
    grid=grid,
    Te=T0_ev,
    n0=n0,
    gamma=gamma,
    n_floor=1.0e10,
    plasma_resistivity=0.0,
    include_hall_term=False,
    include_electron_pressure_term=False,
)

nonlinear_solver = picmi.NewtonNonlinearSolver(
    verbose=True,
    max_iterations=12,
    relative_tolerance=1.0e-8,
    absolute_tolerance=1.0e-11,
)

evolve_scheme = picmi.ThetaImplicitMHDEvolveScheme(
    nonlinear_solver=nonlinear_solver,
    theta=1.0,
    mass_density=f"{rho0}*(1 + {density_slope}*z)",
    electron_pressure=f"{Pe0}*(1 + {pressure_slope}*z)",
    reference_mass_density=rho0,
    reference_magnetic_field=B0,
    gamma_e=gamma,
    gamma_i=gamma,
    reference_ion_pressure=0.0,
    mass_density_floor=1.0e-4 * rho0,
    electron_pressure_floor=1.0e-4 * Pe0,
    fluid_flux="hlld",
)

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=2,
    verbose=1,
)
sim.evolve_scheme = evolve_scheme

initial_field = picmi.AnalyticInitialField(
    Bx_expression=0.0,
    By_expression=0.0,
    Bz_expression=B0,
)
sim.add_applied_field(initial_field)

sim.step(2)


def full_array(name):
    """Return the (r, z) array of a register scalar including ghosts."""
    mf = sim.fields.get(name, level=0)
    arr = np.array(mf[()])
    return np.squeeze(arr), mf.n_grow_vect[1]


# --- nodal-in-z: even mirror about the boundary node -----------------
pe, ng_pe = full_array("hybrid_electron_pressure_fp")
jb_lo = ng_pe  # first valid node in the padded array
jb_hi = ng_pe + nz  # last valid node
# restrict to valid radial columns (radial ghosts are filled by a later
# pass; the identities hold there too, but keep the assert minimal)
pe_valid_r = pe[ng_pe : pe.shape[0] - ng_pe, :]
# the ramp must actually be there (discriminating test)
assert not np.allclose(
    pe_valid_r[:, jb_lo], pe_valid_r[:, jb_lo + 1], rtol=1.0e-6, atol=0.0
), "no pressure gradient at the end node; test misconfigured"
for depth in range(1, ng_pe + 1):
    source_lo = min(jb_lo + depth, jb_hi)
    source_hi = max(jb_hi - depth, jb_lo)
    np.testing.assert_array_equal(
        pe_valid_r[:, jb_lo - depth],
        pe_valid_r[:, source_lo],
        err_msg=f"nodal z_lo ghost depth {depth} is not the even mirror",
    )
    np.testing.assert_array_equal(
        pe_valid_r[:, jb_hi + depth],
        pe_valid_r[:, source_hi],
        err_msg=f"nodal z_hi ghost depth {depth} is not the even mirror",
    )
# zero CENTERED gradient at the end nodes (the pre-fix clamp leaves
# half the ramp slope there)
lo_gradient = pe_valid_r[:, jb_lo + 1] - pe_valid_r[:, jb_lo - 1]
hi_gradient = pe_valid_r[:, jb_hi + 1] - pe_valid_r[:, jb_hi - 1]
np.testing.assert_array_equal(lo_gradient, 0.0)
np.testing.assert_array_equal(hi_gradient, 0.0)

# --- cell-centered-in-z: clamp to the boundary cell (unchanged) ------
rho, ng_rho = full_array("implicit_mhd_mass_density")
jc_lo = ng_rho
jc_hi = ng_rho + nz - 1
rho_valid_r = rho[ng_rho : rho.shape[0] - ng_rho, :]
assert not np.allclose(
    rho_valid_r[:, jc_lo], rho_valid_r[:, jc_lo + 1], rtol=1.0e-6, atol=0.0
), "no density gradient at the end cell; test misconfigured"
for depth in range(1, ng_rho + 1):
    np.testing.assert_array_equal(
        rho_valid_r[:, jc_lo - depth],
        rho_valid_r[:, jc_lo],
        err_msg=f"cell-centered z_lo ghost depth {depth} is not the clamp",
    )
    np.testing.assert_array_equal(
        rho_valid_r[:, jc_hi + depth],
        rho_valid_r[:, jc_hi],
        err_msg=f"cell-centered z_hi ghost depth {depth} is not the clamp",
    )

print("PASS")
