#!/usr/bin/env python3
#
# --- Test the open field boundary (zero-gradient ghost continuation).
# --- A uniform axial field threads both z faces of a uniform plasma; the
# --- open continuation is exact for it, so any face-localized artifact
# --- is a regression in the ghost fill.

import numpy as np

from pywarpx import picmi

constants = picmi.constants

# Plasma and field parameters (as in the ohm_solver_em_modes tests)
B0 = 0.5
beta = 0.01
m_ion = 400.0
vA_over_c = 5e-3

M = m_ion * constants.m_e
vA = vA_over_c * constants.c
n0 = (B0 / vA) ** 2 / (constants.mu0 * (M + constants.m_e))
w_ci = constants.q_e * B0 / M
t_ci = 2.0 * np.pi / w_ci
w_pi = np.sqrt(constants.q_e**2 * n0 / (M * constants.ep0))
l_i = constants.c / w_pi
v_ti = np.sqrt(beta / 2.0) * vA

Nr, Nz = 32, 64
Lr, Lz = Nr * 0.4 * l_i, Nz * 0.4 * l_i

grid = picmi.CylindricalGrid(
    number_of_cells=[Nr, Nz],
    warpx_max_grid_size=Nz,
    lower_bound=[0.0, -Lz / 2.0],
    upper_bound=[Lr, Lz / 2.0],
    lower_boundary_conditions=["none", "neumann"],
    upper_boundary_conditions=["dirichlet", "neumann"],
    # reflecting particles keep the plasma uniform so the field solution
    # stays exactly static; the open boundary is exercised by the fields
    lower_boundary_conditions_particles=["none", "reflecting"],
    upper_boundary_conditions_particles=["reflecting", "reflecting"],
)

sim = picmi.Simulation(
    solver=picmi.HybridPICSolver(
        grid=grid,
        Te=0.0,
        n0=n0,
        plasma_resistivity=5e-4,
        substeps=40,
        n_floor=0.05 * n0,
    ),
    time_step_size=0.02 * t_ci,
    max_steps=150,
    verbose=0,
    warpx_serialize_initial_conditions=True,
    warpx_current_deposition_algo="direct",
    particle_shape=1,
)

sim.add_applied_field(picmi.AnalyticInitialField(Bz_expression=f"{B0}"))

sim.add_species(
    picmi.Species(
        name="ions",
        charge="q_e",
        mass=M,
        initial_distribution=picmi.UniformDistribution(
            density=n0, rms_velocity=[v_ti] * 3
        ),
    ),
    layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=100),
)

sim.add_diagnostic(
    picmi.FieldDiagnostic(
        name="diag1",
        grid=grid,
        period=150,
        data_list=["B", "E"],
    )
)

sim.initialize_inputs()

# There is no PICMI vocabulary for the open field boundary; set it (and
# the initial div cleaner, which must accept it) on the input buckets.
import pywarpx  # noqa: E402

pywarpx.boundary.field_lo = ["none", "open"]
pywarpx.boundary.field_hi = ["pec", "open"]
pywarpx.warpx.do_initial_div_cleaning = 1

sim.initialize_warpx()
sim.step(150)
