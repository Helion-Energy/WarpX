#!/usr/bin/env python3
#
# --- Conservation test for the ScrapedParticleEnergy reduced diagnostic.
# --- Uniform ions drift along a uniform B_z under the hybrid-PIC solver
# --- with Te0 = 0 and eta = 0, which makes the generalized-Ohm E field
# --- identically zero (v_e || B so v_e x B = 0; J || B so the Hall term
# --- vanishes; grad Pe = 0 with Te0 = 0; no resistive term), so no field
# --- work is done on the ions: the kinetic energy leaving the live
# --- population through the absorbing z boundaries must reappear exactly
# --- in the scraped-energy tally,
# ---     KE_live(t) + KE_scraped_cumulative(t) = const
# --- to deposition-independent roundoff, at every step.

from pywarpx import picmi

constants = picmi.constants

n0 = 1.0e18  # m^-3
B0 = 0.1  # T
m_ion = 4.0 * constants.m_p

v_drift = 1.0e5  # m/s, along +z
v_th = 1.0e4  # m/s

NX = 16
NZ = 32
LX = 0.5
LZ = 1.0

DT = 5.0e-8
MAX_STEPS = 20

grid = picmi.Cartesian2DGrid(
    number_of_cells=[NX, NZ],
    lower_bound=[-LX / 2.0, -LZ / 2.0],
    upper_bound=[LX / 2.0, LZ / 2.0],
    lower_boundary_conditions=["periodic", "neumann"],
    upper_boundary_conditions=["periodic", "neumann"],
    lower_boundary_conditions_particles=["periodic", "absorbing"],
    upper_boundary_conditions_particles=["periodic", "absorbing"],
    warpx_max_grid_size=32,
)

solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=5.0 / 3.0,
    Te=0.0,
    n0=n0,
    n_floor=0.05 * n0,
    plasma_resistivity=0.0,
    substeps=4,
)

initial_B = picmi.AnalyticInitialField(
    Bx_expression="0",
    By_expression="0",
    Bz_expression=f"{B0}",
)

ions = picmi.Species(
    name="ions",
    charge="q_e",
    mass=m_ion,
    # Perpendicular-cold beam: u_perp = 0 exactly, so the deposited J is
    # parallel to B to roundoff and every Ohm term vanishes (see header).
    # A perpendicular thermal spread would seed noise-level whistler
    # dynamics that the eta = 0 adaptive B substepping cannot damp.
    initial_distribution=picmi.UniformDistribution(
        density=n0,
        directed_velocity=[0.0, 0.0, v_drift],
        rms_velocity=[0.0, 0.0, v_th],
    ),
    warpx_save_particles_at_zlo=True,
    warpx_save_particles_at_zhi=True,
)

sim = picmi.Simulation(
    solver=solver,
    time_step_size=DT,
    max_steps=MAX_STEPS,
    verbose=1,
    particle_shape=1,
    warpx_serialize_initial_conditions=True,
    warpx_random_seed=20260813,
)

sim.add_applied_field(initial_B)
sim.add_species(
    ions,
    layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=4),
)

live_energy = picmi.ReducedDiagnostic(
    diag_type="ParticleEnergy",
    name="live_energy",
    period=1,
)
sim.add_diagnostic(live_energy)

scraped_energy = picmi.ReducedDiagnostic(
    diag_type="ScrapedParticleEnergy",
    name="scraped_energy",
    period=1,
)
sim.add_diagnostic(scraped_energy)

sim.step()
