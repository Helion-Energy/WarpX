#!/usr/bin/env python3
#
# --- Resampling must not delete particles that crossed a non-periodic face.
# --- resample() calls Redistribute(), which drops every particle outside the
# --- domain. If resampling ran before the particle boundary conditions, the
# --- particles that crossed a reflecting face during the push would vanish on
# --- each resampling step instead of being reflected. Velocity coincidence
# --- thinning conserves weight exactly, so with reflecting faces the total
# --- weight must stay constant to round-off.

import numpy as np

from pywarpx import callbacks, particle_containers, picmi

constants = picmi.constants

##########################
# numerics parameters
##########################

nx, ny, nz = 8, 8, 16
dz = 1.0e-3
dt = 1.0e-9
max_steps = 30
resampling_interval = 3

# Thermal speed of half a cell per step: many particles cross a face each step
v_th = 0.5 * dz / dt

z_lo, z_hi = 0.0, nz * dz

##########################
# numerics components
##########################

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[0.0, 0.0, z_lo],
    upper_bound=[nx * dz, ny * dz, z_hi],
    lower_boundary_conditions=["periodic", "periodic", "dirichlet"],
    upper_boundary_conditions=["periodic", "periodic", "dirichlet"],
    lower_boundary_conditions_particles=["periodic", "periodic", "reflecting"],
    upper_boundary_conditions_particles=["periodic", "periodic", "reflecting"],
    warpx_max_grid_size=16,
)

solver = picmi.ElectrostaticSolver(
    grid=grid,
    method="Multigrid",
    required_precision=1e-6,
    warpx_self_fields_verbosity=0,
)

##########################
# physics components
##########################

electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    initial_distribution=picmi.UniformDistribution(
        density=1e12, rms_velocity=[v_th, v_th, v_th]
    ),
    warpx_do_not_deposit=1,
    warpx_do_resampling=True,
    warpx_resampling_algorithm="velocity_coincidence_thinning",
    warpx_resampling_algorithm_delta_ur=10.0 * v_th,
    warpx_resampling_algorithm_n_theta=2,
    warpx_resampling_algorithm_n_phi=2,
    warpx_resampling_min_ppc=2,
    warpx_resampling_trigger_intervals=resampling_interval,
)

##########################
# simulation setup
##########################

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=max_steps,
    verbose=1,
)

sim.add_species(
    electrons,
    layout=picmi.PseudoRandomLayout(n_macroparticles_per_cell=32, grid=grid),
)

sim.initialize_inputs()
sim.initialize_warpx()

##########################
# instrumentation
##########################

wrapper = particle_containers.ParticleContainerWrapper("electrons")


def total_weight():
    return sum(np.sum(w) for w in wrapper.get_particle_weight(copy_to_host=True))


def n_outside():
    return sum(
        int(np.sum((z < z_lo) | (z >= z_hi)))
        for z in wrapper.get_particle_z(copy_to_host=True)
    )


weight_0 = total_weight()
count_0 = wrapper.get_particle_count()
weights = {}
outside = {}


def after_step():
    weights[sim.extension.warpx.getistep(lev=0)] = total_weight()


# particlescraper runs after the push and before the boundary conditions
def at_scraper():
    outside[sim.extension.warpx.getistep(lev=0)] = n_outside()


callbacks.installafterstep(after_step)
callbacks.installparticlescraper(at_scraper)

##########################
# simulation run
##########################

sim.step(max_steps)

##########################
# checks
##########################

steps = np.array(sorted(weights))
deviation = np.array([abs(weights[s] - weight_0) / weight_0 for s in steps])
print("max relative weight deviation:", deviation.max())
print("macroparticles:", count_0, "->", wrapper.get_particle_count())
resampling_steps = [s for s in sorted(outside) if s % resampling_interval == 0]
exposed = {s: outside[s] for s in resampling_steps}
print("particles beyond a z face before the boundary step, on resampling steps:")
print(exposed)

assert deviation.max() < 1e-12, steps[deviation >= 1e-12]
assert wrapper.get_particle_count() < count_0
assert sum(exposed.values()) > 0
