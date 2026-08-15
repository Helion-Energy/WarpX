#!/usr/bin/env python3
#
# --- Repeated-Evolve regression of the split external-field bookkeeping:
# --- Python drivers legitimately call simulation.step(n) several times, and
# --- WarpX re-runs its hybrid re-initialization at the first step of every
# --- Evolve() call. Bfield_fp holds the TOTAL field between steps, so the
# --- fresh-start "add the external field to the totals" must happen exactly
# --- once per run: re-adding grows the total by one full external field per
# --- step() call, a runaway that quickly drives the substepped B advance
# --- unstable (values -> 1e208 within an evolve) and from there corrupts
# --- the particle deposits. This deck drives a uniform external
# --- B_z = 0.1 T (A_theta = 0.05 r via the parser path) across three
# --- separate step() calls and pins max |B_z,total| to the external value
# --- after every one of them: on broken bookkeeping the second evolve
# --- already reads ~0.2 T.

import numpy as np
from mpi4py import MPI as mpi

from pywarpx import picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=False)

NR = 32
NZ = 32
LR = 0.1
LZ = 0.2

B0 = 0.1           # T: uniform B_z from A_theta = 0.5*B0*r
N0 = 1.0e18
T_I = 1.0
NPPC = 4
DT = 1.0e-10
STEPS_PER_CALL = 2
N_CALLS = 3

grid = picmi.CylindricalGrid(
    number_of_cells=[NR, NZ],
    lower_bound=[0.0, -LZ / 2.0],
    upper_bound=[LR, LZ / 2.0],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    warpx_max_grid_size=16,
)
simulation.time_step_size = DT
simulation.max_steps = STEPS_PER_CALL * N_CALLS
simulation.current_deposition_algo = "direct"
simulation.particle_shape = 1
simulation.verbose = False

# A_theta = 0.5*B0*r -> B_z = B0 exactly (including through the discrete curl)
A_ext = {
    "uniform": {
        "Ax_external_function": f"-{0.5 * B0}*y",
        "Ay_external_function": f"{0.5 * B0}*x",
        "Az_external_function": "0",
        "A_time_external_function": "1",
    },
}

solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=5.0 / 3.0,
    Te=T_I,
    n0=N0,
    n_floor=0.01 * N0,
    plasma_resistivity=1.0e-6,
    substeps=4,
    A_external=A_ext,
)
simulation.solver = solver

ions = picmi.Species(
    name="ions",
    charge="q_e",
    mass=constants.m_p,
    initial_distribution=picmi.UniformDistribution(
        density=N0,
        rms_velocity=[np.sqrt(T_I * constants.q_e / constants.m_p)] * 3,
    ),
)
simulation.add_species(
    ions,
    layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=NPPC),
)

simulation.initialize_inputs()
simulation.initialize_warpx()


def max_bz_total():
    bz = simulation.fields.get("Bfield_fp", dir="z", level=0)
    local = 0.0
    for mfi in bz:
        arr = np.array(bz.array(mfi), copy=False)
        local = max(local, float(np.max(np.abs(arr))))
    buf = np.zeros(1)
    comm.Allreduce(np.array([local]), buf, op=mpi.MAX)
    return buf[0]


for call in range(N_CALLS):
    simulation.step(STEPS_PER_CALL)
    b = max_bz_total()
    if comm.rank == 0:
        print(f"after step() call {call + 1}: max |B_z,total| = {b:.6e} T")
    assert np.isfinite(b), f"B blew up after step() call {call + 1}"
    # the plasma response at these parameters is tiny; a re-added external
    # field shows up as an integer multiple of B0 immediately
    assert abs(b - B0) < 0.05 * B0, (
        f"external field mis-accumulated after step() call {call + 1}: "
        f"max |B_z| = {b} vs the external {B0}"
    )

if comm.rank == 0:
    print("multi-evolve external-field test PASSED")
