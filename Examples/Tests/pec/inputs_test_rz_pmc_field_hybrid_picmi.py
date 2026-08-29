#!/usr/bin/env python3
#
# --- Test the reflecting (PMC) field boundary in the hybrid-PIC solver.
# --- A torsional mode with B_theta nodes on the two z faces is set up in
# --- a uniform plasma; the PMC parity must hold the nodes at the faces
# --- through the substepped B-field advance.

import sys

import numpy as np

from pywarpx import picmi

constants = picmi.constants

# The restarted arm of the test pair passes amr.restart='<chk>' on the
# command line; a restart must re-establish the ghost parity that the
# from-scratch initialization seeds.
restart_from = None
for _arg in sys.argv[1:]:
    if _arg.startswith("amr.restart="):
        restart_from = _arg.split("=", 1)[1].strip("'\"")

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
B1 = 0.05 * B0
kz = np.pi / Lz  # B_theta ~ cos(kz z) has nodes on both z faces

grid = picmi.CylindricalGrid(
    number_of_cells=[Nr, Nz],
    warpx_max_grid_size=Nz,
    lower_bound=[0.0, -Lz / 2.0],
    upper_bound=[Lr, Lz / 2.0],
    lower_boundary_conditions=["none", "neumann"],
    upper_boundary_conditions=["dirichlet", "neumann"],
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


def load_initial_field():
    # The RZ parser path hardcodes initial Br/Bt to zero, so the mode is
    # written directly into the evolved B field.
    from pywarpx import fields

    for wrapper, profile in (
        (fields.ByFPExternalWrapper(), "mode"),
        (fields.BzFPExternalWrapper(), "uniform"),
    ):
        arr = wrapper[...]
        nr, nz = arr.shape[0], arr.shape[1]
        r = (np.arange(nr) + (0.0 if nr == Nr + 1 else 0.5)) * (Lr / Nr)
        z = -Lz / 2.0 + (np.arange(nz) + (0.0 if nz == Nz + 1 else 0.5)) * (Lz / Nz)
        R, Z = np.meshgrid(r, z, indexing="ij")
        vals = B1 * (R / Lr) * np.cos(kz * Z) if profile == "mode" else B0 + 0.0 * R
        wrapper[...] = vals.reshape(arr.shape)


sim.add_applied_field(
    picmi.LoadInitialFieldFromPython(
        load_from_python=load_initial_field,
        load_E=False,
    )
)

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
sim.add_diagnostic(picmi.Checkpoint(name="chk", period=75))

if restart_from is not None:
    sim.amr_restart = restart_from

sim.step(150)

# Save the z_lo ghost/valid B_theta pair: the analysis checks that the
# ghosts carry the reflecting parity (ghost = -mirror) after the
# substepped advance, not values frozen at initialization.
from mpi4py import MPI  # noqa: E402

from pywarpx import fields  # noqa: E402

Bt_w = fields.ByFPWrapper()
valid = Bt_w[:, :]
ghost = Bt_w[:, -1j]
valid, ghost = np.squeeze(valid), np.squeeze(ghost)
# cell-centered in z: ghost(-1) mirrors valid(0); nodal: valid(1)
mirror = valid[:, 0] if valid.shape[1] == Nz else valid[:, 1]
if MPI.COMM_WORLD.rank == 0:
    np.save("pmc_ghost_pair.npy", np.stack([ghost, mirror]))
