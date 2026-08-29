#!/usr/bin/env python3
#
# --- Test the 3D C++ coil unit-field fill (circuit.coils -> <name>_Aext):
# --- A_x = -A_theta * y/rho, A_y = A_theta * x/rho of a circular filament
# --- (no filament offset in 3D; same Legendre-parameter clip), evaluated at
# --- each component's own Yee staggering, must match a NumPy replica at the
# --- coil's reference amp-turns over every valid point.

import numpy as np
from mpi4py import MPI as mpi
from scipy.special import ellipe, ellipk

from pywarpx import picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=False)

NC = 16
L = 0.2  # m, box edge (domain [-0.1, 0.1]^3)
DX = L / NC

R_COIL = 0.25  # m (outside the domain)
Z_COIL = 0.05  # m
N_TURNS = 2.0
I_REF = 1.0e4  # A

N0 = 1.0e18
T_I = 1.0
NPPC = 2
DT = 1.0e-10

grid = picmi.Cartesian3DGrid(
    number_of_cells=[NC, NC, NC],
    lower_bound=[-L / 2.0, -L / 2.0, -L / 2.0],
    upper_bound=[L / 2.0, L / 2.0, L / 2.0],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
    lower_boundary_conditions_particles=["periodic"] * 3,
    upper_boundary_conditions_particles=["periodic"] * 3,
    warpx_max_grid_size=8,
)
simulation.time_step_size = DT
simulation.max_steps = 1
simulation.current_deposition_algo = "direct"
simulation.particle_shape = 1
simulation.verbose = False

A_ext = {
    "drive": {
        "Ax_external_function": "0",
        "Ay_external_function": "0",
        "Az_external_function": "0",
        "A_time_external_function": "1.0",
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

import pywarpx  # noqa: E402

circuit = pywarpx.warpx.get_bucket("circuit")
circuit.coils = "drive"
circuit.add_new_attr("drive.r", R_COIL)
circuit.add_new_attr("drive.z", Z_COIL)
circuit.add_new_attr("drive.n_turns", N_TURNS)
circuit.add_new_attr("drive.I_ref", I_REF)

simulation.initialize_warpx()

# ---------------------------------------------------------------------------


def yee_loop_a_theta(rho, z):
    mu0 = 4.0e-7 * np.pi
    d2 = (rho + R_COIL) ** 2 + (z - Z_COIL) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        m = 4.0 * rho * R_COIL / d2
    m = np.clip(m, 1e-15, 1.0 - 1e-12)
    f = (2.0 - m) * ellipk(m) - 2.0 * ellipe(m)
    psi = mu0 / (4.0 * np.pi) * 4.0 * rho * R_COIL / np.sqrt(d2) * f / m
    return np.where(rho > 0.0, psi / np.maximum(rho, 1e-300), 0.0)


AMPS = I_REF * N_TURNS
PLO = -L / 2.0

# Yee E staggering per component: which axes are nodal
NODAL = {
    "x": (False, True, True),
    "y": (True, False, True),
}


def scan(mf, comp):
    nodal = NODAL[comp]
    ng = mf.n_grow_vect
    max_err = 0.0
    max_ref = 0.0
    for mfi in mf:
        vb = mfi.validbox()
        arr = np.array(mf.array(mfi), copy=False)
        lo = [vb.small_end[d] - ng[d] for d in range(3)]
        for k in range(vb.small_end[2], vb.big_end[2] + 1):
            z = PLO + (k if nodal[2] else k + 0.5) * DX
            for j in range(vb.small_end[1], vb.big_end[1] + 1):
                y = PLO + (j if nodal[1] else j + 0.5) * DX
                for i in range(vb.small_end[0], vb.big_end[0] + 1):
                    x = PLO + (i if nodal[0] else i + 0.5) * DX
                    rho = np.sqrt(x * x + y * y)
                    a_th = AMPS * yee_loop_a_theta(rho, z) if rho > 0 else 0.0
                    if comp == "x":
                        ref = -a_th * y / rho if rho > 0 else 0.0
                    else:
                        ref = a_th * x / rho if rho > 0 else 0.0
                    val = arr[0, k - lo[2], j - lo[1], i - lo[0]]
                    max_err = max(max_err, abs(val - ref))
                    max_ref = max(max_ref, abs(ref))
    return max_err, max_ref


fields = simulation.fields
ax = fields.get("drive_Aext", dir="x", level=0)
ay = fields.get("drive_Aext", dir="y", level=0)

err_x, ref_x = scan(ax, "x")
err_y, ref_y = scan(ay, "y")

errors = np.zeros(2)
refs = np.zeros(2)
comm.Allreduce(np.array([err_x, err_y]), errors, op=mpi.MAX)
comm.Allreduce(np.array([ref_x, ref_y]), refs, op=mpi.MAX)

if comm.rank == 0:
    print(f"max abs err (Ax, Ay): {errors}")
    print(f"max |ref|   (Ax, Ay): {refs}")

assert refs[0] > 0.0 and refs[1] > 0.0
rtol = 1.0e-10
assert errors[0] < rtol * refs[0], f"Ax fill wrong: {errors[0]} vs {refs[0]}"
assert errors[1] < rtol * refs[1], f"Ay fill wrong: {errors[1]} vs {refs[1]}"

if comm.rank == 0:
    print("3D coil unit-field test PASSED")
