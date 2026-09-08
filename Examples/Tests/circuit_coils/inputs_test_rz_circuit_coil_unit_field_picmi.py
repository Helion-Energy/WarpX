#!/usr/bin/env python3
#
# --- Test the C++ coil unit-field fill (circuit.coils -> <name>_Aext):
# --- a filament drive coil OUTSIDE the domain wall (the meshless-coil
# --- model). The filled A_theta must match a NumPy replica of the
# --- discrete-convention kernel (quarter-offset filament, Legendre-
# --- parameter clip) at the coil's reference amp-turns, on the valid
# --- region AND the r_max domain-ghost ring, and the split-field B_z
# --- register must match the discrete curl of that replica at drive
# --- scale 1.

import numpy as np
from mpi4py import MPI as mpi
from scipy.special import ellipe, ellipk

from pywarpx import picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=False)

NR = 32
NZ = 32
LR = 0.1  # m
LZ = 0.2  # m
DR = LR / NR
DZ = LZ / NZ

R_COIL = 0.25  # m (outside the domain wall at r = 0.1 m)
Z_COIL = 0.02  # m
N_TURNS = 4.0
I_REF = 2.0e4  # A

N0 = 1.0e18
T_I = 1.0
NPPC = 4
DT = 1.0e-10
MAX_STEPS = 1

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
simulation.max_steps = MAX_STEPS
simulation.current_deposition_algo = "direct"
simulation.particle_shape = 1
simulation.verbose = False

# The coil pairs with this external field; the C++ fill overwrites the
# (default zero) A functions with the ring-kernel unit field at I_ref.
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

# Coil geometry through the generic input bucket (a dedicated picmi object
# arrives with the coupling engine); written after initialize_inputs so no
# solver bucket write can clobber it.
import pywarpx  # noqa: E402

circuit = pywarpx.warpx.get_bucket("circuit")
circuit.coils = "drive"
circuit.add_new_attr("drive.r", R_COIL)
circuit.add_new_attr("drive.z", Z_COIL)
circuit.add_new_attr("drive.n_turns", N_TURNS)
circuit.add_new_attr("drive.I_ref", I_REF)

simulation.initialize_warpx()

# ---------------------------------------------------------------------------


def yee_loop_a_theta(r, z, r_off, z_off):
    """NumPy replica of the discrete-convention loop A_theta per amp."""
    mu0 = 4.0e-7 * np.pi
    d2 = (r + r_off) ** 2 + (z - z_off) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        m = 4.0 * r * r_off / d2
    m = np.clip(m, 1e-15, 1.0 - 1e-12)
    f = (2.0 - m) * ellipk(m) - 2.0 * ellipe(m)
    psi = mu0 / (4.0 * np.pi) * 4.0 * r * r_off / np.sqrt(d2) * f / m
    return np.where(r > 0.0, psi / np.maximum(r, 1e-300), 0.0)


r_off = R_COIL + 0.25 * DR
z_off = Z_COIL + 0.25 * DZ
AMPS = I_REF * N_TURNS

fields = simulation.fields


def scan(mf, prediction_of_rz, nodal_r, nodal_z, ghost_i):
    ng = mf.n_grow_vect
    n_wall = 0
    max_err = 0.0
    max_ref = 0.0
    z_lo_domain = -LZ / 2.0
    for mfi in mf:
        vb = mfi.validbox()
        arr = np.array(mf.array(mfi), copy=False)
        lo_i = vb.small_end[0] - ng[0]
        lo_j = vb.small_end[1] - ng[1]
        i_list = list(range(vb.small_end[0], vb.big_end[0] + 1))
        if vb.big_end[0] == ghost_i - 1:
            n_wall += 1
            i_list.append(ghost_i)
        for j in range(vb.small_end[1], vb.big_end[1] + 1):
            z_j = z_lo_domain + (j if nodal_z else (j + 0.5)) * DZ
            for i in i_list:
                r_i = (i if nodal_r else (i + 0.5)) * DR
                ref = prediction_of_rz(r_i, z_j)
                max_err = max(max_err, abs(arr[0, 0, j - lo_j, i - lo_i] - ref))
                max_ref = max(max_ref, abs(ref))
    return n_wall, max_err, max_ref


# 1) The filled A_theta vs the replica (valid + r_max domain ghost ring).
at = fields.get("drive_Aext", dir="theta", level=0)
n_a, err_a, ref_a = scan(
    at,
    lambda r, z: AMPS * yee_loop_a_theta(r, z, r_off, z_off),
    nodal_r=True,
    nodal_z=True,
    ghost_i=NR + 1,
)

# 2) The split-field B_z register vs the discrete curl of the replica at
# drive scale 1: Bz(i+1/2,j) = (r_{i+1} A_{i+1,j} - r_i A_{i,j})/(r_{i+1/2} dr).
def bz_replica(r_c, z):
    i = r_c / DR - 0.5
    r_lo = i * DR
    r_hi = (i + 1.0) * DR
    a_lo = AMPS * yee_loop_a_theta(r_lo, z, r_off, z_off)
    a_hi = AMPS * yee_loop_a_theta(r_hi, z, r_off, z_off)
    return (r_hi * a_hi - r_lo * a_lo) / (r_c * DR)


bz = fields.get("hybrid_B_fp_external", dir="z", level=0)
n_b, err_b, ref_b = scan(bz, bz_replica, nodal_r=False, nodal_z=True, ghost_i=NR)

totals = np.zeros(2, dtype=np.int64)
errors = np.zeros(2)
refs = np.zeros(2)
comm.Allreduce(np.array([n_a, n_b], dtype=np.int64), totals, op=mpi.SUM)
comm.Allreduce(np.array([err_a, err_b]), errors, op=mpi.MAX)
comm.Allreduce(np.array([ref_a, ref_b]), refs, op=mpi.MAX)

if comm.rank == 0:
    print(f"wall boxes (A, Bz): {totals}")
    print(f"max abs err (A, Bz): {errors}")
    print(f"max |ref|  (A, Bz): {refs}")

assert np.all(totals > 0), "no wall-adjacent boxes found"
assert refs[0] > 0.0 and refs[1] > 0.0
# scipy-vs-AGM elliptic differences and the series branch sit at ~1e-12 rel
rtol = 1.0e-10
assert errors[0] < rtol * refs[0], f"A_theta fill wrong: {errors[0]} vs {refs[0]}"
assert errors[1] < rtol * refs[1], f"Bz_ext wrong: {errors[1]} vs {refs[1]}"

if comm.rank == 0:
    print("coil unit-field test PASSED")
