#!/usr/bin/env python3
#
# --- Test the runtime piecewise-linear scale segments of the hybrid-PIC
# --- external vector potential (external_vector_potential.<name>.python_scale).
# --- A uniform-Bz unit field (A_theta = B0*r/2, exactly representable by the
# --- discrete curl) is scale-driven from a beforestep callback along a linear
# --- ramp s(t) = s0 + t/tau_r. The realized B_ext must equal s(t)*B0 to
# --- roundoff everywhere including the r_max domain-ghost ring, and E_ext
# --- must carry the segment's EXACT constant slope, -ds/dt * A_theta, with no
# --- finite-difference error (the discrete Faraday partner of the linear
# --- B scale). The initial_scale must be realized before any segment is
# --- pushed.

import numpy as np
from mpi4py import MPI as mpi

from pywarpx import callbacks, picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=False)

# Geometry
NR = 32
NZ = 32
LR = 0.1  # m
LZ = 0.2  # m
DR = LR / NR
DZ = LZ / NZ

B0 = 0.05  # T, unit-field strength at scale 1
S0 = 0.25  # initial_scale (pre-ramp hold)
TAU_R = 2.0e-9  # s, ramp time scale: s(t) = S0 + t/TAU_R

N0 = 1.0e18  # m^-3
T_I = 1.0  # eV
NPPC = 4

DT = 1.0e-10  # s
MAX_STEPS = 4

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

A_ext = {
    "coil": {
        "Ax_external_function": "0",
        "Ay_external_function": "B0d*x/2",
        "Az_external_function": "0",
        "python_scale": True,
        "initial_scale": S0,
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
    B0d=B0,
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

libwarpx = simulation.extension
warpx = libwarpx.warpx


def s_of_t(t):
    return S0 + t / TAU_R


# The initial refresh (at t = 0, before any segment is pushed) must realize
# initial_scale exactly.
s_init = warpx.get_external_vector_potential_scale("coil", 0.0)
assert abs(s_init - S0) < 1.0e-15, f"initial_scale not realized: {s_init} vs {S0}"


def push_segment():
    """beforestep: push the coupling interval's exact linear segment."""
    t0 = warpx.gett_new(0)
    t1 = t0 + warpx.getdt(0)
    warpx.set_external_vector_potential_scale("coil", s_of_t(t0), s_of_t(t1), t0, t1)


callbacks.installbeforestep(push_segment)

simulation.step()


def field_errors(mf, prediction_of_rz, nodal_r, nodal_z, ghost_i):
    """Max abs error of component 0 against prediction(r, z) over every valid
    point of wall-adjacent FABs plus their r_max domain-ghost ring at
    global radial index ghost_i. Returns (n_wall_boxes, max_err, max_pred)."""
    ng = mf.n_grow_vect
    n_wall = 0
    max_err = 0.0
    max_pred = 0.0
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
                pred = prediction_of_rz(r_i, z_j)
                err = abs(arr[0, 0, j - lo_j, i - lo_i] - pred)
                max_err = max(max_err, err)
                max_pred = max(max_pred, abs(pred))
    return n_wall, max_err, max_pred


# The final refresh of the last step ran at t_new on the pushed segment, so
# the realized scales are exact linear values with the exact slope.
t_new = warpx.gett_new(0)
s_B = s_of_t(t_new)
slope = 1.0 / TAU_R

fields = simulation.fields

# GetScale must return the exact segment value.
s_get = warpx.get_external_vector_potential_scale("coil", t_new)
assert abs(s_get - s_B) < 1.0e-14 * abs(s_B), f"GetScale wrong: {s_get} vs {s_B}"

# B_z: uniform-field discrete curl is exact -> s_B * B0 everywhere including
# the r_max domain-ghost ring (i = NR for the cell-centered-in-r Bz).
bz = fields.get("hybrid_B_fp_external", dir="z", level=0)
n_bz, err_bz, pred_bz = field_errors(
    bz, lambda r, z: s_B * B0, nodal_r=False, nodal_z=True, ghost_i=NR
)

# E_theta: EXACT constant-slope partner, -ds/dt * A_theta = -slope*B0*r/2,
# with no finite-difference approximation (i = NR+1 ghost ring for nodal r).
et = fields.get("hybrid_E_fp_external", dir="theta", level=0)
n_et, err_et, pred_et = field_errors(
    et,
    lambda r, z: -slope * B0 * r / 2.0,
    nodal_r=True,
    nodal_z=True,
    ghost_i=NR + 1,
)

totals = np.zeros(2, dtype=np.int64)
errors = np.zeros(2)
preds = np.zeros(2)
comm.Allreduce(np.array([n_bz, n_et], dtype=np.int64), totals, op=mpi.SUM)
comm.Allreduce(np.array([err_bz, err_et]), errors, op=mpi.MAX)
comm.Allreduce(np.array([pred_bz, pred_et]), preds, op=mpi.MAX)

if comm.rank == 0:
    print(f"wall-adjacent boxes (Bz, Et): {totals}")
    print(f"max abs errors      (Bz, Et): {errors}")
    print(f"max |prediction|    (Bz, Et): {preds}")

assert np.all(totals > 0), "no wall-adjacent boxes found in the decomposition"
assert preds[0] > 0.1 * B0 and preds[1] > 0.0, "predictions unexpectedly small"

# Both the linear B scale and its exact-slope E partner are realized to
# roundoff -- this is the property the finite-difference parser path cannot
# deliver (its E scale carries an O(eps/(dt/tau)) cancellation error).
rtol = 1.0e-12
assert errors[0] < rtol * preds[0], f"Bz_ext wrong: {errors[0]} vs {preds[0]}"
assert errors[1] < rtol * preds[1], f"Et_ext wrong: {errors[1]} vs {preds[1]}"

if comm.rank == 0:
    print("scale segment test PASSED")
