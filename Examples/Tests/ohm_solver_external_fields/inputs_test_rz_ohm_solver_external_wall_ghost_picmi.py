#!/usr/bin/env python3
#
# --- Test that the external-field registers of the hybrid-PIC model carry
# --- exact values in the non-periodic (r_max) domain ghost cells. The A_ext
# --- ghosts hold exact analytic parser values, so the ghost rings of
# --- <name>_curlAext and of hybrid_{E,B}_fp_external must match the discrete
# --- curl / scaled A evaluated from the analytic vector potential.
# --- FillBoundary cannot reach non-periodic domain ghosts, so without the
# --- grown-box evaluation in ComputeCurlA / AddExternalFieldFromVectorPotential
# --- these rings stay at their allocation value (zero) and every wall-adjacent
# --- stencil that reads them sees a spurious, drive-scaled field jump.

import numpy as np
from mpi4py import MPI as mpi

from pywarpx import picmi

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

# External field: A_theta = r/2 * (B0 + Bt*cos(kz*z)), periodic in z.
# Discrete curl: Bz(i+1/2,j) = (B0 + Bt*cos(kz*z_j)) exactly (A is linear in r);
# Br(i,j+1/2) = -(A(r_i,z_{j+1}) - A(r_i,z_j))/dz exactly (two-point stencil).
B0 = 0.05  # T
Bt = 0.02  # T
KZ = 2.0 * np.pi / LZ
TAU_D = 1.0e-3  # s, drive ramp time scale: f(t) = 1 + t/TAU_D

# Plasma parameters (static, uniform; the plasma is incidental to this test)
N0 = 1.0e18  # m^-3
T_I = 1.0  # eV
T_E = 1.0  # eV
NPPC = 4

DT = 1.0e-10  # s, far below all dynamical time scales
MAX_STEPS = 2

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
    "tilt": {
        "Ax_external_function": "0",
        "Ay_external_function": "x/2*(B0d + Btd*cos(kzd*z))",
        "Az_external_function": "0",
        "A_time_external_function": "1.0 + t/tau_d",
    },
}

solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=5.0 / 3.0,
    Te=T_E,
    n0=N0,
    n_floor=0.01 * N0,
    plasma_resistivity=1.0e-6,
    substeps=4,
    A_external=A_ext,
    B0d=B0,
    Btd=Bt,
    kzd=KZ,
    tau_d=TAU_D,
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

simulation.step()


def a_theta(r, z):
    """Analytic A_theta at unit drive scale."""
    return 0.5 * r * (B0 + Bt * np.cos(KZ * z))


def ghost_ring_errors(mf, prediction, ghost_i, nodal_z):
    """Compare the r_max domain-ghost ring of mf (component 0) against
    prediction(z_positions). Returns (n_wall_boxes, max_abs_err, max_abs_pred).

    ghost_i: global radial index of the first domain ghost ring.
    nodal_z: True if the field is nodal in z (z_j = z_lo + j*DZ), else
             cell-centered (z_j = z_lo + (j+0.5)*DZ).
    """
    ng = mf.n_grow_vect
    n_wall = 0
    max_err = 0.0
    max_pred = 0.0
    z_lo_domain = -LZ / 2.0
    for mfi in mf:
        vb = mfi.validbox()
        # Only FABs whose valid region touches the radial domain edge own
        # the domain-ghost ring; interior box ghosts are exchange ghosts.
        if vb.big_end[0] != ghost_i - 1:
            continue
        n_wall += 1
        arr = np.array(mf.array(mfi), copy=False)  # [comp, k, j, i]
        lo_i = vb.small_end[0] - ng[0]
        lo_j = vb.small_end[1] - ng[1]
        ii = ghost_i - lo_i
        for j in range(vb.small_end[1], vb.big_end[1] + 1):
            jj = j - lo_j
            z_j = z_lo_domain + (j if nodal_z else (j + 0.5)) * DZ
            pred = prediction(z_j)
            err = abs(arr[0, 0, jj, ii] - pred)
            max_err = max(max_err, err)
            max_pred = max(max_pred, abs(pred))
    return n_wall, max_err, max_pred


# The last external-field refresh happened at t_new with a centered interval,
# and the time function is linear, so the realized scales are exact:
t_new = simulation.extension.warpx.gett_new(0)
s_B = 1.0 + t_new / TAU_D
s_E = -1.0 / TAU_D

fields = simulation.fields

# B_z: (cell r, node z). Valid i in [0, NR-1]; first domain ghost ring i = NR.
# Discrete curl of the linear-in-r A is exact: Bz = s_B * (B0 + Bt*cos(KZ z)).
bz = fields.get("hybrid_B_fp_external", dir="z", level=0)
n_bz, err_bz, pred_bz = ghost_ring_errors(
    bz, lambda z: s_B * (B0 + Bt * np.cos(KZ * z)), NR, nodal_z=True
)

# B_r: (node r, node... cell z). Valid i in [0, NR]; first domain ghost i = NR+1.
# Two-point z stencil of the analytic A at r = (NR+1)*DR.
r_ghost_nodal = (NR + 1) * DR


def br_pred(z_cell):
    z_j = z_cell - 0.5 * DZ
    z_jp1 = z_cell + 0.5 * DZ
    return -s_B * (a_theta(r_ghost_nodal, z_jp1) - a_theta(r_ghost_nodal, z_j)) / DZ


br = fields.get("hybrid_B_fp_external", dir="r", level=0)
n_br, err_br, pred_br = ghost_ring_errors(br, br_pred, NR + 1, nodal_z=False)

# E_theta: (node r, node z). Valid i in [0, NR]; first domain ghost i = NR+1.
# E_ext = -(df/dt) * A with the exact linear-slope scale.
et = fields.get("hybrid_E_fp_external", dir="theta", level=0)
n_et, err_et, pred_et = ghost_ring_errors(
    et, lambda z: s_E * a_theta(r_ghost_nodal, z), NR + 1, nodal_z=True
)

# Reduce across ranks: every wall-adjacent FAB must carry the exact ring.
totals = np.zeros(3, dtype=np.int64)
errors = np.zeros(3)
preds = np.zeros(3)
comm.Allreduce(np.array([n_bz, n_br, n_et], dtype=np.int64), totals, op=mpi.SUM)
comm.Allreduce(np.array([err_bz, err_br, err_et]), errors, op=mpi.MAX)
comm.Allreduce(np.array([pred_bz, pred_br, pred_et]), preds, op=mpi.MAX)

if comm.rank == 0:
    print(f"wall-adjacent boxes checked (Bz, Br, Et): {totals}")
    print(f"max ghost-ring abs errors  (Bz, Br, Et): {errors}")
    print(f"max ghost-ring |prediction| (Bz, Br, Et): {preds}")

# Sanity: the decomposition must actually contain wall-adjacent boxes and the
# predictions must be far from zero, otherwise the test tests nothing.
assert np.all(totals > 0), "no wall-adjacent boxes found in the decomposition"
assert preds[0] > 0.5 * B0, "Bz ghost prediction unexpectedly small"
assert preds[1] > 1.0e-4, "Br ghost prediction unexpectedly small"
assert preds[2] > 1.0e-4 * abs(s_E) * B0 * LR, "Et ghost prediction unexpectedly small"

# The B rings are filled by exact expressions (linear-in-r discrete curl,
# exact two-point z stencil): agreement to roundoff. The E ring goes through
# the centered finite difference of the time function, whose evaluation at
# t +- dt/4 near f = 1 cancels ~dt/(2*tau_d) relative headroom, leaving a
# relative error of order eps_machine/(dt/(2*tau_d)) ~ 1e-9 here - hence the
# looser E tolerance. Unfixed code leaves all rings at zero, failing by
# 100 percent of the prediction.
rtol_b = 1.0e-12
rtol_e = 1.0e-7
assert errors[0] < rtol_b * preds[0], f"Bz ghost ring wrong: {errors[0]} vs {preds[0]}"
assert errors[1] < rtol_b * max(preds[1], 1e-30), (
    f"Br ghost ring wrong: {errors[1]} vs {preds[1]}"
)
assert errors[2] < rtol_e * max(preds[2], 1e-30), (
    f"Et ghost ring wrong: {errors[2]} vs {preds[2]}"
)

if comm.rank == 0:
    print("external wall-ghost test PASSED")
