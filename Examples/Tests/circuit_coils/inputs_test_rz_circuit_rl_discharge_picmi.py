#!/usr/bin/env python3
#
# --- Test the per-substep circuit coupling engine (circuit.engine =
# --- callbacks) end to end with an exactly integrable stand-in circuit:
# --- a series RL loop discharging through a drive coil in a (near-)vacuum
# --- domain. The Python engine implements the coupling contract
# --- (circuitbeginstep / circuitpredict / circuitcorrect / circuitfinish),
# --- advancing I(t) with the exact exponential integrator per coupling
# --- interval, so the realized coil scale must equal I0 exp(-R t / L) / I_ref
# --- to roundoff at every step regardless of the (adaptive) substep sizes.
# --- The vacuum identity pins the grid: B_ext == s(t) * (discrete curl of
# --- the unit A) everywhere including the r_max domain-ghost ring. Hook
# --- accounting verifies the predictor-corrector loop ran.

import numpy as np
from mpi4py import MPI as mpi
from scipy.special import ellipe, ellipk

from pywarpx import callbacks, picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=False)

NR = 32
NZ = 32
LR = 0.1
LZ = 0.2
DR = LR / NR
DZ = LZ / NZ

R_COIL = 0.25
Z_COIL = 0.0
N_TURNS = 1.0
I0 = 5.0e4          # A, initial circuit current
I_REF = I0          # initial_scale = 1

# Series RL discharge: I(t) = I0 exp(-R t / L)
L_CIRC = 5.0e-6     # H
R_CIRC = 0.5        # Ohm

N0 = 1.0e18
T_I = 1.0
NPPC = 4
DT = 1.0e-10  # below the whistler stability limit of this grid/density
MAX_STEPS = 5

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
    "drive": {
        "Ax_external_function": "0",
        "Ay_external_function": "0",
        "Az_external_function": "0",
        "python_scale": True,
        "initial_scale": 1.0,
    },
}

coil = picmi.CircuitCoil(
    name="drive",
    r=R_COIL,
    z=Z_COIL,
    n_turns=N_TURNS,
    I_ref=I_REF,
    probe="disk",
)
circuit = picmi.CircuitCoupling(
    coils=[coil],
    engine="callbacks",
    corrector_iterations=1,
    corrector_rtol=1.0e-6,
)

solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=5.0 / 3.0,
    Te=T_I,
    n0=N0,
    n_floor=0.01 * N0,
    plasma_resistivity=1.0e-6,
    substeps=4,
    A_external=A_ext,
    circuit=circuit,
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


class SeriesRLEngine:
    """The coupling-contract stand-in: a series RL loop advanced with the
    exact per-interval exponential integrator. The measured plasma linkage
    enters as a back-EMF source (negligible in this vacuum deck; its
    handling exercises the corrector path)."""

    def __init__(self):
        self.i_accepted = I0     # last accepted circuit state
        self.i_step_start = I0
        self.t0_committed = None
        self.i_pending = I0
        self.lam0 = 0.0
        self._held_eps = 0.0     # last corrected EMF estimate [V]
        self.n_begin = 0
        self.n_predict = 0
        self.n_correct = 0
        self.n_finish = 0
        self.max_abs_lambda = 0.0

    def _advance(self, t0, t1, eps):
        # L dI/dt + R I = -eps, with eps = d(lambda_phys)/dt in volts
        tau = L_CIRC / R_CIRC
        decay = np.exp(-(t1 - t0) / tau)
        i_forced = -eps / R_CIRC
        self.i_pending = i_forced + (self.i_accepted - i_forced) * decay
        s_old = self.i_accepted / I_REF
        s_new = self.i_pending / I_REF
        warpx.set_external_vector_potential_scale("drive", s_old, s_new, t0, t1)

    def beginstep(self):
        self.n_begin += 1
        self.i_step_start = self.i_accepted
        self.lam0 = warpx.get_coil_flux_linkage("drive")

    def predict(self):
        self.n_predict += 1
        t0, t1, substep, it = warpx.get_coupling_interval()
        if self.t0_committed is not None and t0 > self.t0_committed:
            # the previous interval was accepted: roll the state forward
            self.i_accepted = self.i_pending
        self.t0_committed = t0
        self.lam0 = warpx.get_coil_flux_linkage("drive")
        self._advance(t0, t1, self._held_eps)

    def correct(self):
        self.n_correct += 1
        t0, t1, substep, it = warpx.get_coupling_interval()
        lam1 = warpx.get_coil_flux_linkage("drive")
        self.max_abs_lambda = max(self.max_abs_lambda, abs(lam1))
        eps = (lam1 - self.lam0) / (t1 - t0) / (I_REF * N_TURNS)
        self._held_eps = eps
        self._advance(t0, t1, eps)

    def finish(self):
        self.n_finish += 1
        # commit the last interval of the step
        self.i_accepted = self.i_pending
        self.t0_committed = None


engine = SeriesRLEngine()
callbacks.installcallback("circuitbeginstep", engine.beginstep)
callbacks.installcallback("circuitpredict", engine.predict)
callbacks.installcallback("circuitcorrect", engine.correct)
callbacks.installcallback("circuitfinish", engine.finish)

simulation.step()

# ---------------------------------------------------------------------------

t_final = warpx.gett_new(0)
s_expect = np.exp(-R_CIRC * t_final / L_CIRC)
s_real = warpx.get_external_vector_potential_scale("drive", t_final)

if comm.rank == 0:
    print(f"hooks: begin {engine.n_begin} predict {engine.n_predict} "
          f"correct {engine.n_correct} finish {engine.n_finish}")
    print(f"scale at t_final: realized {s_real:.15e} expected {s_expect:.15e}")
    print(f"max |lambda| measured: {engine.max_abs_lambda:.3e} Wb*A")

# Hook accounting: one begin/finish per step; at least one substep per step;
# with corrector_iterations = 1 every successful predict gets one correct.
assert engine.n_begin == MAX_STEPS and engine.n_finish == MAX_STEPS
assert engine.n_predict >= MAX_STEPS
assert 0 < engine.n_correct <= engine.n_predict

# The vacuum back-EMF is a numerical residue: far below the drive scale
# lambda ~ L_disc * I0^2-equivalents. Use the coil's own linked flux scale.
lam_scale = 1.4596352993e-6 * I0 * I0  # L_disc(R=0.25) at unit turns, x I_ref
assert engine.max_abs_lambda < 1.0e-3 * lam_scale, (
    f"vacuum linkage unexpectedly large: {engine.max_abs_lambda}"
)

# The exact exponential integrator + exact segment realization: the realized
# scale is the analytic RL discharge to roundoff, independent of the
# (adaptive) substep sizes.
assert abs(s_real - s_expect) < 1.0e-12, f"{s_real} vs {s_expect}"


# Vacuum identity on the grid: B_ext == s(t_final) * discrete-curl(unit A),
# including the r_max domain-ghost ring.
def yee_loop_a_theta(r, z, r_off, z_off):
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


def bz_replica(r_c, z):
    i = r_c / DR - 0.5
    r_lo = i * DR
    r_hi = (i + 1.0) * DR
    a_lo = AMPS * yee_loop_a_theta(r_lo, z, r_off, z_off)
    a_hi = AMPS * yee_loop_a_theta(r_hi, z, r_off, z_off)
    return s_real * (r_hi * a_hi - r_lo * a_lo) / (r_c * DR)


fields = simulation.fields
bz = fields.get("hybrid_B_fp_external", dir="z", level=0)

max_err = 0.0
max_ref = 0.0
n_wall = 0
ng = bz.n_grow_vect
for mfi in bz:
    vb = mfi.validbox()
    arr = np.array(bz.array(mfi), copy=False)
    lo_i = vb.small_end[0] - ng[0]
    lo_j = vb.small_end[1] - ng[1]
    i_list = list(range(vb.small_end[0], vb.big_end[0] + 1))
    if vb.big_end[0] == NR - 1:
        n_wall += 1
        i_list.append(NR)
    for j in range(vb.small_end[1], vb.big_end[1] + 1):
        z_j = -LZ / 2.0 + j * DZ
        for i in i_list:
            ref = bz_replica((i + 0.5) * DR, z_j)
            max_err = max(max_err, abs(arr[0, 0, j - lo_j, i - lo_i] - ref))
            max_ref = max(max_ref, abs(ref))

vals = np.zeros(2)
comm.Allreduce(np.array([max_err, max_ref]), vals, op=mpi.MAX)
walls = comm.allreduce(n_wall)

if comm.rank == 0:
    print(f"vacuum identity: max err {vals[0]:.3e} vs max ref {vals[1]:.3e} "
          f"({walls} wall boxes)")

assert walls > 0
assert vals[0] < 1.0e-10 * vals[1], f"vacuum identity broken: {vals}"

if comm.rank == 0:
    print("circuit RL discharge test PASSED")
