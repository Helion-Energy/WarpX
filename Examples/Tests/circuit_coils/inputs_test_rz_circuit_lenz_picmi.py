#!/usr/bin/env python3
#
# --- Lenz-sign test of the circuit coupling's back-EMF measurement: a coil
# --- current ramped up over a resistive plasma column must induce plasma
# --- screening currents whose measured flux linkage OPPOSES the ramp
# --- (lambda_p < 0, growing in magnitude while the ramp lasts). The engine
# --- prescribes the ramp (current-stiff drive: it ignores the EMF), so the
# --- probes and the predictor-corrector machinery are exercised with real
# --- plasma physics while the drive stays exactly known. The outer radial
# --- boundary is the Green's-function open (free-space) condition and the
# --- probe is the free-space reciprocity dot -- the production pairing.
# --- (With a conducting wall between plasma and coil the domain is a flux
# --- conserver and an exterior coil measures exactly nothing: the
# --- full-domain disk flux is conserved to roundoff by the wall's zero
# --- tangential E. That shielding is physical, not a probe defect.)

import numpy as np
from mpi4py import MPI as mpi

from pywarpx import callbacks, picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=False)

NR = 32
NZ = 32
LR = 0.1
LZ = 0.2

R_COIL = 0.25
Z_COIL = 0.0
N_TURNS = 1.0
I_REF = 5.0e4      # A: interior field of order 0.1 T at scale 1

TAU_RAMP = 1.0e-7  # s: s(t) = 1 + t / TAU_RAMP

N0 = 1.0e19        # m^-3, conducting column
T_I = 5.0
NPPC = 4
ETA = 1.0e-7       # Ohm m, low resistivity -> strong screening currents
DT = 5.0e-11
MAX_STEPS = 8

grid = picmi.CylindricalGrid(
    number_of_cells=[NR, NZ],
    lower_bound=[0.0, -LZ / 2.0],
    upper_bound=[LR, LZ / 2.0],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["open", "periodic"],
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

solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=5.0 / 3.0,
    Te=T_I,
    n0=N0,
    n_floor=0.01 * N0,
    plasma_resistivity=ETA,
    substeps=4,
    A_external=A_ext,
    # the coil fill is exactly divergence-free (A_theta only) and happens
    # after initialization; the MLMG cleaner also rejects open boundaries
    do_external_diva_cleaning=False,
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
# default probe -> reciprocity (the open boundary is active)
circuit.engine = "callbacks"
circuit.add_new_attr("coupling.corrector_iterations", 1)
circuit.add_new_attr("coupling.corrector_rtol", 1.0e-6)

# the coupling-power ledger (asserted against the double entry below)
pywarpx.warpx.reduced_diags_names = "circdiag"
circdiag = pywarpx.warpx.get_bucket("circdiag")
circdiag.type = "CircuitCoupling"
circdiag.intervals = 1

simulation.initialize_warpx()

libwarpx = simulation.extension
warpx = libwarpx.warpx


class RampEngine:
    """Current-stiff prescribed ramp: s(t) = 1 + t/TAU_RAMP (ignores the
    measured EMF -- a stiff current source)."""

    def __init__(self):
        self.n_predict = 0
        self.lambda_trace = []   # (t1, lambda_p) at each accepted predict

    def s_of_t(self, t):
        return 1.0 + t / TAU_RAMP

    def push(self):
        t0, t1, substep, it = warpx.get_coupling_interval()
        warpx.set_external_vector_potential_scale(
            "drive", self.s_of_t(t0), self.s_of_t(t1), t0, t1)

    def beginstep(self):
        pass

    def predict(self):
        self.n_predict += 1
        t0, t1, substep, it = warpx.get_coupling_interval()
        self.lambda_trace.append((t0, warpx.get_coil_flux_linkage("drive")))
        self.push()

    def correct(self):
        # current-stiff: the corrected EMF does not change the drive
        self.push()

    def finish(self):
        pass


engine = RampEngine()
callbacks.installcallback("circuitbeginstep", engine.beginstep)
callbacks.installcallback("circuitpredict", engine.predict)
callbacks.installcallback("circuitcorrect", engine.correct)
callbacks.installcallback("circuitfinish", engine.finish)

simulation.step()

lam_final = warpx.get_coil_flux_linkage("drive")
times = np.array([t for (t, lam) in engine.lambda_trace])
lams = np.array([lam for (t, lam) in engine.lambda_trace])

if comm.rank == 0:
    print(f"predicts: {engine.n_predict}")
    print("lambda_p trace (t0, lambda):")
    for t, lam in engine.lambda_trace:
        print(f"  {t:.3e}  {lam:+.6e}")
    print(f"lambda_p final: {lam_final:+.6e} Wb*A")

# Lenz: the ramp-up induces plasma screening currents whose linkage OPPOSES
# the drive -> lambda_p strictly negative and growing in magnitude while the
# ramp lasts. Scale: the coil's own linked flux at I_ref is
# L_disc * I_ref^2 ~ 3.6 Wb*A; the screening response over the run is a
# small but well-resolved fraction of it.
lam_scale = 1.4596352993e-6 * I_REF * I_REF

assert lam_final < 0.0, f"Lenz sign violated: lambda_p = {lam_final}"
assert abs(lam_final) > 1.0e-8 * lam_scale, (
    f"screening response unexpectedly small: {lam_final} vs scale {lam_scale}"
)
# Monotonic growth of the screening magnitude across the recorded predicts
# (skip the first few entries where lambda_p is still near the noise floor).
tail = lams[len(lams) // 2 :]
assert np.all(np.diff(tail) < 0.0), (
    "screening linkage not monotonically strengthening: " + str(tail)
)

# The coupling-power double entry: with the constant ramp slope the two
# ledger columns P_circuit = -sum (ds/dt) lambda and
# P_field = Int J_p . E_ext dV are the same reciprocity sum evaluated two
# ways, so they must agree to roundoff, and the power flows INTO the plasma
# (positive) while the ramp drives screening currents.
if comm.rank == 0:
    data = np.loadtxt("diags/reducedfiles/circdiag.txt")
    last = data[-1] if data.ndim > 1 else data
    s_col, dsdt_col, lam_col, p_circ, p_field = last[2], last[3], last[4], last[5], last[6]
    print(f"ledger: s {s_col:.6f} dsdt {dsdt_col:.3e} lambda {lam_col:+.6e} "
          f"P_circuit {p_circ:+.6e} P_field {p_field:+.6e}")
    assert p_field > 0.0, f"coupling power not into the plasma: {p_field}"
    assert abs(p_circ - p_field) < 1.0e-9 * abs(p_field), (
        f"coupling-power double entry broken: {p_circ} vs {p_field}"
    )
    assert abs(lam_col - lam_final) < 1.0e-12 * abs(lam_final) + 1.0e-30, (
        f"ledger lambda disagrees with the register: {lam_col} vs {lam_final}"
    )

if comm.rank == 0:
    print("circuit Lenz test PASSED")
