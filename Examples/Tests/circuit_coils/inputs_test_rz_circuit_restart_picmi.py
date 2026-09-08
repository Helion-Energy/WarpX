#!/usr/bin/env python3
#
# --- Restart continuity of the circuit coupling: a checkpoint taken mid-run
# --- must restore the live scale segments of the circuit-driven external
# --- fields (they are runtime state, not inputs: without them the drive
# --- would reset to initial_scale and silently jump). The same deck runs
# --- both arms; the restarted arm passes amr.restart='<chk>' on the command
# --- line, continues to the same final step, and its coupling ledger must
# --- match the uninterrupted run's final row.

import sys

import numpy as np
from mpi4py import MPI as mpi

from pywarpx import callbacks, picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

# The restarted arm passes amr.restart='<chk>' on the command line.
restart_from = None
for _arg in list(sys.argv[1:]):
    if _arg.startswith("amr.restart="):
        restart_from = _arg.split("=", 1)[1].strip("'\"")
        sys.argv.remove(_arg)

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=False)

NR = 32
NZ = 32
LR = 0.1
LZ = 0.2

R_COIL = 0.25
Z_COIL = 0.0
N_TURNS = 1.0
I_REF = 5.0e4

TAU_RAMP = 1.0e-7  # s: s(t) = 1 + t / TAU_RAMP

N0 = 1.0e19
T_I = 5.0
NPPC = 4
ETA = 1.0e-7
DT = 5.0e-11
MAX_STEPS = 8
CHK_STEP = 4

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
if restart_from is not None:
    simulation.amr_restart = restart_from

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

checkpoint = picmi.Checkpoint(name="chk", period=CHK_STEP)
simulation.add_diagnostic(checkpoint)

simulation.initialize_inputs()

import pywarpx  # noqa: E402

circuit = pywarpx.warpx.get_bucket("circuit")
circuit.coils = "drive"
circuit.add_new_attr("drive.r", R_COIL)
circuit.add_new_attr("drive.z", Z_COIL)
circuit.add_new_attr("drive.n_turns", N_TURNS)
circuit.add_new_attr("drive.I_ref", I_REF)
circuit.engine = "callbacks"
circuit.add_new_attr("coupling.corrector_iterations", 1)
circuit.add_new_attr("coupling.corrector_rtol", 1.0e-6)

pywarpx.warpx.reduced_diags_names = "circdiag"
circdiag = pywarpx.warpx.get_bucket("circdiag")
circdiag.type = "CircuitCoupling"
circdiag.intervals = 1

simulation.initialize_warpx()

libwarpx = simulation.extension
warpx = libwarpx.warpx


class RampEngine:
    """Current-stiff prescribed ramp (stateless in time, so the engine
    itself needs no restart handling -- exactly what isolates the segment
    restoration under test)."""

    def s_of_t(self, t):
        return 1.0 + t / TAU_RAMP

    def push(self):
        t0, t1, substep, it = warpx.get_coupling_interval()
        warpx.set_external_vector_potential_scale(
            "drive", self.s_of_t(t0), self.s_of_t(t1), t0, t1)

    def beginstep(self):
        pass

    def predict(self):
        self.push()

    def correct(self):
        self.push()

    def finish(self):
        pass


engine = RampEngine()
callbacks.installcallback("circuitbeginstep", engine.beginstep)
callbacks.installcallback("circuitpredict", engine.predict)
callbacks.installcallback("circuitcorrect", engine.correct)
callbacks.installcallback("circuitfinish", engine.finish)

# The restored segment is live BEFORE the first restarted step: the drive
# must not have reset to initial_scale (s = 1).
if restart_from is not None:
    t_restart = warpx.gett_new(0)
    s_restored = warpx.get_external_vector_potential_scale("drive", t_restart)
    s_expected = 1.0 + t_restart / TAU_RAMP
    if comm.rank == 0:
        print(f"restored scale at t = {t_restart:.3e}: {s_restored:.15f} "
              f"(expected {s_expected:.15f})")
    assert abs(s_restored - s_expected) < 1.0e-12, (
        f"scale segment not restored: {s_restored} vs {s_expected}"
    )

# step(n) advances n MORE steps: the restarted arm runs only the remainder
if restart_from is not None:
    simulation.step(MAX_STEPS - CHK_STEP)
else:
    simulation.step()

if restart_from is not None and comm.rank == 0:
    mine = np.loadtxt("diags/reducedfiles/circdiag.txt")
    ref = np.loadtxt(
        "../test_rz_circuit_restart_picmi/diags/reducedfiles/circdiag.txt")
    mine = np.atleast_2d(mine)
    ref = np.atleast_2d(ref)
    # compare at the absolute final step (robust to any overrun)
    last_mine = mine[mine[:, 0] == MAX_STEPS][-1]
    last_ref = ref[ref[:, 0] == MAX_STEPS][-1]
    print("restart ledger:", last_mine)
    print("full-run ledger:", last_ref)
    # s, dsdt, lambda, P_circuit, P_field of the final step must agree:
    # the fields are bit-restored and the stateless engine repushes the
    # identical segments.
    for col in range(2, 7):
        denom = max(abs(last_ref[col]), 1e-300)
        rel = abs(last_mine[col] - last_ref[col]) / denom
        assert rel < 1.0e-10, (
            f"ledger column {col} diverged after restart: "
            f"{last_mine[col]} vs {last_ref[col]} (rel {rel})"
        )
    print("circuit restart test PASSED")
