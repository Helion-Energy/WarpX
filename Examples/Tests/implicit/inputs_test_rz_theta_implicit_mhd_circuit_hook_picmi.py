#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Driven circuit-coupled deck for implicit_mhd.circuit_hook_scope.

A uniform static plasma in a uniform Bz is driven by ONE painted
external coil (A_theta = 0.5*Bc*r, python_scale) whose scale follows a
toy driven RL loop advanced INSIDE the residual through the
"externalcoiltheta" hook (implicit_mhd.external_field_iteration = 1):

    L_C dI/dt + R_C I = V0 - eps,   eps = d(lambda)/dt,

with lambda a fixed linear functional of the plasma-response J_theta
(a mutual-inductance surrogate: the same smooth state->circuit map the
production coupler applies, minus the geometry). Backward-Euler
circuit advance from the per-step committed snapshot, so the pushed
scale segment is a pure function of the current iterate -- exactly the
production ImplicitCircuitCoupler contract.

Run twice by CTest (--scope residual | newton):

* residual: the hook fires on EVERY residual evaluation (default,
  bit-compatible with the pre-knob behavior);
* newton:   the hook fires ONLY at accepted Newton iterates; Jacobian
  probes and line-search trials reuse the cached coil scales
  (lagged-circuit quasi-Newton).

The deck writes circuit_hook_history.csv (per step: hook calls, the
committed scale, lambda, eps). The newton-scope test's analysis
(analysis_mhd_circuit_hook_scope.py) asserts the committed per-step
scales match the residual run to solver tolerance -- the converged
ANSWER is scope-independent -- while the hook-call counts collapse.
"""

import argparse
import sys

import numpy as np

from pywarpx import callbacks, fields, libwarpx, picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument(
    "--scope",
    choices=["residual", "newton"],
    default="residual",
    help="implicit_mhd.circuit_hook_scope under test",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

# --- plasma and grid --------------------------------------------------
n0 = 1.0e20
rho0 = n0 * constants.m_p
T0_ev = 10.0
Pe0 = n0 * T0_ev * constants.q_e
Pi0 = n0 * T0_ev * constants.q_e
gamma = 5.0 / 3.0
B0 = 0.02

nr = 8
nz = 16
rmax = 0.1
zmax = 0.2

sound_speed = np.sqrt(gamma * (Pe0 + Pi0) / rho0)
alfven_speed = B0 / np.sqrt(constants.mu0 * rho0)
dt = 0.5 * (2.0 * zmax / nz) / np.sqrt(sound_speed**2 + alfven_speed**2)
max_steps = 5
theta = 1.0

# --- toy driven RL circuit --------------------------------------------
# Unit reference current: the coil scale IS the loop current. V0/R_C = 1
# targets s -> 1 with an L/R time of three steps, so the scale segments
# move every step; C_M sizes the plasma back-EMF term (the coupling
# under test) without any pretense of real geometry. Its budget: the
# state->scale loop gain carries C_M/(theta dt) through eps and another
# 1/dt through the pushed E-scale slope, so C_M ~ 1e-5 keeps the
# coupled nonlinearity comfortably inside the Newton basin while the
# measured lambda stays far above roundoff for the parity asserts.
R_C = 1.0
L_C = 3.0 * dt * R_C
V0 = 1.0
C_M = 1.0e-5
Bc = 0.25 * B0  # painted coil Bz per unit scale

grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz],
    warpx_max_grid_size=64,
    lower_bound=[0.0, -zmax],
    upper_bound=[rmax, zmax],
    lower_boundary_conditions=["none", "none"],
    upper_boundary_conditions=["dirichlet", "none"],
    lower_boundary_conditions_particles=["none", "absorbing"],
    upper_boundary_conditions_particles=["reflecting", "absorbing"],
)

solver = picmi.HybridPICSolver(
    grid=grid,
    Te=T0_ev,
    n0=n0,
    gamma=gamma,
    n_floor=1.0e10,
    plasma_resistivity=1.0e-6,
    include_hall_term=False,
    include_electron_pressure_term=False,
    A_external={
        "drive": {
            "Ax_external_function": "0.0",
            "Ay_external_function": f"0.5*{Bc}*x",
            "Az_external_function": "0.0",
            "python_scale": True,
            "initial_scale": 0.0,
        }
    },
    do_external_diva_cleaning=False,
)

nonlinear_solver = picmi.NewtonNonlinearSolver(
    verbose=True,
    max_iterations=20,
    relative_tolerance=1.0e-10,
    absolute_tolerance=0.0,
)

evolve_scheme = picmi.ThetaImplicitMHDEvolveScheme(
    nonlinear_solver=nonlinear_solver,
    theta=theta,
    mass_density=f"{rho0}",
    electron_pressure=f"{Pe0}",
    ion_pressure=f"{Pi0}",
    ion_closure="total_energy",
    reference_mass_density=rho0,
    reference_magnetic_field=B0,
    reference_ion_pressure=Pi0,
    gamma_e=gamma,
    gamma_i=gamma,
    mass_density_floor=1.0e-4 * rho0,
    electron_pressure_floor=1.0e-4 * Pe0,
    ion_pressure_floor=1.0e-4 * Pi0,
    fluid_flux="central",
    viscosity=3.0e3,  # the central flux requires the explicit viscous stabilization
    external_field_iteration=1,
    circuit_hook_scope=args.scope,
)

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=max_steps,
    verbose=1,
)
sim.evolve_scheme = evolve_scheme

initial_field = picmi.AnalyticInitialField(
    Bx_expression=0.0,
    By_expression=0.0,
    Bz_expression=B0,
)
sim.add_applied_field(initial_field)

sim.initialize_inputs()
sim.initialize_warpx()

# --- circuit coupler ---------------------------------------------------
warpx = libwarpx.libwarpx_so.get_instance()
jt_wrapper = None
radii = None

# Per-step committed circuit state and instrumentation. lam_n is the
# committed linkage; i_n the committed loop current; hook counts reset
# per step by the finish hook.
state = {"i_n": 0.0, "lam_n": 0.0, "t_n": 0.0, "step": 0}
counts = {"theta": 0}
history = []


def measure_lambda():
    """Fixed linear functional of the plasma-response J_theta.

    lambda = C_M * mu0 * sum(J_theta * r) * dr * dz -- a smooth
    mutual-inductance surrogate; hybrid_current_fp_plasma holds the
    plasma-response current at every hook firing (recomputed from the
    iterate's B just before the hook, and left plasma-only through the
    end-of-step commit).
    """
    global jt_wrapper, radii
    if jt_wrapper is None:
        jt_wrapper = fields.JyFPPlasmaWrapper(level=0)
    jt = np.squeeze(np.asarray(jt_wrapper[...]))
    dr = rmax / nr
    dz = 2.0 * zmax / nz
    if radii is None:
        radii = (np.arange(jt.shape[0]) + 0.5) * dr
    return C_M * constants.mu0 * float(np.sum(jt * radii[:, None])) * dr * dz


def advance_circuit(lam):
    """Backward-Euler RL advance from the committed snapshot."""
    eps = (lam - state["lam_n"]) / (theta * dt)
    i_new = (state["i_n"] + (dt / L_C) * (V0 - eps)) / (1.0 + dt * R_C / L_C)
    return i_new, eps


def push_segment(i_new):
    warpx.set_external_vector_potential_scale(
        "drive", state["i_n"], i_new, state["t_n"], state["t_n"] + dt
    )


def hook_theta():
    counts["theta"] += 1
    i_new, _ = advance_circuit(measure_lambda())
    push_segment(i_new)


def hook_finish():
    lam_end = measure_lambda()
    i_new, eps = advance_circuit(lam_end)
    push_segment(i_new)
    state["i_n"] = i_new
    state["lam_n"] = lam_end
    state["t_n"] += dt
    state["step"] += 1
    history.append((state["step"], state["t_n"], counts["theta"], i_new, lam_end, eps))
    counts["theta"] = 0


for name, function in (("externalcoiltheta", hook_theta), ("externalcoilfinish", hook_finish)):
    if name not in callbacks.callback_instances:
        callbacks.callback_instances[name] = callbacks.CallbackFunctions(name=name)
    callbacks.installcallback(name, function)

sim.step(max_steps)

# --- record and sanity-check -------------------------------------------
rows = np.array(history)
np.savetxt(
    "circuit_hook_history.csv",
    rows,
    header=f"scope={args.scope}\nstep t_end hook_calls scale lambda eps",
)

assert len(history) == max_steps, "the finish hook must fire once per step"
assert np.all(rows[:, 2] >= 1), "the theta hook must fire at least once per step"
# The drive must actually ramp and the plasma back-react (a live
# coupling channel, not a decorative one).
assert rows[-1, 3] > 0.1, "the RL drive did not ramp the coil scale"
assert np.any(rows[:, 4] != 0.0), "no plasma flux linkage was ever measured"

print("--- circuit hook history (scope = {}) ---".format(args.scope))
for step, t_end, calls, scale, lam, eps in history:
    print(
        f"step {int(step)}: hook_calls = {int(calls)}, scale = {scale:.12e}, "
        f"lambda = {lam:.6e}, eps = {eps:.6e}"
    )
print("PASS")
