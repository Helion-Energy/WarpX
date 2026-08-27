#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Driven circuit-coupled deck for the theta-implicit MHD circuit hooks.

A uniform static plasma in a uniform Bz is driven by ONE painted
external coil (A_theta = 0.5*Bc*r, python_scale) whose scale follows a
toy driven RL loop

    L_C dI/dt + R_C I = V0 - eps,   eps = d(lambda)/dt,

with lambda the DISK flux linkage of the plasma-response Bz through the
coil circle -- the exact functional of the C++ engine's DiskFluxLinkage
probe (quarter-cell filament offset, strict r_cell < r_off test,
unclamped linear interpolation between the bracketing node planes) --
advanced by one backward-Euler step per coupling interval from the
per-step committed snapshot: the pushed scale segment is a pure function
of the current iterate, the production coupling contract.

Three axes, selected by CLI flags (combined into the CTest arms):

--scope residual|newton   implicit_mhd.circuit_hook_scope: fire the
    coupling on every residual evaluation (default) or only at accepted
    Newton iterates (lagged-circuit quasi-Newton).

--driver python|native    implicit_mhd.circuit_driver: the RL loop
    advanced in python through the externalcoiltheta/externalcoilfinish
    hooks (default), or by the compiled rl_test_circuit plugin (the
    IDENTICAL backward-Euler map) behind the C++ circuit-coupling
    engine, with the engine's batched device flux probes replacing the
    python measurement. The parity analyses assert the committed
    per-step coil scales agree across scopes and across drivers.

--extra-coils N           N additional measured-only coils (zero painted
    field, disk probes) to exercise the multi-coil batched measurement;
    with --crosscheck, circuit.probe_crosscheck pins every batched
    linkage against the single-coil reference probes (native only).

--r-open                  Green's-function open boundary at r_hi and
    reciprocity probes on the extra coils (ring-kernel unit fields):
    exercises the batched reciprocity path (native only).

The deck writes circuit_hook_history.csv (per step: hook calls, the
committed scale, lambda, eps).
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
parser.add_argument(
    "--driver",
    choices=["python", "native"],
    default="python",
    help="implicit_mhd.circuit_driver under test",
)
parser.add_argument(
    "--extra-coils",
    type=int,
    default=0,
    help="additional measured-only coils (native driver only)",
)
parser.add_argument(
    "--crosscheck",
    action="store_true",
    help="enable circuit.probe_crosscheck (native driver only)",
)
parser.add_argument(
    "--r-open",
    action="store_true",
    help="Green's open r_hi boundary + reciprocity extra-coil probes",
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
dr = rmax / nr
dz = 2.0 * zmax / nz

sound_speed = np.sqrt(gamma * (Pe0 + Pi0) / rho0)
alfven_speed = B0 / np.sqrt(constants.mu0 * rho0)
dt = 0.5 * (2.0 * zmax / nz) / np.sqrt(sound_speed**2 + alfven_speed**2)
max_steps = 5
theta = 1.0

# --- toy driven RL circuit --------------------------------------------
# Unit reference current: the coil scale IS the loop current. V0/R_C = 1
# targets s -> 1 with an L/R time of three steps, so the scale segments
# move every step. The plasma back-EMF eps is the honest d/dt of the
# response disk flux; V0 = R_C sized well above its measured magnitude
# keeps the coupled nonlinearity inside the Newton basin while the
# linkage stays far above roundoff for the parity asserts.
R_C = 2000.0
L_C = 3.0 * dt * R_C
V0 = 2000.0
Bc = 0.25 * B0  # painted coil Bz per unit scale
coil_r = 0.06
coil_z = 0.0

grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz],
    warpx_max_grid_size=8 if args.extra_coils > 0 else 64,
    lower_bound=[0.0, -zmax],
    upper_bound=[rmax, zmax],
    lower_boundary_conditions=["none", "none"],
    upper_boundary_conditions=["open" if args.r_open else "dirichlet", "none"],
    lower_boundary_conditions_particles=["none", "absorbing"],
    upper_boundary_conditions_particles=["reflecting", "absorbing"],
)

A_external = {
    "drive": {
        "Ax_external_function": "0.0",
        "Ay_external_function": f"0.5*{Bc}*x",
        "Az_external_function": "0.0",
        "python_scale": True,
        "initial_scale": 0.0,
    }
}
extra_names = [f"extra{k}" for k in range(args.extra_coils)]
for name in extra_names:
    A_external[name] = {
        "Ax_external_function": "0.0",
        "Ay_external_function": "0.0",
        "Az_external_function": "0.0",
        "python_scale": True,
        "initial_scale": 0.0,
    }

circuit = None
if args.driver == "native":
    from pathlib import Path

    # The fixture plugin is emitted into the runtime output directory
    # (build/bin); ctest runs each test in build/bin/<test_name>/.
    plugin = Path.cwd().resolve().parent / "librl_test_circuit.so"
    coils = [
        picmi.CircuitCoil(
            name="drive",
            r=coil_r,
            z=coil_z,
            n_turns=1.0,
            I_ref=1.0,
            fill_unit_field=False,  # keep the painted analytic A
            probe="disk",
        )
    ]
    for k, name in enumerate(extra_names):
        if args.r_open:
            # Reciprocity rows: real ring-kernel unit fields on coils
            # outside the wall, valid under the Green's open boundary.
            coils.append(
                picmi.CircuitCoil(
                    name=name,
                    r=0.12 + 0.01 * k,
                    z=-0.1 + 0.05 * k,
                    n_turns=1.0,
                    I_ref=1.0,
                    fill_unit_field=True,
                    probe="reciprocity",
                )
            )
        else:
            # Measured-only disk rows (zero painted field).
            coils.append(
                picmi.CircuitCoil(
                    name=name,
                    r=0.03 + 0.005 * k,
                    z=-0.1 + 0.05 * k,
                    n_turns=1.0,
                    I_ref=1.0,
                    fill_unit_field=False,
                    probe="disk",
                )
            )
    circuit = picmi.CircuitCoupling(
        coils=coils,
        engine="external",
        plugin_library=str(plugin),
        # the value carries '=' and ',': pywarpx double-quotes it whole
        # for ParmParse; plain floats (numpy reprs are not parseable)
        plugin_config=f"R={float(R_C)},L={float(L_C)},V0={float(V0)}",
        probe_crosscheck=args.crosscheck,
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
    A_external=A_external,
    do_external_diva_cleaning=False,
    circuit=circuit,
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
    circuit_driver=args.driver,
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
bz_wrapper = None

# Per-step committed circuit state and instrumentation. lam_n is the
# committed linkage; i_n the committed loop current; hook counts reset
# per step by the finish hook.
state = {"i_n": 0.0, "lam_n": 0.0, "t_n": 0.0, "step": 0}
counts = {"theta": 0}
history = []


def measure_lambda():
    """DiskFluxLinkage of the plasma-response Bz, mirrored exactly.

    lambda = I_ref * n_turns * 2 pi dr *
             sum_{j in {j0, j0+1}} w_j sum_{r_c < r_off} Bz(i, j) r_c,

    with the quarter-cell filament offset, the STRICT r_c < r_off cell
    test and the unclamped linear weight between the bracketing node
    planes -- the identical staircase conventions as the C++ probe, so
    the python and native drivers integrate the same measurement.
    Bfield_fp holds the plasma response at every firing point.
    """
    global bz_wrapper
    if bz_wrapper is None:
        bz_wrapper = fields.BzFPWrapper(level=0)
    bz = np.squeeze(np.asarray(bz_wrapper[...]))  # (nr cells, nz+1 nodes)
    r_off = coil_r + 0.25 * dr
    z_off = coil_z + 0.25 * dz
    j0 = int(np.floor((z_off - (-zmax)) / dz))
    j0 = min(max(j0, 0), nz - 1)
    w = (z_off - (-zmax + j0 * dz)) / dz
    r_c = (np.arange(nr) + 0.5) * dr
    inside = r_c < r_off
    plane_sum = lambda j: float(np.sum(bz[inside, j] * r_c[inside]))  # noqa: E731
    lam = (1.0 - w) * plane_sum(j0) + w * plane_sum(j0 + 1)
    return 2.0 * np.pi * dr * lam  # I_ref = n_turns = 1


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


def record_native_step():
    """Per-step committed record of the native driver (afterstep)."""
    state["t_n"] += dt
    state["step"] += 1
    scale = warpx.get_external_vector_potential_scale("drive", state["t_n"])
    lam_end = warpx.get_coil_flux_linkage("drive")
    eps = (lam_end - state["lam_n"]) / (theta * dt)
    state["lam_n"] = lam_end
    history.append((state["step"], state["t_n"], -1, scale, lam_end, eps))


if args.driver == "python":
    for name, function in (
        ("externalcoiltheta", hook_theta),
        ("externalcoilfinish", hook_finish),
    ):
        if name not in callbacks.callback_instances:
            callbacks.callback_instances[name] = callbacks.CallbackFunctions(name=name)
        callbacks.installcallback(name, function)
    # Seed the committed linkage at t = 0 exactly like the native
    # driver's step-entry measurement: between steps Bfield_fp holds the
    # totals, which at scale 0 equal the plasma response.
    state["lam_n"] = measure_lambda()
else:
    callbacks.installcallback("afterstep", record_native_step)
    state["lam_n"] = measure_lambda()

sim.step(max_steps)

# --- record and sanity-check -------------------------------------------
rows = np.array(history)
np.savetxt(
    "circuit_hook_history.csv",
    rows,
    header=f"scope={args.scope} driver={args.driver}\n"
    "step t_end hook_calls scale lambda eps",
)

assert len(history) == max_steps, "the per-step commit must fire once per step"
if args.driver == "python":
    assert np.all(rows[:, 2] >= 1), "the theta hook must fire at least once per step"
# The drive must actually ramp and the plasma back-react (a live
# coupling channel, not a decorative one).
assert rows[-1, 3] > 0.1, "the RL drive did not ramp the coil scale"
assert np.any(rows[:, 4] != 0.0), "no plasma flux linkage was ever measured"

print(f"--- circuit hook history (scope = {args.scope}, driver = {args.driver}) ---")
for step, t_end, calls, scale, lam, eps in history:
    print(
        f"step {int(step)}: hook_calls = {int(calls)}, scale = {scale:.12e}, "
        f"lambda = {lam:.6e}, eps = {eps:.6e}"
    )
print("PASS")
