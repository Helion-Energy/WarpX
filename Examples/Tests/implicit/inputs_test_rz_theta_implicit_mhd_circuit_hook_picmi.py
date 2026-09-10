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

--loop-probe              with --r-open: the extra coils sit INSIDE the
    domain, measured by the analytic-loop probe (circuit.<coil>.probe =
    loop, exclusion radius 2 dr) instead of reciprocity on a filled unit
    field; with --crosscheck the batched loop weights are pinned against
    the single-coil LoopLinkage reference (native only).

--eps-tau-steps X         EMF low-pass with time constant X*dt on the
    drive coil's back-EMF, the production coupler's one-pole EMA whose
    memory is committed only by the finish hook (python) /
    circuit.eps_lowpass_tau (native). 0 = off.

--theta T                 implicit_evolve.theta (default 1). The python
    reference differences eps over theta*dt and advances the FULL step
    (the production hook's semantics); the native driver matches that
    only with circuit.residual_advance = full_step (--full-step-advance),
    its default advancing to the theta-stage time.

--open-loop-first-step    python: no lambda^n before the first finish
    hook (eps = 0 throughout step 1, the production coupler's
    convention); native: circuit.linkage_reference = accepted.

--full-step-advance       native: circuit.residual_advance = full_step.

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
parser.add_argument(
    "--loop-probe",
    action="store_true",
    help="with --r-open: analytic-loop probes (with exclusion mask) on in-domain extra coils",
)
parser.add_argument(
    "--eps-tau-steps",
    type=float,
    default=0.0,
    help="EMF low-pass time constant in units of dt (0 = off)",
)
parser.add_argument(
    "--theta",
    type=float,
    default=1.0,
    help="implicit_evolve.theta (the python reference advances the full step)",
)
parser.add_argument(
    "--open-loop-first-step",
    action="store_true",
    help="python: lambda^n undefined before the first finish hook; native: linkage_reference = accepted",
)
parser.add_argument(
    "--full-step-advance",
    action="store_true",
    help="native: circuit.residual_advance = full_step",
)
parser.add_argument(
    "--drive-probe",
    choices=["disk", "loop"],
    default="disk",
    help="probe of the driven coil: the disk B_z flux (default) or the "
    "analytic-loop J_theta probe with a two-cell exclusion mask (needs --r-open)",
)
parser.add_argument(
    "--probe-weight",
    choices=["none", "physical_share"],
    default="none",
    help="physical-share weight w = eta_phys / eta_field on the J-based drive "
    "probe: native circuit.probe_weight, python the solver-filled "
    "circuit_linkage_weight register (implicit_mhd.circuit_linkage_weight_fill); "
    "needs --drive-probe loop",
)
parser.add_argument(
    "--vacuum-shell",
    action="store_true",
    help="density step to --vacuum-fraction rho0 outside r_step with the "
    "density-keyed vacuum boost --vacuum-diffusivity keyed to that density",
)
parser.add_argument(
    "--vacuum-fraction",
    type=float,
    default=0.05,
    help="--vacuum-shell: shell density as a fraction of rho0 (deeper shells "
    "are Newton-hostile in this undamped dust deck)",
)
parser.add_argument(
    "--vacuum-diffusivity",
    type=float,
    default=30.0,
    help="--vacuum-shell: implicit_mhd.vacuum_resistivity_diffusivity in m^2/s",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left
if args.drive_probe == "loop" and not args.r_open:
    parser.error("--drive-probe loop needs the Green's open boundary (--r-open)")
if args.probe_weight != "none" and args.drive_probe != "loop":
    parser.error("--probe-weight acts on the J-based probes: pass --drive-probe loop")

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

# --vacuum-shell: a tenuous shell (rho_vac = f rho0, pressure unchanged) outside
# r_step = 0.075 (cells 6 and 7 of 8) with the density-keyed field-eta boost
# eta_vac = mu0 D (rho_ref / rho)^2 keyed to rho_ref = rho_vac (the static
# base of the shared vacuum reference): eta_field = sqrt(eta^2 + eta_vac^2),
# so the physical share w = eta / eta_field is W_SHELL = eta / sqrt(eta^2 +
# (mu0 D)^2) in the shell and W_PLASMA = eta / sqrt(eta^2 + (mu0 D f^2)^2)
# in the plasma. The defaults (f = 0.05, D = 30 m^2/s, eta = 1e-6 ohm m)
# give W_SHELL = 0.027 (boost class) and W_PLASMA = 0.996 (physical
# class): the linkage weight has teeth -- an unfilled register (w = 1
# everywhere) is detectably NOT the solver's weight -- while the shell's
# Alfven speed stays within a factor 4.5 of the plasma's (a 1e-3 shell
# stalls the unpreconditioned GMRES of this dust deck).
r_step = 0.075
rho_vac_frac = args.vacuum_fraction if args.vacuum_shell else 1.0
rho_vac = rho_vac_frac * rho0
vacuum_diffusivity = args.vacuum_diffusivity if args.vacuum_shell else 0.0
eta_user = 1.0e-6
eta_vac_shell = constants.mu0 * vacuum_diffusivity
W_SHELL = eta_user / np.sqrt(eta_user**2 + eta_vac_shell**2)
W_PLASMA = eta_user / np.sqrt(eta_user**2 + (eta_vac_shell * rho_vac_frac**2) ** 2)


def stepped(inside_value):
    """Parser expression of a radial step (x = r in RZ): inside_value for
    r < r_step, the vacuum fraction of it outside."""
    if not args.vacuum_shell:
        return f"{inside_value}"
    return f"if(x < {r_step}, {inside_value}, {rho_vac_frac * inside_value})"


sound_speed = np.sqrt(gamma * (Pe0 + Pi0) / rho0)
alfven_speed = B0 / np.sqrt(constants.mu0 * rho0)
dt = 0.5 * (2.0 * zmax / nz) / np.sqrt(sound_speed**2 + alfven_speed**2)
max_steps = 5
theta = args.theta
eps_tau = args.eps_tau_steps * dt

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
            # --drive-probe loop: the analytic-loop J_theta probe of the
            # in-domain filament (two-cell exclusion mask, upper axial
            # plane excluded), the python reference's loop integrand
            probe=args.drive_probe,
            probe_exclusion_radius=(2.0 * dr if args.drive_probe == "loop" else None),
        )
    ]
    for k, name in enumerate(extra_names):
        if args.r_open and args.loop_probe:
            # Analytic-loop rows: measured-only coils INSIDE the domain
            # (zero painted field), the python reference's loop-probe
            # integrand with a two-cell exclusion mask around each
            # filament, valid under the Green's open boundary.
            coils.append(
                picmi.CircuitCoil(
                    name=name,
                    r=0.03 + 0.005 * k,
                    z=-0.1 + 0.05 * k,
                    n_turns=1.0,
                    I_ref=1.0,
                    fill_unit_field=False,
                    probe="loop",
                    probe_exclusion_radius=2.0 * dr,
                )
            )
        elif args.r_open:
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
        eps_lowpass_tau=(eps_tau if eps_tau > 0.0 else None),
        linkage_reference=("accepted" if args.open_loop_first_step else None),
        residual_advance=("full_step" if args.full_step_advance else None),
        # the engine weights its J-based probes with the solver's nodal
        # circuit_linkage_weight register (filled right before every
        # measurement, NeedsLinkageWeight)
        probe_weight=(args.probe_weight if args.probe_weight != "none" else None),
    )

solver = picmi.HybridPICSolver(
    grid=grid,
    Te=T0_ev,
    n0=n0,
    gamma=gamma,
    n_floor=1.0e10,
    plasma_resistivity=eta_user,
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
    # the shell steps the DENSITY only (pressure uniform: no expansion of
    # the plasma into the shell over the run, the shell stays tenuous and
    # boost-dominated for every step)
    mass_density=stepped(rho0),
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
    # --vacuum-shell: the field-eta boost keyed to the shell density
    vacuum_resistivity_diffusivity=(vacuum_diffusivity if args.vacuum_shell else None),
    vacuum_reference_base_density=(rho_vac if args.vacuum_shell else None),
    # python driver + physical_share: the solver refreshes the nodal
    # circuit_linkage_weight register right before each externalcoiltheta /
    # externalcoilfinish callback (the native driver's in-residual fill);
    # without it the register keeps its unit initialization and a python
    # coupler reading it is silently unweighted
    circuit_linkage_weight_fill=(
        1 if (args.driver == "python" and args.probe_weight != "none") else None
    ),
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
# EMF low-pass memory (--eps-tau-steps): "mem" is the committed filter
# state every evaluation of the step reads; "pending" the filtered value
# of the latest evaluation, committed by the finish hook only -- the
# production coupler's per-step-frozen contract.
lowpass = {"mem": 0.0, "pending": None}


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


# --- analytic-loop probe mirror (--drive-probe loop) -------------------
# ablastr::coils' double-precision constants (pi and the CLASSICAL mu0 =
# 4e-7 pi, deliberately ~1.3e-10 off PhysConst::mu0), mirrored exactly.
PI_RING = 3.14159265358979323846
MU0_RING = 4.0e-7 * PI_RING
# f(m)/pi = sum c_k m^k for m <= 0.1 (ablastr::coils::KernelBracket)
KERNEL_SERIES = np.array(
    [
        0.0,
        0.0,
        1.0 / 16.0,
        3.0 / 64.0,
        75.0 / 2048.0,
        245.0 / 8192.0,
        6615.0 / 262144.0,
        22869.0 / 1048576.0,
        1288287.0 / 67108864.0,
        4601025.0 / 268435456.0,
        265939245.0 / 17179869184.0,
        969738055.0 / 68719476736.0,
        28510298817.0 / 2199023255552.0,
        105468168351.0 / 8796093022208.0,
        3138933581875.0 / 281474976710656.0,
        11734782467625.0 / 1125899906842624.0,
        1409850293610375.0 / 144115188075855872.0,
    ]
)


def complete_elliptic_integrals(m):
    """K(m), E(m) by the arithmetic-geometric mean, iteration for
    iteration ablastr::coils::CompleteEllipticIntegrals (the per-element
    early exit at |c| < 1e-16 included, so the last update matches)."""
    m = np.asarray(m, dtype=float)
    a = np.ones_like(m)
    b = np.sqrt(1.0 - m)
    c = np.sqrt(m)
    total = 0.5 * m
    pow2 = 1.0
    for _ in range(64):
        active = np.abs(c) >= 1.0e-16
        if not np.any(active):
            break
        an = 0.5 * (a + b)
        cn = 0.5 * (a - b)
        bn = np.sqrt(a * b)
        pow2 *= 2.0
        a = np.where(active, an, a)
        b = np.where(active, bn, b)
        c = np.where(active, cn, c)
        total = np.where(active, total + 0.5 * pow2 * c * c, total)
    K = PI_RING / (2.0 * a)
    E = K * (1.0 - total)
    return K, E


def kernel_bracket(m):
    """f(m) = (2 - m) K(m) - 2 E(m), the m <= 0.1 series branch included."""
    m = np.asarray(m, dtype=float)
    poly = np.full_like(m, KERNEL_SERIES[16])
    for k in range(15, -1, -1):
        poly = poly * m + KERNEL_SERIES[k]
    K, E = complete_elliptic_integrals(np.where(m <= 0.1, 0.0, m))
    return np.where(m <= 0.1, PI_RING * poly, (2.0 - m) * K - 2.0 * E)


def yee_loop_a_theta(r, z, r_c, z_c):
    """warpx::circuit::YeeLoopATheta: loop A_theta per unit current of the
    filament (r_c, z_c) at the node (r, z), m clipped to [1e-15, 1 - 1e-12],
    exactly 0 on the axis."""
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    d2 = (r + r_c) ** 2 + (z - z_c) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.clip(4.0 * r * r_c / d2, 1.0e-15, 1.0 - 1.0e-12)
        psi = MU0_RING / (4.0 * PI_RING) * 4.0 * r * r_c / np.sqrt(d2) * kernel_bracket(m) / m
        a = np.where(r > 0.0, psi / r, 0.0)
    return a


jtheta_wrapper = None
weight_wrapper = None
loop_table = None


def loop_weight_table():
    """The batched loop probe's static integrand table for the drive coil:
    w_r r (2 pi dr dz) I_ref n_turns A_theta(r_i, z_j; r_c, z_c) on the
    (nr + 1) x nz nodal J_theta mesh (upper axial plane j = nz excluded),
    0 within the exclusion radius -- LinkageBatch::BuildPack's loop row."""
    global loop_table
    if loop_table is None:
        r = np.arange(nr + 1) * dr
        z = -zmax + np.arange(nz) * dz
        R, Z = np.meshgrid(r, z, indexing="ij")
        w_r = np.where((np.arange(nr + 1) == 0) | (np.arange(nr + 1) == nr), 0.5, 1.0)
        excl2 = (2.0 * dr) ** 2
        d2 = (R - coil_r) ** 2 + (Z - coil_z) ** 2
        table = w_r[:, None] * R * (2.0 * PI_RING * dr * dz) * yee_loop_a_theta(R, Z, coil_r, coil_z)
        loop_table = np.where(d2 < excl2, 0.0, table)
    return loop_table


def read_nodal(wrapper_name, idir=None):
    """Global (nr + 1) x (nz + 1) view of a nodal RZ MultiFab register."""
    wrapper = fields.MultiFabWrapper(mf_name=wrapper_name, idir=idir, level=0)
    arr = np.squeeze(np.asarray(wrapper[...]))
    assert arr.shape == (nr + 1, nz + 1), f"{wrapper_name}: unexpected shape {arr.shape}"
    return arr


def measure_lambda_loop(weighted):
    """LoopLinkage of the plasma current, mirrored exactly: the loop table
    dotted with the nodal J_theta of hybrid_current_fp_plasma (the plasma
    current the residual evaluation just computed), times the solver's
    circuit_linkage_weight register when weighted (the physical share w =
    eta_phys / eta_field the native engine folds into the same probe)."""
    jt = read_nodal("hybrid_current_fp_plasma", idir=1)[:, :nz]
    if weighted:
        w = read_nodal("circuit_linkage_weight")[:, :nz]
        jt = jt * w
    return float(np.sum(loop_weight_table() * jt))


def measure_drive():
    """The drive coil's linkage in the probe convention under test."""
    if args.drive_probe == "loop":
        return measure_lambda_loop(args.probe_weight != "none")
    return measure_lambda()


def weight_check_row(step):
    """Per-step (step, lambda_weighted, lambda_unweighted, w_min, w_max,
    boost-class node count) of the drive coil's loop integrand from the
    register: the weight's teeth (an unfilled register is w = 1 everywhere)."""
    w = read_nodal("circuit_linkage_weight")[:, :nz]
    jt = read_nodal("hybrid_current_fp_plasma", idir=1)[:, :nz]
    table = loop_weight_table()
    lam_w = float(np.sum(table * (jt * w)))
    lam_u = float(np.sum(table * jt))
    return (step, lam_w, lam_u, float(w.min()), float(w.max()), int(np.count_nonzero(w < 0.1)))


weight_checks = []
# Loop-probe seed of the committed linkage (python driver): the plasma
# current exists only once the first residual evaluation has computed it
# (hybrid_current_fp_plasma is not filled at t = 0, and the initial state's
# open-face current sheet under the Green's boundary is part of it), so
# lambda^n of step 1 is the FIRST iterate's measurement -- the native
# driver's BeginStepMeasured convention (the first evaluation sees eps = 0
# exactly). The disk probe keeps its t = 0 B_z seed (unchanged tests).
seed = {"pending": False}


def advance_circuit(lam, fraction):
    """Backward-Euler RL advance from the committed snapshot.

    eps is differenced over fraction*dt -- the THETA interval the
    iterate's linkage lives on in the theta hook, the full step in the
    finish hook (the accepted state at t^{n+1}) -- while the loop always
    advances the full step: the production coupler's hook semantics
    (_advance_with(lam, theta) / _advance_with(lam_end, 1.0)). With no
    committed linkage yet (--open-loop-first-step) the step runs open
    loop, eps = 0.
    """
    eps = 0.0 if state["lam_n"] is None else (lam - state["lam_n"]) / (fraction * dt)
    if eps_tau > 0.0:
        sigma = dt / (dt + eps_tau)
        eps = sigma * eps + (1.0 - sigma) * lowpass["mem"]
        lowpass["pending"] = eps
    i_new = (state["i_n"] + (dt / L_C) * (V0 - eps)) / (1.0 + dt * R_C / L_C)
    return i_new, eps


def push_segment(i_new):
    warpx.set_external_vector_potential_scale(
        "drive", state["i_n"], i_new, state["t_n"], state["t_n"] + dt
    )


def hook_theta():
    counts["theta"] += 1
    lam = measure_drive()
    if seed["pending"]:
        state["lam_n"] = lam
        seed["pending"] = False
    i_new, _ = advance_circuit(lam, theta)
    push_segment(i_new)


def hook_finish():
    lam_end = measure_drive()
    if args.drive_probe == "loop":
        weight_checks.append(weight_check_row(state["step"] + 1))
    i_new, eps = advance_circuit(lam_end, 1.0)
    push_segment(i_new)
    state["i_n"] = i_new
    state["lam_n"] = lam_end
    if lowpass["pending"] is not None:
        lowpass["mem"] = lowpass["pending"]  # committed from the accepted state
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
    if args.drive_probe == "loop":
        # the register and current of the accepted state the engine's
        # finish measurement integrated (afterstep leaves both untouched)
        weight_checks.append(weight_check_row(state["step"]))
    # informational only (the engine computes its own eps): the raw
    # finish-hook difference over the full step
    eps = 0.0 if state["lam_n"] is None else (lam_end - state["lam_n"]) / dt
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
    # totals, which at scale 0 equal the plasma response -- unless the
    # production coupler's convention is under test (--open-loop-first-
    # step): no lambda^n before the first finish hook.
    if args.open_loop_first_step:
        state["lam_n"] = None
    elif args.drive_probe == "loop":
        state["lam_n"] = None
        seed["pending"] = True
    else:
        state["lam_n"] = measure_drive()
else:
    callbacks.installcallback("afterstep", record_native_step)
    state["lam_n"] = measure_drive()

sim.step(max_steps)

# --- record and sanity-check -------------------------------------------
rows = np.array(history)
np.savetxt(
    "circuit_hook_history.csv",
    rows,
    header=f"scope={args.scope} driver={args.driver}\n"
    "step t_end hook_calls scale lambda eps",
)

if args.drive_probe == "loop":
    checks = np.array(weight_checks)
    np.savetxt(
        "circuit_weight_check.csv",
        checks,
        header=f"driver={args.driver} probe_weight={args.probe_weight} "
        f"vacuum_shell={int(args.vacuum_shell)}\n"
        "step lambda_weighted lambda_unweighted w_min w_max n_boost_nodes",
    )
    assert checks.shape[0] == max_steps
    if args.probe_weight != "none":
        # The register must be the solver's weight, not its unit init: the
        # vacuum shell holds boost-dominated nodes (w ~ 8e-4), the plasma
        # stays physical (w > 0.99), and the weighted linkage differs from
        # the unweighted one. Under the python driver these fail exactly
        # when implicit_mhd.circuit_linkage_weight_fill is not honored.
        assert W_SHELL < 0.1 < 0.99 < W_PLASMA, "the vacuum shell is not boost-class"
        assert np.all(np.abs(checks[:, 3] - W_SHELL) < 0.3 * W_SHELL), (
            f"register minimum {checks[:, 3]} != the shell weight {W_SHELL:.4e}"
        )
        assert np.all(np.abs(checks[:, 4] - W_PLASMA) < 1.0e-3), (
            f"register maximum {checks[:, 4]} != the plasma weight {W_PLASMA:.6f}"
        )
        assert np.all(checks[:, 5] > 0), "no boost-class node counted"
        assert np.any(checks[:, 1] != checks[:, 2]), "the weight changed no linkage"
        # the committed linkages ARE the weighted loop linkages
        assert np.allclose(rows[:, 4], checks[:, 1], rtol=1.0e-10, atol=0.0), (
            "committed linkage != the weighted loop integrand"
        )
    else:
        assert np.all(checks[:, 3] == 1.0) or args.vacuum_shell or args.driver == "native", (
            "register changed without a weight request"
        )
        assert np.allclose(rows[:, 4], checks[:, 2], rtol=1.0e-10, atol=0.0), (
            "committed linkage != the unweighted loop integrand"
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
