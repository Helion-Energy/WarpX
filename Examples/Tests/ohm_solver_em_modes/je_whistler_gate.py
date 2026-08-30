#!/usr/bin/env python3
#
# --- Whistler-dispersion gate for the Je-form generalized-Ohm's-law
# --- E solve with electron inertia (hybrid_pic_model.esolve = je_form,
# --- relax advance; Hewett & Nielson JCP 29 (1978) 219, regularizers
# --- from Amano JCP 275 (2014) 197, Fig-1 class).
# --- 1D uniform plasma with a Bz guide field and a single seeded
# --- transverse mode at k*d_e ~ 1, dumped every step through a line
# --- probe. Run twice: --esolve e_form is the massless-electron control
# --- (the standard explicit hybrid E solve), --esolve je_form exercises
# --- the dynamical (E, J_e) advance, whose mode must land on the
# --- inertia-corrected whistler branch (rollover
# --- w -> w_ce k^2 d_e^2 / (1 + k^2 d_e^2)) and exclude the massless
# --- branch. Fit and branch discrimination: je_whistler_analysis.py.
# --- Configuration follows the inertia-seeded variant of the
# --- ohm_solver_em_modes test (Munoz et al. (2018) base parameters).

import argparse
import json
import sys

import numpy as np
from mpi4py import MPI as mpi

import pywarpx
from pywarpx import picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

parser = argparse.ArgumentParser()
parser.add_argument(
    "--esolve",
    choices=["e_form", "je_form"],
    default="je_form",
    help="E-solve form: e_form = massless control (the standard explicit "
    "hybrid E solve), je_form = dynamical electron-current solve "
    "(hybrid_pic_model.esolve)",
)
parser.add_argument(
    "--je-c-frac",
    type=float,
    default=0.5,
    help="relax advance: artificial-light-speed CFL fraction "
    "(hybrid_pic_model.je_c_frac)",
)
parser.add_argument(
    "--je-qn-frac",
    type=float,
    default=-1.0,
    help="relax advance: quasineutral relaxation rate as a fraction of "
    "1/dt (hybrid_pic_model.je_qn_frac; pins the longitudinal sector to "
    "the one-count-guarded Ohm value, plasma-blended). <0 = C++ default "
    "(1.0); 0 = off (unstable honest-E_L dynamics, study only)",
)
parser.add_argument(
    "--filter",
    choices=["default", "on", "off"],
    default="default",
    help="bilinear (binomial) filtering of the deposited J/rho "
    "(warpx.use_filter). NOTE: the WarpX default is ON for this 1D "
    "explicit configuration, so 'default' and 'on' run the same "
    "config -- 'off' is the only true A/B arm",
)
parser.add_argument(
    "--je-yee",
    action="store_true",
    help="Yee-coupled relax advance (hybrid_pic_model.je_yee_coupling): "
    "fields on the standard Yee staggering, nodal relax core coupled "
    "through the second-order adjoint gather/scatter pair",
)
parser.add_argument(
    "--je-advance",
    choices=["cn", "be"],
    default="cn",
    help="relax advance: cn = the time-centered Crank-Nicolson default "
    "(hybrid_pic_model.je_time_stagger = 1), be = the legacy "
    "backward-Euler Lie advance",
)
parser.add_argument(
    "--steps", type=int, default=100, help="number of steps (dumped every step)"
)
parser.add_argument("-v", "--verbose", action="store_true")
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

# --- plasma / box parameters (the seeded-whistler configuration) -----------
B0 = 0.25  # guide field [T]
beta = 0.01
m_ion = 100.0  # ion mass [electron masses]
vA_over_c = 1e-4
Nz = 256
DZ = 1.0 / 10.0  # cell size [ion skin depths] = d_e at M/m = 100
DT = 5e-4  # time step [ion cyclotron periods]
NPPC = 1024
eta = 1e-7  # resistivity, damps the mode excitation
substeps = 40
MODE_M = 41  # seeded mode number: k d_e ~ 1
SEED_AMP = 2.0e-3  # seed amplitude [B0]

M = m_ion * constants.m_e
w_ci = constants.q_e * B0 / M
t_ci = 2.0 * np.pi / w_ci
vA = vA_over_c * constants.c
n_plasma = (B0 / vA) ** 2 / (constants.mu0 * (M + constants.m_e))
w_pi = np.sqrt(constants.q_e**2 * n_plasma / (M * constants.ep0))
l_i = constants.c / w_pi
v_ti = np.sqrt(beta / 2.0) * vA
T_plasma = v_ti**2 * M / constants.q_e  # eV

dz = DZ * l_i
Lz = Nz * dz
dt = DT * t_ci
k_seed = 2.0 * np.pi * MODE_M / Lz

# The Eq-27 variable electron mass must sit at the PHYSICAL m_e at this
# dt, else the mode lands on an artificially heavy branch: check the
# stability bound at the B-substep cadence the E solve uses.
d_e = l_i * np.sqrt(constants.m_e / M)
v_Ae = B0 / np.sqrt(constants.mu0 * n_plasma * constants.m_e)
dt_sub = dt / (2.0 * substeps)
alpha = 0.5
me_ratio = (v_Ae * dt_sub / (2.0 * alpha * dz)) ** 2
assert me_ratio < 1.0, (
    f"Eq-27 variable mass would exceed m_e (ratio {me_ratio:.2e}): "
    "reduce dt or raise substeps so the gate measures the physical branch"
)

if comm.rank == 0:
    print(
        f"[gate] {args.esolve} arm: n = {n_plasma:.3e} m^-3, "
        f"Te = {T_plasma:.3f} eV, k*d_e = {k_seed * d_e:.4f}, "
        f"dz/d_e = {dz / d_e:.3f}, Eq-27 mass ratio = {me_ratio:.2e}",
        flush=True,
    )

# --- grid / solver ----------------------------------------------------------
grid = picmi.Cartesian1DGrid(
    number_of_cells=[Nz],
    warpx_max_grid_size=Nz,
    lower_bound=[0],
    upper_bound=[Lz],
    lower_boundary_conditions=["periodic"],
    upper_boundary_conditions=["periodic"],
)

solver = picmi.HybridPICSolver(
    grid=grid,
    Te=T_plasma,
    n0=n_plasma,
    plasma_resistivity=eta,
    substeps=substeps,
)

simulation = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=args.steps,
    particle_shape=1,
    verbose=1 if args.verbose else 0,
    warpx_serialize_initial_conditions=True,
    warpx_current_deposition_algo="direct",
    warpx_grid_type=None if args.je_yee else "collocated",
    warpx_use_filter={"default": None, "on": 1, "off": 0}[args.filter],
)

B_ext = picmi.AnalyticInitialField(
    Bx_expression=f"{SEED_AMP * B0}*cos({k_seed}*z)",
    By_expression=0.0,
    Bz_expression=B0,
)
simulation.add_applied_field(B_ext)

ions = picmi.Species(
    name="ions",
    charge="q_e",
    mass=M,
    initial_distribution=picmi.UniformDistribution(
        density=n_plasma,
        rms_velocity=[v_ti] * 3,
    ),
)
simulation.add_species(
    ions,
    layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=NPPC),
)

line_diag = picmi.ReducedDiagnostic(
    diag_type="FieldProbe",
    probe_geometry="Line",
    z_probe=0,
    z1_probe=Lz,
    resolution=Nz - 1,
    name="par_field_data",
    period=1,
    path="diags/",
)
simulation.add_diagnostic(line_diag)

simulation.initialize_inputs()

if args.esolve == "je_form":
    pywarpx.hybridpicmodel.esolve = "je_form"
    if args.je_yee:
        pywarpx.hybridpicmodel.je_yee_coupling = 1
    # uniform plasma: the one-count level only guards true vacuum; keep it
    # far below n0 so neither the denominator guard nor the Eq-30 vacuum
    # damping blend touches the mode
    pywarpx.hybridpicmodel.je_n_min = 1e-6 * n_plasma
    pywarpx.hybridpicmodel.je_c_frac = args.je_c_frac
    pywarpx.hybridpicmodel.je_time_stagger = 1 if args.je_advance == "cn" else 0
    if args.je_qn_frac >= 0.0:
        pywarpx.hybridpicmodel.je_qn_frac = args.je_qn_frac

simulation.initialize_warpx()

if comm.rank == 0:
    with open("gate_params.json", "w") as f:
        json.dump(
            {
                "esolve": args.esolve,
                "je_advance": "relax",
                "Lz": Lz,
                "dz": dz,
                "dt": dt,
                "B0": B0,
                "m_ion": m_ion,
                "n_plasma": n_plasma,
                "eta": eta,
                "mode_m": MODE_M,
                "substeps": substeps,
            },
            f,
        )

simulation.step()
