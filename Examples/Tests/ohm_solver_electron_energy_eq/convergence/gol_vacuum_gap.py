#!/usr/bin/env python3
#
# --- Vacuum-gap gate for the GOL E solve: a 1D plasma slab bounded by
# --- TRUE vacuum (no density floor, no pedestal — the configuration the
# --- massless Ohm solve cannot represent at all). A seeded transverse
# --- mode drives E through the gap, where the screened solve degenerates
# --- to the Laplace problem whose warm-started Jacobi convergence is the
# --- open question (the implicit arm quantified GMRES 15 -> 595 across
# --- wide gaps). Gates:
# ---   (a) vacuum expansion: the run completes with E finite everywhere;
# ---   (b) per-sweep residual decay across the gap, from the [gol] sweep
# ---       lines (needs warpx verbose >= 2);
# ---   (c) sweep-ladder under-convergence: rerun with --gol-sweeps 16+
# ---       and compare final fields (relL2 = the sweeps-4 solve error).

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
parser.add_argument("--gol-sweeps", type=int, default=4)
parser.add_argument("--nppc", type=int, default=256)
parser.add_argument("--gol-form", choices=["jacobi", "relax"], default="jacobi")
parser.add_argument("--gol-c-frac", type=float, default=0.5)
parser.add_argument(
    "--substeps",
    type=int,
    default=40,
    help="B substeps (relax self-convergence: more substeps = higher "
    "c_art = less retardation, all errors refine together)",
)
parser.add_argument(
    "--gol-div-clean-frac",
    type=float,
    default=-1.0,
    help="relax form: vacuum-blended Marder divergence-cleaning strength "
    "(hybrid_pic_model.gol_div_clean_frac, diffusion-CFL fraction). "
    "<0 = C++ default (0.5); 0 = off",
)
parser.add_argument(
    "--gol-qn-frac",
    type=float,
    default=-1.0,
    help="relax form: quasineutral relaxation rate as a fraction of 1/dt "
    "(hybrid_pic_model.gol_qn_frac; pins the longitudinal sector to the "
    "one-count-guarded Ohm value, plasma-blended). <0 = C++ default (1.0); "
    "0 = off (unstable honest-E_L dynamics, study only)",
)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument(
    "--slab-frac", type=float, default=1.0 / 3.0, help="slab width / Lz"
)
parser.add_argument("-v", "--verbose", type=int, default=2)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

# --- parameters: whistler-gate normalization, shorter box --------------------
B0 = 0.25
beta = 0.01
m_ion = 100.0
vA_over_c = 1e-4
Nz = 128
DZ = 1.0 / 10.0  # cell size [ion skin depths] = d_e at M/m = 100
DT = 5e-4
NPPC = args.nppc
eta = 1e-7
substeps = args.substeps

M = m_ion * constants.m_e
w_ci = constants.q_e * B0 / M
t_ci = 2.0 * np.pi / w_ci
vA = vA_over_c * constants.c
n_plasma = (B0 / vA) ** 2 / (constants.mu0 * (M + constants.m_e))
w_pi = np.sqrt(constants.q_e**2 * n_plasma / (M * constants.ep0))
l_i = constants.c / w_pi
v_ti = np.sqrt(beta / 2.0) * vA
T_plasma = v_ti**2 * M / constants.q_e

dz = DZ * l_i
Lz = Nz * dz
dt = DT * t_ci

slab_w = args.slab_frac * Lz
z_lo = 0.5 * (Lz - slab_w)
z_hi = 0.5 * (Lz + slab_w)
gap_cells = int((Lz - slab_w) / dz)

if comm.rank == 0:
    print(
        f"[gap gate] slab [{z_lo:.3e}, {z_hi:.3e}] of Lz {Lz:.3e} m, "
        f"vacuum gap {gap_cells} cells (periodic wrap), "
        f"sweeps {args.gol_sweeps}",
        flush=True,
    )

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
    verbose=args.verbose,
    warpx_serialize_initial_conditions=True,
    warpx_current_deposition_algo="direct",
    warpx_grid_type="collocated",
)

# global seeded transverse mode (mode 3): finite curl through the gap
k_seed = 2.0 * np.pi * 3.0 / Lz
B_ext = picmi.AnalyticInitialField(
    Bx_expression=f"{2.0e-3 * B0}*cos({k_seed}*z)",
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
        lower_bound=[None, None, z_lo],
        upper_bound=[None, None, z_hi],
    ),
)
simulation.add_species(
    ions,
    layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=NPPC),
)

field_diag = picmi.FieldDiagnostic(
    name="field_diags",
    grid=grid,
    period=args.steps,
    data_list=["B", "E", "rho"],
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
)
simulation.add_diagnostic(field_diag)

pywarpx.hybridpicmodel.esolve = "gol"
pywarpx.hybridpicmodel.gol_sweeps = args.gol_sweeps
# one-count level = one macroparticle's deposit
pywarpx.hybridpicmodel.gol_n_min = n_plasma / NPPC
if args.gol_form == "relax":
    pywarpx.hybridpicmodel.gol_form = "relax"
    pywarpx.hybridpicmodel.gol_c_frac = args.gol_c_frac
    if args.gol_qn_frac >= 0.0:
        pywarpx.hybridpicmodel.gol_qn_frac = args.gol_qn_frac
    if args.gol_div_clean_frac >= 0.0:
        pywarpx.hybridpicmodel.gol_div_clean_frac = args.gol_div_clean_frac

simulation.initialize_inputs()
simulation.initialize_warpx()

if comm.rank == 0:
    with open("gate_params.json", "w") as f:
        json.dump(
            {
                "Lz": Lz,
                "dz": dz,
                "dt": dt,
                "B0": B0,
                "n_plasma": n_plasma,
                "gap_cells": gap_cells,
                "gol_sweeps": args.gol_sweeps,
                "gol_form": args.gol_form,
                "substeps": substeps,
                "z_lo": z_lo,
                "z_hi": z_hi,
            },
            f,
        )

simulation.step()
