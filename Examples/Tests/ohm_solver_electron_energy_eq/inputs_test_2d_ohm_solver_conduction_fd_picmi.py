#!/usr/bin/env python3
#
# --- Anisotropic electron heat conduction on closed flux surfaces
# --- (grid-FD operator): a hot Gaussian blob at the O-point of a
# --- cat's-eye field
# ---
# ---     A_y ~ cos(k x) cos(k z),
# ---     Bx = B0 cos(k x) sin(k z),  Bz = -B0 sin(k x) cos(k z),
# ---
# --- conducting with kappa_perp / kappa_par = 1e-6. The blob sits on a
# --- mid-island flux surface (offset from the O-point): parallel
# --- conduction wraps it into a RING along that surface while the
# --- perpendicular channel is negligible, so the heat must stay INSIDE
# --- the separatrix (the A = 0 contour through the X-points): the
# --- classic magnetic-island heat-confinement problem
# --- (NIMROD/Sovinec anisotropic-diffusion benchmark class).
# ---
# --- The ions are a static heavy background (mass_factor 1e6) at a
# --- density where the cat's-eye current keeps B quasi-static, so
# --- conduction is the only T_e dynamics. The analysis asserts, across
# --- all diagnostic dumps:
# ---   1. maximum principle: no new Te extrema (monotone FD operator),
# ---   2. conservation: Sigma(Te) drift at the deposit-noise floor,
# ---   3. confinement: excess-Te fraction outside the separatrix stays
# ---      at the pollution level while the along-surface spread grows
# ---      (conduction demonstrably active).
# --- Analyse with analysis_conduction_fd.py.

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI as mpi

from pywarpx import callbacks, fields, picmi

constants = picmi.constants
comm = mpi.COMM_WORLD

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="test-suite mode (fixed diags)")
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

# ----------------------------------------------------------------------
# Problem definition
# ----------------------------------------------------------------------
L = 1.0
N = 64
n0 = 1.0e21
Te0_eV = 10.0
gamma = 5.0 / 3.0
mass_factor = 1.0e6
B0 = 2.0e-5

chi_par = 2.0e3  # parallel spread sigma = sqrt(2 chi t) ~ 0.2 L over the run
eps_perp = 1.0e-6
nsteps = 64
diag_steps = 16
tfinal = 1.0e-5
dt = tfinal / nsteps

kmode = 2.0 * np.pi / L
# blob OFFSET from the O-point onto a mid-island flux surface: parallel
# conduction wraps it into a ring along the surface (a blob centered ON
# the O-point is already surface-uniform and nothing visible happens)
xb, zb = 0.65 * L, 0.5 * L
s0 = 0.04 * L
blob_amp = 4.0

qe = constants.q_e
kb = constants.kb
Te0_K = Te0_eV * qe / kb
kappa_par_expr = f"{1.5 * n0 * kb * chi_par:.16e}"
kappa_perp_expr = f"{1.5 * n0 * kb * eps_perp * chi_par:.16e}"

mu0 = 4.0e-7 * np.pi
om_wh = (np.pi * N / L) ** 2 * B0 / (mu0 * qe * n0)
substeps = max(4, int(np.ceil(om_wh * dt / 0.05)))

# ----------------------------------------------------------------------
# PICMI setup
# ----------------------------------------------------------------------
grid = picmi.Cartesian2DGrid(
    number_of_cells=[N, N],
    lower_bound=[0.0, 0.0],
    upper_bound=[L, L],
    lower_boundary_conditions=["periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic"],
    warpx_max_grid_size=N // 2,
)

solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=gamma,
    Te=Te0_eV,
    n0=n0,
    n_floor=0.01 * n0,
    plasma_resistivity=0.0,
    substeps=substeps,
    solve_electron_energy_equation=True,
)

ions = picmi.Species(
    particle_type="H",
    name="ions",
    charge_state=1,
    mass=mass_factor * constants.m_p,
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=f"{n0}",
        momentum_expressions=["0.0", "0.0", "0.0"],
    ),
)

B_field = picmi.AnalyticInitialField(
    Bx_expression=f"{B0}*cos({kmode}*x)*sin({kmode}*z)",
    By_expression="0.0",
    Bz_expression=f"(-{B0})*sin({kmode}*x)*cos({kmode}*z)",
    warpx_do_initial_div_cleaning=False,
)

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=nsteps,
    verbose=0,
    particle_shape=1,
    warpx_serialize_initial_conditions=True,
    warpx_current_deposition_algo="esirkepov",
    warpx_use_filter=False,
)
sim.add_species(
    ions,
    layout=picmi.GriddedLayout(n_macroparticle_per_cell=[2, 2], grid=grid),
)
sim.add_applied_field(B_field)

if comm.rank == 0 and Path("diags").exists():
    shutil.rmtree("diags")
comm.Barrier()

field_diag = picmi.FieldDiagnostic(
    name="field_diag",
    grid=grid,
    period=diag_steps,
    data_list=["rho", "Te", "B"],
    write_dir="diags",
    warpx_file_prefix="field_diags",
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
)
sim.add_diagnostic(field_diag)

sim.initialize_inputs()

import pywarpx  # noqa: E402

pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_par(n,Te,t)", kappa_par_expr)
pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_perp(n,Te,t)", kappa_perp_expr)
pywarpx.hybridpicmodel.qdsmc_conduction_flux_limit_factor = 0.0
pywarpx.hybridpicmodel.qdsmc_conduction_operator = "fd"

sim.initialize_warpx()

# ----------------------------------------------------------------------
# Te blob poke at step 0 (before the first step, after the closure seed)
# ----------------------------------------------------------------------
Te_wrap = fields.MultiFabWrapper(mf_name="hybrid_electron_temperature_fp", level=0)


def poke_te():
    from pywarpx import libwarpx

    if libwarpx.libwarpx_so.get_instance().getistep(0) != 0:
        return
    x = np.linspace(0.0, L, N + 1)
    xg, zg = np.meshgrid(x, x, indexing="ij")
    prof = np.zeros_like(xg)
    for mx in range(-1, 2):
        for mz in range(-1, 2):
            dx_ = xg - xb + mx * L
            dz_ = zg - zb + mz * L
            prof += np.exp(-(dx_**2 + dz_**2) / (2.0 * s0**2))
    Te_wrap[:, :] = Te0_K * (1.0 + blob_amp * prof)


callbacks.installparticleinjection(poke_te)

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
sim.step(nsteps)

if comm.rank == 0:
    print(
        f"[conduction-fd] done: N={N} steps={nsteps} chi_par={chi_par:g} "
        f"eps_perp={eps_perp:g} substeps={substeps}"
    )
