#!/usr/bin/env python3
#
# --- Liftoff-class slab for the QDSMC electron energy equation + the
# --- insulating EB wall (the third acceptance item of the InsulatingEB
# --- scope, and the production stack of the Ohmic-hotspot problem).
# ---
# --- Ported from the ECT-campaign theta-pinch liftoff testbed
# --- (ohm_solver_plasma_cylinder_liftoff, eb_ect_yee_followup branch):
# --- a thin 3D slab (periodic in z) of an annular deuterium column inside
# --- a conducting cylindrical wall, threaded by a bias field that a
# --- Hermite-ramped external vector potential reverses -- the column lifts
# --- off the wall and implodes. ECT/conformal knobs are stripped (this
# --- branch's wall is the staircase insulating EB); the Python
# --- install_wall_scraper standoff hook of the original is REPLACED by
# --- boundary.eb_type = insulating (the C++ wall it prototyped).
# ---
# --- On top of the original ion dynamics this deck runs the electron
# --- energy equation with the production sources:
# ---   * Joule heating (eta J^2, the density-scaled power-law resistivity
# ---     of the formation runs),
# ---   * Spitzer parallel conduction kappa_par = C * Te^2.5 (+ optional
# ---     perpendicular fraction), the piece that drains the Ohmic hotspot
# ---     (without it the source cannot diffuse away and pressure grows
# ---     locally without bound -- the production blocker this validates),
# ---   * the insulating wall: standoff collection (tallied) + zero-normal-
# ---     gradient Te fill + the spill-only deposit fold.
# ---
# --- Probes: max/median Te history (hotspot growth vs conduction), the
# --- open-set electron energy, the collected-charge/energy tallies, and
# --- the usual field diagnostics.

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI as mpi

import pywarpx
from pywarpx import callbacks, fields, picmi

constants = picmi.constants
comm = mpi.COMM_WORLD

# ----------------------------------------------------------------------------
# Parameters (formation-section midplane slab, as in the ECT testbed)
# ----------------------------------------------------------------------------
M_AMU = 2  # deuterium
N_I = 1.5e20  # reference ion density (m^-3)
T_I0 = 5.0  # ion temperature (eV)
T_E0 = 5.0  # electron temperature (eV)

R_INNER = 0.65
R_OUTER = 0.75
R_PART = 0.385
ANNULUS_SMOOTH_CELLS = 2.0
R_WALL = 0.8

BZ_BIAS = -0.01  # T
BZ_REV = 0.5  # T
TAU_RAMP = 10.0e-6  # s

ETA_PLASMA = 1.0e-6  # Ohm m
ETA_VAC_FRAC = 5.0e-2
ETA_POWER = 3.0
N_TRANSITION_FRAC = 0.4

F_T_CI = 0.01
NZ = 8

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", "-r", type=int, default=64)
parser.add_argument("--nppc", type=int, default=12)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--diag-period", type=int, default=100)
parser.add_argument(
    "--standoff",
    type=float,
    default=3.0,
    help="insulating-wall standoff [cells]; 3 = the collocated liftoff "
    "survival point of the explicit-stiffness study",
)
parser.add_argument("--n-floor-frac", type=float, default=0.05)
parser.add_argument(
    "--load-margin",
    type=float,
    default=1.5,
    help="load clearance beyond the collection shell [cells]: the annulus "
    "outer radius is capped at R_WALL - (standoff+margin)*dx. Loading "
    "into the shell amputates the annulus at step 1 and the resulting "
    "grad-Pe cliff field drives runaway edge erosion into the collector "
    "(measured: ~0.3 C/step, 90%% inventory loss in 20 steps)",
)
parser.add_argument("--fill-frac", type=float, default=0.2, help="n_fill/N_I")
parser.add_argument(
    "--kappa",
    choices=["spitzer", "off"],
    default="spitzer",
    help="electron conduction: Spitzer kappa_par = C*Te^2.5 (lnL=10) with "
    "kappa_perp = perp-frac * kappa_par, or off (the Ohmic-hotspot "
    "control arm)",
)
parser.add_argument(
    "--kappa-perp-frac",
    type=float,
    default=1.0e-2,
    help="kappa_perp / kappa_par (cross-field drain at nulls/current sheets)",
)
parser.add_argument("--substeps", type=int, default=256)
parser.add_argument("--substep-rtol", type=float, default=1.0e-3)
parser.add_argument("--advance", choices=["euler", "leapfrog", "pc"], default="pc")
parser.add_argument("--equilibrium-b", type=int, choices=[0, 1], default=1)
parser.add_argument("--grid-type", default="collocated")
parser.add_argument("--openpmd", action="store_true", default=True)
parser.add_argument("--out", type=str, default="liftoff_slab")
parser.add_argument("--verbose", type=int, default=1)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

m_i = M_AMU * constants.m_p
n_floor = args.n_floor_frac * N_I
n_fill = args.fill_frac * N_I
vth = np.sqrt(constants.q_e * T_I0 / m_i)

resolution = args.resolution
dx = 2.0 / resolution
annulus_w = ANNULUS_SMOOTH_CELLS * dx

# cap the annulus outer radius clear of the collection shell (see
# --load-margin help)
r_outer_eff = min(R_OUTER, R_WALL - (args.standoff + args.load_margin) * dx)
if comm.rank == 0 and r_outer_eff < R_OUTER:
    print(
        f"[liftoff] annulus outer radius capped {R_OUTER} -> "
        f"{r_outer_eff:.4f} m (shell clearance at this resolution/standoff)",
        flush=True,
    )
assert r_outer_eff > R_INNER + annulus_w, (
    "standoff+margin leaves no room for the annulus at this resolution; "
    "reduce --standoff or raise --resolution"
)
lz = NZ * dx
zmin, zmax = 5.0 - lz / 2.0, 5.0 + lz / 2.0

w_ci = constants.q_e * abs(BZ_REV) / m_i
dt = F_T_CI * 2.0 * np.pi / w_ci

w_pi = np.sqrt(constants.q_e**2 * N_I / (constants.ep0 * m_i))
l_i = constants.c / w_pi
vA = abs(BZ_REV) / np.sqrt(constants.mu0 * N_I * m_i)
dL2 = 1.0 / (2.0 / dx**2 + 1.0 / dx**2)
eta_max = constants.mu0 * dL2 / (2.0 * dt)
eta_hyper = constants.mu0 * 0.2 * l_i * vA * dL2

# Spitzer kappa_par = C * Te[eV]^2.5 [W/(m K)], lnLambda = 10:
#   kappa = 3.16 n k_B (q_e Te/m_e) tau_e,  tau_e = 3.44e11 Te^1.5/(n lnL)
# (n cancels).  C(lnL=10) ~ 0.264; kappa(10 eV) ~ 84 W/(m K).
LN_LAMBDA = 10.0
KAPPA_C = 3.16 * constants.kb * (constants.q_e / constants.m_e) * 3.44e11 / LN_LAMBDA

# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
grid = picmi.Cartesian3DGrid(
    number_of_cells=[resolution, resolution, NZ],
    lower_bound=[-1.0, -1.0, zmin],
    upper_bound=[1.0, 1.0, zmax],
    lower_boundary_conditions=["dirichlet", "dirichlet", "periodic"],
    upper_boundary_conditions=["dirichlet", "dirichlet", "periodic"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
    warpx_max_grid_size=2048,
    warpx_max_grid_size_x=max(resolution // 2, 8),
    warpx_blocking_factor=8,
)


def hermite_ramp_expression(b0, b1, tau):
    s = f"min(max(t/{tau:.9e},0),1)"
    return f"({b0:.9e} + ({b1 - b0:.9e})*({s})*({s})*(3-2*({s})))"


A_ext = {
    "uniform_reversal": {
        "Ax_external_function": "-0.5*y",
        "Ay_external_function": "0.5*x",
        "Az_external_function": "0",
        "A_time_external_function": hermite_ramp_expression(BZ_BIAS, BZ_REV, TAU_RAMP),
    },
}

# density-scaled power-law resistivity (formation-run form)
a_pl = (args.n_floor_frac / N_TRANSITION_FRAC) ** ETA_POWER
eta_vac = ETA_VAC_FRAC * eta_max
res_str = (
    f"eta_plasma + (eta_vac - eta_plasma)"
    f"*max(0.0, (rho_f/max(rho,rho_f))**({ETA_POWER:.6g}) - ({a_pl:.12g}))"
    f"/({1.0 - a_pl:.12g})"
)

solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=5.0 / 3.0,
    Te=T_E0,
    n0=N_I,
    n_floor=n_floor,
    plasma_resistivity=res_str,
    eta_plasma=ETA_PLASMA,
    eta_vac=eta_vac,
    rho_f=constants.q_e * n_floor,
    plasma_hyper_resistivity=eta_hyper,
    substeps=args.substeps,
    use_rkf45=True,
    substep_rtol=args.substep_rtol,
    substep_atol=1.0e-8,
    max_substep_attempts=1000,
    solve_electron_energy_equation=True,
    include_joule_heating=True,
    A_external=A_ext,
)

embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function="(x**2+y**2-R_w**2)",
    R_w=R_WALL,
    eb_type="insulating",
    eb_standoff_cells=args.standoff,
)

simulation = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=args.steps,
    particle_shape=1,
    verbose=args.verbose,
    warpx_embedded_boundary=embedded_boundary,
    warpx_current_deposition_algo="direct",
    warpx_grid_type=args.grid_type,
)

# equilibrium diamagnetic Bz seed (removes the startup transient; the
# external add supplies the bias part -- see the ECT deck's double-count
# note, upstream #6047)
n_annulus = N_I * R_PART**2 / (r_outer_eff**2 - R_INNER**2)
if args.equilibrium_b:
    k_eq = 2.0 * constants.mu0 * (T_I0 + T_E0) * constants.q_e
    sgn = "" if BZ_BIAS >= 0 else "-"
    rr = "sqrt(x*x+y*y)"
    e_in = f"0.5*(1.0+tanh(({rr}-{R_INNER})/{annulus_w}))"
    e_out = f"0.5*(1.0+tanh(({r_outer_eff}-{rr})/{annulus_w}))"
    n_of_r = f"({n_fill}*(1.0-{e_in})+{n_annulus}*{e_in}*{e_out})"
    bz_seed = f"{sgn}sqrt({BZ_BIAS**2}+{k_eq}*({n_annulus}-{n_of_r}))-({BZ_BIAS})"
else:
    bz_seed = "0"
simulation.add_applied_field(
    picmi.AnalyticInitialField(
        Bx_expression="0",
        By_expression="0",
        Bz_expression=bz_seed,
        warpx_do_initial_div_cleaning=False,
    )
)

# annular column + interior fill, tanh-softened edges. The outer radius is
# capped so the load (and its tanh tail) stays clear of the collection
# shell; the R_PART inventory formula below rescales n_annulus so the
# column carries the same line density regardless of the cap.
r_expr = "sqrt(x*x+y*y)"
edge_in = f"0.5*(1.0+tanh(({r_expr}-R_in)/sw))"
edge_out = f"0.5*(1.0+tanh((R_out-{r_expr})/sw))"
ions = picmi.Species(
    name="ions",
    mass=m_i,
    charge="q_e",
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=f"n_f*(1.0-{edge_in})+n_a*{edge_in}*{edge_out}",
        momentum_expressions=["0", "0", "0"],
        warpx_momentum_spread_expressions=[f"{vth}"] * 3,
        warpx_density_min=0.05 * n_fill,
        n_a=n_annulus,
        n_f=n_fill,
        R_in=R_INNER,
        R_out=r_outer_eff,
        sw=annulus_w,
    ),
)
simulation.add_species(
    ions,
    layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=args.nppc),
)

ion_ion_coulomb = picmi.CoulombCollisions(
    name="ion_ion_Coulomb", species=[ions, ions], CoulombLog=12
)
simulation.collisions = [ion_ion_coulomb]

if comm.rank == 0 and Path("diags").exists():
    shutil.rmtree("diags")
comm.Barrier()

field_diag = picmi.FieldDiagnostic(
    name="field_diag",
    grid=grid,
    period=args.diag_period,
    data_list=["B", "E", "rho", "J", "Te"],
    write_dir="diags",
    warpx_file_prefix="field_diags",
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
)
simulation.add_diagnostic(field_diag)

simulation.initialize_inputs()

# branch-side energy-equation knobs (pywarpx bucket, annulus-deck pattern)
pywarpx.hybridpicmodel.qdsmc_time_advance = args.advance
if args.kappa == "spitzer":
    pywarpx.hybridpicmodel.__setattr__(
        "qdsmc_kappa_par(n,Te,t)", f"{KAPPA_C:.6e}*Te**2.5"
    )
    if args.kappa_perp_frac > 0.0:
        pywarpx.hybridpicmodel.__setattr__(
            "qdsmc_kappa_perp(n,Te,t)",
            f"{args.kappa_perp_frac * KAPPA_C:.6e}*Te**2.5",
        )

simulation.initialize_warpx()

# ----------------------------------------------------------------------------
# Probes: hotspot growth, open-set energy, wall tallies
# ----------------------------------------------------------------------------
Te_wrap = fields.MultiFabWrapper(mf_name="hybrid_electron_temperature_fp", level=0)
rho_wrap = fields.MultiFabWrapper(mf_name="rho_fp", level=0)
qe = constants.q_e
K_PER_EV = constants.q_e / constants.kb

history = {"step": [], "te_max": [], "te_med": [], "u_open": []}


def probe():
    step = simulation.extension.warpx.getistep(lev=0)
    if step % 20 != 0:
        return
    te = Te_wrap[:, :, :] / K_PER_EV  # K -> eV
    ne = rho_wrap[:, :, :] / qe
    open_mask = ne > n_floor
    if comm.rank == 0 and open_mask.any():
        history["step"].append(step)
        history["te_max"].append(float(te[open_mask].max()))
        history["te_med"].append(float(np.median(te[open_mask])))
        history["u_open"].append(float(np.sum(ne[open_mask] * te[open_mask])))


callbacks.installafterstep(probe)

simulation.step(args.steps)

wx = simulation.extension.warpx
q_col = wx.get_eb_collected_charge("ions")
e_col = wx.get_eb_collected_energy("ions")

if comm.rank == 0:
    h = {k: np.asarray(v) for k, v in history.items()}
    print(
        f"[liftoff] done: N={resolution} steps={args.steps} kappa={args.kappa} "
        f"| Te max {h['te_max'][0]:.2f} -> {h['te_max'][-1]:.2f} eV "
        f"(median {h['te_med'][-1]:.2f}) "
        f"| collected q={q_col:.4e} C E={e_col:.4e} J",
        flush=True,
    )
    np.savez_compressed(
        f"{args.out}.npz",
        resolution=resolution,
        steps=args.steps,
        dt=dt,
        kappa=args.kappa,
        kappa_C=KAPPA_C,
        kappa_perp_frac=args.kappa_perp_frac,
        standoff=args.standoff,
        te_final=Te_wrap[:, :, :] / K_PER_EV,
        rho_final=rho_wrap[:, :, :],
        q_collected=q_col,
        e_collected=e_col,
        **h,
    )
