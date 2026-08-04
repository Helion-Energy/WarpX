#!/usr/bin/env python3
"""Phase-0 convergence harness for the QDSMC electron energy equation.

Sets up a 2D periodic hybrid-PIC run where the electron fluid velocity V_e is
controlled through the ion current (B = 0, uniform density => V_e = J_i/rho =
the ion drift field exactly), pokes a Gaussian electron-temperature blob into
hybrid_electron_temperature_fp at step 0 (via the particleinjection callback,
which fires AFTER HybridPICInitializeRhoJandB's closure mirror), advances, and
saves the final Te field for error analysis.

Modes
-----
at_rest   : v = 0 everywhere. Te must be stationary; any change is the
            per-step gather/scatter smoothing (plan items E5/E6).
translate : uniform diagonal drift. Exact solution = periodically shifted
            blob; time discretization is exact (v constant along
            trajectories), so the error is purely spatial remap error.
rotate    : tapered rigid-body vortex (rigid core, cosine taper to zero well
            inside the periodic box). Velocity varies along trajectories =>
            measures the time order via dt self-convergence.

Physics knobs are chosen so the closure-implied Te is uniform (= Te0):
gamma = 5/3 with n0_ref = n0 and uniform density, so the step-0 closure
mirror writes Te = Te0 everywhere and the poked blob rides on a consistent
background. Ions are given a huge mass so they free-stream without
acceleration on the test timescale.

Usage (from this directory, with the warpx-qdsmc venv python):
  python3 qdsmc_advection_test.py --mode rotate --ncell 128 --nsteps 128 --out r128.npz
"""

import argparse
import sys

import numpy as np

from pywarpx import callbacks, fields, picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["at_rest", "translate", "rotate"], required=True)
parser.add_argument("--ncell", type=int, default=64, help="cells per direction")
parser.add_argument("--nsteps", type=int, default=64, help="number of steps to run")
parser.add_argument(
    "--tfinal",
    type=float,
    default=None,
    help="total simulated time [s]; dt = tfinal/nsteps. Default: 0.5*L/v_max "
    "(translate/rotate) or 64 * (L/64)/v_max (at_rest).",
)
parser.add_argument("--vmax", type=float, default=1.0e5, help="peak |V_e| [m/s]")
parser.add_argument(
    "--blob-sigma", type=float, default=0.04, help="Te blob sigma as a fraction of L"
)
parser.add_argument(
    "--advance",
    choices=["euler", "leapfrog", "pc"],
    default="euler",
    help="hybrid_pic_model.qdsmc_time_advance scheme",
)
parser.add_argument(
    "--grad-deposit",
    type=int,
    choices=[0, 1],
    default=0,
    help="hybrid_pic_model.qdsmc_gradient_deposit (B1)",
)
parser.add_argument("--out", type=str, required=True, help="output .npz path")
parser.add_argument("--verbose", type=int, default=0)
args = parser.parse_args()

# ----------------------------------------------------------------------
# Problem definition
# ----------------------------------------------------------------------
L = 1.0  # domain edge [m]
N = args.ncell
n0 = 1.0e18  # ion/electron density [m^-3]
Te0_eV = 10.0  # background electron temperature [eV]
gamma = 5.0 / 3.0
mass_factor = 1.0e6  # ion mass multiplier: freeze ion dynamics
v_max = args.vmax

# Rotation geometry (rotate mode): rigid core out to r1, cosine taper to r2.
xc = zc = 0.5 * L  # vortex center
r1, r2 = 0.25 * L, 0.45 * L
omega0 = v_max / r2  # peak edge speed ~ v_max

# Te blob: compact Gaussian riding on Te0. In rotate mode it sits inside the
# rigid core (support ~ rb + 3*sb < r1) so it rotates rigidly.
rb = 0.12 * L  # blob center offset from vortex center (rotate)
sb = args.blob_sigma * L  # blob sigma
blob_amp = 1.0  # peak fractional bump: Te_peak = Te0*(1+A)

if args.tfinal is None:
    if args.mode == "at_rest":
        tfinal = 64.0 * (L / 64.0) / v_max
    else:
        tfinal = 0.5 * L / v_max
else:
    tfinal = args.tfinal
dt = tfinal / args.nsteps

if args.mode == "translate":
    vdrift = v_max / np.sqrt(2.0)
    ux_expr = f"{vdrift}"
    uz_expr = f"{vdrift}"
    xb, zb = 0.3 * L, 0.3 * L
elif args.mode == "rotate":
    # omega_eff(r): rigid inside r1, cosine taper to zero at r2.
    om = (
        f"({omega0}*if(sqrt((x-{xc})^2+(z-{zc})^2)<{r1},1,"
        f"if(sqrt((x-{xc})^2+(z-{zc})^2)>{r2},0,"
        f"0.5*(1+cos(pi*(sqrt((x-{xc})^2+(z-{zc})^2)-{r1})/({r2}-{r1}))))))"
    )
    ux_expr = f"(-{om}*(z-{zc}))"
    uz_expr = f"({om}*(x-{xc}))"
    xb, zb = xc + rb, zc
else:  # at_rest
    ux_expr = "0.0"
    uz_expr = "0.0"
    xb, zb = 0.5 * L, 0.5 * L

# ----------------------------------------------------------------------
# PICMI setup
# ----------------------------------------------------------------------
grid = picmi.Cartesian2DGrid(
    number_of_cells=[N, N],
    lower_bound=[0.0, 0.0],
    upper_bound=[L, L],
    lower_boundary_conditions=["periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic"],
    warpx_max_grid_size=N,
)

solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=gamma,
    Te=Te0_eV,
    n0=n0,
    n_floor=0.01 * n0,
    plasma_resistivity=0.0,
    substeps=4,
    solve_electron_energy_equation=True,
)

ions = picmi.Species(
    particle_type="H",
    name="ions",
    charge_state=1,
    mass=mass_factor * constants.m_p,
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=f"{n0}",
        momentum_expressions=[ux_expr, "0.0", uz_expr],
    ),
)

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=args.nsteps,
    verbose=args.verbose,
    particle_shape=1,
    warpx_serialize_initial_conditions=True,
    warpx_current_deposition_algo="esirkepov",
    warpx_use_filter=False,
)
sim.add_species(
    ions,
    layout=picmi.GriddedLayout(n_macroparticle_per_cell=[2, 2], grid=grid),
)

sim.initialize_inputs()

# Time-advance scheme (bake-off switch). Not a picmi kwarg, so set the
# bucket directly after initialize_inputs (picmi only clobbers attributes it
# writes itself); verify in warpx_used_inputs.
import pywarpx  # noqa: E402

pywarpx.hybridpicmodel.qdsmc_time_advance = args.advance
pywarpx.hybridpicmodel.qdsmc_gradient_deposit = args.grad_deposit

sim.initialize_warpx()

# ----------------------------------------------------------------------
# Te blob poke (step 0, after the closure mirror) + initial state capture
# ----------------------------------------------------------------------
Te_wrap = fields.MultiFabWrapper(mf_name="hybrid_electron_temperature_fp", level=0)

K_per_eV = picmi.constants.q_e / picmi.constants.kb
state = {"initial": None, "sum0": None}


def _node_coords():
    te = Te_wrap[:, :]
    nx, nz = te.shape
    x = np.linspace(0.0, L, nx)
    z = np.linspace(0.0, L, nz)
    return np.meshgrid(x, z, indexing="ij")


def blob(xg, zg, x0, z0):
    """Periodically wrapped Gaussian bump (fractional)."""
    dxp = (xg - x0 + 0.5 * L) % L - 0.5 * L
    dzp = (zg - z0 + 0.5 * L) % L - 0.5 * L
    return blob_amp * np.exp(-(dxp**2 + dzp**2) / (2.0 * sb**2))


def poke_te():
    if libwarpx_istep() != 0:
        return
    xg, zg = _node_coords()
    te = Te0_eV * K_per_eV * (1.0 + blob(xg, zg, xb, zb))
    Te_wrap[:, :] = te
    state["initial"] = te.copy()
    state["sum0"] = te[:-1, :-1].sum()
    if args.verbose:
        print(f"[harness] poked Te blob at step 0: max {te.max():.6e} K")


def libwarpx_istep():
    from pywarpx import libwarpx

    return libwarpx.libwarpx_so.get_instance().getistep(0)


callbacks.installparticleinjection(poke_te)

# ----------------------------------------------------------------------
# Run and save
# ----------------------------------------------------------------------
sim.step(args.nsteps)

te_final = Te_wrap[:, :]
np.savez_compressed(
    args.out,
    mode=args.mode,
    advance=args.advance,
    ncell=N,
    nsteps=args.nsteps,
    dt=dt,
    tfinal=tfinal,
    L=L,
    vmax=v_max,
    Te0_eV=Te0_eV,
    blob=[xb, zb, sb, blob_amp],
    rotate=[xc, zc, r1, r2, omega0],
    te_initial=state["initial"],
    te_final=te_final,
    te_sum0=state["sum0"],
    te_sum1=te_final[:-1, :-1].sum(),
)
print(
    f"[harness] done: mode={args.mode} N={N} dt={dt:.4e} steps={args.nsteps} "
    f"sum(Te) drift = {(te_final[:-1, :-1].sum() - state['sum0']) / state['sum0']:.3e}"
)
sys.stdout.flush()
