#!/usr/bin/env python3
"""Steady anisotropic-diffusion pollution benchmark (Sovinec/NIMROD form,
the Chacon et al. paper's Sec. 3.1 configuration, explicit-feasible at
eps >= ~1e-3).

Domain [0,1]^2 with isothermal walls pinned at Te0 and the flux function

    psi = sin(pi x) sin(pi z)       (O-point at the center,
    Bx = -B0 sin(pi x) cos(pi z),    separatrix ON the walls,
    Bz =  B0 cos(pi x) sin(pi z))    b tangent to the walls)

A volumetric source S = 2 pi^2 chi_perp * a * sin(pi x) sin(pi z) [K/s]
(applied per step, operator-split) balances perpendicular diffusion of
the same mode shape, whose steady solution is dTe = a sin(pi x)
sin(pi z) when the operator carries NO parallel pollution. The measured
steady central amplitude gives the paper's metric directly:

    chi_eff / chi_perp = a / dTe(center)_ss      (their 1/T(0,0))
    delta_chi          = (chi_eff - chi_perp) / chi_par

Conduction-only skeleton (static heavy ions, high density, quasi-static
B); walls are dirichlet (PEC) field BCs with the P2 subcycle-cadence
isothermal pins carrying the thermal boundary condition.

Usage:
  python3 qdsmc_nimrod_steady_test.py --eps 1e-2 --fd-order 2 --out s.npz
"""

import argparse
import sys

import numpy as np

from pywarpx import callbacks, fields, picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument("--ncell", type=int, default=64)
parser.add_argument("--chi-perp", type=float, default=127.0, help="chi_perp [m^2/s]")
parser.add_argument("--eps", type=float, default=1.0e-2, help="chi_perp/chi_par ratio")
parser.add_argument(
    "--amp", type=float, default=1.0, help="target steady amplitude [eV]"
)
parser.add_argument(
    "--taus", type=float, default=6.0, help="run length in relaxation times"
)
parser.add_argument("--nsteps", type=int, default=256)
parser.add_argument("--fd-order", type=int, choices=[2, 4], default=2)
parser.add_argument(
    "--fd-limiter", choices=["none", "upwind1", "smart"], default="smart"
)
parser.add_argument("--fd-time", choices=["ssprk2", "rkf45"], default="ssprk2")
parser.add_argument("--fd-cfl", type=float, default=0.4)
parser.add_argument(
    "--iso-b", type=float, default=-1.0, help="iso_B blend [T] (<0 = off)"
)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--verbose", type=int, default=0)
args = parser.parse_args()

# ----------------------------------------------------------------------
# Problem definition
# ----------------------------------------------------------------------
L = 1.0
N = args.ncell
n0 = 1.0e21
Te0_eV = 10.0
gamma = 5.0 / 3.0
mass_factor = 1.0e6
B0 = 2.0e-6  # quasi-static direction field; low so whistler substeps stay cheap

chi_perp = args.chi_perp
chi_par = chi_perp / args.eps
kpi = np.pi / L

qe = constants.q_e
kb = constants.kb
K_per_eV = qe / kb
Te0_K = Te0_eV * K_per_eV
a_K = args.amp * K_per_eV

# relaxation time of the source mode (k^2 = 2 pi^2 for sin*sin)
tau = 1.0 / (2.0 * kpi**2 * chi_perp)
tfinal = args.taus * tau
dt = tfinal / args.nsteps

kappa_par_expr = f"{1.5 * n0 * kb * chi_par:.16e}"
kappa_perp_expr = f"{1.5 * n0 * kb * chi_perp:.16e}"

mu0 = 4.0e-7 * np.pi
om_wh = (np.pi * N / L) ** 2 * B0 / (mu0 * qe * n0)
substeps = max(4, int(np.ceil(om_wh * dt / 0.05)))

# ----------------------------------------------------------------------
# PICMI setup (walls: dirichlet fields + isothermal conduction pins)
# ----------------------------------------------------------------------
grid = picmi.Cartesian2DGrid(
    number_of_cells=[N, N],
    lower_bound=[0.0, 0.0],
    upper_bound=[L, L],
    lower_boundary_conditions=["dirichlet", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["reflecting", "reflecting"],
    upper_boundary_conditions_particles=["reflecting", "reflecting"],
    warpx_max_grid_size=N,
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
    Bx_expression=f"(-{B0})*sin({kpi}*x)*cos({kpi}*z)",
    By_expression="0.0",
    Bz_expression=f"{B0}*cos({kpi}*x)*sin({kpi}*z)",
    warpx_do_initial_div_cleaning=False,
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
sim.add_applied_field(B_field)

sim.initialize_inputs()

import pywarpx  # noqa: E402

pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_par(n,Te,t)", kappa_par_expr)
pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_perp(n,Te,t)", kappa_perp_expr)
pywarpx.hybridpicmodel.qdsmc_conduction_flux_limit_factor = 0.0
pywarpx.hybridpicmodel.qdsmc_conduction_operator = "fd"
pywarpx.hybridpicmodel.qdsmc_conduction_fd_order = args.fd_order
pywarpx.hybridpicmodel.qdsmc_conduction_fd_limiter = args.fd_limiter
pywarpx.hybridpicmodel.qdsmc_conduction_fd_cfl = args.fd_cfl
pywarpx.hybridpicmodel.qdsmc_conduction_fd_time = args.fd_time
pywarpx.hybridpicmodel.qdsmc_conduction_bc_lo = ["isothermal", "isothermal"]
pywarpx.hybridpicmodel.qdsmc_conduction_bc_hi = ["isothermal", "isothermal"]
pywarpx.hybridpicmodel.qdsmc_conduction_bc_Te_lo = [Te0_eV, Te0_eV]
pywarpx.hybridpicmodel.qdsmc_conduction_bc_Te_hi = [Te0_eV, Te0_eV]
if args.iso_b > 0.0:
    pywarpx.hybridpicmodel.qdsmc_conduction_iso_B = args.iso_b

sim.initialize_warpx()

# ----------------------------------------------------------------------
# Volumetric source (operator-split, per step) + steady-amplitude probe
# ----------------------------------------------------------------------
Te_wrap = fields.MultiFabWrapper(mf_name="hybrid_electron_temperature_fp", level=0)

state = {"S": None, "shape": None, "amp_t": [], "t": []}


def _grids():
    x = np.linspace(0.0, L, N + 1)
    xg, zg = np.meshgrid(x, x, indexing="ij")
    return np.sin(kpi * xg) * np.sin(kpi * zg)


def add_source():
    if state["S"] is None:
        state["shape"] = _grids()
        state["S"] = 2.0 * kpi**2 * chi_perp * a_K * state["shape"]
    te = Te_wrap[:, :]
    Te_wrap[:, :] = te + state["S"] * dt

    from pywarpx import libwarpx

    it = libwarpx.libwarpx_so.get_instance().getistep(0)
    shp = state["shape"]
    nrm = (shp[1:-1, 1:-1] ** 2).sum()
    amp = ((te[1:-1, 1:-1] - Te0_K) * shp[1:-1, 1:-1]).sum() / nrm
    state["amp_t"].append(float(amp) / K_per_eV)
    state["t"].append(it * dt)


callbacks.installafterstep(add_source)

# ----------------------------------------------------------------------
# Run and score
# ----------------------------------------------------------------------
sim.step(args.nsteps)

t = np.array(state["t"])
amp = np.array(state["amp_t"])  # [eV]

# steady amplitude from the last relaxation time
sel = t > (args.taus - 1.0) * tau
amp_ss = float(amp[sel].mean())
chi_ratio = args.amp / amp_ss  # chi_eff / chi_perp = 1/T(0,0) in paper units
delta_chi = (chi_ratio - 1.0) * chi_perp / chi_par

np.savez_compressed(
    args.out,
    ncell=N,
    nsteps=args.nsteps,
    dt=dt,
    tfinal=tfinal,
    tau=tau,
    chi_perp=chi_perp,
    chi_par=chi_par,
    eps=args.eps,
    amp_target=args.amp,
    fd_order=args.fd_order,
    fd_limiter=args.fd_limiter,
    fd_time=args.fd_time,
    iso_b=args.iso_b,
    substeps=substeps,
    t=t,
    amp_t=amp,
    amp_ss=amp_ss,
    chi_ratio=chi_ratio,
    delta_chi=delta_chi,
)
print(
    f"[harness] nimrod-steady done: order={args.fd_order} N={N} "
    f"eps={args.eps:.1e} amp_ss={amp_ss:.4f} eV (target {args.amp:g}) "
    f"chi_eff/chi_perp={chi_ratio:.4f} (paper 1/T(0,0)) "
    f"delta_chi/chi_par={delta_chi:+.3e}"
)
sys.stdout.flush()
