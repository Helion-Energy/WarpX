#!/usr/bin/env python3
"""Ion-acoustic conduction-crossover instrument (closure benchmark).

A cold-ion, quiet-start ion-acoustic standing wave in a uniform plasma.
The electron closure sets the wave speed:

    polytropic / energy-eq with chi = 0 :  omega = k * sqrt(5/3) * c_iso
    energy-eq  with strong parallel chi :  omega = k * c_iso
    (c_iso^2 = kB Te0 / m_i; cold ions, quasi-neutral, v_A << c_iso)

with the full conduction bridge given by the linear dispersion relation

    x^3 + i R x^2 - (5/3) x - i R = 0,   x = omega/(k c_iso),
    R = chi k / c_iso

(from continuity + ion momentum + electron energy with q = -kappa dTe/dz;
the damped-acoustic root has Re x in [1, sqrt(5/3)] and Im x < 0 with a
damping maximum near R ~ 1). The k || B arm walks the bridge as chi
rises; the k _|_ B arm must NOT shift (kappa_perp = 0; conduction along
B sees no gradient) — anisotropic conduction expressed in ION dynamics.

The wave is seeded as an exact velocity perturbation on a gridded quiet
start (no deposit noise), amplitude ~1e-3 c_s, and measured by
projecting the x-averaged ion charge density and Te onto exp(i k z)
each step. Fitting (frequency/damping vs the cubic root) lives in the
driver, not here.

Usage:
  python3 qdsmc_acoustic_test.py --closure ee --chi 5e4 --b-orient par --out a.npz
"""

import argparse
import sys

import numpy as np

from pywarpx import callbacks, fields, picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument("--closure", choices=["poly", "ee"], required=True)
parser.add_argument(
    "--chi",
    type=float,
    default=0.0,
    help="parallel thermal diffusivity chi [m^2/s] (0 = conduction off; "
    "ee closure only)",
)
parser.add_argument("--b-orient", choices=["par", "perp"], default="par")
parser.add_argument("--ncell-z", type=int, default=64)
parser.add_argument("--ncell-x", type=int, default=16)
parser.add_argument("--mode", type=int, default=1, help="wave mode number in z")
parser.add_argument("--amp", type=float, default=1.0e-3, help="u0 / c_iso")
parser.add_argument("--periods", type=float, default=4.0)
parser.add_argument(
    "--steps-per-period", type=int, default=64, help="isothermal-period sampling"
)
parser.add_argument(
    "--conduction-op",
    choices=["sde", "fd"],
    default="fd",
    help="conduction operator for the ee arm",
)
parser.add_argument("--fd-cfl", type=float, default=0.4)
parser.add_argument("--advance", choices=["euler", "leapfrog", "pc"], default="pc")
parser.add_argument("--depo", choices=["esirkepov", "direct"], default="esirkepov")
parser.add_argument("--measure", type=int, choices=[0, 1], default=1)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--verbose", type=int, default=0)
args = parser.parse_args()

# ----------------------------------------------------------------------
# Problem definition
# ----------------------------------------------------------------------
Lz = 1.0
Lx = 0.25
Nz = args.ncell_z
Nx = args.ncell_x
n0 = 1.0e18
Te0_eV = 10.0
gamma = 5.0 / 3.0
B0 = 2.0e-5  # v_A/c_iso ~ 0.014: acoustic, not magnetosonic

qe = constants.q_e
kb = constants.kb
m_i = constants.m_p
Te0_K = Te0_eV * qe / kb

c_iso = np.sqrt(kb * Te0_K / m_i)
kw = 2.0 * np.pi * args.mode / Lz
u0 = args.amp * c_iso

# time base: the binding constraint is the ACOUSTIC CFL AT THE GRID
# SCALE, not the wave period — with mobile real-mass ions the grid-scale
# mode has omega_max = k_max c_ad, and the explicit ion push + algebraic
# Ohm E is unstable for omega_max*dt > ~2 (measured: roundoff grid noise
# detonates in ~30 steps at omega_max*dt ~ 4). Take the stricter of the
# CFL and the requested per-period sampling.
T_iso = 2.0 * np.pi / (kw * c_iso)
c_ad = np.sqrt(gamma) * c_iso
dx_min = min(Lx / Nx, Lz / Nz)
dt_cfl = 0.3 * dx_min / c_ad
dt = min(T_iso / args.steps_per_period, dt_cfl)
nsteps = int(np.ceil(args.periods * T_iso / dt))

# R = chi k / c_iso, the crossover parameter (recorded for the driver)
R_par = args.chi * kw / c_iso

kappa_par_expr = f"{1.5 * n0 * kb * args.chi:.16e}"

mu0 = 4.0e-7 * np.pi
# the acoustic dt is ~6x the parabolic-instrument dt, so the whistler
# substep count must be set with margin: om_wh*dt_sub <= 0.05 (at 0.09
# the machine-eps whistler seed grows and NaNs the run within ~30 steps)
om_wh = (np.pi * Nz / Lz) ** 2 * B0 / (mu0 * qe * n0)
substeps = max(4, int(np.ceil(om_wh * dt / 0.05)))

# ----------------------------------------------------------------------
# PICMI setup
# ----------------------------------------------------------------------
grid = picmi.Cartesian2DGrid(
    number_of_cells=[Nx, Nz],
    lower_bound=[0.0, 0.0],
    upper_bound=[Lx, Lz],
    lower_boundary_conditions=["periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic"],
    warpx_max_grid_size=max(Nx, Nz),
)

solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=gamma,
    Te=Te0_eV,
    n0=n0,
    n_floor=0.01 * n0,
    plasma_resistivity=0.0,
    substeps=substeps,
    solve_electron_energy_equation=(args.closure == "ee"),
)

ions = picmi.Species(
    particle_type="H",
    name="ions",
    charge_state=1,
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=f"{n0}",
        momentum_expressions=["0.0", "0.0", f"{u0:.16e}*sin({kw:.16e}*z)"],
    ),
)

if args.b_orient == "par":
    bx_expr, bz_expr = "0.0", f"{B0}"
else:
    bx_expr, bz_expr = f"{B0}", "0.0"
B_field = picmi.AnalyticInitialField(
    Bx_expression=bx_expr, By_expression="0.0", Bz_expression=bz_expr
)

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=nsteps,
    verbose=args.verbose,
    particle_shape=1,
    warpx_serialize_initial_conditions=True,
    warpx_current_deposition_algo=args.depo,
    warpx_use_filter=False,
)
sim.add_species(
    ions,
    layout=picmi.GriddedLayout(n_macroparticle_per_cell=[2, 2], grid=grid),
)
sim.add_applied_field(B_field)

sim.initialize_inputs()

if args.closure == "ee":
    import pywarpx  # noqa: E402

    pywarpx.hybridpicmodel.qdsmc_time_advance = args.advance
    pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_par(n,Te,t)", kappa_par_expr)
    pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_perp(n,Te,t)", "0.0")
    pywarpx.hybridpicmodel.qdsmc_conduction_flux_limit_factor = 0.0
    pywarpx.hybridpicmodel.qdsmc_conduction_operator = args.conduction_op
    pywarpx.hybridpicmodel.qdsmc_conduction_fd_cfl = args.fd_cfl

sim.initialize_warpx()

# ----------------------------------------------------------------------
# Per-step mode projection of the x-averaged density and Te
# ----------------------------------------------------------------------
rho_wrap = fields.MultiFabWrapper(mf_name="rho_fp", level=0)
Te_wrap = (
    fields.MultiFabWrapper(mf_name="hybrid_electron_temperature_fp", level=0)
    if args.closure == "ee"
    else None
)

series = {"t": [], "n_re": [], "n_im": [], "te_re": [], "te_im": []}
mode_basis = {}


def _basis(nz_nodes):
    if "e" not in mode_basis:
        z = np.linspace(0.0, Lz, nz_nodes)[:-1]
        mode_basis["e"] = np.exp(-1j * kw * z)
    return mode_basis["e"]


def project():
    from pywarpx import libwarpx

    it = libwarpx.libwarpx_so.get_instance().getistep(0)
    rho = rho_wrap[:, :]
    prof = rho[:-1, :-1].mean(axis=0) / qe  # x-averaged n_i(z)
    e = _basis(rho.shape[1])
    amp_n = 2.0 * (prof * e).mean() / n0
    series["t"].append(it * dt)
    series["n_re"].append(amp_n.real)
    series["n_im"].append(amp_n.imag)
    if Te_wrap is not None:
        te = Te_wrap[:, :]
        tprof = te[:-1, :-1].mean(axis=0)
        amp_t = 2.0 * (tprof * e).mean() / Te0_K
        series["te_re"].append(amp_t.real)
        series["te_im"].append(amp_t.imag)
    else:
        series["te_re"].append(0.0)
        series["te_im"].append(0.0)


if args.measure:
    callbacks.installafterstep(project)

# ----------------------------------------------------------------------
# Run and save
# ----------------------------------------------------------------------
sim.step(nsteps)

t = np.array(series["t"])
n_amp = np.array(series["n_re"]) + 1j * np.array(series["n_im"])
te_amp = np.array(series["te_re"]) + 1j * np.array(series["te_im"])

np.savez_compressed(
    args.out,
    closure=args.closure,
    chi=args.chi,
    R=R_par,
    b_orient=args.b_orient,
    conduction_op=args.conduction_op,
    ncell_z=Nz,
    ncell_x=Nx,
    mode=args.mode,
    amp=args.amp,
    dt=dt,
    nsteps=nsteps,
    substeps=substeps,
    n0=n0,
    Te0_eV=Te0_eV,
    c_iso=c_iso,
    kw=kw,
    t=t,
    n_amp=n_amp,
    te_amp=te_amp,
)
peak = float(np.abs(n_amp).max())
print(
    f"[harness] acoustic done: closure={args.closure} chi={args.chi:g} "
    f"R={R_par:.3g} orient={args.b_orient} N=({Nx},{Nz}) steps={nsteps} "
    f"|n_k|max={peak:.3e} (seed amp {args.amp:g})"
)
sys.stdout.flush()
