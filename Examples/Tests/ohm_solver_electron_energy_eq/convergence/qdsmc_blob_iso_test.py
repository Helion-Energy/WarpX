#!/usr/bin/env python3
#
# --- Point-source conduction isotropy instrument: a hot Gaussian Te blob
# --- diffusing in a uniform, static plasma with b out-of-plane (2D XZ,
# --- uniform By), so chi_perp acts isotropically in the simulation plane.
# --- The axis-pair quadrature launch imprints a cos(4 theta) fourth-moment
# --- anisotropy on the spreading blob (the star pattern);
# --- qdsmc_conduction_isotropic_launch adds the 45-degree-rotated lattice
# --- at half variance, which cancels it. This deck measures the a4
# --- anisotropy amplitude on the half-max contour, A/B.

import argparse
import sys

import numpy as np

import pywarpx
from pywarpx import callbacks, fields, picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument("--ncell", type=int, default=128)
parser.add_argument("--nsteps", type=int, default=200)
parser.add_argument("--chi", type=float, default=550.0, help="chi [m^2/s]")
parser.add_argument("--iso-launch", type=int, choices=[0, 1], default=0)
parser.add_argument(
    "--form", choices=["scatter", "layer", "fluxform"], default="fluxform"
)
parser.add_argument("--npts", type=int, default=3)
parser.add_argument(
    "--conduction-op",
    choices=["sde", "fd"],
    default="sde",
    help="conduction operator (fd has no launch/deposit hop scale: the a4 "
    "star-pattern class should be absent)",
)
parser.add_argument("--out", type=str, required=True)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

N = args.ncell
L = 1.0
dx = L / N
n0 = 1.0e20
Te0_eV = 5.0
B0 = 0.1
mass_factor = 1.0e12  # truly static ion background (slab-instrument idiom)
dt = 1.0e-8

qe = constants.q_e
kb = constants.kb
K_per_eV = qe / kb

# kappa parser value for a CONSTANT chi: chi = kappa / (1.5 n kB)
kappa_const = 1.5 * n0 * kb * args.chi

grid = picmi.Cartesian2DGrid(
    number_of_cells=[N, N],
    lower_bound=[-L / 2.0, -L / 2.0],
    upper_bound=[L / 2.0, L / 2.0],
    lower_boundary_conditions=["periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic"],
    lower_boundary_conditions_particles=["periodic", "periodic"],
    upper_boundary_conditions_particles=["periodic", "periodic"],
    warpx_max_grid_size=N,
)

solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=5.0 / 3.0,
    Te=Te0_eV,
    n0=n0,
    n_floor=1.0e-6 * n0,
    plasma_resistivity=0.0,
    substeps=4,
    solve_electron_energy_equation=True,
)

simulation = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=args.nsteps,
    particle_shape=1,
    verbose=0,
    warpx_serialize_initial_conditions=True,
    warpx_current_deposition_algo="direct",
)

# uniform out-of-plane B: the perp plane of the conduction tensor IS the
# simulation plane (frame: e1 = xhat, e2 = -zhat)
simulation.add_applied_field(
    picmi.AnalyticInitialField(
        Bx_expression="0",
        By_expression=f"{B0}",
        Bz_expression="0",
        warpx_do_initial_div_cleaning=False,
    )
)

ions = picmi.Species(
    name="ions",
    charge="q_e",
    mass=mass_factor * constants.m_p,
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=f"{n0}",
        momentum_expressions=["0", "0", "0"],
    ),
)
simulation.add_species(
    ions,
    layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=4),
)

simulation.initialize_inputs()

pywarpx.hybridpicmodel.qdsmc_conduction_form = args.form
pywarpx.hybridpicmodel.qdsmc_conduction_reconstruction = "ppm"
pywarpx.hybridpicmodel.qdsmc_conduction_quadrature_points = args.npts
pywarpx.hybridpicmodel.qdsmc_conduction_flux_limit_factor = 0.0
pywarpx.hybridpicmodel.qdsmc_conduction_isotropic_launch = args.iso_launch
pywarpx.hybridpicmodel.qdsmc_conduction_operator = args.conduction_op
pywarpx.hybridpicmodel.__setattr__("qdsmc_kappa_par(n,Te,t)", f"{kappa_const:.8e}")
pywarpx.hybridpicmodel.__setattr__("qdsmc_kappa_perp(n,Te,t)", f"{kappa_const:.8e}")

simulation.initialize_warpx()

Te_wrap = fields.MultiFabWrapper(mf_name="hybrid_electron_temperature_fp", level=0)


def wx_instance():
    return simulation.extension.warpx


def node_coords():
    ax = np.linspace(-L / 2.0, L / 2.0, N + 1)
    return np.meshgrid(ax, ax, indexing="ij")


state = {}


def capture0():
    if wx_instance().getistep(0) != 0 or "u0" in state:
        return
    XG, ZG = node_coords()
    s0 = 2.5 * dx
    te = Te0_eV * K_per_eV * (1.0 + np.exp(-(XG**2 + ZG**2) / (2.0 * s0**2)))
    Te_wrap[:, :] = te
    state["u0"] = float(Te_wrap[:, :].sum())


callbacks.installparticleinjection(capture0)

simulation.step(args.nsteps)

# ---- anisotropy measurement: a4 on the half-max excess contour ----------
te = Te_wrap[:, :] / K_per_eV - Te0_eV  # excess Te [eV]
XG, ZG = node_coords()
R = np.sqrt(XG**2 + ZG**2)

# radius where the azimuthal mean of the excess falls to half its center
rbins = np.arange(0.5 * dx, 0.25, dx)
prof = np.array([te[(R >= r - 0.5 * dx) & (R < r + 0.5 * dx)].mean() for r in rbins])
half = 0.5 * te[N // 2, N // 2]
ihalf = int(np.argmin(np.abs(prof - half)))
rstar = rbins[ihalf]

# bilinear sample of the excess on the r* circle; Fourier in theta
thetas = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
xs = rstar * np.cos(thetas)
zs = rstar * np.sin(thetas)
fi = (xs + L / 2.0) / dx
fj = (zs + L / 2.0) / dx
i0 = np.floor(fi).astype(int)
j0 = np.floor(fj).astype(int)
fx = fi - i0
fz = fj - j0
ring = (
    te[i0, j0] * (1 - fx) * (1 - fz)
    + te[i0 + 1, j0] * fx * (1 - fz)
    + te[i0, j0 + 1] * (1 - fx) * fz
    + te[i0 + 1, j0 + 1] * fx * fz
)
a0 = ring.mean()
a4 = 2.0 * np.abs(np.mean(ring * np.exp(-4j * thetas))) / a0

u1 = float(Te_wrap[:, :].sum())
drift = (u1 - state["u0"]) / state["u0"]

print(
    f"[blob-iso] op={args.conduction_op} form={args.form} npts={args.npts} "
    f"iso={args.iso_launch} | a4(r*={rstar:.3f}) = {a4:.4e} "
    f"| Sigma(Te) drift {drift:+.3e}",
    flush=True,
)
np.savez_compressed(
    f"{args.out}.npz",
    te_final=te,
    a4=a4,
    rstar=rstar,
    drift=drift,
    iso=args.iso_launch,
    form=args.form,
    conduction_op=args.conduction_op,
    chi=args.chi,
)
