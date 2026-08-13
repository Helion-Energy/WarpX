#!/usr/bin/env python3
"""GFD6 maximum-principle stress instrument: hot blob at a magnetic null.

2D periodic hybrid-PIC run on the verified conduction-only skeleton
(uniform density, static heavy ions, resistivity 0) with an in-plane
cat's-eye field from the flux function

    A_y / B0 = (1/k) cos(k x) cos(k z),      k = 2 pi / L
    Bx =  B0 cos(k x) sin(k z),   Bz = -B0 sin(k x) cos(k z)

which carries X-point nulls at (L/4, L/4)-class points and O-point nulls
at (0, 0)/(L/2, L/2)-class points. A hot Gaussian blob is seeded ON a null
(--blob-at x|o) and conducts with a strongly anisotropic tensor
(kappa_perp = eps_perp * kappa_par, eps_perp down to 1e-7). Optionally a
seeded, divergence-free, periodic mode-sum perturbation (--b-noise, RMS
fraction of B0) corrupts b-hat — near the null the direction becomes pure
noise, which is exactly the stress the operator must survive.

What is measured (per step, not just at the end):
  overshoot  = max_t max_x Te - max_x Te(0)   [new-maximum minting]
  undershoot = min_t min_x Te - min_x Te(0)   [new-minimum minting]
  sum drift  = Sigma Te(t_final) / Sigma Te(0) - 1
A monotone (maximum-principle) operator keeps overshoot <= 0 and
undershoot >= 0 to roundoff for ANY tensor orientation, noise, or
anisotropy. The unlimited control arm (--fd-limiter none) is expected to
violate at sharp fronts = mechanism attribution.

Density is uniform: the cat's-eye field carries J_y ~ B0 k / mu0, so the
wiggle-instrument idiom applies (n0 = 1e21 keeps the Hall evolution of B
at the 1e-3 B0 level over the run; conduction reads only the direction).

Usage:
  python3 qdsmc_xpoint_test.py --ncell 128 --nsteps 128 --conduction-op fd --out x.npz
"""

import argparse
import sys

import numpy as np

from pywarpx import callbacks, fields, picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument("--ncell", type=int, default=128)
parser.add_argument("--nsteps", type=int, default=128)
parser.add_argument("--tfinal", type=float, default=1.0e-5, help="total time [s]")
parser.add_argument("--chi", type=float, default=250.0, help="chi_par at Te0 [m^2/s]")
parser.add_argument(
    "--eps-perp",
    type=float,
    default=1.0e-4,
    help="kappa_perp / kappa_par anisotropy ratio",
)
parser.add_argument(
    "--kappa-form",
    choices=["const", "spitzer"],
    default="const",
    help="const: kappa(Te) fixed; spitzer: kappa ~ (Te/Te0)^2.5",
)
parser.add_argument(
    "--b-noise",
    type=float,
    default=0.0,
    help="RMS divergence-free b-hat noise as a fraction of B0 (0 = clean)",
)
parser.add_argument("--seed", type=int, default=7, help="noise mode RNG seed")
parser.add_argument(
    "--blob-at",
    choices=["x", "o"],
    default="x",
    help="blob centered on the X-point (L/4,L/4) or the O-point (L/2,L/2)",
)
parser.add_argument("--blob-sigma", type=float, default=0.06, help="sigma / L")
parser.add_argument("--amp", type=float, default=4.0, help="blob amplitude / Te0")
parser.add_argument(
    "--npts", type=int, nargs="+", default=[3], help="SDE GH quadrature points"
)
parser.add_argument("--flux-limit", type=float, default=0.0)
parser.add_argument("--max-hop", type=float, default=6.0)
parser.add_argument(
    "--conduction-op",
    choices=["sde", "fd"],
    default="fd",
    help="conduction operator (hybrid_pic_model.qdsmc_conduction_operator)",
)
parser.add_argument(
    "--fd-order", type=int, choices=[2, 4], default=2, help="FD spatial order"
)
parser.add_argument(
    "--fd-limiter",
    choices=["none", "upwind1", "smart"],
    default="smart",
    help="FD cross-flux limiter (none = unlimited QUICK control arm)",
)
parser.add_argument(
    "--fd-time",
    choices=["ssprk2", "rkf45"],
    default="ssprk2",
    help="FD subcycle integrator",
)
parser.add_argument(
    "--fd-cfl", type=float, default=0.4, help="FD subcycle CFL fraction"
)
parser.add_argument(
    "--iso-b",
    type=float,
    default=-1.0,
    help="isotropic blend below this |B| [T] (<0 = off)",
)
parser.add_argument(
    "--form",
    choices=["scatter", "layer", "fluxform"],
    default="scatter",
    help="SDE-arm grid-transfer form",
)
parser.add_argument("--advance", choices=["euler", "leapfrog", "pc"], default="pc")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--verbose", type=int, default=0)
args = parser.parse_args()

# ----------------------------------------------------------------------
# Problem definition
# ----------------------------------------------------------------------
L = 1.0
N = args.ncell
n0 = 1.0e21  # wiggle idiom: cat's-eye J_y Hall evolution frozen at high n
Te0_eV = 10.0
gamma = 5.0 / 3.0
mass_factor = 1.0e6
B0 = 2.0e-5

kmode = 2.0 * np.pi / L
s0 = args.blob_sigma * L
if args.blob_at == "x":
    xb, zb = 0.25 * L, 0.25 * L
else:
    xb, zb = 0.5 * L, 0.5 * L
tfinal = args.tfinal
dt = tfinal / args.nsteps
chi0 = args.chi

kb = constants.kb
kappa0 = 1.5 * n0 * kb * chi0
if args.kappa_form == "spitzer":
    kappa_par_expr = f"{kappa0:.16e}*((Te/{Te0_eV:.6e})^2.5)"
    kappa_perp_expr = f"{args.eps_perp * kappa0:.16e}*((Te/{Te0_eV:.6e})^2.5)"
else:
    kappa_par_expr = f"{kappa0:.16e}"
    kappa_perp_expr = f"{args.eps_perp * kappa0:.16e}"

npts = args.npts if len(args.npts) > 1 else [args.npts[0], args.npts[0]]

# whistler-stable B substep count (wiggle trap 3)
mu0 = 4.0e-7 * np.pi
om_wh = (np.pi * N / L) ** 2 * B0 / (mu0 * constants.q_e * n0)
substeps = max(4, int(np.ceil(om_wh * dt / 0.1)))

# ----------------------------------------------------------------------
# Field expressions: cat's-eye + optional div-free periodic mode noise
# ----------------------------------------------------------------------
bx_terms = [f"{B0}*cos({kmode}*x)*sin({kmode}*z)"]
bz_terms = [f"(-{B0})*sin({kmode}*x)*cos({kmode}*z)"]
if args.b_noise > 0.0:
    rng = np.random.default_rng(args.seed)
    nmodes = 16
    picked = set()
    while len(picked) < nmodes:
        mx, mz = int(rng.integers(-6, 7)), int(rng.integers(-6, 7))
        if (mx, mz) != (0, 0) and (mx, mz) not in picked:
            picked.add((mx, mz))
    cm = args.b_noise * B0 * np.sqrt(2.0 / nmodes)
    for mx, mz in sorted(picked):
        kx, kz = kmode * mx, kmode * mz
        kn = np.hypot(kx, kz)
        ph = float(rng.uniform(0.0, 2.0 * np.pi))
        theta = f"({kx:.16e}*x+{kz:.16e}*z+{ph:.16e})"
        # psi_m = (cm/kn) cos(theta):  dBx = -dpsi/dz, dBz = +dpsi/dx
        bx_terms.append(f"({cm * kz / kn:.16e})*sin{theta}")
        bz_terms.append(f"({-cm * kx / kn:.16e})*sin{theta}")

B_field = picmi.AnalyticInitialField(
    Bx_expression="+".join(bx_terms),
    By_expression="0.0",
    Bz_expression="+".join(bz_terms),
    warpx_do_initial_div_cleaning=False,
)

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

pywarpx.hybridpicmodel.qdsmc_time_advance = args.advance
pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_par(n,Te,t)", kappa_par_expr)
pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_perp(n,Te,t)", kappa_perp_expr)
pywarpx.hybridpicmodel.qdsmc_conduction_quadrature_points = npts
pywarpx.hybridpicmodel.qdsmc_conduction_flux_limit_factor = args.flux_limit
pywarpx.hybridpicmodel.qdsmc_conduction_max_hop = args.max_hop
pywarpx.hybridpicmodel.qdsmc_conduction_form = args.form
pywarpx.hybridpicmodel.qdsmc_conduction_operator = args.conduction_op
pywarpx.hybridpicmodel.qdsmc_conduction_fd_order = args.fd_order
pywarpx.hybridpicmodel.qdsmc_conduction_fd_limiter = args.fd_limiter
pywarpx.hybridpicmodel.qdsmc_conduction_fd_cfl = args.fd_cfl
pywarpx.hybridpicmodel.qdsmc_conduction_fd_time = args.fd_time
if args.iso_b > 0.0:
    pywarpx.hybridpicmodel.qdsmc_conduction_iso_B = args.iso_b

sim.initialize_warpx()

# ----------------------------------------------------------------------
# Te poke (step 0) + per-step extremum tracking
# ----------------------------------------------------------------------
Te_wrap = fields.MultiFabWrapper(mf_name="hybrid_electron_temperature_fp", level=0)

K_per_eV = constants.q_e / constants.kb
Te0_K = Te0_eV * K_per_eV
state = {
    "initial": None,
    "sum0": None,
    "max0": None,
    "min0": None,
    "peak_t": [],
    "run_max": -np.inf,
    "run_min": np.inf,
}


def _node_coords():
    te = Te_wrap[:, :]
    nx, nz = te.shape
    x = np.linspace(0.0, L, nx)
    z = np.linspace(0.0, L, nz)
    return np.meshgrid(x, z, indexing="ij")


def poke_te():
    if libwarpx_istep() != 0:
        return
    xg, zg = _node_coords()
    prof = np.zeros_like(xg)
    for mx in range(-1, 2):
        for mz in range(-1, 2):
            dx_ = xg - xb + mx * L
            dz_ = zg - zb + mz * L
            prof += np.exp(-(dx_**2 + dz_**2) / (2.0 * s0**2))
    te = Te0_K * (1.0 + args.amp * prof)
    Te_wrap[:, :] = te
    state["initial"] = te.copy()
    state["sum0"] = te[:-1, :-1].sum()
    state["max0"] = te.max()
    state["min0"] = te.min()
    if args.verbose:
        print(f"[harness] poked blob at ({xb}, {zb}): max {te.max():.6e} K")


def track_extrema():
    te = Te_wrap[:, :]
    state["run_max"] = max(state["run_max"], float(te.max()))
    state["run_min"] = min(state["run_min"], float(te.min()))
    state["peak_t"].append(float(te.max()))


def libwarpx_istep():
    from pywarpx import libwarpx

    return libwarpx.libwarpx_so.get_instance().getistep(0)


callbacks.installparticleinjection(poke_te)
callbacks.installafterstep(track_extrema)

# ----------------------------------------------------------------------
# Run and score
# ----------------------------------------------------------------------
sim.step(args.nsteps)

te_final = Te_wrap[:, :]
sum1 = te_final[:-1, :-1].sum()

overshoot = (state["run_max"] - state["max0"]) / Te0_K
undershoot = (state["run_min"] - state["min0"]) / Te0_K
sum_drift = (sum1 - state["sum0"]) / state["sum0"]

# monotone peak decay: count of step-to-step peak increases beyond roundoff
peak = np.array(state["peak_t"])
peak_rise_steps = int(np.sum(np.diff(peak) > 1.0e-9 * Te0_K))

np.savez_compressed(
    args.out,
    ncell=N,
    nsteps=args.nsteps,
    dt=dt,
    tfinal=tfinal,
    L=L,
    chi0=chi0,
    eps_perp=args.eps_perp,
    kappa_form=args.kappa_form,
    b_noise=args.b_noise,
    seed=args.seed,
    blob_at=args.blob_at,
    amp=args.amp,
    blob_sigma=s0,
    npts=npts,
    substeps=substeps,
    conduction_op=args.conduction_op,
    fd_limiter=args.fd_limiter,
    fd_time=args.fd_time,
    fd_order=args.fd_order,
    iso_b=args.iso_b,
    flux_limit=args.flux_limit,
    te_initial=state["initial"],
    te_final=te_final,
    peak_trajectory=peak,
    overshoot=overshoot,
    undershoot=undershoot,
    peak_rise_steps=peak_rise_steps,
    te_sum0=state["sum0"],
    te_sum1=sum1,
    sum_drift=sum_drift,
)
print(
    f"[harness] xpoint done: op={args.conduction_op} lim={args.fd_limiter} "
    f"time={args.fd_time} N={N} eps_perp={args.eps_perp:.1e} "
    f"noise={args.b_noise} at={args.blob_at} kappa={args.kappa_form} "
    f"overshoot={overshoot:+.3e} undershoot={undershoot:+.3e} "
    f"peak_rises={peak_rise_steps} sum drift={sum_drift:+.3e}"
)
sys.stdout.flush()
