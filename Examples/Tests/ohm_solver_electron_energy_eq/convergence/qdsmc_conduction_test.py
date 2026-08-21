#!/usr/bin/env python3
"""Thrust-C (G3) harness: Ito tensor thermal conduction on the QDSMC platform.

2D periodic hybrid-PIC run engineered so conduction is the ONLY T_e dynamics:
uniform density, ions at rest with huge mass (J_i = 0), uniform B (curl B = 0
=> J_plasma = 0 => V_e = 0, and with uniform n the E-solve's grad Pe has zero
discrete curl, so B stays exactly uniform). kappa_par is a constant chosen so
chi_par = chi0; kappa_perp = 0 unless requested. The advection transport still
runs every step (markers at rest => B0 identity), so what is measured is the
conduction substep riding on the full pc Strang loop.

Modes
-----
aligned : B along z, 1D Gaussian T_e bump in z (uniform in x). Exact solution:
          amplitude A*s0/st, variance st^2 = s0^2 + 2 chi0 t (periodically
          summed images).
tilted  : B tilted by --angle deg in the x-z plane, 2D isotropic Gaussian
          blob, kappa_perp = 0. Exact: Gaussian spreads along b only. Also
          measures the spurious perpendicular spread => chi_perp_num/chi_par
          (the anisotropy-pollution gate).

Usage:
  python3 qdsmc_conduction_test.py --mode aligned --ncell 64 --nsteps 128 --out a64.npz
"""

import argparse
import sys

import numpy as np

from pywarpx import callbacks, fields, picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["aligned", "tilted"], required=True)
parser.add_argument("--ncell", type=int, default=64)
parser.add_argument("--nsteps", type=int, default=128)
parser.add_argument("--tfinal", type=float, default=1.0e-5, help="total time [s]")
parser.add_argument(
    "--chi",
    type=float,
    default=None,
    help="chi_par [m^2/s]; default sets 2*chi*tfinal = sigma0^2",
)
parser.add_argument(
    "--blob-sigma",
    type=float,
    default=0.10,
    help="initial blob sigma as a fraction of L",
)
parser.add_argument(
    "--angle",
    type=float,
    default=30.0,
    help="B tilt from z in the x-z plane [deg] (tilted mode)",
)
parser.add_argument(
    "--npts",
    type=int,
    nargs="+",
    default=[3],
    help="GH quadrature points: <both> or <par> <perp>",
)
parser.add_argument(
    "--flux-limit",
    type=float,
    default=0.0,
    help="free-streaming limiter factor f (0 = off; keep off "
    "for convergence runs, it biases chi at the ~5%% level)",
)
parser.add_argument(
    "--max-hop",
    type=float,
    default=4.0,
    help="hop cap in cells (keep the test hops well below it)",
)
parser.add_argument(
    "--kappa-perp",
    type=float,
    default=0.0,
    help="chi_perp [m^2/s] (converted to kappa internally)",
)
parser.add_argument(
    "--conduction-op",
    choices=["sde", "fd"],
    default="sde",
    help="conduction operator (hybrid_pic_model.qdsmc_conduction_operator): "
    "sde = QDSMC daughter/remap forms; fd = grid finite-difference operator "
    "(Chacon et al. CPC 313 (2025) 109646)",
)
parser.add_argument(
    "--fd-order",
    type=int,
    choices=[2, 4],
    default=2,
    help="FD operator spatial order (hybrid_pic_model.qdsmc_conduction_fd_order)",
)
parser.add_argument(
    "--fd-limiter",
    choices=["none", "upwind1", "smart"],
    default="smart",
    help="FD cross-flux limiter (hybrid_pic_model.qdsmc_conduction_fd_limiter); "
    "none = unlimited QUICK control arm",
)
parser.add_argument(
    "--fd-time",
    choices=["ssprk2", "rkf45"],
    default="ssprk2",
    help="FD subcycle integrator (hybrid_pic_model.qdsmc_conduction_fd_time): "
    "ssprk2 = monotone embedded 2(1) pair (keeps the max principle); "
    "rkf45 = Fehlberg 4(5), NOT SSP",
)
parser.add_argument(
    "--fd-cfl",
    type=float,
    default=0.4,
    help="FD subcycle CFL fraction (hybrid_pic_model.qdsmc_conduction_fd_cfl); "
    "keep <= 0.5 -- SSP-RK2 Nyquist damping goes neutral at the edge "
    "(deposit-noise persistence, the alpha<=0.25 rule from Helion's internal hybrid python prototype)",
)
parser.add_argument(
    "--iso-full",
    type=int,
    choices=[0, 1],
    default=0,
    help="isotropic conduction at the cross-field rate everywhere "
    "(hybrid_pic_model.qdsmc_conduction_isotropic)",
)
parser.add_argument(
    "--iso-b",
    type=float,
    default=-1.0,
    help="isotropic blend below this |B| [T] "
    "(hybrid_pic_model.qdsmc_conduction_iso_B); <0 = off",
)
parser.add_argument("--advance", choices=["euler", "leapfrog", "pc"], default="pc")
parser.add_argument("--grad-deposit", type=int, choices=[0, 1], default=1)
parser.add_argument(
    "--form",
    choices=["scatter", "layer", "fluxform"],
    default="scatter",
    help="conduction grid-transfer form (layer = Milstein gather)",
)
parser.add_argument(
    "--interp",
    choices=["linear", "monocubic", "keys"],
    default="monocubic",
    help="layer-form interpolant",
)
parser.add_argument(
    "--curved-feet",
    type=int,
    choices=[0, 1],
    default=0,
    help="layer form: midpoint-rotated feet (measured harmful: the div-D "
    "drift already carries the mean curvature; rotation double-counts)",
)
parser.add_argument(
    "--deposit-kernel",
    choices=["hat", "keys"],
    default="hat",
    help="scatter-form deposit kernel (keys = conservative cubic, "
    "no B1 correction, not monotone)",
)
parser.add_argument(
    "--compensate",
    type=int,
    choices=[0, 1],
    default=0,
    help="scatter hat + bookkept Boris-Book FCT antidiffusion pass "
    "(requires --grad-deposit 0)",
)
parser.add_argument(
    "--fct-limiter",
    choices=["bb", "zalesak", "none"],
    default="bb",
    help="FCT limiter for --compensate: bb (Boris-Book), zalesak "
    "(bounds include the pre-deposit field), none (unlimited control)",
)
parser.add_argument(
    "--recon",
    choices=["plm", "ppm"],
    default="plm",
    help="fluxform sweep reconstruction (ppm = Colella-Woodward parabolic)",
)
parser.add_argument(
    "--ff-unsplit",
    type=int,
    choices=[0, 1],
    default=0,
    help="fluxform: unsplit per-donor transport (C.7d fork (b), corner images)",
)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--verbose", type=int, default=0)
args = parser.parse_args()

# ----------------------------------------------------------------------
# Problem definition
# ----------------------------------------------------------------------
L = 1.0
N = args.ncell
n0 = 1.0e18
Te0_eV = 10.0
gamma = 5.0 / 3.0
mass_factor = 1.0e6
# Uniform |B| [T]: only the DIRECTION matters for the conduction tensor, but
# the magnitude must keep the explicit B-substep advance whistler-stable
# against machine-eps curl(E) seeds: omega_wh(k_max)*dt_sub ~
# (pi*N/L)^2 * B/(mu0 q n0) * dt/substeps ~ 0.08 at B0 = 2e-5 for the
# default parameters (and invariant under the parabolic dt ~ dx^2 sweep).
B0 = 2.0e-5

s0 = args.blob_sigma * L
blob_amp = 1.0
tfinal = args.tfinal
dt = tfinal / args.nsteps
chi0 = args.chi if args.chi is not None else s0**2 / (2.0 * tfinal)

kb = constants.kb
# chi = kappa / (3/2 n kB)  =>  constant-kappa parser value for constant chi
kappa_par_expr = f"{1.5 * n0 * kb * chi0:.16e}"
kappa_perp_expr = f"{1.5 * n0 * kb * args.kappa_perp:.16e}"

theta = np.radians(args.angle) if args.mode == "tilted" else 0.0
bx, bz = np.sin(theta), np.cos(theta)
xb, zb = 0.5 * L, 0.5 * L

npts = args.npts if len(args.npts) > 1 else [args.npts[0], args.npts[0]]

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
        momentum_expressions=["0.0", "0.0", "0.0"],
    ),
)

B_field = picmi.AnalyticInitialField(
    Bx_expression=f"{B0 * bx}",
    By_expression="0.0",
    Bz_expression=f"{B0 * bz}",
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

# Conduction knobs are not picmi kwargs: write the bucket after
# initialize_inputs (verify in warpx_used_inputs).
import pywarpx  # noqa: E402

pywarpx.hybridpicmodel.qdsmc_time_advance = args.advance
pywarpx.hybridpicmodel.qdsmc_gradient_deposit = args.grad_deposit
pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_par(n,Te,t)", kappa_par_expr)
pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_perp(n,Te,t)", kappa_perp_expr)
pywarpx.hybridpicmodel.qdsmc_conduction_quadrature_points = npts
pywarpx.hybridpicmodel.qdsmc_conduction_flux_limit_factor = args.flux_limit
pywarpx.hybridpicmodel.qdsmc_conduction_max_hop = args.max_hop
pywarpx.hybridpicmodel.qdsmc_conduction_form = args.form
pywarpx.hybridpicmodel.qdsmc_conduction_fluxform_unsplit = args.ff_unsplit
pywarpx.hybridpicmodel.qdsmc_conduction_reconstruction = args.recon
pywarpx.hybridpicmodel.qdsmc_conduction_interp = args.interp
pywarpx.hybridpicmodel.qdsmc_conduction_curved_feet = args.curved_feet
pywarpx.hybridpicmodel.qdsmc_conduction_deposit_kernel = args.deposit_kernel
pywarpx.hybridpicmodel.qdsmc_conduction_compensate = args.compensate
pywarpx.hybridpicmodel.qdsmc_conduction_fct_limiter = args.fct_limiter
pywarpx.hybridpicmodel.qdsmc_conduction_operator = args.conduction_op
pywarpx.hybridpicmodel.qdsmc_conduction_fd_order = args.fd_order
pywarpx.hybridpicmodel.qdsmc_conduction_fd_limiter = args.fd_limiter
pywarpx.hybridpicmodel.qdsmc_conduction_fd_cfl = args.fd_cfl
pywarpx.hybridpicmodel.qdsmc_conduction_fd_time = args.fd_time
if args.iso_full:
    pywarpx.hybridpicmodel.qdsmc_conduction_isotropic = 1
if args.iso_b > 0.0:
    pywarpx.hybridpicmodel.qdsmc_conduction_iso_B = args.iso_b

sim.initialize_warpx()

# ----------------------------------------------------------------------
# Te poke (step 0, after the closure mirror) + initial capture
# ----------------------------------------------------------------------
Te_wrap = fields.MultiFabWrapper(mf_name="hybrid_electron_temperature_fp", level=0)

K_per_eV = constants.q_e / constants.kb
state = {"initial": None, "sum0": None}


def _node_coords():
    te = Te_wrap[:, :]
    nx, nz = te.shape
    x = np.linspace(0.0, L, nx)
    z = np.linspace(0.0, L, nz)
    return np.meshgrid(x, z, indexing="ij")


def profile(xg, zg, sig_par, sig_perp):
    """Periodically image-summed blob, fractional amplitude.

    aligned: 1D Gaussian in z (sig_perp unused). tilted: 2D Gaussian with
    sigma sig_par along b = (sin t, cos t) and sig_perp along the in-plane
    perpendicular.
    """
    out = np.zeros_like(xg)
    amp = blob_amp * s0 / sig_par
    if args.mode == "aligned":
        for mz in range(-3, 4):
            dz = zg - zb + mz * L
            out += amp * np.exp(-(dz**2) / (2.0 * sig_par**2))
    else:
        for mx in range(-2, 3):
            for mz in range(-2, 3):
                dx = xg - xb + mx * L
                dz = zg - zb + mz * L
                xi = dx * bx + dz * bz
                eta = dx * bz - dz * bx
                out += amp * np.exp(
                    -(xi**2) / (2.0 * sig_par**2) - (eta**2) / (2.0 * sig_perp**2)
                )
    return out


def poke_te():
    if libwarpx_istep() != 0:
        return
    xg, zg = _node_coords()
    te = Te0_eV * K_per_eV * (1.0 + profile(xg, zg, s0, s0))
    Te_wrap[:, :] = te
    state["initial"] = te.copy()
    state["sum0"] = te[:-1, :-1].sum()
    if args.verbose:
        print(f"[harness] poked Te profile at step 0: max {te.max():.6e} K")


def libwarpx_istep():
    from pywarpx import libwarpx

    return libwarpx.libwarpx_so.get_instance().getistep(0)


callbacks.installparticleinjection(poke_te)

# ----------------------------------------------------------------------
# Run, compare to exact, save
# ----------------------------------------------------------------------
sim.step(args.nsteps)

te_final = Te_wrap[:, :]
xg, zg = _node_coords()
st = np.sqrt(s0**2 + 2.0 * chi0 * tfinal)
te_exact = Te0_eV * K_per_eV * (1.0 + profile(xg, zg, st, s0))

# unique nodes only (drop the periodic seam duplicates)
diff = (te_final - te_exact)[:-1, :-1]
rel_l2 = np.sqrt((diff**2).mean()) / (Te0_eV * K_per_eV)

# moment-based spread of the perturbation (unique nodes)
w = (te_final[:-1, :-1] - Te0_eV * K_per_eV).clip(min=0.0)
xu, zu = xg[:-1, :-1], zg[:-1, :-1]
xi = (xu - xb) * bx + (zu - zb) * bz
eta = (xu - xb) * bz - (zu - zb) * bx
wsum = w.sum()
sig_par_sq = (w * xi**2).sum() / wsum - ((w * xi).sum() / wsum) ** 2
sig_perp_sq = (w * eta**2).sum() / wsum - ((w * eta).sum() / wsum) ** 2
chi_par_meas = (sig_par_sq - s0**2) / (2.0 * tfinal)
chi_perp_meas = (sig_perp_sq - s0**2) / (2.0 * tfinal)

np.savez_compressed(
    args.out,
    mode=args.mode,
    advance=args.advance,
    form=args.form,
    interp=args.interp,
    curved_feet=args.curved_feet,
    deposit_kernel=args.deposit_kernel,
    ncell=N,
    nsteps=args.nsteps,
    dt=dt,
    tfinal=tfinal,
    L=L,
    chi0=chi0,
    chi_perp_in=args.kappa_perp,
    npts=npts,
    angle_deg=args.angle,
    Te0_eV=Te0_eV,
    blob=[xb, zb, s0, blob_amp],
    te_initial=state["initial"],
    te_final=te_final,
    te_exact=te_exact,
    rel_l2=rel_l2,
    sig_par_sq=sig_par_sq,
    sig_perp_sq=sig_perp_sq,
    chi_par_meas=chi_par_meas,
    chi_perp_meas=chi_perp_meas,
    te_sum0=state["sum0"],
    te_sum1=te_final[:-1, :-1].sum(),
)
print(
    f"[harness] done: mode={args.mode} N={N} steps={args.nsteps} npts={npts} "
    f"relL2={rel_l2:.4e} chi_par_meas/chi0={chi_par_meas / chi0:.4f} "
    f"chi_perp_num/chi0={chi_perp_meas / chi0:.3e} "
    f"sum(Te) drift={(te_final[:-1, :-1].sum() - state['sum0']) / state['sum0']:.3e}"
)
sys.stdout.flush()
