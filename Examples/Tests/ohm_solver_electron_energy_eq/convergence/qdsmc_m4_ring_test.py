#!/usr/bin/env python3
"""m=4 grid-anisotropy instrument for the QDSMC conduction executor (rung 1).

2D (x,z) periodic hybrid-PIC run with a UNIFORM out-of-plane field
B = B0 yhat, uniform density, and an axisymmetric Gaussian Te ring at
radius r0 around the box center. With b = yhat the field-aligned frame
falls back to e1 = xhat, e2 = -zhat (the same axis-locked frame the
production Bz slab has in its (x,y) perp plane), kappa_par transport is
out-of-plane (an in-place identity in 2D), and kappa_perp conducts
ISOTROPICALLY in-plane: the exact continuum solution stays axisymmetric
for all time. Any angular structure that develops is executor-generated.

Measured per step (callback): angular multipole moments of w = Te - Te0,
    a_m = sum_nodes w * exp(-i m phi),   m in {0, 2, 4, 8},
both globally (band-limited to the ring band) and per radial shell
(dr = 1 cell). The headline metric is |a4|/a0 growth per step; arg(a4)
distinguishes axis-aligned (cos 4phi, phase ~ 0) from diagonal lobes
(phase ~ pi). The liftoff observation this instrument decomposes:
grid-aligned four-fold Te structure, strongest where conduction acts
hardest (kappa x4 arm), suspected axis-factorized transport executor
(per-axis limiter clipping / split-remap donor identity loss / per-axis
hop clamp / per-face flux cap) rather than missing diagonal quadrature.

Arms (see run_m4_ring.py): split-ppm (production), split-plm, layer,
unsplit-plm (periodic-only), plus controls isolating one component each:
slope_limiter=none, wide-open hop clamp, isotropic_launch.

Usage:
  python3 qdsmc_m4_ring_test.py --ncell 128 --nsteps 128 --out m4.npz
"""

import argparse
import sys

import numpy as np

from pywarpx import callbacks, fields, picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument("--ncell", type=int, default=128)
parser.add_argument("--nsteps", type=int, default=128)
parser.add_argument(
    "--tfinal",
    type=float,
    default=1.25e-6,
    help="total time [s]; default gives diffusion length ~0.05 L at chi=1e3",
)
parser.add_argument("--chi", type=float, default=1.0e3, help="chi_perp [m^2/s]")
parser.add_argument(
    "--ring-r0", type=float, default=0.25, help="ring radius / L (box-centered)"
)
parser.add_argument(
    "--ring-sigma", type=float, default=0.04, help="ring Gaussian sigma / L"
)
parser.add_argument(
    "--npts",
    type=int,
    nargs="+",
    default=[3],
    help="GH quadrature points: <both> or <par> <perp>",
)
parser.add_argument(
    "--max-hop",
    type=float,
    default=2.0,
    help="hop cap in cells (production-like default; wide-open control: 1e9)",
)
parser.add_argument("--flux-limit", type=float, default=0.0)
parser.add_argument("--slope-limiter", choices=["mc", "none"], default="mc")
parser.add_argument("--advance", choices=["euler", "leapfrog", "pc"], default="pc")
parser.add_argument("--grad-deposit", type=int, choices=[0, 1], default=1)
parser.add_argument(
    "--form", choices=["scatter", "layer", "fluxform"], default="fluxform"
)
parser.add_argument(
    "--interp", choices=["linear", "monocubic", "keys"], default="monocubic"
)
parser.add_argument("--recon", choices=["plm", "ppm"], default="ppm")
parser.add_argument("--ff-unsplit", type=int, choices=[0, 1], default=0)
parser.add_argument(
    "--iso-launch",
    type=int,
    choices=[0, 1],
    default=0,
    help="qdsmc_conduction_isotropic_launch (45-deg rotated quadrature pair)",
)
parser.add_argument(
    "--probe-every", type=int, default=1, help="multipole probe cadence [steps]"
)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--verbose", type=int, default=0)
args = parser.parse_args()

# ----------------------------------------------------------------------
# Problem definition (wiggle-test skeleton: frozen heavy ions, uniform n)
# ----------------------------------------------------------------------
L = 1.0
N = args.ncell
n0 = 1.0e21  # uniform B + uniform n -> J = 0, no field evolution
Te0_eV = 10.0
gamma = 5.0 / 3.0
mass_factor = 1.0e6
B0 = 2.0e-5

r0 = args.ring_r0 * L
sr = args.ring_sigma * L
xc = zc = 0.5 * L
ring_amp = 1.0
tfinal = args.tfinal
dt = tfinal / args.nsteps
chi0 = args.chi

kb = constants.kb
# kappa_par transports along yhat = out-of-plane: an in-place identity in
# 2D (feet displace in y only), so its value is inert; keep it equal to
# kappa_perp so the tensor-product quadrature lattice matches production.
kappa_par_expr = f"{1.5 * n0 * kb * chi0:.16e}"
kappa_perp_expr = f"{1.5 * n0 * kb * chi0:.16e}"

npts = args.npts if len(args.npts) > 1 else [args.npts[0], args.npts[0]]

# Whistler-stable substep count for this dt (uniform B carries no J, but
# keep the standard rule so the field push is regime-safe).
mu0 = 4.0e-7 * np.pi
om_wh = (np.pi * N / L) ** 2 * B0 / (mu0 * constants.q_e * n0)
substeps = max(4, int(np.ceil(om_wh * dt / 0.1)))

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

B_field = picmi.AnalyticInitialField(
    Bx_expression="0.0",
    By_expression=f"{B0}",
    Bz_expression="0.0",
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
pywarpx.hybridpicmodel.qdsmc_gradient_deposit = args.grad_deposit
pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_par(n,Te,t)", kappa_par_expr)
pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_perp(n,Te,t)", kappa_perp_expr)
pywarpx.hybridpicmodel.qdsmc_conduction_quadrature_points = npts
pywarpx.hybridpicmodel.qdsmc_conduction_flux_limit_factor = args.flux_limit
pywarpx.hybridpicmodel.qdsmc_conduction_max_hop = args.max_hop
pywarpx.hybridpicmodel.qdsmc_conduction_form = args.form
pywarpx.hybridpicmodel.qdsmc_conduction_fluxform_unsplit = args.ff_unsplit
pywarpx.hybridpicmodel.qdsmc_conduction_reconstruction = args.recon
pywarpx.hybridpicmodel.qdsmc_conduction_slope_limiter = args.slope_limiter
pywarpx.hybridpicmodel.qdsmc_conduction_interp = args.interp
pywarpx.hybridpicmodel.qdsmc_conduction_isotropic_launch = args.iso_launch

sim.initialize_warpx()

# ----------------------------------------------------------------------
# Te ring poke (step 0) + per-step multipole probe
# ----------------------------------------------------------------------
Te_wrap = fields.MultiFabWrapper(mf_name="hybrid_electron_temperature_fp", level=0)

K_per_eV = constants.q_e / constants.kb
Te0_K = Te0_eV * K_per_eV
state = {"initial": None}
history = {"step": [], "a0": []}
for m in (2, 4, 8):
    history[f"a{m}_re"] = []
    history[f"a{m}_im"] = []
history["shell_a4"] = []  # per-shell |a4|/a0 profile snapshots
history["shell_step"] = []

MMODES = (2, 4, 8)


def _node_grids():
    te = Te_wrap[:, :]
    nx, nz = te.shape
    x = np.linspace(0.0, L, nx)
    z = np.linspace(0.0, L, nz)
    xg, zg = np.meshgrid(x, z, indexing="ij")
    return xg, zg


def libwarpx_istep():
    from pywarpx import libwarpx

    return libwarpx.libwarpx_so.get_instance().getistep(0)


def poke_te():
    if libwarpx_istep() != 0:
        return
    xg, zg = _node_grids()
    r = np.sqrt((xg - xc) ** 2 + (zg - zc) ** 2)
    te = Te0_K * (1.0 + ring_amp * np.exp(-((r - r0) ** 2) / (2.0 * sr**2)))
    Te_wrap[:, :] = te
    state["initial"] = te.copy()
    if args.verbose:
        print(f"[m4] poked ring at step 0: max {te.max():.6e} K", flush=True)


# Precomputed on first probe call (unique nodes only: drop the duplicated
# periodic last row/column).
geo = {}


def _geo():
    if geo:
        return geo
    xg, zg = _node_grids()
    xu, zu = xg[:-1, :-1], zg[:-1, :-1]
    r = np.sqrt((xu - xc) ** 2 + (zu - zc) ** 2)
    phi = np.arctan2(zu - zc, xu - xc)
    band = (r > max(r0 - 4.0 * sr, 1.0e-9)) & (r < r0 + 4.0 * sr)
    dx = L / N
    shell_edges = np.arange(max(r0 - 4.0 * sr, 0.0), r0 + 4.0 * sr + dx, dx)
    shell_idx = np.digitize(r, shell_edges) - 1
    nshell = len(shell_edges) - 1
    geo.update(
        r=r,
        phi=phi,
        band=band,
        shell_idx=shell_idx,
        nshell=nshell,
        shell_edges=shell_edges,
    )
    return geo


def probe():
    step = libwarpx_istep()
    if step % args.probe_every != 0 and step != args.nsteps:
        return
    g = _geo()
    te = Te_wrap[:, :][:-1, :-1]
    w = te - Te0_K
    wb = np.where(g["band"], w, 0.0)
    a0 = wb.sum()
    history["step"].append(step)
    history["a0"].append(a0)
    for m in MMODES:
        am = (wb * np.exp(-1j * m * g["phi"])).sum()
        history[f"a{m}_re"].append(am.real)
        history[f"a{m}_im"].append(am.imag)
    # per-shell |a4|/a0 profile (cheap; store every probe for the movie)
    sidx, nshell = g["shell_idx"], g["nshell"]
    valid = g["band"] & (sidx >= 0) & (sidx < nshell)
    s0 = np.bincount(sidx[valid], weights=w[valid], minlength=nshell)
    s4r = np.bincount(
        sidx[valid], weights=(w * np.cos(4 * g["phi"]))[valid], minlength=nshell
    )
    s4i = np.bincount(
        sidx[valid], weights=(w * np.sin(4 * g["phi"]))[valid], minlength=nshell
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        prof = np.where(np.abs(s0) > 0, np.sqrt(s4r**2 + s4i**2) / np.abs(s0), 0.0)
    history["shell_a4"].append(prof)
    history["shell_step"].append(step)


callbacks.installparticleinjection(poke_te)
callbacks.installafterstep(probe)

# ----------------------------------------------------------------------
# Run and save
# ----------------------------------------------------------------------
sim.step(args.nsteps)

te_final = Te_wrap[:, :]

h = {k: np.asarray(v) for k, v in history.items()}
a0 = h["a0"]
a4 = np.sqrt(h["a4_re"] ** 2 + h["a4_im"] ** 2)
m4_over_m0 = np.where(np.abs(a0) > 0, a4 / np.abs(a0), 0.0)
# growth per step: linear fit over the second half (skip the deposit-noise
# floor settling of the first remaps)
half = len(m4_over_m0) // 2
if len(m4_over_m0) - half >= 2:
    growth = np.polyfit(h["step"][half:], m4_over_m0[half:], 1)[0]
else:
    growth = 0.0

print(
    f"[m4] form={args.form} recon={args.recon} unsplit={args.ff_unsplit} "
    f"limiter={args.slope_limiter} hop={args.max_hop:g} iso={args.iso_launch} "
    f"| final m4/m0 = {m4_over_m0[-1]:.4e}, growth = {growth:.4e} /step "
    f"| m2/m0 = {np.sqrt(h['a2_re'][-1] ** 2 + h['a2_im'][-1] ** 2) / abs(a0[-1]):.4e} "
    f"m8/m0 = {np.sqrt(h['a8_re'][-1] ** 2 + h['a8_im'][-1] ** 2) / abs(a0[-1]):.4e} "
    f"| arg(a4) = {np.arctan2(h['a4_im'][-1], h['a4_re'][-1]):+.3f} rad",
    flush=True,
)

np.savez_compressed(
    args.out,
    ncell=N,
    nsteps=args.nsteps,
    dt=dt,
    tfinal=tfinal,
    L=L,
    chi0=chi0,
    ring_r0=r0,
    ring_sigma=sr,
    npts=npts,
    substeps=substeps,
    max_hop=args.max_hop,
    flux_limit=args.flux_limit,
    slope_limiter=args.slope_limiter,
    advance=args.advance,
    form=args.form,
    interp=args.interp,
    recon=args.recon,
    ff_unsplit=args.ff_unsplit,
    iso_launch=args.iso_launch,
    m4_over_m0=m4_over_m0,
    m4_growth_per_step=growth,
    te_initial=state["initial"],
    te_final=te_final,
    shell_edges=_geo()["shell_edges"],
    **h,
)
sys.exit(0)
