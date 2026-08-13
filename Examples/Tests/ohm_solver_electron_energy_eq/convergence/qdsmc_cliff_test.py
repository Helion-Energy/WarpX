#!/usr/bin/env python3
"""Density-cliff conduction stress instrument (qualification-battery arm).

Aligned uniform B (z-hat) with a PERIODIC density slab: a dense region
n = n0 for z in [L/4, 3L/4] dropping by --ratio across tanh cliffs of
--width-cells (default 1 = unresolved) to n0/ratio outside. kappa is a
CONSTANT (n-independent, Spitzer-like), so chi = kappa/(1.5 n kB) JUMPS
by x ratio in the low-density region — conduction is ratio-times faster
outside the slab, and every flux crossing a cliff connects nodes with
strongly mismatched heat capacity.

Arms:
  atrest : uniform Te. ANY structure minted from the density gradient
           alone is an operator defect (the K-pump class: remapping
           K = Te n^(1-gamma) across a cliff re-materializes Te amplified
           by (n_hi/n_lo)^(gamma-1), ~7.4x at ratio 20). A direct
           T-space flux operator has F == 0 identically here.
  blob   : x-uniform hot Gaussian slab at z = L/2 (dense side) conducting
           across both cliffs. Maximum principle: no Te anywhere, at any
           step, may exceed the initial max or undershoot the background.

Both arms track per-step extrema (not just final) and conservation of
the conduction invariant Sigma(n_e Te) (energy form u = 1.5 n kB Te),
with n_e read from the solver's own deposited rho_fp.

The Te slab is x-uniform and n varies only in z, so grad Pe is curl-free
and B stays exactly static (ZK-instrument idiom).

Usage:
  python3 qdsmc_cliff_test.py --arm blob --conduction-op fd --out c.npz
"""

import argparse
import sys

import numpy as np

from pywarpx import callbacks, fields, picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=["atrest", "blob"], required=True)
parser.add_argument("--ncell", type=int, default=128)
parser.add_argument("--nsteps", type=int, default=128)
parser.add_argument("--tfinal", type=float, default=1.0e-5)
parser.add_argument(
    "--chi-dense",
    type=float,
    default=30.0,
    help="chi in the DENSE region [m^2/s]; low side runs at ratio * this",
)
parser.add_argument("--ratio", type=float, default=20.0, help="n_hi / n_lo")
parser.add_argument(
    "--width-cells",
    type=float,
    default=1.0,
    help="tanh cliff width in cells (1 = unresolved, the campaign class)",
)
parser.add_argument("--amp", type=float, default=4.0, help="blob amplitude / Te0")
parser.add_argument("--blob-sigma", type=float, default=0.05, help="sigma / L")
parser.add_argument("--npts", type=int, nargs="+", default=[3])
parser.add_argument("--flux-limit", type=float, default=0.0)
parser.add_argument("--max-hop", type=float, default=6.0)
parser.add_argument(
    "--conduction-op",
    choices=["sde", "fd"],
    default="fd",
    help="conduction operator (hybrid_pic_model.qdsmc_conduction_operator)",
)
parser.add_argument("--fd-order", type=int, choices=[2, 4], default=2)
parser.add_argument(
    "--fd-limiter", choices=["none", "upwind1", "smart"], default="smart"
)
parser.add_argument("--fd-time", choices=["ssprk2", "rkf45"], default="ssprk2")
parser.add_argument("--fd-cfl", type=float, default=0.4)
parser.add_argument(
    "--form",
    choices=["scatter", "layer", "fluxform"],
    default="scatter",
    help="SDE-arm grid-transfer form",
)
parser.add_argument("--advance", choices=["euler", "leapfrog", "pc"], default="pc")
parser.add_argument("--grad-deposit", type=int, choices=[0, 1], default=1)
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
B0 = 2.0e-5

dx = L / N
w = args.width_cells * dx
f_lo = 1.0 / args.ratio
z1, z2 = 0.25 * L, 0.75 * L
zb = 0.5 * L
s0 = args.blob_sigma * L
tfinal = args.tfinal
dt = tfinal / args.nsteps

kb = constants.kb
# constant kappa set by the DENSE-side chi; chi_lo = ratio * chi_dense
kappa_par_expr = f"{1.5 * n0 * kb * args.chi_dense:.16e}"
kappa_perp_expr = "0.0"

# periodic density slab: dense in [z1, z2], low outside
dens_expr = f"{n0}*({f_lo}+{1.0 - f_lo}*0.5*(tanh((z-{z1})/{w})-tanh((z-{z2})/{w})))"

npts = args.npts if len(args.npts) > 1 else [args.npts[0], args.npts[0]]

mu0 = 4.0e-7 * np.pi
om_wh = (np.pi * N / L) ** 2 * B0 / (mu0 * constants.q_e * f_lo * n0)
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
    n_floor=0.005 * n0,
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
        density_expression=dens_expr,
        momentum_expressions=["0.0", "0.0", "0.0"],
    ),
)

B_field = picmi.AnalyticInitialField(
    Bx_expression="0.0", By_expression="0.0", Bz_expression=f"{B0}"
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
    layout=picmi.GriddedLayout(n_macroparticle_per_cell=[4, 4], grid=grid),
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
pywarpx.hybridpicmodel.qdsmc_conduction_operator = args.conduction_op
pywarpx.hybridpicmodel.qdsmc_conduction_fd_order = args.fd_order
pywarpx.hybridpicmodel.qdsmc_conduction_fd_limiter = args.fd_limiter
pywarpx.hybridpicmodel.qdsmc_conduction_fd_cfl = args.fd_cfl
pywarpx.hybridpicmodel.qdsmc_conduction_fd_time = args.fd_time

sim.initialize_warpx()

# ----------------------------------------------------------------------
# Te poke (step 0) + per-step extremum / conservation tracking
# ----------------------------------------------------------------------
Te_wrap = fields.MultiFabWrapper(mf_name="hybrid_electron_temperature_fp", level=0)
rho_wrap = fields.MultiFabWrapper(mf_name="rho_fp", level=0)

K_per_eV = constants.q_e / constants.kb
Te0_K = Te0_eV * K_per_eV
state = {
    "initial": None,
    "sum0": None,
    "usum0": None,
    "max0": None,
    "min0": None,
    "run_max": -np.inf,
    "run_min": np.inf,
    "peak_t": [],
}


def _node_coords():
    te = Te_wrap[:, :]
    nx, nz = te.shape
    x = np.linspace(0.0, L, nx)
    z = np.linspace(0.0, L, nz)
    return np.meshgrid(x, z, indexing="ij")


def _usum():
    """conduction invariant Sigma(n_e Te) from the solver's own density"""
    te = Te_wrap[:, :][:-1, :-1]
    ne = rho_wrap[:, :][:-1, :-1] / constants.q_e
    return float((ne * te).sum())


def poke_te():
    if libwarpx_istep() != 0:
        return
    xg, zg = _node_coords()
    if args.arm == "atrest":
        te = Te0_K * np.ones_like(zg)
    else:
        prof = np.zeros_like(zg)
        for mz in range(-2, 3):
            dz_ = zg - zb + mz * L
            prof += np.exp(-(dz_**2) / (2.0 * s0**2))
        te = Te0_K * (1.0 + args.amp * prof)
    Te_wrap[:, :] = te
    state["initial"] = te.copy()
    state["sum0"] = te[:-1, :-1].sum()
    state["usum0"] = _usum()
    state["max0"] = te.max()
    state["min0"] = te.min()


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
usum1 = _usum()

overshoot = (state["run_max"] - state["max0"]) / Te0_K
undershoot = (state["run_min"] - state["min0"]) / Te0_K
sum_drift = (sum1 - state["sum0"]) / state["sum0"]
usum_drift = (usum1 - state["usum0"]) / state["usum0"]

# atrest arm: any deviation from uniformity is minted structure
mint = float(np.abs(te_final[:-1, :-1] - Te0_K).max() / Te0_K)

np.savez_compressed(
    args.out,
    arm=args.arm,
    ncell=N,
    nsteps=args.nsteps,
    dt=dt,
    tfinal=tfinal,
    L=L,
    chi_dense=args.chi_dense,
    ratio=args.ratio,
    width_cells=args.width_cells,
    amp=args.amp,
    blob_sigma=s0,
    npts=npts,
    substeps=substeps,
    conduction_op=args.conduction_op,
    fd_limiter=args.fd_limiter,
    fd_time=args.fd_time,
    fd_order=args.fd_order,
    form=args.form,
    flux_limit=args.flux_limit,
    te_initial=state["initial"],
    te_final=te_final,
    peak_trajectory=np.array(state["peak_t"]),
    overshoot=overshoot,
    undershoot=undershoot,
    mint=mint,
    te_sum0=state["sum0"],
    te_sum1=sum1,
    sum_drift=sum_drift,
    usum0=state["usum0"],
    usum1=usum1,
    usum_drift=usum_drift,
)
print(
    f"[harness] cliff done: arm={args.arm} op={args.conduction_op} "
    f"lim={args.fd_limiter} time={args.fd_time} N={N} ratio={args.ratio} "
    f"w={args.width_cells} overshoot={overshoot:+.3e} "
    f"undershoot={undershoot:+.3e} mint={mint:.3e} "
    f"sum(Te) drift={sum_drift:+.3e} sum(neTe) drift={usum_drift:+.3e}"
)
sys.stdout.flush()
