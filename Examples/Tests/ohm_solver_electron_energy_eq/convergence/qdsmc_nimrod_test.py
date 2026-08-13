#!/usr/bin/env python3
"""GFD2b anisotropy-pollution instrument (NIMROD-class, transient form).

The steady NIMROD benchmark needs an implicit solve at tokamak
anisotropy; this is the transient, periodic-box equivalent with the
same analytic content. Cat's-eye field (tangent to the isolines of
A ~ cos(k x) cos(k z), k = 2 pi / L) with the temperature initialized
ON the flux function:

    Te(x, z, 0) = Te0 (1 + a cos(k x) cos(k z))

grad(Te) is parallel to grad(A), i.e. EXACTLY perpendicular to b
everywhere, so the parallel channel transports nothing analytically and
the mode decays by the perpendicular channel alone:

    d(amp)/dt = -2 k^2 chi_perp amp        (laplacian(coscos) = -2k^2 coscos)

Any additional decay is numerical cross-field pollution. The measured

    chi_perp_eff = -d ln(amp)/dt / (2 k^2)
    delta_chi    = (chi_perp_eff - chi_perp) / chi_par

is the paper's pollution metric (their Fig. 2 scaling in eps = 1/R and
resolution), including the on-grid X/O-point nulls where b-hat is
undefined and the unmag fallback / iso_B semantics engage.

SDE reference arms are only meaningful at mild anisotropy (the parallel
hop scale sqrt(2 chi_par dt) exceeds the box at eps << 1e-2), so this
instrument is FD-focused by default.

Usage:
  python3 qdsmc_nimrod_test.py --ncell 128 --eps 1e-6 --fd-order 4 --out n.npz
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
parser.add_argument("--chi-perp", type=float, default=127.0, help="chi_perp [m^2/s]")
parser.add_argument(
    "--eps",
    type=float,
    default=1.0e-4,
    help="anisotropy ratio chi_perp/chi_par (chi_par = chi_perp/eps)",
)
parser.add_argument("--amp", type=float, default=0.2, help="mode amplitude / Te0")
parser.add_argument(
    "--conduction-op",
    choices=["sde", "fd"],
    default="fd",
    help="conduction operator (sde only meaningful at eps >= ~1e-2)",
)
parser.add_argument("--fd-order", type=int, choices=[2, 4], default=2)
parser.add_argument(
    "--fd-limiter", choices=["none", "upwind1", "smart"], default="smart"
)
parser.add_argument("--fd-time", choices=["ssprk2", "rkf45"], default="ssprk2")
parser.add_argument("--fd-cfl", type=float, default=0.4)
parser.add_argument(
    "--iso-b",
    type=float,
    default=-1.0,
    help="isotropic blend below this |B| [T] (<0 = off): null-semantics knob",
)
parser.add_argument("--npts", type=int, nargs="+", default=[3])
parser.add_argument("--max-hop", type=float, default=6.0)
parser.add_argument("--advance", choices=["euler", "leapfrog", "pc"], default="pc")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--verbose", type=int, default=0)
args = parser.parse_args()

# ----------------------------------------------------------------------
# Problem definition (cat's-eye skeleton, wiggle density idiom)
# ----------------------------------------------------------------------
L = 1.0
N = args.ncell
n0 = 1.0e21
Te0_eV = 10.0
gamma = 5.0 / 3.0
mass_factor = 1.0e6
B0 = 2.0e-5

kmode = 2.0 * np.pi / L
tfinal = args.tfinal
dt = tfinal / args.nsteps
chi_perp = args.chi_perp
chi_par = chi_perp / args.eps

kb = constants.kb
kappa_par_expr = f"{1.5 * n0 * kb * chi_par:.16e}"
kappa_perp_expr = f"{1.5 * n0 * kb * chi_perp:.16e}"

npts = args.npts if len(args.npts) > 1 else [args.npts[0], args.npts[0]]

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
    Bx_expression=f"{B0}*cos({kmode}*x)*sin({kmode}*z)",
    By_expression="0.0",
    Bz_expression=f"(-{B0})*sin({kmode}*x)*cos({kmode}*z)",
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

pywarpx.hybridpicmodel.qdsmc_time_advance = args.advance
pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_par(n,Te,t)", kappa_par_expr)
pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_perp(n,Te,t)", kappa_perp_expr)
pywarpx.hybridpicmodel.qdsmc_conduction_quadrature_points = npts
pywarpx.hybridpicmodel.qdsmc_conduction_flux_limit_factor = 0.0
pywarpx.hybridpicmodel.qdsmc_conduction_max_hop = args.max_hop
pywarpx.hybridpicmodel.qdsmc_conduction_operator = args.conduction_op
pywarpx.hybridpicmodel.qdsmc_conduction_fd_order = args.fd_order
pywarpx.hybridpicmodel.qdsmc_conduction_fd_limiter = args.fd_limiter
pywarpx.hybridpicmodel.qdsmc_conduction_fd_cfl = args.fd_cfl
pywarpx.hybridpicmodel.qdsmc_conduction_fd_time = args.fd_time
if args.iso_b > 0.0:
    pywarpx.hybridpicmodel.qdsmc_conduction_iso_B = args.iso_b

sim.initialize_warpx()

# ----------------------------------------------------------------------
# Te poke (step 0) + per-step mode-amplitude projection
# ----------------------------------------------------------------------
Te_wrap = fields.MultiFabWrapper(mf_name="hybrid_electron_temperature_fp", level=0)

K_per_eV = constants.q_e / constants.kb
Te0_K = Te0_eV * K_per_eV
state = {"initial": None, "sum0": None, "amp_t": [], "t": []}
mode = {"grid": None, "norm": None}


def _node_coords():
    te = Te_wrap[:, :]
    nx, nz = te.shape
    x = np.linspace(0.0, L, nx)
    z = np.linspace(0.0, L, nz)
    return np.meshgrid(x, z, indexing="ij")


def mode_shape():
    if mode["grid"] is None:
        xg, zg = _node_coords()
        m = np.cos(kmode * xg) * np.cos(kmode * zg)
        mode["grid"] = m[:-1, :-1]
        mode["norm"] = (mode["grid"] ** 2).sum()
    return mode["grid"], mode["norm"]


def project_amp():
    m, nrm = mode_shape()
    te = Te_wrap[:, :][:-1, :-1]
    return float((te * m).sum() / nrm) / Te0_K


def poke_te():
    if libwarpx_istep() != 0:
        return
    xg, zg = _node_coords()
    te = Te0_K * (1.0 + args.amp * np.cos(kmode * xg) * np.cos(kmode * zg))
    Te_wrap[:, :] = te
    state["initial"] = te.copy()
    state["sum0"] = te[:-1, :-1].sum()
    state["amp_t"].append(project_amp())
    state["t"].append(0.0)


def track_amp():
    state["amp_t"].append(project_amp())
    state["t"].append(libwarpx_istep() * dt)


def libwarpx_istep():
    from pywarpx import libwarpx

    return libwarpx.libwarpx_so.get_instance().getistep(0)


callbacks.installparticleinjection(poke_te)
callbacks.installafterstep(track_amp)

# ----------------------------------------------------------------------
# Run and score
# ----------------------------------------------------------------------
sim.step(args.nsteps)

t = np.array(state["t"])
amp = np.array(state["amp_t"])
sel = amp > 0.0
# least-squares exponential decay rate over the run
slope = np.polyfit(t[sel], np.log(amp[sel]), 1)[0]
chi_perp_eff = -slope / (2.0 * kmode**2)
delta_chi = (chi_perp_eff - chi_perp) / chi_par

te_final = Te_wrap[:, :]
sum1 = te_final[:-1, :-1].sum()
sum_drift = (sum1 - state["sum0"]) / state["sum0"]

# analytic check: amplitude after tfinal by chi_perp alone
amp_exact = args.amp * np.exp(-2.0 * kmode**2 * chi_perp * tfinal)

np.savez_compressed(
    args.out,
    ncell=N,
    nsteps=args.nsteps,
    dt=dt,
    tfinal=tfinal,
    L=L,
    chi_perp=chi_perp,
    chi_par=chi_par,
    eps=args.eps,
    amp0=args.amp,
    npts=npts,
    substeps=substeps,
    conduction_op=args.conduction_op,
    fd_order=args.fd_order,
    fd_limiter=args.fd_limiter,
    fd_time=args.fd_time,
    iso_b=args.iso_b,
    amp_t=amp,
    t=t,
    amp_exact_final=amp_exact,
    chi_perp_eff=chi_perp_eff,
    delta_chi=delta_chi,
    te_sum0=state["sum0"],
    te_sum1=sum1,
    sum_drift=sum_drift,
)
print(
    f"[harness] nimrod done: op={args.conduction_op} order={args.fd_order} "
    f"lim={args.fd_limiter} N={N} eps={args.eps:.1e} "
    f"chi_perp_eff/chi_perp={chi_perp_eff / chi_perp:.4f} "
    f"delta_chi/chi_par={delta_chi:+.3e} sum drift={sum_drift:+.3e}"
)
sys.stdout.flush()
