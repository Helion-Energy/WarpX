#!/usr/bin/env python3
"""C.7 Zeldovich nonlinear front: kappa ~ Te^{5/2} against a 1D reference.

Aligned uniform B (the verified G3 skeleton): a hot Gaussian slab in z
(uniform in x) conducts with the Spitzer-like nonlinear parallel
conductivity kappa_par = kappa0 (Te/Te0)^{5/2}. The sharp
Zeldovich-Kompaneets front is the acid test of the T-dependent kappa
parser, the source-slope deposit on steep fronts, and the free-streaming
limiter (with --flux-limit f the front speed must obey
v_front <= f * v_te(T_front)).

Scoring is against a 1D flux-conservative explicit reference integration
of dT/dt = d/dz [ chi0 (T/T0)^{5/2} dT/dz ] with the same IC and a
T0-floored background (the analytic ZK similarity solution assumes a cold
background; the reference includes the floor honestly, and itself
reproduces the ZK exponent z_f ~ t^{2/9} while T_front >> T0).

Usage:
  python3 qdsmc_zeldovich_test.py --ncell 128 --nsteps 256 --out z.npz
"""

import argparse
import sys

import numpy as np

from pywarpx import callbacks, fields, picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument("--ncell", type=int, default=128)
parser.add_argument("--nsteps", type=int, default=256)
parser.add_argument("--tfinal", type=float, default=1.0e-5)
parser.add_argument(
    "--chi0",
    type=float,
    default=3.0,
    help="chi at the background Te0 [m^2/s]; peak chi = chi0 (1+amp)^{5/2}",
)
parser.add_argument("--amp", type=float, default=15.0, help="blob amplitude / Te0")
parser.add_argument(
    "--blob-sigma", type=float, default=0.04, help="initial slab sigma / L"
)
parser.add_argument("--power", type=float, default=2.5, help="kappa ~ Te^power")
parser.add_argument("--npts", type=int, nargs="+", default=[3])
parser.add_argument(
    "--flux-limit",
    type=float,
    default=0.0,
    help="free-streaming limiter f (0 = off). With f > 0 the front speed "
    "is checked against f * v_te.",
)
parser.add_argument("--max-hop", type=float, default=6.0)
parser.add_argument("--advance", choices=["euler", "leapfrog", "pc"], default="pc")
parser.add_argument("--grad-deposit", type=int, choices=[0, 1], default=1)
parser.add_argument(
    "--form",
    choices=["scatter", "layer"],
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
    "--conserve-fixup",
    type=int,
    choices=[0, 1],
    default=0,
    help="layer form: global proportional Sigma(rho T) restore per gather",
)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--verbose", type=int, default=0)
args = parser.parse_args()

# ----------------------------------------------------------------------
# Problem definition (aligned skeleton: uniform B0 zhat, uniform density)
# ----------------------------------------------------------------------
L = 1.0
N = args.ncell
n0 = 1.0e18
Te0_eV = 10.0
gamma = 5.0 / 3.0
mass_factor = 1.0e6
B0 = 2.0e-5

s0 = args.blob_sigma * L
zb = 0.5 * L
amp = args.amp
pwr = args.power
tfinal = args.tfinal
dt = tfinal / args.nsteps
chi0 = args.chi0

kb = constants.kb
# kappa(n, Te, t) with Te in eV: chi(Te) = chi0 * (Te/Te0)^power
kappa_par_expr = f"{1.5 * n0 * kb * chi0:.16e}*((Te/{Te0_eV:.6e})^{pwr:.4f})"
kappa_perp_expr = "0.0"

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
pywarpx.hybridpicmodel.qdsmc_conduction_interp = args.interp
pywarpx.hybridpicmodel.qdsmc_conduction_curved_feet = args.curved_feet
pywarpx.hybridpicmodel.qdsmc_conduction_conserve_fixup = args.conserve_fixup

sim.initialize_warpx()

# ----------------------------------------------------------------------
# Te poke (step 0) + per-step front tracking
# ----------------------------------------------------------------------
Te_wrap = fields.MultiFabWrapper(mf_name="hybrid_electron_temperature_fp", level=0)

K_per_eV = constants.q_e / constants.kb
Te0_K = Te0_eV * K_per_eV
state = {"initial": None, "sum0": None, "front_t": [], "front_z": []}


def _node_coords():
    te = Te_wrap[:, :]
    nx, nz = te.shape
    x = np.linspace(0.0, L, nx)
    z = np.linspace(0.0, L, nz)
    return x, z


def zprofile(te):
    """x-averaged z profile (the problem is 1D in z)."""
    return te[:-1, :-1].mean(axis=0)


FRONT_LEVEL = 2.0  # front = outermost z where Te > FRONT_LEVEL * Te0


def front_pos(prof, z):
    hot = np.where(prof > FRONT_LEVEL * Te0_K)[0]
    if len(hot) == 0:
        return np.nan
    zr = z[:-1][hot]
    return 0.5 * ((zr.max() - zb) + (zb - zr.min()))  # symmetric half-width


def poke_te():
    if libwarpx_istep() != 0:
        return
    x, z = _node_coords()
    zg = np.broadcast_to(z, (len(x), len(z)))
    te = Te0_K * (1.0 + amp * np.exp(-((zg - zb) ** 2) / (2.0 * s0**2)))
    Te_wrap[:, :] = te
    state["initial"] = te.copy()
    state["sum0"] = te[:-1, :-1].sum()


def track_front():
    x, z = _node_coords()
    prof = zprofile(Te_wrap[:, :])
    it = libwarpx_istep()
    state["front_t"].append(it * dt)
    state["front_z"].append(front_pos(prof, z))


def libwarpx_istep():
    from pywarpx import libwarpx

    return libwarpx.libwarpx_so.get_instance().getistep(0)


callbacks.installparticleinjection(poke_te)
callbacks.installafterstep(track_front)

# ----------------------------------------------------------------------
# 1D reference: flux-conservative explicit integration on a fine grid
# ----------------------------------------------------------------------


def reference_profile(z_nodes):
    nz = 2048
    zf = np.linspace(0.0, L, nz, endpoint=False)
    dzf = L / nz
    T = 1.0 + amp * np.exp(-((zf - zb) ** 2) / (2.0 * s0**2))  # units of Te0
    chi_pk = chi0 * (1.0 + amp) ** pwr
    dt_ref = 0.2 * dzf * dzf / chi_pk
    nsub = int(np.ceil(tfinal / dt_ref))
    dt_ref = tfinal / nsub
    for _ in range(nsub):
        Tm = 0.5 * (T + np.roll(T, -1))  # face midpoint value
        chi_f = chi0 * Tm**pwr
        flux = chi_f * (np.roll(T, -1) - T) / dzf
        T = T + dt_ref * (flux - np.roll(flux, 1)) / dzf
    return Te0_K * np.interp(z_nodes % L, zf, T, period=L)


# ----------------------------------------------------------------------
# Run and score
# ----------------------------------------------------------------------
sim.step(args.nsteps)

x, z = _node_coords()
prof_final = zprofile(Te_wrap[:, :])
ref_final = reference_profile(z[:-1])

rel_l2 = np.sqrt(((prof_final - ref_final) ** 2).mean()) / (
    np.sqrt((ref_final**2).mean())
)

# front trajectory exponent from the second half of the run (early time is
# IC-shape dominated)
ft = np.array(state["front_t"])
fz = np.array(state["front_z"])
sel = (ft > 0.3 * tfinal) & np.isfinite(fz) & (fz > 0)
front_exp = (
    float(np.polyfit(np.log(ft[sel]), np.log(fz[sel]), 1)[0])
    if sel.sum() > 4
    else np.nan
)

# front speed vs the free-streaming bound (limiter legs): max step-to-step
# speed over the tracked trajectory vs f * v_te at the CURRENT front level
vfront = np.diff(fz) / np.diff(ft)
vfront_max = np.nanmax(vfront) if len(vfront) else np.nan
v_te_front = np.sqrt(constants.kb * FRONT_LEVEL * Te0_K / constants.m_e)

te_final = Te_wrap[:, :]
sum1 = te_final[:-1, :-1].sum()
np.savez_compressed(
    args.out,
    ncell=N,
    nsteps=args.nsteps,
    dt=dt,
    tfinal=tfinal,
    L=L,
    chi0=chi0,
    amp=amp,
    power=pwr,
    npts=npts,
    substeps=substeps,
    flux_limit=args.flux_limit,
    max_hop=args.max_hop,
    advance=args.advance,
    form=args.form,
    interp=args.interp,
    curved_feet=args.curved_feet,
    blob_sigma=s0,
    front_level=FRONT_LEVEL,
    te_initial=state["initial"],
    te_final=te_final,
    prof_final=prof_final,
    ref_final=ref_final,
    front_t=ft,
    front_z=fz,
    front_exp=front_exp,
    vfront_max=vfront_max,
    v_te_front=v_te_front,
    rel_l2=rel_l2,
    te_sum0=state["sum0"],
    te_sum1=sum1,
)
print(
    f"[harness] zeldovich done: N={N} steps={args.nsteps} npts={npts} "
    f"chi0={chi0} amp={amp} flim={args.flux_limit} "
    f"relL2(prof)={rel_l2:.4e} front_exp={front_exp:.3f} "
    f"vfront_max={vfront_max:.3e} m/s (f*v_te(front)="
    f"{args.flux_limit * v_te_front if args.flux_limit > 0 else 0:.3e}) "
    f"sum drift={(sum1 - state['sum0']) / state['sum0']:.3e}"
)
sys.stdout.flush()
