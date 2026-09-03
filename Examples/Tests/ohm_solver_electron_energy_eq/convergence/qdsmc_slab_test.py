#!/usr/bin/env python3
"""Thrust D / gate G4 slab instrument: wall conduction BCs + tallies.

2D XZ slab, x non-periodic (dirichlet fields, reflecting particles),
z periodic (thin, nz = 8). Uniform heavy ions at rest, B = 0 (the
unmagnetized branch conducts isotropically at chi_par), constant kappa.

Modes
-----
isothermal : T(x=0) = T1, T(x=L) = T2 via qdsmc_conduction_bc_* pins.
             Steady state = linear profile; q_wall = kappa (T1-T2)/L.
flux       : prescribed q0 [W/m^2] into x_lo, adiabatic x_hi. Energy
             rises at the injected rate.

G4 checks (printed + saved; loose asserts, see --assert):
1. BUDGET CLOSURE: Delta Sigma(u) == sum of wall tallies to round-off
   (u = 3/2 n kB Te summed over unique nodes; the tallies are
   accumulated in the same discrete norm by ApplyQdsmcConductionWallBCs
   and read back via warpx.get_qdsmc_wall_tally(dim, side); the metric
   normalizes by the GROSS |tally| so a steady state with zero net
   exchange cannot inflate it).
2. PROFILE: final Te(x) vs the analytic steady profile (isothermal:
   linear between the pinned rows).
3. WALL FLUX: physical q from the steady tally rate,
   q_num = (dTally/dt) * dx / n_zrows, vs kappa (T1 - T2)/L.

Measured operating envelope (2026-08-05, fluxform+ppm, N=32): wall-flux
fidelity follows the WING hop size x_max*sp (sp = sqrt(2 chi dt_c)):
q/q_exact = 0.9998 at sp/dx = 0.63 (npts=3) but sags to 0.962 at 0.89
and 0.746 at 1.26 with an even-odd near-wall stagger (wing hops > 1
cell couple sublattices weakly -- the same hop-aliasing family as the
curvature-leak plateau; npts=2's shorter wings stay accurate there,
1.0041 at sp/dx = 0.89). Keep x_max*sp <~ 0.8 dx near walls; the
defaults sit at sp/dx = 0.63. The fluxform specular fold-back at walls
(method of images) is what makes this work at all: the earlier hard
clamp read as a wall contact resistance (interior saw effective wall
temperatures ~9.7/4.3 eV instead of the pinned 15/5; scatter -- clamp
unfixed, control arm -- still shows q/q_exact ~ 0.49 here).

The PEC/dirichlet + rho pairing trap (ring-test record) is handled by
reflecting particle BCs + a tiny n_floor; the harness prints the wall-row
rho so a floored wall row (which would freeze the pins behind closed
floor faces) is visible immediately.
"""

import argparse
import sys

import numpy as np

from pywarpx import callbacks, fields, picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument("--ncell", type=int, default=32, help="x cells")
parser.add_argument("--nz", type=int, default=8)
parser.add_argument("--nsteps", type=int, default=1536)
parser.add_argument("--tfinal", type=float, default=3.0e-4, help="total time [s]")
parser.add_argument("--chi", type=float, default=2.0e3, help="chi [m^2/s]")
parser.add_argument("--bc", choices=["isothermal", "flux"], default="isothermal")
parser.add_argument("--T1", type=float, default=15.0, help="x_lo wall Te [eV]")
parser.add_argument("--T2", type=float, default=5.0, help="x_hi wall Te [eV]")
parser.add_argument("--q0", type=float, default=None, help="flux mode q [W/m^2]")
parser.add_argument("--npts", type=int, nargs="+", default=[3])
parser.add_argument("--max-hop", type=float, default=2.0)
parser.add_argument(
    "--form", choices=["scatter", "layer", "fluxform"], default="fluxform"
)
parser.add_argument("--recon", choices=["plm", "ppm"], default="ppm")
parser.add_argument("--advance", choices=["euler", "leapfrog", "pc"], default="pc")
parser.add_argument("--assert", dest="do_assert", type=int, default=1)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--verbose", type=int, default=0)
parser.add_argument(
    "--conduction-op",
    choices=["sde", "fd"],
    default="sde",
    help="conduction operator (hybrid_pic_model.qdsmc_conduction_operator)",
)
args = parser.parse_args()

# ----------------------------------------------------------------------
# Problem definition
# ----------------------------------------------------------------------
L = 1.0
N = args.ncell
NZ = args.nz
n0 = 1.0e18
Te0_eV = 10.0
gamma = 5.0 / 3.0
# 1e12 (not the usual 1e6): the pinned wall rows hold a permanent sharp
# grad Pe whose E field slowly accelerates 1e6-heavy ions -- the rho and
# V_e drift then moves Sigma(rho Te) without conduction doing anything,
# polluting the closure metric at ~1e-4/step. 1e12 makes the background
# truly static so closure isolates conduction + pins alone.
mass_factor = 1.0e12

tfinal = args.tfinal
dt = tfinal / args.nsteps
chi0 = args.chi
kb = constants.kb
qe = constants.q_e
K_per_eV = qe / kb
kappa = 1.5 * n0 * kb * chi0  # W/(m K), constant
kappa_par_expr = f"{kappa:.16e}"

# default flux-mode drive: the steady isothermal wall flux, for scale
q0 = args.q0 if args.q0 is not None else kappa * (args.T1 - args.T2) * K_per_eV / L

npts = args.npts if len(args.npts) > 1 else [args.npts[0], args.npts[0]]

# ----------------------------------------------------------------------
# PICMI setup (x walls, z periodic, B = 0 -> isotropic at chi_par)
# ----------------------------------------------------------------------
grid = picmi.Cartesian2DGrid(
    number_of_cells=[N, NZ],
    lower_bound=[0.0, 0.0],
    upper_bound=[L, L * NZ / N],
    lower_boundary_conditions=["dirichlet", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["reflecting", "periodic"],
    upper_boundary_conditions_particles=["reflecting", "periodic"],
    warpx_max_grid_size=max(N, NZ),
)

solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=gamma,
    Te=Te0_eV,
    n0=n0,
    n_floor=1.0e-6 * n0,
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

import pywarpx  # noqa: E402

pywarpx.hybridpicmodel.qdsmc_time_advance = args.advance
pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_par(n,Te,t)", kappa_par_expr)
pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_perp(n,Te,t)", "0.0")
pywarpx.hybridpicmodel.qdsmc_conduction_quadrature_points = npts
pywarpx.hybridpicmodel.qdsmc_conduction_flux_limit_factor = 0.0
pywarpx.hybridpicmodel.qdsmc_conduction_max_hop = args.max_hop
pywarpx.hybridpicmodel.qdsmc_advection_form = args.form
pywarpx.hybridpicmodel.qdsmc_conduction_operator = args.conduction_op
pywarpx.hybridpicmodel.qdsmc_advection_reconstruction = args.recon
if args.bc == "isothermal":
    pywarpx.hybridpicmodel.qdsmc_conduction_bc_lo = ["isothermal", "adiabatic"]
    pywarpx.hybridpicmodel.qdsmc_conduction_bc_hi = ["isothermal", "adiabatic"]
    pywarpx.hybridpicmodel.qdsmc_conduction_bc_Te_lo = [args.T1, 0.0]
    pywarpx.hybridpicmodel.qdsmc_conduction_bc_Te_hi = [args.T2, 0.0]
else:
    pywarpx.hybridpicmodel.qdsmc_conduction_bc_lo = ["flux", "adiabatic"]
    pywarpx.hybridpicmodel.qdsmc_conduction_bc_hi = ["adiabatic", "adiabatic"]
    pywarpx.hybridpicmodel.qdsmc_conduction_bc_q_lo = [q0, 0.0]

sim.initialize_warpx()

# ----------------------------------------------------------------------
# Wrappers + bookkeeping
# ----------------------------------------------------------------------
Te_wrap = fields.MultiFabWrapper(mf_name="hybrid_electron_temperature_fp", level=0)
rho_wrap = fields.MultiFabWrapper(mf_name="rho_fp", level=0)


def wx_instance():
    from pywarpx import libwarpx

    return libwarpx.libwarpx_so.get_instance()


def tallies():
    wx = wx_instance()
    return np.array([[wx.get_qdsmc_wall_tally(d, s) for s in (0, 1)] for d in (0, 1)])


def sigma_u():
    """Sigma(3/2 n kB Te) over unique nodes (x all physical, z drop seam)."""
    te = Te_wrap[:, :][:, :-1]
    ne = rho_wrap[:, :][:, :-1] / qe
    return (1.5 * kb * ne * te).sum()


state = {}


def capture0():
    if wx_instance().getistep(0) != 0:
        return
    state["u0"] = sigma_u()
    state["tal0"] = tallies()
    rho = rho_wrap[:, :]
    if args.verbose:
        print(
            f"[harness] wall-row rho/qe/n0: lo {rho[0, :].mean() / qe / n0:.3f} "
            f"hi {rho[-1, :].mean() / qe / n0:.3f}",
            flush=True,
        )


callbacks.installparticleinjection(capture0)

# ----------------------------------------------------------------------
# Run: first 3/4 to (near) steady state, then a tally-rate window
# ----------------------------------------------------------------------
n_a = (3 * args.nsteps) // 4
sim.step(n_a)
tal_a = tallies()
t_a = wx_instance().gett_new(0)
sim.step(args.nsteps - n_a)
tal_b = tallies()
t_b = wx_instance().gett_new(0)

te_final = Te_wrap[:, :]
rho_final = rho_wrap[:, :]
u1 = sigma_u()

# 1. budget closure (discrete norm; tallies and Sigma(u) share it).
# Normalize by the GROSS tally: at steady state the NET exchange is ~0
# and would inflate a net-normalized metric from pure round-off.
d_sigma = u1 - state["u0"]
tally_sum = (tal_b - state["tal0"]).sum()
tally_gross = np.abs(tal_b - state["tal0"]).sum()
closure = abs(d_sigma - tally_sum) / max(tally_gross, abs(d_sigma), 1e-300)

# 2. profile vs analytic steady state (x profile, z-averaged)
nx = te_final.shape[0]
x = np.linspace(0.0, L, nx)
prof = te_final[:, :-1].mean(axis=1)
if args.bc == "isothermal":
    exact = (args.T1 + (args.T2 - args.T1) * x / L) * K_per_eV
else:
    # flux mode has no steady state (energy rises); compare nothing
    exact = np.full_like(prof, np.nan)
rel_l2 = (
    float(np.sqrt(((prof - exact) ** 2).mean()) / (Te0_eV * K_per_eV))
    if args.bc == "isothermal"
    else float("nan")
)

# 3. physical wall flux from the steady tally-rate window
dx = L / N
nzrows = te_final.shape[1] - 1
rate = (tal_b - tal_a) / (t_b - t_a)  # u-units / s per (dim, side)
q_lo = rate[0, 0] * dx / nzrows
q_hi = rate[0, 1] * dx / nzrows
q_exact = kappa * (args.T1 - args.T2) * K_per_eV / L if args.bc == "isothermal" else q0

np.savez_compressed(
    args.out,
    bc=args.bc,
    form=args.form,
    recon=args.recon,
    ncell=N,
    nz=NZ,
    nsteps=args.nsteps,
    dt=dt,
    tfinal=tfinal,
    chi0=chi0,
    kappa=kappa,
    T1=args.T1,
    T2=args.T2,
    q0=q0,
    npts=npts,
    te_final=te_final,
    rho_final=rho_final,
    profile=prof,
    profile_exact=exact,
    rel_l2=rel_l2,
    closure=closure,
    d_sigma=d_sigma,
    tally_sum=tally_sum,
    tallies_final=tal_b,
    q_lo=q_lo,
    q_hi=q_hi,
    q_exact=q_exact,
)
print(
    f"[harness] slab done: bc={args.bc} form={args.form} N={N} "
    f"steps={args.nsteps} | closure={closure:.3e} "
    f"| profile relL2={rel_l2:.4e} "
    f"| q_lo={q_lo:.4e} q_hi={q_hi:.4e} q_exact={q_exact:.4e} "
    f"(q_lo/q_exact={q_lo / q_exact:+.4f}, q_hi/q_exact={q_hi / q_exact:+.4f})",
    flush=True,
)

if args.do_assert:
    assert closure < 1.0e-9, f"budget closure {closure:.3e} above round-off"
    if args.bc == "isothermal":
        assert rel_l2 < 5.0e-3, f"steady profile relL2 {rel_l2:.3e}"
        assert abs(q_lo / q_exact - 1.0) < 0.02, f"q_lo off: {q_lo / q_exact:.4f}"
        assert abs(q_hi / q_exact + 1.0) < 0.02, f"q_hi off: {q_hi / q_exact:.4f}"
    else:
        assert abs(tally_sum / (q0 * t_b * nzrows / dx) - 1.0) < 0.02
sys.stdout.flush()
