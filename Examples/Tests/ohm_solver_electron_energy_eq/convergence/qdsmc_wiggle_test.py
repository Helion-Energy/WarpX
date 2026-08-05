#!/usr/bin/env python3
"""C.7 curved-field-line leak, wiggle-field instrument (G3a primary).

2D periodic hybrid-PIC run with a smooth snaking field
    Bx = eps * B0 * sin(k z),  Bz = B0,       k = 2 pi m / L
(field lines x(z) = x0 - (eps/k) cos(k z), curvature kappa(s) ~ eps k cos(k z),
R_c,min = (1 + eps^2)/(eps k)), uniform density, kappa_perp = 0, and a
Gaussian hot blob. Heat must stay on its field line: the flux function
    A(x, z) = x + (eps/k) cos(k z)      (A_y / B0; |grad A| = 1 + O(eps^2))
is exactly invariant under parallel conduction, so the curved-field-line
leak is simply the growth of the w-weighted Var(A):
    chi_perp_num = [Var_w(A)(t) - Var_w(A)(0)] / (2 t).

Why this instrument (and not the Sharma-Hammett ring, qdsmc_ring_test.py):
every ring variant needs boundaries or chi/b transitions, and each of those
is a trap/drain for the scatter-form daughters at the few-cell scale
(measured: periodic seam kicks, blend-annulus stirring, PEC-wall vacuum
fast-front cliff, chi-switch one-way traps -- see the ring script's
docstring). The wiggle box is fully periodic and statistically uniform:
no boundaries, no switches, every node equivalent; the verified
aligned/tilted machinery applies unchanged.

Theory to test (chord-vs-arc, quadrature form): the div-D Ito drift cancels
the QUADRATURE MEAN of the per-hop cross-field chord offset x_q^2 sigma^2
kappa / 2; what leaks is the quadrature VARIANCE:
    chi_perp_num / chi_par ~ (Var_q[x^2] / 4) * sigma_par^2 * <kappa^2>,
sigma_par^2 = 2 chi dt_c. npts = 2 has Var_q[x^2] = 0 (|x_q| = 1
deterministic): the leading-order leak should VANISH; npts >= 3 have
Var_q[x^2] = 2 (Gaussian moments): leak ~ chi dt <kappa^2> independent of
x_max. An eps = 0 run measures the straight-field estimator/remap floor.

Usage:
  python3 qdsmc_wiggle_test.py --ncell 128 --nsteps 64 --npts 3 --out w.npz
"""

import argparse
import sys

import numpy as np

from pywarpx import callbacks, fields, picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument("--ncell", type=int, default=128)
parser.add_argument("--nsteps", type=int, default=64)
parser.add_argument("--tfinal", type=float, default=1.0e-5, help="total time [s]")
parser.add_argument(
    "--chi",
    type=float,
    default=1.0e3,
    help="chi_par [m^2/s]",
)
parser.add_argument(
    "--eps", type=float, default=0.42, help="wiggle amplitude Bx/B0 (0 = floor run)"
)
parser.add_argument(
    "--mwiggle", type=int, default=6, help="wiggle periods across the box"
)
parser.add_argument(
    "--blob-sigma", type=float, default=0.05, help="initial blob sigma / L"
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
    default=12.0,
    help="hop cap in cells (large: the cap must NOT engage here)",
)
parser.add_argument("--flux-limit", type=float, default=0.0)
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
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--verbose", type=int, default=0)
args = parser.parse_args()

# ----------------------------------------------------------------------
# Problem definition
# ----------------------------------------------------------------------
L = 1.0
N = args.ncell
# n0 = 1e21 (not 1e18): the wiggle carries a background J_y ~ eps B0 k/mu0
# whose Hall E would evolve B by ~0.5 B0 over the run at 1e18; ~1e-3 B0 at
# 1e21. Conduction reads only the b direction; chi0 enters via the kappa
# parser (~ n0), so the conduction problem is unchanged.
n0 = 1.0e21
Te0_eV = 10.0
gamma = 5.0 / 3.0
mass_factor = 1.0e6
B0 = 2.0e-5

eps = args.eps
kw = 2.0 * np.pi * args.mwiggle / L
s0 = args.blob_sigma * L
xb, zb = 0.5 * L, 0.5 * L
blob_amp = 1.0
tfinal = args.tfinal
dt = tfinal / args.nsteps
chi0 = args.chi

kb = constants.kb
kappa_par_expr = f"{1.5 * n0 * kb * chi0:.16e}"
kappa_perp_expr = "0.0"

npts = args.npts if len(args.npts) > 1 else [args.npts[0], args.npts[0]]

# Whistler-stable B substep count for THIS dt (trap 3; the fixed B0 = 2e-5
# rule assumed parabolic dt ~ dx^2 sweeps, a dt sweep at fixed N must
# rescale): omega_wh(k_max) = (pi N/L)^2 B0/(mu0 q n0), om*dt_sub <= 0.1.
mu0 = 4.0e-7 * np.pi
om_wh = (np.pi * N / L) ** 2 * B0 / (mu0 * constants.q_e * n0)
substeps = max(4, int(np.ceil(om_wh * dt / 0.1)))

# ----------------------------------------------------------------------
# PICMI setup (periodic, uniform density: the verified G3 skeleton)
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
    Bx_expression=f"{eps * B0}*sin({kw}*z)",
    By_expression="0.0",
    Bz_expression=f"{B0}",
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
pywarpx.hybridpicmodel.qdsmc_conduction_deposit_kernel = args.deposit_kernel
pywarpx.hybridpicmodel.qdsmc_conduction_compensate = args.compensate
pywarpx.hybridpicmodel.qdsmc_conduction_fct_limiter = args.fct_limiter
pywarpx.hybridpicmodel.qdsmc_conduction_conserve_fixup = args.conserve_fixup

sim.initialize_warpx()

# ----------------------------------------------------------------------
# Te poke (step 0) + moment machinery in the flux coordinate A
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


def flux_coord(xg, zg):
    """A_y / B0 = x + (eps/k) cos(k z): exact in-plane flux function of the
    wiggle field; heat is confined to A = const up to the leak."""
    if eps == 0.0:
        return xg.copy()
    return xg + (eps / kw) * np.cos(kw * zg)


def poke_te():
    if libwarpx_istep() != 0:
        return
    xg, zg = _node_coords()
    prof = blob_amp * np.exp(-((xg - xb) ** 2 + (zg - zb) ** 2) / (2.0 * s0**2))
    te = Te0_eV * K_per_eV * (1.0 + prof)
    Te_wrap[:, :] = te
    state["initial"] = te.copy()
    state["sum0"] = te[:-1, :-1].sum()
    if args.verbose:
        print(f"[harness] poked blob at step 0: max {te.max():.6e} K")


def libwarpx_istep():
    from pywarpx import libwarpx

    return libwarpx.libwarpx_so.get_instance().getistep(0)


callbacks.installparticleinjection(poke_te)

# ----------------------------------------------------------------------
# Run, measure Var_w(A) growth, save
# ----------------------------------------------------------------------
sim.step(args.nsteps)

te_final = Te_wrap[:, :]
xg, zg = _node_coords()
Te0_K = Te0_eV * K_per_eV
A = flux_coord(xg, zg)[:-1, :-1]


def a_moments(te):
    """w-weighted mean and variance of the flux coordinate A (unique
    nodes). x is NOT wrapped: the blob starts at mid-box and spreads along
    lines whose x-excursion is eps/k << L; the same estimator on the
    initial field defines the baseline."""
    w = (te[:-1, :-1] - Te0_K).clip(min=0.0)
    wsum = w.sum()
    abar = (w * A).sum() / wsum
    avar = (w * A**2).sum() / wsum - abar**2
    return abar, avar


abar0, avar0 = a_moments(state["initial"])
abar1, avar1 = a_moments(te_final)

chi_perp_num = (avar1 - avar0) / (2.0 * tfinal)
adrift = abar1 - abar0

# parallel-spread sanity: arc length s ~ z to O(eps^2); use the w-weighted
# z variance (wrapped-normal via the circular moment, period L)
w1 = (te_final[:-1, :-1] - Te0_K).clip(min=0.0)
w0 = (state["initial"][:-1, :-1] - Te0_K).clip(min=0.0)
zu = zg[:-1, :-1]


def z_var(w):
    c = (w * np.exp(2j * np.pi * zu / L)).sum() / w.sum()
    return -2.0 * np.log(max(abs(c), 1e-300)) * (L / (2.0 * np.pi)) ** 2


chi_par_meas = (z_var(w1) - z_var(w0)) / (2.0 * tfinal)

sum1 = te_final[:-1, :-1].sum()
np.savez_compressed(
    args.out,
    ncell=N,
    nsteps=args.nsteps,
    dt=dt,
    tfinal=tfinal,
    L=L,
    chi0=chi0,
    eps=eps,
    mwiggle=args.mwiggle,
    kwiggle=kw,
    npts=npts,
    substeps=substeps,
    blob_sigma=s0,
    max_hop=args.max_hop,
    advance=args.advance,
    form=args.form,
    interp=args.interp,
    curved_feet=args.curved_feet,
    deposit_kernel=args.deposit_kernel,
    te_initial=state["initial"],
    te_final=te_final,
    avar0=avar0,
    avar1=avar1,
    abar0=abar0,
    abar1=abar1,
    adrift=adrift,
    chi_perp_num=chi_perp_num,
    chi_par_meas=chi_par_meas,
    te_sum0=state["sum0"],
    te_sum1=sum1,
)
print(
    f"[harness] wiggle done: N={N} steps={args.nsteps} npts={npts} "
    f"eps={eps} m={args.mwiggle} substeps={substeps} "
    f"chi_perp_num/chi0={chi_perp_num / chi0:.3e} adrift={adrift / L:.3e} L "
    f"chi_par_meas/chi0={chi_par_meas / chi0:.4f} "
    f"sum drift={(sum1 - state['sum0']) / state['sum0']:.3e}"
)
sys.stdout.flush()
