#!/usr/bin/env python3
"""C.7 Sharma-Hammett ring test: curved-field-line leak of the Ito conduction.

2D NON-periodic (dirichlet fields, absorbing particles) hybrid-PIC run with
circular field lines B = B0 * phi_hat around the domain center,
kappa_perp = 0, and a hot patch at (r0, phi0): heat must spread ONLY along
the circles. Any growth of the blob's radial variance is the
curved-field-line leak the straight-hop tilted test cannot see.

Design notes from the N=64 smoke campaign (why non-periodic and unblended):
- A periodic box is UNUSABLE: phi_hat flips sign across the periodic seam,
  and the D-tensor jump (Dxz) makes the seam/corners a div-D kick line
  that shreds the far field within one run.
- Hermite-blending b to straight zhat near the seam does NOT rescue it:
  b then rotates 90 deg over a few cells, the blend annulus is itself in
  the under-resolved-curvature regime (R_eff ~ blend width ~ hop), and its
  O(1) local stirring random-walks ~0.05 L into the measurement band.
- With walls instead: the E7-clamp pile-up is confined ALONG wall-grazing
  circles (r > 0.5), which never intersect the band's circles (< 0.45);
  cross-circle contamination is limited by the (small) leak itself.
- Even so, bare walls + a bare axis still pollute: circles that pass
  within a hop of a wall drain into the rho = 0 (PEC dirichlet) wall rows,
  and the under-resolved-curvature stirring at the axis (R_c ~ dx < hop)
  creeps hot ridges out to r ~ 0.17. Both are killed EXACTLY by conducting
  only in an annulus: the density is loaded as n0 in [R_KLO, R_KHI] and
  0.3 n0 outside, and kappa_par = 1.5 kB chi0 * n * (n > 0.6 n0), so
  chi = chi0 exactly in the annulus and 0 outside (chi = 0 is the verified
  identity). The trick is exact for a phi_hat field: grad ln n is radial
  and perpendicular to b, so the Ito density drift D.grad(ln n) vanishes,
  and div(chi(r) phi phi) has no dchi/dr term, so the transitions are
  kick-free; with Te = T0 there, grad Pe keeps zero curl and B stays
  static.
- Do NOT give the axis node an out-of-plane b floor: a b || yhat node in
  2D receives deposits and never emits (one-way trap, Te grew 25x T0).
- n0 = 1e24 (not the 1e18 of the aligned/tilted decks) kills the Hall
  feedback of the ring field's physical background J_y = B0/(mu0 r), which
  at 1e18 evolved B by ~16 B0 near the axis within one run. dB/B0 ~ 1/n0;
  conduction only reads the b DIRECTION and chi0 enters via the kappa
  parser (~ n0), so the conduction problem is unchanged.

Theory (chord-vs-arc): a parallel hop of length l = x_q * sigma lands at
radius r + x_q^2 sigma^2/(2 r); the Ito div-D drift is -chi dt/r =
-sigma^2/(2 r), exactly the quadrature MEAN of the offsets, so only the
quadrature VARIANCE of x^2 leaks:
    chi_perp_num / chi_par ~ Var_q[x^2] * sigma^2 / (4 r0^2),
i.e. ~ chi dt / r0^2 for npts >= 3 (Var[x^2] = 2, Gaussian-exact) and ~ 0 at
npts = 2 (|x| = 1 deterministic -> the drift cancels every hop, not just the
mean; only O(l^3/R^2) residuals and the grid remap floor remain).

The ring also validates the discrete div-D drift itself: if it were missing
or wrong, the blob's MEAN radius would migrate at d<r>/dt ~ chi/r.

Exact parallel solution (uniform chi, per circle of radius r, arc s = r*phi):
Gaussian in arc with sigma_t^2 = sigma_phi^2 + 2 chi t, amplitude
sigma_phi/sigma_t, radial profile unchanged. All moments are scored against
the exact field with the SAME estimator (trap 7: wrapped moments lie at the
1e-3 level), so estimator bias cancels; the reported leak is
(sig_r^2_meas - sig_r^2_exact) / (2 tfinal).

Whistler guard: the dt sweep at fixed N breaks the parabolic-invariance of
the B0 = 2e-5 rule, so the B substep count is auto-scaled to keep
omega_wh(k_max) * dt/substeps <= 0.1.

Usage:
  python3 qdsmc_ring_test.py --ncell 128 --nsteps 64 --npts 3 --out r64.npz
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
    default=None,
    help="chi_par [m^2/s]; default spreads the arc sigma 3x over tfinal",
)
parser.add_argument("--r0", type=float, default=0.32, help="ring radius / L")
parser.add_argument("--sigma-r", type=float, default=0.03, help="radial blob sigma / L")
parser.add_argument(
    "--sigma-phi", type=float, default=0.05, help="initial arc-length sigma / L"
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
    default=20.0,
    help="hop cap in cells (large: the cap must NOT engage in this test)",
)
parser.add_argument("--flux-limit", type=float, default=0.0)
parser.add_argument("--advance", choices=["euler", "leapfrog", "pc"], default="pc")
parser.add_argument("--grad-deposit", type=int, choices=[0, 1], default=1)
parser.add_argument(
    "--no-cond",
    action="store_true",
    help="debug control: do not set the kappa parsers (no conduction "
    "substep at all; isolates the transport/E7 wall behavior)",
)
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
    "--deposit-kernel",
    choices=["hat", "keys"],
    default="hat",
    help="scatter-form deposit kernel (keys = conservative cubic, "
    "no B1 correction, not monotone)",
)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--verbose", type=int, default=0)
args = parser.parse_args()

# ----------------------------------------------------------------------
# Problem definition
# ----------------------------------------------------------------------
L = 1.0
N = args.ncell
n0 = 1.0e24
Te0_eV = 10.0
gamma = 5.0 / 3.0
mass_factor = 1.0e6
B0 = 2.0e-5

# conduction annulus (units of L): chi = chi0 for r in [R_KLO, R_KHI]
# via the density-keyed kappa switch, 0 outside; the moment band
# (blob +- 3.3 sigma_r) must stay inside with margin. The same radii
# bound the noise-diagnostic zones.
R_KLO, R_KHI = 0.15, 0.465
N_OUT_FRAC = 0.3  # out-of-annulus density (>> floor, < the kappa switch)
KAPPA_SWITCH_FRAC = 0.6  # kappa on where n > this * n0
TRANS_W = 0.015  # tanh half-width of the density transitions [L]

r0 = args.r0 * L
sr = args.sigma_r * L
sp = args.sigma_phi * L
xc = zc = 0.5 * L
phi0 = 0.0  # patch at (xc + r0, zc)
blob_amp = 1.0
tfinal = args.tfinal
dt = tfinal / args.nsteps
# default: arc sigma triples over the run, sigma_t(tf) = 3 sigma_phi
chi0 = args.chi if args.chi is not None else 4.0 * sp**2 / tfinal

kb = constants.kb
# chi = kappa/(1.5 n kB) = chi0 * (n > switch): exact chi0 in the annulus
# (the n cancels), exactly 0 outside.
kappa_par_expr = f"{1.5 * kb * chi0:.16e}*n*(n>{KAPPA_SWITCH_FRAC * n0:.6e})"
kappa_perp_expr = "0.0"

npts = args.npts if len(args.npts) > 1 else [args.npts[0], args.npts[0]]

# Whistler-stable B substep count for THIS dt (the trap-3 B0 = 2e-5 rule
# assumed dt ~ dx^2 sweeps; a dt sweep at fixed N must rescale substeps):
# omega_wh(k_max) = (pi N / L)^2 * B0 / (mu0 q n0), keep om*dt_sub <= 0.1.
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
    lower_boundary_conditions=["dirichlet", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
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

# Annular density: n0 in the conduction annulus, N_OUT_FRAC*n0 outside
# (tanh edges). Keys the kappa switch; constant on every circle, so the
# per-circle exact solution is unchanged.
_rr = f"sqrt((x-{xc})*(x-{xc})+(z-{zc})*(z-{zc}))"
_ei = f"0.5*(1.0+tanh(({_rr}-{R_KLO * L})/{TRANS_W * L}))"
_eo = f"0.5*(1.0+tanh(({R_KHI * L}-{_rr})/{TRANS_W * L}))"
dens_expr = f"{n0}*({N_OUT_FRAC}+{1.0 - N_OUT_FRAC}*({_ei})*({_eo}))"
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

# Circular field lines everywhere: B = B0 * phi_hat (uniform |B|), guarded
# within a quarter cell of the axis (the exact-center node then sees
# |B| = 0 and takes the isotropic-unmagnetized fallback, which re-emits --
# harmless, unlike an out-of-plane b floor).
dxc = L / N
rr = f"sqrt((x-{xc})*(x-{xc})+(z-{zc})*(z-{zc}))"
rs = f"max({rr},{0.25 * dxc})"
B_field = picmi.AnalyticInitialField(
    Bx_expression=f"-{B0}*(z-{zc})/{rs}",
    By_expression="0.0",
    Bz_expression=f"{B0}*(x-{xc})/{rs}",
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
if not args.no_cond:
    pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_par(n,Te,t)", kappa_par_expr)
    pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_perp(n,Te,t)", kappa_perp_expr)
pywarpx.hybridpicmodel.qdsmc_conduction_quadrature_points = npts
pywarpx.hybridpicmodel.qdsmc_conduction_flux_limit_factor = args.flux_limit
pywarpx.hybridpicmodel.qdsmc_conduction_max_hop = args.max_hop
pywarpx.hybridpicmodel.qdsmc_conduction_form = args.form
pywarpx.hybridpicmodel.qdsmc_conduction_interp = args.interp
pywarpx.hybridpicmodel.qdsmc_conduction_curved_feet = args.curved_feet
pywarpx.hybridpicmodel.qdsmc_conduction_deposit_kernel = args.deposit_kernel
# MUST be off here (default on): the dirichlet/PEC field BC zeroes rho on
# the wall rows, fast-front then boosts their chi to the hop-cap ceiling,
# and the D cliff's div-D drift kicks the row-1 daughters into the wall
# where the floored-node recovery skip deletes the energy from Te
# (measured: Te(row +-1) -> 0 in ONE substep, Sigma(Te) -6% even at
# chi = 0). Real Thrust-D wall BCs must handle this pairing properly.
pywarpx.hybridpicmodel.qdsmc_conduction_vacuum_fast_front = 0

sim.initialize_warpx()

# ----------------------------------------------------------------------
# Te poke (step 0) + exact solution, both in polar form
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


def profile(xg, zg, sig_arc):
    """Ring patch, fractional amplitude: radial Gaussian (r-r0, sigma_r)
    times a wrapped arc-length Gaussian at each node's OWN radius (period
    2 pi r), so per-circle diffusion in arc is exact with sigma_arc(t)^2 =
    sigma_phi^2 + 2 chi t and amplitude sigma_phi/sigma_arc."""
    r = np.sqrt((xg - xc) ** 2 + (zg - zc) ** 2)
    phi = np.arctan2(zg - zc, xg - xc)  # phi0 = 0 at +x
    dphi = (phi - phi0 + np.pi) % (2.0 * np.pi) - np.pi
    amp = blob_amp * sp / sig_arc
    rad = np.exp(-((r - r0) ** 2) / (2.0 * sr**2))
    arc = np.zeros_like(xg)
    for m in range(-4, 5):
        s_arc = r * (dphi + 2.0 * np.pi * m)
        arc += np.exp(-(s_arc**2) / (2.0 * sig_arc**2))
    return amp * rad * arc


def poke_te():
    if libwarpx_istep() != 0:
        return
    xg, zg = _node_coords()
    te = Te0_eV * K_per_eV * (1.0 + profile(xg, zg, sp))
    Te_wrap[:, :] = te
    state["initial"] = te.copy()
    state["sum0"] = te[:-1, :-1].sum()
    if args.verbose:
        print(f"[harness] poked ring patch at step 0: max {te.max():.6e} K")


def libwarpx_istep():
    from pywarpx import libwarpx

    return libwarpx.libwarpx_so.get_instance().getistep(0)


callbacks.installparticleinjection(poke_te)

# ----------------------------------------------------------------------
# Run, score moments against the exact-field estimator, save
# ----------------------------------------------------------------------
sim.step(args.nsteps)

te_final = Te_wrap[:, :]
xg, zg = _node_coords()
st = np.sqrt(sp**2 + 2.0 * chi0 * tfinal)
te_exact = Te0_eV * K_per_eV * (1.0 + profile(xg, zg, st))

Te0_K = Te0_eV * K_per_eV
diff = (te_final - te_exact)[:-1, :-1]
rel_l2 = np.sqrt((diff**2).mean()) / Te0_K


BAND_LO = max(r0 - 3.3 * sr, (R_KLO + 2.0 * TRANS_W) * L)
BAND_HI = min(r0 + 3.3 * sr, (R_KHI - 2.0 * TRANS_W) * L)


def ring_moments(te):
    """Band-restricted w-weighted radial moments + circular arc spread.

    The band tracks the blob (r0 +- 3.3 sigma_r) and is clamped inside the
    pure phi_hat zone; the SAME estimator is applied to the measured and
    exact fields so its (truncation and wrapping) bias subtracts out.
    """
    w = (te[:-1, :-1] - Te0_K).clip(min=0.0)
    r = np.sqrt((xg[:-1, :-1] - xc) ** 2 + (zg[:-1, :-1] - zc) ** 2)
    phi = np.arctan2(zg[:-1, :-1] - zc, xg[:-1, :-1] - xc)
    band = (r > BAND_LO) & (r < BAND_HI)
    wb, rb, phib = w[band], r[band], phi[band]
    wsum = wb.sum()
    rbar = (wb * rb).sum() / wsum
    sig_r_sq = (wb * rb**2).sum() / wsum - rbar**2
    # wrapped-normal arc spread from the circular first moment
    c = (wb * np.exp(1j * (phib - phi0))).sum() / wsum
    sig_arc_sq = -2.0 * np.log(max(abs(c), 1e-300)) * rbar**2
    out_frac = 1.0 - wsum / max(w.sum(), 1e-300)
    return rbar, sig_r_sq, sig_arc_sq, out_frac


rbar_m, sig_r_sq_m, sig_arc_sq_m, out_frac_m = ring_moments(te_final)
rbar_x, sig_r_sq_x, sig_arc_sq_x, out_frac_x = ring_moments(te_exact)

chi_perp_num = (sig_r_sq_m - sig_r_sq_x) / (2.0 * tfinal)
rdrift = rbar_m - rbar_x
chi_par_meas = (sig_arc_sq_m - sp**2) / (2.0 * tfinal)

# out-of-band noise diagnostics: the chi = 0 zones at the axis and walls
# must stay at the identity level
r_all = np.sqrt((xg[:-1, :-1] - xc) ** 2 + (zg[:-1, :-1] - zc) ** 2)
dv_all = np.abs(te_final[:-1, :-1] - Te0_K) / Te0_K
axis_noise = dv_all[r_all < R_KLO * L].max()
wall_noise = dv_all[r_all > R_KHI * L].max()

sum1 = te_final[:-1, :-1].sum()
np.savez_compressed(
    args.out,
    ncell=N,
    nsteps=args.nsteps,
    dt=dt,
    tfinal=tfinal,
    L=L,
    chi0=chi0,
    npts=npts,
    substeps=substeps,
    r0=r0,
    sigma_r=sr,
    sigma_phi=sp,
    max_hop=args.max_hop,
    advance=args.advance,
    form=args.form,
    interp=args.interp,
    curved_feet=args.curved_feet,
    deposit_kernel=args.deposit_kernel,
    te_initial=state["initial"],
    te_final=te_final,
    te_exact=te_exact,
    rel_l2=rel_l2,
    rbar_meas=rbar_m,
    rbar_exact=rbar_x,
    rdrift=rdrift,
    sig_r_sq_meas=sig_r_sq_m,
    sig_r_sq_exact=sig_r_sq_x,
    sig_arc_sq_meas=sig_arc_sq_m,
    sig_arc_sq_exact=sig_arc_sq_x,
    chi_perp_num=chi_perp_num,
    chi_par_meas=chi_par_meas,
    out_frac=out_frac_m,
    axis_noise=axis_noise,
    wall_noise=wall_noise,
    te_sum0=state["sum0"],
    te_sum1=sum1,
)
print(
    f"[harness] ring done: N={N} steps={args.nsteps} npts={npts} "
    f"substeps={substeps} chi0={chi0:.1f} "
    f"chi_perp_num/chi0={chi_perp_num / chi0:.3e} "
    f"rdrift={rdrift / L:.3e} L "
    f"chi_par_meas/chi0={chi_par_meas / chi0:.4f} relL2={rel_l2:.4e} "
    f"axis_noise={axis_noise:.2e} wall_noise={wall_noise:.2e} "
    f"sum drift={(sum1 - state['sum0']) / state['sum0']:.3e}"
)
sys.stdout.flush()
