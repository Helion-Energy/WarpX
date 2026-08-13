#!/usr/bin/env python3
"""RZ harness for the grid-FD thermal conduction operator (curvilinear form).

RZ analog of qdsmc_conduction_test.py, restricted to the FD operator (the
SDE forms are Cartesian-only and parse-guarded in RZ). Same conduction-only
construction: uniform density, ions at rest with huge mass (J_i = 0),
uniform B = B0 zhat (curl B = 0 => J_plasma = 0 => V_e = 0; grad Pe has
zero discrete curl in RZ too -- the theta components vanish and the
in-plane curl is a commutator of difference operators -- so B stays exactly
uniform). The kappa parsers are chosen PROPORTIONAL TO n so chi =
kappa/(1.5 n kB) = chi0 exactly, independent of the RZ-deposited density
(volume-scaled deposition is not bit-uniform near the axis).

Modes
-----
radial  : Gaussian T_e bump in r CENTERED ON THE AXIS, uniform in z.
          Exact cylindrical solution T = Te0 (1 + A s0^2/st^2 *
          exp(-r^2/(2 st^2))), st^2 = s0^2 + 2 chi0 t -- the axis
          regularity + radial metric gate. Adiabatic outer wall (keep the
          blob tail negligible there).
axial   : Gaussian bump in z, uniform in r, z periodic. Exact Cartesian
          1D solution (image-summed); any r-dependence of the error is
          metric leakage into the z fluxes (axis column included).
annulus : rmin > 0 (no axis), blob mid-annulus; no closed-form solution
          with walls, so the gate is the conservation telescoping sum
          (and the run must stay finite/monotone).

All modes report the discrete conserved sum Sigma ne*Te*Vr (Vr = r dr
interior/wall nodes, dr^2/8 for the axis node -- the operator's dual-cell
measure).

Usage:
  python3 qdsmc_conduction_test_rz.py --mode radial --nr 64 --nsteps 128 --out r64.npz
"""

import argparse
import sys

import numpy as np

from pywarpx import callbacks, fields, picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["radial", "axial", "annulus"], required=True)
parser.add_argument("--nr", type=int, default=64)
parser.add_argument(
    "--nz", type=int, default=None, help="default: 8 (radial/annulus) or 64 (axial)"
)
parser.add_argument("--nsteps", type=int, default=128)
parser.add_argument("--tfinal", type=float, default=1.0e-5, help="total time [s]")
parser.add_argument(
    "--chi",
    type=float,
    default=None,
    help="chi [m^2/s]; default sets 2*chi*tfinal = sigma0^2",
)
parser.add_argument(
    "--blob-sigma",
    type=float,
    default=0.10,
    help="initial blob sigma as a fraction of the blob dimension's length",
)
parser.add_argument(
    "--rmin",
    type=float,
    default=0.0,
    help="inner radius [m] (annulus mode; must be > 0 there)",
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
    help="FD cross-flux limiter (inert for b || z, kept for parity)",
)
parser.add_argument(
    "--fd-time",
    choices=["ssprk2", "rkf45"],
    default="ssprk2",
    help="FD subcycle integrator (hybrid_pic_model.qdsmc_conduction_fd_time)",
)
parser.add_argument(
    "--fd-cfl",
    type=float,
    default=0.4,
    help="FD subcycle CFL fraction (hybrid_pic_model.qdsmc_conduction_fd_cfl)",
)
parser.add_argument(
    "--flux-limit",
    type=float,
    default=0.0,
    help="free-streaming limiter factor f (0 = off for convergence runs)",
)
parser.add_argument("--advance", choices=["euler", "leapfrog", "pc"], default="pc")
parser.add_argument("--grad-deposit", type=int, choices=[0, 1], default=1)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--verbose", type=int, default=0)
args = parser.parse_args()

# ----------------------------------------------------------------------
# Problem definition
# ----------------------------------------------------------------------
L = 1.0  # blob dimension's length (r extent for radial/annulus, z for axial)
NR = args.nr
n0 = 1.0e18
Te0_eV = 10.0
gamma = 5.0 / 3.0
mass_factor = 1.0e6
B0 = 2.0e-5  # see the 2D harness: whistler-stability floor, direction only

if args.mode == "axial":
    NZ = args.nz if args.nz is not None else 64
    Lz = L
    Lr = (L / NZ) * NR  # keep dr = dz
else:
    NZ = args.nz if args.nz is not None else 8
    Lr = L
    Lz = (L / NR) * NZ
rmin = args.rmin
if args.mode == "annulus":
    assert rmin > 0.0, "annulus mode needs --rmin > 0"
else:
    assert rmin == 0.0, "radial/axial modes run to the axis (rmin = 0)"

s0 = args.blob_sigma * L
blob_amp = 1.0
tfinal = args.tfinal
dt = tfinal / args.nsteps
chi0 = args.chi if args.chi is not None else s0**2 / (2.0 * tfinal)

kb = constants.kb
# chi = kappa / (3/2 n kB): kappa PROPORTIONAL TO n makes chi = chi0 exact,
# independent of the deposited-density noise (RZ axis deposition).
kappa_expr = f"{1.5 * kb * chi0:.16e}*n"

# blob centers
rb = 0.0 if args.mode == "radial" else 0.5 * (rmin + Lr)
zb = 0.5 * Lz

npts_unused = None  # SDE quadrature does not apply to the FD operator

# ----------------------------------------------------------------------
# PICMI setup
# ----------------------------------------------------------------------
grid = picmi.CylindricalGrid(
    number_of_cells=[NR, NZ],
    warpx_max_grid_size=max(NR, NZ),
    lower_bound=[rmin, 0.0],
    upper_bound=[Lr, Lz],
    lower_boundary_conditions=["none" if rmin == 0.0 else "dirichlet", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=[
        "none" if rmin == 0.0 else "reflecting",
        "periodic",
    ],
    upper_boundary_conditions_particles=["reflecting", "periodic"],
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
    Bx_expression="0.0",
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
    warpx_current_deposition_algo="direct",
    warpx_use_filter=False,
)
sim.add_species(
    ions,
    layout=picmi.GriddedLayout(n_macroparticle_per_cell=[2, 2, 2], grid=grid),
)
sim.add_applied_field(B_field)

sim.initialize_inputs()

# Conduction knobs are not picmi kwargs: write the bucket after
# initialize_inputs (verify in warpx_used_inputs).
import pywarpx  # noqa: E402

pywarpx.hybridpicmodel.qdsmc_time_advance = args.advance
pywarpx.hybridpicmodel.qdsmc_gradient_deposit = args.grad_deposit
pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_par(n,Te,t)", kappa_expr)
pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_perp(n,Te,t)", kappa_expr)
pywarpx.hybridpicmodel.qdsmc_conduction_flux_limit_factor = args.flux_limit
pywarpx.hybridpicmodel.qdsmc_conduction_operator = "fd"
pywarpx.hybridpicmodel.qdsmc_conduction_fd_order = args.fd_order
pywarpx.hybridpicmodel.qdsmc_conduction_fd_limiter = args.fd_limiter
pywarpx.hybridpicmodel.qdsmc_conduction_fd_cfl = args.fd_cfl
pywarpx.hybridpicmodel.qdsmc_conduction_fd_time = args.fd_time

sim.initialize_warpx()

# ----------------------------------------------------------------------
# Te poke (step 0, after the closure mirror) + initial capture
# ----------------------------------------------------------------------
Te_wrap = fields.MultiFabWrapper(mf_name="hybrid_electron_temperature_fp", level=0)
rho_wrap = fields.MultiFabWrapper(mf_name="rho_fp", level=0)

K_per_eV = constants.q_e / constants.kb
state = {"initial": None, "sum0": None}


def _node_coords():
    te = Te_wrap[:, :]
    nr_pts, nz_pts = te.shape
    r = np.linspace(rmin, Lr, nr_pts)
    z = np.linspace(0.0, Lz, nz_pts)
    return np.meshgrid(r, z, indexing="ij")


def _vr_weights(rg):
    """Dual-cell radial measure of the FD operator: Vr = r dr for
    interior AND wall nodes (full-dx wall convention), dr^2/8 for the
    axis node."""
    dr = (Lr - rmin) / NR
    w = rg * dr
    if rmin == 0.0:
        w[rg == 0.0] = dr * dr / 8.0
    return w


def profile(rg, zg, sig):
    """Fractional-amplitude blob at spread sig.

    radial/annulus: Gaussian in r about rb (radial: 2D cylindrical
    amplitude decay s0^2/sig^2; annulus: initial shape only). axial:
    1D Gaussian in z about zb, image-summed over the periodic images.
    """
    if args.mode == "axial":
        out = np.zeros_like(zg)
        amp = blob_amp * s0 / sig
        for mz in range(-3, 4):
            dz = zg - zb + mz * Lz
            out += amp * np.exp(-(dz**2) / (2.0 * sig**2))
        return out
    if args.mode == "radial":
        amp = blob_amp * s0**2 / sig**2
        return amp * np.exp(-((rg - rb) ** 2) / (2.0 * sig**2))
    return blob_amp * np.exp(-((rg - rb) ** 2) / (2.0 * sig**2))


def poke_te():
    if libwarpx_istep() != 0:
        return
    rg, zg = _node_coords()
    te = Te0_eV * K_per_eV * (1.0 + profile(rg, zg, s0))
    Te_wrap[:, :] = te
    state["initial"] = te.copy()
    # the operator's conserved quantity is Sigma ne Te Vr (u = 1.5 ne kB
    # Te in flux form); the RZ-deposited ne is not bit-uniform, so weigh
    # by it rather than assuming n0
    ne = rho_wrap[:, :] / constants.q_e
    state["sum0"] = (_vr_weights(rg)[:, :-1] * ne[:, :-1] * te[:, :-1]).sum()
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
rg, zg = _node_coords()
st = np.sqrt(s0**2 + 2.0 * chi0 * tfinal)
te_exact = Te0_eV * K_per_eV * (1.0 + profile(rg, zg, st))

# unique nodes only (drop the periodic z seam); r keeps all nodes
diff = (te_final - te_exact)[:, :-1]
rel_l2 = np.sqrt((diff**2).mean()) / (Te0_eV * K_per_eV)

# conserved sum (the operator's own discrete measure, ne-weighted)
w_r = _vr_weights(rg)[:, :-1]
ne_final = rho_wrap[:, :] / constants.q_e
sum1 = (w_r * ne_final[:, :-1] * te_final[:, :-1]).sum()
cons_drift = (sum1 - state["sum0"]) / state["sum0"]

# spread measurement of the perturbation
w = (te_final[:, :-1] - Te0_eV * K_per_eV).clip(min=0.0)
ru, zu = rg[:, :-1], zg[:, :-1]
if args.mode == "axial":
    # 1D z second moment, volume-weighted (dz uniform; weight by r measure)
    wz = w * _vr_weights(rg)[:, :-1]
    dz_c = zu - zb
    wsum = wz.sum()
    sig_sq = (wz * dz_c**2).sum() / wsum - ((wz * dz_c).sum() / wsum) ** 2
    chi_meas = (sig_sq - s0**2) / (2.0 * tfinal)
    # metric-leak indicator: per-r-row measured variance spread
    row_sig = np.array(
        [
            (wz[i] * dz_c[i] ** 2).sum() / wz[i].sum()
            - ((wz[i] * dz_c[i]).sum() / wz[i].sum()) ** 2
            for i in range(wz.shape[0])
        ]
    )
    row_spread = (row_sig.max() - row_sig.min()) / sig_sq
else:
    # cylindrical radial second moment: <r^2> = 2 sig^2 for the
    # axis-centered Gaussian (area measure r dr dz)
    wa = w * ru
    wsum = wa.sum()
    if args.mode == "radial":
        sig_sq = 0.5 * (wa * ru**2).sum() / wsum
    else:
        dr_c = ru - rb
        sig_sq = (wa * dr_c**2).sum() / wsum - ((wa * dr_c).sum() / wsum) ** 2
    chi_meas = (sig_sq - s0**2) / (2.0 * tfinal)
    row_spread = 0.0

np.savez_compressed(
    args.out,
    mode=args.mode,
    advance=args.advance,
    nr=NR,
    nz=NZ,
    rmin=rmin,
    nsteps=args.nsteps,
    dt=dt,
    tfinal=tfinal,
    L=L,
    chi0=chi0,
    fd_order=args.fd_order,
    fd_time=args.fd_time,
    fd_cfl=args.fd_cfl,
    Te0_eV=Te0_eV,
    blob=[rb, zb, s0, blob_amp],
    te_initial=state["initial"],
    te_final=te_final,
    te_exact=te_exact,
    rel_l2=rel_l2,
    sig_sq=sig_sq,
    chi_meas=chi_meas,
    cons_drift=cons_drift,
    row_spread=row_spread,
)
print(
    f"[harness] done: mode={args.mode} NR={NR} NZ={NZ} steps={args.nsteps} "
    f"fd_order={args.fd_order} relL2={rel_l2:.4e} "
    f"chi_meas/chi0={chi_meas / chi0:.4f} cons_drift={cons_drift:.3e} "
    f"row_spread={row_spread:.3e}"
)
sys.stdout.flush()
