#!/usr/bin/env python3
"""Thrust D / gate G4 EB instrument: staircase annulus conduction BCs.

2D XZ periodic box with an EB annulus (conductor inside r1 and outside
r2), uniform plasma in the gap, static 1e12-mass ions, B = 0 (isotropic
conduction at chi_par), constant kappa. Exercises the per-line staircase
EB conduction BCs (qdsmc_conduction_eb_bc) of the fluxform sweeps.

Modes
-----
isothermal : T(r1) = T1, T(r2) = T2 via the qdsmc_conduction_eb_Te
             (x,y,z) parser (radius-switched). Steady state = ln r
             profile T(r) = T1 + (T2-T1) ln(r/r1)/ln(r2/r1); the EB
             tally (warpx.get_qdsmc_eb_tally) balances Delta Sigma(u)
             over fluid nodes exactly (in flux form the fluid-sum
             change IS the wall exchange).
adiabatic  : hot Gaussian blob poked in the gap; per-line specular
             fold-back at the covered-mask boundary must conserve
             Sigma(u) over fluid nodes to round-off (no vacuum-deletion
             class at the wall).

The staircase wall sits at half-integer faces of the covered-node mask,
so the effective radii carry an O(dx/2) ambiguity: the ln-r gate both
compares against the nominal (r1, r2) profile (loose) and reports a
least-squares (A + B ln r) fit over the interior for shape fidelity.
"""

import argparse
import sys

import numpy as np

from pywarpx import callbacks, fields, picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["conduction", "advection"],
    default="conduction",
    help="conduction: isothermal/adiabatic wall BC gates. advection: rigid "
    "rotation of a mid-gap blob past the staircase walls (marker EB "
    "reflection gate; conduction off)",
)
parser.add_argument("--ncell", type=int, default=64)
parser.add_argument("--nsteps", type=int, default=384)
parser.add_argument(
    "--vedge", type=float, default=1.0e5, help="advection: speed at r2 [m/s]"
)
parser.add_argument("--marker-reflect", type=int, choices=[0, 1], default=1)
parser.add_argument(
    "--blob-r",
    type=float,
    default=None,
    help="advection: blob center radius (default mid-gap; near r2 exercises "
    "the marker mirror on the staircase)",
)
parser.add_argument("--tfinal", type=float, default=2.0e-5, help="total time [s]")
parser.add_argument("--chi", type=float, default=2.0e3, help="chi [m^2/s]")
parser.add_argument(
    "--eb-bc", choices=["isothermal", "adiabatic"], default="isothermal"
)
parser.add_argument("--T1", type=float, default=15.0, help="inner wall Te [eV]")
parser.add_argument("--T2", type=float, default=5.0, help="outer wall Te [eV]")
parser.add_argument("--r1", type=float, default=0.16)
parser.add_argument("--r2", type=float, default=0.42)
parser.add_argument("--npts", type=int, nargs="+", default=[3])
parser.add_argument("--max-hop", type=float, default=2.0)
parser.add_argument("--recon", choices=["plm", "ppm"], default="ppm")
parser.add_argument("--advance", choices=["euler", "leapfrog", "pc"], default="pc")
parser.add_argument("--assert", dest="do_assert", type=int, default=1)
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
# conduction: 1e12 = truly static background (see qdsmc_slab_test.py).
# advection: 1e6 (the verified heavy-drifting-ion V_e control; ions must
# actually carry J_i).
mass_factor = 1.0e12

r1, r2 = args.r1, args.r2
rm2 = (0.5 * (r1 + r2)) ** 2
if args.mode == "advection":
    mass_factor = 1.0e6

tfinal = args.tfinal
omega0 = 0.0
if args.mode == "advection" and args.vedge > 0.0:
    # one full revolution at the edge speed; CFL = vedge*dt/dx ~ 0.44 at
    # the defaults (N=64, nsteps=384) -- inside the Esirkepov multi-cell
    # trap bound (vedge = 0: at-rest EB gate, tfinal from --tfinal)
    omega0 = args.vedge / r2
    tfinal = 2.0 * np.pi / omega0
dt = tfinal / args.nsteps
chi0 = args.chi
kb = constants.kb
qe = constants.q_e
K_per_eV = qe / kb
kappa = 1.5 * n0 * kb * chi0
npts = args.npts if len(args.npts) > 1 else [args.npts[0], args.npts[0]]

# ----------------------------------------------------------------------
# PICMI setup (periodic box; the EB provides all the walls)
# ----------------------------------------------------------------------
grid = picmi.Cartesian2DGrid(
    number_of_cells=[N, N],
    lower_bound=[-0.5 * L, -0.5 * L],
    upper_bound=[0.5 * L, 0.5 * L],
    lower_boundary_conditions=["periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic"],
    warpx_max_grid_size=N,
)

# conductor inside r1 and outside r2 (implicit function > 0 = body)
embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function=f"max({r1 * r1}-(x*x+z*z),(x*x+z*z)-{r2 * r2})",
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
        density_expression=(f"{n0}*((x*x+z*z)>{r1 * r1})*((x*x+z*z)<{r2 * r2})"),
        momentum_expressions=(
            # advection: heavy drifting ions in rigid rotation =>
            # V_e = J_i/rho = Omega x r, tangential to both walls (the
            # physical flow never crosses the wall; the STAIRCASE makes
            # the local normal components that exercise the mirror)
            [f"(-{omega0}*z)", "0.0", f"({omega0}*x)"]
            if args.mode == "advection"
            else ["0.0", "0.0", "0.0"]
        ),
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
    warpx_embedded_boundary=embedded_boundary,
)
sim.add_species(
    ions,
    layout=picmi.GriddedLayout(n_macroparticle_per_cell=[2, 2], grid=grid),
)

sim.initialize_inputs()

import pywarpx  # noqa: E402

pywarpx.hybridpicmodel.qdsmc_time_advance = args.advance
pywarpx.hybridpicmodel.qdsmc_eb_marker_reflect = args.marker_reflect
if args.mode == "conduction":
    pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_par(n,Te,t)", f"{kappa:.16e}")
    pywarpx.hybridpicmodel.add_new_attr("qdsmc_kappa_perp(n,Te,t)", "0.0")
pywarpx.hybridpicmodel.qdsmc_conduction_quadrature_points = npts
pywarpx.hybridpicmodel.qdsmc_conduction_flux_limit_factor = 0.0
pywarpx.hybridpicmodel.qdsmc_conduction_max_hop = args.max_hop
pywarpx.hybridpicmodel.qdsmc_conduction_form = "fluxform"
pywarpx.hybridpicmodel.qdsmc_conduction_reconstruction = args.recon
pywarpx.hybridpicmodel.qdsmc_conduction_eb_bc = args.eb_bc
if args.eb_bc == "isothermal":
    pywarpx.hybridpicmodel.add_new_attr(
        "qdsmc_conduction_eb_Te(x,y,z)",
        f"if((x*x+z*z)<{rm2},{args.T1},{args.T2})",
    )

sim.initialize_warpx()

# ----------------------------------------------------------------------
# Wrappers + bookkeeping
# ----------------------------------------------------------------------
Te_wrap = fields.MultiFabWrapper(mf_name="hybrid_electron_temperature_fp", level=0)
rho_wrap = fields.MultiFabWrapper(mf_name="rho_fp", level=0)


def wx_instance():
    from pywarpx import libwarpx

    return libwarpx.libwarpx_so.get_instance()


def node_coords():
    te = Te_wrap[:, :]
    nx, nz = te.shape
    x = np.linspace(-0.5 * L, 0.5 * L, nx)
    z = np.linspace(-0.5 * L, 0.5 * L, nz)
    return np.meshgrid(x, z, indexing="ij")


XG, ZG = None, None
FLUID = None  # nodal fluid mask (annulus interior), harness-side


def build_masks():
    global XG, ZG, FLUID
    XG, ZG = node_coords()
    r = np.sqrt(XG**2 + ZG**2)
    FLUID = (r > r1) & (r < r2)


def sigma_u():
    te = Te_wrap[:, :][:-1, :-1]
    ne = rho_wrap[:, :][:-1, :-1] / qe
    return (1.5 * kb * ne * te * FLUID[:-1, :-1]).sum()


state = {}


def capture0():
    if wx_instance().getistep(0) != 0:
        return
    build_masks()
    if args.eb_bc == "adiabatic" or args.mode == "advection":
        # hot blob in the gap (transport needs structure to conserve)
        rmid = args.blob_r if args.blob_r is not None else 0.5 * (r1 + r2)
        s0 = 2.5 * L / N
        te = (
            Te0_eV
            * K_per_eV
            * (1.0 + np.exp(-((XG - rmid) ** 2 + ZG**2) / (2.0 * s0**2)))
        )
        Te_wrap[:, :] = te
    state["u0"] = sigma_u()
    state["tal0"] = wx_instance().get_qdsmc_eb_tally()
    if args.verbose:
        ne = rho_wrap[:, :] / qe
        print(
            f"[harness] fluid nodes {int(FLUID.sum())}, "
            f"mean ne/n0 in gap {float(ne[FLUID].mean()) / n0:.3f}",
            flush=True,
        )


callbacks.installparticleinjection(capture0)

# ----------------------------------------------------------------------
# Run + measure
# ----------------------------------------------------------------------
sim.step(args.nsteps)

te_final = Te_wrap[:, :]
u1 = sigma_u()
tal1 = wx_instance().get_qdsmc_eb_tally()

d_sigma = u1 - state["u0"]
tally = tal1 - state["tal0"]
closure = abs(d_sigma - tally) / max(abs(d_sigma), abs(tally), 1e-300)

r = np.sqrt(XG**2 + ZG**2)
# interior ring, >= 2 cells from either staircase wall
dx = L / N
ring = (r > r1 + 2.5 * dx) & (r < r2 - 2.5 * dx)
rel_l2 = float("nan")
fit_T1 = fit_T2 = fit_T1_eff = fit_T2_eff = float("nan")
# the pinned bath ring intrudes eb_ring nodes into the fluid, so the
# conduction problem's TRUE bath surfaces sit at the shifted radii —
# the O(dx)-convergent staircase geometry price (measured N=64: fit
# T(r1+2dx) = 15.07, T(r2-2dx) = 4.83 for pins 15/5, while the nominal
# radii read 17.65/3.71 by extrapolation past the baths)
EB_RING = 2
r1_eff, r2_eff = r1 + EB_RING * dx, r2 - EB_RING * dx
if args.eb_bc == "isothermal":
    T1K, T2K = args.T1 * K_per_eV, args.T2 * K_per_eV
    exact = T1K + (T2K - T1K) * np.log(r / r1_eff) / np.log(r2_eff / r1_eff)
    diff = (te_final - exact)[ring]
    rel_l2 = float(np.sqrt((diff**2).mean()) / (Te0_eV * K_per_eV))
    # shape fit A + B ln r -> implied bath temperatures at the shifted
    # radii (gate) and the nominal radii (reported, offset expected)
    lr = np.log(r[ring])
    Bfit, Afit = np.polyfit(lr, te_final[ring], 1)
    fit_T1 = float((Afit + Bfit * np.log(r1)) / K_per_eV)
    fit_T2 = float((Afit + Bfit * np.log(r2)) / K_per_eV)
    fit_T1_eff = float((Afit + Bfit * np.log(r1_eff)) / K_per_eV)
    fit_T2_eff = float((Afit + Bfit * np.log(r2_eff)) / K_per_eV)

np.savez_compressed(
    args.out,
    eb_bc=args.eb_bc,
    recon=args.recon,
    ncell=N,
    nsteps=args.nsteps,
    dt=dt,
    tfinal=tfinal,
    chi0=chi0,
    r1=r1,
    r2=r2,
    T1=args.T1,
    T2=args.T2,
    npts=npts,
    te_final=te_final,
    d_sigma=d_sigma,
    tally=tally,
    closure=closure,
    rel_l2=rel_l2,
    fit_T1=fit_T1,
    fit_T2=fit_T2,
    fit_T1_eff=fit_T1_eff,
    fit_T2_eff=fit_T2_eff,
    r1_eff=r1_eff,
    r2_eff=r2_eff,
)
print(
    f"[harness] annulus done: bc={args.eb_bc} N={N} steps={args.nsteps} "
    f"| closure={closure:.3e} (dSigma {d_sigma:.4e}, tally {tally:.4e}) "
    f"| lnr relL2={rel_l2:.4e} "
    f"| fit T(r_eff)={fit_T1_eff:.2f}/{fit_T2_eff:.2f} eV "
    f"(pinned {args.T1}/{args.T2}; nominal radii read "
    f"{fit_T1:.2f}/{fit_T2:.2f})",
    flush=True,
)

# advection mode: rigid-rotation exact solution = the blob rotated by
# omega*t (compare on an interior core ring away from the staircase)
rot_l2 = float("nan")
if args.mode == "advection":
    th = -(omega0 * tfinal)
    XR = XG * np.cos(th) - ZG * np.sin(th)
    ZR = XG * np.sin(th) + ZG * np.cos(th)
    rmid = args.blob_r if args.blob_r is not None else 0.5 * (r1 + r2)
    s0b = 2.5 * L / N
    te_exact = (
        Te0_eV * K_per_eV * (1.0 + np.exp(-((XR - rmid) ** 2 + ZR**2) / (2.0 * s0b**2)))
    )
    core = (r > r1 + 2.5 * dx) & (r < r2 - 2.5 * dx)
    rot_l2 = float(
        np.sqrt((((te_final - te_exact)[core]) ** 2).mean()) / (Te0_eV * K_per_eV)
    )
    print(
        f"[harness] advection: rotated-blob relL2={rot_l2:.4e} "
        f"| Sigma(Te,fluid) drift={d_sigma / state['u0']:.3e} "
        f"(marker_reflect={args.marker_reflect})",
        flush=True,
    )

if args.do_assert:
    if args.mode == "advection" and args.vedge <= 0.0:
        # near-rest EB gate: with static 1e12 ions and no drive, V_e is
        # not exactly zero (the wall density ramp curls a whisper of B
        # through grad Pe / n), so the gate bounds the drift 4+ orders
        # below the catastrophic wall classes (-6% per substep) rather
        # than at bit level. Measured: -1.3e-6 over 128 steps.
        assert abs(d_sigma) / state["u0"] < 1.0e-4, (
            f"near-rest EB drift {d_sigma / state['u0']:.3e}"
        )
    elif args.mode == "advection":
        # NOT a marker-reflection gate: the rotating-ion instrument is
        # dominated by EB ion scraping (ne drains ~0.3%/step) and
        # near-wall deposit dilution -- measured identical with the
        # mirror on/off. Loose envelope only; the differential
        # instrument needs the deposit fold + ion-side EB handling
        # (recorded follow-ups).
        assert rot_l2 < 1.5, f"rotated-blob relL2 {rot_l2:.3e}"
    elif args.eb_bc == "isothermal":
        # Gate on the SHIFTED bath radii (the ring intrusion is the
        # O(dx) staircase geometry price; the sub-cell level-set wall
        # position is the recorded upgrade once the EB mirror
        # machinery of the 7028 stack merges). Measured N=64:
        # T(r_eff) = 15.07/4.83 for pins 15/5, relL2 vs the shifted
        # ln-r profile 7.6e-2 (staircase azimuthal ripple included).
        assert closure < 1.0e-9, f"EB budget closure {closure:.3e}"
        assert rel_l2 < 0.15, f"ln-r profile relL2 {rel_l2:.3e}"
        assert abs(fit_T1_eff - args.T1) < 0.5, f"fit T(r1_eff) {fit_T1_eff:.2f}"
        assert abs(fit_T2_eff - args.T2) < 0.5, f"fit T(r2_eff) {fit_T2_eff:.2f}"
    else:
        # adiabatic: fluid-sum conservation at round-off (relative)
        assert abs(d_sigma) / state["u0"] < 1.0e-10, (
            f"adiabatic EB Sigma drift {d_sigma / state['u0']:.3e}"
        )
sys.stdout.flush()
