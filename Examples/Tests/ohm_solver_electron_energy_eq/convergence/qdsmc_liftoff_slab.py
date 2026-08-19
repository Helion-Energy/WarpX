#!/usr/bin/env python3
#
# --- Liftoff-class slab for the QDSMC electron energy equation + the
# --- insulating EB wall (the third acceptance item of the InsulatingEB
# --- scope, and the production stack of the Ohmic-hotspot problem).
# ---
# --- Ported from the ECT-campaign theta-pinch liftoff testbed
# --- (ohm_solver_plasma_cylinder_liftoff, eb_ect_yee_followup branch):
# --- a thin 3D slab (periodic in z) of an annular deuterium column inside
# --- a conducting cylindrical wall, threaded by a bias field that a
# --- Hermite-ramped external vector potential reverses -- the column lifts
# --- off the wall and implodes. ECT/conformal knobs are stripped (this
# --- branch's wall is the staircase insulating EB); the Python
# --- install_wall_scraper standoff hook of the original is REPLACED by
# --- boundary.eb_type = insulating (the C++ wall it prototyped).
# ---
# --- On top of the original ion dynamics this deck runs the electron
# --- energy equation with the production sources:
# ---   * Joule heating (eta J^2, the density-scaled power-law resistivity
# ---     of the formation runs),
# ---   * Spitzer parallel conduction kappa_par = C * Te^2.5 (+ optional
# ---     perpendicular fraction), the piece that drains the Ohmic hotspot
# ---     (without it the source cannot diffuse away and pressure grows
# ---     locally without bound -- the production blocker this validates),
# ---   * the insulating wall: standoff collection (tallied) + zero-normal-
# ---     gradient Te fill + the spill-only deposit fold.
# ---
# --- Probes: max/median Te history (hotspot growth vs conduction), the
# --- open-set electron energy, the collected-charge/energy tallies, and
# --- the usual field diagnostics.

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI as mpi

import pywarpx
from pywarpx import callbacks, fields, picmi

constants = picmi.constants
comm = mpi.COMM_WORLD

# ----------------------------------------------------------------------------
# Parameters (formation-section midplane slab, as in the ECT testbed)
# ----------------------------------------------------------------------------
M_AMU = 2  # deuterium
N_I = 1.5e20  # reference ion density (m^-3)
T_I0 = 5.0  # ion temperature (eV)
T_E0 = 5.0  # electron temperature (eV)

R_INNER = 0.65
R_OUTER = 0.75
R_PART = 0.385
ANNULUS_SMOOTH_CELLS = 2.0
R_WALL = 0.8

BZ_BIAS = -0.01  # T
BZ_REV = 0.5  # T
TAU_RAMP = 10.0e-6  # s

ETA_PLASMA = 1.0e-6  # Ohm m
ETA_VAC_FRAC = 5.0e-2
ETA_POWER = 3.0
N_TRANSITION_FRAC = 0.4

F_T_CI = 0.01
NZ = 8

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", "-r", type=int, default=64)
parser.add_argument("--nppc", type=int, default=12)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--diag-period", type=int, default=100)
parser.add_argument(
    "--standoff",
    type=float,
    default=3.0,
    help="insulating-wall standoff [cells]; 3 = the collocated liftoff "
    "survival point of the explicit-stiffness study",
)
parser.add_argument("--n-floor-frac", type=float, default=0.05)
parser.add_argument(
    "--load-margin",
    type=float,
    default=1.5,
    help="load clearance beyond the collection shell [cells]: the annulus "
    "outer radius is capped at R_WALL - (standoff+margin)*dx. Loading "
    "into the shell amputates the annulus at step 1 and the resulting "
    "grad-Pe cliff field drives runaway edge erosion into the collector "
    "(measured: ~0.3 C/step, 90%% inventory loss in 20 steps)",
)
parser.add_argument("--fill-frac", type=float, default=0.2, help="n_fill/N_I")
parser.add_argument(
    "--kappa",
    choices=["spitzer", "off"],
    default="spitzer",
    help="electron conduction: Spitzer kappa_par = C*Te^2.5 (lnL=10) with "
    "kappa_perp = perp-frac * kappa_par, or off (the Ohmic-hotspot "
    "control arm)",
)
parser.add_argument(
    "--kappa-perp-frac",
    type=float,
    default=1.0e-2,
    help="kappa_perp / kappa_par (cross-field drain at nulls/current sheets)",
)
parser.add_argument(
    "--kappa-mult",
    type=float,
    default=1.0,
    help="multiplier on the Spitzer conduction coefficient (par and perp)",
)
parser.add_argument(
    "--joule-redirect-te",
    type=float,
    default=-1.0,
    help="Te threshold [eV] above which the eta*J^2 Joule source deposits on "
    "ions instead of electrons (hybrid_pic_model.joule_redirect_Te_threshold); "
    "<0 = off",
)
parser.add_argument(
    "--resistivity",
    choices=["powerlaw", "chodura"],
    default="powerlaw",
    help="E-solve resistivity: density-scaled power law (formation-run form), "
    "or Chodura current-driven anomalous form capped at eta_vac",
)
parser.add_argument(
    "--chodura-fc",
    type=float,
    default=0.1,
    help="Chodura anomalous-collisionality fraction f_C of omega_pi",
)
parser.add_argument(
    "--chodura-vcrit",
    type=float,
    default=0.0,
    help="Chodura drift-speed threshold [m/s]; 0 = vA(BZ_REV, N_I)",
)
parser.add_argument(
    "--tau-ramp",
    type=float,
    default=TAU_RAMP,
    help="reversal-drive Hermite ramp time [s] (bias -0.01 T -> +0.5 T); "
    "peak dB/dt = 1.5*dB/tau at tau/2",
)
parser.add_argument(
    "--drive-scale",
    type=float,
    default=1.0,
    help="scale BOTH the bias field and the reversal peak by this factor "
    "(gentler drive). dt follows via w_ci(BZ_REV), so omega_ci*dt stays "
    "fixed (0.5 -> dt doubles); vA, eta_vac, eta_hyper and the "
    "pressure-balance equilibrium seed all rescale coherently",
)
parser.add_argument(
    "--filter-passes",
    type=int,
    default=0,
    help="binomial smoothing passes per direction on deposited rho/J "
    "(warpx.use_filter + filter_npass_each_dir); 0 = off (the non-PSATD "
    "WarpX default). J_plasma, the Ohm E-solve, and the Joule source all "
    "consume the filtered deposits",
)
parser.add_argument(
    "--filter-comp",
    type=int,
    choices=[0, 1],
    default=1,
    help="apply the filter compensation pass (warpx.use_filter_compensation)",
)
parser.add_argument(
    "--relax-rate",
    choices=["off", "spitzer"],
    default="off",
    help="electron-ion relaxation channel nu_ei(rho,Te): off, or Spitzer "
    "(NRL nu_eps/2, lnL=10, Z=1, mu=M_AMU). Supplies the OU drag toward "
    "u_e that pairs with the Joule-redirect ion kicks, plus the global "
    "Q_ei Te<->Ti coupling",
)
parser.add_argument(
    "--joule-heating-eta",
    choices=["esolve", "plasma", "spitzer"],
    default="esolve",
    help="resistivity used by the Joule/redirect HEATING source "
    "(hybrid_pic_model.joule_heating_resistivity(rho,J,Te,t)): esolve = the "
    "E-solve eta incl. the numerical eta_vac ramp (legacy), plasma = constant "
    "ETA_PLASMA, spitzer = Spitzer eta_par (lnL=10, Z=1) from the local Te, "
    "capped at eta_vac. The E-solve resistivity is unchanged either way",
)
parser.add_argument(
    "--joule-heating-n-min-frac",
    type=float,
    default=-1.0,
    help="independent Joule-heating density gate as a fraction of N_I "
    "(hybrid_pic_model.joule_heating_n_min); <0 = off (gate stays at n_floor)",
)
parser.add_argument(
    "--redirect-n-min-factor",
    type=float,
    default=0.0,
    help="redirect density gate factor g: redirected energy only staged at "
    "n >= g*n_floor (hybrid_pic_model.joule_redirect_n_min_factor); 0 = off",
)
parser.add_argument(
    "--redirect-kick-cap",
    type=float,
    default=-1.0,
    help="per-particle redirect kick cap f: sigma_redir <= f*v_th,i "
    "(hybrid_pic_model.joule_redirect_kick_cap_vth_frac); <0 = off",
)
parser.add_argument(
    "--transport-op",
    choices=["markers", "grid"],
    default="markers",
    help="energy-equation transport operator "
    "(hybrid_pic_model.qdsmc_transport_operator): grid + --conduction-op fd "
    "= fully grid-based Te path (no marker machinery)",
)
parser.add_argument(
    "--conduction-op",
    choices=["sde", "fd"],
    default="sde",
    help="conduction operator (hybrid_pic_model.qdsmc_conduction_operator): "
    "fd = grid FD operator with SMART-limited cross fluxes (max principle "
    "-- cannot mint new Te extrema at the b-hat-noise null); sde = QDSMC "
    "production default",
)
parser.add_argument(
    "--fd-order",
    type=int,
    choices=[2, 4],
    default=2,
    help="FD operator spatial order (hybrid_pic_model.qdsmc_conduction_fd_order)",
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
    "--fd-limiter",
    choices=["none", "upwind1", "smart"],
    default="smart",
    help="FD cross-flux limiter (hybrid_pic_model.qdsmc_conduction_fd_limiter)",
)
parser.add_argument(
    "--iso-conduction",
    type=int,
    choices=[0, 1],
    default=0,
    help="single-temperature-style ISOTROPIC conduction at the cross-field "
    "rate everywhere (hybrid_pic_model.qdsmc_conduction_isotropic): "
    "sidesteps the b-hat-undefined problem at the reconnection null",
)
parser.add_argument(
    "--iso-b",
    type=float,
    default=-1.0,
    help="surgical isotropic blend below this |B| [T] "
    "(hybrid_pic_model.qdsmc_conduction_iso_B): chi_par -> chi_perp where "
    "b-hat is noise (the null region); <0 = off. Suggested ~4x the bias",
)
parser.add_argument(
    "--wall-te",
    type=float,
    default=-1.0,
    help="dielectric-wall temperature Dirichlet BC for the conduction "
    "energy equation [eV] (hybrid_pic_model.qdsmc_conduction_eb_bc = "
    "isothermal + qdsmc_conduction_eb_Te): the wall-adjacent ring is held "
    "at this temperature, so conduction drains into the dielectric -- a "
    "cooling channel, exchange tallied (wall_bath in the dropped print). "
    "<0 = adiabatic zero-wall-heat-flux (the default). Room temp = 0.025",
)
parser.add_argument(
    "--wall-ring",
    type=int,
    default=2,
    choices=[1, 2, 3],
    help="isothermal wall-bath ring depth in cells "
    "(hybrid_pic_model.qdsmc_conduction_eb_ring; depth 2 puts the "
    "unresolved wall density ramp inside the bath)",
)
parser.add_argument(
    "--wall-bc",
    choices=["isothermal", "drain"],
    default="isothermal",
    help="wall thermal mode when --wall-te is set: isothermal = two-sided "
    "ring pin (legacy); drain = one-sided free-streaming-limited "
    "temperature drain (MHD wall_thermal_bc port -- never heats plasma)",
)
parser.add_argument(
    "--te-shunt",
    type=float,
    default=-1.0,
    help="general Te limiter with ion shunt [eV] "
    "(hybrid_pic_model.Te_shunt_threshold): any-channel excess above the "
    "cap goes to the ions as stochastic kicks (kick cap / density gate / "
    "relaxation guard apply); <0 = off",
)
parser.add_argument(
    "--te-abort",
    type=float,
    default=-1.0,
    help="graceful Te-runaway abort ceiling [eV] "
    "(hybrid_pic_model.Te_abort_threshold); <0 = off. Replaces the external "
    "te_watchdog.sh kill for arms that should stop themselves",
)
parser.add_argument(
    "--contam-boundary-frac",
    type=float,
    default=-1.0,
    help="quarantine-contamination tally class boundary as a fraction of N_I "
    "(hybrid_pic_model.qdsmc_contamination_n_boundary = frac*N_I); set above "
    "--n-floor-frac to watch the eta_vac ignition band; <0 = off",
)
parser.add_argument(
    "--grad-deposit",
    type=int,
    choices=[0, 1],
    default=1,
    help="half-gradient-corrected (antidiffusive B1) QDSMC deposit "
    "(hybrid_pic_model.qdsmc_gradient_deposit; the branch default). 0 = "
    "plain hat deposit -- the discriminating control for the up-cliff "
    "remap-residual energy injection (Q0d budget: resid_band +5.6 J vs "
    "comp_band -0.13 J at ignition)",
)
parser.add_argument(
    "--chi-vac",
    type=float,
    default=0.0,
    help="halo/band conductivity boost [m^2/s]: isotropic chi -> chi_vac in "
    "a smooth exp(-(n/4n_floor)^2) window around the density floor "
    "(MHD-halo-port experiment). 0 = off (parsers unchanged)",
)
parser.add_argument(
    "--pedestal-frac",
    type=float,
    default=0.0,
    help="MHD-shaped conduction pedestal fraction "
    "(hybrid_pic_model.qdsmc_conduction_pedestal_fraction); pair with "
    "--chi-vac and a wall drain for the halo exhaust chain. 0 = off",
)
parser.add_argument(
    "--eta-vac-d",
    type=float,
    default=0.0,
    help="code-side vacuum resistivity boost D_vac [m^2/s] "
    "(hybrid_pic_model.vacuum_resistivity_diffusivity); the E-solve eta "
    "gains mu0*D_vac*(rho_ref/rho)^2 in quadrature, heating keeps the raw "
    "parser. 0 = off",
)
parser.add_argument(
    "--hyper-vac-d",
    type=float,
    default=0.0,
    help="vacuum hyper-resistivity boost D4 [m^4/s] "
    "(hybrid_pic_model.vacuum_hyper_resistivity_diffusivity): halo damper "
    "for grid-scale field turbulence, biharmonic-CFL-capped. 0 = off",
)
parser.add_argument(
    "--band-drain-rate",
    type=float,
    default=0.0,
    help="band-scoped rectified Te drain rate [1/s] "
    "(hybrid_pic_model.qdsmc_band_drain_rate; MHD halo-relaxation port). "
    "0 = off",
)
parser.add_argument(
    "--band-drain-te",
    type=float,
    default=-1.0,
    help="band drain target Te [eV] (hybrid_pic_model.qdsmc_band_drain_Te)",
)
parser.add_argument(
    "--band-drain-nhi",
    type=float,
    default=-1.0,
    help="band window top in units of n_floor "
    "(hybrid_pic_model.qdsmc_band_drain_n_hi_factor); <0 = C++ default 1.25. "
    "The liftoff ignition band is (1,4)x floor -- use ~4 here",
)
parser.add_argument(
    "--ve-midpoint",
    type=int,
    choices=[0, 1],
    default=1,
    help="sub-cell midpoint V_e re-gather for the marker push "
    "(hybrid_pic_model.qdsmc_ve_midpoint_gather; the branch default). 0 = "
    "markers advect with their home-node V_e for the whole step "
    "(piecewise-constant velocity sampling) -- the discriminating control "
    "for sub-cell interpolation of the 1/rho-amplified V_e field at "
    "floor-band cliff edges",
)
parser.add_argument(
    "--cliff-deposit",
    type=int,
    choices=[0, 1],
    default=0,
    help="cliff-limited entropy deposit "
    "(hybrid_pic_model.qdsmc_cliff_limited_deposit): isothermal spill "
    "across unresolved density jumps -- the K-diffusion heat-pump fix",
)
parser.add_argument(
    "--cliff-r1",
    type=float,
    default=-1.0,
    help="cliff-deposit blend start |ln(n_dest/n_home)| "
    "(hybrid_pic_model.qdsmc_cliff_deposit_r1); <0 = C++ default 0.35",
)
parser.add_argument(
    "--cliff-r2",
    type=float,
    default=-1.0,
    help="cliff-deposit fully-isothermal bound "
    "(hybrid_pic_model.qdsmc_cliff_deposit_r2); <0 = C++ default 1.4",
)
parser.add_argument(
    "--energy-budget",
    type=int,
    choices=[0, 1],
    default=0,
    help="per-stage open-set energy budget (hybrid_pic_model."
    "qdsmc_energy_budget): cumulative dU per Strang stage "
    "(advection+compression / conduction / sources), bulk vs band split at "
    "the contamination boundary",
)
parser.add_argument("--substeps", type=int, default=256)
parser.add_argument("--substep-rtol", type=float, default=1.0e-3)
parser.add_argument("--advance", choices=["euler", "leapfrog", "pc"], default="pc")
parser.add_argument("--equilibrium-b", type=int, choices=[0, 1], default=1)
parser.add_argument("--grid-type", default="collocated")
parser.add_argument("--openpmd", action="store_true", default=True)
parser.add_argument("--out", type=str, default="liftoff_slab")
parser.add_argument("--verbose", type=int, default=1)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

# Gentler-drive scaling: everything downstream (w_ci -> dt, vA, eta_vac,
# eta_hyper, the Hermite ramp, the diamagnetic equilibrium seed) derives
# from these two constants, so scaling them here keeps the run coherent.
BZ_BIAS *= args.drive_scale
BZ_REV *= args.drive_scale

m_i = M_AMU * constants.m_p
n_floor = args.n_floor_frac * N_I
n_fill = args.fill_frac * N_I
vth = np.sqrt(constants.q_e * T_I0 / m_i)

resolution = args.resolution
dx = 2.0 / resolution
annulus_w = ANNULUS_SMOOTH_CELLS * dx

# cap the annulus outer radius clear of the collection shell (see
# --load-margin help)
r_outer_eff = min(R_OUTER, R_WALL - (args.standoff + args.load_margin) * dx)
if comm.rank == 0 and r_outer_eff < R_OUTER:
    print(
        f"[liftoff] annulus outer radius capped {R_OUTER} -> "
        f"{r_outer_eff:.4f} m (shell clearance at this resolution/standoff)",
        flush=True,
    )
assert r_outer_eff > R_INNER + annulus_w, (
    "standoff+margin leaves no room for the annulus at this resolution; "
    "reduce --standoff or raise --resolution"
)
lz = NZ * dx
zmin, zmax = 5.0 - lz / 2.0, 5.0 + lz / 2.0

w_ci = constants.q_e * abs(BZ_REV) / m_i
dt = F_T_CI * 2.0 * np.pi / w_ci

w_pi = np.sqrt(constants.q_e**2 * N_I / (constants.ep0 * m_i))
l_i = constants.c / w_pi
vA = abs(BZ_REV) / np.sqrt(constants.mu0 * N_I * m_i)
dL2 = 1.0 / (2.0 / dx**2 + 1.0 / dx**2)
eta_max = constants.mu0 * dL2 / (2.0 * dt)
eta_hyper = constants.mu0 * 0.2 * l_i * vA * dL2

# Spitzer kappa_par = C * Te[eV]^2.5 [W/(m K)], lnLambda = 10:
#   kappa = 3.16 n k_B (q_e Te/m_e) tau_e,  tau_e = 3.44e11 Te^1.5/(n lnL)
# (n cancels).  C(lnL=10) ~ 0.264; kappa(10 eV) ~ 84 W/(m K).
LN_LAMBDA = 10.0
KAPPA_C = 3.16 * constants.kb * (constants.q_e / constants.m_e) * 3.44e11 / LN_LAMBDA

# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
grid = picmi.Cartesian3DGrid(
    number_of_cells=[resolution, resolution, NZ],
    lower_bound=[-1.0, -1.0, zmin],
    upper_bound=[1.0, 1.0, zmax],
    lower_boundary_conditions=["dirichlet", "dirichlet", "periodic"],
    upper_boundary_conditions=["dirichlet", "dirichlet", "periodic"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
    warpx_max_grid_size=2048,
    warpx_max_grid_size_x=max(resolution // 2, 8),
    warpx_blocking_factor=8,
)


def hermite_ramp_expression(b0, b1, tau):
    s = f"min(max(t/{tau:.9e},0),1)"
    return f"({b0:.9e} + ({b1 - b0:.9e})*({s})*({s})*(3-2*({s})))"


A_ext = {
    "uniform_reversal": {
        "Ax_external_function": "-0.5*y",
        "Ay_external_function": "0.5*x",
        "Az_external_function": "0",
        "A_time_external_function": hermite_ramp_expression(
            BZ_BIAS, BZ_REV, args.tau_ramp
        ),
    },
}

# density-scaled power-law resistivity (formation-run form)
a_pl = (args.n_floor_frac / N_TRANSITION_FRAC) ** ETA_POWER
eta_vac = ETA_VAC_FRAC * eta_max
if args.resistivity == "powerlaw":
    res_str = (
        f"eta_plasma + (eta_vac - eta_plasma)"
        f"*max(0.0, (rho_f/max(rho,rho_f))**({ETA_POWER:.6g}) - ({a_pl:.12g}))"
        f"/({1.0 - a_pl:.12g})"
    )
else:
    # Chodura current-driven anomalous resistivity:
    #   eta_C = f_C * m_e nu_an / (n e^2), nu_an = w_pi(rho)*(1 - exp(-v_de/v_c)),
    #   v_de = |J|/rho.  Capped at eta_vac so the RKF45 stability envelope of
    #   the power-law scheme is an upper bound; grid-scale whistler damping in
    #   true vacuum is left to plasma_hyper_resistivity.
    v_crit = args.chodura_vcrit if args.chodura_vcrit > 0.0 else vA
    chodura_pref = (
        args.chodura_fc
        * (constants.m_e / constants.q_e)
        * np.sqrt(constants.q_e / (constants.ep0 * m_i))
    )
    res_str = (
        f"eta_plasma + min(eta_vac - eta_plasma, "
        f"{chodura_pref:.9e}/sqrt(max(rho,rho_f))"
        f"*(1.0 - exp(-J/(max(rho,rho_f)*{v_crit:.9e}))))"
    )

solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=5.0 / 3.0,
    Te=T_E0,
    n0=N_I,
    n_floor=n_floor,
    plasma_resistivity=res_str,
    eta_plasma=ETA_PLASMA,
    eta_vac=eta_vac,
    rho_f=constants.q_e * n_floor,
    plasma_hyper_resistivity=eta_hyper,
    substeps=args.substeps,
    use_rkf45=True,
    substep_rtol=args.substep_rtol,
    substep_atol=1.0e-8,
    max_substep_attempts=1000,
    solve_electron_energy_equation=True,
    include_joule_heating=True,
    A_external=A_ext,
)

embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function="(x**2+y**2-R_w**2)",
    R_w=R_WALL,
    eb_type="insulating",
    eb_standoff_cells=args.standoff,
)

simulation = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=args.steps,
    particle_shape=1,
    verbose=args.verbose,
    warpx_embedded_boundary=embedded_boundary,
    warpx_current_deposition_algo="direct",
    warpx_grid_type=args.grid_type,
)

# equilibrium diamagnetic Bz seed (removes the startup transient; the
# external add supplies the bias part -- see the ECT deck's double-count
# note, upstream #6047)
n_annulus = N_I * R_PART**2 / (r_outer_eff**2 - R_INNER**2)
if args.equilibrium_b:
    k_eq = 2.0 * constants.mu0 * (T_I0 + T_E0) * constants.q_e
    sgn = "" if BZ_BIAS >= 0 else "-"
    rr = "sqrt(x*x+y*y)"
    e_in = f"0.5*(1.0+tanh(({rr}-{R_INNER})/{annulus_w}))"
    e_out = f"0.5*(1.0+tanh(({r_outer_eff}-{rr})/{annulus_w}))"
    n_of_r = f"({n_fill}*(1.0-{e_in})+{n_annulus}*{e_in}*{e_out})"
    bz_seed = f"{sgn}sqrt({BZ_BIAS**2}+{k_eq}*({n_annulus}-{n_of_r}))-({BZ_BIAS})"
else:
    bz_seed = "0"
simulation.add_applied_field(
    picmi.AnalyticInitialField(
        Bx_expression="0",
        By_expression="0",
        Bz_expression=bz_seed,
        warpx_do_initial_div_cleaning=False,
    )
)

# annular column + interior fill, tanh-softened edges. The outer radius is
# capped so the load (and its tanh tail) stays clear of the collection
# shell; the R_PART inventory formula below rescales n_annulus so the
# column carries the same line density regardless of the cap.
r_expr = "sqrt(x*x+y*y)"
edge_in = f"0.5*(1.0+tanh(({r_expr}-R_in)/sw))"
edge_out = f"0.5*(1.0+tanh((R_out-{r_expr})/sw))"
ions = picmi.Species(
    name="ions",
    mass=m_i,
    charge="q_e",
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=f"n_f*(1.0-{edge_in})+n_a*{edge_in}*{edge_out}",
        momentum_expressions=["0", "0", "0"],
        warpx_momentum_spread_expressions=[f"{vth}"] * 3,
        warpx_density_min=0.05 * n_fill,
        n_a=n_annulus,
        n_f=n_fill,
        R_in=R_INNER,
        R_out=r_outer_eff,
        sw=annulus_w,
    ),
)
simulation.add_species(
    ions,
    layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=args.nppc),
)

ion_ion_coulomb = picmi.CoulombCollisions(
    name="ion_ion_Coulomb", species=[ions, ions], CoulombLog=12
)
simulation.collisions = [ion_ion_coulomb]

if comm.rank == 0 and Path("diags").exists():
    shutil.rmtree("diags")
comm.Barrier()

field_diag = picmi.FieldDiagnostic(
    name="field_diag",
    grid=grid,
    period=args.diag_period,
    data_list=["B", "E", "rho", "J", "Te"],
    write_dir="diags",
    warpx_file_prefix="field_diags",
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
)
simulation.add_diagnostic(field_diag)

simulation.initialize_inputs()

if args.filter_passes > 0:
    pywarpx.warpx.use_filter = 1
    pywarpx.warpx.use_filter_compensation = args.filter_comp
    pywarpx.warpx.filter_npass_each_dir = [args.filter_passes] * 3

# branch-side energy-equation knobs (pywarpx bucket, annulus-deck pattern)
pywarpx.hybridpicmodel.qdsmc_time_advance = args.advance
# PRODUCTION conduction form: the code default ("scatter", #6982-compat
# control arm) has no closed floor faces and no EB machinery -- with the
# insulating wall's maintained band Te it acts as a perpetual bath donor
# (measured on this deck: +12.8% open-set energy per 400 steps). The
# fluxform split sweeps with ppm/npts=3 are the gate-passed defaults.
pywarpx.hybridpicmodel.qdsmc_conduction_form = "fluxform"
pywarpx.hybridpicmodel.qdsmc_conduction_reconstruction = "ppm"
pywarpx.hybridpicmodel.qdsmc_conduction_quadrature_points = 3
pywarpx.hybridpicmodel.qdsmc_transport_operator = args.transport_op
pywarpx.hybridpicmodel.qdsmc_conduction_operator = args.conduction_op
pywarpx.hybridpicmodel.qdsmc_conduction_fd_order = args.fd_order
pywarpx.hybridpicmodel.qdsmc_conduction_fd_limiter = args.fd_limiter
pywarpx.hybridpicmodel.qdsmc_conduction_fd_time = args.fd_time
if args.iso_conduction:
    pywarpx.hybridpicmodel.qdsmc_conduction_isotropic = 1
if args.iso_b > 0.0:
    pywarpx.hybridpicmodel.qdsmc_conduction_iso_B = args.iso_b
if args.wall_te > 0.0:
    # Temperature wall: the dielectric absorbs conducted heat (two-sided
    # pin, or the one-sided limited drain that can never heat plasma).
    pywarpx.hybridpicmodel.qdsmc_conduction_eb_bc = args.wall_bc
    pywarpx.hybridpicmodel.__setattr__(
        "qdsmc_conduction_eb_Te(x,y,z)", f"{args.wall_te:.6e}"
    )
    pywarpx.hybridpicmodel.qdsmc_conduction_eb_ring = args.wall_ring
else:
    # Zero wall heat flux (the campaign default).
    pywarpx.hybridpicmodel.qdsmc_conduction_eb_bc = "adiabatic"
if args.kappa == "spitzer":
    kappa_c_eff = args.kappa_mult * KAPPA_C
    pywarpx.hybridpicmodel.__setattr__(
        "qdsmc_kappa_par(n,Te,t)", f"{kappa_c_eff:.6e}*Te**2.5"
    )
    if args.kappa_perp_frac > 0.0:
        pywarpx.hybridpicmodel.__setattr__(
            "qdsmc_kappa_perp(n,Te,t)",
            f"{args.kappa_perp_frac * kappa_c_eff:.6e}*Te**2.5",
        )
    # First-class halo/band conductivity boost (composed into the compiled
    # parsers code-side; window 4 x n_floor by default). 0 = off.
    if args.chi_vac > 0.0:
        pywarpx.hybridpicmodel.qdsmc_conduction_vacuum_chi = args.chi_vac
    if args.pedestal_frac > 0.0:
        pywarpx.hybridpicmodel.qdsmc_conduction_pedestal_fraction = args.pedestal_frac
if args.eta_vac_d > 0.0:
    # Code-side vacuum resistivity boost (eta split): E-solve eta gains the
    # density-keyed vacuum term; Joule heating keeps the raw parser. NOTE:
    # this deck's plasma_resistivity still carries its own legacy ramp --
    # migrating that to this knob is a deliberate deck change, not implied.
    pywarpx.hybridpicmodel.vacuum_resistivity_diffusivity = args.eta_vac_d
if args.hyper_vac_d > 0.0:
    pywarpx.hybridpicmodel.vacuum_hyper_resistivity_diffusivity = args.hyper_vac_d
if args.band_drain_rate > 0.0:
    pywarpx.hybridpicmodel.qdsmc_band_drain_rate = args.band_drain_rate
    pywarpx.hybridpicmodel.qdsmc_band_drain_Te = args.band_drain_te
    if args.band_drain_nhi > 0.0:
        pywarpx.hybridpicmodel.qdsmc_band_drain_n_hi_factor = args.band_drain_nhi
if args.joule_redirect_te >= 0.0:
    pywarpx.hybridpicmodel.joule_redirect_Te_threshold = args.joule_redirect_te
    if args.relax_rate == "off":
        # The guard rail aborts the undamped redirect by default (measured
        # anti-stabilizing, arm SR); keep the no-drag control arm launchable.
        pywarpx.hybridpicmodel.joule_redirect_allow_undamped = 1
if args.relax_rate == "spitzer":
    # NRL equilibration nu_eps = 3.2e-9 Z^2 lnL n_i[cm^-3] / (mu Te[eV]^1.5);
    # the branch relaxes Te -> Ti at rate 2 nu_ei, so nu_ei = nu_eps / 2.
    # rho is the charge density [C/m^3]; Te floor guards the cold-cell pole
    # (the OU update is exact/exp-form, so large nu is stable regardless).
    nu_coef = 3.2e-9 * LN_LAMBDA / (2.0 * M_AMU * 1.0e6 * constants.q_e)
    pywarpx.hybridpicmodel.__setattr__(
        "electron_ion_relaxation_rate(rho,Te,Ti,t)",
        f"{nu_coef:.6e}*rho/max(Te,0.1)**1.5",
    )
if args.joule_heating_eta == "plasma":
    # Physical-eta heating: constant bulk eta. The E-solve keeps the
    # numerical eta_vac ramp; only the heat source changes.
    pywarpx.hybridpicmodel.__setattr__(
        "joule_heating_resistivity(rho,J,Te,t)", f"{ETA_PLASMA:.6e}"
    )
elif args.joule_heating_eta == "spitzer":
    # NRL Spitzer parallel resistivity eta_par = 5.25e-5 Z lnL / Te[eV]^1.5
    # Ohm m; Te floored at 0.1 eV and capped at eta_vac (cold-cell sanity,
    # same cap philosophy as the Chodura arm).
    eta_sp_coef = 5.25e-5 * LN_LAMBDA
    pywarpx.hybridpicmodel.__setattr__(
        "joule_heating_resistivity(rho,J,Te,t)",
        f"min({eta_vac:.6e}, {eta_sp_coef:.6e}/max(Te,0.1)**1.5)",
    )
if args.joule_heating_n_min_frac >= 0.0:
    pywarpx.hybridpicmodel.joule_heating_n_min = args.joule_heating_n_min_frac * N_I
if args.redirect_n_min_factor > 0.0:
    pywarpx.hybridpicmodel.joule_redirect_n_min_factor = args.redirect_n_min_factor
if args.redirect_kick_cap > 0.0:
    pywarpx.hybridpicmodel.joule_redirect_kick_cap_vth_frac = args.redirect_kick_cap
if args.te_shunt > 0.0:
    pywarpx.hybridpicmodel.Te_shunt_threshold = args.te_shunt
if args.te_abort > 0.0:
    pywarpx.hybridpicmodel.Te_abort_threshold = args.te_abort
if args.contam_boundary_frac > 0.0:
    pywarpx.hybridpicmodel.qdsmc_contamination_n_boundary = (
        args.contam_boundary_frac * N_I
    )
if args.energy_budget:
    pywarpx.hybridpicmodel.qdsmc_energy_budget = 1
if args.grad_deposit == 0:
    pywarpx.hybridpicmodel.qdsmc_gradient_deposit = 0
if args.ve_midpoint == 0:
    pywarpx.hybridpicmodel.qdsmc_ve_midpoint_gather = 0
if args.cliff_deposit:
    pywarpx.hybridpicmodel.qdsmc_cliff_limited_deposit = 1
    if args.cliff_r1 >= 0.0:
        pywarpx.hybridpicmodel.qdsmc_cliff_deposit_r1 = args.cliff_r1
    if args.cliff_r2 > 0.0:
        pywarpx.hybridpicmodel.qdsmc_cliff_deposit_r2 = args.cliff_r2

simulation.initialize_warpx()

# ----------------------------------------------------------------------------
# Probes: hotspot growth, open-set energy, wall tallies
# ----------------------------------------------------------------------------
Te_wrap = fields.MultiFabWrapper(mf_name="hybrid_electron_temperature_fp", level=0)
rho_wrap = fields.MultiFabWrapper(mf_name="rho_fp", level=0)
qe = constants.q_e
K_PER_EV = constants.q_e / constants.kb

history = {"step": [], "te_max": [], "te_med": [], "u_open": []}


def probe():
    step = simulation.extension.warpx.getistep(lev=0)
    if step % 20 != 0:
        return
    te = Te_wrap[:, :, :] / K_PER_EV  # K -> eV
    ne = rho_wrap[:, :, :] / qe
    open_mask = ne > n_floor
    if comm.rank == 0 and open_mask.any():
        history["step"].append(step)
        history["te_max"].append(float(te[open_mask].max()))
        history["te_med"].append(float(np.median(te[open_mask])))
        history["u_open"].append(float(np.sum(ne[open_mask] * te[open_mask])))
        if step % 200 == 0:
            # live line for watchdogs (the 2026-08 runaways were invisible
            # between 400-step dumps)
            print(
                f"[liftoff] step {step} te_max={history['te_max'][-1]:.3e} eV "
                f"te_med={history['te_med'][-1]:.3e} eV",
                flush=True,
            )


callbacks.installafterstep(probe)

simulation.step(args.steps)

wx = simulation.extension.warpx
q_col = wx.get_eb_collected_charge("ions")
e_col = wx.get_eb_collected_energy("ions")

if comm.rank == 0:
    h = {k: np.asarray(v) for k, v in history.items()}
    print(
        f"[liftoff] done: N={resolution} steps={args.steps} kappa={args.kappa} "
        f"kmult={args.kappa_mult:g} resist={args.resistivity} "
        f"redirect={args.joule_redirect_te:g} relax={args.relax_rate} "
        f"heateta={args.joule_heating_eta} "
        f"heatnmin={args.joule_heating_n_min_frac:g} "
        f"redirgate={args.redirect_n_min_factor:g} "
        f"kickcap={args.redirect_kick_cap:g} teabort={args.te_abort:g} "
        f"filter={args.filter_passes}p/c{args.filter_comp} "
        f"tau={args.tau_ramp:.3g} drive={args.drive_scale:g} "
        f"| Te max {h['te_max'][0]:.2f} -> {h['te_max'][-1]:.2f} eV "
        f"(median {h['te_med'][-1]:.2f}) "
        f"| collected q={q_col:.4e} C E={e_col:.4e} J",
        flush=True,
    )
    np.savez_compressed(
        f"{args.out}.npz",
        resolution=resolution,
        steps=args.steps,
        dt=dt,
        kappa=args.kappa,
        kappa_C=KAPPA_C,
        kappa_mult=args.kappa_mult,
        resistivity=args.resistivity,
        joule_redirect_te=args.joule_redirect_te,
        relax_rate=args.relax_rate,
        joule_heating_eta=args.joule_heating_eta,
        joule_heating_n_min_frac=args.joule_heating_n_min_frac,
        redirect_n_min_factor=args.redirect_n_min_factor,
        redirect_kick_cap=args.redirect_kick_cap,
        te_abort=args.te_abort,
        filter_passes=args.filter_passes,
        filter_comp=args.filter_comp,
        tau_ramp=args.tau_ramp,
        drive_scale=args.drive_scale,
        kappa_perp_frac=args.kappa_perp_frac,
        standoff=args.standoff,
        te_final=Te_wrap[:, :, :] / K_PER_EV,
        rho_final=rho_wrap[:, :, :],
        q_collected=q_col,
        e_collected=e_col,
        **h,
    )

# Hold every rank until the npz is on disk (GPU-node teardown race
# killed rank 0 mid-savez on arm C, 2026-08-07).
comm.Barrier()
