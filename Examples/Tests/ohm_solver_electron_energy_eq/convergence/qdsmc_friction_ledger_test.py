#!/usr/bin/env python3
#
# --- Friction/battery ledger instrument for the solve_for_Faraday gating.
# ---
# --- momentum arms (force-free B, uniform n, real-mass D ions at rest,
# --- constant eta; electrons carry J so u_e = -J/(e n) != 0):
# ---   drag-off : E* push, no drag operator -- the ion-side friction lives
# ---              in the dropped eta J (exact single-species cancellation);
# ---              bulk ion momentum must stay ~0.
# ---   drag-on  : Q_ei drag operator on -> the push now uses the TRUE E
# ---              (eta J included), whose force cancels the drag kick;
# ---              bulk ion momentum must stay ~0 AND far below the
# ---              single-booked drag rate nu*u_e*t (the old E*+drag double
# ---              booking would show the full rate).
# ---
# --- biermann arm: crossed gradients n(x), Te(z) (energy equation on,
# --- conduction/sources off, B(0) = 0): with include_biermann_battery=1
# --- the Faraday E keeps -grad Pe/(e n) and B_y must grow at the analytic
# --- battery rate (kB/e) * dTe/dz * (dn/dx)/n; with it off, B_y stays 0.

import argparse
import sys

import numpy as np

import pywarpx
from pywarpx import fields, particle_containers, picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=["drag-off", "drag-on", "biermann"], required=True)
parser.add_argument("--biermann-flag", type=int, choices=[0, 1], default=1)
parser.add_argument("--ncell", type=int, default=64)
parser.add_argument("--nsteps", type=int, default=100)
parser.add_argument("--out", type=str, required=True)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

N = args.ncell
L = 1.0
dx = L / N
n0 = 1.0e20
Te0_eV = 10.0
dt = 1.0e-8
qe = constants.q_e
kb = constants.kb
K_per_eV = qe / kb
m_i = 2.0 * constants.m_p

momentum_arm = args.arm in ("drag-off", "drag-on")

B0 = 0.05
ETA = 1.0e-5
NU_EI = 1.0e5  # constant relaxation rate [1/s] for the drag-on arm

grid = picmi.Cartesian2DGrid(
    number_of_cells=[N, N],
    lower_bound=[-L / 2.0, -L / 2.0],
    upper_bound=[L / 2.0, L / 2.0],
    lower_boundary_conditions=["periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic"],
    lower_boundary_conditions_particles=["periodic", "periodic"],
    upper_boundary_conditions_particles=["periodic", "periodic"],
    warpx_max_grid_size=N,
)

solver_kwargs = {}
if args.arm == "drag-on":
    solver_kwargs["electron_ion_relaxation_rate"] = f"{NU_EI}"
solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=5.0 / 3.0,
    Te=Te0_eV,
    n0=n0,
    n_floor=1.0e-6 * n0,
    plasma_resistivity=ETA if momentum_arm else 0.0,
    substeps=4,
    solve_electron_energy_equation=True,
    **solver_kwargs,
)

simulation = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=args.nsteps,
    particle_shape=1,
    verbose=0,
    warpx_serialize_initial_conditions=True,
    warpx_current_deposition_algo="direct",
)

k = 2.0 * np.pi / L
if momentum_arm:
    # force-free field: J = curl B / mu0 uniform in magnitude, J x B = 0
    simulation.add_applied_field(
        picmi.AnalyticInitialField(
            Bx_expression="0",
            By_expression=f"{B0}*sin({k}*x)",
            Bz_expression=f"{B0}*cos({k}*x)",
            warpx_do_initial_div_cleaning=False,
        )
    )
    density_expression = f"{n0}"
    mass = m_i  # real-mass ions: the momentum response is the measurement
    nppc = 16
else:
    density_expression = f"{n0}*(1.0+0.3*sin({k}*x))"
    mass = 1.0e12 * constants.m_p  # static background: isolate the battery
    nppc = 8

ions = picmi.Species(
    name="ions",
    charge="q_e",
    mass=mass,
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=density_expression,
        momentum_expressions=["0", "0", "0"],
    ),
)
simulation.add_species(
    ions,
    layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=nppc),
)

simulation.initialize_inputs()
if args.arm == "biermann":
    pywarpx.hybridpicmodel.include_biermann_battery = args.biermann_flag
simulation.initialize_warpx()

Te_wrap = fields.MultiFabWrapper(mf_name="hybrid_electron_temperature_fp", level=0)


def ion_mean_velocity():
    pw = particle_containers.ParticleContainerWrapper("ions")
    sums = np.zeros(4)
    for comp, idx in (("w", 0), ("ux", 1), ("uy", 2), ("uz", 3)):
        for arr in pw.get_particle_real_arrays(comp, level=0, copy_to_host=True):
            if idx == 0:
                sums[0] += arr.sum()
            else:
                pass
    # weighted momentum: need w*u sums
    wtot = 0.0
    pu = np.zeros(3)
    ws = pw.get_particle_real_arrays("w", level=0, copy_to_host=True)
    us = [
        pw.get_particle_real_arrays(c, level=0, copy_to_host=True)
        for c in ("ux", "uy", "uz")
    ]
    for tile in range(len(ws)):
        wtot += ws[tile].sum()
        for d in range(3):
            pu[d] += (ws[tile] * us[d][tile]).sum()
    return pu / max(wtot, 1.0), wtot


dTe_eV = 3.0
state = {}


def capture0():
    # impose Te(z) AFTER the adiabat seed (which runs at Evolve entry and
    # would wipe a pre-step write to Te(x) ~ the n adiabat, killing the
    # crossed gradients) -- the annulus-deck idiom
    if simulation.extension.warpx.getistep(lev=0) != 0 or "done" in state:
        return
    ax_n = np.linspace(-L / 2.0, L / 2.0, N + 1)
    XG, ZG = np.meshgrid(ax_n, ax_n, indexing="ij")
    Te_wrap[:, :] = (Te0_eV + dTe_eV * np.sin(k * ZG)) * K_per_eV
    state["done"] = True


if args.arm == "biermann":
    from pywarpx import callbacks

    callbacks.installparticleinjection(capture0)

u0, w0 = (np.zeros(3), 0.0) if args.arm == "biermann" else ion_mean_velocity()

simulation.step(args.nsteps)

T_tot = args.nsteps * dt
if momentum_arm:
    u1, w1 = ion_mean_velocity()
    du = u1 - u0
    # single-booked drag drift the OLD E*+drag code would show:
    J = B0 * k / constants.mu0
    u_e = J / (qe * n0)
    du_bug = NU_EI * u_e * T_tot
    print(
        f"[ledger] arm={args.arm} | mean ion du = ({du[0]:+.3e}, {du[1]:+.3e}, "
        f"{du[2]:+.3e}) m/s | single-booked drag drift {du_bug:.3e} m/s "
        f"| ratio {np.linalg.norm(du) / du_bug:.3e}",
        flush=True,
    )
    np.savez_compressed(f"{args.out}.npz", arm=args.arm, du=du, du_bug=du_bug)
else:
    By_wrap = fields.ByFPWrapper(level=0)
    by = By_wrap[:, :]
    ax = np.linspace(-L / 2.0, L / 2.0, by.shape[0])
    XG, ZG = np.meshgrid(ax, np.linspace(-L / 2.0, L / 2.0, by.shape[1]), indexing="ij")
    n_of_x = n0 * (1.0 + 0.3 * np.sin(k * XG))
    dndx = n0 * 0.3 * k * np.cos(k * XG)
    dTedz_K = dTe_eV * K_per_eV * k * np.cos(k * ZG)
    # battery rate for Pe = n kB Te: curl(grad Pe/(e n))_y with Te=Te(z), n=n(x):
    #   dBy/dt = -(kB/e) * dTe/dz * (dn/dx) / n
    by_pred = -(kb / qe) * dTedz_K * dndx / n_of_x * T_tot
    num = float(np.sqrt(np.mean(by**2)))
    pred = float(np.sqrt(np.mean(by_pred**2)))
    print(
        f"[ledger] arm=biermann flag={args.biermann_flag} | rms By {num:.4e} T "
        f"| analytic battery {pred:.4e} T | ratio {num / max(pred, 1e-30):.3f}",
        flush=True,
    )
    np.savez_compressed(f"{args.out}.npz", by=by, by_pred=by_pred, rms=num, pred=pred)
