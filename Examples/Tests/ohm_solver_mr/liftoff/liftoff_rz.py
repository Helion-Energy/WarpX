#!/usr/bin/env python3
#
# --- RZ reference arm for the liftoff m=4 seam battery: quasi-1D radial
# --- theta-pinch, single level (RZ+MR does not exist), same annulus and
# --- external-field ramp as the 3D slab. The conducting wall is realized as
# --- the r-max DOMAIN boundary (Dirichlet field BC + absorbing particles) at
# --- r = 0.8 m -- the RZ cylinder-compression deck precedent -- rather than
# --- an EB (RZ+EB+hybrid not exercised on this branch; recorded choice).
# --- A_theta = (r/2) * f(t) via the analytic-parser A_external path (RZ
# --- parser convention: expressions in (x, y, z) evaluated at y=0, x=r, so
# --- Ay maps to A_theta). Purpose: the axisymmetric bulk-implosion reference
# --- (column radius vs t, peak density); m=4 is structurally absent.

import argparse
import sys

import numpy as np

import pywarpx
from pywarpx import picmi

constants = picmi.constants

M_AMU = 2
N_I = 1.5e20
N_FLOOR_FRAC = 0.03
T_I0 = 5.0
T_E0 = 0.0

R_INNER = 0.6
R_OUTER = 0.7
R_PART = 0.7
R_WALL = 0.8  # r-max domain boundary (the wall; recorded choice)

BZ_BIAS = -0.1
BZ_REV = 1.5
TAU_RAMP = 4.0e-6

ETA_PLASMA = 1.0e-6
ETA_VAC_FRAC = 5.0e-2
ETA_POWER = 2.0
N_TRANSITION_FRAC = 0.7

F_T_CI = 0.01
NZ = 8

SEED = 20260809


def hermite_ramp_expression(b0, b1, tau):
    s = f"min(max(t/{tau:.9e},0),1)"
    return f"({b0:.9e} + ({b1 - b0:.9e})*({s})*({s})*(3-2*({s})))"


def power_law_resistivity(
    eta_plasma, eta_vac, power, n_floor_frac, n_transition_frac, n0
):
    a = (n_floor_frac / n_transition_frac) ** power
    res_str = (
        f"eta_plasma + (eta_vac - eta_plasma)"
        f"*max(0.0, (rho_f/max(rho,rho_f))**({power:.6g}) - ({a:.12g}))"
        f"/({1.0 - a:.12g})"
    )
    return {
        "plasma_resistivity": res_str,
        "eta_plasma": eta_plasma,
        "eta_vac": eta_vac,
        "rho_f": constants.q_e * n0 * n_floor_frac,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nr", help="radial cells", type=int, default=80)
    parser.add_argument("--nppc", help="macroparticles per cell", type=int, default=64)
    parser.add_argument("--steps", help="number of steps", type=int, default=4000)
    parser.add_argument(
        "--diag-steps", help="field diagnostic period", type=int, default=50
    )
    parser.add_argument(
        "--substeps",
        help="RK4 substep count (pre-registered 104)",
        type=int,
        default=104,
    )
    parser.add_argument(
        "--pin-numerics-n",
        help="evaluate eta_vac and eta_hyper at this 3D base-grid N",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--eta-vac-frac",
        type=float,
        default=ETA_VAC_FRAC,
    )
    parser.add_argument("--tau-ramp", type=float, default=TAU_RAMP)
    parser.add_argument(
        "--holmstrom",
        help="zero the Ohm's-law Hall/pressure E in vacuum regions",
        action="store_true",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("-v", "--verbose", type=int, default=0)
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    m_i = M_AMU * constants.m_p
    n_floor = N_FLOOR_FRAC * N_I
    vth = np.sqrt(constants.q_e * T_I0 / m_i)

    dr = R_WALL / args.nr
    lz = NZ * dr
    zmin, zmax = 5.0 - lz / 2.0, 5.0 + lz / 2.0

    w_ci = constants.q_e * abs(BZ_REV) / m_i
    t_ci = 2.0 * np.pi / w_ci
    dt = F_T_CI * t_ci

    # numerics pinned to the 3D base grid (battery comparability)
    dx_pin = 2.0 / args.pin_numerics_n
    w_pi = np.sqrt(constants.q_e**2 * N_I / (constants.ep0 * m_i))
    l_i = constants.c / w_pi
    vA = abs(BZ_REV) / np.sqrt(constants.mu0 * N_I * m_i)
    dL2 = dx_pin**2 / 3.0
    eta_max = constants.mu0 * dL2 / (2.0 * dt)
    eta_hyper = constants.mu0 * 0.2 * l_i * vA * dL2

    grid = picmi.CylindricalGrid(
        number_of_cells=[args.nr, NZ],
        lower_bound=[0.0, zmin],
        upper_bound=[R_WALL, zmax],
        lower_boundary_conditions=["none", "periodic"],
        upper_boundary_conditions=["dirichlet", "periodic"],
        lower_boundary_conditions_particles=["none", "periodic"],
        upper_boundary_conditions_particles=["absorbing", "periodic"],
        warpx_max_grid_size=2048,
    )

    sim = picmi.Simulation(
        time_step_size=dt,
        max_steps=args.steps,
        particle_shape=1,
        verbose=args.verbose,
    )
    sim.current_deposition_algo = "direct"
    sim.grid_type = "staggered"

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

    sim.solver = picmi.HybridPICSolver(
        grid=grid,
        gamma=5.0 / 3.0,
        Te=T_E0,
        n0=N_I,
        n_floor=n_floor,
        plasma_hyper_resistivity=eta_hyper,
        substeps=args.substeps,
        substep_rtol=1.0e-3,
        substep_atol=1.0e-8,
        max_substep_attempts=1000,
        holmstrom_vacuum_region=True if args.holmstrom else None,
        A_external=A_ext,
        **power_law_resistivity(
            ETA_PLASMA,
            args.eta_vac_frac * eta_max,
            ETA_POWER,
            N_FLOOR_FRAC,
            N_TRANSITION_FRAC,
            N_I,
        ),
    )

    n_annulus = N_I * R_PART**2 / (R_OUTER**2 - R_INNER**2)
    n_fill = 2.0 * n_floor
    r_expr = "sqrt(x*x+y*y)"
    ions = picmi.Species(
        name="ions",
        mass=m_i,
        charge="q_e",
        initial_distribution=picmi.AnalyticDistribution(
            density_expression=(
                f"n_a*(({r_expr}>=R_in)*({r_expr}<=R_out))+n_f*({r_expr}<R_in)"
            ),
            momentum_expressions=["0", "0", "0"],
            warpx_momentum_spread_expressions=[f"{vth}"] * 3,
            warpx_density_min=0.5 * n_fill,
            n_a=n_annulus,
            n_f=n_fill,
            R_in=R_INNER,
            R_out=R_OUTER,
        ),
    )
    sim.add_species(
        ions,
        layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=args.nppc),
    )

    ion_ion_coulomb = picmi.CoulombCollisions(
        name="ion_ion_Coulomb",
        species=[ions, ions],
        CoulombLog=12,
    )
    sim.collisions = [ion_ion_coulomb]

    field_diag = picmi.FieldDiagnostic(
        name="diag1",
        grid=grid,
        period=args.diag_steps,
        data_list=["B", "E", "rho", "J"],
        write_dir="diags",
        warpx_format="plotfile",
    )
    sim.add_diagnostic(field_diag)

    print(
        f"liftoff_rz: NR={args.nr} nppc={args.nppc} steps={args.steps} "
        f"dt={dt:.4e} substeps={args.substeps} "
        f"eta_vac={args.eta_vac_frac * eta_max:.4e} eta_hyper={eta_hyper:.4e}",
        flush=True,
    )

    sim.initialize_inputs()
    pywarpx.warpx.random_seed = args.seed
    sim.initialize_warpx()
    sim.step(args.steps)


if __name__ == "__main__":
    main()
