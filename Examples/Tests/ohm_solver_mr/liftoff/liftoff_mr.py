#!/usr/bin/env python3
#
# --- Theta-pinch liftoff of an annular plasma column (hybrid-PIC solver),
# --- ported to the MR+EB stack (branch hybrid_mr_eb) for the m=4 seam
# --- battery: a thin 3D slab (NZ=8 cubic cells, periodic z) with an annular
# --- deuterium column inside a conducting cylindrical wall (EB staircase
# --- freeze, extended per level by the MR stack), reversed by an external
# --- vector potential with a Hermite-smoothstep time ramp (-0.1 T -> +1.5 T
# --- over 4 us).
# ---
# --- Port deltas vs the hybrid_ohms_law_global_A_recovery original:
# ---  * use_conformal_eb / use_global_A_recovery / eb_deposit_fold /
# ---    eb_rho_dirichlet dropped (do not exist on this branch);
# ---  * RKF45 replaced by fixed-substep RK4 sized for the finest grid
# ---    (RKF45 aborts with MR; lockstep substep rule);
# ---  * optional static ratio-2 level-1 patch (--refined-core), square
# ---    in-plane, full periodic z: the square seam is an m=4-shaped
# ---    perturbation by construction;
# ---  * numerics pinning (--pin-numerics-n): eta_vac and the Chacon
# ---    hyper-resistivity floor evaluated at a fixed base-grid dx so all
# ---    battery arms solve the same physical problem.

import argparse
import sys

import numpy as np

import pywarpx
from pywarpx import picmi

constants = picmi.constants

# ----------------------------------------------------------------------------
# Physics parameters: slab at the z = 5 m midplane of the formation section
# ----------------------------------------------------------------------------
M_AMU = 2  # deuterium
N_I = 1.5e20  # reference ion density (m^-3)
N_FLOOR_FRAC = 0.03
T_I0 = 5.0  # ion temperature (eV)
T_E0 = 0.0  # electron temperature (eV): no electron pressure

# Annular column (radii in m) and low-density interior fill
R_INNER = 0.6
R_OUTER = 0.7
R_PART = 0.7  # the annulus carries the inventory of a full column of this radius
R_WALL = 0.8  # conducting wall radius (embedded boundary)

# External field ramp: Hermite smoothstep from the bias field to the
# (reversed) main field over TAU_RAMP
BZ_BIAS = -0.1  # T
BZ_REV = 1.5  # T
TAU_RAMP = 4.0e-6  # s

# Resistivity (density-scaled power law) and hyper-resistivity
ETA_PLASMA = 1.0e-6  # Ohm m, bulk plasma
ETA_VAC_FRAC = 5.0e-2  # vacuum resistivity as a fraction of the CFL limit
ETA_POWER = 2.0
N_TRANSITION_FRAC = 0.7

# Time stepping
F_T_CI = 0.01  # dt as a fraction of the (reversed-field) ion cyclotron period

NZ = 8  # one blocking factor of cells: thin periodic slab

# MR patch: target realized half-width of the square level-1 tag region (m)
PATCH_HALF_TARGET = 0.46

SEED = 20260809


def hermite_ramp_expression(b0, b1, tau):
    """Time scaling of the unit-Bz external vector potential: Hermite
    smoothstep h(s) = s^2*(3-2*s) from b0 (bias) to b1 (reversed field)."""
    s = f"min(max(t/{tau:.9e},0),1)"
    return f"({b0:.9e} + ({b1 - b0:.9e})*({s})*({s})*(3-2*({s})))"


def power_law_resistivity(
    eta_plasma, eta_vac, power, n_floor_frac, n_transition_frac, n0
):
    """Density-scaled resistivity interpolating from eta_plasma in the bulk
    (n >= n_transition) to eta_vac at the density floor, as a power law of
    the density (see the formation simulation this case is derived from)."""
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


def whistler_substeps(dx_finest, dt, margin=2.5):
    """Pre-registered RK4 substep rule: staggered-grid whistler ceiling at
    the density floor and B = |BZ_REV| on the finest grid; RK4 imaginary-axis
    stability 2*sqrt(2); margin covers compression overshoot of B."""
    M_AMU * constants.m_p
    n_floor = N_FLOOR_FRAC * N_I
    k2_eff = 3 * 4.0 / dx_finest**2  # staggered curl Nyquist, 3 cubic dims
    omega = abs(BZ_REV) * k2_eff / (constants.mu0 * n_floor * constants.q_e)
    n_min = omega * dt / (2.0 * np.sqrt(2.0))
    n_sub = int(np.ceil(margin * n_min / 2.0) * 2)
    return max(n_sub, 4)


def setup_simulation(args):
    """Create the PICMI simulation object; returns (sim, info dict)."""
    resolution = args.resolution
    m_i = M_AMU * constants.m_p
    n_floor = N_FLOOR_FRAC * N_I
    vth = np.sqrt(constants.q_e * T_I0 / m_i)

    # cell sizes (cubic cells; thin z slab centered on the z = 5 m midplane)
    dx = 2.0 / resolution
    lz = NZ * dx
    zmin, zmax = 5.0 - lz / 2.0, 5.0 + lz / 2.0

    # time step from the reversed-field ion cyclotron period
    w_ci = constants.q_e * abs(BZ_REV) / m_i
    t_ci = 2.0 * np.pi / w_ci
    dt = F_T_CI * t_ci

    # ion skin depth and Alfven speed at the reversed field for the
    # Chacon-style hyper-resistivity floor; CFL-limited vacuum resistivity.
    # PINNED NUMERICS: dL2 is evaluated at the --pin-numerics-n base grid
    # (default: this run's own resolution) so every battery arm uses the
    # same eta_vac and eta_hyper values.
    pin_n = args.pin_numerics_n if args.pin_numerics_n is not None else resolution
    dx_pin = 2.0 / pin_n
    w_pi = np.sqrt(constants.q_e**2 * N_I / (constants.ep0 * m_i))
    l_i = constants.c / w_pi
    vA = abs(BZ_REV) / np.sqrt(constants.mu0 * N_I * m_i)
    dL2 = dx_pin**2 / 3.0  # 1/(1/dx^2+1/dy^2+1/dz^2), cubic
    eta_max = constants.mu0 * dL2 / (2.0 * dt)
    eta_hyper = constants.mu0 * 0.2 * l_i * vA * dL2

    # RK4 substeps sized for the finest grid present (lockstep rule)
    dx_finest = dx / 2.0 if args.refined_core else dx
    substeps = (
        args.substeps if args.substeps is not None else whistler_substeps(dx_finest, dt)
    )

    grid_kw = {}
    if args.refined_core:
        # level-1 boxes must stay small enough that the blocking factor does
        # not round the patch out toward the wall
        grid_kw["warpx_blocking_factor"] = 4
    else:
        grid_kw["warpx_blocking_factor"] = 8
    grid = picmi.Cartesian3DGrid(
        number_of_cells=[resolution, resolution, NZ],
        lower_bound=[-1.0, -1.0, zmin],
        upper_bound=[1.0, 1.0, zmax],
        lower_boundary_conditions=["dirichlet", "dirichlet", "periodic"],
        upper_boundary_conditions=["dirichlet", "dirichlet", "periodic"],
        lower_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
        upper_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
        warpx_max_grid_size=2048,
        warpx_max_grid_size_x=args.max_grid_size_xy,
        warpx_max_grid_size_y=args.max_grid_size_xy,
        **grid_kw,
    )

    patch_cells = 0
    if args.refined_core:
        # Square in-plane static level-1 patch, full periodic z. Realized
        # half-width = round-to-even(target/dx) coarse cells; the tag box is
        # shrunk by 1.5 cells to absorb the AMReX one-cell error buffer.
        patch_cells = int(round(args.patch_half / dx))
        if patch_cells % 2 == 1:
            patch_cells += 1
        tag_half = (patch_cells - 1.5) * dx
        grid.add_refined_region(
            level=1,
            lo=[-tag_half, -tag_half, zmin],
            hi=[tag_half, tag_half, zmax],
        )

    sim = picmi.Simulation(
        time_step_size=dt,
        max_steps=args.steps,
        particle_shape=1,
        verbose=args.verbose,
    )
    sim.current_deposition_algo = "direct"
    sim.grid_type = "staggered"

    # External reversal field: uniform Bz through A = (-y/2, x/2, 0) * f(t),
    # with the Hermite ramp from the bias to the reversed field in f(t)
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
        substeps=substeps,
        # use_rkf45 stays False (RK4 path; RKF45 aborts with MR). On a NaN
        # the single-level arms auto-restart the half-step with RKF45: give
        # that fallback the original deck's tolerances and attempt budget.
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

    sim.embedded_boundary = picmi.EmbeddedBoundary(
        implicit_function="(x**2+y**2-R_w**2)", R_w=R_WALL
    )

    # Annular column carrying the inventory of a full column of radius
    # R_PART at N_I, plus a low-density interior fill (no plasma between the
    # annulus and the wall)
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
        data_list=["B", "E", "rho", "J", "divB"],
        write_dir="diags",
        warpx_format="plotfile",
    )
    sim.add_diagnostic(field_diag)

    info = dict(
        dx=dx,
        dt=dt,
        substeps=substeps,
        eta_vac=args.eta_vac_frac * eta_max,
        eta_hyper=eta_hyper,
        patch_cells=patch_cells,
        patch_half=patch_cells * dx,
    )
    return sim, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        help="run as a short CI-size smoke (32x32x8, 4 ppc, 10 steps)",
        action="store_true",
    )
    parser.add_argument(
        "-n", "--resolution", help="cells along x and y", type=int, default=128
    )
    parser.add_argument("--nppc", help="macroparticles per cell", type=int, default=16)
    parser.add_argument("--steps", help="number of steps", type=int, default=None)
    parser.add_argument(
        "--diag-steps", help="field diagnostic period", type=int, default=50
    )
    parser.add_argument(
        "--refined-core",
        help="add the static ratio-2 level-1 square patch (full z)",
        action="store_true",
    )
    parser.add_argument(
        "--patch-half",
        help="target realized half-width of the level-1 patch (m)",
        type=float,
        default=PATCH_HALF_TARGET,
    )
    parser.add_argument(
        "--no-divb-audit",
        help="disable hybrid_pic_model.mr_check_div_b (per-substep global "
        "reductions are costly at high substep counts)",
        action="store_true",
    )
    parser.add_argument(
        "--mr-emf-matching",
        help="override hybrid_pic_model.mr_emf_matching (arm M0 uses 0)",
        type=int,
        choices=[0, 1],
        default=None,
    )
    parser.add_argument(
        "--substeps",
        help="override the RK4 substep count (default: pre-registered rule)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--pin-numerics-n",
        help="evaluate eta_vac and eta_hyper at this base-grid N (battery "
        "pinning; default: this run's own resolution)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--eta-vac-frac",
        help="vacuum resistivity as a fraction of the (pinned) CFL limit",
        type=float,
        default=ETA_VAC_FRAC,
    )
    parser.add_argument(
        "--tau-ramp",
        help="ramp time (s); shrink for the substep-stability smoke",
        type=float,
        default=TAU_RAMP,
    )
    parser.add_argument(
        "--seed", help="random seed (shared across arms)", type=int, default=SEED
    )
    parser.add_argument(
        "--holmstrom",
        help="zero the Ohm's-law Hall/pressure E in vacuum regions",
        action="store_true",
    )
    parser.add_argument(
        "--max-grid-size-xy",
        help="max grid size along x and y (parallel decomposition)",
        type=int,
        default=24,
    )
    parser.add_argument("-v", "--verbose", help="WarpX verbosity", type=int, default=0)
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    if args.test:
        args.resolution = 32
        args.nppc = 4
        if args.steps is None:
            args.steps = 10
        args.diag_steps = min(args.diag_steps, args.steps)
    if args.steps is None:
        m_i = M_AMU * constants.m_p
        t_ci = 2.0 * np.pi * m_i / (constants.q_e * abs(BZ_REV))
        args.steps = int((TAU_RAMP + t_ci) / (F_T_CI * t_ci))

    sim, info = setup_simulation(args)

    print(
        f"liftoff_mr: N={args.resolution} nppc={args.nppc} steps={args.steps} "
        f"dt={info['dt']:.4e} substeps={info['substeps']} "
        f"eta_vac={info['eta_vac']:.4e} eta_hyper={info['eta_hyper']:.4e} "
        f"refined={args.refined_core} patch_half={info['patch_half']:.4f} m "
        f"({info['patch_cells']} coarse cells) mr_emf_matching={args.mr_emf_matching}",
        flush=True,
    )

    # Bucket writes between initialize_inputs and initialize_warpx (the only
    # reliable path for hybrid_pic_model knobs; verify in warpx_used_inputs)
    sim.initialize_inputs()
    pywarpx.warpx.random_seed = args.seed
    if args.refined_core:
        pywarpx.hybridpicmodel.mr_check_div_b = 0 if args.no_divb_audit else 1
        pywarpx.warpx.refine_plasma_init = 1
        if args.mr_emf_matching is not None:
            pywarpx.hybridpicmodel.mr_emf_matching = args.mr_emf_matching
    sim.initialize_warpx()
    sim.step(args.steps)


if __name__ == "__main__":
    main()
