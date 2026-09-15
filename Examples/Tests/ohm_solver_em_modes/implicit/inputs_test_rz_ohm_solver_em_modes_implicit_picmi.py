#!/usr/bin/env python3
#
# --- Test script for the kinetic-fluid hybrid model in WarpX wherein ions are
# --- treated as kinetic particles and electrons as an isothermal, inertialess
# --- background fluid. The script is set up to produce parallel normal EM modes
# --- in a metallic cylinder and is run in RZ geometry.
# --- As a CI test only a small number of steps are taken.
# --- This version uses the theta-implicit hybrid solver.

import argparse
import sys
from pathlib import Path

import dill
import numpy as np
from mpi4py import MPI as mpi

from pywarpx import amr, picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(verbose=0)


class CylindricalNormalModes(object):
    """The following runs a simulation of an uniform plasma at a set ion
    temperature (and Te = 0) with an external magnetic field applied in the
    z-direction (parallel to domain).
    The analysis script (in this same directory) analyzes the output field
    data for EM modes.
    """

    # Applied field parameters
    B0 = 0.5  # Initial magnetic field strength (T)
    beta = 0.01  # Plasma beta, used to calculate temperature

    # Plasma species parameters
    m_ion = 400.0  # Ion mass (electron masses)
    vA_over_c = 5e-3  # ratio of Alfven speed and the speed of light

    # Spatial domain
    Nz = 512  # number of cells in z direction
    Nr = 128  # number of cells in r direction

    # Temporal domain (if not run as a CI test)
    LT = 800.0  # Simulation temporal length (ion cyclotron periods)

    # Numerical parameters
    NPPC = 800  # Seed number of particles per cell
    DZ = 0.4  # Cell size (ion skin depths)
    DR = 0.4  # Cell size (ion skin depths)
    DT = 0.002  # Time step (ion cyclotron periods)

    # Plasma resistivity - used to dampen the mode excitation
    eta = 5e-4
    # Number of substeps used to update B
    substeps = 40

    def __init__(
        self,
        test,
        verbose,
        pc_bb=False,
        d1=False,
        dt_mult=1.0,
        steps=None,
        mm_jacobian=False,
        mm_pc=None,
    ):
        """Get input parameters for the specific case desired."""
        self.test = test
        self.verbose = verbose or self.test
        self.pc_bb = pc_bb
        self.d1 = d1
        self.dt_mult = dt_mult

        # The one/two-rank D1 pair keeps the sole BoxArray box on rank zero;
        # a fixed seed makes its particle realization an exact common input.
        if self.d1 or mm_pc is not None:
            simulation.random_seed = 20260914

        # calculate various plasma parameters based on the simulation input
        self.get_plasma_quantities()

        if not self.test:
            self.total_steps = int(self.LT / self.DT)
            # output diagnostics 50 times per cyclotron period
            self.diag_steps = max(10, int(1.0 / 50 / self.DT))
        else:
            # if this is a test case run for only a small number of steps
            self.total_steps = steps if steps else 50
            # and make the grid and particle count smaller
            self.Nz = 128
            self.Nr = 64
            self.NPPC = 200
            self.diag_steps = 10

        self.Lz = self.Nz * self.DZ * self.l_i
        self.Lr = self.Nr * self.DR * self.l_i

        self.dt = self.DT * self.t_ci * self.dt_mult

        # dump all the current attributes to a dill pickle file
        if comm.rank == 0:
            with open("sim_parameters.dpkl", "wb") as f:
                dill.dump(self, f)

        # print out plasma parameters
        if comm.rank == 0:
            print(
                f"Initializing simulation with input parameters:\n"
                f"\tT = {self.T_plasma:.3f} eV\n"
                f"\tn = {self.n_plasma:.1e} m^-3\n"
                f"\tB0 = {self.B0:.2f} T\n"
                f"\tM/m = {self.m_ion:.0f}\n"
            )
            print(
                f"Plasma parameters:\n"
                f"\tl_i = {self.l_i:.1e} m\n"
                f"\tt_ci = {self.t_ci:.1e} s\n"
                f"\tv_ti = {self.v_ti:.1e} m/s\n"
                f"\tvA = {self.vA:.1e} m/s\n"
            )
            print(
                f"Numerical parameters:\n"
                f"\tdt = {self.dt:.1e} s\n"
                f"\tdiag steps = {self.diag_steps:d}\n"
                f"\ttotal steps = {self.total_steps:d}\n",
                flush=True,
            )
        self.setup_run(mm_jacobian=mm_jacobian, mm_pc=mm_pc)

    def get_plasma_quantities(self):
        """Calculate various plasma parameters based on the simulation input."""
        # Ion mass (kg)
        self.M = self.m_ion * constants.m_e

        # Cyclotron angular frequency (rad/s) and period (s)
        self.w_ci = constants.q_e * abs(self.B0) / self.M
        self.t_ci = 2.0 * np.pi / self.w_ci

        # Alfven speed (m/s): vA = B / sqrt(mu0 * n * (M + m)) = c * omega_ci / w_pi
        self.vA = self.vA_over_c * constants.c
        self.n_plasma = (self.B0 / self.vA) ** 2 / (
            constants.mu0 * (self.M + constants.m_e)
        )

        # Ion plasma frequency (Hz)
        self.w_pi = np.sqrt(constants.q_e**2 * self.n_plasma / (self.M * constants.ep0))

        # Skin depth (m)
        self.l_i = constants.c / self.w_pi

        # Ion thermal velocity (m/s) from beta = 2 * (v_ti / vA)**2
        self.v_ti = np.sqrt(self.beta / 2.0) * self.vA

        # Temperature (eV) from thermal speed: v_ti = sqrt(kT / M)
        self.T_plasma = self.v_ti**2 * self.M / constants.q_e  # eV

        # Larmor radius (m)
        self.rho_i = self.v_ti / self.w_ci

    def setup_run(self, mm_jacobian=False, mm_pc=None):
        """Setup simulation components."""

        #######################################################################
        # Set geometry and boundary conditions                                #
        #######################################################################

        if self.test and self.pc_bb:
            # AMReX otherwise refines a one-box layout until it has at least
            # one box per MPI rank. Keep this arm at one BoxArray box so its
            # second CTest rank has no local box and exercises the resolved
            # local path for precond.bb_global=-1.
            amr.refine_grid_layout = False

        self.grid = picmi.CylindricalGrid(
            number_of_cells=[self.Nr, self.Nz],
            warpx_max_grid_size=self.Nz,
            lower_bound=[0, -self.Lz / 2.0],
            upper_bound=[self.Lr, self.Lz / 2.0],
            lower_boundary_conditions=["none", "periodic"],
            upper_boundary_conditions=["dirichlet", "periodic"],
            lower_boundary_conditions_particles=["none", "periodic"],
            upper_boundary_conditions_particles=["reflecting", "periodic"],
        )
        simulation.time_step_size = self.dt
        simulation.max_steps = self.total_steps
        simulation.current_deposition_algo = "direct"
        simulation.particle_shape = 1
        simulation.verbose = self.verbose

        #######################################################################
        # Field solver and external field                                     #
        #######################################################################

        self.solver = picmi.HybridPICSolver(
            grid=self.grid,
            Te=0.0,
            n0=self.n_plasma,
            plasma_resistivity=self.eta,
            substeps=self.substeps,
            n_floor=self.n_plasma * 0.05,
        )
        simulation.solver = self.solver

        B_ext = picmi.AnalyticInitialField(Bz_expression=self.B0)
        simulation.add_applied_field(B_ext)

        #######################################################################
        # Implicit solver setup                                               #
        #######################################################################

        # Create GMRES linear solver for Newton (JFNK)
        gmres_solver = picmi.GMRESLinearSolver(
            verbose_int=1,
            max_iterations=20,
            relative_tolerance=1.0e-4,
            absolute_tolerance=0.0,
        )

        # Block-banded direct preconditioner arm: the built-in verification
        # gates (extraction self-check, LU/Apply round-trips, FD-JVP
        # comparison) run at every rebuild and abort on failure, and Newton
        # convergence is required rather than best-effort. The operator is
        # rebuilt every 10 steps: this deck is quasi-static so the frozen
        # coefficients hold, it exercises the stale-operator amortization
        # path used in production, and it keeps the rebuild cost (probe +
        # factor, ~2 s/rebuild at this scale on a full host) inside the CI
        # budget on a 2-core runner while still running the verification
        # gates at five distinct states across the run. In test mode the
        # explicit refine_grid_layout=False setting above keeps the 64x128
        # domain in one BoxArray box. The two-rank CTest therefore leaves one
        # rank with no local box and explicitly exercises auto-global
        # resolution to local mode.
        pc = None
        if self.pc_bb:
            pc = picmi.BlockBandedPreconditioner(
                verify=True, update_interval=10, global_solve=-1
            )
        if self.dt_mult > 1.0:
            # whistler-stiff arm: the line search rescues the large-dt Newton
            # basin (without it the unpreconditioned dt x4/x8 solves can
            # diverge), keeping the arm a pure linear-solver measurement
            import pywarpx

            pywarpx.warpx.get_bucket("newton").line_search = 1

        # Create nonlinear solver using Newton (JFNK)
        nonlinear_solver = picmi.NewtonNonlinearSolver(
            verbose=True,
            max_iterations=20,
            relative_tolerance=1.0e-6,
            absolute_tolerance=0.0,
            require_convergence=self.pc_bb,
            linear_solver=gmres_solver,
            max_particle_iterations=21,
            particle_tolerance=1.0e-10,
            use_mass_matrices_jacobian=(True if (self.d1 or mm_jacobian) else None),
            use_mass_matrices_pc=mm_pc,
            pc_type=pc,
            diagnostic_file="diags/newton_diag.txt" if self.test else None,
        )

        if self.d1:
            import pywarpx

            newton = pywarpx.warpx.get_bucket("newton")
            newton.d1_operator_partition = 1
            newton.d1_step = 0
            newton.d1_iteration = 0
            newton.d1_once = 1
            newton.d1_diagnostic_file = "diags/d1_operator_partition.txt"

        # Create the theta-implicit hybrid evolve scheme
        evolve_scheme = picmi.ThetaImplicitHybridEvolveScheme(
            theta=0.5,
            nonlinear_solver=nonlinear_solver,
        )
        simulation.evolve_scheme = evolve_scheme

        #######################################################################
        # Particle types setup                                                #
        #######################################################################

        self.ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=self.M,
            initial_distribution=picmi.UniformDistribution(
                density=self.n_plasma,
                rms_velocity=[self.v_ti] * 3,
            ),
        )
        simulation.add_species(
            self.ions,
            layout=picmi.PseudoRandomLayout(
                grid=self.grid, n_macroparticles_per_cell=self.NPPC
            ),
        )

        #######################################################################
        # Add diagnostics                                                     #
        #######################################################################

        one_step_bb_diagnostic = self.test and self.pc_bb and self.total_steps == 1
        if not one_step_bb_diagnostic:
            field_diag = picmi.FieldDiagnostic(
                name="field_diag",
                grid=self.grid,
                period=self.diag_steps,
                data_list=["B", "E"],
                write_dir="diags",
                warpx_file_prefix="field_diags",
                warpx_format="openpmd",
                warpx_openpmd_backend="h5",
            )
            simulation.add_diagnostic(field_diag)

        # add particle diagnostic for checksum
        # The one-step BB diagnostic arms have checksums disabled and inspect
        # only their solver record streams, so avoid unrelated plotfile I/O.
        if self.test and not one_step_bb_diagnostic:
            part_diag = picmi.ParticleDiagnostic(
                name="diag1",
                period=self.total_steps,
                species=[self.ions],
                data_list=["ux", "uy", "uz", "weighting"],
            )
            simulation.add_diagnostic(part_diag)


##########################
# parse input parameters
##########################

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--test",
    help="toggle whether this script is run as a short CI test",
    action="store_true",
)
parser.add_argument(
    "-v",
    "--verbose",
    help="Verbose output",
    action="store_true",
)
parser.add_argument(
    "--pc-bb",
    help="use the block-banded direct preconditioner (pc_block_banded) "
    "with its verification gates enabled",
    action="store_true",
)
parser.add_argument(
    "--d1",
    help="emit one schema-v1 D1 common-direction operator partition at internal step 0, Newton iteration 0",
    action="store_true",
)
parser.add_argument(
    "--mm-jacobian",
    help="explicitly enable the mass-matrix Jacobian for the MM-PC twin",
    action="store_true",
)
parser.add_argument(
    "--mm-pc",
    choices=("default", "on", "off"),
    default="default",
    help="tri-state mass-matrix preconditioner control; default emits no setting",
)
parser.add_argument(
    "--dt-mult",
    help="time-step multiplier (>1 also enables the Newton line search)",
    type=float,
    default=1.0,
)
parser.add_argument(
    "--steps",
    help="override the number of steps in test mode",
    type=int,
    default=None,
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

mm_pc = None if args.mm_pc == "default" else args.mm_pc == "on"
if mm_pc is not None:
    if not args.test or not args.pc_bb or args.d1:
        parser.error("explicit --mm-pc requires --test --pc-bb and excludes --d1")
    if not args.mm_jacobian:
        parser.error("explicit --mm-pc requires --mm-jacobian")
    if args.steps is None or args.steps <= 0:
        parser.error("explicit --mm-pc requires a positive --steps value")
elif args.mm_jacobian:
    parser.error("--mm-jacobian is reserved for an explicit --mm-pc twin")

# Newton diagnostics are append-only during normal runs. The explicit CI twin
# owns this one-step file and must start from a clean record on every replay so
# a prior CTest invocation cannot become either evidence or a false failure.
if mm_pc is not None:
    if comm.rank == 0:
        Path("diags/newton_diag.txt").unlink(missing_ok=True)
    comm.Barrier()

run = CylindricalNormalModes(
    test=args.test,
    verbose=args.verbose,
    pc_bb=args.pc_bb or args.d1,
    d1=args.d1,
    dt_mult=args.dt_mult,
    steps=args.steps,
    mm_jacobian=args.mm_jacobian,
    mm_pc=mm_pc,
)
simulation.step()
