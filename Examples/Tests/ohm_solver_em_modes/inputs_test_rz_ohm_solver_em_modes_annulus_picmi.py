#!/usr/bin/env python3
#
# --- Test script for the kinetic-fluid hybrid model in WarpX wherein ions are
# --- treated as kinetic particles and electrons as an isothermal, inertialess
# --- background fluid. Variant of the cylindrical normal-modes test with a
# --- VACUUM ANNULUS: the uniform plasma column fills r < R_p = 0.5 r_max and
# --- true vacuum (no particles, no density floor pedestal) fills the annulus
# --- out to the conducting wall at r_max. The script is set up to produce
# --- parallel normal EM modes of the partially filled metallic cylinder and
# --- is run in RZ geometry.
# --- With --esolve je_form the run exercises the Je-form generalized-Ohm's-law
# --- solve (dynamical electron current, relax advance), whose vacuum limit is
# --- regular by construction; --je-vacuum selects the vacuum treatment
# --- (uniform deck resistivity, or the local vacuum current-decay term with
# --- zero resistivity).
# --- As a CI test only a small number of steps are taken.

import argparse
import sys

import dill
import numpy as np
from mpi4py import MPI as mpi

import pywarpx
from pywarpx import picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(verbose=0)


class CylindricalNormalModesAnnulus(object):
    """The following runs a simulation of a plasma column at a set ion
    temperature (and Te = 0) with an external magnetic field applied in the
    z-direction (parallel to domain), surrounded by a vacuum annulus and a
    conducting wall.
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
    LT = 200.0  # Simulation temporal length (ion cyclotron periods)

    # Numerical parameters
    NPPC = 8000  # Seed number of particles per cell (in the column)
    DZ = 0.4  # Cell size (ion skin depths)
    DR = 0.4  # Cell size (ion skin depths)
    DT = 0.02  # Time step (ion cyclotron periods)

    # Plasma resistivity - used to dampen the mode excitation
    eta = 5e-4
    # Number of substeps used to update B
    substeps = 40

    def __init__(self, test, verbose, esolve="e_form", je_vacuum="eta"):
        """Get input parameters for the specific case desired."""
        self.test = test
        self.verbose = verbose or self.test
        self.esolve = esolve
        self.je_vacuum = je_vacuum

        # calculate various plasma parameters based on the simulation input
        self.get_plasma_quantities()

        if not self.test:
            self.total_steps = int(self.LT / self.DT)
        else:
            # if this is a test case run for only a small number of steps
            self.total_steps = 100
            # and make the grid and particle count smaller
            self.Nz = 128
            self.Nr = 64
            self.NPPC = 200
        # output diagnostics 5 times per cyclotron period
        self.diag_steps = max(10, int(1.0 / 5 / self.DT))

        self.Lz = self.Nz * self.DZ * self.l_i
        self.Lr = self.Nr * self.DR * self.l_i

        # plasma column radius: the vacuum annulus fills R_p .. Lr with the
        # conducting wall at Lr; the column edge is smoothed over one cell
        self.R_p = 0.5 * self.Lr
        self.delta_p = self.DR * self.l_i

        # the Je-form vacuum treatment: with the local current-decay term
        # armed the deck resistivity is removed entirely
        if self.esolve == "je_form" and self.je_vacuum == "gamma":
            self.eta = 0.0

        self.dt = self.DT * self.t_ci

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
                f"Column geometry:\n"
                f"\tR_p = {self.R_p:.4e} m ({self.R_p / self.Lr:.2f} * r_max)\n"
                f"\tdelta_p = {self.delta_p:.4e} m\n"
                f"\tr_max (wall) = {self.Lr:.4e} m\n"
            )
            print(
                f"Numerical parameters:\n"
                f"\tdt = {self.dt:.1e} s\n"
                f"\tdiag steps = {self.diag_steps:d}\n"
                f"\ttotal steps = {self.total_steps:d}\n",
                flush=True,
            )
        self.setup_run()

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

    def setup_run(self):
        """Setup simulation components."""

        #######################################################################
        # Set geometry and boundary conditions                                #
        #######################################################################

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
        # Particle types setup                                                #
        #######################################################################

        self.ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=self.M,
            initial_distribution=picmi.AnalyticDistribution(
                density_expression="n0_p/(1+exp((sqrt(x*x+y*y)-R_p)/delta_p))",
                momentum_expressions=["0", "0", "0"],
                warpx_momentum_spread_expressions=[f"{self.v_ti}"] * 3,
                warpx_density_min=0.01 * self.n_plasma,
                R_p=self.R_p,
                delta_p=self.delta_p,
                n0_p=self.n_plasma,
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

        # add field and particle diagnostics for checksum
        if self.test:
            checksum_field_diag = picmi.FieldDiagnostic(
                name="diag1",
                grid=self.grid,
                period=self.total_steps,
                data_list=["B", "E"],
            )
            simulation.add_diagnostic(checksum_field_diag)

            checksum_part_diag = picmi.ParticleDiagnostic(
                name="diag1",
                period=self.total_steps,
                species=[self.ions],
                data_list=["ux", "uy", "uz", "weighting"],
            )
            simulation.add_diagnostic(checksum_part_diag)


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
    "--esolve",
    choices=["e_form", "je_form"],
    default="e_form",
    help="E-solve form (hybrid_pic_model.esolve): e_form = the legacy "
    "algebraic solve with its density floor, je_form = the dynamical "
    "electron-current solve (relax advance, collocated grid) whose "
    "vacuum limit needs no floor",
)
parser.add_argument(
    "--je-vacuum",
    choices=["eta", "gamma"],
    default="eta",
    help="je_form vacuum treatment: eta = keep the uniform deck "
    "resistivity, gamma = zero resistivity with the local vacuum "
    "current-decay term (hybrid_pic_model.je_vacuum_gamma_frac = 0.5)",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

if args.esolve == "je_form":
    # the Je-form relax advance requires a collocated grid
    simulation = picmi.Simulation(verbose=0, warpx_grid_type="collocated")

run = CylindricalNormalModesAnnulus(
    test=args.test, verbose=args.verbose, esolve=args.esolve, je_vacuum=args.je_vacuum
)
simulation.initialize_inputs()
if args.esolve == "je_form":
    pywarpx.hybridpicmodel.esolve = "je_form"
    # the one-count density level (one macroparticle's deposit): the
    # contract value for je_n_min -- never lower
    pywarpx.hybridpicmodel.je_n_min = run.n_plasma / run.NPPC
    if args.je_vacuum == "gamma":
        pywarpx.hybridpicmodel.je_vacuum_gamma_frac = 0.5
simulation.initialize_warpx()
simulation.step()
