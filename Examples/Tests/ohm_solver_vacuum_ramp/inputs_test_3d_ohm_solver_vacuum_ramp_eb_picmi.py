#!/usr/bin/env python3
#
# --- Test script for the kinetic-fluid hybrid model: inductive ramp of a
# --- uniform axial magnetic field in 3D Cartesian geometry with an embedded
# --- conducting cylinder along z. The domain sits below the Ohm's-law
# --- density floor, so the run isolates the external-drive plumbing in 3D
# --- and the stair-step embedded-boundary response: the flux threading the
# --- conductor stays frozen (eddy/image currents at the masked surface)
# --- while the far exterior tracks the programmed ramp.
# ---
# --- Validation deck, run manually (see analysis_vacuum_ramp_eb.py): the
# --- Darwin variant needs Newton + restart-free GMRES for the vacuum
# --- resistivity and runs ~1 h on 4 ranks, so it is not registered as a
# --- CI test until the hybrid preconditioner lands.

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI as mpi

from pywarpx import picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=False)


class VacuumRampEB3d(object):
    # Uniform initial field along z, doubled by the ramp
    B0 = 0.1  # T
    dB = 0.1  # T

    # Dilute plasma far below the Ohm's-law density floor: every cell takes
    # the vacuum branch of the generalized Ohm's law and the ions respond as
    # test particles (see the RZ vacuum-ramp deck for the floor rationale).
    n0 = 1e20  # m^-3
    n_floor_fac = 64.0
    T_i = 0.5  # eV
    T_e = 0.5

    # Domain (m) and embedded conducting cylinder along z
    L_perp = 0.5
    L_z = 0.125
    R_cond = 0.08

    # Ramp timing (ion cyclotron periods at B0) and run length
    t0_ramp_ci = 1.0
    tau_ramp_ci = 4.0
    LT = 6.0
    # The slowest Picard mode of the embedded-boundary configuration scales
    # with omega_ci*dt and crosses unity near the end of the ramp (B ->
    # 2 B0) at DT = 0.015; this value keeps it contracting through the
    # full doubling.
    DT = 0.010

    N_perp = 32
    NZ = 8
    NPPC = 8

    substeps = 10

    def __init__(self, test, verbose, darwin=False):
        self.test = test
        self.verbose = verbose or self.test
        self.darwin = darwin
        if self.test:
            # CI-sized window: exercises the drive, the embedded conductor
            # and the flux bookkeeping through the onset of the ramp.
            self.LT = 4.0

        self.get_plasma_quantities()

        self.dt = self.DT * self.t_ci
        self.t0_ramp = self.t0_ramp_ci * self.t_ci
        self.tau_ramp = self.tau_ramp_ci * self.t_ci
        self.total_steps = int(self.LT / self.DT)
        self.diag_steps = self.total_steps // 5

        if comm.rank == 0:
            print(
                f"Initializing simulation with input parameters:\n"
                f"\tTi = {self.T_i:.2f} eV\n"
                f"\tn0 = {self.n0:.1e} m^-3\n"
                f"\tB0 = {self.B0:.2f} T -> {self.B0 + self.dB:.2f} T\n"
                f"\tconductor radius = {self.R_cond:.3f} m\n"
                f"\tt_ci = {self.t_ci:.2e} s\n"
                f"\ttotal steps = {self.total_steps:d}\n"
            )

        self.setup_run()

    def get_plasma_quantities(self):
        """Calculate various plasma parameters based on the simulation input."""
        self.M = constants.m_p

        self.w_ci = constants.q_e * abs(self.B0) / self.M
        self.t_ci = 2.0 * np.pi / self.w_ci

        self.w_pi = np.sqrt(constants.q_e**2 * self.n0 / (self.M * constants.ep0))
        self.l_i = constants.c / self.w_pi

        self.vi_th = np.sqrt(self.T_i * constants.q_e / self.M)

    def load_fields(self):
        Bx = simulation.fields.get("Bfield_fp_external", dir="x", level=0)
        By = simulation.fields.get("Bfield_fp_external", dir="y", level=0)
        Bz = simulation.fields.get("Bfield_fp_external", dir="z", level=0)

        Bx[:, :, :] = 0.0
        By[:, :, :] = 0.0
        Bz[:, :, :] = self.B0
        comm.Barrier()

    def setup_run(self):
        """Setup simulation components."""

        self.grid = picmi.Cartesian3DGrid(
            number_of_cells=[self.N_perp, self.N_perp, self.NZ],
            lower_bound=[-self.L_perp / 2.0, -self.L_perp / 2.0, -self.L_z / 2.0],
            upper_bound=[self.L_perp / 2.0, self.L_perp / 2.0, self.L_z / 2.0],
            lower_boundary_conditions=["dirichlet", "dirichlet", "periodic"],
            upper_boundary_conditions=["dirichlet", "dirichlet", "periodic"],
            lower_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
            upper_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
            warpx_max_grid_size=self.N_perp // 2,
        )
        simulation.time_step_size = self.dt
        simulation.max_steps = self.total_steps
        simulation.current_deposition_algo = "direct"
        simulation.particle_shape = 1
        simulation.use_filter = True
        simulation.verbose = self.verbose

        # Embedded conducting cylinder along z (positive inside the body)
        simulation.embedded_boundary = picmi.EmbeddedBoundary(
            implicit_function=f"-(x**2 + y**2 - {self.R_cond}**2)"
        )

        # Uniform compression field ramped with a sigmoid in time
        A_ext = {
            "uniform_analytical": {
                "Ax_external_function": f"-0.5*y*{self.dB}",
                "Ay_external_function": f"0.5*x*{self.dB}",
                "Az_external_function": "0",
                "A_time_external_function": (
                    "1/(1+exp(5*(1-(t-t0_ramp)*sqrt(2)/tau_ramp)))"
                ),
            },
        }

        # Darwin unified drive: the external flux enters through the
        # boundary values of the evolved vector potential and diffuses
        # inward through a resistive halo (Hewett, J. Comput. Phys. 38
        # (1980) vacuum-resistivity treatment); mu0*L^2/eta ~ 0.25 t_ci.
        # The split-field (non-Darwin) drive imposes the field
        # volumetrically and keeps the plasma resistivity.
        eta = 2.0 if self.darwin else 1e-6

        self.solver = picmi.HybridPICSolver(
            grid=self.grid,
            gamma=1.0,
            Te=self.T_e,
            n0=self.n0,
            n_floor=self.n_floor_fac * self.n0,
            plasma_resistivity=eta,
            substeps=self.substeps,
            A_external=A_ext,
            tau_ramp=self.tau_ramp,
            t0_ramp=self.t0_ramp,
            darwin=self.darwin,
        )
        simulation.solver = self.solver

        B_ext = picmi.LoadInitialFieldFromPython(
            load_from_python=self.load_fields,
            warpx_do_divb_cleaning_external=True,
            load_B=True,
            load_E=False,
        )
        simulation.add_applied_field(B_ext)

        import pywarpx

        pywarpx.particles.max_grid_crossings = (
            int(np.ceil(0.5 * self.DT / 1.0e-3)) + 1
        )
        if self.darwin:
            pywarpx.hybridpicmodel.darwin_poisson_rtol = 1.0e-8
        if self.darwin:
            # The vacuum resistivity puts the resistive gain far beyond
            # the Picard contraction bound: Newton with restart-free GMRES.
            nonlinear_solver = picmi.NewtonNonlinearSolver(
                verbose=self.verbose,
                max_iterations=20,
                relative_tolerance=1.0e-6,
                absolute_tolerance=0.0,
                require_convergence=True,
                linear_solver=picmi.GMRESLinearSolver(
                    verbose_int=0,
                    max_iterations=600,
                    restart_length=600,
                    relative_tolerance=1.0e-6,
                    absolute_tolerance=0.0,
                ),
                max_particle_iterations=21,
                particle_tolerance=1.0e-10,
            )
        else:
            # The masked embedded-boundary cells slow the Picard
            # contraction to ~0.92 per iteration on this configuration
            # (worsening toward ~0.96 as the ramp doubles B); the
            # iterations are cheap at this grid size, so give the
            # (steadily converging) fixed point room.
            nonlinear_solver = picmi.PicardNonlinearSolver(
                verbose=self.verbose,
                max_iterations=600,
                relative_tolerance=1.0e-6,
                absolute_tolerance=0.0,
                require_convergence=True,
            )
        simulation.evolve_scheme = picmi.ThetaImplicitHybridEvolveScheme(
            theta=0.5,
            nonlinear_solver=nonlinear_solver,
        )

        #######################################################################
        # Particle types setup                                                #
        #######################################################################

        self.ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=self.M,
            initial_distribution=picmi.UniformDistribution(
                density=self.n0,
                rms_velocity=[self.vi_th] * 3,
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
            name="diag1",
            grid=self.grid,
            period=self.diag_steps,
            data_list=["B", "E", "rho"],
            write_dir="diags",
            warpx_format="plotfile",
        )
        simulation.add_diagnostic(field_diag)

        #######################################################################
        # Initialize                                                          #
        #######################################################################

        if comm.rank == 0:
            if Path.exists(Path("diags")):
                shutil.rmtree("diags")
            Path("diags").mkdir(parents=True, exist_ok=True)

        simulation.initialize_inputs()
        simulation.initialize_warpx()


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
    "--darwin",
    help="use the Darwin (magnetoinductive) field split",
    action="store_true",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = VacuumRampEB3d(test=args.test, verbose=args.verbose, darwin=args.darwin)
simulation.step()
