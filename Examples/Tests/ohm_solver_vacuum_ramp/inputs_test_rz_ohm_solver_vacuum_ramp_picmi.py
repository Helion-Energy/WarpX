#!/usr/bin/env python3
#
# --- Test script for the kinetic-fluid hybrid model: inductive ramp of a
# --- uniform axial magnetic field over a domain that sits entirely below the
# --- Ohm's-law density floor. The external field enters through a
# --- time-dependent vector potential; the entire domain is "vacuum" for the
# --- generalized Ohm's law, so the test isolates the external-drive plumbing
# --- (flux advance, absence of spurious longitudinal fields) and the drift
# --- response of a dilute ion population (betatron compression, r^2 B
# --- conserved per drift orbit).

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


class VacuumInductiveRamp(object):
    # Uniform initial field, doubled by the ramp
    B0 = 0.1  # T
    dB = 0.1  # T

    # Dilute plasma: the Ohm's-law density floor sits far ABOVE the actual
    # density everywhere, so every cell takes the vacuum branch of the
    # generalized Ohm's law for the entire run and the ions respond as test
    # particles. The floored electron back-EMF scales as n/n_floor, so the
    # large factor keeps the spurious drift correction at the percent level
    # (it also weakens the floored whistler response, widening the Picard
    # contraction margin).
    n0 = 1e20  # m^-3
    n_floor_fac = 64.0
    T_i = 0.5  # eV
    T_e = 0.5

    # Domain
    LR = 0.5  # m
    LZ = 0.125  # m

    # Ramp timing (ion cyclotron periods at B0) and run length
    t0_ramp_ci = 1.0
    tau_ramp_ci = 4.0
    LT = 6.0

    # Time step (ion cyclotron periods at B0): keeps the vacuum whistler
    # phase advance theta*(k_max*l_i_floor)**2*omega_ci*dt below ~0.5
    # through the end of the ramp on the default grid.
    DT = 0.015

    NR = 32
    NZ = 8
    NPPC = 16

    substeps = 10

    def __init__(
        self,
        test,
        verbose,
        implicit=False,
        darwin=False,
        recovery=False,
        recovery_cadence="half",
    ):
        self.test = test
        self.verbose = verbose or self.test
        self.implicit = implicit
        self.darwin = darwin
        self.recovery = recovery
        self.recovery_cadence = recovery_cadence

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
                f"\tl_i = {self.l_i:.2e} m (floored: {self.l_i_floor:.2e} m)\n"
                f"\tt_ci = {self.t_ci:.2e} s\n"
                f"\tdt = {self.dt:.2e} s\n"
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
        self.l_i_floor = self.l_i / np.sqrt(self.n_floor_fac)

        self.vi_th = np.sqrt(self.T_i * constants.q_e / self.M)

    def load_fields(self):
        Br = simulation.fields.get("Bfield_fp_external", dir="r", level=0)
        Bt = simulation.fields.get("Bfield_fp_external", dir="theta", level=0)
        Bz = simulation.fields.get("Bfield_fp_external", dir="z", level=0)

        Br[:, :] = 0.0
        Bt[:, :] = 0.0
        Bz[:, :] = self.B0
        comm.Barrier()

    def setup_run(self):
        """Setup simulation components."""

        self.grid = picmi.CylindricalGrid(
            number_of_cells=[self.NR, self.NZ],
            lower_bound=[0.0, -self.LZ / 2.0],
            upper_bound=[self.LR, self.LZ / 2.0],
            lower_boundary_conditions=["none", "periodic"],
            upper_boundary_conditions=["dirichlet", "periodic"],
            lower_boundary_conditions_particles=["none", "periodic"],
            upper_boundary_conditions_particles=["absorbing", "periodic"],
            warpx_max_grid_size=self.NZ,
        )
        simulation.time_step_size = self.dt
        simulation.max_steps = self.total_steps
        simulation.current_deposition_algo = "direct"
        simulation.particle_shape = 1
        simulation.use_filter = True
        simulation.verbose = self.verbose

        # Uniform compression field ramped with a sigmoid in time; only the
        # vector potential of the CHANGE is prescribed (the initial B0 is
        # loaded directly).
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

        # With the Darwin (unified) drive the external flux enters through
        # the boundary values of the evolved vector potential and must
        # PENETRATE the low-density halo. A vacuum resistivity turns the
        # halo into a fast resistive diffuser (Hewett, J. Comput. Phys. 38
        # (1980): eta -> large in vacuum regions), giving a penetration
        # time mu0*R^2/eta ~ 0.25 t_ci << tau_ramp. The split-field drive
        # (explicit and non-Darwin implicit) imposes the external field
        # volumetrically instead, so the plasma value suffices there.
        # The vacuum A recovery replaces the transport problem outright
        # (the masked cells take the magnetostatic solution each
        # application), so the recovery variant runs at the PLASMA eta:
        # the ramp asserts then prove the recovery delivers the flux.
        eta = 2.0 if (self.darwin and not self.recovery) else 1e-6

        recovery_kwargs = {}
        if self.recovery:
            recovery_kwargs = dict(
                darwin_vacuum_recovery=True,
                darwin_vacuum_recovery_cadence=self.recovery_cadence,
            )

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
            **recovery_kwargs,
        )
        simulation.solver = self.solver

        B_ext = picmi.LoadInitialFieldFromPython(
            load_from_python=self.load_fields,
            warpx_do_divb_cleaning_external=True,
            load_B=True,
            load_E=False,
        )
        simulation.add_applied_field(B_ext)

        if self.implicit:
            import pywarpx

            pywarpx.particles.max_grid_crossings = (
                int(np.ceil(0.5 * self.DT / 1.0e-3)) + 1
            )
            if self.darwin:
                # The E_L asserts run at the percent level; the default
                # 1e-10 Poisson tolerance only slows the CI run down.
                pywarpx.hybridpicmodel.darwin_poisson_relative_tolerance = 1.0e-8
            if self.darwin:
                # The vacuum resistivity puts the resistive gain
                # eta*k_max^2*theta*dt/mu0 (~300) far beyond the Picard
                # contraction bound, so the resistive diffusion must be
                # solved with Newton-GMRES. The Krylov space needs the
                # full diffusion spectrum: restart-free GMRES.
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
                nonlinear_solver = picmi.PicardNonlinearSolver(
                    verbose=self.verbose,
                    max_iterations=100,
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
        particle_diag = picmi.ParticleDiagnostic(
            name="diag1",
            period=self.diag_steps,
            species=[self.ions],
            data_list=["x", "z", "weighting"],
            write_dir="diags",
            warpx_format="plotfile",
        )
        simulation.add_diagnostic(particle_diag)

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
    "--implicit",
    help="use the theta-implicit hybrid evolve scheme",
    action="store_true",
)
parser.add_argument(
    "--darwin",
    help="use the Darwin (magnetoinductive) field split (implicit only)",
    action="store_true",
)
parser.add_argument(
    "--recovery",
    help="enable the vacuum vector-potential recovery (darwin only); the "
    "deck then runs at the plasma resistivity instead of the Hewett "
    "vacuum value, so the ramp asserts prove the recovery delivers the "
    "external flux",
    action="store_true",
)
parser.add_argument(
    "--recovery-cadence",
    help="vacuum recovery cadence",
    choices=["half", "full"],
    default="half",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = VacuumInductiveRamp(
    test=args.test,
    verbose=args.verbose,
    implicit=args.implicit,
    darwin=args.darwin,
    recovery=args.recovery,
    recovery_cadence=args.recovery_cadence,
)
simulation.step()
