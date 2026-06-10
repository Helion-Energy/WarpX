#!/usr/bin/env python3
#
# --- Test script for the global vector potential (A) recovery option of the
# --- kinetic-fluid hybrid model in WarpX. A current-carrying plasma column
# --- (z-pinch) is surrounded by vacuum. The azimuthal magnetic field in the
# --- vacuum region is recovered from the global vector Poisson solve
# --- del^2 A = -mu0*J and must follow the analytic mu0*I/(2*pi*r) profile.

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


class PlasmaZPinch(object):
    # Plasma parameters
    n0 = 1e20  # plasma density (m^-3)
    T_i = 10.0  # ion temperature (eV)
    T_e = 10.0  # electron temperature (eV)

    # Magnetic field strength at the plasma surface, chosen so that the
    # magnetic pressure is comparable to the plasma pressure
    B_a = np.sqrt(2.0 * constants.mu0 * n0 * constants.q_e * (10.0 + 10.0))

    # Domain size (m)
    LR = 0.5
    LZ = 0.125

    # Plasma column radius (m) and edge width (m)
    R_p = 0.4 * LR
    delta_p = 0.01

    # Resolution and macroparticle count
    NR = 48
    NZ = 32
    NPPC = 32

    # Simulation length and timestep in ion cyclotron periods
    DT = 1e-3

    def __init__(self, test, verbose):
        self.test = test
        self.verbose = verbose or self.test

        self.DR = self.LR / self.NR
        self.DZ = self.LZ / self.NZ

        # Total axial current implied by the initial azimuthal field
        # B_theta(R_p) = B_a
        self.I_total = 2.0 * np.pi * self.R_p * self.B_a / constants.mu0
        # Uniform current density inside the column
        self.J_z = self.I_total / (np.pi * self.R_p**2)
        # Ion drift speed carrying the current
        self.v_drift = self.J_z / (self.n0 * constants.q_e)

        # calculate various plasma parameters based on the simulation input
        self.get_plasma_quantities()

        self.dt = self.DT * self.t_ci

        self.total_steps = 10
        self.diag_steps = self.total_steps

        # print out plasma parameters
        if comm.rank == 0:
            print(
                f"Initializing simulation with input parameters:\n"
                f"\tTi = {self.T_i:.1f} eV\n"
                f"\tn0 = {self.n0:.1e} m^-3\n"
                f"\tB_a = {self.B_a:.3f} T\n"
                f"\tI = {self.I_total:.3e} A\n"
                f"\tv_drift = {self.v_drift:.3e} m/s\n"
            )
            print(
                f"Plasma parameters:\n"
                f"\tl_i = {self.l_i:.1e} m\n"
                f"\tt_ci = {self.t_ci:.1e} s\n"
                f"\tv_ti = {self.vi_th:.1e} m/s\n"
                f"\tvA = {self.vA:.1e} m/s\n"
            )
            print(
                f"Numerical parameters:\n"
                f"\tdr = {self.DR / self.l_i:.3f} c/w_pi\n"
                f"\tdt = {self.dt:.1e} s\n"
                f"\ttotal steps = {self.total_steps:d}\n"
            )

        self.setup_run()

    def get_plasma_quantities(self):
        """Calculate various plasma parameters based on the simulation input."""

        # Ion mass (kg)
        self.M = constants.m_p

        # Cyclotron angular frequency (rad/s) and period (s)
        self.w_ci = constants.q_e * abs(self.B_a) / self.M
        self.t_ci = 2.0 * np.pi / self.w_ci

        # Ion plasma frequency (Hz)
        self.w_pi = np.sqrt(constants.q_e**2 * self.n0 / (self.M * constants.ep0))

        # Ion skin depth (m)
        self.l_i = constants.c / self.w_pi

        # Alfven speed (m/s)
        self.vA = abs(self.B_a) / np.sqrt(
            constants.mu0 * self.n0 * (constants.m_e + self.M)
        )

        # calculate thermal speeds
        self.vi_th = np.sqrt(self.T_i * constants.q_e / self.M)

    def Bt(self, r):
        """Initial azimuthal magnetic field of the uniform-current column.

        The far vacuum region (r > 1.4*R_p, where the deposited charge
        density is strictly zero) is deliberately initialized with a *wrong*
        profile (a constant continuation instead of the analytic B_a*R_p/r
        decay of the enclosed current). The local field update has no
        mechanism to correct this, while the global A recovery determines
        the vacuum field from the plasma current alone and must restore the
        analytic 1/r profile, which is what the analysis script checks.
        """
        r = np.maximum(r, 1e-12)
        r_w = 1.4 * self.R_p
        return np.where(
            r <= self.R_p,
            self.B_a * r / self.R_p,
            self.B_a * self.R_p / np.minimum(r, r_w),
        )

    def load_fields(self):
        Br = simulation.fields.get("Bfield_fp_external", dir="r", level=0)
        Bt = simulation.fields.get("Bfield_fp_external", dir="theta", level=0)
        Bz = simulation.fields.get("Bfield_fp_external", dir="z", level=0)

        Br[:, :] = 0.0
        Bz[:, :] = 0.0

        RM, ZM = np.meshgrid(Bt.mesh("r"), Bt.mesh("z"), indexing="ij")
        Bt[:, :] = self.Bt(RM)
        comm.Barrier()

    def setup_run(self):
        """Setup simulation components."""

        #######################################################################
        # Set geometry and boundary conditions                                #
        #######################################################################

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

        #######################################################################
        # Field solver                                                        #
        #######################################################################

        self.solver = picmi.HybridPICSolver(
            grid=self.grid,
            gamma=5.0 / 3.0,
            Te=self.T_e,
            n0=self.n0,
            n_floor=0.05 * self.n0,
            plasma_resistivity=1e-4 * constants.mu0 * self.LR * self.vA,
            plasma_hyper_resistivity=1e-9,
            substeps=10,
            use_global_A_recovery=True,
        )
        simulation.solver = self.solver

        # Add field loader callback for the initial pinch field
        B_init = picmi.LoadInitialFieldFromPython(
            load_from_python=self.load_fields,
            warpx_do_divb_cleaning_external=True,
            load_B=True,
            load_E=False,
        )
        simulation.add_applied_field(B_init)

        #######################################################################
        # Particle types setup                                                #
        #######################################################################

        self.ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=self.M,
            initial_distribution=picmi.AnalyticDistribution(
                density_expression="n0_p/(1+exp((sqrt(x*x+y*y)-R_p)/delta_p))",
                momentum_expressions=["0", "0", f"{self.v_drift}"],
                warpx_momentum_spread_expressions=[f"{self.vi_th}"] * 3,
                warpx_density_min=0.05 * self.n0,
                R_p=self.R_p,
                delta_p=self.delta_p,
                n0_p=self.n0,
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

        # Initialize inputs and WarpX instance
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
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = PlasmaZPinch(test=args.test, verbose=args.verbose)
simulation.step()
