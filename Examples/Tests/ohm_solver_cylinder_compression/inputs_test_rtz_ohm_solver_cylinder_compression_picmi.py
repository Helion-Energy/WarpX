#!/usr/bin/env python3
#
# --- Test script for the kinetic-fluid hybrid model in WarpX wherein ions are
# --- treated as kinetic particles and electrons as an isothermal, inertialess
# --- background fluid. This is the RTZ (3D real-space cylindrical r-theta-z)
# --- variant of the plasma-cylinder adiabatic-compression test: the same
# --- axisymmetric Grad-Shafranov-like setup is run on a full 3D cylindrical grid
# --- (theta is a real discretized coordinate, not an azimuthal-mode
# --- decomposition). With theta-uniform initial conditions the result should
# --- closely reproduce the RZ (m=0) case; it additionally exercises the new
# --- theta-derivative operators of the RTZ Ohm's-law hybrid solver.

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


class PlasmaCylinderCompression(object):
    # B0 is chosen with all other quantities scaled by it
    n0 = 1e20
    T_i = 10  # eV
    T_e = 10
    p0 = n0 * constants.q_e * (T_i + T_e)

    B0 = np.sqrt(2 * constants.mu0 * p0)  # External magnetic field strength (T)

    # Do a 2x uniform B-field compression
    dB = B0

    # Flux Conserver radius
    R_c = 0.5

    # Plasma Radius (These values control the analytical GS solution)
    R_p = 0.25
    delta_p = 0.025

    # Domain parameters
    LR = R_c  # m
    LZ = 0.25 * R_c  # m
    # theta domain: full 2*pi, in [-pi, pi) (matches the atan2 range used to store
    # the particle azimuthal position) with periodic boundaries
    LT = 2.0 * np.pi  # radians

    LT_cyc = 10  # ion cyclotron periods (total run time, full run)
    DT = 1e-3  # ion cyclotron periods

    # Resolution parameters
    NR = 128
    NT = 16  # number of azimuthal (theta) cells
    NZ = 32

    # Starting number of particles per cell
    NPPC = 8

    # Number of substeps used to update B
    substeps = 60

    def Bz(self, r):
        return np.sqrt(
            self.B0**2
            - 2.0
            * constants.mu0
            * self.n0
            * constants.q_e
            * (self.T_i + self.T_e)
            / (1.0 + np.exp((r - self.R_p) / self.delta_p))
        )

    def __init__(self, test, verbose):
        self.test = test
        self.verbose = verbose or self.test

        self.Lr = self.LR
        self.Lz = self.LZ

        self.DR = self.LR / self.NR
        self.DZ = self.LZ / self.NZ

        # calculate various plasma parameters based on the simulation input
        self.get_plasma_quantities()

        self.dt = self.DT * self.t_ci

        # run very low resolution as a CI test
        if self.test:
            self.total_steps = 20
            self.diag_steps = self.total_steps // 5
            self.NR = 32
            self.NT = 8
            self.NZ = 16
        else:
            self.total_steps = int(self.LT_cyc / self.DT)
            self.diag_steps = 100

        self.DR = self.LR / self.NR
        self.DZ = self.LZ / self.NZ

        if comm.rank == 0:
            print(
                f"Initializing RTZ simulation with input parameters:\n"
                f"\tTi = {self.T_i:.1f} eV\n"
                f"\tn0 = {self.n0:.1e} m^-3\n"
                f"\tB0 = {self.B0:.2f} T\n"
                f"\tNR x NT x NZ = {self.NR} x {self.NT} x {self.NZ}\n"
                f"\tdt = {self.dt:.1e} s\n"
                f"\ttotal steps = {self.total_steps:d}\n"
            )

        self.setup_run()

    def get_plasma_quantities(self):
        """Calculate various plasma parameters based on the simulation input."""
        # Ion mass (kg)
        self.M = constants.m_p
        # Cyclotron angular frequency (rad/s) and period (s)
        self.w_ci = constants.q_e * abs(self.B0) / self.M
        self.t_ci = 2.0 * np.pi / self.w_ci
        # Ion plasma frequency (Hz)
        self.w_pi = np.sqrt(constants.q_e**2 * self.n0 / (self.M * constants.ep0))
        # Ion skin depth (m)
        self.l_i = constants.c / self.w_pi
        # Alfven speed (m/s)
        self.vA = abs(self.B0) / np.sqrt(
            constants.mu0 * self.n0 * (constants.m_e + self.M)
        )
        # thermal speed
        self.vi_th = np.sqrt(self.T_i * constants.q_e / self.M)
        # Ion Larmor radius (m)
        self.rho_i = self.vi_th / self.w_ci

    def load_fields(self):
        # External B field on the full 3D cylindrical (r, theta, z) grid.
        # The initial equilibrium is axisymmetric, so Bz depends only on r.
        Br = simulation.fields.get("Bfield_fp_external", dir="r", level=0)
        Bt = simulation.fields.get("Bfield_fp_external", dir="theta", level=0)
        Bz = simulation.fields.get("Bfield_fp_external", dir="z", level=0)

        Br[...] = 0.0
        Bt[...] = 0.0

        RM, TM, ZM = np.meshgrid(
            Bz.mesh("r"), Bz.mesh("theta"), Bz.mesh("z"), indexing="ij"
        )
        Bz[...] = self.Bz(RM) * (RM <= self.R_c)
        comm.Barrier()

    def setup_run(self):
        """Setup simulation components."""

        #######################################################################
        # Set geometry and boundary conditions                                #
        #######################################################################

        # Full 3D cylindrical grid: axes are (r, theta, z).
        self.grid = picmi.CylindricalGrid3D(
            number_of_cells=[self.NR, self.NT, self.NZ],
            lower_bound=[0.0, -np.pi, -self.Lz / 2.0],
            upper_bound=[self.Lr, np.pi, self.Lz / 2.0],
            lower_boundary_conditions=["none", "periodic", "periodic"],
            upper_boundary_conditions=["dirichlet", "periodic", "periodic"],
            lower_boundary_conditions_particles=["none", "periodic", "periodic"],
            upper_boundary_conditions_particles=["absorbing", "periodic", "periodic"],
            warpx_max_grid_size=self.NZ,
        )
        simulation.time_step_size = self.dt
        simulation.max_steps = self.total_steps
        simulation.current_deposition_algo = "direct"
        simulation.particle_shape = 1
        simulation.use_filter = True
        simulation.verbose = self.verbose

        #######################################################################
        # Field solver and external field                                     #
        #######################################################################
        # Uniform axial compression field, ramped in time, applied analytically.
        # A = 0.25*dB*(-y, x, 0) -> B = curl(A) = (0, 0, 0.5*dB) uniform Bz.
        A_ext = {
            "uniform_analytical": {
                # In RTZ the parser's (x, y) are the GRID coordinates (r, theta):
                # the Cartesian-form Ar = -0.25*y*dB would be a theta-dependent,
                # seam-discontinuous A_r (harmless only while the curl-A theta
                # terms were missing). The uniform-Bz drive needs only A_theta.
                "Ax_external_function": "0",
                "Ay_external_function": f"0.25*x*{self.dB}",
                "Az_external_function": "0",
                "A_time_external_function": "1/(1+exp(5*(1-(t-t0_ramp)*sqrt(2)/tau_ramp)))",
            },
        }

        self.solver = picmi.HybridPICSolver(
            grid=self.grid,
            gamma=5.0 / 3.0,
            Te=self.T_e,
            n0=self.n0,
            n_floor=0.05 * self.n0,
            plasma_resistivity=1e-4 * constants.mu0 * self.R_c * self.vA,
            plasma_hyper_resistivity=1e-9,
            substeps=self.substeps,
            A_external=A_ext,
            tau_ramp=20e-6,
            t0_ramp=5e-6,
            # The analytical uniform-Bz vector potential is already divergence
            # free, and the projection div-cleaner is not yet implemented for the
            # RTZ (3D cylindrical) grid (it would require a 3D cylindrical Poisson
            # solve); disable it here. See the RZ case for the cleaned path.
            do_external_diva_cleaning=False,
        )
        simulation.solver = self.solver

        # Add field loader callback for the initial equilibrium B field
        B_ext = picmi.LoadInitialFieldFromPython(
            load_from_python=self.load_fields,
            # The loaded equilibrium B is axisymmetric and z-independent (only Bz(r)
            # is nonzero), so it is already divergence free. The projection
            # div-cleaner is not yet implemented for the RTZ (3D cylindrical) grid
            # (it would require a 3D cylindrical Poisson solve), so disable it.
            warpx_do_initial_div_cleaning=False,
            load_B=True,
            load_E=False,
        )
        simulation.add_applied_field(B_ext)

        #######################################################################
        # Particle types setup                                                #
        #######################################################################
        r_omega = "(sqrt(x*x+y*y)*q_e*B0/m_p)"
        dlnndr = "((-1/delta_p)/(1+exp(-(sqrt(x*x+y*y)-R_p)/delta_p)))"
        vth = f"0.5*(-{r_omega}+sqrt({r_omega}*{r_omega}+4*q_e*T_i*{dlnndr}/m_p))"

        momentum_expr = [f"y*{vth}", f"-x*{vth}", "0"]

        self.ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=self.M,
            warpx_do_temperature_deposition=True,
            initial_distribution=picmi.AnalyticDistribution(
                density_expression="n0_p/(1+exp((sqrt(x*x+y*y)-R_p)/delta_p))",
                momentum_expressions=momentum_expr,
                warpx_momentum_spread_expressions=[f"{str(self.vi_th)}"] * 3,
                warpx_density_min=0.05 * self.n0,
                R_p=self.R_p,
                delta_p=self.delta_p,
                n0_p=self.n0,
                B0=self.B0,
                T_i=self.T_i,
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

        particle_diag = picmi.ParticleDiagnostic(
            name="diag1",
            period=self.diag_steps,
            species=[self.ions],
            data_list=["ux", "uy", "uz", "x", "y", "z", "weighting"],
            write_dir="diags",
            warpx_format="plotfile",
        )
        simulation.add_diagnostic(particle_diag)
        field_diag = picmi.FieldDiagnostic(
            name="diag1",
            grid=self.grid,
            period=self.diag_steps,
            data_list=["B", "E", "rho", "Tr_ions", "Tt_ions", "Tz_ions"],
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
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = PlasmaCylinderCompression(test=args.test, verbose=args.verbose)
simulation.step()
