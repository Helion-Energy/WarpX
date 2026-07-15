#!/usr/bin/env python3
#
# --- RZ variant of the adiabatic-compression test for the hybrid-PIC
# --- (Ohm's law) electron energy equation. Same physics as the 2D deck:
# --- with all sources off the equation reduces to the transport terms,
# ---
# ---     dU_e/dt + div(U_e V_e) + P_e div(V_e) = 0,
# ---
# --- solved by advecting the electron entropy K_e = T_e n_e^(1-gamma) with
# --- QDSMC markers moving at V_e. A sinusoidal AXIAL ion-velocity
# --- perturbation v_z = V0 sin(k z) drives an ion-acoustic wave along z
# --- (uniform in r), so the pointwise adiabat
# ---
# ---     T_e(r,z,t) = T_e0 * ( n(r,z,t) / n0 )^(gamma_e - 1)
# ---
# --- must hold everywhere. This exercises the RZ-specific pieces of the
# --- QDSMC machinery: theta=0-plane markers, azimuthal-mode deposits, and
# --- the radial-geometry inverse-volume scaling of the charge densities
# --- used to seed and recover the entropy. Analyse with analysis_adiabat.py.

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI as mpi

from pywarpx import picmi

constants = picmi.constants
comm = mpi.COMM_WORLD

simulation = None


class AdiabaticCompressionRZ(object):
    # ---- Plasma parameters ------------------------------------------------
    te_eV = 100.0  # initial (uniform) electron temperature (eV)
    ti_eV = 10.0  # ion temperature (eV)
    gamma_e = 5.0 / 3.0  # electron adiabatic index
    n0 = 2.0e20  # uniform density (m^-3)

    # ---- Perturbation -----------------------------------------------------
    pert_frac = 0.30  # ion velocity amplitude V0 = pert_frac * c_s
    n_wave = 1  # wavelengths across Lz

    # ---- Geometry ---------------------------------------------------------
    Lz = 0.5  # domain length in z (m); the wave runs along z
    NZ = 128
    NR = 16

    # ---- Numerics ---------------------------------------------------------
    NPPC = 800
    periods = 2.0
    steps_per_period = 400
    substeps = 10

    def __init__(self, test, verbose, implicit=False, nlsolver="picard", theta=0.5):
        self.test = test
        self.verbose = verbose or test
        self.implicit = implicit
        self.nlsolver = nlsolver
        self.theta = theta

        if self.test:
            self.NZ = 32
            self.NR = 8
            self.NPPC = 64
            self._steps_override = 60
            self.ndiag = 10
        else:
            self._steps_override = None
            self.ndiag = 40

        self.get_plasma_quantities()
        if comm.rank == 0:
            self._print_params()
        self.setup_run()

    def get_plasma_quantities(self):
        mi = constants.m_p
        self.dz = self.Lz / self.NZ
        self.Lr = self.dz * self.NR
        self.k = 2.0 * np.pi * self.n_wave / self.Lz

        self.c_s = np.sqrt(self.gamma_e * constants.q_e * self.te_eV / mi)
        self.omega = self.k * self.c_s
        self.T_period = 2.0 * np.pi / self.omega
        self.V0 = self.pert_frac * self.c_s

        self.dt = self.T_period / self.steps_per_period
        if self._steps_override is not None:
            self.total_steps = self._steps_override
        else:
            self.total_steps = int(self.periods * self.steps_per_period)
        self.diag_steps = max(1, self.total_steps // self.ndiag)

        self.vi_th = np.sqrt(constants.q_e * self.ti_eV / mi)
        self.eta = 0.0  # no resistivity -> no Joule heating, pure LHS

    def _print_params(self):
        print(
            f"\n[setup] RZ adiabatic-compression (electron-energy-equation LHS) test\n"
            f"  Te0 = {self.te_eV:.1f} eV,  Ti = {self.ti_eV:.1f} eV,  gamma_e = {self.gamma_e:.4f}\n"
            f"  n0        = {self.n0:.3e} m^-3\n"
            f"  c_s       = {self.c_s:.3e} m/s\n"
            f"  V0        = {self.V0:.3e} m/s   (= {self.pert_frac:.2f} c_s, axial)\n"
            f"  Grid      = {self.NR} x {self.NZ} (r x z),  Lr x Lz = {self.Lr:.4f} x {self.Lz:.3f} m\n"
            f"  dt        = {self.dt:.3e} s   ({self.steps_per_period}/period)\n"
            f"  steps     = {self.total_steps},  diag every {self.diag_steps}\n"
            f"  scheme    = {'theta-implicit hybrid' if self.implicit else 'explicit hybrid'}\n"
            f"  CHECK:  T_e(r,z,t) = Te0 (n/n0)^(gamma_e-1)  pointwise\n"
        )

    def setup_run(self):
        global simulation

        simulation = picmi.Simulation(
            time_step_size=self.dt,
            max_steps=self.total_steps,
            verbose=self.verbose,
            particle_shape=1,
            warpx_serialize_initial_conditions=True,
            warpx_current_deposition_algo="direct",
            warpx_use_filter=True,
        )

        self.grid = picmi.CylindricalGrid(
            number_of_cells=[self.NR, self.NZ],
            warpx_max_grid_size=self.NZ,
            lower_bound=[0, -self.Lz / 2.0],
            upper_bound=[self.Lr, self.Lz / 2.0],
            lower_boundary_conditions=["none", "periodic"],
            upper_boundary_conditions=["dirichlet", "periodic"],
            lower_boundary_conditions_particles=["none", "periodic"],
            upper_boundary_conditions_particles=["reflecting", "periodic"],
        )

        self.solver = picmi.HybridPICSolver(
            grid=self.grid,
            gamma=self.gamma_e,
            Te=self.te_eV,
            n0=self.n0,
            n_floor=0.05 * self.n0,
            plasma_resistivity=self.eta,
            substeps=self.substeps,
            solve_electron_energy_equation=True,
            include_joule_heating=False,
        )
        simulation.solver = self.solver

        if self.implicit:
            # Theta-implicit hybrid scheme with the theta-centered QDSMC
            # energy stage inside the nonlinear iteration (see the 2D adiabat
            # deck). Picard is the default: Newton/JFNK is not robust in
            # unmagnetized configurations.
            if self.nlsolver == "picard":
                nonlinear_solver = picmi.PicardNonlinearSolver(
                    verbose=self.verbose,
                    max_iterations=100,
                    relative_tolerance=1.0e-8,
                    absolute_tolerance=0.0,
                    require_convergence=True,
                )
            else:
                gmres_solver = picmi.GMRESLinearSolver(
                    verbose_int=1,
                    max_iterations=100,
                    relative_tolerance=1.0e-6,
                    absolute_tolerance=0.0,
                )
                nonlinear_solver = picmi.NewtonNonlinearSolver(
                    verbose=self.verbose,
                    max_iterations=20,
                    relative_tolerance=1.0e-8,
                    absolute_tolerance=0.0,
                    require_convergence=True,
                    linear_solver=gmres_solver,
                    max_particle_iterations=21,
                    particle_tolerance=1.0e-12,
                )
            simulation.evolve_scheme = picmi.ThetaImplicitHybridEvolveScheme(
                theta=self.theta,
                nonlinear_solver=nonlinear_solver,
            )

        # Uniform density; sinusoidal axial velocity perturbation
        # v_z = V0 sin(k z). Uniform n0 and Te0 -> uniform initial entropy.
        # NOTE: do_temperature_deposition is intentionally NOT enabled here.
        # It is not needed for the source-free adiabat check (no Q_ei), and
        # the shape-aware temperature deposition currently crashes in RZ on
        # multi-rank runs (latent issue in the variance-deposition kernels;
        # tracked separately -- it blocks RZ Q_ei runs, not this test).
        self.ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=constants.m_p,
            initial_distribution=picmi.AnalyticDistribution(
                density_expression="n0",
                momentum_expressions=["0", "0", f"({self.V0})*sin(({self.k})*z)"],
                warpx_momentum_spread_expressions=[str(self.vi_th)] * 3,
                n0=self.n0,
            ),
        )
        simulation.add_species(
            self.ions,
            layout=picmi.PseudoRandomLayout(
                grid=self.grid, n_macroparticles_per_cell=self.NPPC
            ),
        )

        if comm.rank == 0 and Path("diags").exists():
            shutil.rmtree("diags")
        comm.Barrier()

        field_diag = picmi.FieldDiagnostic(
            name="field_diag",
            grid=self.grid,
            period=self.diag_steps,
            data_list=["rho", "Te", "J", "B"],
            write_dir="diags",
            warpx_file_prefix="field_diags",
            warpx_format="openpmd",
            warpx_openpmd_backend="h5",
        )
        simulation.add_diagnostic(field_diag)

        simulation.initialize_inputs()
        simulation.initialize_warpx()


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
    help="use the theta-implicit hybrid evolve scheme (theta = 0.5)",
    action="store_true",
)
parser.add_argument(
    "--nlsolver",
    help="nonlinear solver for the implicit scheme",
    choices=["newton", "picard"],
    default="picard",
)
parser.add_argument(
    "--theta",
    type=float,
    default=0.5,
    help="implicit time-biasing parameter (0.5 = time-centered)",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = AdiabaticCompressionRZ(
    test=args.test,
    verbose=args.verbose,
    implicit=args.implicit,
    nlsolver=args.nlsolver,
    theta=args.theta,
)
simulation.step()
