#!/usr/bin/env python3
#
# --- Verification test for the QDSMC isotropic electron thermal conduction
# --- pass (Gauss-Hermite sampled Gaussian kernel; Albright et al., Phys.
# --- Plasmas 9, 1898 (2002)). A Gaussian electron-temperature bump rides a
# --- static, uniform, current-free plasma with a constant conductivity, so
# --- the exact solution stays Gaussian with variance sigma^2(t) =
# --- sigma_0^2 + 2 D t per axis, D = 2 kappa / (3 n_e k_B). The analysis
# --- fits the variance growth and checks the discrete maximum principle
# --- and the energy ledger.

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


class ConductionTest(object):
    # Static uniform plasma
    n0 = 1e18  # m^-3
    T_i = 0.1  # eV
    T_e = 100.0  # eV (background)

    # Gaussian temperature bump
    bump_amp = 1.0  # relative amplitude
    bump_sigma_cells = 6.0

    # Diffusivity in m^2/s; kappa follows from D = 2 kappa/(3 n k_B).
    # Operating point: sqrt(2 D dt/2) ~ 2.3 dx per half-pass -- the
    # LARGE-kick regime the kernel is built for. Two numerical-diffusion
    # scales bracket it (Albright et al. 2002): the advection stage's
    # gather-deposit remap ~ dx^2/(4 dt) pushes dt UP (2% of D here), and
    # the kernel's deposit granularity ~ (dx/sigma)^2 relative pushes the
    # kick width up (measured 1.6% at this point; it PEAKS near
    # sigma ~ dx, so mid-range kicks verify worse -- the dt ~ dx
    # refinement protocol in reverse).
    D = 1.65e4

    L = 0.5  # m
    NX = 64
    NZ = 64
    NPPC = 16

    DT = 2.0e-8  # s
    total_steps = 40

    def __init__(self, test, verbose):
        self.test = test
        self.verbose = verbose or self.test

        self.dx = self.L / self.NX
        self.bump_sigma = self.bump_sigma_cells * self.dx
        self.kappa = 1.5 * self.n0 * constants.kb * self.D
        self.diag_steps = self.total_steps // 5

        if comm.rank == 0:
            print(
                f"Initializing QDSMC conduction test:\n"
                f"\tD = {self.D:.3e} m^2/s (kappa = {self.kappa:.3e} W/(m K))\n"
                f"\tsigma_0 = {self.bump_sigma:.3e} m\n"
                f"\tkick/dx per half-pass = "
                f"{np.sqrt(self.D * self.DT) / self.dx:.3f}\n"
                f"\ttotal steps = {self.total_steps:d}\n"
            )

        self.setup_run()

    def load_bump(self):
        """Overwrite the uniform initial T_e with the Gaussian bump.

        Runs after initialize_warpx (the hybrid model seeds a uniform T_e
        during InitData, after the external-field hooks), so the step-0
        diagnostic still shows the uniform seed; the analysis fits from
        the later dumps.
        """
        Te = simulation.fields.get("hybrid_electron_temperature_fp", level=0)
        # Nodal mesh coordinates spanning the (periodic) box.
        x = np.linspace(-self.L / 2.0, self.L / 2.0, self.NX + 1)
        z = np.linspace(-self.L / 2.0, self.L / 2.0, self.NZ + 1)
        X, Z = np.meshgrid(x, z, indexing="ij")
        T0_K = self.T_e * constants.q_e / constants.kb
        prof = T0_K * (
            1.0
            + self.bump_amp
            * np.exp(-(X**2 + Z**2) / (2.0 * self.bump_sigma**2))
        )
        Te[:, :] = prof
        comm.Barrier()

    def setup_run(self):
        self.grid = picmi.Cartesian2DGrid(
            number_of_cells=[self.NX, self.NZ],
            lower_bound=[-self.L / 2.0, -self.L / 2.0],
            upper_bound=[self.L / 2.0, self.L / 2.0],
            lower_boundary_conditions=["periodic", "periodic"],
            upper_boundary_conditions=["periodic", "periodic"],
            lower_boundary_conditions_particles=["periodic", "periodic"],
            upper_boundary_conditions_particles=["periodic", "periodic"],
            warpx_max_grid_size=self.NX,
        )
        simulation.time_step_size = self.DT
        simulation.max_steps = self.total_steps
        simulation.current_deposition_algo = "direct"
        simulation.particle_shape = 1
        simulation.verbose = self.verbose

        self.solver = picmi.HybridPICSolver(
            grid=self.grid,
            gamma=1.0,
            Te=self.T_e,
            n0=self.n0,
            n_floor=0.01 * self.n0,
            plasma_resistivity=1e-6,
            substeps=4,
            solve_electron_energy_equation=True,
            qdsmc_conduction="isotropic",
            qdsmc_conduction_kappa=f"{self.kappa}",
        )
        simulation.solver = self.solver

        self.ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=constants.m_p,
            initial_distribution=picmi.UniformDistribution(
                density=self.n0,
                rms_velocity=[np.sqrt(self.T_i * constants.q_e / constants.m_p)]
                * 3,
            ),
        )
        simulation.add_species(
            self.ions,
            layout=picmi.PseudoRandomLayout(
                grid=self.grid, n_macroparticles_per_cell=self.NPPC
            ),
        )

        field_diag = picmi.FieldDiagnostic(
            name="diag1",
            grid=self.grid,
            period=self.diag_steps,
            data_list=["Te", "rho"],
            write_dir="diags",
            warpx_format="plotfile",
        )
        simulation.add_diagnostic(field_diag)

        if comm.rank == 0:
            if Path.exists(Path("diags")):
                shutil.rmtree("diags")
            Path("diags").mkdir(parents=True, exist_ok=True)

        simulation.initialize_inputs()
        simulation.initialize_warpx()
        self.load_bump()


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

run = ConductionTest(test=args.test, verbose=args.verbose)
simulation.step()
