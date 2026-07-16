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

    def __init__(self, test, verbose, front=False, rz=False, circle=False):
        self.test = test
        self.verbose = verbose or self.test
        self.front = front
        self.rz = rz
        self.circle = circle
        if self.rz and not self.front:
            raise ValueError("the RZ variant runs the front mode")
        if self.circle and (self.front or self.rz):
            raise ValueError("--circle is a standalone mode")

        if self.circle:
            # Parallel-conduction verification on circular in-plane field
            # lines: a hot patch on a ring spreads ALONG the line as 1D
            # arc-length diffusion and must not leak across (the grid-free
            # transport of the field-line Green's-function method;
            # del-Castillo-Negrete & Chacon, PRL 106, 195004 (2011)).
            self.NX = 64
            self.NZ = 64
            self.L = 0.5
            self.Lz = 0.5
            self.bump_sigma_cells = 3.0
            self.ring_r0_cells = 16.0
            self.total_steps = 40
            self.diag_steps = 8

        if self.front:
            # Zel'dovich-Barenblatt slab front: kappa ~ T^{5/2} releases a
            # hot slab into a cold background; the self-similar front obeys
            # x_f ~ t^{2/9} (porous-medium exponent m = 7/2, 1D). The cold
            # background (T_bg = T_e) has D smaller by (T_bg/T_peak)^{5/2}
            # = 1e-5, so it is effectively inert.
            if self.rz:
                # RZ: cylindrically uniform slab front along z. The same
                # 1D Zel'dovich exponent applies, while the pass runs the
                # full RZ machinery (off-plane kick fold into the radius,
                # cylindrical deposit volumes, axis and wall handling).
                #
                # KNOWN ISSUE (unregistered): the discrete radial weights
                # (2 pi r contents vs the nodal deposit volumes and the
                # Verboncoeur axis factors) pump a uniform temperature
                # radially at the axis scale, so this variant does not yet
                # hold the RZ invariance. Same convention family as the
                # RZ filter volume-weighting work; to be resolved with it.
                self.NX = 8
                self.NZ = 128
                self.L = 0.03125
                self.Lz = 0.5
            else:
                self.NX = 128
                self.NZ = 8
                self.L = 0.5
                self.Lz = 0.03125
            self.T_e = 1.0  # eV background
            self.bump_amp = 99.0  # peak T = 100 eV
            self.bump_sigma_cells = 6.0
            self.total_steps = 300
            self.diag_steps = 25
        else:
            self.Lz = self.L
            self.diag_steps = self.total_steps // 5

        self.dx = (self.Lz / self.NZ) if self.rz else (self.L / self.NX)
        self.bump_sigma = self.bump_sigma_cells * self.dx
        if self.circle:
            self.ring_r0 = self.ring_r0_cells * self.dx
        if self.circle:
            # sub-kick ~ 0.7 dx along the line (4 substeps per half-pass);
            # sized so the angular spread stays well under a radian and
            # the circular-statistics variance remains linear.
            self.D = self.dx**2 / self.DT * 2.0
            self.kappa = f"{1.5 * self.n0 * constants.kb * self.D}"
        elif self.front:
            # The kick width at the initial peak sits near dx (the front
            # exponent is set by scaling and conservation, not the kernel
            # prefactor); the run length keeps the front inside the box.
            D_peak = 0.2 * (2.3 * self.dx) ** 2 / self.DT
            T_peak_K = (
                (1.0 + self.bump_amp) * self.T_e * constants.q_e / constants.kb
            )
            kappa0 = 1.5 * self.n0 * constants.kb * D_peak
            self.kappa = f"{kappa0}*(T/{T_peak_K})**2.5"
        else:
            self.kappa = f"{1.5 * self.n0 * constants.kb * self.D}"

        if comm.rank == 0:
            print(
                f"Initializing QDSMC conduction test (front={self.front}):\n"
                f"\tkappa(T,n) = {self.kappa} W/(m K)\n"
                f"\tsigma_0 = {self.bump_sigma:.3e} m\n"
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
        if self.rz:
            x = np.linspace(0.0, self.L, self.NX + 1)
        else:
            x = np.linspace(-self.L / 2.0, self.L / 2.0, self.NX + 1)
        z = np.linspace(-self.Lz / 2.0, self.Lz / 2.0, self.NZ + 1)
        X, Z = np.meshgrid(x, z, indexing="ij")
        T0_K = self.T_e * constants.q_e / constants.kb
        if self.circle:
            r2 = (X - self.ring_r0) ** 2 + Z**2
        elif self.front:
            r2 = Z**2 if self.rz else X**2
        else:
            r2 = X**2 + Z**2
        prof = T0_K * (
            1.0 + self.bump_amp * np.exp(-r2 / (2.0 * self.bump_sigma**2))
        )
        Te[:, :] = prof
        comm.Barrier()

    def setup_run(self):
        if self.rz:
            self.grid = picmi.CylindricalGrid(
                number_of_cells=[self.NX, self.NZ],
                lower_bound=[0.0, -self.Lz / 2.0],
                upper_bound=[self.L, self.Lz / 2.0],
                lower_boundary_conditions=["none", "periodic"],
                upper_boundary_conditions=["dirichlet", "periodic"],
                lower_boundary_conditions_particles=["none", "periodic"],
                upper_boundary_conditions_particles=["reflecting", "periodic"],
                warpx_max_grid_size=self.NZ,
            )
        else:
            self.grid = picmi.Cartesian2DGrid(
                number_of_cells=[self.NX, self.NZ],
                lower_bound=[-self.L / 2.0, -self.Lz / 2.0],
                upper_bound=[self.L / 2.0, self.Lz / 2.0],
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

        if self.circle:
            # Azimuthal in-plane rings about the box center,
            # b = (-z, 0, x)/r, with a hybrid-friendly radial profile: a
            # rigid-rotor core (B ~ r: uniform current, no field null), a
            # current-free 1/r annulus where the test patch lives, and a
            # cos^2 taper to zero before the box edge (no current sheets
            # at the periodic seams; the zero-field corners deposit in
            # place through the unmagnetized gate).
            r_c = 6.0 * self.dx
            r_t = 0.30 * self.L
            r_e = 0.45 * self.L
            rr = f"max(sqrt(x*x+z*z),{0.01 * self.dx})"
            ramp = f"min({rr}/{r_c},1.0)"
            taper = (f"cos(1.5707963267948966*"
                     f"min(max(({rr}-{r_t})/{r_e - r_t},0.0),1.0))**2")
            # 1 mT: the parallel kernel only consumes the field DIRECTION,
            # and the whistler frequency at this grid/dt must stay under
            # the explicit B-substepping stability limit
            # (omega ~ k_max^2 B / (mu0 n e); 0.1 T would put
            # omega dt_sub ~ 40 and detonate within two steps).
            prof = f"(1.0e-3*{ramp}*{taper})"
            b_init = picmi.AnalyticInitialField(
                Bx_expression=f"-z/{rr}*{prof}",
                By_expression="0.0",
                Bz_expression=f"x/{rr}*{prof}",
            )
            simulation.add_applied_field(b_init)

        self.solver = picmi.HybridPICSolver(
            grid=self.grid,
            gamma=1.0,
            Te=self.T_e,
            n0=self.n0,
            n_floor=0.01 * self.n0,
            # Strong resistivity: with the inert-ion lattice nothing
            # shorts the bump's grad(Pe), which would otherwise drive a
            # secular Faraday growth of B and spin up spurious electron
            # advection. The resistive diffusion time of B at the bump
            # scale is << the run, so the field never builds; eta*J does
            # not enter the temperature (Joule heating off).
            plasma_resistivity=0.1,
            substeps=50 if self.circle else 4,
            solve_electron_energy_equation=True,
            qdsmc_conduction="parallel" if self.circle else "isotropic",
            qdsmc_conduction_kappa=self.kappa,
            # Sub-kicks of ~1 dx: the 3-node Gauss-Hermite rule imprints
            # its node triplet on the field when a single kick spans
            # several cells; composing sub-cell kicks (central limit)
            # recovers the smooth Gaussian kernel.
            qdsmc_conduction_substeps=4,
        )
        simulation.solver = self.solver

        # Inert heavy ions: the temperature bump's pressure gradient would
        # otherwise ballistically accelerate light cold ions until one
        # crosses the deposition guard cells in a single step. The test
        # isolates conduction on a static density lattice.
        self.ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=1.0e5 * constants.m_p,
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
parser.add_argument(
    "--front",
    help="run the Zel'dovich nonlinear-front variant",
    action="store_true",
)
parser.add_argument(
    "--rz",
    help="run the front variant on an RZ grid (front along z)",
    action="store_true",
)
parser.add_argument(
    "--circle",
    help="parallel conduction on circular field lines",
    action="store_true",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = ConductionTest(
    test=args.test, verbose=args.verbose, front=args.front, rz=args.rz,
    circle=args.circle,
)
simulation.step()
