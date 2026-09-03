#!/usr/bin/env python3
#
# --- Manufactured-solution test for the PHYSICAL anisotropic electron
# --- viscosity of the hybrid-PIC (Ohm's law) electron energy equation.
# ---
# --- A 2D Cartesian (x, z) periodic box holds a uniform guide field tilted
# --- by theta in the x-z plane, plus a small y-perturbation varying in x:
# ---
# ---     B(x) = [ B0 sin(theta), b1 sin(k x), B0 cos(theta) ],  k = 2 pi n / Lx
# ---
# --- so that curl(B) = [0, 0, b1 k cos(k x)] and, with cold ions at rest,
# --- the electron fluid carries all of it:
# ---
# ---     u_e(x) = -curl(B)/(mu0 e n0) = [0, 0, -U cos(k x)],
# ---     U = b1 k / (mu0 e n0).
# ---
# --- The only nonzero velocity gradient is G_xz = d u_ez / dx = U k sin(k x),
# --- so the flow is divergence-free (no compressional heating) and the
# --- traceless rate-of-strain tensor has the single pair W_xz = W_zx = G_xz.
# --- The Braginskii field-aligned strain is then
# ---
# ---     S = 2 b_x b_z G_xz = sin(2 theta) U k sin(k x),
# ---
# --- and with W:W = 2 G_xz^2 the viscous heating reduces to the closed form
# ---
# ---     Q_nu(x) = [ (3/4) sin^2(2 theta) mu_par
# ---               + (1/4) (4 - 3 sin^2(2 theta)) mu_perp ] (U k)^2 sin^2(k x).
# ---
# --- Two properties make this the right fixture:
# ---
# ---   * theta is a pure ANISOTROPY dial. The strain is fixed; only its
# ---     projection on b-hat changes. At theta = 0 the field is exactly
# ---     perpendicular to the strain plane's parallel direction (b_x = 0),
# ---     S vanishes IDENTICALLY -- discretely too, since B_x is zero
# ---     everywhere -- and a correct parallel viscosity must deposit no
# ---     heat at all. That is the null arm.
# ---   * u_e is set by curl(B) on a smooth analytic field rather than by a
# ---     deposited quantity, and the ions are laid down on a regular grid
# ---     with no thermal spread, so rho is uniform to round-off. The
# ---     heating is a positive-definite QUADRATIC in grad u_e, which
# ---     rectifies deposition noise into a spurious heat source; a
# ---     shot-noise-free fixture is what makes the absolute rate assertable.
# ---
# --- Everything else in the energy equation is off (no resistivity, no
# --- Joule heating, no conduction, no e-i relaxation), so the measured
# --- T_e rise is the viscous channel and nothing else. The free-streaming
# --- flux limiter is disabled so the closed form above is exact.
# ---
# --- analysis_viscosity.py checks the domain-mean heating rate against the
# --- closed form (active arm) or against zero (null arm), plus the
# --- sin^2(k x) profile shape.

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI as mpi

import pywarpx
from pywarpx import picmi

constants = picmi.constants
comm = mpi.COMM_WORLD

simulation = None


class ElectronViscosity(object):
    # ---- Plasma parameters ------------------------------------------------
    te_eV = 100.0  # initial electron temperature (eV), uniform
    gamma_e = 5.0 / 3.0  # electron adiabatic index
    n0 = 1.0e19  # uniform density (m^-3)
    coulomb_log = 10.0  # ln(Lambda) entering the Braginskii tau_e

    # ---- Magnetic field ---------------------------------------------------
    B0 = 0.1  # guide-field magnitude (T)
    b1 = 0.005  # y-perturbation amplitude (T); b1/B0 = 5% keeps b-hat
    #   effectively fixed while still driving the current
    n_wave = 2  # full wavelengths of the perturbation across Lx

    # ---- Geometry ---------------------------------------------------------
    Lx = 0.3  # domain length in x (m); the field varies in x
    NX = 32  # cells in x
    NZ = 8  # cells in z (everything is z-independent; periodic)

    # ---- Numerics ---------------------------------------------------------
    NPPC = 4  # gridded layout, cold ions -> rho is uniform to round-off,
    #   so a handful of markers per cell is enough
    TOTAL_STEPS = 20
    substeps = 20
    TARGET_HEATING = 0.03  # T_e rise over the run, as a fraction of Te0

    def __init__(self, test, verbose, tilt_deg, nu_par):
        self.test = test
        self.verbose = verbose or test
        self.tilt = np.deg2rad(tilt_deg)
        # nu_par < 0 selects the built-in Braginskii coefficients; a positive
        # value selects the parser model with that constant nu_par [m^2/s].
        self.nu_par_override = nu_par

        self.total_steps = self.TOTAL_STEPS
        self.diag_steps = self.TOTAL_STEPS

        self.get_plasma_quantities()
        if comm.rank == 0:
            self._print_params()
        self.setup_run()

    def braginskii_nu_par(self, te_eV):
        """Braginskii (1965) field-aligned kinematic electron viscosity [m^2/s].

        nu_par = 0.73 k_B T_e tau_e / m_e, with tau_e from Eq. (2.5e). Written
        out here independently of the C++ implementation -- that is the point
        of the test.
        """
        te_J = te_eV * constants.q_e
        four_pi_eps0 = 4.0 * np.pi * 8.8541878188e-12
        tau_e = (
            3.0
            * np.sqrt(constants.m_e)
            * te_J**1.5
            * four_pi_eps0**2
            / (
                4.0
                * np.sqrt(2.0 * np.pi)
                * self.n0
                * constants.q_e**4
                * self.coulomb_log
            )
        )
        return 0.73 * te_J * tau_e / constants.m_e

    def get_plasma_quantities(self):
        mi = constants.m_p

        self.dx = self.Lx / self.NX
        self.Lz = self.dx * self.NZ  # square cells
        self.k = 2.0 * np.pi * self.n_wave / self.Lx

        # Electron drift amplitude carrying curl(B) (ions start at rest).
        self.U = self.b1 * self.k / (constants.mu0 * constants.q_e * self.n0)

        # Ion cyclotron period: the run must stay well inside it so the J x B
        # force does not appreciably rearrange the manufactured state.
        self.w_ci = constants.q_e * self.B0 / mi
        self.t_ci = 2.0 * np.pi / self.w_ci

        # Rate coefficient of the closed form, evaluated at the initial Te:
        #   <dTe/dt> = (gamma-1) <Q_nu> / (n0 kB)
        #            = 0.25 sin^2(2 theta) (m_e/kB) nu_par (U k)^2
        # using <sin^2> = 1/2 and neglecting the ~1e-12 perpendicular channel.
        self.nu_par0 = (
            self.braginskii_nu_par(self.te_eV)
            if self.nu_par_override < 0.0
            else self.nu_par_override
        )
        self.dTe_dt_pred = (
            0.25
            * np.sin(2.0 * self.tilt) ** 2
            * (constants.m_e / constants.kb)
            * self.nu_par0
            * (self.U * self.k) ** 2
        )

        # Size dt from the heating itself: the run ends once T_e has risen by
        # TARGET_HEATING. That fraction is the fixture's lifetime budget, not
        # a signal-to-noise one (the fixture is shot-noise free). The
        # perturbation is a standing whistler that rotates B_y into B_z at
        # omega_w = k^2 B_par/(mu0 e n0); the closed form holds only while
        # that rotation is small, and 3% heating lands at a whistler phase of
        # ~0.15 rad, where the measured drift from the closed form is ~1%.
        te0_K = self.te_eV * constants.q_e / constants.kb
        target = self.TARGET_HEATING * te0_K
        rate = max(self.dTe_dt_pred, 1.0e-30)
        self.dt = target / (rate * self.total_steps)
        # The null arm has no heating to size against; borrow the active arm's
        # dt so both arms run the identical trajectory.
        if np.sin(2.0 * self.tilt) ** 2 < 1.0e-12:
            nu_ref = self.nu_par0
            self.dt = target / (
                0.25
                * (constants.m_e / constants.kb)
                * nu_ref
                * (self.U * self.k) ** 2
                * self.total_steps
            )

    def _print_params(self):
        print(
            f"\n[setup] Anisotropic electron-viscosity manufactured solution\n"
            f"  Te0       = {self.te_eV:.1f} eV,  n0 = {self.n0:.3e} m^-3\n"
            f"  B0        = {self.B0:.3e} T,  b1 = {self.b1:.3e} T "
            f"(b1/B0 = {self.b1 / self.B0:.3f})\n"
            f"  tilt      = {np.rad2deg(self.tilt):.1f} deg  "
            f"(sin^2(2 theta) = {np.sin(2.0 * self.tilt) ** 2:.4f})\n"
            f"  k         = {self.k:.4e} 1/m  ({self.n_wave} wavelengths, "
            f"k dx = {self.k * self.dx:.4f})\n"
            f"  U         = {self.U:.4e} m/s,  U k = {self.U * self.k:.4e} 1/s\n"
            f"  nu_par    = {self.nu_par0:.4e} m^2/s "
            f"({'braginskii' if self.nu_par_override < 0 else 'parser'})\n"
            f"  Grid      = {self.NX} x {self.NZ},  dx = {self.dx:.3e} m\n"
            f"  dt        = {self.dt:.4e} s  "
            f"({self.dt * self.total_steps / self.t_ci:.5f} t_ci total)\n"
            f"  ----\n"
            f"  PREDICTED <dTe/dt> = {self.dTe_dt_pred:.4e} K/s\n"
        )

    def load_initial_B(self):
        """Set B(x) = [B0 sin(theta), b1 sin(k x), B0 cos(theta)].

        div(B) = 0 analytically (B_x and B_z are uniform, B_y depends only on
        x), so no cleaning is needed. WarpX folds Bfield_fp_external into
        Bfield_fp at initialization, after which it evolves self-consistently.
        """
        Bx = simulation.fields.get("Bfield_fp_external", dir="x", level=0)
        By = simulation.fields.get("Bfield_fp_external", dir="y", level=0)
        Bz = simulation.fields.get("Bfield_fp_external", dir="z", level=0)

        XBy, _ = np.meshgrid(By.mesh("x"), By.mesh("z"), indexing="ij")
        Bx[:, :] = self.B0 * np.sin(self.tilt)
        By[:, :] = self.b1 * np.sin(self.k * XBy)
        Bz[:, :] = self.B0 * np.cos(self.tilt)
        comm.Barrier()

    def setup_run(self):
        global simulation

        self.grid = picmi.Cartesian2DGrid(
            number_of_cells=[self.NX, self.NZ],
            lower_bound=[0.0, -self.Lz / 2.0],
            upper_bound=[self.Lx, self.Lz / 2.0],
            lower_boundary_conditions=["periodic", "periodic"],
            upper_boundary_conditions=["periodic", "periodic"],
            lower_boundary_conditions_particles=["periodic", "periodic"],
            upper_boundary_conditions_particles=["periodic", "periodic"],
            warpx_max_grid_size=self.NX // 2,
        )

        # Electron energy equation with the viscous source and NOTHING else:
        # no resistivity, no Joule heating, no conduction, no relaxation, so
        # the measured dTe is the viscous channel alone.
        self.solver = picmi.HybridPICSolver(
            grid=self.grid,
            gamma=self.gamma_e,
            Te=self.te_eV,
            n0=self.n0,
            n_floor=0.05 * self.n0,
            plasma_resistivity=0.0,
            plasma_hyper_resistivity=0.0,
            substeps=self.substeps,
            solve_electron_energy_equation=True,
            include_joule_heating=False,
        )

        simulation = picmi.Simulation(
            solver=self.solver,
            time_step_size=self.dt,
            max_steps=self.total_steps,
            verbose=self.verbose,
            particle_shape=1,
            warpx_serialize_initial_conditions=True,
            warpx_current_deposition_algo="direct",
            warpx_use_filter=False,
        )

        B_init = picmi.LoadInitialFieldFromPython(
            load_from_python=self.load_initial_B,
            load_B=True,
            load_E=False,
        )
        simulation.add_applied_field(B_init)

        # Cold ions on a regular sub-grid: rho is uniform to round-off and
        # J_i vanishes, so u_e = -curl(B)/(mu0 e n0) exactly.
        self.ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=constants.m_p,
            initial_distribution=picmi.AnalyticDistribution(
                density_expression="n0",
                momentum_expressions=["0", "0", "0"],
                n0=self.n0,
            ),
        )
        simulation.add_species(
            self.ions,
            layout=picmi.GriddedLayout(grid=self.grid, n_macroparticle_per_cell=[2, 2]),
        )

        # Remove diags from a previous run in the same directory so stale
        # openPMD dumps cannot mix into the analysis of this run.
        if comm.rank == 0 and Path("diags").exists():
            shutil.rmtree("diags")
        comm.Barrier()

        field_diag = picmi.FieldDiagnostic(
            name="field_diag",
            grid=self.grid,
            period=self.diag_steps,
            data_list=["B", "rho", "Te"],
            write_dir="diags",
            warpx_file_prefix="field_diags",
            warpx_format="openpmd",
            warpx_openpmd_backend="h5",
        )
        simulation.add_diagnostic(field_diag)

        # Viscosity knobs. There is no picmi wrapper for these, so they are
        # set straight on the hybrid bucket -- which also means the C++
        # defaults are the ONLY defaults: nothing is restated here that the
        # solver could disagree with.
        if self.nu_par_override < 0.0:
            pywarpx.hybridpicmodel.qdsmc_viscosity_model = "braginskii"
            pywarpx.hybridpicmodel.qdsmc_viscosity_coulomb_log = self.coulomb_log
        else:
            pywarpx.hybridpicmodel.qdsmc_viscosity_model = "parser"
            pywarpx.hybridpicmodel.add_new_attr(
                "qdsmc_nu_par(n,Te,B)", f"{self.nu_par_override:.16e}"
            )
        # The closed form above is the unlimited Braginskii stress; the
        # free-streaming cap would make the coefficient Te-dependent in a
        # second way and is exercised elsewhere.
        pywarpx.hybridpicmodel.qdsmc_viscosity_flux_limit_factor = 0.0
        # Centred grad u_e: the fixture is smooth and shot-noise-free, so the
        # MC bound is not needed and its clipping near the zeros of sin(k x)
        # would perturb the exact comparison.
        pywarpx.hybridpicmodel.qdsmc_viscosity_limiter = "none"

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
    "--tilt-deg",
    type=float,
    default=45.0,
    help="angle of the guide field out of the z axis, in degrees. 45 puts the "
    "strain fully on the field-aligned channel; 0 is the exact null (b_x = 0, "
    "so the Braginskii parallel strain S vanishes identically).",
)
parser.add_argument(
    "--nu-par",
    type=float,
    default=-1.0,
    help="constant nu_par [m^2/s] through the parser viscosity model; the "
    "default (negative) selects the built-in Braginskii coefficients.",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = ElectronViscosity(
    test=args.test,
    verbose=args.verbose,
    tilt_deg=args.tilt_deg,
    nu_par=args.nu_par,
)
simulation.step()
