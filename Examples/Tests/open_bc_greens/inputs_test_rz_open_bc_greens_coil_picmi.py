#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""Coil-set drive against the RZ Green's-function open boundary.

A pair of circular filament coils OUTSIDE the domain (Rc > r_max) ramps up
with a Hann engagement through the hybrid solver's split-field external
vector potential (parser A_time path). The Green's-function open boundary
is active at r_hi, so this deck exercises the one interaction the ring and
transient tests cannot: the split-field ride-through contract at the open
face. The external coil field must pass through the boundary untouched
(it is advanced analytically, never through the discrete boundary
operators), while the Green's ghost fill acts on the plasma-response field
only -- whose sources are exactly the interior currents the fill maps.

Two arms:

- default (vacuum ride-through): no particles at all. Every cell sits on
  the vacuum branch of the generalized Ohm's law with zero plasma current,
  so the response field is identically zero and the total field must equal
  s(t) * curl A^unit -- the discrete Yee curl of the sampled analytic loop
  vector potential -- to roundoff, at every dump, including mid-ramp.
  Any contamination from the open-face ghost fill, the external
  accumulation, or the Faraday pairing of (scale_E, scale_B) shows up
  directly against this analytic baseline.

- --plasma (Lenz response): a super-floor plasma column on axis, with a
  0.5 m standoff from the open face (the analytic loop field is exact in
  free space, so the column must not touch the boundary; sub-floor cells
  between column and face stay on the vacuum branch and let both the coil
  flux and the response field pass). The ramp drives azimuthal eddy
  currents through -E_ext in Ohm's law; the response opposes the applied
  flux (Lenz) and decays resistively. Gated on the Lenz sign and on the
  implicit arm landing on the explicit subcycled trajectory at truncation
  level (the transient-guard pattern).

The coil filaments sit outside the domain, so the parser loop expressions
need no finite-conductor softening and no quarter-cell offset. The
analysis twin (analysis_coil.py) evaluates the same loop A_theta with
scipy elliptic integrals at the Yee staggerings and applies the same
discrete RZ curl; deck and analysis constants must match.
"""

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


class CoilGreensDrive(object):
    # Coil pair: circular filaments outside the domain in r, normalized so
    # each coil's isolated on-axis field at its own loop plane is BC
    # (I = 2 Rc BC / mu0). These constants are duplicated in
    # analysis_coil.py and must match.
    RC = 1.3  # m
    ZC = 0.45  # m (pair at +-ZC)
    BC = 0.03  # T

    # Domain
    LR = 1.0  # m
    LZ = 2.0  # m
    NR = 32
    NZ = 64

    # Hann engagement over one reference cyclotron period, then hold
    # (zero slope at t = 0 and at the plateau join).
    T_RAMP_CI = 1.0
    LT = 2.0  # total run length [t_ci]
    DT = 0.02  # step [t_ci]

    # Plasma arm: super-floor column, hard standoff from the open face.
    n0 = 1.0e21  # m^-3
    n_floor_fac = 64.0  # floor = n0 / 64
    T_i = 0.1  # eV
    T_e = 1.0  # eV
    R_COL = 0.5  # column radius / half-length [m]
    W_COL = 0.06  # tanh edge width [m]
    NPPC = 16

    def __init__(self, test, verbose, implicit=False, plasma=False):
        self.test = test
        self.verbose = verbose or self.test
        self.implicit = implicit
        self.plasma = plasma

        # Reference field: both coils' on-axis contribution at the domain
        # center (pure algebra, no elliptic integrals needed on axis).
        self.B_ref = 2.0 * self.BC * self.RC**3 / (self.RC**2 + self.ZC**2) ** 1.5
        self.w_ci = constants.q_e * self.B_ref / constants.m_p
        self.t_ci = 2.0 * np.pi / self.w_ci

        self.dt = self.DT * self.t_ci
        self.t_ramp = self.T_RAMP_CI * self.t_ci
        self.total_steps = int(round(self.LT / self.DT))
        self.diag_steps = self.total_steps // 4

        if comm.rank == 0:
            print(
                f"Initializing coil-drive Green's-boundary run:\n"
                f"\tB_ref (domain center, plateau) = {self.B_ref:.4e} T\n"
                f"\tt_ci = {self.t_ci:.4e} s, dt = {self.dt:.4e} s\n"
                f"\tramp = {self.t_ramp:.4e} s, steps = {self.total_steps}\n"
                f"\tarm: {'plasma (Lenz)' if plasma else 'vacuum ride-through'}"
                f"{' / implicit' if implicit else ' / explicit'}\n"
            )

        self.setup_run()

    def _loop_A_expressions(self, Rc, zc):
        """Parser strings (Ax, Ay) for the vector potential of a circular
        current loop at (Rc, zc), normalized so the on-axis field at the
        loop plane is BC (I = 2 Rc BC / mu0). WarpX evaluates the Cartesian
        parsers at (x = r, y = 0, z), mapping Ay -> A_theta. Filaments are
        outside the domain, so no finite-conductor softening is needed."""
        C = f"(-2*{Rc}*{self.BC}/pi)"
        zz = f"(z-{zc})"
        a2 = f"({Rc}*{Rc}+x*x+y*y+{zz}*{zz}-2*{Rc}*sqrt(x*x+y*y))"
        b2 = f"({Rc}*{Rc}+x*x+y*y+{zz}*{zz}+2*{Rc}*sqrt(x*x+y*y))"
        k2 = f"(1-{a2}/{b2})"
        t1 = f"((2-{k2})*comp_ellint_1(sqrt{k2}))"
        t2 = f"(2*comp_ellint_2(sqrt{k2}))"
        eps = 1e-12
        Ax = f"(((y*{Rc}*{C})/((sqrt({b2}*(x*x+y*y))*({k2}))+{eps}))*({t1}-{t2}))"
        Ay = f"(((x*{Rc}*{C})/((sqrt({b2}*(x*x+y*y))*({k2}))+{eps}))*({t2}-{t1}))"
        return Ax, Ay

    def setup_run(self):
        """Setup simulation components."""

        self.grid = picmi.CylindricalGrid(
            number_of_cells=[self.NR, self.NZ],
            lower_bound=[0.0, -self.LZ / 2.0],
            upper_bound=[self.LR, self.LZ / 2.0],
            # r_hi carries the Green's-function open boundary (the hybrid
            # solver's BC map passes 'open' through instead of PML); the
            # z caps are PEC, which acts on the plasma RESPONSE only --
            # the split external field rides through all walls unchanged.
            lower_boundary_conditions=["none", "dirichlet"],
            upper_boundary_conditions=["open", "dirichlet"],
            lower_boundary_conditions_particles=["none", "reflecting"],
            upper_boundary_conditions_particles=["absorbing", "reflecting"],
            n_azimuthal_modes=1,
            warpx_max_grid_size=32,
        )
        simulation.time_step_size = self.dt
        simulation.max_steps = self.total_steps
        simulation.current_deposition_algo = "direct"
        simulation.particle_shape = 1
        simulation.use_filter = self.plasma
        simulation.verbose = self.verbose

        # Hann engagement: s(0) = s'(0) = 0, s(T) = 1, s'(T) = 0, C^1 at
        # the plateau join -- no step-function kick into E_ext and no
        # harmonic comb from segment pushes (parser A_time path).
        s_of_t = f"if(t<{self.t_ramp!r},0.5-0.5*cos(pi*t/{self.t_ramp!r}),1.0)"

        A_ext = {}
        for name, zc in (("coil_zp", self.ZC), ("coil_zm", -self.ZC)):
            Ax, Ay = self._loop_A_expressions(self.RC, zc)
            A_ext[name] = {
                "Ax_external_function": Ax,
                "Ay_external_function": Ay,
                "Az_external_function": "0",
                "A_time_external_function": s_of_t,
            }

        # The loop A_theta is analytically divergence-free (pure m = 0
        # azimuthal component); the MLMG projection pass would only distort
        # the analytic baseline (and stalls in RZ at this resolution).
        eta = 1.0e-2 if self.plasma else 1.0e-6
        n_floor = (
            self.n0 / self.n_floor_fac if self.plasma else self.n_floor_fac * self.n0
        )
        self.solver = picmi.HybridPICSolver(
            grid=self.grid,
            gamma=1.0,
            Te=self.T_e,
            n0=self.n0,
            n_floor=n_floor,
            plasma_resistivity=eta,
            substeps=32 if self.plasma else 10,
            A_external=A_ext,
            do_external_diva_cleaning=False,
        )
        simulation.solver = self.solver

        if self.implicit:
            if self.plasma:
                # The driven near-perfect flux excluder (skin time >> ramp)
                # is exactly the overshooting-Newton configuration class of
                # the driven floored-whistler decks: without backtracking
                # the iteration limit-cycles just above tolerance at the
                # 20-iteration cap. The line search is the validated fix
                # (not picmi-managed; bucket write lands as newton.*).
                import pywarpx

                pywarpx.warpx.get_bucket("newton").line_search = 1
            # The vacuum arm is a precision plumbing gate: Newton converges
            # each step through an E_ext-scale opening residual, and the
            # relative-tolerance crumbs it leaves behind seed a floored
            # whistler field that theta = 0.5 propagates without damping at
            # the vacuum eta -- the ride-through residual accumulates
            # linearly in step count at ~4e-9 x (rel_tol / 1e-8) per step.
            # 1e-10 keeps the 100-step gate two orders below any physical
            # leak while costing at most one extra Newton iteration. On the
            # drive plateau the opening residual itself sits near the
            # machine floor, where a relative test alone stalls (measured:
            # abs 6e-16 cannot reach rel 1e-10 of its own opening), so the
            # vacuum arm also carries an absolute escape far below any
            # gated signal. The plasma arm runs the standard 1e-8.
            newton_rtol = 1.0e-8 if self.plasma else 1.0e-10
            newton_atol = 0.0 if self.plasma else 1.0e-12
            nonlinear_solver = picmi.NewtonNonlinearSolver(
                verbose=self.verbose,
                max_iterations=20,
                relative_tolerance=newton_rtol,
                absolute_tolerance=newton_atol,
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
            simulation.evolve_scheme = picmi.ThetaImplicitHybridEvolveScheme(
                theta=0.5,
                nonlinear_solver=nonlinear_solver,
            )

        #######################################################################
        # Particle types setup (plasma arm only)                             #
        #######################################################################

        if self.plasma:
            # Smooth-edged super-floor column with a 0.5 m standoff from the
            # open face and from the PEC z caps: the density at the walls is
            # ~1e-8 n0, far below the floor, so the boundary cells stay on
            # the vacuum branch (plasma OFF the open face).
            r_str = "sqrt(x*x+y*y)"
            profile = (
                f"{self.n0}"
                f"*(0.5-0.5*tanh(({r_str}-{self.R_COL})/{self.W_COL}))"
                f"*(0.5-0.5*tanh((abs(z)-{self.R_COL})/{self.W_COL}))"
            )
            self.ions = picmi.Species(
                name="ions",
                charge="q_e",
                mass=constants.m_p,
                initial_distribution=picmi.AnalyticDistribution(
                    density_expression=profile,
                    # Do not load macroparticles in the numerical vacuum:
                    # the tanh tail otherwise populates every cell with
                    # near-zero-weight ions that ride the unshielded
                    # vacuum-branch drive field, reach multi-cell-per-step
                    # speeds by mid-run, and trip the implicit deposit
                    # range guard. Cut an order below the Ohm floor. (This
                    # does NOT remove the marginal edge band at n ~ n_floor
                    # -- see the scheme-divergence note in the CI
                    # registration of the implicit Lenz arm.)
                    warpx_density_min=self.n0 / (10.0 * self.n_floor_fac),
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
    "--implicit",
    help="use the theta-implicit hybrid evolve scheme",
    action="store_true",
)
parser.add_argument(
    "--plasma",
    help="Lenz arm: super-floor plasma column with standoff from the open "
    "face (default: speciesless vacuum ride-through)",
    action="store_true",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = CoilGreensDrive(
    test=args.test,
    verbose=args.verbose,
    implicit=args.implicit,
    plasma=args.plasma,
)
simulation.step()
