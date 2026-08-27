#!/usr/bin/env python3
#
# --- Circuit-coupled version of the RZ plasma-cylinder compression test:
# --- the SAME physics configuration as
# --- inputs_test_rz_ohm_solver_cylinder_compression_picmi.py
# --- (in its CI test mode), but with both compression drive fields (the
# --- openPMD-file-loaded half and the parser-analytical half) converted to
# --- SetScale-driven circuit ports and driven through the per-substep
# --- coupling engine by a current-stiff Python engine that pushes the exact
# --- same sigmoid ramp as segments. The realized run must be equivalent to
# --- the parser-driven original: the drives differ only by the
# --- piecewise-linear per-substep segment realization (with its exact
# --- -slope E partner) vs the pointwise sigmoid (with its centered-FD E),
# --- an O((dt/tau)^2) difference. The final fields are compared against the
# --- original test's plotfile, which must exist (declared as a test
# --- dependency; a scratch run can pass ref_dir=<path> on the command line).
# ---
# --- This exercises, on real compression physics: python_scale on a
# --- file-loaded AND a parser-loaded field, two coupled coils, the
# --- per-substep predictor-corrector hook contract, disk probes under a
# --- conducting wall, and the coupling ledger.

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import openpmd_api as io
from mpi4py import MPI as mpi

from pywarpx import callbacks, picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=False)

# The reference (parser-driven) run's output; the ctest dependency layout
# puts it in a sibling directory.
REF_DIR = "../test_rz_ohm_solver_cylinder_compression_picmi/diags/diag1000020"
for _arg in list(sys.argv[1:]):
    if _arg.startswith("ref_dir="):
        REF_DIR = _arg.split("=", 1)[1]
        sys.argv.remove(_arg)

# The drive ramp of the original deck (sigmoid starting around 2.5 us)
TAU_RAMP = 20e-6
T0_RAMP = 5e-6


def s_of_t(t):
    return 1.0 / (1.0 + np.exp(5.0 * (1.0 - (t - T0_RAMP) * np.sqrt(2.0) / TAU_RAMP)))


class PlasmaCylinderCompression(object):
    # Identical parameters to the original test deck
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

    LT = 10  # ion cyclotron periods
    DT = 1e-3  # ion cyclotron periods

    NR = 128
    NZ = 32

    NPPC = 100
    substeps = 60

    # Effective circuit ports for the two drive halves (no unit-field fill:
    # the spatial A stays the file/parser shape; the port drives its scale)
    R_COIL = 0.45
    Z_COIL = 0.0
    I_REF = 5.0e4

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

        # Write A to OpenPMD for a uniform B field, identically to the
        # original deck (exercises python_scale on a file-loaded field)
        if comm.rank == 0:
            mvec = np.array([0])
            rvec = np.linspace(0, 2 * self.LR, num=2 * self.NR)
            zvec = np.linspace(-self.LZ, self.LZ, num=2 * self.NZ)
            MM, RM, ZM = np.meshgrid(mvec, rvec, zvec, indexing="ij")

            Ar_data = np.zeros_like(RM)
            Az_data = np.zeros_like(RM)

            # Only include half of the compression field here
            At_data = 0.25 * RM * self.dB

            series = io.Series("Afield.h5", io.Access.create)

            it = series.iterations[0]

            A = it.meshes["A"]
            A.geometry = io.Geometry.thetaMode
            A.geometry_parameters = "m=0"
            A.grid_spacing = [self.DR, self.DZ]
            A.grid_global_offset = [0.0, -self.LZ]
            A.grid_unit_SI = 1.0
            A.axis_labels = ["r", "z"]
            A.data_order = "C"
            A.unit_dimension = {
                io.Unit_Dimension.M: 1.0,
                io.Unit_Dimension.T: -2.0,
                io.Unit_Dimension.I: -1.0,
                io.Unit_Dimension.L: -1.0,
            }

            Ar = A["r"]
            At = A["t"]
            Az = A["z"]

            Ar.position = [0.0, 0.0]
            At.position = [0.0, 0.0]
            Az.position = [0.0, 0.0]

            Ar.reset_dataset(io.Dataset(Ar_data.dtype, Ar_data.shape))
            At.reset_dataset(io.Dataset(At_data.dtype, At_data.shape))
            Az.reset_dataset(io.Dataset(Az_data.dtype, Az_data.shape))

            Ar.store_chunk(Ar_data)
            At.store_chunk(At_data)
            Az.store_chunk(Az_data)

            series.flush()
            series.close()

        comm.Barrier()

        self.get_plasma_quantities()

        self.dt = self.DT * self.t_ci

        # run very low resolution as a CI test
        if self.test:
            self.total_steps = 20
            self.diag_steps = self.total_steps // 5
            self.NR = 64
            self.NZ = 16
        else:
            self.total_steps = int(self.LT / self.DT)
            self.diag_steps = 100

        if comm.rank == 0:
            print(
                f"Initializing circuit-coupled compression:\n"
                f"\tB0 = {self.B0:.4f} T, dt = {self.dt:.3e} s, "
                f"steps = {self.total_steps}\n"
                f"\ts(0) = {s_of_t(0.0):.6e}"
            )

        self.setup_run()

    def get_plasma_quantities(self):
        self.M = constants.m_p
        self.w_ci = constants.q_e * abs(self.B0) / self.M
        self.t_ci = 2.0 * np.pi / self.w_ci
        self.w_pi = np.sqrt(constants.q_e**2 * self.n0 / (self.M * constants.ep0))
        self.l_i = constants.c / self.w_pi
        self.vA = abs(self.B0) / np.sqrt(
            constants.mu0 * self.n0 * (constants.m_e + self.M)
        )
        self.vi_th = np.sqrt(self.T_i * constants.q_e / self.M)
        self.rho_i = self.vi_th / self.w_ci

    def load_fields(self):
        Br = simulation.fields.get("Bfield_fp_external", dir="r", level=0)
        Bt = simulation.fields.get("Bfield_fp_external", dir="theta", level=0)
        Bz = simulation.fields.get("Bfield_fp_external", dir="z", level=0)

        Br[:, :] = 0.0
        Bt[:, :] = 0.0

        RM, ZM = np.meshgrid(Bz.mesh("r"), Bz.mesh("z"), indexing="ij")

        Bz[:, :] = self.Bz(RM) * (RM <= self.R_c)
        comm.Barrier()

    def setup_run(self):
        self.grid = picmi.CylindricalGrid(
            number_of_cells=[self.NR, self.NZ],
            lower_bound=[0.0, -self.Lz / 2.0],
            upper_bound=[self.Lr, self.Lz / 2.0],
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

        # The two drive halves of the original deck, as SetScale-driven
        # circuit ports (the spatial A is identical; only the time scale
        # moves from the parser to the engine's segments)
        s0 = s_of_t(0.0)
        A_ext = {
            "uniform_file": {
                "read_from_file": True,
                "path": "Afield.h5",
                "python_scale": True,
                "initial_scale": s0,
            },
            "uniform_analytical": {
                "Ax_external_function": f"-0.25*y*{self.dB}",
                "Ay_external_function": f"0.25*x*{self.dB}",
                "Az_external_function": "0",
                "python_scale": True,
                "initial_scale": s0,
            },
        }

        coil_a = picmi.CircuitCoil(
            name="bank_a",
            r=self.R_COIL,
            z=self.Z_COIL,
            n_turns=1.0,
            I_ref=self.I_REF,
            field_name="uniform_file",
            fill_unit_field=False,
            probe="disk",
        )
        coil_b = picmi.CircuitCoil(
            name="bank_b",
            r=self.R_COIL,
            z=self.Z_COIL,
            n_turns=1.0,
            I_ref=self.I_REF,
            field_name="uniform_analytical",
            fill_unit_field=False,
            probe="disk",
        )
        circuit = picmi.CircuitCoupling(
            coils=[coil_a, coil_b],
            engine="callbacks",
            corrector_iterations=1,
            corrector_rtol=1.0e-6,
        )

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
            circuit=circuit,
        )
        simulation.solver = self.solver

        B_ext = picmi.LoadInitialFieldFromPython(
            load_from_python=self.load_fields,
            warpx_do_divb_cleaning_external=True,
            load_B=True,
            load_E=False,
        )
        simulation.add_applied_field(B_ext)

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

        particle_diag = picmi.ParticleDiagnostic(
            name="diag1",
            period=self.diag_steps,
            species=[self.ions],
            data_list=["ux", "uy", "uz", "x", "z", "weighting"],
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

        if comm.rank == 0:
            if Path.exists(Path("diags")):
                shutil.rmtree("diags")
            Path("diags").mkdir(parents=True, exist_ok=True)

        simulation.initialize_inputs()

        # the coupling ledger
        import pywarpx

        pywarpx.warpx.reduced_diags_names = "circdiag"
        circdiag = pywarpx.warpx.get_bucket("circdiag")
        circdiag.type = "CircuitCoupling"
        circdiag.intervals = 1

        simulation.initialize_warpx()


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test", action="store_true",
                    help="toggle whether this script is run as a short CI test")
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = PlasmaCylinderCompression(test=args.test, verbose=args.verbose)

warpx = simulation.extension.warpx


class SigmoidBankEngine:
    """Current-stiff drive: both bank ports follow the prescribed sigmoid
    exactly (the measured linkage is recorded in the ledger but does not
    react back -- the equivalence target is the parser-driven original)."""

    def push(self):
        t0, t1, substep, it = warpx.get_coupling_interval()
        for name in ("uniform_file", "uniform_analytical"):
            warpx.set_external_vector_potential_scale(
                name, s_of_t(t0), s_of_t(t1), t0, t1)

    def beginstep(self):
        pass

    def predict(self):
        self.push()

    def correct(self):
        self.push()

    def finish(self):
        pass


engine = SigmoidBankEngine()
callbacks.installcallback("circuitbeginstep", engine.beginstep)
callbacks.installcallback("circuitpredict", engine.predict)
callbacks.installcallback("circuitcorrect", engine.correct)
callbacks.installcallback("circuitfinish", engine.finish)

simulation.step()

# ---------------------------------------------------------------------------
# Equivalence against the parser-driven original: the realized drive differs
# only by the per-substep piecewise-linear segments (exact -slope E) vs the
# pointwise sigmoid (centered-FD E), so the final fields must agree far
# below any physical scale of the run.
# ---------------------------------------------------------------------------
t_final = warpx.gett_new(0)
s_real = warpx.get_external_vector_potential_scale("uniform_file", t_final)
s_expect = s_of_t(t_final)

if comm.rank == 0:
    print(f"scale at t_final: realized {s_real:.12e} expected {s_expect:.12e}")
    assert abs(s_real - s_expect) < 1e-12, "sigmoid segments not realized"

    import yt

    yt.funcs.mylog.setLevel(0)

    def load(fn):
        ds = yt.load(fn)
        grid = ds.covering_grid(
            level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
        )
        return ds, grid

    ds_c, g_c = load("diags/diag1000020")
    ds_r, g_r = load(REF_DIR)

    print("field equivalence vs the parser-driven run:")
    failures = 0
    # tolerance: the drive realizations differ at O((dt/tau)^2); measured
    # differences are orders below this bound (see the field printout)
    tols = {"B": 1e-6, "E": 1e-5, "rho": 1e-5}
    for field, comps in (("B", ["Br", "Bt", "Bz"]),
                         ("E", ["Er", "Et", "Ez"]),
                         ("rho", ["rho"])):
        scale = max(
            float(np.abs(g_r["boxlib", c].v).max()) for c in comps)
        for c in comps:
            diff = float(
                np.abs(g_c["boxlib", c].v - g_r["boxlib", c].v).max())
            rel = diff / scale
            print(f"  {c}: max |diff| = {diff:.3e}  rel = {rel:.3e}")
            if rel > tols[field]:
                failures += 1
                print(f"    FAIL (tol {tols[field]:.0e})")
    assert failures == 0, "circuit-driven run diverged from the parser-driven run"

    led = np.atleast_2d(np.loadtxt("diags/reducedfiles/circdiag.txt"))
    print(f"ledger rows: {led.shape[0]}, s_final = {led[-1, 2]:.6e}")

    print("circuit cylinder compression test PASSED")
