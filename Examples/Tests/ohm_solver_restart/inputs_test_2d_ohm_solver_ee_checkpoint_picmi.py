#!/usr/bin/env python3
#
# --- Checkpoint/restart equivalence test for the hybrid-PIC (Ohm's law)
# --- solver WITH the QDSMC electron energy equation. Same drive and
# --- restart-hostile ingredients as the base checkpoint deck, plus Joule
# --- heating so T_e evolves away from the adiabat seed before the
# --- checkpoint is written. T_e is evolved state under the energy
# --- equation: the checkpoint must carry it and the restart must restore
# --- it WITHOUT re-running the adiabat seed (the m_qdsmc_te_seeded latch)
# --- — a restart that re-seeds T_e from rho diverges from the
# --- uninterrupted run in Te (and, through Pe, in the fields) at the
# --- first step after the checkpoint.

import sys

from pywarpx import picmi

constants = picmi.constants

# The restarted arm of the test pair passes amr.restart='<chk>' on the
# command line (the native-input idiom); forward it into PICMI explicitly.
restart_from = None
for _arg in sys.argv[1:]:
    if _arg.startswith("amr.restart="):
        restart_from = _arg.split("=", 1)[1].strip("'\"")

# Plasma parameters (arbitrary but hybrid-reasonable)
n0 = 1.0e19  # m^-3
B0 = 0.1  # T
T_i = 10.0  # eV
T_e = 10.0  # eV
m_ion = 4.0 * constants.m_p

NX = 48
NZ = 48
LX = 1.0
LZ = 1.0

DT = 2.0e-9
MAX_STEPS = 10

vth_i = (constants.q_e * T_i / m_ion) ** 0.5

grid = picmi.Cartesian2DGrid(
    number_of_cells=[NX, NZ],
    lower_bound=[-LX / 2.0, -LZ / 2.0],
    upper_bound=[LX / 2.0, LZ / 2.0],
    lower_boundary_conditions=["periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic"],
    lower_boundary_conditions_particles=["periodic", "periodic"],
    upper_boundary_conditions_particles=["periodic", "periodic"],
    warpx_max_grid_size=48,
)

# External vector potential with a time envelope: Az(x) ramps in, driving a
# uniform in-plane E_y = -dAz/dt and B_y = -dAz/dx through the split-field
# (subtract/add telescoping) machinery whose restart path this test guards.
A_ext = {
    "uniform_ramp": {
        "Ax_external_function": "0",
        "Ay_external_function": "0",
        "Az_external_function": f"0.05*{B0}*x",
        "A_time_external_function": f"sin(0.5*pi*t/({MAX_STEPS}*{DT}))",
    },
}

solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=5.0 / 3.0,
    Te=T_e,
    n0=n0,
    n_floor=0.05 * n0,
    plasma_resistivity=1.0e-6,
    plasma_hyper_resistivity=1.0e-8,
    substeps=10,
    A_external=A_ext,
    solve_electron_energy_equation=True,
    include_joule_heating=True,
    # Heating eta >> E-solve eta (the split-resistivity path): drives T_e
    # far from the adiabat seed by the checkpoint step, so a restart that
    # loses the evolved T_e (re-seeding it from rho) fails the comparison
    # decisively instead of hiding inside the re-entry tolerance.
    joule_heating_resistivity=1.0e-3,
)

# Parsed analytic initial B: uniform Bz plus an out-of-plane By modulation
# (any By(x,z) is discretely divergence-free in 2D XZ, so the initial
# projection clean stays off). The restart must NOT re-apply this seed on
# top of the checkpoint-restored evolved field.
initial_B = picmi.AnalyticInitialField(
    Bx_expression="0",
    By_expression=f"0.2*{B0}*cos(2*pi*x/{LX})",
    Bz_expression=f"{B0}",
    warpx_do_initial_div_cleaning=False,
)

ions = picmi.Species(
    name="ions",
    charge="q_e",
    mass=m_ion,
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=f"{n0}*(1+0.1*cos(2*pi*x/{LX})*cos(2*pi*z/{LZ}))",
        momentum_expressions=["0", "0", "0"],
        warpx_momentum_spread_expressions=[f"{vth_i}"] * 3,
    ),
)

sim = picmi.Simulation(
    solver=solver,
    time_step_size=DT,
    max_steps=MAX_STEPS,
    verbose=1,
    particle_shape=1,
)

if restart_from is not None:
    sim.amr_restart = restart_from

sim.add_applied_field(initial_B)
sim.add_species(
    ions,
    layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=4),
)

field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=10,
    data_list=["B", "E", "rho", "J", "Te"],
)
sim.add_diagnostic(field_diag)

checkpoint = picmi.Checkpoint(name="chk", period=5)
sim.add_diagnostic(checkpoint)

sim.step()
