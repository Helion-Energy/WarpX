#!/usr/bin/env python3
"""Background MCC collision rate for a projectile as heavy as the gas.

Two cold, monoenergetic ion beams cross a cold background of the same mass.
The only process is elastic scattering reversed in the centre of mass, which
for equal masses is a velocity swap: a collided ion leaves with the sampled
neutral's velocity, near zero at 1 K. So the uncollided fraction of each beam
decays as exp(-nu t) with nu = n * sigma * v_rel exactly, and the analysis
reads nu back from the survivors.

The cross section is flat, sigma0 from 0 to 5000 eV. The collision energy WarpX
tabulates against is the reduced-mass energy E = mu v_rel^2 / 2, so the beams
sit at E = 1000 eV and 4000 eV on that axis.

The high beam is the point of the test. The null-collision majorant is scanned
over E up to 5000 eV, and it must convert each E to v_rel with the reduced mass
mu = m/2. Converting with the projectile mass instead underestimates the
majorant by sqrt(2), and above E = 2500 eV the true frequency then exceeds it:
the high beam collides at 5000/sqrt(2 * 4000 * 5000) = 0.79 of its rate, while
the low beam, below that energy, is unaffected. For electrons on a heavy gas
mu is m_e to within 1e-4, which is why nothing else in the suite sees this.

Fields are not evolved, so nothing but the collisions changes a velocity.
"""

import numpy as np
from scipy import constants

from pywarpx import picmi

# must match the analysis script
m_ion = 4.0 * constants.m_p  # kg, projectile and background
mu = 0.5 * m_ion  # reduced mass of the pair
n_gas = 1.0e20  # m^-3
sigma0 = 1.0e-19  # m^2
E_low = 1000.0  # eV, reduced-mass collision energy of the low beam
E_high = 4000.0  # eV, of the high beam
dt = 1.5e-9  # s
max_steps = 108

# nu_max * dt = 0.0104 here, so the linearised null-collision acceptance
# costs ~0.5% of the rate, well inside the analysis tolerance. 108 steps
# leaves e^-1 of the high beam and e^-0.5 of the low one.


def v_rel(energy_ev):
    """Relative speed at which the reduced-mass energy is energy_ev."""
    return np.sqrt(2.0 * energy_ev * constants.e / mu)


# Flat to the 5000 eV end of the majorant scan, so the scan's upper limit is
# the default one and the high beam is the only thing above 2500 eV.
xsec = "flat_cross_section.dat"
energies = np.linspace(0.0, 5000.0, 5001)
np.savetxt(xsec, np.c_[energies, np.full_like(energies, sigma0)])

grid = picmi.Cartesian3DGrid(
    number_of_cells=[8, 8, 8],
    lower_bound=[0.0, 0.0, 0.0],
    upper_bound=[1.0, 1.0, 1.0],
    lower_boundary_conditions=["periodic"] * 3,
    upper_boundary_conditions=["periodic"] * 3,
    lower_boundary_conditions_particles=["periodic"] * 3,
    upper_boundary_conditions_particles=["periodic"] * 3,
    warpx_max_grid_size=8,
)

# PICMI requires a solver object; setting the method to "none" after
# construction is what reaches algo.maxwell_solver, so fields never evolve.
solver = picmi.ElectromagneticSolver(grid=grid, method=None)
solver.method = "none"

beams = []
collisions = []
for label, energy in (("low", E_low), ("high", E_high)):
    beam = picmi.Species(
        name=f"ions_{label}",
        charge="q_e",
        mass=m_ion,
        initial_distribution=picmi.UniformDistribution(
            density=1.0e3,
            directed_velocity=[0.0, 0.0, v_rel(energy)],
        ),
    )
    beams.append(beam)
    collisions.append(
        picmi.MCCCollisions(
            name=f"mcc_{label}",
            species=beam,
            background_density=n_gas,
            background_temperature=1.0,
            background_mass=m_ion,
            scattering_processes={
                "elastic": {
                    "cross_section": xsec,
                    "scattering_angle_model": "backward",
                }
            },
        )
    )

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=max_steps,
    warpx_collisions=collisions,
    warpx_serialize_initial_conditions=True,
    warpx_use_filter=False,
    verbose=1,
)
for beam in beams:
    sim.add_species(
        beam,
        layout=picmi.PseudoRandomLayout(n_macroparticles_per_cell=200, grid=grid),
    )

sim.add_diagnostic(
    picmi.ParticleDiagnostic(
        name="diag1",
        period=max_steps,
        species=beams,
        data_list=["uz", "weighting"],
        warpx_format="openpmd",
    )
)

sim.step(max_steps)
