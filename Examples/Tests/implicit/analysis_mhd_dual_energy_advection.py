#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# Cold supersonic advection of a transverse-velocity ripple under
# ion_closure = dual_energy: div u = 0, the exact solution advects
# everything unchanged, and only the small prescribed viscous shear
# heating acts. The blended ion temperature must stay cold (the
# total_energy contrast twin measures the catastrophic-cancellation
# heating of the recovered pressure on the identical flow, see
# analysis_mhd_dual_energy_advection_total.py).

import sys

import numpy as np
import warpx_constants as constants
from analysis_mhd_dual_energy_common import blended_pressure, load_fluid_state

gamma_i = 5.0 / 3.0
Ti0_eV = 1.0
n0 = 1.0e20
P0 = n0 * Ti0_eV * constants.elementary_charge
pressure_floor = 1.0e-6 * P0

state = load_fluid_state(sys.argv[1])
assert state["ion_internal_energy"] is not None

pressure = blended_pressure(
    state["ion_energy"],
    state["kinetic_energy"],
    state["ion_internal_energy"],
    gamma_i,
    pressure_floor,
)
number_density = state["rho"] / constants.proton_mass
Ti_eV = pressure / (number_density * constants.elementary_charge)

# The exact solution advects the ripple unchanged (div u = 0) and keeps
# the uniform pressure at P0. The scheme numerically dissipates ripple
# kinetic energy each step; the conservative E_i books it as heat, but
# the mixmaster re-sync drains that drift into the U_i anchor every
# step, so the blended pressure sits at a BOUNDED AR(1) equilibrium of
# O(one step's truncation) above P0 (measured 24% at these parameters)
# instead of accumulating -- the total_energy twin on the identical flow
# integrates the full drift to ~27x Ti0 (see the contrast analysis).
pressure_error = np.abs(pressure / P0 - 1.0)
print(f"dual_energy advection: max |p/P0 - 1| = {pressure_error.max():.4e}")
print(f"dual_energy advection: max Ti = {Ti_eV.max():.6e} eV (Ti0 = {Ti0_eV})")

assert pressure_error.max() < 0.35, "dual_energy pressure bound violated"
assert Ti_eV.max() < 1.4 * Ti0_eV, "dual_energy Ti overheated"

# The auxiliary internal energy stayed positive everywhere.
assert state["ion_internal_energy"].min() > 0.0

print("dual_energy advection: PASS")
