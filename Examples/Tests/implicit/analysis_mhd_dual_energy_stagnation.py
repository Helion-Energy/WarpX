#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# 1D converging-flow stagnation under ion_closure = dual_energy: uniform
# initial entropy, so the smooth compression must keep the ion adiabat
# p_i / rho^gamma at its initial value -- the dual closure heats ONLY
# through the physical blended PdV work (plus the small prescribed
# viscous heating), and the blended pressure is checked against the
# analytic adiabat cell by cell. The total_energy contrast twin measures
# the additional cancellation heating on the identical flow
# (analysis_mhd_dual_energy_stagnation_total.py).

import sys

import numpy as np
import warpx_constants as constants
from analysis_mhd_dual_energy_common import blended_pressure, load_fluid_state

gamma_i = 5.0 / 3.0
Ti0_eV = 10.0
n0 = 1.0e20
rho0 = n0 * constants.proton_mass
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

# The run must have compressed appreciably for the check to mean anything.
compression = state["rho"].max() / rho0
print(f"dual_energy stagnation: max compression rho/rho0 = {compression:.4f}")
assert compression > 1.2, "stagnation test failed to compress"

adiabat = pressure / state["rho"] ** gamma_i
adiabat0 = P0 / rho0**gamma_i
adiabat_error = np.abs(adiabat / adiabat0 - 1.0)
print(f"dual_energy stagnation: max adiabat error = {adiabat_error.max():.4e}")

# Physical-PdV-only heating: the blended pressure follows the analytic
# adiabat to within the prescribed viscous heating and the scheme's
# truncation (measured 4.6% at these parameters; tolerance ~1.7x that).
assert adiabat_error.max() < 0.08, "dual_energy adiabat violated"

print("dual_energy stagnation: PASS")
