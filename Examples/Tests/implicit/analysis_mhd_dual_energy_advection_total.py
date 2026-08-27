#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# total_energy contrast twin of the cold supersonic advection test: on
# the identical flow the recovered pressure p(E_i - KE) integrates the
# total-minus-kinetic cancellation error of the ~450x KE-dominated state
# into spurious heating, while the dual_energy run (read from the
# dependency's output) stays cold -- the measured production signature
# (the ~600 eV artificial axis column) this closure exists to remove.

import sys

import warpx_constants as constants
from analysis_mhd_dual_energy_common import (
    blended_pressure,
    load_fluid_state,
    recovered_pressure,
)

gamma_i = 5.0 / 3.0
Ti0_eV = 1.0
n0 = 1.0e20
P0 = n0 * Ti0_eV * constants.elementary_charge
pressure_floor = 1.0e-6 * P0

total_state = load_fluid_state(sys.argv[1])
dual_state = load_fluid_state(sys.argv[2])

pressure_total_run = recovered_pressure(
    total_state["ion_energy"],
    total_state["kinetic_energy"],
    gamma_i,
    pressure_floor,
)
Ti_total = pressure_total_run / (
    total_state["rho"] / constants.proton_mass * constants.elementary_charge
)

pressure_dual_run = blended_pressure(
    dual_state["ion_energy"],
    dual_state["kinetic_energy"],
    dual_state["ion_internal_energy"],
    gamma_i,
    pressure_floor,
)
Ti_dual = pressure_dual_run / (
    dual_state["rho"] / constants.proton_mass * constants.elementary_charge
)

print(f"total_energy advection: max Ti = {Ti_total.max():.6e} eV")
print(f"dual_energy  advection: max Ti = {Ti_dual.max():.6e} eV")
print(
    "cancellation heating contrast: "
    f"{(Ti_total.max() - Ti0_eV) / max(Ti_dual.max() - Ti0_eV, 1.0e-30):.3e}"
)

# The total_energy run visibly overheats where the dual run stays cold.
assert Ti_total.max() > 2.0 * Ti0_eV, (
    "expected the total_energy closure to show cancellation heating "
    f"(max Ti = {Ti_total.max():.3e} eV)"
)
assert Ti_total.max() > 5.0 * Ti_dual.max(), (
    "expected the total_energy heating to dominate the dual_energy "
    f"temperature ({Ti_total.max():.3e} vs {Ti_dual.max():.3e} eV)"
)

print("total_energy advection contrast: PASS")
