#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# Absolute low-internal-energy guard of the dual-energy closure
# (implicit_mhd.dual_energy_internal_guard) on the converging-flow
# stagnation deck with the guard above every reachable U_i: (i) the
# end-of-step rewrite makes E_i = KE + U_i in every cell to round-off (the
# window is exactly 0, the blend is the pure internal-energy pressure, and
# the U_i overwrite is skipped); (ii) the ion adiabat p_i / rho^gamma of the
# pure internal-energy closure is held at least as well as the dual run's
# (heating only through the blended PdV work and the prescribed viscous
# heating); (iii) the guard ledger has one row per step, books every cell,
# and its cumulative discard is finite.

import sys

import numpy as np
import warpx_constants as constants
from analysis_mhd_dual_energy_common import load_fluid_state, smooth_positive_floor

gamma_i = 5.0 / 3.0
Ti0_eV = 10.0
n0 = 1.0e20
nz = 128
max_step = 32
rho0 = n0 * constants.proton_mass
P0 = n0 * Ti0_eV * constants.elementary_charge
pressure_floor = 1.0e-6 * P0

state = load_fluid_state(sys.argv[1])
initial = load_fluid_state(sys.argv[2])
U = state["ion_internal_energy"]
assert U is not None
E = state["ion_energy"]
K = state["kinetic_energy"]

# (i) guarded rewrite: E_i = KE + smooth-floor((gamma-1) U_i)/(gamma-1) = KE + U_i
# away from the pressure floor (every cell here sits far above it).
internal_from_U = smooth_positive_floor((gamma_i - 1.0) * U, pressure_floor) / (
    gamma_i - 1.0
)
identity_error = np.abs(E - K - internal_from_U) / np.maximum(U, 1e-300)
print(f"dual_energy guard: max |E_i - KE - U_i|/U_i = {identity_error.max():.3e}")
assert identity_error.max() < 1.0e-10, "guarded rewrite E_i = KE + U_i violated"
assert np.isfinite(U).all() and (U > 0).all()

# (ii) the pure internal-energy closure holds the ion adiabat
compression = state["rho"].max() / rho0
print(f"dual_energy guard: max compression rho/rho0 = {compression:.4f}")
assert compression > 1.2, "stagnation flow failed to compress"
pressure = (gamma_i - 1.0) * U
adiabat_error = np.abs(pressure / state["rho"] ** gamma_i / (P0 / rho0**gamma_i) - 1.0)
print(f"dual_energy guard: max adiabat error = {adiabat_error.max():.4e}")
assert adiabat_error.max() < 0.08, "guarded (pure internal) adiabat violated"

# (iii) the guard ledger: rows = steps, every cell guarded, finite cumulative discard
rows = np.loadtxt("diags/dual_energy_guard_ledger.txt", ndmin=2)
print(
    f"dual_energy guard: ledger rows = {rows.shape[0]}, cells = {int(rows[:, 1].min())}..{int(rows[:, 1].max())}, "
    f"cumulative discard = {rows[-1, 2]:.6e} J/m^2 (domain U_i = {U.sum() * 1.0 / nz:.6e} J/m^3 mean)"
)
assert rows.shape[0] == max_step, "ledger must hold one row per step"
assert (rows[:, 0] == np.arange(1, max_step + 1)).all(), "ledger step column"
assert (rows[:, 1] == nz).all(), "the guard must cover every cell of this deck"
assert np.isfinite(rows[:, 2]).all(), "ledger discard must be finite"

# (iv) value gate: in this periodic box the only non-conservative change of
# sum(E_i + U_e) is the guarded rewrite, so the domain budget closes on the
# ledger: [sum(E_i + U_e)](32) - [sum(E_i + U_e)](0) = -discarded_cum (per unit
# area in 1D: both sides carry the same dz).
dz = 1.0 / nz
budget = (np.sum(E + state["electron_energy"]) - np.sum(initial["ion_energy"] + initial["electron_energy"])) * dz
print(f"dual_energy guard: value gate: energy change {budget:.9e} vs -ledger {-rows[-1, 2]:.9e} J/m^2 "
      f"(rel {abs(budget + rows[-1, 2]) / abs(rows[-1, 2]):.3e})")
np.testing.assert_allclose(budget, -rows[-1, 2], rtol=1.0e-9)

print("dual_energy guard: PASS")
