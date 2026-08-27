#!/usr/bin/env python3
"""Calibrated the reference code's electron-ion temperature equilibration.

A uniform static two-temperature state (u = 0, B = 0, eta = 0) is
exactly inert in every flux and work term, so the only dynamics is the
the reference code's eq_brate exchange (ntb.f90 658-700, z_eff = 1),

    dU_e/dt = +nu (p_i - p_e) = -dU_i/dt,
    nu = 4.75e-15 lnLambda (m_p/m_i) n / Te[eV]^{3/2},

with the reference code's NRL-style lnLambda branches and the rate FROZEN per
step from the step-old (n, Te). Because Te itself relaxes, nu changes
step to step: the reference solution below replicates the exact
per-step frozen-coefficient theta recurrence (branch logic, zero clamp,
monotonicity cap, theta amplification of the pair difference), which
independently pins the implemented coefficient, the branch structure,
the exchange antisymmetry, the per-step rate refresh, and the theta
weighting. U_e + U_i is conserved to round-off by construction (the
same product deposits with opposite signs).

Modes: "total" (E_i channel), "dual" (E_i AND the auxiliary U_i, which
the u = 0 Enzo re-sync pins equal), "cgl" (the isotropic -Q/3 / -2Q/3
pair split, which keeps an isotropic state exactly isotropic while
p_eff follows the same recurrence). The "total" deck also runs a
reversed variant (cold electrons, low-Te lnLambda branch) checking the
opposite relaxation direction.
"""

import sys

import numpy as np
import yt

from warpx_constants import m_p
from warpx_constants import elementary_charge as q_e

yt.set_log_level(50)

# Must match the input decks.
THETA = 0.5
NSTEPS = 30
GAMMA = 5.0 / 3.0
SCALE = 1.0

mode = sys.argv[3] if len(sys.argv) > 3 else "total"
assert mode in ("total", "dual", "cgl")

FIELDS = {
    "total": ("implicit_mhd_mass_density", "implicit_mhd_electron_energy",
              "implicit_mhd_ion_energy"),
    "dual": ("implicit_mhd_mass_density", "implicit_mhd_electron_energy",
             "implicit_mhd_ion_energy", "implicit_mhd_ion_internal_energy"),
    "cgl": ("implicit_mhd_mass_density", "implicit_mhd_electron_energy",
            "implicit_mhd_ion_parallel_energy",
            "implicit_mhd_ion_perp_energy"),
}[mode]


def load(path):
    ds = yt.load(path)
    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    data = {f: np.asarray(grid["boxlib", f]).squeeze() for f in FIELDS}
    return ds, data


def ion_internal(state):
    # u = 0 everywhere: E_i is purely internal under total/dual.
    if mode == "cgl":
        return (state["implicit_mhd_ion_parallel_energy"] +
                state["implicit_mhd_ion_perp_energy"])
    return state["implicit_mhd_ion_energy"]


ds_initial, initial = load(sys.argv[1])
ds_final, final = load(sys.argv[2])
dt = float(ds_final.current_time) / NSTEPS

number_density = np.mean(initial["implicit_mhd_mass_density"]) / m_p
pe = (GAMMA - 1.0) * np.mean(initial["implicit_mhd_electron_energy"])
pi = (GAMMA - 1.0) * np.mean(ion_internal(initial))

# --- Exact frozen-coefficient theta recurrence (the implemented map) ---
THREE_LN_TEN = 3.0 * np.log(10.0)
BRANCH_TE = np.exp(2.0)
RATE_COEFFICIENT = SCALE * 4.75e-15  # times m_p/m_i = 1 (proton deck)
RATE_CAP = 1.0 / ((2.0 * GAMMA - 2.0) * (1.0 - THETA) * dt)


def frozen_rate(pe_old):
    te = pe_old / (number_density * q_e)
    if te <= BRANCH_TE:
        lnl = 23.0 + THREE_LN_TEN - 0.5 * np.log(number_density) \
            + 1.5 * np.log(te)
    else:
        lnl = 24.0 + THREE_LN_TEN - 0.5 * np.log(number_density) \
            + np.log(te)
    lnl = max(lnl, 0.0)
    return min(RATE_COEFFICIENT * lnl * number_density / te**1.5, RATE_CAP)


pressure_sum = pe + pi
difference = pi - pe
for _ in range(NSTEPS):
    decay = (2.0 * GAMMA - 2.0) * frozen_rate(pe) * dt
    difference *= (1.0 - (1.0 - THETA) * decay) / (1.0 + THETA * decay)
    pe = 0.5 * (pressure_sum - difference)
predicted_difference = difference

measured_difference = (GAMMA - 1.0) * (
    np.mean(ion_internal(final)) -
    np.mean(final["implicit_mhd_electron_energy"]))
relative_error = abs(measured_difference - predicted_difference) / \
    abs(predicted_difference)

initial_total = np.mean(initial["implicit_mhd_electron_energy"] +
                        ion_internal(initial))
final_total = np.mean(final["implicit_mhd_electron_energy"] +
                      ion_internal(final))
conservation = abs(final_total - initial_total) / initial_total
uniformity = np.max(np.abs(
    final["implicit_mhd_electron_energy"] -
    np.mean(final["implicit_mhd_electron_energy"]))) / initial_total

initial_difference = (GAMMA - 1.0) * (
    np.mean(ion_internal(initial)) -
    np.mean(initial["implicit_mhd_electron_energy"]))

print(f"initial p_i - p_e          {initial_difference:.6e} Pa")
print(f"predicted final p_i - p_e  {predicted_difference:.6e} Pa")
print(f"measured final p_i - p_e   {measured_difference:.6e} Pa")
print(f"relative error             {relative_error:.3e}")
print(f"U_e + U_i drift            {conservation:.3e}")
print(f"final spatial nonuniformity {uniformity:.3e}")

# The exchange must have acted (the difference decayed substantially)
# and in the right DIRECTION (toward equal temperatures, never past).
assert abs(measured_difference) < 0.5 * abs(initial_difference)
assert measured_difference * initial_difference > 0.0
# Electrons moved toward the ions: U_e changed with the sign of
# (T_i - T_e), in BOTH deck variants.
electron_change = np.mean(final["implicit_mhd_electron_energy"]) - \
    np.mean(initial["implicit_mhd_electron_energy"])
assert electron_change * initial_difference > 0.0

assert relative_error < 1.0e-6, (
    f"equilibration rate off by {relative_error:.3e}"
)
assert conservation < 1.0e-11
assert uniformity < 1.0e-11

if mode == "dual":
    # u = 0: the end-of-step Enzo re-sync pins the auxiliary internal
    # channel to the conservative stock, so a missing U_i deposit would
    # still show up as a blend/recurrence error above; assert the pair
    # explicitly anyway.
    sync_error = np.max(np.abs(
        final["implicit_mhd_ion_energy"] -
        final["implicit_mhd_ion_internal_energy"])) / \
        np.mean(final["implicit_mhd_ion_energy"])
    print(f"dual E_i vs U_i mismatch   {sync_error:.3e}")
    assert sync_error < 1.0e-9

if mode == "cgl":
    # The isotropic (1/3, 2/3) deposit keeps dT_par = dT_perp exactly:
    # an initially isotropic state must stay isotropic to round-off.
    p_par = 2.0 * final["implicit_mhd_ion_parallel_energy"]
    p_perp = final["implicit_mhd_ion_perp_energy"]
    anisotropy = np.max(np.abs(p_perp - p_par)) / np.mean(p_perp)
    print(f"cgl residual anisotropy    {anisotropy:.3e}")
    assert anisotropy < 1.0e-11

print("PASS")
