#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Gate-inactive bit-identity of the electron-inertia dust gate.

Same deck as analysis_mhd_stiff_whistler_inertia_drift.py with the dust
gate armed BELOW the uniform density (electron_inertia_linear_below =
rho0/4): the blend weight w = (1 + tanh(s (1 + s^2)))/2 with
s = (rho - rho_c)/(0.3 rho_c) evaluates at s = 10 to
w = (1 + tanh(1010))/2 -- EXACTLY 1.0 in double precision (tanh saturates
to 1.0 beyond ~19.06, i.e. beyond rho ~ 1.77 rho_c), so an armed-but-
inactive gate must reproduce the ungated baseline BIT FOR BIT. The
analytic above-threshold tail of the blend, 1 - w <= exp(-2 s (1 + s^2)),
is exp(-606) ~ 1e-263 at 3 rho_c -- far inside the 1e-12 construction
requirement, proven numerically below.

Three gates:
  1. np.array_equal bit-identity of every output field against the
     ungated baseline test's plotfile (dependency).
  2. The same tight drift-Doppler capped dispersion as the baseline --
     under an adversarially FLIPPED blend (linear above, full below) this
     run executes the LINEAR inertia form at the seeded density and lands
     on the W_lin branch instead: a 1.26e-1 relative dispersion mismatch
     against a 7.3e-14 pass residual. This is the tripwire.
  3. The Newton/GMRES iteration history must match the baseline exactly
     (bit-identical residuals converge identically).
"""

import re
import sys
from pathlib import Path

import numpy as np
import warpx_constants as constants
import yt

baseline_directory = "../test_1d_theta_implicit_mhd_stiff_whistler_inertia_drift"


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


def input_value(name):
    # Command-line overrides append AFTER the input-file entry in the
    # dumped table; the last occurrence is the effective value.
    matches = re.findall(
        rf"^{re.escape(name)}\s*=\s*([^#\s]+)", used_inputs, re.MULTILINE
    )
    assert matches
    return matches[-1]


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])
_, baseline_final = get_data(f"{baseline_directory}/diags/diag000001")
used_inputs = Path("warpx_used_inputs").read_text()

number_density = 1.0e20
mass_density_reference = number_density * constants.proton_mass
background_field = 0.1
drift_velocity = 5.0e4
theta = 0.5
effective_electron_mass = constants.proton_mass
electron_skin_depth = np.sqrt(
    effective_electron_mass
    / (constants.mu_0 * number_density * constants.elementary_charge**2)
)

# The gate must be armed at rho0/4 (deep in the exact-full-form regime).
assert input_value("hybrid_pic_model.electron_inertia_linear_below") == "rho_gate"
assert input_value("my_constants.rho_gate") == "rho0/4"
gate_density = mass_density_reference / 4.0


# ---- Blend-construction proof (the 1e-12-at-3x requirement) ----
# w(rho) = (1 + tanh(s (1 + s^2)))/2, s = (rho - rho_c)/(0.3 rho_c).
def blend_weight(rho, rho_c):
    s = (rho - rho_c) / (0.3 * rho_c)
    return 0.5 * (1.0 + np.tanh(s * (1.0 + s * s)))


# Exactly 1.0 in double precision at this deck's density (4x the gate),
# and already within 1e-12 of 1 at 3x the gate (analytically exp(-606)).
assert blend_weight(mass_density_reference, gate_density) == 1.0
assert 1.0 - blend_weight(3.0 * gate_density, gate_density) < 1.0e-12
# The analytic tail bound at 3x: 1 - w <= exp(-2 s (1 + s^2)), s = 20/3.
s3 = (3.0 - 1.0) / 0.3
analytic_tail_log10 = -2.0 * s3 * (1.0 + s3 * s3) / np.log(10.0)
assert analytic_tail_log10 < -12.0
# And exactly 0.0 deep below (the gated dust-band regime).
assert blend_weight(0.2 * gate_density, gate_density) == 0.0

# ---- Gate 1: bit-identity against the ungated baseline ----
for field in (
    "Bx",
    "By",
    "Bz",
    "Ex",
    "Ey",
    "Ez",
    "implicit_mhd_mass_density",
    "implicit_mhd_momentum_density",
    "implicit_mhd_electron_energy",
):
    mine = final["boxlib", field].value.ravel()
    theirs = baseline_final["boxlib", field].value.ravel()
    assert np.array_equal(mine, theirs), (
        f"gate-inactive run differs from the ungated baseline in {field}: "
        f"max |diff| = {np.max(np.abs(mine - theirs)):.3e} (the armed dust "
        "gate perturbed above-threshold physics)"
    )

# ---- Gate 2: the tight drift-Doppler dispersion (the flip tripwire) ----
initial_bx = initial["boxlib", "Bx"].value.ravel()
initial_by = initial["boxlib", "By"].value.ravel()
final_bx = final["boxlib", "Bx"].value.ravel()
final_by = final["boxlib", "By"].value.ravel()

number_of_cells = initial_bx.size
domain_length = float((initial_ds.domain_right_edge - initial_ds.domain_left_edge)[0])
cell_size = domain_length / number_of_cells
dt = float(final_ds.current_time - initial_ds.current_time)
hall_diffusivity = background_field / (
    constants.mu_0 * constants.elementary_charge * number_density
)

wavenumbers = 2.0 * np.pi * np.fft.fftfreq(number_of_cells, d=cell_size)
effective_wavenumbers = 2.0 * np.sin(0.5 * wavenumbers * cell_size) / cell_size
central_wavenumbers = np.sin(wavenumbers * cell_size) / cell_size
kappa = (effective_wavenumbers * electron_skin_depth) ** 2
omega_dt = hall_diffusivity * effective_wavenumbers**2 * dt
doppler_dt = central_wavenumbers * drift_velocity * dt


def advance(total_omega_dt):
    return ((1.0 + kappa) + 1.0j * (1.0 - theta) * total_omega_dt) / (
        (1.0 + kappa) - 1.0j * theta * total_omega_dt
    )


amplification_full = advance(omega_dt - (1.0 + kappa) * doppler_dt)
amplification_linear = advance(omega_dt - doppler_dt)

initial_polarization = initial_bx + 1.0j * initial_by
final_polarization = final_bx + 1.0j * final_by
initial_spectrum = np.fft.fft(initial_polarization)
expected_polarization = np.fft.ifft(amplification_full * initial_spectrum)
linear_polarization = np.fft.ifft(amplification_linear * initial_spectrum)

np.testing.assert_allclose(
    final_polarization,
    expected_polarization,
    rtol=2.0e-8,
    atol=2.0e-12,
)
distance_to_full = np.linalg.norm(final_polarization - expected_polarization)
distance_to_linear = np.linalg.norm(final_polarization - linear_polarization)
linear_branch_margin = distance_to_linear / max(distance_to_full, 1e-30)
assert linear_branch_margin > 1.0e4, (
    f"flipped-blend tripwire: margin {linear_branch_margin:.3e} (the "
    "gate-inactive run does not carry the full inertia form's advective "
    "Doppler at the seeded modes)"
)

# ---- Gate 3: identical solver history ----
# newton.txt appends across reruns of a build tree, so compare this
# step's (last) solve line, which both tests wrote exactly once per run.
newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
baseline_history = np.atleast_2d(np.loadtxt(f"{baseline_directory}/diags/newton.txt"))
assert np.array_equal(newton_history[-1], baseline_history[-1]), (
    "gate-inactive Newton/GMRES solve differs from the ungated baseline: "
    f"{newton_history[-1]} vs {baseline_history[-1]}"
)
last_solve = newton_history[-1]

print(f"gate density rho_c={gate_density:.6g} kg/m3 (deck density = 4 rho_c)")
print(
    f"blend weight at deck density={blend_weight(mass_density_reference, gate_density)!r}"
)
print(f"analytic blend tail at 3 rho_c: 10^{analytic_tail_log10:.1f}")
print("bit-identity to ungated baseline: PASS (all fields, exact)")
print(f"full-branch |B - G_full B0|={distance_to_full:.6e}")
print(f"linear-branch margin={linear_branch_margin:.6e}")
print(f"Newton iterations={int(last_solve[2])}")
print(f"GMRES iterations={int(last_solve[7])}")
