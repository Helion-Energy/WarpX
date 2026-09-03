#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Analytic gate of the time-staged ion viscosity (implicit_mhd.viscosity(t)).

The deck is a decaying transverse shear: a uniform column at rest
carrying a single transverse Fourier mode v_y = A sin(2 pi z / Lz),
uniform density and pressures, and a dynamically inert guide field. With
fluid_flux = central (no Riemann dissipation) the ONLY momentum sink is
the explicit viscous face stress, and the mode is a discrete eigenmode of
the face-difference Laplacian, so the amplitude obeys exactly

    dA/dt = -nu(t) k_eff^2 A,   k_eff^2 = (4/dz^2) sin^2(k dz/2)

with nu(t) frozen per step at the theta-stage time by
UpdateStagedViscosity (the reference code's glob_mod card semantics: constant
within a segment). Measuring -d ln A/dt between consecutive outputs
therefore reads back the tabulated viscosity directly.

Modes:

  scalar   -- the anchor deck (implicit_mhd.viscosity = 2000, a plain
              scalar): every per-step rate equals nu_hi k_eff^2, and the
              cumulative decay matches exp(-nu_hi k_eff^2 t).
  table    -- the staged deck (implicit_mhd.viscosity(t), the tanh blend
              2000 -> 500 at t_switch that the production formation deck's
              --viscosity-table compiler emits for '2000,500@8'):
              (a) both PLATEAUS match their analytic rate,
              (b) every per-step rate tracks nu(t_n + theta dt) k_eff^2
                  through the transition,
              (c) the transition is RESOLVED and smooth -- it takes many
                  steps, the rate is monotone across it, and the half-way
                  crossing lands on t_switch (a hard switch would jump
                  between the two plateaus in a single step).
  identity -- the staged deck with the table collapsed to the single
              value nu_hi: every output field of every plotfile is
              BIT-IDENTICAL (np.array_equal) to the scalar anchor twin,
              and so is the Newton iteration history. This pins the
              claim that the new parser path costs nothing when the
              table has one entry.

Usage:
    analysis_mhd_viscosity_table.py <mode> <initial plotfile> <final plotfile>
"""

import sys

import numpy as np
import yt

# Deck constants (inputs_test_1d_theta_implicit_mhd_viscosity_shear and
# its staged twin); m_p/q_e are the WarpX parser constants (CODATA 2022).
m_p = 1.67262192595e-27
q_e = 1.602176634e-19
n0 = 1.0e20
rho0 = n0 * m_p
Ti = 100.0
P0 = n0 * Ti * q_e
gamma = 5.0 / 3.0
nz = 64
Lz = 1.0
dz = Lz / nz
sound_speed = np.sqrt(gamma * P0 / rho0)
shear = 1.0e-3 * sound_speed
nu_hi = 2000.0
nu_lo = 500.0
t_switch = 8.0e-6
tau_switch = 2.0e-6
dt = 2.5e-7
theta = 0.5
n_steps = 64

# The discrete eigenvalue of the second-order face-difference Laplacian
# for the seeded mode (0.08% below the continuum k^2 at this resolution).
k_mode = 2.0 * np.pi / Lz
k_eff2 = 4.0 / dz**2 * np.sin(0.5 * k_mode * dz) ** 2

baseline_directory = "../test_1d_theta_implicit_mhd_viscosity_shear"

FIELDS = [
    "Bz",
    "implicit_mhd_mass_density",
    "implicit_mhd_momentum_y",
    "implicit_mhd_ion_energy",
    "implicit_mhd_electron_energy",
]


def staged_viscosity(t):
    """The deck's tanh table, evaluated exactly as the parser does."""
    return nu_hi + 0.5 * (nu_lo - nu_hi) * (1.0 + np.tanh((t - t_switch) / tau_switch))


def get_fields(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return {name: data["boxlib", name].value.ravel() for name in FIELDS}


def mode_amplitude(fields):
    """Project v_y = m_y/rho onto the seeded sine (an exact eigenmode)."""
    velocity = fields["implicit_mhd_momentum_y"] / fields["implicit_mhd_mass_density"]
    z_centers = (np.arange(nz) + 0.5) * dz
    basis = np.sin(k_mode * z_centers)
    return 2.0 * np.dot(velocity, basis) / nz


mode = sys.argv[1]
assert mode in ("scalar", "table", "identity"), f"unknown mode {mode}"
initial_directory = sys.argv[2]
prefix = initial_directory[: -len("000000")]
snapshots = [get_fields(f"{prefix}{step:06d}") for step in range(n_steps + 1)]

# The seeded state must be the clean eigenmode: the projection recovers
# the full amplitude, so nothing else is hiding in v_y.
amplitudes = np.array([mode_amplitude(snapshot) for snapshot in snapshots])
assert np.isclose(amplitudes[0], shear, rtol=1.0e-10, atol=0.0), (
    f"seeded amplitude {amplitudes[0]:.9e} is not the deck's {shear:.9e}"
)
assert np.all(amplitudes > 0.0), "the shear mode changed sign (not a diffusive decay)"

# Per-step momentum-diffusion rate and the analytic prediction: the
# viscosity is frozen at the theta-stage time of each step, so the
# expected rate for step n (advancing t_n -> t_{n+1}) is
# nu(t_n + theta dt) k_eff^2.
step_times = dt * np.arange(n_steps)
measured_rate = -np.diff(np.log(amplitudes)) / dt
if mode == "table":
    expected_nu = staged_viscosity(step_times + theta * dt)
else:
    expected_nu = np.full(n_steps, nu_hi)
expected_rate = expected_nu * k_eff2
measured_nu = measured_rate / k_eff2

relative = np.abs(measured_rate / expected_rate - 1.0)
assert np.max(relative) < 0.01, (
    "the measured momentum-diffusion rate does not track the tabulated "
    f"viscosity: max relative error {np.max(relative):.3e} at step "
    f"{int(np.argmax(relative))}"
)
print(
    f"[{mode}] per-step rate tracks nu(t) k_eff^2 to "
    f"{np.max(relative):.2e} over {n_steps} steps "
    f"(k_eff^2 = {k_eff2:.4f} 1/m^2)"
)

# Cumulative decay: an independent, integral check of the same schedule.
expected_amplitude = amplitudes[0] * np.exp(-np.cumsum(expected_rate) * dt)
cumulative = np.abs(amplitudes[1:] / expected_amplitude - 1.0)
assert np.max(cumulative) < 0.01, (
    f"cumulative decay off the analytic schedule by {np.max(cumulative):.3e}"
)
print(
    f"[{mode}] cumulative decay A/A0 = {amplitudes[-1] / amplitudes[0]:.6f} "
    f"vs analytic {expected_amplitude[-1] / amplitudes[0]:.6f}"
)

if mode == "table":
    # (a) The two PLATEAUS, measured 2.5 tanh widths clear of the switch
    # (where the table is within 1% of its plateau value).
    early = step_times + theta * dt < t_switch - 2.5 * tau_switch
    late = step_times + theta * dt > t_switch + 2.5 * tau_switch
    assert early.sum() >= 5 and late.sum() >= 5, "plateaus are too short to measure"
    nu_early = measured_nu[early].mean()
    nu_late = measured_nu[late].mean()
    assert np.isclose(nu_early, nu_hi, rtol=0.01, atol=0.0), (
        f"early plateau viscosity {nu_early:.3f} != {nu_hi}"
    )
    assert np.isclose(nu_late, nu_lo, rtol=0.01, atol=0.0), (
        f"late plateau viscosity {nu_late:.3f} != {nu_lo}"
    )
    print(
        f"[table] plateaus: {nu_early:.3f} m^2/s (analytic {nu_hi}) -> "
        f"{nu_late:.3f} m^2/s (analytic {nu_lo})"
    )

    # (b) The transition is RESOLVED, not a hard switch: many steps must
    # land strictly between the plateaus, the measured schedule must be
    # monotone to the measurement noise, and the half-way crossing must
    # sit on t_switch (a hard switch would jump the whole 1500 m^2/s in a
    # single step and put no sample in between).
    midpoint = 0.5 * (nu_hi + nu_lo)
    interior = (measured_nu < nu_hi - 0.05 * (nu_hi - nu_lo)) & (
        measured_nu > nu_lo + 0.05 * (nu_hi - nu_lo)
    )
    assert interior.sum() >= 8, (
        f"the transition is unresolved: only {interior.sum()} steps lie "
        "strictly between the two plateaus (a hard switch would give 0-1)"
    )
    noise = np.max(np.abs(measured_nu - expected_nu))
    assert np.all(np.diff(measured_nu) < 2.0 * noise), (
        "the staged viscosity is not monotone in time across the transition"
    )
    # measured_nu descends, so reverse for np.interp's ascending contract.
    t_half = np.interp(midpoint, measured_nu[::-1], (step_times + theta * dt)[::-1])
    assert abs(t_half - t_switch) < 0.15 * tau_switch, (
        f"the half-way crossing is at {t_half:.4e} s, not the tabulated "
        f"t_switch = {t_switch:.4e} s"
    )
    print(
        f"[table] transition resolved over {int(interior.sum())} steps "
        f"(tanh width {tau_switch:.1e} s = {tau_switch / dt:.0f} steps), "
        f"monotone, half-way crossing at {t_half:.4e} s"
    )

if mode == "identity":
    # Bit-identity against the scalar anchor twin: a one-entry table must
    # cost exactly nothing.
    for step in range(n_steps + 1):
        twin = get_fields(f"{baseline_directory}/{prefix}{step:06d}")
        for name in FIELDS:
            assert np.array_equal(snapshots[step][name], twin[name]), (
                f"{name} at step {step} differs from the scalar-viscosity "
                "anchor: the one-entry table is not bit-identical"
            )
    with open("diags/newton.txt") as handle:
        staged_history = handle.read()
    with open(f"{baseline_directory}/diags/newton.txt") as handle:
        anchor_history = handle.read()
    assert staged_history == anchor_history, (
        "the Newton iteration history differs from the scalar anchor"
    )
    print(
        f"[identity] all {len(FIELDS)} fields bit-identical to the scalar "
        f"anchor over {n_steps + 1} plotfiles, Newton history identical"
    )

print("PASS")
