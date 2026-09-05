#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Viscous-heating energy ledger of the theta-implicit MHD ion closures.

Deck: inputs_test_1d_theta_implicit_mhd_viscous_heating -- a decaying
circularly polarized transverse shear v_x = A cos(kz), v_y = A sin(kz) on
a uniform column at rest, B = 0, central flux, explicit viscosity nu.
Both components are discrete eigenmodes of the viscous face-difference
Laplacian and their face shears add to the same value on every face, so
the dissipation is uniform, the pressure stays uniform, div u = 0 and
v_z = 0 to round-off, and the ONLY energy exchange is transverse kinetic
energy -> ion internal energy through the viscous stress.

Modes (<booking>_<stage>):

  stress_work_cn  dual_energy, dual_energy_viscous_heating = stress_work,
                  viscous_theta = theta = 0.5
  stress_work_be  ... viscous_theta = 1.0 (the staged, backward-Euler
                  stress); the register books the exact KE removed, so
                  the identity still holds to round-off
  legacy_cn       dual_energy, the pre-2026-09 pointwise
                  rho_f nu |du/dn|^2 booking at theta_v = 0.5: exact here
                  (interior, donor states) -- documents that the two
                  bookings agree where the old one was right
  legacy_be       the same at viscous_theta = 1.0: the old booking heats
                  by exactly (1 + nu k_eff^2 dt/2) x the KE removed
  stress_work_cap dual_energy, stress_work, viscous_theta = 0.5, with the
                  free-streaming viscous cap binding
                  (viscous_flux_limit_factor = 0.3, decay 3.4 % slower):
                  the analytic decay-factor check is skipped (the cap
                  changes the operator) and the ledger identity must still
                  hold -- a register built from the pre-cap stress is off
                  by 3.3 %
  total_cn/be     ion_closure = total_energy: E_i is a conservative
                  register, sum(E_i) is constant to round-off (for either
                  stage) and its implied internal part gains the KE loss

Checks (all modes): the per-step decay factor of BOTH transverse
amplitudes equals the analytic theta-scheme factor of the staged viscous
operator; density and v_z stay at rest; sum(E_i) is constant.
Dual modes: sum(dU_i) against -sum(dKE) per plotfile (the identity, or
the legacy_be factor); the deposit is uniform across cells and
non-negative every step.

Usage:
    analysis_mhd_viscous_heating.py <mode> <initial plotfile> <final plotfile>
"""

import sys

import numpy as np
import yt

# Deck constants; m_p/q_e are the WarpX parser constants (CODATA 2022).
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
shear = 1.0e-1 * sound_speed
nu = 2000.0
dt = 2.5e-7
theta = 0.5
n_steps = 64

k_mode = 2.0 * np.pi / Lz
k_eff2 = 4.0 / dz**2 * np.sin(0.5 * k_mode * dz) ** 2
x_step = nu * k_eff2 * dt

FIELDS = [
    "implicit_mhd_mass_density",
    "implicit_mhd_momentum_x",
    "implicit_mhd_momentum_y",
    "implicit_mhd_momentum_z",
    "implicit_mhd_ion_energy",
    "implicit_mhd_electron_energy",
]

mode = sys.argv[1]
booking, stage = mode.rsplit("_", 1)
assert booking in ("stress_work", "legacy", "total"), f"unknown booking {booking}"
assert stage in ("cn", "be", "cap"), f"unknown stage {stage}"
dual = booking != "total"
viscous_theta = 1.0 if stage == "be" else 0.5
capped = stage == "cap"

initial_directory = sys.argv[2]
prefix = initial_directory[: -len("000000")]


def get_fields(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    names = FIELDS + (["implicit_mhd_ion_internal_energy"] if dual else [])
    return {name: data["boxlib", name].value.ravel() for name in names}


snapshots = [get_fields(f"{prefix}{step:06d}") for step in range(n_steps + 1)]
z_centers = (np.arange(nz) + 0.5) * dz


def amplitude(fields, component, basis):
    velocity = fields[f"implicit_mhd_momentum_{component}"] / fields[
        "implicit_mhd_mass_density"
    ]
    return 2.0 * np.dot(velocity, basis) / nz


def kinetic_energy(fields):
    rho = fields["implicit_mhd_mass_density"]
    return 0.5 * (
        fields["implicit_mhd_momentum_x"] ** 2
        + fields["implicit_mhd_momentum_y"] ** 2
        + fields["implicit_mhd_momentum_z"] ** 2
    ) / rho


# 1. The flow stays exactly incompressible and at rest along z: the two
# shears are seeded so that their dissipation is uniform, so nothing can
# drive an acoustic response. What remains is Newton-tolerance noise
# (relative_tolerance 1e-11 on the normalized solver vector, i.e. up to
# ~1e-11 rho0 c_s in m_z = 1e-10 rho0 shear), nine orders below the
# signal; a genuine acoustic response would be O(dU/U) = 5e-3.
# With the free-streaming cap binding the two components are capped
# SEPARATELY (the harmonic cap acts per component of the stress), so the
# capped dissipation is no longer the same on every face: a weak, O(1e-5)
# acoustic response develops. The at-rest guards are loosened to that
# level in the cap mode; the ledger identity below stays at 1e-8 (the PdV
# exchange it introduces is second order in the acoustic amplitude).
density_drift = max(
    np.max(np.abs(f["implicit_mhd_mass_density"] / rho0 - 1.0)) for f in snapshots
)
axial_momentum = max(
    np.max(np.abs(f["implicit_mhd_momentum_z"])) for f in snapshots
) / (rho0 * shear)
density_tolerance = 1.0e-4 if capped else 1.0e-11
momentum_tolerance = 1.0e-3 if capped else 1.0e-9
assert density_drift < density_tolerance, f"density moved: {density_drift:.3e}"
assert axial_momentum < momentum_tolerance, (
    f"axial momentum appeared: max |m_z| = {axial_momentum:.3e} rho0 shear"
)
print(
    f"[all] density uniform to {density_drift:.1e}, max |m_z| = "
    f"{axial_momentum:.1e} rho0 shear over the run "
    + ("(the per-component cap's weak acoustic response)" if capped
       else "(solver-tolerance noise)")
)

# 2. Both transverse amplitudes decay by the analytic per-step factor of
# the staged viscous operator: the stress is formed from
# u^{n+theta_v} = (theta_v/theta) u^{n+theta} + (1 - theta_v/theta) u^n,
# so on the eigenmode m^{n+1} - m^n = -x [w m^{n+1} + (1 - w) m^n] with
# w = theta_v and x = nu k_eff^2 dt.
expected_factor = (1.0 - (1.0 - viscous_theta) * x_step) / (1.0 + viscous_theta * x_step)
cos_basis = np.cos(k_mode * z_centers)
sin_basis = np.sin(k_mode * z_centers)
amplitudes_x = np.array([amplitude(f, "x", cos_basis) for f in snapshots])
amplitudes_y = np.array([amplitude(f, "y", sin_basis) for f in snapshots])
assert np.isclose(amplitudes_x[0], shear, rtol=1.0e-10, atol=0.0)
assert np.isclose(amplitudes_y[0], shear, rtol=1.0e-10, atol=0.0)
for name, amplitudes in (("x", amplitudes_x), ("y", amplitudes_y)):
    factors = amplitudes[1:] / amplitudes[:-1]
    worst = np.max(np.abs(factors / expected_factor - 1.0))
    if capped:
        # The cap divides the stress by 1 + |Pi|/(f p_i): the decay is no
        # longer the linear operator's, and it must be measurably slower
        # (the cap binding is what makes this twin discriminating): every
        # step strictly slower, and the run's total decay slower by more
        # than 1 % (measured 3.4 % at f = 0.3: the per-step factor moves
        # by only x times the rate reduction, 2.7e-4 here).
        assert np.all(factors > expected_factor), (
            f"the viscous cap is not binding on v_{name}: per-step factor "
            f"{factors.min():.9f} vs uncapped {expected_factor:.9f}"
        )
        cumulative = amplitudes[-1] / amplitudes[0]
        assert cumulative > expected_factor**n_steps * 1.01, (
            f"the viscous cap barely binds on v_{name}: total decay "
            f"{cumulative:.6f} vs uncapped {expected_factor**n_steps:.6f}"
        )
        continue
    assert worst < 1.0e-7, (
        f"v_{name} per-step decay factor off the analytic {expected_factor:.9f} "
        f"by {worst:.3e} (viscous_theta = {viscous_theta})"
    )
if capped:
    print(
        f"[{mode}] cap binding: total decay {amplitudes_x[-1] / amplitudes_x[0]:.6f} "
        f"vs uncapped {expected_factor**n_steps:.6f}; mean per-step factor "
        f"{(amplitudes_x[1:] / amplitudes_x[:-1]).mean():.9f} vs "
        f"{expected_factor:.9f} (x = nu k_eff^2 dt = {x_step:.5f})"
    )
else:
    print(
        f"[{mode}] per-step decay factor {expected_factor:.9f} "
        f"(viscous_theta = {viscous_theta}, x = nu k_eff^2 dt = {x_step:.5f}) "
        f"reproduced to {worst:.2e} on both components"
    )

# 3. Energy ledger per plotfile.
ke_total = np.array([np.sum(kinetic_energy(f)) for f in snapshots])
ei_total = np.array([np.sum(f["implicit_mhd_ion_energy"]) for f in snapshots])
ke_loss = ke_total[0] - ke_total
ei_drift = np.max(np.abs(ei_total - ei_total[0])) / ke_loss[-1]
assert ei_drift < 1.0e-9, (
    "the conservative E_i register moved: max |sum(E_i) - sum(E_i)_0| is "
    f"{ei_drift:.3e} of the kinetic energy dissipated over the run"
)
print(
    f"[{mode}] sum(E_i) constant to {ei_drift:.2e} of the dissipated KE "
    f"(KE loss over the run: {ke_loss[-1] / ke_total[0]:.4f} of KE_0, "
    f"{ke_loss[-1] / (ei_total[0] - ke_total[0]):.3e} of the internal energy)"
)

if dual:
    ui = np.array([f["implicit_mhd_ion_internal_energy"] for f in snapshots])
    ui_gain = np.sum(ui, axis=1) - np.sum(ui[0])
    if booking == "legacy" and stage == "be":
        # The old pointwise source heats with |grad u^{n+1/2}|^2 while the
        # staged stress removes grad u^{n+1} . grad u^{n+1/2}; on the
        # eigenmode the ratio is exactly (1 + a)/(2 a) = 1 + x/2 with
        # a = 1/(1 + x) the BE decay factor -- the same every step.
        expected_ratio = 1.0 + 0.5 * x_step
        label = "legacy booking = (1 + nu k_eff^2 dt/2) x the KE removed"
    else:
        expected_ratio = 1.0
        label = "internal gain = kinetic loss"
    ratio = ui_gain[1:] / ke_loss[1:]
    worst = np.max(np.abs(ratio / expected_ratio - 1.0))
    assert worst < 1.0e-8, (
        f"sum(dU_i)/(-sum(dKE)) = {ratio[-1]:.12f} vs expected "
        f"{expected_ratio:.12f} (worst per-plotfile deviation {worst:.3e}): "
        f"{label} FAILS"
    )
    print(
        f"[{mode}] {label}: ratio {ratio[-1]:.12f} vs {expected_ratio:.12f}, "
        f"worst deviation {worst:.2e} over {n_steps} plotfiles"
    )
    # The deposit is uniform (the seeded dissipation is the same on every
    # face and half of each face goes to each neighbour) and non-negative.
    # Under the per-component cap the dissipation is no longer uniform
    # along z (see above), so only positivity is asserted there.
    increments = np.diff(ui, axis=0)
    assert np.all(increments > 0.0), "a cell's U_i decreased in a viscous decay"
    nonuniformity = np.max(np.abs(increments / increments.mean(axis=1, keepdims=True) - 1.0))
    if not capped:
        assert nonuniformity < 1.0e-8, (
            f"the viscous deposit is not uniform across cells: {nonuniformity:.3e}"
        )
    print(
        f"[{mode}] U_i deposit positive in every cell every step, "
        + (f"non-uniform by {nonuniformity:.2e} (per-component cap)" if capped
           else f"uniform to {nonuniformity:.2e}")
    )
else:
    # total_energy: the internal part is the recovery E_i - KE, so with
    # sum(E_i) constant its gain IS the KE loss; report it for the record.
    internal_gain = (ei_total - ke_total) - (ei_total[0] - ke_total[0])
    print(
        f"[{mode}] implied internal gain / KE loss = "
        f"{internal_gain[-1] / ke_loss[-1]:.12f}"
    )

print("PASS")
