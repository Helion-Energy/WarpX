#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Analytic gate of the reference code's density eater port.

The deck is a STATIC contact (uniform total pressure, uniform tangential
velocity, zero normal velocity, uniform Bz) with two over-dense bumps:
one inside the eater band (the first 4 cell planes at z_lo), one far
outside it. The only evolution is the eater's, so every faithful-contract
clause is checked analytically (the reference code's ntb.f90:773-807 semantics):

  1. one-sided relaxation: the band excess above the target
     rho_t = 0.01 rho0 decays by the factor (1 - rate) = 0.8 per step
     (en_lim = 0.8 MAX(en, en0/100) + 0.2 en0/100; en = MIN(en, en_lim));
     the step-1 band state is EXACT to solver roundoff (the solve
     precedes the eater and preserves the static state);
  2. velocity preservation: the tangential velocity vy = m_y/rho in the
     band is invariant (the reference code evolves velocities; the eater does not
     touch them, so the momentum DENSITY scales with the mass);
  3. ion energy-density preservation: E_i in the band is INVARIANT (the
     the reference shot's closure keeps wio and wik: the removed mass gives up
     nothing, so the per-particle ion temperature RISES -- checked
     against the analytic (1 - rate)^-n factor). NOTE: this is the
     faithful the reference code's semantics -- the eater is NOT a cold-dilution
     refill;
  4. electron temperature preservation: U_e/rho in the band is invariant
     (te is the reference code's electron state under flg_wie = false, so U_e
     scales with the mass);
  5. band restriction: the far out-of-band bump is untouched to solver
     tolerance;
  6. ledger closure: the cumulative removed mass equals the domain mass
     deficit (periodic conservative fluxes remove nothing else) and the
     removed electron energy matches the domain U_e deficit.

With the optional third argument (the reference code's eaten_type 0 -> 1 card flip,
implicit_mhd.density_eater_start_time) the same deck must additionally
show a clean TIME GATE: the band is untouched to solver round-off on
every step before the gate -- no relaxation, no ledger row -- and the
analytic (1 - rate)^n decay then starts from the first gated step, with
every other clause above unchanged. The gate is exact, not smoothed: the
eater is an operator-split projection of the committed end-of-step state,
outside every Newton solve.

Usage: analysis_mhd_density_eater.py diags/diag000000 diags/diag000010
                                     [first eaten step]
"""

import sys

import numpy as np
import yt

# Deck constants (inputs_test_1d_theta_implicit_mhd_density_eater);
# m_p is the WarpX parser constant (CODATA 2022).
m_p = 1.67262192595e-27
q_e = 1.602176634e-19
n0 = 1.0e20
rho0 = n0 * m_p
Te = 0.01
gamma = 5.0 / 3.0
P0 = n0 * 100.0 * q_e
bump = 4.0
nz = 64
Lz = 1.0
dz = Lz / nz
sound_speed = np.sqrt(gamma * P0 / rho0)
vy0 = 0.01 * sound_speed
rate = 0.2
eater_cells = 4
target = 0.01 * rho0
n_steps = 10

band = slice(0, eater_cells)
# Far bump: z in [0.75, 0.875) -> cells 48..55. The restriction gate
# uses its INTERIOR (50..53): the bump's own contact-edge cells carry
# the deck's static-solve noise (measured 7.6e-7 at step 1,
# symmetrically on BOTH edges before anything could propagate from the
# eater band -- the compensating p_i/p_e jump class, not leakage).
far_bump = slice(48, 56)
far_bump_interior = slice(50, 54)


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    fields = {
        name: data["boxlib", name].value.ravel()
        for name in (
            "implicit_mhd_mass_density",
            "implicit_mhd_momentum_y",
            "implicit_mhd_ion_energy",
            "implicit_mhd_electron_energy",
        )
    }
    return fields


initial_dir = sys.argv[1]
# The first step on which the eater is allowed to run (1 = ungated).
first_eaten_step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
n_eaten = n_steps - first_eaten_step + 1
prefix = initial_dir[: -len("000000")]
snapshots = [get_data(f"{prefix}{step:06d}") for step in range(n_steps + 1)]
initial = snapshots[0]
final = snapshots[-1]

rho_initial = initial["implicit_mhd_mass_density"]
assert np.allclose(rho_initial[band], bump * rho0, rtol=1.0e-12, atol=0.0)
assert np.allclose(rho_initial[far_bump], bump * rho0, rtol=1.0e-12, atol=0.0)

# 0. Time gate: every step before the gate leaves the band alone. The
# eater returns before touching any state, so only the deck's own static
# solve noise appears here. Vacuous (an empty range) when ungated.
for step in range(1, first_eaten_step):
    held = snapshots[step]["implicit_mhd_mass_density"][band]
    assert np.allclose(held, bump * rho0, rtol=1.0e-5, atol=0.0), (
        f"the band was eaten at step {step}, before the eater's start "
        f"time: max rel drift {np.max(np.abs(held / (bump * rho0) - 1.0)):.3e}"
    )
if first_eaten_step > 1:
    print(
        f"time gate: band untouched for {first_eaten_step - 1} steps before "
        "the eater's start time"
    )

# 1a. The FIRST eaten step is the pure eater applied to the preserved
# static state: rho = target + (1 - rate)(rho0_band - target), momentum
# and U_e scale by rho_new/rho_old, E_i untouched. The pre-eater solve
# preserves the static contact to first-step Newton/GMRES tolerance noise
# (measured ~2e-7 relative on this deck), so the analytic comparisons
# carry a 1e-5 relative gate (two orders above the noise, five below the
# 0.2/step signal). atol=0 everywhere: the fields span 12 orders of
# magnitude in SI, so any absolute tolerance would be vacuous for some
# block.
rho_step1_expected = target + (1.0 - rate) * (bump * rho0 - target)
scale_step1 = rho_step1_expected / (bump * rho0)
step1 = snapshots[first_eaten_step]
step1_reference = snapshots[first_eaten_step - 1]
assert np.allclose(
    step1["implicit_mhd_mass_density"][band],
    rho_step1_expected,
    rtol=1.0e-5,
    atol=0.0,
), "the first eaten step is not the analytic eater limit"
assert np.allclose(
    step1["implicit_mhd_momentum_y"][band],
    scale_step1 * step1_reference["implicit_mhd_momentum_y"][band],
    rtol=1.0e-5,
    atol=0.0,
), "the first eaten step's band momentum did not scale with the eaten mass"
assert np.allclose(
    step1["implicit_mhd_electron_energy"][band],
    scale_step1 * step1_reference["implicit_mhd_electron_energy"][band],
    rtol=1.0e-5,
    atol=0.0,
), "the first eaten step's band electron energy did not scale with the eaten mass"
# E_i carries the deck's largest static-solve noise (measured 6e-6 at
# the band-edge contact cells, where the compensating p_i/p_e jumps
# meet); 1e-4 is still four orders below the 9x per-particle
# temperature-rise signal the invariance implies.
assert np.allclose(
    step1["implicit_mhd_ion_energy"][band],
    step1_reference["implicit_mhd_ion_energy"][band],
    rtol=1.0e-4,
    atol=0.0,
), "the first eaten step's band ion energy density is not invariant"
print(
    f"step {first_eaten_step}: analytic eater limit exact "
    f"(band rho -> {scale_step1:.6f} x)"
)

# 1b. Sustained analytic exponential relaxation of the band excess. The
# eater's tiny electron-pressure perturbation (p_e/P <= 4 Te/Ti = 4e-4)
# launches weak acoustics, so later steps carry a small contamination;
# 1e-3 relative on the excess is far below the 0.8/step signal.
for step in range(first_eaten_step, n_steps + 1):
    excess = snapshots[step]["implicit_mhd_mass_density"][band] - target
    expected = (bump * rho0 - target) * (1.0 - rate) ** (step - first_eaten_step + 1)
    assert np.allclose(excess, expected, rtol=1.0e-3, atol=0.0), (
        f"band excess at step {step} is not the analytic "
        f"(1 - rate)^n decay: max rel err = "
        f"{np.max(np.abs(excess / expected - 1.0)):.3e}"
    )
ratio = (snapshots[n_steps]["implicit_mhd_mass_density"][band] - target) / (
    snapshots[n_steps - 1]["implicit_mhd_mass_density"][band] - target
)
print(
    f"exponential relaxation: {n_eaten} eaten steps analytic, last-step "
    f"excess ratio {ratio.mean():.6f} (expected {1.0 - rate})"
)

# 2. Velocity preservation in the band across the full run.
vy_final = (
    final["implicit_mhd_momentum_y"][band] / final["implicit_mhd_mass_density"][band]
)
assert np.allclose(vy_final, vy0, rtol=1.0e-5, atol=0.0), (
    "band tangential velocity is not preserved by the eater: "
    f"max rel err = {np.max(np.abs(vy_final / vy0 - 1.0)):.3e}"
)
print(
    f"velocity preservation: band vy = vy0 to {np.max(np.abs(vy_final / vy0 - 1.0)):.2e}"
)

# 3. Ion energy DENSITY invariance (the wio/wik preservation) and the
# implied per-particle temperature rise.
ion_e_drift = (
    final["implicit_mhd_ion_energy"][band] / initial["implicit_mhd_ion_energy"][band]
    - 1.0
)
assert np.max(np.abs(ion_e_drift)) < 1.0e-4, (
    f"band ion energy density drifted: max rel = {np.max(np.abs(ion_e_drift)):.3e}"
)
kinetic_initial = (
    0.5
    * initial["implicit_mhd_momentum_y"][band] ** 2
    / (initial["implicit_mhd_mass_density"][band])
)
kinetic_final = (
    0.5
    * final["implicit_mhd_momentum_y"][band] ** 2
    / (final["implicit_mhd_mass_density"][band])
)
temperature_rise = (
    (final["implicit_mhd_ion_energy"][band] - kinetic_final)
    / final["implicit_mhd_mass_density"][band]
) / (
    (initial["implicit_mhd_ion_energy"][band] - kinetic_initial)
    / initial["implicit_mhd_mass_density"][band]
)
expected_rise = (bump * rho0) / (
    target + (bump * rho0 - target) * (1.0 - rate) ** n_eaten
)
assert np.all(temperature_rise > 0.9 * expected_rise), (
    "the eaten band's per-particle ion temperature did not rise as the "
    f"faithful energy-density-preserving semantics require (measured "
    f"{temperature_rise.min():.3f}x, analytic {expected_rise:.3f}x)"
)
print(
    f"ion closure: E_i invariant to {np.max(np.abs(ion_e_drift)):.2e}, "
    f"T_i rose {temperature_rise.mean():.2f}x"
)

# 4. Electron temperature preservation: U_e/rho invariant in the band.
Te_ratio = (
    final["implicit_mhd_electron_energy"][band]
    / final["implicit_mhd_mass_density"][band]
) / (
    initial["implicit_mhd_electron_energy"][band]
    / initial["implicit_mhd_mass_density"][band]
)
assert np.allclose(Te_ratio, 1.0, rtol=1.0e-4, atol=0.0), (
    f"band electron temperature drifted: max rel = {np.max(np.abs(Te_ratio - 1.0)):.3e}"
)
print(f"electron closure: Te preserved to {np.max(np.abs(Te_ratio - 1.0)):.2e}")

# 5. Band restriction: the far bump's interior (out of acoustic reach,
# away from its own contact-edge solve noise) is untouched to solver
# tolerance.
far_drift = (
    final["implicit_mhd_mass_density"][far_bump_interior]
    / initial["implicit_mhd_mass_density"][far_bump_interior]
    - 1.0
)
assert np.max(np.abs(far_drift)) < 1.0e-6, (
    f"the eater leaked outside its band: far-bump density drift = "
    f"{np.max(np.abs(far_drift)):.3e}"
)
print(f"band restriction: far bump drift {np.max(np.abs(far_drift)):.2e}")

# 6. Ledger closure ("step mass energy" rows of cumulative removals; 1D
# units are per unit cross-section, i.e. kg/m^2 = sum(drho) dz).
ledger = np.atleast_2d(np.loadtxt("diags/eater_ledger.txt"))
assert ledger.shape[0] == n_eaten, (
    f"ledger rows ({ledger.shape[0]}) != eaten steps ({n_eaten}): the "
    "gate must suppress the ledger row too"
)
assert np.all(np.diff(ledger[:, 1]) >= 0.0) and np.all(np.diff(ledger[:, 2]) >= 0.0), (
    "the removal ledger must be cumulative (nondecreasing)"
)
mass_deficit = dz * np.sum(
    initial["implicit_mhd_mass_density"] - final["implicit_mhd_mass_density"]
)
assert np.isclose(ledger[-1, 1], mass_deficit, rtol=1.0e-6, atol=0.0), (
    f"ledger mass {ledger[-1, 1]:.9e} != domain mass deficit "
    f"{mass_deficit:.9e} (periodic conservative fluxes remove nothing)"
)
# U_e also exchanges a tiny amount of pdV work with the weak acoustics,
# so the energy column closes at the acoustic-contamination level.
energy_deficit = dz * np.sum(
    initial["implicit_mhd_electron_energy"] - final["implicit_mhd_electron_energy"]
)
assert np.isclose(ledger[-1, 2], energy_deficit, rtol=1.0e-3, atol=0.0), (
    f"ledger energy {ledger[-1, 2]:.9e} != domain U_e deficit {energy_deficit:.9e}"
)
print(
    f"ledger closure: mass {ledger[-1, 1]:.6e} kg/m^2, energy {ledger[-1, 2]:.6e} J/m^2"
)

print("PASS")
