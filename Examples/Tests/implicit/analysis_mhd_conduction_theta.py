#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Conduction-stage L-stability (implicit_mhd.conduction_theta).

The deck seeds a grid-Nyquist electron-temperature ripple on a
pressure-uniform state with a hot Gaussian Te spike over a low-density
halo band, and runs 10 steps at the stiff grid diffusion number
theta dt chi / dx^2 = 1e3 (z_Nyquist = 4 dt chi/dx^2 = 8e3) with the ion
fluid frozen, so the electron energy sees the conduction operator in
isolation.

crank_nicolson variant (conduction_theta = theta = 0.5, the default
path): the trapezoidal rule is A-stable but NOT L-stable -- the Nyquist
mode's amplification is

    g = (1 - z/2)/(1 + z/2) -> -1  as  z -> infinity,

|g| = 0.99950 at z = 8e3, i.e. the checkerboard SURVIVES with
|amplification| ~ 1 and flips sign every step (the production halo
temperature checkerboard that preceded the Newton freeze-guard aborts).
After 10 steps the seeded ripple must retain essentially its full
amplitude (and its sign: (-1)^10 = +1) while the well-resolved modes
partially decay, so the Nyquist-band share of the fluctuation power
GROWS.

backward_euler variant (conduction_theta = 1): backward-Euler
conduction damps the Nyquist mode as g = 1/(1 + z) ~ 1.2e-4 in a
single step; every fluctuation mode is stiff here (z_1 ~ 19), so the
profile must land on the analytic fully-diffused limit -- uniform
specific internal energy e_inf = sum(U_e)/sum(rho) (the equilibrium of
the conservative flux -chi rho_f d(e_spec)/dz at frozen rho) -- while
conduction, a pure conservative face flux, closes sum(U_e) to
round-off.

Usage: analysis_mhd_conduction_theta.py <initial_plotfile>
       <final_plotfile> {backward_euler|crank_nicolson}
"""

import sys

import numpy as np
import yt

yt.set_log_level(50)


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])
variant = sys.argv[3]
assert variant in ("backward_euler", "crank_nicolson")

number_of_cells = 64
number_of_steps = 10

rho_initial = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
rho_final = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
energy_initial = initial["boxlib", "implicit_mhd_electron_energy"].value.ravel()
energy_final = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()
assert rho_initial.size == number_of_cells

# the ion fluid is frozen (evolve_ion_fluid = false)
assert np.array_equal(rho_initial, rho_final)

# specific internal energy (the diffused quantity, up to gamma - 1)
e_spec_initial = energy_initial / rho_initial
e_spec_final = energy_final / rho_final

# --- Nyquist amplitude: projection onto (-1)^i ---
alternating = np.where(np.arange(number_of_cells) % 2 == 0, 1.0, -1.0)


def nyquist_amplitude(values):
    return np.mean(values * alternating)


nyquist_initial = nyquist_amplitude(e_spec_initial)
nyquist_final = nyquist_amplitude(e_spec_final)
# signed: Crank-Nicolson flips the sign every step, (-1)^10 = +1
nyquist_ratio = nyquist_final / nyquist_initial
print(f"Nyquist projection initial   = {nyquist_initial:.6e}")
print(f"Nyquist projection final     = {nyquist_final:.6e}")
print(f"Nyquist amplitude ratio      = {nyquist_ratio:.6e}")

# --- Nyquist-band fluctuation-power fraction (top quarter of the
# spectrum, the production checkerboard diagnostic) ---


def nyquist_band_fraction(values):
    spectrum = np.abs(np.fft.rfft(values)) ** 2
    fluctuation_power = np.sum(spectrum[1:])
    band_power = np.sum(spectrum[24:])
    return band_power / fluctuation_power, band_power


fraction_initial, band_power_initial = nyquist_band_fraction(e_spec_initial)
fraction_final, band_power_final = nyquist_band_fraction(e_spec_final)
band_power_ratio = band_power_final / band_power_initial
print(f"Nyquist-band power fraction  = {fraction_initial:.4f} (initial) "
      f"-> {fraction_final:.4f} (final)")
print(f"Nyquist-band power ratio     = {band_power_ratio:.6e}")

# --- conservation: conduction is a conservative face flux and the only
# active term (frozen fluid, u = 0, J = 0), so sum(U_e) closes to the
# Newton-tolerance round-off (measured 6e-12 BE / 1.5e-11 CN at
# newton.absolute_tolerance = 1e-11) ---
energy_drift = abs(np.sum(energy_final) - np.sum(energy_initial)) / np.sum(
    energy_initial
)
print(f"sum(U_e) relative drift      = {energy_drift:.3e}")
assert energy_drift < 1.0e-9

if variant == "backward_euler":
    # L-stability: the seeded checkerboard must be OBLITERATED --
    # theory: a SINGLE-step factor 1/(1 + 8e3) = 1.2e-4; measured after
    # 10 steps: exactly 0 (the leftover round-off profile is
    # even-symmetric about the spike and cancels the odd projection).
    # 1e-5 fails any centering with |g| ~ 1.
    assert abs(nyquist_ratio) < 1.0e-5
    # measured 5e-30
    assert band_power_ratio < 1.0e-10
    # analytic fully-diffused asymptote: uniform e_spec at the
    # rho-weighted mean (the slowest fluctuation mode retains
    # (1/(1 + 19.3))^10 ~ 1e-13 of its initial contrast); measured
    # landing 5.7e-12, gated with ~500x margin.
    e_uniform = np.sum(energy_initial) / np.sum(rho_initial)
    profile_deviation = np.max(np.abs(e_spec_final - e_uniform)) / e_uniform
    print(f"asymptotic profile deviation = {profile_deviation:.3e}")
    assert profile_deviation < 1.0e-9
else:
    # Crank-Nicolson survival: |g|^10 = 0.99950^10 = 0.99502; measured
    # 0.99505, positive sign after the even number of steps -- the
    # checkerboard rides through with |amplification| ~ 1 per step.
    per_step = abs(nyquist_ratio) ** (1.0 / number_of_steps)
    print(f"measured |amplification|/step = {per_step:.6f}")
    assert 0.985 < nyquist_ratio < 1.001
    # the band keeps its power (measured ratio 0.990) while the
    # resolved modes decay, so the Nyquist share of the fluctuation
    # power GROWS (measured 0.032 -> 0.062, factor 1.93): the emerging-
    # checkerboard signature of the production halo.
    assert band_power_ratio > 0.9
    assert fraction_final > 1.5 * fraction_initial

print(f"{variant}: all conduction-theta gates passed")
