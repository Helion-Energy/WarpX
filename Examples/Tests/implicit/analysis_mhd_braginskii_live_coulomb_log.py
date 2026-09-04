#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Braginskii conduction with the LIVE Coulomb logarithm, both channels.

The deck of test_1d_theta_implicit_mhd_braginskii_isolimit (unmagnetized
limit, small ion and electron temperature ripples relaxing diffusively at
chi_par k^2), run with implicit_mhd.conduction_coulomb_log = -1: the
constant lnLambda = 10 of the collision times is replaced by the two-branch
fits evaluated at the face state (n in m^-3, T in eV; the 3 ln 10 converts
the fits' cm^-3 density; the same electron form the electron-ion
equilibration rate already uses):

    electron:  lnL = 23 + 3 ln 10 - ln(n)/2 + 3 ln(Te)/2   (Te <= e^2 eV)
               lnL = 24 + 3 ln 10 - ln(n)/2 + ln(Te)       (Te >  e^2 eV)
    ion-ion:   lnL = 23 + 3 ln 10 - ln(2)/2 - ln(n)/2 + 3 ln(Ti)/2

each floored at 1. At the deck's uniform state (n = 1e18 m^-3, Te = 10 eV
on the upper electron branch, Ti = 40 eV) these are lnL_e = 12.487 and
lnL_i = 14.371, so relative to the constant-lnLambda twin every collision
time, and with it every chi, is scaled by 10/lnL: 0.801 (electron) and
0.696 (ion). The 0.5 percent temperature ripple modulates lnL by less than
0.1 percent, so each mode still decays at a single analytic rate. Neither
live chi reaches the knee of the conduction_chi_max soft cap (the ion
channel, capped in the twin, passes through uncapped here), so the
measured rates are pure lnLambda-ratio shifts of the raw Braginskii
coefficients: -20 percent (electron) and -30 percent (ion), both far
outside the 5 percent gate. The test also asserts that the measurement
REJECTS the constant-lnLambda prediction, so a knob that silently kept
lnLambda = 10 fails. Conduction stays a conservative face flux:
sum(E_i) + sum(U_e) must close to roundoff.

Usage: analysis_mhd_braginskii_live_coulomb_log.py <initial_plotfile> <final_plotfile>
"""

import sys

import numpy as np
import warpx_constants as constants
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

# constants from inputs_test_1d_theta_implicit_mhd_braginskii_isolimit
number_density = 1.0e18
te0_eV = 10.0
ti0_eV = 40.0
constant_coulomb_log = 10.0  # the twin's constant, rejected below
chi_max = 4.6667e6  # scaled by (gamma - 1) with the convention fix
gamma = 5.0 / 3.0
relative_amplitude = 0.005
number_of_cells = 64
wavenumber = 2.0 * np.pi
rho0 = number_density * constants.proton_mass
electron_pressure0 = number_density * te0_eV * constants.elementary_charge
ion_pressure0 = number_density * ti0_eV * constants.elementary_charge

# the two-branch Coulomb logarithms as the solver evaluates them
three_ln_ten = 3.0 * np.log(10.0)
branch_te_eV = np.exp(2.0)


def coulomb_log_electron(n_m3, te_eV):
    if te_eV <= branch_te_eV:
        value = 23.0 + three_ln_ten - 0.5 * np.log(n_m3) + 1.5 * np.log(te_eV)
    else:
        value = 24.0 + three_ln_ten - 0.5 * np.log(n_m3) + np.log(te_eV)
    return max(value, 1.0)


def coulomb_log_ion(n_m3, ti_eV):
    value = (
        23.0
        + three_ln_ten
        - 0.5 * np.log(2.0)
        - 0.5 * np.log(n_m3)
        + 1.5 * np.log(ti_eV)
    )
    return max(value, 1.0)


coulomb_log_e = coulomb_log_electron(number_density, te0_eV)
coulomb_log_i = coulomb_log_ion(number_density, ti0_eV)
# the deck sits on the UPPER electron branch (Te = 10 eV > e^2 = 7.39 eV),
# and the values must agree with the cm^-3 forms of the fits
assert te0_eV > branch_te_eV
n_cm3 = number_density * 1.0e-6
np.testing.assert_allclose(
    coulomb_log_e, 24.0 - np.log(np.sqrt(n_cm3) / te0_eV), rtol=1.0e-12
)
np.testing.assert_allclose(
    coulomb_log_i, 23.0 - np.log(np.sqrt(2.0 * n_cm3) * ti0_eV**-1.5), rtol=1.0e-12
)
print(f"lnLambda_e(n0, Te0) = {coulomb_log_e:.6f}")
print(f"lnLambda_i(n0, Ti0) = {coulomb_log_i:.6f}")


def braginskii_chi_parallel(
    temperature_eV, coulomb_log, tau_prefactor, coefficient, mass
):
    kbt = temperature_eV * constants.elementary_charge
    tau_shared = (
        np.pi**1.5
        * constants.epsilon_0**2
        / (constants.elementary_charge**4 * coulomb_log)
    )
    tau = tau_prefactor * tau_shared * kbt * np.sqrt(kbt) / number_density
    # (gamma - 1): the operator is q = -rho chi grad(e_spec), so realizing
    # Braginskii's kappa/(n kB) coefficient requires chi_code = (gamma - 1) chi
    return (gamma - 1.0) * coefficient * kbt * tau / mass


electron_prefactor = 6.0 * np.sqrt(2.0) * np.sqrt(constants.electron_mass)
ion_prefactor = 12.0 * np.sqrt(constants.proton_mass)


def chi_electron(coulomb_log):
    return braginskii_chi_parallel(
        te0_eV, coulomb_log, electron_prefactor, 3.16, constants.electron_mass
    )


def chi_ion(coulomb_log):
    return braginskii_chi_parallel(
        ti0_eV, coulomb_log, ion_prefactor, 3.9, constants.proton_mass
    )


# the C^2 soft cap of implicit_mhd.conduction_chi_max (knee 0.9 chi_max,
# width chi_max/10): BOTH live values must sit below the knee so the cap
# is inert and the measurement isolates the Coulomb logarithm
knee = 0.9 * chi_max
width = 0.1 * chi_max
chi_electron_live = chi_electron(coulomb_log_e)
chi_ion_live = chi_ion(coulomb_log_i)
assert chi_electron_live < knee
assert chi_ion_live < knee
# the constant-lnLambda twin's coefficients, which this run must reject:
# raw values, and the capped ion value the twin actually realizes
chi_electron_constant = chi_electron(constant_coulomb_log)
chi_ion_constant_raw = chi_ion(constant_coulomb_log)
assert chi_ion_constant_raw > knee
chi_ion_constant_capped = knee + width * np.tanh((chi_ion_constant_raw - knee) / width)
print(f"chi_par_e live/constant = {chi_electron_live:.6e} / {chi_electron_constant:.6e} m^2/s")
print(
    f"chi_par_i live/constant(raw, capped) = {chi_ion_live:.6e} / "
    f"{chi_ion_constant_raw:.6e}, {chi_ion_constant_capped:.6e} m^2/s"
)

elapsed_time = float(final_ds.current_time - initial_ds.current_time)
assert elapsed_time > 0.0

z = (np.arange(number_of_cells) + 0.5) / number_of_cells
mode = np.sin(wavenumber * z)


def mode_amplitude(values):
    return 2.0 * np.mean(values * mode)


initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
initial_ion_energy = initial["boxlib", "implicit_mhd_ion_energy"].value.ravel()
final_ion_energy = final["boxlib", "implicit_mhd_ion_energy"].value.ravel()
initial_electron_energy = initial[
    "boxlib", "implicit_mhd_electron_energy"
].value.ravel()
final_electron_energy = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()

# the initial condition must carry exactly the intended modes on a uniform
# static background (E = p/(gamma - 1) at u = 0)
np.testing.assert_allclose(
    mode_amplitude(initial_ion_energy),
    relative_amplitude * ion_pressure0 / (gamma - 1.0),
    rtol=1.0e-9,
)
np.testing.assert_allclose(
    mode_amplitude(initial_electron_energy),
    relative_amplitude * electron_pressure0 / (gamma - 1.0),
    rtol=1.0e-9,
)
np.testing.assert_allclose(initial_density, rho0, rtol=1.0e-12, atol=0.0)

for label, initial_values, final_values, chi_live, chi_constant_raw, chi_constant, ratio in (
    (
        "electron",
        initial_electron_energy,
        final_electron_energy,
        chi_electron_live,
        chi_electron_constant,
        chi_electron_constant,
        constant_coulomb_log / coulomb_log_e,
    ),
    (
        "ion",
        initial_ion_energy,
        final_ion_energy,
        chi_ion_live,
        chi_ion_constant_raw,
        chi_ion_constant_capped,
        constant_coulomb_log / coulomb_log_i,
    ),
):
    initial_amplitude = mode_amplitude(initial_values)
    final_amplitude = mode_amplitude(final_values)
    assert 0.05 < final_amplitude / initial_amplitude < 0.5, (
        f"{label} decay left the resolvable window: "
        f"{final_amplitude / initial_amplitude:.4f}"
    )
    measured_rate = np.log(initial_amplitude / final_amplitude) / elapsed_time
    analytic_rate = chi_live * wavenumber**2
    print(f"{label}: measured rate            = {measured_rate:.6e} 1/s")
    print(f"{label}: analytic live-lnL rate   = {analytic_rate:.6e} 1/s")
    print(f"{label}: ratio                    = {measured_rate / analytic_rate:.6f}")
    assert abs(measured_rate / analytic_rate - 1.0) < 0.05, (
        f"{label} conductive decay rate off by "
        f"{measured_rate / analytic_rate - 1.0:+.3%}"
    )
    # the shift relative to the RAW constant-lnLambda coefficient is the
    # Coulomb-logarithm ratio itself (no cap on either side)
    measured_ratio = measured_rate / (chi_constant_raw * wavenumber**2)
    print(f"{label}: measured/constant-lnL    = {measured_ratio:.6f} (predicted {ratio:.6f})")
    assert abs(measured_ratio / ratio - 1.0) < 0.05, (
        f"{label} live/constant rate ratio {measured_ratio:.4f} is not the "
        f"Coulomb-logarithm ratio {ratio:.4f}"
    )
    # and the gate must genuinely SEE the live value: the same measurement
    # rejects the rate the constant-lnLambda twin realizes
    constant_rate = chi_constant * wavenumber**2
    assert abs(measured_rate / constant_rate - 1.0) > 0.08, (
        f"{label} rate cannot distinguish live from constant lnLambda: "
        f"{measured_rate / constant_rate - 1.0:+.3%}"
    )

# conduction is a conservative face flux: the fluid energy ledger closes
initial_total = np.sum(initial_ion_energy) + np.sum(initial_electron_energy)
final_total = np.sum(final_ion_energy) + np.sum(final_electron_energy)
energy_drift = abs(final_total - initial_total) / initial_total
print(f"relative fluid-energy drift = {energy_drift:.3e}")
assert energy_drift < 1.0e-11

# the stiff diffusion suppresses the acoustic response: density stays
# uniform far below the temperature-mode scale
assert np.max(np.abs(final_density / rho0 - 1.0)) < 1.0e-4

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history[-1, 2] >= 1
assert newton_history[-1, 4] < 1.0e-9

print("PASS")
