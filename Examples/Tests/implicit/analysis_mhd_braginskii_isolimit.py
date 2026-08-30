#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Braginskii conduction in the unmagnetized (isotropic) limit, both channels.

With B = 0 everywhere the Braginskii tensor must collapse exactly to the
isotropic flux at chi = chi_par (chi_perp = chi_par at x = (Omega tau)^2 = 0
and the bhat projection carries no field), so small ion and electron
temperature ripples must decay at the SAME analytic chi k^2 rate the
isotropic model is verified against by
test_1d_theta_implicit_mhd_central_conduction -- here with chi the
Braginskii (1965) Z = 1 parallel diffusivities computed from first
principles:

    chi_par_e = 3.16 kB Te tau_e / m_e,
    tau_e = 6 sqrt(2) pi^1.5 eps0^2 sqrt(m_e) (kB Te)^1.5 / (n e^4 lnL),
    chi_par_i = 3.9  kB Ti tau_i / m_i,
    tau_i = 12 pi^1.5 eps0^2 sqrt(m_i) (kB Ti)^1.5 / (n e^4 lnL).

The deck also arms the conduction_chi_max soft cap BETWEEN the two
channel values: the ion chi (~7.88e6 m^2/s) lands on the tanh rolloff
(effective ~6.98e6, a -11.4% shift the 5% gate must SEE applied), while
the electron chi (~6.05e6) sits below the knee (0.9 chi_max) and must
pass through EXACTLY. A cap that leaks onto the electron channel (e.g.
a harmonic cap instead of the soft clip: -47%) or an ignored ion cap
(+12.9%) both fail the gates; so do wrong Braginskii prefactors (a
missing sqrt(2) in tau_e is +41%, swapping 3.16/3.9 is +/-23%, a
gamma-factor confusion is 67%). Conduction is a conservative face flux:
sum(E_i) + sum(U_e) must close to roundoff.

Usage: analysis_mhd_braginskii_isolimit.py <initial_plotfile> <final_plotfile>
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
coulomb_log = 10.0
chi_max = 4.6667e6  # scaled by (gamma - 1) with the convention fix
gamma = 5.0 / 3.0
relative_amplitude = 0.005
number_of_cells = 64
wavenumber = 2.0 * np.pi
rho0 = number_density * constants.proton_mass
electron_pressure0 = number_density * te0_eV * constants.elementary_charge
ion_pressure0 = number_density * ti0_eV * constants.elementary_charge


def braginskii_chi_parallel(temperature_eV, tau_prefactor, coefficient, mass):
    kbt = temperature_eV * constants.elementary_charge
    tau = tau_prefactor * kbt * np.sqrt(kbt) / number_density
    return coefficient * kbt * tau / mass


tau_shared = (
    np.pi**1.5 * constants.epsilon_0**2 / (constants.elementary_charge**4 * coulomb_log)
)
# (gamma - 1): the operator is q = -rho chi grad(e_spec), so realizing
# Braginskii's kappa/(n kB) coefficient requires chi_code = (gamma - 1) chi
# (the 2026-08-29 convention fix; the solver applies the same factor).
chi_electron = (gamma - 1.0) * braginskii_chi_parallel(
    te0_eV,
    6.0 * np.sqrt(2.0) * np.sqrt(constants.electron_mass) * tau_shared,
    3.16,
    constants.electron_mass,
)
chi_ion_raw = (gamma - 1.0) * braginskii_chi_parallel(
    ti0_eV,
    12.0 * np.sqrt(constants.proton_mass) * tau_shared,
    3.9,
    constants.proton_mass,
)
# the C^2 soft cap of implicit_mhd.conduction_chi_max (knee 0.9 chi_max,
# width chi_max/10); the electron chi sits below the knee -> untouched
knee = 0.9 * chi_max
width = 0.1 * chi_max
assert chi_electron < knee
assert chi_ion_raw > knee
chi_ion = knee + width * np.tanh((chi_ion_raw - knee) / width)
print(f"chi_par_e            = {chi_electron:.6e} m^2/s (below the cap knee)")
print(f"chi_par_i raw/capped = {chi_ion_raw:.6e} / {chi_ion:.6e} m^2/s")

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

for label, initial_values, final_values, chi in (
    ("electron", initial_electron_energy, final_electron_energy, chi_electron),
    ("ion", initial_ion_energy, final_ion_energy, chi_ion),
):
    initial_amplitude = mode_amplitude(initial_values)
    final_amplitude = mode_amplitude(final_values)
    assert 0.05 < final_amplitude / initial_amplitude < 0.5, (
        f"{label} decay left the resolvable window: "
        f"{final_amplitude / initial_amplitude:.4f}"
    )
    measured_rate = np.log(initial_amplitude / final_amplitude) / elapsed_time
    analytic_rate = chi * wavenumber**2
    print(f"{label}: measured rate = {measured_rate:.6e} 1/s")
    print(f"{label}: analytic rate = {analytic_rate:.6e} 1/s")
    print(f"{label}: ratio         = {measured_rate / analytic_rate:.6f}")
    assert abs(measured_rate / analytic_rate - 1.0) < 0.05, (
        f"{label} conductive decay rate off by "
        f"{measured_rate / analytic_rate - 1.0:+.3%}"
    )
    if label == "ion":
        # the gate must genuinely SEE the cap: the same measurement must
        # reject the uncapped coefficient
        raw_rate = chi_ion_raw * wavenumber**2
        assert abs(measured_rate / raw_rate - 1.0) > 0.08, (
            "ion rate cannot distinguish capped from raw chi: "
            f"{measured_rate / raw_rate - 1.0:+.3%}"
        )

# conduction is a conservative face flux: the fluid energy ledger closes
initial_total = np.sum(initial_ion_energy) + np.sum(initial_electron_energy)
final_total = np.sum(final_ion_energy) + np.sum(final_electron_energy)
energy_drift = abs(final_total - initial_total) / initial_total
print(f"relative fluid-energy drift = {energy_drift:.3e}")
assert energy_drift < 1.0e-11

# the stiff diffusion (chi k / c_s ~ 400) suppresses the acoustic response:
# density stays uniform far below the temperature-mode scale
assert np.max(np.abs(final_density / rho0 - 1.0)) < 1.0e-4

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history[-1, 2] >= 1
assert newton_history[-1, 4] < 1.0e-9

print("PASS")
