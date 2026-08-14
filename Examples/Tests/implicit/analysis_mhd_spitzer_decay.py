#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Live Spitzer-keyed resistive decay: eta = etaC Te^-1.5 on the parser Te.

A circularly-polarized transverse B mode decays resistively on a warm
uniform state with Joule heating on, so the electron temperature RISES
as the mode decays and the decay must SLOW by exactly (Te/Te0)^-1.5.
Window-by-window asserts:

* the measured decay rate of each snapshot interval matches
  eta(Te_window) k_eff^2 / mu0 evaluated through the same smooth-floor
  parser formula at the window-mean Te;
* the late/early rate ratio brackets the (Te_late/Te_early)^-1.5
  prediction (the live-coupling signature: a frozen eta(Te0) fails this
  by the full Te contrast);
* Te genuinely rises (the reservoir sizing makes it roughly double).
"""

import glob
import os
import sys

import numpy as np
import warpx_constants as constants
import yt


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


plotfiles = sorted(glob.glob(os.path.join(sys.argv[1], "diag[0-9]*")))
assert len(plotfiles) >= 10, "need the full snapshot series"

n0 = 1.0e18
te0_ev = 2.0
gamma_e = 5.0 / 3.0
number_of_cells = 64
domain_length = 1.0
cell_size = domain_length / number_of_cells
wavenumber = 2.0 * np.pi / domain_length
k_eff = 2.0 * np.sin(0.5 * wavenumber * cell_size) / cell_size
kelvin_per_ev = 11604.51812
tf_ev = 0.02
eta_c = 1.0e-6 * te0_ev**1.5


def eta_of_te_kelvin(te_kelvin):
    """The exact smooth-floor parser formula."""
    te_ev = te_kelvin / kelvin_per_ev
    return eta_c * (te_ev**2 + tf_ev**2) ** (-0.75)


times = []
amplitudes = []
te_kelvin = []
for plotfile in plotfiles:
    ds, data = get_data(plotfile)
    bx = data["boxlib", "Bx"].value.ravel()
    by = data["boxlib", "By"].value.ravel()
    ue = data["boxlib", "implicit_mhd_electron_energy"].value.ravel()
    rho = data["boxlib", "implicit_mhd_mass_density"].value.ravel()
    times.append(float(ds.current_time))
    amplitudes.append(np.sqrt(np.mean(bx**2 + by**2)))
    # temperature-primary Te (K): p_e/(n kB) from the dumped moments
    n_e = rho / constants.m_p
    te = (gamma_e - 1.0) * ue / (n_e * constants.k)
    # uniform heating: Te must stay uniform to high accuracy
    np.testing.assert_allclose(te, np.mean(te), rtol=1.0e-6)
    te_kelvin.append(float(np.mean(te)))

times = np.array(times)
amplitudes = np.array(amplitudes)
te_kelvin = np.array(te_kelvin)

# the reservoir sizing must genuinely heat the electrons (discriminating)
te_rise = te_kelvin[-1] / te_kelvin[0]
print(
    f"Te rise over run: {te_rise:.3f}x ({te_kelvin[0]:.1f} K -> {te_kelvin[-1]:.1f} K)"
)
assert te_rise > 1.4, "electrons did not heat; sizing broken"

# per-interval decay rates vs the live eta(Te) prediction
rates = -np.diff(np.log(amplitudes)) / np.diff(times)
te_mid = 0.5 * (te_kelvin[1:] + te_kelvin[:-1])
rates_predicted = eta_of_te_kelvin(te_mid) * k_eff**2 / constants.mu_0
worst = np.max(np.abs(rates / rates_predicted - 1.0))
print(f"worst per-interval rate error vs eta(Te_mid) k_eff^2/mu0: {worst:.3e}")
# measured worst 1.7e-2 (the first interval, where Te changes fastest and
# the interval-mean Te lags the theta-stage value)
np.testing.assert_allclose(rates, rates_predicted, rtol=0.03)

# late/early ratio brackets the Te^-1.5 prediction. Windows: the first
# and last snapshot intervals; the bracket uses the window Te extremes
# (the mid-point prediction sits strictly inside it).
ratio_measured = rates[-1] / rates[0]
te_e_lo, te_e_hi = te_kelvin[0], te_kelvin[1]
te_l_lo, te_l_hi = te_kelvin[-2], te_kelvin[-1]
bracket_lo = eta_of_te_kelvin(te_l_hi) / eta_of_te_kelvin(te_e_lo)
bracket_hi = eta_of_te_kelvin(te_l_lo) / eta_of_te_kelvin(te_e_hi)
print(
    f"rate ratio late/early = {ratio_measured:.4f}, "
    f"Te^-1.5 bracket = [{bracket_lo:.4f}, {bracket_hi:.4f}], "
    f"midpoint prediction = {(te_mid[-1] / te_mid[0]) ** (-1.5):.4f}"
)
assert bracket_lo * 0.98 < ratio_measured < bracket_hi * 1.02, (
    "decay did not slow by the live Te^-1.5 factor"
)
# a frozen eta(Te0) would keep the ratio at 1: require the full contrast
assert ratio_measured < 0.75, "no live-Te slowing detected (frozen eta?)"

# Newton health: the JFNK picked up the state-dependent eta smoothly
newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert np.all(newton_history[:, 2] <= 8), "Newton iterations degraded"

print("PASS")
