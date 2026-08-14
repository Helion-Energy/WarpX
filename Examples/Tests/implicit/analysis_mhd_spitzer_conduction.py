#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Live Spitzer-keyed conduction: chi_e = chiC Te^2.5 on the parser Te.

A small electron-temperature ripple at k2 relaxes diffusively while
uniform Joule heating (circular B pair at k1, constant eta) raises the
base Te, so the ripple relaxation must SPEED UP by (Te/Te0)^2.5.
Interval-by-interval asserts mirror the eta = Te^-1.5 decay test:

* each snapshot interval's measured ripple decay rate matches
  chi_e(Te_mid) k2_eff^2 evaluated through the same smooth-floor parser
  formula at the interval-mean base Te;
* the late/early rate ratio brackets the (Te_late/Te_early)^2.5
  prediction (a frozen chi_e(Te0) would keep it at 1);
* the base Te genuinely rises.
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


plotfiles = sorted(
    glob.glob(os.path.join(sys.argv[1], "diag" + "[0-9]" * 6))
)  # exactly diagNNNNNN: skip the diag*.old.* dirs WarpX leaves on reruns
assert len(plotfiles) >= 10, "need the full snapshot series"

n0 = 1.0e18
te0_ev = 2.0
gamma_e = 5.0 / 3.0
number_of_cells = 64
domain_length = 1.0
cell_size = domain_length / number_of_cells
k1 = 2.0 * np.pi / domain_length
k2 = 2.0 * k1
k2_eff = 2.0 * np.sin(0.5 * k2 * cell_size) / cell_size
k1_eff = 2.0 * np.sin(0.5 * k1 * cell_size) / cell_size
kelvin_per_ev = 11604.51812
tf_ev = 0.02
eta0 = 1.0e-6
gamma_b = eta0 * k1_eff**2 / constants.mu_0
chi_c = 2.0 * gamma_b / (k2_eff**2 * te0_ev**2.5)


def chi_of_te_kelvin(te_kelvin):
    """The exact smooth-floor parser formula."""
    te_ev = te_kelvin / kelvin_per_ev
    return chi_c * (te_ev**2 + tf_ev**2) ** 1.25


z = (np.arange(number_of_cells) + 0.5) * cell_size
mode = np.sin(k2 * z)
mode_norm = np.dot(mode, mode)

times = []
ripple = []
te_kelvin = []
for plotfile in plotfiles:
    ds, data = get_data(plotfile)
    ue = data["boxlib", "implicit_mhd_electron_energy"].value.ravel()
    rho = data["boxlib", "implicit_mhd_mass_density"].value.ravel()
    n_e = rho / constants.m_p
    te = (gamma_e - 1.0) * ue / (n_e * constants.k)
    times.append(float(ds.current_time))
    # base Te = mean; ripple = the k2 Fourier amplitude of Te
    te_kelvin.append(float(np.mean(te)))
    ripple.append(float(np.dot(te - np.mean(te), mode) / mode_norm))

times = np.array(times)
ripple = np.abs(np.array(ripple))
te_kelvin = np.array(te_kelvin)

te_rise = te_kelvin[-1] / te_kelvin[0]
print(f"base Te rise: {te_rise:.3f}x ({te_kelvin[0]:.1f} K -> {te_kelvin[-1]:.1f} K)")
assert te_rise > 1.4, "electrons did not heat; sizing broken"
assert np.all(ripple > 0.0), "ripple amplitude lost"

# per-interval relaxation rates vs the live chi_e(Te) prediction
rates = -np.diff(np.log(ripple)) / np.diff(times)
te_mid = 0.5 * (te_kelvin[1:] + te_kelvin[:-1])
rates_predicted = chi_of_te_kelvin(te_mid) * k2_eff**2
worst = np.max(np.abs(rates / rates_predicted - 1.0))
print(f"worst per-interval rate error vs chi_e(Te_mid) k2_eff^2: {worst:.3e}")
np.testing.assert_allclose(rates, rates_predicted, rtol=0.05)

# late/early ratio brackets the Te^2.5 prediction (conduction SPEEDS UP)
ratio_measured = rates[-1] / rates[0]
bracket_lo = chi_of_te_kelvin(te_kelvin[-2]) / chi_of_te_kelvin(te_kelvin[1])
bracket_hi = chi_of_te_kelvin(te_kelvin[-1]) / chi_of_te_kelvin(te_kelvin[0])
print(
    f"rate ratio late/early = {ratio_measured:.4f}, "
    f"Te^2.5 bracket = [{bracket_lo:.4f}, {bracket_hi:.4f}], "
    f"midpoint prediction = {(te_mid[-1] / te_mid[0]) ** 2.5:.4f}"
)
assert bracket_lo * 0.95 < ratio_measured < bracket_hi * 1.05, (
    "relaxation did not speed up by the live Te^2.5 factor"
)
assert ratio_measured > 2.0, "no live-Te speedup detected (frozen chi?)"

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert np.all(newton_history[:, 2] <= 10), "Newton iterations degraded"

print("PASS")
