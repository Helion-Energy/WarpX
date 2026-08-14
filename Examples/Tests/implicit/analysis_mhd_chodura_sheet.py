#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Chodura anomalous-resistivity smoke on the tangential-discontinuity sheet.

100 steps with the run32-form current-keyed anomalous resistivity
(smooth density guards, smooth exponential saturation) on the hlld
current sheets. Asserts: the run stays finite, Newton converges within
its budget every step (the JFNK sees the J- and rho-keyed eta through
the matrix-free probes, no hard branches), mass is exactly conserved
(periodic telescoping), the sheets genuinely dissipate (peak |J|
decreases, magnetic energy decreases), and the dissipated magnetic
energy shows up as electron Joule heat.
"""

import sys

import numpy as np
import yt


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


# keep the dataset handles alive: the covering grid only weak-refs them
initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])


def fields(data):
    by = data["boxlib", "By"].value.ravel()
    rho = data["boxlib", "implicit_mhd_mass_density"].value.ravel()
    ue = data["boxlib", "implicit_mhd_electron_energy"].value.ravel()
    ei = data["boxlib", "implicit_mhd_ion_energy"].value.ravel()
    # the sheet current is the plasma current curl B/mu0 (the dumped j*
    # fields are the ION fluid current, zero for the static state)
    dz = 1.6 / by.size
    jx = np.diff(by, append=by[0]) / dz
    return by, jx, rho, ue, ei


by0, jx0, rho0, ue0, ei0 = fields(initial)
by1, jx1, rho1, ue1, ei1 = fields(final)

for name, arr in [
    ("By", by1),
    ("jx", jx1),
    ("rho", rho1),
    ("Ue", ue1),
    ("Ei", ei1),
]:
    assert np.all(np.isfinite(arr)), f"non-finite {name} after 100 steps"
assert np.all(rho1 > 0.0), "density lost positivity"
assert np.all(ue1 > 0.0), "electron energy lost positivity"

# periodic flux divergence telescopes exactly
np.testing.assert_allclose(np.sum(rho1), np.sum(rho0), rtol=5.0e-14, atol=0.0)

# the anomalous term must genuinely act on the sheets
peak_current_ratio = np.max(np.abs(jx1)) / np.max(np.abs(jx0))
magnetic_ratio = np.sum(by1**2) / np.sum(by0**2)
joule_heat = np.sum(ue1 - ue0)
print(f"peak |J| ratio  = {peak_current_ratio:.4f}")
print(f"magnetic energy ratio = {magnetic_ratio:.4f}")
print(f"total electron heating = {joule_heat:.4e} (sum over cells)")
assert peak_current_ratio < 0.98, "sheets did not dissipate (eta_an inert?)"
assert magnetic_ratio < 0.999, "magnetic energy did not decay"
assert joule_heat > 0.0, "no Joule heat deposited"

# Newton health across all 100 steps: converged within budget, smoothly
# (newton.txt appends across local reruns; take this run's block)
newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history.shape[0] >= 100, "solver did not complete 100 steps"
newton_history = newton_history[-100:]
assert np.all(newton_history[:, 2] <= 20), "Newton hit its iteration cap"
print(f"max Newton iterations/step = {int(np.max(newton_history[:, 2]))}")

print("PASS")
