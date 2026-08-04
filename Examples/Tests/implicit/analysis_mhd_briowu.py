#!/usr/bin/env python3
"""Brio & Wu MHD shock tube for fluid_flux = hlld (robustness + sanity).

The primary role of this test is Newton-lock coverage: with the
consistent fan closure at the default kappa_bn this configuration
deterministically locks the Newton line search at step 18 (the CI-scale
reproducer of the total-energy lock class), so completing 200 steps
proves the hlld_fan_closure = barotropic + hlld_kappa_bn robustness
configuration stays load-bearing.

Solution sanity is asserted through conservation (exact for the
telescoping face fluxes on a periodic mesh) and the calibrated
post-slow-shock plateau of the right-moving problem launched at
z = zr = 1.15 (cells 214-228 at nz = 256). The compound-wave region is
deliberately NOT pinned tightly: the barotropic-fan rendering of the
180-degree rotational structure carries a known local density
overshoot; tighten these asserts when the compound rendering improves.
"""

import sys

import numpy as np
import yt

from warpx_constants import m_p, mu_0, e as q_e

yt.set_log_level(50)

N0 = 1.0e20
TI = 100.0
RHO0 = N0 * m_p
P0 = N0 * TI * q_e
BT0 = np.sqrt(mu_0 * P0)


def load(path):
    ds = yt.load(path)
    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    fields = ["Bx", "By", "implicit_mhd_mass_density"]
    return {f: np.asarray(grid["boxlib", f]).ravel() for f in fields}


initial = load(sys.argv[1])
final = load(sys.argv[2])

rho_initial = initial["implicit_mhd_mass_density"] / RHO0
rho = final["implicit_mhd_mass_density"] / RHO0
by = final["By"] / BT0

# global conservation (periodic mesh, telescoping fluxes)
mass_drift = abs(np.sum(rho) - np.sum(rho_initial)) / np.sum(rho_initial)
print(f"relative mass drift: {mass_drift:.3e}")
assert mass_drift < 1.0e-12

# post-slow-shock plateau of the right-moving problem (calibrated:
# rho = 0.1148 +- 0.0003, By = -0.872 at 200 steps, nz = 256)
plateau = slice(214, 228)
rho_plateau = np.mean(rho[plateau])
by_plateau = np.mean(by[plateau])
print(f"post-slow-shock plateau: rho = {rho_plateau:.4f} "
      f"(std {np.std(rho[plateau]):.4f}), By = {by_plateau:.4f}")
assert 0.105 < rho_plateau < 0.125
assert np.std(rho[plateau]) < 2.0e-3
assert -0.95 < by_plateau < -0.80

# bounds over the measured window: no vacuum states, bounded overshoot
window = slice(150, 250)
print(f"window density range: [{np.min(rho[window]):.4f}, "
      f"{np.max(rho[window]):.4f}]")
assert np.min(rho[window]) > 0.10
assert np.max(rho[window]) < 1.05

# the B_t sign reversal must have propagated into the low-density side
assert by_plateau < -0.5
assert np.max(np.abs(by[window])) < 1.1

print("PASS")
