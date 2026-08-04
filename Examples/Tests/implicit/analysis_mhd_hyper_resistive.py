#!/usr/bin/env python3
"""Calibrated hyper-resistive decay under the hlld recast (1D).

A transverse standing perturbation Bx = b cos(kz) with no guide field
and u = 0 is contact-inert for the hlld fan, so the only dynamics is
the solver-assembled E = -eta_H laplacian(J). The composite Yee
operator (downward curl -> Dzz -> upward curl) damps the mode at
exactly

    gamma_d = eta_H * kd^4 / mu0,   kd = (2/dz) sin(k dz/2),

and the theta-implicit step multiplies the amplitude by

    g = (1 - (1-theta) gamma_d dt) / (1 + theta gamma_d dt)

per step. The measured per-run log-decay must match N ln(g) to a
tolerance far below the leading physical contaminant (the O(b^2)
magnetic-pressure fluid response, ~1e-7 relative here).
"""

import sys

import numpy as np
import yt

from warpx_constants import mu_0

yt.set_log_level(50)

# Must match the input deck.
B0 = 1.0e-5
NZ = 64
LZ = 1.6
DZ = LZ / NZ
KZ = 2.0 * np.pi / LZ
KD = 2.0 * np.sin(KZ * DZ / 2.0) / DZ
ETA_H = 0.5
GAMMA_D = ETA_H * KD**4 / mu_0
DT = 0.1 / GAMMA_D
THETA = 0.5
NSTEPS = 40

GROWTH = (1.0 - (1.0 - THETA) * GAMMA_D * DT) / \
         (1.0 + THETA * GAMMA_D * DT)


def load(path):
    ds = yt.load(path)
    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    return {
        f: np.asarray(grid["boxlib", f]).squeeze()
        for f in ("Bx", "By", "Bz", "implicit_mhd_mass_density")
    }


def mode_amplitude(bx):
    z = (np.arange(NZ) + 0.5) * DZ
    cos_part = 2.0 * np.mean(bx * np.cos(KZ * z))
    sin_part = 2.0 * np.mean(bx * np.sin(KZ * z))
    return np.hypot(cos_part, sin_part)


initial = load(sys.argv[1])
final = load(sys.argv[2])

amplitude_initial = mode_amplitude(initial["Bx"])
amplitude_final = mode_amplitude(final["Bx"])

measured_log_decay = np.log(amplitude_final / amplitude_initial)
expected_log_decay = NSTEPS * np.log(GROWTH)
relative_error = abs(measured_log_decay - expected_log_decay) / \
    abs(expected_log_decay)

print(f"initial amplitude        {amplitude_initial:.6e}")
print(f"final amplitude          {amplitude_final:.6e}")
print(f"measured log decay       {measured_log_decay:.10f}")
print(f"expected log decay       {expected_log_decay:.10f}")
print(f"relative error           {relative_error:.3e}")

assert amplitude_initial > 0.99 * B0
assert amplitude_final < amplitude_initial
assert relative_error < 1.0e-5, (
    f"hyper-resistive decay rate off by {relative_error:.3e}"
)

# The transverse-y and guide channels must stay identically inert, and
# the O(b^2) fluid response must stay far below the rate tolerance.
by_max = np.max(np.abs(final["By"]))
bz_max = np.max(np.abs(final["Bz"]))
rho_drift = np.max(np.abs(
    final["implicit_mhd_mass_density"] -
    initial["implicit_mhd_mass_density"]
)) / np.max(initial["implicit_mhd_mass_density"])
print(f"max |By|                 {by_max:.3e}")
print(f"max |Bz|                 {bz_max:.3e}")
print(f"relative density drift   {rho_drift:.3e}")
assert by_max < 1.0e-14 * B0
assert bz_max < 1.0e-14 * B0
assert rho_drift < 1.0e-8

print("PASS")
