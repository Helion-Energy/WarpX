#!/usr/bin/env python3
"""Circularly polarized Alfven wave (CPAW) checks for fluid_flux = hlld.

The CPAW is an exact nonlinear solution of ideal MHD at any amplitude, so
it pins the conservative-form recast end to end: the Riemann induction
EMF, the Yee curl, and the Maxwell-stress momentum coupling. Asserted, at
the calibrated 128-cell resolution (values measured from the validated
implementation; adversarial margins both ways):

1. Propagation: after one full crossing the complex tangential field
   B_x + i B_y must return to the initial pattern (correlation modulus
   > 0.998, phase < 0.05 rad), and at half crossing it must be phase
   shifted by pi (a wrong-sign EMF fails immediately).
2. Dissipation scale: the amplitude retention after one crossing must
   sit in the first-order upwind band [0.80, 0.92] (measured 0.858, the
   analytic rotational-layer estimate exp(-(2 pi)^2 / (2 nz)) = 0.857):
   a regression to Rusanov-class diffusion or an unstable anti-diffusive
   scheme both fail.
3. No magnetosonic contamination: |B_t| is constant for the exact
   solution, so the density must stay uniform to round-off
   (< 1e-10 relative) -- a stress/pressure imbalance fails this.
4. B_n invariance: dBz/dt = 0 identically in 1D; Bz must not drift.
5. Total-energy budget: the damped wave energy is dissipated by the
   upwind fan and (by design of the component hybrid) NOT thermalized;
   the total (E_i + U_e + B^2/2mu0) drift must stay at the calibrated
   truncation scale (|drift| < 3e-3), not grow.
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
B0 = np.sqrt(mu_0 * P0)


def load(path):
    ds = yt.load(path)
    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    fields = [
        "Bx",
        "By",
        "Bz",
        "implicit_mhd_mass_density",
        "implicit_mhd_ion_energy",
        "implicit_mhd_electron_energy",
    ]
    return {f: np.asarray(grid["boxlib", f]).ravel() for f in fields}


initial = load(sys.argv[1])
half = load(sys.argv[2])
final = load(sys.argv[3])

bt_initial = initial["Bx"] + 1j * initial["By"]
bt_half = half["Bx"] + 1j * half["By"]
bt_final = final["Bx"] + 1j * final["By"]

# 1. propagation (phase) checks
correlation_final = np.vdot(bt_initial, bt_final) / (
    np.linalg.norm(bt_initial) * np.linalg.norm(bt_final)
)
correlation_half = np.vdot(bt_initial, bt_half) / (
    np.linalg.norm(bt_initial) * np.linalg.norm(bt_half)
)
phase_final = np.angle(correlation_final)
phase_half = np.angle(correlation_half)
print(f"full-crossing correlation: |c| = {abs(correlation_final):.6f}, "
      f"phase = {phase_final:.4f} rad")
print(f"half-crossing correlation: |c| = {abs(correlation_half):.6f}, "
      f"phase = {phase_half:.4f} rad (expect +-pi)")
assert abs(correlation_final) > 0.998
assert abs(phase_final) < 0.05
assert abs(correlation_half) > 0.998
assert abs(abs(phase_half) - np.pi) < 0.05

# 2. dissipation scale
retention = np.sqrt(np.mean(np.abs(bt_final) ** 2) /
                    np.mean(np.abs(bt_initial) ** 2))
print(f"amplitude retention after one crossing: {retention:.4f}")
assert 0.80 < retention < 0.92

# 3. no magnetosonic contamination
rho = final["implicit_mhd_mass_density"]
density_perturbation = np.max(np.abs(rho / RHO0 - 1.0))
print(f"max relative density perturbation: {density_perturbation:.3e}")
assert density_perturbation < 1.0e-10

# 4. B_n invariance
bz_drift = np.max(np.abs(final["Bz"] - initial["Bz"])) / B0
print(f"relative Bz drift: {bz_drift:.3e}")
assert bz_drift < 1.0e-12

# 5. total-energy budget
def total_energy(data):
    magnetic = (data["Bx"] ** 2 + data["By"] ** 2 + data["Bz"] ** 2) / (
        2.0 * mu_0
    )
    return np.sum(
        data["implicit_mhd_ion_energy"]
        + data["implicit_mhd_electron_energy"]
        + magnetic
    )


drift = (total_energy(final) - total_energy(initial)) / total_energy(initial)
print(f"relative total-energy drift: {drift:.3e}")
assert abs(drift) < 3.0e-3

print("PASS")
