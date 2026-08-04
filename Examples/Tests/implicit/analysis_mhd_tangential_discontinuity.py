#!/usr/bin/env python3
"""Static magnetized tangential-discontinuity preservation.

Modes (third argument):
  hlld    -- every field must be preserved EXACTLY (machine zero): the
             fan is inert at B_n = 0 (chi = 0) with a static balanced
             interface (q(0) = 0), the ideal EMF vanishes with u = 0,
             and the ion pressure is uniform so the LLF E_i channel sees
             no jump. This is the FRC-separatrix invariant through the
             full assembly.
  rusanov -- calibration variant: the SAME state must diffuse under the
             Rusanov flux (alpha ~ c_s at a static interface), proving
             the hlld assert is discriminating.
"""

import sys

import numpy as np
import yt

from warpx_constants import m_p, e as q_e

yt.set_log_level(50)

N0 = 1.0e20
TI = 100.0
RHO0 = N0 * m_p
P0 = N0 * TI * q_e


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
final = load(sys.argv[2])
mode = sys.argv[3]

rho_drift = np.max(np.abs(final["implicit_mhd_mass_density"] -
                          initial["implicit_mhd_mass_density"])) / RHO0
by_drift = np.max(np.abs(final["By"] - initial["By"])) / np.max(
    np.abs(initial["By"]))
ue_drift = np.max(np.abs(final["implicit_mhd_electron_energy"] -
                         initial["implicit_mhd_electron_energy"])) / P0
ei_drift = np.max(np.abs(final["implicit_mhd_ion_energy"] -
                         initial["implicit_mhd_ion_energy"])) / P0

print(f"mode = {mode}: max relative drift rho {rho_drift:.3e}, "
      f"By {by_drift:.3e}, U_e {ue_drift:.3e}, E_i {ei_drift:.3e}")

if mode == "hlld":
    assert rho_drift == 0.0
    assert by_drift == 0.0
    assert ue_drift == 0.0
    assert ei_drift == 0.0
elif mode == "rusanov":
    # the diffusive flux must move the density visibly, proving the
    # exactness assert above is discriminating
    assert rho_drift > 1.0e-3
else:
    raise ValueError(f"unknown mode {mode}")

print(f"mode = {mode}: PASS")
