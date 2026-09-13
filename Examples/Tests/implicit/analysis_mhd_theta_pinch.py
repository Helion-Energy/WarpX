#!/usr/bin/env python3
"""Static theta-pinch hold in RZ (the equilibrium-dissipation gate).

A pressure-balanced Bz(r) column (pe carries pe + Bz^2/2mu0 = const,
u = 0, Br = Btheta = 0) is exactly stationary for the hlld path: the
r-face jumps are pure contact mode (B_n = 0, S_M = 0), and the
UCT-HLLD corner E_theta dissipation is built on the fan's rotational
bounds, which collapse to zero here. Fast-bound (UCT-HLL) corner
coefficients diffuse this profile visibly within 100 steps, which is
exactly the mechanism that destroyed the FRC free-boundary halo, so
every field must be preserved to machine zero.
"""

import sys

import numpy as np
import yt

from warpx_constants import m_p

yt.set_log_level(50)

N0 = 1.0e20
RHO0 = N0 * m_p

FIELDS = [
    "Br",
    "Bt",
    "Bz",
    "implicit_mhd_mass_density",
    "implicit_mhd_ion_energy",
    "implicit_mhd_electron_energy",
]


def load(path):
    ds = yt.load(path)
    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    return {f: np.asarray(grid["boxlib", f]) for f in FIELDS}


initial = load(sys.argv[1])
final = load(sys.argv[2])
# Optional mode: "exact" (default; every field machine-preserved) or
# "diffuse" (calibration: the central flux's fast-speed Rusanov penalty
# acts on the electron-energy profile that carries the pressure balance,
# so it must measurably drift -- proving the exact gate discriminates).
mode = sys.argv[3] if len(sys.argv) > 3 else "exact"
assert mode in ("exact", "diffuse")

rho_drift = np.max(np.abs(final["implicit_mhd_mass_density"] -
                          initial["implicit_mhd_mass_density"])) / RHO0
print(f"mode = {mode}: relative rho drift {rho_drift:.3e}")
electron_energy_drift = np.max(
    np.abs(final["implicit_mhd_electron_energy"] -
           initial["implicit_mhd_electron_energy"])
) / np.max(np.abs(initial["implicit_mhd_electron_energy"]))
print(f"relative electron-energy drift {electron_energy_drift:.3e}")
for f in FIELDS:
    drift = float(np.max(np.abs(final[f] - initial[f])))
    print(f"  {f:35s} max drift = {drift:.3e}")
    if mode == "exact":
        assert drift == 0.0, f"{f} drifted by {drift:.3e}"
if mode == "diffuse":
    assert electron_energy_drift > 1.0e-3, (
        f"fast-speed calibration did not diffuse: {electron_energy_drift:.3e}"
    )

print(f"mode = {mode}: PASS")
