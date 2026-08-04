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

rho_drift = np.max(np.abs(final["implicit_mhd_mass_density"] -
                          initial["implicit_mhd_mass_density"])) / RHO0
print(f"relative rho drift {rho_drift:.3e}")
for f in FIELDS:
    drift = float(np.max(np.abs(final[f] - initial[f])))
    print(f"  {f:35s} max drift = {drift:.3e}")
    assert drift == 0.0, f"{f} drifted by {drift:.3e}"

print("PASS")
