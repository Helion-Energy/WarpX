#!/usr/bin/env python3
"""Static RADIAL contact preservation in RZ (the cylindrical-metric gate).

A density annulus with uniform pressures, uniform Bz, and zero velocity
is exactly stationary for the hlld path only if the r_face/r_center
weighted divergence and the -Pi_theta_theta/r geometric source (hydro
AND Maxwell-stress parts) cancel cell-by-cell. Modes:

  hlld    -- every field must be preserved EXACTLY (machine zero).
  rusanov -- calibration variant: the SAME state must diffuse radially,
             proving the exactness assert is discriminating.
  central_flow -- the production central flux with the limited Rusanov
             penalty (central_dissipation = 1) and the entropy channels
             penalised at the smoothed FLOW speed
             (central_dissipation_entropy_speed = flow), which is exactly
             zero at u = 0: the same machine-zero gates as hlld.
  central_fast -- the same penalty at the default fast speed: the annulus
             diffuses radially at the acoustic speed (the rusanov gate).
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
mode = sys.argv[3]
assert mode in ("hlld", "rusanov", "central_flow", "central_fast")

rho_drift = np.max(np.abs(final["implicit_mhd_mass_density"] -
                          initial["implicit_mhd_mass_density"])) / RHO0
field_drift = {
    f: float(np.max(np.abs(final[f] - initial[f]))) for f in FIELDS
}
print(f"mode = {mode}: relative rho drift {rho_drift:.3e}")
for f, drift in field_drift.items():
    print(f"  {f:35s} max drift = {drift:.3e}")

if mode in ("hlld", "central_flow"):
    for f, drift in field_drift.items():
        assert drift == 0.0, f"{f} drifted by {drift:.3e}"
else:
    assert rho_drift > 1.0e-3, (
        f"Rusanov calibration did not diffuse: {rho_drift:.3e}"
    )

print(f"mode = {mode}: PASS")
