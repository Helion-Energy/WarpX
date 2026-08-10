#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""External-field r_max ghost regression (RZ, hlld, open r boundary).

A static uniform plasma sits in a curl-free split external field whose
discrete curl is exactly r-independent in B_z and z-independent in B_r,
with B_r threading the open r_max wall. Every field jump the corner
UCT-HLLD EMF can see is zero and u = 0 kills the four-state EMF
average, so the assembled E_theta vanishes -- PROVIDED the r_max
domain ghosts of the external B field carry their analytic values. If
the ghost ring is left at its allocation value (zero) instead, the
wall corners see a spurious external-Bz jump of order B0, and the
(armed, since B_r != 0 at the wall) rotational dissipation coefficient
converts it into a dt-INDEPENDENT E_theta injection of ~5-13 V/m at
the wall ring from the first evaluation on. The scheme's genuine
response (the reflect wall's PEC-image stress against the interior
B_r*B_z stress kicks the wall ring's momentum, whose -(u x B)_theta
then shows in E_theta) scales linearly with dt; at the deck's dt
(1000x below the explicit CFL) it sits ~4 decades below the injection.

Usage: analysis_mhd_external_wall_ghost.py <initial_plotfile> <final_plotfile>
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


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])

# Characteristic scales of the deck.
B0 = 0.01
n0 = 1.0e20
m_p = 1.67262192369e-27
mu0 = 4.0e-7 * np.pi
alfven_speed = B0 / np.sqrt(mu0 * n0 * m_p)
E_char = alfven_speed * B0

# The static drive must produce (almost) no electric field: the
# scheme's genuine wall-stress response scales linearly with the tiny
# dt (measured ~5e-4 V/m here) while an unfilled external-Bz ghost ring
# injects a dt-independent |E_theta| ~ 5-13 V/m at the wall ring
# (E_char ~ 218 V/m).
final_Et = np.squeeze(final["boxlib", "Et"].value)
max_Et = np.abs(final_Et).max()
print(f"max |E_theta|: {max_Et:.3e} V/m (tolerance {1.0e-4 * E_char:.3e})")
assert max_Et < 1.0e-4 * E_char, (
    f"E_theta injection on a static external drive (max |E_theta| = "
    f"{max_Et:.3e} V/m): the external-field r_max ghost ring is not "
    f"carrying its analytic value"
)

# The magnetic field must stay the analytic external field (the initial
# plotfile holds the plasma response only, so compare against the
# analytic values; the cell-centered average of the staggered field is
# exact for this linear-in-r/z configuration).
Lr = 0.1
Cr = 0.2 * B0 / Lr
nr, nz = np.squeeze(final["boxlib", "Br"].value).shape
dr = Lr / nr
dz = 0.4 / nz
r_cc = (np.arange(nr) + 0.5) * dr
z_cc = -0.2 + (np.arange(nz) + 0.5) * dz
analytic = {
    "Br": -Cr * r_cc[:, None] * np.ones((1, nz)),
    "Bt": np.zeros((nr, nz)),
    "Bz": (B0 + 2.0 * Cr * z_cc[None, :]) * np.ones((nr, 1)),
}
for name, expected in analytic.items():
    last = np.squeeze(final["boxlib", name].value)
    deviation = np.abs(last - expected).max() / B0
    print(f"relative deviation of {name} from the external field: {deviation:.3e}")
    assert deviation < 1.0e-6, (
        f"{name} deviates from the static external field (relative "
        f"deviation {deviation:.3e})"
    )

# The fluid state must stay frozen at the dt-scaled level.
for name, scale in [
    ("implicit_mhd_mass_density", n0 * m_p),
    ("implicit_mhd_ion_energy", None),
    ("implicit_mhd_electron_energy", None),
]:
    first = np.squeeze(initial["boxlib", name].value)
    last = np.squeeze(final["boxlib", name].value)
    reference = scale if scale is not None else np.abs(first).max()
    drift = np.abs(last - first).max() / reference
    print(f"relative drift of {name}: {drift:.3e}")
    assert drift < 1.0e-6, (
        f"{name} drifted (relative change {drift:.3e}) on a static "
        f"external drive"
    )

print("External-field wall-ghost regression: all assertions passed.")
