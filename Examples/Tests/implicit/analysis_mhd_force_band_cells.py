#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""The band-cells force mask: the last N live cells at the shaped wall feel no magnetic force.

The RZ external-current deck (a uniform static column, the split external
field's in-domain current j_theta,ext = -8 c r/mu0 pushing the fluid with
f_r = j_theta,ext B_z,total under lorentz_force_current = total) with the
flat dielectric wall at r = 0.065 (masked cells i >= 10 of 16, rigid under
wall_thermal_bc = zero_flux) and the column density raised to 1e20 so the
pressure response to a force difference stays at the percent level over
the run.

mode "twin" (lorentz_force_band_cells = 0): the live cells 1..8 move under
f_r t (rms within 5 %); the wall-adjacent live cell 9 is reported but not
gated (its outer face is the stair interface, whose absorb-image stress is
not the free-space force -- the same reason the parent test excludes the
domain-boundary cells; it reaches ~45 % of f_r t here). The audit's
force_withheld is exactly zero.

mode "band" (lorentz_force_band_cells = N, twin plotfile as the 3rd
argument, N as the 4th): the band cells i in [10 - N, 9] stay at rest
(rms(m_r) below 5 % of the twin's rms(m_r) over the same cells), the
twin-minus-band momentum difference vanishes in the interior cells
1..9-N (rms below 3 % of the band's f_r t): the mask removes the magnetic
force exactly where it says and nowhere else. The band run's
force_withheld is positive and its booked lorentz work is below the
twin's.

Usage: analysis_mhd_force_band_cells.py <twin|band> <final_plotfile>
       [<twin_final_plotfile> <N>]
"""

import re
import sys
from pathlib import Path

import numpy as np
import yt

yt.set_log_level(50)

mode, final_plotfile = sys.argv[1], sys.argv[2]
assert mode in ("twin", "band"), mode
used_inputs = Path("warpx_used_inputs").read_text()


def input_value(name, default=None):
    match = re.search(rf"^{re.escape(name)}\s*=\s*([^#\n]+)", used_inputs, re.MULTILINE)
    if match is None:
        assert default is not None, name
        return default
    return match.group(1).strip().strip('"')


mu0 = 4.0e-7 * np.pi
B0 = float(input_value("my_constants.B0"))
c_ext = float(input_value("my_constants.c_ext"))
dt = float(input_value("my_constants.dt"))
max_step = int(input_value("max_step"))
band_cells = int(input_value("implicit_mhd.lorentz_force_band_cells", "0"))
t_final = max_step * dt
NAMES = ("implicit_mhd_mass_density", "implicit_mhd_momentum_r", "implicit_mhd_momentum_z")


def load(plotfile):
    ds = yt.load(plotfile)
    dims = ds.domain_dimensions
    data = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)
    nr, nz = int(dims[0]), int(dims[1])
    r_lo = float(ds.domain_left_edge[0].value)
    dr = (float(ds.domain_right_edge[0].value) - r_lo) / nr
    r = r_lo + (np.arange(nr) + 0.5) * dr
    fields = {name: np.array(data["boxlib", name].value).reshape(nr, nz) for name in NAMES}
    return float(ds.current_time), r, dr, fields


def read_audit(path):
    columns, rows = None, []
    with open(path) as handle:
        for line in handle:
            if line.startswith("# columns:"):
                columns = line[len("# columns:"):].split()
            elif line.startswith("#") or not line.strip():
                continue
            else:
                rows.append([float(x) for x in line.split()])
    assert columns is not None, path
    data = np.array(rows)
    return {name: data[:, i] for i, name in enumerate(columns)}


t1, r, dr, f1 = load(final_plotfile)
assert abs(t1 - t_final) <= 1.0e-3 * t_final, (t1, t_final)
nr = len(r)
# the code's wall classification: the first masked cell is the first with
# (i + 0.5) dr >= r_wall - 1e-3 dr (wall_flat_r065.csv)
r_wall = 0.065
first_masked = int(np.argmax(r >= r_wall - 1.0e-3 * dr))
assert first_masked == 10, first_masked
bz_ext = 4.0 * c_ext * r ** 2
jt_ext = -8.0 * c_ext * r / mu0
f_r = (jt_ext * (B0 + bz_ext))[:, None] * np.ones_like(f1["implicit_mhd_momentum_r"])
expected = f_r * t_final
live = np.zeros(nr, dtype=bool)
live[1:first_masked] = True   # the axis cell excluded, as in the parent test
mr = f1["implicit_mhd_momentum_r"]
audit = read_audit("diags/energy_audit.txt")
lorentz_sum = float(np.sum(audit["lorentz"]))
withheld_sum = float(np.sum(audit["force_withheld"]))
print(f"mode {mode}: band_cells {band_cells}, first masked cell {first_masked}; "
      f"audit sum lorentz {lorentz_sum:.6e} J, sum force_withheld {withheld_sum:.6e} J")
profile = mr.mean(axis=1)
print("radial profile (z-mean): i r f_r*t m_r")
for i in range(nr):
    print(f"  {i:2d} {r[i]:.4f} {expected[i, 0]:12.4e} {profile[i]:12.4e}"
          f"{'' if live[i] else '  (masked / boundary)'}")

if mode == "twin":
    assert band_cells == 0, band_cells
    gated = live.copy()
    gated[first_masked - 1] = False   # the wall-adjacent cell: reported, not gated
    rms_expected = float(np.sqrt(np.mean(expected[gated] ** 2)))
    error = float(np.sqrt(np.mean((mr - expected)[gated] ** 2)) / rms_expected)
    wall_cell = float(np.mean(mr[first_masked - 1]) / expected[first_masked - 1, 0])
    print(f"twin: rms(m_r - f_r t)/rms(f_r t) over cells 1..{first_masked - 2} = {error:.3e}; "
          f"wall-adjacent cell {first_masked - 1}: m_r/(f_r t) = {wall_cell:.3f} (not gated)")
    assert error < 0.05, error
    assert withheld_sum == 0.0, withheld_sum
    assert lorentz_sum > 0.0, lorentz_sum
else:
    twin_plotfile, n_band = sys.argv[3], int(sys.argv[4])
    assert band_cells == n_band, (band_cells, n_band)
    _, _, _, ft = load(twin_plotfile)
    difference = ft["implicit_mhd_momentum_r"] - mr
    band = np.zeros(nr, dtype=bool)
    band[first_masked - n_band:first_masked] = True
    interior = live & ~band
    rms_band_expected = float(np.sqrt(np.mean(expected[band] ** 2)))
    rms_band_twin = float(np.sqrt(np.mean(ft["implicit_mhd_momentum_r"][band] ** 2)))
    band_motion = float(np.sqrt(np.mean(mr[band] ** 2)) / rms_band_twin)
    interior_residual = float(np.sqrt(np.mean(difference[interior] ** 2)) / rms_band_expected)
    print(f"band: cells {np.flatnonzero(band).tolist()}: rms(m_r)/rms(m_r twin) = {band_motion:.3e} "
          f"(twin rms(m_r) {rms_band_twin:.3e} vs f_r t {rms_band_expected:.3e}); interior cells "
          f"{np.flatnonzero(interior).tolist()}: rms(d)/rms(f_r t) = {interior_residual:.3e}")
    assert rms_band_twin > 0.3 * rms_band_expected, ("the twin's band did not move", rms_band_twin)
    assert band_motion < 0.05, band_motion
    assert interior_residual < 0.03, interior_residual
    assert withheld_sum > 0.0, withheld_sum
    twin_audit = read_audit(str(Path(twin_plotfile).parents[1] / "diags/energy_audit.txt"))
    twin_lorentz = float(np.sum(twin_audit["lorentz"]))
    print(f"band: booked lorentz {lorentz_sum:.6e} vs twin {twin_lorentz:.6e} J")
    assert lorentz_sum < twin_lorentz, (lorentz_sum, twin_lorentz)
print(f"band-cells force mask ({mode}): PASS")
