#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""The 1D external current: the same discriminant as the RZ test, without walls or axis.

A uniform static plasma column (u = 0, uniform pressure, no plasma current) in
the uniform B_x = B0 plus the split external vector potential A_y(z) =
A0 sin(k z): B_x,ext = -A0 k cos(k z) and the in-domain external current
j_y,ext = A0 k^2 sin(k z) / mu0. Periodic in z, so every cell is interior.
Nothing physical acts on the fluid.

mode "total" (implicit_mhd.lorentz_force_current = total, the default): the
conservative rows integrate -div of the TOTAL Maxwell stress and the fluid
moves along z under the external current's own force f_z = -j_y,ext B_x,total:
m_z(t) must follow f_z t (rms of the difference within 10 %: the second-order
stencil misses 1-4 % per cell), with the ion entropy proxy constant.
mode "plasma" (implicit_mhd.lorentz_force_current = plasma): the external
force is subtracted cell by cell, so the fluid stays at rest to truncation:
rms(m_z) below 1 % of the "total" twin's (measured 1e-3), the entropy proxies
constant. The final B_x must carry the external field in both modes (the 1D
split external vector potential produces its field and its current; a missing
field reads 1.0 of the external amplitude, the moving "total" fluid compresses
the field by up to 5 % of it). The work pairing itself is gated by the RZ tests;
here the sign, the centring and an ignored knob are the attacks (x10-x1000).

Usage: analysis_mhd_external_current_1d.py <total|plasma> <initial_plotfile>
       <final_plotfile> [<twin_final_plotfile>]
"""

import re
import sys
from pathlib import Path

import numpy as np
import yt

yt.set_log_level(50)

mode, initial_plotfile, final_plotfile = sys.argv[1], sys.argv[2], sys.argv[3]
twin_plotfile = sys.argv[4] if len(sys.argv) > 4 else None
assert mode in ("total", "plasma"), mode

used_inputs = Path("warpx_used_inputs").read_text()


def input_value(name, default=None):
    match = re.search(rf"^{re.escape(name)}\s*=\s*([^#\n]+)", used_inputs, re.MULTILINE)
    if match is None:
        assert default is not None, name
        return default
    return match.group(1).strip().strip('"')


mu0 = 4.0e-7 * np.pi
B0 = float(input_value("my_constants.B0"))
A0 = float(eval(input_value("my_constants.A0"), {"__builtins__": {}, "pi": np.pi}))  # the deck writes 0.01 / (2*pi)
Lz = float(input_value("my_constants.Lz"))
k = 2.0 * np.pi / Lz
dt = float(input_value("my_constants.dt"))
max_step = int(input_value("max_step"))
knob = input_value("implicit_mhd.lorentz_force_current", "total")
assert knob == mode, (knob, mode)
t_final = max_step * dt

NAMES = ("implicit_mhd_mass_density", "implicit_mhd_momentum_x", "implicit_mhd_momentum_y",
         "implicit_mhd_momentum_z", "implicit_mhd_ion_energy", "implicit_mhd_electron_energy", "Bx")


def load(plotfile):
    ds = yt.load(plotfile)
    dims = ds.domain_dimensions
    data = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)
    nz = int(dims[0])
    z_lo = float(ds.domain_left_edge[0].value)
    dz = (float(ds.domain_right_edge[0].value) - z_lo) / nz
    z = z_lo + (np.arange(nz) + 0.5) * dz
    fields = {name: np.array(data["boxlib", name].value).ravel() for name in NAMES}
    return float(ds.current_time), z, fields


t0, z, f0 = load(initial_plotfile)
t1, _, f1 = load(final_plotfile)
assert abs(t1 - t_final) <= 1.0e-3 * t_final, (t1, t_final)


def internal(f):
    kinetic = 0.5 * (f["implicit_mhd_momentum_x"] ** 2 + f["implicit_mhd_momentum_y"] ** 2 +
                     f["implicit_mhd_momentum_z"] ** 2) / f["implicit_mhd_mass_density"]
    return f["implicit_mhd_ion_energy"] - kinetic


rho_init = f0["implicit_mhd_mass_density"]
mz = f1["implicit_mhd_momentum_z"]
bx_ext = -A0 * k * np.cos(k * z)
jy_ext = A0 * k ** 2 * np.sin(k * z) / mu0
f_z = -jy_ext * (B0 + bx_ext)
expected = f_z * t_final
rms = lambda a: float(np.sqrt(np.mean(a ** 2)))  # noqa: E731
rms_expected, rms_mz = rms(expected), rms(mz)
print("profile: i z f_z*t m_z m_z/(f_z*t) Bx_final Bx_expected")
for i in range(len(z)):
    denom = expected[i]
    print(f"  {i:2d} {z[i]:.4f} {denom:12.4e} {mz[i]:12.4e} {mz[i] / denom if denom != 0 else float('nan'):8.3f} "
          f"{f1['Bx'][i]:.5f} {B0 + bx_ext[i]:.5f}")
print(f"mode {mode}: B0 {B0} A0 k {A0 * k:.4e} T, peak j_y,ext {np.max(np.abs(jy_ext)):.3e} A/m^2, t_final {t_final:.3e} s; "
      f"rms(m_z) {rms_mz:.4e}, rms(f_ext t) {rms_expected:.4e} kg/m^2/s")
# the external field is present in the plotted total field (cell-centred B_x against the analytic external field)
bx_error = float(np.max(np.abs(f1["Bx"] - (B0 + bx_ext))) / (A0 * k))
print(f"external field: max |B_x - (B0 + B_x,ext)| / (A0 k) = {bx_error:.3e}")

gamma = 5.0 / 3.0


def proxies(f):
    rho = f["implicit_mhd_mass_density"]
    return internal(f) / rho ** gamma, f["implicit_mhd_electron_energy"] / rho ** gamma


si0, se0 = proxies(f0)
si1, se1 = proxies(f1)
internal_change = float(np.max(np.abs((si1 - si0) / si0)))
electron_change = float(np.max(np.abs((se1 - se0) / se0)))
print(f"max relative change of the entropy proxies: ion {internal_change:.3e}, electron {electron_change:.3e}")

# a missing external field reads exactly 1.0; the moving "total" fluid compresses the field by up to 5 % of A0 k
assert bx_error < (0.2 if mode == "total" else 0.02), ("the 1D split external vector potential produced no field", bx_error)
if mode == "total":
    error = rms(mz - expected) / rms_expected
    print(f"total: rms(m_z - f_ext t)/rms(f_ext t) = {error:.3e}")
    assert error < 0.1, error
    assert internal_change < 1.0e-4, internal_change
else:
    ratio = rms_mz / rms_expected
    print(f"plasma: rms(m_z)/rms(f_ext t) = {ratio:.3e}")
    assert ratio < 1.0e-2, ratio
    assert internal_change < 1.0e-5, internal_change
    if twin_plotfile is not None:
        _, _, ft = load(twin_plotfile)
        rms_twin = rms(ft["implicit_mhd_momentum_z"])
        print(f"plasma vs total twin: rms(m_z) {rms_mz:.3e} vs {rms_twin:.3e}")
        assert rms_twin > 0.5 * rms_expected, ("the total twin did not move: the external field is not acting", rms_twin)
        assert rms_mz < 1.0e-2 * rms_twin, (rms_mz, rms_twin)
assert electron_change < 1.0e-5, electron_change
print(f"external current force, 1D ({mode}): PASS")
