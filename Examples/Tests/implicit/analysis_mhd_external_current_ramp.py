#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""The ramped external field: the column screens it, and j_ext must be refreshed every residual.

Same uniform static ideal column and the same in-domain external current as
analysis_mhd_external_current.py, but the split external vector potential
ramps linearly over the whole run, A_time(t) = t / t_ramp with t_ramp =
max_step dt. The split scheme advances the total field by -curl E_ohm and
E_ohm = 0 for the ideal static column, so the column SCREENS the ramp: B_z
stays B0 inside (to 1e-4 while nothing moves) and the plasma carries
j_plasma = -j_ext(t).

mode "total" (implicit_mhd.lorentz_force_current = total, the default): the
rows integrate -div of the TOTAL Maxwell stress, (j_plasma + j_ext) x B = 0 --
the conductor feels no force at all and the fluid must not move: rms(m_r)
over the interior cells below 1 % of the screened integral (measured 1.5e-3).

mode "plasma" (implicit_mhd.lorentz_force_current = plasma): the fluid feels
the screening current's real force j_plasma x B0 = -j_ext(t) x B0 and after
the ramp carries m_r = int_0^T 8 c(t) r B0/mu0 dt = 4 c_ext r B0 t_ramp/mu0:
rms(m_r - that) over the interior cells below 1 % of it (measured 0.3 %; the
moving fluid carries its frozen-in field, so the interior B_z drops by 2e-3).
Because j_ext(t) changes every step this is also the gate on the per-residual
refresh of the external current: a j_ext frozen at the first residual (1/32
of the final one) leaves the plasma run at 0.83 of the screened integral,
the sign flipped reads -1.

The interior here is cells 1-9 of 16: the PEC r_max wall's influence reaches
six cells in under the ramp (the face stress at the wall carries the boundary
image of B) and the axis cell is excluded as in the static test. The entropy
proxies are gated as in the static test.

Usage: analysis_mhd_external_current_ramp.py <total|plasma> <initial_plotfile>
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
c_ext = float(input_value("my_constants.c_ext"))
dt = float(input_value("my_constants.dt"))
max_step = int(input_value("max_step"))
t_ramp = float(input_value("my_constants.t_ramp"))
knob = input_value("implicit_mhd.lorentz_force_current", "total")
assert knob == mode, (knob, mode)
t_final = max_step * dt
assert abs(t_ramp - t_final) <= 1.0e-9 * t_final, ("the ramp must end with the run", t_ramp, t_final)
assert "t_ramp" in input_value("external_vector_potential.sheet.A_time_external_function(t)"), "not the ramped deck"

NAMES = ("implicit_mhd_mass_density", "implicit_mhd_momentum_r", "implicit_mhd_momentum_t",
         "implicit_mhd_momentum_z", "implicit_mhd_ion_energy", "implicit_mhd_electron_energy", "Bz")


def load(plotfile):
    ds = yt.load(plotfile)
    dims = ds.domain_dimensions
    data = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)
    nr, nz = int(dims[0]), int(dims[1])
    r_lo = float(ds.domain_left_edge[0].value)
    dr = (float(ds.domain_right_edge[0].value) - r_lo) / nr
    r = r_lo + (np.arange(nr) + 0.5) * dr
    fields = {name: np.array(data["boxlib", name].value).reshape(nr, nz) for name in NAMES}
    return float(ds.current_time), r, fields


t0, r, f0 = load(initial_plotfile)
t1, _, f1 = load(final_plotfile)
assert abs(t1 - t_final) <= 1.0e-3 * t_final, (t1, t_final)


def internal(f):
    kinetic = 0.5 * (f["implicit_mhd_momentum_r"] ** 2 + f["implicit_mhd_momentum_t"] ** 2 +
                     f["implicit_mhd_momentum_z"] ** 2) / f["implicit_mhd_mass_density"]
    return f["implicit_mhd_ion_energy"] - kinetic


nr = len(r)
interior = np.zeros(nr, dtype=bool)
interior[1:10] = True
rho_init = f0["implicit_mhd_mass_density"]
mr = f1["implicit_mhd_momentum_r"]
# the screened integral: the plasma carries j_theta,plasma = -j_theta,ext(t) = 8 c(t) r/mu0 with
# c(t) = c_ext t/t_ramp, in the frozen interior field B0: m_r = int_0^T 8 c(t) r B0/mu0 dt = 4 c_ext r B0 T/mu0
expected = (4.0 * c_ext * r * B0 * t_ramp / mu0)[:, None] * np.ones_like(mr)
rms = lambda a: float(np.sqrt(np.mean(a[interior] ** 2)))  # noqa: E731
rms_expected, rms_mr = rms(expected), rms(mr)
profile = mr.mean(axis=1)
bz_profile = f1["Bz"].mean(axis=1)
print("radial profile (z-mean): i r screened_integral m_r m_r/screened Bz_final")
for i in range(nr):
    tag = "" if interior[i] else "  (boundary cell, not gated)"
    denom = expected[i, 0]
    print(f"  {i:2d} {r[i]:.4f} {denom:12.4e} {profile[i]:12.4e} {profile[i] / denom:8.4f} {bz_profile[i]:.6f}{tag}")
print(f"mode {mode}: B0 {B0} c_ext {c_ext} t_ramp {t_ramp:.3e} s; interior rms(m_r) {rms_mr:.4e}, "
      f"rms(screened integral) {rms_expected:.4e} kg/m^2/s")
# screening: the total field inside the column stays B0 (the response cancels the ramping external field)
bz_dev = float(np.max(np.abs(f1["Bz"][interior] - B0)) / B0)
print(f"screening: interior max |B_z,total - B0|/B0 at the end of the ramp = {bz_dev:.3e}")

gamma = 5.0 / 3.0


def proxies(f):
    rho = f["implicit_mhd_mass_density"]
    return internal(f) / rho ** gamma, f["implicit_mhd_electron_energy"] / rho ** gamma


si0, se0 = proxies(f0)
si1, se1 = proxies(f1)
internal_change = float(np.max(np.abs((si1 - si0) / si0)[interior]))
electron_change = float(np.max(np.abs((se1 - se0) / se0)[interior]))
print(f"interior max relative change of the entropy proxies: ion {internal_change:.3e}, electron {electron_change:.3e}")

GATE = 1.0e-2
# screening: under "total" nothing moves and the interior field stays B0 to 1e-4; under "plasma" the fluid
# moves outward and carries its frozen-in field with it (the interior B_z drops by 2e-3)
assert bz_dev < (1.0e-3 if mode == "total" else 1.0e-2), ("the column does not screen the ramp", bz_dev)
if mode == "total":
    ratio = rms_mr / rms_expected
    print(f"total: interior rms(m_r)/rms(screened integral) = {ratio:.3e} (the conductor must feel no force)")
    assert ratio < GATE, ratio
    assert internal_change < 1.0e-4, internal_change
else:
    error = rms(mr - expected) / rms_expected
    print(f"plasma: interior rms(m_r - screened integral)/rms(screened integral) = {error:.3e} "
          f"(a j_ext frozen at the first residual reads 0.17, the sign flipped 2)")
    assert error < GATE, error
    assert internal_change < 1.0e-5, internal_change
    if twin_plotfile is not None:
        _, _, ft = load(twin_plotfile)
        rms_twin = rms(ft["implicit_mhd_momentum_r"])
        print(f"plasma vs total twin (interior): rms(m_r) {rms_mr:.3e} vs {rms_twin:.3e}")
        assert rms_twin < GATE * rms_expected, ("the total twin moved under the screened ramp", rms_twin)
        assert rms_mr > 10.0 * rms_twin, (rms_mr, rms_twin)
assert electron_change < 1.0e-5, electron_change
print(f"ramped external current ({mode}): PASS")
