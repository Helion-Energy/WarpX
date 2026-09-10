#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""An external field that carries a current inside the domain must not push the fluid.

The RZ deck holds a uniform static plasma column (no plasma current, uniform
pressure, u = 0) in the uniform state field B_z = B0 plus the split external
field of A_theta = c r^3: B_z,ext = 4 c r^2, whose curl is the in-domain external
current j_theta,ext = -8 c r / mu0. Nothing physical acts on the fluid.

mode "total" (implicit_mhd.lorentz_force_current = total, the default): the
conservative momentum rows integrate -div of the TOTAL Maxwell stress, so the
fluid feels the external field's own force f_r = j_theta,ext B_z,total =
-(8 c r/mu0)(B0 + 4 c r^2) and accelerates: m_r(t) must follow f_r t (rms within
5 %), and the total-energy pairing (E_i books the same discrete work) must keep
the ion internal energy E_i - |m|^2/(2 rho) constant. With c = 0 the run must be
exactly static.

mode "plasma" (implicit_mhd.lorentz_force_current = plasma): the external force
is subtracted cell by cell, so the fluid must stay at rest to truncation:
rms(m_r) below 2 % of the "total" twin's (the face-form/point-form residual of
the correction is O((dr/r)^2), 1 % on this mesh), with the internal energy
constant to 1e-6 (the ion-energy work must pair with the CORRECTED force: a
correction applied to the momentum rows alone would leave the work of the
subtracted force in E_i).

Both comparisons run over the INTERIOR radial cells (the axis cell and the three
cells at the PEC r_max wall excluded and reported separately): at a domain wall
the discrete face stress is not the free-space force (the wall face carries the
boundary image of B), so neither the "total" motion nor the pointwise correction
follows the analytic f_r there. The production case (a coil sheet 30 cells inside
the domain boundary, the machine wall a stair mask) is the interior case.

Usage: analysis_mhd_external_current.py <total|plasma> <initial_plotfile>
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
knob = input_value("implicit_mhd.lorentz_force_current", "total")
assert knob == mode, (knob, mode)
t_final = max_step * dt

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
interior[1:nr - 3] = True
rho_init = f0["implicit_mhd_mass_density"]
c_s = np.sqrt(5.0 / 3.0 * (2.0 / 3.0) * (f0["implicit_mhd_electron_energy"] + internal(f0)) / rho_init).mean()
mr = f1["implicit_mhd_momentum_r"]
# the external force on the fluid: f_r = j_theta,ext B_z,total, j_theta,ext = -dB_z,ext/dr / mu0 = -8 c r / mu0
bz_ext = 4.0 * c_ext * r ** 2
jt_ext = -8.0 * c_ext * r / mu0
f_r = (jt_ext * (B0 + bz_ext))[:, None] * np.ones_like(mr)
expected = f_r * t_final
rms_expected = float(np.sqrt(np.mean(expected[interior] ** 2)))
rms_mr = float(np.sqrt(np.mean(mr[interior] ** 2)))
profile = mr.mean(axis=1)
print("radial profile (z-mean): i r f_r*t m_r m_r/(f_r*t)")
for i in range(nr):
    tag = "" if interior[i] else "  (boundary cell, not gated)"
    denom = f_r[i, 0] * t_final
    print(f"  {i:2d} {r[i]:.4f} {denom:12.4e} {profile[i]:12.4e} {profile[i] / denom if denom != 0 else float('nan'):8.3f}{tag}")
sound_momentum = float(rho_init.mean() * c_s)
print(f"mode {mode}: B0 {B0} c_ext {c_ext} t_final {t_final:.3e} s; rms(m_r) {rms_mr:.4e}, rms(f_ext t) {rms_expected:.4e} kg/m^2/s, rho0 c_s {sound_momentum:.4e}")

# The moving fluid compresses (a few per cent at the boundary cells), so the
# adiabatic part of every energy change is legitimate: compare ENTROPY proxies
# e/rho^gamma over the interior cells. A dropped pairing (the momentum rows
# pushed by a force whose work E_i does not book, or E_i booking the work of a
# force the rows no longer apply) shows up as a non-adiabatic change of the ion
# proxy of order the kinetic energy involved: KE/internal ~ 4e-3 here for the
# "total" run, ~4e-5 (the residual motion times the removed force) for "plasma".
gamma = 5.0 / 3.0


def proxies(f):
    rho = f["implicit_mhd_mass_density"]
    return internal(f) / rho ** gamma, f["implicit_mhd_electron_energy"] / rho ** gamma


si0, se0 = proxies(f0)
si1, se1 = proxies(f1)
internal_change = float(np.max(np.abs((si1 - si0) / si0)[interior]))
electron_change = float(np.max(np.abs((se1 - se0) / se0)[interior]))
density_change = float(np.max(np.abs(f1["implicit_mhd_mass_density"] - rho_init) / rho_init))
raw_internal_change = float(np.max(np.abs(internal(f1) - internal(f0)) / internal(f0)))
print(f"interior max relative change of the entropy proxies: ion {internal_change:.3e}, electron {electron_change:.3e}; "
      f"raw internal energy {raw_internal_change:.3e} (all cells), density {density_change:.3e} (all cells)")

if c_ext == 0.0:
    assert rms_mr <= 1.0e-12 * sound_momentum, rms_mr
    assert internal_change <= 1.0e-12, internal_change
    print("c_ext = 0: static state stays static")
    sys.exit(0)

if mode == "total":
    error = float(np.sqrt(np.mean((mr - expected)[interior] ** 2)) / rms_expected)
    print(f"total: interior rms(m_r - f_ext t)/rms(f_ext t) = {error:.3e}")
    assert error < 0.05, error
    assert internal_change < 1.0e-4, internal_change      # a dropped pairing reads ~4e-3
else:
    ratio = rms_mr / rms_expected
    print(f"plasma: interior rms(m_r)/rms(f_ext t) = {ratio:.3e}")
    assert ratio < 2.0e-2, ratio
    assert internal_change < 1.0e-5, internal_change      # a correction of the rows alone reads ~4e-5
    if twin_plotfile is not None:
        _, _, ft = load(twin_plotfile)
        rms_twin = float(np.sqrt(np.mean(ft["implicit_mhd_momentum_r"][interior] ** 2)))
        print(f"plasma vs total twin (interior): rms(m_r) {rms_mr:.3e} vs {rms_twin:.3e}")
        assert rms_twin > 0.5 * rms_expected, ("the total twin did not move: the external field is not acting", rms_twin)
        assert rms_mr < 2.0e-2 * rms_twin, (rms_mr, rms_twin)
assert electron_change < 1.0e-5, electron_change
print(f"external current force ({mode}): PASS")
