#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""The physical-share force weight: a mock current pushes the fluid with its physical share only.

The 1D deck holds a uniform static column carrying the plasma-response current of
B_x(z) = B0 (1 + a cos kz), j_y = B0 a k sin(kz)/mu0, force f_z = j_y B_x, while
the field advance runs on the density-keyed vacuum resistivity (eta_field ~ 1257
eta0), so the current's physical share is w = eta0/eta_field ~ 8e-4 everywhere.

mode "plasma" (implicit_mhd.lorentz_force_current = plasma; the external field is
present but carries no current, so this is the full force): m_z(t) follows
f_z t (rms within 5 %) and the entropy proxies stay constant to 1e-4.

mode "physical": the rows feel w f_z: m_z follows w f_z t (rms within 10 %),
hence ~1e-3 of the twin's (with implicit_mhd.lorentz_force_weight_eta = 100 eta0
the weight is a constant-reference density statement, w = 7.9e-2, and the same
checks apply with that w); both entropy proxies stay constant to 1e-6 (the E_i
work pairs with the WEIGHTED force); the energy audit's force_withheld column is
positive and the summed lorentz/force_withheld ratio equals w/(1 - w) to a few
1e-3 (per cell the ratio is exact; the compression modulates w by O(1e-3)). The
plasma twin's force_withheld must be exactly zero.

Usage: analysis_mhd_physical_force_1d.py <plasma|physical> <final_plotfile>
       [<plasma_twin_final_plotfile>]
"""

import re
import sys
from pathlib import Path

import numpy as np
import yt

yt.set_log_level(50)

mode, final_plotfile = sys.argv[1], sys.argv[2]
twin_plotfile = sys.argv[3] if len(sys.argv) > 3 else None
assert mode in ("plasma", "physical"), mode

used_inputs = Path("warpx_used_inputs").read_text()


def input_value(name, default=None):
    match = re.search(rf"^{re.escape(name)}\s*=\s*([^#\n]+)", used_inputs, re.MULTILINE)
    if match is None:
        assert default is not None, name
        return default
    return match.group(1).strip().strip('"')


mu0 = 4.0e-7 * np.pi
m_p = 1.67262192369e-27
n0 = float(input_value("my_constants.n0"))
B0 = float(input_value("my_constants.B0"))
a_mod = float(input_value("my_constants.a_mod"))
Lz = float(input_value("my_constants.Lz"))
dt = float(input_value("my_constants.dt"))
eta0 = float(input_value("my_constants.eta0"))
D_vac = float(input_value("my_constants.D_vac"))
ref_factor = float(input_value("my_constants.ref_factor"))
max_step = int(input_value("max_step"))
knob = input_value("implicit_mhd.lorentz_force_current")
assert knob == mode, (knob, mode)
t_final = max_step * dt
k_mod = 2.0 * np.pi / Lz
rho0 = n0 * m_p

# the physical share of the current: the weight's reference eta (the live user
# eta = eta0 here, or the constant implicit_mhd.lorentz_force_weight_eta when
# set) over the vacuum-boosted field eta
eta_vac = mu0 * D_vac * ref_factor ** 2
weight_eta = float(input_value("implicit_mhd.lorentz_force_weight_eta", "-1"))
eta_ref = weight_eta if weight_eta >= 0.0 else eta0
w = eta_ref / np.sqrt(eta_ref ** 2 + eta_vac ** 2)
print(f"eta0 {eta0:.3e}, weight reference eta {eta_ref:.3e} (lorentz_force_weight_eta {weight_eta}), "
      f"eta_vac {eta_vac:.3e} ohm m -> physical share w = {w:.4e}")

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
    fields = {name: np.array(data["boxlib", name].value).reshape(nz) for name in NAMES}
    return float(ds.current_time), z, fields


t1, z, f1 = load(final_plotfile)
assert abs(t1 - t_final) <= 1.0e-3 * t_final, (t1, t_final)


def internal(f):
    kinetic = 0.5 * (f["implicit_mhd_momentum_x"] ** 2 + f["implicit_mhd_momentum_y"] ** 2 +
                     f["implicit_mhd_momentum_z"] ** 2) / f["implicit_mhd_mass_density"]
    return f["implicit_mhd_ion_energy"] - kinetic


# the analytic force of the response current on the column
bx = B0 * (1.0 + a_mod * np.cos(k_mod * z))
jy = B0 * a_mod * k_mod * np.sin(k_mod * z) / mu0
f_z = jy * bx
expected_full = f_z * t_final
rms_full = float(np.sqrt(np.mean(expected_full ** 2)))
mz = f1["implicit_mhd_momentum_z"]
rms_mz = float(np.sqrt(np.mean(mz ** 2)))
gamma = 5.0 / 3.0
rho = f1["implicit_mhd_mass_density"]
si1 = internal(f1) / rho ** gamma
se1 = f1["implicit_mhd_electron_energy"] / rho ** gamma
# the initial proxies: uniform P0/(gamma-1) over rho0^gamma for both species
P0 = n0 * float(input_value("my_constants.Te")) * 1.602176634e-19
si0 = (P0 / (gamma - 1.0)) / rho0 ** gamma
se0 = si0
internal_change = float(np.max(np.abs(si1 - si0) / si0))
electron_change = float(np.max(np.abs(se1 - se0) / se0))
print(f"mode {mode}: t_final {t_final:.3e} s; rms(m_z) {rms_mz:.4e}, rms(f_z t) {rms_full:.4e}, "
      f"w rms(f_z t) {w * rms_full:.4e} kg/m^2/s; entropy proxy changes ion {internal_change:.3e} "
      f"electron {electron_change:.3e}")


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


audit = read_audit("diags/energy_audit.txt")
lorentz_sum = float(np.sum(audit["lorentz"]))
withheld_sum = float(np.sum(audit["force_withheld"]))
print(f"audit: sum lorentz {lorentz_sum:.6e} J, sum force_withheld {withheld_sum:.6e} J, "
      f"cum column {audit['force_withheld_cum'][-1]:.6e} J")

if mode == "plasma":
    error = float(np.sqrt(np.mean((mz - expected_full) ** 2)) / rms_full)
    print(f"plasma: rms(m_z - f_z t)/rms(f_z t) = {error:.3e}")
    assert error < 0.05, error
    assert internal_change < 1.0e-4, internal_change
    assert electron_change < 1.0e-4, electron_change
    assert withheld_sum == 0.0, withheld_sum
    assert lorentz_sum > 0.0, lorentz_sum
else:
    expected = w * expected_full
    error = float(np.sqrt(np.mean((mz - expected) ** 2)) / (w * rms_full))
    print(f"physical: rms(m_z - w f_z t)/rms(w f_z t) = {error:.3e}; rms(m_z)/rms(f_z t) = {rms_mz / rms_full:.3e}")
    assert error < 0.10, error
    assert rms_mz < 3.0 * w * rms_full, (rms_mz, w * rms_full)
    assert internal_change < 1.0e-6, internal_change
    assert electron_change < 1.0e-6, electron_change
    assert withheld_sum > 0.0, withheld_sum
    assert lorentz_sum > 0.0, lorentz_sum
    ratio = lorentz_sum / withheld_sum
    print(f"physical: sum lorentz / sum force_withheld = {ratio:.6e}, w/(1-w) = {w / (1.0 - w):.6e}")
    assert abs(ratio - w / (1.0 - w)) < 5.0e-3 * w / (1.0 - w), (ratio, w / (1.0 - w))
    assert abs(audit["force_withheld_cum"][-1] - withheld_sum) <= 1.0e-9 * withheld_sum
    if twin_plotfile is not None:
        _, _, ft = load(twin_plotfile)
        rms_twin = float(np.sqrt(np.mean(ft["implicit_mhd_momentum_z"] ** 2)))
        print(f"physical vs plasma twin: rms(m_z) {rms_mz:.3e} vs {rms_twin:.3e} (ratio {rms_mz / rms_twin:.3e})")
        assert rms_twin > 0.5 * rms_full, ("the plasma twin did not move", rms_twin)
        assert rms_mz < 3.0 * w * rms_twin, (rms_mz, rms_twin)
print(f"physical-share force ({mode}): PASS")
