#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""The change-of-variables pedestal, first-order discriminant: a velocity pulse on a pure background.

inputs_test_1d_theta_implicit_mhd_pedestal_cov_pulse (dual_energy, central, viscosity 1; the hlld
variant drops the viscosity) and ..._pulse_cgl (cgl, hlld, no viscosity): every cell sits EXACTLY
on the background (n_ped = 0.01 n0, T_e = T_i = 2 eV, zero deviation everywhere) and carries a
Gaussian velocity pulse (u0 = 1e3 m/s, KE/e_int = 1.7e-3) on a 256-cell periodic line for 8
theta = 1 steps. Where the deviation is zero the shifted mass and energy fluxes vanish identically
and the pressure work (gamma - 1) D(U) div u is zero, so the flow can neither transport nor
compress the background -- while in the total-variable form (any single channel's shift dropped)
the same flow moves rho, U_e or E_i by div(u) dt ~ 3.5e-4 per step where the pulse sits. Gates:

  1. rho and U_e BITWISE static between step 3 and the last step in EVERY cell (the load-time
     sanitize slack re-lands the ion energies once at steps 1-2; density and electron energy are
     bitwise from step 1, gated from 3 with the ions). Under CGL the pair U_par, U_perp is bitwise
     static too: purely internal energies, zero work on a zero deviation, zero relaxation on an
     isotropic background.
  2. The momentum evolves (the test is not vacuous): m_z changes by more than 1e-6 relative.
  3. dual_energy: the ion internal energies (E_i - KE and U_i) change by less than --ion-tol
     relative to e_ped (true build: the viscous heating on the totals and the O(KE div u) kinetic
     bookkeeping, ~1e-6; a dropped shift gives ~3.5e-4).
  4. rho within 1e-12 of rho_ped and T_e within 1e-5 of T_ped_e (on the background, not merely
     static).
  5. Every step's Newton exit is a convergence exit (exit_status 2, 3, 4 or 5): the deck does not
     require convergence so that a sabotaged build shows its damage in the fields instead of
     aborting; the true build must converge every step.
  6. The pedestal ledger books no raise (8 rows of zeros) and the injected fields are identically
     zero.

Usage: analysis_mhd_pedestal_cov_pulse.py <step3_plotfile> <final_plotfile> <ledger> [--cgl] [--ion-tol X]
"""

import sys

import numpy as np
import warpx_constants as constants
import yt

yt.set_log_level(50)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
first_plotfile, final_plotfile, ledger_file = args[:3]
cgl = "--cgl" in sys.argv
ion_tol = float(sys.argv[sys.argv.index("--ion-tol") + 1]) if "--ion-tol" in sys.argv else 5.0e-6
n0 = 1.0e18
f_ped = 0.01
T_ped_e = 2.0
gamma = 5.0 / 3.0
nz = 256
steps = 8
rho_ped = f_ped * n0 * constants.proton_mass
q_over_m = constants.elementary_charge / constants.proton_mass
e_ped = rho_ped * q_over_m * 2.0 / (gamma - 1.0)


def load(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    names = [name for kind, name in ds.field_list if kind == "boxlib"]
    return {name: np.array(data["boxlib", name].value).ravel() for name in names}


first = load(first_plotfile)
final = load(final_plotfile)

# 1. bitwise static background blocks
static_blocks = ["implicit_mhd_mass_density", "implicit_mhd_electron_energy"]
if cgl:
    static_blocks += ["implicit_mhd_ion_parallel_energy", "implicit_mhd_ion_perp_energy"]
for name in static_blocks:
    a, b = first[name], final[name]
    changed = int(np.sum(a != b))
    worst = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
    print(f"{name}: bitwise static {changed == 0} (cells changed {changed}, max relative change {worst:.3e})")
    assert changed == 0, (name, changed, worst)

# 2. the pulse moved
mz_a, mz_b = first["implicit_mhd_momentum_z"], final["implicit_mhd_momentum_z"]
mz_change = float(np.max(np.abs(mz_a - mz_b)) / np.max(np.abs(mz_a)))
print(f"momentum_z: max |m_z| {np.max(np.abs(mz_a)):.4e} -> {np.max(np.abs(mz_b)):.4e}, max relative change {mz_change:.3e}")
assert mz_change > 1.0e-6, mz_change

# 3. ion internal energies (dual_energy); the pulse is axial, so the kinetic
# energy is m_z^2 / (2 rho) (no field to turn it: B_ref only)
if not cgl:
    def kinetic(fields):
        m = fields["implicit_mhd_momentum_z"]
        return m * m / (2.0 * fields["implicit_mhd_mass_density"])
    e_int_a = first["implicit_mhd_ion_energy"] - kinetic(first)
    e_int_b = final["implicit_mhd_ion_energy"] - kinetic(final)
    print(f"KE/e_int max {np.max(kinetic(first) / e_int_a):.2e}")
    for label, a, b in (("E_i - KE", e_int_a, e_int_b),
                        ("U_i", first["implicit_mhd_ion_internal_energy"], final["implicit_mhd_ion_internal_energy"])):
        worst = float(np.max(np.abs(a - b)) / e_ped)
        print(f"ion internal energy {label}: max change {worst:.3e} of e_ped (tolerance {ion_tol:.1e})")
        assert worst < ion_tol, (label, worst)

# 4. on the background
rho_f = final["implicit_mhd_mass_density"]
np.testing.assert_allclose(rho_f, rho_ped, rtol=1.0e-12)
te = (gamma - 1.0) * final["implicit_mhd_electron_energy"] / (rho_f * q_over_m)
np.testing.assert_allclose(te, T_ped_e, rtol=1.0e-5)
print(f"background: rho = rho_ped to {np.max(np.abs(rho_f / rho_ped - 1.0)):.1e}, T_e = {te.min():.6f}..{te.max():.6f} eV")

# 5. every step converged: the exit code where the diagnostic carries one
# (2 abs, 3 rel, 4 / 5 active-set exits), else the final norms against the
# deck's tolerances
with open("diags/newton.txt") as f:
    header = [line for line in f if line.startswith("#")][-1]
columns = {name: int(index) for index, name in
           (token[1:].split("]") for token in header.split() if token.startswith("["))}
newton = np.loadtxt("diags/newton.txt", comments="#", ndmin=2)
assert newton.shape[0] == steps, newton.shape
iterations = newton[:, columns["iters"]].astype(int).tolist()
if "exit_status" in columns:
    exit_status = newton[:, columns["exit_status"]].astype(int)
    print(f"Newton exits per step: {exit_status.tolist()}; iterations {iterations}")
    assert np.all(np.isin(exit_status, (2, 3, 4, 5))), exit_status
else:
    import re
    used_inputs = open("warpx_used_inputs").read()
    def knob(name):
        return float(re.search(rf"^newton\.{name}\s*=\s*(\S+)", used_inputs, re.MULTILINE).group(1))
    norm_abs = newton[:, columns["norm_abs"]]
    norm_rel = newton[:, columns["norm_rel"]]
    converged = (norm_abs < knob("absolute_tolerance")) | (norm_rel < knob("relative_tolerance"))
    print(f"Newton final relative norms per step: {norm_rel.tolist()}; iterations {iterations}; converged {converged.tolist()}")
    assert np.all(converged), (norm_abs, norm_rel)

# 6. no raise, zero injected fields
ledger = np.loadtxt(ledger_file, ndmin=2)
assert ledger.shape == (steps, 5), ledger.shape
assert np.all(ledger[:, 1] == 0), ledger[:, 1]
assert np.all(ledger[:, 2:] == 0.0), ledger[-1]
for name in ("implicit_mhd_pedestal_injected_mass",
             "implicit_mhd_pedestal_injected_electron_energy",
             "implicit_mhd_pedestal_injected_ion_energy"):
    assert np.all(final[name] == 0.0), name
print("change of variables: the pulse moves nothing but the momentum -- background bitwise static, no injection")
