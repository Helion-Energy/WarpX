#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""The halo lift of the Braginskii parallel ceiling and its hard start-time gate.

Entropy-mode column with the field along z (pure parallel conduction along
z), density-keyed as a halo (factor 100), both channels cap-bound, so each
channel's e_spec ripple decays at k_eff^2 (gamma - 1) chi_ceiling -- a
state-independent rate: (gamma - 1) 100 = 66.7 m^2/s at the base ceiling
(implicit_mhd.conduction_chi_par_max), (gamma - 1) 1e4 = 6667 m^2/s at the
halo ceiling (implicit_mhd.conduction_chi_par_max_halo). Ten steps of 3e-8 s,
plotfiles at steps 0, 5, 10.

"nolift": no halo key -- both channels decay at the BASE rate over 0-10
          (measured chi below 10 % of the lifted value; the lift never
          applies without the key).
"lift":   halo ceiling 1e4 from step one -- both channels at the LIFTED rate
          over 0-5 and 5-10 (within 25 %; ~46 % decay per window).
"gated":  halo ceiling 1e4 with conduction_chi_par_max_halo_start_time =
          5.2 dt -- the step-5 plotfile is BYTE-IDENTICAL to the nolift arm's
          (the gate is exact: below the start time the lift does not exist),
          the 0-5 window decays at the base rate and the 5-10 window at the
          lifted rate (within 25 % of the lift arm's own 5-10 rate).

Every arm closes sum(U_e) and sum(U_i) separately (conservative face flux,
no equilibration, no Joule).

Usage: analysis_mhd_conduction_halo_lift.py <arm> diags/diag000000 diags/diag000005 diags/diag000010
           [<nolift diag000005> <lift diag000005> <lift diag000010>]
"""

import sys

import numpy as np
import yt

yt.set_log_level(50)

arm = sys.argv[1]
assert arm in ("nolift", "lift", "gated"), arm
plotfiles = sys.argv[2:5]

# constants from the base deck and this test's overrides
gamma = 5.0 / 3.0
number_of_cells = 64
domain_length = 1.0
ripple_mode = 4
chi_base_defined = 100.0
chi_halo_defined = 1.0e4
convention = gamma - 1.0   # kappa/(n kB) -> operator convention
chi_base = convention * chi_base_defined
chi_lifted = convention * chi_halo_defined
cell_size = domain_length / number_of_cells
z_centers = (np.arange(number_of_cells) + 0.5) * cell_size
k_ripple = 2.0 * np.pi * ripple_mode / domain_length
k_eff_sq = (2.0 / cell_size * np.sin(0.5 * k_ripple * cell_size)) ** 2
basis = np.column_stack([np.ones(number_of_cells), np.sin(k_ripple * z_centers),
                         np.cos(k_ripple * z_centers)])


def load(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    rho = data["boxlib", "implicit_mhd_mass_density"].value.ravel()
    ue = data["boxlib", "implicit_mhd_electron_energy"].value.ravel()
    ui = data["boxlib", "implicit_mhd_ion_internal_energy"].value.ravel()
    return float(ds.current_time), rho, ue, ui


def all_fields(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    names = [name for kind, name in ds.field_list if kind == "boxlib"]
    return {name: np.array(data["boxlib", name].value) for name in names}


def ripple_amplitude(values):
    coefficients = np.linalg.lstsq(basis, values, rcond=None)[0]
    return np.hypot(coefficients[1], coefficients[2])


def measured_chi(state_a, state_b):
    """Effective chi of each channel over the window a -> b from the e_spec ripple decay."""
    ta, rho_a, ue_a, ui_a = state_a
    tb, rho_b, ue_b, ui_b = state_b
    dt = tb - ta
    assert dt > 0.0, (ta, tb)
    out = {}
    for name, ea, eb in (("electron", ue_a, ue_b), ("ion", ui_a, ui_b)):
        retention = ripple_amplitude(eb / rho_b) / ripple_amplitude(ea / rho_a)
        out[name] = (-np.log(retention) / (k_eff_sq * dt), retention)
    return out


states = [load(p) for p in plotfiles]
t0, t5, t10 = (s[0] for s in states)
print(f"arm {arm}: times {t0:.3e} {t5:.3e} {t10:.3e} s; operator chi base {chi_base:.4e}, "
      f"lifted {chi_lifted:.4e} m^2/s")

# conservation of each channel over the whole run
volume = cell_size
for name, idx in (("U_e", 2), ("U_i", 3)):
    total_0 = states[0][idx].sum() * volume
    total_10 = states[2][idx].sum() * volume
    drift = abs(total_10 - total_0) / abs(total_0)
    print(f"{name} conservation drift = {drift:.3e}")
    assert drift < 1.0e-5, drift

early = measured_chi(states[0], states[1])
late = measured_chi(states[1], states[2])
for window, chis in (("0-5", early), ("5-10", late)):
    for name, (chi, retention) in chis.items():
        print(f"  window {window} {name}: measured chi {chi:.4e} m^2/s (retention {retention:.5f})")


def assert_lifted(chis, label):
    for name, (chi, retention) in chis.items():
        assert 0.75 * chi_lifted < chi < 1.25 * chi_lifted, (label, name, chi, chi_lifted)
        assert 0.3 < retention < 0.8, (label, name, retention)


def assert_base(chis, label):
    for name, (chi, _) in chis.items():
        # the base decay is 0.6 % per window: a loose bound suffices to show the
        # ceiling was NOT lifted (the lifted rate is 100x)
        assert chi < 0.1 * chi_lifted, (label, name, chi, chi_lifted)


if arm == "nolift":
    assert_base(early, "nolift 0-5")
    assert_base(late, "nolift 5-10")
    whole = measured_chi(states[0], states[2])
    for name, (chi, retention) in whole.items():
        print(f"  window 0-10 {name}: measured chi {chi:.4e} (retention {retention:.5f})")
        assert chi > 0.0, (name, chi)
elif arm == "lift":
    assert_lifted(early, "lift 0-5")
    assert_lifted(late, "lift 5-10")
else:
    nolift_step5, lift_step5, lift_step10 = sys.argv[5:8]
    mine = all_fields(plotfiles[1])
    twin = all_fields(nolift_step5)
    common = sorted(set(mine) & set(twin))
    assert common, (set(mine), set(twin))
    for name in common:
        same = np.array_equal(mine[name], twin[name])
        print(f"  step 5 vs the nolift arm, {name}: identical {same}")
        assert same, name
    assert_base(early, "gated 0-5")
    assert_lifted(late, "gated 5-10")
    lift_late = measured_chi(load(lift_step5), load(lift_step10))
    for name, (chi, _) in late.items():
        reference = lift_late[name][0]
        print(f"  gated 5-10 {name}: {chi:.4e} vs the lift arm's 5-10 {reference:.4e}")
        assert abs(chi - reference) < 0.25 * reference, (name, chi, reference)

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history[-1, 2] >= 1
print(f"conduction halo lift ({arm}): PASS")
