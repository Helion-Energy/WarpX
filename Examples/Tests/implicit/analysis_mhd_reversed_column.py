#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Reversed-field column hold in RZ: the closed-flux-mass gate of the
per-channel Rusanov penalty speed and of the preconditioner's upwind rows.

The deck (inputs_test_rz_theta_implicit_mhd_reversed_column) is an exact
equilibrium at u = 0: Bz(r) reverses through a null at r_s, the electron
pressure carries the total-pressure balance, the density steps down across
the null. The mass inside r_s per radian, M_in = sum_{r < r_s} rho r dr dz,
is this column's closed-flux mass. Modes:

  exact   -- flow-speed penalty (or no penalty) on the static hold: every
             field must be preserved to machine zero (the flow speed is
             exactly zero at u = 0).
  diffuse -- calibration: the SAME hold under the fast-speed penalty must
             lose closed-flux mass measurably (the field-blind Fickian
             diffusion of the density step at |u_n| + c_f).
  pulse   -- perturbed hold (u_pert > 0): completion within the Newton budget
             (the deck requires convergence, so completion is convergence),
             M_in and the Newton / GMRES work printed; with a 5th argument
             (the final plotfile of another arm) the closed-flux mass loss
             of this arm must be at most half that arm's.
  pulse_pc -- pulse plus the preconditioner contract: jacobian.pc_type =
             pc_mhd_block with pc_mhd_block.fluid_upwind set, the run at the
             residual's zero-dissipation default, cumulative GMRES bounded.

Usage: analysis_mhd_reversed_column.py <initial_plotfile> <final_plotfile> <mode> [<reference_final_plotfile>]
"""

import re
import sys
from pathlib import Path

import numpy as np
import yt

yt.set_log_level(50)

FIELDS = [
    "Br",
    "Bt",
    "Bz",
    "implicit_mhd_mass_density",
    "implicit_mhd_ion_energy",
    "implicit_mhd_electron_energy",
    "implicit_mhd_momentum_r",
    "implicit_mhd_momentum_z",
]


def load(path):
    ds = yt.load(path)
    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    fields = {f: np.asarray(grid["boxlib", f]) for f in FIELDS}
    r_lo = float(ds.domain_left_edge[0])
    r_hi = float(ds.domain_right_edge[0])
    z_lo = float(ds.domain_left_edge[1])
    z_hi = float(ds.domain_right_edge[1])
    nr, nz = ds.domain_dimensions[0], ds.domain_dimensions[1]
    dr = (r_hi - r_lo) / nr
    dz = (z_hi - z_lo) / nz
    r_center = r_lo + (np.arange(nr) + 0.5) * dr
    return fields, r_center, dr, dz


def closed_mass(fields, r_center, dr, dz, r_null):
    """Mass per radian inside the field null (cells whose centre lies inside)."""
    rho = fields["implicit_mhd_mass_density"].reshape(fields["Bz"].shape)
    inside = r_center < r_null
    weights = (r_center * dr * dz)[inside]
    return float(np.sum(rho[inside, ...].reshape(inside.sum(), -1) * weights[:, None]))


def field_null_radius(fields, r_center):
    """First radius where the z-averaged initial Bz crosses zero."""
    bz = fields["Bz"].reshape(fields["Bz"].shape)
    bz_profile = bz.reshape(bz.shape[0], -1).mean(axis=1)
    crossing = np.where(np.diff(np.sign(bz_profile)) != 0)[0]
    assert crossing.size >= 1, "no field reversal in the initial Bz"
    i = crossing[0]
    # linear interpolation of the zero crossing between cell centres
    return float(
        r_center[i] - bz_profile[i] * (r_center[i + 1] - r_center[i]) /
        (bz_profile[i + 1] - bz_profile[i])
    )


initial, r_center, dr, dz = load(sys.argv[1])
final, _, _, _ = load(sys.argv[2])
mode = sys.argv[3]
assert mode in ("exact", "diffuse", "pulse", "pulse_pc", "pulse_guard")

r_null = field_null_radius(initial, r_center)
m_in_initial = closed_mass(initial, r_center, dr, dz, r_null)
m_in_final = closed_mass(final, r_center, dr, dz, r_null)
m_in_change = (m_in_final - m_in_initial) / m_in_initial
total_initial = closed_mass(initial, r_center, dr, dz, np.inf)
total_final = closed_mass(final, r_center, dr, dz, np.inf)
print(f"mode = {mode}: field null at r = {r_null:.4f} m")
print(f"closed-flux mass per radian {m_in_initial:.6e} -> {m_in_final:.6e} "
      f"(relative change {m_in_change:+.3e})")
print(f"total mass per radian {total_initial:.6e} -> {total_final:.6e} "
      f"(relative change {(total_final - total_initial) / total_initial:+.3e})")
field_drift = {f: float(np.max(np.abs(final[f] - initial[f]))) for f in FIELDS}
for f, drift in field_drift.items():
    print(f"  {f:35s} max drift = {drift:.3e}")

# total mass: conservative face fluxes, PEC wall and axis, periodic z
np.testing.assert_allclose(total_final, total_initial, rtol=1.0e-11, atol=0.0)

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
newton_per_step = newton_history[:, 2]
total_gmres = newton_history[-1, 7]
print(f"Newton per step: mean {newton_per_step.mean():.2f}, max {newton_per_step.max():.0f}; "
      f"cumulative GMRES {total_gmres:.0f}")

if mode == "exact":
    for f, drift in field_drift.items():
        assert drift == 0.0, f"{f} drifted by {drift:.3e}"
elif mode == "diffuse":
    assert m_in_change < -1.0e-4, (
        f"fast-speed calibration did not leak closed-flux mass: {m_in_change:+.3e}"
    )
else:
    # measured maxima: 11 (fast), 12 (flow), 13 (zero dissipation with the
    # preconditioner's upwind rows) against the deck's 20; the bound catches
    # a stall, not a fluctuation
    assert newton_per_step.max() <= 16, (
        f"Newton budget exceeded: max {newton_per_step.max():.0f} per step"
    )
    if mode == "pulse_pc":
        used_inputs = Path("warpx_used_inputs").read_text()

        def input_value(name):
            match = re.search(rf"^{re.escape(name)}\s*=\s*([^#\s]+)", used_inputs,
                              re.MULTILINE)
            assert match is not None, name
            return match.group(1).strip('"')

        assert input_value("jacobian.pc_type") == "pc_mhd_block"
        assert input_value("pc_mhd_block.fluid_upwind") in ("rusanov", "upwind")
        assert "implicit_mhd.central_dissipation" not in used_inputs or (
            float(input_value("implicit_mhd.central_dissipation")) == 0.0
        )
        # the measured pc-on envelope of the zero-dissipation pulse: 3.8e4
        # cumulative GMRES over the 100 steps (the fast-speed pc-off twin
        # 3.0e4, the flow-speed one 3.9e4); a singular or non-stationary
        # preconditioner would blow far past it
        assert 0 < total_gmres <= 5.0e4, (
            f"cumulative GMRES {total_gmres:.0f} exceeds the pc-on budget"
        )
    if mode == "pulse_guard":
        # The local penalty guard on the reduced-penalty pulse: the density
        # switch (n_g = 3e19, the low-density outer region fully guarded,
        # the step at the null in its taper) restores the full penalty on
        # the guarded faces, so the run must complete in the Newton budget
        # like its unguarded twin and its ledger must book the transport
        # the guard added -- mass across the density step at the null and
        # ion energy through the pulse -- on every step.
        rows = [
            line.split()
            for line in open("diags/guard_ledger.txt")
            if line.strip() and not line.startswith("#")
        ]
        table = np.array(rows, dtype=float)
        faces, guard_mass, guard_ei = table[:, 1], table[:, 2], table[:, 4]
        print(f"guard ledger: {len(rows)} rows; guarded faces "
              f"{faces.min():.0f}..{faces.max():.0f}; guard mass per step "
              f"{guard_mass.min():.3e}..{guard_mass.max():.3e} kg (cumulative "
              f"{table[-1, 10]:.3e}); guard E_i per step {guard_ei.min():.3e}.."
              f"{guard_ei.max():.3e} J (cumulative {table[-1, 12]:.3e})")
        # one row per step (the newton file's last step number, not its row
        # count: the solver appends to newton.txt across reruns in one dir)
        assert len(rows) == int(newton_history[-1, 0]), "the ledger missed a step"
        assert np.all(faces > 0), "the guard booked no guarded face"
        assert np.all(guard_mass > 0.0) and np.all(guard_ei > 0.0), (
            "the guard booked no transport"
        )
    if len(sys.argv) > 4:
        reference, _, _, _ = load(sys.argv[4])
        m_in_reference = closed_mass(reference, r_center, dr, dz, r_null)
        reference_change = (m_in_reference - m_in_initial) / m_in_initial
        print(f"reference arm closed-flux mass change {reference_change:+.3e}")
        if mode == "pulse_guard":
            # reported, not gated: the guard restores the full penalty on
            # the density step at the null, so the guarded arm leaks MORE
            # closed-flux mass than its reduced-penalty twin by design
            print(f"guarded / unguarded closed-flux mass loss ratio "
                  f"{abs(m_in_change) / max(abs(reference_change), 1e-300):.2f}")
        else:
            assert abs(m_in_change) <= 0.5 * abs(reference_change), (
                f"closed-flux mass loss {m_in_change:+.3e} is not below half the "
                f"reference arm's {reference_change:+.3e}"
            )

print(f"mode = {mode}: PASS")
