#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Wall-drain gate anchored at the reachable set (temperature-floor image).

The deck (see inputs_test_rz_theta_implicit_mhd_wall_floor_gate) puts a
quiet uniform plasma at T0 = 10 eV inside a flat dielectric wall with
wall_thermal_bc = outflow_limited and a reservoir far BELOW the temperature
floors (T_wall = 0.5 eV << T_floor = 9.5 eV), the one-way ratchet
engaged everywhere from step 0, and dt far above the wall-face
free-streaming time.

REPRODUCER CONTRACT (the bug this test pins): a wall drain gated on the
reservoir value alone never closes while the admissibility projection's
temperature ratchet pins the same components at the floor bound -- the
first solve descends the live column to the bound, every later solve is
fully pinned (Armijo cannot decrease, the free subspace is at round-off
so the noise gate skips the rescue), iteration-0 frozen steps accumulate
within a handful of steps, and the run ABORTS on newton.max_frozen_steps
long before max_step (newton.txt then never reaches it).

WITH the gate anchored at max(reservoir, temperature-floor image) the
drain closes smoothly exactly where the projection stops the state:

  1. the run COMPLETES all its steps (newton.txt reaches max_step);
  2. zero frozen solves: the first solve genuinely works (iters >= 1)
     and EVERY solve converges (final norms at tolerance -- a pinned
     deadlock plateaus at the drain-demand defect instead);
  3. the wall-adjacent live ring settles AT the floor FROM ABOVE, in
     both the electron channel and the dual_energy ion pair (the twin
     U_i booking): the drain ran (ring temperature well below T0) but
     never cooled through the floor, and it did NOT stop at the
     reservoir-anchored value (which would leave the ring essentially
     at T0/at the old gate);
  4. the masked band sits on the rigid clamp image (locates the ring).

Usage: analysis_mhd_wall_floor_gate.py <initial plotfile> <final plotfile>
"""

import sys

import numpy as np
import yt

proton_mass = 1.67262192595e-27  # WarpX parser m_p (ablastr::constant)
qe = 1.602176634e-19
n0 = 1.0e20
rho0 = n0 * proton_mass
T0_ev = 10.0
Tfloor_ev = 9.5
gamma = 5.0 / 3.0
MAX_STEP = 16

FIELDS = (
    "implicit_mhd_mass_density",
    "implicit_mhd_electron_energy",
    "implicit_mhd_ion_internal_energy",
)


def load_state(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, {f: np.squeeze(data["boxlib", f].value) for f in FIELDS}


def temperature_ev(energy, rho):
    """T [eV] of a specific internal-energy block (n kB T = rho (q/m) T_ev)."""
    return (gamma - 1.0) * energy * proton_mass / (rho * qe)


ds0, initial = load_state(sys.argv[1])
dsf, final = load_state(sys.argv[2])

# Wall geometry of the deck: flat polyline at r_w = 0.065 m between the
# cell centers at 0.0625 (live, i = 12) and 0.0675 (masked, i = 13).
nr = int(ds0.domain_dimensions[0])
dr = (float(ds0.domain_right_edge[0]) - float(ds0.domain_left_edge[0])) / nr
r_centers = float(ds0.domain_left_edge[0]) + (np.arange(nr) + 0.5) * dr
masked = (r_centers + 1.0e-3 * dr) >= 0.065
i_ring = int(np.argmax(masked)) - 1  # last live cell = 12

# 1. Completion: the unfixed solver aborts on the frozen-step trip
# within a handful of steps; newton.txt then never reaches MAX_STEP.
history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
steps = history[:, 0]
iters = history[:, 2]
norm_abs = history[:, 4]
assert steps[-1] == MAX_STEP, (
    f"run did not complete: newton.txt ends at step {steps[-1]:.0f} of {MAX_STEP}"
)

# 2. Zero frozen solves. The first solve carries the whole gate-closing
# descent (a genuine Newton update: an iteration-0 stagnation logs 0
# iters), and every solve must CONVERGE -- a pinned deadlock instead
# plateaus at the drain-demand defect (production: 1.5e-2 scaled; the
# tolerances here are 1e-13 abs / 1e-10 rel, so 1e-6 discriminates by
# >4 decades while leaving late quiet-step round-off plenty of room).
assert iters[0] >= 1, (
    "the first solve accepted no update (iters == 0): the gate-closing "
    "descent never happened"
)
assert norm_abs.max() < 1.0e-6, (
    f"a solve ended at norm {norm_abs.max():.3e}: the wall-drain demand "
    "never resolved (the pinned-deadlock fingerprint)"
)
print(
    f"newton: {len(steps)} solves, iters[0] = {iters[0]:.0f}, "
    f"max final norm = {norm_abs.max():.3e}"
)

# 4 (needed first, to trust i_ring). Masked band on the rigid clamp
# image: rho = mass floor (1e-4 rho0), 4 decades below the live column.
band_rho = final["implicit_mhd_mass_density"][masked, :]
assert np.allclose(band_rho, 1.0e-4 * rho0, rtol=1.0e-6), (
    "masked band left the rigid clamp image: the ring index is untrusted"
)

# 3. The wall-adjacent live ring settled AT the floor from above, in
# BOTH channels (electron U_e and the dual_energy auxiliary U_i -- the
# twin booking of the identical gated wall flux).
rho_ring = final["implicit_mhd_mass_density"][i_ring, :]
checks = (
    ("Te", final["implicit_mhd_electron_energy"][i_ring, :]),
    ("Ti", final["implicit_mhd_ion_internal_energy"][i_ring, :]),
)
for name, ring_energy in checks:
    t_ring = temperature_ev(ring_energy, rho_ring)
    print(
        f"wall ring {name}: min = {t_ring.min():.4f} eV, "
        f"max = {t_ring.max():.4f} eV (floor {Tfloor_ev}, IC {T0_ev})"
    )
    # never through the floor (the ratchet-consistent gate closed)
    assert t_ring.min() >= Tfloor_ev * (1.0 - 1.0e-9), (
        f"{name} cooled through the floor: {t_ring.min():.4f} eV"
    )
    # settled AT the floor: measured ring rest point 9.52-9.56 eV
    # (gate-vs-conduction balance a fraction of a smoothstep width above
    # the anchor); 2% headroom keeps the assert far from both the floor
    # (9.5) and any reservoir-anchored stall (which parks near T0).
    assert t_ring.max() <= Tfloor_ev * 1.02, (
        f"{name} did not settle at the floor: {t_ring.max():.4f} eV"
    )
    # and the drain genuinely ran (discriminates an over-closed gate
    # that would leave the ring at the 10 eV IC)
    assert t_ring.max() < 0.99 * T0_ev, (
        f"{name} never drained: {t_ring.max():.4f} eV (IC {T0_ev} eV)"
    )

# IC sanity: the ratchet was engaged from step 0 (IC at T0 > floor).
t_init = temperature_ev(
    initial["implicit_mhd_electron_energy"][i_ring, :],
    initial["implicit_mhd_mass_density"][i_ring, :],
)
assert np.allclose(t_init, T0_ev, rtol=1.0e-6), "IC ring temperature wrong"

print("wall floor gate: PASS")
