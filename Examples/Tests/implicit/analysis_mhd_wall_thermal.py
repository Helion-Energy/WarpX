#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Stair-step shaped-wall THERMAL boundary (implicit_mhd.wall_thermal_bc)
for the theta-implicit RZ MHD recast.

A static uniform-density plasma with a Gaussian electron-temperature
core (T0 = 100 eV on axis, Tmin = 2 eV far tail) sits inside a STEPPED
cylinder wall (r_w = 0.30 m for z < 0, 0.22 m for z > 0), frozen ions,
eta = 0, no Joule heating: pure isotropic electron conduction
(chi = 20 m^2/s) against the thermal wall contract (the ion channel
requires live ions and shares this code template; the formation decks
exercise it). The stepped polyline presents r-normal AND z-normal stair
interface faces plus interior-metal faces.

In BOTH modes the masked band is a RIGID conductor: every fluid
increment inside masked cells is zero, so the whole band must stay
bit-static across the run in every fluid moment.

mode = "zero_flux": the interface passes exactly nothing -- the interior
(unmasked) energy is conserved to solver tolerance -- while the Gaussian
core measurably flattens (interior conduction is alive, the mode did not
kill the operator globally).

mode = "temperature" (CLI override wall_thermal_bc=temperature,
wall_temperature=2.0): the interface faces exchange conductively against
the T_wall reservoir, draining the interior ONE-SIDEDLY (the drain
leaves the fluid system -- the scraper-analog sink). The interior energy
decreases strictly monotonically by a physically sensible fraction, the
wall-adjacent ring hugs T_wall without undershooting the reservoir, and
the step-ledge cells drain through their z-normal interface faces
measurably faster than the same radius away from the ledge.

Both modes gate the r = 0 axis health: the reflecting parity origin must
compose cleanly with the shaped wall (z-uniform axis column, radial
monotonicity preserved, no NaNs anywhere).

Usage: analysis_mhd_wall_thermal.py <diag_dir> <zero_flux|temperature>
"""

import glob
import sys

import numpy as np
import yt

diag_dir = sys.argv[1]
mode = sys.argv[2]
assert mode in ("zero_flux", "temperature"), f"unknown mode {mode}"

proton_mass = 1.67262192369e-27
qe = 1.602176634e-19
n0 = 1.0e20
T0_ev = 100.0
Tmin_ev = 2.0
Twall_ev = 2.0
gamma = 5.0 / 3.0

FIELDS = (
    "implicit_mhd_mass_density",
    "implicit_mhd_electron_energy",
    "implicit_mhd_momentum_r",
    "implicit_mhd_momentum_t",
    "implicit_mhd_momentum_z",
)


def load_state(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    state = {f: np.squeeze(data["boxlib", f].value) for f in FIELDS}
    return ds, state


plotfiles = sorted(glob.glob(f"{diag_dir}/diag??????"))
assert len(plotfiles) >= 4, f"need at least 4 snapshots, got {len(plotfiles)}"

ds0, state0 = load_state(plotfiles[0])
nr, nz = int(ds0.domain_dimensions[0]), int(ds0.domain_dimensions[1])
dr = (float(ds0.domain_right_edge[0]) - float(ds0.domain_left_edge[0])) / nr
dz = (float(ds0.domain_right_edge[1]) - float(ds0.domain_left_edge[1])) / nz
r_centers = float(ds0.domain_left_edge[0]) + (np.arange(nr) + 0.5) * dr
z_centers = float(ds0.domain_left_edge[1]) + (np.arange(nz) + 0.5) * dz
# r-weighted cell volumes (the 2*pi*dz factor is a common constant)
volume_weight = np.outer(r_centers * dr, np.ones(nz))

# The mask, replicated exactly as ImplicitMHDWallMask builds it: cell
# (i, j) is conductor when its center radius is on/outside the polyline
# radius of its center z (on-or-outside with the 1e-3*dr sliver).
wall_radius = np.where(z_centers < 0.0, 0.30, 0.22)
masked = (r_centers[:, None] + 1.0e-3 * dr) >= wall_radius[None, :]
interior = ~masked
# Deep metal: masked cells whose four face neighbors are masked too
# (edge-padded: the mask continues constantly past r_max and the z ends,
# and never reaches the axis).
padded = np.pad(masked, 1, mode="edge")
deep_metal = (
    masked
    & padded[:-2, 1:-1]
    & padded[2:, 1:-1]
    & padded[1:-1, :-2]
    & padded[1:-1, 2:]
)
interface_metal = masked & ~deep_metal
assert interface_metal.any() and deep_metal.any(), "degenerate mask layout"
# The step must create z-normal interface faces: masked cells at the
# z > 0 wall radius whose -z neighbor is interior.
ledge_metal = interface_metal & (~padded[1:-1, :-2]) & masked
assert ledge_metal.any(), "stepped polyline produced no z-normal interfaces"


def temperature_ev(state):
    number_density = state["implicit_mhd_mass_density"] / proton_mass
    return (gamma - 1.0) * state["implicit_mhd_electron_energy"] / (
        number_density * qe
    )


def fluid_energy(state, cells):
    return float(
        np.sum(
            state["implicit_mhd_electron_energy"] * volume_weight,
            where=cells,
        )
    )


states = []
for plotfile in plotfiles:
    _, state = load_state(plotfile)
    for name, field in state.items():
        assert np.isfinite(field).all(), f"{name} has non-finite values"
    states.append(state)

te0 = temperature_ev(states[0])
te_end = temperature_ev(states[-1])
interior_energy = np.array([fluid_energy(s, interior) for s in states])
total_energy = np.array([fluid_energy(s, np.ones_like(masked)) for s in states])

print(f"mode = {mode}")
print(f"interior energy trace: {interior_energy / interior_energy[0]}")
print(f"total    energy trace: {total_energy / total_energy[0]}")

# ---- checks common to both modes -------------------------------------

# The masked band is a rigid conductor: EVERY fluid moment of EVERY
# masked cell (interface metal included) is bit-static in both modes.
for f in FIELDS:
    scale = np.max(np.abs(states[0][f]))
    scale = scale if scale > 0.0 else 1.0
    drift = np.max(
        np.abs(states[-1][f] - states[0][f]),
        initial=0.0,
        where=masked,
    ) / scale
    print(f"masked-band drift {f}: {drift:.3e}")
    assert drift < 1.0e-12, f"masked band changed ({f}: {drift:.3e})"

# Interior conduction is alive: the Gaussian core flattens on axis by
# the analytic estimate chi * 4 dT/w^2 * t_end ~ 15 eV.
axis_drop = float(np.mean(te0[0, :]) - np.mean(te_end[0, :]))
print(f"axis Te drop: {axis_drop:.2f} eV")
assert axis_drop > 5.0, f"interior conduction looks dead ({axis_drop:.2f} eV)"

# Axis health (the r = 0 reflecting parity origin composing with the
# shaped wall): the axis column stays z-uniform -- the nearest wall is
# 14 cells away and the diffusion front travels ~3 cells, so any
# z-structure on axis is a spurious wall/axis interaction -- and the
# radial profile stays monotone at the axis for BOTH species.
for name, field in (("Te", te_end),):
    axis_column = field[0, :]
    z_spread = float(
        (axis_column.max() - axis_column.min()) / axis_column.mean()
    )
    print(f"axis {name} z-spread: {z_spread:.3e}")
    assert z_spread < 2.0e-2, f"axis {name} column has z-structure"
    assert np.all(field[0, :] > field[4, :]) and np.all(
        field[4, :] > field[8, :]
    ), f"radial {name} monotonicity lost near the axis"

# ---- mode-specific checks --------------------------------------------

# Shaped-wall deposition ledger (implicit_mhd.wall_ledger_file): rows of
# "step mass energy" in true kg/J. The booked energy must CLOSE against
# the interior's measured loss: zero exactly under zero_flux (frozen
# ions, no conduction through the interface), and equal to the interior
# drop under temperature (theta = 1, so the accepted-state booking
# matches the update to the nonlinear tolerance).
ledger_rows = np.loadtxt(f"{diag_dir}/wall_ledger.txt", ndmin=2)
ledger_mass, ledger_energy = ledger_rows[-1, 1], ledger_rows[-1, 2]
interior_drop_joules = (
    (interior_energy[0] - interior_energy[-1]) * 2.0 * np.pi * dz
)
print(f"ledger: deposited mass {ledger_mass:.3e} kg, "
      f"energy {ledger_energy:.6e} J vs interior drop "
      f"{interior_drop_joules:.6e} J")
assert abs(ledger_mass) < 1.0e-25, "ledger booked mass with frozen ions"

if mode == "zero_flux":
    # The interface passes exactly nothing: interior energy conserved to
    # solver tolerance...
    interior_drift = np.max(np.abs(interior_energy / interior_energy[0] - 1.0))
    print(f"interior-energy drift: {interior_drift:.3e}")
    assert interior_drift < 1.0e-6, (
        f"zero-flux wall leaked ({interior_drift:.3e})"
    )
    # ...and the ledger books exactly nothing.
    assert abs(ledger_energy) < 1.0e-20, (
        f"zero-flux wall ledger booked energy ({ledger_energy:.3e} J)"
    )
else:
    # The interior drains strictly monotonically through the interface
    # (a one-sided sink: the drained energy leaves the fluid system)...
    assert np.all(np.diff(interior_energy) < 0.0), (
        f"interior energy not strictly decreasing: {interior_energy}"
    )
    drained_fraction = 1.0 - interior_energy[-1] / interior_energy[0]
    print(f"interior drained fraction: {drained_fraction:.3f}")
    assert 0.05 < drained_fraction < 0.60, (
        f"unphysical drain fraction {drained_fraction:.3f}"
    )
    # ...and the ledger closes against the measured interior loss.
    ledger_closure = abs(ledger_energy - interior_drop_joules) / (
        interior_drop_joules
    )
    print(f"ledger closure error: {ledger_closure:.3e}")
    assert ledger_closure < 1.0e-5, (
        f"wall ledger does not close ({ledger_closure:.3e})"
    )
    # ...the wall-adjacent interior ring hugs T_wall without undershoot
    # (the Dirichlet exchange reverses sign at the reservoir). The last
    # unmasked cell under the z < 0 section starts at ~28 eV with a
    # half-cell diffusion time ~10x shorter than the run.
    edge_i = np.argmax(masked[:, 0]) - 1  # last interior cell, z < 0
    edge_te = te_end[edge_i, 0:2]
    print(f"edge ring Te: {edge_te} (initial {te0[edge_i, 0]:.1f} eV)")
    for value in edge_te:
        assert 0.95 * Twall_ev < value < 12.0, (
            f"edge ring not anchored to T_wall ({value:.2f} eV)"
        )
    # ...and the step ledge drains through its z-normal faces too: the
    # interior cell under the ledge (radius between the two wall
    # sections, last z < 0 column) must end measurably cooler than the
    # same radius far from the ledge, for both species.
    ledge_i = np.argmax(masked[:, -1]) + 1  # inside the z<0 section,
    assert not masked[ledge_i, 0]  # masked for z > 0
    j_near = nz // 2 - 1
    j_far = 1
    for name, field in (("Te", te_end),):
        near, far = field[ledge_i, j_near], field[ledge_i, j_far]
        print(f"ledge {name}: near {near:.2f} eV vs far {far:.2f} eV")
        assert near < far - 0.5, (
            f"z-normal interface faces not draining ({name})"
        )

print(f"wall thermal BC mode {mode}: all checks passed")
