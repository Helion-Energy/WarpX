#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Shaped-wall no-slip LIFT-OFF / REPLENISHMENT gates.

See inputs_test_rz_theta_implicit_mhd_wall_no_slip_lift.  A no-slip wall
constrains the TANGENTIAL slip at the wall FACE.  Implementing it as a
volumetric pin of all three momentum components of the wall-adjacent
LIVE cells instead does two unphysical things:

  - it forbids the fluid from moving NORMAL to the wall (no lift-off),
  - it sets the face velocities on BOTH faces of the outermost live
    cell to zero, sealing that finite control volume off from the rest
    of the domain: no advective replenishment of mass OR energy.

A sealed cell that also sits against a cold wall reservoir (and has no
volumetric ion source, as in the production arm where
joule_ion_fraction = -1 routes all Joule to the electrons) is left with
a one-sided drain and no advective supply to balance it.  These gates
measure the SEAL and the corrupted budget directly; this is not a crash
reproducer, and it runs on the production flux (fluid_flux = central),
where the pinned configuration completes the window with no positivity
violations.

Gates, against the wall_no_slip = 0 twin of the identical deck
(measured face condition | measured volumetric pin | threshold):

  1. LIFT-OFF: the wall-adjacent live cell keeps a normal (radial)
     momentum comparable to the twin's.
     0.9982 | 0.0 exactly | > 0.5.
  2. REPLENISHMENT: its mass density follows the imposed -2 Ar
     expansion drift instead of being held at its initial value, i.e.
     the cell still exchanges fluid with its neighbours.
     +3.125e-2 | -1.002e-1 (SIGN REVERSED: the frozen cell is a rigid
     obstacle the expansion piles mass against) | > 0.5 x the twin's
     +3.126e-2.  At band width 2 -- the setting the production arm ran
     -- the pinned cell's density is bit-frozen for the entire run.
  3. ION-ENERGY BUDGET: its ion energy tracks the twin's.  The
     no-slip wall imposes a ZERO face velocity, so it does no work: the
     tangential kinetic energy its shear removes becomes internal energy
     in the same cell and the cell's ion-energy budget is unchanged.  A
     sealed cell instead has the cold-wall drain as its only ion-energy
     channel and a completely different budget.
     1.9e-3 | 1.47e-1 | < 3e-2.
  4. TANGENTIAL DRAG: the no-slip condition still removes tangential
     momentum at the wall face relative to the twin (17.1 %; the pin
     removes 100 %, so this gate alone does not separate them), and the
     far interior is untouched -- the volumetric pin instead reflects
     the expansion as a compression wave that REVERSES the far
     interior's radial momentum.
     worst far-interior error 1.1e-3 | 8.6e-1 | < 1e-2.

Usage:
  analysis_mhd_wall_no_slip_lift.py <final_plotfile> off
  analysis_mhd_wall_no_slip_lift.py <final_plotfile> on <off_final_plotfile>
"""

import sys

import numpy as np
import warpx_constants as constants
import yt

yt.set_log_level(50)

# constants from inputs_test_rz_theta_implicit_mhd_wall_no_slip_lift
NUMBER_DENSITY = 1.0e20
RHO0 = NUMBER_DENSITY * constants.proton_mass
TEMPERATURE_EV = 100.0
GAMMA = 5.0 / 3.0
EXPANSION_RATE = 5.0e4
SHEAR_RATE = 5.0e4
WALL_RADIUS = 0.30
NUMBER_OF_CELLS_R = 24
RADIAL_EXTENT = 0.48
CELL_SIZE = RADIAL_EXTENT / NUMBER_OF_CELLS_R
FINAL_TIME = 12 * 8.0e-8

# Mask geometry (the implementation's cell-centered convention: first
# cell whose center (i + 1/2) dr sits on/outside the polyline radius).
FIRST_MASKED = int(np.ceil(WALL_RADIUS / CELL_SIZE - 0.5 - 1.0e-3))
assert FIRST_MASKED == 15
WALL_CELL = FIRST_MASKED - 1  # = 14, the outermost LIVE cell
FAR_INTERIOR = slice(2, 9)  # rows 2..8, well inside any wall layer

FIELDS = (
    "implicit_mhd_mass_density",
    "implicit_mhd_ion_energy",
    "implicit_mhd_momentum_r",
    "implicit_mhd_momentum_t",
    "implicit_mhd_momentum_z",
)

# The initial ion energy is purely the thermal part plus the kinetic
# part of the imposed (u_r, u_z) = (Ar r, Az r) field.
RADIUS = (np.arange(NUMBER_OF_CELLS_R) + 0.5) * CELL_SIZE
ION_ENERGY_0 = NUMBER_DENSITY * TEMPERATURE_EV * constants.elementary_charge / (
    GAMMA - 1.0
) + (0.5 * RHO0 * (EXPANSION_RATE**2 + SHEAR_RATE**2) * RADIUS**2)


def load(plotfile):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return {name: grid["boxlib", name].value[:, :, 0] for name in FIELDS}


def row_mean(state, name, index):
    return float(np.mean(state[name][index, :]))


state = load(sys.argv[1])
mode = sys.argv[2]
assert mode in ("on", "off"), f"unknown mode {mode}"

for name, field in state.items():
    assert np.isfinite(field).all(), f"{name} has non-finite values"

# The imposed uniform expansion div u = 2 Ar drains every live cell's
# density at the same rate; the wall-adjacent cell is no exception
# unless the pin has sealed it.
expected_drift = 1.0 - np.exp(-2.0 * EXPANSION_RATE * FINAL_TIME)

density = row_mean(state, "implicit_mhd_mass_density", WALL_CELL)
density_drift = (RHO0 - density) / RHO0
momentum_r = row_mean(state, "implicit_mhd_momentum_r", WALL_CELL)
momentum_z = row_mean(state, "implicit_mhd_momentum_z", WALL_CELL)
ion_energy = row_mean(state, "implicit_mhd_ion_energy", WALL_CELL)
free_momentum_r = RHO0 * EXPANSION_RATE * RADIUS[WALL_CELL]
free_momentum_z = RHO0 * SHEAR_RATE * RADIUS[WALL_CELL]

print(f"wall-adjacent live cell i = {WALL_CELL} (mode {mode})")
print(
    f"  mass_density  {density:.8e} "
    f"(drift {density_drift:.4e}, uniform expansion {expected_drift:.4e})"
)
print(
    f"  momentum_r    {momentum_r:.8e} ({momentum_r / free_momentum_r:.4e} of rho Ar r)"
)
print(
    f"  momentum_z    {momentum_z:.8e} ({momentum_z / free_momentum_z:.4e} of rho Az r)"
)
print(
    f"  ion_energy    {ion_energy:.8e} ({ion_energy / ION_ENERGY_0[WALL_CELL]:.4e} of E_i(0))"
)

if mode == "on":
    assert len(sys.argv) > 3, "on mode needs the wall_no_slip = 0 twin"
    reference = load(sys.argv[3])

    reference_density = row_mean(reference, "implicit_mhd_mass_density", WALL_CELL)
    reference_drift = (RHO0 - reference_density) / RHO0
    reference_momentum_r = row_mean(reference, "implicit_mhd_momentum_r", WALL_CELL)
    reference_momentum_z = row_mean(reference, "implicit_mhd_momentum_z", WALL_CELL)
    reference_ion_energy = row_mean(reference, "implicit_mhd_ion_energy", WALL_CELL)

    # --- Gate 1: normal lift-off ---
    # A no-slip wall constrains tangential slip only.  The wall-adjacent
    # cell must be free to move normal to the wall; the volumetric pin
    # zeroed this component exactly.
    lift = momentum_r / reference_momentum_r
    print(f"gate 1 lift-off: momentum_r / twin = {lift:.6f}")
    assert lift > 0.5, (
        f"the wall-adjacent live cell kept only {lift:.3e} of the free "
        "twin's NORMAL momentum: the no-slip wall must not pin the "
        "component that lifts fluid off the wall"
    )

    # --- Gate 2: advective replenishment ---
    # The uniform expansion drains every live cell.  A cell whose two
    # faces both carry zero velocity cannot follow it at all -- and a
    # cell whose momentum is pinned while its neighbours flow becomes a
    # rigid obstacle the expansion piles mass against, reversing the
    # sign of the drift.
    print(
        f"gate 2 replenishment: density drift {density_drift:.6e} vs "
        f"twin {reference_drift:.6e} (uniform {expected_drift:.6e})"
    )
    assert density_drift > 0.5 * reference_drift, (
        f"the wall-adjacent live cell drifted {density_drift:.3e} against "
        f"the twin's {reference_drift:.3e}: the wall condition sealed the "
        "cell off from its neighbours (no advective replenishment)"
    )

    # --- Gate 3: no ion-energy collapse ---
    # The no-slip wall imposes a ZERO face velocity, so it transports no
    # energy: the cell's ion-energy budget must be the twin's.  A sealed
    # cell has a completely different budget -- the cold-wall drain and
    # nothing else.
    energy_error = abs(ion_energy / reference_ion_energy - 1.0)
    print(
        f"gate 3 ion energy: {ion_energy:.6e} vs twin "
        f"{reference_ion_energy:.6e} (error {energy_error:.6e})"
    )
    assert energy_error < 3.0e-2, (
        f"the wall-adjacent live cell's ion energy is off the twin's by "
        f"{energy_error:.3e}: its energy budget is no longer the advective "
        "one, so the cold-wall drain is the only channel it has left"
    )

    # --- Gate 4: the no-slip condition is real, and local ---
    drag = (reference_momentum_z - momentum_z) / reference_momentum_z
    print(f"gate 4 tangential drag: {drag:.6e} of the twin's momentum_z")
    assert drag > 1.0e-2, (
        "wall_no_slip = 1 removed no tangential momentum at the wall "
        "face: the no-slip condition is inert"
    )
    # Locality: over this window sound crosses ~9 cells, so the interior
    # is acoustically connected to the wall layer and a strictly exact
    # test is meaningless; 1 % of the field scale separates the face
    # condition (measured 1.2e-3) from the volumetric pin, which
    # reflects the expansion as a compression wave and REVERSES the far
    # interior's radial momentum (measured > 1 of the scale).
    # The three momentum components share one scale: the setup drives no
    # azimuthal flow at all, so momentum_t's own scale is round-off and
    # a relative test against it would be meaningless.
    momentum_scale = max(
        float(np.max(np.abs(reference[f"implicit_mhd_momentum_{c}"])))
        for c in ("r", "t", "z")
    )
    for name in FIELDS:
        scale = (
            momentum_scale
            if name.startswith("implicit_mhd_momentum_")
            else float(np.max(np.abs(reference[name])))
        )
        far_error = (
            float(
                np.max(
                    np.abs(
                        state[name][FAR_INTERIOR, :] - reference[name][FAR_INTERIOR, :]
                    )
                )
            )
            / scale
        )
        print(f"  far interior |d {name}|max / scale = {far_error:.3e}")
        assert far_error < 1.0e-2, (
            f"the wall condition changed the far interior {name} by "
            f"{far_error:.3e} of its scale: it is leaking past the wall layer"
        )
else:
    # The free-slip twin must actually be free: the wall-adjacent cell
    # keeps its imposed normal and tangential momentum and follows the
    # expansion drift.
    assert momentum_r > 0.5 * free_momentum_r, (
        "the wall_no_slip = 0 twin lost its normal momentum at the wall"
    )
    assert momentum_z > 0.5 * free_momentum_z, (
        "the wall_no_slip = 0 twin is not free-slip"
    )
    assert density_drift > 0.2 * expected_drift, (
        "the wall_no_slip = 0 twin did not follow the expansion drift"
    )

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history[-1, 2] >= 1

print(f"wall no-slip lift-off/replenishment ({mode}): all gates passed")
