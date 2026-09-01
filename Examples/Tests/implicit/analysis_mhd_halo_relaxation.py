#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Calibrated checks of the windowed halo temperature-relaxation outlet.

Modes (sys.argv[1]):

  on -- three-region static column (hot sub-window band at 100 eV, an
    above-window band at the SAME pressure, a cold sub-window band at
    0.5 eV) with the outlet on (temperature target, T_med = 1 eV):

      * hot-band cells drain BOTH channels toward the cold-medium
        targets at the local density as the theta-discretized linear
        decay, per step
            (u - u_target) -> (u - u_target) * A,
            A = (1 - (1-theta) nu dt) / (1 + theta nu dt),
        exact in the fully-engaged regime of the one-sided gate (the
        band stays at ~100x the target, where the rectifier is
        identically 1) -- the measured rate must match nu to < 2%
        (theta bias accounted through the exact inverse map);
      * the ion drain acts on the internal part relative to the CURRENT
        kinetic energy, so momentum -- and with it the kinetic part of
        E_i -- must stay invariant;
      * above-window cells (n = 2.5 n_max >= 1.25 n_max, window weight
        exactly zero) are untouched despite sitting far above target
        (loaded at the hot band's pressure: a leaky window would drain
        them by ~3% here);
      * cold cells (0.5 eV < T_med) are NOT heated: the rectifier is
        exactly closed below the target;
      * the ledger closes against the domain (U_e + E_i) loss to
        < 1e-6 relative (a converged solve books the removal exactly).

  off -- the knob-off NULL TWIN of the same column: everything interior
    must be unchanged (the hot band would have lost ~4% otherwise) and
    no ledger file may exist -- the bit-identical-when-off contract at
    regression strength.

  ion_target -- completely uniform sub-window column, Te = 100 eV,
    Ti = 5 eV, halo_relaxation_target = ion (the reference code's te <= tm valve,
    vp.f90:731, in relaxed form): U_e decays toward the LOCAL frozen
    ion-temperature image n kB Ti / (gamma_e - 1) at the exact theta
    rate, the ion channel/momentum/density are untouched (which also
    keeps the per-step frozen target constant), and the ledger closes
    against the electron-energy loss.

Tolerances (on/off column): the cold band carries a static pressure dip
whose acoustic response stays confined to the contact-adjacent cells
over 8 steps (sound crosses ~1e-3 of a cell per step); interior cells
(>= 3 cells from every contact) see only solver-tolerance noise plus
second-order leakage, as in the halo-pedestal calibrated tests. The
ion_target column is exactly uniform: every cell is interior.
"""

import os
import sys

import numpy as np
import warpx_constants as constants
import yt


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


mode = sys.argv[1]
assert mode in ("on", "off", "ion_target"), mode
initial_ds, initial = get_data(sys.argv[2])
final_ds, final = get_data(sys.argv[3])

number_of_cells = 64
n_hot = 1.0e18
gamma = 5.0 / 3.0
transverse_velocity = 2.0e4
relaxation_rate = 5.0e6
dt = 1.0e-9
theta = 0.5
steps = 8
ledger_file = "halo_relaxation_ledger.txt"

# Theta-discretized per-step decay factor of the relaxation ODE
# u' = -nu (u - u_target).
x = relaxation_rate * dt
decay = (1.0 - (1.0 - theta) * x) / (1.0 + theta * x)


def measured_rate(deviation_final, deviation_initial):
    """Invert the theta map for the effective rate over the run."""
    ratio = np.mean(deviation_final) / np.mean(deviation_initial)
    per_step = ratio ** (1.0 / steps)
    return (1.0 - per_step) / (dt * (1.0 - theta + theta * per_step))


initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
initial_momentum = initial["boxlib", "implicit_mhd_momentum_density"].value.ravel()
final_momentum = final["boxlib", "implicit_mhd_momentum_density"].value.ravel()
initial_electron = initial["boxlib", "implicit_mhd_electron_energy"].value.ravel()
final_electron = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()
initial_ion = initial["boxlib", "implicit_mhd_ion_energy"].value.ravel()
final_ion = final["boxlib", "implicit_mhd_ion_energy"].value.ravel()

kinetic_initial = 0.5 * initial_momentum**2 / initial_density


def check_ledger_closure(domain_loss):
    """Cumulative booked removal vs the measured domain energy loss."""
    ledger = np.loadtxt(ledger_file, ndmin=2)
    assert ledger.shape[0] == steps, ledger.shape
    assert int(ledger[-1, 0]) == steps
    booked = ledger[-1, 1]
    closure = abs(booked - domain_loss) / abs(domain_loss)
    print(
        f"ledger booked {booked:.9e} J/m^2, domain loss "
        f"{domain_loss:.9e} J/m^2, closure {closure:.3e}"
    )
    assert closure < 1.0e-6, closure


if mode == "ion_target":
    # Completely uniform column: every cell is interior.
    ion_temperature = 5.0
    electron_target = (
        n_hot * ion_temperature * constants.elementary_charge / (gamma - 1.0)
    )
    # Density, momentum, and the ion channel are untouched (the valve
    # caps electrons only) -- this also keeps the frozen per-step
    # target constant, making the decay exactly the theta map.
    np.testing.assert_allclose(final_density, initial_density, rtol=1.0e-9)
    np.testing.assert_allclose(final_momentum, initial_momentum, rtol=1.0e-9)
    np.testing.assert_allclose(final_ion, initial_ion, rtol=1.0e-9)
    prediction = electron_target + (initial_electron - electron_target) * (decay**steps)
    np.testing.assert_allclose(final_electron, prediction, rtol=1.0e-6)
    nu_electron = measured_rate(
        final_electron - electron_target, initial_electron - electron_target
    )
    print(
        f"ion_target: measured electron rate {nu_electron:.6e} 1/s "
        f"vs nu {relaxation_rate:.6e} "
        f"({abs(nu_electron / relaxation_rate - 1.0):.3e} relative)"
    )
    assert abs(nu_electron / relaxation_rate - 1.0) < 0.02
    # Ledger closes against the electron-energy loss alone.
    cell_size = 1.0 / number_of_cells
    domain_loss = np.sum(initial_electron - final_electron) * cell_size
    check_ledger_closure(domain_loss)
    sys.exit(0)

# --- Three-region column (modes "on" and "off") ---
medium_temperature = 1.0
electron_target = (
    n_hot * medium_temperature * constants.elementary_charge / (gamma - 1.0)
)
ion_target = electron_target  # gamma_e = gamma_i in the deck

cells = np.arange(number_of_cells)
hot = cells < 28
bulk = (cells >= 28) & (cells < 44)
cold = cells >= 44
# Contacts sit at cell boundaries 0/63 (periodic wrap), 27|28, and
# 43|44: keep cells >= 3 cells away from every contact.
contact_distance = np.minimum.reduce(
    [
        np.abs(cells - (-0.5)),
        np.abs(cells - 27.5),
        np.abs(cells - 43.5),
        np.abs(cells - 63.5),
    ]
)
interior = contact_distance >= 3.0
hot_interior = hot & interior
bulk_interior = bulk & interior
cold_interior = cold & interior

internal_initial = initial_ion - kinetic_initial

if mode == "off":
    # NULL TWIN: nothing may move away from the cold-band contacts (the
    # outlet would have drained the hot band by ~4% -- four orders of
    # magnitude above the tolerance), and no ledger may exist. The
    # acoustic bleed of the static cold-band pressure dip decays ~40x
    # per cell (measured: 9.5e-7 at 3 cells, 8.5e-12 at 6 cells), so a
    # second, machine-tight tier certifies the legacy path is
    # numerically silent where the physics is silent.
    deep = contact_distance >= 6.0
    for name, initial_field, final_field in (
        ("density", initial_density, final_density),
        ("momentum", initial_momentum, final_momentum),
        ("electron energy", initial_electron, final_electron),
        ("ion energy", initial_ion, final_ion),
    ):
        np.testing.assert_allclose(
            final_field[interior],
            initial_field[interior],
            rtol=4.0e-6,
            err_msg=name,
        )
        np.testing.assert_allclose(
            final_field[deep],
            initial_field[deep],
            rtol=1.0e-10,
            err_msg=name,
        )
    assert not os.path.exists(ledger_file), (
        "knob-off run must not produce a relaxation ledger"
    )
    print("off: interior state unchanged, no ledger file -- null twin holds")
    sys.exit(0)

# --- mode == "on" ---
# Density and momentum are invariant away from the cold-band contacts:
# the outlet drains internal energies only (the KE-following ion target
# leaves the kinetic channel untouched).
np.testing.assert_allclose(
    final_density[interior], initial_density[interior], rtol=1.0e-6
)
np.testing.assert_allclose(
    final_momentum[interior], initial_momentum[interior], rtol=1.0e-6
)

# Hot band: both channels decay toward the local cold-medium targets at
# the exact theta-discretized rate.
electron_prediction = electron_target + (initial_electron - electron_target) * (
    decay**steps
)
ion_prediction = (
    kinetic_initial + ion_target + (internal_initial - ion_target) * decay**steps
)
np.testing.assert_allclose(
    final_electron[hot_interior], electron_prediction[hot_interior], rtol=1.0e-6
)
np.testing.assert_allclose(
    final_ion[hot_interior], ion_prediction[hot_interior], rtol=1.0e-6
)
nu_electron = measured_rate(
    final_electron[hot_interior] - electron_target,
    initial_electron[hot_interior] - electron_target,
)
internal_final = final_ion - kinetic_initial
nu_ion = measured_rate(
    internal_final[hot_interior] - ion_target,
    internal_initial[hot_interior] - ion_target,
)
print(
    f"on: measured electron rate {nu_electron:.6e} 1/s, ion rate "
    f"{nu_ion:.6e} 1/s vs nu {relaxation_rate:.6e} "
    f"(rel {abs(nu_electron / relaxation_rate - 1.0):.3e} / "
    f"{abs(nu_ion / relaxation_rate - 1.0):.3e})"
)
assert abs(nu_electron / relaxation_rate - 1.0) < 0.02
assert abs(nu_ion / relaxation_rate - 1.0) < 0.02

# Above-window band: window weight exactly zero -- untouched despite
# sitting far above target (a leaky window would drain ~3% of U_e).
np.testing.assert_allclose(
    final_electron[bulk_interior], initial_electron[bulk_interior], rtol=1.0e-6
)
np.testing.assert_allclose(
    final_ion[bulk_interior], initial_ion[bulk_interior], rtol=1.0e-6
)

# Cold band: below-target cells are NEVER heated (one-sided rectifier
# exactly closed at 0.5 eV < T_med).
np.testing.assert_allclose(
    final_electron[cold_interior], initial_electron[cold_interior], rtol=1.0e-6
)
np.testing.assert_allclose(
    final_ion[cold_interior], initial_ion[cold_interior], rtol=1.0e-6
)
assert np.all(
    final_electron[cold_interior] <= initial_electron[cold_interior] * (1.0 + 1.0e-6)
)

# Ledger closure against the FULL domain (U_e + E_i) loss: every other
# term telescopes (periodic mesh) or cancels pairwise, so a converged
# solve books the outlet's removal exactly.
cell_size = 1.0 / number_of_cells
domain_loss = (
    np.sum(initial_electron - final_electron) + np.sum(initial_ion - final_ion)
) * cell_size
check_ledger_closure(domain_loss)

# Nothing electromagnetic happens: B stays zero, and E = -u_e x B = 0.
for data in (initial, final):
    for field_name in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"):
        np.testing.assert_allclose(
            data["boxlib", field_name].value, 0.0, rtol=0.0, atol=1.0e-20
        )
