#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Calibrated checks of the COLD halo pedestal image
(implicit_mhd.halo_pedestal_temperature_e / _i) on the three-band static
column of inputs_test_1d_theta_implicit_mhd_halo_pedestal_cold.

Setup: bulk (rho0, pressures P_bulk) in z in [0.25, 0.75); a sub-pedestal
halo (rho_halo = 1e-3 rho0, two decades below the pedestal
rho_ped = f max(rho_peak, rho_ref) = 0.1 rho0) on both sides, loaded
COLD (0.1 P_bulk, z < 0.25) on one side and HOT (3 P_bulk, z >= 0.75) on
the other. The cold image is in pressure balance with the bulk
(n_ped kB T_ped = n0 kB T_bulk), so

    U_e,ped = rho_ped (q/m) T_ped_e / (gamma - 1) = P_e,bulk / (gamma - 1),
    e_i,ped = rho_ped (q/m) T_ped_i / (gamma - 1) = P_i,bulk / (gamma - 1).

Modes (sys.argv[1]):

  reset -- the default raise form: at the first step every halo cell is
    raised onto the pedestal STATE (rho = rho_ped, U_e = U_e,ped, and
    E_i = U_i = e_i,ped: dual-energy sync exact, zero momentum), the hot
    band INCLUDED (its excess heat is removed). The column is then a
    uniform-pressure static contact whose residual vanishes identically,
    so the raised state rides unchanged to the last step: exact-equality
    checks (rtol 1e-13) of every raised block, the bulk untouched
    (1e-14). Discrimination: the peak image f x max(U) would lift the
    cold band only to 0.3 P_bulk / (gamma - 1) (max(U) sits in the hot
    band) and leave the hot band at 3 P_bulk; the floor form would leave
    the hot band at 3 P_bulk. Ledger: five rows (one per step); row 1
    books 32 raised cells, the injected mass 32 (rho_ped - rho_halo) dz
    and the SIGNED energy change per species
        dz [16 (U_ped - 0.1 U_ped) + 16 (U_ped - 3 U_ped)] = -1.1 x 16 dz U_ped
    (the hot band gives back more than the cold band receives); rows 2-5
    book nothing more (no cell is sub-pedestal again).

  floor -- implicit_mhd.halo_pedestal_cold_raise = floor: the raise is
    max(own, image). Checked after the FIRST step only (the hot band is
    then a pressure jump and the column evolves): the cold band's
    interior sits exactly on the cold image, the hot band's interior
    keeps 3 P_bulk / (gamma - 1) for both species (rtol 1e-9: only the
    contact-adjacent cells move in one step at CFL ~1e-2), and the
    ledger's row 1 books a pure SOURCE, +0.9 x 16 dz U_ped per species.

  total_energy / cgl -- the reset form under the other ion closures
    (E_i alone, or the CGL pair U_par = rho_ped (q/m) T_ped_i / 2 and
    U_perp = rho_ped (q/m) T_ped_i, i.e. P_i,bulk / 2 and P_i,bulk):
    same static-contact exactness.

Usage: analysis_mhd_halo_pedestal_cold.py <mode> <initial_plotfile> <final_plotfile>
"""

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


def field(data, name):
    return np.array(data["boxlib", name].value).ravel()


mode = sys.argv[1]
assert mode in ("reset", "floor", "total_energy", "cgl", "static", "cov"), mode
initial_ds, initial = get_data(sys.argv[2])
final_ds, final = get_data(sys.argv[3])

number_of_cells = 64
domain_length = 1.0
cell_size = domain_length / number_of_cells
gamma = 5.0 / 3.0
number_density = 1.0e18
rho0 = number_density * constants.proton_mass
rho_halo = 1.0e-3 * rho0
pedestal_fraction = 0.1
reference_density = 0.5 * rho0
electron_temperature_bulk = 1.0
ion_temperature_bulk = 2.0
electron_pressure_bulk = number_density * electron_temperature_bulk * constants.elementary_charge
ion_pressure_bulk = number_density * ion_temperature_bulk * constants.elementary_charge
# The static variant pins the pedestal at n_static = 0.05 n0 (half the
# peak-keyed value) and its cold image at the matching pressure-balance
# temperatures.
static_density_fraction = 0.05
if mode == "static":
    pedestal_temperature_e = electron_temperature_bulk / static_density_fraction
    pedestal_temperature_i = ion_temperature_bulk / static_density_fraction
else:
    pedestal_temperature_e = electron_temperature_bulk / pedestal_fraction
    pedestal_temperature_i = ion_temperature_bulk / pedestal_fraction
ledger_file = "halo_pedestal_ledger.txt"
# Every deck runs five steps (one ledger row each); the floor mode reads
# the plotfile after the first step only.
steps = 5

z = (np.arange(number_of_cells) + 0.5) * cell_size
bulk = (z >= 0.25) & (z < 0.75)
cold_band = z < 0.25
hot_band = z >= 0.75
halo = ~bulk
assert cold_band.sum() == 16 and hot_band.sum() == 16

# The dynamic pedestal keys to the instantaneous peak, not the (lower)
# reference density.
if mode == "static":
    pedestal = static_density_fraction * rho0
    # Discrimination: the fraction alone would give twice this.
    assert abs(pedestal - pedestal_fraction * rho0) > 0.4 * pedestal
else:
    pedestal = pedestal_fraction * max(rho0, reference_density)
    assert pedestal == pedestal_fraction * rho0
# The cold image through n kB T = rho (q/m) T[eV]; in pressure balance
# with the bulk by construction of the deck.
charge_to_mass = constants.elementary_charge / constants.proton_mass
electron_image = pedestal * charge_to_mass * pedestal_temperature_e / (gamma - 1.0)
ion_image = pedestal * charge_to_mass * pedestal_temperature_i / (gamma - 1.0)
np.testing.assert_allclose(electron_image, electron_pressure_bulk / (gamma - 1.0), rtol=1e-13)
np.testing.assert_allclose(ion_image, ion_pressure_bulk / (gamma - 1.0), rtol=1e-13)
cgl_parallel_image = 0.5 * pedestal * charge_to_mass * pedestal_temperature_i
cgl_perp_image = pedestal * charge_to_mass * pedestal_temperature_i

initial_density = field(initial, "implicit_mhd_mass_density")
final_density = field(final, "implicit_mhd_mass_density")
initial_electron = field(initial, "implicit_mhd_electron_energy")
final_electron = field(final, "implicit_mhd_electron_energy")

# Step 0 (written before the first OneStep): the loaded column.
np.testing.assert_allclose(initial_density[bulk], rho0, rtol=1.0e-14)
np.testing.assert_allclose(initial_density[halo], rho_halo, rtol=1.0e-14)
np.testing.assert_allclose(initial_electron[bulk], electron_image, rtol=1.0e-13)
np.testing.assert_allclose(initial_electron[cold_band], 0.1 * electron_image, rtol=1.0e-13)
np.testing.assert_allclose(initial_electron[hot_band], 3.0 * electron_image, rtol=1.0e-13)

if mode == "cgl":
    initial_parallel = field(initial, "implicit_mhd_ion_parallel_energy")
    final_parallel = field(final, "implicit_mhd_ion_parallel_energy")
    initial_perp = field(initial, "implicit_mhd_ion_perp_energy")
    final_perp = field(final, "implicit_mhd_ion_perp_energy")
    np.testing.assert_allclose(initial_parallel[bulk], cgl_parallel_image, rtol=1.0e-13)
    np.testing.assert_allclose(initial_perp[bulk], cgl_perp_image, rtol=1.0e-13)
    np.testing.assert_allclose(initial_parallel[hot_band], 3.0 * cgl_parallel_image, rtol=1.0e-13)
    np.testing.assert_allclose(initial_perp[hot_band], 3.0 * cgl_perp_image, rtol=1.0e-13)
    # Per-cell ion "energy" booked by the ledger under cgl: U_par + U_perp.
    initial_ion = initial_parallel + initial_perp
    ion_image_booked = cgl_parallel_image + cgl_perp_image
else:
    initial_ion = field(initial, "implicit_mhd_ion_energy")
    final_ion = field(final, "implicit_mhd_ion_energy")
    np.testing.assert_allclose(initial_ion[bulk], ion_image, rtol=1.0e-13)
    np.testing.assert_allclose(initial_ion[cold_band], 0.1 * ion_image, rtol=1.0e-13)
    np.testing.assert_allclose(initial_ion[hot_band], 3.0 * ion_image, rtol=1.0e-13)
    ion_image_booked = ion_image

# Momentum and fields are exactly zero throughout (static contact, no
# drive): the raise never touches momentum. Under the change of variables
# the sanitize parks the lifted halo a 2e-6 slack above the background,
# a 2e-6 pressure step that drives a correspondingly tiny flow: momentum
# is bounded by 1e-5 of rho_ped c_s there instead.
sound_speed = np.sqrt(gamma * (electron_pressure_bulk + ion_pressure_bulk) / rho0)
for data in (initial, final):
    momentum = data["boxlib", "implicit_mhd_momentum_density"].value
    if mode == "cov" and data is final:
        assert np.max(np.abs(momentum)) < 1.0e-5 * pedestal * sound_speed, np.max(np.abs(momentum))
    else:
        np.testing.assert_allclose(momentum, 0.0, rtol=0.0, atol=0.0)
    for field_name in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"):
        np.testing.assert_allclose(
            data["boxlib", field_name].value, 0.0, rtol=0.0, atol=1.0e-20
        )

# Band interiors (>= 3 cells from every contact): in the floor mode the
# column is dynamic after the raise and only these cells carry exact
# values after one step.
interior = np.zeros(number_of_cells, dtype=bool)
for band in (cold_band, hot_band, bulk):
    idx = np.flatnonzero(band)
    interior[idx[3:-3]] = True

# The density raise is the same in every mode: the halo rides exactly ON
# the pedestal (interior point of the admissible set), the bulk is
# untouched. Exact on the static column; on the floor mode's dynamic
# column the contact-adjacent cells move at the 1e-7 level in one step,
# so the exact check is restricted to the band interiors there.
if mode == "cov":
    # Change of variables: the sanitize lifted the halo onto the background
    # (its slack margin: 2e-6, a pressure step that moves the contact cells
    # at the 1e-5 level), the pedestal never raised anything.
    np.testing.assert_allclose(final_density[interior & bulk], rho0, rtol=1.0e-9)
    np.testing.assert_allclose(final_density[bulk], rho0, rtol=1.0e-4)
    np.testing.assert_allclose(final_density[interior & halo], pedestal, rtol=1.0e-5)
    np.testing.assert_allclose(final_density[halo], pedestal, rtol=1.0e-3)
elif mode == "floor":
    np.testing.assert_allclose(final_density[interior & bulk], rho0, rtol=1.0e-9)
    np.testing.assert_allclose(final_density[interior & halo], pedestal, rtol=1.0e-9)
    np.testing.assert_allclose(final_density[halo], pedestal, rtol=1.0e-5)
else:
    np.testing.assert_allclose(final_density[bulk], rho0, rtol=1.0e-14)
    np.testing.assert_allclose(final_density[halo], pedestal, rtol=1.0e-14)

ledger = np.loadtxt(ledger_file, ndmin=2)
print("ledger rows:\n", ledger)
assert ledger.shape == (steps, 5), ledger.shape
assert int(ledger[0, 0]) == 1 and int(ledger[-1, 0]) == steps
if mode == "cov":
    # No raise ever: every row books zero cells and zero totals; the
    # injected fields are identically zero; the halo rides on the
    # background (within the sanitize slack) at the image temperatures.
    assert np.all(ledger[:, 1] == 0) and np.all(ledger[:, 2:] == 0.0), ledger
    for name in ("implicit_mhd_pedestal_injected_mass",
                 "implicit_mhd_pedestal_injected_electron_energy",
                 "implicit_mhd_pedestal_injected_ion_energy"):
        assert np.all(field(final, name) == 0.0), name
    # No reset under the change of variables: the COLD band (loaded below
    # the background) was lifted onto it by the sanitize; the HOT band
    # (loaded at 3x the image) keeps its energy -- the floor form -- and
    # drives a slow flow into its neighbours, so the checks sit on the band
    # interiors after the first step.
    final_internal = field(final, "implicit_mhd_ion_internal_energy")
    np.testing.assert_allclose(final_electron[interior & cold_band], electron_image, rtol=1.0e-5)
    np.testing.assert_allclose(final_ion[interior & cold_band], ion_image, rtol=1.0e-5)
    np.testing.assert_allclose(final_internal[interior & cold_band], ion_image, rtol=1.0e-5)
    np.testing.assert_allclose(final_electron[interior & hot_band], 3.0 * electron_image, rtol=1.0e-5)
    np.testing.assert_allclose(final_ion[interior & hot_band], 3.0 * ion_image, rtol=1.0e-5)
    np.testing.assert_allclose(final_internal[interior & hot_band], 3.0 * ion_image, rtol=1.0e-5)
    # (the bulk sits one ulp below the background image in floating point,
    # so the sanitize lifts it by its 2e-6 slack as well)
    np.testing.assert_allclose(final_electron[interior & bulk], electron_image, rtol=1.0e-5)
    np.testing.assert_allclose(final_ion[interior & bulk], ion_image, rtol=1.0e-5)
    print("cov: cold band lifted onto the background by the sanitize, hot band kept, no injection, ledger zero")
    sys.exit(0)
assert int(ledger[0, 1]) == 32, ledger[0]
injected_mass = 32 * (pedestal - rho_halo) * cell_size
np.testing.assert_allclose(ledger[0, 2], injected_mass, rtol=1.0e-12)

if mode == "floor":
    # One step: contact-adjacent cells may move; the band interiors carry
    # the raise's exact values.
    final_internal = field(final, "implicit_mhd_ion_internal_energy")
    # Cold band: lifted onto the image (below it at load).
    np.testing.assert_allclose(final_electron[interior & cold_band], electron_image, rtol=1.0e-9)
    np.testing.assert_allclose(final_ion[interior & cold_band], ion_image, rtol=1.0e-9)
    np.testing.assert_allclose(final_internal[interior & cold_band], ion_image, rtol=1.0e-9)
    # Hot band: max(own, image) keeps the loaded energy (3x the image) --
    # NOT cooled; this is what discriminates floor from reset.
    np.testing.assert_allclose(final_electron[interior & hot_band], 3.0 * electron_image, rtol=1.0e-9)
    np.testing.assert_allclose(final_ion[interior & hot_band], 3.0 * ion_image, rtol=1.0e-9)
    np.testing.assert_allclose(final_internal[interior & hot_band], 3.0 * ion_image, rtol=1.0e-9)
    np.testing.assert_allclose(final_electron[interior & bulk], electron_image, rtol=1.0e-9)
    # Pure source: only the cold band's deficit is booked.
    np.testing.assert_allclose(ledger[0, 3], 16 * 0.9 * electron_image * cell_size, rtol=1.0e-12)
    np.testing.assert_allclose(ledger[0, 4], 16 * 0.9 * ion_image * cell_size, rtol=1.0e-12)
    assert ledger[0, 3] > 0.0 and ledger[0, 4] > 0.0
    print("floor mode: cold band on the image, hot band kept at 3x, source-only ledger")
    sys.exit(0)

# Reset form: every halo cell sits exactly on the cold image, bulk
# untouched, and the state is invariant afterwards (zero residual).
np.testing.assert_allclose(final_electron[halo], electron_image, rtol=1.0e-13)
np.testing.assert_allclose(final_electron[bulk], electron_image, rtol=1.0e-13)
if mode == "cgl":
    np.testing.assert_allclose(final_parallel[halo], cgl_parallel_image, rtol=1.0e-13)
    np.testing.assert_allclose(final_perp[halo], cgl_perp_image, rtol=1.0e-13)
    np.testing.assert_allclose(final_parallel[bulk], cgl_parallel_image, rtol=1.0e-13)
    np.testing.assert_allclose(final_perp[bulk], cgl_perp_image, rtol=1.0e-13)
else:
    np.testing.assert_allclose(final_ion[halo], ion_image, rtol=1.0e-13)
    np.testing.assert_allclose(final_ion[bulk], ion_image, rtol=1.0e-13)
    if mode == "reset":
        # Dual-energy sync on the raised cells: E_i - |m|^2/(2 rho) == U_i
        # exactly (zero momentum, so E_i == U_i).
        final_internal = field(final, "implicit_mhd_ion_internal_energy")
        np.testing.assert_allclose(final_internal[halo], ion_image, rtol=1.0e-13)
        np.testing.assert_array_equal(final_internal[halo], final_ion[halo])

# Discrimination against the peak image (which would leave the hot band
# at 3x and lift the cold band to at most 0.3x) and the floor form (hot
# band at 3x): the hot band was COOLED onto the image.
assert np.all(final_electron[hot_band] < 0.5 * initial_electron[hot_band])
assert np.all(final_electron[cold_band] > 5.0 * initial_electron[cold_band])

# Ledger: signed energy change of the raise, per species, booked exactly
# (the raise writes the image, so the deltas are exact arithmetic).
expected_electron = cell_size * (
    16 * (electron_image - 0.1 * electron_image) + 16 * (electron_image - 3.0 * electron_image)
)
expected_ion = cell_size * (
    16 * (ion_image_booked - 0.1 * ion_image_booked)
    + 16 * (ion_image_booked - 3.0 * ion_image_booked)
)
assert expected_electron < 0.0 and expected_ion < 0.0
np.testing.assert_allclose(ledger[0, 3], expected_electron, rtol=1.0e-12)
np.testing.assert_allclose(ledger[0, 4], expected_ion, rtol=1.0e-12)
# Nothing sub-pedestal after the first raise: rows 2-5 add nothing.
assert np.all(ledger[1:, 1] == 0), ledger[:, 1]
np.testing.assert_array_equal(ledger[1:, 2:], np.tile(ledger[0, 2:], (steps - 1, 1)))
# The per-cell injected fields integrate to the ledger exactly (cell
# measure dz in 1D): mass, electron energy, ion energy.
injected_mass = field(final, "implicit_mhd_pedestal_injected_mass")
injected_electron = field(final, "implicit_mhd_pedestal_injected_electron_energy")
injected_ion = field(final, "implicit_mhd_pedestal_injected_ion_energy")
np.testing.assert_allclose(np.sum(injected_mass) * cell_size, ledger[-1, 2], rtol=1.0e-12)
np.testing.assert_allclose(np.sum(injected_electron) * cell_size, ledger[-1, 3], rtol=1.0e-12)
np.testing.assert_allclose(np.sum(injected_ion) * cell_size, ledger[-1, 4], rtol=1.0e-12)
# ... and sit on the raised cells only (the bulk was never sub-pedestal).
np.testing.assert_array_equal(injected_mass[bulk], 0.0)
np.testing.assert_allclose(injected_mass[halo], pedestal - rho_halo, rtol=1.0e-12)

# Closure against the domain: the ledger IS the total change of each
# booked block over the run (static contact afterwards).
np.testing.assert_allclose(
    ledger[-1, 2], np.sum(final_density - initial_density) * cell_size, rtol=1.0e-12
)
np.testing.assert_allclose(
    ledger[-1, 3], np.sum(final_electron - initial_electron) * cell_size, rtol=1.0e-12
)
if mode == "cgl":
    final_ion_booked = final_parallel + final_perp
else:
    final_ion_booked = final_ion
np.testing.assert_allclose(
    ledger[-1, 4], np.sum(final_ion_booked - initial_ion) * cell_size, rtol=1.0e-12
)
print(f"{mode}: raised state exact, ledger closes (mass {ledger[-1, 2]:.6e}, "
      f"U_e {ledger[-1, 3]:.6e}, ion {ledger[-1, 4]:.6e})")
