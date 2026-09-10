#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""RZ gates of the COLD halo pedestal image on a dynamic deck
(inputs_test_rz_theta_implicit_mhd_halo_pedestal_cold; also run under the
production solver stack).

  1. Ledger closure. Periodic z, reflecting outer wall, no wall drain:
     the domain mass changes only through the pedestal refresh, so the
     ledger's cumulative injected mass must equal the domain mass change
     between the first and last plotfile (RZ measure 2 pi r dr dz) to the
     nonlinear solver tolerance.
  2. Signs. The halo is loaded hot and sparse (300 eV at 1e-5 rho0):
     n_halo T_halo = 3e-3 n0 eV > n_ped T_ped_e = 2e-3 n0 eV, so the
     electron ledger is NEGATIVE (the reset raise removes heat), while
     n_ped T_ped_i = 2e-2 n0 eV > 3e-3 n0 eV makes the ion ledger
     POSITIVE.
  3. The cold band. At the last step the cells riding the pedestal
     (rho <= 1.1 f_ped max rho) must be the majority of the domain and
     their mass-weighted electron / ion temperatures must sit within 25 %
     of the cold image (2 eV / 20 eV). The peak image would put the band
     at the core temperature (~50 eV for both species), 25x / 2.5x off.
  4. One ledger row per step, raised-cell count > 0 on the first row.

Usage: analysis_mhd_halo_pedestal_cold_rz.py <initial_plotfile> <final_plotfile> <ledger>
"""

import sys

import numpy as np
import warpx_constants as constants
import yt

yt.set_log_level(50)

initial_plotfile, final_plotfile, ledger_file = sys.argv[1:4]
# Optional: --static <n_static m^-3> (the pedestal density is then fixed, not f max n) and --floor (floor raise: source-only
# bookings; the hot loaded electrons are kept so the band's electron gate is skipped).
static_density = float(sys.argv[sys.argv.index("--static") + 1]) if "--static" in sys.argv else None

n0 = 1.0e19
rho0 = n0 * constants.proton_mass
gamma = 5.0 / 3.0
f_ped = 1.0e-3
T_ped_e = float(sys.argv[sys.argv.index("--te") + 1]) if "--te" in sys.argv else 2.0
T_ped_i = float(sys.argv[sys.argv.index("--ti") + 1]) if "--ti" in sys.argv else 20.0
steps = 8
charge_to_mass = constants.elementary_charge / constants.proton_mass


FIELDS = (
    "implicit_mhd_mass_density",
    "implicit_mhd_electron_energy",
    "implicit_mhd_ion_internal_energy",
    "implicit_mhd_pedestal_injected_mass",
    "implicit_mhd_pedestal_injected_electron_energy",
    "implicit_mhd_pedestal_injected_ion_energy",
)


def load(plotfile):
    """Fields as (nr, nz) arrays plus the RZ cell volumes 2 pi r dr dz."""
    ds = yt.load(plotfile)
    dims = ds.domain_dimensions
    data = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)
    fields = {name: np.array(data["boxlib", name].value)[:, :, 0] for name in FIELDS}
    r_lo = float(ds.domain_left_edge[0].value)
    r_hi = float(ds.domain_right_edge[0].value)
    z_lo = float(ds.domain_left_edge[1].value)
    z_hi = float(ds.domain_right_edge[1].value)
    nr, nz = int(dims[0]), int(dims[1])
    dr = (r_hi - r_lo) / nr
    dz = (z_hi - z_lo) / nz
    r = r_lo + (np.arange(nr) + 0.5) * dr
    volume = (2.0 * np.pi * r * dr * dz)[:, None] * np.ones((1, nz))
    return fields, volume


initial, volume = load(initial_plotfile)
final, volume_final = load(final_plotfile)
np.testing.assert_allclose(volume_final, volume)

rho_i = initial["implicit_mhd_mass_density"]
rho_f = final["implicit_mhd_mass_density"]
ue_f = final["implicit_mhd_electron_energy"]
ui_f = final["implicit_mhd_ion_internal_energy"]

ledger = np.loadtxt(ledger_file, ndmin=2)
print("ledger:\n", ledger)
assert ledger.shape == (steps, 5), ledger.shape
assert int(ledger[0, 0]) == 1 and int(ledger[-1, 0]) == steps
assert ledger[0, 1] > 0, ledger[0]
# The first refresh raises the whole sub-pedestal halo (78 % of the
# volume, 2/3 of the cells).
assert ledger[0, 1] >= 0.5 * rho_i.size, (ledger[0, 1], rho_i.size)

# 1. Mass closure: domain mass change == booked injected mass.
mass_change = np.sum((rho_f - rho_i) * volume)
booked = ledger[-1, 2]
closure = abs(mass_change - booked) / abs(booked)
print(f"mass change {mass_change:.9e} kg, booked {booked:.9e} kg, closure {closure:.3e}")
assert booked > 0.0
assert closure < 1.0e-6, closure

# 1b. The per-cell injected fields integrate (RZ measure) to the ledger
# columns exactly: the ledger IS their domain sum.
for name, col in (("implicit_mhd_pedestal_injected_mass", 2),
                  ("implicit_mhd_pedestal_injected_electron_energy", 3),
                  ("implicit_mhd_pedestal_injected_ion_energy", 4)):
    total = np.sum(final[name] * volume)
    print(f"{name}: domain sum {total:.9e} vs ledger {ledger[-1, col]:.9e}")
    np.testing.assert_allclose(total, ledger[-1, col], rtol=1.0e-12)

# 2. Signs of the energy bookings (reset form on a hot sparse halo).
print(f"energy ledger: electrons {ledger[-1, 3]:.6e} J, ions {ledger[-1, 4]:.6e} J")
floor_raise = "--floor" in sys.argv
if floor_raise:
    # floor form: max(own, image) -- pure source, every booking >= 0 (the
    # hot electrons are kept, the cold ions lifted).
    assert ledger[-1, 3] >= 0.0 and ledger[-1, 4] > 0.0, ledger[-1, 3:5]
else:
    assert ledger[-1, 3] < 0.0, ledger[-1, 3]
    assert ledger[-1, 4] > 0.0, ledger[-1, 4]

# 3. The cold band rides near the cold image. The band cells next to the
# column edge are heated by the capped perpendicular conduction from the
# hot edge (a few cells over 8 steps), so the MEDIAN is the statistic of
# the raised state; the mass-weighted mean must still sit far below the
# peak image (the core temperature).
if static_density is not None:
    pedestal = static_density * constants.proton_mass
    # Discrimination: the fraction-keyed pedestal would be a third of this.
    assert abs(pedestal - f_ped * rho_f.max()) > 0.4 * pedestal
else:
    pedestal = f_ped * rho_f.max()
band = rho_f <= 1.1 * pedestal
print(f"band cells {band.sum()} of {band.size}")
assert band.sum() >= 0.4 * band.size
n_band = rho_f[band] / constants.proton_mass
te_band = (gamma - 1.0) * ue_f[band] / (n_band * constants.elementary_charge)
ti_band = (gamma - 1.0) * ui_f[band] / (n_band * constants.elementary_charge)
weights = rho_f[band] * volume[band]
te_mean = np.sum(weights * te_band) / np.sum(weights)
ti_mean = np.sum(weights * ti_band) / np.sum(weights)
te_median = np.median(te_band)
ti_median = np.median(ti_band)
print(f"band <Te>_m {te_mean:.4f} eV, median {te_median:.4f} (image {T_ped_e}); "
      f"<Ti>_m {ti_mean:.4f} eV, median {ti_median:.4f} (image {T_ped_i})")
print(f"band Te range {te_band.min():.4f}..{te_band.max():.4f}, Ti range {ti_band.min():.4f}..{ti_band.max():.4f}")
if not floor_raise:
    assert abs(te_median / T_ped_e - 1.0) < 0.25, te_median
    # Far from the peak image (the core temperature, 50 eV) even on average.
    assert te_mean < 0.3 * 50.0, te_mean
    # The raise lands exactly on the image: the coldest band cells sit on it.
    np.testing.assert_allclose(te_band.min(), T_ped_e, rtol=1.0e-6)
assert abs(ti_median / T_ped_i - 1.0) < 0.25, ti_median
# The coldest band cells sit on the ion image: exactly where the pedestal
# rises every step (re-raise at the start of the last step), within the
# 8-step conductive drift (~1 %) when the pedestal is static (no re-raise
# after the first step).
np.testing.assert_allclose(ti_band.min(), T_ped_i, rtol=(5.0e-2 if static_density is not None else 1.0e-6))
print("cold pedestal image (RZ): ledger closes, signs right, band cold")
