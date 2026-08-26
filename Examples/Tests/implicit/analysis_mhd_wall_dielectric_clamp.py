#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""Shaped-wall EXTERIOR CLAMP (rigid-vacuum band image) for the
theta-implicit RZ MHD recast.

The deck deliberately contaminates the wall band: uniform density rho0
(2000x the mass floor) everywhere plus an axial velocity painted outside
r = 0.31 m -- the T5 formation-ladder failure in miniature (a loaded IC
left the frozen dielectric band electrically conductive). The solver
must scrape it at first-step sanitize time and keep it scraped:

1. CLAMP IMAGE: from the first post-init snapshot on, every masked cell
   sits exactly on the fixed rigid-vacuum image -- rho = mass floor,
   u = 0, electron/ion energies on their floor-consistent values (the
   configured 1 eV temperature floors at the clamp density), each a
   fixed 2e-6 slack above its bound.
2. LEDGER HEADER: the '#'-comment header row of wall_ledger.txt books
   the scraped cell count (the full masked band) and the net mass
   removed, closing against this script's own integral of the
   diag000000 exterior with exact 2 pi r dr dz cell volumes.
3. BIT-STATIC BAND: the exterior stays BIT-IDENTICAL between the two
   post-clamp snapshots with the halo pedestal (0.05 rho0, far above
   the band density), the electron/ion temperature floors (whose
   end-of-step restorations re-land a value-derived margin -- strictly
   increasing on a bound-resident cell), and the floor-consistency
   supply all armed: every band-repopulation path must skip the mask.
4. INTERIOR UNTOUCHED: the quiet uniform interior equilibrium stays at
   its IC values to solver precision (the clamp acts outside the
   contour only).

Usage: analysis_mhd_wall_dielectric_clamp.py <diag_dir>
"""

import glob
import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(0)

diag_dir = sys.argv[1]

# Deck constants (inputs_test_rz_theta_implicit_mhd_wall_dielectric_clamp)
proton_mass = 1.67262192595e-27  # WarpX parser m_p (ablastr::constant)
qe = 1.602176634e-19
kb = 1.380649e-23
n0 = 1.0e18
rho0 = n0 * proton_mass
T0_ev = 100.0
Ti_ev = 50.0
Tmin_ev = 2.0
gamma = 5.0 / 3.0
mass_floor = 5.0e-4 * rho0
electron_pressure_floor = 1.0e-4 * n0 * qe * Tmin_ev
ion_pressure_floor = 1.0e-4 * n0 * qe * Tmin_ev
te_floor_ev = 1.0
ti_floor_ev = 1.0
SLACK = 2.0e-6

# The clamp image, exactly as ClampWallExteriorState builds it: the
# fixed slack above each admissibility bound, energies through the
# temperature-floor coefficients at the clamp density (n kB T_floor /
# (gamma - 1), quasi-neutral n = rho/m_p).
rho_image = mass_floor * (1.0 + SLACK)
n_image = rho_image / proton_mass


def energy_image(pressure_floor, t_floor_ev):
    bound = max(pressure_floor, n_image * qe * t_floor_ev) / (gamma - 1.0)
    return bound * (1.0 + SLACK)


ue_image = energy_image(electron_pressure_floor, te_floor_ev)
ei_image = energy_image(ion_pressure_floor, ti_floor_ev)  # KE = 0

FIELDS = (
    "implicit_mhd_mass_density",
    "implicit_mhd_electron_energy",
    "implicit_mhd_ion_energy",
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
assert len(plotfiles) == 3, f"need snapshots at steps 0/4/8, got {len(plotfiles)}"
ds0, state_ic = load_state(plotfiles[0])
_, state_mid = load_state(plotfiles[1])
_, state_end = load_state(plotfiles[2])

nr, nz = int(ds0.domain_dimensions[0]), int(ds0.domain_dimensions[1])
dr = (float(ds0.domain_right_edge[0]) - float(ds0.domain_left_edge[0])) / nr
dz = (float(ds0.domain_right_edge[1]) - float(ds0.domain_left_edge[1])) / nz
r_centers = float(ds0.domain_left_edge[0]) + (np.arange(nr) + 0.5) * dr
z_centers = float(ds0.domain_left_edge[1]) + (np.arange(nz) + 0.5) * dz

# The mask, replicated exactly as ImplicitMHDWallMask builds it.
wall_radius = np.where(z_centers < 0.0, 0.30, 0.22)
masked = (r_centers[:, None] + 1.0e-3 * dr) >= wall_radius[None, :]
interior = ~masked
assert masked.any() and interior.any(), "degenerate mask layout"

for name, field in state_end.items():
    assert np.isfinite(field).all(), f"{name} has non-finite values"

# --- 0. the IC really contaminated the band (the teeth) ------------------
ic_band_max_rho = np.max(state_ic["implicit_mhd_mass_density"], where=masked,
                         initial=0.0)
ic_band_max_uz = np.max(
    np.abs(state_ic["implicit_mhd_momentum_z"]), where=masked, initial=0.0
)
print(f"IC band: max rho {ic_band_max_rho:.3e} (floor {mass_floor:.3e}), "
      f"max |mom_z| {ic_band_max_uz:.3e}")
assert ic_band_max_rho > 100.0 * mass_floor, "IC band not contaminated"
assert ic_band_max_uz > 0.0, "IC band carries no momentum contamination"

# --- 1. post-clamp exterior == the rigid vacuum image --------------------
image = {
    "implicit_mhd_mass_density": rho_image,
    "implicit_mhd_electron_energy": ue_image,
    "implicit_mhd_ion_energy": ei_image,
    "implicit_mhd_momentum_r": 0.0,
    "implicit_mhd_momentum_t": 0.0,
    "implicit_mhd_momentum_z": 0.0,
}
for f, value in image.items():
    scale = max(abs(value), np.max(np.abs(state_mid[f])), 1.0e-300)
    err = np.max(np.abs(state_mid[f] - value), where=masked, initial=0.0)
    print(f"clamp image mismatch {f}: {err / scale:.3e} of scale")
    assert err / scale < 1.0e-13, (
        f"exterior not on the clamp image ({f}: {err:.3e}, expected {value:.6e})"
    )

# --- 2. ledger header closes against the measured scrape -----------------
with open(f"{diag_dir}/wall_ledger.txt", encoding="utf-8") as ledger:
    header = ledger.readline().strip()
print(f"ledger header: {header}")
assert header.startswith("# exterior clamp:"), "missing clamp header row"
tokens = header.split()
cells_booked = int(tokens[4])
mass_booked = float(tokens[6])
cells_expected = int(np.count_nonzero(masked))
volume = 2.0 * np.pi * r_centers[:, None] * dr * dz
mass_expected = float(
    np.sum(
        (state_ic["implicit_mhd_mass_density"] - rho_image) * volume,
        where=masked,
    )
)
print(f"clamp: booked {cells_booked} cells / {mass_booked:.6e} kg, "
      f"expected {cells_expected} cells / {mass_expected:.6e} kg")
assert cells_booked == cells_expected, "clamped-cell count mismatch"
assert abs(mass_booked - mass_expected) < 1.0e-10 * abs(mass_expected), (
    "scraped-mass booking does not close"
)
# The remaining ledger rows must keep the frozen-ion-free live fluid's
# deposition bookkeeping intact (numeric rows still parse).
ledger_rows = np.loadtxt(f"{diag_dir}/wall_ledger.txt", ndmin=2)
assert ledger_rows.shape[0] >= 1, "ledger has no numeric rows"

# --- 3. exterior BIT-IDENTICAL across post-clamp steps -------------------
for f in FIELDS:
    same = np.array_equal(
        np.where(masked, state_end[f], 0.0), np.where(masked, state_mid[f], 0.0)
    )
    drift = np.max(
        np.abs(state_end[f] - state_mid[f]), where=masked, initial=0.0
    )
    print(f"band drift {f}: {drift:.3e} (bitwise match: {same})")
    assert same, (
        f"exterior not bit-static ({f}: {drift:.3e}) -- a floor/pedestal "
        "path is re-creating state outside the wall"
    )

# --- 4. interior untouched by the clamp ----------------------------------
# Physics scales (the interior momenta are IDENTICALLY zero in the IC,
# so a max-|IC| scale would divide roundoff noise by zero): the
# acoustic momentum scale rho0 c_s for the momenta.
sound_speed = np.sqrt(gamma * n0 * qe * (T0_ev + Ti_ev) / rho0)
interior_scale = {
    "implicit_mhd_mass_density": rho0,
    "implicit_mhd_electron_energy": n0 * qe * T0_ev / (gamma - 1.0),
    "implicit_mhd_ion_energy": n0 * qe * Ti_ev / (gamma - 1.0),
    "implicit_mhd_momentum_r": rho0 * sound_speed,
    "implicit_mhd_momentum_t": rho0 * sound_speed,
    "implicit_mhd_momentum_z": rho0 * sound_speed,
}
for f in FIELDS:
    err = np.max(
        np.abs(state_mid[f] - state_ic[f]), where=interior, initial=0.0
    ) / interior_scale[f]
    print(f"interior drift {f}: {err:.3e}")
    assert err < 1.0e-8, f"interior state moved ({f}: {err:.3e})"

print("shaped-wall exterior clamp: all checks passed")
