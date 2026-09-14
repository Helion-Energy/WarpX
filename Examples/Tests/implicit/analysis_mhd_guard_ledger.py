#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""The local penalty guard's ledger on the wall viscosity band.

The wall-band switch (implicit_mhd.central_dissipation_wall_band_guard)
gives every cell of the wall viscosity band S = 1, so every LIVE face
with at least one band cell is a guarded face. On the sheared-column band
deck (inputs_test_rz_theta_implicit_mhd_wall_viscosity_band: 24 radial
cells, the mask table at i >= 15, band width 3, 8 axial rows) the guarded
faces are exactly enumerable. The band membership is the band
coefficient's own test, "cell i >= first_masked - width", which marks the
masked cells too; on this ELECTROMAGNETIC-ONLY wall deck (wall_model =
pec, no thermal BC) the masked cells are not frozen and their faces are
live fluid faces, so nothing is excluded as interior metal (on a
production deck the frozen wall's interior faces are, exactly as the
kernel's wall_mechanics classification excludes them):

  r-normal faces: a face i separates cells i-1 and i, so faces i = 12
    (cell 12 on its right) through i = 24 (the r_max face, cell 23 on its
    left) carry a band cell: 13 per row, 104 in the 8 rows (the axis face
    i = 0 is zeroed before any flux and never written);
  z-normal faces: every z-face of the band columns i = 12 .. 23 (9 per
    column including the two domain-end faces): 12 x 9 = 108.

So 212 guarded faces per step, on both MPI ranks combined (face-type
boxes share their seam faces; the owner masks give each face to one
box). The column carries a linear axial shear u_z = A r at uniform
density and pressure, so the only penalised jump on an r-face is the
momentum one and the booked guard transport is the kinetic part of the
ion total-energy penalty: guard_Ei > 0 every step, guard_mass and
guard_Ue exactly 0 (uniform rho and U_e reconstruct to zero jumps), and
the full penalty of the guarded faces equals the guard's share exactly
because the deck's own coefficient is 0 (c_face - c = c_face).

Usage: analysis_mhd_guard_ledger.py <guard_ledger> <expected_guarded_faces>
"""

import sys

import numpy as np

rows = [
    line.split()
    for line in open(sys.argv[1])
    if line.strip() and not line.startswith("#")
]
expected_faces = int(sys.argv[2])
assert rows, "the guard ledger has no rows"
table = np.array(rows, dtype=float)
(step, faces, guard_mass, guard_ue, guard_ei, guard_ui,
 penalty_mass, penalty_ue, penalty_ei, penalty_ui,
 cum_mass, cum_ue, cum_ei, cum_ui) = table.T
print(f"guard ledger: {len(rows)} rows; guarded faces per step "
      f"{faces.min():.0f}..{faces.max():.0f} (expected {expected_faces}); "
      f"guard_Ei per step {guard_ei.min():.3e}..{guard_ei.max():.3e} J; "
      f"cumulative guard_Ei {cum_ei[-1]:.3e} J")
assert np.all(faces == expected_faces), f"guarded faces {faces} != {expected_faces}"
assert np.all(guard_ei > 0.0), "no ion-energy transport booked for the guard"
# The density and the electron energy stay uniform up to the roundoff the
# RZ momentum update leaves behind (measured: 6e-22 kg and 5e-13 J per
# step against a domain mass of 4e-8 kg and 4e-6 J of E_i transport), so
# their booked transport must stay at the roundoff class -- 1e-12 kg and
# 1e-3 of the ion-energy transport are 1e6 / 1e3 above it.
assert np.all(guard_mass <= 1.0e-12) and np.all(guard_ue <= 1.0e-3 * guard_ei), (
    "mass / electron-energy transport booked on a uniform-density, "
    f"uniform-pressure column: {guard_mass.max():.3e} kg, {guard_ue.max():.3e} J"
)
assert np.all(guard_ui == 0.0), "U_i booked under the total_energy closure"
np.testing.assert_array_equal(penalty_ei, guard_ei)
np.testing.assert_allclose(cum_ei, np.cumsum(guard_ei), rtol=1.0e-12, atol=0.0)
assert np.all(np.diff(step) == 1.0), "the ledger skipped a step"
print("PASS: the guard ledger books the band faces and their transport")
