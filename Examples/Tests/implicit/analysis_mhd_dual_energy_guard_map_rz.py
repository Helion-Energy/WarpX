#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# Per-cell dual-energy guard discard register on the RZ change-of-variables
# deck: the register implicit_mhd_dual_energy_guard_discard (cumulative
# J/m^3 the step-end rewrite discarded in each guarded cell) integrates
# with the RZ measure 2 pi r dr dz to the guard ledger's cumulative column
# at the last step (the ledger IS its domain sum); the guard splits the
# domain (some cells guarded, some not); one ledger row per step.
#
# Usage: analysis_mhd_dual_energy_guard_map_rz.py <final_plotfile> <ledger>

import sys

import numpy as np
import yt

final_plotfile, ledger_file = sys.argv[1:3]
steps = 8
REG = "implicit_mhd_dual_energy_guard_discard"

ds = yt.load(final_plotfile)
dims = ds.domain_dimensions
data = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)
register = np.array(data["boxlib", REG].value)[:, :, 0]
internal = np.array(data["boxlib", "implicit_mhd_ion_internal_energy"].value)[:, :, 0]
r_lo = float(ds.domain_left_edge[0].value)
r_hi = float(ds.domain_right_edge[0].value)
z_lo = float(ds.domain_left_edge[1].value)
z_hi = float(ds.domain_right_edge[1].value)
nr, nz = int(dims[0]), int(dims[1])
dr = (r_hi - r_lo) / nr
dz = (z_hi - z_lo) / nz
r = r_lo + (np.arange(nr) + 0.5) * dr
volume = (2.0 * np.pi * r * dr * dz)[:, None] * np.ones((1, nz))

ledger = np.loadtxt(ledger_file, ndmin=2)
print("guard ledger:\n", ledger)
assert ledger.shape == (steps, 3), ledger.shape
assert int(ledger[0, 0]) == 1 and int(ledger[-1, 0]) == steps
guarded = ledger[:, 1]
assert (guarded > 0).all() and (guarded < nr * nz).all(), "the guard must split the domain"

total = float(np.sum(register * volume))
booked = float(ledger[-1, 2])
print(f"register: domain integral {total:.12e} J vs ledger cumulative {booked:.12e} J; "
      f"cells with a nonzero register {int((register != 0).sum())} of {nr * nz}; "
      f"max |register| {np.abs(register).max():.6e} J/m^3")
assert np.isfinite(register).all()
np.testing.assert_allclose(total, booked, rtol=1.0e-12, atol=1.0e-300)

# The register lives only where the window is below 1 (the step-old U_i
# below 2 G); report the overlap with the end-of-run cold set.
guard = 1.0
cold = internal < 2.0 * guard
print(f"cells below 2 G at the end: {int(cold.sum())}; nonzero register outside them: "
      f"{int(((register != 0) & ~cold).sum())}")

print("dual_energy guard map (RZ): register integrates to the ledger")
