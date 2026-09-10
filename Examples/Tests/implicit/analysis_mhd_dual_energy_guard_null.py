#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# Null test of the absolute low-internal-energy guard: with the guard set
# far BELOW every internal energy in the domain the window is exactly 1
# everywhere and fk x 1.0 is bit-preserving, so the run must reproduce the
# unguarded dual_energy_stagnation run BIT FOR BIT (every fluid block and
# the field), not merely to round-off.

import sys

import numpy as np
import yt

fields = (
    "implicit_mhd_mass_density",
    "implicit_mhd_momentum_z",
    "implicit_mhd_electron_energy",
    "implicit_mhd_ion_energy",
    "implicit_mhd_ion_internal_energy",
    "Bz",
)


def load(plotfile):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return {field: grid["boxlib", field].value for field in fields}


guarded = load(sys.argv[1])
base = load(sys.argv[2])
for field in fields:
    same = np.array_equal(guarded[field], base[field])
    print(f"guard null {field}: bitwise identical = {same}")
    assert same, f"{field} differs with an inactive guard"
print("dual_energy guard null: PASS (bitwise)")
