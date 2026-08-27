#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# Restart round-trip of the dual-energy closure: the run restarted from
# the mid-run checkpoint must reproduce the uninterrupted run's final
# state -- including the auxiliary ion internal-energy block U_i, which
# is checkpointed like every other fluid block -- to round-off.

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
    # extract while the dataset is alive (the covering grid holds only a
    # weak reference to it)
    return {field: grid["boxlib", field].value for field in fields}


restarted = load(sys.argv[1])
uninterrupted = load(sys.argv[2])

for field in fields:
    a = restarted[field]
    b = uninterrupted[field]
    scale = np.abs(b).max()
    difference = np.abs(a - b).max()
    print(
        f"restart round-trip {field}: max |diff| = {difference:.3e} (scale {scale:.3e})"
    )
    assert difference <= 1.0e-10 * scale, (
        f"{field} failed the dual-energy restart round-trip"
    )

print("dual_energy restart round-trip: PASS")
