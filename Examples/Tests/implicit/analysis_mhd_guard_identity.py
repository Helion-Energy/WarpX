#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Off-plateau identity of the local penalty guard.

The guard raises the central flux's Rusanov coefficient to
c_face = c + (1 - c) S with S the max over the two cells of three
C-infinity switches. Its density switch is an exact unit step in
log10(n_old / n_g): exactly 1 at or below n_g, exactly 0 at or above
n_g 10^width. On a deck whose density never falls below n_g 10^width the
switch is therefore exactly zero on every face, c_face = c + 0 is c bit
for bit, and the guarded run must reproduce the unguarded twin BITWISE --
not to a tolerance: the guard sits on the JFNK residual hot path of every
production deck at c < 1, and this is the statement that a guard whose
arguments stay on the off plateau is no guard at all. The guard ledger of
the same run must then book zero guarded faces and zero transport in
every row.

Usage: analysis_mhd_guard_identity.py <final_plotfile> <twin_final_plotfile> [<guard_ledger>]
"""

import sys

import numpy as np
import yt

yt.set_log_level(50)


def get_fields(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    names = sorted(name for kind, name in ds.field_list if kind == "boxlib")
    return {name: np.asarray(data["boxlib", name]).ravel().copy() for name in names}


guarded = get_fields(sys.argv[1])
twin = get_fields(sys.argv[2])
assert set(guarded) == set(twin), (sorted(guarded), sorted(twin))
assert len(guarded) >= 2, "the plotfile publishes fewer than two fields"
for name in sorted(guarded):
    identical = np.array_equal(guarded[name], twin[name])
    print(f"  {name:35s} bitwise identical: {identical}")
    assert identical, f"{name} differs between the guarded run and its twin"

if len(sys.argv) > 3:
    rows = [
        line.split()
        for line in open(sys.argv[3])
        if line.strip() and not line.startswith("#")
    ]
    assert rows, "the guard ledger has no rows"
    table = np.array(rows, dtype=float)
    print(f"guard ledger: {table.shape[0]} rows, max guarded faces "
          f"{table[:, 1].max():.0f}, max |transport| "
          f"{np.abs(table[:, 2:]).max():.3e}")
    assert np.all(table[:, 1] == 0.0), "the ledger booked guarded faces"
    assert np.all(table[:, 2:] == 0.0), "the ledger booked guard transport"
print("PASS: the guard on its off plateau is bitwise no guard")
