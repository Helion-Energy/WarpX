#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""Shaped-wall FIELD freeze no-op twin (implicit_mhd.wall_field_freeze)
-- work item B, T-B2.

With no external drive there are no band currents for the freeze to
kill: under the pec wall of the thermal deck every masked E row is
already projected to exactly zero, so the Faraday rhs of every frozen
magnetic-field face is exactly zero BEFORE the freeze zeroes it, the
frozen rows were already exact identities, and the whole run --
interior conduction-drain physics included -- must be BIT-IDENTICAL
with the freeze on vs off. Any mismatch means the freeze touched a
live row (the interior-physics regression this test guards).

Compares every common plotfile of the freeze-on run against the
freeze-off reference (test_rz_theta_implicit_mhd_wall_thermal_zero_flux),
every plotted field, bitwise (np.array_equal).

Usage:
  analysis_mhd_wall_field_freeze_noop.py <diags_dir> <reference_diags_dir>
"""

import glob
import os
import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(0)


def load_all_fields(plotfile):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    names = sorted(name for kind, name in ds.field_list if kind == "boxlib")
    return {name: np.squeeze(grid["boxlib", name].value) for name in names}


diags_dir, reference_dir = sys.argv[1], sys.argv[2]
plotfiles = sorted(glob.glob(os.path.join(diags_dir, "diag" + "[0-9]" * 6)))
assert len(plotfiles) >= 2, f"expected at least 2 plotfiles in {diags_dir}"

for plotfile in plotfiles:
    reference_plotfile = os.path.join(
        reference_dir, os.path.basename(plotfile)
    )
    assert os.path.isdir(reference_plotfile), (
        f"reference snapshot {reference_plotfile} is missing"
    )
    state = load_all_fields(plotfile)
    reference = load_all_fields(reference_plotfile)
    assert state.keys() == reference.keys(), "plotted field sets differ"
    for name in state:
        equal = np.array_equal(state[name], reference[name])
        difference = (
            0.0
            if equal
            else float(np.max(np.abs(state[name] - reference[name])))
        )
        print(f"{os.path.basename(plotfile)}: {name} bitwise "
              f"{'EQUAL' if equal else f'DIFFERS (max {difference:.3e})'}")
        assert equal, (
            f"{os.path.basename(plotfile)}/{name}: wall_field_freeze "
            "changed a quiescent no-drive run (the freeze must be an "
            "exact no-op when the band carries no currents)"
        )

print("shaped-wall field freeze no-op test PASSED")
