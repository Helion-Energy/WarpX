#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Bit-identity twin check for implicit_mhd.wall_heat_flux_cap.

The knob is an OVERRIDE of the per-mode wall-cap defaults, and its
contract is that every existing configuration is reproduced bit for bit:

  * wall_heat_flux_cap unset  == wall_heat_flux_cap = none under the
    hard dirichlet pin (both uncapped) and under z_wall_conduction;
  * wall_heat_flux_cap = free_streaming on the hard dirichlet pin ==
    wall_thermal_bc = dirichlet_limited with the knob unset (the
    existing free-streaming wall cap with the legacy factor).

This script compares one run's diagnostics against a reference run's:
the Newton diagnostic file (every column, exactly), every field of
every plotfile pair (np.array_equal, so NaN-free exact equality), and
the shaped-wall ledger when both runs wrote one. Any difference fails.

Usage: analysis_mhd_wall_heat_flux_cap_identity.py <diag_dir> <reference_diag_dir>
"""

import glob
import os
import sys

import numpy as np
import yt

yt.set_log_level(50)

diag_dir = sys.argv[1]
reference_dir = sys.argv[2]

failures = []


def last_run(rows):
    """The solver APPENDS its per-step diagnostic tables when a work
    directory is reused (a ctest rerun), so compare the LAST run's rows:
    from the last row whose step column reads 1."""
    starts = np.flatnonzero(rows[:, 0] == 1)
    return rows[starts[-1]:] if len(starts) else rows


def compare_text_table(name):
    mine = os.path.join(diag_dir, name)
    theirs = os.path.join(reference_dir, name)
    have = os.path.exists(mine), os.path.exists(theirs)
    if have != (True, True):
        if have == (False, False):
            print(f"{name}: absent in both runs (skipped)")
            return
        failures.append(f"{name}: present in only one run {have}")
        return
    a = last_run(np.atleast_2d(np.loadtxt(mine)))
    b = last_run(np.atleast_2d(np.loadtxt(theirs)))
    if a.shape != b.shape:
        failures.append(f"{name}: shape {a.shape} vs {b.shape}")
        return
    if not np.array_equal(a, b):
        failures.append(
            f"{name}: max abs difference {np.max(np.abs(a - b)):.3e}"
        )
    print(f"{name}: {a.shape[0]} rows, identical = {np.array_equal(a, b)}")


compare_text_table("newton.txt")
compare_text_table("wall_ledger.txt")

mine = sorted(glob.glob(f"{diag_dir}/diag??????"))
theirs = sorted(glob.glob(f"{reference_dir}/diag??????"))
assert len(mine) >= 2, f"need at least 2 plotfiles, got {len(mine)}"
if [os.path.basename(p) for p in mine] != [os.path.basename(p) for p in theirs]:
    failures.append(f"plotfile sets differ: {mine} vs {theirs}")
else:
    for a_path, b_path in zip(mine, theirs):
        ds_a = yt.load(a_path)
        ds_b = yt.load(b_path)
        assert (ds_a.domain_dimensions == ds_b.domain_dimensions).all()
        grid_a = ds_a.covering_grid(
            level=0, left_edge=ds_a.domain_left_edge, dims=ds_a.domain_dimensions
        )
        grid_b = ds_b.covering_grid(
            level=0, left_edge=ds_b.domain_left_edge, dims=ds_b.domain_dimensions
        )
        fields_a = sorted(f for t, f in ds_a.field_list if t == "boxlib")
        fields_b = sorted(f for t, f in ds_b.field_list if t == "boxlib")
        if fields_a != fields_b:
            failures.append(f"{a_path}: field lists differ")
            continue
        for f in fields_a:
            a = np.asarray(grid_a["boxlib", f].value)
            b = np.asarray(grid_b["boxlib", f].value)
            if not np.array_equal(a, b):
                failures.append(
                    f"{os.path.basename(a_path)}/{f}: max abs difference "
                    f"{np.max(np.abs(a - b)):.3e}"
                )
        print(f"{os.path.basename(a_path)}: {len(fields_a)} fields compared")

if failures:
    print("BIT-IDENTITY FAILURES:")
    for line in failures:
        print("  " + line)
    sys.exit(1)

print("wall_heat_flux_cap identity twin: bit-identical to the reference run")
