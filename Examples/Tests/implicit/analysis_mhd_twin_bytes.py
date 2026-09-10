#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""A switched-on knob whose operand is identically zero must be the plain solver.

Compares this run's final plotfile (every field, to the last bit) and its
Newton diagnostic file (byte for byte) with a twin run without the knob.

Usage: analysis_mhd_twin_bytes.py <final_plotfile> <twin_final_plotfile> [<twin_newton_txt>]
"""

import sys
from pathlib import Path

import numpy as np
import yt

yt.set_log_level(50)

final_plotfile, twin_plotfile = sys.argv[1], sys.argv[2]
twin_newton = sys.argv[3] if len(sys.argv) > 3 else str(Path(twin_plotfile).parents[1] / "diags/newton.txt")

mine = Path("diags/newton.txt").read_bytes()
twin = Path(twin_newton).read_bytes()
print(f"newton.txt: {len(mine)} bytes, twin {len(twin)} bytes, byte-identical: {mine == twin}")
assert mine == twin


def fields(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    names = [name for kind, name in ds.field_list if kind == "boxlib"]
    return {name: np.array(data["boxlib", name].value) for name in names}


mine_fields, twin_fields = fields(final_plotfile), fields(twin_plotfile)
common = sorted(set(mine_fields) & set(twin_fields))
assert common, (set(mine_fields), set(twin_fields))
for name in common:
    same = np.array_equal(mine_fields[name], twin_fields[name])
    print(f"{name}: identical {same}")
    assert same, name
print("null twin: byte-identical to the plain run")
