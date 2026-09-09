#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""The change-of-variables pedestal with a ZERO background is the plain solver.

implicit_mhd.pedestal_fraction > 0 with implicit_mhd.pedestal_reference_density
= 0 arms every branch of the change of variables (the flux shifts, the
pressure-work shifts, the background floors, the per-step refresh) with a zero
background: every shift is exactly 0.0 and every floor is the positivity guard,
so the run must be BYTE-IDENTICAL to the plain twin -- Newton diagnostic file
byte-equal, plotfile fields identical to the last bit. This gates the
"exactly zero when off" construction of every shift at once.

Usage: analysis_mhd_pedestal_cov_null.py <final_plotfile> <twin_final_plotfile> [<twin_newton_txt>]
"""

import re
import sys
from pathlib import Path

import numpy as np
import yt

yt.set_log_level(50)

final_plotfile = sys.argv[1]
twin_plotfile = sys.argv[2]
twin_newton = (
    sys.argv[3] if len(sys.argv) > 3 else str(Path(twin_plotfile).parents[1] / "diags/newton.txt")
)

used_inputs = Path("warpx_used_inputs").read_text()
fraction = re.search(r"^implicit_mhd\.pedestal_fraction\s*=\s*(\S+)", used_inputs, re.MULTILINE)
reference = re.search(
    r"^implicit_mhd\.pedestal_reference_density\s*=\s*(\S+)", used_inputs, re.MULTILINE
)
assert fraction is not None and float(fraction.group(1)) > 0.0, fraction
assert reference is not None and float(reference.group(1)) == 0.0, reference

mine = Path("diags/newton.txt").read_bytes()
twin = Path(twin_newton).read_bytes()
print(f"newton.txt: {len(mine)} bytes, twin {len(twin)} bytes, byte-identical: {mine == twin}")
assert mine == twin


def fields(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    names = [name for kind, name in ds.field_list if kind == "boxlib"]
    return {name: np.array(data["boxlib", name].value) for name in names}


mine_fields = fields(final_plotfile)
twin_fields = fields(twin_plotfile)
common = sorted(set(mine_fields) & set(twin_fields))
assert common, (set(mine_fields), set(twin_fields))
for name in common:
    same = np.array_equal(mine_fields[name], twin_fields[name])
    print(f"{name}: identical {same}")
    assert same, name
print("change-of-variables null: byte-identical to the plain twin")
