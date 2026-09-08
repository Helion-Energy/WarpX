#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Fused elementwise vector operations must be bit-identical.

implicit_evolve.fused_vector_ops = 1 runs every elementwise solver-vector
operation (copy, increment, linear combination, scale, set, pack/unpack)
as one kernel over the fused layout of the state instead of one kernel per
MultiFab, with the per-element expressions unchanged. This run and its
twin (same deck, fused_vector_ops = 0) must therefore agree exactly: the
Newton diagnostic file is byte-identical and the final plotfile fields are
identical to the last bit.

Usage: analysis_mhd_fused_identity.py <final_plotfile> <twin_final_plotfile>
       [<twin_newton_txt>]
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
match = re.search(r"^implicit_evolve\.fused_vector_ops\s*=\s*(\S+)", used_inputs, re.MULTILINE)
assert match is not None and match.group(1) == "1", match

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
assert set(mine_fields) == set(twin_fields), (set(mine_fields), set(twin_fields))
for name in sorted(mine_fields):
    identical = np.array_equal(mine_fields[name], twin_fields[name])
    print(f"{name:40s} identical: {identical}")
    assert identical, name
print("fused elementwise operations: bit-identical PASS")
