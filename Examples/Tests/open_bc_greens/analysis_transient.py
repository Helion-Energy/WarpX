#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""Temporal-coupling guard of the RZ Green's-function open boundary.

The Green's ghost fill is an instantaneous linear constraint,
psi_ghost = K . J_theta[B], and must be evaluated INSIDE every RK stage of
the subcycled B integrator (part of the right-hand side), per the design.
Then the semi-discrete problem is independent of the substep count, and two
runs of the same mid-transient diffusion problem with different substep
counts agree to the integrator's order (measured ~3e-13 on this setup).
The forbidden lagged variant (ghost values frozen over each substep, which
makes the boundary partially reflecting for wave-dominated fields) makes
the transient substep-dependent at O(dt_sub): measured ~4e-6 on this
setup, seven orders of magnitude above per-stage. The assert is placed
between the two.

Arguments: plotfile of the 80-substep run, plotfile of the 20-substep run.
"""

import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(0)


def load_B(fn):
    ds = yt.load(fn)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return (
        np.stack([grid["boxlib", c].v.squeeze() for c in ("Br", "Bt", "Bz")], axis=0),
        int(ds.domain_dimensions[0]),
    )


B_80, nr = load_B(sys.argv[1])
B_20, _ = load_B(sys.argv[2])

scale = np.abs(B_80).max()
diff = np.abs(B_80 - B_20) / scale

# the boundary-lag signature is largest at the open face: check the
# outermost cells separately from the bulk
diff_bulk = diff.max()
diff_wall = diff[:, nr - 2 :, :].max()
print(f"substep-invariance of the transient: bulk = {diff_bulk:.3e}")
print(f"substep-invariance of the transient: wall = {diff_wall:.3e}")

# per-stage ghost fill measured ~3e-13; lagged (once-per-substep) fill
# measured ~4e-6; assert well inside the gap
assert diff_bulk < 1.0e-7, f"transient depends on substep count: {diff_bulk:.3e}"
assert diff_wall < 1.0e-7, (
    f"boundary transient depends on substep count: {diff_wall:.3e} "
    "(is the Green's ghost fill still applied inside every RK stage?)"
)

print("Transient substep-invariance test PASSED")
