#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Default-path identity of implicit_mhd.central_dissipation (production flux).

The reconstruction is a NEW branch inside the recast face-flux kernel,
on the JFNK residual hot path of every production deck. Its default
("none") must therefore leave that path untouched to the last bit, not
merely to a tolerance: the entire existing regression suite and every
committed checksum depend on it.

Two arms of the same base deck, differing only in whether the knob is
written at all:

  * the base deck with NO implicit_mhd.central_dissipation line, and
  * the same deck with implicit_mhd.central_dissipation = 0.

Every published field of the final plotfile must agree BITWISE. That is
a stronger statement than "the branch is skipped": it also gates the
parse path (a mis-parsed knob that silently selected a limiter, or a
default that drifted away from none, fails here) and the new axis ghost
fill of the cell-centered current, which is deliberately gated on the
reconstruction being active for exactly this reason.

Usage: analysis_mhd_reconstruction_identity.py <final>
       (the default-line twin is read from
        ../test_1d_theta_implicit_mhd_central_dissipation_identity_default)
"""

import sys

import numpy as np
import yt

yt.set_log_level(50)

TWIN = "../test_1d_theta_implicit_mhd_central_dissipation_identity_default/diags/diag000030"
# The order deck publishes density and electron energy (it is the smooth
# sine rig; see the CMake comment for why the identity runs there rather
# than on the square-contact deck).
FIELDS = (
    "implicit_mhd_mass_density",
    "implicit_mhd_electron_energy",
)


def get_fields(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return {name: data["boxlib", name].value.ravel().copy() for name in FIELDS}


explicit_none = get_fields(sys.argv[1])
default = get_fields(TWIN)

for name in FIELDS:
    left = explicit_none[name]
    right = default[name]
    assert left.shape == right.shape, name
    # Bitwise: no tolerance, no allclose.
    identical = np.array_equal(left, right)
    if not identical:
        difference = np.max(np.abs(left - right))
        raise AssertionError(
            f"{name} is not bit-identical between the default and the "
            f"explicit central_dissipation = 0 arm "
            f"(max |diff| = {difference:.6e})"
        )
    assert np.all(np.isfinite(left)), name
    print(f"{name:36s} bit-identical over {left.size} cells")

# The run must actually have DONE something (a dead deck would pass the
# identity vacuously).
assert np.std(explicit_none["implicit_mhd_mass_density"]) > 0.0
