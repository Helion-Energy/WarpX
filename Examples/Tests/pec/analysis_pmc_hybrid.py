#!/usr/bin/env python3
#
# Check that the reflecting (PMC) parity holds the B_theta nodes on the
# two z faces through the hybrid solver's substepped B-field advance.

import sys

import numpy as np
import yt

ds = yt.load(sys.argv[1])
ad = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
Bt = np.asarray(ad["boxlib", "Bt"])[:, :, 0]

face = max(np.abs(Bt[:, 0]).max(), np.abs(Bt[:, -1]).max())
bulk = np.abs(Bt).max()
print(f"max|Bt|: faces {face:.3e} T, bulk {bulk:.3e} T, ratio {face / bulk:.3f}")

# the torsional mode must survive (B1 = 25 mT initially, reversed in
# sign by step 150) ...
assert bulk > 1.0e-2
# ... with its face nodes held by the boundary parity
assert face < 0.15 * bulk

# the z_lo ghost cells must carry the reflecting parity (ghost = -mirror)
# after the substepped advance, not values frozen at initialization
ghost, mirror = np.load("pmc_ghost_pair.npy")
parity_err = np.abs(ghost + mirror).max()
print(f"ghost parity: max|ghost + mirror| {parity_err:.3e} T")
assert parity_err < 0.02 * bulk
