#!/usr/bin/env python3
#
# A uniform axial field is exact under the open (zero-gradient) boundary,
# so the field must stay uniform: a face-localized deviation is a
# regression in the ghost continuation.

import sys

import numpy as np
import yt

B0 = 0.5

ds = yt.load(sys.argv[1])
ad = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
Bz = np.asarray(ad["boxlib", "Bz"])[:, :, 0]
Br = np.asarray(ad["boxlib", "Br"])[:, :, 0]
Bt = np.asarray(ad["boxlib", "Bt"])[:, :, 0]

dBz = np.abs(Bz - B0)
face = dBz[:, [0, 1, -2, -1]].max()
interior = dBz[:, 8:-8].max()
perp = max(np.abs(Br).max(), np.abs(Bt).max())
print(
    f"|Bz - B0|: faces {face:.3e} T, interior {interior:.3e} T; max|Bperp| {perp:.3e} T"
)

assert dBz.max() < 0.1 * B0
assert perp < 0.1 * B0
# no artifact pinned to the open faces (PIC noise is face-uniform)
assert face < 3.0 * interior
