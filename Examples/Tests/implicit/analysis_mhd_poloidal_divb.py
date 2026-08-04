#!/usr/bin/env python3
"""RZ poloidal-dynamics invariants for fluid_flux = hlld.

A poloidal velocity ring bends an initially uniform (discretely
divergence-free) Bz, generating Br/Bz dynamics through the corner UCT
EMF and the direct Er/Ez face fluxes. Asserted from the RAW staggered
fields:

1. div B stays at round-off: the discrete cylindrical divergence,
   (r_hi Br_hi - r_lo Br_lo)/(r_c dr) + (Bz_hi - Bz_lo)/dz, is exactly
   preserved by the Yee curl of any EMF, so growth flags a violation of
   the constrained-transport structure (e.g. a boundary or MPI seam in
   the corner EMF; the test runs on 2 ranks).
2. B_theta stays EXACTLY zero: axisymmetric poloidal dynamics have a
   purely toroidal current, so Er and Ez are pure eta J = 0 here; any
   drift flags an Er/Ez assembly error.
3. The dynamics really happened: |Br| must exceed a calibrated fraction
   of B0 (a scheme that suppressed the EMF entirely would pass 1-2).
"""

import sys

import numpy as np
import yt

yt.set_log_level(50)


def stats(path):
    ds = yt.load(path)
    io = ds.index.io
    dr = float((ds.domain_right_edge[0] - ds.domain_left_edge[0]) /
               ds.domain_dimensions[0])
    dz = float((ds.domain_right_edge[1] - ds.domain_left_edge[1]) /
               ds.domain_dimensions[1])
    rlo = float(ds.domain_left_edge[0])
    divb_max = bz_max = bt_max = br_max = 0.0
    for grid in ds.index.grids:
        br = io._read_raw_field(grid, ("raw", "Bx_fp"))  # (r-nodal, z-cc)
        bt = io._read_raw_field(grid, ("raw", "By_fp"))  # (cc, cc)
        bz = io._read_raw_field(grid, ("raw", "Bz_fp"))  # (r-cc, z-nodal)
        ilo = int(grid.get_global_startindex()[0])
        nrc = bt.shape[0]
        i = np.arange(nrc)
        rf_lo = rlo + (ilo + i) * dr
        rf_hi = rlo + (ilo + i + 1) * dr
        rc = rlo + (ilo + i + 0.5) * dr
        div = ((rf_hi[:, None] * br[1:, :] - rf_lo[:, None] * br[:-1, :]) /
               (rc[:, None] * dr) + (bz[:, 1:] - bz[:, :-1]) / dz)
        divb_max = max(divb_max, float(np.max(np.abs(div))))
        bz_max = max(bz_max, float(np.max(np.abs(bz))))
        bt_max = max(bt_max, float(np.max(np.abs(bt))))
        br_max = max(br_max, float(np.max(np.abs(br))))
        assert np.all(np.isfinite(br)) and np.all(np.isfinite(bz))
    return divb_max, bz_max, bt_max, br_max, dr


divb_0, bz_0, bt_0, br_0, dr = stats(sys.argv[1])
divb_1, bz_1, bt_1, br_1, _ = stats(sys.argv[2])

print(f"t = 0  : max|divB| = {divb_0:.3e} T/m, max|Btheta| = {bt_0:.3e}, "
      f"max|Br| = {br_0:.3e}")
print(f"t = end: max|divB| = {divb_1:.3e} T/m, max|Btheta| = {bt_1:.3e}, "
      f"max|Br| = {br_1:.3e}")
print(f"normalized divB(end) dr / max|B| = {divb_1 * dr / bz_1:.3e}")

assert divb_0 == 0.0
assert divb_1 * dr / bz_1 < 1.0e-12
assert bt_0 == 0.0
assert bt_1 == 0.0
assert br_1 > 5.0e-3 * bz_1

print("PASS")
