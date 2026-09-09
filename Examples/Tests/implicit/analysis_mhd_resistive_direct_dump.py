#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Structure gate of the direct resistive block's matrix dump.

Physics: reruns the stiff RZ vacuum-resistive baseline's assertions
unchanged (the dump is a diagnostic; the fields must be the baseline's).

Dump: ``implicit_mhd.resistive_direct_dump_prefix`` wrote the assembled
sparse rows of ``I + theta_r dt curl((eta/mu0) curl .)`` at the second
assembly. The file must (1) be self-consistent (header sizes, row offsets,
column ranges, one canonical unknown per face of every participating
component); (2) carry the identity diagonal on every row and only
identity rows where the stencil emits nothing (the axis and outer-wall B_r
faces of this deck); (3) split into exactly two coupled blocks -- the poloidal (B_r,
B_z) and the azimuthal (B_theta) components decouple in axisymmetry
without the Hall term -- plus those identity rows; (4) be exactly
symmetric under the geometric row scaling d_i a_ij = d_j a_ji (the
cylindrical face-volume weights make the curl-curl self-adjoint), while
the raw matrix is visibly asymmetric: the property the symmetric/Cholesky
backends of the block depend on, and the sabotage the check would catch
(a lost metric factor or a one-sided boundary emission breaks it).
"""

import os
import runpy
import struct
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

# Same structural physics assertions as the unpreconditioned baseline.
runpy.run_path(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "analysis_mhd_vacuum_resistive_rz.py",
    ),
    run_name="__main__",
)

prefix = "diags/resistive_direct"

with open(prefix + "_matrix.bin", "rb") as f:
    assert f.read(8) == b"WXCSR001"
    n, nnz = struct.unpack("<qq", f.read(16))
    spacedim, ncomp = struct.unpack("<ii", f.read(8))
    comps = []
    for _ in range(ncomp):
        active, offset = struct.unpack("<iq", f.read(12))
        low = struct.unpack("<iii", f.read(12))
        high = struct.unpack("<iii", f.read(12))
        comps.append((active, offset, low, high))
    row_offsets = np.frombuffer(f.read(4 * (n + 1)), dtype=np.int32)
    columns = np.frombuffer(f.read(4 * nnz), dtype=np.int32)
    values = np.frombuffer(f.read(8 * nnz), dtype=np.float64)
    assert f.read(1) == b"", "trailing bytes in the matrix dump"

# (1) self-consistency
assert spacedim == 2 and ncomp == 3
sizes = []
for active, offset, low, high in comps:
    if active:
        assert offset == sum(sizes), (offset, sizes)
        sizes.append((high[0] - low[0] + 1) * (high[1] - low[1] + 1))
assert len(sizes) == 3, "the RZ resistive block carries all three face components"
assert n == sum(sizes), (n, sizes)
assert row_offsets[0] == 0 and row_offsets[-1] == nnz
assert np.all(np.diff(row_offsets) >= 1), "every canonical row is assembled"
assert columns.min() >= 0 and columns.max() < n
a = sp.csr_matrix((values.copy(), columns.copy(), row_offsets.copy()), shape=(n, n))
meta = open(prefix + "_meta.txt").read()
assert f"rows {n}" in meta and f"nonzeros {nnz}" in meta and "assembly 2" in meta, meta

# (2) identity diagonal on every row; identity rows only where nothing is emitted
diagonal = a.diagonal()
assert np.all(diagonal >= 1.0 - 1e-12), (diagonal.min(), "I + A has a unit-or-larger diagonal")
row_nnz = np.diff(row_offsets)
identity_rows = np.flatnonzero(row_nnz == 1)
assert np.allclose(diagonal[identity_rows], 1.0)
# the radial-boundary B_r faces (component 0 at the axis node and at the
# outer conducting wall node, where the tangential E vanishes) are the
# identity rows of this deck, and nothing else is
active0, offset0, low0, high0 = comps[0]
len_r0 = high0[0] - low0[0] + 1
radial_index = np.arange(sizes[0]) % len_r0
boundary_rows = offset0 + np.flatnonzero((radial_index == 0) | (radial_index == len_r0 - 1))
assert set(identity_rows.tolist()) == set(boundary_rows.tolist()), (
    len(identity_rows), len(boundary_rows))

# (3) two coupled blocks + the identity rows
pattern = (abs(a) + abs(a.T)).tocsr()
ncomponents, labels = csg.connected_components(pattern, directed=False)
block_sizes = np.sort(np.bincount(labels))[::-1]
assert ncomponents == 2 + len(identity_rows), (ncomponents, len(identity_rows))
assert block_sizes[0] == sizes[0] + sizes[2] - len(identity_rows), block_sizes[:3]
assert block_sizes[1] == sizes[1], block_sizes[:3]

# (4) exact symmetry under the geometric row scaling
coo = a.tocoo()
keys = coo.row.astype(np.int64) * n + coo.col.astype(np.int64)
mirror_keys = coo.col.astype(np.int64) * n + coo.row.astype(np.int64)
order = np.argsort(keys)
sorted_keys, sorted_values = keys[order], coo.data[order]
position = np.minimum(np.searchsorted(sorted_keys, mirror_keys), len(sorted_keys) - 1)
has_mirror = sorted_keys[position] == mirror_keys
off_diagonal = coo.row != coo.col
assert np.all(has_mirror[off_diagonal]), "the pattern is structurally symmetric"
ratio_ok = off_diagonal & (coo.data != 0.0) & (sorted_values[position] != 0.0)
ratio = coo.data[ratio_ok] / sorted_values[position][ratio_ok]  # a_ij / a_ji = d_j / d_i
graph = sp.csr_matrix((ratio, (coo.row[ratio_ok], coo.col[ratio_ok])), shape=(n, n))
d = np.ones(n)
for root in np.unique(labels, return_index=True)[1]:
    if np.sum(labels == labels[root]) == 1:
        continue
    bfs, predecessor = csg.breadth_first_order(
        graph, root, directed=False, return_predecessors=True)
    for node in bfs[1:]:
        i = predecessor[node]
        d[node] = d[i] * graph[i, node]
assert np.all(d > 0.0)
scaled = (sp.diags(d) @ a).tocsr()
scale = abs(scaled).max()
scaled_asymmetry = abs(scaled - scaled.T).max() / scale
raw_asymmetry = abs(a - a.T).max() / abs(a).max()
print(
    f"resistive direct dump: {n} rows, {nnz} nonzeros, {len(identity_rows)} identity rows, "
    f"blocks {block_sizes[:2].tolist()}; raw asymmetry {raw_asymmetry:.3e}, "
    f"row-scaled asymmetry {scaled_asymmetry:.3e}, d in [{d.min():.3g}, {d.max():.3g}]"
)
assert raw_asymmetry > 1.0e-3, "the raw RZ operator is not symmetric (metric factors)"
assert scaled_asymmetry < 1.0e-12, "the row-scaled operator must be symmetric to roundoff"
