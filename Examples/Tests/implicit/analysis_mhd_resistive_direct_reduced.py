#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Partition gate of the reduced solve of the direct resistive block.

Usage: analysis_mhd_resistive_direct_reduced.py <threshold> <margin> <min dropped fraction> <min column-only rows>

``implicit_mhd.resistive_direct_row_threshold`` splits the assembled rows
into a factorized sub-system and rows solved as b_i / a_ii. The dump of
the full matrix and of the sub-system (``<prefix>_matrix.bin``,
``<prefix>_reduced_matrix.bin``, ``<prefix>_reduced_rows.bin``) must
satisfy the rule exactly: a row is kept iff some off-diagonal coupling in
its ROW or its COLUMN exceeds (threshold / margin) times the diagonal of
that row/column; the sub-system is the full matrix restricted to the kept
rows (values bit-equal); the dropped couplings all lie within the build
bound; at least the given fraction of the rows is dropped (so the
partition is not trivial on this deck); and -- the exactness of the
trivial rows -- no dropped row is referenced by a kept row beyond the
bound (at threshold 0: not referenced at all), since a trivial row's
solution enters the kept rows' equations only through such references.

The COLUMN criterion is what keeps rows that have no couplings of their
own but are referenced by neighbours (frozen wall faces read by live
faces; the one-sided emissions of the hyper-resistive chain and the wall
seam): a row-only rule drops them and truncates O(1) couplings of the
kept rows. Decks whose coupling pattern is structurally symmetric cannot
see that (every row referenced is also coupling), so the last argument is
the number of rows the deck must keep by the column rule ALONE -- the
test refuses a deck that does not discriminate. On such a deck a missing
column criterion fails three checks: the kept set (equality with the
rule), the referenced-dropped-row check, and the count itself.
"""

import struct
import sys

import numpy as np
import scipy.sparse as sp

threshold = float(sys.argv[1])
margin = float(sys.argv[2])
min_dropped_fraction = float(sys.argv[3])
min_column_only_rows = int(sys.argv[4]) if len(sys.argv) > 4 else 0
prefix = "diags/resistive_direct"


def read_csr(path):
    with open(path, "rb") as f:
        assert f.read(8) == b"WXCSR001", path
        n, nnz = struct.unpack("<qq", f.read(16))
        spacedim, ncomp = struct.unpack("<ii", f.read(8))
        for _ in range(ncomp):
            f.read(12 + 12 + 12)
        row_offsets = np.frombuffer(f.read(4 * (n + 1)), dtype=np.int32)
        columns = np.frombuffer(f.read(4 * nnz), dtype=np.int32)
        values = np.frombuffer(f.read(8 * nnz), dtype=np.float64)
        assert f.read(1) == b"", "trailing bytes in " + path
    return sp.csr_matrix((values.copy(), columns.copy(), row_offsets.copy()), shape=(n, n))


def read_vec(path):
    with open(path, "rb") as f:
        assert f.read(8) == b"WXVEC001", path
        (n,) = struct.unpack("<q", f.read(8))
        return np.frombuffer(f.read(8 * n), dtype=np.float64).copy()


a = read_csr(prefix + "_matrix.bin")
sub = read_csr(prefix + "_reduced_matrix.bin")
rows = read_vec(prefix + "_reduced_rows.bin").astype(np.int64)
n = a.shape[0]

# the rule, recomputed from the full matrix
build_threshold = threshold / margin
diagonal = np.abs(a.diagonal())
assert np.all(diagonal > 0.0)
coo = a.tocoo()
off = coo.row != coo.col
r, c, v = coo.row[off], coo.col[off], np.abs(coo.data[off])
keep_by_row = np.zeros(n, dtype=bool)
keep_by_row[r[v > build_threshold * diagonal[r]]] = True
keep_by_column = np.zeros(n, dtype=bool)
keep_by_column[c[v > build_threshold * diagonal[c]]] = True
keep = keep_by_row | keep_by_column
column_only_rows = int(np.sum(keep_by_column & ~keep_by_row))
expected_rows = np.flatnonzero(keep)
assert rows.shape == expected_rows.shape and np.array_equal(rows, expected_rows), (
    len(rows), len(expected_rows))
assert np.all(np.diff(rows) > 0), "kept rows are listed in canonical order"

# the sub-system is the exact restriction (pattern and values bit-equal)
restricted = a[rows][:, rows].tocsr()
restricted.sort_indices()
sub.sort_indices()
assert sub.shape == restricted.shape
assert np.array_equal(sub.indptr, restricted.indptr)
assert np.array_equal(sub.indices, restricted.indices)
assert np.array_equal(sub.data, restricted.data), "sub-system values are the full matrix's"

# every dropped coupling is within the build bound (both sides)
dropped = ~keep
touching = dropped[r] | dropped[c]
if touching.any():
    ratio_r = np.where(dropped[r[touching]], v[touching] / diagonal[r[touching]], 0.0)
    ratio_c = np.where(dropped[c[touching]], v[touching] / diagonal[c[touching]], 0.0)
    worst = max(ratio_r.max(), ratio_c.max())
    assert worst <= build_threshold * (1.0 + 1.0e-12), (worst, build_threshold)
else:
    worst = 0.0

# exactness of the trivial rows: a kept row may reference a DROPPED row only
# within the bound (the dumped row list decides what is dropped, so a code
# that dropped referenced rows fails here independently of the rule check)
dumped_dropped = np.ones(n, dtype=bool)
dumped_dropped[rows] = False
referenced = dumped_dropped[c] & ~dumped_dropped[r]
if referenced.any():
    worst_reference = np.max(v[referenced] / diagonal[c[referenced]])
    assert worst_reference <= build_threshold * (1.0 + 1.0e-12), (
        worst_reference, build_threshold, "a kept row references a dropped row beyond the bound")
assert column_only_rows >= min_column_only_rows, (
    column_only_rows, min_column_only_rows, "the deck does not discriminate the column criterion")

dropped_fraction = 1.0 - len(rows) / n
print(
    f"reduced solve partition: threshold {threshold:g} (build {build_threshold:g}), "
    f"{len(rows)} of {n} rows kept, dropped fraction {dropped_fraction:.3f}, "
    f"sub-system nonzeros {sub.nnz} of {a.nnz}, worst dropped coupling {worst:.3e}, "
    f"rows kept by the column rule alone {column_only_rows}"
)
assert dropped_fraction >= min_dropped_fraction, (dropped_fraction, min_dropped_fraction)
