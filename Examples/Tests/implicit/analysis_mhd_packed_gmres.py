#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Packed GMRES twin of an implicit MHD test.

The run uses newton.linear_solver = packed_gmres with packed_gmres.self_test
= 1 (the in-run check of the packed inner products, norms, axpy/lincomb,
the pack/unpack round trip and the GEMV kernels against the solver-vector
operations aborts the run on a mismatch), on the deck and solver settings
of a twin that ran amrex_gmres. Gates against that twin:

1. The solver selection is recorded in warpx_used_inputs (contract).

2. Same Krylov process up to round-off: per recorded step the Newton
   iteration count agrees within one, the GMRES iteration count within
   max(3, 10%), and the exit residual norms differ by less than the
   Newton convergence target max(atol, rtol * norm0) (an exit residual
   below the target is a round-off-level quantity; both solvers implement
   two-pass classical Gram-Schmidt and only the summation order of the
   inner products differs, so the iterates agree to round-off and both
   exits sit below the target).

3. Same converged state: the electron energy of the final plotfiles agrees
   to 1e-8 relative (L2 and max), i.e. far inside the Newton tolerance of
   the decks used with this analysis.

Usage: analysis_mhd_packed_gmres.py <final_plotfile> <twin_final_plotfile>
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

# 1. contract
used_inputs = Path("warpx_used_inputs").read_text()
match = re.search(r"^newton\.linear_solver\s*=\s*(\S+)", used_inputs, re.MULTILINE)
assert match is not None and match.group(1) == "packed_gmres", match
match = re.search(r"^packed_gmres\.self_test\s*=\s*(\S+)", used_inputs, re.MULTILINE)
assert match is not None and match.group(1) in ("1", "true"), match
match = re.search(r"^newton\.relative_tolerance\s*=\s*(\S+)", used_inputs, re.MULTILINE)
newton_rtol = float(match.group(1)) if match is not None else 1.0e-6
match = re.search(r"^newton\.absolute_tolerance\s*=\s*(\S+)", used_inputs, re.MULTILINE)
newton_atol = float(match.group(1)) if match is not None else 0.0

# 2. Krylov process
# newton.txt columns: [0] step [1] time [2] iters [3] total_iters [4] norm_abs
# [5] norm_rel [6] gmres_iters [7] gmres_total_iters [8] gmres_last_res ...
mine = np.atleast_2d(np.loadtxt("diags/newton.txt"))
twin = np.atleast_2d(np.loadtxt(twin_newton))
assert mine.shape[0] == twin.shape[0], (mine.shape, twin.shape)
print("step  newton(packed/amrex)  gmres(packed/amrex)  norm_abs(packed/amrex)  |d|/target")
worst_newton = 0
worst_gmres = 0.0
worst_norm = 0.0
for row_p, row_a in zip(mine, twin):
    assert int(row_p[0]) == int(row_a[0])
    dn = abs(int(row_p[2]) - int(row_a[2]))
    dg = abs(int(row_p[6]) - int(row_a[6]))
    # norm0 of the twin's solve from its exit norms; the convergence target
    # is what both solves were asked to reach
    norm0 = row_a[4] / row_a[5] if row_a[5] > 0.0 else 0.0
    target = max(newton_atol, newton_rtol * norm0, 1.0e-300)
    dnorm = abs(row_p[4] - row_a[4]) / target
    print(
        f"{int(row_p[0]):4d}  {int(row_p[2]):3d} / {int(row_a[2]):3d}"
        f"  {int(row_p[6]):5d} / {int(row_a[6]):5d}"
        f"  {row_p[4]:.3e} / {row_a[4]:.3e}  {dnorm:.3e}"
    )
    worst_newton = max(worst_newton, dn)
    worst_gmres = max(worst_gmres, dg / max(3.0, 0.1 * int(row_a[6])))
    worst_norm = max(worst_norm, dnorm)
    assert dn <= 1, (row_p, row_a)
    assert dg <= max(3, 0.1 * int(row_a[6])), (row_p, row_a)
    assert dnorm <= 1.0, (row_p, row_a, target)
print(
    f"cumulative GMRES iterations: packed = {int(mine[-1, 7])}, amrex = {int(twin[-1, 7])}; "
    f"worst |dNewton| = {worst_newton}, worst |dGMRES|/bound = {worst_gmres:.3f}, "
    f"worst |dnorm_abs|/target = {worst_norm:.3e}"
)

# 3. converged state
def electron_energy(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return np.squeeze(data["boxlib", "implicit_mhd_electron_energy"].value)


ee_p = electron_energy(final_plotfile)
ee_a = electron_energy(twin_plotfile)
assert ee_p.shape == ee_a.shape, (ee_p.shape, ee_a.shape)
scale_l2 = np.sqrt(np.sum(ee_a**2))
scale_max = np.max(np.abs(ee_a))
rel_l2 = np.sqrt(np.sum((ee_p - ee_a) ** 2)) / scale_l2
rel_max = np.max(np.abs(ee_p - ee_a)) / scale_max
print(f"electron energy, packed vs amrex: rel. L2 {rel_l2:.3e}, rel. max {rel_max:.3e}")
assert rel_l2 <= 1.0e-8, rel_l2
assert rel_max <= 1.0e-8, rel_max
print("packed GMRES twin: PASS")
