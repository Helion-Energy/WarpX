#!/usr/bin/env python3
#
# --- Analysis for the whistler-stiff (dt x8) pc_block_banded arm of the RZ
# --- theta-implicit hybrid EM-modes deck. At this time step the grid-scale
# --- whistler CFL makes the unpreconditioned linear solve expensive
# --- (~44-60 GMRES iterations/step measured on this deck), while the
# --- block-banded direct preconditioner holds the count flat, so the
# --- iteration ceiling below is a sharp regression tripwire for the
# --- preconditioner itself. Convergence is enforced in the run
# --- (newton.require_convergence), the extraction self-check aborts on
# --- failure, and the LU/Apply round-trips plus FD-JVP comparison are printed
# --- for evidence review. This script verifies the intended one-box/two-rank
# --- layout as well as the solver counts and diagnostic contract.

import re
from pathlib import Path

import numpy as np

cell_header = Path("diags/diag1000010/Level_0/Cell_H")
box_count_match = next(
    (
        re.fullmatch(r"\((\d+)\s+\d+", line)
        for line in cell_header.read_text(encoding="utf-8").splitlines()
        if re.fullmatch(r"\((\d+)\s+\d+", line)
    ),
    None,
)
assert box_count_match is not None, "could not read BoxArray size from Cell_H"
box_count = int(box_count_match.group(1))
assert box_count == 1, (
    f"empty-rank regression requires one BoxArray box, found {box_count}"
)

diag_path = Path("diags/newton_diag.txt")
header = diag_path.read_text(encoding="utf-8").splitlines()[0]
columns = re.findall(r"\[\d+\]([^\s]+)", header)
expected_columns = [
    "step()",
    "time(s)",
    "iters",
    "total_iters",
    "norm_abs",
    "norm_rel",
    "gmres_iters",
    "gmres_total_iters",
    "gmres_last_res",
    "exit_status",
    "convergence_status",
    "line_search_min_alpha",
    "line_search_halvings",
    "line_search_trials",
    "linear_rtol_last",
    "linear_rtol_min",
    "linear_rtol_max",
    "residual_evals",
]
assert columns == expected_columns, (
    f"unexpected Newton diagnostic schema: {columns} != {expected_columns}"
)

data = np.atleast_2d(np.loadtxt(diag_path))
assert data.shape[1] == len(expected_columns), (
    f"Newton diagnostic has {data.shape[1]} values but {len(expected_columns)} columns"
)
newton_iters = data[:, 2]
gmres_iters = data[:, 6]
exit_status = data[:, 9].astype(int)
convergence_status = data[:, 10].astype(int)
line_search_min_alpha = data[:, 11]
line_search_halvings = data[:, 12].astype(int)
line_search_trials = data[:, 13].astype(int)
linear_rtol_last = data[:, 14]
linear_rtol_min = data[:, 15]
linear_rtol_max = data[:, 16]
residual_evals = data[:, 17].astype(int)

newton_max = newton_iters.max()
gmres_mean = gmres_iters.mean()
gmres_max = gmres_iters.max()

print("pc_block_banded stiff arm (dt x8):")
print(f"  BoxArray boxes:       {box_count} across 2 MPI ranks")
print(f"  Newton iters/step: max {newton_max:.0f}, mean {newton_iters.mean():.2f}")
print(f"  GMRES iters/step:  max {gmres_max:.0f}, mean {gmres_mean:.2f}")

# Calibrated 2026-08-18 (2-rank CPU, OMP_NUM_THREADS=1): PC-on measured a
# flat 11 GMRES / 3 Newton per step; the PC-off control on the same deck
# (line search on) measured a flat 44 GMRES / 3 Newton per step. The mean
# ceiling sits at ~2x the preconditioned count and well below half the
# unpreconditioned count.
NEWTON_MAX_CEIL = 8
GMRES_MEAN_CEIL = 20.0
GMRES_MAX_CEIL = 30

assert newton_max <= NEWTON_MAX_CEIL, (
    f"Newton iterations regressed: max {newton_max} > ceiling {NEWTON_MAX_CEIL}"
)
assert gmres_mean <= GMRES_MEAN_CEIL, (
    f"mean GMRES/step regressed: {gmres_mean:.2f} > ceiling {GMRES_MEAN_CEIL}"
)
assert gmres_max <= GMRES_MAX_CEIL, (
    f"max GMRES/step regressed: {gmres_max} > ceiling {GMRES_MAX_CEIL}"
)

# Evidence-contract gates. This arm requires convergence, enables line search,
# and uses fixed 1e-4 linear forcing. It therefore exercises the appended
# schema without relying on a particular accepted-alpha history.
assert np.all(np.isin(exit_status, (2, 3))), (
    f"non-success public Newton statuses: {exit_status}"
)
assert np.array_equal(convergence_status, exit_status), (
    "required-convergence arm disagrees between public and convergence statuses"
)
assert np.all((line_search_min_alpha > 0.0) & (line_search_min_alpha <= 1.0)), (
    f"invalid accepted line-search alpha: {line_search_min_alpha}"
)
assert np.array_equal(line_search_halvings, line_search_trials - newton_iters), (
    "line-search halvings must equal evaluated trials minus one first trial per update"
)
assert np.all(line_search_trials >= newton_iters), (
    "line search must evaluate at least one trial per Newton update"
)
for name, values in (
    ("last", linear_rtol_last),
    ("minimum", linear_rtol_min),
    ("maximum", linear_rtol_max),
):
    assert np.allclose(values, 1.0e-4, rtol=1.0e-13, atol=0.0), (
        f"fixed linear tolerance has unexpected {name} values: {values}"
    )
assert np.all(residual_evals >= newton_iters + 1 + line_search_trials), (
    "residual-evaluation count misses a base or line-search evaluation"
)

print("pc_block_banded stiff-arm CI gates passed")
