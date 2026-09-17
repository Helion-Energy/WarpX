#!/usr/bin/env python3
"""Validate a default-preserving block-banded MM-PC off/on twin."""

import argparse
import math
import re
from pathlib import Path

N_COLUMNS = 18
MM_PC_RE = re.compile(r"^implicit_evolve\.use_mass_matrices_pc = ([01])$")


def read_diag(root):
    path = root / "diags" / "newton_diag.txt"
    raw = path.read_bytes()
    rows = []
    for lineno, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        assert len(fields) == N_COLUMNS, (
            f"{path}:{lineno}: expected {N_COLUMNS} columns, got {len(fields)}"
        )
        values = [float(value) for value in fields]
        assert all(math.isfinite(value) for value in values), (
            f"{path}:{lineno}: non-finite solver record"
        )
        rows.append(values)
    assert len(rows) == 1, f"{path}: expected one solver row, got {len(rows)}"
    row = rows[0]
    assert int(row[0]) == 1, f"{path}: expected step 1"
    assert int(row[9]) == 3 and int(row[10]) == 3, (
        f"{path}: expected relative convergence status 3, got {row[9:11]}"
    )
    assert row[2] > 0 and row[6] > 0 and row[17] > 0, (
        f"{path}: expected positive Newton, GMRES, and residual-evaluation counts"
    )
    return raw


def read_and_normalize_inputs(root, expected):
    path = root / "warpx_used_inputs"
    lines = path.read_text(encoding="utf-8").splitlines()
    matches = [(index, MM_PC_RE.fullmatch(line)) for index, line in enumerate(lines)]
    matches = [(index, match) for index, match in matches if match is not None]
    assert len(matches) == 1, f"{path}: expected one explicit MM-PC setting"
    index, match = matches[0]
    actual = "on" if match.group(1) == "1" else "off"
    assert actual == expected, f"{path}: expected MM-PC {expected}, got {actual}"
    assert "implicit_evolve.use_mass_matrices_jacobian = 1" in lines, (
        f"{path}: mass-matrix Jacobian is not enabled"
    )
    assert 'jacobian.pc_type = "pc_block_banded"' in lines, (
        f"{path}: block-banded preconditioner is not selected"
    )
    lines[index] = "implicit_evolve.use_mass_matrices_pc = <TWIN>"
    return "\n".join(lines) + "\n"


def validate(root, expected):
    assert root.is_dir(), f"missing run directory: {root}"
    return {
        "diag": read_diag(root),
        "inputs": read_and_normalize_inputs(root, expected),
        "parameters": (root / "sim_parameters.dpkl").read_bytes(),
    }


parser = argparse.ArgumentParser()
parser.add_argument("--expect", choices=("off", "on"), required=True)
parser.add_argument("--reference", type=Path)
args = parser.parse_args()

current = validate(Path("."), args.expect)
if args.reference is not None:
    other = "off" if args.expect == "on" else "on"
    reference = validate(args.reference, other)
    assert current["diag"] == reference["diag"], (
        "MM-PC off/on Newton diagnostic records are not byte-identical"
    )
    assert current["inputs"] == reference["inputs"], (
        "MM-PC off/on used inputs differ outside the expected setting"
    )
    assert current["parameters"] == reference["parameters"], (
        "MM-PC off/on serialized simulation parameters differ"
    )
    print("MM-PC off/on isolated equality gate passed")
else:
    print(f"MM-PC {args.expect} self-check passed")
