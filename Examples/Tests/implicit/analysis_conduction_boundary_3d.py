#!/usr/bin/env python3
"""Cartesian native boundary-energy and rejection-rollback qualification."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--executable", type=Path, required=True)
p.add_argument("--output-parent", type=Path, required=True)
p.add_argument("--launcher-json", default="[]")
a = p.parse_args()
r = Path(__file__).resolve().parent
launcher = json.loads(a.launcher_json)
assert isinstance(launcher, list) and all(isinstance(x, str) for x in launcher)
a.output_parent.mkdir(parents=True, exist_ok=True)
out = Path(tempfile.mkdtemp(prefix="attempt-", dir=a.output_parent))
rows = []
for case in json.loads((r / "conduction_boundary_3d_cases.json").read_text()):
    cwd = out / case["id"]
    cwd.mkdir()
    command = [
        *launcher,
        str(a.executable.resolve()),
        str(r / "inputs_test_3d_conduction_boundary"),
        *case["args"],
        "amrex.the_arena_init_size=0",
        "amrex.the_arena_is_managed=0",
    ]
    (cwd / "COMMAND.json").write_text(json.dumps(command, indent=2) + "\n")
    with (cwd / "stdout.log").open("x") as log:
        try:
            code = subprocess.run(
                command,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "1"},
                timeout=180,
            ).returncode
        except subprocess.TimeoutExpired:
            code = -999
    results = [
        json.loads(line.split("CARTESIAN_BOUNDARY_RESULT ", 1)[1])
        for line in (cwd / "stdout.log").read_text().splitlines()
        if line.startswith("CARTESIAN_BOUNDARY_RESULT ")
    ]
    passed = code == 0 and len(results) == 1 and results[0]["passed"] is True
    row = dict(case=case, returncode=code, results=results, passed=passed)
    rows.append(row)
    (cwd / "RECEIPT.json").write_text(json.dumps(row, indent=2) + "\n")
    print(case["id"], "PASS" if passed else "FAIL", flush=True)
report = dict(passed=all(row["passed"] for row in rows), rows=rows)
(out / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
if not report["passed"]:
    raise SystemExit(f"Boundary fixture failed; evidence: {out}")
print(f"BOUNDARY_MATRIX_PASS {len(rows)} {out}", flush=True)
