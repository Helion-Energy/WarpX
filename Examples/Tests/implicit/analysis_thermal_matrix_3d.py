#!/usr/bin/env python3
"""Execute a frozen native3Dthermal case manifest and preserve every outcome."""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--executable", type=Path, required=True)
p.add_argument("--manifest", type=Path, required=True)
p.add_argument("--inputs", type=Path, required=True)
p.add_argument("--output-parent", type=Path, required=True)
p.add_argument("--launcher-json", default="[]")
a = p.parse_args()
launcher = json.loads(a.launcher_json)
assert isinstance(launcher, list) and all(isinstance(x, str) for x in launcher)
a.output_parent.mkdir(parents=True, exist_ok=True)
out = Path(tempfile.mkdtemp(prefix="attempt-", dir=a.output_parent))
rows = []
for case in json.loads(a.manifest.read_text()):
    cwd = out / case["name"]
    cwd.mkdir()
    cmd = [
        *launcher,
        str(a.executable.resolve()),
        str(a.inputs.resolve()),
        *case["args"],
        "amrex.the_arena_init_size=0",
        "amrex.the_arena_is_managed=0",
    ]
    (cwd / "COMMAND.json").write_text(json.dumps(cmd, indent=2) + "\n")
    with (cwd / "stdout.log").open("x") as log:
        try:
            code = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "1"},
                timeout=1200,
            ).returncode
        except subprocess.TimeoutExpired:
            code = -999
    raw = (cwd / "stdout.log").read_text()
    records = [
        json.loads(line.split(" ", 1)[1])
        for line in raw.splitlines()
        if line.startswith(tuple(case["prefixes"]))
    ]
    # AMReX wraps assertion messages with a leading # on continuation lines.
    # Match the full diagnostic after removing formatting, retaining the
    # nonzero-exit and Assertion requirements.
    diagnostic = " ".join(re.sub(r"(?m)^\s*#\s*", " ", raw).split())
    if case.get("reject"):
        passed = (
            code not in (0, -999)
            and "Assertion" in raw
            and case["reject_message"] in diagnostic
        )
    else:
        passed = code == 0 and len(records) == case["record_count"]
    row = dict(case=case, returncode=code, results=records, passed=passed)
    rows.append(row)
    (cwd / "RECEIPT.json").write_text(json.dumps(row, indent=2) + "\n")
    (out / "RESULTS.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(case["name"], "PASS" if passed else "FAIL", flush=True)
passed = bool(rows) and all(x["passed"] for x in rows)
(out / ("PASS.json" if passed else "FAIL.json")).write_text(
    json.dumps(dict(passed=passed, cases=rows), indent=2) + "\n"
)
if not passed:
    raise SystemExit(f"Native thermal matrix failed: {out}")
