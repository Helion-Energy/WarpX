#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from analyze import run_analysis

p = argparse.ArgumentParser()
p.add_argument("--executable", type=Path, required=True)
p.add_argument("--input", type=Path, required=True)
p.add_argument("--case", required=True)
p.add_argument("--ranks", type=int, default=1)
p.add_argument("--mpi", default="mpiexec")
p.add_argument("--negative", default="")
p.add_argument("--override", action="append", default=[])
a = p.parse_args()
run = Path(tempfile.mkdtemp(prefix=a.case + "-", dir=Path.cwd()))
cmd = (
    ([a.mpi, "--bind-to", "none", "-n", str(a.ranks)] if a.ranks > 1 else [])
    + [str(a.executable.absolute()), str(a.input.absolute())]
    + a.override
)
if a.negative:
    cmd += ["fixture.negative=" + a.negative]
env = os.environ.copy()
env["OMP_NUM_THREADS"] = "1"
(run / "COMMAND.json").write_text(json.dumps(cmd, indent=2) + "\n")
with (run / "run.log").open("w") as f:
    result = subprocess.run(cmd, cwd=run, env=env, stdout=f, stderr=subprocess.STDOUT)
assert result.returncode == 0, (result.returncode, str(run / "run.log"))
if a.negative:
    text = (run / "run.log").read_text()
    marker = "NATIVE_SPATIAL_STOPPING_NEGATIVE_PASS case=" + a.negative
    assert (
        text.count(marker) == 1
        and "purity=exact" in text
        and "NATIVE_SPATIAL_STOPPING_PASS " not in text
    )
    expected = {
        "budget": "event work/capability gate",
        "speed": "event candidate inadmissible",
        "nonuniform": "nonuniform or unconstrained spatial context",
        "faraday": "event Faraday gate",
        "external_current": "unsupported spatial stopping scope",
        "external_particle": "unsupported spatial stopping scope",
    }
    assert "reason=" + expected[a.negative] in text
    report = {"pass": True, "rejected": a.negative, "reason": expected[a.negative]}
else:
    report = run_analysis(run)
(run / "ANALYSIS.json").write_text(json.dumps(report, indent=2) + "\n")
print(run)
print(json.dumps(report))
