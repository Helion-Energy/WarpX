"""Fresh-directory native RZ event check; never overwrites simulation evidence."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from analyze import analyze

p = argparse.ArgumentParser()
p.add_argument("--executable", type=Path, required=True)
p.add_argument("--input", type=Path, required=True)
p.add_argument("--case", required=True)
p.add_argument("--override", action="append", default=[])
p.add_argument("--negative")
p.add_argument("--ranks", type=int, default=1)
p.add_argument("--mpi")
a = p.parse_args()
root = Path(tempfile.mkdtemp(prefix="native-rz-event-" + a.case + "-", dir=Path.cwd()))
cmd = (
    ([a.mpi, "--bind-to", "none", "-n", str(a.ranks)] if a.ranks > 1 else [])
    + [str(a.executable.absolute()), str(a.input.absolute())]
    + a.override
)
if a.negative:
    cmd += ["fixture.negative=" + a.negative]
(root / "PLAN.json").write_text(
    json.dumps(
        {
            "command": cmd,
            "executable_sha256": hashlib.sha256(a.executable.read_bytes()).hexdigest(),
            "input_sha256": hashlib.sha256(a.input.read_bytes()).hexdigest(),
        },
        indent=2,
    )
    + "\n"
)
env = os.environ.copy()
env["OMP_NUM_THREADS"] = "1"
with (root / "run.log").open("w") as out:
    r = subprocess.run(cmd, cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT)
(root / "RESULT.json").write_text(json.dumps({"exit": r.returncode}) + "\n")
assert r.returncode == 0, root
if a.negative:
    log = (root / "run.log").read_text()
    assert (
        f"NATIVE_RZ_SPATIAL_STOPPING_NEGATIVE_PASS case={a.negative}" in log
        and "purity=exact" in log
    )
else:
    (root / "ANALYSIS.json").write_text(json.dumps(analyze(root), indent=2) + "\n")
print(root)
