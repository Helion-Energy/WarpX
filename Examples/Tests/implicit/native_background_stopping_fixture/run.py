import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--executable", required=True)
p.add_argument("--case", required=True)
p.add_argument("--mode", default="coupled_jfnk")
p.add_argument("--pmc", action="store_true")
p.add_argument("--zero", action="store_true")
p.add_argument("--negative", choices=["ampere", "magnetic", "wall", "stale", "alias"])
p.add_argument("--grid", type=int, default=8)
p.add_argument("--ranks", type=int, default=1)
p.add_argument("--mpi")
p.add_argument("--steps", type=int, default=0)
a = p.parse_args()
source = Path(__file__).resolve().parent
root = Path(tempfile.mkdtemp(prefix="background-" + a.case + "-", dir=Path.cwd()))
exe = Path(a.executable).resolve()
cmd = [
    str(exe),
    str(source / "inputs_rz"),
    "max_step=" + str(a.steps),
    "amr.n_cell=" + str(a.grid) + " " + str(a.grid),
    "hybrid_pic_model.electron_energy_mode=" + a.mode,
]
if a.pmc:
    cmd += [
        "boundary.field_lo=none pmc",
        "boundary.field_hi=pec pmc",
        "boundary.particle_lo=none reflecting",
        "boundary.particle_hi=reflecting reflecting",
    ]
if a.zero:
    cmd += ["fixture.background_zero=1"]
if a.negative:
    cmd += ["fixture.background_negative=" + a.negative]
if a.ranks > 1:
    assert a.mpi
    cmd = [a.mpi, "-n", str(a.ranks), *cmd]
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
with (root / "run.log").open("wb") as log:
    rc = subprocess.run(
        cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
    ).returncode
text = (root / "run.log").read_text()
marker = (
    "BACKGROUND_NEGATIVE_PASS kind=" + a.negative
    if a.negative
    else "BACKGROUND_EVENT_PASS"
)
ok = (
    rc == 0
    and marker in text
    and "BACKGROUND_CONTINUATION_PASS steps=" + str(a.steps) in text
)
if ok and not a.negative and not a.zero:
    ok = (
        subprocess.run(
            [
                sys.executable,
                str(source / "analyze.py"),
                str(root),
                str(root / "INDEPENDENT.json"),
            ]
        ).returncode
        == 0
    )
result = dict(
    case=a.case,
    command=cmd,
    returncode=rc,
    pass_=ok,
    executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
    input_sha256=hashlib.sha256((source / "inputs_rz").read_bytes()).hexdigest(),
    log_sha256=hashlib.sha256((root / "run.log").read_bytes()).hexdigest(),
    scope="CPU fixed-density background event, strict native constraints and full arithmetic accounting; no live event dispatch or zero-PIC-energy claim",
)
(root / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
print(root)
print(json.dumps(result))
sys.exit(0 if ok else 1)
