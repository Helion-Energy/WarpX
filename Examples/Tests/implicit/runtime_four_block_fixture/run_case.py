"""Run one pinned native runtime probe in a new output directory."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--binary", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--eta", choices=["zero", "constant", "temperature"], default="zero")
p.add_argument("--ranks", type=int, default=1, choices=[1, 2])
p.add_argument("overrides", nargs="*")
a = p.parse_args()
binary = a.binary.resolve()
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
inputs = Path(__file__).with_name("inputs_runtime").read_text()
if a.eta == "constant":
    inputs = inputs.replace(
        "plasma_resistivity(rho,J,t) = 0.0", "plasma_resistivity(rho,J,t) = 1.e-3"
    )
elif a.eta == "temperature":
    inputs = inputs.replace(
        "plasma_resistivity(rho,J,t) = 0.0",
        'plasma_resistivity(rho,J,Te,t) = "1.e-3*(Te/1160451.812155)^2"',
    )
(out / "inputs").write_text(inputs)
launch = (
    []
    if a.ranks == 1
    else [
        "mpirun",
        "--mca",
        "pml",
        "ob1",
        "--mca",
        "btl",
        "self,tcp",
        "-np",
        str(a.ranks),
    ]
)
command = launch + [str(binary), str(out / "inputs"), *a.overrides]
env = os.environ.copy()
env["OMP_NUM_THREADS"] = "1"
(out / "COMMAND.json").write_text(json.dumps(command, indent=2) + "\n")
with (out / "run.log").open("w") as log:
    result = subprocess.run(
        command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180
    )
log = (out / "run.log").read_text()
records = (
    [json.loads(x) for x in (out / "PROBES.jsonl").read_text().splitlines()]
    if (out / "PROBES.jsonl").exists()
    else []
)
summary = {
    "exit_code": result.returncode,
    "complete": result.returncode == 0 and "PASS RUNTIME_FOUR_BLOCK" in log,
    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    "input_sha256": hashlib.sha256(inputs.encode()).hexdigest(),
    "ranks": a.ranks,
    "eta": a.eta,
    "records": len(records),
    "OMP_NUM_THREADS": 1,
}
(out / "RUN_RESULT.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
if not summary["complete"]:
    print("\n".join(log.splitlines()[-35:]))
    raise SystemExit(1)
from analyze_case import validate

print("NATIVE_GATES", json.dumps(validate(out)), flush=True)
print(json.dumps(records[0]), flush=True)
for direction in ["E", "U", "mixed"]:
    r = next(x for x in records if x.get("direction") == direction)
    print(
        direction,
        r["full_derivative"],
        r["derivative_difference"],
        r["full_moments"],
        r["moment_derivative_difference"],
        flush=True,
    )
