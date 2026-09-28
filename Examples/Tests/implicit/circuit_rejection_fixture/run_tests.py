"""Native coupler rejection/retry matrix; no plugin disk checkpoint calls."""

import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build", type=Path, required=True)
parser.add_argument("--legacy-plugin", type=Path, required=True)
parser.add_argument("--c3-plugin", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--mpiexec", type=Path, required=True)
parser.add_argument("--cases", nargs="+")
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
fixture = Path(__file__).resolve().parent
binary = args.build.resolve() / "test_circuit_rejection"
cases = "replay resize phase device_view absent bad_api snapshot_failure active_view cancel_failure accept_failure token legacy_library c3_library finish_unaccepted".split()
if args.cases:
    cases = args.cases
jobs = []
for ranks in [1, 2]:
    for case in cases:
        command = (
            []
            if ranks == 1
            else [
                str(args.mpiexec),
                "--mca",
                "pml",
                "ob1",
                "--mca",
                "btl",
                "self,tcp",
                "-n",
                "2",
            ]
        )
        command += [
            str(binary),
            "test.case=" + case,
            "test.seed=" + str(fixture / "initial"),
        ]
        if case == "legacy_library":
            command += ["test.library=" + str(args.legacy_plugin.resolve())]
        if case == "c3_library":
            command += ["test.library=" + str(args.c3_plugin.resolve())]
        jobs.append((f"ranks{ranks}-{case}", command, case))


def run(job):
    label, command, case = job
    run_dir = out / label
    run_dir.mkdir(exist_ok=True)
    result = subprocess.run(
        command,
        env=dict(os.environ, OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        cwd=run_dir,
    )
    output = result.stdout + result.stderr
    (out / (label + ".log")).write_text(output)
    if case == "finish_unaccepted":
        passed = (
            result.returncode != 0
            and "Cannot use FinishStep to cancel an unaccepted native circuit" in output
            and "m_rejection.Irreversible()" in output
        )
    else:
        passed = result.returncode == 0 and "PASS circuit_rejection " + case in output
    record = {
        "label": label,
        "command": command,
        "returncode": result.returncode,
        "expected_rejection": case == "finish_unaccepted",
        "passed": passed,
    }
    print(label, "PASS" if passed else "FAIL", flush=True)
    return record


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    records = list(pool.map(run, jobs))
(out / "TEST_COMMANDS.json").write_text(json.dumps(records, indent=2) + "\n")
raise SystemExit(any(not record["passed"] for record in records))
