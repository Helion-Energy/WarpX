"""Run independent native MM-PC gates in fresh per-case directories."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--ranks", type=int, default=1, choices=[1, 2])
parser.add_argument("--case", help="Run only this named case")
args = parser.parse_args()
binary = args.binary.resolve()
output = args.output.resolve()
output.mkdir(exist_ok=False, parents=True)
inputs = Path(__file__).with_name("inputs_mm_pc").resolve()
env = os.environ.copy()
env["OMP_NUM_THREADS"] = "1"
launch = (
    []
    if args.ranks == 1
    else [
        "mpirun",
        "--mca",
        "pml",
        "ob1",
        "--mca",
        "btl",
        "self,tcp",
        "-np",
        str(args.ranks),
    ]
)
cases = []
for mode in ["pc_only", "fallback", "matrix"]:
    jac = int(mode in ["fallback", "matrix"])
    for boxes in [32, 8]:
        cases.append(
            (
                f"{mode}-periodic-grid{boxes}",
                mode,
                [
                    f"amr.max_grid_size={boxes}",
                    f"implicit_evolve.use_mass_matrices_jacobian={jac}",
                ],
                None,
            )
        )
    cases.append(
        (
            f"{mode}-pmc-grid8",
            mode,
            [
                f"implicit_evolve.use_mass_matrices_jacobian={jac}",
                "boundary.field_lo=none neumann",
                "boundary.field_hi=pec neumann",
                "boundary.particle_lo=none reflecting",
                "boundary.particle_hi=reflecting reflecting",
            ],
            None,
        )
    )
cases += [
    (
        "none",
        "none",
        ["implicit_evolve.use_mass_matrices_pc=0", "jacobian.pc_type=none"],
        None,
    ),
    ("effective-none", "effective_none", ["jacobian.pc_type=none"], None),
    (
        "bad-width",
        "bad_width",
        [],
        "Esirkepov current-response band widths are inconsistent",
    ),
    (
        "bad-band",
        "bad_band",
        [],
        "Esirkepov MM-PC requires all nine complete current-response bands",
    ),
]
if args.case:
    cases = [c for c in cases if c[0] == args.case]
    if not cases:
        parser.error("Unknown case")
summary = {
    "binary": str(binary),
    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    "inputs_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
    "ranks": args.ranks,
    "env": {"OMP_NUM_THREADS": "1"},
    "cases": [],
}
for name, mode, options, rejection in cases:
    directory = output / name
    directory.mkdir()
    command = launch + [str(binary), str(inputs), f"mm_probe.mode={mode}", *options]
    (directory / "COMMAND.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "run.log").open("w") as log:
        result = subprocess.run(
            command,
            cwd=directory,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    log = (directory / "run.log").read_text()
    passed = (
        result.returncode != 0 and rejection in log
        if rejection
        else result.returncode == 0 and f"PASS MM_PC {mode}" in log
    )
    record = {
        "name": name,
        "exit_code": result.returncode,
        "passed": passed,
        "expected_rejection": rejection,
        "result": [
            line
            for line in log.splitlines()
            if line.startswith("PASS MM_PC") or (rejection and rejection in line)
        ],
    }
    summary["cases"].append(record)
    (output / "RESULTS.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(name, "PASS" if passed else "FAIL", record["result"], flush=True)
    if not passed:
        print("\n".join(log.splitlines()[-35:]))
        raise SystemExit(1)
