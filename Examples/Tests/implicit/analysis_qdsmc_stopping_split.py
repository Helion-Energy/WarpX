#!/usr/bin/env python3
"""Check raw nodal energy delivery and the complete explicit PC source placement."""

import argparse
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--split-exe", type=Path, required=True)
parser.add_argument("--order-exe", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--ranks", type=int, default=1)
parser.add_argument("--mpiexec", default="mpiexec")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
out = Path(tempfile.mkdtemp(prefix="attempt-", dir=args.output.resolve()))
fixture = Path(__file__).resolve().parent
rows = []


def run(tag, executable, inputs, extra):
    directory = out / tag
    directory.mkdir()
    command = [str(executable.resolve()), str(fixture / inputs), *extra]
    if args.ranks > 1:
        command = [args.mpiexec, "-n", str(args.ranks), *command]
    (directory / "COMMAND.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "run.log").open("w") as log:
        proc = subprocess.run(
            command,
            cwd=directory,
            env=os.environ,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    match = re.search(
        r"(?:STOPPING_SPLIT|PC_SOURCE_ORDER) (\{.*\})",
        (directory / "run.log").read_text(),
    )
    data = json.loads(match[1]) if match else None
    row = dict(tag=tag, returncode=proc.returncode, diagnostic=data)
    rows.append(row)
    (out / "RESULTS.json").write_text(json.dumps(rows, indent=2) + "\n")
    assert proc.returncode == 0 and data is not None, row
    return data


# Distinct seam partials and periodic endpoint copies; capacities differ 4x.
# Keep at most eight boxes per rank, including CUDA performance guards.
layouts = (
    [(16, 16, 16), (16, 16, 8), (32, 16, 8)]
    if args.ranks == 1
    else [(16, 16, 8), (32, 32, 8)]
)
for nr, nz, grid in layouts:
    for pedestal in [0, 1]:
        for mode, fraction, cooling in [
            ("halves", 0.5, 0),
            ("cooling", 0.5, 1),
            ("full", 1.0, 0),
        ]:
            run(
                f"{mode}-cells{nr}x{nz}-grid{grid}-ped{pedestal}",
                args.split_exe,
                "inputs_qdsmc_source_stage_capacity",
                [
                    f"amr.n_cell={nr} {nz}",
                    f"amr.max_grid_size={grid}",
                    f"audit.fraction={fraction}",
                    f"audit.cooling={cooling}",
                    f"hybrid_pic_model.density_pedestal={pedestal}",
                ],
            )

# Q*dt is prescribed deposited energy. This tests the real PC driver, with
# density evolution/compression, against an independent analytic thermal ODE.
orders = {}
for capacity in [0, 1]:
    errors = []
    for steps in [16, 32, 64]:
        result = run(
            f"pc-cap{capacity}-steps{steps}",
            args.order_exe,
            "inputs_qdsmc_source_stage_order",
            [
                "amr.max_grid_size=8",
                "audit.stopping=1",
                f"audit.steps={steps}",
                f"hybrid_pic_model.qdsmc_source_stage_density={capacity}",
            ],
        )
        assert result["stopping"]
        errors.append(result["max_error_K"])
    orders[capacity] = math.log2(errors[-2] / errors[-1])
assert 0.9 < orders[0] < 1.2, orders
assert 1.9 < orders[1] < 2.1, orders
summary = dict(
    pass_checks=True, ranks=args.ranks, orders=orders, cases=rows, attempt=str(out)
)
(out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"pass": True, "orders": orders, "attempt": str(out)}), flush=True)
