#!/usr/bin/env python3
"""Qualify synchronized density floors through native explicit thermal operators."""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--executable", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--ranks", type=int, default=1)
parser.add_argument("--mpiexec", default="mpiexec")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
out = Path(tempfile.mkdtemp(prefix="attempt-", dir=args.output.resolve()))
inputs = Path(__file__).resolve().with_name("inputs_qdsmc_density_floor_sync")
rows = []


def run(tag, overrides, rejection=None):
    directory = out / tag
    directory.mkdir()
    command = [str(args.executable.resolve()), str(inputs), *overrides]
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
    text = (directory / "run.log").read_text()
    match = re.search(r"FLOOR_SYNC (\{.*\})", text)
    result = json.loads(match[1]) if match else None
    row = dict(tag=tag, returncode=proc.returncode, diagnostic=result)
    rows.append(row)
    (out / "RESULTS.json").write_text(json.dumps(rows, indent=2) + "\n")
    if rejection is None:
        assert proc.returncode == 0 and result and result["pass"], row
        steps = [json.loads(m) for m in re.findall(r"FLOOR_SYNC_STEP (\{.*\})", text)]
        assert len(steps) == 4, row
        assert [step["floor"] for step in steps] == [8e18, 5e17, 8e18, 1e18]
        row["steps"] = steps
    else:
        assert proc.returncode != 0 and rejection in text, row
        row["expected_rejection"] = rejection


run("linked-fixed-pedestal", [])
run("linked-no-pedestal", ["hybrid_pic_model.density_pedestal=0"])
run("linked-fixed-profile", ["hybrid_pic_model.density_pedestal_profile(x,y,z)=4.e18"])
run("linked-following-pedestal", ["hybrid_pic_model.density_pedestal_track_floor=1"])
run(
    "linked-explicit-equal-inputs",
    [
        "hybrid_pic_model.qdsmc_n_floor=1.e18",
        "hybrid_pic_model.qdsmc_te_n_floor=1.e18",
    ],
)
run(
    "independent-controls",
    [
        "hybrid_pic_model.qdsmc_sync_density_floors=0",
        "hybrid_pic_model.qdsmc_n_floor=2.e18",
        "hybrid_pic_model.qdsmc_te_n_floor=3.e18",
        "audit.initial_cond=2.e18",
        "audit.initial_te=3.e18",
    ],
)
run(
    "independent-defaults",
    [
        "hybrid_pic_model.qdsmc_sync_density_floors=0",
        "audit.initial_cond=1",
    ],
)
# Rejections require the specific diagnostic, not just an arbitrary process failure.
if args.ranks == 1:
    for parameter in ["qdsmc_n_floor", "qdsmc_te_n_floor"]:
        run(
            f"input-conflict-{parameter}",
            [f"hybrid_pic_model.{parameter}=2.e18"],
            "to be omitted or equal to n_floor",
        )
    run(
        "runtime-conflict",
        ["audit.mode=conflict"],
        "update all three through set_hybrid_pic_density_floor",
    )
    run(
        "runtime-invalid",
        ["audit.mode=invalid"],
        "SetDensityFloor requires a finite floor",
    )

summary = dict(pass_checks=True, ranks=args.ranks, cases=rows, attempt=str(out))
(out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"pass": True, "cases": len(rows), "attempt": str(out)}), flush=True)
