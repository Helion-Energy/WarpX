#!/usr/bin/env python3
"""Run native field-only tests; immutable originals are optional regression oracles."""

import argparse
import json
import os
import subprocess
from pathlib import Path

p = argparse.ArgumentParser(__doc__)
p.add_argument("--rz-updated", type=Path)
p.add_argument("--rz-original", type=Path)
p.add_argument("--cartesian-updated", type=Path)
p.add_argument("--cartesian-original", type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument(
    "--mpi", action="store_true", help="Add two-rank multi-box and empty-rank arms"
)
p.add_argument("--mpiexec", default="mpiexec")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
inputs = Path(__file__).resolve().parent
env = os.environ.copy()
env["OMP_NUM_THREADS"] = "1"
results = []


def run(dim, exe, nodal, label, ranks=1, box=8, negative=None):
    wd = a.output.resolve() / label
    wd.mkdir()
    cmd = ([a.mpiexec, "-n", str(ranks)] if ranks > 1 else []) + [
        str(exe.resolve()),
        str(inputs / ("inputs_" + dim)),
        "eta_capture.nodal=" + str(int(nodal)),
        "amr.max_grid_size=" + str(box),
        "amr.refine_grid_layout=0",
    ]
    if negative:
        cmd += ["eta_capture.case=" + negative]
    with (wd / "run.log").open("w") as log:
        proc = subprocess.run(
            cmd, cwd=wd, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    log = (wd / "run.log").read_text()
    rows = {}
    for line in log.splitlines():
        if line.startswith("ETA_CAPTURE "):
            values = dict(s.split("=", 1) for s in line.split()[1:] if "=" in s)
            rows[values["case"]] = values
    if negative:
        passed = (
            proc.returncode != 0
            and "Global eta field capture requires independent" in log
        )
    else:
        passed = proc.returncode == 0 and len(rows) == 14
        passed = passed and all(
            float(v["mirror_error"]) < 2.0e-12
            and float(v["physical_error"]) < 2.0e-12
            and float(v["input_error"]) == 0
            for v in rows.values()
        )
    entry = dict(
        label=label, cmd=cmd, cwd=str(wd), rc=proc.returncode, passed=passed, rows=rows
    )
    results.append(entry)
    (a.output / "RESULTS.json").write_text(json.dumps(results, indent=2) + "\n")
    print(label, "PASS" if passed else "FAIL", flush=True)
    return entry


for dim, updated, original in [
    ("rz", a.rz_updated, a.rz_original),
    ("3d", a.cartesian_updated, a.cartesian_original),
]:
    if updated is None:
        continue
    for nodal in [False] if dim == "rz" else [False, True]:
        for ranks, box in [(1, 8), (2, 8), (2, 64)] if a.mpi else [(1, 8)]:
            base = f"{dim}-{'nodal' if nodal else 'yee'}-mpi{ranks}-box{box}"
            live = run(dim, updated, nodal, base + "-updated", ranks, box)
            if original:
                ref = run(dim, original, nodal, base + "-original", ranks, box)
                exact = (
                    live["passed"]
                    and ref["passed"]
                    and all(
                        live["rows"][key]["field_checksum"]
                        == ref["rows"][key]["field_checksum"]
                        for key in live["rows"]
                    )
                )
                results.append(dict(label=base + "-original-checksum", passed=exact))
        for negative in ["alias", "guards"]:
            run(
                dim, updated, nodal, f"{dim}-{int(nodal)}-{negative}", negative=negative
            )
(a.output / "RESULTS.json").write_text(json.dumps(results, indent=2) + "\n")
if not results or not all(r["passed"] for r in results):
    raise SystemExit(1)
