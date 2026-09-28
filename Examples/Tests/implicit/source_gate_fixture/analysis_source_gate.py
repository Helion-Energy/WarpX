#!/usr/bin/env python3
"""Cross-layout test of the production recorded-final-gather eta receipt.

The native callback runs only before the initial E solve. All residuals,
particle pushes, deposition, OU preparation/acceptance and work gathers use
production code. No debug fields or DTA/reducer instrumentation are needed.
"""

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("executable", type=Path)
p.add_argument("input", type=Path)
p.add_argument(
    "--mode", choices=["coupled_jfnk", "decoupled_jfnk"], default="coupled_jfnk"
)
p.add_argument("--ranks", type=int, choices=[1, 2], default=1)
a = p.parse_args()
prefix = []
if a.ranks > 1:
    prefix = shlex.split(os.environ.get("MPIEXEC", "mpiexec"))
    prefix += shlex.split(os.environ.get("MPIEXEC_PREFLAGS", ""))
    prefix += ["-n", str(a.ranks)]
records = []
for grid in [16, 8]:
    cwd = Path(tempfile.mkdtemp(prefix=f"box{grid}-", dir=Path.cwd())).resolve()
    command = prefix + [
        str(a.executable.resolve()),
        str(a.input.resolve()),
        f"hybrid_pic_model.electron_energy_mode={a.mode}",
        f"amr.max_grid_size={grid}",
    ]
    proc = subprocess.run(
        command,
        cwd=cwd,
        env=dict(os.environ, OMP_NUM_THREADS="1"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    (cwd / "run.log").write_text(proc.stdout)
    row = dict(command=command, returncode=proc.returncode, cwd=str(cwd), grid=grid)
    records.append(row)
    (cwd / "command.json").write_text(json.dumps(row, indent=2) + "\n")
    assert proc.returncode == 0, proc.stdout[-5000:]
    receipt = [
        line
        for line in proc.stdout.splitlines()
        if line.startswith("Eulerian accepted ion electric work [J]:")
    ]
    assert len(receipt) == 1 and "species=ions " in receipt[0], receipt
    row["receipt"] = {
        key: float(value)
        for key, value in re.findall(r"([a-z_]+)=([^ ]+)", receipt[0])
        if key != "species"
    }
    assert all(math.isfinite(value) for value in row["receipt"].values())
    assert row["receipt"]["global_eta"] > 1.0e-9, "zero source cannot exercise corners"
    assert row["receipt"]["resistive_push_correction"] == 1
    assert "Expected OU population thermal partner [J]:" in proc.stdout

first, second = [row["receipt"] for row in records]
errors = {}
# Every work uses production recorded final gather positions, not a reconstructed
# midpoint. The unchanged weight and old K check identical physical loading.
for key in [
    "global_eta",
    "nr_global_eta",
    "global_eta_abs",
    "physical_weight",
    "kinetic_old",
    "kinetic_endpoint",
    "total_field",
    "kinetic_change",
]:
    scale = max(abs(first[key]), abs(second[key]))
    assert scale > 0
    errors[key] = abs(first[key] - second[key]) / scale
# Before the corner repair, global_eta differs by4.58e-5. The strict native
# solve/roundoff allowance here is1e-10; no production solver gate is relaxed.
passed = max(errors.values()) < 1.0e-10
result = dict(
    pass_all=passed,
    mode=a.mode,
    ranks=a.ranks,
    relative_differences=errors,
    records=records,
)
Path("RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
assert passed, result
print("SOURCE_CORNER_PASS", json.dumps(errors, sort_keys=True))
