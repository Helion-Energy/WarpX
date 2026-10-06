#!/usr/bin/env python3
"""Check actual PC source-stage order and independently integrated receiver energy."""
import argparse
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--order-exe", type=Path, required=True)
p.add_argument("--capacity-exe", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--ranks", type=int, default=1)
p.add_argument("--full-pedestal", action="store_true")
p.add_argument("--mpiexec", default="mpiexec")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
out = Path(tempfile.mkdtemp(prefix="attempt-", dir=a.output.resolve()))
fixture = Path(__file__).resolve().parent
rows = []

def run(tag, exe, inputs, extra, expected_code=0):
    d = out / tag
    d.mkdir()
    cmd = [str(exe.resolve()), str(inputs), *extra]
    if a.full_pedestal:
        cmd.append("hybrid_pic_model.qdsmc_full_pedestal_transport=1")
    if a.ranks > 1:
        cmd = [a.mpiexec, "-n", str(a.ranks), *cmd, "amr.max_grid_size=8"]
    (d / "COMMAND.json").write_text(json.dumps(cmd, indent=2) + "\n")
    with (d / "run.log").open("w") as log:
        proc = subprocess.run(cmd, cwd=d, env={**os.environ, "OMP_NUM_THREADS": "1"},
                              stdout=log, stderr=subprocess.STDOUT)
    match = re.search(r"(?:ENERGY_RECOVERY|PC_SOURCE_ORDER) (\{.*\})",
                      (d / "run.log").read_text())
    data = json.loads(match[1]) if match else None
    row = dict(tag=tag, returncode=proc.returncode, expected_returncode=expected_code,
               diagnostic=data)
    rows.append(row)
    (out / "RESULTS.json").write_text(json.dumps(rows, indent=2) + "\n")
    assert proc.returncode == expected_code and data is not None, row
    return data

orders = {}
for enabled in [0, 1]:
    errors = []
    for steps in [16, 32, 64]:
        r = run(f"pc-{enabled}-{steps}", a.order_exe,
                fixture / "inputs_qdsmc_source_stage_order",
                ["amr.n_cell=16 16", "amr.max_grid_size=16", f"audit.steps={steps}",
                 f"hybrid_pic_model.qdsmc_source_stage_density={enabled}"])
        errors.append(r["max_error_K"])
    orders[enabled] = math.log2(errors[-2] / errors[-1])
assert 0.9 < orders[0] < 1.2, orders
assert 1.9 < orders[1] < 2.1, orders

cases = {
    "static": [],
    "drop": ["audit.nold=1e20", "audit.nnew=1.01e18"],
    "rise": ["audit.nold=1.01e18", "audit.nnew=1e20"],
    "empty-old-pedestal": ["audit.nold=0", "audit.nnew=1e20"],
    "below-old-no-pedestal": ["audit.nold=1e15", "audit.nnew=1e20",
                                "hybrid_pic_model.density_pedestal=0"],
    "closed-physical-gate": ["audit.nold=1e20", "audit.nnew=1e15"],
    "physical-coefficient-density": ["audit.nold=1e20", "audit.nnew=1.01e18",
        "audit.eta_factor=0.101",
        'hybrid_pic_model.joule_heating_resistivity(rho,J,Te,t)=1.e-5*rho/(1.602176634e-19*1.e19)'],
    "static-empty": ["audit.nold=0", "audit.nnew=0", "audit.heat=0"],
}
for tag, extra in cases.items():
    r = run(tag, a.capacity_exe, fixture / "inputs_qdsmc_source_stage_capacity",
            ["audit.old_receiver=1", *extra])
    assert r["native_map_verified"] and r["source_on_old_state_contract_pass"], r

result = dict(pass_checks=True, orders=orders, cases=rows, attempt=str(out))
(out / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"pass": True, "orders": orders, "attempt": str(out)}), flush=True)
