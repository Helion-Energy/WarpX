#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Native nonlinear time and checkpoint regression for mixed conduction BCs."""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--executable", type=Path, required=True)
parser.add_argument("--inputs", type=Path, required=True)
parser.add_argument("--suite", choices=["time", "restart"], required=True)
parser.add_argument("--output-parent", type=Path, required=True)
a = parser.parse_args()
a.output_parent.mkdir(parents=True, exist_ok=True)
output = Path(tempfile.mkdtemp(prefix="attempt-", dir=a.output_parent))
print(f"Native conduction evidence: {output}", flush=True)
common = [
    "boundary.field_lo=none pec",
    "boundary.field_hi=pec pec",
    "boundary.particle_lo=none reflecting",
    "boundary.particle_hi=reflecting reflecting",
    "ions.num_particles_per_cell_each_dim=1 1 1",
    "warpx.verbose=0",
    "hybrid_pic_model.qdsmc_conduction_fd_time=rkl2",
    "hybrid_pic_model.qdsmc_conduction_Te_floor=0.25",
    "hybrid_pic_model.qdsmc_conduction_flux_limit_factor=0",
    "hybrid_pic_model.qdsmc_conduction_chi_max=0",
    "hybrid_pic_model.qdsmc_kappa_par(n,Te,t)=1.5*1.380649e-23*100*n*(Te/2)^2.5",
    "hybrid_pic_model.qdsmc_kappa_perp(n,Te,t)=1.5*1.380649e-23*100*n*(Te/2)^2.5",
]


def run(name, options):
    dest = output / name
    dest.mkdir()
    command = [str(a.executable.resolve()), str(a.inputs.resolve()), *common, *options]
    (dest / "COMMAND.json").write_text(json.dumps(command, indent=2) + "\n")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    with (dest / "run.log").open("w") as log:
        result = subprocess.run(
            command, cwd=dest, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=90
        )
    assert result.returncode == 0, (dest, result.returncode)
    lines = [
        line
        for line in (dest / "run.log").read_text().splitlines()
        if line.startswith("BOUNDARY_EVOLVE")
    ]
    assert len(lines) == 1, (dest, lines)
    record = dict(re.findall(r"(\w+)=([^ ]+)", lines[0]))
    assert float(record["floor_J"]) == 0.0, record
    assert float(record["max_relative_residual"]) < 1.0e-11, record
    values = np.loadtxt(dest / "nodal_temperature.csv", delimiter=",", skiprows=1)[:, 2]
    return dest, values, record


if a.suite == "time":
    values = []
    records = []
    for steps in [4, 8, 16, 32, 256, 512]:
        _, state, record = run(
            f"steps-{steps}",
            [
                "amr.n_cell=32 64",
                "amr.max_grid_size=32",
                f"boundary_evolve.steps={steps}",
                f"boundary_evolve.dt={1.28e-5 / steps:.17g}",
            ],
        )
        values.append(state)
        records.append(record)
    reference = values[-1]
    norm = np.linalg.norm(reference)
    errors = [float(np.linalg.norm(v - reference) / norm) for v in values[:-2]]
    reference_delta = float(np.linalg.norm(values[-2] - reference) / norm)
    orders = np.log2(np.array(errors[:-1]) / errors[1:]).tolist()
    result = dict(
        errors=errors, orders=orders, reference_delta=reference_delta, records=records
    )
    (output / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    assert reference_delta < 0.1 * errors[-1], result
    assert min(orders[-2:]) >= 1.8, result
else:
    # The dedicated input uses the explicit hybrid solver. The native
    # checkpoint comparison advances conduction alone with frozen B/rho.
    common += [
        "boundary_evolve.beta=2",
        "boundary_evolve.eta=1e-5",
        "boundary_evolve.dt=2e-7",
        "hybrid_pic_model.qdsmc_kappa_perp(n,Te,t)=1.5*1.380649e-23*.01*n*(Te/2)^2.5",
    ]
    _, reference, full = run("continuous", ["boundary_evolve.steps=12"])
    first, _, before = run(
        "first",
        [
            "boundary_evolve.steps=6",
            "boundary_evolve.mode=restart_write",
            "diagnostics.enable=1",
            "diagnostics.diags_names=restart",
            "restart.diag_type=Full",
            "restart.format=checkpoint",
            "restart.file_prefix=checkpoint",
            "restart.intervals=1:1",
        ],
    )
    checkpoints = [
        p for p in first.glob("checkpoint*") if (p / "WarpXHeader").is_file()
    ]
    assert len(checkpoints) == 1, checkpoints
    _, restored, after = run(
        "second",
        [
            "boundary_evolve.steps=6",
            "boundary_evolve.mode=restart_read",
            f"amr.restart={checkpoints[0]}",
        ],
    )
    error = float(np.max(np.abs(restored - reference)) / np.max(np.abs(reference)))
    heat_errors = {}
    for channel in range(7):
        key = f"heat{channel}_J"
        total, left, right = (float(record[key]) for record in [full, before, after])
        heat_errors[key] = abs(total - left - right)
        assert heat_errors[key] <= 1.0e-11 * max(
            abs(total), abs(left) + abs(right), 1.0e-30
        )
    result = dict(max_relative_temperature_error=error, split_heat_errors_J=heat_errors)
    (output / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    assert error <= 1.0e-10, result
print(json.dumps(result), flush=True)
