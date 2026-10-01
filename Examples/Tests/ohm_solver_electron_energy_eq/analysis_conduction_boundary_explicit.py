#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Exercise mixed conduction boundaries through the full explicit PC advance."""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--executable", type=Path, required=True)
parser.add_argument("--inputs", type=Path, required=True)
parser.add_argument("--output-parent", type=Path, required=True)
parser.add_argument("--ranks", type=int, default=1)
parser.add_argument("--mpiexec", default="mpiexec")
a = parser.parse_args()
a.output_parent.mkdir(parents=True, exist_ok=True)
dest = Path(tempfile.mkdtemp(prefix="attempt-", dir=a.output_parent))
command = [
    str(a.executable.resolve()),
    str(a.inputs.resolve()),
    "max_step=12",
    "warpx.const_dt=1e-10",
    "hybrid_pic_model.elec_temp=2",
    "ions.num_particles_per_cell_each_dim=1 1 1",
    "hybrid_pic_model.qdsmc_conduction_fd_time=rkl2",
    "hybrid_pic_model.qdsmc_conduction_Te_floor=0.25",
    "hybrid_pic_model.qdsmc_conduction_chi_max=0",
    "hybrid_pic_model.qdsmc_conduction_flux_limit_factor=0",
    "hybrid_pic_model.qdsmc_conduction_bc_lo=adiabatic leg",
    "hybrid_pic_model.qdsmc_conduction_bc_hi=isothermal leg",
    "hybrid_pic_model.qdsmc_conduction_bc_Te_hi=2 0.5",
    "hybrid_pic_model.qdsmc_conduction_leg_length=6",
    "hybrid_pic_model.qdsmc_conduction_leg_Te_wall=0.5",
    "hybrid_pic_model.qdsmc_conduction_leg_flux_limit=1e-5",
    "hybrid_pic_model.qdsmc_kappa_par(n,Te,t)=1.5*1.380649e-23*100*n*(Te/2)^2.5",
    "hybrid_pic_model.qdsmc_kappa_perp(n,Te,t)=1.5*1.380649e-23*.01*n*(Te/2)^2.5",
    "hybrid_pic_model.qdsmc_energy_budget=1",
    "hybrid_pic_model.qdsmc_source_stage_audit_interval=1",
    "amrex.signal_handling=0",
]
if a.ranks > 1:
    command = [a.mpiexec, "-n", str(a.ranks), *command]
env = dict(
    os.environ,
    OMP_NUM_THREADS="1",
    WARPX_QDSMC_COND_STATS="1",
    WARPX_QDSMC_PHASE_MINTE="1",
)
(dest / "COMMAND.json").write_text(
    json.dumps(
        dict(
            argv=command,
            environment={
                k: env[k]
                for k in [
                    "OMP_NUM_THREADS",
                    "WARPX_QDSMC_COND_STATS",
                    "WARPX_QDSMC_PHASE_MINTE",
                ]
            },
        ),
        indent=2,
    )
    + "\n"
)
with (dest / "run.log").open("w") as log:
    result = subprocess.run(
        command, cwd=dest, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=90
    )
assert result.returncode == 0, (dest, result.returncode)
text = (dest / "run.log").read_text()
used = (dest / "warpx_used_inputs").read_text()
assert re.search(r"algo.evolve_scheme\s*=\s*explicit", used, re.I), dest
records = [
    dict(re.findall(r"(\w+)=([^\s]+)", line))
    for line in text.splitlines()
    if "cond rk stats:" in line
]
assert len(records) == 24, (dest, len(records))
for half in [1, 2]:
    group = [r for r in records if int(r["half"]) == half]
    assert len(group) == 12, (half, len(group))
for r in records:
    assert float(r["t_done_frac"]) == 1 and float(r["floor_J"]) == 0, r
    assert abs(float(r["residual_J"])) <= 1e-11 * float(r["E_cap_J"]), r
phases = re.findall(r"phase=(\w+) minTe=([^\s]+)", text)
assert {"pc_conduction_half1", "pc_conduction_half2"} <= {p for p, _ in phases}, phases
minimum = min(float(t) for _, t in phases)
assert minimum > 0.25 * 1.602176634e-19 / 1.380649e-23, minimum
result = dict(
    scheme="explicit",
    steps=12,
    conduction_halves=len(records),
    ranks=a.ranks,
    min_temperature_K=minimum,
    max_relative_conduction_residual=max(
        abs(float(r["residual_J"])) / float(r["E_cap_J"]) for r in records
    ),
    records=records,
)
(dest / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
print(f"EXPLICIT_PC_BOUNDARY_PASS {dest}")
