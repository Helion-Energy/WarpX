#!/usr/bin/env python3
"""Check independent native Joule energy accounting and physical-source routing.

Preserve each invocation in a new attempt directory. Tolerances are enforced in
C++ using an independent physical dual-volume integral, not a result checksum.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

cases = {
    "species-small-raw-scale": ["joule_test.raw_species_scale=1.e-6", "joule_test.check_relaxation=1"],
    "species-large-raw-scale": ["joule_test.raw_species_scale=1.e6", "joule_test.check_relaxation=1"],
    "species-unit-raw-scale": ["joule_test.check_relaxation=1"],
    "species-small-raw-redirect": [
        "joule_test.raw_species_scale=1.e-6", "joule_test.redirect=1",
        "hybrid_pic_model.joule_redirect_Te_threshold=50",
        "joule_test.accepted_fraction=0", "joule_test.redirected_fraction=1",
    ],
    "receiving-thin": ["joule_test.receiving_density_ratio=0.2"],
    "receiving-dense": ["joule_test.receiving_density_ratio=10"],
    "receiving-redirect": [
        "joule_test.receiving_density_ratio=10", "joule_test.redirect=1",
        "hybrid_pic_model.joule_redirect_Te_threshold=50",
        "joule_test.accepted_fraction=0", "joule_test.redirected_fraction=1",
    ],
    "pedestal-off": ["hybrid_pic_model.density_pedestal=0"],
    "pedestal-zero": ["joule_test.pedestal_fraction=0"],
    "pedestal-p2": [],
    "pedestal-p5": ["joule_test.pedestal_fraction=0.5"],
    "variable-capacity": ["joule_test.variable=1"],
    "below-gate": ["joule_test.ne=5e16", "joule_test.accepted_fraction=0"],
    "at-gate": ["joule_test.ne=1e17", "joule_test.accepted_fraction=0"],
    "taper-half": [
        "joule_test.ne=1.5e17",
        "hybrid_pic_model.joule_heating_taper=1",
        "joule_test.accepted_fraction=0.5",
        "joule_test.declined_fraction=0.5",
    ],
    "source-taper-half": [
        "hybrid_pic_model.qdsmc_source_taper_n=6.666666666666667e18",
        "joule_test.accepted_fraction=0.5",
        "joule_test.declined_fraction=0.5",
    ],
    "redirect": [
        "joule_test.redirect=1",
        "hybrid_pic_model.joule_redirect_Te_threshold=50",
        "joule_test.accepted_fraction=0",
        "joule_test.redirected_fraction=1",
    ],
    "redirect-taper": [
        "joule_test.redirect=1",
        "hybrid_pic_model.joule_redirect_Te_threshold=50",
        "joule_test.ne=1.5e17",
        "hybrid_pic_model.joule_heating_taper=1",
        "joule_test.accepted_fraction=0",
        "joule_test.redirected_fraction=0.5",
        "joule_test.declined_fraction=0.5",
    ],
    "redirect-gated": [
        "joule_test.redirect=1",
        "hybrid_pic_model.joule_redirect_Te_threshold=50",
        "hybrid_pic_model.joule_redirect_n_min_factor=2",
        "joule_test.ne=1.5e17",
        "joule_test.accepted_fraction=0",
        "joule_test.declined_fraction=1",
    ],
}

# Actual two-species particle heating, not just staged redirected energy.
_pair = [
    "joule_test.check_pair_energy=1",
    "ions.num_particles_per_cell_each_dim=4 16 4",
    "ions2.num_particles_per_cell_each_dim=4 16 4",
    "ions.do_not_push=0", "ions2.do_not_push=0",
    "hybrid_pic_model.include_joule_heating=0",
    "hybrid_pic_model.qdsmc_viscosity_model=parser",
    "hybrid_pic_model.qdsmc_nu_par(n,Te,B)=0",
    "hybrid_pic_model.qdsmc_nu_perp(n,Te,B)=0",
    "hybrid_pic_model.qdsmc_viscosity_heating=strain",
    "hybrid_pic_model.qdsmc_viscosity_in_ohms_law=0",
]
for _name, _args in {
    "pedestal-zero": ["joule_test.pedestal_fraction=0"],
    "pedestal-p2": [],
    "pedestal-five": ["joule_test.pedestal_fraction=5"],
    "receiving-thin": ["joule_test.receiving_density_ratio=.2"],
    "receiving-dense": ["joule_test.receiving_density_ratio=10"],
    "stage-p2": ["joule_test.pair_via_stage=1"],
    "stage-thin": ["joule_test.pair_via_stage=1", "joule_test.receiving_density_ratio=.2"],
    "stage-dense": ["joule_test.pair_via_stage=1", "joule_test.receiving_density_ratio=10"],
}.items():
    cases["relaxation-pair-"+_name] = _pair + _args

p = argparse.ArgumentParser()
p.add_argument("--executable", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--ranks", type=int, default=1)
p.add_argument("--mpiexec", default="mpiexec")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
out = Path(tempfile.mkdtemp(prefix="attempt-", dir=a.output.resolve()))
template = Path(__file__).with_name("inputs_test_rz_joule_pedestal").read_text()
rows = []
for tag, extra in cases.items():
    d = out / tag
    d.mkdir()
    (d / "inputs").write_text(template)
    # One box on GPU; the independent MPI matrix separately uses many boxes.
    cmd = [
        str(a.executable.resolve()),
        str(d / "inputs"),
        "amr.max_grid_size=64",
        *extra,
    ]
    if a.ranks > 1:
        cmd = [a.mpiexec, "-n", str(a.ranks), *cmd, "amr.max_grid_size=16"]
    (d / "COMMAND.json").write_text(json.dumps(cmd, indent=2) + "\n")
    with (d / "run.log").open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd=d,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    match = re.search(r"JOULE_PEDESTAL (\{.*\})", (d / "run.log").read_text())
    data = json.loads(match[1]) if match else None
    pair_match = re.search(r"RELAXATION_PAIR (\{.*\})", (d / "run.log").read_text())
    pair = json.loads(pair_match[1]) if pair_match else None
    rows.append({"tag": tag, "returncode": proc.returncode, "diagnostic": data,
                 "relaxation_pair": pair})
    (out / "RESULTS.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(tag, proc.returncode, data, flush=True)
passed = all(
    row["returncode"] == 0
    and row["diagnostic"] is not None
    and row["diagnostic"]["pass"]
    for row in rows
)
print("Joule capacity/routing:", "PASS" if passed else "FAIL", out)
raise SystemExit(0 if passed else 2)
