"""Run the bounded source-free RZ runtime block-probe qualification."""

import argparse
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--binary", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
base = [
    "warpx.const_dt=1.e-8",
    "implicit_evolve.thermal.push_relative_tolerance=1.e-12",
    "warpx.Bz_external_grid_function(x,y,z)=0.1*(1+0.05*(x/0.25)^2)",
]
pmc = [
    "boundary.field_lo=none neumann",
    "boundary.field_hi=pec neumann",
    "boundary.particle_lo=none reflecting",
    "boundary.particle_hi=reflecting reflecting",
]
cases = [
    (
        "zero-b-curlfree",
        "zero",
        1,
        ["runtime_probe.curlfree=1", "warpx.Bz_external_grid_function(x,y,z)=0"],
    ),
    ("eta-temperature", "temperature", 1, base),
    ("eta-constant", "constant", 1, base),
    ("eta-temperature-halfdt", "temperature", 1, base + ["warpx.const_dt=5.e-9"]),
    ("eta-temperature-quarterdt", "temperature", 1, base + ["warpx.const_dt=2.5e-9"]),
    (
        "eta-temperature-no-density-projection",
        "temperature",
        1,
        base + ["implicit_evolve.mass_matrices_density_projection=0"],
    ),
    ("eta-temperature-onebox", "temperature", 1, base + ["amr.max_grid_size=32"]),
    ("eta-temperature-mpi2", "temperature", 2, base),
    (
        "eta-temperature-particle-control",
        "temperature",
        1,
        base
        + [
            "implicit_evolve.use_mass_matrices_jacobian=0",
            "implicit_evolve.mass_matrices_density_projection=0",
        ],
    ),
    ("eta-temperature-pmc", "temperature", 1, base + pmc),
    ("eta-temperature-pmc-mpi2", "temperature", 2, base + pmc),
]
for name, eta, ranks, options in cases:
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_case.py")),
        "--binary",
        str(a.binary.resolve()),
        "--output",
        str((a.output / name).resolve()),
        "--eta",
        eta,
        "--ranks",
        str(ranks),
        *options,
    ]
    subprocess.run(command, check=True)
