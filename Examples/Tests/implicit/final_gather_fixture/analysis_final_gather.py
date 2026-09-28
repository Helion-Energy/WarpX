#!/usr/bin/env python3
"""Native final Boris gather: capture/default-off parity, crop, guards, checkpoint."""

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("executable", type=Path)
p.add_argument("input", type=Path)
p.add_argument("case", choices=["pair", "crop", "iterated", "negative"])
p.add_argument("--mode", default="decoupled_jfnk")
p.add_argument("--grid", type=int, default=8)
p.add_argument("--ranks", type=int, default=1)
p.add_argument("--cells", type=int, default=8)
p.add_argument("--baseline", type=Path)
a = p.parse_args()
exe = a.executable.resolve()
deck = a.input.resolve()
rz = "_rz_" in deck.name
# Optional MPI replays use the caller's launcher and vendor-specific flags.
prefix = []
if a.ranks != 1:
    prefix = shlex.split(os.environ.get("MPIEXEC", "mpiexec"))
    prefix += ["-n", str(a.ranks)]
    prefix += shlex.split(os.environ.get("MPIEXEC_PREFLAGS", ""))

common = [
    str(deck),
    "max_step=0",
    "amr.n_cell=" + (" ".join([str(a.cells)] * (2 if rz else 3))),
    "amr.blocking_factor=4",
    "amr.max_grid_size=" + str(a.grid),
    "amr.refine_grid_layout=0",
    "warpx.const_dt=2.e-10",
    "diagnostics.enable=0",
    "hybrid_pic_model.electron_energy_mode=" + a.mode,
]
results = []


def run(name, extra, expected=None, binary=exe):
    cwd = Path(tempfile.mkdtemp(prefix=name + "-", dir=Path.cwd())).absolute()
    cmd = prefix + [str(binary)] + common + extra
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1")),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    (cwd / "run.log").write_text(proc.stdout)
    rec = dict(command=cmd, returncode=proc.returncode, cwd=str(cwd), expected=expected)
    (cwd / "command.json").write_text(json.dumps(rec, indent=2) + "\n")
    results.append(rec)
    if expected:
        normalized = " ".join(proc.stdout.replace("\n#", " ").split())
        assert proc.returncode != 0 and expected in normalized, (
            name,
            proc.returncode,
            proc.stdout[-2500:],
        )
    else:
        assert proc.returncode == 0 and "FINAL_GATHER_PASS" in proc.stdout, (
            name,
            proc.returncode,
            proc.stdout[-2500:],
        )
    return cwd


def physics(path):
    return sorted(
        line
        for f in path.glob("physics-rank*.txt")
        for line in f.read_text().splitlines()
    )


on = ["implicit_evolve.thermal.audit_ion_electric_work=1"]
off = ["implicit_evolve.thermal.audit_ion_electric_work=0"]
if a.case == "negative":
    for name, key, message in [
        ("no_push", "ions.do_not_push=1", "does not support do_not_push"),
        ("no_gather", "ions.do_not_gather=1", "does not support do_not_gather"),
        (
            "field_ionization",
            "ions.do_field_ionization=1",
            "does not support field ionization",
        ),
        (
            "radiation",
            ["ions.species_type=electron", "ions.do_classical_radiation_reaction=1"],
            "classical radiation momentum changes",
        ),
        (
            "variable_charge",
            "gather_test.negative=variable_charge",
            "requires fixed species charge",
        ),
        (
            "direct_suborbit",
            "gather_test.negative=direct_suborbit",
            "does not support direct suborbit advancement",
        ),
        (
            "wall",
            "implicit_evolve.reflect_particles_at_rmax=1",
            "wall reflection momentum changes",
        ),
        (
            "suborbits",
            "implicit_evolve.particle_suborbits=1",
            "Implicit conservative density requires",
        ),
        (
            "late_no_push",
            "gather_test.negative=late_no_push",
            "does not support do_not_push",
        ),
        (
            "late_no_gather",
            "gather_test.negative=late_no_gather",
            "does not support do_not_gather",
        ),
        (
            "late_wall",
            "gather_test.negative=late_reflection",
            "wall reflection momentum changes",
        ),
        (
            "late_suborbit",
            "gather_test.negative=late_suborbit",
            "does not support particle suborbits",
        ),
        (
            "zero_iterations",
            "gather_test.negative=zero_iterations",
            "nonempty Boris shape3 MC iteration",
        ),
    ]:
        run(name, on + ([key] if isinstance(key, str) else key), message)
    run(
        "legacy",
        on + ["hybrid_pic_model.electron_energy_mode=legacy"],
        "requires an Eulerian electron energy mode",
    )
else:
    options = []
    if a.case == "crop":
        options = ["particles.crop_on_PEC_boundary=1", "gather_test.cropped=1"]
        if not rz:
            options += [
                "boundary.field_lo=pec pec pec",
                "boundary.field_hi=pec pec pec",
                "boundary.particle_lo=reflecting reflecting reflecting",
                "boundary.particle_hi=reflecting reflecting reflecting",
            ]
    if a.case == "iterated":
        options += ["gather_test.iterations=50"]
    disabled = run("off", off + options)
    enabled = run(
        "on", on + options + ["diagnostics.enable=1", "gather_test.checkpoint=1"]
    )
    assert physics(disabled) == physics(enabled), "capture changed physical particles"
    headers = list(enabled.glob("checkpoint*/ions/Header"))
    assert headers, "native full checkpoint missing"
    for header in headers:
        assert "implicit_final_gather" not in header.read_text(), (
            "scratch leaked into native checkpoint"
        )
    if a.baseline:
        baseline = run("baseline", off + options, binary=a.baseline.resolve())
        assert physics(baseline) == physics(disabled), (
            "default-off changed baseline physical update"
        )
Path("RESULT.json").write_text(
    json.dumps(
        dict(
            pass_all=True,
            case=a.case,
            mode=a.mode,
            grid=a.grid,
            ranks=a.ranks,
            results=results,
        ),
        indent=2,
    )
    + "\n"
)
print("FINAL_GATHER_GROUP_PASS", a.case, a.mode, a.grid, a.ranks)
