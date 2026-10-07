#!/usr/bin/env python3
"""Native source-guard matrix; failures retain their full logs and exports."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


def fields(line):
    return dict(re.findall(r"(\w+)=([^\s]+)", line))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launcher", default="")
    parser.add_argument("--seeds", type=int, default=0)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    executable = str(args.executable.resolve())
    inputs = str(Path(__file__).with_name("inputs_source_guard_rz").resolve())
    prefix = shlex.split(args.launcher)
    base = ["hybrid_pic_model.source_guard.export_interval=0"]
    cases = [
        ("dense", []),
        ("at_count", ["source_test.thermal=0", "source_test.stopping=0"]),
        ("above_count", ["hybrid_pic_model.source_guard.minimum_resident_markers=15"]),
        ("below_count", ["hybrid_pic_model.source_guard.minimum_resident_markers=17"]),
        ("cutoff8", ["hybrid_pic_model.source_guard.minimum_resident_markers=8"]),
        ("cutoff32", ["hybrid_pic_model.source_guard.minimum_resident_markers=32"]),
        ("vacuum_pedestal", ["source_test.n=0", "source_test.defect=0"]),
        ("below_density", ["source_test.n=0.999999e17", "source_test.defect=0"]),
        ("at_density", ["source_test.n=1.e17", "source_test.defect=0"]),
        ("above_density", ["source_test.n=1.000001e17", "source_test.defect=0"]),
        ("defect_above", ["source_test.bad_n=1.000001e17"]),
        ("unequal_weights", ["source_test.unequal=1"]),
        ("redirect", ["source_test.redirect=1", "source_test.thermal=0", "source_test.stopping=0"]),
        ("redirect_dust", ["source_test.redirect=1", "source_test.thermal=0", "source_test.stopping=0", "hybrid_pic_model.source_guard.minimum_resident_markers=17"]),
        ("signed_stopping", ["source_test.electron_drift=6.e6"]),
        ("changed_mask", ["source_test.changed_mask=1"]),
        ("off", ["hybrid_pic_model.source_guard.enabled=0"]),
        ("ohmic_only", ["hybrid_pic_model.source_guard.thermal_relaxation=0", "hybrid_pic_model.source_guard.alpha_stopping=0"]),
        ("thermal_only", ["hybrid_pic_model.source_guard.ohmic=0", "hybrid_pic_model.source_guard.alpha_stopping=0"]),
        ("stopping_only", ["hybrid_pic_model.source_guard.ohmic=0", "hybrid_pic_model.source_guard.thermal_relaxation=0"]),
    ]
    bad = [
        ("zero_count", "hybrid_pic_model.source_guard.minimum_resident_markers=0", "invalid threshold"),
        ("negative_density", "hybrid_pic_model.source_guard.n_min=-1", "invalid threshold"),
        ("negative_axis", "hybrid_pic_model.source_guard.axis_cells=-1", "invalid threshold"),
        ("all_axis", "hybrid_pic_model.source_guard.axis_cells=16", "single-level RZ"),
        ("unknown_species", "hybrid_pic_model.source_guard.background_species=missing", "unknown background"),
        ("duplicate_species", "hybrid_pic_model.source_guard.background_species=ions ions", "explicit and unique"),
        ("nonthermal_background", "hybrid_pic_model.source_guard.background_species=alpha", "thermal ion"),
    ]
    rows = []
    for name, opts in cases:
        case = out / name
        case.mkdir()
        command = prefix + [executable, inputs] + base + opts
        start = time.monotonic()
        with (case / "stdout.log").open("w") as log:
            result = subprocess.run(command, cwd=case, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "OMP_NUM_THREADS": "1"}, timeout=120)
        text = (case / "stdout.log").read_text()
        rows.append({"case": name, "command": command, "returncode": result.returncode,
                     "seconds": time.monotonic()-start,
                     "pass": result.returncode == 0 and "SOURCE_GUARD_PASS" in text,
                     "records": [fields(x) for x in text.splitlines()
                                 if x.startswith(("[source_guard_exchange]", "[source_guard_expectation]", "SOURCE_GUARD_PASS", "SOURCE_GUARD_STOPPING"))]})
        (out / "RESULTS.json").write_text(json.dumps(rows, indent=2)+"\n")
    for name, option, expected in bad:
        case = out / name
        case.mkdir()
        command = prefix + [executable, inputs] + base + [option]
        with (case / "stdout.log").open("w") as log:
            result = subprocess.run(command, cwd=case, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "OMP_NUM_THREADS": "1"}, timeout=120)
        text = (case / "stdout.log").read_text()
        rows.append({"case": name, "command": command, "returncode": result.returncode,
                     "pass": result.returncode != 0 and expected in text, "rejection": expected})
        (out / "RESULTS.json").write_text(json.dumps(rows, indent=2)+"\n")
    # Source sampling study: report measured systematic and stochastic residuals;
    # it is deliberately not a false exact-conservation gate.
    for ppc, dims in [(16, "2 2 4"), (64, "4 4 4"), (256, "8 4 8")]:
        for seed in range(args.seeds):
            case = out / f"thermal_ppc{ppc}_seed{seed}"
            case.mkdir()
            command = prefix + [executable, inputs] + base + [
                "source_test.stopping=0", "source_test.defect=0", f"source_test.ppc={ppc}",
                f"ions.num_particles_per_cell_each_dim={dims}",
                f"ions2.num_particles_per_cell_each_dim={dims}", f"warpx.random_seed={1024+seed}",
            ]
            with (case / "stdout.log").open("w") as log:
                result = subprocess.run(command, cwd=case, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, "OMP_NUM_THREADS": "1"}, timeout=120)
            text = (case / "stdout.log").read_text()
            rows.append({"case": case.name, "command": command, "returncode": result.returncode,
                         "pass": result.returncode == 0 and "SOURCE_GUARD_PASS" in text,
                         "ppc": ppc, "seed": seed,
                         "records": [fields(x) for x in text.splitlines() if x.startswith(("[source_guard_exchange]", "[source_guard_expectation]"))]})
            (out / "RESULTS.json").write_text(json.dumps(rows, indent=2)+"\n")
    failed = [r["case"] for r in rows if not r["pass"]]
    print(json.dumps({"cases": len(rows), "failed": failed, "output": str(out)}))
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
