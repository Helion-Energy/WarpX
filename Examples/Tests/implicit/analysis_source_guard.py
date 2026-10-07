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
import tempfile


def fields(line):
    return dict(re.findall(r"(\w+)=([^\s]+)", line))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launcher", default="")
    parser.add_argument("--seeds", type=int, default=0)
    parser.add_argument("--case", help="Run one named case (including expected rejections)")
    parser.add_argument("--new-attempt", action="store_true", help="Preserve a fresh attempt under --output")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.new_attempt)
    if args.new_attempt:
        out = Path(tempfile.mkdtemp(prefix="attempt-", dir=out))
    executable = str(args.executable.resolve())
    inputs = str(Path(__file__).with_name("inputs_source_guard_rz").resolve())
    prefix = shlex.split(args.launcher)
    base = ["hybrid_pic_model.source_guard.export_interval=0", "amrex.the_arena_init_size=67108864"]
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
        ("unequal_counts", ["ions2.num_particles_per_cell_each_dim=1 2 4", "source_test.ppc2=8"]),
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
    # Check independently configured physical cutoffs, both physical Z ends,
    # periodic wrapping, and current particle ownership across an MPI seam.
    for label, option in [("configured", "hybrid_pic_model.source_guard.n_min"),
                          ("ohmic", "hybrid_pic_model.joule_heating_n_min")]:
        for side, density in [("below", "1.999999e17"), ("at", "2.e17"),
                              ("above", "2.000001e17")]:
            cases.append((f"{label}_floor_{side}", [f"{option}=2.e17",
                         f"source_test.n={density}", "source_test.defect=0"]))
    nonperiodic = ["boundary.field_lo=none pec", "boundary.field_hi=none pec",
                  "boundary.particle_lo=none reflecting", "boundary.particle_hi=reflecting reflecting"]
    for j in [0, 9, 16]:
        cases.append((f"nonperiodic_defect{j}", nonperiodic+[f"source_test.defect_j={j}"]))
    cases += [
        ("periodic_wrap_defect", ["source_test.defect_j=0"]),
        ("axis_disabled", ["hybrid_pic_model.source_guard.axis_cells=0"]),
        ("axis_two_cells", ["hybrid_pic_model.source_guard.axis_cells=2"]),
        ("empty_background", ["ions.injection_style=none", "ions2.injection_style=none",
                              "source_test.ppc=0", "source_test.n=0", "source_test.defect=0"]),
        ("moving_count_epoch", ["source_test.move_epoch=1", "source_test.defect=0",
                                "hybrid_pic_model.source_guard.diagnostic_interval=1000"]),
        ("nonperiodic_moving_epoch", nonperiodic+["source_test.move_epoch=1", "source_test.defect=0"]),
    ]
    invalid_data = [
        ("thermal_nan", ["source_test.bad_ti=1", "source_test.stopping=0"],
         "invalid eligible relaxation moments"),
        ("thermal_negative_rate", ["hybrid_pic_model.electron_ion_relaxation_rate(rho,Te,Ti,t)=-1",
                                   "source_test.stopping=0"], "invalid eligible relaxation rate"),
        ("ohmic_nan", ["source_test.bad_current=1", "source_test.thermal=0", "source_test.stopping=0"],
         "invalid eligible Ohmic current"),
        ("stopping_nan_te", ["source_test.bad_stopping_te=1", "source_test.thermal=0"],
         "invalid eligible stopping thermodynamics"),
        ("stopping_nan_velocity", ["source_test.bad_stopping_ve=1", "source_test.thermal=0"],
         "invalid eligible stopping velocity or particle"),
    ]
    for name, options, _ in invalid_data:
        cases.append((f"masked_{name}", options+["hybrid_pic_model.source_guard.minimum_resident_markers=17"]))
    bad = [
        ("zero_count", "hybrid_pic_model.source_guard.minimum_resident_markers=0", "invalid threshold"),
        ("negative_density", "hybrid_pic_model.source_guard.n_min=-1", "invalid threshold"),
        ("negative_axis", "hybrid_pic_model.source_guard.axis_cells=-1", "invalid threshold"),
        ("all_axis", "hybrid_pic_model.source_guard.axis_cells=16", "single-level RZ"),
        ("unknown_species", "hybrid_pic_model.source_guard.background_species=missing", "unknown background"),
        ("duplicate_species", "hybrid_pic_model.source_guard.background_species=ions ions", "explicit and unique"),
        ("nonthermal_background", "hybrid_pic_model.source_guard.background_species=alpha", "thermal ion"),
    ]
    bad = [(name, [option], expected) for name, option, expected in bad]
    bad += [(f"eligible_{name}", options, expected) for name, options, expected in invalid_data]
    if args.case:
        names = {x[0] for x in cases+bad}
        if args.case not in names or args.seeds:
            parser.error("--case requires a known case and --seeds=0")
        cases = [x for x in cases if x[0] == args.case]
        bad = [x for x in bad if x[0] == args.case]
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
                                 if x.startswith(("[source_guard_exchange]", "[source_guard_expectation]", "[source_guard_skipped]", "SOURCE_GUARD_PASS", "SOURCE_GUARD_STOPPING"))]})
        (out / "RESULTS.json").write_text(json.dumps(rows, indent=2)+"\n")
    for name, options, expected in bad:
        case = out / name
        case.mkdir()
        command = prefix + [executable, inputs] + base + options
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
