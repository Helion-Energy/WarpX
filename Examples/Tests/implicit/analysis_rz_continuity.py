#!/usr/bin/env python3
"""Native deposited continuity: historical negative control and compatible filter.

Every case uses WarpX deposition, volume scaling, filtering, guard sums and BCs.
The independent invariants are charge balance and zero continuity residual for
source-free constant-momentum trajectories, including an axis crossing.
"""
import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


def maps(path):
    result = {}
    for source in sorted(path.glob("*_rank*.csv")):
        name = source.stem.split("_rank")[0]
        values = result.setdefault(name, {})
        with source.open() as stream:
            for row in csv.DictReader(stream):
                point = int(row["i"]), int(row["j"])
                value = float(row["value"])
                assert math.isfinite(value)
                if point in values:
                    assert values[point] == value, (source, point, "inconsistent shared node")
                values[point] = value
    return result


def run(executable, inputs, parent, label, args, mpi=None):
    path = parent / label
    path.mkdir()
    command = ([mpi, "-n", "2"] if mpi else []) + [str(executable), str(inputs)] + args
    (path / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    with (path / "run.log").open("w") as log:
        process = subprocess.run(command, cwd=path, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=60)
    rows = [json.loads(line.split("CONTINUITY_TEST ", 1)[1])
            for line in (path / "run.log").read_text().splitlines()
            if line.startswith("CONTINUITY_TEST ")]
    assert process.returncode == 0 and len(rows) == 1, (label, process.returncode, rows)
    return rows[0], maps(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--mpi-exec")
    args = parser.parse_args()
    args.output_parent.mkdir(parents=True, exist_ok=True)
    parent = Path(tempfile.mkdtemp(prefix="matrix-", dir=args.output_parent))
    records = []
    reference = {}
    zero_pass = {}
    for axis, passes, boxes in itertools.product((0, 1), (0, 1, 2), (32, 8)):
        for mode in (0, 1, 2):
            label = f"axis{axis}-pass{passes}-box{boxes}-mode{mode}"
            options = [f"boundary.verboncoeur_axis_correction={axis}",
                       f"warpx.filter_npass_each_dir={passes} {passes}",
                       f"amr.max_grid_size={boxes}",
                       f"warpx.use_filter={int(mode != 0)}",
                       f"warpx.rz_continuity_filter={int(mode == 2)}",
                       f"continuity_test.mode={mode}", "continuity_test.write_maps=1"]
            row, data = run(args.executable.resolve(), args.inputs.resolve(), parent, label, options)
            row.update(axis=axis, passes=passes, boxes=boxes, mode=mode, label=label)
            row["rho_min"] = min(min(data[name].values()) for name in ("rho_old", "rho_new"))
            if mode != 1:
                assert row["rho_min"] >= 0., (label, "negative filtered density")
            if mode == 1 and axis == 0 and passes == 1:
                assert row["rho_min"] < 0., (label, "missing scalar positivity counterexample")
            records.append(row)
            # Counts/charge are unchanged by native filtering, to arithmetic precision.
            assert abs(row["new_charge_C"] / row["old_charge_C"] - 1) < 2.e-13, row
            key = axis, passes, mode
            if boxes == 32:
                reference[key] = data
            else:
                for name in data:
                    a, b = data[name], reference[key][name]
                    assert a.keys() == b.keys()
                    scale = max(abs(v) for v in b.values())
                    if name == "residual":
                        scale = row["rho_scale"] / 1.e-9
                    assert max(abs(a[k] - b[k]) for k in a) <= 2.e-13 * max(scale, 1.e-100), (label, name)
            if passes == 0:
                if mode == 0:
                    zero_pass[axis, boxes] = data
                else:
                    assert data == zero_pass[axis, boxes], (label, "zero-pass control changed")
            if mode == 1 and passes > 0:
                assert row["relative_step_residual"] > 1.e-5, (label, "missing negative control")
            print(label, row["relative_step_residual"], flush=True)
            if args.mpi_exec and boxes == 8 and passes == 2:
                parallel, pmap = run(args.executable.resolve(), args.inputs.resolve(), parent,
                                    label + "-mpi2", options, args.mpi_exec)
                records.append(dict(parallel, axis=axis, passes=passes, boxes=boxes,
                                    mode=mode, label=label + "-mpi2"))
                assert pmap == data, (label, "MPI changed fields")
    profiles = {
        "wrap": ["ions.multiple_particles_pos_z=.0119 .0159 .0079 .0159 .0319",
                 "ions.multiple_particles_uz=.0005 -.0004 .0007 .001 .001"],
        "rwall": ["ions.multiple_particles_pos_x=.0001 .0007 .0079 .0159 .0318",
                  "ions.multiple_particles_pos_y=0 0 0 0 0"],
        "zwall": ["boundary.field_lo=none pec", "boundary.field_hi=pec pec",
                  "boundary.particle_lo=none reflecting",
                  "boundary.particle_hi=reflecting reflecting",
                  "ions.multiple_particles_pos_z=.0119 .0159 .0079 .0159 .0318"],
    }
    profiles["rwall_absorb"] = profiles["rwall"] + ["boundary.particle_hi=absorbing periodic"]
    profiles["zwall_absorb"] = profiles["zwall"] + ["boundary.particle_lo=none absorbing",
                                                    "boundary.particle_hi=absorbing absorbing"]
    profiles["corner"] = profiles["zwall"] + profiles["rwall"]
    profiles["pmc"] = profiles["rwall"] + ["boundary.field_hi=pmc periodic"]
    for profile, changes in profiles.items():
        for mode in (0, 1, 2):
            label = f"boundary-{profile}-mode{mode}"
            options = ["amr.max_grid_size=8", f"warpx.use_filter={int(mode != 0)}",
                       f"warpx.rz_continuity_filter={int(mode == 2)}",
                       f"continuity_test.mode={mode}"] + changes
            row, _ = run(args.executable.resolve(), args.inputs.resolve(), parent, label, options)
            if "absorb" not in profile:
                assert abs(row["new_charge_C"] / row["old_charge_C"] - 1) < 2.e-13, row
            # PEC image charge with absorbing particles has physical boundary
            # flux. Its local continuity is tested; its inventory need not be constant.
            records.append(dict(row, label=label))
    (parent / "RESULT.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"PASS: {len(records)} native cases; evidence {parent}")


if __name__ == "__main__":
    main()
