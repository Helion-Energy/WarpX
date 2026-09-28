#!/usr/bin/env python3
"""Independent fixed-time orbit gates; retain every run in a fresh directory."""

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path


def check_ladder(rows):
    assert len(rows) == 5
    rows = sorted(rows, key=lambda x: x["steps_per_period"])
    assert [x["steps_per_period"] for x in rows] == [16, 32, 64, 128, 256]
    kind = rows[0]["case"]
    final_time = rows[0]["final_time_s"]
    for row in rows:
        assert row["case"] == kind and row["final_time_s"] == final_time
        assert row["unconverged_particles"] == 0
        assert row["final_gather_support_valid"] and row["endpoint_inside"]
        assert row["native_current_samples"] == 3
        assert row["zero_component_work_exact"]
        assert row["max_continuity_relative"] < 1.0e-10
        assert row["max_electric_work_defect_over_initial_K"] < 1.0e-12
        assert row["inner_iterations_max"] <= 50
        assert sum(row["inner_histogram"]) == 12 * row["steps"]
        assert row["min_boundary_distance_m"] > 0.0
        for value in row.values():
            if isinstance(value, float):
                assert math.isfinite(value)
        if kind == "gyro":
            assert row["max_energy_relative"] < 1.0e-10
            assert row["max_guiding_center_relative"] < 1.0e-10
            assert row["max_discrete_phase_defect_rad"] < 5.0e-10
        elif kind == "circular":
            assert row["max_total_energy_relative"] < 1.0e-8
    if kind == "gyro":
        key = "max_phase_error_rad"
        assert rows[-1][key] < 0.002
    elif kind == "circular":
        key = "max_position_relative"
        assert rows[-1][key] < 0.002
    else:
        key = "max_velocity_relative"
        assert rows[-1][key] < 1.0e-5
    ratio = rows[-2][key] / rows[-1][key]
    assert 3.5 < ratio < 4.5, (kind, key, ratio)
    return {
        "case": kind,
        "error_metric": key,
        "finest_error": rows[-1][key],
        "last_refinement_ratio": ratio,
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("case", choices=["gyro", "circular", "drift"])
    args = parser.parse_args()
    executable, inputs = args.executable.resolve(), args.inputs.resolve()
    directory = Path(tempfile.mkdtemp(prefix="alpha-orbits-", dir=Path.cwd()))
    records = []
    for n in [16, 32, 64, 128, 256]:
        work = directory / f"n{n}"
        work.mkdir()
        command = [
            str(executable),
            str(inputs),
            f"orbit.case={args.case}",
            f"orbit.steps_per_period={n}",
            "amr.max_grid_size=8",
        ]
        with (work / "run.log").open("w") as stream:
            process = subprocess.run(
                command,
                cwd=work,
                env=dict(os.environ, OMP_NUM_THREADS="1"),
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=60,
                check=False,
            )
        text = (work / "run.log").read_text()
        assert process.returncode == 0, (command, process.returncode, text[-3000:])
        results = [
            json.loads(line.split(" ", 1)[1])
            for line in text.splitlines()
            if line.startswith("ALPHA_ORBIT_RESULT ")
        ]
        assert len(results) == 1
        records.append(results[0])
    (directory / "RESULTS.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(check_ladder(records), sort_keys=True))


if __name__ == "__main__":
    main()
