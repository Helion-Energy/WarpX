#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Native RZ common-state viscous-source and instrumentation A/A control.

The registered control is double-precision CPU. An optional JSON MPI prefix adds
2-rank replays of the same serial-produced particles, markers and full field FABs.
No production equation, input physics, source tolerance or raw marker is changed.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from viscous_common_state_io import fields_compare, mf, particle_compare, particles

QE = 1.602176634e-19
KB = 1.380649e-23


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def complete_state(a, b):
    result = {
        "from": str(a),
        "to": str(b),
        "fields": fields_compare(a, b),
        "exact_clock": (a / "clock.txt").read_bytes() == (b / "clock.txt").read_bytes(),
        "particles": [particle_compare(a, b, p) for p in ["particles_ions", "markers"]],
    }
    result["passed"] = all(
        x["exact_all_fabs_with_ghosts"] for x in result["fields"]
    ) and all(
        x["exact_ids"] and x["exact_all_real"] and x["exact_all_int"] and x["finite"]
        for x in result["particles"]
    )
    return result


def nodal(path):
    """Unique valid RZ nodes; drop the periodic high-z image, not radial walls."""
    out = np.full((65, 33), np.nan)
    for (lo, hi, typ), fab in mf(path).items():
        assert typ == (1, 1)
        ir = slice(lo[0], hi[0] + 1)
        iz = slice(lo[1], hi[1] + 1)
        lr, lz = lo[0] - fab["fab_lo"][0], lo[1] - fab["fab_lo"][1]
        data = fab["array"][0, lz : lz + hi[1] - lo[1] + 1, lr : lr + hi[0] - lo[0] + 1]
        prior = out[iz, ir]
        existing = np.isfinite(prior)
        assert np.array_equal(prior[existing], data[existing]), "valid overlap differs"
        out[iz, ir] = data
    assert np.isfinite(out).all()
    assert np.array_equal(out[0], out[-1]), "periodic nodal images differ"
    return out[:-1]


def masks_and_volume():
    radius = np.arange(33) * (0.04 / 32)
    low = np.maximum(0.0, radius - 0.04 / 64)
    high = np.minimum(0.04, radius + 0.04 / 64)
    volume = np.broadcast_to(np.pi * (high**2 - low**2) * (0.08 / 64), (64, 33))
    masks = {
        "all": np.ones(33, dtype=bool),
        "axis4": np.arange(33) < 4,
        "outer4": np.arange(33) >= 29,
        "interior": (np.arange(33) >= 4) & (np.arange(33) < 29),
    }
    return masks, volume


def temperature_norms(a, b):
    name = "hybrid_electron_temperature_fp[level=0]"
    delta = (nodal(b / "fields" / name) - nodal(a / "fields" / name)) * KB / QE
    masks, vol = masks_and_volume()
    return {
        name: {
            "volume_rms_eV": float(
                np.sqrt(
                    np.sum(vol[:, mask] * delta[:, mask] ** 2) / np.sum(vol[:, mask])
                )
            ),
            "point_rms_eV": float(np.sqrt(np.mean(delta[:, mask] ** 2))),
            "max_abs_eV": float(np.max(abs(delta[:, mask]))),
            "volume_fraction": float(np.sum(vol[:, mask]) / np.sum(vol)),
        }
        for name, mask in masks.items()
    }


def source_bracket(case):
    records = [
        json.loads(x.split(" ", 1)[1])
        for x in (case / "run.log").read_text().splitlines()
        if x.startswith("V1_BRACKET ")
    ]
    assert len(records) == 1
    record = records[0]
    assert record["instrument"] == int(case.name[-1])
    assert (case / "before/clock.txt").read_bytes() == (
        case / "after/clock.txt"
    ).read_bytes()
    assert record["dt_s"] == 0.5 * float(
        (case / "before/clock.txt").read_text().split()[2]
    )
    before, after = case / "before/fields", case / "after/fields"
    temp = "hybrid_electron_temperature_fp[level=0]"
    rho = nodal(before / "rho_fp[level=0]")
    assert np.array_equal(rho, nodal(after / "rho_fp[level=0]"))
    ne = rho / QE
    assert np.all(ne > 5e18), "this fixture has no floor-band cells"
    te0, te1 = nodal(before / temp), nodal(after / temp)
    u0 = 1.5 * ne * KB * te0
    # Form the local difference first, independently of subtracting two global sums.
    du = 1.5 * ne * KB * (te1 - te0)
    work = record["dt_s"] * nodal(after / "hybrid_qdsmc_visc_work_fp[level=0]")
    strain = record["dt_s"] * nodal(after / "hybrid_qdsmc_visc_heating_fp[level=0]")
    masks, volume = masks_and_volume()
    regions = {}
    for name, mask in masks.items():
        v = volume[:, mask]
        initial = float(np.sum(v * u0[:, mask]))
        observed = float(np.sum(v * du[:, mask]))
        expected = float(np.sum(v * work[:, mask]))
        bound = 128 * np.finfo(float).eps * abs(initial) + 2e-11 * abs(expected)
        regions[name] = {
            "U0_J": initial,
            "local_sum_dU_J": observed,
            "nodal_work_J": expected,
            "strain_J": float(np.sum(v * strain[:, mask])),
            "residual_J": observed - expected,
            "bound_J": bound,
            "passed": bool(abs(observed - expected) <= bound),
        }
    assert record["clamp_J"] == 0.0, "nonzero clamp needs a spatial source attribution"
    assert all(x["passed"] for x in regions.values())
    assert abs(regions["all"]["local_sum_dU_J"] - record["dU_J"]) <= record["bound_J"]
    assert (
        abs(regions["all"]["nodal_work_J"] - record["nodal_work_J"])
        <= record["bound_J"]
    )
    assert abs(regions["all"]["strain_J"] - record["strain_J"]) <= record["bound_J"]
    if record["instrument"]:
        assert abs(record["dU_J"] - record["staggered_work_J"]) <= record["bound_J"]
        assert (
            abs(record["nodal_work_J"] - record["staggered_work_J"])
            <= record["bound_J"]
        )
        assert abs(record["native_bracket_J"] - record["dU_J"]) <= record["bound_J"]
    unchanged = [
        particle_compare(case / "before", case / "after", p)
        for p in ["particles_ions", "markers"]
    ]
    assert all(
        x["exact_ids"] and x["exact_all_real"] and x["exact_all_int"] for x in unchanged
    )
    return {
        "native": record,
        "regions": regions,
        "unchanged_particles_and_markers": unchanged,
    }


def analyze(root):
    common = root / "producer/common"
    ids, ion_data, _ = particles(common, "particles_ions")
    marker_ids, marker_data, _ = particles(common, "markers")
    assert len(ids) == 131072 and len(marker_ids) == 2112
    assert ion_data.shape[0] == 7 and marker_data.shape[0] == 16
    cases = sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith(("source-", "advance-"))
    )
    expected = {
        f"{mode}-serial-i{instrument}"
        for mode in ["source", "advance"]
        for instrument in [0, 1]
    }
    if any("mpi2" in p.name for p in cases):
        expected |= {
            f"{mode}-mpi2-i{instrument}"
            for mode in ["source", "advance"]
            for instrument in [0, 1]
        }
    assert {p.name for p in cases} == expected
    producer_receipt = json.loads((root / "producer/RESULT.json").read_text())
    assert producer_receipt["exit_code"] == 0
    for case in cases:
        receipt = json.loads((case / "RESULT.json").read_text())
        assert receipt["exit_code"] == 0
        assert (
            receipt["input_sha256"]
            == producer_receipt["input_sha256"]
            == sha(case / "inputs")
        )
    loads = {p.name: complete_state(common, p / "before") for p in cases}
    assert all(x["passed"] for x in loads.values())
    source = {p.name: source_bracket(p) for p in cases if p.name.startswith("source-")}
    aa = {}
    for layout in ["serial", "mpi2"]:
        for mode in ["source", "advance"]:
            a, b = root / f"{mode}-{layout}-i0", root / f"{mode}-{layout}-i1"
            if a.exists():
                result = complete_state(a / "after", b / "after")
                assert result[
                    "passed"
                ], f"instrumentation changes state: {mode}/{layout}"
                result["temperature_norms"] = temperature_norms(
                    a / "after", b / "after"
                )
                aa[f"{mode}-{layout}"] = result
        a = root / f"advance-{layout}-i0"
        if a.exists():
            start = (a / "before/clock.txt").read_text().split()
            end = (a / "after/clock.txt").read_text().split()
            assert int(start[0]) == 20 and int(end[0]) == 30
            assert float(end[1]) > float(start[1])
            assert not complete_state(a / "before", a / "after")[
                "passed"
            ], "continuation did no work"
    layout_comparison = None
    if (root / "advance-mpi2-i1").exists():
        assert complete_state(
            root / "source-serial-i1/after", root / "source-mpi2-i1/after"
        )["passed"]
        # Different reduction layouts need not be bitwise identical after evolution.
        # This reports measured differences; it does not replace an A/A threshold.
        layout_comparison = temperature_norms(
            root / "advance-serial-i1/after", root / "advance-mpi2-i1/after"
        )
    return {
        "passed": True,
        "scope": "common-state CPU DP native source/instrumentation gate only",
        "counts": {"ions": len(ids), "raw_markers": len(marker_ids)},
        "loads": loads,
        "source_brackets": source,
        "instrumentation_AA": aa,
        "continuation_layout_temperature_differences": layout_comparison,
        "norms": "Volume RMS uses clipped physical nodal annuli times dz; point RMS uses unique valid nodes. Periodic high-z image excluded. axis4=i0..3,outer4=i29..32,interior=i4..28.",
        "limitations": [
            "No production equation/input physics change",
            "Only registered field FABs and particle/marker attributes are serialized; uncheckpointed solver/cache state is reinitialized equally in both arms, not matched to an uninterrupted trajectory",
            "End-step raw marker entropy, weight and velocity are natively reset to zero",
            "Qnu strain diagnostic is not added to the separately booked J.Evisc work",
            "Drag field is excluded from ion push; this does not certify full plasma energy conservation",
            "No paired FRC physics attribution or second-order full-PIC closure",
            "CUDA, single and mixed precision replay are unqualified",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path(__file__).with_name("inputs_viscous_common_state"),
    )
    parser.add_argument(
        "--mpi-prefix-json",
        default="[]",
        help='optional argv such as ["mpiexec","-n","2"]',
    )
    parser.add_argument("--analyze-existing", action="store_true")
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="preserve prior CTest evidence in a fresh timestamped subdirectory",
    )
    args = parser.parse_args()
    root = args.output.resolve()
    if args.new_run:
        assert not args.analyze_existing
        root /= datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
    if not args.analyze_existing:
        assert args.executable is not None
        executable = args.executable.resolve()
        root.mkdir(parents=True, exist_ok=False)
        prefix = json.loads(args.mpi_prefix_json)
        assert isinstance(prefix, list) and all(isinstance(x, str) for x in prefix)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

        def run(tag, options, launcher=()):
            case = root / tag
            case.mkdir()
            (case / "inputs").write_bytes(args.inputs.read_bytes())
            command = [
                *launcher,
                str(executable),
                str(case / "inputs"),
                "ions.num_particles_per_cell=64",
                "amr.max_grid_size=16",
                *options,
            ]
            receipt = {
                "command": command,
                "cwd": str(case),
                "exe_sha256": sha(executable),
                "input_sha256": sha(case / "inputs"),
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            (case / "COMMAND.json").write_text(json.dumps(receipt, indent=2) + "\n")
            start = time.monotonic()
            with (case / "run.log").open("w") as log:
                completed = subprocess.run(
                    command,
                    cwd=case,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=90,
                )
            receipt.update(
                exit_code=completed.returncode, wall_s=time.monotonic() - start
            )
            (case / "RESULT.json").write_text(json.dumps(receipt, indent=2) + "\n")
            assert completed.returncode == 0, (case / "run.log").read_text()[-10000:]

        run(
            "producer",
            [
                "v1.mode=0",
                "v1.steps=20",
                "v1.instrument=1",
                "diagnostics.enable=1",
                "diagnostics.diags_names=checkpoint",
                "checkpoint.diag_type=Full",
                "checkpoint.format=checkpoint",
                "checkpoint.intervals=20",
                "checkpoint.file_prefix=native_chk",
                "checkpoint.file_min_digits=6",
            ],
        )
        layouts = [("serial", [])] + ([("mpi2", prefix)] if prefix else [])
        for layout, launcher in layouts:
            for mode, steps, tag in [(1, 0, "source"), (2, 10, "advance")]:
                for instrument in [0, 1]:
                    run(
                        f"{tag}-{layout}-i{instrument}",
                        [
                            f"v1.mode={mode}",
                            f"v1.steps={steps}",
                            f"v1.instrument={instrument}",
                            f"v1.restore={root}/producer/common",
                            f"amr.restart={root}/producer/native_chk000020",
                            f"hybrid_pic_model.qdsmc_energy_budget={instrument}",
                        ],
                        launcher,
                    )
    result = analyze(root)
    (root / "ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "analysis": str(root / "ANALYSIS.json"),
                "loads": len(result["loads"]),
                "source_brackets": len(result["source_brackets"]),
                "instrumentation_AA": len(result["instrumentation_AA"]),
            }
        )
    )


if __name__ == "__main__":
    main()
