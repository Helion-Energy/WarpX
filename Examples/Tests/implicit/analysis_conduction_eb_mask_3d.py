#!/usr/bin/env python3
"""Native covered-temperature independence; preserve per-case numerical receipts."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


def arguments(case):
    geometry = {
        "x_plane": "x-0.03125",
        "y_plane": "y-0.03125",
        "z_plane": "z-0.03125",
        "oblique_plane": "x+0.7*y+0.3*z-0.03125",
        "cylinder_z": "sqrt(x*x+y*y)-0.7",
        "sphere": "sqrt(x*x+y*y+z*z)-0.7",
    }
    orientation = case.get("field_orientation", "normal")
    b = {"normal": "1 0 0", "tangent": "0 0 1", "oblique_xyz": "1 1 1"}[orientation]
    ratio = case.get("anisotropy", 1)
    wall = case.get("wall", "isothermal_capped")
    # T01 retains its prescribed 300/3000 K pair. Cap and anisotropy rows
    # exercise the production 500 eV wall rather than a room-temperature proxy.
    temperature = (
        300.0 if case["family"] == "T01" else 500 * 1.602176634e-19 / 1.380649e-23
    )
    result = [
        f"warpx.eb_implicit_function={geometry[case['geometry']]}",
        f"hybrid_pic_model.qdsmc_conduction_fd_order={case['fd_order']}",
        f"hybrid_pic_model.qdsmc_conduction_fd_time={case['integrator']}",
        f"eb_test.temperature_K={temperature:.17g}",
        f"eb_test.magnetic={b}",
        "eb_test.amplitude=" + ("0" if case["profile"] == "uniform" else "0.2"),
        "eb_test.require_invariance=1",
        f"hybrid_pic_model.qdsmc_kappa_perp(n,Te,t)=1.5*1.380649e-23*n/{ratio}",
        "amrex.the_arena_init_size=0",
        "amrex.the_arena_is_managed=0",
    ]
    if wall != "adiabatic":
        result.extend(
            [
                "hybrid_pic_model.qdsmc_conduction_eb_bc=isothermal",
                f"hybrid_pic_model.qdsmc_conduction_eb_Te(x,y,z)={temperature:.17g}*1.380649e-23/1.602176634e-19",
                f"hybrid_pic_model.qdsmc_conduction_eb_flux_limit={case.get('wall_flux_cap', 0)}",
                f"hybrid_pic_model.qdsmc_conduction_eb_ring={case.get('wall_ring', 2)}",
            ]
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path(__file__).with_name("inputs_test_3d_conduction_eb_mask"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("conduction_eb_mask_cases.json"),
    )
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--launcher-json", default="[]")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--write-fields", action="store_true")
    args = parser.parse_args()
    assert 0 <= args.shard_index < args.shards
    launcher = json.loads(args.launcher_json)
    assert isinstance(launcher, list) and all(isinstance(s, str) for s in launcher)
    cases = json.loads(args.cases.read_text())["cases"]
    assert len(set(c["id"] for c in cases)) == len(cases)
    selected = cases[args.shard_index :: args.shards]
    assert selected
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="attempt-", dir=args.output_parent))
    receipts = []
    failed = []
    for case in selected:
        cwd = output / case["id"]
        cwd.mkdir()
        command = [
            *launcher,
            str(args.executable.resolve()),
            str(args.inputs.resolve()),
            *arguments(case),
        ]
        if args.write_fields:
            command.append("eb_test.output=" + str(cwd / "temperature"))
        (cwd / "COMMAND.json").write_text(json.dumps(command, indent=2) + "\n")
        with (cwd / "stdout.log").open("x") as log:
            try:
                proc = subprocess.run(
                    command,
                    cwd=cwd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "OMP_NUM_THREADS": "1"},
                    timeout=180,
                )
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                returncode = -999
        records = [
            json.loads(line.split("EB_MASK_RESULT ", 1)[1])
            for line in (cwd / "stdout.log").read_text().splitlines()
            if line.startswith("EB_MASK_RESULT ")
        ]
        passed = returncode == 0 and len(records) == 1 and records[0]["passed"] is True
        receipt = dict(
            case=case,
            command=command,
            returncode=returncode,
            results=records,
            passed=passed,
        )
        (cwd / "RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
        receipts.append(receipt)
        if not passed:
            failed.append(case["id"])
        print(case["id"], "PASS" if passed else "FAIL", flush=True)
    report = dict(
        passed=not failed,
        failed=failed,
        expected=len(selected),
        completed=len(receipts),
        executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
        inputs_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        results=receipts,
    )
    (output / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    if failed:
        raise SystemExit(
            f"Covered-temperature regression failures: {failed}; evidence: {output}"
        )
    print(f"EB_MATRIX_PASS {len(receipts)} {output}", flush=True)


if __name__ == "__main__":
    main()
