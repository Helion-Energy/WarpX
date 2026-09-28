#!/usr/bin/env python3
"""Exercise accepted floor updates and full native controller checkpoint replay."""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("mode", choices=["decoupled_jfnk", "coupled_jfnk"])
    parser.add_argument("pedestal", choices=["tracked", "static", "none"])
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="density-control-", dir=Path.cwd()))
    common = [f"hybrid_pic_model.electron_energy_mode={args.mode}"]
    if args.pedestal != "tracked":
        common.append("hybrid_pic_model.density_pedestal_track_floor=0")
    if args.pedestal == "none":
        common.append("hybrid_pic_model.density_pedestal=0")
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"

    def run(label, options, rejection=None):
        directory = root / label
        directory.mkdir()
        command = [str(args.executable.resolve()), str(args.inputs.resolve())]
        command.extend(common + options)
        with (directory / "run.log").open("w") as log:
            result = subprocess.run(
                command, cwd=directory, env=environment, stdout=log, stderr=log
            )
        text = (directory / "run.log").read_text()
        if rejection:
            assert result.returncode != 0 and rejection in text, (
                label,
                result.returncode,
                text,
            )
        else:
            assert result.returncode == 0, (label, result.returncode, text)
        return directory

    for change, factor in [("same", "1"), ("down", ".25"), ("up", "2")]:
        run(
            change,
            [
                "control_test.controller=0",
                f"control_test.factor={factor}",
                "control_test.stop=2",
            ],
        )
    full = run("full", ["control_test.stop=3"])
    first = run("first", ["control_test.stop=1", "control_test.checkpoint=1"])
    checkpoint = first / "checkpoint000001"
    assert (checkpoint / "density_control.dat").is_file()
    restart = [f"amr.restart={checkpoint}", "control_test.stop=3"]
    run("restart", restart + [f"control_test.reference={full}/final"])
    run(
        "different-controller",
        restart + ["control_test.configuration=changed-controller"],
        "Restart density controller configuration differs",
    )
    run(
        "unregistered-controller",
        restart + ["control_test.controller=0"],
        "restored controller registration",
    )
    broken = root / "missing-control-checkpoint"
    shutil.copytree(checkpoint, broken)
    (broken / "density_control.dat").unlink()
    run(
        "missing-state",
        [f"amr.restart={broken}"],
        "Eulerian restart requires density_control.dat",
    )
    if args.pedestal == "tracked":
        # Step three falls inside the deadband: last and EMA are different.
        # Persist that distinction, then resume the next native sample.
        deadband = [
            "control_test.deadband=.03",
            "control_test.configuration=fixture-alpha0.2-deadband0.03",
        ]
        full = run("deadband-full", deadband + ["control_test.stop=4"])
        first = run(
            "deadband-first",
            deadband + ["control_test.stop=3", "control_test.checkpoint=3"],
        )
        run(
            "deadband-restart",
            deadband
            + [
                "control_test.stop=4",
                f"amr.restart={first}/checkpoint000003",
                f"control_test.reference={full}/final",
            ],
        )
    print(f"DENSITY_CONTROL_RUNTIME_PASS mode={args.mode} evidence={root}")


if __name__ == "__main__":
    main()
