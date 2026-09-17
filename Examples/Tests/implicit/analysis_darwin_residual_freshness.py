#!/usr/bin/env python3
"""Detect density/history leaks in the Darwin split's nonlinear residual."""

import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


PATTERN = re.compile(
    r"RESIDUAL_CHECK: step = (\d+), iter = (\d+), (\w+), "
    r"difference = ([^,]+), reference = (\S+)"
)
LABELS = {
    "saved_vs_fresh", "repeat", "stage_base_vs_full",
    "stage_probe_vs_full", "excursion_return",
}


def check_output(output):
    rows = PATTERN.findall(output)
    assert rows, "residual instrumentation did not run"
    assert {row[2] for row in rows} == LABELS
    assert all(math.isfinite(float(value)) for row in rows for value in row[3:])
    assert {int(row[0]) for row in rows} == {0, 1}, "two steps required"
    assert any(int(row[1]) > 0 for row in rows), "must check a changed iterate"
    worst = max(float(row[3]) / max(1.0, float(row[4])) for row in rows)
    # Fresh, repeated and excursion-return residuals agree at the particle/
    # floating-point accuracy. The old density component produces ~2e-2.
    assert worst < 1.0e-7, f"residual depends on evaluation history: {worst}"
    return worst


def main():
    executable, inputs, output_root = map(Path, sys.argv[1:])
    output_root.mkdir(parents=True, exist_ok=True)
    for energy in (0, 1):
        run = Path(tempfile.mkdtemp(prefix=f"energy{energy}_", dir=output_root))
        environment = dict(os.environ, OMP_NUM_THREADS="1", WARPX_RESIDUAL_CHECK="1")
        command = [str(executable.resolve()), str(inputs.resolve()),
                   f"hybrid_pic_model.solve_electron_energy_equation={energy}"]
        result = subprocess.run(command, cwd=run, env=environment,
                                capture_output=True, text=True, timeout=90)
        output = result.stdout + result.stderr
        (run / "stdout.txt").write_text(output)
        assert result.returncode == 0, f"simulation failed; see {run}"
        worst = check_output(output)
        rows = [line.split() for line in (run / "newton.txt").read_text().splitlines()
                if line and not line.startswith("#")]
        assert rows and {int(row[0]) for row in rows} == {1, 2}
        assert all(int(row[10]) in (2, 3) for row in rows), "nonlinear failure"
        print(f"energy={energy}, worst normalized residual drift={worst}; {run}")


if __name__ == "__main__":
    main()
