#!/usr/bin/env python3
"""Check device arithmetic and mandatory refusals, including release builds."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

executable, root = map(Path, sys.argv[1:3])
root.mkdir(parents=True, exist_ok=True)
for mode in ("valid", "sign", "band", "entry", "interval", "nonfinite", "convergence"):
    run = Path(tempfile.mkdtemp(prefix=mode + "_", dir=root))
    result = subprocess.run(
        [str(executable.resolve()), f"mode={mode}"],
        cwd=run,
        env=dict(os.environ, OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=40,
    )
    output = result.stdout + result.stderr
    (run / "stdout.txt").write_text(output)
    if mode == "valid":
        assert result.returncode == 0 and "DEVICE_CIRCUIT_PASS" in output, run
    else:
        assert result.returncode != 0, f"{mode} failed to refuse; {run}"
        assert "A required refusal did not occur" not in output, run
        expected = {
            "entry": "entry current disagrees",
            "interval": "Invalid device circuit affine packet",
        }.get(mode, "Device circuit rejected")
        assert expected in output, (mode, run, output)
    print(f"Device circuit {mode}: PASS; {run}")
