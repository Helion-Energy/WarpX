#!/usr/bin/env python3
"""Directly exercise both device scale consumers, including EB masks."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

exe, plugin, inputs, root = map(Path, sys.argv[1:5])
root.mkdir(parents=True, exist_ok=True)
text = inputs.read_text().replace(
    "hybrid_pic_model.deterministic_pressure_bc = 1\n", ""
)
extra = [
    line.replace("uniform_analytical.", "background.")
    for line in text.splitlines()
    if line.startswith("external_vector_potential.uniform_analytical.")
    and ".python_scale" not in line
    and ".initial_scale" not in line
]
text = text.replace(
    "fields = uniform_analytical", "fields = uniform_analytical background"
)
text += "\n" + "\n".join(extra) + "\n"
for eb in (False, True) if sys.argv[5] == "ON" else (False,):
    run = Path(tempfile.mkdtemp(prefix="eb_" if eb else "regular_", dir=root))
    deck = text + ('\nwarpx.eb_implicit_function = "0.08**2-x*x-y*y"\n' if eb else "")
    (run / "inputs").write_text(deck)
    result = subprocess.run(
        [str(exe.resolve()), "inputs", f"circuit.plugin_library={plugin.resolve()}"],
        cwd=run,
        env=dict(os.environ, OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    (run / "stdout.txt").write_text(output)
    assert result.returncode == 0 and "DEVICE_EXTERNAL_FIELDS_PASS" in output, run
    print(f"Direct E/B scale consumers EB={eb}: PASS; {run}")
