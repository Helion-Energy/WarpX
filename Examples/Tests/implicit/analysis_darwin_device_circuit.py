#!/usr/bin/env python3
"""Compare the native host oracle and device response through complete steps."""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yt

exe, plugin, no_affine, inputs, root = map(Path, sys.argv[1:6])
eb = sys.argv[6]
root.mkdir(parents=True, exist_ok=True)
campaign = Path(tempfile.mkdtemp(prefix="comparison_", dir=root))
analysis = Path(__file__).with_name("analysis_darwin_native_circuit.py")
for backend in ("host", "device_affine"):
    deck = campaign / f"inputs_{backend}"
    deck.write_text(
        inputs.read_text()
        + f"\ncircuit.trial_backend = {backend}\ncircuit.device_iterations = 20\n"
    )
    command = [
        sys.executable,
        str(analysis),
        str(exe),
        str(plugin),
        str(deck),
        str(campaign / backend),
        eb,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=400)
    (campaign / f"{backend}.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, (backend, campaign, result.stdout, result.stderr)

names = ["energy", "polytropic", "reference2", "two_ports"]
if eb == "ON":
    names.append("embedded_boundary")
yt.set_log_level(50)
for name in names:
    runs = [
        next((campaign / backend).glob(name + "_*"))
        for backend in ("host", "device_affine")
    ]
    device_log = (runs[1] / "stdout.txt").read_text()
    assert device_log.count("Device circuit accepted:") == 2
    assert device_log.count("host_trial_calls=0") == 2
    currents = [
        re.findall(
            r"NATIVE_RL_PORT: (\d+) (\d+) (\S+) (\S+) (\S+) (\S+)",
            (run / "stdout.txt").read_text(),
        )
        for run in runs
    ]
    assert len(currents[0]) == len(currents[1])
    for a, b in zip(*currents):
        assert a[:3] == b[:3]
        # Complete nonlinear trajectories agree to the solve tolerance;
        # same-EMF map/exact agreement is checked separately at 1e-14.
        assert np.isclose(float(a[4]), float(b[4]), rtol=1e-8, atol=1e-15), (name, a, b)
    for step in (1, 2):
        grids = []
        datasets = []
        for run in runs:
            ds = yt.load(str(run / "diags" / f"gate{step:06d}"))
            datasets.append(ds)
            grids.append(ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions))
        for field in ("Ex", "Ey", "Ez", "Bx", "By", "Bz", "Te", "Pe", "rho"):
            a, b = [np.asarray(grid["boxlib", field]) for grid in grids]
            assert np.isfinite(a).all() and np.isfinite(b).all()
            delta = np.max(np.abs(a - b))
            gate = 1e-2 if field.startswith("E") else 3e-7
            if field in ("Te", "Pe", "rho"):
                gate = (1e-8 if field == "rho" else 1e-6) * np.max(np.abs(a))
            assert delta < gate, (name, step, field, delta, gate)
    print(f"{name}: host/device field and current gates PASS")

for name, library, args, message in (
    ("missing_capability", no_affine, [], "affine"),
    (
        "insufficient_iterations",
        plugin,
        ["circuit.device_iterations=1"],
        "Device circuit rejected",
    ),
):
    run = campaign / name
    run.mkdir()
    result = subprocess.run(
        [
            str(exe.resolve()),
            str(inputs.resolve()),
            f"circuit.plugin_library={library.resolve()}",
            "circuit.trial_backend=device_affine",
            *args,
        ],
        cwd=run,
        env=dict(os.environ, OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout + result.stderr
    (run / "stdout.txt").write_text(output)
    assert result.returncode != 0 and message in output, (name, run)
    assert "NATIVE_RL_COMMIT:" not in output, (name, "refusal accepted a step")
    print(f"{name}: required refusal PASS")
print(f"Device circuit integration PASS; {campaign}")
