#!/usr/bin/env python3
"""Native compiled-plugin replay and reference-current normalization gates."""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yt
from analysis_darwin_residual_freshness import check_output


def main():
    executable, plugin, inputs, root = map(Path, sys.argv[1:5])
    root.mkdir(parents=True, exist_ok=True)
    completed = {}
    cases = [
        ("energy", 1, 1),
        ("polytropic", 0, 1),
        ("reference2", 1, 2),
        ("two_ports", 1, 1),
    ]
    if len(sys.argv) > 5 and sys.argv[5] == "ON":
        cases.append(("embedded_boundary", 1, 1))
    for name, energy, reference in cases:
        run = Path(tempfile.mkdtemp(prefix=name + "_", dir=root))
        text = inputs.read_text()
        if reference == 2:
            text = text.replace("I_ref = 1", "I_ref = 2").replace("*0.01", "*0.02")
        port_count = 1
        if name == "two_ports":
            text = text.replace(
                "fields = uniform_analytical", "fields = uniform_analytical secondary"
            )
            text = text.replace(
                "coils = uniform_analytical", "coils = uniform_analytical secondary"
            )
            extra = [
                line.replace("uniform_analytical.", "secondary.")
                for line in text.splitlines()
                if line.startswith(
                    (
                        "external_vector_potential.uniform_analytical.",
                        "circuit.uniform_analytical.",
                    )
                )
            ]
            text += (
                "\n"
                + "\n".join(extra).replace(
                    "secondary.probe = disk", "secondary.probe = none"
                )
                + "\n"
            )
            port_count = 2
        if name == "embedded_boundary":
            text += '\nwarpx.eb_implicit_function = "0.08**2 - x*x - y*y"\n'
        (run / "inputs").write_text(text)
        command = [
            str(executable.resolve()),
            "inputs",
            f"circuit.plugin_library={plugin.resolve()}",
            f"hybrid_pic_model.solve_electron_energy_equation={energy}",
        ]
        environment = dict(os.environ, OMP_NUM_THREADS="1", WARPX_RESIDUAL_CHECK="1")
        result = subprocess.run(
            command,
            cwd=run,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        (run / "stdout.txt").write_text(output)
        assert result.returncode == 0, f"native simulation failed; see {run}"
        worst = check_output(output)
        assert output.count("NATIVE_RL_COMMIT:") == 2, "one circuit commit per step"
        assert "Circuit coupling engine: external" in output
        current = [0.0] * port_count
        ports = re.findall(
            r"NATIVE_RL_PORT: (\d+) (\d+) (\S+) (\S+) (\S+) (\S+)", output
        )
        assert len(ports) == 2 * port_count
        for step, port, dt, emf, measured, i_ref in ports:
            dt, emf, measured, i_ref = map(float, (dt, emf, measured, i_ref))
            port = int(port)
            expected = (current[port] + dt / 0.01 * (5e5 - emf)) / (1 + dt / 0.01)
            assert np.isclose(measured, expected, rtol=1e-14, atol=0)
            assert i_ref == reference
            current[port] = measured
        assert any(abs(float(row[3])) > 1e-6 for row in ports), (
            "feedback must be nonzero"
        )
        print(f"{name}: native replay defect={worst}; {run}")
        completed[name] = run
    yt.set_log_level(50)
    for step in (1, 2):
        grids = []
        datasets = []
        for name in ("energy", "reference2"):
            dataset = yt.load(str(completed[name] / "diags" / f"gate{step:06d}"))
            datasets.append(dataset)
            grids.append(
                dataset.covering_grid(
                    0, dataset.domain_left_edge, dataset.domain_dimensions
                )
            )
        for field in ("Ex", "Ey", "Ez", "Bx", "By", "Bz", "Te", "rho"):
            a, b = [np.asarray(grid["boxlib", field]) for grid in grids]
            assert np.isfinite(a).all() and np.isfinite(b).all()
            delta = np.max(np.abs(a - b))
            gate = 1e-2 if field.startswith("E") else 3e-7
            if field in ("Te", "rho"):
                gate = (1e-6 if field == "Te" else 1e-8) * np.max(np.abs(a))
            assert delta < gate, (step, field, delta, gate)
    print("Native plugin reference-current normalization passed")


if __name__ == "__main__":
    main()
