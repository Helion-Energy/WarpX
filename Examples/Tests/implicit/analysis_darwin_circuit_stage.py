#!/usr/bin/env python3
"""Installed nonzero circuit feedback must preserve the residual's replay contract."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from analysis_darwin_residual_freshness import PATTERN, check_output


def main():
    executable, inputs, output_root = map(Path, sys.argv[1:])
    output_root.mkdir(parents=True, exist_ok=True)
    cases = (
        ("legacy", 1, False),
        ("energy", 1, True),
        ("polytropic", 0, True),
        ("two_coils", 1, True),
    )
    for name, energy, consistent in cases:
        run = Path(tempfile.mkdtemp(prefix=f"{name}_", dir=output_root))
        environment = dict(os.environ, OMP_NUM_THREADS="1", WARPX_RESIDUAL_CHECK="1")
        case_inputs = inputs.resolve()
        if name == "two_coils":
            case_inputs = run / "inputs"
            text = inputs.read_text().replace(
                "fields = uniform_analytical", "fields = uniform_analytical secondary"
            )
            extra = [
                line.replace("uniform_analytical.", "secondary.")
                for line in text.splitlines()
                if line.startswith("external_vector_potential.uniform_analytical.")
            ]
            case_inputs.write_text(text + "\n" + "\n".join(extra) + "\n")
        command = [
            str(executable.resolve()),
            str(case_inputs),
            f"hybrid_pic_model.solve_electron_energy_equation={energy}",
        ]
        if not consistent:
            command.append("implicit_evolve.darwin_circuit_consistent_stage=0")
        result = subprocess.run(
            command,
            cwd=run,
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        output = result.stdout + result.stderr
        (run / "stdout.txt").write_text(output)
        if not consistent:
            # Reproduce the old stage map rather than accepting any arbitrary crash.
            assert result.returncode != 0 and "Nonlinear solver failed" in output
            rows = PATTERN.findall(output)
            assert rows, "legacy control did not execute the replay diagnostic"
            worst = max(float(row[3]) / max(1.0, float(row[4])) for row in rows)
            assert worst > 1.0e-7, "legacy control did not reproduce the replay defect"
        else:
            assert result.returncode == 0, f"simulation failed; see {run}"
            worst = check_output(output)
            assert "CIRCUIT_STAGE_TEST: starts=2, commits=2" in output
            print(f"{name}: scaled replay defect={worst}; {run}")


if __name__ == "__main__":
    main()
