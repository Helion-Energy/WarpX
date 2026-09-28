"""Specific production guard, not just a crash."""

import os
import subprocess
import sys

case = sys.argv[2]
expected = {
    "mode": "must be legacy, decoupled_jfnk or coupled_jfnk",
    "conflict": "conflicts with solve_electron_energy_equation=0",
    "eb": "one level and no EB",
    "driver": "requires Darwin theta_implicit_hybrid",
    "amr": "one level and no EB",
    "floor": "hard common Ohm/energy density floor",
    "gamma": "requires gamma>1",
    "metadata": "invalid Eulerian electron energy checkpoint metadata",
    "ghost": "all required ghosts",
    "alias": "independent nodal storage",
}[case]
if sys.argv[3] == "3D" and case in ["eb", "driver", "amr", "floor", "gamma"]:
    expected = "currently requires RZ"
r = subprocess.run(
    [sys.argv[1], f"test.case=reject_{case}"],
    env={**os.environ, "OMP_NUM_THREADS": "1"},
    capture_output=True,
    text=True,
    timeout=30,
    check=False,
)
print(r.stdout, r.stderr)
assert r.returncode != 0
assert expected in r.stdout + r.stderr
assert "ERROR guard absent" not in r.stdout + r.stderr
