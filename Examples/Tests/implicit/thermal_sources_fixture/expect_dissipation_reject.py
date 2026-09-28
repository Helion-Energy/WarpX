"""Require the documented dissipation capability or purity rejection."""

import subprocess
import sys

result = subprocess.run(
    [sys.argv[1], "test.reject=" + sys.argv[2]], capture_output=True, text=True
)
output = result.stdout + result.stderr
expected = {
    "boundary": "Unsupported axial dissipation boundary",
    "limited": "Matched viscosity requires centered gradients",
    "parser": "Hyperresistivity needs a live parser",
    "alias": "inputs must not alias owned scratch",
}.get(sys.argv[2], "requires single-level m0/no-EB and additive")
assert result.returncode != 0 and expected in output, output
print("explicit dissipation rejection:", sys.argv[2])
