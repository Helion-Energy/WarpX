"""Require an explicit stopping capability/options/alias assertion."""

import subprocess
import sys

result = subprocess.run(
    [sys.argv[1], "test.reject=" + sys.argv[2]], capture_output=True, text=True
)
output = result.stdout + result.stderr
expected = {
    "gamma": "EulerianStoppingTransfer invalid options",
    "alias": "input aliases owned scratch",
}.get(sys.argv[2], "requires single-level m0/no-EB")
assert result.returncode != 0 and expected in output, output
print("explicit stopping rejection:", sys.argv[2])
