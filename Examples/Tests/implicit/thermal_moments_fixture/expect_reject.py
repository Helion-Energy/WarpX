"""Require the intended fail-closed diagnostic, rather than any crash."""

import os
import subprocess
import sys

expected = {
    "current_boundary": "geometry-consistent current images",
    "modes": "RZ requires m=0",
    "floor": "finite positive density floor",
    "boundary": "Explicit current boundary",
    "layout": "matching native staggering/box order/DM",
    "alias": "trial inputs must be independent",
}
case = sys.argv[2]
result = subprocess.run(
    [sys.argv[1], f"test.case=reject_{case}"],
    env={**os.environ, "OMP_NUM_THREADS": "1"},
    capture_output=True,
    text=True,
    timeout=30,
    check=False,
)
print(result.stdout, result.stderr)
assert result.returncode != 0
assert expected[case] in result.stdout + result.stderr
assert "ERROR unsupported fixture" not in result.stdout + result.stderr
