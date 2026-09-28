"""Check explicit native-species capability/state assertions."""

import subprocess
import sys

p = subprocess.run(
    [sys.argv[1], "test.reject=" + sys.argv[2]], capture_output=True, text=True
)
expected = {
    "mm": "aggregate MM has no per-species response",
    "old": "requires fixed-old Ti",
    "descriptor": "invalid charged-species descriptor",
    "eb": "requires single-level m0/no-EB",
}
assert p.returncode != 0 and expected[sys.argv[2]] in p.stdout + p.stderr, (
    p.stdout + p.stderr
)
print("SPECIES_REJECTION_PASS", sys.argv[2])
