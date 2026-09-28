"""Require an explicit unsupported-source or purity diagnostic."""

import subprocess
import sys

result = subprocess.run(
    [sys.argv[1], "test.reject=" + sys.argv[2]], capture_output=True, text=True
)
text = result.stdout + result.stderr
assert result.returncode != 0, text
expected = {
    "parser": "enabled source requires a live compiled parser",
    "alias": "outputs must be disjoint scratch (including aliases)",
}.get(sys.argv[2], "require separately derived thermal stages and are unsupported")
assert expected in text, text
print("explicit rejection:", sys.argv[2])
