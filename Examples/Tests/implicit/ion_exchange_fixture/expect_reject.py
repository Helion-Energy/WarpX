#!/usr/bin/env python3
import subprocess
import sys

result = subprocess.run(
    [sys.argv[1], "test.case=guard_" + sys.argv[2]], capture_output=True, text=True
)
expected = (
    "kick-cap/shunt transforms unsupported"
    if sys.argv[2] in {"cap", "shunt"}
    else "requires single-level m0/no-EB"
)
if result.returncode == 0 or expected not in result.stdout + result.stderr:
    print(result.stdout + result.stderr)
    raise SystemExit("expected capability rejection absent")
print("PASS guarded", sys.argv[2])
