import subprocess
import sys

result = subprocess.run(
    [sys.argv[1], "test.case=reject"], capture_output=True, text=True
)
assert result.returncode != 0, result.stdout
assert "requires physical E-form Ohm response" in result.stdout + result.stderr
print("PASS explicit transformed Ohm rejection")
