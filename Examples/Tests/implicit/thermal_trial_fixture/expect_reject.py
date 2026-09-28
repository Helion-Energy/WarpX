import subprocess
import sys

r = subprocess.run([sys.argv[1], "test.case=reject"], capture_output=True, text=True)
assert r.returncode != 0
assert "requires explicit AllContributors closure" in r.stdout + r.stderr
print("PASS explicit closure selection")
