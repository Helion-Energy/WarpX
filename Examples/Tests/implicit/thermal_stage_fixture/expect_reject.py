"""Invalid contexts must fail explicitly, with their own diagnostic."""

import subprocess
import sys

messages = {
    "leg": "leg/source coupling has not been derived",
    "bfloor": "Invalid Eulerian thermal stage parameters",
    "density": "requires finite fields and positive n",
    "periodic_wall": "periodic thermal direction cannot also carry",
}
result = subprocess.run(
    [sys.argv[1], f"test.case=reject_{sys.argv[2]}"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    check=False,
)
print(result.stdout)
assert result.returncode != 0 and messages[sys.argv[2]] in result.stdout
