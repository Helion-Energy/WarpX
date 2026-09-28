import pathlib
import subprocess
import sys
import tempfile

exe, case = sys.argv[1:]
messages = {
    "replay": "PASS native RNG replay",
    "metadata": "requires matching ranks",
    "truncated": "Corrupt or truncated",
    "checksum": "Corrupt or truncated",
    "missing": "missing per-rank random state",
}
with tempfile.TemporaryDirectory(prefix="thermal-random-", dir=".") as d:
    result = subprocess.run(
        [exe, str(pathlib.Path(d).resolve()), case],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(result.stdout)
    assert (
        (result.returncode == 0) if case == "replay" else (result.returncode != 0)
    ) and messages[case] in result.stdout
