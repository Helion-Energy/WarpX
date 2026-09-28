"""Assert invalid layouts fail explicitly, including when AMReX aborts by signal."""

import re
import subprocess
import sys

run = subprocess.run(
    [sys.argv[1], f"test.reject={sys.argv[2]}"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    check=False,
)
print(run.stdout)
assert run.returncode != 0, "invalid layout was accepted"
expected = {
    "empty": "at least one field block",
    "duplicate": "block names must be unique",
    "scalar_alias": "distinct from the vector/scalar blocks",
    "nan_scale": "finite and positive",
    "inf_scale": "finite and positive",
    "zero_scale": "finite and positive",
    "bad_layout": "incompatible MultiFab layouts",
}
message = re.sub(r"\s+", " ", run.stdout.replace("#", ""))
assert expected[sys.argv[2]] in message, "failure was unrelated to expected validation"
