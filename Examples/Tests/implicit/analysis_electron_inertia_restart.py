"""Validate the native elliptic restart round trip and specific rejected states."""
import subprocess
import sys

ERRORS = {
    "missing": "Exact native restart requires elliptic inertia warm-start state",
    "component": "Missing native elliptic correction component",
    "config": "Missing or incompatible elliptic warm-start configuration",
    "layout": "Elliptic warm-start geometry or rank/box/guard layout mismatch",
    "payload": "Native elliptic correction payload layout mismatch",
}


def validate(returncode, output, case):
    if case in ERRORS:
        assert returncode != 0, "Invalid native state was accepted"
        assert ERRORS[case] in output, "Failure did not reach the intended state check"
    else:
        assert returncode == 0, output[-4000:]
        assert f"INERTIA_RESTART_PASS case={case}" in output


if __name__ == "__main__":
    executable, inputs, case = sys.argv[1:]
    process = subprocess.run(
        [executable, inputs, "test.case=" + case],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(process.stdout)
    validate(process.returncode, process.stdout, case)
