"""Build the actual coupler against a pinned existing native WarpX library.

Only --output is written. No full WarpX instance is constructed: native coupler
and ExternalVectorPotential lifecycle methods use an explicit real scale owner.
"""

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--native-build", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
build = args.native_build.resolve()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
source = Path(__file__).resolve().parents[4]
recipe = (build / "CMakeFiles/lib_rz.dir/flags.make").read_text()
variables = dict(line.split(" = ", 1) for line in recipe.splitlines() if " = " in line)
includes = shlex.split(variables["CXX_INCLUDES"])
project = [i for i, value in enumerate(includes) if value.startswith("-I")][:2]
includes[project[0]] = "-I" + str(source / "Source")
includes[project[1]] = "-I" + str(build / "Source")
link = shlex.split((build / "CMakeFiles/app_rz.dir/link.txt").read_text())
link = link[link.index("-o") + 2 :]
manifest = []
for i, item in enumerate(link):
    if item.startswith("lib/"):
        original = build / item
        destination = out / "reference" / item
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = hashlib.sha256(original.read_bytes()).hexdigest()
        shutil.copy2(original, destination)
        after = hashlib.sha256(original.read_bytes()).hexdigest()
        copied = hashlib.sha256(destination.read_bytes()).hexdigest()
        assert before == after == copied, "Native library changed during pin"
        manifest.append(
            {"original": str(original), "copy": str(destination), "sha256": copied}
        )
        link[i] = str(destination)
link.insert(0, "-Wl,-rpath," + str(out / "reference/lib"))
command = [
    "/usr/bin/c++",
    *shlex.split(variables["CXX_DEFINES"]),
    *includes,
    *shlex.split(variables["CXX_FLAGS"]),
    "-O1",
    "-ffunction-sections",
    "-fdata-sections",
    str(Path(__file__).with_name("test_circuit_rejection.cpp")),
    str(source / "Source/Circuit/CircuitCoupler.cpp"),
    str(source / "Source/Circuit/DarwinDeviceCoupling.cpp"),
    "-Wl,--gc-sections",
    "-o",
    str(out / "test_circuit_rejection"),
    *link,
]
(out / "NATIVE_REFERENCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
(out / "compile-command.json").write_text(json.dumps(command, indent=2) + "\n")
with (out / "compile.log").open("w") as log:
    code = subprocess.run(
        command, cwd=source, stdout=log, stderr=subprocess.STDOUT
    ).returncode
print("actual native coupler fixture compile:", code)
raise SystemExit(code)
