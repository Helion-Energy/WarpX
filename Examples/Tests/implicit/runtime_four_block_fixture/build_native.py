"""Link the independent oracle against a completed native WarpX Makefile build.

Use a stable matching header/library snapshot when another worker is rebuilding.
Only the selected output directory is written. The existing build is read-only.
"""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--native-source", type=Path, required=True)
parser.add_argument("--native-build", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = args.native_source.resolve()
build = args.native_build.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
worker = Path(__file__).resolve().parents[4]
recipe = (build / "CMakeFiles/lib_rz.dir/flags.make").read_text()
variables = dict(line.split(" = ", 1) for line in recipe.splitlines() if " = " in line)
includes = shlex.split(variables["CXX_INCLUDES"])
# CMake's first two project include roots are Source and generated Source.
# Dependency includes retain their original (read-only) AMReX locations.
project = [i for i, value in enumerate(includes) if value.startswith("-I")][:2]
assert len(project) == 2 and all(includes[i].endswith("/Source") for i in project)
includes[project[0]] = "-I" + str(source / "Source")
includes[project[1]] = "-I" + str(build / "Source")
link = shlex.split((build / "CMakeFiles/app_rz.dir/link.txt").read_text())
link = link[link.index("-o") + 2 :]
link = [str(build / item) if item.startswith("lib/") else item for item in link]
command = [
    "/usr/bin/c++",
    *shlex.split(variables["CXX_DEFINES"]),
    *includes,
    *shlex.split(variables["CXX_FLAGS"]),
    str(Path(__file__).with_name("test_runtime_four_block.cpp")),
    "-o",
    str(output / "test_runtime_four_block"),
    "-Wl,--wrap=_ZN5warpx7thermal18SolveThermalSystemERNS0_24ThermalNonlinearOperatorER14WarpXSolverVecRKNS0_19ThermalSolveOptionsE",
    "-Wl,--wrap=_ZN5WarpX23PushParticlesandDepositEdb16PositionPushType16MomentumPushTypePK15ImplicitOptions",
    "-Wl,--wrap=_ZN14ImplicitSolver8PreRHSOpEdib",
    *link,
]
(output / "compile-command.json").write_text(json.dumps(command, indent=2) + "\n")
with (output / "compile.log").open("w") as log:
    result = subprocess.run(command, cwd=worker, stdout=log, stderr=subprocess.STDOUT)
print("native four-block runtime compile:", result.returncode)
raise SystemExit(result.returncode)
