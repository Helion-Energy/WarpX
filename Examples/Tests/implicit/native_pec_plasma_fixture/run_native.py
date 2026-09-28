import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--executable", required=True)
p.add_argument("--input", required=True)
p.add_argument("--case", required=True)
p.add_argument("--mode", default="coupled_jfnk")
p.add_argument("--boundary", choices=["periodic", "pmc"], default="periodic")
p.add_argument("--negative", choices=["alias", "width", "eta", "mm"])
p.add_argument("--ranks", type=int, default=1)
p.add_argument("--mpi")
p.add_argument("--grid", type=int, default=8)
args = p.parse_args()
root = Path(tempfile.mkdtemp(prefix="native-pec-" + args.case + "-", dir=Path.cwd()))
source = Path(__file__).resolve().parent
cmd = [
    str(Path(args.executable).resolve()),
    str(Path(args.input).resolve()),
    "fixture.event=1",
    "fixture.kernel_oracle=1",
    "implicit_evolve.darwin_longitudinal_schur=1",
    "hybrid_pic_model.electron_energy_mode=" + args.mode,
    "amr.n_cell=" + str(args.grid) + " " + str(args.grid),
]
if args.boundary == "pmc":
    cmd += [
        "boundary.field_lo=none pmc",
        "boundary.field_hi=pec pmc",
        "boundary.particle_lo=none reflecting",
        "boundary.particle_hi=reflecting reflecting",
    ]
patterns = {
    "alias": "Pressure/Hall capture requires independent matching E scratch and all guards",
    "width": "Pressure/Hall capture requires independent matching E scratch and all guards",
    "eta": "!m.HasResistivity()",
    "mm": "PEC plasma boundary constitutive response requires full particle Jv",
}
if args.negative in ["alias", "width"]:
    cmd += ["fixture.event=0", "fixture.capture_negative=" + args.negative]
elif args.negative == "eta":
    cmd += ["fixture.event=0", "hybrid_pic_model.plasma_resistivity(rho,J,t)=0.01"]
elif args.negative == "mm":
    cmd += ["fixture.event=0", "implicit_evolve.use_mass_matrices_jacobian=1"]
if args.ranks > 1:
    assert args.mpi
    cmd = [args.mpi, "-n", str(args.ranks), *cmd]
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
with (root / "run.log").open("wb") as out:
    rc = subprocess.run(
        cmd, cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT
    ).returncode
log = (root / "run.log").read_text(errors="replace")
if args.negative:
    passed = (
        rc != 0
        and patterns[args.negative] in log
        and "PEC_PLASMA_CONTINUATION_PASS" not in log
    )
else:
    passed = (
        rc == 0
        and "PEC_EVENT_COMMIT_PASS" in log
        and "PEC_KERNEL_ORACLE_PASS" in log
        and "PEC_PLASMA_CONTINUATION_PASS steps=2" in log
    )
    if passed:
        passed = (
            subprocess.run(
                [
                    sys.executable,
                    str(source / "analyze_phase.py"),
                    str(root),
                    str(root / "PHASE.json"),
                ]
            ).returncode
            == 0
        )
result = {
    "case": args.case,
    "command": cmd,
    "returncode": rc,
    "pass": passed,
    "executable_sha256": hashlib.sha256(Path(args.executable).read_bytes()).hexdigest(),
    "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
    "log_sha256": hashlib.sha256((root / "run.log").read_bytes()).hexdigest(),
    "scope": "native CPU continuation and algebraic constitutive/phase identities, not zero physical energy defect",
}
(root / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
print(root)
print(json.dumps(result))
sys.exit(0 if passed else 1)
