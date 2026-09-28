import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--executable", required=True)
p.add_argument("--case", required=True)
p.add_argument("--mode", default="coupled_jfnk")
p.add_argument("--pmc", action="store_true")
p.add_argument("--steps", type=int, default=1)
p.add_argument("--reject-step", type=int, default=0)
p.add_argument("--fault", choices=["decision", "companion"], default="decision")
p.add_argument("--ranks", type=int, default=1)
p.add_argument("--mpi")
p.add_argument("--seam", action="store_true")
p.add_argument("--veto-rank", type=int, default=-1)
p.add_argument("--preflight", action="store_true")
p.add_argument("--diagnostics", action="store_true")
a = p.parse_args()
source = Path(__file__).resolve().parent
root = Path(tempfile.mkdtemp(prefix="retained-" + a.case + "-", dir=Path.cwd()))
exe = Path(a.executable).resolve()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


steps = 0 if a.preflight else a.steps
assert a.ranks > 0 and (a.preflight or 0 <= a.reject_step < steps)
assert not a.seam or a.ranks == 2
plan = {
    "contract": "CPU exact complete in-process rollback and retry versus fresh clean, unchanged native constraints. Seam is rejected-only post-endpoint relocation, not accepted common-support crossing.",
    "executable": str(exe),
    "executable_sha256": sha(exe),
    "input_sha256": sha(source / "inputs_rz"),
    "arguments": vars(a),
}
(root / "PLAN.json").write_text(json.dumps(plan, indent=2) + "\n")
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
records = []
for arm in ["preflight"] if a.preflight else ["retry", "clean"]:
    d = root / arm
    d.mkdir()
    cmd = [
        str(exe),
        str(source / "inputs_rz"),
        "max_step=0",
        "implicit_evolve.native_retained_acceptance=1",
        "fixture.steps=" + str(steps),
        "fixture.reject=" + str(int(arm == "retry")),
        "fixture.reject_step=" + str(a.reject_step),
        "fixture.fault=" + a.fault,
        "fixture.preflight=" + str(int(a.preflight)),
        "fixture.veto_rank=" + str(a.veto_rank),
        "hybrid_pic_model.electron_energy_mode=" + a.mode,
    ]
    if a.diagnostics:
        cmd += [
            "fixture.diagnostics=1",
            "warpx.reduced_diags_names=poynting",
            "poynting.type=FieldPoyntingFlux",
            "poynting.intervals=1",
        ]
    if a.pmc:
        cmd += [
            "boundary.field_lo=none pmc",
            "boundary.field_hi=pec pmc",
            "boundary.particle_lo=none reflecting",
            "boundary.particle_hi=reflecting reflecting",
        ]
    if a.seam:
        cmd += [
            "fixture.seam=1",
            "amr.n_cell=16 16",
            "amr.max_grid_size=16",
            "amr.max_grid_size_y=8",
        ]
    if a.ranks > 1:
        assert a.mpi
        cmd = [a.mpi, "-n", str(a.ranks), *cmd]
    with (d / "run.log").open("wb") as log:
        rc = subprocess.run(
            cmd, cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT
        ).returncode
    text = (d / "run.log").read_text()
    m = re.search(
        r"RETAINED_ACCEPTANCE_PASS steps=(\d+) rejected=(\d+) before_calls=(\d+) after_calls=(\d+) migrated=(\d+)",
        text,
    )
    expected_reject = int(arm == "retry")
    expected_before = steps + expected_reject
    expected_after = steps + int(arm == "retry" and a.fault == "decision")
    ok = (
        rc == 0
        and bool(m)
        and tuple(map(int, m.groups()[:4]))
        == (steps, expected_reject, expected_before, expected_after)
        and text.count("Eulerian accepted energy [J]:") == steps
    )
    if arm == "retry":
        ok = ok and text.count("RETAINED_ROLLBACK_PASS") == 1
    if a.diagnostics:
        ok = (
            ok
            and text.count("RETAINED_MIDPOINT_DIAGNOSTIC integrated_abs=")
            == expected_before
        )
    if a.seam and arm == "retry":
        ok = ok and bool(m) and int(m.group(5)) > 0
    if a.preflight:
        ok = (
            ok
            and "RETAINED_PREFLIGHT_PASS unsupported=1 invalid_density=1 exact=1 operator_calls=0"
            in text
        )
    records.append(
        {
            "arm": arm,
            "command": cmd,
            "returncode": rc,
            "pass": bool(ok),
            "log_sha256": sha(d / "run.log"),
            "counts": list(map(int, m.groups())) if m else None,
        }
    )
    if not ok:
        break
comparisons = []
if all(x["pass"] for x in records) and not a.preflight and len(records) == 2:
    for q in sorted((root / "clean").glob("accepted_*.bin")):
        b = root / "retry" / q.name
        comparisons.append(
            {
                "file": q.name,
                "exact": b.is_file() and b.read_bytes() == q.read_bytes(),
                "clean_sha256": sha(q),
                "retry_sha256": sha(b) if b.exists() else None,
            }
        )
checkpoint_comparisons = []
if a.diagnostics and all(x["pass"] for x in records):
    for step in range(steps):
        suffix = "diagnostic-accepted_" + str(step) + "/FieldPoyntingFlux_data.txt"
        x = root / "retry" / suffix
        y = root / "clean" / suffix
        checkpoint_comparisons.append(
            dict(
                kind="accepted",
                step=step,
                exact=x.read_bytes() == y.read_bytes(),
                retry_sha256=sha(x),
                clean_sha256=sha(y),
            )
        )
    x = (
        root
        / "retry"
        / ("diagnostic-old_" + str(a.reject_step))
        / "FieldPoyntingFlux_data.txt"
    )
    y = (
        root
        / "retry"
        / ("diagnostic-restored_" + str(a.reject_step))
        / "FieldPoyntingFlux_data.txt"
    )
    checkpoint_comparisons.append(
        dict(
            kind="rollback",
            exact=x.read_bytes() == y.read_bytes(),
            old_sha256=sha(x),
            restored_sha256=sha(y),
        )
    )
ok = (
    all(x["exact"] for x in checkpoint_comparisons)
    and all(x["pass"] for x in records)
    and (
        a.preflight
        or len(comparisons) == steps * a.ranks
        and all(x["exact"] for x in comparisons)
    )
)
result = {
    "case": a.case,
    "pass": ok,
    "records": records,
    "accepted_states": comparisons,
    "diagnostic_checkpoints": checkpoint_comparisons,
    "scope": "CPU exact recomputation; no GPU byte-recompute or physical event admission claim",
}
(root / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
print(root)
print(json.dumps(result))
raise SystemExit(0 if ok else 1)
