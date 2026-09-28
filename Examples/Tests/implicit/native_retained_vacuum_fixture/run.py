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
p.add_argument("--reference-executable")
p.add_argument("--case", required=True)
p.add_argument(
    "--mode", choices=["coupled_jfnk", "decoupled_jfnk"], default="coupled_jfnk"
)
p.add_argument("--mm", type=int, choices=[0, 1], default=0)
p.add_argument("--steps", type=int, default=2)
p.add_argument("--reject-step", type=int, default=0)
p.add_argument(
    "--fault",
    choices=["decision", "before", "density", "nonfinite"],
    default="decision",
)
p.add_argument("--ranks", type=int, default=1)
p.add_argument("--mpi")
p.add_argument("--relocate", action="store_true")
p.add_argument("--veto-rank", type=int, default=-1)
p.add_argument("--preflight", action="store_true")
p.add_argument(
    "--negative",
    choices=["projection_without_mm", "joule", "driver", "nonzero", "segment"],
)
a = p.parse_args()
source = Path(__file__).resolve().parent
root = Path(tempfile.mkdtemp(prefix="retained-vacuum-" + a.case + "-", dir=Path.cwd()))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


exe = Path(a.executable).resolve()
steps = 0 if a.preflight else a.steps
assert a.ranks > 0 and (a.preflight or 0 <= a.reject_step < steps)
assert not a.relocate or a.ranks == 2
plan = dict(
    arguments=vars(a),
    executable=str(exe),
    executable_sha256=sha(exe),
    input_sha256=sha(source / "inputs_rz"),
    contract="Exact accepted fields/history/full particles/RNG and reduced diagnostics; the explicitly identified nine native tangent bands are derived scratch, invalidated on retry. No physical crossing, driver or source admission.",
)
(root / "PLAN.json").write_text(json.dumps(plan, indent=2) + "\n")
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
records = []
arms = (
    ["negative"]
    if a.negative
    else ["preflight"]
    if a.preflight
    else ["retry", "clean", "nonretained"]
)
for arm in arms:
    d = root / arm
    d.mkdir()
    program = (
        Path(a.reference_executable).resolve()
        if arm == "nonretained" and a.reference_executable
        else exe
    )
    cmd = [
        str(program),
        str(source / "inputs_rz"),
        "fixture.steps=" + str(steps),
        "fixture.reject=" + str(int(arm == "retry")),
        "fixture.reject_step=" + str(a.reject_step),
        "fixture.fault=" + a.fault,
        "fixture.seam=" + str(int(a.relocate)),
        "fixture.veto_rank=" + str(a.veto_rank),
        "fixture.preflight=" + str(int(a.preflight)),
        "implicit_evolve.native_retained_acceptance=" + str(int(arm != "nonretained")),
        "hybrid_pic_model.electron_energy_mode=" + a.mode,
        "implicit_evolve.use_mass_matrices_jacobian=" + str(a.mm),
        "implicit_evolve.mass_matrices_density_projection=" + str(a.mm),
    ]
    if a.negative == "projection_without_mm":
        cmd += [
            "implicit_evolve.use_mass_matrices_jacobian=0",
            "implicit_evolve.mass_matrices_density_projection=1",
        ]
    if a.negative == "joule":
        cmd += ["hybrid_pic_model.include_joule_heating=1"]
    if a.negative == "driver":
        cmd += [
            "external_vector_potential.uniform_analytical.A_time_external_function(t)=t"
        ]
    if a.negative == "nonzero":
        cmd += [
            "external_vector_potential.uniform_analytical.A_time_external_function(t)=1"
        ]
    if a.negative == "segment":
        cmd += [
            "external_vector_potential.uniform_analytical.python_scale=1",
            "external_vector_potential.uniform_analytical.initial_scale=0",
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
        r"RETAINED_VACUUM_ACCEPTANCE_PASS steps=(\d+) rejected=(\d+) before_calls=(\d+) after_calls=(\d+) migrated=(\d+)",
        text,
    )
    reject = int(arm == "retry")
    nbefore = 0 if arm == "nonretained" else steps + reject
    nafter = (
        0
        if arm == "nonretained"
        else steps + int(arm == "retry" and a.fault == "decision")
    )
    ok = (
        rc == 0
        and bool(m)
        and tuple(map(int, m.groups()[:4])) == (steps, reject, nbefore, nafter)
        and text.count("Eulerian accepted energy [J]:") == steps
    )
    if arm == "retry":
        ok = ok and text.count("RETAINED_ROLLBACK_PASS") == 1
    if a.relocate and arm == "retry":
        ok = ok and bool(m) and int(m.group(5)) > 0
    if a.preflight:
        ok = (
            ok
            and "RETAINED_VACUUM_PREFLIGHT_PASS density=2 mask=2 time=1 layout=1 exact=1"
            in text
        )
    if a.negative:
        expected = {
            "projection_without_mm": "Hybrid MM density projection requires a MM Jacobian",
            "joule": "Retained acceptance requires source-free native Yee",
            "driver": "Correlated curl rate requires static or analytic prescribed fields",
            "nonzero": "Retained acceptance requires source-free native Yee",
            "segment": "Correlated curl rate requires static or analytic prescribed fields",
        }[a.negative]
        ok = (
            rc != 0 and expected in text and "Eulerian accepted energy [J]:" not in text
        )
    records.append(
        dict(
            arm=arm,
            command=cmd,
            executable_sha256=sha(program),
            returncode=rc,
            pass_=bool(ok),
            log_sha256=sha(d / "run.log"),
            counts=list(map(int, m.groups())) if m else None,
        )
    )
    if not ok:
        break
comparisons = []
if all(x["pass_"] for x in records) and len(records) == 3:
    for arm in ["retry", "nonretained"]:
        for q in sorted((root / "clean").glob("accepted_*.bin")):
            b = root / arm / q.name
            comparisons.append(
                dict(
                    kind="accepted",
                    arm=arm,
                    file=q.name,
                    exact=b.is_file() and b.read_bytes() == q.read_bytes(),
                    clean_sha256=sha(q),
                    other_sha256=sha(b) if b.exists() else None,
                )
            )
        for step in range(steps):
            suffix = "diagnostic-accepted_" + str(step) + "/FieldPoyntingFlux_data.txt"
            x = root / "clean" / suffix
            y = root / arm / suffix
            comparisons.append(
                dict(
                    kind="diagnostic_checkpoint",
                    arm=arm,
                    step=step,
                    exact=x.read_bytes() == y.read_bytes(),
                    clean_sha256=sha(x),
                    other_sha256=sha(y),
                )
            )
    for q in (root / "retry").glob("restored_*.bin"):
        b = q.with_name(q.name.replace("restored_", "old_"))
        comparisons.append(
            dict(
                kind="rollback",
                file=q.name,
                exact=q.read_bytes() == b.read_bytes(),
                old_sha256=sha(b),
                restored_sha256=sha(q),
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
    comparisons.append(
        dict(
            kind="diagnostic_rollback",
            exact=x.read_bytes() == y.read_bytes(),
            old_sha256=sha(x),
            restored_sha256=sha(y),
        )
    )
ok = all(x["pass_"] for x in records) and (
    bool(a.preflight or a.negative)
    or len(records) == 3
    and len(comparisons) >= 4 * steps
    and all(x["exact"] for x in comparisons)
)
result = dict(
    case=a.case,
    pass_=ok,
    records=records,
    comparisons=comparisons,
    scope="CPU exact recomputation in bounded static joined-vacuum branch; GPU arithmetic, drivers and physical events remain separate",
)
(root / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
print(root)
print(json.dumps(result))
raise SystemExit(0 if ok else 1)
