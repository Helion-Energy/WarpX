import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from negative_check import negative_pass

p = argparse.ArgumentParser()
p.add_argument("--plugin", required=True)
p.add_argument("--config", required=True)
p.add_argument("--old-plugin")
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
p.add_argument("--empty-rank", action="store_true")
p.add_argument("--negative", choices=["provider", "combined_vacuum"])
a = p.parse_args()
for required in [a.executable, a.plugin, a.config] + (
    [a.old_plugin] if a.negative == "provider" else []
):
    assert required and Path(required).is_file(), (
        "missing pinned prerequisite",
        required,
    )
source = Path(__file__).resolve().parent
root = Path(tempfile.mkdtemp(prefix="retained-c3-" + a.case + "-", dir=Path.cwd()))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


exe = Path(a.executable).resolve()
steps = a.steps
assert a.ranks > 0 and (0 <= a.reject_step < steps)
assert not a.relocate or a.ranks == 2
plan = dict(
    arguments=vars(a),
    executable=str(exe),
    executable_sha256=sha(exe),
    input_sha256=sha(source / "inputs_rz"),
    contract="Exact accepted fields/history/full particles/RNG and reduced diagnostics; the explicitly identified nine native tangent bands are derived scratch, invalidated on retry. No physical crossing, extra prescribed driver or source admission.",
)
(root / "PLAN.json").write_text(json.dumps(plan, indent=2) + "\n")
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
records = []
arms = ["negative"] if a.negative else ["retry", "clean", "nonretained"]
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
        "circuit.plugin_library="
        + str(Path(a.old_plugin if a.negative == "provider" else a.plugin).resolve()),
        "circuit.plugin_config=" + str(Path(a.config).resolve()),
        "implicit_evolve.native_retained_acceptance=" + str(int(arm != "nonretained")),
        "hybrid_pic_model.electron_energy_mode=" + a.mode,
        "implicit_evolve.use_mass_matrices_jacobian=" + str(a.mm),
        "implicit_evolve.mass_matrices_density_projection=" + str(a.mm),
    ]
    if a.empty_rank:
        cmd += [
            "amr.max_grid_size=32",
            "amr.refine_grid_layout=0",
            "fixture.require_empty_rank=1",
        ]
    if a.negative == "combined_vacuum":
        cmd += [
            "endpoint_diagnostic.joined_vacuum=1",
            "hybrid_pic_model.darwin_vacuum_recovery=1",
            "hybrid_pic_model.darwin_vacuum_recovery_components=all",
            "hybrid_pic_model.darwin_vacuum_recovery_operator=poisson",
            "hybrid_pic_model.darwin_vacuum_recovery_mask=vacuum",
            "hybrid_pic_model.darwin_vacuum_recovery_relaxation_time=0",
            "endpoint_diagnostic.vacuum_edge_policy=native_edge_candidate",
            "hybrid_pic_model.darwin_vacuum_recovery_frozen_mask=1",
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
        r"RETAINED_C3_ACCEPTANCE_PASS steps=(\d+) rejected=(\d+) before_calls=(\d+) after_calls=(\d+) migrated=(\d+)",
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
        and text.count("Device circuit accepted:") == steps
    )
    if arm == "retry":
        ok = ok and text.count("RETAINED_ROLLBACK_PASS") == 1
    if arm != "nonretained" and not a.negative:
        ok = ok and text.count("RETAINED_C3_CLOSED_CANDIDATE") == nbefore + nafter
    if a.empty_rank and not a.negative:
        ok = ok and text.count("RETAINED_C3_EMPTY_RANK count=1") == 1
    if a.relocate and arm == "retry":
        ok = ok and bool(m) and int(m.group(5)) > 0
    if a.negative:
        ok = negative_pass(text, rc, a.negative)
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
        for q in sorted((root / "clean").glob("circuit-accepted_*-rank*/*")):
            b = root / arm / q.parent.name / q.name
            comparisons.append(
                dict(
                    kind="native_circuit_accepted",
                    arm=arm,
                    file=str(q.relative_to(root / "clean")),
                    exact=q.read_bytes() == b.read_bytes(),
                    clean_sha256=sha(q),
                    other_sha256=sha(b),
                )
            )
    for q in sorted((root / "retry").glob("circuit-restored_*-rank*/*")):
        b = Path(str(q).replace("circuit-restored_", "circuit-old_"))
        comparisons.append(
            dict(
                kind="native_circuit_rollback",
                file=str(q.relative_to(root / "retry")),
                exact=q.read_bytes() == b.read_bytes(),
                old_sha256=sha(b),
                restored_sha256=sha(q),
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
    bool(a.negative)
    or len(records) == 3
    and len(comparisons) >= 4 * steps
    and all(x["exact"] for x in comparisons)
)
result = dict(
    case=a.case,
    pass_=ok,
    records=records,
    comparisons=comparisons,
    scope="CPU exact recomputation in positive-density native C3 branch; GPU arithmetic, combined vacuum and physical events remain separate",
)
(root / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
print(root)
print(json.dumps(result))
raise SystemExit(0 if ok else 1)
