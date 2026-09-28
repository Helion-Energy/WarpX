import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path

p = argparse.ArgumentParser()
for name in ["executable", "input", "plugin", "config"]:
    p.add_argument("--" + name, required=True)
p.add_argument(
    "--case", choices=["density", "nonfinite", "stale", "retry"], required=True
)
p.add_argument(
    "--mode", default="coupled_jfnk", choices=["coupled_jfnk", "decoupled_jfnk"]
)
p.add_argument("--ranks", type=int, default=1)
p.add_argument("--mpi")
a = p.parse_args()
root = Path(tempfile.mkdtemp(prefix="circuit-mm-" + a.case + "-", dir=".")).resolve()


def sha(q):
    return hashlib.sha256(Path(q).read_bytes()).hexdigest()


def run(label, extra):
    d = root / label
    d.mkdir()
    shutil.copy2(a.input, d / "inputs")
    cmd = [
        str(Path(a.executable).resolve()),
        "inputs",
        "circuit.plugin_library=" + str(Path(a.plugin).resolve()),
        "circuit.plugin_config=" + str(Path(a.config).resolve()),
        "hybrid_pic_model.electron_energy_mode=" + a.mode,
        "implicit_evolve.use_mass_matrices_jacobian=1",
        "implicit_evolve.mass_matrices_density_projection=1",
        "implicit_evolve.fused_mass_matrices_deposit=0",
        *extra,
    ]
    if a.ranks > 1:
        assert a.mpi
        cmd = [a.mpi, "-n", str(a.ranks), *cmd]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    record = dict(
        command=cmd,
        executable_sha256=sha(a.executable),
        input_sha256=sha(d / "inputs"),
        provider_sha256=sha(a.plugin),
        OMP_NUM_THREADS="1",
    )
    (d / "COMMAND.json").write_text(json.dumps(record, indent=2) + "\n")
    start = time.time()
    with (d / "run.log").open("w") as log:
        r = subprocess.run(
            cmd, cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=850
        )
    text = (d / "run.log").read_text()
    (d / "RESULT.json").write_text(
        json.dumps(
            dict(
                returncode=r.returncode,
                seconds=time.time() - start,
                log_sha256=sha(d / "run.log"),
            ),
            indent=2,
        )
        + "\n"
    )
    return d, r.returncode, text


def records(q):
    out = {}
    with q.open("rb") as f:
        while h := f.read(16):
            assert len(h) == 16
            n, m = struct.unpack("<QQ", h)
            name = f.read(n).decode()
            value = f.read(m)
            assert len(value) == m and name not in out
            out[name] = value
    return out


if a.case == "retry":
    clean, rc, log = run("clean", ["retry_test.fail_first=0"])
    assert rc == 0 and "COMPANION_RETRY_PASS" in log, log[-4000:]
    trial, rc, log = run("retry", ["retry_test.fail_first=1"])
    assert rc == 0 and all(
        x in log
        for x in [
            "COMPANION_RETRY_PASS",
            "COMPANION_ROLLBACK_PASS",
            "INJECTED_AFTER_RETAINED_CURRENT",
        ]
    ), log[-4000:]
    summary = []
    for rank in range(a.ranks):
        x = records(clean / f"accepted.rank{rank}.bin")
        y = records(trial / f"accepted.rank{rank}.bin")
        assert x == y, "accepted retry differs from clean"
        summary.append(
            dict(
                rank=rank,
                records=len(x),
                bytes=sum(map(len, x.values())),
                all_exact=True,
            )
        )
        for f in (clean / f"circuit-accepted-rank{rank}").glob("*"):
            assert (
                f.read_bytes()
                == (trial / f"circuit-accepted-rank{rank}" / f.name).read_bytes()
            )
    for f in (clean / "rng-accepted").glob("*"):
        assert f.read_bytes() == (trial / "rng-accepted" / f.name).read_bytes()
    (root / "COMPARISON.json").write_text(json.dumps(summary, indent=2) + "\n")
else:
    extra = [
        "density_test.stale=" + str(int(a.case == "stale")),
        "density_test.nonfinite=" + str(int(a.case == "nonfinite")),
    ]
    d, rc, log = run(a.case, extra)
    if a.case == "stale":
        assert (
            rc != 0
            and "MM continuity requires a fresh retained current increment" in log
        ), log[-4000:]
    else:
        assert rc == 0 and "DENSITY_INCREMENT_PASS" in log, log[-4000:]
        if a.case == "nonfinite":
            assert "DENSITY_NONFINITE propagated=1" in log
(root / "PASS.json").write_text(
    json.dumps(dict(pass_=True, case=a.case, mode=a.mode, ranks=a.ranks), indent=2)
    + "\n"
)
print(root)
