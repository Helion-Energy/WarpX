#!/usr/bin/env python3
"""Check ion-step order, subcycle independence and the old-form negative control."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

exe, inputs = map(str, map(Path, sys.argv[1:3]))
root = Path(tempfile.mkdtemp(prefix="inertia-transport-", dir="."))
# Keep an expected MPI_Abort inside its own launch step. GPU/Slurm harnesses
# run this controller outside srun and pass an argv prefix as JSON; serial
# CTest needs no launcher. No shell expansion is performed.
launcher = json.loads(os.environ.get("WARPX_TEST_LAUNCHER_JSON", "[]"))
assert isinstance(launcher, list) and all(isinstance(a, str) for a in launcher)


def run(name, *overrides, fails=False):
    work = root / name
    work.mkdir()
    result = subprocess.run(
        [*launcher, exe, inputs, *overrides], cwd=work, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "OMP_NUM_THREADS": "1"}, check=False,
    )
    (work / "run.log").write_text(result.stdout)
    rows = [json.loads(line.split(" ", 1)[1]) for line in result.stdout.splitlines()
            if line.startswith("EDGE_TRANSPORT {")]
    assert rows and rows[-1]["crossed_cells"] > 2, result.stdout[-4000:]
    if fails:
        assert result.returncode != 0
        assert "The moving-edge inertia work must close its energy budget" in result.stdout
        assert max(row["energy_relative_defect"] for row in rows) > 0.01
    else:
        assert result.returncode == 0, result.stdout[-4000:]
        assert "EDGE_TRANSPORT_COMPLETE" in result.stdout
    print(json.dumps({"case": name, "exit": result.returncode, "last": rows[-1]}), flush=True)
    return rows


base = run("base")
half = run("half_dt", "warpx.const_dt=1.25e-8", "edge_transport.steps=200")
sub8 = run("sub8", "hybrid_pic_model.substeps=8")
ratio = base[-1]["energy_relative_defect"] / half[-1]["energy_relative_defect"]
assert 3.8 < ratio < 4.2, ratio
for quantity in ("electron_bulk_J", "work_J", "E_max_V_m", "radius_rms_m"):
    a, b = base[-1][quantity], sub8[-1][quantity]
    assert abs(a-b) <= 1e-10 * max(abs(a), abs(b), 1e-30), (quantity, a, b)
run("old_split", "hybrid_pic_model.electron_inertia_momentum_flux=0", fails=True)
print(f"TRANSPORT_ORDER_AND_NEGATIVE_CONTROL_PASS ratio={ratio:.12g}")
