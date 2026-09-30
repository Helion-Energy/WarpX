#!/usr/bin/env python3
"""Native nodal FD4 probe runner; fixed physical profiles, no plot averaging.

The regression gate here is deliberately the independent-normal first slice.
Radial and variable-density rows are reported to expose remaining closure work;
they are never silently accepted as a fourth-order full-operator qualification.
"""

import argparse
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--executable", type=Path, required=True)
parser.add_argument(
    "--inputs",
    type=Path,
    default=Path(__file__).with_name("inputs_test_rz_implicit_thermal"),
)
output_group = parser.add_mutually_exclusive_group(required=True)
output_group.add_argument("--output", type=Path)
output_group.add_argument("--output-parent", type=Path)
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--geometry", choices=["rz", "3d"], default="rz")
parser.add_argument("--ranks", type=int, default=1)
parser.add_argument("--max-grid", type=int, default=32)
parser.add_argument(
    "--profiles", nargs="+", default=["axial", "r2", "r4", "closed", "mixed"]
)
parser.add_argument("--sizes", nargs="+", type=int, default=[32, 64, 128])
parser.add_argument("--density-slope", type=float, default=0.0)
parser.add_argument("--limiter", type=float, default=0.0)
parser.add_argument("--perpendicular", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=0.0)
a = parser.parse_args()
if a.output_parent is not None:
    a.output_parent.mkdir(parents=True, exist_ok=True)
    a.output = Path(tempfile.mkdtemp(prefix="attempt-", dir=a.output_parent))
else:
    a.output.mkdir(parents=True, exist_ok=False)
print(f"Native conduction evidence: {a.output}", flush=True)
results = []
for profile in a.profiles:
    for n in a.sizes:
        run = a.output / f"{profile}-{n}"
        run.mkdir()
        cmd = [
            str(a.executable.resolve()),
            str(a.inputs.resolve()),
            f"amr.n_cell={n} {n}" if a.geometry == "rz" else f"amr.n_cell=8 8 {n}",
            f"amr.max_grid_size={a.max_grid}",
            "geometry.prob_lo=0 0" if a.geometry == "rz" else "geometry.prob_lo=0 0 0",
            "geometry.prob_hi=1 1" if a.geometry == "rz" else "geometry.prob_hi=1 1 1",
            "hybrid_pic_model.density_pedestal=0",
            "hybrid_pic_model.density_pedestal_track_floor=0",
            "amrex.signal_handling=0",
            "hybrid_pic_model.qdsmc_conduction_fd_order=4",
            "hybrid_pic_model.qdsmc_kappa_par(n,Te,t)=1.5*1.380649e-23*n",
            f"hybrid_pic_model.qdsmc_kappa_perp(n,Te,t)=1.5*1.380649e-23*n*{a.perpendicular}",
            "ions.num_particles_per_cell_each_dim=1 1 1",
            f"conduction_test.profile={profile}",
            f"conduction_test.density_slope={a.density_slope}",
            f"conduction_test.limiter={a.limiter}",
            f"conduction_test.perpendicular={a.perpendicular}",
            f"conduction_test.beta={a.beta}",
        ]
        if a.ranks > 1:
            cmd = ["mpiexec", "-n", str(a.ranks), *cmd]
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", WARPX_QDSMC_COND_STATS="1")
        with (run / "run.log").open("w") as log:
            p = subprocess.run(
                cmd, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120
            )
        (run / "COMMAND.json").write_text(json.dumps(cmd, indent=2) + "\n")
        text = (run / "run.log").read_text()
        if p.returncode:
            raise RuntimeError(f"native probe failed: {run}/run.log ({p.returncode})")
        records = re.findall(r"CONDUCTION_ORDER_RESULT (\{.*\})", text)
        assert len(records) == 1, (run, records)
        stats = re.findall(
            r"cond rk stats:.*accepted=(\d+) attempts=(\d+).*t_done_frac=([^\s]+)", text
        )
        assert len(stats) == 3 and all(s == ("1", "1", "1") for s in stats), (
            run,
            stats,
        )
        r = json.loads(records[0])
        r.update(native_single_step_verified=True, ranks=a.ranks, max_grid=a.max_grid)
        assert r["energy_relative"] < 1e-11, r
        assert r["min_temperature"] > 0, r
        if profile == "axial":
            assert r["min_temperature"] >= 1.0 - 1.0e-12, r
            assert r["max_temperature"] <= 3.0 + 1.0e-12, r
        results.append(r)
        print(json.dumps(r), flush=True)
for profile in a.profiles:
    group = [r for r in results if r["profile"] == profile]
    for previous, current in zip(group, group[1:]):
        for key in [
            "axis_error",
            "row1_error",
            "row2_error",
            "linf_error",
            "weighted_l2_error",
        ]:
            if previous[key] > 0 and current[key] > 0:
                current[key + "_order"] = math.log2(previous[key] / current[key])
    if (
        profile == "axial"
        and a.limiter == 0
        and a.density_slope == 0
        and a.beta == 0
        and len(group) > 1
    ):
        last = group[-1]
        # Fixed gate: test the claimed first slice, not a relabelled radial closure.
        for key in ["axis_error", "row1_error", "linf_error"]:
            if a.baseline:
                assert 1.8 < last[key + "_order"] < 2.2, last
            else:
                assert last[key + "_order"] >= 3.7, last
        assert last["temporal_sensitivity"] < 0.03 * last["axis_error"], last
report = {
    "scope": "normal-eligibility first slice; radial closure is diagnostic only",
    "baseline": a.baseline,
    "results": results,
    "passed": True,
}
(a.output / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
