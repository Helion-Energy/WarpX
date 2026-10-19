import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path


def vector(values):
    return " ".join(map(str, values))


r = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description="Native Cartesian Fourier-mode convergence")
ap.add_argument("--executable", type=Path, required=True)
ap.add_argument("--output-parent", type=Path, required=True)
ap.add_argument("--launcher-json", default="[]")
a = ap.parse_args()
a.output_parent.mkdir(parents=True, exist_ok=True)
out = Path(tempfile.mkdtemp(prefix="attempt-", dir=a.output_parent))
launcher = json.loads(a.launcher_json)
assert isinstance(launcher, list) and all(isinstance(x, str) for x in launcher)
rows = []
for profile in ["x", "y", "z", "oblique"]:
    for n in [32, 64, 128]:
        name = f"{profile}-{n}"
        cwd = out / name
        cwd.mkdir()
        axis = {"x": 0, "y": 1, "z": 2}.get(profile)
        if axis is None:
            shape = [n, n, n]
            mode = [1, 1, 1]
            B = [1, 2, 3]
            boundary = ["periodic"] * 3
            particles = boundary.copy()
            perp = 0.01
        else:
            shape = [8, 8, 8]
            shape[axis] = n
            mode = [0, 0, 0]
            mode[axis] = 1
            B = mode.copy()
            boundary = ["pec"] * 3
            boundary[axis] = "periodic"
            particles = ["reflecting"] * 3
            particles[axis] = "periodic"
            perp = 1
        overrides = [
            f"amr.n_cell={vector(shape)}",
            "amr.max_grid_size=32",
            "geometry.prob_lo=0 0 0",
            "geometry.prob_hi=1 1 1",
            f"boundary.field_lo={vector(boundary)}",
            f"boundary.field_hi={vector(boundary)}",
            f"boundary.particle_lo={vector(particles)}",
            f"boundary.particle_hi={vector(particles)}",
            f"order_test.mode={vector(mode)}",
            f"order_test.magnetic={vector(B)}",
            f"order_test.perpendicular={perp}",
            "hybrid_pic_model.qdsmc_kappa_par(n,Te,t)=1.5*1.380649e-23*n",
            f"hybrid_pic_model.qdsmc_kappa_perp(n,Te,t)=1.5*1.380649e-23*n*{perp}",
            "hybrid_pic_model.qdsmc_conduction_Te_floor=0",
        ]
        cmd = [
            *launcher,
            str(a.executable.resolve()),
            str(r / "inputs_test_3d_conduction_boundary"),
            *overrides,
            "amrex.the_arena_init_size=0",
            "amrex.the_arena_is_managed=0",
        ]
        (cwd / "COMMAND.json").write_text(json.dumps(cmd, indent=2) + "\n")
        with (cwd / "stdout.log").open("x") as f:
            proc = subprocess.run(
                cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, timeout=300
            )
        hits = [
            json.loads(x.split("CARTESIAN_ORDER_RESULT ", 1)[1])
            for x in (cwd / "stdout.log").read_text().splitlines()
            if x.startswith("CARTESIAN_ORDER_RESULT ")
        ]
        row = dict(profile=profile, n=n, returncode=proc.returncode, results=hits)
        rows.append(row)
        (cwd / "RECEIPT.json").write_text(json.dumps(row, indent=2) + "\n")
        print(row, flush=True)
        if proc.returncode or len(hits) != 1:
            raise SystemExit("Native convergence fixture failed: " + name)
failures = []
orders = []
for profile in ["x", "y", "z", "oblique"]:
    group = [x for x in rows if x["profile"] == profile]
    for before, after in zip(group, group[1:]):
        a = before["results"][0]
        b = after["results"][0]
        order = math.log2(a["linf_error"] / b["linf_error"])
        orders.append(dict(profile=profile, n=after["n"], order=order))
        if profile == "oblique" and not b["linf_error"] < a["linf_error"]:
            failures.append([profile, "nondecreasing", after["n"]])
    last = group[-1]["results"][0]
    if profile != "oblique":
        if orders[-1]["order"] < 3.7:
            failures.append([profile, "order", orders[-1]["order"]])
        if last["temporal_sensitivity"] >= 0.03 * last["linf_error"]:
            failures.append([profile, "temporal_sensitivity", last])
report = dict(passed=not failures, failures=failures, orders=orders, rows=rows)
(out / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
if failures:
    raise SystemExit("Convergence acceptance failed")
