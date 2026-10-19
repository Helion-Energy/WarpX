import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

r = Path(__file__).resolve().parent
p = argparse.ArgumentParser(
    description="Native3Dsource-stage and marker-capacity qualification"
)
p.add_argument("--order-executable", type=Path, required=True)
p.add_argument("--capacity-executable", type=Path, required=True)
p.add_argument("--output-parent", type=Path, required=True)
p.add_argument("--launcher-json", default="[]")
p.add_argument("--filter", type=int, default=1, choices=[0, 1])
a = p.parse_args()
launcher = json.loads(a.launcher_json)
assert isinstance(launcher, list) and all(isinstance(x, str) for x in launcher)
a.output_parent.mkdir(parents=True, exist_ok=True)
out = Path(tempfile.mkdtemp(prefix="attempt-", dir=a.output_parent))
rows = []


def run(name, kind, extra, expected=0):
    cwd = out / name
    cwd.mkdir()
    cmd = [
        *launcher,
        str(
            (a.order_executable if kind == "order" else a.capacity_executable).resolve()
        ),
        str(r / ("inputs_3d_source_stage_" + kind)),
        *extra,
        f"warpx.use_filter={a.filter}",
        "amr.max_grid_size=8",
        "amrex.the_arena_init_size=0",
        "amrex.the_arena_is_managed=0",
    ]
    (cwd / "COMMAND.json").write_text(json.dumps(cmd, indent=2) + "\n")
    with (cwd / "stdout.log").open("x") as f:
        proc = subprocess.run(
            cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, timeout=300
        )
    records = [
        json.loads(line.split(" ", 1)[1])
        for line in (cwd / "stdout.log").read_text().splitlines()
        if line.startswith(("PC_SOURCE_ORDER ", "ENERGY_RECOVERY "))
    ]
    row = dict(
        name=name,
        kind=kind,
        returncode=proc.returncode,
        expected=expected,
        results=records,
    )
    rows.append(row)
    (cwd / "RECEIPT.json").write_text(json.dumps(row, indent=2) + "\n")
    (out / "RESULTS.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(row, flush=True)
    assert proc.returncode == expected and len(records) == 1
    return records[0]


orders = []
for stopping in [0, 1]:
    for stage in [0, 1]:
        errors = []
        for steps in [16, 32, 64]:
            result = run(
                f"order-stopping{stopping}-stage{stage}-steps{steps}",
                "order",
                [
                    f"audit.steps={steps}",
                    f"audit.stopping={stopping}",
                    f"hybrid_pic_model.qdsmc_source_stage_density={stage}",
                ],
            )
            errors.append(result["max_error_K"])
        order = math.log2(errors[-2] / errors[-1])
        assert (0.9 < order < 1.2) if stage == 0 else (1.9 < order < 2.1)
        orders.append(dict(stopping=stopping, stage_density=stage, order=order))
cases = {
    "static": [],
    "drop": ["audit.nold=1e20", "audit.nnew=1.01e18"],
    "rise": ["audit.nold=1.01e18", "audit.nnew=1e20"],
    "empty-old-pedestal": ["audit.nold=0", "audit.nnew=1e20"],
    "below-old-no-pedestal": [
        "audit.nold=1e15",
        "audit.nnew=1e20",
        "hybrid_pic_model.density_pedestal=0",
    ],
    "closed-physical-gate": ["audit.nold=1e20", "audit.nnew=1e15"],
    "physical-coefficient-density": [
        "audit.nold=1e20",
        "audit.nnew=1.01e18",
        "audit.eta_factor=0.101",
        "hybrid_pic_model.joule_heating_resistivity(rho,J,Te,t)=1.e-5*rho/(1.602176634e-19*1.e19)",
    ],
    "static-empty": ["audit.nold=0", "audit.nnew=0", "audit.heat=0"],
}
for pedestal in [0, 1]:
    for name, extra in cases.items():
        result = run(
            f"capacity-fullped{pedestal}-{name}",
            "capacity",
            [
                "audit.old_receiver=1",
                f"hybrid_pic_model.qdsmc_full_pedestal_transport={pedestal}",
                *extra,
            ],
        )
        assert (
            result["native_map_verified"]
            and result["source_on_old_state_contract_pass"]
        )
result = run(
    "negative-wrong-receiver",
    "capacity",
    ["audit.old_receiver=0", "audit.nold=1e20", "audit.nnew=1.01e18"],
    expected=2,
)
assert result["native_map_verified"] and not result["source_on_old_state_contract_pass"]
(out / "PASS.json").write_text(
    json.dumps(dict(passed=True, orders=orders, cases=rows), indent=2) + "\n"
)
print("THREE_DIMENSIONAL_SOURCE_STAGE_PASS", len(rows), flush=True)
