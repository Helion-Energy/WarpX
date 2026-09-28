#!/usr/bin/env python3
"""Read-only full native inventory audit; a nonzero physical defect is reported.

Passing inventory checks does NOT assert exact closed-box conservation or full
method temporal order. All source/field parameters remain in the input deck.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
EPS = sys.float_info.epsilon

RECEIPT_PREFIXES = (
    "Eulerian accepted",
    "Expected OU endpoint",
    "Expected OU NR",
    "Expected OU population thermal partner [J]:",
)
ASSIGNMENT = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)=([^\s;]+)")
ACTIVE_SOURCE_CHANNELS = {"electron_total", "joule", "relaxation"}
ENDPOINT_MISMATCH_LIMIT = 1.0e-11  # Native CommitIonExchange quality gate.


def values(line):
    return {k: float(v) for k, v in re.findall(r"([A-Za-z_]+)=(" + NUMBER + r")", line)}


def one(lines, prefix):
    found = [line for line in lines if line.startswith(prefix)]
    assert len(found) == 1, (prefix, found)
    return values(found[0])


def physics_receipts(record):
    return [
        line
        for line in (Path(record["cwd"]) / "run.log").read_text().splitlines()
        if line.startswith(RECEIPT_PREFIXES)
    ]


def parse_receipt(line):
    """Preserve labels/text exactly; recognize and reject nonfinite numbers."""
    numbers = {}

    def replace(match):
        key, token = match.groups()
        if key in ("species", "channel"):
            return match.group(0)
        assert key not in numbers, ("Duplicate receipt field", key, line)
        value = float(token)
        assert math.isfinite(value), ("Nonfinite receipt field", key, line)
        numbers[key] = value
        return key + "=<number>"

    signature = ASSIGNMENT.sub(replace, line)
    assert numbers, ("Missing numeric receipt fields", line)
    return signature, numbers


def receipt_budget(lines, gamma):
    """Positive scales for this closed, cold, single-proton fixture.

    The count-derived gamma is shared only by callback twins with the same
    grid/population. Every positive work/inventory/weight scale is independently
    read from that run. No observed difference enters a scale or allowance.
    """
    for line in lines:
        parse_receipt(line)
    work = one(lines, "Eulerian accepted ion electric work [J]:")
    exchange = one(lines, "Eulerian accepted ion exchange [J]:")
    expected = one(lines, "Expected OU endpoint audit [J]:")
    moments = one(lines, "Expected OU NR moment work [J]:")
    energy = one(lines, "Eulerian accepted energy [J]:")
    source = [
        values(line)["energy"]
        for line in lines
        if line.startswith("Eulerian accepted source [J]:")
    ]
    # K_old+K_endpoint bounds signed electric work. The thermal and bulk
    # constituents are uniformly signed in this fixture, as in analyze().
    # Absolute component work and Cauchy scales cover signed eta/remainder
    # channels. A signed net work alone is never used as its own scale.
    constituents = dict(
        kinetic_old=abs(work["kinetic_old"]),
        kinetic_endpoint=abs(work["kinetic_endpoint"]),
        realized_exchange=abs(exchange["realized_total"]),
        conditional_relaxation=abs(expected["relaxation"]),
        conditional_redirect=abs(expected["redirect"]),
        thermal_relaxation=abs(moments["thermal_relaxation"]),
        deterministic_bulk=abs(moments["deterministic_bulk"]),
        source_channels=math.fsum(abs(v) for v in source),
    )
    for key in (
        "global_eta_abs",
        "applied_remainder_abs",
        "global_eta_cauchy_bound",
        "remainder_cauchy_bound",
    ):
        assert work[key] >= 0, (key, work[key])
        constituents[key] = work[key]
    scales = dict(
        energy=math.fsum(constituents.values()),
        stored_U=abs(energy["old"]) + abs(energy["new"]),
        stored_K=abs(work["kinetic_old"]) + abs(work["kinetic_endpoint"]),
        weight=abs(work["physical_weight"]),
        dimensionless=1.0,
    )
    assert 0 < gamma < 1 and all(math.isfinite(v) and v >= 0 for v in scales.values())
    return dict(
        gamma=gamma,
        positive_constituents=constituents,
        scales=scales,
        allowances={key: gamma * v for key, v in scales.items()},
    )


def receipt_field_family(line, key):
    # Only actual flags are exact flags. In particular, the endpoint relative
    # mismatch is a dimensionless residual, not a boolean or prescribed zero.
    if key == "resistive_push_correction":
        return "flag"
    if key in ("particles", "count", "step"):
        return "count"
    if line.startswith("Expected OU endpoint work relative mismatch="):
        assert key == "mismatch"
        return "dimensionless"
    channel = re.search(r"channel=(\S+)", line)
    if (channel and channel.group(1) not in ACTIVE_SOURCE_CHANNELS) or key in (
        "requested_redirect",
        "expected_redirect",
        "redirect",
        "nr_minus_selected",
        "quadrature_estimate",
    ):
        return "zero"
    if line.startswith("Eulerian accepted energy [J]:"):
        if key in ("conduction", "absorption"):
            return "zero"
        if key in ("old", "new", "trajectory_endpoint"):
            return "stored_U"
    if key in ("kinetic_old", "kinetic_endpoint"):
        return "stored_K"
    if key == "physical_weight":
        return "weight"
    assert "[J]:" in line, ("Unclassified numeric receipt", key, line)
    return "energy"


def compare_roundoff_receipts(left, right, gamma, label):
    """Same fixed bounds for audit-off and an unchanged audit-on repeat."""
    assert len(left) == len(right), (
        label,
        "Receipt count changed",
        len(left),
        len(right),
    )
    lb, rb = receipt_budget(left, gamma), receipt_budget(right, gamma)
    checks = []
    quality = []
    for index, (ll, rr) in enumerate(zip(left, right)):
        ls, lv = parse_receipt(ll)
        rs, rv = parse_receipt(rr)
        assert ls == rs and lv.keys() == rv.keys(), (label, ll, rr)
        for key, x in lv.items():
            y = rv[key]
            family = receipt_field_family(ll, key)
            if family == "flag":
                assert x in (0.0, 1.0) and y in (0.0, 1.0), (label, key, x, y)
                allowed = 0.0
            elif family == "count":
                assert x >= 0 and y >= 0 and x.is_integer() and y.is_integer(), (
                    label,
                    key,
                    x,
                    y,
                )
                allowed = 0.0
            elif family == "zero":
                assert x == y == 0.0, (label, "Configured-zero channel", key, x, y)
                allowed = 0.0
            else:
                allowed = lb["allowances"][family] + rb["allowances"][family]
            if family == "dimensionless":
                assert (
                    0 <= x <= ENDPOINT_MISMATCH_LIMIT
                    and 0 <= y <= ENDPOINT_MISMATCH_LIMIT
                ), (
                    label,
                    "Native endpoint quality gate",
                    x,
                    y,
                    ENDPOINT_MISMATCH_LIMIT,
                )
                quality.append(
                    dict(
                        receipt=index,
                        field=key,
                        left=x,
                        right=y,
                        upper_limit=ENDPOINT_MISMATCH_LIMIT,
                    )
                )
            difference = y - x
            assert abs(difference) <= allowed, (label, key, difference, allowed, ll, rr)
            checks.append(
                dict(
                    receipt=index,
                    signature=ls,
                    field=key,
                    family=family,
                    left=x,
                    right=y,
                    difference=difference,
                    allowance=allowed,
                )
            )
    return dict(
        label=label,
        observed_text_equal=left == right,
        left_budget=lb,
        right_budget=rb,
        endpoint_quality=quality,
        checks=checks,
        maximum_allowance_fraction=max(
            (
                abs(c["difference"]) / c["allowance"]
                for c in checks
                if c["allowance"] > 0
            ),
            default=0.0,
        ),
    )


def analyze(record, *, include_reported_solver_defect=False):
    lines = (Path(record["cwd"]) / "run.log").read_text().splitlines()
    old = [
        values(line) for line in lines if line.startswith("SOURCE_GATE_INVENTORY_OLD")
    ]
    new = [
        values(line) for line in lines if line.startswith("SOURCE_GATE_INVENTORY_NEW")
    ]
    assert [int(v["component"]) for v in old] == list(range(7))
    assert [int(v["component"]) for v in new] == list(range(7))
    ions = [
        values(line) for line in lines if line.startswith("SOURCE_GATE_ION_INVENTORY")
    ]
    assert len(ions) == 1, "This closed matched-proton gate has exactly one ion species"
    ion = ions[0]
    total = one(lines, "SOURCE_GATE_TOTAL_INVENTORY")
    exchange = one(lines, "Eulerian accepted ion exchange [J]:")
    expected = one(lines, "Expected OU endpoint audit [J]:")
    moments = one(lines, "Expected OU NR moment work [J]:")
    thermal = one(lines, "Expected OU population thermal partner [J]:")
    work = one(lines, "Eulerian accepted ion electric work [J]:")
    energy = one(lines, "Eulerian accepted energy [J]:")
    sources = {}
    for line in lines:
        if line.startswith("Eulerian accepted source [J]:"):
            channel = re.search(r"channel=(\S+)", line).group(1)
            sources[channel] = values(line)["energy"]
    # Conservative summation allowance. Stable differences use the absolute
    # differences actually reduced, not O(1) total energies that cancel later.
    visits = sum(v["entries"] for v in old + new) + 2 * ion["particles"]
    k = 32 * visits + 256
    assert k * EPS < 1
    gamma = k * EPS / (1 - k * EPS)
    inventory_scale = sum(abs(v["energy"]) for v in old + new) + ion["old"] + ion["new"]
    norm_bound = gamma * inventory_scale
    stable_scale = sum(v["absolute_delta"] for v in new) + ion["old"] + ion["new"]
    # The population thermal and bulk decomposition bounds the absolute
    # conditional work for this cold, uniformly signed matched source fixture.
    stable_scale += abs(moments["thermal_relaxation"]) + abs(
        moments["deterministic_bulk"]
    )
    stable_scale += abs(exchange["realized_total"]) + sum(
        abs(v) for v in sources.values()
    )
    bound = gamma * stable_scale
    checks = {}

    def check(label, defect, allowed=bound):
        checks[label] = dict(defect=defect, allowance=allowed)
        assert math.isfinite(defect) and abs(defect) <= allowed, (
            record["label"],
            label,
            defect,
            allowed,
        )

    physical_volume = math.pi * 0.25**2 * 0.25
    for n, (a, b) in enumerate(zip(old, new)):
        check(f"native_norm_old_{n}", a["energy"] - a["native"], norm_bound)
        check(f"native_norm_new_{n}", b["energy"] - b["native"], norm_bound)
        check(
            f"stable_subtraction_{n}", b["stable_delta"] - b["difference"], norm_bound
        )
        check(
            f"physical_volume_{n}",
            a["clipped_volume"] - physical_volume,
            gamma * physical_volume,
        )
    for line in lines:
        if line.startswith("SOURCE_GATE_BOUNDARY"):
            b = values(line)
            assert (
                b["lost_weight"]
                == b["lost_charge"]
                == b["lost_energy"]
                == b["tangential_wall_E"]
                == 0
            ), b
    check("weight", ion["new_weight"] - ion["old_weight"], gamma * ion["old_weight"])
    check("initial_K", ion["old"] - work["kinetic_old"])
    check(
        "endpoint_plus_kick_K",
        ion["delta"]
        - (work["kinetic_endpoint"] - work["kinetic_old"] + exchange["realized_total"]),
    )
    check("old_U", old[6]["energy"] - energy["old"], norm_bound)
    check("new_U", new[6]["energy"] - energy["new"], norm_bound)
    check(
        "cell_U_balance",
        new[6]["stable_delta"]
        - sum(
            energy[k]
            for k in (
                "source",
                "conduction",
                "advection",
                "compression",
                "absorption",
                "defect",
            )
        ),
    )
    for key, value in sources.items():
        if key not in ("electron_total", "joule", "relaxation"):
            assert value == 0, (key, value)
    for key in ("conduction", "absorption"):
        assert energy[key] == 0, (key, energy[key])
    assert (
        exchange["requested_redirect"]
        == exchange["expected_redirect"]
        == moments["redirect"]
        == 0
    )
    check(
        "thermal_pair",
        thermal["electron_relaxation"] + thermal["ion_population_thermal"],
        64
        * EPS
        * (
            abs(thermal["electron_relaxation"]) + abs(thermal["ion_population_thermal"])
        ),
    )
    delta_b = sum(v["stable_delta"] for v in new[:3])
    delta_el = sum(v["stable_delta"] for v in new[3:6])
    delta_u = new[6]["stable_delta"]
    measured = delta_b + delta_el + delta_u + ion["delta"]
    check("actual_inventory", measured - total["actual_total_defect"])
    fluctuation = (
        exchange["realized_total"] - expected["relaxation"] - expected["redirect"]
    )
    nr_defect = measured - fluctuation
    # The total expected defect is independently reconstructed from field,
    # exact particle electric work, thermal PDE flux/source and OU moments.
    budget = dict(
        magnetic=delta_b,
        longitudinal=delta_el,
        joule=sources["joule"],
        ion_electric=work["total_field"],
        ou_bulk=moments["deterministic_bulk"],
        compression=energy["compression"],
        advection=energy["advection"],
        conduction=energy["conduction"],
    )
    if include_reported_solver_defect:
        # The multi-step inertia audit books the independently reported
        # accepted cell-U residual. The original one-step gate stays unchanged.
        budget["thermal_solver_defect"] = energy["defect"]
    reconstructed = math.fsum(budget.values())
    check("conditional_budget", nr_defect - reconstructed)
    check(
        "thermal_source_partition",
        sources["electron_total"] - sources["joule"] - sources["relaxation"],
    )
    transfer_scale = math.fsum(abs(v) for v in budget.values())
    split = dict(
        magnetic_plus_joule=delta_b + sources["joule"],
        longitudinal=delta_el,
        non_eta_ion_electric=work["total_field"] - work["global_eta"],
        eta_plus_ou_bulk=work["global_eta"] + moments["deterministic_bulk"],
        compression=energy["compression"],
        advection=energy["advection"],
        actual_remainder=work["applied_remainder"],
    )
    # NR-minus-relativistic interval, so subtract its upper/lower endpoints.
    interval = [
        nr_defect - expected["convention_upper"],
        nr_defect - expected["convention_lower"],
    ]
    return dict(
        label=record["label"],
        inventory_checks_pass=True,
        dt=total["time"],
        old=old,
        new=new,
        ions=ions,
        total=total,
        energy=energy,
        sources=sources,
        exchange=exchange,
        expected=expected,
        moments=moments,
        thermal=thermal,
        work=work,
        actual_total_defect=measured,
        realized_minus_NR_expectation=fluctuation,
        conditional_NR_total_defect=nr_defect,
        conditional_relativistic_defect_interval=interval,
        arithmetic_allowance=bound,
        roundoff_gamma=gamma,
        norm_arithmetic_allowance=norm_bound,
        conditional_defect_resolved=abs(nr_defect)
        > bound
        + max(abs(expected["convention_lower"]), abs(expected["convention_upper"])),
        conditional_defect_per_second=nr_defect / total["time"],
        conditional_relative_transfer_defect=nr_defect / transfer_scale,
        transfer_scale=transfer_scale,
        budget=budget,
        split=split,
        checks=checks,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("executable", type=Path, nargs="?")
    p.add_argument("input", type=Path, nargs="?")
    p.add_argument(
        "--manifest", type=Path, help="Analyze already preserved native runs"
    )
    p.add_argument("--with-mpi", action="store_true")
    p.add_argument(
        "--receipt-check",
        choices=("exact", "roundoff"),
        default="exact",
        help="Exact accepted text (CPU default), or derived numeric bounds plus an unchanged repeat",
    )
    p.add_argument("--output", type=Path, default=Path("FULL_ENERGY_ANALYSIS.json"))
    a = p.parse_args()
    if a.manifest:
        records = json.loads(a.manifest.read_text())
    else:
        assert a.executable and a.input
        base = a.input.read_text()
        cases = []
        for n in (16, 32, 64):
            for d, dt in enumerate((1e-9, 5e-10, 2.5e-10)):
                cases.append(
                    (
                        f"full-energy-n{n}-dt{d}",
                        1,
                        f"amr.n_cell = {n} {n}\nions.num_particles_per_cell_each_dim = {64 // n} 4 {64 // n}\nwarpx.const_dt = {dt:.17g}\n",
                        True,
                    )
                )
        cases += [
            (
                "full-energy-decoupled",
                1,
                "hybrid_pic_model.electron_energy_mode = decoupled_jfnk\n",
                True,
            ),
            ("full-energy-box8", 1, "amr.max_grid_size = 8\n", True),
            ("full-energy-audit-off", 1, "", False),
        ]
        if a.with_mpi:
            cases.append(("full-energy-mpi2", 2, "amr.max_grid_size = 8\n", True))
        if a.receipt_check == "roundoff":
            cases.append(("full-energy-repeat", 1, "", True))
        records = []
        for label, ranks, extra, audit in cases:
            cwd = Path(tempfile.mkdtemp(prefix=label + "-", dir=Path.cwd())).resolve()
            inp = base + "\n" + extra + f"source_gate.energy_audit = {int(audit)}\n"
            (cwd / "inputs").write_text(inp)
            prefix = []
            if ranks > 1:
                prefix = (
                    shlex.split(os.environ.get("MPIEXEC", "mpiexec"))
                    + shlex.split(os.environ.get("MPIEXEC_PREFLAGS", ""))
                    + ["-n", str(ranks)]
                )
            cmd = prefix + [str(a.executable.resolve()), "inputs"]
            with (cwd / "run.log").open("w") as log:
                proc = subprocess.run(
                    cmd,
                    cwd=cwd,
                    env=dict(os.environ, OMP_NUM_THREADS="1"),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            records.append(
                dict(
                    label=label,
                    rc=proc.returncode,
                    cwd=str(cwd),
                    cmd=cmd,
                    input_sha256=hashlib.sha256(inp.encode()).hexdigest(),
                )
            )
            Path("FULL_ENERGY_COMMANDS.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
            assert proc.returncode == 0, (label, cwd)
    assert all(record["rc"] == 0 for record in records), records
    rows = [analyze(record) for record in records if "audit-off" not in record["label"]]

    def get(label):
        return next(row for row in rows if row["label"] == label)

    suffix = "-r02" if any(row["label"].endswith("-r02") for row in rows) else ""
    reference = get("full-energy-n16-dt1" + suffix)
    for label in ("full-energy-decoupled", "full-energy-box8", "full-energy-mpi2"):
        twin = next((r for r in rows if r["label"] == label + suffix), None)
        if twin:
            allowance = twin["arithmetic_allowance"] + reference["arithmetic_allowance"]
            assert (
                abs(
                    twin["conditional_NR_total_defect"]
                    - reference["conditional_NR_total_defect"]
                )
                <= allowance
            )
    # Same executable, seed, input, box and native update. CPU keeps its original
    # exact receipt gate; CUDA can explicitly select numeric roundoff checks.
    on = next(r for r in records if r["label"] == "full-energy-n16-dt1" + suffix)
    off = next(r for r in records if r["label"] == "full-energy-audit-off" + suffix)
    on_receipts, off_receipts = physics_receipts(on), physics_receipts(off)
    receipt_checks = []
    if a.receipt_check == "exact":
        assert on_receipts == off_receipts, (
            "Read-only energy audit changed accepted physics receipts"
        )
    else:
        repeat = next(r for r in records if r["label"] == "full-energy-repeat" + suffix)
        repeated = get("full-energy-repeat" + suffix)
        # The off twin has no diagnostic count output; its fixed input differs
        # only by the audit switch. The audited repeat verifies the same count.
        assert repeated["roundoff_gamma"] == reference["roundoff_gamma"]
        for record in (off, repeat):
            receipt_checks.append(
                compare_roundoff_receipts(
                    on_receipts,
                    physics_receipts(record),
                    reference["roundoff_gamma"],
                    record["label"],
                )
            )
    temporal = []
    for n in (16, 32, 64):
        rr = [get(f"full-energy-n{n}-dt{d}" + suffix) for d in range(3)]
        defects = [r["conditional_NR_total_defect"] for r in rr]
        temporal.append(
            dict(
                cells=n,
                defects=defects,
                observed_local_powers=[
                    math.log2(abs(defects[j] / defects[j + 1])) for j in range(2)
                ],
                not_whole_method_order=True,
            )
        )
    result = dict(
        inventory_checks_pass=True,
        exact_energy_conservation_claim=False,
        receipt_check=a.receipt_check,
        GPU_bitwise_claim=False,
        audit_on_off_physics_receipts_exact=on_receipts == off_receipts,
        numeric_receipt_checks=receipt_checks,
        rows=rows,
        temporal=temporal,
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "NATIVE_ENERGY_INVENTORY_PASS",
        len(rows),
        "audited runs; deterministic physical defects remain reported",
    )


if __name__ == "__main__":
    main()
