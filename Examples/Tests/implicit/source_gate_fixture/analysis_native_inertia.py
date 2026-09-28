#!/usr/bin/env python3
"""Accepted native dJe-only energy identity; no zero-total-energy assertion.

The observer uses physical electron mass and production measured Je history.
The residual/Jv, field and source algorithms are unchanged. The native energy
inventory's existing arithmetic gates are reused without relaxation.
"""

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import analysis_native_energy as energy

EPS = energy.EPS
PHYSICAL_ME = 9.1093837139e-31


def one(lines, prefix):
    found = [line for line in lines if line.startswith(prefix)]
    assert len(found) == 1, (prefix, found)
    result = {}
    for key, token in energy.ASSIGNMENT.findall(found[0]):
        if key == "measure":
            continue
        assert key not in result
        result[key] = float(token)
        assert math.isfinite(result[key])
    return result


def gamma(count):
    k = count * EPS
    assert 0 < k < 1
    return k / (1 - k)


def checked(defect, allowance, name):
    assert math.isfinite(defect) and math.isfinite(allowance) and allowance >= 0
    assert abs(defect) <= allowance, (name, defect, allowance)
    return dict(defect=defect, allowance=allowance)


def split_steps(lines):
    starts = [
        i
        for i, line in enumerate(lines)
        if line.startswith("SOURCE_GATE_INVENTORY_OLD component=0 ")
    ]
    assert starts
    return [lines[i:j] for i, j in zip(starts, starts[1:] + [len(lines)])]


def inertia_identity(lines, inventory):
    context = one(lines, "SOURCE_GATE_INERTIA_CONTEXT")
    assert context["captures"] == 1 and context["regular_intervals_active"] == 0
    assert context["measured_history"] == 1 and context["theta"] == 0.5
    assert context["dt"] > 0
    if not context["enabled"]:
        assert context["mass"] == 0
        assert not any(
            line.startswith(
                (
                    "SOURCE_GATE_INERTIA_NODAL",
                    "SOURCE_GATE_INERTIA_EDGE",
                    "SOURCE_GATE_INERTIA_CURRENT_MAP",
                )
            )
            for line in lines
        )
        return dict(
            context=context,
            storage_delta=0.0,
            inertia_work=0.0,
            conditional_total_with_storage=inventory["conditional_NR_total_defect"],
        )
    assert context["enabled"] == 1 and context["mass"] == PHYSICAL_ME
    current = one(lines, "SOURCE_GATE_INERTIA_CURRENT_MAP")
    assert current["scale"] > 0
    current_check = checked(
        current["error"], gamma(128) * current["scale"], "stage current map"
    )
    measures = {}
    for measure in ("native", "clipped"):
        node = one(lines, "SOURCE_GATE_INERTIA_NODAL measure=" + measure)
        edge = one(lines, "SOURCE_GATE_INERTIA_EDGE measure=" + measure)
        assert all(math.isfinite(v) for v in list(node.values()) + list(edge.values()))
        assert node["storage_old"] > 0 and node["storage_new"] > 0
        assert node["identity_scale"] > 0 and node["field_law_scale"] > 0
        assert node["theta_term"] == 0
        # Each point uses <512 elementary operations. Positive work/storage
        # constituents cover cancellation, then a separate gamma_N covers
        # reduction of the pointwise L1 errors and global identity terms.
        local = gamma(512)
        summation = gamma(32 * node["entries"] + 256)
        bound = (local + summation) * node["identity_scale"]
        law_bound = (local + summation) * node["field_law_scale"]
        residual = node["work"] - math.fsum(
            node[k]
            for k in ("storage_delta", "theta_term", "coefficient_term", "history_term")
        )
        checks = dict(
            local_identity=checked(node["identity_l1"], bound, "pointwise identity"),
            field_law=checked(
                node["field_law_l1"], law_bound, "captured native Ei law"
            ),
            reduced_identity=checked(residual, bound, "reduced identity"),
            stored_difference=checked(
                node["storage_delta"] - (node["storage_new"] - node["storage_old"]),
                bound,
                "stable storage difference",
            ),
            projected_difference=checked(
                edge["projected_work"] - edge["raw_work"] - edge["projection_delta"],
                (local + summation) * (edge["raw_abs"] + edge["projected_abs"]),
                "private projection",
            ),
        )
        metric = edge["raw_work"] - node["work"]
        base_actual = (
            inventory["actual_total_defect"]
            if measure == "native"
            else inventory["total"]["clipped_total_defect"]
        )
        base_conditional = base_actual - inventory["realized_minus_NR_expectation"]
        actual = base_actual + node["storage_delta"]
        conditional = base_conditional + node["storage_delta"]
        # These are reported defects, not conservation gates. In particular,
        # a homogeneous projected Ei is not a complete recovered-field audit.
        remainder = base_conditional + edge["projected_work"]
        measures[measure] = dict(
            node=node,
            edge=edge,
            metric_transfer=metric,
            identity_allowance=bound,
            checks=checks,
            actual_total_with_storage=actual,
            conditional_total_with_storage=conditional,
            conditional_plus_projected_inertia_work=remainder,
            EL_entry_jump_already_in_field_inventory=edge["EL_entry_jump"],
        )
    return dict(
        context=context,
        current_map=current,
        current_check=current_check,
        measures=measures,
    )


def analyze_run(record):
    assert record["rc"] == 0, record
    cwd = Path(record["cwd"])
    lines = (cwd / "run.log").read_text().splitlines()
    result = []
    for step, segment in enumerate(split_steps(lines), 1):
        # Preserve original inventory gates and allowances. The new multi-step
        # budget explicitly books the measured accepted thermal solver defect.
        directory = cwd / ("analysis-step-%d" % step)
        directory.mkdir(exist_ok=True)
        (directory / "run.log").write_text("\n".join(segment) + "\n")
        inventory = energy.analyze(
            dict(cwd=str(directory), label=record["label"] + f"-step{step}"),
            include_reported_solver_defect=True,
        )
        accepted = inventory["energy"]
        # This signed/L1 pair is reduced from the accepted cell residual in
        # MeasureThermalEnergyBalance before endpoint mutation. Independently
        # integrate the delivered cell-U change and compare the same account;
        # never choose this term from a full-energy observed mismatch.
        independently_integrated = inventory["new"][6]["stable_delta"] - math.fsum(
            accepted[k]
            for k in ("source", "conduction", "advection", "compression", "absorption")
        )
        signed_l1 = checked(
            accepted["defect"],
            (1 + inventory["roundoff_gamma"]) * accepted["absolute_defect"],
            "accepted signed/L1 thermal residual",
        )
        thermal_account = dict(
            signed=accepted["defect"],
            l1=accepted["absolute_defect"],
            independent_cell_inventory_residual=independently_integrated,
            signed_l1_check=signed_l1,
            independent_check=checked(
                independently_integrated - accepted["defect"],
                inventory["arithmetic_allowance"],
                "independent accepted thermal residual",
            ),
        )
        row = dict(
            step=step, inventory=inventory, thermal_solver_account=thermal_account
        )
        if record["audit"]:
            row["inertia"] = inertia_identity(segment, inventory)
            expected_enabled = record.get("enabled", record.get("inertia"))
            assert row["inertia"]["context"]["enabled"] == expected_enabled
            assert row["inertia"]["context"]["step"] == inventory["total"]["step"]
        else:
            assert not any(line.startswith("SOURCE_GATE_INERTIA_") for line in segment)
        row["segment"] = segment
        result.append(row)
    assert len(result) == 2, "The native inertia fixture requires both accepted steps"
    return result


def compare_callback(left, right, mode):
    assert len(left) == len(right)
    comparisons = []
    for left_step, right_step in zip(left, right):
        ll = [x for x in left_step["segment"] if x.startswith(energy.RECEIPT_PREFIXES)]
        rr = [x for x in right_step["segment"] if x.startswith(energy.RECEIPT_PREFIXES)]
        if mode == "exact":
            assert ll == rr, "Observer changed accepted CPU receipts"
            prefix = (
                "SOURCE_GATE_INVENTORY_",
                "SOURCE_GATE_ION_INVENTORY",
                "SOURCE_GATE_TOTAL_INVENTORY",
                "SOURCE_GATE_BOUNDARY",
            )
            assert [x for x in left_step["segment"] if x.startswith(prefix)] == [
                x for x in right_step["segment"] if x.startswith(prefix)
            ], "Observer changed native CPU inventories"
            comparisons.append(
                dict(
                    step=left_step["step"],
                    receipt_text_equal=True,
                    inventory_text_equal=True,
                )
            )
        else:
            same_count_gamma = left_step["inventory"]["roundoff_gamma"]
            assert same_count_gamma == right_step["inventory"]["roundoff_gamma"]
            receipt = energy.compare_roundoff_receipts(
                ll, rr, same_count_gamma, f"inertia-step{left_step['step']}"
            )
            checks = []
            # Independent positive inventory scales, not a fitted difference.
            norm_bound = (
                left_step["inventory"]["norm_arithmetic_allowance"]
                + right_step["inventory"]["norm_arithmetic_allowance"]
            )
            transfer_bound = (
                left_step["inventory"]["arithmetic_allowance"]
                + right_step["inventory"]["arithmetic_allowance"]
            )
            for key in ("old", "new"):
                for x, y in zip(
                    left_step["inventory"][key], right_step["inventory"][key]
                ):
                    assert x.keys() == y.keys()
                    for field in x:
                        if field in ("component", "entries"):
                            allowance = 0.0
                        elif field in ("volume", "clipped_volume"):
                            allowance = same_count_gamma * (
                                abs(x[field]) + abs(y[field])
                            )
                        else:
                            allowance = norm_bound
                        checks.append(
                            dict(
                                group=key,
                                component=x["component"],
                                field=field,
                                **checked(
                                    y[field] - x[field],
                                    allowance,
                                    "native callback inventory",
                                ),
                            )
                        )
            for key in (
                "actual_total_defect",
                "conditional_NR_total_defect",
                "realized_minus_NR_expectation",
            ):
                checks.append(
                    dict(
                        field=key,
                        **checked(
                            right_step["inventory"][key] - left_step["inventory"][key],
                            transfer_bound,
                            "callback transfer",
                        ),
                    )
                )
            comparisons.append(
                dict(step=left_step["step"], receipts=receipt, inventory_checks=checks)
            )
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, nargs="?")
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--with-mpi", action="store_true")
    parser.add_argument(
        "--receipt-check", choices=("exact", "roundoff"), default="exact"
    )
    args = parser.parse_args()
    if args.manifest:
        records = json.loads(args.manifest.read_text())
    else:
        assert args.executable and args.input
        # Fixed tiny two-step source-on physical-me gate; no temporal-order
        # claim. Zero inertia changes only that physical switch. Every case
        # uses the production measured-history default (no override).
        cases = []
        for mode in ("coupled_jfnk", "decoupled_jfnk"):
            cases += [
                (mode + "-on", mode, 1, True, 1, 16),
                (mode + "-off", mode, 1, False, 1, 16),
            ]
            if args.receipt_check == "roundoff":
                cases += [(mode + "-repeat", mode, 1, True, 1, 16)]
        cases += [
            ("zero-on", "coupled_jfnk", 0, True, 1, 16),
            ("zero-off", "coupled_jfnk", 0, False, 1, 16),
            ("box8", "coupled_jfnk", 1, True, 1, 8),
        ]
        if args.with_mpi:
            cases += [("mpi2", "coupled_jfnk", 1, True, 2, 8)]
        if args.receipt_check == "roundoff":
            cases += [("zero-repeat", "coupled_jfnk", 0, True, 1, 16)]
        records = []
        for label, mode, enabled, audit, ranks, box in cases:
            cwd = Path(tempfile.mkdtemp(prefix=label + "-", dir=Path.cwd())).resolve()
            extra = f"\nmax_step=2\nsource_gate.energy_audit=1\nsource_gate.inertia_audit={int(audit)}\n"
            extra += f"hybrid_pic_model.include_electron_inertia={enabled}\nhybrid_pic_model.electron_energy_mode={mode}\namr.max_grid_size={box}\n"
            inp = args.input.read_text() + extra
            (cwd / "inputs").write_text(inp)
            prefix = []
            if ranks > 1:
                prefix = (
                    shlex.split(os.environ.get("MPIEXEC", "mpiexec"))
                    + shlex.split(os.environ.get("MPIEXEC_PREFLAGS", ""))
                    + ["-n", str(ranks)]
                )
            command = prefix + [str(args.executable.resolve()), "inputs"]
            with (cwd / "run.log").open("w") as log:
                proc = subprocess.run(
                    command,
                    cwd=cwd,
                    env=dict(os.environ, OMP_NUM_THREADS="1"),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            records.append(
                dict(
                    label=label,
                    cwd=str(cwd),
                    command=command,
                    rc=proc.returncode,
                    mode=mode,
                    enabled=enabled,
                    audit=audit,
                    ranks=ranks,
                    box=box,
                    input_sha256=hashlib.sha256(inp.encode()).hexdigest(),
                )
            )
            Path("INERTIA_COMMANDS.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
            assert proc.returncode == 0, records[-1]
    analyzed = {r["label"]: analyze_run(r) for r in records}
    callback = {}
    for mode in ("coupled_jfnk", "decoupled_jfnk", "zero"):
        callback[mode + "-off"] = compare_callback(
            analyzed[mode + "-on"], analyzed[mode + "-off"], args.receipt_check
        )
        if args.receipt_check == "roundoff":
            callback[mode + "-repeat"] = compare_callback(
                analyzed[mode + "-on"], analyzed[mode + "-repeat"], args.receipt_check
            )
    # Compare the first-step deterministic stage before any layout-dependent
    # random OU realization can affect a subsequent step. Native nonlinear
    # tolerances remain unchanged. This is not a stochastic trajectory match.
    reference = analyzed["coupled_jfnk-on"][0]["inertia"]["measures"]["native"]
    comparisons = {}
    for label in ("decoupled_jfnk-on", "box8", "mpi2"):
        if label not in analyzed:
            continue
        current = analyzed[label][0]["inertia"]["measures"]["native"]
        checks = {}
        for group, key, scale in (
            ("node", "work", "identity_scale"),
            ("node", "storage_delta", "identity_scale"),
            ("node", "history_term", "identity_scale"),
            ("edge", "projected_work", "projected_abs"),
        ):
            allowance = 1.0e-10 * (reference[group][scale] + current[group][scale])
            checks[group + "_" + key] = checked(
                current[group][key] - reference[group][key],
                allowance,
                "native mode/layout",
            )
        comparisons[label] = checks
    # Nonzero history/coefficients/map terms are exposed and never forced to
    # zero. At least one resolved history term proves the gate did exercise
    # the production stored-versus-virtual distinction.
    assert any(
        abs(s["inertia"]["measures"]["native"]["node"]["history_term"])
        > s["inertia"]["measures"]["native"]["identity_allowance"]
        for s in analyzed["coupled_jfnk-on"]
    )
    for steps in analyzed.values():
        for row in steps:
            del row["segment"]
    result = dict(
        pass_all=True,
        scope="physical-me dJe-only accepted native identity; full energy defect remains reported",
        receipt_check=args.receipt_check,
        records=records,
        analyzed=analyzed,
        callback=callback,
        first_step_mode_layout=comparisons,
    )
    Path("INERTIA_ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "NATIVE_INERTIA_IDENTITY_PASS",
        len(records),
        "native runs; no zero-total-energy claim",
    )


if __name__ == "__main__":
    main()
