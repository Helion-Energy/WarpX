#!/usr/bin/env python3
"""Validate schema-v1 D1 common-direction operator-partition evidence."""

import argparse
import math
from pathlib import Path

SCHEMAS = {
    "D1_BEGIN": {
        "schema",
        "step",
        "iteration",
        "direction",
        "direction_norm",
        "epsilon_relative_top",
        "rungs",
        "destructive",
        "timing_valid",
    },
    "D1_DIRECTION": {
        "schema",
        "step",
        "iteration",
        "component",
        "norm",
        "max_abs",
    },
    "D1_METRIC": {
        "schema",
        "step",
        "iteration",
        "rung",
        "eps_relative",
        "eps_absolute",
        "quantity",
        "norm",
        "reference_norm",
        "relative",
    },
    "D1_COMPONENT": {
        "schema",
        "step",
        "iteration",
        "rung",
        "eps_relative",
        "eps_absolute",
        "quantity",
        "component",
        "norm",
        "reference_norm",
        "relative",
        "max_abs",
        "max_r",
        "max_z",
    },
    "D1_END": {"schema", "step", "iteration", "status", "rungs"},
}

RUNG_QUANTITIES = {
    "A",
    "B_g",
    "B_s",
    "C",
    "A_minus_B_s",
    "B_s_minus_C",
    "B_g_minus_B_s",
    "A_minus_C",
    "full_saved_drift_over_eps",
    "closure",
}
NOISE_QUANTITIES = {"stage_repeat", "full_repeat", "full_saved_repeat"}
FLOAT_FIELDS = {
    "direction_norm",
    "epsilon_relative_top",
    "norm",
    "max_abs",
    "eps_relative",
    "eps_absolute",
    "reference_norm",
    "relative",
}
INT_FIELDS = {
    "schema",
    "step",
    "iteration",
    "rungs",
    "destructive",
    "timing_valid",
    "component",
    "rung",
    "max_r",
    "max_z",
}


def parse(path):
    records = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.startswith("D1_"):
            continue
        words = raw.split()
        kind = words[0]
        assert kind in SCHEMAS, f"{path}:{lineno}: unknown record {kind}"
        fields = {}
        for word in words[1:]:
            assert "=" in word, f"{path}:{lineno}: non-key/value token {word!r}"
            key, value = word.split("=", 1)
            assert key not in fields, f"{path}:{lineno}: duplicate key {key}"
            fields[key] = value
        assert set(fields) == SCHEMAS[kind], (
            f"{path}:{lineno}: {kind} schema keys {set(fields)} != {SCHEMAS[kind]}"
        )
        for key in INT_FIELDS & fields.keys():
            fields[key] = int(fields[key])
        for key in FLOAT_FIELDS & fields.keys():
            fields[key] = float(fields[key])
            assert math.isfinite(fields[key]), f"{path}:{lineno}: non-finite {key}"
        assert fields["schema"] == 1, f"{path}:{lineno}: unsupported schema"
        records.append((kind, fields))
    assert records, f"{path}: no D1 records"
    return records


def validate(path):
    records = parse(path)
    assert records[0][0] == "D1_BEGIN", "D1_BEGIN must be the first D1 record"
    assert records[-1][0] == "D1_END", "D1_END must be the last D1 record"
    by_kind = {kind: [f for k, f in records if k == kind] for kind in SCHEMAS}
    assert len(by_kind["D1_BEGIN"]) == 1, "expected exactly one D1_BEGIN"
    assert len(by_kind["D1_END"]) == 1, "expected exactly one D1_END"
    begin = by_kind["D1_BEGIN"][0]
    end = by_kind["D1_END"][0]
    assert begin["direction"] == "rz_periodic_hash_v1"
    assert math.isclose(begin["direction_norm"], 1.0, rel_tol=5e-13, abs_tol=5e-15)
    assert begin["rungs"] == end["rungs"] == 3
    assert begin["destructive"] == 1 and begin["timing_valid"] == 0
    assert end["status"] == "complete"
    assert (begin["step"], begin["iteration"]) == (end["step"], end["iteration"])
    identity = (begin["step"], begin["iteration"])
    assert all(
        (fields["step"], fields["iteration"]) == identity for _, fields in records
    ), "all D1 records must share the D1_BEGIN step/iteration identity"

    directions = by_kind["D1_DIRECTION"]
    assert len(directions) == 3
    assert {r["component"] for r in directions} == {0, 1, 2}
    assert all(r["norm"] > 0.0 and r["max_abs"] > 0.0 for r in directions)

    metrics = by_kind["D1_METRIC"]
    components = by_kind["D1_COMPONENT"]
    metric_key = {(r["rung"], r["quantity"]): r for r in metrics}
    assert len(metric_key) == len(metrics), "duplicate D1_METRIC key"
    expected = {(-1, q) for q in NOISE_QUANTITIES}
    expected |= {(rung, q) for rung in range(3) for q in RUNG_QUANTITIES}
    assert set(metric_key) == expected, (
        f"metric inventory mismatch: missing={expected - set(metric_key)}, "
        f"extra={set(metric_key) - expected}"
    )

    component_key = {(r["rung"], r["quantity"], r["component"]): r for r in components}
    assert len(component_key) == len(components), "duplicate D1_COMPONENT key"
    expected_components = {(r, q, c) for r, q in expected for c in range(3)}
    assert set(component_key) == expected_components, "component inventory mismatch"

    eps_rel = [metric_key[(r, "A")]["eps_relative"] for r in range(3)]
    eps_abs = [metric_key[(r, "A")]["eps_absolute"] for r in range(3)]
    assert eps_rel[0] < eps_rel[1] < eps_rel[2]
    assert eps_abs[0] < eps_abs[1] < eps_abs[2]
    assert math.isclose(eps_rel[-1], begin["epsilon_relative_top"], rel_tol=1e-14)
    for rung in range(3):
        for quantity in RUNG_QUANTITIES:
            rec = metric_key[(rung, quantity)]
            assert rec["eps_relative"] == eps_rel[rung]
            assert rec["eps_absolute"] == eps_abs[rung]
        # C is one frozen action reused across the ladder.
        if rung:
            assert math.isclose(
                metric_key[(rung, "C")]["norm"],
                metric_key[(0, "C")]["norm"],
                rel_tol=2e-14,
                abs_tol=0.0,
            )
        closure = metric_key[(rung, "closure")]
        assert closure["relative"] <= 1e-10, (
            f"rung {rung}: four-operator closure {closure['relative']:.3e}"
        )
    for quantity in NOISE_QUANTITIES:
        rec = metric_key[(-1, quantity)]
        assert rec["eps_relative"] == rec["eps_absolute"] == 0.0

    print(
        f"D1 schema-v1 gate passed: step {begin['step']} iteration "
        f"{begin['iteration']}, 3 rungs, A/B_g/B_s/C + noise/localization complete"
    )
    return records


def indexed(records):
    result = {}
    for kind, fields in records:
        if kind == "D1_DIRECTION":
            key = (kind, fields["component"])
        elif kind == "D1_METRIC":
            key = (kind, fields["rung"], fields["quantity"])
        elif kind == "D1_COMPONENT":
            key = (kind, fields["rung"], fields["quantity"], fields["component"])
        else:
            continue
        result[key] = fields
    return result


def compare(left, right, rtol):
    left_begin = next(fields for kind, fields in left if kind == "D1_BEGIN")
    right_begin = next(fields for kind, fields in right if kind == "D1_BEGIN")
    for name in {"step", "iteration", "direction", "rungs"}:
        assert left_begin[name] == right_begin[name], (
            f"one/two-rank D1_BEGIN mismatch for {name}"
        )
    assert math.isclose(
        left_begin["epsilon_relative_top"],
        right_begin["epsilon_relative_top"],
        rel_tol=rtol,
        abs_tol=0.0,
    ), "one/two-rank epsilon_relative_top mismatch"

    a = indexed(left)
    b = indexed(right)
    assert set(a) == set(b), "one/two-rank record inventory differs"
    numeric = {
        "norm",
        "reference_norm",
        "relative",
        "max_abs",
        "eps_relative",
        "eps_absolute",
    }
    for key in sorted(a, key=str):
        for name in numeric & a[key].keys():
            av, bv = a[key][name], b[key][name]
            scale = max(abs(av), abs(bv), 1e-300)
            assert abs(av - bv) <= rtol * scale, (
                f"one/two-rank mismatch {key} {name}: {av:.17e} vs {bv:.17e}"
            )
        for name in {"max_r", "max_z"} & a[key].keys():
            assert a[key][name] == b[key][name], (
                f"one/two-rank location mismatch {key} {name}"
            )
    print(f"D1 one/two-rank agreement passed at relative tolerance {rtol:.1e}")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--file", type=Path, default=Path("diags/d1_operator_partition.txt")
)
parser.add_argument("--compare", type=Path)
parser.add_argument("--compare-rtol", type=float, default=1e-9)
parser.add_argument("--assert-default-off", action="store_true")
args = parser.parse_args()

if args.assert_default_off:
    assert not args.file.exists(), "default-off run created a D1 diagnostic file"
    used_inputs = Path("warpx_used_inputs").read_text(encoding="utf-8")
    assert "newton.d1_" not in used_inputs, (
        "default-off run unexpectedly configured D1 parameters"
    )
    print("D1 default-off/no-output gate passed")
    raise SystemExit(0)

primary = validate(args.file)
if args.compare is not None:
    compare(primary, validate(args.compare), args.compare_rtol)
