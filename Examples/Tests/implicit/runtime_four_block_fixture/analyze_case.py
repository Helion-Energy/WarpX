"""Check native block-probe invariants and smooth directional convergence."""

import argparse
import json
import sys
from pathlib import Path


def validate(directory):
    directory = Path(directory)
    run = json.loads((directory / "RUN_RESULT.json").read_text())
    rows = [
        json.loads(x) for x in (directory / "PROBES.jsonl").read_text().splitlines()
    ]
    assert run["complete"] and len(rows) == 29
    anchor = rows[0]
    probes = [r for r in rows if r["kind"] == "direction"]
    purity = max(v for r in probes for key in ["full_ABA", "mm_ABA"] for v in r[key])
    purity_relative = max(
        r[key][b] / max(anchor["residual_norms"][b], 1.0)
        for r in probes
        for key in ["full_ABA", "mm_ABA"]
        for b in range(2)
    )
    assert purity_relative < 128 * sys.float_info.epsilon, (
        "residual depends on previous probes",
        purity_relative,
    )
    assert (
        max(
            anchor["mm_zero_difference"][b] / max(anchor["residual_norms"][b], 1.0)
            for b in range(2)
        )
        < 128 * sys.float_info.epsilon
    ), "MM base differs from physical base"
    assert max(anchor["moment_mm_zero_difference"]) < 1.0e-9
    first = {
        d: next(r for r in probes if r["direction"] == d) for d in ["E", "U", "mixed"]
    }
    assert all(x > 0 for r in first.values() for x in r["full_derivative"])
    u = [r for r in probes if r["direction"] == "U"]
    ratios = [
        u[i]["full_even_remainder"][1] / u[i + 1]["full_even_remainder"][1]
        for i in range(3)
    ]
    assert all(3.95 < r < 4.05 for r in ratios), ("thermal Taylor convergence", ratios)
    current = max(first["U"]["full_moments"][:3])
    if run["eta"] == "temperature":
        assert current > 1.0e-3, "live eta(T) failed to change particle current"
        assert max(first["U"]["moment_derivative_difference"][:3]) < 0.01 * current
    else:
        assert current < 1.0e-8, (
            "constant eta produced a spurious T-only particle response"
        )
    linearity = rows[-1]
    relative_linearity = {
        kind: [
            linearity[kind][b] / max(first[d]["full_derivative"][b] for d in first)
            for b in range(2)
        ]
        for kind in ["full", "mm"]
    }
    assert max(x for v in relative_linearity.values() for x in v) < 1.0e-6
    # Common resolved window, before small cross derivatives become dominated
    # by subtraction noise. This triangle sum bounds the difference between
    # two measured finite differences, not the unknown analytic truncation error.
    full_accuracy = {}
    for direction in ["E", "U", "mixed"]:
        ladder = [r for r in probes if r["direction"] == direction]
        trial = next(i for i, r in enumerate(ladder) if r["epsilon"] == 0.00625)
        reference = next(i for i, r in enumerate(ladder) if r["epsilon"] == 0.0001)
        norm = ladder[trial]["full_derivative"]
        bound = [
            sum(
                r["full_refinement_change"][b]
                for r in ladder[trial + 1 : reference + 1]
            )
            / norm[b]
            for b in range(2)
        ]
        adjacent = [
            ladder[trial]["full_refinement_change"][b] / norm[b] for b in range(2)
        ]
        fine_noise = [
            ladder[-1]["full_refinement_change"][b] / norm[b] for b in range(2)
        ]
        assert max(bound + adjacent) < 1.0e-5, (
            "full-particle Q6 window",
            direction,
            bound,
            adjacent,
        )
        full_accuracy[direction] = {
            "relative_difference_bound_to_reference": bound,
            "relative_adjacent_difference": adjacent,
            "relative_last_refinement_difference": fine_noise,
        }
    particle_control = anchor["probe_pushes"] > 0
    if particle_control:
        assert all(max(r["derivative_difference"]) == 0 for r in probes)
    else:
        assert all(r["probe_pushes"] == 0 for r in probes)
    report = {
        "passed": True,
        "residual_purity_error": purity,
        "relative_residual_purity_error": purity_relative,
        "anchor_error": anchor["mm_zero_difference"],
        "temperature_column_current_response_A_per_m2": current,
        "thermal_even_remainder_halving_ratios": ratios,
        "relative_mixed_linearity": relative_linearity,
        "full_particle_probe_control": particle_control,
        "anchor_particle_calls": anchor["full_pushes"],
        "probe_particle_calls": anchor["probe_pushes"],
        "probe_stage_calls": anchor["probe_stage_calls"],
        "full_particle_Q6": {
            "scope": "smooth deterministic epsilon-window consistency, not an analytic Jacobian oracle",
            "trial_epsilon": 0.00625,
            "reference_epsilon": 0.0001,
            "relative_tolerance": 1.0e-5,
            "directions": full_accuracy,
        },
        "relative_block_derivative_errors_at_epsilon_0_1": {
            d: [
                first[d]["derivative_difference"][b] / first[d]["full_derivative"][b]
                for b in range(2)
            ]
            for d in first
        },
    }
    (directory / "GATES.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    print(json.dumps(validate(a.directory), indent=2))
