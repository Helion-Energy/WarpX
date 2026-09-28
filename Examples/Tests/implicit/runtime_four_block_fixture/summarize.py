"""Recompute gate summaries, timestep orders and plots from immutable probe logs."""

import argparse
import json
import math
from pathlib import Path

from analyze_case import validate


def read(case):
    return [json.loads(x) for x in (case / "PROBES.jsonl").read_text().splitlines()]


def direction(rows, name, epsilon):
    return next(
        r for r in rows if r.get("direction") == name and r["epsilon"] == epsilon
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cases = {
        d.name: validate(d)
        for d in sorted(a.runs.iterdir())
        if (d / "RUN_RESULT.json").exists()
    }
    names = ["eta-temperature", "eta-temperature-halfdt", "eta-temperature-quarterdt"]
    steps = []
    for name, dt in zip(names, [1.0e-8, 5.0e-9, 2.5e-9]):
        rows = read(a.runs / name)
        blocks = {}
        for d in ["E", "U", "mixed"]:
            r = direction(rows, d, 0.05)
            blocks[d] = {
                "full_norms": r["full_derivative"],
                "absolute_mm_gap": r["derivative_difference"],
                "relative_mm_gap": [
                    x / y
                    for x, y in zip(r["derivative_difference"], r["full_derivative"])
                ],
            }
        steps.append({"dt": dt, "directions": blocks})
    orders = {
        d: [
            [
                math.log(
                    steps[i]["directions"][d]["absolute_mm_gap"][b]
                    / steps[i + 1]["directions"][d]["absolute_mm_gap"][b],
                    2,
                )
                for b in range(2)
            ]
            for i in range(2)
        ]
        for d in ["E", "U", "mixed"]
    }
    summary = {
        "cases": cases,
        "timestep_mm_errors_at_epsilon_0_05": steps,
        "absolute_mm_gap_halving_orders": orders,
        "Q6_accuracy_definition": "Triangle bound on measured D(.00625)-D(.0001), per-block normalized; not an exact analytic Jacobian error",
        "full_particle_Q6_maximum": max(
            x
            for case in cases.values()
            for d in case["full_particle_Q6"]["directions"].values()
            for x in d["relative_difference_bound_to_reference"]
        ),
    }
    (a.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read(a.runs / "eta-temperature")
    fig, ax = plt.subplots(1, 4, figsize=(15.4, 3.6), constrained_layout=True)
    for d, b, label in [("E", 0, "EE"), ("E", 1, "UE"), ("U", 0, "EU"), ("U", 1, "UU")]:
        rr = [r for r in rows if r.get("direction") == d]
        ax[0].loglog(
            [r["epsilon"] for r in rr],
            [r["derivative_difference"][b] / r["full_derivative"][b] for r in rr],
            "o-",
            label=label,
        )
        rr = rr[1:]
        ax[1].loglog(
            [r["epsilon"] for r in rr],
            [r["full_refinement_change"][b] / r["full_derivative"][b] for r in rr],
            "o-",
            label=label,
        )
    for p in ax[:2]:
        p.axhline(1.0e-5, color="gray", ls="--", label="Q6 threshold")
        p.set_xlabel("dimensionless epsilon")
        p.legend(fontsize=7)
    ax[0].set_title("MM/full gap, dt = 10 ns")
    ax[0].set_ylabel("relative block difference")
    ax[1].set_title("Full-particle adjacent FD difference")
    u = [r for r in rows if r.get("direction") == "U"][:6]
    ax[2].loglog(
        [r["epsilon"] for r in u],
        [r["full_even_remainder"][1] for r in u],
        "o-",
        label="thermal remainder",
    )
    ax[2].loglog(
        [r["epsilon"] for r in u],
        [
            u[0]["full_even_remainder"][1] * (r["epsilon"] / u[0]["epsilon"]) ** 2
            for r in u
        ],
        "--",
        label="epsilon squared",
    )
    ax[2].set_title("Smooth nonlinear thermal response")
    ax[2].set_xlabel("dimensionless epsilon")
    ax[2].set_ylabel("J/m³")
    ax[2].legend(fontsize=7)
    for d, label in [("E", "EE"), ("U", "EU")]:
        ax[3].loglog(
            [r["dt"] for r in steps],
            [r["directions"][d]["absolute_mm_gap"][0] for r in steps],
            "o-",
            label=label,
        )
    ax[3].loglog(
        [r["dt"] for r in steps],
        [
            steps[0]["directions"]["E"]["absolute_mm_gap"][0]
            * (r["dt"] / steps[0]["dt"]) ** 3
            for r in steps
        ],
        "--",
        label="dt cubed",
    )
    ax[3].set_title("MM field error under timestep reduction")
    ax[3].set_xlabel("dt [s]")
    ax[3].set_ylabel("V/m per direction unit")
    ax[3].legend(fontsize=7)
    for p in ax:
        p.grid(True, which="both", alpha=0.2)
    fig.savefig(a.output / "directional-convergence.png", dpi=170)
    fig.savefig(a.output / "directional-convergence.pdf")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "full_Q6_bound_max": summary["full_particle_Q6_maximum"],
                "orders": orders,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
