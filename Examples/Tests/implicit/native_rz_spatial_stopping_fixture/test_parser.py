"""Deliberate independent-inventory corruptions, no simulation or source changes."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from analyze import analyze

p = argparse.ArgumentParser()
p.add_argument("run", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
out = []


def ledger(key, value):
    def change(r):
        p = r / "LEDGER.json"
        x = json.loads(p.read_text())
        x[key] = value
        p.write_text(json.dumps(x))

    return change


def replace_record(stem, condition, change):
    def mutate(r):
        for p in sorted(r.glob(stem + ".rank*.txt")):
            rows = p.read_text().splitlines()
            for i, line in enumerate(rows):
                words = line.split()
                if condition(words):
                    rows[i] = change(words)
                    p.write_text("\n".join(rows) + "\n")
                    return
        raise AssertionError("mutation site missing")

    return mutate


cases = {
    "negative_heat": ledger("heat", -1.0),
    "forged_actual_total": ledger("actual_energy_defect", 1.0),
    "nonzero_axis_work": ledger("axis_reaction_work", 1.0e-100),
    "nonzero_conductor_work": ledger("conductor_work", 1.0e-100),
    "too_strict_relativistic_budget": ledger("relative_budget", 1.0e-20),
    "wrong_axial_transfer": ledger("axial_electric_transfer", 1.0),
    "wrong_mass_energy": ledger("electron_energy_change", 1.0),
    "rho_mutation": replace_record(
        "fields_after",
        lambda a: a[0] == "rho",
        lambda a: " ".join(a[:4] + [str(float(a[4]) * 1.1)] + a[5:]),
    ),
    "wrong_dual_volume": replace_record(
        "fields_before",
        lambda a: a[0] == "B0",
        lambda a: " ".join(a[:7] + [str(float(a[7]) * 1.01)]),
    ),
    "forged_particle_kick": replace_record(
        "after",
        lambda a: a[0] == "fast",
        lambda a: " ".join(a[:7] + [str(float(a[7]) + 10000)] + a[8:]),
    ),
    "wall_outside_support": replace_record(
        "Jwall", lambda a: a[1] == "1", lambda a: " ".join(a[:3] + ["1"])
    ),
    "electric_aux_stale": replace_record(
        "ElectricAux",
        lambda a: a[0] == "1" and a[1] == "1",
        lambda a: " ".join(a[:3] + [str(float(a[3]) + 1)]),
    ),
    "stopping_velocity_stale": replace_record(
        "StopVe",
        lambda a: a[0] == "2" and a[1] == "1",
        lambda a: " ".join(a[:3] + [str(float(a[3]) + 100000)]),
    ),
}
for name, mutation in cases.items():
    with tempfile.TemporaryDirectory(
        prefix="rz-parser-" + name + "-", dir=a.output.parent
    ) as d:
        r = Path(d)
        for p in a.run.iterdir():
            if p.is_file() and (
                p.name.endswith(".txt")
                or p.name in ["LEDGER.json", "GEOMETRY.json", "run.log"]
            ):
                shutil.copyfile(p, r / p.name)
        mutation(r)
        try:
            analyze(r)
        except (AssertionError, ValueError, KeyError, ArithmeticError) as e:
            out.append({"case": name, "rejected": True, "reason": str(e)[:500]})
        else:
            raise AssertionError("accepted corrupted receipt " + name)
a.output.write_text(
    json.dumps({"pass": True, "negative_count": len(out), "cases": out}, indent=2)
    + "\n"
)
print("parser negatives", len(out), "PASS")
