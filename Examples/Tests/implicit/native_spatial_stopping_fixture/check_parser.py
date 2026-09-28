#!/usr/bin/env python3
"""Deliberate malformed physical receipts, with no simulation reruns."""

import argparse
import json
import shutil
from pathlib import Path

from analyze import run_analysis


def alter_field(run, label, amount=None, sign=False):
    p = next(run.glob("fields_after.rank*.txt"))
    lines = p.read_text().splitlines()
    for i, line in enumerate(lines):
        values = line.split()
        if values[0] == label and (not sign or float(values[-1]) != 0):
            values[-1] = str(-float(values[-1]) if sign else float(values[-1]) + amount)
            lines[i] = " ".join(values)
            break
    else:
        raise AssertionError(label)
    p.write_text("\n".join(lines) + "\n")


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    a = cli.parse_args()
    run_analysis(a.reference)
    a.output.mkdir(exist_ok=False)
    names = [
        "heat",
        "quality",
        "budget",
        "convention",
        "particle",
        "thermal",
        "electron",
        "magnetic",
        "potential",
        "displacement",
        "duplicate_owner",
        "duplicate_particle",
        "range_mean",
        "range_bound",
        "accounting",
        "nonfinite",
    ]
    rows = []
    for name in names:
        d = a.output / name
        d.mkdir()
        for p in a.reference.iterdir():
            if p.name in ["LEDGER.json", "run.log"] or p.name.startswith(
                ("before.rank", "after.rank", "fields_before.rank", "fields_after.rank")
            ):
                shutil.copy2(p, d / p.name)
        p = d / "LEDGER.json"
        ledger = json.loads(p.read_text())
        if name == "heat":
            ledger["heat"] = -1
        elif name == "quality":
            ledger["spatial_residual"] = 1e-4
        elif name == "budget":
            ledger["relative_budget"] = 1e-30
        elif name == "convention":
            ledger["relativistic_bound"] = 0
        elif name == "range_mean":
            ledger["current_range_mean"] = 2 * ledger["current_range_bound"] + 1
        elif name == "range_bound":
            ledger["current_range_bound"] *= 2
        elif name == "accounting":
            ledger["accounting_defect"] = 1
        elif name == "nonfinite":
            ledger["heat"] = float("nan")
        elif name == "particle":
            f = next(d.glob("after.rank*.txt"))
            lines = f.read_text().splitlines()
            parts = lines[0].split()
            parts[7] = str(float(parts[7]) + 1000)
            lines[0] = " ".join(parts)
            f.write_text("\n".join(lines) + "\n")
        elif name == "thermal":
            alter_field(d, "U", 1)
        elif name == "electron":
            alter_field(d, "J0", 1e5)
        elif name == "magnetic":
            alter_field(d, "B2", sign=True)
        elif name == "potential":
            alter_field(d, "A0", 1e-4)
        elif name == "displacement":
            alter_field(d, "D2", 1e5)
        elif name.startswith("duplicate"):
            f = next(
                d.glob(
                    "fields_after.rank*.txt"
                    if name == "duplicate_owner"
                    else "after.rank*.txt"
                )
            )
            text = f.read_text()
            f.write_text(text + text.splitlines()[0] + "\n")
        p.write_text(json.dumps(ledger, indent=2) + "\n")
        try:
            run_analysis(d)
        except (AssertionError, ValueError) as error:
            rows.append({"case": name, "rejected": True, "reason": str(error)})
        else:
            raise AssertionError("accepted malformed " + name)
    out = {"pass": True, "cases": rows}
    (a.output / "RESULTS.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
