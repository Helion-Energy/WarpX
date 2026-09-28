#!/usr/bin/env python3
"""Independent represented-input native stopping inventory; never calibrates bounds."""

import argparse
import decimal
import itertools
import json
import math
from pathlib import Path

D = decimal.Decimal
decimal.getcontext().prec = 70


def number(text):
    x = float(text)
    assert math.isfinite(x), f"nonfinite input {text}"
    return D.from_float(x)


def records(run, stem):
    result = {}
    for path in sorted(run.glob(stem + ".rank*.txt")):
        for line in path.read_text().splitlines():
            a = line.split()
            key = a[0], int(a[1])
            assert key not in result, f"duplicate native particle {key}"
            result[key] = list(map(number, a[2:]))
    assert result
    return result


def fields(run, stem):
    result = {}
    for path in sorted(run.glob(stem + ".rank*.txt")):
        for line in path.read_text().splitlines():
            a = line.split()
            key = a[0], *(int(x) for x in a[1:4])
            assert key not in result, f"duplicate physical owner {key}"
            result[key] = number(a[4])
    assert result
    return result


def run_analysis(run):
    raw = json.loads((run / "LEDGER.json").read_text())
    assert raw["pass"] is True
    scalar = {k: number(v) for k, v in raw.items() if type(v) in (float, int)}
    ledger = scalar
    log = (run / "run.log").read_text()
    assert log.count("NATIVE_STOPPING_ZERO_PSI_ORACLE exact=1") == 1
    assert log.count("NATIVE_UNIFORM_STOPPING_PASS ") == 1
    for marker in [
        "purity=exact",
        "rollback=exact",
        "retry=exact",
        "once_only=1",
        "stale=reject",
        "wrong_heat=reject",
        "wrong_work=reject",
        "wrong_current=reject",
    ]:
        assert marker in log
    old, new = records(run, "before"), records(run, "after")
    assert old.keys() == new.keys()
    f0, f1 = fields(run, "fields_before"), fields(run, "fields_after")
    assert f0.keys() == f1.keys()
    n = 1 + max(key[1] for key in f0 if key[0] == "rho")
    volume = D(1) / D(n**3)
    expected = {
        (label, i, j, k)
        for label in ["rho", "U", "J0", "J1", "J2"]
        for i, j, k in itertools.product(range(n), repeat=3)
    }
    assert f0.keys() == expected, "complete unique physical native DOFs"
    for key in f0:
        if key[0] == "rho":
            assert f0[key] == f1[key]
    qe, me, c = ledger["q_e"], ledger["m_e"], ledger["c"]
    c2 = c * c

    def gamma(u):
        return (D(1) + sum(x * x for x in u) / c2).sqrt()

    def norm(u):
        return sum(x * x for x in u).sqrt()

    k0 = D(0)
    dki = D(0)
    momentum = [D(0)] * 3
    for key in old:
        a, b = old[key], new[key]
        assert a[:5] == b[:5] and a[-1] == b[-1], "fixed mass/charge/position/weight"
        m, q = a[:2]
        ua = a[5:8]
        ub = b[5:8]
        w = a[8]
        ga, gb = gamma(ua), gamma(ub)
        k0 += w * m * sum(x * x for x in ua) / (ga + 1)
        dki += w * m * sum((y - x) * (y + x) for x, y in zip(ua, ub)) / (ga + gb)
        for j in range(3):
            momentum[j] += w * m * (ub[j] - ua[j])
        assert norm(ua) <= ledger["cap"] and norm(ub) <= ledger["cap"]
    ke0 = D(0)
    dke = D(0)
    electron_p = [D(0)] * 3
    heat = sum((f1[key] - value) * volume for key, value in f0.items() if key[0] == "U")
    u0 = sum(value * volume for key, value in f0.items() if key[0] == "U")
    for comp in range(3):
        for i, j, k in itertools.product(range(n), repeat=3):
            xyz = [i, j, k]
            other = xyz.copy()
            other[comp] = (other[comp] + 1) % n
            rho = (f0[("rho", *xyz)] + f0[("rho", *other)]) / 2
            M = me / (qe * rho)
            ja, jb = f0[(f"J{comp}", *xyz)], f1[(f"J{comp}", *xyz)]
            ke0 += volume * ja * M * ja / 2
            dke += volume * (ja + jb) * M * (jb - ja) / 2
            electron_p[comp] -= volume * me / qe * (jb - ja)
    count = D(len(old))
    assert count == ledger["contribution_count"]
    operations = D(4096) * (1 + count)
    assert operations == ledger["operations"]
    eps = D(2) ** -52
    factor = 4 * operations * eps / (1 - operations * eps)
    assert ledger["uniform_bound"] >= factor * (1 - 32 * eps)
    # Independent native inputs provide the positive reference; no observed
    # signed defect or run-to-run spread enters a tolerance.
    kinetic_bound = factor * (k0 + ke0)
    energy_bound = factor * (k0 + ke0 + u0 + ledger["absolute_work"] + abs(dke))
    checks = []

    def check(label, value, bound):
        assert bound >= 0 and abs(value) <= bound, (label, str(value), str(bound))
        checks.append(
            {"name": label, "difference": float(value), "bound": float(bound)}
        )

    check("initial ion inventory", k0 - ledger["ion_initial_energy"], kinetic_bound)
    check("stable native ion deltaK", dki - ledger["ion_energy_change"], kinetic_bound)
    check(
        "native electron initial mass metric",
        ke0 - ledger["electron_initial_energy"],
        kinetic_bound,
    )
    check(
        "native electron stable deltaK",
        dke - ledger["electron_energy_change"],
        kinetic_bound,
    )
    check(
        "native U change",
        heat - ledger["heat"] - ledger["heat_transfer_error"],
        energy_bound,
    )
    actual = dki + dke + heat
    check(
        "independent actual total defect",
        actual - ledger["actual_energy_defect"],
        energy_bound,
    )
    for comp in range(3):
        check(
            f"Cartesian momentum {comp}",
            momentum[comp] + electron_p[comp],
            ledger["momentum_bound"],
        )
    terms = [
        "relativistic_defect",
        "spatial_transfer",
        "constraint_work",
        "mean_current_work",
        "transpose_work",
        "heat_transfer_error",
    ]
    check(
        "independent accounting identity",
        actual - sum(ledger[x] for x in terms),
        energy_bound,
    )
    check(
        "relative heat sign/work",
        ledger["heat"] + ledger["drag_energy_change"] - ledger["drag_bulk_work"],
        energy_bound,
    )
    check(
        "permitted convention",
        ledger["relativistic_defect"],
        ledger["relativistic_bound"] + kinetic_bound,
    )
    assert ledger["relativistic_bound"] <= ledger["relative_budget"] * (k0 + ke0) * (
        1 + 64 * eps
    )
    assert ledger["heat"] >= 0 and ledger["harmonic_residual"] <= D("1e-12")
    assert (
        ledger["grid_residual"] <= ledger["grid_bound"]
        and ledger["momentum_error"] <= ledger["momentum_bound"]
    )
    assert abs(ledger["accounting_defect"]) <= ledger["arithmetic_bound"]
    assert ledger["uniform_error"] <= ledger["uniform_bound"]
    # Recompute the speed-cap convention bound from all actual native charged
    # weights and the uniform electric impulse. Cubic partition-of-unity error
    # is covered by the fixed per-point 4096-operation envelope.
    psi = [number(x) for x in raw["electric_impulse"]]
    bound = (
        sum(a[8] * abs(a[1]) for a in old.values())
        * norm(psi)
        * ledger["cap"] ** 3
        / c2
    )
    check(
        "kinematic bound reconstructed",
        ledger["relativistic_bound"] - bound,
        factor * max(bound, D("1e-300")),
    )
    if ledger["interval"] > 0:
        assert ledger["heat"] > 0 and norm(psi) > 0
        assert norm([number(x) for x in raw["reaction_momentum"]]) > 0
    return {
        "pass": True,
        "scope": "periodic Cartesian3D uniform fixed-density native event only",
        "native_particles": int(count),
        "native_dofs": len(f0),
        "checks": checks,
        "actual_total_defect": float(actual),
        "positive_kinetic_scale": float(k0 + ke0),
        "defect_over_positive_kinetic": float(actual / (k0 + ke0)),
        "relative_heat": float(heat),
        "ion_change": float(dki),
        "electron_change": float(dke),
        "finite_terms": {k: float(ledger[k]) for k in terms},
        "no_total_conservation_claim": True,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    out = {
        "pass": True,
        "cases": [{"run": str(x), "analysis": run_analysis(x)} for x in args.runs],
    }
    text = json.dumps(out, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
