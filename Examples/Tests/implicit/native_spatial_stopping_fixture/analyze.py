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
    assert log.count("NATIVE_SPATIAL_STOPPING_PASS ") == 1
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
        for label in [
            "rho",
            "U",
            *[f"{prefix}{c}" for prefix in ["J", "A", "B", "D", "I"] for c in range(3)],
        ]
        for i, j, k in itertools.product(range(n), repeat=3)
    }
    assert f0.keys() == expected, "complete unique physical native DOFs"
    for key in f0:
        if key[0] == "rho":
            assert f0[key] == f1[key]
    qe, me, c = ledger["q_e"], ledger["m_e"], ledger["c"]
    mu0 = ledger["mu0"]
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
    _wb0 = sum(
        value * value * volume / (2 * mu0)
        for key, value in f0.items()
        if key[0].startswith("B")
    )
    dwb = sum(
        (f1[key] - value) * (f1[key] + value) * volume / (2 * mu0)
        for key, value in f0.items()
        if key[0].startswith("B")
    )
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
    energy_bound = factor * (
        k0 + ke0 + u0 + ledger["absolute_work"] + abs(dke) + abs(dwb)
    )
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
    check(
        "native magnetic stable storage",
        dwb - ledger["magnetic_energy_change"],
        energy_bound,
    )
    actual = dki + dke + heat + dwb
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
        "curl_work",
        "displacement_work",
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
    assert ledger["heat"] >= 0 and ledger["spatial_residual"] <= D("1e-12")
    assert (
        ledger["grid_residual"] <= ledger["grid_bound"]
        and ledger["momentum_error"] <= ledger["momentum_bound"]
    )
    assert abs(ledger["accounting_defect"]) <= ledger["arithmetic_bound"]
    assert ledger["uniform_error"] <= ledger["uniform_bound"]
    assert D(0) <= ledger["range_ratio"] <= D(1) and ledger["range_calls"] > 0
    depth = D(n**3 + 32)
    g = depth * eps / (1 - depth * eps)
    for channel in ["current", "impulse"]:
        scale = ledger[channel + "_range_scale"]
        mean = ledger[channel + "_range_mean"]
        stated = ledger[channel + "_range_bound"]
        expected_range = 16 * g * scale
        check(
            channel + " periodic range bound",
            stated - expected_range,
            64 * eps * expected_range,
        )
        check(channel + " removed divergence mean", mean, stated)
    assert (
        ledger["faraday_error"] <= ledger["faraday_bound"]
        and ledger["displacement_curl"] <= ledger["displacement_curl_bound"]
    )
    # Independently reconstruct the actual native MC electric gather: native
    # Yee-to-node averaging then the cardinal cubic basis, using represented A.
    psi_grid = {
        (comp, i, j, k): -(f1[(f"A{comp}", i, j, k)] - f0[(f"A{comp}", i, j, k)])
        for comp in range(3)
        for i, j, k in itertools.product(range(n), repeat=3)
    }

    def B3(x):
        x = abs(x)
        if x < 1:
            return D(2) / 3 - x * x + x * x * x / 2
        if x < 2:
            return (2 - x) ** 3 / 6
        return D(0)

    def gather(a):
        coord = [x * n for x in a[2:5]]
        result = [D(0)] * 3
        indices = [
            list(
                range(
                    int(x.to_integral_value(rounding=decimal.ROUND_FLOOR)) - 1,
                    int(x.to_integral_value(rounding=decimal.ROUND_FLOOR)) + 3,
                )
            )
            for x in coord
        ]
        for xyz in itertools.product(*indices):
            weight = math.prod(B3(coord[d] - xyz[d]) for d in range(3))
            key = [v % n for v in xyz]
            for comp in range(3):
                neighbor = key.copy()
                neighbor[comp] = (neighbor[comp] - 1) % n
                aux = (psi_grid[(comp, *key)] + psi_grid[(comp, *neighbor)]) / 2
                result[comp] += weight * aux
        return result

    endpoint_work = D(0)
    bound = D(0)
    electric_p = [D(0)] * 3
    for key, a in old.items():
        b = new[key]
        gathered = gather(a)
        ua = a[5:8]
        ub = b[5:8]
        ga = gamma(ua)
        gb = gamma(ub)
        endpoint_work += (
            a[8]
            * a[1]
            * sum(gathered[c] * (ua[c] / ga + ub[c] / gb) / 2 for c in range(3))
        )
        bound += a[8] * abs(a[1]) * norm(gathered) * ledger["cap"] ** 3 / c2
        for c in range(3):
            electric_p[c] += a[8] * a[1] * gathered[c]
    check(
        "native MC endpoint work reconstructed",
        endpoint_work - ledger["endpoint_particle_work"],
        energy_bound,
    )
    check(
        "kinematic bound reconstructed",
        ledger["relativistic_bound"] - bound,
        factor * max(bound, D("1e-300")),
    )

    def curl_b(f, c, xyz):
        # Native Yee downward Ampere differences, periodic physical DOFs.
        a = (c + 1) % 3
        b = (c + 2) % 3
        q = list(xyz)
        qa = q.copy()
        qb = q.copy()
        qa[a] = (qa[a] - 1) % n
        qb[b] = (qb[b] - 1) % n
        return (
            n
            * (
                (f[(f"B{b}", *q)] - f[(f"B{b}", *qa)])
                - (f[(f"B{a}", *q)] - f[(f"B{a}", *qb)])
            )
            / mu0
        )

    gridwork = D(0)
    curlwork = dwb
    constraint = D(0)
    dwork = D(0)
    maxrow = D(0)
    faraday = D(0)
    gauge = D(0)
    dcurl = D(0)
    electric_grid = [D(0)] * 3
    for xyz in itertools.product(range(n), repeat=3):
        div = D(0)
        for comp in range(3):
            p = psi_grid[(comp, *xyz)]
            oldc = curl_b(f0, comp, xyz)
            newc = curl_b(f1, comp, xyz)
            ia = (f0[(f"I{comp}", *xyz)] + f1[(f"I{comp}", *xyz)]) / 2
            ja = (f0[(f"J{comp}", *xyz)] + f1[(f"J{comp}", *xyz)]) / 2
            da = (f0[(f"D{comp}", *xyz)] + f1[(f"D{comp}", *xyz)]) / 2
            neighbor = list(xyz)
            neighbor[comp] = (neighbor[comp] + 1) % n
            rho_edge = (f0[("rho", *xyz)] + f0[("rho", *neighbor)]) / 2
            electric_grid[comp] += volume * rho_edge * p
            gridwork += volume * ia * p
            curlwork += volume * (oldc + newc) * p / 2
            constraint += volume * (ia + ja + da - (oldc + newc) / 2) * p
            dwork -= volume * da * p
            row = (
                newc
                - f1[(f"I{comp}", *xyz)]
                - f1[(f"J{comp}", *xyz)]
                - f1[(f"D{comp}", *xyz)]
            )
            maxrow = max(maxrow, abs(row))
            prev = list(xyz)
            prev[comp] = (prev[comp] - 1) % n
            div += n * (p - psi_grid[(comp, *prev)])
            a = (comp + 1) % 3
            b = (comp + 2) % 3
            qa = list(xyz)
            qb = list(xyz)
            qa[a] = (qa[a] + 1) % n
            qb[b] = (qb[b] + 1) % n
            curlpsi = n * (
                (psi_grid[(b, *qa)] - psi_grid[(b, *xyz)])
                - (psi_grid[(a, *qb)] - psi_grid[(a, *xyz)])
            )
            faraday = max(
                faraday, abs(f1[(f"B{comp}", *xyz)] - f0[(f"B{comp}", *xyz)] + curlpsi)
            )
            curlD = n * (
                (f1[(f"D{b}", *qa)] - f1[(f"D{b}", *xyz)])
                - (f1[(f"D{a}", *qb)] - f1[(f"D{a}", *xyz)])
            )
            dcurl = max(dcurl, abs(curlD))
        gauge = max(gauge, abs(div))
    check(
        "MC/native current transfer reconstructed",
        endpoint_work - gridwork - ledger["spatial_transfer"],
        energy_bound,
    )
    check("curl work reconstructed", curlwork - ledger["curl_work"], energy_bound)
    check(
        "displacement work reconstructed",
        dwork - ledger["displacement_work"],
        energy_bound,
    )
    check(
        "constraint work reconstructed",
        constraint - ledger["constraint_work"],
        energy_bound,
    )
    check("actual native Ampere rows", maxrow, ledger["grid_bound"])
    dscale = sum(
        2 * n * max(abs(value) for key, value in f1.items() if key[0] == f"D{c}")
        for c in range(3)
    )
    check("native displacement longitudinal space", dcurl, 4096 * eps * dscale)
    electric_transfer = max(abs(electric_p[c] - electric_grid[c]) for c in range(3))
    check(
        "electric gather charge momentum transfer",
        electric_transfer - ledger["electric_momentum_transfer"],
        ledger["momentum_bound"],
    )
    psi_scale = max(abs(v) for v in psi_grid.values())
    check(
        "native Faraday stencil",
        faraday,
        D(4096) * eps * n * max(psi_scale, D("1e-300")),
    )
    check(
        "native transverse impulse",
        gauge,
        (D("1e-12") + D(4096) * eps)
        * n
        * max(ledger["m_e"] / qe * ledger["cap"], psi_scale),
    )
    psi = [number(x) for x in raw["electric_impulse"]]
    if ledger["interval"] > 0:
        assert ledger["heat"] > 0 and norm(psi) > 0
        assert norm([number(x) for x in raw["reaction_momentum"]]) > 0
    return {
        "pass": True,
        "scope": "periodic Cartesian3D spatial fixed-density native event only",
        "native_particles": int(count),
        "native_dofs": len(f0),
        "checks": checks,
        "actual_total_defect": float(actual),
        "positive_kinetic_scale": float(k0 + ke0),
        "defect_over_positive_kinetic": float(actual / (k0 + ke0)),
        "relative_heat": float(heat),
        "ion_change": float(dki),
        "electron_change": float(dke),
        "magnetic_change": float(dwb),
        "ampere_row": float(maxrow),
        "faraday_error": float(faraday),
        "div_impulse": float(gauge),
        "displacement_curl": float(dcurl),
        "displacement_max": float(
            max(abs(v) for k, v in f1.items() if k[0].startswith("D"))
        ),
        "electric_momentum_transfer": float(electric_transfer),
        "range_max_ratio": float(ledger["range_ratio"]),
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
