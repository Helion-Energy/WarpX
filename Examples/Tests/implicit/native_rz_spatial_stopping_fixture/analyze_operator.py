"""Independent Decimal reconstruction of actual native RZ impulse operators."""

import argparse
import json
import re
from decimal import Decimal as D
from decimal import getcontext
from pathlib import Path

getcontext().prec = 70
EPS = D(2) ** -52
PI = D("3.141592653589793238462643383279502884197169399375105820974944592307816")


def value(s):
    v = D.from_float(float(s))
    assert v.is_finite()
    return v


def analyze(root):
    root = Path(root)
    rows = [
        x
        for x in (root / "run.log").read_text().splitlines()
        if x.startswith("RZ_EVENT_OPERATOR ")
    ]
    assert len(rows) == 1
    meta = dict(re.findall(r"(\w+)=([^ ]+)", rows[0]))
    n = int(meta["n"])
    per = bool(int(meta["periodic"]))
    h = D(1) / n
    mu = value(meta["mu0"])
    assert meta["exact_fields_particles_rng"] == "1"
    assert value(meta["scalar_residual"]) <= value(meta["scalar_target"])
    checks = []

    def gate(name, error, bound):
        assert error.is_finite() and bound.is_finite() and bound >= 0
        assert abs(error) <= bound, (name, str(error), str(bound))
        checks.append(dict(name=name, difference=float(error), bound=float(bound)))

    tables = {}
    types_e = [(0, 1), (1, 1), (1, 0)]
    types_b = [(1, 0), (0, 0), (0, 1)]

    def vol(i, j, t):
        r = (D(i) + D(1 - t[0]) / 2) * h
        lo = max(D(0), r - h / 2)
        hi = min(D(1), r + h / 2)
        z = (D(j) + D(1 - t[1]) / 2) * h
        dz = h if per else min(D(1), z + h / 2) - max(D(0), z - h / 2)
        return PI * (hi * hi - lo * lo) * dz

    for name in ["A", "B", "C", "PL", "PT", "BT", "CT"]:
        table = {}
        types = types_b if name in ["B", "BT"] else types_e
        for file in sorted(root.glob(name + "-rank*.txt")):
            for row in file.read_text().splitlines():
                c, i, j, nr, nz, x, v = row.split()
                c, i, j, nr, nz = map(int, [c, i, j, nr, nz])
                assert (nr, nz) == types[c]
                if per:
                    j %= n
                key = (c, i, j)
                assert key not in table, (name, "duplicate", key)
                vv = vol(i, j, types[c])
                gate("volume " + name + str(key), value(v) - vv, 64 * EPS * vv)
                table[key] = (value(x), vv)
        expected = {
            (c, i, j)
            for c, t in enumerate(types)
            for i in range(n + t[0])
            for j in range(n + (0 if per else t[1]))
        }
        assert set(table) == expected, (name, "incomplete")
        tables[name] = table

    def get(name, c, i, j):
        types = types_b if name in ["B", "BT"] else types_e
        t = types[c]
        sign = D(1)
        if per:
            j %= n
        elif j < 0 or j > n - 1 + t[1]:
            j = -j - (1 - t[1]) if j < 0 else 2 * (n - 1 + t[1]) - j + (1 - t[1])
            # E/A: cell-normal component odd. Magnetic: cell-normal tangential odd.
            if not t[1]:
                sign = -sign
        if i < 0:
            i = -i - (1 - t[0])
            sign *= 1 if c == 2 else -1
        if i > n - 1 + t[0]:
            if name in ["B", "BT", "C", "CT", "PL"]:
                return D(0)
            if t[0]:
                return D(0)
            i = 2 * (n - 1) - i + 1
            sign = -sign
        return sign * tables[name][c, i, j][0]

    maxrow = {}
    for an, bn, cn in [("A", "B", "C"), ("PT", "BT", "CT")]:
        maxerr = D(0)
        for (c, i, j), (actual, _) in tables[bn].items():
            if c == 0:
                terms = (
                    [] if i == 0 else [-get(an, 1, i, j + 1) / h, get(an, 1, i, j) / h]
                )
            elif c == 1:
                terms = [
                    get(an, 0, i, j + 1) / h,
                    -get(an, 0, i, j) / h,
                    -get(an, 2, i + 1, j) / h,
                    get(an, 2, i, j) / h,
                ]
            else:
                terms = [
                    D(i + 1) * get(an, 1, i + 1, j) / ((D(i) + D(".5")) * h),
                    -D(i) * get(an, 1, i, j) / ((D(i) + D(".5")) * h),
                ]
            error = actual - sum(terms, D(0))
            bound = 4096 * EPS * sum(map(abs, terms), D(0))
            gate("curl " + bn + str((c, i, j)), error, bound)
            maxerr = max(maxerr, abs(error))
        maxrow[bn] = float(maxerr)
        maxerr = D(0)
        for (c, i, j), (actual, _) in tables[cn].items():
            if c == 0:
                terms = [-get(bn, 1, i, j) / (h * mu), get(bn, 1, i, j - 1) / (h * mu)]
            elif c == 1:
                terms = (
                    []
                    if i == 0
                    else [
                        -get(bn, 2, i, j) / (h * mu),
                        get(bn, 2, i - 1, j) / (h * mu),
                        get(bn, 0, i, j) / (h * mu),
                        -get(bn, 0, i, j - 1) / (h * mu),
                    ]
                )
            elif i == 0:
                terms = [4 * get(bn, 1, 0, j) / (h * mu)]
            else:
                terms = [
                    (D(i) + D(".5")) * get(bn, 1, i, j) / (D(i) * h * mu),
                    -(D(i) - D(".5")) * get(bn, 1, i - 1, j) / (D(i) * h * mu),
                ]
            error = actual - sum(terms, D(0))
            bound = 4096 * EPS * sum(map(abs, terms), D(0))
            gate("ampere " + cn + str((c, i, j)), error, bound)
            maxerr = max(maxerr, abs(error))
        maxrow[cn] = float(maxerr)
    count = sum(map(len, tables.values()))
    depth = D(4096 + 32 * count) * EPS
    assert depth < D(".01")
    gamma = depth / (1 - depth)

    def dot(a, b):
        return sum(
            (x * tables[b][key][0] * v for key, (x, v) in tables[a].items()), D(0)
        )

    def positive(a, b):
        return sum(
            (abs(x * tables[b][key][0]) * v for key, (x, v) in tables[a].items()), D(0)
        )

    for an, bn, cn in [("A", "B", "C"), ("PT", "BT", "CT")]:
        magnetic = dot(bn, bn) / mu
        work = dot(an, cn)
        gate(
            "native curl adjoint " + an,
            magnetic - work,
            4 * gamma * (positive(bn, bn) / mu + positive(an, cn)),
        )
    gate(
        "native reported curl work",
        value(meta["curl_work"]) - dot("A", "C"),
        4 * gamma * positive("A", "C"),
    )
    # The constraint current is conjugate only to tangential electric wall rows.
    wall_work = sum(
        (
            x * D(1 + i + j) * v
            for (c, i, j), (x, v) in tables["A"].items()
            if i == n and c in (1, 2)
        ),
        D(0),
    )
    gate("PEC constrained current work", wall_work, D(0))
    assert any(
        abs(x) > 0
        for (c, i, j), (x, _) in tables["A"].items()
        if i == n - 1 and c in (1, 2)
    )
    # Integrate the represented radial gradient from phi(R)=0; no solve or fit.
    nz = n if per else n + 1
    phi = {}
    for j in range(nz):
        phi[n, j] = D(0)
        for i in reversed(range(n)):
            phi[i, j] = phi[i + 1, j] - h * get("PL", 0, i, j)

    def divergence(name, i, j):
        radial = (
            4 * get(name, 0, 0, j) / h
            if i == 0
            else (
                (D(i) + D(".5")) * get(name, 0, i, j)
                - (D(i) - D(".5")) * get(name, 0, i - 1, j)
            )
            / (D(i) * h)
        )
        return radial + (get(name, 2, i, j) - get(name, 2, i, j - 1)) / h

    paired = sum(
        (
            phi[i, j] * divergence("PT", i, j) * vol(i, j, (1, 1))
            for i in range(n)
            for j in range(nz)
        ),
        D(0),
    )
    paired_scale = sum(
        (
            abs(phi[i, j] * divergence("PT", i, j)) * vol(i, j, (1, 1))
            for i in range(n)
            for j in range(nz)
        ),
        D(0),
    )
    orth = dot("PL", "PT")
    gate(
        "native projection metric identity",
        orth + paired,
        4 * gamma * (positive("PL", "PT") + paired_scale),
    )
    return dict(
        pass_=True,
        scope="native RZ field/operator only; no event or wall plasma closure",
        periodic=per,
        n=n,
        checks=checks,
        max_stencil_errors=maxrow,
        magnetic_norm=float(dot("B", "B") / mu),
        curl_work=float(dot("A", "C")),
        orthogonality=float(orth),
        independent_phi_div_pt=float(paired),
        max_div_pt=float(
            max(abs(divergence("PT", i, j)) for i in range(n) for j in range(nz))
        ),
        native_scalar_residual=float(value(meta["scalar_residual"])),
        native_scalar_target=float(value(meta["scalar_target"])),
        pinned_pec_rows=sum(i == n and c in (1, 2) for c, i, j in tables["A"]),
        positive_operator_scale=float(positive("A", "C") + positive("B", "B") / mu),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = analyze(args.run)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}))
