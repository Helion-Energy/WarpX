"""Independent represented-input RZm0 event inventory and native maps.

No observed discrepancy sets a bound. Original nonlinear/physical gates and
count/positive-scale accumulation bounds are retained. Finite transfer terms
remain in the actual budget.
"""

import argparse
import json
from decimal import ROUND_FLOOR, getcontext
from decimal import Decimal as D
from pathlib import Path

getcontext().prec = 70
EPS = D(2) ** -52
PI = D("3.141592653589793238462643383279502884197169399375105820974944592307816")
E_TYPES = [(0, 1), (1, 1), (1, 0)]
B_TYPES = [(1, 0), (0, 0), (0, 1)]


def num(x):
    v = D.from_float(float(x))
    assert v.is_finite()
    return v


def norm(x):
    return sum((v * v for v in x), D(0)).sqrt()


def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), D(0))


def trig(x):
    # Independently rounded represented theta, evaluated in Decimal.
    while x > PI:
        x -= 2 * PI
    while x < -PI:
        x += 2 * PI
    sn = x
    cs = D(1)
    a = x
    b = D(1)
    for k in range(1, 70):
        a *= -x * x / D((2 * k) * (2 * k + 1))
        b *= -x * x / D((2 * k - 1) * (2 * k))
        sn += a
        cs += b
    return cs, sn


def particle_table(root, stem):
    out = {}
    for p in sorted(root.glob(stem + ".rank*.txt")):
        for line in p.read_text().splitlines():
            a = line.split()
            key = (a[0], int(a[1]))
            assert key not in out
            out[key] = list(map(num, a[2:]))
            assert len(out[key]) == 9
    assert out
    return out


def analyze(root):
    root = Path(root)
    raw = json.loads((root / "LEDGER.json").read_text())
    meta = json.loads((root / "GEOMETRY.json").read_text())
    assert raw["pass"] is True
    ledger = {k: num(v) for k, v in raw.items() if type(v) in (float, int)}
    n = meta["n"]
    per = meta["periodic"]
    assert type(n) is int and n >= 8 and type(per) is bool
    h = D(1) / n
    qe, me, c, mu = (ledger[x] for x in ["q_e", "m_e", "c", "mu0"])
    c2 = c * c

    def gamma(u):
        return (1 + dot(u, u) / c2).sqrt()

    log = (root / "run.log").read_text()
    assert log.count("NATIVE_RZ_SPATIAL_STOPPING_PASS ") == 1
    assert (
        log.count(
            "NATIVE_STOPPING_ZERO_PSI_ORACLE native_cylindrical_kick_exact=1 handler_particle_restore_exact=1"
        )
        == 1
    )
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
        assert marker in log, marker
    old, new = particle_table(root, "before"), particle_table(root, "after")
    assert old.keys() == new.keys()
    count = D(len(old))
    assert count == ledger["contribution_count"]
    ops = 4096 * (1 + count)
    assert ops == ledger["operations"] and ops * EPS < D(".01")
    factor = 4 * ops * EPS / (1 - ops * EPS)
    checks = []

    def gate(name, error, bound):
        assert error.is_finite() and bound.is_finite() and bound >= 0
        assert abs(error) <= bound, (name, str(error), str(bound))
        checks.append(dict(name=name, difference=float(error), bound=float(bound)))

    def typ(label):
        if label == "rho":
            return (1, 1)
        if label == "U":
            return (0, 0)
        return (B_TYPES if label[0] == "B" else E_TYPES)[int(label[-1])]

    def volume(i, j, t):
        r = (D(i) + D(1 - t[0]) / 2) * h
        z = (D(j) + D(1 - t[1]) / 2) * h
        low = max(D(0), r - h / 2)
        high = min(D(1), r + h / 2)
        width = h if per else min(D(1), z + h / 2) - max(D(0), z - h / 2)
        return PI * (high * high - low * low) * width

    def fields(stem):
        out = {}
        for p in sorted(root.glob(stem + ".rank*.txt")):
            for row in p.read_text().splitlines():
                a = row.split()
                label = a[0]
                i, j, k = map(int, a[1:4])
                assert k == 0
                key = label, i, j
                assert key not in out
                nt = tuple(map(int, a[5:7]))
                assert nt == typ(label)
                vv = volume(i, j, nt)
                gate("physical measure " + str(key), num(a[7]) - vv, 64 * EPS * vv)
                out[key] = num(a[4])
        labels = ["rho", "U"] + [
            f"{p}{c}" for p in ["J", "A", "B", "D", "I"] for c in range(3)
        ]
        expected = {
            (label, i, j)
            for label in labels
            for i in range(n + typ(label)[0])
            for j in range(n + (0 if per else typ(label)[1]))
        }
        assert out.keys() == expected, "complete unique field ownership"
        return out

    f0, f1 = fields("fields_before"), fields("fields_after")
    assert f0.keys() == f1.keys()
    for key in f0:
        if key[0] == "rho":
            assert f0[key] == f1[key]

    def read_vector(stem, types=E_TYPES):
        out = {}
        for p in sorted(root.glob(stem + ".rank*.txt")):
            for line in p.read_text().splitlines():
                a = line.split()
                key = tuple(map(int, a[:3]))
                assert key not in out
                out[key] = num(a[3])
        expected = {
            (c, i, j)
            for c, t in enumerate(types)
            for i in range(n + t[0])
            for j in range(n + (0 if per else t[1]))
        }
        assert out.keys() == expected, (stem, "complete unique prepared operand")
        return out

    psi = read_vector("Psi")
    wall = read_vector("Jwall")
    mean = read_vector("MeanJe")
    ve = read_vector("StopVe")
    pi = read_vector("StopPi")
    aux = read_vector("ElectricAux", [(1, 1)] * 3)

    def get(table, comp, i, j, kind="electric", types=E_TYPES):
        t = types[comp]
        sign = D(1)
        if per:
            j %= n
        elif j < 0 or j > n - 1 + t[1]:
            j = -j - (1 - t[1]) if j < 0 else 2 * (n - 1 + t[1]) - j + (1 - t[1])
            sign *= (
                -1
                if (not t[1] if kind in ["magnetic", "potential"] else comp == 2)
                else 1
            )
        if i < 0:
            i = -i - (1 - t[0])
            sign *= 1 if comp == 2 else -1
        if i > n - 1 + t[0]:
            if kind == "magnetic":
                return D(0)
            if kind == "potential" and t[0]:
                return D(0)
            i = 2 * (n - 1 + t[0]) - i + (1 - t[0])
            sign *= (
                (-1 if comp == 0 else 1)
                if kind == "velocity"
                else (-1 if comp != 0 or kind == "potential" else 1)
            )
        return sign * table[comp, i, j]

    mass = {}
    for key in psi:
        comp, i, j = key
        r = f0["rho", i, j]
        if comp != 1:
            ni, nj = (i + 1, j) if comp == 0 else (i, (j + 1) % n if per else j + 1)
            r = (r + f0["rho", ni, nj]) / 2
        assert r > qe * D("1e16")
        mass[key] = me / (qe * r)
    k0 = dki = D(0)
    momentum = [D(0)] * 3
    for key, a in old.items():
        b = new[key]
        assert a[:5] == b[:5] and a[8] == b[8]
        m, q = a[:2]
        u = a[5:8]
        v = b[5:8]
        w = a[8]
        ga, gb = gamma(u), gamma(v)
        k0 += w * m * dot(u, u) / (ga + 1)
        dki += (
            w
            * m
            * dot([y - x for x, y in zip(u, v)], [y + x for x, y in zip(u, v)])
            / (ga + gb)
        )
        for c0 in range(3):
            momentum[c0] += w * m * (v[c0] - u[c0])
        assert norm(u) <= ledger["cap"] and norm(v) <= ledger["cap"]
    ke0 = dke = D(0)
    ep = [D(0)] * 3
    for key, M in mass.items():
        co, i, j = key
        a = f0[f"J{co}", i, j]
        b = f1[f"J{co}", i, j]
        vol = volume(i, j, E_TYPES[co])
        ke0 += vol * a * M * a / 2
        dke += vol * (a + b) * M * (b - a) / 2
        ep[co] -= vol * me / qe * (b - a)
    u0 = sum(
        (v * volume(i, j, (0, 0)) for (label, i, j), v in f0.items() if label == "U"),
        D(0),
    )
    heat = sum(
        (
            (f1[key] - v) * volume(key[1], key[2], (0, 0))
            for key, v in f0.items()
            if key[0] == "U"
        ),
        D(0),
    )
    dwb = sum(
        (
            (f1[key] - v)
            * (f1[key] + v)
            * volume(key[1], key[2], typ(key[0]))
            / (2 * mu)
            for key, v in f0.items()
            if key[0][0] == "B"
        ),
        D(0),
    )
    kinetic_bound = factor * (k0 + ke0)
    energy_bound = factor * (
        k0 + ke0 + u0 + ledger["absolute_work"] + abs(dke) + abs(dwb)
    )
    for name, x, reported in [
        ("stable ion deltaK", dki, "ion_energy_change"),
        ("initial ion energy", k0, "ion_initial_energy"),
        ("initial electron energy", ke0, "electron_initial_energy"),
        ("native mass electron deltaK", dke, "electron_energy_change"),
    ]:
        gate(name, x - ledger[reported], kinetic_bound)
    gate(
        "cell U heat",
        heat - ledger["heat"] - ledger["heat_transfer_error"],
        energy_bound,
    )
    gate(
        "native magnetic storage", dwb - ledger["magnetic_energy_change"], energy_bound
    )
    actual = dki + dke + heat + dwb
    gate("actual full inventory", actual - ledger["actual_energy_defect"], energy_bound)
    terms = [
        "relativistic_defect",
        "spatial_transfer",
        "constraint_work",
        "mean_current_work",
        "transpose_work",
        "heat_transfer_error",
        "curl_work",
        "displacement_work",
        "conductor_work",
    ]
    gate(
        "independent signed accounting",
        actual - sum((ledger[x] for x in terms), D(0)),
        energy_bound,
    )
    gate(
        "relative heat law",
        ledger["heat"] + ledger["drag_energy_change"] - ledger["drag_bulk_work"],
        energy_bound,
    )
    gate(
        "NR convention",
        ledger["relativistic_defect"],
        ledger["relativistic_bound"] + kinetic_bound,
    )
    assert ledger["relativistic_bound"] <= ledger["relative_budget"] * (k0 + ke0) * (
        1 + 64 * EPS
    )
    assert ledger["spatial_residual"] <= D("1e-12") and ledger["gauge_error"] <= D(
        "1e-12"
    )
    assert (
        ledger["grid_residual"] <= ledger["grid_bound"]
        and ledger["momentum_error"] <= ledger["momentum_bound"]
    )
    assert (
        ledger["faraday_error"] <= ledger["faraday_bound"]
        and ledger["displacement_curl"] <= ledger["displacement_curl_bound"]
    )
    assert (
        ledger["range_calls"] == 0
        and ledger["current_range_mean"] == ledger["impulse_range_mean"] == 0
    )
    assert (
        ledger["heat"] >= 0
        and abs(ledger["accounting_defect"]) <= ledger["arithmetic_bound"]
    )
    # Reconstruct native auxiliary centering on every physical node from actual
    # E-image parity, not the different A curl extension.
    for (co, i, j), value in aux.items():
        expected = (
            get(psi, co, i, j)
            if co == 1
            else (
                (get(psi, co, i, j) + get(psi, co, i - 1, j)) / 2
                if co == 0
                else (get(psi, co, i, j) + get(psi, co, i, j - 1)) / 2
            )
        )
        gate(
            "native MC auxiliary " + str((co, i, j)),
            value - expected,
            4096 * EPS * max(abs(expected), max(map(abs, psi.values()))),
        )

    def b3(x):
        x = abs(x)
        if x < 1:
            return D(2) / 3 - x * x + x * x * x / 2
        if x < 2:
            return (2 - x) ** 3 / 6
        return D(0)

    def gather(table, a, kind, types):
        radius, theta, z = a[2:5]
        co, si = trig(theta)
        cyl = []
        for comp, t in enumerate(types):
            r = radius * n - D(1 - t[0]) / 2
            zz = z * n - D(1 - t[1]) / 2
            ir = int(r.to_integral_value(rounding=ROUND_FLOOR))
            iz = int(zz.to_integral_value(rounding=ROUND_FLOOR))
            value = D(0)
            for i in range(ir - 1, ir + 3):
                for j in range(iz - 1, iz + 3):
                    value += (
                        b3(r - i) * b3(zz - j) * get(table, comp, i, j, kind, types)
                    )
            cyl.append(value)
        return [co * cyl[0] - si * cyl[1], si * cyl[0] + co * cyl[1], cyl[2]]

    def scalar_rho(a):
        r = a[2] * n
        z = a[4] * n
        i = int(r.to_integral_value(rounding=ROUND_FLOOR))
        j = int(z.to_integral_value(rounding=ROUND_FLOOR))
        ar = r - i
        az = z - j
        return sum(
            (
                f0["rho", i + di, (j + dj) % n if per else j + dj]
                * (ar if di else 1 - ar)
                * (az if dj else 1 - az)
                for di in (0, 1)
                for dj in (0, 1)
            ),
            D(0),
        )

    def dk(a, b, m):
        return (
            m
            * dot([y - x for x, y in zip(a, b)], [y + x for x, y in zip(a, b)])
            / (gamma(a) + gamma(b))
        )

    endpoint = drag = bulk = ew = thermal = nrbound = D(0)
    electric_p = [D(0)] * 3
    reaction_p = [D(0)] * 3
    rate_map_error = D(0)
    for key, a in old.items():
        b = new[key]
        m, q = a[:2]
        w = a[8]
        ua = a[5:8]
        ub = b[5:8]
        pg = gather(aux, a, "electric", [(1, 1)] * 3)
        vg = gather(ve, a, "velocity", E_TYPES)
        minus = [x + q * y / (2 * m) for x, y in zip(ua, pg)]
        dragged = [x - q * y / (2 * m) for x, y in zip(ub, pg)]
        imp = [m * (y - x) for x, y in zip(minus, dragged)]
        ddrag = w * dk(minus, dragged, m)
        bwork = w * dot(imp, vg)
        drag += ddrag
        bulk += bwork
        thermal -= ddrag - bwork
        ew += w * (dk(ua, minus, m) + dk(dragged, ub, m))
        endpoint += (
            w
            * q
            * dot(pg, [(x / gamma(ua) + y / gamma(ub)) / 2 for x, y in zip(ua, ub)])
        )
        nrbound += w * abs(q) * norm(pg) * ledger["cap"] ** 3 / c2
        for c0 in range(3):
            electric_p[c0] += w * q * pg[c0]
            reaction_p[c0] += w * imp[c0]
        # Actual native law in exact represented-input arithmetic; Cartesian and
        # collision-frame isotropic maps differ only by bounded native rotations.
        expected = minus
        if key[0] == "fast" and ledger["interval"] > 0:
            rho = scalar_rho(a)
            te = num(meta["te_kelvin"]) * num(meta["kb"])
            ep0 = num(meta["epsilon0"])
            clog = num(meta["clog"])
            rate = (
                D(2).sqrt()
                * (rho / qe)
                * q
                * q
                * qe
                * qe
                * me.sqrt()
                * clog
                / (12 * PI * PI.sqrt() * ep0 * ep0 * m * te * te.sqrt())
            )
            decay = (-rate * ledger["interval"]).exp()
            gm = gamma(minus)
            v = [e + (u / gm - e) * decay for u, e in zip(minus, vg)]
            gg = 1 / (1 - dot(v, v) / c2).sqrt()
            expected = [gg * x for x in v]
        error = norm([x - y for x, y in zip(expected, dragged)])
        rate_map_error = max(rate_map_error, error)
        gate(
            "native finite kick " + str(key),
            error,
            factor * (norm(ua) + norm(minus) + norm(vg) + norm(dragged)),
        )
    for name, x, label in [
        ("actual MC endpoint work", endpoint, "endpoint_particle_work"),
        ("electric half-kick work", ew, "electric_particle_work"),
        ("native drag work", drag, "drag_energy_change"),
        ("stopping bulk work", bulk, "drag_bulk_work"),
        ("relative heat independently reconstructed", thermal, "heat"),
    ]:
        gate(name, x - ledger[label], energy_bound)
    gate(
        "NR positive bound reconstructed",
        nrbound - ledger["relativistic_bound"],
        factor * max(nrbound, D("1e-300")),
    )
    B = {
        (co, i, j): f1[f"B{co}", i, j]
        for co, t in enumerate(B_TYPES)
        for i in range(n + t[0])
        for j in range(n + (0 if per else t[1]))
    }
    A = {
        (co, i, j): f1[f"A{co}", i, j]
        for co, t in enumerate(E_TYPES)
        for i in range(n + t[0])
        for j in range(n + (0 if per else t[1]))
    }
    Dv = {
        (co, i, j): f1[f"D{co}", i, j]
        for co, t in enumerate(E_TYPES)
        for i in range(n + t[0])
        for j in range(n + (0 if per else t[1]))
    }

    def ampere(comp, i, j):
        def g(c, i, j):
            return get(B, c, i, j, "magnetic", B_TYPES)

        if comp == 0:
            return (-g(1, i, j) + g(1, i, j - 1)) / (h * mu)
        if comp == 1:
            return (
                D(0)
                if i == 0
                else (-g(2, i, j) + g(2, i - 1, j) + g(0, i, j) - g(0, i, j - 1))
                / (h * mu)
            )
        return (
            4 * g(1, 0, j) / (h * mu)
            if i == 0
            else ((D(i) + D(".5")) * g(1, i, j) - (D(i) - D(".5")) * g(1, i - 1, j))
            / (D(i) * h * mu)
        )

    def curl(field, comp, i, j):
        def g(c, i, j):
            return get(field, c, i, j, "potential")

        if comp == 0:
            return D(0) if i == 0 else (-g(1, i, j + 1) + g(1, i, j)) / h
        if comp == 1:
            return (g(0, i, j + 1) - g(0, i, j) - g(2, i + 1, j) + g(2, i, j)) / h
        return ((i + 1) * g(1, i + 1, j) - i * g(1, i, j)) / ((D(i) + D(".5")) * h)

    maxrow = faraday = dcurl = D(0)
    grid = curlwork = dwork = constraint = meanwork = trialwork = wallwork = (
        axiswork
    ) = D(0)
    pi_z = mass_psi_z = D(0)
    for key, p in psi.items():
        co, i, j = key
        vol = volume(i, j, E_TYPES[co])
        C = ampere(co, i, j)
        ion_current = (f0[f"I{co}", i, j] + f1[f"I{co}", i, j]) / 2
        J = (f0[f"J{co}", i, j] + f1[f"J{co}", i, j]) / 2
        Dv1 = Dv[key]
        W = wall[key]
        assert W == 0 if not (co != 0 and i == n) else p == 0
        if co == 1 and i == 0:
            assert p == 0 and mean[key] == ve[key] == f1[f"J{co}", i, j] == 0
        grid += vol * p * ion_current
        curlwork += vol * p * C / 2
        dwork -= vol * p * Dv1 / 2
        wallwork -= vol * p * W / 2
        constraint += vol * p * (ion_current + J + (Dv1 + W - C) / 2)
        b = mass[key] * qe / me * (0 if co == 1 and i == 0 else pi[key])
        meanwork += vol * (J - mean[key]) * b
        trialwork += vol * mean[key] * b
        maxrow = max(maxrow, abs(C - f1[f"I{co}", i, j] - f1[f"J{co}", i, j] - Dv1 - W))
        reaction_row = f1[f"J{co}", i, j] - f0[f"J{co}", i, j] - (p + b) / mass[key]
        gate("electron impulse row " + str(key), reaction_row, ledger["grid_bound"])
        if co == 1 and i == 0:
            axiswork += vol * pi[key] * ve[key]
        if co == 2:
            pi_z += vol * pi[key]
            mass_psi_z += vol * me / qe * p / mass[key]
    for (co, i, j), x in B.items():
        faraday = max(faraday, abs(x - curl(A, co, i, j)))
        dcurl = max(dcurl, abs(curl(Dv, co, i, j)))
    curlwork += dwb
    for name, x, label in [
        ("MC/current transfer", endpoint - grid, "spatial_transfer"),
        ("curl boundary work", curlwork, "curl_work"),
        ("displacement work", dwork, "displacement_work"),
        ("all-row constraint work", constraint, "constraint_work"),
        ("trial mean work", meanwork, "mean_current_work"),
        ("stopping transpose work", bulk + trialwork, "transpose_work"),
    ]:
        gate(name, x - ledger[label], energy_bound)
    gate("native Ampere with explicit conductor", maxrow, ledger["grid_bound"])
    gate("native Faraday", faraday, ledger["faraday_bound"])
    gate("longitudinal D", dcurl, ledger["displacement_curl_bound"])
    gate("conductor constrained work", wallwork, D(0))
    gate("axis constrained work", axiswork, D(0))
    assert ledger["conductor_work"] == ledger["axis_reaction_work"] == 0
    axial = momentum[2] + ep[2]
    er = electric_p[2] - mass_psi_z
    sr = reaction_p[2] - pi_z
    gate(
        "axial represented transfer identity", axial - er - sr, ledger["momentum_bound"]
    )
    gate(
        "axial defect receipt",
        axial - ledger["axial_momentum_defect"],
        ledger["momentum_bound"],
    )
    gate(
        "axial electric transfer receipt",
        er - ledger["axial_electric_transfer"],
        ledger["momentum_bound"],
    )
    gate(
        "axial stopping image receipt",
        sr - ledger["axial_image_transfer"],
        ledger["momentum_bound"],
    )

    def divergence(i, j):
        rad = (
            4 * get(psi, 0, 0, j) / h
            if i == 0
            else (
                (D(i) + D(".5")) * get(psi, 0, i, j)
                - (D(i) - D(".5")) * get(psi, 0, i - 1, j)
            )
            / (D(i) * h)
        )
        return rad + (get(psi, 2, i, j) - get(psi, 2, i, j - 1)) / h

    maxdiv = max(abs(divergence(i, j)) for i in range(n) for j in range(n + (not per)))
    gate(
        "independent transverse impulse",
        maxdiv,
        4
        * n
        * (D("1e-12") + 4096 * EPS)
        * max(me / qe * ledger["cap"], max(map(abs, psi.values()))),
    )
    if ledger["interval"] > 0:
        assert (
            ledger["heat"] > 0
            and dwb > 0
            and max(map(abs, wall.values())) > 0
            and max(map(abs, Dv.values())) > 0
        )
    return dict(
        pass_=True,
        scope="fixed-density RZm0 standalone event; conductor reaction separate from plasma Je",
        n=n,
        periodic=per,
        particles=int(count),
        checks=checks,
        actual_total_defect=float(actual),
        positive_kinetic_scale=float(k0 + ke0),
        defect_over_positive_kinetic=float(actual / (k0 + ke0)),
        heat=float(heat),
        ion_change=float(dki),
        electron_change=float(dke),
        magnetic_change=float(dwb),
        ampere_row=float(maxrow),
        faraday_error=float(faraday),
        displacement_curl=float(dcurl),
        displacement_max=float(max(map(abs, Dv.values()))),
        div_impulse=float(maxdiv),
        native_rate_map_error=float(rate_map_error),
        axial_momentum_defect=float(axial),
        axial_electric_transfer=float(er),
        axial_stopping_image_transfer=float(sr),
        conductor_norm=float(max(map(abs, wall.values()))),
        finite_terms={key: float(ledger[key]) for key in terms},
        no_total_conservation_claim=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.run)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}))
