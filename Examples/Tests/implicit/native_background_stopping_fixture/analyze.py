import hashlib
import json
import math
import sys
from collections import defaultdict
from decimal import Decimal as D
from decimal import getcontext
from pathlib import Path

from native_fab import name, read

getcontext().prec = 70
root = Path(sys.argv[1])
output = Path(sys.argv[2])


def F(x):
    return D.from_float(float(x))


L = json.loads((root / "LEDGER.json").read_text())
BG = json.loads((root / "BACKGROUND_LEDGER.json").read_text())
pins = {}
q = F(L["q_e"])
me = F(L["m_e"])
c = F(L["c"])
mu = F(L["mu0"])
eps = F(2.220446049250313e-16)
pi = F(math.pi)
used = (root / "warpx_used_inputs").read_text()
periodic = any(
    "periodic" in line
    for line in used.splitlines()
    if line.startswith("boundary.field_hi")
)
pieces = read(name(root / "background_before", "hybrid_electron_energy_fp"))
N = max(x["valid"][1][0] for x in pieces) + 1
Nz = max(x["valid"][1][1] for x in pieces) + 1
dr = D(1) / N
dz = D(1) / Nz
checks = []


def pin(p):
    pins[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()


def check(label, error, bound):
    ok = abs(error) <= bound
    checks.append(dict(label=label, error=str(error), bound=str(bound), pass_=ok))
    assert ok, checks[-1]


def table(path, valid=True):
    result = {}
    parts = read(path)
    pin(path.with_name(path.name + "_H"))
    for p in path.parent.glob(path.name + "_D_*"):
        pin(p)
    for part in parts:
        lo, hi, st = (
            part["valid"] if valid else (part["lo"], part["hi"], part["stagger"])
        )
        for i in range(lo[0], hi[0] + 1):
            for j in range(lo[1], hi[1] + 1):
                key = i, j % Nz if valid and periodic else j
                v = F(part["array"][i - part["lo"][0], j - part["lo"][1], 0])
                if key in result and valid:
                    check("duplicate valid owner", v - result[key], D(0))
                result[key] = v
    return result, st


def volume(k, st):
    i, j = k
    r = (D(i) + (D(0) if st[0] else D(".5"))) * dr
    z = (D(j) + (D(0) if st[1] else D(".5"))) * dz
    return (
        pi
        * (min(D(1), r + dr / 2) ** 2 - max(D(0), r - dr / 2) ** 2)
        * (dz if periodic else min(D(1), z + dz / 2) - max(D(0), z - dz / 2))
    )


def particles(folder):
    data = {}
    for path in folder.glob("particles.rank*.txt"):
        pin(path)
        for line in path.read_text().splitlines():
            x = line.split()
            key = x[0], int(x[1])
            assert key not in data
            data[key] = list(map(F, x[2:]))
    return data


old = particles(root / "background_before")
new = particles(root / "background_after_recovery")
assert old.keys() == new.keys()
count = len(old)
ops = D(4096 * (count + 1))
factor = 4 * ops * eps / (1 - ops * eps)
work = defaultdict(D)
grid = defaultdict(D)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def gamma(u):
    return (1 + dot(u, u) / (c * c)).sqrt()


def dk(u, v, m):
    return m * (dot(v, v) - dot(u, u)) / (gamma(v) + gamma(u))


points = {}
for path in root.glob("background_points.rank*.txt"):
    pin(path)
    for line in path.read_text().splitlines():
        x = line.split()
        points[x[0], int(x[1])] = list(map(F, x[2:]))
assert points.keys() == old.keys()
for key, a in old.items():
    b = new[key]
    mass, charge, r, theta, z = a[:5]
    u = a[5:8]
    v = b[5:8]
    weight = a[-1]
    assert a[:5] == b[:5] and weight == b[-1]
    x = points[key]
    psi = x[3:6]
    ve = x[6:9]
    half = [charge * p / (2 * mass) for p in psi]
    minus = [aa + bb for aa, bb in zip(u, half)]
    drag = [aa - bb for aa, bb in zip(v, half)]
    delta = weight * dk(u, v, mass)
    dw = weight * dk(minus, drag, mass)
    bulk = weight * mass * dot([bb - aa for aa, bb in zip(minus, drag)], ve)
    ew = weight * (dk(u, minus, mass) + dk(drag, v, mass))
    vm = [(aa / gamma(u) + bb / gamma(v)) / 2 for aa, bb in zip(u, v)]
    endpoint = weight * charge * dot(vm, psi)
    for label, value in dict(
        ion_energy_change=delta,
        drag_energy_change=dw,
        drag_bulk_work=bulk,
        electric_particle_work=ew,
        endpoint_particle_work=endpoint,
        heat=-dw + bulk,
        relativistic_defect=ew - endpoint,
    ).items():
        work[label] += value
        work["positive_" + label] += abs(value)
for component in range(3):

    def fld(prefix):
        return table(root / f"background_{prefix}_{component}")[0]

    P, st = table(root / f"background_Psi_{component}")
    I0 = fld("I0")
    I1 = fld("I1")
    J0, _ = table(
        name(root / "background_before", "diagnostic_Je_endpoint_fp", component)
    )
    J1 = fld("Je1")
    Mean = fld("Mean")
    Pi = fld("Pi")
    M = fld("M")
    D0, _ = table(
        name(root / "background_before", "diagnostic_D_endpoint_fp", component)
    )
    dD = fld("dD")
    W0, _ = table(
        name(root / "background_before", "native_PEC_wall_endpoint_fp", component)
    )
    dW = fld("dW")
    C0 = fld("C0")
    dC = fld("dC")
    for k, p in P.items():
        measure = volume(k, st)
        jm = (J0[k] + J1[k]) / 2
        im = (I0[k] + I1[k]) / 2
        reaction = M[k] * q / me * Pi[k]
        reaction = D(0) if component == 1 and k[0] == 0 else reaction
        values = dict(
            ion_grid_work=im * p,
            electron_energy_change=jm * M[k] * (J1[k] - J0[k]),
            constraint_work=(
                im + jm + D0[k] + dD[k] / 2 + W0[k] + dW[k] / 2 - C0[k] - dC[k] / 2
            )
            * p,
            mean_current_work=(jm - Mean[k]) * reaction,
            reaction_mean=Mean[k] * reaction,
            curl_pair=(C0[k] + dC[k] / 2) * p,
            displacement_work=-(D0[k] + dD[k] / 2) * p,
            conductor_work=-(W0[k] + dW[k] / 2) * p,
            initial_constraint_work=(I0[k] + J0[k] + D0[k] + W0[k] - C0[k]) * p,
            initial_displacement_work=-D0[k] * p,
            initial_conductor_work=-W0[k] * p,
        )
        for label, value in values.items():
            grid[label] += value * measure
            grid["positive_" + label] += abs(value * measure)
    for field, prefix in [
        ("hybrid_A_fp", "A"),
        ("Bfield_fp", "B"),
        ("diagnostic_D_endpoint_fp", "D"),
        ("native_PEC_wall_endpoint_fp", "W"),
    ]:
        f0, st = table(name(root / "background_before", field, component))
        df = fld("d" + prefix)
        f1 = fld(prefix + "1")
        delta_error = max(abs(f1[k] - f0[k] - df[k]) for k in f0)
        scale = max(abs(f0[k]) + abs(df[k]) for k in f0)
        check(
            "full " + prefix + str(component) + " equals saved plus increment",
            delta_error,
            2 * eps * scale,
        )
        if prefix == "B":
            for k, db in df.items():
                v = volume(k, st)
                cross = f0[k] * db / mu * v
                self = db * db / (2 * mu) * v
                initial = f0[k] * f0[k] / (2 * mu) * v
                actual = (f1[k] - f0[k]) * (f1[k] + f0[k]) / (2 * mu) * v
                for label, val in dict(
                    magnetic_cross_work=cross,
                    magnetic_self_work=self,
                    magnetic_initial_energy=initial,
                    magnetic_actual=actual,
                ).items():
                    grid[label] += val
                    grid["positive_" + label] += abs(val)
u0, st = table(name(root / "background_before", "hybrid_electron_energy_fp"))
u1, _ = table(root / "background_U1")
for k, val in u0.items():
    grid["U_change"] += (u1[k] - val) * volume(k, st)
    grid["positive_U_change"] += abs((u1[k] - val) * volume(k, st))
    grid["U_old"] += abs(val) * volume(k, st)
ind = dict(
    (k, work[k])
    for k in [
        "ion_energy_change",
        "drag_energy_change",
        "drag_bulk_work",
        "electric_particle_work",
        "endpoint_particle_work",
        "heat",
        "relativistic_defect",
    ]
)
ind.update(
    (k, grid[k])
    for k in [
        "electron_energy_change",
        "constraint_work",
        "mean_current_work",
        "displacement_work",
        "conductor_work",
    ]
)
ind.update(
    spatial_transfer=work["endpoint_particle_work"] - grid["ion_grid_work"],
    transpose_work=work["drag_bulk_work"] + grid["reaction_mean"],
    heat_transfer_error=grid["U_change"] - work["heat"],
    magnetic_energy_change=grid["magnetic_cross_work"] + grid["magnetic_self_work"],
    curl_work=grid["magnetic_cross_work"]
    + grid["magnetic_self_work"]
    + grid["curl_pair"],
)
positive = sum(v for k, v in work.items() if k.startswith("positive_")) + sum(
    v for k, v in grid.items() if k.startswith("positive_")
)
bound = factor * positive
for label, value in ind.items():
    check("native ledger " + label, value - F(L[label]), bound)
for label in [
    "magnetic_cross_work",
    "magnetic_self_work",
    "magnetic_initial_energy",
    "initial_constraint_work",
    "initial_displacement_work",
    "initial_conductor_work",
]:
    check("background ledger " + label, grid[label] - F(BG[label]), bound)
actual = (
    work["ion_energy_change"]
    + grid["electron_energy_change"]
    + grid["magnetic_actual"]
    + grid["U_change"]
)
terms = sum(
    ind[k]
    for k in [
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
)
check("stable actual inventory versus nine-term identity", actual - terms, bound)
check(
    "actual inventory versus native ledger",
    actual - F(L["actual_energy_defect"]),
    bound,
)
check(
    "full delta magnetic versus cross plus self",
    grid["magnetic_actual"] - ind["magnetic_energy_change"],
    factor
    * (
        grid["positive_magnetic_initial_energy"]
        + grid["positive_magnetic_cross_work"]
        + grid["positive_magnetic_self_work"]
    ),
)
assert grid["magnetic_cross_work"] != 0 and grid["magnetic_initial_energy"] > 0
result = dict(
    pass_=all(x["pass_"] for x in checks),
    precision=70,
    particles=count,
    periodic=periodic,
    checks=checks,
    independent={k: str(v) for k, v in ind.items()},
    grid={k: str(v) for k, v in grid.items()},
    actual_total_change_J=str(actual),
    sum_explicit_defects_J=str(terms),
    identity_remainder_J=str(actual - terms),
    positive_work_operand_scale_J=str(positive),
    bound_J=str(bound),
    arithmetic_factor=str(factor),
    pins=pins,
    scope="Accounting and matched native heat; no zero total energy or temporal order claim",
)
output.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            k: result[k]
            for k in [
                "pass_",
                "particles",
                "periodic",
                "actual_total_change_J",
                "identity_remainder_J",
                "bound_J",
                "independent",
            ]
        },
        indent=2,
    )
)
