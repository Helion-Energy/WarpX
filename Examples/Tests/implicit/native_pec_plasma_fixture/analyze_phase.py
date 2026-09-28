import json
import sys
from pathlib import Path

import numpy as np
from native_fab import name, read, values

root = Path(sys.argv[1])
eps = np.finfo(float).eps
dt = 1.0e-12
old = root / (
    "event_after_recovery"
    if (root / "event_after_recovery").exists()
    else "initialized"
)
checks = []


def record(kind, error, bound, **rest):
    checks.append(
        dict(
            kind=kind,
            error=float(error),
            bound=float(bound),
            pass_=bool(error <= bound),
            **rest,
        )
    )


for step in [1, 2]:
    now = root / f"accepted_{step}"
    for field, stage in [
        ("diagnostic_Je_endpoint_fp", "diagnostic_Je_stage_fp"),
        ("diagnostic_D_endpoint_fp", "diagnostic_D_stage_fp"),
        ("native_PEC_wall_endpoint_fp", "native_PEC_wall_stage_fp"),
    ]:
        for c in range(3):
            a = values(name(old, field, c), False)
            b = values(name(now, stage, c), False)
            out = values(name(now, field, c), False)
            assert a.keys() == b.keys() == out.keys()
            maxratio = 0.0
            e = 0.0
            bd = 0.0
            for k in a:
                av, bv, ov = map(np.longdouble, [a[k], b[k], out[k]])
                err = abs(ov - (2 * bv - av))
                bound = 8 * eps * (abs(ov) + 2 * abs(bv) + abs(av))
                e = max(e, err)
                bd = max(bd, bound)
                maxratio = max(
                    maxratio,
                    float(err / bound)
                    if bound
                    else (0.0 if err == 0 else float("inf")),
                )
            record(
                "full_grow1_affine_phase",
                maxratio,
                1.0,
                step=step,
                field=field,
                component=c,
                absolute_error=float(e),
                maximum_bound=float(bd),
            )
    for c in range(3):
        wall = values(name(now, "native_PEC_wall_endpoint_fp", c))
        parts = read(name(now, "native_PEC_wall_endpoint_fp", c))
        imax = max(p["valid"][1][0] for p in parts)
        imax = imax if c else imax + 1
        record(
            "wall_support",
            max(abs(v) for (_, i, j), v in wall.items() if c == 0 or i != imax),
            0.0,
            step=step,
            component=c,
        )
        if c == 0:
            continue
        f = values(name(now, "native_PEC_pressure_hall_fp", c))
        m = values(name(now, "native_PEC_mass_fp", c))
        rate = values(name(now, "native_PEC_plasma_rate_fp", c))
        q = values(name(now, "diagnostic_Je_inertia_stage_fp", c))
        q0 = values(name(old, "diagnostic_Je_endpoint_fp", c))
        E = values(name(now, "Efield_fp", c))
        ew = 0.0
        ratios = [0.0, 0.0]
        errors = [0.0, 0.0]
        for k in q:
            if k[1] != imax:
                continue
            F, M, R, Q, Q0 = map(np.longdouble, [f[k], m[k], rate[k], q[k], q0[k]])
            error = [abs(F + M * R), abs(Q - Q0 - np.longdouble(dt / 2) * R)]
            bound = [
                8 * eps * (abs(F) + abs(M * R)),
                64 * eps * (abs(Q) + abs(Q0) + abs(np.longdouble(dt / 2) * R)),
            ]
            for n in range(2):
                ratios[n] = max(
                    ratios[n],
                    float(error[n] / bound[n])
                    if bound[n]
                    else (0.0 if error[n] == 0 else float("inf")),
                )
                errors[n] = max(errors[n], float(error[n]))
            ew = max(ew, abs(E[k] * wall[k]))
        record(
            "PEC_constitutive_force",
            ratios[0],
            1.0,
            step=step,
            component=c,
            absolute_error=errors[0],
        )
        record(
            "PEC_constitutive_increment",
            ratios[1],
            1.0,
            step=step,
            component=c,
            absolute_error=errors[1],
        )
        record("PEC_electric_wall_work", ew, 0.0, step=step, component=c)
    old = now
out = {"root": str(root), "checks": checks, "pass": all(x["pass_"] for x in checks)}
Path(sys.argv[2]).write_text(json.dumps(out, indent=2) + "\n")
print("PHASE", out["pass"], len(checks))
print([x for x in checks if not x["pass_"]])
raise SystemExit(0 if out["pass"] else 1)
