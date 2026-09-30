#!/usr/bin/env python3
"""Native common-state baseline/guard x axis-gate attribution; no copied solver."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

QE = 1.602176634e-19
MU0 = 1.2566370612685e-6  # exact value in Source/ablastr/constant.H
ETA = 1e-5
GATE_KEYS = {
    "hybrid_pic_model.holmstrom_vacuum_region",
    "hybrid_pic_model.holmstrom_transition_width",
    "hybrid_pic_model.holmstrom_axis_radius",
    "hybrid_pic_model.holmstrom_axis_rolloff",
}
POLICY = {
    "declared_before_matrix": True,
    "raw_gate_error_bound_relative_to_max_raw_E": 2e-11,
    "unchanged_fields": "bitwise maps within each native runtime",
    "unchanged_conduction": "bitwise T0,T(h),T(h/2),T(h/4) within each runtime",
    "inactive_control": "20*n_floor, tanh saturation at double precision",
    "source_receiver_probes": "Separate native gates; every nonzero return preserved, never a hidden source correction",
    "scope": "Manufactured fixed state, one rank/box; no evolved FRC or physical moving-ion viscosity certificate",
}


def read_map(path):
    with path.open("rb") as f:
        nx, nz, ir, iz = np.fromfile(f, dtype=np.int64, count=4)
        a = np.fromfile(f, dtype=np.float64).reshape((nx, nz), order="F")
    return a, (int(ir), int(iz))


def params(path):
    p = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            p[k.strip()] = v.strip()
    return p


def load(path):
    lines = path.joinpath("run.log").read_text().splitlines()
    rows = [
        json.loads(x.split("HOLMSTROM_AXIS ", 1)[1])
        for x in lines
        if x.startswith("HOLMSTROM_AXIS ")
    ]
    assert len(rows) == 1, f"Missing native result in {path}"
    maps = {p.stem: read_map(p) for p in path.glob("*.bin")}
    assert all(np.isfinite(a).all() for a, _ in maps.values())
    return {"meta": rows[0], "maps": maps, "params": params(path / "warpx_used_inputs")}


def coords_volume(a, stagger, meta):
    nr, nz = meta["nr"], meta["nz"]
    radius, length = meta["domain_radius_m"], meta["domain_length_m"]
    dr, dz = radius / nr, length / nz
    r = (np.arange(a.shape[0]) + (0 if stagger[0] else 0.5)) * dr
    v = np.pi * (np.minimum(r + dr / 2, radius) ** 2 - np.maximum(r - dr / 2, 0) ** 2)
    v = np.broadcast_to(v[:, None] * dz, a.shape).copy()
    if stagger[1]:
        v[:, -1] = 0  # periodic duplicate, native sum_unique convention
    return r, v


def gate_from_native_rho(rho, c, meta):
    raw = (
        (rho[:-1] + rho[1:]) / 2
        if c == 0
        else (rho[:, :-1] + rho[:, 1:]) / 2
        if c == 2
        else rho
    )
    dr = meta["domain_radius_m"] / meta["nr"]
    r = (np.arange(raw.shape[0]) + (0.5 if c == 0 else 0)) * dr
    radial = 0.5 * (1 - np.tanh((r - 0.006) / 0.003))
    density = 0.5 * (
        1 + np.tanh((raw / QE - meta["n_floor"]) / (0.5 * meta["n_floor"]))
    )
    return 1 - radial[:, None] * (1 - density), radial, raw / QE


def powers(run, field_name):
    power = {"J_W": 0.0, "Ji_W": 0.0, "Je_W": 0.0}
    for c in range(3):
        a, stagger = run["maps"][field_name + str(c)]
        _, v = coords_volume(a, stagger, run["meta"])
        jp = run["maps"]["J" + str(c)][0]
        ji = run["maps"]["Ji" + str(c)][0]
        power["J_W"] += float(np.sum(v * jp * a))
        power["Ji_W"] += float(np.sum(v * ji * a))
        power["Je_W"] += float(np.sum(v * (jp - ji) * a))
    return power


def state_metrics(run):
    m = run["meta"]
    maps = run["maps"]
    magnetic = 0.0
    magnetic_rate = 0.0
    for c in range(3):
        b, stagger = maps["B" + str(c)]
        _, v = coords_volume(b, stagger, m)
        magnetic += float(np.sum(v * b**2) / (2 * MU0))
        magnetic_rate += float(np.sum(v * b * maps["dBdt" + str(c)][0]) / MU0)
    applied = powers(run, "applied_E")
    result = {
        "native": m,
        "magnetic_energy_J": magnetic,
        "magnetic_rate_W": magnetic_rate,
        "applied": applied,
        "magnetic_rate_plus_JE_W": magnetic_rate + applied["J_W"],
        "raw": powers(run, "raw_E"),
        "hyper": powers(run, "hyper_E"),
        "visc": powers(run, "visc_E"),
    }
    h = m["conduction_dt"]
    t0, t1, t2, t4 = [maps[x][0] for x in ("T0", "T1", "T2", "T3")]
    rhs = (4 * (t2 - t0) - (t1 - t0)) / h
    rhs_half = (4 * (t4 - t0) - (t2 - t0)) / (h / 2)
    result["rhs_extrapolation_l2_relative"] = float(
        np.linalg.norm(rhs - rhs_half) / np.linalg.norm(rhs_half)
    )
    result["rhs_axis4_max_eV_us"] = float(
        np.max(np.abs(rhs[:4])) * 1.380649e-23 / QE * 1e-6
    )
    return result, rhs


def compare(on, off):
    meta = on["meta"]
    changed_inputs = {
        k: [off["params"].get(k), on["params"].get(k)]
        for k in set(on["params"]) | set(off["params"])
        if on["params"].get(k) != off["params"].get(k)
    }
    assert set(changed_inputs) == GATE_KEYS, changed_inputs
    checks = {}
    unchanged = ["T0", "rho", "pedestal", "Pe", "Qnu", "T1", "T2", "T3"]
    if not meta["active"]:
        unchanged += ["applied_E" + str(c) for c in range(3)]
    unchanged += [
        stem + str(c)
        for stem in ("B", "J", "Ji", "visc_E", "hyper_E")
        for c in range(3)
    ]
    for name in unchanged:
        checks[name] = bool(np.array_equal(on["maps"][name][0], off["maps"][name][0]))
    components = []
    bands = {
        k: {"J_delta_W": 0.0, "Ji_delta_W": 0.0, "activation_volume_sum_m3": 0.0}
        for k in ("all", "first4_sites", "r_le_6mm", "r_6_9mm", "r_ge_15mm")
    }
    maxraw = max(
        float(np.max(np.abs(off["maps"]["raw_E" + str(c)][0]))) for c in range(3)
    )
    threshold = POLICY["raw_gate_error_bound_relative_to_max_raw_E"] * maxraw
    for c in range(3):
        raw_on = on["maps"]["raw_E" + str(c)][0]
        raw_off, st = off["maps"]["raw_E" + str(c)]
        gate, mask, ne = gate_from_native_rho(off["maps"]["rho"][0], c, meta)
        r, v = coords_volume(raw_off, st, meta)
        j = off["maps"]["J" + str(c)][0]
        ji = off["maps"]["Ji" + str(c)][0]
        diss = (
            ETA * j
            + off["maps"]["hyper_E" + str(c)][0]
            + off["maps"]["visc_E" + str(c)][0]
        )
        expected = diss + gate * (raw_off - diss)
        err = float(np.max(np.abs(raw_on - expected)))
        perterm = {}
        termfields = []
        for arm in (on, off):
            full = arm["maps"]["raw_E" + str(c)][0]
            hall = full - arm["maps"]["nohall_E" + str(c)][0]
            pressure = full - arm["maps"]["nope_E" + str(c)][0]
            motion = full - hall - pressure - diss
            termfields.append({"hall": hall, "pressure": pressure, "motion": motion})
        for term in ("hall", "pressure", "motion"):
            er = float(np.max(np.abs(termfields[0][term] - gate * termfields[1][term])))
            perterm[term] = {
                "max_gate_error_V_m": er,
                "off_max_V_m": float(np.max(np.abs(termfields[1][term]))),
                "pass": er <= threshold,
            }
        applied_delta = (
            on["maps"]["applied_E" + str(c)][0] - off["maps"]["applied_E" + str(c)][0]
        )
        masks = {
            "all": np.ones_like(r, dtype=bool),
            "first4_sites": np.arange(len(r)) < 4,
            "r_le_6mm": r <= 0.006,
            "r_6_9mm": (r > 0.006) & (r <= 0.009),
            "r_ge_15mm": r >= 0.015,
        }
        for name, band in masks.items():
            bands[name]["J_delta_W"] += float(
                np.sum(v[band] * j[band] * applied_delta[band])
            )
            bands[name]["Ji_delta_W"] += float(
                np.sum(v[band] * ji[band] * applied_delta[band])
            )
            bands[name]["activation_volume_sum_m3"] += float(
                np.sum(v[band] * (1 - gate[band]))
            )
        components.append(
            {
                "component": c,
                "min_gate": float(gate.min()),
                "max_removed_fraction": float((1 - gate).max()),
                "axis_mask_at_first_site": float(mask[0]),
                "min_ne_over_floor": float(ne.min() / meta["n_floor"]),
                "max_gate_error_V_m": err,
                "bound_V_m": threshold,
                "raw_pass": err <= threshold,
                "term_checks": perterm,
                "applied_delta_max_V_m": float(np.max(np.abs(applied_delta))),
                "tail_removed_fraction_r_ge_15mm": float(
                    np.max((1 - gate)[r >= 0.015])
                ),
            }
        )
    return {
        "input_diff": changed_inputs,
        "unchanged": checks,
        "components": components,
        "bands": bands,
        "pass": all(checks.values())
        and all(
            x["raw_pass"] and all(t["pass"] for t in x["term_checks"].values())
            for x in components
        ),
    }


def run_case(executable, inputs, root, active, on, size, source_probe="none"):
    root.mkdir(parents=True, exist_ok=False)
    changes = {
        "amr.n_cell": f"{size} {size}",
        "amr.max_grid_size": str(2 * size),
        "holmstrom_test.active": str(int(active)),
        "hybrid_pic_model.holmstrom_vacuum_region": str(int(on)),
        "hybrid_pic_model.holmstrom_transition_width": ".5" if on else "0",
        "hybrid_pic_model.holmstrom_axis_radius": ".006" if on else "0",
        "hybrid_pic_model.holmstrom_axis_rolloff": ".003" if on else "0",
    }
    text = inputs.read_text()
    for k, v in changes.items():
        text = re.sub(r"(?m)^" + re.escape(k) + r"\s*=.*$", f"{k} = {v}", text)
    text += f"\nholmstrom_test.source_probe = {source_probe}\n"
    (root / "inputs").write_text(text)
    cmd = [str(executable), "inputs"]
    (root / "COMMAND.json").write_text(json.dumps(cmd, indent=2) + "\n")
    start = time.monotonic()
    with (root / "run.log").open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            check=False,
        )
    outcome = {"returncode": proc.returncode, "elapsed_s": time.monotonic() - start}
    (root / "PROCESS.json").write_text(json.dumps(outcome, indent=2) + "\n")
    return outcome


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    out_group = p.add_mutually_exclusive_group(required=True)
    out_group.add_argument("--output", type=Path)
    out_group.add_argument("--output-parent", type=Path)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--source-probes", action="store_true")
    args = p.parse_args()
    if args.output_parent:
        args.output_parent.mkdir(parents=True, exist_ok=True)
        root = Path(
            tempfile.mkdtemp(prefix="matrix-", dir=args.output_parent)
        ).resolve()
    else:
        root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "POLICY.json").write_text(json.dumps(POLICY, indent=2) + "\n")
    result = {
        "executables": {},
        "processes": {},
        "states": {},
        "on_off": {},
        "baseline_guard": {},
    }
    inputs = Path(__file__).with_name("inputs_test_rz_holmstrom_axis")
    if not args.analyze_only:
        for runtime, exe in (("baseline", args.baseline), ("guard", args.candidate)):
            exe = exe.resolve()
            result["executables"][runtime] = {
                "path": str(exe),
                "sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
            }
            for active in (False, True):
                for on in (False, True):
                    label = f"{runtime}-{'active' if active else 'inactive'}-{'on' if on else 'off'}"
                    result["processes"][label] = run_case(
                        exe, inputs, root / label, active, on, args.size
                    )
        (root / "LAUNCHES.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        result.update(json.loads((root / "LAUNCHES.json").read_text()))
    states = {}
    rhss = {}
    for label, proc in result["processes"].items():
        assert proc["returncode"] == 0, (label, proc)
        states[label] = load(root / label)
        result["states"][label], rhss[label] = state_metrics(states[label])
    for runtime in ("baseline", "guard"):
        for active in ("inactive", "active"):
            label = f"{runtime}-{active}"
            result["on_off"][label] = compare(
                states[label + "-on"], states[label + "-off"]
            )
    for active in ("inactive", "active"):
        for on in ("off", "on"):
            label = f"{active}-{on}"
            a = states["baseline-" + label]
            b = states["guard-" + label]
            rhs_a = rhss["baseline-" + label]
            rhs_b = rhss["guard-" + label]
            result["baseline_guard"][label] = {
                "RHS_difference_l2_relative": float(
                    np.linalg.norm(rhs_b - rhs_a) / np.linalg.norm(rhs_a)
                ),
                "E_bitwise_equal": all(
                    np.array_equal(
                        a["maps"]["applied_E" + str(c)][0],
                        b["maps"]["applied_E" + str(c)][0],
                    )
                    for c in range(3)
                ),
                "axis4_RHS_difference_max_eV_us": float(
                    np.max(np.abs((rhs_b - rhs_a)[:4])) * 1.380649e-23 / QE * 1e-6
                ),
            }
    result["pass"] = all(x["pass"] for x in result["on_off"].values()) and all(
        x["E_bitwise_equal"] for x in result["baseline_guard"].values()
    )
    (root / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.source_probes and not args.analyze_only:
        probes = {}
        for active in (False, True):
            for on in (False, True):
                label = f"source-visc-{'active' if active else 'inactive'}-{'on' if on else 'off'}"
                probes[label] = run_case(
                    args.candidate.resolve(),
                    inputs,
                    root / label,
                    active,
                    on,
                    args.size,
                    "visc",
                )
        (root / "SOURCE_PROBES.json").write_text(json.dumps(probes, indent=2) + "\n")
    print(
        json.dumps(
            {
                "pass": result["pass"],
                "on_off": {k: v["pass"] for k, v in result["on_off"].items()},
                "baseline_guard": result["baseline_guard"],
            },
            indent=2,
        )
    )
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
