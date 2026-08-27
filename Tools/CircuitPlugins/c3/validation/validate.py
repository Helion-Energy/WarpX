#!/usr/bin/env python3
"""Validation gate for the c3 external-circuit plugin.

Runs the C++ plugin (through the real dlopen ABI path, with perturbed
non-accepted evaluations exercising restore-and-re-advance) side by side
with the Python reference stepper on bit-identical inputs:

  (a) the reference shot's open-loop pre-roll + machine [0, +12 us], eps = 0
  (b) same window, sinusoidal eps on coil_F09 / coil_A07 / coil_C05
  (c) lock-ring config: machine-0 stage swap -- locked fluxes + station
      sums at arming, post-swap R = 0 linked-flux conservation
  (d) checkpoint round trip mid-window (post-swap): restart trajectory
      and end-state checkpoint must match bit for bit
  (e) alternate-case circuits (reverse-recovery + segment-3/5 coverage),
      open loop machine [0, +100 us]

Bar for (a), (b), (c): per-coil max |ds| / max |s_ref| <= 1e-10.
Bar for (d): bitwise.
Bar for (e): median <= 1e-10, max <= 1e-8. The alternate-case compression banks'
reverse-recovery turn-off sits on the ring current's zero crossing, so
the trajectory amplifies solver pivot-order roundoff through the switch
timing: the measured intrinsic sensitivity -- the SAME python stepper
with scipy dense lu_factor instead of sparse splu -- is 1.1e-9 on the
same C coils, the class the C++ port lands in (and the port is closer
to python-dense than the two python variants are to each other).

Run (deck venv; CPU only):
    HETOOLS_USE_GPU=0 ~/.env/warpx-mhd/bin/python3 validate.py \
        --workdir /path/to/work [--cases a,b,c,d,e] [--skip-export]

Any python interpreter works for this driver; the reference/exporter
subprocesses use --python (default ~/.env/warpx-mhd/bin/python3).
"""

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PLUGIN_DIR = HERE.parent
VENV_PY = str(pathlib.Path.home() / ".env/warpx-mhd/bin/python3")

# window conventions
DT_WARPX = 5.0e-7          # deck WarpX step (subcycle 10 at dt_circ 50 ns)
T_END_REF_SHOT = 14.5e-6        # machine +12 us at shift 2.5 us
CKPT_T = 8.5e-6            # machine +6 us (post-swap)
T_END_ALT_CASE = 100.0e-6       # shift 0: sim == machine
BAR = 1.0e-10
# case (e) max bar: the C-bank rr turn-off rides the ring current-zero
# crossing and amplifies LU pivot-order roundoff into switch-timing
# jitter; python-sparse vs python-dense measures 1.1e-9 intrinsic (see
# the module docstring), so the max bar is the 1e-8 sensitivity class
BAR_E_MAX = 1.0e-8

# case-b eps program: (manifest name, amplitude [V/turn], frequency [Hz])
EPS_PORTS = (("coil_F09", 150.0, 1.5e5),
             ("coil_A07", 100.0, 0.9e5),
             ("coil_C05", 80.0, 0.6e5))


def run(cmd, env_extra=None, log=None):
    env = dict(os.environ, HETOOLS_USE_GPU="0")
    if env_extra:
        env.update(env_extra)
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    with open(log, "w") if log else open(os.devnull, "w") as lf:
        subprocess.run([str(c) for c in cmd], check=True, env=env,
                       stdout=lf, stderr=subprocess.STDOUT)


def read_csv(path):
    """(t, values) from a hexfloat CSV (rows: t v0 v1 ...)."""
    t, rows = [], []
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        toks = line.split()
        t.append(float.fromhex(toks[0]))
        rows.append([float.fromhex(x) for x in toks[1:]])
    return t, rows


def traj_metric(csv_cpp, csv_ref):
    """Per-coil max |delta| / max |ref| over the window; returns
    (worst, median, n_coils)."""
    t_a, a = read_csv(csv_cpp)
    t_b, b = read_csv(csv_ref)
    assert len(a) == len(b) and len(a[0]) == len(b[0]), \
        (len(a), len(b), "row/col mismatch")
    n = len(a[0])
    rel = []
    for k in range(n):
        dmax = max(abs(ra[k] - rb[k]) for ra, rb in zip(a, b))
        pk = max(abs(rb[k]) for rb in b)
        rel.append(dmax / pk if pk > 0.0 else dmax)
    rel_sorted = sorted(rel)
    return max(rel), rel_sorted[n // 2], n


def make_eps_table(path, n_steps):
    with open(path, "w") as f:
        f.write("ports " + " ".join(nm for nm, _, _ in EPS_PORTS) + "\n")
        for i in range(n_steps):
            t0 = i * DT_WARPX
            vals = [amp * math.sin(2.0 * math.pi * fr * t0)
                    for _, amp, fr in EPS_PORTS]
            f.write(" ".join(float(v).hex() for v in vals) + "\n")


def config_string(case_dir, preroll=True):
    # inline form (exercises the non-INI config path on restart legs)
    cfg = {}
    for line in (case_dir / "config.ini").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    if not preroll:
        cfg["preroll"] = "0"
    return ",".join(f"{k}={v}" for k, v in cfg.items())


def check(tag, ok, detail):
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}: {detail}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--cases", default="a,b,c,d,e")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--reuse-csv", action="store_true",
                    help="re-check existing trajectory CSVs without "
                         "re-running the engines")
    ap.add_argument("--python", default=VENV_PY)
    args = ap.parse_args()
    wd = pathlib.Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    cases = set(args.cases.split(","))
    harness = PLUGIN_DIR / "build" / "c3_harness"
    plugin = PLUGIN_DIR / "build" / "libc3circuit.so"
    assert harness.is_file() and plugin.is_file(), \
        "build the plugin first: cmake -S . -B build && cmake --build build"

    if not args.skip_export:
        need = ",".join(sorted({"a" if c in "ab" else "c" if c in "cd"
                                else "e" for c in cases}))
        run([args.python, HERE / "export_case.py", "--workdir", wd,
             "--cases", need], log=wd / "export.log")

    n_ref_shot_steps = round(T_END_REF_SHOT / DT_WARPX)
    results = {}

    def compare_case(tag, case_dir, t_end, eps=None, lock=False):
        cs = config_string(case_dir)
        cpp = wd / f"cpp_{tag}.csv"
        ref = wd / f"ref_{tag}.csv"
        if args.reuse_csv and cpp.is_file() and ref.is_file():
            return traj_metric(cpp, ref)
        hcmd = [harness, "--mode", "traj", "--plugin", plugin,
                "--config", cs, "--manifest", case_dir / "manifest.txt",
                "--t-end", t_end, "--dt", DT_WARPX, "--out", cpp]
        rcmd = [args.python, HERE / "reference_run.py", "--case", case_dir,
                "--t-end", t_end, "--dt", DT_WARPX, "--out", ref]
        if eps:
            hcmd += ["--eps", eps]
            rcmd += ["--eps", eps]
        if lock:
            # lock diagnostics need direct mode: run the ABI (dlopen) leg
            # for the scale trajectories, then a direct leg for the lock
            # currents/fluxes + the arming record
            run(hcmd, log=wd / f"harness_{tag}.log")
            hcmd2 = [harness, "--mode", "traj", "--config", cs,
                     "--manifest", case_dir / "manifest.txt",
                     "--t-end", t_end, "--dt", DT_WARPX,
                     "--out", wd / f"cpp_{tag}_direct.csv",
                     "--lock-out", wd / f"cpp_{tag}_lock.csv"]
            run(hcmd2, log=wd / f"harness_{tag}_direct.log")
            rcmd += ["--lock-out", wd / f"ref_{tag}_lock.csv",
                     "--arming-out", wd / f"ref_{tag}_arming.json"]
        else:
            run(hcmd, log=wd / f"harness_{tag}.log")
        run(rcmd, log=wd / f"reference_{tag}.log")
        return traj_metric(cpp, ref)

    if "a" in cases:
        print("== case (a): the reference shot's open loop, machine [0, +12 us] ==")
        worst, med, n = compare_case("a", wd / "case_a", T_END_REF_SHOT)
        results["a"] = check("a", worst <= BAR,
                             f"max rel {worst:.3e}, median {med:.3e} "
                             f"({n} coils, bar {BAR:.0e})")

    if "b" in cases:
        print("== case (b): eps-driven (3 sinusoidal ports) ==")
        eps_path = wd / "eps_b.txt"
        make_eps_table(eps_path, n_ref_shot_steps)
        worst, med, n = compare_case("b", wd / "case_a", T_END_REF_SHOT,
                                     eps=eps_path)
        results["b"] = check("b", worst <= BAR,
                             f"max rel {worst:.3e}, median {med:.3e} "
                             f"({n} coils)")

    if "c" in cases:
        print("== case (c): machine-0 lock swap + conservation ==")
        worst, med, n = compare_case("c", wd / "case_c", T_END_REF_SHOT,
                                     lock=True)
        ok = check("c-traj", worst <= BAR,
                   f"max rel {worst:.3e}, median {med:.3e} ({n} coils)")
        # the dlopen and direct legs run the same code: bit-identical
        same = ((wd / "cpp_c.csv").read_bytes()
                == (wd / "cpp_c_direct.csv").read_bytes())
        ok &= check("c-dlopen", same,
                    "dlopen vs direct instantiation scale CSVs bit-match")
        # arming: parse the C++ "# armed" record from the direct-leg lock
        # CSV, compare each ring's locked flux + station sums to python's
        arming = json.loads((wd / "ref_c_arming.json").read_text())
        locked_ref = {k: float.fromhex(v)
                      for k, v in arming["locked"].items()}
        armed_line = next(
            ln for ln in (wd / "cpp_c_lock.csv").read_text().splitlines()
            if ln.startswith("# armed"))
        locked_cpp = {}
        for tok in armed_line.split()[3:]:
            nm, v = tok.split("=")
            locked_cpp[nm] = float.fromhex(v)
        worst_arm = max(abs(locked_cpp[k] - locked_ref[k])
                        / max(abs(locked_ref[k]), 1e-30)
                        for k in locked_ref)
        st_ref, st_cpp = {}, {}
        for k in locked_ref:
            st_ref[k[3]] = st_ref.get(k[3], 0.0) + locked_ref[k]
            st_cpp[k[3]] = st_cpp.get(k[3], 0.0) + locked_cpp[k]
        st_txt = ", ".join(
            f"{s}: {st_cpp[s] * 1e3:+.4f} vs {st_ref[s] * 1e3:+.4f} mWb"
            for s in sorted(st_ref))
        ok &= check("c-arming", worst_arm <= BAR,
                    f"max rel {worst_arm:.3e}; station sums {st_txt}")
        # post-swap conservation, both sides independently: R = 0 rings
        # hold their linked flux to BE round-off accumulation (~N eps)
        cons = {}
        for side in ("cpp", "ref"):
            _, rows = read_csv(wd / f"{side}_c_lock.csv")
            fluxes = [[r[2 * j + 1] for r in rows if any(r)]
                      for j in range(len(locked_ref))]
            drift = 0.0
            for j, nm in enumerate(sorted(locked_ref)):
                lam = [f for f in fluxes[j] if f != 0.0]
                if not lam:
                    continue
                lam0 = locked_ref[nm] if side == "ref" else locked_cpp[nm]
                drift = max(drift, max(abs(v - lam0) for v in lam)
                            / max(abs(lam0), 1e-30))
            cons[side] = drift
        ok &= check("c-conserve",
                    cons["cpp"] <= 1e-11 and cons["ref"] <= 1e-11,
                    f"post-swap flux drift cpp {cons['cpp']:.3e}, "
                    f"ref {cons['ref']:.3e} (BE round-off class)")
        results["c"] = ok

    if "d" in cases:
        print("== case (d): checkpoint round trip (post-swap, +6 us) ==")
        case_dir = wd / "case_c"
        cs = config_string(case_dir)
        cs2 = config_string(case_dir, preroll=False)
        ckpt = wd / "ckpt_d"
        run([harness, "--mode", "ckpt", "--plugin", plugin,
             "--config", cs, "--config2", cs2,
             "--manifest", case_dir / "manifest.txt",
             "--t-end", T_END_REF_SHOT, "--dt", DT_WARPX,
             "--ckpt-t", CKPT_T, "--ckpt-dir", ckpt,
             "--out", wd / "cpp_d_straight.csv",
             "--out2", wd / "cpp_d_restart.csv"],
            log=wd / "harness_d.log")
        t_s, rows_s = read_csv(wd / "cpp_d_straight.csv")
        t_r, rows_r = read_csv(wd / "cpp_d_restart.csv")
        tail = [r for t, r in zip(t_s, rows_s) if t > CKPT_T]
        bit_traj = tail == rows_r
        end_a = (ckpt / "end_a" / "c3_external_circuit.dat").read_bytes()
        end_b = (ckpt / "end_b" / "c3_external_circuit.dat").read_bytes()
        bit_end = end_a == end_b
        results["d"] = check(
            "d", bit_traj and bit_end,
            f"restart rows bit-match {bit_traj} "
            f"({len(rows_r)} rows), end checkpoints bit-match {bit_end}")

    if "e" in cases:
        print("== case (e): the alternate reference case (rr + seg3/5), machine [0, +100 us]"
              " ==")
        worst, med, n = compare_case("e", wd / "case_e", T_END_ALT_CASE)
        results["e"] = check(
            "e", worst <= BAR_E_MAX and med <= BAR,
            f"max rel {worst:.3e} (rr-timing sensitivity bar "
            f"{BAR_E_MAX:.0e}), median {med:.3e} ({n} coils)")

    print("\n== summary ==")
    for k in sorted(results):
        print(f"  ({k}) {'PASS' if results[k] else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
