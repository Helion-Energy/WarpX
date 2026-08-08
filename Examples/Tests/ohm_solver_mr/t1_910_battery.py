#!/usr/bin/env python3
"""Serial battery driver for the T1.9 (Mach-cone angle) and T1.10
(cold-beam finite-grid instability at the seam) hybrid-PIC MR tests.

Same conventions as t1_34_battery.py: each case gets a directory under
t1_runs/, plotfiles are reduced (T1.10: x-averaged lineouts via
t1_pltreader; T1.9: full 2D By/rho maps per level -> t1_maps.npz) and
deleted, a .done/.failed marker makes the battery resumable, and a run
that goes non-finite is a RESULT, not an error.  The measured s/step
(WarpX "Total Time" / steps) is recorded in the marker.

Run two driver processes with disjoint --only selections for
2-concurrent execution (resource budget: T1.9 np <= 24, T1.10 np <= 16).

Usage:
    python t1_910_battery.py --list
    python t1_910_battery.py --only t19_mr --nprocs 24
    python t1_910_battery.py --only t110 --nprocs 16
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "t1_runs")
sys.path.insert(0, HERE)
import t1_pltreader  # noqa: E402

ETA_BARE = "1.0e-10"
ETA_FLOOR = "1.0e-7"  # the long-horizon stabilization floor (control leg only)

T19_BASE = ["--dt-fac", "0.005", "--steps", "1750", "--plot-int", "10"]
T110_BASE = [
    "--dt-fac",
    "0.012",
    "--steps",
    "2500",
    "--plot-int",
    "5",
    "--v0-fac",
    "2.0",
]


def cases():
    """Return [(name, deck_script, args, reduction, b0_for_monitor)]."""
    cs = []
    # ---------------- T1.9 ------------------------------------------
    cs.append(
        ("t19_mr", "t1_9_machcone.py", T19_BASE + ["--max-level", "1"], "maps2d", 0.25)
    )
    cs.append(
        ("t19_ctl", "t1_9_machcone.py", T19_BASE + ["--max-level", "0"], "maps2d", 0.25)
    )
    # ---------------- T1.10 -----------------------------------------
    for name, extra in [
        ("t110_mr", ["--max-level", "1", "--eta", ETA_BARE]),
        ("t110_ctl", ["--max-level", "0", "--eta", ETA_BARE]),
        ("t110_fine", ["--max-level", "0", "--resolution", "fine", "--eta", ETA_BARE]),
        ("t110_mr_eta7", ["--max-level", "1", "--eta", ETA_FLOOR]),
        ("t110_ctl_eta7", ["--max-level", "0", "--eta", ETA_FLOOR]),
    ]:
        cs.append((name, "t1_10_beam.py", T110_BASE + extra, "lineouts", 0.25))
    return cs


# ----------------------------------------------------------------------
# T1.9 reduction: full 2D By/rho maps per level
# ----------------------------------------------------------------------
MAP_FIELDS = ["Bx", "By", "Bz", "Ex", "Ey", "Ez", "rho"]


def convert_run_maps(run_dir, b0=0.25, out_name="t1_maps.npz"):
    plts = sorted(glob.glob(os.path.join(run_dir, "diags", "diag1??????")))
    assert plts, f"no plotfiles under {run_dir}"
    t_list = []
    lev = [dict(by=[], rho=[], amax=[]) for _ in range(2)]
    nlev = None
    lo1 = None
    for p in plts:
        time, levels = t1_pltreader.read_plotfile(p, MAP_FIELDS)
        if nlev is None:
            nlev = len(levels)
            if nlev > 1:
                lo1 = levels[1]["lo"]
        t_list.append(time)
        for iL, ld in enumerate(levels):
            a = ld["arr"]
            lev[iL]["by"].append(a["By"].astype(np.float32))
            lev[iL]["rho"].append(a["rho"].astype(np.float32))
            lev[iL]["amax"].append(
                [
                    np.abs(a["Ex"]).max(),
                    np.abs(a["Ey"]).max(),
                    np.abs(a["Ez"]).max(),
                    np.abs(a["By"] - b0).max(),
                    np.abs(a["Bx"]).max(),
                    np.abs(a["Bz"]).max(),
                ]
            )
    out = dict(
        t=np.array(t_list),
        by0=np.array(lev[0]["by"]),
        rho0=np.array(lev[0]["rho"]),
        amax0=np.array(lev[0]["amax"], dtype=np.float32),
    )
    if nlev > 1:
        out.update(
            by1=np.array(lev[1]["by"]),
            rho1=np.array(lev[1]["rho"]),
            amax1=np.array(lev[1]["amax"], dtype=np.float32),
            lo1=np.array(lo1),
        )
    npz_path = os.path.join(run_dir, out_name)
    np.savez_compressed(npz_path, **out)
    return npz_path


def check_finite_maps(npz_path):
    d = np.load(npz_path)
    for key in ("amax0", "amax1"):
        if key not in d:
            continue
        bad = ~np.isfinite(d[key]).all(axis=1)
        if bad.any():
            return False, int(np.argmax(bad))
    return True, -1


def check_finite_lineouts(npz_path):
    d = np.load(npz_path)
    for key in ("bx0", "by0", "bz0", "bx1", "by1", "bz1"):
        if key not in d:
            continue
        bad = ~np.isfinite(d[key]).all(axis=1)
        if bad.any():
            return False, int(np.argmax(bad))
    return True, -1


def s_per_step(log_path, steps):
    """Parse WarpX 'Total Time' from run.log -> seconds per step."""
    if not os.path.exists(log_path):
        return None
    txt = open(log_path, errors="replace").read()
    m = re.findall(r"Total Time\s*:\s*([0-9.eE+-]+)", txt)
    if not m:
        return None
    return float(m[-1]) / steps


def run_case(name, script, args, reduction, b0, nprocs, keep_plt):
    out_dir = os.path.join(RUNS, name)
    done = os.path.join(out_dir, ".done")
    failed = os.path.join(out_dir, ".failed")
    if os.path.exists(done) or os.path.exists(failed):
        print(f"[battery910] {name}: already finished, skipping")
        return
    cmd = [
        sys.executable,
        os.path.join(HERE, script),
        "--name",
        name,
        "--out",
        out_dir,
        "--nprocs",
        str(nprocs),
    ] + args
    rc = subprocess.run(cmd).returncode
    log = os.path.join(out_dir, "run.log")
    nan_in_log = False
    if os.path.exists(log):
        with open(log, "rb") as f:
            tail = f.read()[-4000:].decode(errors="replace").lower()
        nan_in_log = ("nan" in tail) or ("erroneous arithmetic" in tail)
    npz = None
    ok, bad_step = False, -1
    if glob.glob(os.path.join(out_dir, "diags", "diag1??????")):
        if reduction == "maps2d":
            npz = convert_run_maps(out_dir, b0=b0)
            ok, bad_step = check_finite_maps(npz)
        else:
            npz = t1_pltreader.convert_run(out_dir, b0=b0)
            ok, bad_step = check_finite_lineouts(npz)
    steps = None
    try:
        steps = json.load(open(os.path.join(out_dir, "params.json")))["steps"]
    except Exception:
        pass
    sps = s_per_step(log, steps) if steps else None
    status = dict(
        name=name,
        rc=rc,
        nan_in_log=nan_in_log,
        finite=ok,
        first_bad_step=bad_step,
        npz=npz,
        s_per_step=sps,
        nprocs=nprocs,
    )
    marker = done if (rc == 0 and ok and not nan_in_log) else failed
    with open(marker, "w") as f:
        json.dump(status, f, indent=1)
    print(
        f"[battery910] {name}: rc={rc} finite={ok} "
        f"(first bad output {bad_step}) nan_in_log={nan_in_log} "
        f"s/step={sps if sps is None else round(sps, 4)} "
        f"-> {os.path.basename(marker)}"
    )
    if npz and not keep_plt:
        for p in glob.glob(os.path.join(out_dir, "diags", "diag1??????")):
            shutil.rmtree(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only", default=None, help="comma-separated substrings selecting case names"
    )
    p.add_argument("--list", action="store_true")
    p.add_argument("--nprocs", type=int, default=16)
    p.add_argument("--keep-plt", action="store_true")
    a = p.parse_args()

    cs = cases()
    if a.only:
        keys = a.only.split(",")
        cs = [c for c in cs if any(k in c[0] for k in keys)]
    if a.list:
        for n, s, ar, red, b0 in cs:
            print(n, s, " ".join(ar), f"({red}, monitor b0={b0})")
        return
    os.makedirs(RUNS, exist_ok=True)
    for n, s, ar, red, b0 in cs:
        run_case(n, s, ar, red, b0, a.nprocs, a.keep_plt)


if __name__ == "__main__":
    main()
