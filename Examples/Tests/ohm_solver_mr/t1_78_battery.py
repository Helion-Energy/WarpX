#!/usr/bin/env python3
"""Serial battery driver for the T1.7 (low-density seam) and T1.8
(late-time soak) hybrid-PIC MR tests.

Same conventions as t1_run_battery.py / t1_34_battery.py: each case gets a
directory under t1_runs/, plotfiles are reduced to an npz and deleted, a
.done/.failed marker makes the battery resumable, and a run that goes
non-finite is a RESULT (seam instability), not an error.

T1.7 arms (deck t1_7_lowden.py, reducer t1_7_reduce.py): quiet plasma with
a density gap down to the solver floor (n_floor = 0.05 n0, holmstrom
vacuum ON), ramp-position scan with the patch seams fixed at 48/80 l_i:
  inramp : gap [48, 64] l_i -- ramp midpoint exactly on the lower seam
  floor  : gap [40, 56] l_i -- lower seam dead-center in the floor region
  above  : gap [56, 72] l_i -- both ramps inside the patch, seams at n0
  nogap  : uniform n0 (same solver settings) -- attribution baseline
eta = 1e-7 legs run 667 steps (8 t_ci); the eta = 3e-8 threshold legs run
1300 steps (15.6 t_ci -- the measured uniform-density quiet-seam stability
horizon), plus a holmstrom = 0 attribution pair.

T1.8 arms (deck t1_3_alfven.py, reducer t1_pltreader.py): the T1.3
stabilized configuration (eta = 1e-7 Ohm m, dt = 0.012 t_ci,
refine_plasma_init = 1) extended to 60000 steps = 720 t_ci = 36.2 patch
transits, MR + uniform-coarse control.

Usage:
    python t1_78_battery.py --list
    python t1_78_battery.py [--only substr1,substr2] [--nprocs 8]
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "t1_runs")
sys.path.insert(0, HERE)
import t1_7_reduce  # noqa: E402
import t1_pltreader  # noqa: E402

# T1.7 production: eta = 1e-6 Ohm m is the MEASURED stability floor for
# fine-resolution floor-density cells under MR (envelope mapping in
# t1_7_8_results.md: 3e-7 + eta_h 1.0 still blows up, 1e-6 alone is
# stable; the planned 1e-7/3e-8 legs are unviable in any fine-containing
# arm and survive only as threshold-documentation legs).
T17_E6 = ["--eta", "1.0e-6", "--steps", "667", "--plot-int", "5"]
T17_E7 = ["--eta", "1.0e-7", "--steps", "667", "--plot-int", "5"]
GAP = {
    "inramp": ["--gap-za-li", "48", "--gap-zb-li", "64"],
    "floor": ["--gap-za-li", "40", "--gap-zb-li", "56"],
    "above": ["--gap-za-li", "56", "--gap-zb-li", "72"],
    "nogap": [],
}

# T1.8: T1.3 stabilized configuration + refine_plasma_init (mandatory
# amendment; byte-identical no-op on the max_level=0 control).
T18 = [
    "--eta",
    "1.0e-7",
    "--dt-fac",
    "0.012",
    "--steps",
    "60000",
    "--plot-int",
    "50",
    "--extra-inputs",
    "warpx.refine_plasma_init = 1",
]


def cases():
    """Return [(name, deck_script, args, b0_for_monitor, reducer)]."""
    cs = []
    # ---------------- T1.7 ------------------------------------------
    # main stabilized scan
    for pos in ("inramp", "floor", "above", "nogap"):
        for lev, arm in ((1, "mr"), (0, "ctl")):
            cs.append(
                (
                    f"t17_{pos}_e6_{arm}",
                    "t1_7_lowden.py",
                    T17_E6 + GAP[pos] + ["--max-level", str(lev)],
                    0.25,
                    "t17",
                )
            )
    # uniform-fine reference at the harshest position
    cs.append(
        (
            "t17_floor_e6_fine",
            "t1_7_lowden.py",
            T17_E6 + GAP["floor"] + ["--max-level", "0", "--resolution", "fine"],
            0.25,
            "t17",
        )
    )
    # holmstrom = 0 attribution pair at the stabilized point
    for lev, arm in ((1, "mr"), (0, "ctl")):
        cs.append(
            (
                f"t17_floor_e6_holm0_{arm}",
                "t1_7_lowden.py",
                T17_E6 + GAP["floor"] + ["--holmstrom", "0", "--max-level", str(lev)],
                0.25,
                "t17",
            )
        )
    # stabilized soak leg (15.6 t_ci, the T1.1-battery stability horizon)
    cs.append(
        (
            "t17_floor_e6long_mr",
            "t1_7_lowden.py",
            ["--eta", "1.0e-6", "--steps", "1300", "--plot-int", "10"]
            + GAP["floor"]
            + ["--max-level", "1"],
            0.25,
            "t17",
        )
    )
    # eta = 1e-7 threshold-documentation legs (fine-containing floor arms
    # are EXPECTED to die < 100 steps -- that is the recorded finding;
    # nogap_e7_mr completing is the floor-attribution control)
    cs.append(
        (
            "t17_floor_e7_mr",
            "t1_7_lowden.py",
            T17_E7 + GAP["floor"] + ["--max-level", "1"],
            0.25,
            "t17",
        )
    )
    cs.append(
        (
            "t17_floor_e7_ctl",
            "t1_7_lowden.py",
            T17_E7 + GAP["floor"] + ["--max-level", "0"],
            0.25,
            "t17",
        )
    )
    cs.append(
        (
            "t17_floor_e7_fine",
            "t1_7_lowden.py",
            T17_E7 + GAP["floor"] + ["--max-level", "0", "--resolution", "fine"],
            0.25,
            "t17",
        )
    )
    cs.append(
        (
            "t17_nogap_e7_mr",
            "t1_7_lowden.py",
            T17_E7 + GAP["nogap"] + ["--max-level", "1"],
            0.25,
            "t17",
        )
    )
    # ---------------- T1.8 ------------------------------------------
    cs.append(("t18_mr", "t1_3_alfven.py", T18 + ["--max-level", "1"], 0.25, "t1"))
    cs.append(("t18_ctl", "t1_3_alfven.py", T18 + ["--max-level", "0"], 0.25, "t1"))
    return cs


def check_finite(npz_path):
    """Return (ok, first_bad_step or -1)."""
    d = np.load(npz_path)
    for key in ("bx0", "by0", "bz0", "bx1", "by1", "bz1", "emax0", "emax1"):
        if key not in d:
            continue
        bad = ~np.isfinite(d[key]).all(axis=1)
        if bad.any():
            return False, int(np.argmax(bad))
    return True, -1


def run_case(name, script, args, b0, reducer, nprocs, keep_plt):
    out_dir = os.path.join(RUNS, name)
    done = os.path.join(out_dir, ".done")
    failed = os.path.join(out_dir, ".failed")
    if os.path.exists(done) or os.path.exists(failed):
        print(f"[battery78] {name}: already finished, skipping")
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
        if reducer == "t17":
            npz = t1_7_reduce.convert_run(out_dir, b0=b0)
        else:
            npz = t1_pltreader.convert_run(out_dir, b0=b0)
        ok, bad_step = check_finite(npz)
    status = dict(
        name=name,
        rc=rc,
        nan_in_log=nan_in_log,
        finite=ok,
        first_bad_step=bad_step,
        npz=npz,
    )
    marker = done if (rc == 0 and ok and not nan_in_log) else failed
    with open(marker, "w") as f:
        json.dump(status, f, indent=1)
    print(
        f"[battery78] {name}: rc={rc} finite={ok} "
        f"(first bad output {bad_step}) nan_in_log={nan_in_log} "
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
    p.add_argument("--nprocs", type=int, default=8)
    p.add_argument("--keep-plt", action="store_true")
    a = p.parse_args()

    cs = cases()
    if a.only:
        keys = a.only.split(",")
        cs = [c for c in cs if any(k in c[0] for k in keys)]
    if a.list:
        for n, s, ar, b0, red in cs:
            print(n, s, " ".join(ar), f"(monitor b0={b0}, reducer={red})")
        return
    os.makedirs(RUNS, exist_ok=True)
    for n, s, ar, b0, red in cs:
        run_case(n, s, ar, b0, red, a.nprocs, a.keep_plt)


if __name__ == "__main__":
    main()
