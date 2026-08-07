#!/usr/bin/env python3
"""Serial battery driver for the T1.3 (Alfven-wave patch transits) and
T1.4 (advected tangential discontinuity) hybrid-PIC MR tests.

Same conventions as t1_run_battery.py: each case gets a directory under
t1_runs/, plotfiles are reduced to t1_lineouts.npz (t1_pltreader) and
deleted, a .done/.failed marker makes the battery resumable, and a run
that goes non-finite is a RESULT (seam instability), not an error.

Arms (T1.3):
  bare pair at dt = 0.006 t_ci (eta = 1e-10; valid window ends at the
  seam-instability noise) and a stabilized triple at dt = 0.012 t_ci with
  plasma_resistivity = 3e-8 Ohm m -- the measured stabilization floor --
  applied IDENTICALLY to the MR arm, the uniform-coarse control and the
  uniform-fine reference, so all comparisons stay fair. The stabilized
  horizon is 18700 steps = 224 t_ci = 11.3 patch transits.

Arms (T1.4):
  stabilized (eta = 3e-8) MR / uniform-coarse / uniform-fine triple over
  1900 steps = 22.8 t_ci (full slab passage + downwind window), plus a
  bare (eta = 1e-10) MR/ctl pair whose valid window covers the first
  seam crossing.

Usage:
    python t1_34_battery.py --list
    python t1_34_battery.py [--only substr1,substr2] [--nprocs 8]
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
import t1_pltreader  # noqa: E402

ETA_STAB = "3.0e-8"  # measured seam-instability stabilization floor (Ohm m)
ETA_BARE = "1.0e-10"

# The quiet-seam stabilization floor eta = 3e-8 Ohm m does NOT hold once
# the 2% wave feeds the seam: measured blowups at ~5 t_ci (dt = 0.012,
# t13_mr_stab) and ~7 t_ci (dt = 0.006, dbg13_dt006_eta3e8) -- the
# wave-driven channel does not scale down with dt like the quiet one.
# eta = 1e-7 (the next measured-stable quiet value) is flat-stable with
# the wave for >= 48 t_ci (dbg13_dt012_eta1e7), so the long arms carry it,
# identically in MR / ctl / fine.
ETA_STAB13 = "1.0e-7"
T13_BARE = [
    "--eta",
    ETA_BARE,
    "--dt-fac",
    "0.006",
    "--steps",
    "1300",
    "--plot-int",
    "5",
]
T13_STAB = [
    "--eta",
    ETA_STAB13,
    "--dt-fac",
    "0.012",
    "--steps",
    "18700",
    "--plot-int",
    "20",
]
# 256 ppc: at 64 ppc the x-averaged |B| PIC noise is ~2.6% of B0 rms,
# which would drown a ~1% wake-ringing signal; 256 ppc halves it and the
# runs stay minutes-cheap. Applied to all T1.4 arms identically.
T14_BASE = ["--dt-fac", "0.012", "--steps", "1900", "--plot-int", "5", "--ppc", "256"]


def cases():
    """Return [(name, deck_script, args, b0_for_monitor)]."""
    cs = []
    # ---------------- T1.3 ------------------------------------------
    cs.append(("t13_mr_bare", "t1_3_alfven.py", T13_BARE + ["--max-level", "1"], 0.25))
    cs.append(("t13_ctl_bare", "t1_3_alfven.py", T13_BARE + ["--max-level", "0"], 0.25))
    # t13_mr_stab (eta = 3e-8, dt = 0.012) is the KEPT negative result --
    # its .failed marker documents that the quiet floor fails with the wave
    cs.append(
        ("t13_mr_stab_e7", "t1_3_alfven.py", T13_STAB + ["--max-level", "1"], 0.25)
    )
    cs.append(
        ("t13_ctl_stab_e7", "t1_3_alfven.py", T13_STAB + ["--max-level", "0"], 0.25)
    )
    cs.append(
        (
            "t13_fine_stab_e7",
            "t1_3_alfven.py",
            T13_STAB + ["--max-level", "0", "--resolution", "fine"],
            0.25,
        )
    )
    # ---------------- T1.4 ------------------------------------------
    cs.append(
        (
            "t14_mr",
            "t1_4_td.py",
            T14_BASE + ["--eta", ETA_STAB, "--max-level", "1"],
            0.0,
        )
    )
    cs.append(
        (
            "t14_ctl",
            "t1_4_td.py",
            T14_BASE + ["--eta", ETA_STAB, "--max-level", "0"],
            0.0,
        )
    )
    cs.append(
        (
            "t14_fine",
            "t1_4_td.py",
            T14_BASE + ["--eta", ETA_STAB, "--max-level", "0", "--resolution", "fine"],
            0.0,
        )
    )
    cs.append(
        (
            "t14_mr_bare",
            "t1_4_td.py",
            T14_BASE + ["--eta", ETA_BARE, "--max-level", "1"],
            0.0,
        )
    )
    cs.append(
        (
            "t14_ctl_bare",
            "t1_4_td.py",
            T14_BASE + ["--eta", ETA_BARE, "--max-level", "0"],
            0.0,
        )
    )
    return cs


def check_finite(npz_path):
    """Return (ok, first_bad_step or -1)."""
    d = np.load(npz_path)
    for key in ("bx0", "by0", "bz0", "bx1", "by1", "bz1"):
        if key not in d:
            continue
        bad = ~np.isfinite(d[key]).all(axis=1)
        if bad.any():
            return False, int(np.argmax(bad))
    return True, -1


def run_case(name, script, args, b0, nprocs, keep_plt):
    out_dir = os.path.join(RUNS, name)
    done = os.path.join(out_dir, ".done")
    failed = os.path.join(out_dir, ".failed")
    if os.path.exists(done) or os.path.exists(failed):
        print(f"[battery34] {name}: already finished, skipping")
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
        f"[battery34] {name}: rc={rc} finite={ok} "
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
        for n, s, ar, b0 in cs:
            print(n, s, " ".join(ar), f"(monitor b0={b0})")
        return
    os.makedirs(RUNS, exist_ok=True)
    for n, s, ar, b0 in cs:
        run_case(n, s, ar, b0, a.nprocs, a.keep_plt)


if __name__ == "__main__":
    main()
