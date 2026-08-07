#!/usr/bin/env python3
"""Serial battery driver for the T1 whistler seam-reflection tests.

Runs the t1_whistler.py cases listed below (T1.1 dispersion/reflection scan
below the coarse Nyquist, MR vs max_level=0 control, plus one uniform-fine
reference; T1.2 above-coarse-Nyquist packet with a hyper-resistivity scan),
converts each run's per-step plotfiles to a compact npz lineout file
(t1_pltreader.convert_run) and then deletes the plotfiles. A run that
already has a .done marker is skipped, so the battery can be resumed or
chunked; a run that goes non-finite is marked .failed and the battery
continues (a seam-driven instability is a result, not an error).

Usage:
    python t1_run_battery.py --list
    python t1_run_battery.py [--only substr1,substr2] [--nprocs 8]
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

# T1.1 runs at dt = 0.006 t_ci for 6 t_ci: the bare coarse-fine seam has a
# numerical instability whose growth rate scales ~linearly with dt
# (gamma ~ 0.09 w_ci at dt = 0.012 t_ci, ~0.045 at 0.006; blowup at ~5 and
# ~7.6 t_ci respectively, from noise), so the battery runs inside the
# stability window instead of polluting the measurement with a stabilizer.
KFACS = [0.1, 0.2, 0.3, 0.45, 0.6, 0.7, 0.8, 0.9]
ETAH_FACS = [0.0, 0.03, 0.2, 1.0]
T11_BASE = ["--mode", "global", "--dt-fac", "0.006", "--steps", "1000"]
T12_BASE = ["--kfac", "1.27", "--mode", "packet", "--dt-fac", "0.005", "--steps", "800"]


def cases():
    cs = []
    for kf in KFACS:
        tag = f"t11_k{int(round(kf * 1000)):04d}"
        base = ["--kfac", str(kf)] + T11_BASE
        cs.append((f"{tag}_mr", base + ["--max-level", "1"]))
        cs.append((f"{tag}_ctl", base + ["--max-level", "0"]))
    # uniform-fine reference (patch removed, fine resolution everywhere)
    cs.append(
        (
            "t11_k0600_fine",
            ["--kfac", "0.6"] + T11_BASE + ["--max-level", "0", "--resolution", "fine"],
        )
    )
    for eh in ETAH_FACS:
        tag = f"t12_pk127_eh{int(round(eh * 100)):03d}_mr"
        cs.append((tag, T12_BASE + ["--max-level", "1", "--eta-h-fac", str(eh)]))
    # free-dispersal reference: same packet, uniform fine, no patch
    cs.append(
        ("t12_pk127_fine", T12_BASE + ["--max-level", "0", "--resolution", "fine"])
    )
    # null (amp = 0) twins for coherent PIC-noise subtraction: identical
    # seed/layout, so b_signal = b_run - b_null removes the thermal
    # magnetic noise that otherwise dwarfs the packet energy budget
    for eh in ETAH_FACS:
        tag = f"t12_null_eh{int(round(eh * 100)):03d}_mr"
        cs.append(
            (
                tag,
                T12_BASE
                + ["--max-level", "1", "--amp-fac", "0", "--eta-h-fac", str(eh)],
            )
        )
    cs.append(
        (
            "t12_null_fine",
            T12_BASE + ["--max-level", "0", "--resolution", "fine", "--amp-fac", "0"],
        )
    )
    return cs


def check_finite(npz_path):
    """Return (ok, first_bad_step or -1)."""
    d = np.load(npz_path)
    for key in ("bx0", "by0", "bx1", "by1"):
        if key not in d:
            continue
        bad = ~np.isfinite(d[key]).all(axis=1)
        if bad.any():
            return False, int(np.argmax(bad))
    return True, -1


def run_case(name, args, nprocs, keep_plt):
    out_dir = os.path.join(RUNS, name)
    done = os.path.join(out_dir, ".done")
    failed = os.path.join(out_dir, ".failed")
    if os.path.exists(done) or os.path.exists(failed):
        print(f"[battery] {name}: already finished, skipping")
        return
    cmd = [
        sys.executable,
        os.path.join(HERE, "t1_whistler.py"),
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
        npz = t1_pltreader.convert_run(out_dir)
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
        f"[battery] {name}: rc={rc} finite={ok} "
        f"(first bad step {bad_step}) nan_in_log={nan_in_log} "
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
        cs = [(n, ar) for n, ar in cs if any(k in n for k in keys)]
    if a.list:
        for n, ar in cs:
            print(n, " ".join(ar))
        return
    os.makedirs(RUNS, exist_ok=True)
    for n, ar in cs:
        run_case(n, ar, a.nprocs, a.keep_plt)


if __name__ == "__main__":
    main()
