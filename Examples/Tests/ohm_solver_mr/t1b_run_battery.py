#!/usr/bin/env python3
"""Serial battery driver for the T1b graded seam-dissipation-band tests.

Builds on the T1 whistler battery (t1_whistler.py deck, t1_pltreader.py
converter): every case below is a T1 case plus the seam-band knobs
    hybrid_pic_model.mr_seam_band_width  (W, fine cells)
    hybrid_pic_model.mr_seam_eta         (Ohm m, target at the patch edge)
    hybrid_pic_model.mr_seam_eta_h       (Ohm m^3 literal, or "coarse-matched")
passed through --extra-inputs.

Experiments (see T1B_RESULTS.md):
  E1  instability kill test: bare seam vs eta_h band vs eta band vs both,
      at the T1 unstable configuration (dt = 0.012 t_ci, kfac 0.1 seed,
      bulk eta = 1e-10, bulk eta_h = 0), 20 t_ci horizon. Band width scan
      W = 3 and 6 for the winning arm.
  E2  reflection regression: T1.1 pairs at kfac 0.3 / 0.6 / 0.8 and the
      T1.2 packet arm (plus its null twin), with the winning band on.
  E3  bulk-physics null: run separately with t1b_smoke_null (uses the
      t_smoke_inputs_2d_uniform deck, not this driver).

Usage:
    python t1b_run_battery.py --list
    python t1b_run_battery.py [--only substr1,substr2] [--nprocs 8]
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
RUNS = os.path.join(HERE, "t1b_runs")
sys.path.insert(0, HERE)
import t1_pltreader  # noqa: E402
import t1_whistler  # noqa: E402

# seam-band targets:
# eta_h seam target = the coarse level's recommended absorber strength,
# 0.2 * eta_h*(coarse) (T1.2 result); with per-level Nyquist-scaled bulk
# eta_h this is exactly the "coarse-matched" value (16x the fine bulk).
SEAM_ETA_H = 0.2 * t1_whistler.ETA_H_STAR  # = 1.532e-18 Ohm m^3
# fine-referenced bulk absorber for the coarse-matched parser-path arm:
# 0.2 * eta_h*(fine) = SEAM_ETA_H / 16
BULK_ETA_H_FINE = SEAM_ETA_H / 16.0
# eta seam target = the measured global stabilization threshold
SEAM_ETA = 3.0e-8  # Ohm m

E1_BASE = [
    "--kfac",
    "0.1",
    "--mode",
    "global",
    "--dt-fac",
    "0.012",
    "--steps",
    "1667",
    "--max-level",
    "1",
]
T11_BASE = ["--mode", "global", "--dt-fac", "0.006", "--steps", "1000"]
T12_BASE = ["--kfac", "1.27", "--mode", "packet", "--dt-fac", "0.005", "--steps", "800"]


def band_args(width=None, eta=None, eta_h=None):
    """--extra-inputs argument list for the seam-band knobs."""
    out = []
    if width is not None:
        out += ["--extra-inputs", f"hybrid_pic_model.mr_seam_band_width = {width}"]
    if eta is not None:
        out += ["--extra-inputs", f"hybrid_pic_model.mr_seam_eta = {eta}"]
    if eta_h is not None:
        out += ["--extra-inputs", f"hybrid_pic_model.mr_seam_eta_h = {eta_h}"]
    return out


def cases():
    cs = []
    # ---------------- E1: instability kill test -----------------------
    cs.append(("e1_bare", E1_BASE))
    for w in (6, 3):
        cs.append(
            (f"e1_ehband_w{w}", E1_BASE + band_args(w, eta_h=f"{SEAM_ETA_H:.6e}"))
        )
        cs.append((f"e1_etaband_w{w}", E1_BASE + band_args(w, eta=f"{SEAM_ETA:.6e}")))
        cs.append(
            (
                f"e1_both_w{w}",
                E1_BASE
                + band_args(w, eta=f"{SEAM_ETA:.6e}", eta_h=f"{SEAM_ETA_H:.6e}"),
            )
        )
    # coarse-matched parser path: per-level Nyquist-scaled bulk eta_h
    # (fine-referenced absorber everywhere) + coarse-matched seam band
    cs.append(
        (
            "e1_ehband_cm_w6",
            E1_BASE
            + ["--eta-h-fac", str(BULK_ETA_H_FINE / t1_whistler.ETA_H_STAR)]
            + band_args(6, eta_h="coarse-matched"),
        )
    )
    # mechanism localization: wider band (covers all C-F operator reach)
    # and stronger targets at fixed width
    cs.append(("e1_etaband_w24", E1_BASE + band_args(24, eta=f"{SEAM_ETA:.6e}")))
    cs.append(("e1_etastrong_w6", E1_BASE + band_args(6, eta="1.0e-7")))
    cs.append(("e1_ehstrong_w6", E1_BASE + band_args(6, eta_h=f"{5 * SEAM_ETA_H:.6e}")))
    # control anchor: global (both-level) eta at the T1 stabilization
    # threshold, no band -- the coarse side is dissipative here
    cs.append(("e1_globaleta_ctl", E1_BASE + ["--eta", f"{SEAM_ETA:.6e}"]))

    # ---------------- E2: reflection regression (winning band) --------
    for kf in (0.3, 0.6, 0.8):
        tag = f"t11b_k{int(round(kf * 1000)):04d}"
        base = ["--kfac", str(kf)] + T11_BASE
        cs.append(
            (
                f"{tag}_mr",
                base + ["--max-level", "1"] + band_args(6, eta_h=f"{SEAM_ETA_H:.6e}"),
            )
        )
        cs.append((f"{tag}_ctl", base + ["--max-level", "0"]))
    # T1.2 packet with the band + identical-seed null twin
    cs.append(
        (
            "t12b_pk127_band",
            T12_BASE + ["--max-level", "1"] + band_args(6, eta_h=f"{SEAM_ETA_H:.6e}"),
        )
    )
    cs.append(
        (
            "t12b_null_band",
            T12_BASE
            + ["--max-level", "1", "--amp-fac", "0"]
            + band_args(6, eta_h=f"{SEAM_ETA_H:.6e}"),
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
        print(f"[t1b] {name}: already finished, skipping")
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
        f"[t1b] {name}: rc={rc} finite={ok} "
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
