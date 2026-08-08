#!/usr/bin/env python3
"""m=4 rung-1 sweep driver: run the ring-anisotropy instrument across the
executor arms and controls, then print the attribution table.

Arms (qdsmc_m4_dev_prompt.md rung 1):
  split-ppm     production default (fluxform split sweeps, ppm, MC limiter)
  split-plm     reconstruction control
  layer         gather-form control (historical arm)
  unsplit-plm   CSLAM-style unsplit donor transport (periodic-only)
Controls (isolate one executor component each, on split-ppm):
  nolimiter     qdsmc_conduction_slope_limiter = none
  widehop       qdsmc_conduction_max_hop = 1e9
  isolaunch     qdsmc_conduction_isotropic_launch = 1 (45-deg rotated pair)
  nolim-widehop both released (residual = remap/donor identity loss)

Usage:
  python3 run_m4_ring.py --python <py> --outdir m4_ring [--ncell 128]
"""

import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
TEST = HERE / "qdsmc_m4_ring_test.py"

ARMS = {
    "split-ppm": [],
    "split-plm": ["--recon", "plm"],
    "layer": ["--form", "layer"],
    "unsplit-plm": ["--ff-unsplit", "1", "--recon", "plm"],
    "nolimiter": ["--slope-limiter", "none"],
    "widehop": ["--max-hop", "1e9"],
    "isolaunch": ["--iso-launch", "1"],
    "nolim-widehop": ["--slope-limiter", "none", "--max-hop", "1e9"],
}

parser = argparse.ArgumentParser()
parser.add_argument("--python", default=sys.executable)
parser.add_argument("--outdir", default="m4_ring")
parser.add_argument("--ncell", type=int, default=128)
parser.add_argument("--nsteps", type=int, default=128)
parser.add_argument("--arms", nargs="+", default=list(ARMS.keys()))
parser.add_argument("--extra", nargs="*", default=[], help="extra deck args, all arms")
args = parser.parse_args()

outdir = pathlib.Path(args.outdir)
outdir.mkdir(exist_ok=True)

results = {}
for arm in args.arms:
    npz = outdir / f"{arm}.npz"
    log = outdir / f"{arm}.log"
    cmd = (
        [
            args.python,
            str(TEST),
            "--ncell",
            str(args.ncell),
            "--nsteps",
            str(args.nsteps),
            "--out",
            str(npz),
        ]
        + ARMS[arm]
        + args.extra
    )
    print(f"[m4-sweep] {arm}: {' '.join(cmd)}", flush=True)
    with open(log, "w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=outdir)
    if r.returncode != 0:
        print(f"[m4-sweep] {arm} FAILED (exit {r.returncode}), see {log}", flush=True)
        results[arm] = None
        continue
    import numpy as np

    d = np.load(npz)
    results[arm] = (
        float(d["m4_over_m0"][-1]),
        float(d["m4_growth_per_step"]),
        float(np.sqrt(d["a2_re"][-1] ** 2 + d["a2_im"][-1] ** 2) / abs(d["a0"][-1])),
        float(np.sqrt(d["a8_re"][-1] ** 2 + d["a8_im"][-1] ** 2) / abs(d["a0"][-1])),
        float(np.arctan2(d["a4_im"][-1], d["a4_re"][-1])),
    )

print(
    f"\n{'arm':<15} {'m4/m0 final':>12} {'growth/step':>12} "
    f"{'m2/m0':>10} {'m8/m0':>10} {'arg(a4)':>9}",
    flush=True,
)
for arm, v in results.items():
    if v is None:
        print(f"{arm:<15} {'FAILED':>12}")
        continue
    m4, g, m2, m8, ph = v
    print(f"{arm:<15} {m4:>12.4e} {g:>12.4e} {m2:>10.3e} {m8:>10.3e} {ph:>+9.3f}")
