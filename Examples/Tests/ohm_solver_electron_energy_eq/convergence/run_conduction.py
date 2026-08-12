#!/usr/bin/env python3
"""G3 driver: conduction convergence sweeps on qdsmc_conduction_test.py.

Sections
--------
aligned2 : aligned-B Gaussian spread, parabolic refinement (dt ~ dx^2,
           nsteps = 32*(N/32)^2), npts = 3 -> L2 slope vs exact.
aligned2_np2 : same at npts = 2 (weak-order-1 quadrature; with dt ~ dx^2 its
           O(dt) defect scales like dx^2, so the slope should hold ~2 with a
           larger constant).
nograd   : same as aligned2 but --grad-deposit 0 -> shows the daughter-remap
           chi_num floor the half-gradient correction removes.
tilted   : B at 30 deg, N = 64, npts = 3: tensor rotation + anisotropy
           pollution (chi_perp_num / chi_par).

Usage: run_conduction.py <section> [...sections]
"""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(HERE, "cond_out")
ENV = dict(
    os.environ,
    LD_LIBRARY_PATH="/usr/local/openmpi5/lib:/usr/local/hdf5/lib",
    OMP_NUM_THREADS="2",
)


def run_case(tag, **kw):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{tag}.npz")
    if os.path.exists(path):
        return np.load(path)
    cmd = [PY, os.path.join(HERE, "qdsmc_conduction_test.py"), "--out", path]
    for key, val in kw.items():
        cmd.append("--" + key.replace("_", "-"))
        if isinstance(val, (list, tuple)):
            cmd += [str(v) for v in val]
        else:
            cmd.append(str(val))
    print("[run ]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=ENV, cwd=HERE)
    return np.load(path)


def slope_table(title, cases):
    print(f"\n## {title}\n")
    print("| N | steps | rel L2 | order | chi_par/chi0 |")
    print("|---|-------|--------|-------|--------------|")
    prev = None
    errs, ns = [], []
    for d in cases:
        n, e = int(d["ncell"]), float(d["rel_l2"])
        r = float(d["chi_par_meas"] / d["chi0"])
        o = f"{np.log(prev[1] / e) / np.log(n / prev[0]):.2f}" if prev else ""
        print(f"| {n} | {int(d['nsteps'])} | {e:.4e} | {o} | {r:.4f} |")
        prev = (n, e)
        errs.append(e)
        ns.append(n)
    if len(ns) > 2:
        fit = np.polyfit(np.log(ns), np.log(errs), 1)[0]
        print(f"\nfit slope (in dx): **{-fit:.2f}**")


def sweep(section, npts, grad):
    cases = []
    for n in (24, 32, 48, 64, 96, 128):
        steps = max(8, round(32 * (n / 32) ** 2))
        cases.append(
            run_case(
                f"{section}_N{n}",
                mode="aligned",
                ncell=n,
                nsteps=steps,
                npts=npts,
                grad_deposit=grad,
            )
        )
    slope_table(section, cases)


if __name__ == "__main__":
    sections = sys.argv[1:] or ["aligned2", "aligned2_np2", "nograd", "tilted"]
    for s in sections:
        if s == "aligned2":
            sweep("aligned2", [3], 1)
        elif s == "aligned2_np2":
            sweep("aligned2_np2", [2], 1)
        elif s == "nograd":
            sweep("nograd", [3], 0)
        elif s == "tilted":
            d = run_case(
                "tilted_N64",
                mode="tilted",
                ncell=64,
                nsteps=128,
                npts=[3],
                grad_deposit=1,
            )
            print("\n## tilted 30 deg, N=64, npts=3\n")
            print(f"rel L2 vs exact        : {float(d['rel_l2']):.4e}")
            print(
                f"chi_par_meas / chi0    : {float(d['chi_par_meas'] / d['chi0']):.4f}"
            )
            print(
                f"chi_perp_num / chi0    : {float(d['chi_perp_meas'] / d['chi0']):.3e}"
            )
            print(
                f"sum(Te) drift          : "
                f"{float((d['te_sum1'] - d['te_sum0']) / d['te_sum0']):.3e}"
            )
        else:
            raise SystemExit(f"unknown section {s}")
