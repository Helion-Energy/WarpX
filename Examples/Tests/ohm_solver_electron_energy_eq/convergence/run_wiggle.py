#!/usr/bin/env python3
"""C.7/G3a driver: curved-field-line leak scaling on qdsmc_wiggle_test.py.

The wiggle instrument measures chi_perp_num = growth of Var_w(A)/2t on a
snaking field B = B0 (eps sin(kz), 0, 1). Early probing showed the leak is
NOT the O(chi^2 dt kappa^2) chord-vs-arc term: it grows with 1/dt at fixed
dx (a per-remap effect -- the E6 remap floor dx^2/4dt reopened by
curvature, only partially cancelled by the B1 source-slope correction).
These sections pin the empirical scaling
    chi_leak / chi0 = C * (dx^2/(4 chi0 dt_c))^a * (eps)^b * (k dx)^c * ...
and the npts / correction-on-off structure, to extrapolate to liftoff
(R_c ~ 0.35 m) for gate G3a.

Sections: keps (k and eps scan), dxdt (N and nsteps scan), quad (npts,
gradient-deposit control), floor (eps = 0 controls). Usage:
  run_wiggle.py [section ...]      (default: all)
"""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(HERE, "wiggle_out")
ENV = dict(
    os.environ,
    LD_LIBRARY_PATH="/usr/local/openmpi5/lib:/usr/local/hdf5/lib",
    OMP_NUM_THREADS="2",
)

BASE = dict(ncell=64, nsteps=64, npts=[3], eps=0.2, mwiggle=2)


def run_case(tag, **kw):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{tag}.npz")
    if os.path.exists(path):
        return np.load(path)
    cmd = [PY, os.path.join(HERE, "qdsmc_wiggle_test.py"), "--out", path]
    for key, val in kw.items():
        cmd.append("--" + key.replace("_", "-"))
        if isinstance(val, (list, tuple)):
            cmd += [str(v) for v in val]
        else:
            cmd.append(str(val))
    print("[run ]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=ENV, cwd=HERE)
    return np.load(path)


def row(d):
    ncell = int(d["ncell"])
    nsteps = int(d["nsteps"])
    dx = float(d["L"]) / ncell
    dt_c = float(d["dt"]) / 2.0
    chi0 = float(d["chi0"])
    floorx = dx * dx / (4.0 * dt_c)  # raw remap-floor scale
    leak = float(d["chi_perp_num"]) / chi0
    return (
        f"| {ncell} | {nsteps} | {float(d['eps']):.2f} | {int(d['mwiggle'])} "
        f"| {'/'.join(str(int(v)) for v in d['npts'])} "
        f"| {leak:.3e} | {leak * chi0 / floorx:.3e} "
        f"| {float(d['adrift']):.2e} | {float(d['chi_par_meas']) / chi0:.3f} |"
    )


HDR = (
    "| N | steps | eps | m | npts | leak/chi0 | leak/floor | adrift | chi_par |\n"
    "|---|-------|-----|---|------|-----------|------------|--------|---------|"
)


def fit(tag, xs, ys):
    if len(xs) > 1 and all(y > 0 for y in ys):
        p = np.polyfit(np.log(xs), np.log(ys), 1)[0]
        print(f"    {tag} power-law exponent: {p:+.2f}")


def main(sections):
    if "keps" in sections:
        print("\n## k (mwiggle) and eps scaling  [N=64, ns=64, npts=3]\n")
        print(HDR)
        ms, ml = [], []
        for m in (1, 2, 4):
            d = run_case(f"k_m{m}", **{**BASE, "mwiggle": m})
            print(row(d))
            ms.append(m)
            ml.append(float(d["chi_perp_num"]) / float(d["chi0"]))
        fit("k", ms, ml)
        es, el = [], []
        for e in (0.1, 0.2, 0.3, 0.42):
            d = run_case(f"eps_{e}", **{**BASE, "eps": e})
            print(row(d))
            es.append(e)
            el.append(float(d["chi_perp_num"]) / float(d["chi0"]))
        fit("eps", es, el)

    if "dxdt" in sections:
        print("\n## dx and dt scaling  [eps=0.2, m=2, npts=3]\n")
        print(HDR)
        ns_, nl = [], []
        for n in (48, 64, 96, 128):
            d = run_case(f"dx_N{n}", **{**BASE, "ncell": n})
            print(row(d))
            ns_.append(n)
            nl.append(float(d["chi_perp_num"]) / float(d["chi0"]))
        fit("1/dx", ns_, nl)
        ts, tl = [], []
        for s in (32, 64, 128, 256):
            d = run_case(f"dt_ns{s}", **{**BASE, "nsteps": s})
            print(row(d))
            ts.append(s)
            tl.append(float(d["chi_perp_num"]) / float(d["chi0"]))
        fit("1/dt", ts, tl)

    if "quad" in sections:
        print("\n## quadrature and correction structure  [eps=0.2, m=2]\n")
        print(HDR)
        for np_ in (2, 3, 5):
            d = run_case(f"np{np_}", **{**BASE, "npts": [np_]})
            print(row(d))
        d = run_case("nograd", **{**BASE, "grad_deposit": 0})
        print(row(d), "   <- gradient correction OFF")

    if "floor" in sections:
        print("\n## eps = 0 floors\n")
        print(HDR)
        for n in (64, 128):
            d = run_case(f"floor_N{n}", **{**BASE, "ncell": n, "eps": 0.0})
            print(row(d))


if __name__ == "__main__":
    main(sys.argv[1:] or ["keps", "dxdt", "quad", "floor"])
