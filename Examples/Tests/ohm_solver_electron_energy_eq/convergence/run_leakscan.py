#!/usr/bin/env python3
"""Curvature-leak quantification scan (C.7d closeout, Eric 2026-08-05).

Eric's decision: the explicit (fluxform) arm is production and the
curvature leak is ACCEPTED — the parallel quasi-shorting condition keeps
field lines near-isothermal, so modest cross-line leakage mostly
relabels heat among nearly-isothermal lines. The remaining obligation is
QUANTIFICATION: an engineering scaling of

    leak == chi_perp_num / chi_par = f(sigma_b/dx, eps, m, sp/dx, npts, N_remap)

with sigma_b = blob (feature) cross-field width, sp = sqrt(2 chi dt_c)
the per-substep hop sigma (dt_c = dt/2 in the pc scheme), eps and m the
wiggle amplitude/periods (curvature dx/R_c ~ eps k dx / (1 + eps^2)).

Sections (all fluxform/PLM unless noted; N=64 rows ~15 s each, N=128
liftoff rows are minutes-long):
  width   : sigma_b/dx = 0.8 .. 12.8 at eps=0.2   [resolution axis]
  eps     : eps = 0.1 .. 0.45 at sigma_b/dx = 3.2 [curvature-amp axis]
  hop     : chi = 125 .. 8000 -> sp/dx = 0.28 .. 2.26 [hop/aliasing axis]
  npts    : npts = 2/3/5 at the standard point    [quadrature axis]
  mper    : m = 1/2/4 wiggle periods              [curvature-k axis]
  liftoff : the N=128/ns=224/eps=0.45 point: width scan + npts=5
            (does width-convergence survive at eps=0.45?)
  fit     : print power-law fits of the clean axes from cached rows

Caches per-case npz under leakscan_out/. Reference floors: eps=0 floor
3.6e-6; layer arm at the standard point 4.7e-4; layer at liftoff
2.56e-4 (npts=2).
"""

import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(HERE, "leakscan_out")
ENV = dict(
    os.environ,
    LD_LIBRARY_PATH="/usr/local/openmpi5/lib:/usr/local/hdf5/lib",
    OMP_NUM_THREADS="2",
)

BASE = dict(ncell=64, nsteps=64, npts=[3], eps=0.2, mwiggle=2, form="fluxform")


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
    t0 = time.time()
    subprocess.run(cmd, check=True, env=ENV, cwd=HERE)
    print(f"[time] {tag}: {time.time() - t0:.1f} s", flush=True)
    return np.load(path)


def row(tag, d, extra=""):
    leak = float(d["chi_perp_num"]) / float(d["chi0"])
    # hop sigma per conduction substep (pc: dt_c = dt/2), in cells
    dt_c = float(d["tfinal"]) / int(d["nsteps"]) / 2.0
    sp_dx = np.sqrt(2.0 * float(d["chi0"]) * dt_c) * int(d["ncell"])
    sb_dx = float(d["blob_sigma"]) * int(d["ncell"])
    return (
        f"| {tag} | {int(d['ncell'])} | {int(d['nsteps'])} "
        f"| {float(d['eps']):.2f} | {int(d['mwiggle'])} "
        f"| {sb_dx:.1f} | {sp_dx:.2f} "
        f"| {'/'.join(str(int(v)) for v in d['npts'])} "
        f"| {leak:.3e} | {float(d['adrift']):.2e} {extra}|"
    )


HDR = (
    "| tag | N | steps | eps | m | sb/dx | sp/dx | npts | leak/chi0 | adrift |\n"
    "|-----|---|-------|-----|---|-------|-------|------|-----------|--------|"
)


def fit_slope(xs, ys, label):
    lx, ly = np.log(np.asarray(xs)), np.log(np.asarray(ys))
    p = np.polyfit(lx, ly, 1)
    print(
        f"  {label}: exponent {p[0]:+.2f}  (points: "
        + ", ".join(f"({x:.3g}, {y:.2e})" for x, y in zip(xs, ys))
        + ")"
    )
    return p[0]


def main(sections):
    if "width" in sections:
        print("\n## feature-width axis  [eps=0.2, m=2, sp/dx=0.80]\n")
        print(HDR)
        for sb in (0.0125, 0.025, 0.05, 0.10, 0.15, 0.20):
            d = run_case(f"w_sb{sb}", **{**BASE, "blob_sigma": sb})
            print(row("width", d))

    if "eps" in sections:
        print("\n## curvature-amplitude axis  [sb/dx=3.2, m=2, sp/dx=0.80]\n")
        print(HDR)
        for ep in (0.1, 0.2, 0.3, 0.45):
            d = run_case(f"e_{ep}", **{**BASE, "eps": ep})
            print(row("eps", d))

    if "hop" in sections:
        print("\n## hop-size axis  [sb/dx=3.2, eps=0.2; chi scan at fixed dt]\n")
        print(HDR)
        for chi in (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0):
            d = run_case(f"c_{int(chi)}", **{**BASE, "chi": chi})
            print(row("hop", d))

    if "npts" in sections:
        print("\n## quadrature axis  [standard point]\n")
        print(HDR)
        for np_ in (2, 3, 5):
            d = run_case(f"q_np{np_}", **{**BASE, "npts": [np_]})
            print(row("npts", d))

    if "mper" in sections:
        print("\n## curvature-k axis  [sb/dx=3.2, eps=0.2]\n")
        print(HDR)
        for m in (1, 2, 4):
            d = run_case(f"m_{m}", **{**BASE, "mwiggle": m})
            print(row("mper", d))

    if "liftoff" in sections:
        print("\n## liftoff point  [N=128, ns=224, eps=0.45, m=2, sp/dx=0.86]\n")
        print(HDR)
        lift = dict(ncell=128, nsteps=224, eps=0.45, mwiggle=2, form="fluxform")
        for np_, sb in (
            (3, 0.05),
            (3, 0.10),
            (3, 0.15),
            (2, 0.05),
            (2, 0.10),
            (5, 0.05),
        ):
            d = run_case(
                f"l_np{np_}_sb{sb}", **{**lift, "npts": [np_], "blob_sigma": sb}
            )
            print(row("liftoff", d))
        print("(layer references: 2.56e-4 at npts=2 sb=0.05; eps=0 floor 3.6e-6)")

    if "fit" in sections:
        print("\n## power-law fits (cached rows)\n")

        def leak_of(tag):
            d = np.load(os.path.join(OUT, f"{tag}.npz"))
            return float(d["chi_perp_num"] / d["chi0"])

        try:
            sbs = (0.05, 0.10, 0.15, 0.20)
            fit_slope(
                [s * 64 for s in sbs],
                [leak_of(f"w_sb{s}") for s in sbs],
                "width sb/dx (resolved branch, >= 3.2 cells)",
            )
            eps = (0.1, 0.2, 0.3, 0.45)
            fit_slope(eps, [leak_of(f"e_{e}") for e in eps], "eps")
            chis = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
            fit_slope(
                [np.sqrt(2 * c * (1e-5 / 64 / 2)) * 64 for c in chis],
                [leak_of(f"c_{int(c)}") for c in chis],
                "hop sp/dx (leak already chi-normalized)",
            )
            ms = (1, 2, 4)
            fit_slope(ms, [leak_of(f"m_{m}") for m in ms], "m (k)")
            sbl = (0.05, 0.10, 0.15)
            fit_slope(
                [s * 128 for s in sbl],
                [leak_of(f"l_np3_sb{s}") for s in sbl],
                "liftoff width sb/dx (eps=0.45)",
            )
        except FileNotFoundError as err:
            print(f"  (skipping fit: missing cache {err})")


if __name__ == "__main__":
    main(sys.argv[1:] or ["width", "eps", "hop", "npts", "mper", "liftoff", "fit"])
