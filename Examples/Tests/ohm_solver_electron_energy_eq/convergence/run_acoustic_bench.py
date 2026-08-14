#!/usr/bin/env python3
"""Closure benchmark driver: polytropic vs energy-equation vs FD-conduction
on the ion-acoustic crossover (qdsmc_acoustic_test.py).

Arms:
  poly_par        polytropic closure           -> x = sqrt(5/3), undamped
  ee_chi0_par     energy eq, conduction off    -> x = sqrt(5/3) (regression:
                                                  the energy equation must
                                                  reproduce the closure it
                                                  generalizes)
  ee_R*_par       energy eq + FD conduction    -> the conduction bridge
  ee_R10_perp     k _|_ B at strong chi        -> x = sqrt(5/3) unchanged
                                                  (anisotropy in ion dynamics)

Scoring: fit A e^{-g t} sin(w t + phi) to Re(n_k)(t) and compare
(w, g) against the damped-acoustic root of

    x^3 + i R x^2 - (5/3) x - i R = 0,  x = (w - i g)/(k c_iso).

Usage:  python3 run_acoustic_bench.py [--jobs 8]
"""

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(HERE, "acoustic_out")
ENV = dict(
    os.environ,
    LD_LIBRARY_PATH="/usr/local/openmpi5/lib:/usr/local/hdf5/lib",
    OMP_NUM_THREADS="2",
)

parser = argparse.ArgumentParser()
parser.add_argument("--jobs", type=int, default=8)
args = parser.parse_args()

R_SWEEP = (0.3, 1.0, 3.0, 10.0)

# chi from R: chi = R * c_iso / k with the deck's fixed parameters
KB = 1.380649e-23
QE = 1.602176634e-19
MI = 1.67262192369e-27
TE0 = 10.0 * QE / KB
C_ISO = np.sqrt(KB * TE0 / MI)
KW = 2.0 * np.pi  # mode 1, Lz = 1


def chi_of(R):
    return R * C_ISO / KW


def build_cases():
    cases = [
        ("poly_par", dict(closure="poly", b_orient="par")),
        ("ee_chi0_par", dict(closure="ee", chi=0.0, b_orient="par")),
    ]
    for R in R_SWEEP:
        cases.append(
            (f"ee_R{R:g}_par", dict(closure="ee", chi=chi_of(R), b_orient="par"))
        )
    cases.append(("ee_R10_perp", dict(closure="ee", chi=chi_of(10.0), b_orient="perp")))
    return cases


def run_case(tag, kw):
    path = os.path.join(OUT, f"{tag}.npz")
    if os.path.exists(path):
        return tag
    log = os.path.join(OUT, f"{tag}.log")
    cmd = [PY, os.path.join(HERE, "qdsmc_acoustic_test.py"), "--out", path]
    for key, val in kw.items():
        cmd += ["--" + key.replace("_", "-"), str(val)]
    with open(log, "w") as fh:
        r = subprocess.run(cmd, env=ENV, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
    print(f"[{'done' if r.returncode == 0 else 'FAIL'}] {tag}", flush=True)
    return tag


def dispersion_root(R):
    """Damped-acoustic root x of x^3 + iRx^2 - (5/3)x - iR = 0
    (Re x > 0; e^{-i w t} convention so Im x <= 0 is damping)."""
    roots = np.roots([1.0, 1j * R, -5.0 / 3.0, -1j * R])
    cand = [r for r in roots if r.real > 0.1]
    return (
        min(cand, key=lambda r: abs(r.imag))
        if R == 0
        else max(cand, key=lambda r: r.real)
    )


def fit_wave(t, y):
    """least-squares A e^{-g t} sin(w t + phi); returns (w, g)."""
    # initial guesses from zero crossings / envelope
    w0 = 2.0 * np.pi / (t[-1] / 4.0)
    yn = y / (np.abs(y).max() + 1e-300)

    def model(tt, A, g, w, ph):
        return A * np.exp(-g * tt) * np.sin(w * tt + ph)

    best = None
    for wg in (w0, 1.29 * KW * C_ISO, 1.0 * KW * C_ISO):
        try:
            p, _ = curve_fit(model, t, yn, p0=[1.0, 0.0, wg, 0.0], maxfev=20000)
            resid = float(np.sum((model(t, *p) - yn) ** 2))
            if best is None or resid < best[1]:
                best = (p, resid)
        except RuntimeError:
            continue
    p = best[0]
    return abs(p[2]), p[1]


def report():
    print("\n## Acoustic crossover: measured vs linear theory\n")
    print(
        "| arm | R | x_meas (w/k c_iso) | x_theory | g_meas/k c_iso "
        "| g_theory | verdict |"
    )
    print("|---|---|---|---|---|---|---|")
    x_ad = np.sqrt(5.0 / 3.0)
    for tag, _ in build_cases():
        path = os.path.join(OUT, f"{tag}.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        t = d["t"]
        y = d["n_amp"].real
        w, g = fit_wave(t, y)
        xm = w / (KW * C_ISO)
        gm = g / (KW * C_ISO)
        if tag.startswith("poly") or "chi0" in tag or tag.endswith("perp"):
            xt, gt = x_ad, 0.0
        else:
            R = float(d["R"])
            root = dispersion_root(R)
            xt, gt = root.real, -root.imag
        ok = abs(xm - xt) / xt < 0.03 and abs(gm - gt) < 0.05 * x_ad
        print(
            f"| {tag} | {float(d['R']):.3g} | {xm:.4f} | {xt:.4f} "
            f"| {gm:+.4f} | {gt:.4f} | {'PASS' if ok else 'CHECK'} |"
        )
    print(
        "\nBridge endpoints: adiabatic x = 1.2910, isothermal x = 1.0000; "
        "perp arm must sit at the adiabatic endpoint at ANY chi."
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = build_cases()
    todo = [c for c in cases if not os.path.exists(os.path.join(OUT, c[0] + ".npz"))]
    print(f"[bench] {len(cases)} arms ({len(todo)} to run), jobs={args.jobs}")
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for _ in cf.as_completed([pool.submit(run_case, *c) for c in todo]):
            pass
    report()


if __name__ == "__main__":
    main()
