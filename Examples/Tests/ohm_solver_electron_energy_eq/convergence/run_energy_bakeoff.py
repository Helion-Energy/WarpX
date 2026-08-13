#!/usr/bin/env python3
"""Energy-equation transport bake-off: QDSMC markers vs direct grid
integration (SMART flux-form entropy pair through the shared adaptive RK
integrator, per QDSMC_FD_CONDUCTION_PLAN.md).

Arms x cases over qdsmc_advection_test.py (exact solutions + conservation
probes) with wall-clock from the harness perf line. Both arms consume
identical inputs (V_e, rho, Te seed) and share QDSMCInitializeKe /
QDSMCUpdateTe, so the comparison isolates the transport operator.

Usage: python3 run_energy_bakeoff.py [--ncells 64 128] [--nsteps 64]
Outputs bakeoff_out/<tag>.npz + a markdown table on stdout.
"""

import argparse
import pathlib
import re
import subprocess
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--ncells", type=int, nargs="+", default=[64, 128])
parser.add_argument("--nsteps", type=int, default=64)
parser.add_argument("--modes", nargs="+", default=["translate", "rotate"])
parser.add_argument("--outdir", default="bakeoff_out")
args = parser.parse_args()

HERE = pathlib.Path(__file__).parent
OUT = HERE / args.outdir
OUT.mkdir(exist_ok=True)

ARMS = {
    "markers": ["--transport-op", "markers", "--advance", "pc", "--grad-deposit", "1"],
    "grid-rkf45": ["--transport-op", "grid", "--fd-time", "rkf45"],
    "grid-ssprk2": ["--transport-op", "grid", "--fd-time", "ssprk2"],
}

perf_re = re.compile(r"\[harness\] perf: ([0-9.]+) s for (\d+) steps")

K_PER_EV = 11604.5


def l2(a, b):
    d = (a - b)[:-1, :-1]  # drop duplicated periodic nodal row/col
    return float(np.sqrt(np.mean(d**2)))


def rel_l2(d):
    """relL2 vs exact (translate) or vs the initial blob (rotate: one probe
    period returns the pattern; here we use the shifted/initial exact)."""
    mode = str(d["mode"])
    L = float(d["L"])
    xb, zb, sb, amp = d["blob"]
    t = float(d["tfinal"])
    n = d["te_final"].shape[0]
    x = np.linspace(0.0, L, n)
    xg, zg = np.meshgrid(x, x, indexing="ij")

    def wrap_blob(x0, z0):
        dxp = (xg - x0 + 0.5 * L) % L - 0.5 * L
        dzp = (zg - z0 + 0.5 * L) % L - 0.5 * L
        return amp * np.exp(-(dxp**2 + dzp**2) / (2.0 * sb**2))

    if mode == "translate":
        v = float(d["vmax"]) / np.sqrt(2.0)
        te_exact = d["Te0_eV"] * K_PER_EV * (1.0 + wrap_blob(xb + v * t, zb + v * t))
    else:
        # rotate: exact = rotated blob about the box center
        if "rotate" not in d:
            return np.nan
        om = float(np.atleast_1d(d["rotate"])[-1])  # last entry = omega
        c = 0.5 * L
        th = om * t
        x0 = c + (xb - c) * np.cos(th) - (zb - c) * np.sin(th)
        z0 = c + (xb - c) * np.sin(th) + (zb - c) * np.cos(th)
        te_exact = d["Te0_eV"] * K_PER_EV * (1.0 + wrap_blob(x0, z0))
    return l2(d["te_final"], te_exact) / (d["Te0_eV"] * K_PER_EV)


rows = []
for mode in args.modes:
    for N in args.ncells:
        for arm, extra in ARMS.items():
            tag = f"{mode}_{N}_{arm}"
            out = OUT / tag
            cmd = [
                sys.executable,
                str(HERE / "qdsmc_advection_test.py"),
                "--mode",
                mode,
                "--ncell",
                str(N),
                "--nsteps",
                str(args.nsteps),
                "--out",
                str(out),
            ] + extra
            print(f"--- {tag}: {' '.join(cmd[1:])}", flush=True)
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(res.stdout[-2000:], res.stderr[-2000:])
                rows.append((mode, N, arm, np.nan, np.nan, np.nan))
                continue
            m = perf_re.search(res.stdout)
            wall = float(m.group(1)) / int(m.group(2)) * 1e3 if m else np.nan
            d = np.load(str(out) + ".npz")
            try:
                rel = rel_l2(d)
            except Exception:
                rel = np.nan
            if "te_sum0" in d and "te_sum1" in d:
                drift = (float(d["te_sum1"]) - float(d["te_sum0"])) / float(
                    d["te_sum0"]
                )
            else:
                drift = np.nan
            rows.append((mode, N, arm, rel, drift, wall))
            print(
                f"    relL2={rel:.4e} sum(Te) drift={drift:.3e} {wall:.2f} ms/step",
                flush=True,
            )

print("\n# Energy-equation transport bake-off\n")
print("| mode | N | arm | relL2 | sum(Te) drift | ms/step |")
print("|---|---|---|---|---|---|")
for mode, N, arm, rel, drift, wall in rows:
    print(f"| {mode} | {N} | {arm} | {rel:.4e} | {drift:.3e} | {wall:.2f} |")

# order read per arm (translate mode has the exact solution)
if "translate" in args.modes and len(args.ncells) >= 2:
    print("\n| arm | order (translate) |")
    print("|---|---|")
    for arm in ARMS:
        es = [r[3] for r in rows if r[0] == "translate" and r[2] == arm]
        Ns = [r[1] for r in rows if r[0] == "translate" and r[2] == arm]
        if len(es) >= 2 and all(np.isfinite(es)):
            sl = np.log(es[0] / es[-1]) / np.log(Ns[-1] / Ns[0])
            print(f"| {arm} | {sl:.2f} |")
