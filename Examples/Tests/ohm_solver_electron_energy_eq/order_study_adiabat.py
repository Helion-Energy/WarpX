#!/usr/bin/env python3
#
# --- Temporal order-of-accuracy study for the theta-implicit hybrid scheme
# --- with the QDSMC electron-energy equation (not a CI test; run manually).
#
# Runs the adiabatic-compression deck at dt, dt/2, dt/4 with the final time
# held fixed, on the same grid with identical (serialized) initial conditions,
# and measures the self-convergence of the final electron-temperature field:
#
#     e(dt) = || T_e(dt) - T_e(dt/2) ||_2 ,   order = log2( e(dt) / e(dt/2) ).
#
# At theta = 0.5 the scheme is time-centered throughout (implicit-midpoint
# particle push, midpoint characteristic for the QDSMC markers, midpoint
# coefficient states) and the dt-error is second order. At theta = 1
# (backward Euler biasing) the slope drops to 1.
#
# Interpretation caveat: on a FIXED grid the measured error is the composite
#     e(dt) ~= A dt^2 + B dx^p / dt,
# where the second term is the per-step marker remap (gather/scatter)
# smoothing accumulated over T/dt steps (Albright et al., Phys. Plasmas 9,
# 1898 (2002)). Pointwise slopes therefore approach 2 only in the
# dt-dominated bracket (large --dt-scales such that A s^3 >> B) and saturate
# toward the remap floor as dt shrinks; refining dt alone below the floor
# HELPS NOTHING (and joint dt ~ dx self-convergence is unmeasurable at
# fixed particles/cell, since each resolution carries a different PIC noise
# realization). Representative theta = 0.5 measurement on the shipped deck
# (dt-scales 10/5/2.5, which divide the 60-step horizon exactly -- pick
# scales that keep total_steps integral or the runs end at different times):
# slope 1.6 in the [10, 5] bracket falling to 0.8 in [5, 2.5], with the
# 3-point (A, B) fit consistent with a second-order A-term; theta = 1 gives
# slope ~0.9 throughout with ~5x larger absolute error at matched dt.
#
# Realization averaging (marker-noise floor mitigation): with --seeds the
# study runs each (theta, dt-scale) case once per seed (warpx.random_seed on
# the particle load) and AVERAGES the final fields across seeds before taking
# successive differences -- the stage's noise injection decorrelates across
# loadings (floor drops ~ 1/sqrt(R)) while the coherent dt-error is shared.
# Judge the energy-equation dynamics on rho (--field rho): the closure ladder
# established rho as the clean second-order control metric.
#
# Usage:  python3 order_study_adiabat.py [--np 2] [--dt-scales 10 5 2.5]
#         python3 order_study_adiabat.py --thetas 0.5 --dt-scales 4 2 1 0.5 0.25 \
#             --seeds 101 102 103 104 105 106 107 108 --field rho \
#             [--segregated] [--low-pass 2 4 8] [--jobs 4]

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--np", type=int, default=2, help="MPI ranks per run")
parser.add_argument(
    "--dt-scales",
    type=float,
    nargs="+",
    default=[1.0, 0.5, 0.25],
    help="successive dt scalings (each should halve the previous)",
)
parser.add_argument(
    "--thetas", type=float, nargs="+", default=[0.5, 1.0], help="theta values to test"
)
parser.add_argument(
    "--workdir", default="order_study_runs", help="scratch directory for the runs"
)
parser.add_argument(
    "--no-energy-eq",
    action="store_true",
    help="bisect: run with the algebraic closure and compare rho instead of Te",
)
parser.add_argument(
    "--refine-grid",
    action="store_true",
    help="refine the grid together with dt (dt ~ dx protocol): each case gets "
    "--grid-scale dt_scales[0]/s, and fields are compared on the coarsest "
    "grid by stride subsampling (the fields are nodal, so refined-grid nodes "
    "coincide with coarse ones)",
)
parser.add_argument(
    "--segregated",
    action="store_true",
    help="run the implicit scheme with the segregated midpoint-iterated "
    "QDSMC solve (implicit_evolve.qdsmc_segregated_solve)",
)
parser.add_argument(
    "--seeds",
    type=int,
    nargs="+",
    default=None,
    help="realization ensemble: run each case once per seed and average the "
    "final fields across seeds before the ladder (default: one unseeded run)",
)
parser.add_argument(
    "--field",
    default=None,
    help="field to judge the ladder on (default: rho for --no-energy-eq, "
    "else Te; the ORDER_FIELD env var is honored as a fallback)",
)
parser.add_argument(
    "--low-pass",
    type=int,
    nargs="+",
    default=None,
    help="also report the ladder restricted to Fourier modes with all "
    "|k-index| <= K, for each given K (box-scale coherent response)",
)
parser.add_argument(
    "--jobs",
    type=int,
    default=1,
    help="number of cases to run concurrently (each with --np ranks)",
)
args = parser.parse_args()

deck = (
    Path(__file__).parent / "inputs_test_2d_ohm_solver_electron_energy_adiabat_picmi.py"
)
workdir = Path(args.workdir).absolute()


def run_case(theta, dt_scale, seed=None):
    tag = f"theta{theta}_s{dt_scale}" + (f"_seed{seed}" if seed is not None else "")
    case = workdir / tag
    if case.exists():
        shutil.rmtree(case)
    case.mkdir(parents=True)
    cmd = [
        "mpiexec",
        "-n",
        str(args.np),
        sys.executable,
        str(deck),
        "--test",
        "--implicit",
        "--nlsolver",
        "picard",
        "--theta",
        str(theta),
        "--dt-scale",
        str(dt_scale),
    ]
    if args.refine_grid:
        gs = args.dt_scales[0] / dt_scale
        assert abs(gs - round(gs)) < 1e-12, (
            "dt-scales must be power-of-two nested for --refine-grid"
        )
        cmd += ["--grid-scale", str(int(round(gs)))]
    if args.no_energy_eq:
        cmd.append("--no-energy-eq")
    if args.segregated:
        cmd.append("--segregated")
    if seed is not None:
        cmd += ["--seed", str(seed)]
    with open(case / "run.log", "w") as log:
        subprocess.run(cmd, cwd=case, stdout=log, stderr=subprocess.STDOUT, check=True)
    return case


def final_field(case):
    from openpmd_viewer import OpenPMDTimeSeries

    ts = OpenPMDTimeSeries(str(case / "diags" / "field_diags"))
    name = args.field or os.environ.get(
        "ORDER_FIELD", "rho" if args.no_energy_eq else "Te"
    )
    f, _ = ts.get_field(name, iteration=ts.iterations[-1])
    return f


def low_pass(f, kmax):
    """Restrict to the box-scale modes: zero every Fourier mode with any
    wavenumber index above kmax (axis-agnostic; the fields are periodic)."""
    fk = np.fft.fftn(f)
    mask = np.ones_like(fk, dtype=bool)
    for ax, n in enumerate(f.shape):
        k = np.abs(np.fft.fftfreq(n, d=1.0 / n))
        mask &= np.expand_dims(k <= kmax, tuple(i for i in range(f.ndim) if i != ax))
    return np.real(np.fft.ifftn(np.where(mask, fk, 0.0)))


def ladder(fields, label):
    errs = [np.sqrt(np.mean((a - b) ** 2)) for a, b in zip(fields[:-1], fields[1:])]
    print(f"  -- {label} --")
    for i, e in enumerate(errs):
        line = f"  e(dt*{args.dt_scales[i]}) = {e:.6e}"
        if i > 0 and errs[i] > 0:
            line += f"   observed order = {np.log2(errs[i - 1] / errs[i]):.2f}"
        print(line)


seeds = args.seeds if args.seeds is not None else [None]

for theta in args.thetas:
    print(f"\n=== theta = {theta}{' (segregated)' if args.segregated else ''} ===")
    fields = []
    for s in args.dt_scales:
        # One run per seed; realization-average the final fields (the noise
        # floor drops ~ 1/sqrt(R); the coherent dt-error survives averaging).
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            cases = list(pool.map(lambda sd: run_case(theta, s, sd), seeds))
        fields.append(np.mean([final_field(c) for c in cases], axis=0))
        print(f"  ran dt-scale {s} ({len(seeds)} realization(s))")
    if args.refine_grid:
        # Stride-subsample every field onto the coarsest case's nodes.
        base_shape = fields[0].shape
        sub = []
        for f in fields:
            # Periodic nodal arrays hold N values per N cells, so the
            # refined nodes at stride n//b coincide with the coarse nodes.
            strides = tuple(n // b for n, b in zip(f.shape, base_shape))
            sub.append(f[:: strides[0], :: strides[1]])
        fields = sub
    ladder(fields, "full field")
    for kmax in args.low_pass or []:
        ladder([low_pass(f, kmax) for f in fields], f"low-pass |k| <= {kmax}")
