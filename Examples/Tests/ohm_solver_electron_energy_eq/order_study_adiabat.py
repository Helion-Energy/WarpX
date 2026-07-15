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
# Usage:  python3 order_study_adiabat.py [--np 2] [--dt-scales 10 5 2.5]

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import os

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
args = parser.parse_args()

deck = Path(__file__).parent / "inputs_test_2d_ohm_solver_electron_energy_adiabat_picmi.py"
workdir = Path(args.workdir).absolute()


def run_case (theta, dt_scale):
    case = workdir / f"theta{theta}_s{dt_scale}"
    if case.exists():
        shutil.rmtree(case)
    case.mkdir(parents=True)
    cmd = [
        "mpiexec", "-n", str(args.np), sys.executable, str(deck),
        "--test", "--implicit", "--nlsolver", "picard",
        "--theta", str(theta), "--dt-scale", str(dt_scale),
    ]
    if args.refine_grid:
        gs = args.dt_scales[0] / dt_scale
        assert abs(gs - round(gs)) < 1e-12, "dt-scales must be power-of-two nested for --refine-grid"
        cmd += ["--grid-scale", str(int(round(gs)))]
    if args.no_energy_eq:
        cmd.append("--no-energy-eq")
    with open(case / "run.log", "w") as log:
        subprocess.run(cmd, cwd=case, stdout=log, stderr=subprocess.STDOUT, check=True)
    return case


def final_Te (case):
    from openpmd_viewer import OpenPMDTimeSeries

    ts = OpenPMDTimeSeries(str(case / "diags" / "field_diags"))
    name = os.environ.get("ORDER_FIELD", "rho" if args.no_energy_eq else "Te")
    f, _ = ts.get_field(name, iteration=ts.iterations[-1])
    return f


for theta in args.thetas:
    print(f"\n=== theta = {theta} ===")
    fields = []
    for s in args.dt_scales:
        case = run_case(theta, s)
        fields.append(final_Te(case))
        print(f"  ran dt-scale {s}")
    errs = []
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
    for a, b in zip(fields[:-1], fields[1:]):
        errs.append(np.sqrt(np.mean((a - b) ** 2)))
    for i, e in enumerate(errs):
        line = f"  e(dt*{args.dt_scales[i]}) = {e:.6e}"
        if i > 0 and errs[i] > 0:
            line += f"   observed order = {np.log2(errs[i-1]/errs[i]):.2f}"
        print(line)
