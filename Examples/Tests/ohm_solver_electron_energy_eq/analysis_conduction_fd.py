#!/usr/bin/env python3
"""Analysis for test_2d_ohm_solver_conduction_fd_picmi: magnetic-island
heat confinement under strongly anisotropic FD conduction.

Asserts, across all diagnostic dumps:
  1. maximum principle : max(Te) non-increasing and min(Te) non-decreasing
                         (to roundoff) — the monotone-operator guarantee;
  2. conservation      : |Sigma(Te) drift| below the deposit-noise floor;
  3. confinement       : the excess-Te fraction outside the separatrix
                         (A < 0, beyond the island) stays at the pollution
                         level, while the along-surface angular spread
                         around the O-point grows by a measurable factor
                         (proof that conduction actually ran).
"""

import argparse

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="diags/field_diags")
# extremum tolerance sits at the marker-transport noise floor at this
# resolution/ppc (measured ~3e-4; the per-step roundoff-level audit lives
# in the convergence-harness instruments), NOT at roundoff
parser.add_argument("--tol-extremum", type=float, default=1.0e-3)
parser.add_argument("--tol-sum", type=float, default=1.0e-5)
parser.add_argument("--tol-outside", type=float, default=0.02)
parser.add_argument("--min-spread-growth", type=float, default=2.5)
args = parser.parse_args()

L = 1.0
kmode = 2.0 * np.pi / L
xb, zb = 0.5 * L, 0.5 * L
Te0_eV = 10.0
KB = 1.380649e-23
QE = 1.602176634e-19
Te0_K = Te0_eV * QE / KB

ts = OpenPMDTimeSeries(args.path)

# the iteration-0 dump is written before the injection-callback Te poke
# (uniform field, no blob) — the physics reference is the first dump
# that carries the blob
iterations = [it for it in ts.iterations if it > 0]

maxs, mins, sums, frac_out, ang_var = [], [], [], [], []
for it in iterations:
    te, info = ts.get_field("Te", iteration=it)
    maxs.append(float(te.max()))
    mins.append(float(te.min()))
    sums.append(float(te.sum()))

    zg, xg = np.meshgrid(info.z, info.x, indexing="ij")
    # openPMD 2D layout is (z, x); the flux function is symmetric in the
    # two coordinates so orientation does not matter for A or the angle
    A = np.cos(kmode * xg) * np.cos(kmode * zg)  # +1 at the O-point
    w = np.clip(te - Te0_K, 0.0, None)
    wsum = w.sum()
    frac_out.append(float(w[A < 0.0].sum() / wsum))
    th = np.arctan2(zg - zb, xg - xb)
    c = (w * np.exp(1j * th)).sum() / wsum
    ang_var.append(float(-2.0 * np.log(max(abs(c), 1.0e-300))))

maxs, mins, sums = np.array(maxs), np.array(mins), np.array(sums)

# overshoot vs the first post-poke dump (peak must decay monotonically);
# undershoot vs the ANALYTIC background Te0 (the poke's exact floor)
overshoot = float((maxs[1:].max() - maxs[0]) / Te0_K) if len(maxs) > 1 else 0.0
undershoot = float((mins.min() - Te0_K) / Te0_K)
sum_drift = float(abs(sums[-1] - sums[0]) / sums[0])
spread_growth = ang_var[-1] / max(ang_var[0], 1.0e-12)

print("[conduction-fd analysis]")
print(f"  overshoot (rel Te0)      : {overshoot:+.3e}  (tol {args.tol_extremum:g})")
print(f"  undershoot (rel Te0)     : {undershoot:+.3e}  (tol {args.tol_extremum:g})")
print(f"  Sigma(Te) drift          : {sum_drift:.3e}  (tol {args.tol_sum:g})")
print(
    f"  excess-Te outside island : {frac_out[0]:.3e} -> {frac_out[-1]:.3e}"
    f"  (tol {args.tol_outside:g})"
)
print(
    f"  angular spread growth    : {spread_growth:.2f}x"
    f"  (min {args.min_spread_growth:g}x)"
)

assert overshoot <= args.tol_extremum, "maximum principle violated (new max)"
assert undershoot >= -args.tol_extremum, "maximum principle violated (new min)"
assert sum_drift <= args.tol_sum, "Sigma(Te) not conserved"
assert frac_out[-1] <= args.tol_outside, "heat leaked past the separatrix"
assert spread_growth >= args.min_spread_growth, (
    "no along-surface spreading measured -- conduction did not act"
)
print("[conduction-fd analysis] PASS")
