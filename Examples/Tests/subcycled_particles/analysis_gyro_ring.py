#!/usr/bin/env python3
#
# Analysis of the subcycled gyro-ring test.
#
# A single alpha particle gyrates for exactly one gyroperiod, covered by 10
# global steps, in a uniform out-of-plane field B = 1 T (in y). It is pushed
# by the SubcycledParticleContainer (~21 subcycles per step) with
# orbit-averaged deposition. Checks:
#
# 1. Charge conservation of the orbit average at every step: the deposited
#    charge density integrates to exactly q*w (the per-step subcycle weights
#    sum to one).
# 2. The gyroperiod-summed orbit-averaged current integrates to ~zero (closed
#    orbit), while the integral of |J| equals q*w*v (the speed is constant).
# 3. The gyroperiod-summed charge sits on an annulus of radius r_L about the
#    gyrocenter.

import glob
import os
import sys

import numpy as np
import yt

# Physical parameters (must match the input deck)
q_alpha = 3.20435324e-19
m_alpha = 6.64465751e-27
w_p = 1.0e6
v_p = 1.0e6
B0 = 1.0
r_L = m_alpha * v_p / (q_alpha * B0)

last_plotfile = sys.argv[1]
plotfiles = sorted(
    glob.glob(os.path.join(os.path.dirname(last_plotfile), "diag1??????"))
)
# Skip the initial (step 0) output: no orbit average has been deposited yet
plotfiles = [pf for pf in plotfiles if not pf.endswith("000000")]
n_steps = len(plotfiles)
print(f"found {n_steps} step outputs")
assert n_steps == 10, "expected 10 step outputs"

Q_exact = q_alpha * w_p
I_scale = q_alpha * w_p * v_p

rho_sum = jx_sum = jz_sum = None
for pf in plotfiles:
    ds = yt.load(pf)
    ad = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    dx = (ds.domain_right_edge - ds.domain_left_edge) / ds.domain_dimensions
    cell_vol = float(dx[0]) * float(dx[1])

    rho = np.squeeze(ad["boxlib", "rho"].to_ndarray())
    jx = np.squeeze(ad["boxlib", "jx"].to_ndarray())
    jy = np.squeeze(ad["boxlib", "jy"].to_ndarray())
    jz = np.squeeze(ad["boxlib", "jz"].to_ndarray())

    # --- 1. per-step charge conservation of the orbit average -------------
    Q_dep = rho.sum() * cell_vol
    err_Q = abs(Q_dep - Q_exact) / Q_exact
    print(f"{os.path.basename(pf)}: deposited charge rel. err {err_Q:.3e}")
    assert err_Q < 1.0e-10, "orbit-averaged charge is not conserved"

    # Out-of-plane current is zero (no vy)
    net_jy = np.abs(jy).sum() * cell_vol / I_scale
    assert net_jy < 1.0e-12, "out-of-plane current should vanish"

    if rho_sum is None:
        rho_sum = np.zeros_like(rho)
        jx_sum = np.zeros_like(jx)
        jz_sum = np.zeros_like(jz)
    rho_sum += rho / n_steps
    jx_sum += jx / n_steps
    jz_sum += jz / n_steps

# --- 2. full-period currents -------------------------------------------------
# Net current of the closed orbit vanishes (up to the per-step truncated
# final subcycle and the Boris phase error).
net_jx = jx_sum.sum() * cell_vol / I_scale
net_jz = jz_sum.sum() * cell_vol / I_scale
print(f"net in-plane current (normalized): jx {net_jx:.3e}, jz {net_jz:.3e}")
assert abs(net_jx) < 2.0e-2, "net x current of closed orbit too large"
assert abs(net_jz) < 2.0e-2, "net z current of closed orbit too large"

# The integral of the in-plane |J| equals q*w*<|v|> = q*w*v
jmag_int = (np.sqrt(jx_sum**2 + jz_sum**2)).sum() * cell_vol / I_scale
print(f"integral of |J| (normalized): {jmag_int:.6f}")
assert abs(jmag_int - 1.0) < 5.0e-2, "ring current magnitude wrong"

# --- 3. ring geometry ----------------------------------------------------------
x = np.linspace(
    float(ds.domain_left_edge[0]) + 0.5 * float(dx[0]),
    float(ds.domain_right_edge[0]) - 0.5 * float(dx[0]),
    ds.domain_dimensions[0],
)
z = np.linspace(
    float(ds.domain_left_edge[1]) + 0.5 * float(dx[1]),
    float(ds.domain_right_edge[1]) - 0.5 * float(dx[1]),
    ds.domain_dimensions[1],
)
X, Z = np.meshgrid(x, z, indexing="ij")

rho_pos = np.abs(rho_sum)
w_tot = rho_pos.sum()
xc = (rho_pos * X).sum() / w_tot
zc = (rho_pos * Z).sum() / w_tot
print(f"charge centroid: ({xc:.4e}, {zc:.4e}) m (gyrocenter at origin)")
assert abs(xc) < 2.0 * float(dx[0]), "charge centroid off gyrocenter in x"
assert abs(zc) < 2.0 * float(dx[1]), "charge centroid off gyrocenter in z"

R = np.sqrt((X - xc) ** 2 + (Z - zc) ** 2)
r_mean = (rho_pos * R).sum() / w_tot
r_std = np.sqrt((rho_pos * (R - r_mean) ** 2).sum() / w_tot)
print(f"mean ring radius: {r_mean:.6e} m (r_L = {r_L:.6e} m), spread: {r_std:.3e} m")
assert abs(r_mean - r_L) < 1.5 * float(dx[0]), "ring radius does not match r_L"
assert r_std < 2.0 * float(dx[0]), "ring is too diffuse"

print("Subcycled gyro-ring test: all checks passed")
