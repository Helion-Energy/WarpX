#!/usr/bin/env python3
#
# Analysis of the RZ subcycled gyro-ring test.
#
# A single alpha particle gyrates for exactly one gyroperiod (10 global
# steps) about a gyrocenter on the axis in a uniform axial field B = 1 T,
# pushed by the SubcycledParticleContainer with orbit-averaged deposition.
# The orbit is a circle of radius r_L centered on r = 0, so the particle
# sits at constant radius with purely azimuthal velocity, and the deposits
# probe the RZ-specific machinery. Checks:
#
# 1. Charge conservation of the orbit average at every step over the
#    cylindrical measure: integral of rho over dV = 2 pi r dr dz equals
#    exactly q*w (per-step subcycle weights sum to one AND the inverse
#    volume scaling of the orbit-averaged deposit is correct).
# 2. The deposited current is a pure ring current: the signed integral of
#    j_theta equals q*w*v_theta (= -q*w*v here), while j_r and j_z
#    integrate to ~zero.
# 3. The gyroperiod-summed charge sits on a thin annulus at r = r_L,
#    motionless in z.

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

rho_sum = None
jt_int_sum = 0.0
jr_int_sum = 0.0
jz_int_sum = 0.0
for pf in plotfiles:
    ds = yt.load(pf)
    ad = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    nr, nz = int(ds.domain_dimensions[0]), int(ds.domain_dimensions[1])
    rlo = float(ds.domain_left_edge[0])
    zlo = float(ds.domain_left_edge[1])
    dr = (float(ds.domain_right_edge[0]) - rlo) / nr
    dz = (float(ds.domain_right_edge[1]) - zlo) / nz

    # cylindrical volume element per cell, dV = 2 pi r_i dr dz
    r_cc = rlo + (np.arange(nr) + 0.5) * dr
    dV = (2.0 * np.pi * r_cc * dr * dz)[:, np.newaxis]

    rho = np.squeeze(ad["boxlib", "rho"].to_ndarray())
    jr = np.squeeze(ad["boxlib", "jr"].to_ndarray())
    jt = np.squeeze(ad["boxlib", "jt"].to_ndarray())
    jz = np.squeeze(ad["boxlib", "jz"].to_ndarray())

    # --- 1. per-step charge conservation over the cylindrical measure -----
    Q_dep = (rho * dV).sum()
    err_Q = abs(Q_dep - Q_exact) / Q_exact
    print(f"{os.path.basename(pf)}: deposited charge rel. err {err_Q:.3e}")
    assert err_Q < 1.0e-8, "orbit-averaged charge is not conserved in RZ"

    jt_int_sum += (jt * dV).sum() / n_steps
    jr_int_sum += (jr * dV).sum() / n_steps
    jz_int_sum += (jz * dV).sum() / n_steps

    if rho_sum is None:
        rho_sum = np.zeros_like(rho)
    rho_sum += rho / n_steps

# --- 2. ring current -----------------------------------------------------
# v is purely azimuthal (v_theta = -v): the signed j_theta integral is
# -q*w*v, while j_r and j_z vanish (up to the Boris polygon jitter and the
# per-step truncated final subcycle).
jt_norm = jt_int_sum / I_scale
jr_norm = jr_int_sum / I_scale
jz_norm = jz_int_sum / I_scale
print(
    f"normalized current integrals: jt {jt_norm:.6f}, jr {jr_norm:.3e}, jz {jz_norm:.3e}"
)
assert abs(jt_norm + 1.0) < 5.0e-2, "azimuthal ring current magnitude/sign wrong"
assert abs(jr_norm) < 2.0e-2, "net radial current should vanish"
assert abs(jz_norm) < 2.0e-2, "net axial current should vanish"

# --- 3. ring geometry ----------------------------------------------------
nr, nz = rho_sum.shape
r_cc = rlo + (np.arange(nr) + 0.5) * dr
z_cc = zlo + (np.arange(nz) + 0.5) * dz
R = r_cc[:, np.newaxis] * np.ones((1, nz))
Z = np.ones((nr, 1)) * z_cc[np.newaxis, :]

w_cell = np.abs(rho_sum) * (2.0 * np.pi * R * dr * dz)
w_tot = w_cell.sum()
r_mean = (w_cell * R).sum() / w_tot
z_mean = (w_cell * Z).sum() / w_tot
r_std = np.sqrt((w_cell * (R - r_mean) ** 2).sum() / w_tot)
print(
    f"ring radius: {r_mean:.6e} m (r_L = {r_L:.6e} m), spread {r_std:.3e} m, z centroid {z_mean:.3e} m"
)
assert abs(r_mean - r_L) < 1.5 * dr, "ring radius does not match r_L"
assert r_std < 2.0 * dr, "ring is too diffuse in r"
assert abs(z_mean) < 2.0 * dz, "ring drifted in z"

print("RZ subcycled gyro-ring test: all checks passed")
