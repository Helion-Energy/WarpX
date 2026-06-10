#!/usr/bin/env python3
#
# --- Analysis script for the global A recovery z-pinch test. The azimuthal
# --- magnetic field in the vacuum region surrounding the current-carrying
# --- plasma column is recovered from the global vector Poisson solve and
# --- must follow the analytic B_theta = mu0*I/(2*pi*r) profile.

import sys

import numpy as np
import yt
from scipy.constants import mu_0

yt.funcs.mylog.setLevel(50)

# Test parameters, must match the ones in the input script
LR = 0.5
R_p = 0.4 * LR
B_a = np.sqrt(2.0 * mu_0 * 1e20 * 1.602176634e-19 * 20.0)
I_total = 2.0 * np.pi * R_p * B_a / mu_0

fn = sys.argv[1] if len(sys.argv) > 1 else "diags/diag1000010"

ds = yt.load(fn)
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

Br = data["boxlib", "Br"].v.squeeze()
Bt = data["boxlib", "Bt"].v.squeeze()
Bz = data["boxlib", "Bz"].v.squeeze()
Er = data["boxlib", "Er"].v.squeeze()
Et = data["boxlib", "Et"].v.squeeze()
Ez = data["boxlib", "Ez"].v.squeeze()

NR, NZ = Bt.shape
dr = LR / NR
r = (np.arange(NR) + 0.5) * dr

# Average over the periodic z direction
Bt_profile = Bt.mean(axis=1)

# In the vacuum region (away from the plasma edge and the blend seam) the
# recovered field must follow the analytic 1/r profile of the enclosed
# current. Without the global A recovery the vacuum field has no mechanism
# to maintain this profile once the local Faraday update acts on the
# unreliable vacuum E field.
i_vac = np.where((r > 1.5 * R_p) & (r < 0.9 * LR))[0]
Bt_analytic = mu_0 * I_total / (2.0 * np.pi * r[i_vac])
rel_err = np.abs(Bt_profile[i_vac] - Bt_analytic) / Bt_analytic

print(f"max relative error of vacuum B_theta: {rel_err.max():.3e}")
assert rel_err.max() < 0.05, (
    f"Vacuum B_theta deviates from the analytic mu0*I/(2*pi*r) profile by "
    f"{rel_err.max():.3e} (tolerance 5e-2)"
)

# The z-pinch has no Br or Bz; the recovered field must not introduce
# spurious components in the vacuum region
B_spurious = max(np.abs(Br[i_vac, :]).max(), np.abs(Bz[i_vac, :]).max())
print(f"max spurious |Br|, |Bz| in vacuum: {B_spurious:.3e} (B_a = {B_a:.3e})")
assert B_spurious < 0.01 * B_a, (
    f"Spurious Br/Bz of {B_spurious:.3e} found in the vacuum region"
)

# The configuration is quasi-static, so the reconstructed vacuum E field
# (E = -dA/dt - grad(Pe)/rho) must remain small compared to the
# characteristic v_A*B_a field of the plasma
vA = B_a / np.sqrt(mu_0 * 1e20 * 1.67262192e-27)
E0 = vA * B_a
E_vac = max(
    np.abs(Er[i_vac, :]).max(),
    np.abs(Et[i_vac, :]).max(),
    np.abs(Ez[i_vac, :]).max(),
)
print(f"max vacuum |E|: {E_vac:.3e} (vA*B_a = {E0:.3e})")
assert E_vac < 0.05 * E0, (
    f"Vacuum E field of {E_vac:.3e} is not small compared to vA*B_a = {E0:.3e}"
)

print("Global A recovery analysis: all checks passed.")
