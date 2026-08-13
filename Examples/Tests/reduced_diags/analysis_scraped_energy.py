#!/usr/bin/env python3
#
# Analysis for test_2d_reduced_diags_scraped_energy_picmi: the deck is
# constructed so E = 0 identically (see the deck header), so the live
# kinetic energy plus the cumulative scraped energy must be constant in
# time, and the per-step increments must telescope into the cumulative
# columns exactly.

import sys

import numpy as np

live = np.loadtxt("diags/reducedfiles/live_energy.txt")
scraped = np.loadtxt("diags/reducedfiles/scraped_energy.txt")

# live_energy columns: step, time, total(J), ions(J), total_mean, ions_mean
ke_live = live[:, 2]

# scraped_energy columns: step, time,
#   outflow_energy_ions_<boundary>(J) ... , integrated_outflow_ions_<boundary>(J) ...
n_bnd = (scraped.shape[1] - 2) // 2
inc = scraped[:, 2 : 2 + n_bnd]
cum = scraped[:, 2 + n_bnd : 2 + 2 * n_bnd]

# Rows of the two files must correspond to the same steps
assert live.shape[0] == scraped.shape[0]
assert np.array_equal(live[:, 0], scraped[:, 0])

# 1) Increments telescope into the cumulative columns
cum_from_inc = np.cumsum(inc, axis=0)
err_tel = np.max(np.abs(cum_from_inc + (cum[0] - inc[0]) - cum))
print(f"telescoping error = {err_tel:.3e}")
assert err_tel < 1e-12 * max(cum.max(), 1.0)

# 2) A non-trivial amount of energy must actually have been scraped
ke0 = ke_live[0]
scraped_total = cum[-1].sum()
frac = scraped_total / ke0
print(f"scraped fraction of initial KE = {frac:.4f}")
assert frac > 0.01, "test is vacuous: (almost) nothing was scraped"

# 3) Conservation: KE_live + KE_scraped_cum constant at every step
total = ke_live + cum.sum(axis=1)
err_cons = np.max(np.abs(total - total[0])) / total[0]
print(f"conservation error = {err_cons:.3e}")
assert err_cons < 1e-10

# 4) With v_drift >> v_th along +z, virtually everything exits at z_hi.
#    zhi is the last domain boundary column (boundary order: xlo, xhi,
#    zlo, zhi[, eb]); tolerate an eb column that never fills.
zhi_col = 3
lo_leak = cum[-1].sum() - cum[-1][zhi_col]
print(f"non-zhi share = {lo_leak / scraped_total:.3e}")
assert lo_leak / scraped_total < 1e-3

print("scraped-energy ledger checks passed")
sys.exit(0)
