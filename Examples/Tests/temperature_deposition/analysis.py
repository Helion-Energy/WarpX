#!/usr/bin/env python3
#
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""Check the shape-matched temperature deposition against the injected temperature.

The simulation is a uniform, periodic, field-free Maxwellian proton plasma at
Ti = 100 eV with 4 particles per cell, run for 2 steps under the hybrid-PIC
solver (whose deposit loop hosts the temperature deposition). The plasma is
inert over this horizon, so the rho-weighted domain mean of each deposited
temperature component (Tx/Ty/Tz) must read back the injected temperature.

The 3% tolerance is chosen to separate the current unbiased weighted-variance
estimator, T = [sum_p w_p (v_p - vbar)^2 / W] / (1 - sum_p w_p^2 / W^2) with
w_p the per-node effective weights (particle weight x shape factor) and
W = sum_p w_p, from the estimators that preceded it (measured on this exact
deck, 2D, shape factor 2, 4 ppc, 2 MPI ranks):

  ===========================================  =====================
  estimator                                    mean T / injected T
  ===========================================  =====================
  broken 2D deposit indexing (pre-fix)         garbage: deposits
                                               collapse onto each
                                               box's origin rows and
                                               write out of bounds;
                                               measured Tx/Ty read
                                               0.085 of truth and Tz
                                               reads 3.1e5 x truth
  n/(n-1) reliability correction               0.944-0.957 per
                                               component (0.951
                                               overall, biased low
                                               ~5%; 0.945 on the 1D
                                               deck)
  1/(1 - sum(w^2)/W^2) (current, unbiased)     0.992-1.005 per
                                               component (0.999
                                               overall; 1.003 on the
                                               1D deck)
  ===========================================  =====================

The n/(n-1) form under-corrects because with continuous shape weights the
effective number of samples per node, n_eff = W^2/sum(w^2), is smaller than
the raw contributing-particle count n. With strictly equal effective weights
the two corrections coincide (verified separately to ~1e-11 relative on a
collocated 1-ppc regularly spaced deck).

The step-0 plotfile is not used: the temperature deposition runs inside the
hybrid-PIC deposit loop, so the T_<species> fields are only populated from
step 1 onward (this script analyzes the step-2 dump).
"""

import sys

import numpy as np
import yt
from scipy.constants import e, k

# Injected ion temperature (must match the inputs file), converted to kelvin
T_INJECTED_EV = 100.0
T_injected = T_INJECTED_EV * e / k

# Relative tolerance on the rho-weighted domain mean of each component
TOL_MEAN = 0.03
# Relative tolerance on the spread between components (isotropy)
TOL_ISOTROPY = 0.025

plotfile = sys.argv[1]

ds = yt.load(plotfile)
ad = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)

rho = np.abs(ad["boxlib", "rho_ions"].to_ndarray())
assert rho.sum() > 0.0, "rho_ions is identically zero"

mean_T = {}
for comp in ["Tx_ions", "Ty_ions", "Tz_ions"]:
    T = ad["boxlib", comp].to_ndarray()
    mean_T[comp] = (rho * T).sum() / rho.sum()
    ratio = mean_T[comp] / T_injected
    print(
        f"{comp}: rho-weighted mean = {mean_T[comp]:.6e} K, "
        f"ratio to injected = {ratio:.6f}"
    )
    assert abs(ratio - 1.0) < TOL_MEAN, (
        f"{comp} deviates from the injected temperature by more than "
        f"{TOL_MEAN:.0%}: ratio = {ratio:.6f}"
    )

# Isotropy: the injected distribution is isotropic, so the three deposited
# components must agree with each other
T_vals = np.array(list(mean_T.values()))
spread = (T_vals.max() - T_vals.min()) / T_vals.mean()
print(f"component spread (max-min)/mean = {spread:.6f}")
assert spread < TOL_ISOTROPY, (
    f"Deposited temperature components are not isotropic: "
    f"spread = {spread:.6f} > {TOL_ISOTROPY}"
)

print("Temperature deposition analysis: PASS")
