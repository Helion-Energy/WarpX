#!/usr/bin/env python3
"""Check that each beam's collision rate is n * sigma * v_rel.

A collided ion takes the velocity of a 1 K neutral, so it sits orders of
magnitude below v_rel, and anything still above half of v_rel has not
collided. The survivors decay as exp(-nu t), so nu = -ln(f) / t.

Statistical error on nu is ~0.4-0.5% per beam at 102400 ions; the tolerance of
2.5% is 5 sigma. The failure it guards against is 21% on the high beam.
"""

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy import constants

# must match the input file
m_ion = 4.0 * constants.m_p
mu = 0.5 * m_ion
n_gas = 1.0e20
sigma0 = 1.0e-19
dt = 1.5e-9
max_steps = 108
beams = {"low": 1000.0, "high": 4000.0}

tolerance = 0.025

ts = OpenPMDTimeSeries("diags/diag1/")
iteration = ts.iterations[-1]
assert iteration == max_steps, f"last dump is at step {iteration}"
t = iteration * dt

failures = []
for label, energy in beams.items():
    v0 = np.sqrt(2.0 * energy * constants.e / mu)
    (uz,) = ts.get_particle(
        var_list=["uz"], species=f"ions_{label}", iteration=iteration
    )
    # openPMD momentum is in units of m*c; the beam is non-relativistic to 1e-6
    vz = uz * constants.c
    survivors = np.count_nonzero(vz > 0.5 * v0)
    fraction = survivors / vz.size
    assert 0.0 < fraction < 1.0, f"{label} beam: survivor fraction {fraction}"

    nu_measured = -np.log(fraction) / t
    nu_expected = n_gas * sigma0 * v0
    ratio = nu_measured / nu_expected
    print(
        f"{label:>4} beam, E = {energy:6.0f} eV: {survivors}/{vz.size} survive, "
        f"nu = {nu_measured:.5e} /s against n*sigma*v = {nu_expected:.5e} /s, "
        f"ratio {ratio:.4f}"
    )
    if abs(ratio - 1.0) > tolerance:
        failures.append(f"{label} beam collides at {ratio:.4f} of n*sigma*v_rel")

assert not failures, "; ".join(failures)
