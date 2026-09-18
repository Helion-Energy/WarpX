#!/usr/bin/env python3
"""Check Opal energy sharing for MCC electron-impact ionization.

A cold, monoenergetic beam at E0 ionizes a neutral background once per
electron at most, so every electron in the final dump either still carries E0
or carries a share of dE = E0 - E_ioniz given to it by one ionization event.
The two shares of an event are W for the ejected electron and dE - W for the
incident one, with W sampled from the Opal secondary-electron distribution
truncated at dE/2, whose CDF is

    F(W) = arctan(W/w) / arctan(dE/2w).

Mapping every collided electron through min(E, dE - E) recovers the W of its
event whichever of the two it is, so the whole collided population tests that
one distribution, and the mapping is continuous at dE/2 rather than sorting
the two roles by a threshold they could straddle. Equal sharing, the default,
puts every collided electron at exactly dE/2 and so fails this outright.

The energy sum is checked as well, since a distribution can be right in shape
while losing or inventing energy per event.
"""

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy import constants, stats

# must match the input file
E_elec = 200.0  # eV, energy the initial beam momentum was set from
E_ioniz = 20.0  # eV
opal_w = 10.0  # eV

mc2 = constants.m_e * constants.c**2 / constants.e  # eV

# The beam momentum is set as u = sqrt(2*E_elec*q_e/m_e)/c, so its kinetic
# energy is marginally below E_elec; the collisions share out what the
# momentum actually carries, not the round number it was derived from.
u0 = np.sqrt(2.0 * E_elec * constants.e / constants.m_e) / constants.c
E0 = (np.sqrt(1.0 + u0**2) - 1.0) * mc2
dE = E0 - E_ioniz

ts = OpenPMDTimeSeries("diags/diag1/")
iteration = ts.iterations[-1]

ux, uy, uz, w = ts.get_particle(
    var_list=["ux", "uy", "uz", "w"], species="electrons", iteration=iteration
)
w_ions = ts.get_particle(var_list=["w"], species="ions", iteration=iteration)[0]

E = (np.sqrt(1.0 + ux**2 + uy**2 + uz**2) - 1.0) * mc2

# Every ion present was created by an ionization event, and each event also
# created one electron, so the ion count is the event count.
n_events = w_ions.size

# An electron that has not collided is still at E0, far above the dE = 180 eV
# that the most energetic outgoing electron can carry.
collided = E < E0 - 1.0
assert np.allclose(E[~collided], E0, rtol=1.0e-12, atol=0.0), (
    "electrons that should not have collided are not at the initial energy, "
    "so something other than the collisions is changing their energy"
)
assert collided.sum() == 2 * n_events, (
    f"{collided.sum()} electrons lost energy but {n_events} ions were created; "
    "each event must leave exactly two electrons below the beam energy"
)

# Guard against the test passing on a handful of events
assert n_events > 1000, (
    f"only {n_events} ionization events occurred, too few for the "
    "distribution check; raise max_step or the electron density"
)

# Each event puts one electron below dE/2 and the other above it: the ejected
# electron takes the smaller share by construction, and a factor of two lost
# in the sampling would break this before it broke the shape.
below = (E[collided] < 0.5 * dE).sum()
above = (E[collided] > 0.5 * dE).sum()
assert below == n_events and above == n_events, (
    f"{below} collided electrons fall below dE/2 and {above} above it, "
    f"against {n_events} events; the two shares are not paired as expected"
)

# Energy accounting. The weights are uniform, and the electron created by an
# event inherits the weight of the electron that caused it.
assert np.all(w == w[0]), "the electron weights are not uniform"
weight = w[0]
n_initial = E.size - n_events
energy_final = np.sum(E * w)
energy_initial = n_initial * weight * E0
energy_paid = n_events * weight * E_ioniz
energy_error = abs(energy_final + energy_paid - energy_initial) / energy_initial

print(f"ionization events      : {n_events}")
print(f"fraction of beam       : {n_events / n_initial:.3%}")
print(f"energy in electrons    : {energy_final:.10e} eV")
print(f"energy paid to ionize  : {energy_paid:.10e} eV")
print(f"initial energy         : {energy_initial:.10e} eV")
print(f"relative difference    : {energy_error:.3e}")

assert energy_error < 1.0e-10, (
    f"the electron energy plus the ionization cost differs from the initial "
    f"energy by {energy_error:.3e}: the shares do not add up to what is "
    "available"
)

# The two shares of one event map to the same W up to round-off, so the
# empirical CDF of this sample is that of the n_events underlying draws, and
# the Kolmogorov-Smirnov statistic is compared against n_events.
W = np.minimum(E[collided], dE - E[collided])
ks = stats.kstest(W, lambda x: np.arctan(x / opal_w) / np.arctan(0.5 * dE / opal_w))
ks_tolerance = 4.0 / np.sqrt(n_events)

W_mean_theory = (
    0.5 * opal_w * np.log1p((0.5 * dE / opal_w) ** 2) / np.arctan(0.5 * dE / opal_w)
)

print(f"mean ejected energy    : {W.mean():.4f} eV")
print(f"expected               : {W_mean_theory:.4f} eV")
print(f"KS statistic           : {ks.statistic:.4f} (tolerance {ks_tolerance:.4f})")

assert ks.statistic < ks_tolerance, (
    f"the ejected electron energies deviate from the Opal distribution by "
    f"a KS statistic of {ks.statistic:.4f}, above the {ks_tolerance:.4f} "
    "expected from this sample size"
)
assert abs(W.mean() - W_mean_theory) < 0.1 * W_mean_theory, (
    f"the mean ejected energy is {W.mean():.4f} eV against an expected "
    f"{W_mean_theory:.4f} eV"
)
