#!/usr/bin/env python3
"""Calibrated CGL isotropization under the bi-Maxwellian closure.

A uniform static bi-Maxwellian (p_perp = 2 p_par initially, u = 0) is
exactly inert in every flux and work term, so the only dynamics is the
ion-ion relaxation exchange, under which Delta = p_perp - p_par decays
at exactly nu_iso (the effective pressure and density are conserved,
so the ODE is linear) and U_par + U_perp is conserved to round-off.
The measured per-run log-decay must match the theta-discrete
amplification computed from the Braginskii rate re-derived here from
first principles -- this independently pins the implemented collision
coefficient, the exchange antisymmetry, and the theta weighting.
"""

import sys

import numpy as np
import yt

from warpx_constants import epsilon_0, m_p
from warpx_constants import elementary_charge as q_e

yt.set_log_level(50)

# Must match the input deck.
N0 = 1.0e20
TI_EV = 100.0
P0 = N0 * TI_EV * q_e
LNL = 10.0
KT = P0 / N0
NU = (np.sqrt(2.0) * q_e**4 * LNL * N0 /
      (12.0 * np.pi**1.5 * np.sqrt(m_p) * epsilon_0**2 * KT**1.5))
DT = 0.1 / NU
THETA = 0.5
NSTEPS = 40

GROWTH = (1.0 - (1.0 - THETA) * NU * DT) / (1.0 + THETA * NU * DT)


def load(path):
    ds = yt.load(path)
    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    return {
        f: np.asarray(grid["boxlib", f]).squeeze()
        for f in ("implicit_mhd_ion_parallel_energy",
                  "implicit_mhd_ion_perp_energy")
    }


initial = load(sys.argv[1])
final = load(sys.argv[2])


def delta(state):
    p_par = 2.0 * state["implicit_mhd_ion_parallel_energy"]
    p_perp = state["implicit_mhd_ion_perp_energy"]
    return np.mean(p_perp - p_par)


def total(state):
    return np.mean(state["implicit_mhd_ion_parallel_energy"] +
                   state["implicit_mhd_ion_perp_energy"])


measured_log_decay = np.log(delta(final) / delta(initial))
expected_log_decay = NSTEPS * np.log(GROWTH)
relative_error = abs(measured_log_decay - expected_log_decay) / \
    abs(expected_log_decay)
conservation = abs(total(final) - total(initial)) / total(initial)
uniformity = np.max(np.abs(
    final["implicit_mhd_ion_perp_energy"] -
    np.mean(final["implicit_mhd_ion_perp_energy"])
)) / total(initial)

print(f"nu_iso (first principles)  {NU:.6e} 1/s")
print(f"initial anisotropy Delta   {delta(initial):.6e}")
print(f"final anisotropy Delta     {delta(final):.6e}")
print(f"measured log decay         {measured_log_decay:.10f}")
print(f"expected log decay         {expected_log_decay:.10f}")
print(f"relative error             {relative_error:.3e}")
print(f"U_par + U_perp drift       {conservation:.3e}")
print(f"final spatial nonuniformity {uniformity:.3e}")

assert delta(initial) > 0.9 * P0 * (3.0 / 5.0)
assert relative_error < 1.0e-5, (
    f"isotropization rate off by {relative_error:.3e}"
)
assert conservation < 1.0e-12
assert uniformity < 1.0e-12

print("PASS")
