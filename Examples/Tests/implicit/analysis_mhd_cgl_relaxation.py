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

Modes: "collisional" (pure Braginskii rate), "bounded" (instability
bounds pull a mirror-unstable state to marginal stability), "null"
(B = 0 deck: the Stage B+ magnetization-weighted blend adds the
unmagnetized cell-transit rate v_thi/dz, calibrated the same way).
"""

import sys

import numpy as np
import yt

from warpx_constants import epsilon_0, m_p
from warpx_constants import elementary_charge as q_e

yt.set_log_level(50)

# Must match the input decks (cgl_relaxation and cgl_null share all
# state constants; they differ in B, dt, and the active rate).
N0 = 1.0e20
TI_EV = 100.0
P0 = N0 * TI_EV * q_e
LNL = 10.0
KT = P0 / N0
NU = (np.sqrt(2.0) * q_e**4 * LNL * N0 /
      (12.0 * np.pi**1.5 * np.sqrt(m_p) * epsilon_0**2 * KT**1.5))
# Stage B+ null-blend rate of the cgl_null deck: at B = 0 the
# magnetization weight is exactly 0 and the relaxation gains the full
# unmagnetized cell-transit rate v_thi/dz.
DZ = 0.2 / 16
V_THI = np.sqrt(P0 / (N0 * m_p))
NU_NULL = V_THI / DZ
THETA = 0.5
NSTEPS = 40


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
mode = sys.argv[3] if len(sys.argv) > 3 else "collisional"
assert mode in ("collisional", "bounded", "null")

# The decks set dt = 0.1 / (active rate); the calibrated modes measure
# against the theta-discrete amplification of that same rate.
NU_EFF = NU + NU_NULL if mode == "null" else NU
DT = 0.1 / NU_EFF
GROWTH = (1.0 - (1.0 - THETA) * NU_EFF * DT) / (1.0 + THETA * NU_EFF * DT)


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

print(f"nu_iso (first principles)  {NU_EFF:.6e} 1/s"
      + (f"  [nu_null/nu_ii = {NU_NULL / NU:.1f}]" if mode == "null"
         else ""))
print(f"initial anisotropy Delta   {delta(initial):.6e}")
print(f"final anisotropy Delta     {delta(final):.6e}")
print(f"measured log decay         {measured_log_decay:.10f}")
print(f"expected log decay         {expected_log_decay:.10f}")
print(f"relative error             {relative_error:.3e}")
print(f"U_par + U_perp drift       {conservation:.3e}")
print(f"final spatial nonuniformity {uniformity:.3e}")

assert delta(initial) > 0.9 * P0 * (3.0 / 5.0)
assert conservation < 1.0e-12
assert uniformity < 1.0e-12

if mode in ("collisional", "null"):
    # null: at B = 0 the blend weight is exactly 0 and the decay runs
    # at nu_ii + v_thi/dz (~116x collisional), so this also verifies
    # the w -> 0 limit -- a silently vanished null rate would slow the
    # log decay by that factor and fail spectacularly.
    assert relative_error < 1.0e-5, (
        f"isotropization rate off by {relative_error:.3e}"
    )
else:
    # Instability-bounded variant: the A0 = 2 state starts mirror
    # unstable (beta_perp (p_perp/p_par - 1) = 2.4 > 1) and the
    # cyclotron-scale enhancement must pull it to marginal stability
    # far faster than the collisional rate alone
    # (nu_inst dt ~ 0.6 vs nu_ii dt = 0.1 here).
    magnetic_pressure = P0  # deck: B0 = sqrt(mu0 P0), so B0^2/mu0 = P0
    p_par = 2.0 * np.mean(final["implicit_mhd_ion_parallel_energy"])
    p_perp = np.mean(final["implicit_mhd_ion_perp_energy"])
    mirror_measure = 2.0 * p_perp * (p_perp - p_par) / \
        (magnetic_pressure * p_par) - 1.0
    print(f"final mirror measure       {mirror_measure:.4f}")
    assert mirror_measure < 0.2, (
        "instability bound failed to pull the state to marginal "
        f"stability (mirror measure {mirror_measure:.3f})"
    )
    collisional_growth = GROWTH**NSTEPS
    assert delta(final) / delta(initial) < 0.5 * collisional_growth, (
        "bounded relaxation is not faster than collisional"
    )

print("PASS")
