#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Floor-consistency relaxation source under the engineered deadlock.

The SAME deck as the active-set rescue test (a pedestal-resident halo
band whose partnerless ion-energy sink is pinned at an ion
temperature-floor admissibility bound ~500x above its relaxation target)
with implicit_mhd.floor_consistency_rate armed at the 1/(theta dt) cap.
The escalation trigger (production rr13e_C1w): the rescue alone let the
pinned defect GROW 0.468 -> 1.004 (157 -> 241 components) until the
freeze guard fired -- the discrete equations continuously demand
sub-bound drain, a consistency gap no line-search policy can close.

With the source, the reservoir supplies the demanded drain at the bound:
the band descends to the supply == sink equilibrium a fraction of a
rectifier width ABOVE its admissibility image and RIDES there. The
Newton solves converge to machine precision throughout (no plateau, no
pinned components, no rescues, no frozen steps -- the .run test also
FAILs on any "frozen step" warning), and every supplied Joule is booked.

Assertions (newton.txt + floor_ledger.txt + plotfiles + the rescue
baseline test's outputs, a declared dependency):
  1. the run COMPLETED all its steps with every solve progressing;
  2. NO WEDGE: the maximum post-solve residual norm stays below 10% of
     the rescue-only run's measured defect plateau (measured here:
     machine precision vs 2.13e-2 -- a ~1e9 margin);
  3. the band's ion internal energy RIDES AT the bound (within the
     rectifier engagement band above the temperature-floor image) and
     is steady between the mid and final dumps, while the rescue-only
     twin drained THROUGH the bound to ~500x below;
  4. LEDGER CLOSURE: the booked supply's steady-state rate equals the
     analytic sink demand rate at the bound-riding state (the sink rate
     and target are known deck constants; the strongest assert);
  5. total energy accounting: the state's fluid-energy change over the
     steady window balances booked supply minus analytic sink drain;
  6. the booked MASS column is exactly zero (no density bound engages).

Usage: analysis_mhd_floor_consistency.py <initial> <mid> <final plotfile>
"""

import sys

import numpy as np
import yt

baseline_directory = "../test_1d_theta_implicit_mhd_active_set_rescue"

MAX_STEP = 120
# Deck constants (inputs_test_1d_theta_implicit_mhd_floor_consistency).
K_B = 1.380649e-23
M_P = 1.67262192369e-27
Q_E = 1.602176634e-19
N0 = 1.0e18
RHO0 = N0 * M_P
PEDESTAL_FRACTION = 1.0e-3
RHO_HALO = PEDESTAL_FRACTION * RHO0
GAMMA = 5.0 / 3.0
T_FLOOR = 58022.0
SINK_RATE = 5.0e7
DT = 1.0e-9
NZ = 64
DZ = 1.0 / NZ
# The band's temperature-floor admissibility image (internal energy):
# n_halo kB T_floor / (gamma - 1).
E_BOUND = (RHO_HALO / M_P) * K_B * T_FLOOR / (GAMMA - 1.0)
# Rectifier width fraction of the source (ThetaImplicitMHD_K.H).
WIDTH_FRACTION = 0.1


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


def band_fields(data):
    rho = data["boxlib", "implicit_mhd_mass_density"].value.ravel()
    ion_e = data["boxlib", "implicit_mhd_ion_energy"].value.ravel()
    mom = data["boxlib", "implicit_mhd_momentum_density"].value.ravel()
    internal = ion_e - 0.5 * mom**2 / rho
    band = rho < 0.5 * rho.max()
    return rho, internal, band


initial_ds, initial = get_data(sys.argv[1])
mid_ds, mid = get_data(sys.argv[2])
final_ds, final = get_data(sys.argv[3])

def last_session(rows, step_col=0):
    # newton.txt / floor_ledger.txt APPEND across reruns of a test
    # directory; keep only the most recent session (rows after the last
    # step-number reset).
    resets = np.nonzero(np.diff(rows[:, step_col]) < 0)[0]
    return rows[(resets[-1] + 1) if len(resets) else 0 :]


history = last_session(np.atleast_2d(np.loadtxt("diags/newton.txt")))
steps = history[:, 0]
iters = history[:, 2]
norm_abs = history[:, 4]

ledger = last_session(np.atleast_2d(np.loadtxt("diags/floor_ledger.txt")))
ledger_steps = ledger[:, 0]
booked_mass = ledger[:, 1]
booked_energy = ledger[:, 2]

# 1. Completion, with every solve progressing (an iteration-0 stagnation
# logs 0 iters; the .run test additionally fails on any "frozen step"
# warning line).
assert steps[-1] == MAX_STEP, (
    f"run did not complete: newton.txt ends at step {steps[-1]:.0f} of {MAX_STEP}"
)
assert np.all(iters >= 1), "a solve made zero progress (iters == 0)"
assert ledger.shape[0] == history.shape[0], (
    f"ledger rows ({ledger.shape[0]}) != solves ({history.shape[0]})"
)
assert np.all(np.diff(ledger_steps) == 1), "ledger rows are not per-step"

# 2. NO WEDGE, and the pinned defect stays below 10% of the rescue-only
# plateau (the dependency's measured deadlock signature). The rescue
# baseline plateaus at ~2.13e-2; with the supply the solves converge to
# machine precision instead (measured max ~1e-11).
baseline_history = last_session(
    np.atleast_2d(np.loadtxt(f"{baseline_directory}/diags/newton.txt"))
)
baseline_plateau = baseline_history[:, 4].max()
assert baseline_plateau > 5.0e-3, (
    "the rescue baseline lost its deadlock plateau "
    f"(max norm {baseline_plateau:.3e}): this test discriminates nothing"
)
defect_ratio = norm_abs.max() / baseline_plateau
print(
    f"defect: max norm {norm_abs.max():.3e} vs rescue plateau "
    f"{baseline_plateau:.3e} (ratio {defect_ratio:.3e})"
)
assert defect_ratio < 0.1, (
    f"the wedge survived the supply: max residual norm {norm_abs.max():.3e} "
    f"is not below 10% of the rescue-only plateau {baseline_plateau:.3e}"
)

# 3. The band RIDES AT the bound: end-of-step internal energy just above
# the temperature-floor image (measured equilibrium 1.144x -- the theta
# stage rides ~0.7 rectifier widths above its bound, where the supply
# tail balances the sink) and steady between the mid and final dumps
# (measured drift 3.6e-5); the rescue-only twin drained THROUGH the
# bound to ~130x below.
_, internal_mid, band_mid = band_fields(mid)
_, internal_final, band_final = band_fields(final)
band_mid_mean = internal_mid[band_mid].mean()
band_final_mean = internal_final[band_final].mean()
print(
    f"band internal energy / bound image: mid {band_mid_mean / E_BOUND:.4f}, "
    f"final {band_final_mean / E_BOUND:.4f}"
)
assert 1.0 <= band_final_mean / E_BOUND <= 1.0 + 4.0 * WIDTH_FRACTION, (
    f"the band does not ride the bound: internal {band_final_mean:.3e} vs "
    f"image {E_BOUND:.3e}"
)
ride_drift = abs(band_final_mean - band_mid_mean) / E_BOUND
print(f"band ride drift between mid and final dumps: {ride_drift:.3e}")
assert ride_drift < 1.0e-2, (
    f"the band is not riding smoothly (drift {ride_drift:.3e} of the bound)"
)
# The rescue-only twin drained THROUGH the same bound. (Keep the dataset
# handle alive: yt covering grids hold only a weak reference to it.)
rescue_ds, rescue_final = get_data(f"{baseline_directory}/diags/diag000120")
_, rescue_internal, rescue_band = band_fields(rescue_final)
rescue_terminal = rescue_internal[rescue_band].mean()
contrast = band_final_mean / rescue_terminal
print(
    f"terminal band internal energy: supplied {band_final_mean:.3e} vs "
    f"rescue-only {rescue_terminal:.3e} (contrast {contrast:.1f}x)"
)
assert contrast > 50.0, (
    "the supplied band should hold ~130x above the rescue-only terminal "
    f"state; measured contrast only {contrast:.1f}x"
)

# 4. LEDGER CLOSURE (the strongest assert): at the riding equilibrium the
# supply balances the sink exactly, so the booked energy's steady-state
# rate must equal the analytic sink demand
#     sum_band SINK_RATE * (e_int - target) * dz   [J/m^2/s],
# with target the frozen pedestal image (pedestal_fraction x the peak
# internal energy) and the sink's density window and one-sided gate both
# identically 1 at the riding state (rho at the pedestal, e_int >> 2x
# target). Measured closure: 3.2e-5 relative.
target = PEDESTAL_FRACTION * internal_final.max()
analytic_sink_rate = SINK_RATE * np.sum(internal_final[band_final] - target) * DZ
window = 20
booked_rate = (booked_energy[-1] - booked_energy[-1 - window]) / (window * DT)
closure = abs(booked_rate - analytic_sink_rate) / analytic_sink_rate
print(
    f"ledger closure: booked rate {booked_rate:.6e} vs analytic sink "
    f"{analytic_sink_rate:.6e} J/m^2/s (relative gap {closure:.3e})"
)
assert closure < 0.02, (
    f"the booked supply does not close against the analytic sink demand "
    f"(relative gap {closure:.3e})"
)

# 5. Total energy accounting over the steady window [mid, final]: the
# fluid energy change must balance booked supply minus the sink drain
# (all other channels -- periodic fluxes, the exactly-pairing pdV
# couple, zero-eta Joule -- cancel).
def fluid_energy(data):
    u_e = data["boxlib", "implicit_mhd_electron_energy"].value.ravel()
    e_i = data["boxlib", "implicit_mhd_ion_energy"].value.ravel()
    return np.sum(u_e + e_i) * DZ


mid_index = np.searchsorted(ledger_steps, 60)
booked_window = booked_energy[-1] - booked_energy[mid_index]
sink_window = analytic_sink_rate * (MAX_STEP - 60) * DT
state_change = fluid_energy(final) - fluid_energy(mid)
balance = state_change - (booked_window - sink_window)
balance_scale = max(abs(booked_window), abs(sink_window))
print(
    f"energy accounting over steps 60-120: state change {state_change:.6e}, "
    f"booked {booked_window:.6e}, sink {sink_window:.6e}, "
    f"imbalance {balance:.3e} ({abs(balance) / balance_scale:.3e} of the "
    "booked scale)"
)
assert abs(balance) / balance_scale < 0.02, (
    f"energy accounting does not balance: residual {balance:.3e} vs booked "
    f"scale {balance_scale:.3e}"
)

# 6. No density bound engages anywhere: the mass column is EXACTLY zero.
assert np.all(booked_mass == 0.0), (
    f"the ledger booked mass on this deck (final row {booked_mass[-1]:.3e})"
)

print("PASS")
