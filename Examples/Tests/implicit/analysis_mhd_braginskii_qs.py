#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Braginskii quasi-shorting cross-field boost (implicit_mhd.conduction_qs_*).

Strongly magnetized column (uniform Bx perpendicular to z, Omega_e
tau_e ~ 1e4 -> chi_perp_e ~ 0.1 m2/s) with the ion fluid frozen: all z
transport is cross-field electron conduction. Two structures, two
gates, measured against the no-QS twin
(test_1d_theta_implicit_mhd_braginskii_qs_twin, conduction_qs_chi = 0):

1. POCKET GATE -- an entropy-excess pocket (rho -> 0.25 rho0, electron
   pseudo-entropy s = (Te/T0)(rho0/rho)^(2/3) -> 8 >> onset 1.25) must
   drain heat across the flux surfaces at the quasi-shorting rate: the
   electron energy lost through the pocket-window boundary faces
   (chi_add = 2.5e4 m2/s there, s ~ 2.5) must (a) exceed the twin's
   chi_perp-only drain by the expected O(chi_add/chi_perp) ~ 4e5 ratio
   (measured 4.4e5) and (b) match the face-flux prediction assembled
   from the initial profile with the implementation's own formulas,

       chi_add = qs_chi * 0.5 [(s - onset) + sqrt((s - onset)^2 + w^2)],
       w = 0.3 (onset - 1),

   within a calibrated band (measured/predicted = 0.981: the boundary
   profile steepens only slightly over the short run).

2. LEAK GATE (the shifted-ramp bound) -- the background carries an
   ON-ADIABAT density/temperature ripple (Te ~ rho^(2/3), s = 1
   exactly), measured in windows the pocket's heat cannot reach within
   the run (per-step implicit influence length sqrt(chi dt) << dx). Its
   decay excess over the twin isolates chi_add(s = 1): the ramp is
   centered ABOVE the envelope, so an on-adiabat cell must see
   chi_add < 1% of the amplitude (analytic leak
   0.5 [(1 - onset) + sqrt((1 - onset)^2 + w^2)] = 5.504e-3 qs_chi
   here; measured 5.510e-3, matching the smooth-max formula to 0.1%).
   A ramp centered ON the envelope (the production-fatal variant)
   leaks w/2 = 3.75% and FAILS this gate.

Usage: analysis_mhd_braginskii_qs.py <initial_plotfile> <final_plotfile>
       (the twin's final plotfile is read from
        ../test_1d_theta_implicit_mhd_braginskii_qs_twin/diags/diag000010)
"""

import sys

import numpy as np
import warpx_constants as constants
import yt

yt.set_log_level(50)


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])
twin_ds, twin = get_data(
    "../test_1d_theta_implicit_mhd_braginskii_qs_twin/diags/diag000010"
)

# constants from inputs_test_1d_theta_implicit_mhd_braginskii_qs
number_density = 1.0e18
t0_eV = 10.0
gamma = 5.0 / 3.0
b_field = 0.05
coulomb_log = 10.0
qs_chi = 2.0e4
qs_onset = 1.25
qs_width = 0.3 * (qs_onset - 1.0)
number_of_cells = 64
domain_length = 1.0
ripple_mode = 4
n_floor_ohm = 1.0e10  # hybrid_pic_model.n_floor
rho0 = number_density * constants.proton_mass
cell_size = domain_length / number_of_cells
z_centers = (np.arange(number_of_cells) + 0.5) * cell_size

elapsed_time = float(final_ds.current_time - initial_ds.current_time)
assert elapsed_time > 0.0
assert abs(float(twin_ds.current_time) - float(final_ds.current_time)) < 1e-30


def fields(data):
    rho = data["boxlib", "implicit_mhd_mass_density"].value.ravel()
    energy = data["boxlib", "implicit_mhd_electron_energy"].value.ravel()
    return rho, energy


rho_i, energy_i = fields(initial)
rho_f, energy_f = fields(final)
rho_t, energy_t = fields(twin)
# frozen ion fluid: identical densities in all three states
assert np.array_equal(rho_i, rho_f) and np.array_equal(rho_i, rho_t)

e_spec_i = energy_i / rho_i
e_spec_f = energy_f / rho_f
e_spec_t = energy_t / rho_t

# ---- re-derive the implementation's face diffusivities from the
# INITIAL state (the coefficient inputs at theta ~ the initial state
# for this short, weakly-relaxing run) ----
pressure_i = (gamma - 1.0) * energy_i
charge_to_mass = constants.elementary_charge / constants.proton_mass
rho_guard_ohm = n_floor_ohm * constants.elementary_charge / charge_to_mass

# face arrays on interior faces f = 1..N-1 between cells f-1 and f
# (the periodic wrap face is far from every window)
face_density = 0.5 * (rho_i[:-1] + rho_i[1:])
face_charge_density = np.maximum(
    charge_to_mass * face_density, charge_to_mass * rho_guard_ohm
)
face_te = (
    0.5
    * (pressure_i[:-1] + pressure_i[1:])
    * constants.elementary_charge
    / (face_charge_density * constants.k)
)

# Braginskii electron chi_perp (Z = 1 magnetization fit; see
# analysis_mhd_braginskii_isolimit.py for the parallel coefficient)
tau_shared = (
    np.pi**1.5
    * constants.epsilon_0**2
    / (constants.elementary_charge**4 * coulomb_log)
)
kb_te = constants.k * face_te
face_number_density = face_charge_density / constants.elementary_charge
tau_e = (
    6.0
    * np.sqrt(2.0)
    * np.sqrt(constants.electron_mass)
    * tau_shared
    * kb_te
    * np.sqrt(kb_te)
    / face_number_density
)
chi_par = 3.16 * kb_te * tau_e / constants.electron_mass
x_mag = (constants.elementary_charge * b_field / constants.electron_mass * tau_e) ** 2
chi_perp = (
    chi_par
    * (4.664 / 11.92 * x_mag + 1.0)
    / ((x_mag / 3.7703 + 14.79 / 3.7703) * x_mag + 1.0)
)

# quasi-shorting boost at the faces (the implementation formula; the
# smooth density guard is invisible here, rho >> rho_guard)
t0_kelvin = t0_eV * constants.elementary_charge / constants.k
guarded_density = np.sqrt(face_density**2 + rho_guard_ohm**2)
entropy = face_te / t0_kelvin * (rho0 / guarded_density) ** (2.0 / 3.0)
excess = entropy - qs_onset
chi_add = qs_chi * 0.5 * (excess + np.sqrt(excess**2 + qs_width**2))

# ---- gate 1: pocket heat drain ----
# window edge at 1.25 pocket widths: the boundary faces sit where the
# ramp is amplitude-scale (s ~ 2.5, chi_add ~ 2.5e4), so the gate
# probes the boost's AMPLITUDE, not its tail
pocket = np.abs(z_centers - 0.5 * domain_length) <= 0.07
pocket_faces = np.nonzero(np.diff(pocket.astype(int)))[0]  # faces at the edges
face_left, face_right = pocket_faces[0], pocket_faces[1]
gradient = np.diff(e_spec_i) / cell_size  # at interior faces


def boundary_flux(chi_face):
    # q = -chi rho_f d(e_spec)/dz at the window boundary faces;
    # net drain rate of the window = q_left - q_right (per dz)
    q = -chi_face * face_density * gradient
    return q[face_left] - q[face_right]


predicted_qs = boundary_flux(chi_perp + chi_add) * elapsed_time / cell_size
predicted_twin = boundary_flux(chi_perp) * elapsed_time / cell_size
measured_qs = np.sum(energy_f[pocket]) - np.sum(energy_i[pocket])
measured_twin = np.sum(energy_t[pocket]) - np.sum(energy_i[pocket])
flux_ratio = measured_qs / measured_twin
prediction_ratio = measured_qs / predicted_qs
print(f"pocket dU_e (QS)             = {measured_qs:.6e} (pred {predicted_qs:.6e})")
print(f"pocket dU_e (twin)           = {measured_twin:.6e} (pred {predicted_twin:.6e})")
print(f"pocket QS/twin flux ratio    = {flux_ratio:.3e}")
print(f"pocket measured/predicted    = {prediction_ratio:.4f}")
# the pocket must LOSE energy through both boundary faces
assert measured_qs < 0.0 and predicted_qs < 0.0
# (a) the QS drain dwarfs the twin's chi_perp-only drain by the
# expected O(chi_add/chi_perp) ~ 4e5 ratio (measured 4.4e5)
assert flux_ratio > 1.0e4
# (b) quantitative: the time-integrated drain matches the initial-
# profile face-flux prediction (measured 0.981)
assert 0.9 < prediction_ratio < 1.05

# ---- gate 2: on-adiabat leak bound ----
# windows the pocket cannot influence within the run: the per-step
# implicit influence length is sqrt(chi_add,max dt) ~ 0.5 dx
window = np.abs(z_centers - 0.5 * domain_length) > 0.2
k_ripple = 2.0 * np.pi * ripple_mode / domain_length
# discrete diffusion eigenvalue of the ripple mode (face-difference
# operator): keff^2 = (2/dz sin(k dz/2))^2
k_eff_sq = (2.0 / cell_size * np.sin(0.5 * k_ripple * cell_size)) ** 2


def ripple_amplitude(values):
    # least-squares fit of the windowed fluctuations to the ripple
    basis = np.column_stack(
        [
            np.ones(window.sum()),
            np.sin(k_ripple * z_centers[window]),
            np.cos(k_ripple * z_centers[window]),
        ]
    )
    coefficients = np.linalg.lstsq(basis, values[window], rcond=None)[0]
    return np.hypot(coefficients[1], coefficients[2])


amplitude_i = ripple_amplitude(e_spec_i)
amplitude_f = ripple_amplitude(e_spec_f)
amplitude_t = ripple_amplitude(e_spec_t)
# the twin decays only through the tiny Braginskii chi_perp; the QS
# run adds exactly the on-adiabat leak. The ratio cancels everything
# the runs share.
leak_chi = np.log(amplitude_t / amplitude_f) / (k_eff_sq * elapsed_time)
analytic_leak = qs_chi * 0.5 * ((1.0 - qs_onset) + np.hypot(1.0 - qs_onset, qs_width))
print(f"ripple amplitude             = {amplitude_i:.6e} -> "
      f"{amplitude_f:.6e} (QS) / {amplitude_t:.6e} (twin)")
print(f"measured on-adiabat leak chi = {leak_chi:.4e} m^2/s "
      f"(analytic {analytic_leak:.4e}, amplitude {qs_chi:.1e})")
# THE shifted-ramp bound: an on-adiabat cell sees < 1% of the amplitude
assert leak_chi < 0.01 * qs_chi
# and the leak is real and matches the smooth-max formula (a ramp
# centered ON the envelope leaks w/2 = 3.75% and fails both gates)
assert 0.8 * analytic_leak < leak_chi < 1.2 * analytic_leak

print("braginskii quasi-shorting: all gates passed")
