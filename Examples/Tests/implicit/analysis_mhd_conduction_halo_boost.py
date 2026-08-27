#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Density-keyed halo boost of the Braginskii chi_perp
(implicit_mhd.conduction_halo_boost) keyed to the dynamic reference
(implicit_mhd.vacuum_reference_peak_fraction).

Strongly magnetized column (uniform Bx perpendicular to z, frozen ions,
J = 0, eta = 0): all z transport is cross-field electron conduction.
Two density zones -- bulk n0 (z < Lz/2) and halo n0/100 (z >= Lz/2) --
carry the same on-ripple temperature profile Te = T0 (1 + a sin(k4 z)),
whose specific-internal-energy ripple decays diffusively at each zone's
effective cross-field diffusivity.  With the dynamic reference
n_ref = max(n_floor, 0.1 n_peak) = 0.1 n0 and D_boost = 1 m^2/s:

1. HALO gate -- the boost diffusivity D_boost (n_ref/n_halo)^2 =
   100 m^2/s dwarfs the physical chi_perp (~1e-3 m^2/s at the halo
   density): the measured halo ripple-decay diffusivity must match the
   implementation's quadrature composition
   sqrt(chi_perp^2 + chi_boost^2) within a calibrated band.

2. BULK gate -- at the peak density the boost is
   D_boost (n_ref/n0)^2 = 0.01 m^2/s, entering the quadrature BELOW the
   physical chi_perp ~ 0.1 m^2/s: the bulk keeps physical Braginskii
   (the measured bulk decay stays two decades under the halo's, inside
   a loose physical band) -- boosted perp flux in the LOW-DENSITY zone
   only.

Usage: analysis_mhd_conduction_halo_boost.py <initial_plotfile> <final_plotfile>
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

# constants from inputs_test_1d_theta_implicit_mhd_conduction_halo_boost
bulk_number_density = 1.0e18
halo_number_density = 1.0e16
t0_eV = 10.0
gamma = 5.0 / 3.0
b_field = 0.05
coulomb_log = 10.0
boost_diffusivity = 1.0
peak_fraction = 0.1
number_of_cells = 64
domain_length = 1.0
ripple_mode = 4
n_floor_ohm = 1.0e10  # hybrid_pic_model.n_floor
mass_density_floor = 1.0e-6 * bulk_number_density * constants.proton_mass
cell_size = domain_length / number_of_cells
z_centers = (np.arange(number_of_cells) + 0.5) * cell_size

elapsed_time = float(final_ds.current_time - initial_ds.current_time)
assert elapsed_time > 0.0


def fields(data):
    rho = data["boxlib", "implicit_mhd_mass_density"].value.ravel()
    energy = data["boxlib", "implicit_mhd_electron_energy"].value.ravel()
    return rho, energy


rho_i, energy_i = fields(initial)
rho_f, energy_f = fields(final)
# frozen ion fluid: identical densities
assert np.array_equal(rho_i, rho_f)

e_spec_i = energy_i / rho_i
e_spec_f = energy_f / rho_f


# ---- the implementation's effective cross-field diffusivity of a
# uniform-density zone at (n, T0): Braginskii Z = 1 chi_perp composed
# with the halo boost through the vacuum_keyed_resistivity quadrature
# smooth max ----
def braginskii_chi_perp(number_density, te_kelvin):
    tau_shared = (
        np.pi**1.5
        * constants.epsilon_0**2
        / (constants.elementary_charge**4 * coulomb_log)
    )
    kb_te = constants.k * te_kelvin
    tau_e = (
        6.0
        * np.sqrt(2.0)
        * np.sqrt(constants.electron_mass)
        * tau_shared
        * kb_te
        * np.sqrt(kb_te)
        / number_density
    )
    chi_par = 3.16 * kb_te * tau_e / constants.electron_mass
    x_mag = (
        constants.elementary_charge * b_field / constants.electron_mass * tau_e
    ) ** 2
    return (
        chi_par
        * (4.664 / 11.92 * x_mag + 1.0)
        / ((x_mag / 3.7703 + 14.79 / 3.7703) * x_mag + 1.0)
    )


t0_kelvin = t0_eV * constants.elementary_charge / constants.k
# dynamic reference: max(static Ohm guard, fraction * n_peak); the
# frozen step-old peak is the bulk density (frozen ions)
reference_density = max(n_floor_ohm, peak_fraction * bulk_number_density)
assert reference_density == 1.0e17


def composed_chi(number_density):
    rho = number_density * constants.proton_mass
    rho_ref = reference_density * constants.proton_mass
    rho_guarded = np.sqrt(rho**2 + mass_density_floor**2)
    chi_boost = boost_diffusivity * (rho_ref / rho_guarded) ** 2
    chi_perp = braginskii_chi_perp(number_density, t0_kelvin)
    return np.hypot(chi_perp, chi_boost), chi_perp, chi_boost


chi_halo, chi_perp_halo, chi_boost_halo = composed_chi(halo_number_density)
chi_bulk, chi_perp_bulk, chi_boost_bulk = composed_chi(bulk_number_density)
print(
    f"halo: chi_perp {chi_perp_halo:.3e} + boost {chi_boost_halo:.3e} "
    f"-> composed {chi_halo:.3e} m^2/s"
)
print(
    f"bulk: chi_perp {chi_perp_bulk:.3e} + boost {chi_boost_bulk:.3e} "
    f"-> composed {chi_bulk:.3e} m^2/s"
)
# calibration guards: the boost dominates the halo and stays under the
# physical bulk chi_perp
assert chi_boost_halo > 1.0e3 * chi_perp_halo
assert chi_boost_bulk < 0.5 * chi_perp_bulk

# ---- measured ripple decay per zone (windows away from the two zone
# interfaces at z = 0 and z = Lz/2; the run's diffusion length
# sqrt(chi_halo t) ~ 0.024 stays well inside the 0.15 margins) ----
k_ripple = 2.0 * np.pi * ripple_mode / domain_length
# discrete diffusion eigenvalue of the ripple mode (face-difference
# operator): keff^2 = (2/dz sin(k dz/2))^2
k_eff_sq = (2.0 / cell_size * np.sin(0.5 * k_ripple * cell_size)) ** 2


def measured_chi(window):
    basis = np.column_stack(
        [
            np.ones(window.sum()),
            np.sin(k_ripple * z_centers[window]),
            np.cos(k_ripple * z_centers[window]),
        ]
    )

    def amplitude(values):
        coefficients = np.linalg.lstsq(basis, values[window], rcond=None)[0]
        return np.hypot(coefficients[1], coefficients[2])

    return np.log(amplitude(e_spec_i) / amplitude(e_spec_f)) / (k_eff_sq * elapsed_time)


bulk_window = (z_centers > 0.15) & (z_centers < 0.35)
halo_window = (z_centers > 0.65) & (z_centers < 0.85)
assert bulk_window.sum() >= 10 and halo_window.sum() >= 10

chi_halo_measured = measured_chi(halo_window)
chi_bulk_measured = measured_chi(bulk_window)
print(f"halo measured chi = {chi_halo_measured:.4e} (predicted {chi_halo:.4e})")
print(f"bulk measured chi = {chi_bulk_measured:.4e} (predicted {chi_bulk:.4e})")

# ---- gate 1: the halo decays at the composed boost diffusivity ----
assert 0.8 * chi_halo < chi_halo_measured < 1.2 * chi_halo

# ---- gate 2: the bulk keeps physical Braginskii (no boost leak) ----
# two decades of separation, and a loose physical band (the on-ripple
# Te variation moves chi_perp by ~ +/-15%)
assert chi_bulk_measured < 0.02 * chi_halo_measured
assert 0.5 * chi_bulk < chi_bulk_measured < 2.0 * chi_bulk

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history[-1, 2] >= 1

print("conduction halo boost: all gates passed")
