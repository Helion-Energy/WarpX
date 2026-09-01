#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Density-keyed halo boost of the Braginskii ION chi_perp
(implicit_mhd.conduction_halo_boost): the reference code's exact multiplicative
low-density boost (ntb.f90 t_cond ~584,
xip = MAX(xip, xip*(en0/en)^2*dp_mn)), keyed to the shared reference
density (implicit_mhd.vacuum_reference_base_density, the reference code's en00,
raised by implicit_mhd.vacuum_reference_peak_fraction).

Single-zone magnetized column (uniform Bx perpendicular to z: all z
transport is cross-field), ENTROPY-MODE initialized -- both species'
pressures uniform, the density carrying the ripple -- so grad(p) = 0
and each channel's e_spec ripple decays diffusively at that channel's
effective cross-field chi, without an acoustic response. Three arms
share the base deck (each with its own dt sized to ~50% decay of its
target channel):

"boost"    (dp = 1, static base n_ref = 10 n0): the ion chi_perp is
           MULTIPLIED by exactly dp (n_ref/n0)^2 = 100 -- measured
           against the kernel's first-principles raw value; the
           electron channel must stay frozen on this short window
           (an electron leak of the boost decays it by ~16%).
"noop"     (dp = 1, base n_ref = 0.3 n0, boost = 0.09 < 1): the
           multiplicative MAX form is an EXACT no-op -- both channels
           decay at their raw Braginskii chi_perp (an additive or
           quadrature floor composition would still add here).
"ion_only" (dp = 1e6, perp clamps armed, dynamic peak-fraction
           reference): the boosted ion value rides the
           DEFINED-convention perp cap -- the measured ion chi must
           equal the cap itself (clamps cap the BOOSTED value, the
           the reference code's t_cond order) -- while the electron channel stays on
           its raw value: a leak of the boost onto the electron channel
           (the pre-2026-08-31 both-channels form) would drag it onto
           the same cap rate.

"production_stack" (the "boost" arm's exact x100 gate re-run through
           the RZ production routing stack -- explicit
           conduction_theta == theta, conduction_coefficient_state =
           step_old, conduction_flux_limit_factor = 0.3 at an
           amplitude that keeps the limiter inert, wide armed
           per-component + legacy clamps with the x100 target
           interior, and fluid_flux = hlld): none of those knobs may
           re-route or re-scale the boost (the run_h1_halodrain
           masking hunt, 2026-08-31 -- the actual production masker,
           conduction_qs_chi saturating every face onto the perp cap,
           is excluded against the dp boost by the parse guard).

Every arm also closes sum(U_e) and sum(U_i) separately to roundoff
(conservative face flux, no equilibration, no Joule).

Usage: analysis_mhd_conduction_halo_boost.py <arm> <initial_plotfile> <final_plotfile>
"""

import sys

import numpy as np
import warpx_constants as constants
import yt

yt.set_log_level(50)

arm = sys.argv[1]
assert arm in ("boost", "noop", "ion_only", "production_stack")


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


initial_ds, initial = get_data(sys.argv[2])
final_ds, final = get_data(sys.argv[3])

# constants from inputs_base_1d_theta_implicit_mhd_conduction_halo_boost
number_density = 1.0e16
ti_eV = 1.0
te_eV = 10.0
gamma = 5.0 / 3.0
b_field = 5.0e-4
coulomb_log = 10.0
number_of_cells = 64
domain_length = 1.0
ripple_mode = 4
ripple_amplitude_ic = 0.05
mass_density_floor = 1.0e-6 * number_density * constants.proton_mass
cell_size = domain_length / number_of_cells
z_centers = (np.arange(number_of_cells) + 0.5) * cell_size

if arm in ("boost", "production_stack"):
    dp_mn = 1.0
    # static base reference (implicit_mhd.vacuum_reference_base_density)
    reference_density = 10.0 * number_density
elif arm == "noop":
    dp_mn = 1.0
    reference_density = 0.3 * number_density
else:
    dp_mn = 1.0e6
    # max(base 0.02 n0, 0.1 * step-old peak): the rippled step-old peak
    # is n0 (1 + dr) and the fraction path must win the composition
    reference_density = 0.1 * number_density * (1.0 + ripple_amplitude_ic)
# DEFINED-convention perp clamp bounds of the ion_only arm (the
# operator applies the (gamma - 1) convention bridge)
perp_min_defined = 0.001
perp_cap_defined = 100.0

elapsed_time = float(final_ds.current_time - initial_ds.current_time)
assert elapsed_time > 0.0


def fields(data):
    rho = data["boxlib", "implicit_mhd_mass_density"].value.ravel()
    ue = data["boxlib", "implicit_mhd_electron_energy"].value.ravel()
    ui = data["boxlib", "implicit_mhd_ion_internal_energy"].value.ravel()
    return rho, ue, ui


rho_i, ue_i, ui_i = fields(initial)
rho_f, ue_f, ui_f = fields(final)

# ---- conservation gates: conduction is a conservative face flux and
# there is no equilibration/Joule, but the momentum channel is LIVE --
# the pressure ripple regrown by partial temperature-ripple decay
# drives O(dr^2) pdV/kinetic exchange growing with the window (measured
# 1e-12..1.3e-6 across the arms), so the per-channel budgets are gated
# as a sanity bound four decades under the 50%-class decay signal ----
volume = cell_size
for name, total_i, total_f in (
    ("U_e", ue_i.sum() * volume, ue_f.sum() * volume),
    ("U_i", ui_i.sum() * volume, ui_f.sum() * volume),
):
    drift = abs(total_f - total_i) / abs(total_i)
    print(f"{name} conservation drift = {drift:.3e}")
    assert drift < 1.0e-5


# ---- the kernel's own first-principles Braginskii chi_perp of each
# channel (kappa/(n kB) convention; the operator applies the
# (gamma - 1) bridge) ----
def braginskii_chi_perp(species, t_eV):
    kb_t = t_eV * constants.elementary_charge
    tau_shared = (
        np.pi**1.5
        * constants.epsilon_0**2
        / (constants.elementary_charge**4 * coulomb_log)
    )
    if species == "e":
        mass = constants.electron_mass
        tau = (
            6.0
            * np.sqrt(2.0)
            * np.sqrt(mass)
            * tau_shared
            * kb_t
            * np.sqrt(kb_t)
            / number_density
        )
        chi_par = 3.16 * kb_t * tau / mass
        numerator_1 = 4.664 / 11.92
        denominator_1 = 14.79 / 3.7703
        denominator_2 = 1.0 / 3.7703
    else:
        mass = constants.proton_mass
        tau = (
            12.0 * np.sqrt(mass) * tau_shared * kb_t * np.sqrt(kb_t) / number_density
        )
        chi_par = 3.9 * kb_t * tau / mass
        numerator_1 = 2.0 / 2.645
        denominator_1 = 2.70 / 0.677
        denominator_2 = 1.0 / 0.677
    x_mag = (constants.elementary_charge * b_field / mass * tau) ** 2
    return (
        chi_par
        * (numerator_1 * x_mag + 1.0)
        / ((denominator_2 * x_mag + denominator_1) * x_mag + 1.0)
    )


rho = number_density * constants.proton_mass
rho_ref = reference_density * constants.proton_mass
rho_guarded = np.sqrt(rho**2 + mass_density_floor**2)
boost = max(1.0, dp_mn * (rho_ref / rho_guarded) ** 2)

raw_e = braginskii_chi_perp("e", te_eV)
raw_i = braginskii_chi_perp("i", ti_eV)
boosted_i = raw_i * boost
expected_e = raw_e
if arm == "ion_only":
    # armed clamps cap the BOOSTED value (DEFINED convention)
    boosted_i = min(max(boosted_i, perp_min_defined), perp_cap_defined)
    expected_e = min(max(raw_e, perp_min_defined), perp_cap_defined)

convention = gamma - 1.0  # kappa/(n kB) -> operator convention
chi_expected_i = convention * boosted_i
chi_expected_e = convention * expected_e
print(
    f"raw chi_perp_e {raw_e:.4e}, raw chi_perp_i {raw_i:.4e}, "
    f"boost x{boost:.4g} -> expected operator chi: ion "
    f"{chi_expected_i:.4e}, electron {chi_expected_e:.4e} m^2/s"
)

# ---- measured ripple decay per channel (full periodic domain) ----
k_ripple = 2.0 * np.pi * ripple_mode / domain_length
# discrete diffusion eigenvalue of the face-difference operator
k_eff_sq = (2.0 / cell_size * np.sin(0.5 * k_ripple * cell_size)) ** 2
basis = np.column_stack(
    [
        np.ones(number_of_cells),
        np.sin(k_ripple * z_centers),
        np.cos(k_ripple * z_centers),
    ]
)


def ripple_amplitude(values):
    coefficients = np.linalg.lstsq(basis, values, rcond=None)[0]
    return np.hypot(coefficients[1], coefficients[2])


def measure(energy_initial, energy_final):
    retention = ripple_amplitude(energy_final / rho_f) / ripple_amplitude(
        energy_initial / rho_i
    )
    return -np.log(retention) / (k_eff_sq * elapsed_time), retention


chi_ion, retention_ion = measure(ui_i, ui_f)
chi_ele, retention_ele = measure(ue_i, ue_f)
print(f"ion measured chi = {chi_ion:.4e} (predicted {chi_expected_i:.4e}), "
      f"retention {retention_ion:.4f}")
print(f"ele measured chi = {chi_ele:.4e} (predicted {chi_expected_e:.4e}), "
      f"retention {retention_ele:.4f}")

# the arm's window must resolve its target ion rate (~half-decay class)
assert 0.05 < retention_ion < 0.9

# ION gate: the measured chi matches the arm's prediction -- x100
# multiplied raw ("boost"), exactly raw ("noop": a floor composition
# would still add), or exactly the cap ("ion_only": clamps cap the
# boosted value)
assert 0.8 * chi_expected_i < chi_ion < 1.2 * chi_expected_i

# ELECTRON gate: never boosted
if arm in ("boost", "production_stack"):
    # the electron channel is frozen on this short window; an electron
    # leak of the x100 boost decays it by ~16% (retention ~0.84)
    expected_retention = np.exp(-chi_expected_e * k_eff_sq * elapsed_time)
    assert expected_retention > 0.995
    assert retention_ele > 0.99
else:
    # measurable electron decay at the RAW value; in the ion_only arm a
    # leaked boost would drag it onto the cap rate (retention ~0.5)
    assert 0.7 * chi_expected_e < chi_ele < 1.3 * chi_expected_e

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert newton_history[-1, 2] >= 1

print(f"conduction halo boost ({arm}): all gates passed")
