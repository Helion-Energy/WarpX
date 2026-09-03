#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Order of accuracy and numerical diffusivity of the TVD reconstruction
(implicit_mhd.fluid_reconstruction), and the audit's D ~ c_f dx / 2 claim.

A single-wavelength sine density perturbation is advected as an exact
CONTACT (uniform velocity, uniform ion and electron pressure, uniform
field along the flow) on 32 / 64 / 128 cells with hlld, once with
reconstruction OFF (donor cell) and once with the reference code's median limiter.
dt is the SAME on every grid of the family (keyed to the finest), so the
theta = 0.5 temporal error is a common constant and the measured order is
purely spatial.

Two gates and one measurement:

1. ORDER. The L1 density error against the exact translated sine must
   converge at first order with the reconstruction OFF and at second
   order with the median limiter ON. Asserted as observed order <= 1.3
   (off, per-level and overall) and >= 1.5 (on). The 1.5 rather than 2.0
   bar is not slack: a median/minmod limiter clips at the two smooth
   EXTREMA of the sine, which is a first-order defect on an O(1) fraction
   of the cells, so the honest asymptotic L1 order of MUSCL-minmod on a
   profile with extrema is between 1.5 and 2 -- and the point of the gate
   is the separation from the donor-cell arm, which is decisive.

2. DIFFUSIVITY. The fundamental Fourier mode's amplitude decay gives the
   scheme's effective diffusivity directly,

       D_eff = -ln(A_final / A_initial) / (k^2 t),

   which the test prints for every arm and which the ON arms must reduce
   by a large factor at fixed resolution. The parity audit sized the
   first-order hlld dissipation at D ~ c_f dx / 2 (5.0e3 m^2/s in the
   production core at c_f = 1.29e6 m/s, dx = 7.81e-3 m); the OFF arm's
   D_eff is therefore printed against BOTH c_f dx / 2 and |u| dx / 2
   built from this deck's own speeds and cell sizes, and asserted to be
   LINEAR in dx (the signature that makes it a first-order truncation
   error rather than anything else) and O(1) against the |u| scale.

   MEASURED, and it REFINES the audit. The deck is deliberately SUBSONIC
   (Mach 0.531, the production regime) so the fan straddles the contact
   and the kappa_contact-smoothed region selection could leak fast-speed
   dissipation onto it. It does not. The donor-cell arm measures

       n =  32:  D_eff = 4.649e+02,  D/(|u| dx/2) = 0.9919
       n =  64:  D_eff = 2.330e+02,  D/(|u| dx/2) = 0.9943
       n = 128:  D_eff = 1.166e+02,  D/(|u| dx/2) = 0.9949

   -- exactly linear in dx, and exactly the textbook first-order Godunov
   contact coefficient D = |u| dx / 2, with the smoothed region selection
   contributing under 1%. Against the fast speed the same numbers are
   D/(c_f dx/2) = 0.528, i.e. the audit's c_f-scale sizing overstates the
   ADVECTED channels by the Mach-number factor c_f/|u| = 1.88 here. The
   audit's c_f dx / 2 is the right scale for the fast-magnetosonic
   characteristic fields, which a pure contact does not excite; for the
   channels this test measures the scale is |u| dx / 2. Carried to the
   production core (|u_z| = 6.2e5 m/s, dx = 7.81e-3 m) that is
   D = 2.4e3 m^2/s rather than the audit's 5.0e3 -- which does NOT rescue
   the conclusion: it is still above the explicit viscosity nu = 2000
   that was set to match the reference code's d_nur, so first-order hlld is still
   supplying more numerical diffusion than the physics model asks for.

3. CONSERVATION. Every arm must conserve total mass to round-off (the
   reconstruction is applied to the face STATES, so the flux stays
   single-valued and the divergence still telescopes).

Usage: analysis_mhd_reconstruction_order.py <initial> <final>
       (this test is the median n = 128 arm; the other five arms are read
        from ../test_1d_theta_implicit_mhd_reconstruction_order_*/diags)
"""

import os
import sys

import numpy as np
import warpx_constants as constants
import yt

yt.set_log_level(50)

# Deck constants (inputs_base_1d_theta_implicit_mhd_reconstruction_order).
NUMBER_DENSITY = 1.0e20
DENSITY = NUMBER_DENSITY * constants.proton_mass
DENSITY_AMPLITUDE = 0.2 * DENSITY
ELECTRON_TEMPERATURE_EV = 10.0
ELECTRON_PRESSURE = (
    NUMBER_DENSITY * ELECTRON_TEMPERATURE_EV * constants.elementary_charge
)
ION_PRESSURE = ELECTRON_PRESSURE
GAMMA = 5.0 / 3.0
VELOCITY = 3.0e4
MAGNETIC_FIELD = 1.0e-3
DOMAIN_LENGTH = 1.0
WAVENUMBER = 2.0 * np.pi / DOMAIN_LENGTH

TEST_STEM = "test_1d_theta_implicit_mhd_reconstruction_order"
PLOTFILE = "diags/diag000100"


def get_density(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return (
        float(ds.current_time),
        data["boxlib", "implicit_mhd_mass_density"].value.ravel(),
    )


def arm_plotfile(mode, cells):
    if mode == "median" and cells == 128:
        return PLOTFILE
    return os.path.join("..", f"{TEST_STEM}_{mode}_n{cells}", PLOTFILE)


def exact_density(cells, time):
    centers = (np.arange(cells) + 0.5) * DOMAIN_LENGTH / cells
    return DENSITY + DENSITY_AMPLITUDE * np.sin(
        WAVENUMBER * (centers - VELOCITY * time)
    )


def fundamental_amplitude(density):
    # A pure single mode sampled at cell centres has an exact DFT
    # amplitude, so this needs no windowing or aliasing correction.
    spectrum = np.fft.rfft(density - np.mean(density))
    return 2.0 * np.abs(spectrum[1]) / density.size


def observed_orders(errors, cell_counts):
    return [
        np.log(errors[i] / errors[i + 1]) / np.log(cell_counts[i + 1] / cell_counts[i])
        for i in range(len(errors) - 1)
    ]


initial_time, initial_density = get_density(sys.argv[1])
assert initial_time == 0.0

# The initial condition is the parser-sampled sine itself.
np.testing.assert_allclose(
    initial_density, exact_density(initial_density.size, 0.0), rtol=5.0e-13, atol=0.0
)

sound_speed = np.sqrt(GAMMA * (ELECTRON_PRESSURE + ION_PRESSURE) / DENSITY)
alfven_speed = MAGNETIC_FIELD / np.sqrt(constants.mu_0 * DENSITY)
# Field along the flow: the tangential field vanishes, so the fast speed
# degenerates to max(sound, Alfven) -- the fan's own c_f at these states.
fast_speed = max(sound_speed, alfven_speed)

cell_counts = [32, 64, 128]
results = {}
for mode in ("none", "median"):
    errors = []
    diffusivities = []
    for cells in cell_counts:
        time, density = get_density(arm_plotfile(mode, cells))
        reference = exact_density(cells, time)
        errors.append(np.mean(np.abs(density - reference)) / DENSITY_AMPLITUDE)
        amplitude = fundamental_amplitude(density)
        diffusivities.append(
            -np.log(amplitude / DENSITY_AMPLITUDE) / (WAVENUMBER**2 * time)
        )
        # Mass conservation, every arm.
        np.testing.assert_allclose(np.mean(density), DENSITY, rtol=5.0e-12, atol=0.0)
        assert np.all(density > 0.0)
    results[mode] = {
        "errors": errors,
        "diffusivities": diffusivities,
        "orders": observed_orders(errors, cell_counts),
        "overall": np.log(errors[0] / errors[-1])
        / np.log(cell_counts[-1] / cell_counts[0]),
    }

print(f"fast speed c_f = {fast_speed:.4e} m/s, |u| = {VELOCITY:.4e} m/s")
for mode in ("none", "median"):
    entry = results[mode]
    print(f"--- fluid_reconstruction = {mode}")
    for index, cells in enumerate(cell_counts):
        cell_size = DOMAIN_LENGTH / cells
        print(
            f"    n = {cells:4d}  L1/A = {entry['errors'][index]:.6e}"
            f"  D_eff = {entry['diffusivities'][index]:.6e} m^2/s"
            f"  D_eff/(c_f dx/2) = "
            f"{entry['diffusivities'][index] / (0.5 * fast_speed * cell_size):.4f}"
            f"  D_eff/(|u| dx/2) = "
            f"{entry['diffusivities'][index] / (0.5 * VELOCITY * cell_size):.4f}"
        )
    print(f"    per-level orders {np.round(entry['orders'], 3)}")
    print(f"    overall order    {entry['overall']:.3f}")

off = results["none"]
on = results["median"]

# 1. ORDER.
assert off["overall"] <= 1.3, f"donor cell is not first order: {off['overall']}"
assert all(order <= 1.3 for order in off["orders"]), off["orders"]
assert on["overall"] >= 1.5, f"median limiter is not second order: {on['overall']}"
assert all(order >= 1.5 for order in on["orders"]), on["orders"]
# The separation is the point: at the finest grid the limited scheme must
# be a large factor more accurate than donor cell.
assert on["errors"][-1] < 0.2 * off["errors"][-1], (
    on["errors"][-1],
    off["errors"][-1],
)

# 2. DIFFUSIVITY. The donor-cell arm's effective diffusivity must scale
# LINEARLY with dx -- that is the statement "this is a first-order
# truncation error", and it is what makes the single-grid number
# extrapolatable to the production cell size at all.
off_fast_ratios = [
    off["diffusivities"][index] / (0.5 * fast_speed * DOMAIN_LENGTH / cells)
    for index, cells in enumerate(cell_counts)
]
off_flow_ratios = [
    off["diffusivities"][index] / (0.5 * VELOCITY * DOMAIN_LENGTH / cells)
    for index, cells in enumerate(cell_counts)
]
print(f"donor-cell D_eff/(c_f dx/2) = {np.round(off_fast_ratios, 4)}")
print(f"donor-cell D_eff/(|u| dx/2) = {np.round(off_flow_ratios, 4)}")
assert max(off_fast_ratios) / min(off_fast_ratios) < 1.25, (
    "donor-cell D_eff is not linear in dx",
    off_fast_ratios,
)
# Against the CONTACT scale the coefficient must be O(1): a first-order
# Godunov contact cannot dissipate much less than |u| dx / 2 nor much
# more than the fan can supply.
assert 0.3 < np.mean(off_flow_ratios) < 4.0, off_flow_ratios
# The limiter must cut it hard at fixed resolution.
assert on["diffusivities"][-1] < 0.15 * off["diffusivities"][-1], (
    on["diffusivities"][-1],
    off["diffusivities"][-1],
)
assert all(value > 0.0 for value in on["diffusivities"]), on["diffusivities"]
