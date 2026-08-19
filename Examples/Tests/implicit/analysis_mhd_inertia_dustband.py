#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Ungated dust-band baseline for the electron-inertia dust gate.

A whistler packet (carrier mode 10, kappa = (k_eff d_e)^2 ~ 1) sits
inside a wide low-density band at 1.5x the Ohm floor, embedded in a 10x
denser background, with a frozen z-drift streaming the electrons through
the band -- the production dust regime: grid-scale d_e at near-floor
density, where the inertia term's advective piece is O(1) and NOT
representable in the pc_mhd_block fold. This UNGATED baseline must
follow the FULL (drift-Doppler capped) analytic advance built from the
band parameters:

    G_full(k): W dt = w dt - (1+kappa) k_c u0 dt,
    G_lin(k):  W dt = w dt - k_c u0 dt,

(see analysis_mhd_stiff_whistler_inertia_drift.py for the derivation and
staggering chains), applied per Fourier mode of the initial packet and
compared inside a window 2.5 envelope-sigmas wide -- the envelope decays
to ~5e-6 at the band edges, so band-uniform theory is exact up to
interface leakage of the one-step implicit solve (measured 2.2e-7
relative here, and 1e-11 in a uniform-density control run of this deck;
gated at 1e-4). The gated twin (dependency) must land on G_lin instead
and take no more GMRES iterations than this run.

Dust-regime detail: at 1.5x the Ohm floor the recast's C-infinity Ohm
guard is already ACTIVE -- the Hall denominator is
smooth_positive_floor(rho, fl) = fl + (rho - fl + sqrt((rho-fl)^2 +
fl^2))/2 = 1.206 rho at rho = 1.5 fl -- so the band's whistler rate w
uses that guarded density (a 0.829x rotation factor, confirmed exactly
by the run; the inertia term's own 1/rho is a hard max and stays
unguarded above the floor).
"""

import re
import sys
from pathlib import Path

import numpy as np
import warpx_constants as constants
import yt


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


def input_value(name):
    match = re.search(rf"^{re.escape(name)}\s*=\s*([^#\s]+)", used_inputs, re.MULTILINE)
    assert match is not None
    return match.group(1)


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])
used_inputs = Path("warpx_used_inputs").read_text()

band_number_density = 2.0e20
floor_number_density = band_number_density / 1.5
background_field = 0.1
drift_velocity = 1.0e5
envelope_center = 0.5
envelope_sigma = 0.1
theta = 0.5
effective_electron_mass = constants.proton_mass
electron_skin_depth = np.sqrt(
    effective_electron_mass
    / (constants.mu_0 * band_number_density * constants.elementary_charge**2)
)


# The recast's C-infinity Ohm division guard (ThetaImplicitMHD_K.H): the
# Hall term divides by this guarded density; at 1.5x the floor it is
# already active (1.206x the true density).
def smooth_positive_floor(value, floor):
    excess = value - floor
    return floor + 0.5 * (excess + np.sqrt(excess * excess + floor * floor))


hall_number_density = smooth_positive_floor(band_number_density, floor_number_density)

# The deck's dust-regime claims: band at 1.5x the Ohm floor, carrier at
# kappa ~ 1 (d_e ~ 2 cells at the band density).
assert input_value("hybrid_pic_model.n_floor") == "nfl"
assert input_value("my_constants.nfl") == "n_band/1.5"

initial_bx = initial["boxlib", "Bx"].value.ravel()
initial_by = initial["boxlib", "By"].value.ravel()
final_bx = final["boxlib", "Bx"].value.ravel()
final_by = final["boxlib", "By"].value.ravel()
final_bz = final["boxlib", "Bz"].value.ravel()
initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()

number_of_cells = initial_bx.size
domain_length = float((initial_ds.domain_right_edge - initial_ds.domain_left_edge)[0])
cell_size = domain_length / number_of_cells
dt = float(final_ds.current_time - initial_ds.current_time)
# The deck's dt is set from the UNGUARDED band diffusivity (CFL 5)...
nominal_diffusivity = background_field / (
    constants.mu_0 * constants.elementary_charge * band_number_density
)
whistler_cfl = nominal_diffusivity * dt / cell_size**2
np.testing.assert_allclose(whistler_cfl, 5.0, rtol=2.0e-12, atol=0.0)
# ... while the band's whistler rate uses the Ohm-guarded density.
hall_diffusivity = background_field / (
    constants.mu_0 * constants.elementary_charge * hall_number_density
)

z = (
    float(initial_ds.domain_left_edge[0])
    + (np.arange(number_of_cells) + 0.5) * cell_size
)

wavenumbers = 2.0 * np.pi * np.fft.fftfreq(number_of_cells, d=cell_size)
effective_wavenumbers = 2.0 * np.sin(0.5 * wavenumbers * cell_size) / cell_size
central_wavenumbers = np.sin(wavenumbers * cell_size) / cell_size
kappa = (effective_wavenumbers * electron_skin_depth) ** 2
omega_dt = hall_diffusivity * effective_wavenumbers**2 * dt
doppler_dt = central_wavenumbers * drift_velocity * dt

# The dust regime is real: the carrier sits at kappa ~ 1.
carrier_index = 10
carrier_kappa = kappa[carrier_index]
assert 0.7 < carrier_kappa < 1.4, f"carrier kappa = {carrier_kappa}"
# ... and the packet still disperses in the band: the LINEAR capped
# rotation at the carrier is a real angle (0.118 rad; the packet's upper
# modes rotate several times that, so the differential dispersion across
# the packet is substantial).
carrier_rotation = 2.0 * np.arctan(
    theta
    * (omega_dt[carrier_index] - doppler_dt[carrier_index])
    / (1.0 + carrier_kappa)
)
assert abs(carrier_rotation) > 0.1, f"carrier rotation {carrier_rotation}"


def advance(total_omega_dt):
    return ((1.0 + kappa) + 1.0j * (1.0 - theta) * total_omega_dt) / (
        (1.0 + kappa) - 1.0j * theta * total_omega_dt
    )


amplification_full = advance(omega_dt - (1.0 + kappa) * doppler_dt)
amplification_linear = advance(omega_dt - doppler_dt)

initial_polarization = initial_bx + 1.0j * initial_by
final_polarization = final_bx + 1.0j * final_by
initial_spectrum = np.fft.fft(initial_polarization)
full_polarization = np.fft.ifft(amplification_full * initial_spectrum)
linear_polarization = np.fft.ifft(amplification_linear * initial_spectrum)

window = np.abs(z - envelope_center) <= 2.5 * envelope_sigma
window_norm = np.linalg.norm(initial_polarization[window])
error_to_full = (
    np.linalg.norm((final_polarization - full_polarization)[window]) / window_norm
)
error_to_linear = (
    np.linalg.norm((final_polarization - linear_polarization)[window]) / window_norm
)

# Gate 1: the ungated band packet follows the FULL branch (calibrated:
# measured in-window relative error 2.2e-7, dominated by interface
# leakage of the implicit solve -- a uniform-density control run of this
# deck measures 1e-11; gated at ~450x).
assert error_to_full < 1.0e-4, f"in-band full-branch error {error_to_full:.3e}"
# Gate 2: ... and is far from the LINEAR branch (the advective Doppler is
# a large fraction of the rotation here; measured error 3.8e-1, a
# separation of 1.75e6; gated at 1e3).
assert error_to_linear > 1.0e3 * error_to_full, (
    f"branch separation too small: full {error_to_full:.3e} vs linear "
    f"{error_to_linear:.3e}"
)

# Frozen state.
np.testing.assert_allclose(final_bz, background_field, rtol=1.0e-14, atol=0.0)
np.testing.assert_array_equal(final_density, initial_density)

# Measured: 1 Newton iteration (the frozen state is exactly linear) and
# 22 GMRES iterations -- the advective Doppler rows are NOT in the
# pc_mhd_block fold (the dust gate's motivation).
newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
last_solve = newton_history[-1]
assert 1 <= last_solve[2] <= 8
assert (last_solve[4] <= 1.1e-12) or (last_solve[5] <= 1.1e-10)
gmres_iterations = last_solve[7]
assert 0 < gmres_iterations <= 40

print(f"band whistler CFL={whistler_cfl:.12g}")
print(
    f"band d_e={electron_skin_depth:.6g} m ({electron_skin_depth / cell_size:.3g} cells)"
)
print(f"carrier kappa={carrier_kappa:.6g}, linear rotation={carrier_rotation:.4g} rad")
print(f"in-band error to FULL branch={error_to_full:.6e}")
print(f"in-band error to LINEAR branch={error_to_linear:.6e}")
print(f"branch separation ratio={error_to_linear / error_to_full:.6g}")
print(f"Newton iterations={int(last_solve[2])}")
print(f"GMRES iterations={int(gmres_iterations)}")
