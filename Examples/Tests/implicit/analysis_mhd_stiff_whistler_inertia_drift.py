#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Validate the ADVECTIVE electron-inertia piece: drift-Doppler whistler cap.

Same broadband circularly polarized whistler as
analysis_mhd_stiff_whistler_inertia.py, plus a frozen ion z-drift u0.
Because (curl B)_z = 0 in 1D, the electron current must carry
Je_z = -Ji_z, so the electrons stream at exactly u0 as well -- the only
1D configuration in which the inertia term's advective piece
-(Je.grad)(Je/rho) is nonzero (without a z-flow it is structurally zero:
(Je.grad) = Je_z d/dz). The system stays EXACTLY linear in the transverse
B (je_z, u and rho all frozen), so per staggered Fourier mode the theta
step advances the circular eigenmodes by

    G_full = ((1+kappa) + i (1-theta) W dt) / ((1+kappa) - i theta W dt),
    W dt   = w dt - (1+kappa) k_c u0 dt,

the drift-Doppler generalization of the capped branch: the ideal -u x B
advection contributes -k_c u0 (interpolation chain
k_eff cos(k dz/2) = sin(k dz)/dz = k_c) and the inertia advective piece
contributes the kappa-weighted portion -kappa k_c u0 (nodal central 2dz
difference, also exactly k_c) -- together the DISCRETE Galilean shift
(1+kappa) k_c u0 of the capped rotation w/(1+kappa), i.e. the whole
co-moving plasma Dopplers the capped whistler by k u0 in the continuum
limit. The LINEAR (dJe/dt-only) inertia form retains the ideal Doppler
but drops the kappa-weighted advective portion:

    W_lin dt = w dt - k_c u0 dt,

which separates from the full branch by the kappa k_c u0 dt rotation --
this is the discriminator the dust gate's CI leans on (a build whose
blend is flipped runs the linear form at the seeded density and lands on
W_lin, failing the tight G_full gate below).
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

number_density = 1.0e20
mass_density_reference = number_density * constants.proton_mass
background_field = 0.1
perturbation_field = 1.0e-4
drift_velocity = 5.0e4
mode_numbers = np.array([1, 2, 4, 7, 11, 16, 21, 27])
mode_weights = np.array([1.0, 0.7, -0.5, 0.35, -0.25, 0.18, 0.12, -0.08])
theta = 0.5
# reduced_electron_mass_ratio = 1: m_e_eff = m_p.
effective_electron_mass = constants.proton_mass
electron_skin_depth = np.sqrt(
    effective_electron_mass
    / (constants.mu_0 * number_density * constants.elementary_charge**2)
)

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
hall_diffusivity = background_field / (
    constants.mu_0 * constants.elementary_charge * number_density
)
whistler_cfl = hall_diffusivity * dt / cell_size**2
np.testing.assert_allclose(whistler_cfl, 20.0, rtol=2.0e-12, atol=0.0)

z = (
    float(initial_ds.domain_left_edge[0])
    + (np.arange(number_of_cells) + 0.5) * cell_size
)
phases = 2.0 * np.pi * np.outer(mode_numbers, z) / domain_length
initial_bx_expected = perturbation_field * np.sum(
    mode_weights[:, np.newaxis] * np.cos(phases), axis=0
)
initial_by_expected = perturbation_field * np.sum(
    mode_weights[:, np.newaxis] * np.sin(phases), axis=0
)
np.testing.assert_allclose(initial_bx, initial_bx_expected, rtol=3.0e-12, atol=2.0e-15)
np.testing.assert_allclose(initial_by, initial_by_expected, rtol=3.0e-12, atol=2.0e-15)

wavenumbers = 2.0 * np.pi * np.fft.fftfreq(number_of_cells, d=cell_size)
effective_wavenumbers = 2.0 * np.sin(0.5 * wavenumbers * cell_size) / cell_size
central_wavenumbers = np.sin(wavenumbers * cell_size) / cell_size
kappa = (effective_wavenumbers * electron_skin_depth) ** 2
omega_dt = hall_diffusivity * effective_wavenumbers**2 * dt
doppler_dt = central_wavenumbers * drift_velocity * dt


def advance(total_omega_dt):
    return ((1.0 + kappa) + 1.0j * (1.0 - theta) * total_omega_dt) / (
        (1.0 + kappa) - 1.0j * theta * total_omega_dt
    )


amplification_full = advance(omega_dt - (1.0 + kappa) * doppler_dt)
amplification_linear = advance(omega_dt - doppler_dt)
amplification_nodrift = advance(omega_dt)

initial_polarization = initial_bx + 1.0j * initial_by
final_polarization = final_bx + 1.0j * final_by
initial_spectrum = np.fft.fft(initial_polarization)
expected_polarization = np.fft.ifft(amplification_full * initial_spectrum)
linear_polarization = np.fft.ifft(amplification_linear * initial_spectrum)
nodrift_polarization = np.fft.ifft(amplification_nodrift * initial_spectrum)

# Gate 1: the measured rotation matches the drift-Doppler capped branch
# tightly (the transverse system is exactly linear, so the only error is
# the Newton/GMRES tolerance; measured 7.3e-14 relative).
np.testing.assert_allclose(
    final_polarization,
    expected_polarization,
    rtol=2.0e-8,
    atol=2.0e-12,
)

# Gate 2: the LINEAR-form branch (the advective piece dropped -- what a
# flipped dust-gate blend would produce at this above-threshold density)
# must be excluded by many orders. Calibration: the full-branch relative
# error is 7.3e-14 while the linear branch sits at 1.26e-1 relative (the
# kappa k_c u0 dt rotation gap), a separation of 1.7e12; require four
# orders of margin.
distance_to_full = np.linalg.norm(final_polarization - expected_polarization)
distance_to_linear = np.linalg.norm(final_polarization - linear_polarization)
linear_branch_margin = distance_to_linear / max(distance_to_full, 1e-30)
assert linear_branch_margin > 1.0e4, (
    f"advective-piece discrimination failed: |B - G_lin B0| / |B - G_full "
    f"B0| = {linear_branch_margin:.3e} (the measured rotation does not "
    "carry the kappa-weighted electron-drift Doppler of the full inertia "
    "form)"
)

# Gate 3: the drift matters at all (the no-drift capped branch is far) --
# the advective piece has no other 1D coverage.
distance_to_nodrift = np.linalg.norm(final_polarization - nodrift_polarization)
nodrift_margin = distance_to_nodrift / max(distance_to_full, 1e-30)
assert nodrift_margin > 1.0e4, (
    f"drift-Doppler coverage failed: margin {nodrift_margin:.3e}"
)

# Frozen state: Bz stays exactly B0, the fluid untouched.
np.testing.assert_allclose(final_bz, background_field, rtol=1.0e-14, atol=0.0)
np.testing.assert_array_equal(final_density, initial_density)
np.testing.assert_allclose(
    final_density, mass_density_reference, rtol=5.0e-14, atol=0.0
)

assert input_value("jacobian.pc_type") == "pc_mhd_block"
assert input_value("pc_mhd_block.include_electron_inertia_coupling") == "true"
assert input_value("implicit_mhd.fluid_flux") == "central"
assert input_value("implicit_mhd.evolve_ion_fluid") == "false"
assert input_value("hybrid_pic_model.include_electron_inertia") == "true"
assert input_value("implicit_mhd.velocity_z(x,y,z)") == "u0"

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
last_solve = newton_history[-1]
assert 1 <= last_solve[2] <= 8
assert (last_solve[4] <= 1.1e-12) or (last_solve[5] <= 1.1e-10)
gmres_iterations = last_solve[7]
# The advective Doppler rows are NOT in the pc_mhd_block fold (that
# unrepresentability is the dust gate's entire motivation), so this deck
# takes more GMRES iterations than the drift-free inertia test's 1;
# measured 16, gated at ~2x.
assert 0 < gmres_iterations <= 30

relative_wave_error = np.linalg.norm(
    final_polarization - expected_polarization
) / np.linalg.norm(initial_polarization)
print(f"grid whistler CFL={whistler_cfl:.12g}")
print(f"effective electron skin depth d_e={electron_skin_depth:.6g} m")
print(f"drift velocity u0={drift_velocity:.6g} m/s")
print(f"full-branch relative B+ error={relative_wave_error:.12e}")
print(f"|B - G_full B0|={distance_to_full:.6e}")
print(f"|B - G_lin  B0|={distance_to_linear:.6e}")
print(f"|B - G_nodrift B0|={distance_to_nodrift:.6e}")
print(f"linear-branch margin={linear_branch_margin:.6e}")
print(f"no-drift margin={nodrift_margin:.6e}")
print(f"Newton iterations={int(last_solve[2])}")
print(f"GMRES iterations={int(gmres_iterations)}")
