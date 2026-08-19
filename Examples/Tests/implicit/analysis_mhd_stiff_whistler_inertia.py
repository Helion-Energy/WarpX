#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Validate the electron-inertia CAP of the whistler branch at k d_e ~ 1.

Same broadband circularly polarized whistler as
analysis_mhd_stiff_whistler_central.py, on the conservative-form recast
with the electron-inertia Ohm term on and reduced_electron_mass_ratio = 1
(m_e_eff = m_p), so the effective electron skin depth crosses the seeded
spectrum: kappa = (k_eff d_e)^2 runs from 0.02 (mode 1) through 0.96
(mode 7) to 8.0 (mode 27).

With frozen ions and u = 0 the transverse recast system with inertia is
EXACTLY linear: per staggered Fourier mode the theta step advances the
circular eigenmodes by

    G_capped = ((1+kappa) + i (1-theta) w dt) / ((1+kappa) - i theta w dt),

the CAPPED whistler branch (effective rotation w/(1+kappa) -- the
d_e^2-regularized dispersion of Angus et al.); a FLIPPED inertia sign
produces the spurious resonance branch with (1-kappa) in place of
(1+kappa), which only separates from the cap near kappa ~ 1 -- exactly
what this test discriminates (the two branches are indistinguishable at
kappa << 1, so no other test can catch a sign error).
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
mode_numbers = np.array([1, 2, 4, 7, 11, 16, 21, 27])
mode_weights = np.array([1.0, 0.7, -0.5, 0.35, -0.25, 0.18, 0.12, -0.08])
theta = 0.5
# reduced_electron_mass_ratio = 1: m_e_eff = m_p, so
# d_e^2 = m_e_eff / (mu0 n0 e^2).
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

# The residual advances each staggered circular eigenmode B+ = Bx+i*By by
#
#     theta (1 + kappa)(B1 - B0) - i theta w dt B^theta = 0,
#     kappa = (k_eff d_e)^2,  w = D_H k_eff^2,
#
# (the inertia rows act on the INCREMENT: the theta-stage two-point
# dJe/dt is exactly (curl(B1 - B0)/mu0)/dt at theta = 1/2), so
#
#     G_capped = ((1+kappa) + i (1-theta) w dt)/((1+kappa) - i theta w dt),
#
# the same unit-modulus theta form as the massless test with the rotation
# capped by 1/(1 + kappa). The wrong-sign branch replaces (1+kappa) with
# (1-kappa): near-identical at kappa << 1, violently different at the
# mode-7 kappa = 0.96 (rotation 27x instead of 0.51x the massless angle)
# and rotation-reversed for kappa > 1.
wavenumbers = 2.0 * np.pi * np.fft.fftfreq(number_of_cells, d=cell_size)
effective_wavenumbers = 2.0 * np.sin(0.5 * wavenumbers * cell_size) / cell_size
kappa = (effective_wavenumbers * electron_skin_depth) ** 2
omega_dt = hall_diffusivity * effective_wavenumbers**2 * dt
amplification_capped = ((1.0 + kappa) + 1.0j * (1.0 - theta) * omega_dt) / (
    (1.0 + kappa) - 1.0j * theta * omega_dt
)
amplification_wrong = ((1.0 - kappa) + 1.0j * (1.0 - theta) * omega_dt) / (
    (1.0 - kappa) - 1.0j * theta * omega_dt
)

# The seeded spectrum must actually cross the branch-splitting point.
kappa_modes = (
    2.0 * np.sin(np.pi * mode_numbers / number_of_cells) / cell_size
    * electron_skin_depth
) ** 2
assert np.min(np.abs(kappa_modes - 1.0)) < 0.1, (
    f"no seeded mode near kappa = 1 (kappa = {kappa_modes}): the branch "
    "discrimination would be inconclusive"
)
assert np.max(kappa_modes) > 2.0

initial_polarization = initial_bx + 1.0j * initial_by
final_polarization = final_bx + 1.0j * final_by
initial_spectrum = np.fft.fft(initial_polarization)
expected_polarization = np.fft.ifft(amplification_capped * initial_spectrum)
wrong_polarization = np.fft.ifft(amplification_wrong * initial_spectrum)

# Gate 1: the measured rotation matches the CAPPED branch tightly (the
# transverse system is exactly linear, so the only error is the Newton
# tolerance; the massless variant of this deck measures ~1e-14).
np.testing.assert_allclose(
    final_polarization,
    expected_polarization,
    rtol=2.0e-8,
    atol=2.0e-12,
)

# Gate 2 (sign discipline): the wrong-sign branch must be excluded by
# many orders of magnitude. Calibration: the capped-branch residual is
# ~4e-16 T while |final - wrong| is ~2.6e-4 T (the branch gap at the
# kappa ~ 1 and kappa > 1 modes), a separation of ~6e11; require six
# orders of margin.
distance_to_capped = np.linalg.norm(final_polarization - expected_polarization)
distance_to_wrong = np.linalg.norm(final_polarization - wrong_polarization)
branch_margin = distance_to_wrong / max(distance_to_capped, 1e-30)
assert branch_margin > 1.0e6, (
    f"branch discrimination failed: |B - G_wrong B0| / |B - G_capped B0| "
    f"= {branch_margin:.3e} (the measured rotation does not distinguish "
    "the capped whistler from the spurious resonance)"
)

# The 1D transverse curl has no Bz row: Bz stays exactly B0, and frozen
# ions leave the fluid state untouched.
np.testing.assert_allclose(final_bz, background_field, rtol=1.0e-14, atol=0.0)
np.testing.assert_array_equal(final_density, initial_density)
np.testing.assert_allclose(
    final_density, mass_density_reference, rtol=5.0e-14, atol=0.0
)

assert input_value("jacobian.pc_type") == "pc_mhd_block"
assert input_value("pc_mhd_block.include_hall_mhd_coupling") == "true"
assert input_value("pc_mhd_block.include_electron_inertia_coupling") == "true"
assert input_value("pc_mhd_block.resistive_solver") == "banded"
assert input_value("pc_mhd_block.resistive_validate_assembly") == "true"
assert input_value("implicit_mhd.fluid_flux") == "central"
assert input_value("implicit_mhd.evolve_ion_fluid") == "false"
assert input_value("hybrid_pic_model.include_hall_term") == "true"
assert input_value("hybrid_pic_model.include_electron_inertia") == "true"
assert input_value("hybrid_pic_model.reduced_electron_mass_ratio") == "1.0"

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
last_solve = newton_history[-1]
assert 1 <= last_solve[2] <= 8
assert (last_solve[4] <= 1.1e-12) or (last_solve[5] <= 1.1e-10)
# Preconditioner-effectiveness gate, calibrated against the SAME deck run
# with jacobian.pc_type = none (gmres.max_iterations raised so restarts
# cannot mask the count): the unpreconditioned solve of this
# inertia-capped whistler state takes 51 total GMRES iterations. Require
# the banded Hall+inertia block to remove at least 88% of them (gate 6);
# the measured preconditioned count is 1 -- the frozen inertia mass plus
# whistler rows ARE the transverse Jacobian of this linear state -- so
# the margin covers platform variation many times over.
unpreconditioned_gmres_iterations = 51
gmres_iteration_gate = 6
gmres_iterations = last_solve[7]
assert 0 < gmres_iterations <= gmres_iteration_gate

relative_wave_error = np.linalg.norm(
    final_polarization - expected_polarization
) / np.linalg.norm(initial_polarization)
print(f"grid whistler CFL={whistler_cfl:.12g}")
print(f"effective electron skin depth d_e={electron_skin_depth:.6g} m")
print(
    "modal kappa=(k_eff d_e)^2 range="
    f"{np.min(kappa_modes):.6g}..{np.max(kappa_modes):.6g}"
)
print(f"capped-branch relative B+ error={relative_wave_error:.12e}")
print(f"|B - G_capped B0|={distance_to_capped:.6e}")
print(f"|B - G_wrong  B0|={distance_to_wrong:.6e}")
print(f"branch-discrimination margin={branch_margin:.6e}")
print(f"Newton iterations={int(last_solve[2])}")
print(f"GMRES iterations={int(gmres_iterations)}")
print(
    "preconditioned/unpreconditioned GMRES ratio="
    f"{gmres_iterations / unpreconditioned_gmres_iterations:.4f}"
    f" (gate {gmres_iteration_gate})"
)
