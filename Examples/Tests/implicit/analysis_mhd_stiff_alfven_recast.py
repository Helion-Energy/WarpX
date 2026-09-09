#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Gate of the large-CFL Alfven step on the conservative-form recast, with and without the coupling block.

usage: analysis_mhd_stiff_alfven_recast.py <initial plotfile> <final plotfile> <mode> [<baseline newton.txt>]

mode = baseline: today's block preconditioner (which is the identity on this deck: at reference Alfven CFL 20 the
       Faraday corrector stands down and the resistive block is inactive) -- records the Newton and GMRES counts
       (calibrated 2 Newton / 311-320 GMRES per solve) and checks the wave physics;
mode = coupling: pc_mhd_block.coupling_block = alfven_schur -- the same physics gates, and the GMRES total must be at
       most a third of the baseline run's (the block carries the momentum <-> B coupling exactly on this 1D wave);
       Newton at most the baseline's + 1.
The physics gates are those of analysis_mhd_stiff_alfven.py (the exact discrete theta-method prediction of the
broadband linear Alfven wave, mass conservation, positivity): the block changes the Newton iterates, never the
converged state.
"""

import re
import sys
from pathlib import Path

import numpy as np
import warpx_constants as constants
import yt


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    return ds, data


def input_value(name):
    match = re.search(rf"^{re.escape(name)}\s*=\s*([^#\s]+)", used_inputs, re.MULTILINE)
    assert match is not None, name
    return match.group(1)


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])
mode = sys.argv[3]
assert mode in ("baseline", "coupling"), mode
used_inputs = Path("warpx_used_inputs").read_text()

number_density = 1.0e20
mass_density_reference = number_density * constants.proton_mass
pressure_reference = number_density * 10.0 * constants.elementary_charge
gamma = 5.0 / 3.0
background_field = 0.1
alfven_speed = background_field / np.sqrt(constants.mu_0 * mass_density_reference)

initial_bx = initial["boxlib", "Bx"].value.ravel()
final_bx = final["boxlib", "Bx"].value.ravel()
initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_energy = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()

number_of_cells = initial_bx.size
domain_length = float((initial_ds.domain_right_edge - initial_ds.domain_left_edge)[0])
cell_size = domain_length / number_of_cells
dt = float(final_ds.current_time - initial_ds.current_time)
alfven_cfl = alfven_speed * dt / cell_size
np.testing.assert_allclose(alfven_cfl, 20.0, rtol=1.0e-12, atol=0.0)

# Exact discrete theta-method prediction of the linear Alfven eigenmodes (see analysis_mhd_stiff_alfven.py).
wavenumbers = 2.0 * np.pi * np.fft.fftfreq(number_of_cells, d=cell_size)
effective_wavenumbers = np.sin(wavenumbers * cell_size) / cell_size
theta_number = 0.5 * alfven_speed * dt * effective_wavenumbers
amplification = (1.0 - 1.0j * theta_number) / (1.0 + 1.0j * theta_number)
expected_bx = np.fft.ifft(np.fft.fft(initial_bx) * amplification).real
perturbation_ratio = np.max(np.abs(initial_bx)) / background_field
nonlinear_field_scale = 8.0 * background_field * perturbation_ratio**2
# The central recast carries a small explicit viscosity (viscous number 0.01) for its stabilization: allow it 5x the
# legacy scheme's relative tolerance on the wave.
np.testing.assert_allclose(final_bx, expected_bx, rtol=1.0e-2, atol=nonlinear_field_scale)
assert np.linalg.norm(final_bx - initial_bx) > 0.25 * np.linalg.norm(initial_bx)
np.testing.assert_allclose(np.sum(final_density), np.sum(initial_density), rtol=5.0e-14, atol=0.0)
assert np.all(np.isfinite(final_density))
assert np.all(np.isfinite(final_energy))
assert np.min(final_density) > 0.999 * mass_density_reference
assert np.min(final_energy) > 0.999 * pressure_reference / (gamma - 1.0)

assert input_value("implicit_mhd.fluid_flux") == "central"
assert input_value("jacobian.pc_type") == "pc_mhd_block"
coupling_block = input_value("pc_mhd_block.coupling_block") if "pc_mhd_block.coupling_block" in used_inputs else "none"
assert coupling_block == ("alfven_schur" if mode == "coupling" else "none"), coupling_block

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
last_solve = newton_history[-1]
newton_iterations = int(last_solve[2])
gmres_iterations = int(last_solve[7])
assert 1 <= newton_iterations <= 8
assert (last_solve[4] <= 1.1e-12) or (last_solve[5] <= 1.1e-10)
if mode == "baseline":
    # today's block PC is the identity here: 2 Newton, 311-320 GMRES calibrated; ceiling with margin
    assert 0 < gmres_iterations <= 450, gmres_iterations
else:
    baseline = np.atleast_2d(np.loadtxt(sys.argv[4]))[-1]
    baseline_gmres = int(baseline[7])
    baseline_newton = int(baseline[2])
    assert gmres_iterations <= baseline_gmres / 3, (gmres_iterations, baseline_gmres)
    assert newton_iterations <= baseline_newton + 1, (newton_iterations, baseline_newton)
    print(f"baseline GMRES={baseline_gmres} Newton={baseline_newton}")

relative_wave_error = np.linalg.norm(final_bx - expected_bx) / np.linalg.norm(initial_bx)
print(f"mode={mode} coupling_block={coupling_block}")
print(f"Alfven CFL={alfven_cfl:.12g}")
print(f"theta-method relative Bx error={relative_wave_error:.12e}")
print(f"Newton iterations={newton_iterations}")
print(f"GMRES iterations={gmres_iterations}")
