#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Dust-gate effectiveness: linearized band physics + no-worse GMRES.

Same deck as analysis_mhd_inertia_dustband.py with the dust gate armed at
5x the band density (electron_inertia_linear_below = 1e21 m_p): the
blend weight is EXACTLY 0.0 in double precision at the band (s = -8/3,
tanh saturates) and EXACTLY 1.0 at the 2x-gate background, so the band's
inertia assembly keeps only the linear dJe/dt piece -- the response the
pc_mhd_block rows fold -- while the background keeps the bit-identical
full form. Gates:

  (a) the band packet follows the LINEAR capped-whistler analytic
      advance G_lin (the advective Doppler removed), far from the FULL
      branch the ungated baseline (dependency) landed on;
  (b) the gated GMRES iteration count does not exceed the ungated
      baseline's -- below the gate, residual and preconditioner agree by
      construction (the production mechanism; on this small 1D case the
      measured difference is modest: see the printed ratio).
"""

import re
import sys
from pathlib import Path

import numpy as np
import warpx_constants as constants
import yt

baseline_directory = "../test_1d_theta_implicit_mhd_inertia_dustband"


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


def input_value(name):
    # Command-line overrides append AFTER the input-file entry in the
    # dumped table; the last occurrence is the effective value.
    matches = re.findall(
        rf"^{re.escape(name)}\s*=\s*([^#\s]+)", used_inputs, re.MULTILINE
    )
    assert matches
    return matches[-1]


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])
used_inputs = Path("warpx_used_inputs").read_text()

band_number_density = 2.0e20
floor_number_density = band_number_density / 1.5
background_number_density = 2.0e21
gate_number_density = 1.0e21
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


# The recast's C-infinity Ohm division guard (see the baseline analysis):
# the band's whistler rate uses this guarded density (1.206x at 1.5x the
# floor).
def smooth_positive_floor(value, floor):
    excess = value - floor
    return floor + 0.5 * (excess + np.sqrt(excess * excess + floor * floor))


hall_number_density = smooth_positive_floor(band_number_density, floor_number_density)

# The gate must be armed at 1e21 m_p = 5x band, 0.5x background.
assert input_value("hybrid_pic_model.electron_inertia_linear_below") == "rho_gate"
assert input_value("my_constants.rho_gate") == "1.0e21*m_p"


# ---- Blend placement proof (double-precision exactness) ----
def blend_weight(rho, rho_c):
    s = (rho - rho_c) / (0.3 * rho_c)
    return 0.5 * (1.0 + np.tanh(s * (1.0 + s * s)))


rho_band = band_number_density * constants.proton_mass
rho_hi = background_number_density * constants.proton_mass
rho_gate = gate_number_density * constants.proton_mass
assert blend_weight(rho_band, rho_gate) == 0.0  # band: exactly linear
assert blend_weight(rho_hi, rho_gate) == 1.0  # background: exactly full
assert 1.0 - blend_weight(3.0 * rho_gate, rho_gate) < 1.0e-12

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
error_to_linear = (
    np.linalg.norm((final_polarization - linear_polarization)[window]) / window_norm
)
error_to_full = (
    np.linalg.norm((final_polarization - full_polarization)[window]) / window_norm
)

# Gate (a): the gated band packet follows the LINEAR branch (calibrated:
# measured in-window relative error 1.5e-7, interface leakage; gated at
# ~650x) and is far from the FULL branch the ungated baseline follows
# (measured error 3.8e-1, a separation of 2.5e6; gated at 1e3).
assert error_to_linear < 1.0e-4, (
    f"in-band linear-branch error {error_to_linear:.3e} (the gate did not "
    "linearize the band's inertia assembly)"
)
assert error_to_full > 1.0e3 * error_to_linear, (
    f"branch separation too small: linear {error_to_linear:.3e} vs full "
    f"{error_to_full:.3e}"
)

# Frozen state.
np.testing.assert_allclose(final_bz, background_field, rtol=1.0e-14, atol=0.0)
np.testing.assert_array_equal(final_density, initial_density)

# Gate (b): GMRES no worse than the ungated baseline (dependency).
# Measured: 22 gated vs 22 ungated (ratio 1.00) -- on this small linear
# 1D case the count is dominated by the background's (full-form)
# unrepresented advective rows, so the gate is exactly no-worse here; the
# production win is the NONLINEAR 1/rho^2 pieces the frozen PC cannot
# fold at all, which this frozen-state deck cannot exhibit in the count.
newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
baseline_history = np.atleast_2d(np.loadtxt(f"{baseline_directory}/diags/newton.txt"))
last_solve = newton_history[-1]
baseline_solve = baseline_history[-1]
assert 1 <= last_solve[2] <= 8
assert (last_solve[4] <= 1.1e-12) or (last_solve[5] <= 1.1e-10)
gated_gmres = last_solve[7]
ungated_gmres = baseline_solve[7]
assert 0 < gated_gmres <= ungated_gmres, (
    f"gated GMRES count {int(gated_gmres)} exceeds the ungated baseline's "
    f"{int(ungated_gmres)} (the gate must restore PC/residual agreement "
    "in the band, never worsen the solve)"
)

print(
    f"band d_e={electron_skin_depth:.6g} m ({electron_skin_depth / cell_size:.3g} cells)"
)
print(
    f"blend weight at band={blend_weight(rho_band, rho_gate)!r}, "
    f"at background={blend_weight(rho_hi, rho_gate)!r}"
)
print(f"in-band error to LINEAR branch={error_to_linear:.6e}")
print(f"in-band error to FULL branch={error_to_full:.6e}")
print(f"branch separation ratio={error_to_full / error_to_linear:.6g}")
print(
    f"GMRES iterations gated={int(gated_gmres)} ungated={int(ungated_gmres)} "
    f"(ratio {gated_gmres / ungated_gmres:.3f})"
)
print(f"Newton iterations gated={int(last_solve[2])} ungated={int(baseline_solve[2])}")
