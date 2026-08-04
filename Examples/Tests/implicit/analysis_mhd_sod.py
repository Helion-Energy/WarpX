#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Sod shock tube for the implicit-MHD ion total-energy closure.

A periodic double Riemann problem (high state between two diaphragms)
launches mirror-image Sod problems; the right-moving one at the upper
diaphragm is compared against the exact solution of the gamma = 5/3
ideal-gas Riemann problem, computed here by Newton iteration on the star
pressure. Every reference value is derived from the simulation data itself
(left/right states from the initial plotfile), never from physical-constant
libraries, so the asserts are immune to CODATA revisions between the code
and the analysis environment.

Mode "total_energy" asserts the star-region density and pressure plateaus
match the exact solution to a few percent. The pressure is recovered from
the measured E_i and density with the exact star velocity for the kinetic
part, p = (gamma-1) (E_i - rho u_star^2 / 2), i.e. the E_i plateau test
expressed in pressure units. Mode "barotropic" runs the SAME input with the
default barotropic closure, whose isentropic law has the wrong shock jump
conditions, and asserts the density plateaus deviate -- calibrating that
the total_energy tolerance is discriminating rather than trivially
satisfied.

Usage: analysis_mhd_sod.py <initial_plotfile> <final_plotfile> <mode>
"""

import sys

import numpy as np
import yt

GAMMA = 5.0 / 3.0


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


def exact_riemann(rho_l, p_l, rho_r, p_r):
    """Exact ideal-gas Riemann solution for two states at rest (left
    rarefaction, contact, right shock), following Toro Ch. 4."""
    a_l = np.sqrt(GAMMA * p_l / rho_l)
    a_r = np.sqrt(GAMMA * p_r / rho_r)

    def f_and_df(p, rho_k, p_k, a_k):
        if p > p_k:  # shock
            big_a = 2.0 / ((GAMMA + 1.0) * rho_k)
            big_b = (GAMMA - 1.0) / (GAMMA + 1.0) * p_k
            f = (p - p_k) * np.sqrt(big_a / (p + big_b))
            df = np.sqrt(big_a / (p + big_b)) * (1.0 - 0.5 * (p - p_k) / (p + big_b))
        else:  # rarefaction
            exponent = (GAMMA - 1.0) / (2.0 * GAMMA)
            f = 2.0 * a_k / (GAMMA - 1.0) * ((p / p_k) ** exponent - 1.0)
            df = (p / p_k) ** (-(GAMMA + 1.0) / (2.0 * GAMMA)) / (rho_k * a_k)
        return f, df

    p_star = 0.5 * (p_l + p_r)
    for _ in range(40):
        f_l, df_l = f_and_df(p_star, rho_l, p_l, a_l)
        f_r, df_r = f_and_df(p_star, rho_r, p_r, a_r)
        dp = -(f_l + f_r) / (df_l + df_r)
        p_star = max(p_star + dp, 1.0e-10 * p_l)
        if abs(dp) < 1.0e-13 * p_star:
            break
    f_l, _ = f_and_df(p_star, rho_l, p_l, a_l)
    f_r, _ = f_and_df(p_star, rho_r, p_r, a_r)
    u_star = 0.5 * (f_r - f_l)
    assert p_r < p_star < p_l and u_star > 0.0
    rho_star_l = rho_l * (p_star / p_l) ** (1.0 / GAMMA)
    a_star_l = a_l * (p_star / p_l) ** ((GAMMA - 1.0) / (2.0 * GAMMA))
    g = (GAMMA - 1.0) / (GAMMA + 1.0)
    rho_star_r = rho_r * (p_star / p_r + g) / (g * p_star / p_r + 1.0)
    shock_speed = a_r * np.sqrt(
        (GAMMA + 1.0) / (2.0 * GAMMA) * p_star / p_r + (GAMMA - 1.0) / (2.0 * GAMMA)
    )
    return {
        "p_star": p_star,
        "u_star": u_star,
        "rho_star_l": rho_star_l,
        "rho_star_r": rho_star_r,
        "tail_speed": u_star - a_star_l,
        "shock_speed": shock_speed,
    }


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])
mode = sys.argv[3]
assert mode in ("total_energy", "barotropic")

initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()
initial_ion_energy = initial["boxlib", "implicit_mhd_ion_energy"].value.ravel()
final_ion_energy = final["boxlib", "implicit_mhd_ion_energy"].value.ravel()

number_of_cells = initial_density.size
z_lo = float(initial_ds.domain_left_edge[0])
z_hi = float(initial_ds.domain_right_edge[0])
cell_size = (z_hi - z_lo) / number_of_cells
z = z_lo + (np.arange(number_of_cells) + 0.5) * cell_size
time = float(final_ds.current_time - initial_ds.current_time)

# left/right states from the initial data
rho_l = np.max(initial_density)
rho_r = np.min(initial_density)
high = initial_density > 0.5 * (rho_l + rho_r)
if mode == "total_energy":
    p_l = (GAMMA - 1.0) * np.max(initial_ion_energy)
    p_r = (GAMMA - 1.0) * np.min(initial_ion_energy)
else:
    # the barotropic run leaves the (unused) ion-energy field zeroed, so
    # work with normalized states: the deck's p_ratio (a pure number, not
    # a physical constant) and the density ratio measured from the data
    p_l = 1.0
    p_r = 0.1
    rho_r = rho_r / rho_l
    rho_l = 1.0

# upper diaphragm (right-moving problem) from the initial density profile
high_indices = np.where(high)[0]
assert 0 < high_indices[0] and high_indices[-1] < number_of_cells - 1
z_diaphragm = z[high_indices[-1]] + 0.5 * cell_size

exact = exact_riemann(rho_l, p_l, rho_r, p_r)
# barotropic mode uses normalized states; rescale speeds with the sound
# speed is unnecessary because only density RATIOS are asserted there
if mode == "total_energy":
    tail = z_diaphragm + exact["tail_speed"] * time
    contact = z_diaphragm + exact["u_star"] * time
    shock = z_diaphragm + exact["shock_speed"] * time
else:
    # positions require dimensional speeds: recover the left sound speed
    # from the electron-energy field, which the barotropic run does evolve
    # and which the deck sets to pe_over_pi = 1e-3 of the ion pressure
    initial_electron_energy = initial[
        "boxlib", "implicit_mhd_electron_energy"
    ].value.ravel()
    p_l_dim = (GAMMA - 1.0) * np.max(initial_electron_energy) / 1.0e-3
    v0 = np.sqrt(p_l_dim / np.max(initial_density))
    tail = z_diaphragm + exact["tail_speed"] * v0 * time
    contact = z_diaphragm + exact["u_star"] * v0 * time
    shock = z_diaphragm + exact["shock_speed"] * v0 * time

# asymmetric margins: the (first-order, backward-Euler) contact smears over
# more cells than the self-sharpening shock or the rarefaction tail
tail_margin = 4.0 * cell_size
contact_margin = 8.0 * cell_size
shock_margin = 6.0 * cell_size
left_star = (z > tail + tail_margin) & (z < contact - contact_margin)
right_star = (z > contact + contact_margin) & (z < shock - shock_margin)
assert left_star.sum() >= 3 and right_star.sum() >= 2

rho_left_star = np.mean(final_density[left_star])
rho_right_star = np.mean(final_density[right_star])
rho_norm = np.max(initial_density) if mode == "barotropic" else 1.0
density_deviations = (
    abs(rho_left_star / rho_norm / exact["rho_star_l"] - 1.0),
    abs(rho_right_star / rho_norm / exact["rho_star_r"] - 1.0),
)
print(f"mode = {mode}")
print(
    f"rho*_L: measured {rho_left_star / rho_norm:.5e}, "
    f"exact {exact['rho_star_l']:.5e} "
    f"(deviation {density_deviations[0]:.3e})"
)
print(
    f"rho*_R: measured {rho_right_star / rho_norm:.5e}, "
    f"exact {exact['rho_star_r']:.5e} "
    f"(deviation {density_deviations[1]:.3e})"
)

if mode == "total_energy":
    # both star regions move at the exact contact speed u*; subtracting the
    # corresponding kinetic energy turns the measured E_i plateau into a
    # pressure plateau, which must match the exact p*
    kinetic = 0.5 * final_density * exact["u_star"] ** 2
    pressure = (GAMMA - 1.0) * (final_ion_energy - kinetic)
    p_left_star = np.mean(pressure[left_star])
    p_right_star = np.mean(pressure[right_star])
    pressure_deviations = (
        abs(p_left_star / exact["p_star"] - 1.0),
        abs(p_right_star / exact["p_star"] - 1.0),
    )
    print(
        f"p* left window: measured {p_left_star:.5e}, "
        f"exact {exact['p_star']:.5e} "
        f"(deviation {pressure_deviations[0]:.3e})"
    )
    print(
        f"p* right window: measured {p_right_star:.5e}, "
        f"exact {exact['p_star']:.5e} "
        f"(deviation {pressure_deviations[1]:.3e})"
    )

    tolerance = 0.04
    assert max(density_deviations) < tolerance, (
        f"star-region density plateau off by {max(density_deviations):.3e}"
    )
    assert max(pressure_deviations) < tolerance, (
        f"star-region pressure plateau off by {max(pressure_deviations):.3e}"
    )
    # conservative transport: total mass is preserved to solver tolerance
    np.testing.assert_allclose(
        np.sum(final_density), np.sum(initial_density), rtol=1.0e-10, atol=0.0
    )
else:
    # the barotropic closure must MISS the exact plateaus by much more than
    # the total_energy tolerance, or the assert above proves nothing
    assert max(density_deviations) > 0.08, (
        f"barotropic calibration unexpectedly matched the exact solution "
        f"(max deviation {max(density_deviations):.3e}); the total_energy "
        f"tolerance is not discriminating"
    )

print(f"mode = {mode}: PASS")
