#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Calibrated check of the pedestal-band velocity relaxation (drag).

Setup: zero field, zero normal velocity, uniform electron and ion
pressures, and a UNIFORM transverse velocity v0 (in x, the plotted
momentum component) -- a static contact with tangential slip, whose
HLLD fluxes vanish identically -- with the density rho0 in the central
half of the domain and exactly the dynamic pedestal rho_ped = f * rho0
outside. The pedestal-band drag (implicit_mhd.halo_pedestal_drag_rate
= nu) is then the ONLY leading-order dynamics:

  * band cells (rho_old = rho_ped, drag weight exactly 1) relax as the
    theta-discretized linear decay, per step
        m_x -> m_x * A,   A = (1 - (1-theta) nu dt) / (1 + theta nu dt);
  * bulk cells (rho0 = 1000 rho_ped >= 2 rho_ped, drag weight exactly
    zero) are invariant -- the C^1 engagement window closes at twice
    the pedestal;
  * under the total_energy closure the matched kinetic drain makes the
    band's E_i track its kinetic part (internal energy invariant):
    E_i -> internal_0 + kinetic_0 * A^(2N), discretely exact at
    theta = 1/2;
  * density and electron energy are preserved (the drag is a pure
    momentum sink; no mass is demanded from the gated band).

Tolerances: the exact solution of the drag ODE is polluted only by
(i) linear/nonlinear solver tolerance noise, spread over all unknowns
in REFERENCE-scaled units (so it looks largest relative to the tiny
band density), and (ii) the theta-STAGE quadratic kinetic remainder of
the matched drain (the drain is exact for the end-of-step states; the
midpoint state carries an O(|dm|^2) internal-pressure remainder that
drives a tiny genuine acoustic response, ~1e-8 relative over 8 steps).
The asserts leave roughly an order of magnitude of margin over the
measured deviations without admitting any physical drift.
"""

import sys

import numpy as np
import warpx_constants as constants
import yt


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])

number_of_cells = 64
domain_length = 1.0
cell_size = domain_length / number_of_cells
number_density = 1.0e18
rho0 = number_density * constants.proton_mass
pedestal_fraction = 1.0e-3
rho_halo = pedestal_fraction * rho0
transverse_velocity = 2.0e4
drag_rate = 5.0e7
dt = 1.0e-9
theta = 0.5
steps = 8

z = (np.arange(number_of_cells) + 0.5) * cell_size
bulk = (z > 0.25) & (z < 0.75)
# Band cells NOT adjacent to the contact: the two cells touching the
# bulk feel the theta-stage kinetic remainder of the matched drain
# directly (a genuine, tiny acoustic response, see the module
# docstring) and get an order of magnitude more slack.
edge = np.zeros_like(bulk)
edge[1:] |= bulk[:-1] != bulk[1:]
edge[:-1] |= bulk[:-1] != bulk[1:]
band_interior = ~bulk & ~edge
band_edge = ~bulk & edge

# Theta-discretized per-step decay factor of the drag ODE m' = -nu m.
x = drag_rate * dt
decay = (1.0 - (1.0 - theta) * x) / (1.0 + theta * x)

initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()

# The drag is a pure momentum sink: no mass moves anywhere (band noise
# floor: solver tolerance in reference-scaled units plus the theta-stage
# kinetic remainder, measured ~1e-8 relative).
np.testing.assert_allclose(initial_density[bulk], rho0, rtol=1.0e-14)
np.testing.assert_allclose(initial_density[~bulk], rho_halo, rtol=1.0e-12)
np.testing.assert_allclose(final_density, initial_density, rtol=1.0e-7)

# Loaded momentum: rho * v0 in the plotted (x) component.
momentum_initial = initial["boxlib", "implicit_mhd_momentum_density"].value.ravel()
momentum_final = final["boxlib", "implicit_mhd_momentum_density"].value.ravel()
np.testing.assert_allclose(
    momentum_initial, initial_density * transverse_velocity, rtol=1.0e-13
)

# Bulk: rho0 = 1000 rho_ped is far above the 2 rho_ped engagement
# window -- the drag weight is exactly zero there.
np.testing.assert_allclose(momentum_final[bulk], momentum_initial[bulk], rtol=1.0e-9)

# Band: theta-discretized decay at the full rate (contact-adjacent
# cells get an order of magnitude more slack, see above).
np.testing.assert_allclose(
    momentum_final[band_interior],
    momentum_initial[band_interior] * decay**steps,
    rtol=1.0e-7,
)
np.testing.assert_allclose(
    momentum_final[band_edge],
    momentum_initial[band_edge] * decay**steps,
    rtol=1.0e-5,
)
# Discrimination: the drag really acted (decay^8 ~ 0.67).
assert np.all(momentum_final[~bulk] < 0.8 * momentum_initial[~bulk])

# Electron energy: internal only, preserved everywhere.
initial_electron = initial["boxlib", "implicit_mhd_electron_energy"].value.ravel()
final_electron = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()
np.testing.assert_allclose(final_electron, initial_electron, rtol=1.0e-9)

# Ion energy (total_energy closure): the matched kinetic drain keeps
# the internal part invariant, so E_i tracks its kinetic part, which
# decays as decay^(2N) (discretely exact at theta = 1/2).
initial_ion = initial["boxlib", "implicit_mhd_ion_energy"].value.ravel()
final_ion = final["boxlib", "implicit_mhd_ion_energy"].value.ravel()
kinetic_initial = 0.5 * momentum_initial**2 / initial_density
internal_initial = initial_ion - kinetic_initial
ion_prediction = internal_initial + kinetic_initial * decay ** (2 * steps)
np.testing.assert_allclose(final_ion[bulk], initial_ion[bulk], rtol=1.0e-9)
np.testing.assert_allclose(
    final_ion[band_interior], ion_prediction[band_interior], rtol=1.0e-7
)
np.testing.assert_allclose(final_ion[band_edge], ion_prediction[band_edge], rtol=1.0e-5)

# Nothing electromagnetic happens: B stays zero, and E = -u_e x B = 0.
for data in (initial, final):
    for field_name in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"):
        np.testing.assert_allclose(
            data["boxlib", field_name].value, 0.0, rtol=0.0, atol=1.0e-20
        )
