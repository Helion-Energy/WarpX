#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Calibrated check of the pedestal-band ion-energy relaxation.

Setup: zero field, zero normal velocity, uniform electron and ion
pressures, and a UNIFORM transverse velocity v0 (in x, the plotted
momentum component) -- a static contact with tangential slip, whose
HLLD fluxes vanish identically at t = 0 -- with the density rho0 in
the central half of the domain and exactly the dynamic pedestal
rho_ped = f * rho0 outside. The pedestal-band ion-energy relaxation
(implicit_mhd.halo_pedestal_energy_rate = nu, drag OFF) is then the
ONLY leading-order dynamics:

  * band cells (rho_old = rho_ped, mask weight exactly 1) drain their
    INTERNAL energy e = E_i - |m|^2/(2 rho) toward the frozen pedestal
    image e_ped = f * max_cell(e) as the theta-discretized linear
    decay, per step
        (e - e_ped) -> (e - e_ped) * A,
        A = (1 - (1-theta) nu dt) / (1 + theta nu dt),
    exact in the fully-engaged regime of the one-sided gate (the band
    stays >= 670 e_ped >> 2 e_ped here, where the rectifier is
    identically 1; the drain-only companion test covers the closed
    side);
  * the drain acts on the internal part relative to the CURRENT
    kinetic energy (the momentum drag owns the kinetic channel; here
    it is off), so momentum -- and with it the kinetic part of E_i --
    must stay invariant: E_i -> kinetic_0 + e_ped + (e_0 - e_ped) *
    A^N. A wrong implementation that relaxed the TOTAL E_i would land
    ~4% of E_i away from this prediction, orders of magnitude outside
    the asserted tolerances;
  * bulk cells (rho0 = 1000 rho_ped >= 2 rho_ped, mask weight exactly
    zero) are invariant -- the C^1 engagement window closes at twice
    the pedestal;
  * density and electron energy are preserved (the relaxation is a
    pure ion-internal-energy sink; no mass is demanded from the gated
    band).

Tolerances: unlike the drag test, the internal drain itself builds an
ion-pressure DEFICIT in the band (that is its function), so the
contact faces see a growing pressure jump ~(gamma-1) * (e_0 - e_ped)
* (1 - A^N) that drives a genuine, tiny acoustic inflow confined to
the four contact-adjacent cells over these 8 steps (sound crosses
~1e-3 of a cell; measured: 4e-4 relative on the band-edge density,
8e-4 on its momentum -- the inflow carries the SAME uniform u_x, so
momentum tracks the advected mass -- and 4e-7/8e-7 on the bulk-edge
side, weighted by the 1000x density ratio). Interior cells feel only
solver-tolerance noise plus second-order leakage (<= 1.3e-7 relative
everywhere, and the band-interior E_i lands on the theta-discretized
prediction to 1.3e-7). The asserts leave roughly an order of
magnitude of margin over the measured deviations without admitting
any physical drift.
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
electron_temperature = 0.01
ion_pressure = number_density * electron_temperature * constants.elementary_charge
gamma_i = 5.0 / 3.0
internal_energy = ion_pressure / (gamma_i - 1.0)
# Frozen pedestal internal-energy image: f * the cellwise internal
# peak (the invariant bulk), constant across all steps.
internal_pedestal = pedestal_fraction * internal_energy
transverse_velocity = 2.0e4
energy_rate = 5.0e7
dt = 1.0e-9
theta = 0.5
steps = 8

z = (np.arange(number_of_cells) + 0.5) * cell_size
bulk = (z > 0.25) & (z < 0.75)
# Cells NOT adjacent to the contact: the four cells touching it (two
# per side) feel the growing pressure-deficit jump directly (a
# genuine, tiny acoustic inflow, see the module docstring) and get
# more slack on BOTH sides of the contact.
edge = np.zeros_like(bulk)
edge[1:] |= bulk[:-1] != bulk[1:]
edge[:-1] |= bulk[:-1] != bulk[1:]
bulk_interior = bulk & ~edge
bulk_edge = bulk & edge
band_interior = ~bulk & ~edge
band_edge = ~bulk & edge

# Theta-discretized per-step decay factor of the relaxation ODE
# e' = -nu (e - e_ped).
x = energy_rate * dt
decay = (1.0 - (1.0 - theta) * x) / (1.0 + theta * x)

initial_density = initial["boxlib", "implicit_mhd_mass_density"].value.ravel()
final_density = final["boxlib", "implicit_mhd_mass_density"].value.ravel()

# The relaxation is a pure ion-internal-energy sink: away from the
# contact no mass moves (noise floor: solver tolerance in
# reference-scaled units); the four contact-adjacent cells carry the
# genuine acoustic inflow of the growing band pressure deficit
# (measured 4.1e-4 band side, 4.0e-7 bulk side).
np.testing.assert_allclose(initial_density[bulk], rho0, rtol=1.0e-14)
np.testing.assert_allclose(initial_density[~bulk], rho_halo, rtol=1.0e-12)
np.testing.assert_allclose(final_density[~edge], initial_density[~edge], rtol=1.0e-7)
np.testing.assert_allclose(
    final_density[bulk_edge], initial_density[bulk_edge], rtol=4.0e-6
)
np.testing.assert_allclose(
    final_density[band_edge], initial_density[band_edge], rtol=4.0e-3
)

# Loaded momentum: rho * v0 in the plotted (x) component.
momentum_initial = initial["boxlib", "implicit_mhd_momentum_density"].value.ravel()
momentum_final = final["boxlib", "implicit_mhd_momentum_density"].value.ravel()
np.testing.assert_allclose(
    momentum_initial, initial_density * transverse_velocity, rtol=1.0e-13
)

# Momentum is untouched away from the contact (the drag is off and
# the relaxation has no momentum component): this is the
# discrimination that the KE-following target drains only the
# internal part -- the kinetic channel stays with the (inactive)
# drag. The contact-adjacent cells carry the acoustic inflow, which
# advects the uniform u_x with the mass (measured 8.2e-4 band side,
# 8.1e-7 bulk side, i.e. momentum tracks the density change).
np.testing.assert_allclose(
    momentum_final[bulk_interior], momentum_initial[bulk_interior], rtol=1.0e-7
)
np.testing.assert_allclose(
    momentum_final[band_interior], momentum_initial[band_interior], rtol=1.0e-6
)
np.testing.assert_allclose(
    momentum_final[bulk_edge], momentum_initial[bulk_edge], rtol=8.0e-6
)
np.testing.assert_allclose(
    momentum_final[band_edge], momentum_initial[band_edge], rtol=8.0e-3
)

# Electron energy: internal only, preserved everywhere up to the tiny
# acoustic pdV response at the contact (measured 1.7e-6 band edge).
initial_electron = initial["boxlib", "implicit_mhd_electron_energy"].value.ravel()
final_electron = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()
np.testing.assert_allclose(final_electron[~edge], initial_electron[~edge], rtol=1.0e-6)
np.testing.assert_allclose(final_electron[edge], initial_electron[edge], rtol=2.0e-5)

# Ion energy (total_energy closure): the band's INTERNAL part decays
# toward the pedestal image at the exact theta-discretized rate while
# the kinetic part rides along unchanged; the bulk is invariant.
initial_ion = initial["boxlib", "implicit_mhd_ion_energy"].value.ravel()
final_ion = final["boxlib", "implicit_mhd_ion_energy"].value.ravel()
kinetic_initial = 0.5 * momentum_initial**2 / initial_density
internal_initial = initial_ion - kinetic_initial
np.testing.assert_allclose(internal_initial, internal_energy, rtol=1.0e-11)
ion_prediction = (
    kinetic_initial
    + internal_pedestal
    + (internal_initial - internal_pedestal) * decay**steps
)
np.testing.assert_allclose(
    final_ion[bulk_interior], initial_ion[bulk_interior], rtol=1.0e-7
)
np.testing.assert_allclose(final_ion[bulk_edge], initial_ion[bulk_edge], rtol=8.0e-6)
np.testing.assert_allclose(
    final_ion[band_interior], ion_prediction[band_interior], rtol=1.5e-6
)
np.testing.assert_allclose(final_ion[band_edge], ion_prediction[band_edge], rtol=1.5e-3)
# Discrimination: the relaxation really acted (the band's internal
# deviation retains decay^8 ~ 0.67 of itself, i.e. ~29% of E_i is
# gone) and did not overshoot.
internal_final = final_ion - kinetic_initial
ratio = (internal_final[~bulk] - internal_pedestal) / (
    internal_initial[~bulk] - internal_pedestal
)
assert np.all(ratio < 0.75)
assert np.all(ratio > 0.6)

# Nothing electromagnetic happens: B stays zero, and E = -u_e x B = 0.
for data in (initial, final):
    for field_name in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"):
        np.testing.assert_allclose(
            data["boxlib", field_name].value, 0.0, rtol=0.0, atol=1.0e-20
        )
