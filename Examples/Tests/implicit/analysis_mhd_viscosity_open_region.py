#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Two-region gate of the reference code's nu_op viscosity multiplier.

the reference code's step.f90 disip:199-201 scales the ion viscous coefficient by
nu_op_mul wherever the poloidal flux exceeds nu_op_bnd -- the tenuous
region. The deck realizes that as an axially sheared rotation,
v_theta = A sin(2 pi z / Lz) uniform in r, on a uniform column with a
uniform Bz. Three facts make each radius an independent 1D decay:

  * psi(r) = int_0^r Bz r' dr' = B0 r^2/2 is monotone in r, so the gate
    cuts the column at a known radius (r_cut = 0.25 m, the face between
    cells 15 and 16);
  * the initial radial gradient of v_theta is identically zero, the axis
    face carries the r = 0 metric weight, and the r_max PEC ghost mirrors
    v_theta EVENLY -- so no radial viscous flux exists anywhere at t = 0
    and none is forced at either boundary;
  * the guide field is dynamically inert (k v_A is four orders below
    nu k^2), so the rotation does not become a torsional Alfven wave.

Each radius therefore decays at exactly nu(r) k_eff^2 with
k_eff^2 = (4/dz^2) sin^2(k dz/2), and the measured per-radius rate reads
the effective viscosity back out. Radial diffusion does eventually smear
the two plateaus into each other once nu becomes r-dependent; the
measurement bands sit >= 5 cells clear of the cut, where that leakage is
exp(-d^2/(4 nu t)) ~ 7e-5.

Modes:
  open -- nu_op_mul = 0.1: the SELECTED (large-psi, outer) region runs at
          0.1 nu and the complement is untouched at nu.
  off  -- nu_op_mul = 1: both regions run at nu (the mechanism is
          inactive, the pre-existing uniform-viscosity behaviour).

Usage:
    analysis_mhd_viscosity_open_region.py <mode> <initial> <final plotfile>
"""

import sys

import numpy as np
import yt

# Deck constants (inputs_test_rz_theta_implicit_mhd_viscosity_open_region)
m_p = 1.67262192595e-27
q_e = 1.602176634e-19
n0 = 1.0e20
rho0 = n0 * m_p
Ti = 100.0
P0 = n0 * Ti * q_e
gamma = 5.0 / 3.0
nr = 32
nz = 8
Lr = 0.5
Lz = 0.0625
dr = Lr / nr
dz = Lz / nz
sound_speed = np.sqrt(gamma * P0 / rho0)
shear = 1.0e-3 * sound_speed
nu_hi = 2000.0
nu_op_mul = 0.1
r_cut = 0.25
n_steps = 16
dt = 5.0e-9
t_end = n_steps * dt

k_mode = 2.0 * np.pi / Lz
k_eff2 = 4.0 / dz**2 * np.sin(0.5 * k_mode * dz) ** 2

# Measurement bands, both >= 5 cells clear of the cut at cell 16. The
# inner band also skips the first three cells (the axis ring) and the
# outer band runs to the wall, where the even ghost forces no flux.
inner_band = slice(3, 11)
outer_band = slice(21, 32)


def get_amplitudes(plotfile):
    """Per-radius amplitude of the seeded axial mode of v_theta."""
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    momentum = data["boxlib", "implicit_mhd_momentum_t"].value.reshape(nr, nz)
    density = data["boxlib", "implicit_mhd_mass_density"].value.reshape(nr, nz)
    velocity = momentum / density
    z_centers = (np.arange(nz) + 0.5) * dz
    basis = np.sin(k_mode * z_centers)
    return 2.0 * (velocity @ basis) / nz


mode = sys.argv[1]
assert mode in ("open", "off"), f"unknown mode {mode}"
initial = get_amplitudes(sys.argv[2])
final = get_amplitudes(sys.argv[3])

assert np.allclose(initial, shear, rtol=1.0e-10, atol=0.0), (
    "the seeded rotation is not the uniform-in-r eigenmode the gate needs"
)
assert np.all(final > 0.0), "the shear mode changed sign (not a diffusive decay)"

measured_nu = -np.log(final / initial) / (k_eff2 * t_end)

nu_inner_expected = nu_hi
nu_outer_expected = nu_hi * (nu_op_mul if mode == "open" else 1.0)

nu_inner = measured_nu[inner_band]
nu_outer = measured_nu[outer_band]

# The complement (small psi) must be untouched in BOTH modes: this is the
# clause that pins the multiplier to its region.
assert np.allclose(nu_inner, nu_inner_expected, rtol=0.02, atol=0.0), (
    f"the psi <= nu_op_bnd region is not at the unscaled viscosity: "
    f"measured {nu_inner.min():.2f}..{nu_inner.max():.2f} m^2/s, "
    f"expected {nu_inner_expected}"
)
assert np.allclose(nu_outer, nu_outer_expected, rtol=0.03, atol=0.0), (
    f"the psi > nu_op_bnd region is not at the scaled viscosity: "
    f"measured {nu_outer.min():.2f}..{nu_outer.max():.2f} m^2/s, "
    f"expected {nu_outer_expected}"
)
print(
    f"[{mode}] psi <= bnd (r = {dr * 3.5:.3f}..{dr * 10.5:.3f} m): "
    f"nu_eff = {nu_inner.mean():.2f} m^2/s (analytic {nu_inner_expected})"
)
print(
    f"[{mode}] psi >  bnd (r = {dr * 21.5:.3f}..{dr * 31.5:.3f} m): "
    f"nu_eff = {nu_outer.mean():.2f} m^2/s (analytic {nu_outer_expected})"
)

if mode == "open":
    # The measured ratio IS nu_op_mul, independently of the absolute
    # calibration, and the cut lands on the analytic radius.
    ratio = nu_outer.mean() / nu_inner.mean()
    assert np.isclose(ratio, nu_op_mul, rtol=0.03, atol=0.0), (
        f"the region ratio is {ratio:.4f}, not nu_op_mul = {nu_op_mul}"
    )
    # Every cell strictly inside the small-psi region must be closer to
    # the unscaled value than to the scaled one, and vice versa: the
    # boundary sits where psi crosses nu_op_bnd, not somewhere else.
    midpoint = 0.5 * (nu_inner_expected + nu_outer_expected)
    cut_cell = int(round(r_cut / dr))
    assert np.all(measured_nu[:cut_cell][3:] > midpoint), (
        "a small-psi cell was scaled: the gate radius is wrong"
    )
    assert np.all(measured_nu[cut_cell + 5 :] < midpoint), (
        "a large-psi cell was NOT scaled: the gate radius is wrong"
    )
    print(
        f"[open] region ratio {ratio:.4f} (nu_op_mul {nu_op_mul}), gate at "
        f"cell {cut_cell} (r = {r_cut} m)"
    )
else:
    spread = measured_nu[3:].max() / measured_nu[3:].min() - 1.0
    assert spread < 0.02, (
        f"the inactive multiplier still varied the viscosity by {spread:.3e}"
    )
    print(f"[off] uniform viscosity across the column to {spread:.2e}")

print("PASS")
