#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""The mirror-plane density hole under the reduced penalty, unguarded and guarded.

Deck: inputs_test_rz_theta_implicit_mhd_density_hole_mirror -- a uniform
column of gas at the inflow temperature T0 flows toward the z_lo mirror
plane (u_z odd there: the flow stagnates) through a near-floor density
hole one cell row in from the plane, on the production ladder's reduced
penalty (central + van Albada + c = 0.25).

WHAT THIS RIG DOES AND DOES NOT SHOW. The production run whose defect the
guard addresses (a c = 0.25 formation arm) grows a 177 keV cell at the
mirror plane in gas held at the pedestal density; the rig's hole REFILLS
instead -- the dispersive central average of the mass flux is large
enough at c = 0.25 -- and the column's peak temperature (1.6 x T0 at
Mach 0.5) is the stagnation and compression heating of the plane row, not
a runaway (the same rig at c = 1 gives the same 1.6 x). The production
mechanism needs the halo pedestal machinery (the mass gate shut at the
pedestal while momentum and energy keep flowing) that a 100-step column
does not carry; the guard's effect on the runaway is therefore measured
on the production arms, and this test gates what a column CAN gate:

  reference -- the unguarded rig: completes within the Newton budget, its
               peak ion temperature stays within 2 x the inflow temperature
               (a stagnation flow at Mach 0.5, no runaway), the peak is
               reported;
  guarded   -- the same rig with the density switch (n_g = 3e19 m^-3:
               the 1e18 hole fully guarded, the 1e20 ambient exactly not)
               and the mirror-plane switch (4 rows + the 2-row taper): the
               peak stays within 2 x the inflow temperature and at or
               below the unguarded peak (the guard never heats), the
               guard ledger books one row per step with the switch
               geometry -- at least the 208 faces of the mirror-plane
               switch (7 z-faces x 16 columns + 6 rows x 16 r-faces) --
               and positive mass and ion-energy transport.

Usage: analysis_mhd_density_hole_mirror.py <initial_plotfile> <final_plotfile> <mode> [<guard_ledger> <reference_final_plotfile>]
"""

import sys

import numpy as np
import yt
from scipy import constants

yt.set_log_level(50)

FIELDS = [
    "implicit_mhd_mass_density",
    "implicit_mhd_ion_energy",
    "implicit_mhd_electron_energy",
    "implicit_mhd_momentum_r",
    "implicit_mhd_momentum_t",
    "implicit_mhd_momentum_z",
]
GAMMA = 5.0 / 3.0
# the mirror-plane switch of the deck's registration: 4 rows at S = 1 and
# the 2-row taper make rows 0..5 guarded; their z-faces j = 0..6 in each
# of the 16 columns and their r-faces i = 1..16 (the axis face is never
# written) in each of the 6 rows
SYMMETRY_FACES = 7 * 16 + 6 * 16


def load(path):
    ds = yt.load(path)
    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    fields = {f: np.asarray(grid["boxlib", f]) for f in FIELDS}
    return fields, ds


def temperatures_ev(fields):
    """Ion and electron temperatures [eV] per cell (m_i = m_p, singly charged)."""
    rho = fields["implicit_mhd_mass_density"]
    kinetic = 0.5 * (
        fields["implicit_mhd_momentum_r"] ** 2
        + fields["implicit_mhd_momentum_t"] ** 2
        + fields["implicit_mhd_momentum_z"] ** 2
    ) / rho
    u_i = fields["implicit_mhd_ion_energy"] - kinetic
    n = rho / constants.m_p
    t_i = (GAMMA - 1.0) * u_i / n / constants.e
    t_e = (GAMMA - 1.0) * fields["implicit_mhd_electron_energy"] / n / constants.e
    return t_i, t_e


initial, ds0 = load(sys.argv[1])
final, ds1 = load(sys.argv[2])
mode = sys.argv[3]
assert mode in ("reference", "guarded")

t_i0, t_e0 = temperatures_ev(initial)
t_i1, t_e1 = temperatures_ev(final)
# The inflow temperature: the ambient gas of the initial state (its
# maximum; the hole is at or below it by construction).
t_inflow = float(np.max(t_i0))
rho1 = final["implicit_mhd_mass_density"]
nr, nz = ds1.domain_dimensions[0], ds1.domain_dimensions[1]
shape = (nr, nz)
t_i1 = t_i1.reshape(shape)
rho1 = rho1.reshape(shape)
hot = np.unravel_index(np.argmax(t_i1), shape)
print(f"inflow T_i = {t_inflow:.3f} eV; final T_i max = {t_i1.max():.3f} eV "
      f"({t_i1.max() / t_inflow:.2f} x) at (i, j) = {hot}, n there = "
      f"{rho1[hot] / constants.m_p:.3e} m^-3; final T_e max = {t_e1.max():.3f} eV")
print("final T_i / T_inflow of the first four rows above the plane "
      "(radial max): "
      + " ".join(f"{t_i1[:, j].max() / t_inflow:.2f}" for j in range(4)))
print("final n / n0 of the first four rows (radial min): "
      + " ".join(f"{rho1[:, j].min() / initial['implicit_mhd_mass_density'].max():.3e}"
                 for j in range(4)))

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
newton_per_step = newton_history[:, 2]
final_step = int(newton_history[-1, 0])
print(f"Newton per step: mean {newton_per_step.mean():.2f}, max "
      f"{newton_per_step.max():.0f}; cumulative GMRES {newton_history[-1, 7]:.0f}")
# measured 5 (reference) and 8 (guarded) against the deck's 20: the bound
# catches a stall, not a fluctuation
assert newton_per_step.max() <= 16, (
    f"Newton budget exceeded: max {newton_per_step.max():.0f} per step"
)
assert t_i1.max() <= 2.0 * t_inflow, (
    f"the column's peak temperature {t_i1.max():.3f} eV exceeds twice the inflow "
    f"temperature {t_inflow:.3f} eV"
)

if mode == "reference":
    print("PASS: the unguarded column stays within twice the inflow temperature")
else:
    rows = [
        line.split()
        for line in open(sys.argv[4])
        if line.strip() and not line.startswith("#")
    ]
    table = np.array(rows, dtype=float)
    faces, guard_mass, guard_ei, cum_ei = table[:, 1], table[:, 2], table[:, 4], table[:, 12]
    print(f"guard ledger: {len(rows)} rows; guarded faces {faces.min():.0f}.."
          f"{faces.max():.0f} (the mirror-plane switch alone: {SYMMETRY_FACES}); guard "
          f"mass per step {guard_mass.min():.3e}..{guard_mass.max():.3e} kg; guard E_i "
          f"per step {guard_ei.min():.3e}..{guard_ei.max():.3e} J; cumulative guard E_i "
          f"{cum_ei[-1]:.3e} J")
    assert len(rows) == final_step, "the ledger missed a step"
    assert np.all(table[:, 0] == np.arange(1, final_step + 1)), "the ledger's steps are not consecutive"
    assert np.all(faces >= SYMMETRY_FACES), "fewer guarded faces than the mirror-plane switch alone"
    assert np.all(guard_mass > 0.0) and np.all(guard_ei > 0.0), (
        "the guard booked no transport"
    )
    np.testing.assert_allclose(cum_ei, np.cumsum(guard_ei), rtol=1.0e-12, atol=0.0)
    reference, _ = load(sys.argv[5])
    t_i_ref, _ = temperatures_ev(reference)
    print(f"unguarded twin: final T_i max = {t_i_ref.max():.3f} eV "
          f"({t_i_ref.max() / t_inflow:.2f} x)")
    assert t_i1.max() <= 1.001 * t_i_ref.max(), (
        f"the guard raised the peak temperature: {t_i1.max():.3f} eV against the "
        f"unguarded {t_i_ref.max():.3f} eV"
    )
    print("PASS: the guard bounds the column's temperature and books its transport")
