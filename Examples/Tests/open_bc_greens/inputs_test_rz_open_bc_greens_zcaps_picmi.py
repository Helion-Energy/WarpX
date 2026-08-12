#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""Magnetostatic cap gate of the RZ Green's-function open boundary with
open z faces (see README.rst).

A Gaussian ring of azimuthal current, held off-center in z, is relaxed in a
resistive vacuum with ALL free faces open (r_hi, z_lo, z_hi; non-periodic
z, so the isolated kernel branch is exercised). The steady state satisfies
curl B = mu0 J_ext with free-space boundary values. This driver then reads
B directly from the grid INCLUDING guard cells and asserts:

1. the cap-band ghost values of Br and Bz match the analytic free-space
   (isolated, image-free) field of the discrete ring source -- full
   resolution loop-psi superposition, Yee-differenced exactly as the code
   differences its psi table -- to the tolerance of the r-face validation
   (analysis_ring.py), on the full filled bands INCLUDING the corner
   ghosts shared with the r_hi fill;
2. the ghost B_theta vanishes identically (the m = 0 poloidal/toroidal
   systems are decoupled and the source is purely azimuthal);
3. the discrete cylindrical div B vanishes to machine precision on every
   cell whose faces are all evolution- or psi-derived: the interior, the
   deep ghost bands, and in particular the corner-ghost cells where the
   r-face and cap fills meet (they difference one shared psi table). The
   first ghost band mixes an evolved boundary-plane face with psi-derived
   faces, so its div measures the fill accuracy rather than round-off and
   is asserted at the field tolerance instead.
"""

import numpy as np
from scipy.special import ellipe, ellipk

from pywarpx import fields, picmi

constants = picmi.constants
mu0 = constants.mu0

# ---------------------------------------------------------------------------
# Simulation: vacuum resistive relax of a held ring current (the explicit
# hybrid solver with eta = mu0 makes the field advance linear resistive
# vacuum diffusion; same recipe as the input-file tests in this directory).
# ---------------------------------------------------------------------------
NR, NZ = 32, 32
RMIN, RMAX = 0.0, 1.0
ZMIN, ZMAX = -0.5, 0.5
R0, Z0, A0, J0 = 0.4, 0.1, 0.1, 1.0e3
MAX_STEPS = 170

grid = picmi.CylindricalGrid(
    number_of_cells=[NR, NZ],
    n_azimuthal_modes=1,
    lower_bound=[RMIN, ZMIN],
    upper_bound=[RMAX, ZMAX],
    lower_boundary_conditions=["none", "open"],
    upper_boundary_conditions=["open", "open"],
    lower_boundary_conditions_particles=["none", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
)

solver = picmi.HybridPICSolver(
    grid=grid,
    Te=1.0,
    n0=1.0e12,
    gamma=1.0,
    n_floor=1.0e12,
    include_hall_term=False,
    include_electron_pressure_term=False,
    # eta = mu0 * D with D = 1 m^2/s
    plasma_resistivity=mu0,
    substeps=20,
    Jy_external_function=f"{J0}*exp(-((x-{R0})^2 + (z-{Z0})^2)/{A0}^2)",
)

simulation = picmi.Simulation(
    solver=solver,
    time_step_size=6.0e-3,
    max_steps=MAX_STEPS,
    verbose=False,
)

simulation.step(MAX_STEPS)

# ---------------------------------------------------------------------------
# Read B with guard cells (global gather; identical on every rank)
# ---------------------------------------------------------------------------
Br_w = fields.BxFPWrapper()
Bt_w = fields.ByFPWrapper()
Bz_w = fields.BzFPWrapper()
ngr = int(Br_w.mf.n_grow_vect[0])
ngz = int(Br_w.mf.n_grow_vect[1])
# fortran-ordered (i, j[, comp]); m = 0 only
Br = np.squeeze(Br_w[()])  # nodal r, cc z: (NR+1+2*ngr, NZ+2*ngz)
Bt = np.squeeze(Bt_w[()])  # cc r, cc z
Bz = np.squeeze(Bz_w[()])  # cc r, nodal z

dr = (RMAX - RMIN) / NR
dz = (ZMAX - ZMIN) / NZ


def ir(i):  # array index of nodal/cc r index i
    return i + ngr


def jz(j):  # array index of nodal/cc z index j
    return j + ngz


# ---------------------------------------------------------------------------
# Analytic free-space psi of the discrete source (isolated: no image sum),
# on all nodal points of the filled bands, i in [0, NR+ngr], j in
# [-ngz, NZ+ngz].
# ---------------------------------------------------------------------------
r_src = dr * np.arange(NR + 1)
z_src = ZMIN + dz * np.arange(NZ + 1)
RS, ZS = np.meshgrid(r_src, z_src, indexing="ij")
w_src = J0 * np.exp(-(((RS - R0) ** 2) + (ZS - Z0) ** 2) / A0**2) * dr * dz

i_eval = np.arange(0, NR + ngr + 1)
j_eval = np.arange(-ngz, NZ + ngz + 1)
RB = (dr * i_eval)[:, None, None, None]
ZB = (ZMIN + dz * j_eval)[None, :, None, None]
RSb = RS[None, None, :, :]
ZSb = ZS[None, None, :, :]

d2 = (ZB - ZSb) ** 2 + (RB + RSb) ** 2
with np.errstate(divide="ignore", invalid="ignore"):
    m = np.where(d2 > 0.0, 4.0 * RB * RSb / np.where(d2 > 0.0, d2, 1.0), 0.0)
    # exact eval/source coincidences (m = 1, on the shared boundary-plane
    # nodes) carry ~e^-36 of the ring current; drop the singular self-terms
    m = np.where(m >= 1.0, 0.0, m)
    G0 = mu0 / (4.0 * np.pi) * np.sqrt(d2) * ((2.0 - m) * ellipk(m) - 2.0 * ellipe(m))
G0 = np.where((RB > 0.0) & (RSb > 0.0) & (d2 > 0.0) & (m > 0.0), G0, 0.0)
psiA = np.einsum("ijkl,kl->ij", G0, w_src)  # [i_eval, j_eval]


def psiA_at(i, j):
    return psiA[np.asarray(i), np.asarray(j) + ngz]


# analytic Yee-differenced ghosts
r_nodes = dr * np.arange(NR + ngr + 1)


def analytic_Br(i, j_cc):
    II, JJ = np.meshgrid(i, j_cc, indexing="ij")
    out = -(psiA_at(II, JJ + 1) - psiA_at(II, JJ)) / dz
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(II > 0, out / np.where(II > 0, r_nodes[II], 1.0), 0.0)
    return out


def analytic_Bz(i_cc, j):
    II, JJ = np.meshgrid(i_cc, j, indexing="ij")
    rc = (II + 0.5) * dr
    return (psiA_at(II + 1, JJ) - psiA_at(II, JJ)) / (rc * dr)


# ---------------------------------------------------------------------------
# 1. Cap-band ghosts vs analytic (both caps, full radial extent + corners)
# ---------------------------------------------------------------------------
TOL_CAP = 0.02  # the r-face validation tolerance (analysis_ring.py)

for cap, j_br, j_bz in (
    ("z_lo", np.arange(-ngz, 0), np.arange(-ngz, 0)),
    ("z_hi", np.arange(NZ, NZ + ngz), np.arange(NZ + 1, NZ + ngz + 1)),
):
    i_br = np.arange(0, NR + ngr + 1)
    i_bz = np.arange(0, NR + ngr)
    code_br = Br[np.ix_(ir(i_br), jz(j_br))]
    code_bz = Bz[np.ix_(ir(i_bz), jz(j_bz))]
    ana_br = analytic_Br(i_br, j_br)
    ana_bz = analytic_Bz(i_bz, j_bz)
    scale = max(np.abs(ana_br).max(), np.abs(ana_bz).max())
    err_br = np.abs(code_br - ana_br).max() / scale
    err_bz = np.abs(code_bz - ana_bz).max() / scale
    print(f"{cap} cap ghosts: rel err Br = {err_br:.3e}, Bz = {err_bz:.3e}")
    assert err_br < TOL_CAP, f"{cap} ghost Br error {err_br:.3e} > {TOL_CAP}"
    assert err_bz < TOL_CAP, f"{cap} ghost Bz error {err_bz:.3e} > {TOL_CAP}"

# ---------------------------------------------------------------------------
# 2. Ghost B_theta must vanish identically (decoupled toroidal system)
# ---------------------------------------------------------------------------
bt_max = np.abs(Bt).max()
print(f"max |B_theta| anywhere: {bt_max:.3e}")
assert bt_max == 0.0, f"toroidal field contaminated: {bt_max:.3e}"

# ---------------------------------------------------------------------------
# 3. Discrete cylindrical div B
# ---------------------------------------------------------------------------
i_cc = np.arange(0, NR + ngr)  # cells with all faces present (r >= 0)
j_cc = np.arange(-ngz, NZ + ngz)
II, JJ = np.meshgrid(i_cc, j_cc, indexing="ij")
r_lo_f = r_nodes[II]
r_hi_f = r_nodes[II + 1]
rc = (II + 0.5) * dr
divB = (
    r_hi_f * Br[np.ix_(ir(i_cc + 1), jz(j_cc))]
    - r_lo_f * Br[np.ix_(ir(i_cc), jz(j_cc))]
) / (rc * dr) + (
    Bz[np.ix_(ir(i_cc), jz(j_cc + 1))] - Bz[np.ix_(ir(i_cc), jz(j_cc))]
) / dz
b_scale = max(np.abs(Br).max(), np.abs(Bz).max())
div_rel = np.abs(divB) * dr / b_scale

# seam bands: the first ghost band, where one face is the evolved
# boundary-plane value and the others are psi-derived
seam = np.zeros_like(div_rel, dtype=bool)
seam[II == NR] = True  # first radial ghost ring
seam[JJ == -1] = True  # first lo-cap ghost row
seam[JJ == NZ] = True  # first hi-cap ghost row

div_machine = div_rel[~seam].max()
div_seam = div_rel[seam].max()
corner = (II >= NR) & ((JJ <= -1) | (JJ >= NZ))
div_corner = div_rel[corner].max()
print(f"max |div B| dr/|B|: interior+deep ghosts = {div_machine:.3e}, "
      f"corner ghosts = {div_corner:.3e}, seam bands = {div_seam:.3e}")
assert div_machine < 1.0e-11, f"div B not at machine precision: {div_machine:.3e}"
assert div_corner < 1.0e-11, f"corner-ghost div B not machine zero: {div_corner:.3e}"
assert div_seam < TOL_CAP, f"seam-band div B beyond fill tolerance: {div_seam:.3e}"

print("open z-cap magnetostatic gate PASSED")
