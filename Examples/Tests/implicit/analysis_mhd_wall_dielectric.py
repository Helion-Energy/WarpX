#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""Dielectric shaped-wall transparency test (RZ MHD).

A seeded on-axis plasma-response flux column (the z-uniform
Bz = B0*exp(-(r/w0)^2), radially well inside the stepped wall polyline)
spreads through the uniform resistivity eta = mu0*D toward the r_max
PEC boundary, REDISTRIBUTING poloidal flux across every ring. The
per-z-row disk-flux CHANGE dpsi through probe rings DEEP INSIDE the
masked band (the coil-radius stand-ins, r ~ 0.34 and 0.41 m against
wall radii 0.30/0.22 m) is the flux swing a wall-mounted coil contour
integrates as back-EMF -- the quantity the one-way-glass pec_response
contract pins to exactly zero (the measured |lambda| ~ 1e-16 Wb
back-EMF kill) and the dielectric standoff must keep finite and
physical.

mode = "none" (the transparency reference twin): dpsi at the probe
rings is FINITE (the loop's flux measurably reaches the probe ring
through the resistive dust), and the wall-band region's fluid DOES
evolve (interior conduction reaches it) -- the teeth for the
dielectric run's bit-static band check.

mode = "dielectric" (the headline; takes the none twin's end plotfile
as a 4th argument): the probe-ring dpsi is finite AND matches the none
twin to solver precision -- the band imposes no field constraint at
all: no response pinning, no PC row drops, no wall-seam guard (the
hyper-resistive Ohm term, one of the three seam-guarded terms, is on
at a grid-scale weight ~25% of eta, so a wrongly-engaged seam guard
would zero it at seam-adjacent rows and break the match). The full B
field matches the twin pointwise too. Meanwhile the masked-band FLUID
state is bit-static (the pec_response rigid-freeze contract, unchanged
under dielectric).

mode = "pec_response" (the adversarial contrast, same deck): the same
probe-ring dpsi is pinned at zero/roundoff (every masked E_theta ring
freezes its disk flux by discrete Faraday telescoping), and the masked
band is bit-static as well.

Usage:
  analysis_mhd_wall_dielectric.py <diag0> <diag_end> <mode> [<none_end>]
"""

import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(0)

# Must match the inputs file
B0 = 1.0e-5  # [T] seeded flux-column peak Bz
W0 = 0.08  # [m] seeded flux-column radial width
R_WALL_LO = 0.30  # [m] wall radius, z < 0
R_WALL_HI = 0.22  # [m] wall radius, z > 0
R_PROBES = (0.34, 0.40)  # [m] probe-ring radii (deep inside the band)

FLUID_FIELDS = (
    "implicit_mhd_mass_density",
    "implicit_mhd_electron_energy",
    "implicit_mhd_momentum_r",
    "implicit_mhd_momentum_t",
    "implicit_mhd_momentum_z",
)


def load(fn):
    ds = yt.load(fn)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    fields = {
        name: grid["boxlib", name].v.squeeze() for name in ("Br", "Bz") + FLUID_FIELDS
    }
    return ds, fields


ds0, state0 = load(sys.argv[1])
ds1, state1 = load(sys.argv[2])
mode = sys.argv[3]
assert mode in ("none", "dielectric", "pec_response"), f"unknown mode {mode}"

nr, nz = int(ds0.domain_dimensions[0]), int(ds0.domain_dimensions[1])
rmin = float(ds0.domain_left_edge[0])
dr = (float(ds0.domain_right_edge[0]) - rmin) / nr
zmin = float(ds0.domain_left_edge[1])
dz = (float(ds0.domain_right_edge[1]) - zmin) / nz
r_cc = rmin + dr * (np.arange(nr) + 0.5)
z_cc = zmin + dz * (np.arange(nz) + 0.5)

for name, field in state1.items():
    assert np.isfinite(field).all(), f"{name} has non-finite values"

# The mask, replicated exactly as ImplicitMHDWallMask builds it (cell
# center on/outside the polyline radius with the 1e-3*dr sliver).
wall_radius = np.where(z_cc < 0.0, R_WALL_LO, R_WALL_HI)
masked = (r_cc[:, None] + 1.0e-3 * dr) >= wall_radius[None, :]
assert masked.any() and not masked.all(), "degenerate mask layout"

# Flux scale: the seeded column's total disk flux per radian,
# integral of B0 exp(-(r/w0)^2) r dr = B0 w0^2 / 2 (matching the psi
# sums below).
psi_scale = 0.5 * B0 * W0**2

b_scale = B0  # seeded peak Bz


def dpsi_rows(i_ring):
    """Per-z-row change of the disk flux through corner ring i_ring
    (cell-centered plotfile Bz = z-average of the two staggered planes,
    each carrying the exact conserved staggered disk sum)."""
    rows0 = (state0["Bz"][:i_ring, :] * r_cc[:i_ring, None] * dr).sum(axis=0)
    rows1 = (state1["Bz"][:i_ring, :] * r_cc[:i_ring, None] * dr).sum(axis=0)
    return rows1 - rows0


probe_rings = [int(np.ceil((rp - rmin) / dr - 1.0e-12)) for rp in R_PROBES]
dpsi_max = {}
for i_ring in probe_rings:
    r_ring = rmin + i_ring * dr
    assert r_ring + 1.0e-3 * dr >= R_WALL_LO, "probe ring not inside the band"
    dpsi = float(np.abs(dpsi_rows(i_ring)).max())
    dpsi_max[i_ring] = dpsi
    print(
        f"probe ring r = {r_ring:.5f}: max-row |dpsi| = {dpsi:.6e} "
        f"({dpsi / psi_scale:.3e} of the loop flux scale {psi_scale:.3e})"
    )


def band_drift(a, b):
    """Max relative drift of every fluid moment over the masked band."""
    worst = 0.0
    for f in FLUID_FIELDS:
        scale = np.max(np.abs(a[f]))
        scale = scale if scale > 0.0 else 1.0
        drift = np.max(np.abs(b[f] - a[f]), initial=0.0, where=masked) / scale
        print(f"masked-band drift {f}: {drift:.3e}")
        worst = max(worst, drift)
    return worst


# The rigid vacuum clamp image (the shaped-wall exterior clamp, applied
# at first-step sanitize time in every wall_live-freezing mode): deck
# floors mass 5e-4 rho0 and electron pressure 1e-4 n0 q_e Tmin, no
# temperature floors, each value a fixed 2e-6 slack above its bound.
PROTON_MASS = 1.67262192595e-27  # WarpX parser m_p (ablastr::constant)
QE = 1.602176634e-19
N0 = 1.0e18
TMIN_EV = 2.0
GAMMA = 5.0 / 3.0
SLACK = 2.0e-6
CLAMP_IMAGE = {
    "implicit_mhd_mass_density": 5.0e-4 * N0 * PROTON_MASS * (1.0 + SLACK),
    "implicit_mhd_electron_energy": 1.0e-4 * N0 * QE * TMIN_EV
    / (GAMMA - 1.0) * (1.0 + SLACK),
    "implicit_mhd_momentum_r": 0.0,
    "implicit_mhd_momentum_t": 0.0,
    "implicit_mhd_momentum_z": 0.0,
}


def band_image_mismatch(state):
    """Max relative mismatch of the masked band against the clamp image."""
    worst = 0.0
    for f, value in CLAMP_IMAGE.items():
        scale = max(abs(value), float(np.max(np.abs(state[f]))), 1.0e-300)
        err = np.max(np.abs(state[f] - value), initial=0.0, where=masked) / scale
        print(f"clamp image mismatch {f}: {err:.3e} of scale")
        worst = max(worst, err)
    return worst


# Calibrated thresholds (measured on the 2-rank OMP CI layout):
# - the transparent probe-ring linkage lands at 0.496 (r = 0.34) and
#   0.316 (r = 0.41) of the loop flux scale;
# - the dielectric-vs-none twin match lands at machine roundoff
#   (psi 4e-16, pointwise B 1e-16 of scale: the dielectric field path
#   is operation-identical to none -- nothing reads the mask);
# - the pec_response pinning lands at roundoff too (2e-16 of scale,
#   |dpsi| = 6.6e-24 Wb/rad: the masked E_theta rings freeze the disk
#   flux exactly);
# - the dielectric masked-band fluid drift is exactly 0.0 while the
#   none twin's band electron energy drifts by 1.0e-1 (the teeth).
FINITE_MIN = 0.01  # of psi_scale: the linkage is FINITE
TOL_MATCH_PSI = 1.0e-6  # of psi_scale: dielectric == none (psi)
TOL_MATCH_B = 1.0e-6  # of b_scale: dielectric == none (pointwise B)
TOL_PINNED = 1.0e-8  # of psi_scale: pec_response pins dpsi
TOL_STATIC = 1.0e-12  # masked-band fluid moments bit-static
MIN_BAND_EVOLVES = 1.0e-4  # the none twin's band must NOT be static

if mode == "none":
    # --- transparency reference: the linkage is finite ----------------------
    for i_ring, dpsi in dpsi_max.items():
        assert dpsi > FINITE_MIN * psi_scale, (
            f"reference run: no flux reached probe ring {i_ring} "
            f"({dpsi / psi_scale:.3e})"
        )
    # ...and the band region's fluid evolves (nothing freezes it), the
    # teeth of the dielectric run's bit-static check.
    drift = band_drift(state0, state1)
    assert drift > MIN_BAND_EVOLVES, (
        f"reference band unexpectedly static ({drift:.3e}): the "
        "bit-static check would be vacuous"
    )

elif mode == "dielectric":
    assert len(sys.argv) > 4, "dielectric mode needs the none twin's plotfile"
    _, state_none = load(sys.argv[4])

    # --- 1. FINITE response linkage at the in-band probe rings --------------
    for i_ring, dpsi in dpsi_max.items():
        assert dpsi > FINITE_MIN * psi_scale, (
            f"dielectric band blocked the response flux at ring {i_ring} "
            f"({dpsi / psi_scale:.3e} of scale)"
        )

    # --- 2. ... matching the wall_model=none twin to solver precision -------
    for i_ring in probe_rings:
        rows_d = (state1["Bz"][:i_ring, :] * r_cc[:i_ring, None] * dr).sum(axis=0)
        rows_n = (state_none["Bz"][:i_ring, :] * r_cc[:i_ring, None] * dr).sum(axis=0)
        err = float(np.abs(rows_d - rows_n).max()) / psi_scale
        print(f"ring {i_ring} |psi_dielectric - psi_none|max/scale = {err:.3e}")
        assert err < TOL_MATCH_PSI, f"dielectric flux deviates from none: {err:.3e}"
    for name in ("Br", "Bz"):
        err = float(np.abs(state1[name] - state_none[name]).max()) / b_scale
        print(f"pointwise |{name}_dielectric - {name}_none|max/scale = {err:.3e}")
        assert err < TOL_MATCH_B, f"dielectric {name} deviates from none: {err:.3e}"

    # --- 3. masked-band FLUID state on the rigid vacuum clamp image ---------
    # (the pec_response freeze contract plus the first-step exterior
    # clamp: the uniform rho0 IC loaded the band far above the floor
    # and must have been scraped)
    mismatch = band_image_mismatch(state1)
    assert mismatch < TOL_STATIC, (
        f"dielectric masked band off the clamp image ({mismatch:.3e})"
    )

    # ... while the none twin's band DOES evolve un-clamped (the check
    # has teeth; the twins share this run's initial state bit-exactly).
    print("none-twin band evolution (teeth of the clamp-image check):")
    drift_none = band_drift(state0, state_none)
    assert drift_none > MIN_BAND_EVOLVES, (
        f"none twin's band unexpectedly static ({drift_none:.3e})"
    )

else:  # pec_response
    # --- adversarial contrast: the linkage is pinned at zero/roundoff -------
    for i_ring, dpsi in dpsi_max.items():
        assert dpsi < TOL_PINNED * psi_scale, (
            f"pec_response probe-ring flux NOT pinned at ring {i_ring} "
            f"({dpsi / psi_scale:.3e} of scale)"
        )
    # ... and the fluid contract is the same rigid freeze onto the
    # exterior clamp image.
    mismatch = band_image_mismatch(state1)
    assert mismatch < TOL_STATIC, (
        f"pec_response masked band off the clamp image ({mismatch:.3e})"
    )

print(f"dielectric shaped-wall transparency test ({mode}) PASSED")
