#!/usr/bin/env python3
#
# --- Production-path battery of the hybrid solver's per-step diffusive
# --- div(B) Marder clean (hybrid_pic_model.divb_clean_alpha, implemented in
# --- MarderCleanDivergence and driven once per step from the hybrid field
# --- advance), on a planar conducting wall, for BOTH grid types:
# ---
# ---   * a synthetic B with a localized near-wall divergence bump (plus a
# ---     control bump in the deep fluid and a divergence-free tangential
# ---     background) is loaded through the field wrappers after
# ---     initialization,
# ---   * ONE production step is taken -- the deck is configured so the
# ---     Faraday push is an exact no-op (no plasma, holmstrom_vacuum_region
# ---     with zero resistivity gives E = 0 identically), so the only B
# ---     writer inside the step is the clean itself (correction + its
# ---     internal EB mirror re-fill of the wall band),
# ---   * asserted through the wrappers: the near-wall-band divergence (the
# ---     clean's own discrete operator) DECREASED, the deep-fluid B --
# ---     including the deep divergence bump, which the banded clean must
# ---     not reach -- is unchanged to round-off, and on the Yee grid the
# ---     bulk CELL-CENTERED divergence (the quantity the staggered Faraday
# ---     update conserves) received no injection above round-off.
# ---
# --- The Yee guard is the meaningful one: on a staggered grid the clean's
# --- divergence operator (ComputeDivE applied to the face-centered B) is a
# --- NODAL PROXY, not the conserved cell-centered divergence, so its
# --- grad(div) correction writes cell-centered divergence wherever it acts.
# --- The default BANDED mode (divb_clean_band_cells = 4) keeps the
# --- correction out of the bulk, which is what makes the clean Yee-safe; in
# --- the unbounded mode (band_cells <= 0) the correction would act on the
# --- deep-fluid proxy divergence seeded here and the bulk cell-centered
# --- guard below would fire. Do not weaken that assertion.

import argparse
import sys

import numpy as np

from pywarpx import fields, picmi

# grid: cubic cells, wall normal x, y and z periodic
N_X = 32
N_YZ = 8
LO_X, HI_X = -1.0, 1.0
H = (HI_X - LO_X) / N_X
LO_YZ, HI_YZ = -N_YZ * H / 2, N_YZ * H / 2
L_Z = HI_YZ - LO_YZ

# planar wall: conductor x > X_WALL; the 0.3 h node offset matches the other
# EB batteries (a cut cell, no degenerate node-on-surface geometry)
X_WALL = LO_X + (24 + 0.3) * H

B0 = 0.01  # field scale (T)

# Clean configuration under test: alpha and iters set explicitly, the band
# knobs LEFT AT THEIR DEFAULTS (band_cells=4, inner_div=1, inner_corr=2).
ALPHA = 0.1
ITERS = 5

# synthetic divergence bumps (Gaussians in the wall distance, in cells):
# one in the near-wall clean band, one deep in the fluid as a control
S_BUMP_WALL = 2.5
S_BUMP_BULK = 11.5
SIGMA_CELLS = 1.0
BUMP_AMP = 0.7 * B0

# Band-divergence measurement window (node wall-distance in cells). The
# level set is clamped at the roof (nGrow+1)*h = 3h, so the clean's outer
# cutoff is min(band_cells, 3 - 0.5)h = 2.5h and the kept-divergence band is
# phi in [1h, 2.5h] (nodes at 1.3h and 2.3h for this wall). Measuring inside
# that active band keeps the assertion meaningful for any roof >= 3h (a
# larger roof only widens the actively damped band).
BAND_LO, BAND_HI = 1.05, 2.6
BULK_MIN = 7.0  # bulk = beyond the band + the grad(div) stencil reach
X_BULK_MIN = -0.8  # keep clear of the x domain boundary layers


def setup_simulation(grid_type):
    grid = picmi.Cartesian3DGrid(
        number_of_cells=[N_X, N_YZ, N_YZ],
        lower_bound=[LO_X, LO_YZ, LO_YZ],
        upper_bound=[HI_X, HI_YZ, HI_YZ],
        lower_boundary_conditions=["dirichlet", "periodic", "periodic"],
        upper_boundary_conditions=["dirichlet", "periodic", "periodic"],
        lower_boundary_conditions_particles=["absorbing", "periodic", "periodic"],
        upper_boundary_conditions_particles=["absorbing", "periodic", "periodic"],
        warpx_max_grid_size=2048,
        warpx_max_grid_size_x=N_X // 2,
        warpx_blocking_factor=8,
    )

    sim = picmi.Simulation(
        time_step_size=1.0e-9,
        max_steps=1,
        particle_shape=1,
        verbose=0,
        # managed memory so the field-wrapper writes below are host-accessible
        # on a CUDA build (no-op on CPU builds)
        warpx_amrex_the_arena_is_managed=1,
    )
    sim.grid_type = grid_type
    if grid_type == "collocated":
        # the collocated (nodal) hybrid path forbids Esirkepov deposition
        sim.current_deposition_algo = "direct"

    sim.solver = picmi.HybridPICSolver(
        grid=grid,
        gamma=5.0 / 3.0,
        Te=0.0,
        n0=1.0e18,
        n_floor=1.0e16,
        # No plasma (rho = 0 < rho_floor everywhere) + Holmstrom vacuum
        # handling + zero resistivity give E = 0 IDENTICALLY, so the Faraday
        # push inside the production step is an exact no-op and the B change
        # measured across sim.step(1) is the divergence clean alone.
        plasma_resistivity=0.0,
        holmstrom_vacuum_region=True,
        substeps=4,
        # the production per-step div(B) clean under test (defaults for the
        # band/inner knobs; see the header comment for why banded-by-default
        # is the Yee-safe configuration)
        divb_clean_alpha=ALPHA,
        divb_clean_iters=ITERS,
    )

    sim.embedded_boundary = picmi.EmbeddedBoundary(
        implicit_function="(x-xw)", xw=X_WALL
    )

    sim.initialize_inputs()
    sim.initialize_warpx()
    return sim


def load_synthetic_b(grid_type):
    """Load B through the wrappers:

    * Bx (the wall-NORMAL component): two x-Gaussian bumps whose dBx/dx is
      the seeded divergence -- one centered S_BUMP_WALL cells inside the
      wall (in the clean band, mimicking the curved-wall injection of the
      EB mirror fill) and one at S_BUMP_BULK cells (a deep-fluid
      must-not-touch control). Bx is x-nodal on both grid types.
    * By (wall-TANGENTIAL): a divergence-free background B0*sin(2 pi z/L)
      (dBy/dy = 0 exactly). Tangential + constant along x, so the clean's
      internal even-parity mirror fill reproduces it at the wall band
      instead of perturbing the measured divergence there (a NORMAL
      background would be rescaled by the odd fill at the first unowned
      fluid node, which the collocated centered divergence reads).
    * Bz = 0.
    """
    Bx = fields.BxFPWrapper()
    By = fields.ByFPWrapper()
    Bz = fields.BzFPWrapper()

    bx_shape = np.asarray(Bx[...]).shape
    x = LO_X + np.arange(bx_shape[0]) * H  # Bx is x-nodal on both grid types
    s_cells = (X_WALL - x) / H  # wall distance in cells (fluid s > 0)

    def bump(center_cells):
        p = BUMP_AMP * np.exp(-((s_cells - center_cells) ** 2) / (2.0 * SIGMA_CELLS**2))
        p[s_cells <= 0.0] = 0.0  # fluid-side only
        return p

    profile_x = bump(S_BUMP_WALL) + bump(S_BUMP_BULK)
    Bx[...] = np.broadcast_to(profile_x[:, None, None], bx_shape).copy()

    by_shape = np.asarray(By[...]).shape
    if grid_type == "collocated":
        z = LO_YZ + np.arange(by_shape[2]) * H  # By is z-nodal
    else:
        z = LO_YZ + (np.arange(by_shape[2]) + 0.5) * H  # By is z-centered
    by = B0 * np.sin(2.0 * np.pi * (z - LO_YZ) / L_Z)
    By[...] = np.broadcast_to(by[None, None, :], by_shape).copy()
    Bz[...] = 0.0


def read_b():
    return (
        np.array(np.asarray(fields.BxFPWrapper()[...])),
        np.array(np.asarray(fields.ByFPWrapper()[...])),
        np.array(np.asarray(fields.BzFPWrapper()[...])),
    )


def clean_div(bx, by, bz, grid_type):
    """The clean's own discrete divergence at the interior nodes i=1..N_X-1
    (ComputeDivE applied to B): Yee downward differences of the face fields
    (the nodal PROXY divergence) or the collocated centered differences.
    Returns (div, node_wall_distance_in_cells)."""
    if grid_type == "collocated":
        d = (
            (bx[2:, 1:-1, 1:-1] - bx[:-2, 1:-1, 1:-1])
            + (by[1:-1, 2:, 1:-1] - by[1:-1, :-2, 1:-1])
            + (bz[1:-1, 1:-1, 2:] - bz[1:-1, 1:-1, :-2])
        ) / (2.0 * H)
    else:
        d = (
            (bx[1:N_X, 1:N_YZ, 1:N_YZ] - bx[0 : N_X - 1, 1:N_YZ, 1:N_YZ])
            + (by[1:N_X, 1:N_YZ, 1:N_YZ] - by[1:N_X, 0 : N_YZ - 1, 1:N_YZ])
            + (bz[1:N_X, 1:N_YZ, 1:N_YZ] - bz[1:N_X, 1:N_YZ, 0 : N_YZ - 1])
        ) / H
    x_nodes = LO_X + np.arange(1, N_X) * H
    return d, (X_WALL - x_nodes) / H


def yee_cc_div(bx, by, bz):
    """The divergence the staggered Faraday update conserves: the
    cell-centered divergence of the face-centered B, at cells i=0..N_X-1.
    Returns (div_cc, cell_center_wall_distance_in_cells)."""
    d = (
        (bx[1:, :, :] - bx[:-1, :, :])
        + (by[:, 1:, :] - by[:, :-1, :])
        + (bz[:, :, 1:] - bz[:, :, :-1])
    ) / H
    x_cent = LO_X + (np.arange(N_X) + 0.5) * H
    return d, (X_WALL - x_cent) / H


def band_l2(d, s_cells):
    mask = (s_cells >= BAND_LO) & (s_cells <= BAND_HI)
    return float(np.sqrt(np.mean(d[mask, :, :] ** 2)))


def bulk_mask_x(shape0, x_offset_cells):
    """Deep-fluid x mask for an array whose x coordinate is
    LO_X + (i + x_offset_cells)*H (0 for x-nodal, 0.5 for x-centered)."""
    x = LO_X + (np.arange(shape0) + x_offset_cells) * H
    s = (X_WALL - x) / H
    return (s >= BULK_MIN) & (x >= X_BULK_MIN)


class CheckSet:
    """Collect named assertions and report them pytest-style."""

    def __init__(self):
        self.failures = []

    def expect(self, name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
        if not condition:
            self.failures.append(f"{name}: {detail}")

    def finish(self):
        n = len(self.failures)
        print(f"\n{'all checks passed' if n == 0 else f'{n} CHECKS FAILED'}")
        assert n == 0, "\n".join(self.failures)


def run_battery(grid_type):
    sim = setup_simulation(grid_type)
    ck = CheckSet()

    load_synthetic_b(grid_type)
    bx0, by0, bz0 = read_b()
    d0, s_nodes = clean_div(bx0, by0, bz0, grid_type)
    band_before = band_l2(d0, s_nodes)
    dcc0 = None  # set on staggered only; guard (c) below is Yee-only
    if grid_type == "staggered":
        dcc0, _ = yee_cc_div(bx0, by0, bz0)

    # the seed actually landed: the band divergence is O(B0/H), not noise
    ck.expect(
        "divb_clean: seeded band divergence is significant",
        band_before > 0.05 * B0 / H,
        f"band L2 div = {band_before:.4e}, scale B0/H = {B0 / H:.4e}",
    )

    # ONE production step: deposition (no species), the (exactly no-op)
    # Faraday push, the div(B) clean, and the E solve (E = 0 in vacuum).
    sim.step(1)

    bx1, by1, bz1 = read_b()
    d1, _ = clean_div(bx1, by1, bz1, grid_type)
    band_after = band_l2(d1, s_nodes)

    # divergence profile along the wall normal, for the log
    jm, km = d0.shape[1] // 2, d0.shape[2] // 2
    for i in np.nonzero((s_nodes > -1.0) & (s_nodes < 6.0))[0][::-1]:
        print(
            f"  node s={s_nodes[i]:+5.2f}h  div before={d0[i, jm, km]:+.4e}"
            f"  after={d1[i, jm, km]:+.4e}"
        )

    # (0) the clean acted: B in the correction window was modified
    corr = (s_nodes >= 2.0) & (s_nodes <= 3.0)
    d_band_b = float(np.max(np.abs(bx1[1:N_X][corr] - bx0[1:N_X][corr])))
    ck.expect(
        "divb_clean: the band B field was modified by the clean",
        d_band_b > 0.0,
        f"max|dBx| in the correction window = {d_band_b:.3e}",
    )

    # (a) the near-wall-band divergence decreased. The banded grad(div)
    # update is dissipative of the kept band divergence; alpha=0.1 x 5
    # sweeps gives a ~35-55% L2 reduction on this seed, so 0.9 asserts a
    # genuine reduction with margin (not a point value).
    ck.expect(
        "divb_clean: the near-wall band div(B) decreased",
        band_after < 0.9 * band_before,
        f"band L2 div before = {band_before:.4e}, after = {band_after:.4e}, "
        f"ratio = {band_after / band_before:.3f}",
    )

    # (b) the deep fluid -- including the seeded deep divergence bump -- is
    # untouched: the push is an exact no-op and the banded clean must not
    # reach past the band. Round-off tolerance (bit-identical expected).
    tol_b = 1.0e-13 * B0
    for name, b0, b1, off in (
        ("Bx", bx0, bx1, 0.0),
        ("By", by0, by1, 0.5),
        ("Bz", bz0, bz1, 0.5),
    ):
        if grid_type == "collocated":
            off = 0.0  # every component is x-nodal on the collocated grid
        mask = bulk_mask_x(b0.shape[0], off)
        chg = float(np.max(np.abs(b1[mask, :, :] - b0[mask, :, :])))
        ck.expect(
            f"divb_clean: deep-fluid {name} unchanged (band-restricted clean)",
            chg <= tol_b,
            f"max|d{name}| = {chg:.3e}, tol = {tol_b:.1e}",
        )

    # (c) YEE ONLY: no bulk cell-centered divergence injection. The clean's
    # operator is a nodal proxy on the staggered grid, so its correction
    # writes cell-centered divergence wherever it acts; the default BANDED
    # mode keeps it out of the bulk (this guard would fire in the unbounded
    # mode, band_cells <= 0, acting on the deep proxy-divergence bump seeded
    # above). Round-off*scale tolerance -- do not weaken or delete.
    if grid_type == "staggered":
        dcc1, _ = yee_cc_div(bx1, by1, bz1)
        mask_cc = bulk_mask_x(N_X, 0.5)
        inj = float(np.max(np.abs(dcc1[mask_cc, :, :] - dcc0[mask_cc, :, :])))
        tol_cc = 1.0e-12 * B0 / H
        ck.expect(
            "divb_clean: no bulk cell-centered div(B) injection (Yee guard)",
            inj <= tol_cc,
            f"max|d(divcc B)| = {inj:.3e}, tol = {tol_cc:.1e} "
            f"(bulk cc div scale: {float(np.max(np.abs(dcc0[mask_cc, :, :]))):.3e})",
        )

    ck.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid-type",
        choices=["staggered", "collocated"],
        default="staggered",
        help="field staggering of the production div(B)-clean battery",
    )
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    run_battery(args.grid_type)


if __name__ == "__main__":
    main()
