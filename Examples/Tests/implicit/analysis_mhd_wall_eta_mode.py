#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: S. Eric Clark (Helion Energy)
#
# License: BSD-3-Clause-LBNL

"""Wall-band resistivity treatment (implicit_mhd.wall_band_eta_mode) for
the theta-implicit RZ MHD recast: where the field eta steps at the shaped
wall decides where a coil ramp's screening sheet sits.

The deck (inputs_test_rz_theta_implicit_mhd_wall_eta_mode) drives a
uniform external Bz ramp against a dense uniform plasma that reaches the
stepped dielectric wall (r_w = 0.30 m for z < 0, 0.22 m for z > 0); the
composed plasma eta is the constant eta_c = mu0*D (no vacuum boost) and
the override is mu0*1e4 = 500 eta_c. The plotted PC eta register
(implicit_mhd_field_resistivity_e1, the E_theta corner staggering
averaged to cell centres by the diagnostic: cell i = the mean of corner
rows i and i+1 in a flat wall section) is compared against the ANALYTIC
table of each mode, and the screening sheet is located from the numerical
curl of the plotted B.

mode = "override" (the reference twin): corner rows < first_masked carry
eta_c, rows >= first_masked the override (the contour node included under
dielectric); the sheet peaks in the last live cells.

mode = "neumann" (deck default; takes the override twin's plotfile as the
third argument): the masked corner rows within N = 3 of the contour carry
EXACTLY eta_c (the nearest live row's composed eta, bit-equal here), the
override only beyond; the last live cells' current drops, the band
carries the sheet, the total screening current stays comparable.

mode = "transparent": the last N = 3 live corner rows carry
eta_k = eta_c^(k/N) override^((N-k)/N) (k = 0 next to the contour); the
sheet leaves the ramp cells for the first composed cells inward.

Every mode: the eta table is monotone from the plasma to the override
along r in EVERY cell row (stair step included), and the run's own
pc_mhd_block.resistive_validate_assembly asserts the residual/PC twin.

Usage:
  analysis_mhd_wall_eta_mode.py <diag_end> override
  analysis_mhd_wall_eta_mode.py <diag_end> neumann <override_diag_end>
  analysis_mhd_wall_eta_mode.py <diag_end> transparent <override_diag_end>
"""

import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(0)

MU0_WARPX = 1.2566370612685e-06  # WarpX parser mu0 (ablastr::constant)
ETA_C = MU0_WARPX * 20.0  # deck: mu0*D
ETA_OVERRIDE = MU0_WARPX * 1.0e4  # deck: wall_band_eta_override
N_CELLS = 3  # the default neumann / transparent band widths
R_WALL_LO = 0.30  # z < 0
R_WALL_HI = 0.22  # z > 0
FIELDS = ("Br", "Bz", "implicit_mhd_field_resistivity_e1")


def load(plotfile):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, {f: np.squeeze(grid["boxlib", f].value) for f in FIELDS}


ds, state = load(sys.argv[1])
mode = sys.argv[2]
assert mode in ("override", "neumann", "transparent"), f"unknown mode {mode}"

nr, nz = int(ds.domain_dimensions[0]), int(ds.domain_dimensions[1])
r_lo = float(ds.domain_left_edge[0])
z_lo = float(ds.domain_left_edge[1])
dr = (float(ds.domain_right_edge[0]) - r_lo) / nr
dz = (float(ds.domain_right_edge[1]) - z_lo) / nz
r_nodes = r_lo + np.arange(nr + 1) * dr
z_cells = z_lo + (np.arange(nz) + 0.5) * dz

for name, field in state.items():
    assert np.isfinite(field).all(), f"{name} has non-finite values"


def first_masked_node(r_wall):
    """The mask's first masked E_theta corner row (r_node >= r_wall - 1e-3 dr)."""
    return int(np.argmax(r_nodes + 1.0e-3 * dr >= r_wall))


def first_masked_cell(r_wall):
    """The mask's first masked cell (cell centre >= r_wall - 1e-3 dr); the
    cell-centred current profile is indexed by cells, the eta register by
    the corners it averages."""
    r_centres = r_lo + (np.arange(nr) + 0.5) * dr
    return int(np.argmax(r_centres + 1.0e-3 * dr >= r_wall))


def corner_table(mode_, first):
    """Analytic E_theta corner eta per radial node in a flat wall section."""
    eta = np.full(nr + 1, ETA_C)
    eta[first:] = ETA_OVERRIDE
    if mode_ == "neumann":
        eta[first:first + N_CELLS] = ETA_C
    elif mode_ == "transparent":
        for k in range(N_CELLS):
            i = first - 1 - k
            eta[i] = ETA_C ** (k / N_CELLS) * ETA_OVERRIDE ** ((N_CELLS - k) / N_CELLS)
    return eta


def cell_table(mode_, first):
    """The plotted register in a flat section: cell i = mean of nodes i, i+1."""
    eta = corner_table(mode_, first)
    return 0.5 * (eta[:-1] + eta[1:])


eta_plot = state["implicit_mhd_field_resistivity_e1"]
first_lo = first_masked_node(R_WALL_LO)
first_hi = first_masked_node(R_WALL_HI)
print(f"first masked corner row: {first_lo} (z < 0), {first_hi} (z > 0); "
      f"eta_c {ETA_C:.6e}, override {ETA_OVERRIDE:.6e}")

# (a) the analytic table, bit-tight, in cell rows at least N + 1 rows away
# from the stair step at z = 0 (so only the radial rule acts there): the
# two lowest cell rows (z < 0 section) and the two highest (z > 0).
j_step = nz // 2
flat_rows_lo = [0, 1]
flat_rows_hi = [nz - 2, nz - 1]
assert j_step - (flat_rows_lo[-1] + 1) > N_CELLS and flat_rows_hi[0] - j_step > N_CELLS
worst = 0.0
for rows, first in ((flat_rows_lo, first_lo), (flat_rows_hi, first_hi)):
    expected = cell_table(mode, first)
    for j in rows:
        got = eta_plot[:, j]
        rel = np.abs(got - expected) / expected
        worst = max(worst, float(rel.max()))
        assert rel.max() < 1.0e-12, (
            f"{mode}: plotted eta in cell row {j} differs from the analytic table "
            f"(worst rel {rel.max():.3e} at cell {int(rel.argmax())}: "
            f"{got[rel.argmax()]:.16e} vs {expected[rel.argmax()]:.16e})"
        )
print(f"eta table in the flat sections matches the {mode} analytic values "
      f"(worst rel {worst:.2e})")

# (a') monotone from the plasma to the override along r in EVERY row (the
# stair step included: the axial rule may ramp / extend there, but never
# out of order), and bounded by the two values.
log_eta = np.log(eta_plot)
assert (eta_plot >= ETA_C * (1.0 - 1.0e-12)).all()
assert (eta_plot <= ETA_OVERRIDE * (1.0 + 1.0e-12)).all()
for j in range(nz):
    steps = np.diff(log_eta[:, j])
    assert (steps >= -1.0e-9).all(), (
        f"{mode}: eta not monotone along r in cell row {j}: {eta_plot[:, j].tolist()}"
    )
# the largest cell-to-cell log jump in the window around the contour
# (cells first-3 .. first+1; the diagnostic's corner averaging spreads a
# corner step over the two cells sharing that corner) in the flat
# sections: the override's hard step (log 250 between cells first-2 and
# first-1) against the two continuous modes -- neumann pushes the step
# out of the window (to cells first+2 -> first+3), the transparent ramp
# divides it into N equal log steps (exactly log(500)/N per cell pair,
# the cell averaging preserving the geometric ratio).
jumps = []
for rows, first in ((flat_rows_lo, first_lo), (flat_rows_hi, first_hi)):
    for j in rows:
        window = log_eta[first - 3:first + 2, j]
        jumps.append(float(np.diff(window).max()))
contour_jump = max(jumps)
hard_step = np.log(ETA_OVERRIDE / ETA_C)
print(f"largest log-eta jump around the contour (flat sections): {contour_jump:.3f} "
      f"(hard step {hard_step:.3f})")
if mode == "override":
    assert contour_jump > np.log(100.0), "the override twin has no hard step at the contour"
elif mode == "neumann":
    assert contour_jump < 1.0e-9, f"neumann left a step at the contour ({contour_jump:.3e})"
else:
    assert contour_jump <= hard_step / N_CELLS + 1.0e-9, (
        f"transparent: the largest step {contour_jump:.3f} exceeds the ramp's "
        f"{hard_step / N_CELLS:.3f}"
    )


def j_theta(fields):
    """Response |J_theta| = (dBr/dz - dBz/dr) / mu0 from the plotted
    (total) cell-centred fields: the uniform external Bz ramp is
    curl-free, so the numerical curl isolates the screening response."""
    dbr_dz = np.gradient(fields["Br"], dz, axis=1)
    dbz_dr = np.gradient(fields["Bz"], dr, axis=0)
    return (dbr_dz - dbz_dr) / (4.0e-7 * np.pi)


class SheetMetrics:
    """Where the screening sheet sits in one flat wall section: the
    row-mean radial j_theta profile over the section's cell rows, the
    mean |j_theta| of the last two live cells and of the N masked band
    cells, the |j_theta|-weighted centroid (in cells) over the live and
    band cells, the peak cell of the live region and the signed sums.
    `first` is the first masked CELL of the section."""

    def __init__(self, jt, first, rows):
        self.first = first
        self.profile = jt[:, rows].mean(axis=1)
        p = np.abs(self.profile)
        self.last_live = float(p[first - 2:first].mean())
        self.band = float(p[first:first + N_CELLS].mean())
        # the band beyond the contour cell (the contour cell shares its
        # inner corner with the last live corner row, so it carries part
        # of a sheet that sits against the contour)
        self.band_deep = float(p[first + 1:first + N_CELLS + 1].mean())
        w = p[:first + N_CELLS]
        self.centroid = float((w * np.arange(w.size)).sum() / w.sum())
        self.peak = int(np.argmax(p[:first]))
        self.live_sum = float(self.profile[:first].sum())
        self.band_sum = float(self.profile[first:first + N_CELLS].sum())
        self.total = float(self.profile.sum())

    def describe(self, tag):
        cells = " ".join(f"{v:9.1f}" for v in self.profile)
        print(f"{tag}: j_theta by cell [A/m^2]: {cells}")
        print(f"{tag}: last-live mean {self.last_live:.4e}, band mean {self.band:.4e} "
              f"(beyond the contour cell {self.band_deep:.4e}), "
              f"centroid {self.centroid:.2f} cells, live peak cell {self.peak} "
              f"(first masked {self.first}); sums live {self.live_sum:.1f} "
              f"band {self.band_sum:.1f} all {self.total:.1f}")


jt = j_theta(state)
cell_lo = first_masked_cell(R_WALL_LO)
cell_hi = first_masked_cell(R_WALL_HI)
print(f"first masked cell: {cell_lo} (z < 0), {cell_hi} (z > 0)")
sections = ((cell_lo, flat_rows_lo), (cell_hi, flat_rows_hi))
mine = [SheetMetrics(jt, first, rows) for first, rows in sections]
for m, tag in zip(mine, ("z<0", "z>0")):
    m.describe(f"{mode} {tag}")
assert all(m.last_live + m.band > 0.0 for m in mine), "no response current: the drive is dead"

if mode == "override":
    # the sheet sits against the eta step: it peaks in the last two live
    # cells (the outermost live cell shares its outer corner with the
    # overridden contour node, which can move the peak one cell in) and
    # the overridden band carries next to nothing
    for m in mine:
        assert m.peak >= m.first - 2, (
            f"override: the sheet peaks in cell {m.peak}, not in the last live cells "
            f"{m.first - 2}, {m.first - 1}"
        )
        assert m.last_live > 20.0 * m.band_deep, (
            f"override: band current beyond the contour cell {m.band_deep:.3e} not far below "
            f"the live sheet {m.last_live:.3e}"
        )
else:
    _, ref_state = load(sys.argv[3])
    jt_ref = j_theta(ref_state)
    ref = [SheetMetrics(jt_ref, first, rows) for first, rows in sections]
    for m, tag in zip(ref, ("z<0", "z>0")):
        m.describe(f"override twin {tag}")
    for m, r, tag in zip(mine, ref, ("z<0", "z>0")):
        # the total (live + band) screening current is set by the drive
        # and the plasma eta, not by where the eta steps: comparable
        # (measured 0.86 transparent, 1.25 neumann on this deck)
        ratio_total = (m.live_sum + m.band_sum) / (r.live_sum + r.band_sum)
        print(f"{mode} {tag}: total (live + band) screening current ratio vs the override "
              f"twin {ratio_total:.4f}; centroid shift {m.centroid - r.centroid:+.2f} cells; "
              f"peak shift {m.peak - r.peak:+d} cells; last-live ratio "
              f"{m.last_live / r.last_live:.3f}")
        assert 0.5 < ratio_total < 2.0, (
            f"{mode} {tag}: total screening current not comparable to the override twin's "
            f"({ratio_total:.3f})"
        )
        if mode == "neumann":
            # the sheet leaves the fluid cells for the eta-continuous band
            # (measured: centroid +2.4 / +2.8 cells, band 2.2x the live sum,
            # last live cells at 0.5x)
            assert m.centroid > r.centroid + 1.5, (
                f"neumann {tag}: the sheet's centroid moved only {m.centroid - r.centroid:+.2f} cells"
            )
            assert abs(m.band_sum) > abs(m.live_sum), (
                f"neumann {tag}: the band ({m.band_sum:.1f}) does not carry the majority of "
                f"the sheet (live {m.live_sum:.1f})"
            )
            assert m.last_live < 0.7 * r.last_live, (
                f"neumann {tag}: the last live cells still carry "
                f"{m.last_live / r.last_live:.3f} of the override twin's sheet"
            )
        else:
            # the sheet leaves the ramp cells inward, to the first composed
            # cells (measured: peak -3 cells, centroid -2.7 / -2.8 cells,
            # last live cells at 0.07x, band unchanged)
            assert m.peak <= r.peak - 2, (
                f"transparent {tag}: the sheet's peak moved only {m.peak - r.peak:+d} cells"
            )
            assert m.centroid < r.centroid - 1.5, (
                f"transparent {tag}: the sheet's centroid moved only "
                f"{m.centroid - r.centroid:+.2f} cells"
            )
            assert m.last_live < 0.3 * r.last_live, (
                f"transparent {tag}: the ramp cells still carry "
                f"{m.last_live / r.last_live:.3f} of the override twin's sheet"
            )
            assert m.band_deep < 0.02 * r.last_live, (
                f"transparent {tag}: the overridden band carries "
                f"{m.band_deep / r.last_live:.3e} of the override twin's sheet (expected "
                "next to nothing)"
            )

# Newton health over the driven window (a frozen or grinding solve would
# show here); newton.txt APPENDS across reruns: the last six rows are the
# most recent run
newton_rows = np.loadtxt("diags/newton.txt", ndmin=2)
assert len(newton_rows) >= 6, "run did not complete all steps"
newton_rows = newton_rows[-6:]
assert newton_rows[0, 0] == 1, "the last six rows are not one complete run"
newton_iters = newton_rows[:, 2]
print(f"{mode} newton iters/step: {newton_iters.astype(int).tolist()}")
assert np.max(newton_iters) <= 6, (
    f"drive-era Newton grind ({int(np.max(newton_iters))} iters in a step)"
)

print(f"wall-band eta mode test ({mode}) PASSED")
