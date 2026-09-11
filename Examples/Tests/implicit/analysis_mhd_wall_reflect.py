#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Shaped-wall FLUID PERMEABILITY (implicit_mhd.wall_fluid_bc) for the
theta-implicit RZ MHD recast.

See inputs_test_rz_theta_implicit_mhd_wall_reflect: a uniform plasma is
driven into a NOTCHED BOTTLE wall (r_w = 0.30 m, a 0.22 m notch for
|z| < 0.08 m, end plugs down to a one-cell axis clearance for |z| > 0.30 m)
by a radial expansion and by two axial pulses aimed at the notch ledges,
so the stair-step interface presents r-normal faces at three radii and
z-normal faces of both orientations. The plugs close the domain with the
wall itself (no z-end fluid boundary of this solver is mass-tight); the
only non-interface exits are the two axis cells' end faces, whose
advective flux is integrated per step to close the live-mass budget.

mode = "reflect": the masked side presents the ODD-normal-momentum image
(the impermeable no-normal-flow wall of the reference code):
  1. the masked band stays rigid (the exterior clamp; every fluid moment
     bit-static after the first snapshot);
  2. the LIVE fluid mass is conserved to solver tolerance (measured
     drift versus the absorb twin's loss, see below);
  3. the wall ledger books an exactly zero crossing mass (central: the
     arithmetic mean of equal-and-opposite momenta; the Rusanov jump of
     the density is zero) -- round-off allowed for the hlld variant;
  4. the arriving plasma PILES UP: the last live ring at the outer wall
     radius ends denser than it started and denser than the absorb
     twin's (when the twin is given);
  5. reported, not gated: the stair-corner (nook) pockets -- max T_i, T_e,
     E_i and n over the corner-classified live cells per snapshot -- and
     the Newton record (iterations per step, exit residuals) next to the
     twin's.

mode = "absorb" (the default, the dielectric scraper, set on the CLI):
the same deck removes a measurable fraction of the live mass through
the interface, the ledger's cumulative mass matches the loss (closure),
and the band stays rigid.

mode = "reflect_pin" (CLI wall_corner_temperature_pin_rate=1e7; the twin
argument is the UNPINNED reflect run): the reflect gates, plus the four
two-walled nook cells relax toward the 2 eV wall reservoir in both
species -- against this drive's ram and ledge conduction they settle at
36 / 34 eV (T_i / T_e) versus the unpinned twin's 74 / 115 eV, so the
gate is the ratio to the twin -- while the ledge cells that are not
nooks are NOT pinned (within a factor two of the twin's): the
classification is exactly the two-walled set.

Usage:
  analysis_mhd_wall_reflect.py <diag_dir> absorb
  analysis_mhd_wall_reflect.py <diag_dir> reflect [<absorb_diag_dir>]
  analysis_mhd_wall_reflect.py <diag_dir> reflect_pin <reflect_diag_dir>
"""

import glob
import os
import sys

import numpy as np
import yt

diag_dir = sys.argv[1]
mode = sys.argv[2]
twin_dir = sys.argv[3] if len(sys.argv) > 3 else None
assert mode in ("reflect", "absorb", "reflect_pin"), f"unknown mode {mode}"
pin_mode = mode == "reflect_pin"
if pin_mode:
    mode = "reflect"  # the reflect gates apply; the twin argument is the UNPINNED reflect run

proton_mass = 1.67262192595e-27  # WarpX parser m_p (ablastr::constant)
qe = 1.602176634e-19
gamma = 5.0 / 3.0
n0 = 1.0e20
rho0 = n0 * proton_mass

FIELDS = (
    "implicit_mhd_mass_density",
    "implicit_mhd_ion_energy",
    "implicit_mhd_electron_energy",
    "implicit_mhd_momentum_r",
    "implicit_mhd_momentum_t",
    "implicit_mhd_momentum_z",
)


def load_state(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    state = {f: np.squeeze(data["boxlib", f].value) for f in FIELDS}
    return ds, state


def load_series(directory):
    plotfiles = sorted(glob.glob(f"{directory}/diag??????"))
    assert len(plotfiles) >= 3, f"need at least 3 snapshots in {directory}"
    assert len(plotfiles) == 17, (
        f"the budget needs one snapshot per step (17 for 16 steps), got "
        f"{len(plotfiles)} in {directory}")
    ds0, _ = load_state(plotfiles[0])
    states = []
    for plotfile in plotfiles:
        _, state = load_state(plotfile)
        for name, field in state.items():
            assert np.isfinite(field).all(), f"{name} has non-finite values"
        states.append(state)
    return ds0, states


ds0, states = load_series(diag_dir)
nr, nz = int(ds0.domain_dimensions[0]), int(ds0.domain_dimensions[1])
r_lo = float(ds0.domain_left_edge[0])
z_lo = float(ds0.domain_left_edge[1])
dr = (float(ds0.domain_right_edge[0]) - r_lo) / nr
dz = (float(ds0.domain_right_edge[1]) - z_lo) / nz
r_centers = r_lo + (np.arange(nr) + 0.5) * dr
z_centers = z_lo + (np.arange(nz) + 0.5) * dz
# absolute cell volumes 2 pi r dr dz
cell_volume = 2.0 * np.pi * np.outer(r_centers * dr, np.full(nz, dz))

# The mask, replicated as ImplicitMHDWallMask builds it from
# wall_bottle_notch_r30_r22.csv: conductor when the cell-centre radius is
# on/outside the polyline radius at the cell-centre z (1e-3 dr sliver).
wall_radius = np.where(np.abs(z_centers) > 0.30, 0.02,
                       np.where(np.abs(z_centers) < 0.08, 0.22, 0.30))
masked = (r_centers[:, None] + 1.0e-3 * dr) >= wall_radius[None, :]
live = ~masked
first_masked = np.argmax(masked, axis=0)  # first masked radial index per z
assert np.all(masked[first_masked, np.arange(nz)]), "mask never reaches r_max"
# Stair CORNER (nook) cells: the last live cell of a z row whose radial
# index is masked in an axial neighbour row -- TWO wall faces, the
# solver's corner classification for wall_corner_temperature_pin_rate
# (edge rows use the constant continuation). On this bottle that is the
# cell at the outer wall radius next to each notch ledge and next to
# each plug: four cells.
fm_lo = np.concatenate(([first_masked[0]], first_masked[:-1]))
fm_hi = np.concatenate((first_masked[1:], [first_masked[-1]]))
corner = np.zeros_like(masked)
ledge = np.zeros_like(masked)  # live cells with a z-normal interface face
for j in range(nz):
    i_last = first_masked[j] - 1
    if i_last >= fm_lo[j] or i_last >= fm_hi[j]:
        corner[i_last, j] = True
    for i in range(min(fm_lo[j], fm_hi[j]), first_masked[j]):
        ledge[i, j] = True
n_corner = int(corner.sum())
assert n_corner == 4, f"expected 4 two-walled nook cells (2 notch + 2 plug), got {n_corner}"
assert int(ledge.sum()) == 36, f"expected 36 ledge-facing cells (2 x 4 notch + 2 x 14 plug), got {int(ledge.sum())}"
# last live ring at the OUTER wall radius (rows 0.08 < |z| < 0.30)
outer_rows = (np.abs(z_centers) > 0.08) & (np.abs(z_centers) < 0.30)
outer_ring = np.zeros_like(masked)
outer_ring[first_masked[outer_rows][0] - 1, outer_rows] = True
assert outer_ring.sum() == outer_rows.sum()
# the only non-interface exits: the axis cells' faces at the two z ends
axis_end = np.zeros_like(masked)
axis_end[0, 0] = True
axis_end[0, -1] = True


def live_mass(state):
    return float(np.sum(state["implicit_mhd_mass_density"] * cell_volume,
                        where=live))


def temperatures_ev(state):
    rho = state["implicit_mhd_mass_density"]
    number = rho / proton_mass
    kinetic = 0.5 * (state["implicit_mhd_momentum_r"] ** 2 +
                     state["implicit_mhd_momentum_t"] ** 2 +
                     state["implicit_mhd_momentum_z"] ** 2) / rho
    p_i = (gamma - 1.0) * (state["implicit_mhd_ion_energy"] - kinetic)
    p_e = (gamma - 1.0) * state["implicit_mhd_electron_energy"]
    return p_i / (number * qe), p_e / (number * qe), number


def report_series(label, series):
    print(f"--- {label}: {len(series)} snapshots (every 4th printed)")
    for k, state in enumerate(series):
        if k % 4 != 0:
            continue
        t_i, t_e, number = temperatures_ev(state)
        e_i = state["implicit_mhd_ion_energy"]
        print(f"  snap {k}: live mass {live_mass(state):.12e} kg | nook max "
              f"T_i {t_i[corner].max():.1f} eV  T_e {t_e[corner].max():.1f} eV  "
              f"E_i {e_i[corner].max():.3e} J/m^3  n {number[corner].max():.3e} m^-3"
              f" | ledge max T_i {t_i[ledge].max():.1f} eV  n {number[ledge].max():.3e}"
              f" | outer-ring mean n/n0 {number[outer_ring].mean() / n0:.4f}"
              f" | live max T_i {t_i[live].max():.1f} eV"
              f" | axis end-cell |u_z| {np.abs(state['implicit_mhd_momentum_z'][axis_end] / state['implicit_mhd_mass_density'][axis_end]).max():.2e} m/s")


def report_newton(directory):
    path = os.path.join(directory, "newton.txt")
    if not os.path.exists(path):
        print(f"  (no newton diagnostic file at {path})")
        return
    with open(path) as handle:
        lines = [line for line in handle if line.strip()]
    print(f"  newton.txt: {len(lines)} lines; last: {lines[-1].strip()[:160]}")


report_series(mode, states)
report_newton(diag_dir)

# ---- rigid band (both modes): every fluid moment of every masked cell
# is bit-static across the post-clamp snapshots -------------------------
for f in FIELDS:
    scale = np.max(np.abs(states[1][f]))
    scale = scale if scale > 0.0 else 1.0
    drift = np.max(np.abs(states[-1][f] - states[1][f]), initial=0.0,
                   where=masked) / scale
    print(f"masked-band drift {f}: {drift:.3e}")
    assert drift < 1.0e-12, f"masked band changed ({f}: {drift:.3e})"

# ---- ledger: rows of "step mass energy" in kg / J, INTO the wall -------
ledger_rows = np.loadtxt(f"{diag_dir}/wall_ledger.txt", ndmin=2)
ledger_mass, ledger_energy = ledger_rows[-1, 1], ledger_rows[-1, 2]
mass = np.array([live_mass(s) for s in states])
# The exterior clamp at first-step sanitize time touches MASKED cells
# only, so the live budget runs from the initial snapshot.
mass_ref = mass[0]
loss = mass_ref - mass[-1]
print(f"live mass: initial {mass_ref:.12e}, final {mass[-1]:.12e} kg; "
      f"relative loss {loss / mass_ref:.6e}")
print(f"ledger: crossing mass {ledger_mass:.6e} kg ({ledger_mass / mass_ref:.3e} "
      f"of the live mass), energy {ledger_energy:.6e} J")

# The only non-interface exits are the two axis cells' end faces (the
# polyline may not reach the axis, and no z-end fluid boundary of this
# solver is mass-tight). Their advective mass flux is the end cell's own
# m_z (Neumann ghost = copy: central mean of equal momenta, zero Rusanov
# jump), so with theta = 1 and one snapshot per step the leak integrates
# exactly as dt sum_n m_z^{n+1} A over the run (A = pi dr^2), outward
# positive. The live budget must then CLOSE:
#   M(end) - M(0) = -ledger_mass - leak    (to the nonlinear tolerance)
# -- every gram that left the live fluid is accounted for by the ledger
# (the stair faces) or by the axis channels; nothing else.
dt = float(np.loadtxt(f"{diag_dir}/newton.txt", ndmin=2)[0, 1])
axis_area = np.pi * dr * dr
leak = 0.0
for state in states[1:]:
    mz = state["implicit_mhd_momentum_z"]
    leak += dt * axis_area * (mz[0, -1] - mz[0, 0])
budget_gap = (mass[-1] - mass_ref + ledger_mass + leak) / mass_ref
print(f"axis-end leak (outward) {leak:.6e} kg = {leak / mass_ref:.3e} of the "
      f"live mass; budget gap (M_end - M_0 + ledger + leak)/M_0 = {budget_gap:.3e}")
assert abs(budget_gap) < 1.0e-7, (
    f"live-mass budget does not close ({budget_gap:.3e}): mass left or "
    f"entered the live fluid through something other than the stair faces "
    f"and the axis ends")

if mode == "absorb":
    # The scraper removes a measurable fraction of the live mass through
    # the stair faces (measured 1.6e-1 on this deck; require a
    # conservative fraction) and the ledger books it (positive).
    assert loss > 5.0e-2 * mass_ref, (
        f"absorbing wall removed too little ({loss / mass_ref:.3e})")
    assert ledger_mass > 0.5 * loss, (
        f"absorb ledger ({ledger_mass:.3e}) far below the loss ({loss:.3e})")
else:
    # 2. conservation: apart from the axis channels, nothing leaves --
    # the drift of the live mass over the whole run is the axis leak
    # alone (measured 1.5e-5 of the mass with the channels' 2.7e3 m/s
    # end flow) and stays four orders below the twin's loss.
    drift = np.max(np.abs(mass[1:] / mass_ref - 1.0))
    print(f"live-mass drift (reflect): {drift:.3e} (axis leak {abs(leak) / mass_ref:.3e})")
    assert drift < 1.0e-4, f"reflecting wall leaked mass ({drift:.3e})"
    # 3. the ledger's crossing mass is exactly zero (central) or
    # round-off (hlld): no mass ever crossed a stair face.
    assert abs(ledger_mass) < 1.0e-12 * mass_ref, (
        f"reflecting wall ledger booked a crossing mass "
        f"({ledger_mass:.3e} kg = {ledger_mass / mass_ref:.3e} of the live mass)")
    # 4. pile-up: the outer last-live ring ends denser than it started.
    t_i, t_e, number = temperatures_ev(states[-1])
    ring_ratio = number[outer_ring].mean() / n0
    print(f"outer last-live ring mean n/n0 at the end: {ring_ratio:.4f}")
    assert ring_ratio > 1.02, (
        f"no pile-up at the impermeable wall (ring n/n0 {ring_ratio:.4f})")
    if twin_dir is not None and pin_mode:
        # 5. the corner pin: the two-walled nooks relax toward the 2 eV wall
        # reservoir at 0.5 per step (rate 1e7/s x 50 ns) in BOTH species
        # against the pulse's ram and the conduction from the 50 eV ledge
        # neighbours, which hold them well above the target on this drive
        # (measured 36 / 34 eV vs the unpinned twin's 74 / 115 eV, the
        # cooled nook 1.6x denser): gate on the RATIO to the unpinned
        # twin, and require the ledge cells that are not nooks to be left
        # alone (within a factor two of the twin's).
        _, twin = load_series(twin_dir)
        report_series("unpinned reflect twin", twin)
        t_i_twin, t_e_twin, _ = temperatures_ev(twin[-1])
        nook_ti, nook_te = t_i[corner].max(), t_e[corner].max()
        twin_ti, twin_te = t_i_twin[corner].max(), t_e_twin[corner].max()
        print(f"pinned nooks: max T_i {nook_ti:.2f} eV, max T_e {nook_te:.2f} eV; "
              f"unpinned twin nooks: max T_i {twin_ti:.1f}, T_e {twin_te:.1f} eV; "
              f"ratios {nook_ti / twin_ti:.3f}, {nook_te / twin_te:.3f}")
        assert nook_ti < 0.65 * twin_ti and nook_te < 0.5 * twin_te, (
            f"corner pin did not cool the nooks against the unpinned twin "
            f"(T_i {nook_ti:.2f} vs {twin_ti:.1f}, T_e {nook_te:.2f} vs {twin_te:.1f} eV)")
        not_nook = ledge & ~corner
        ratio = t_i[not_nook].max() / t_i_twin[not_nook].max()
        print(f"ledge cells that are not nooks: max T_i pinned/unpinned {ratio:.3f}")
        assert 0.5 < ratio < 2.0, (
            f"the corner pin touched cells outside the two-walled set (ratio {ratio:.3f})")
    elif twin_dir is not None:
        _, twin = load_series(twin_dir)
        report_series("absorb twin", twin)
        report_newton(twin_dir)
        twin_mass = np.array([live_mass(s) for s in twin])
        twin_loss = (twin_mass[0] - twin_mass[-1]) / twin_mass[0]
        print(f"twin relative loss {twin_loss:.6e} vs reflect drift {drift:.3e}")
        assert twin_loss > 100.0 * drift, (
            "the reflect run does not separate from the absorb twin")
        _, _, twin_number = temperatures_ev(twin[-1])
        twin_ratio = twin_number[outer_ring].mean() / n0
        print(f"twin outer-ring mean n/n0: {twin_ratio:.4f}")
        assert ring_ratio > twin_ratio, (
            "the impermeable wall does not hold more gas at the wall than "
            "the scraper")

print(f"wall_fluid_bc = {mode}" + (" + corner pin" if pin_mode else "") + ": all checks passed")
