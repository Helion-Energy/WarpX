#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Braginskii clamp CLASS of the shaped-wall interface conductance
(implicit_mhd.wall_conduction_scale) for the theta-implicit RZ MHD
recast.

A hot static uniform column (T0 = 100 eV both species, B uniform along
z so every RADIAL face has b_n = 0 and the interior tensor nn
projection is chi_perp exactly) sits inside a straight cylinder wall
against a wall_thermal_bc = dirichlet bath at T_wall = 2 eV. The hard
pin carries no free-streaming cap and no one-sided gate, so the ONLY
thing this knob changes is the coefficient that multiplies the
half-cell exchange chi_wall rho (e - e_wall) 2/dr:

  mode = "perp" (the default, and the pre-2026-09-02 behavior): the
  wall face takes the chi_PERP-clamped coefficient -- the same
  coefficient the interior radial faces use, because b_n = 0 there.

  mode = "parallel": the wall face takes the chi_PAR-clamped
  coefficient instead, reproducing the reference code's measured wall conductance
  G (their implicit temperature solve pins every cut-cell vertex at the
  0.5 eV anchor for both species with G bounded by the PARALLEL clamp
  maximum xile_mx = xili_mx = 1e6 n kB). With the deck's production-
  ratio clamps that is a 1e4 stronger wall.

Both per-component clamps are saturated over the whole run, so the
effective diffusivities are the state-independent constants
(gamma - 1) * chi_*_cap to machine precision and, with theta = 1, each
run's conduction subsystem is EXACTLY a linear backward-Euler
cylindrical diffusion column with a half-cell Dirichlet bath on the
wall face. This analysis

  * recomputes the raw Braginskii chi_par/chi_perp of BOTH species from
    the run's own measured face state and asserts the saturation margin
    (so the analytic prediction below cannot silently stop applying);
  * integrates the identical discrete column operator for the mode
    under test and gates the measured per-channel drain against it,
    both after the FIRST step (where the analytic drain is the pure
    wall-face exchange with every interior gradient still exactly zero)
    and over the whole run;
  * checks the masked band stayed a rigid bit-static conductor and the
    wall ledger closes against the interior loss;
  * in "parallel" mode, gates the measured drain against the "perp"
    twin's: the ratio of the first-step drains must be the analytic
    clamp ratio.

Usage: analysis_mhd_wall_conduction_scale.py <diag_dir> perp
       analysis_mhd_wall_conduction_scale.py <diag_dir> parallel \
           <perp_diag_dir>
"""

import glob
import sys

import numpy as np
import yt

yt.set_log_level(50)

diag_dir = sys.argv[1]
mode = sys.argv[2]
assert mode in ("perp", "parallel"), f"unknown mode {mode}"

# WarpX PhysConst / ablastr::constant values (the parser's m_p, q_e, ...)
proton_mass = 1.67262192595e-27
electron_mass = 9.1093837139e-31
qe = 1.602176634e-19
eps0 = 8.8541878128e-12

# ---- deck constants (kept in lockstep with the inputs file) ----------
n0 = 1.0e18
T0_ev = 100.0
Twall_ev = 2.0
gamma = 5.0 / 3.0
B0 = 5.0e-4
Lr = 0.1
Lz = 0.1
nr = 8
nz = 8
dr = Lr / nr
coulomb_log = 10.0
chi_par_cap = 1.0e5  # defined (kappa/(n kB)) convention
chi_perp_cap = chi_par_cap / 1.0e4
chi_op_par = (gamma - 1.0) * chi_par_cap  # operator diffusivities
chi_op_perp = (gamma - 1.0) * chi_perp_cap
dt = 0.125 * dr**2 / chi_op_par
nsteps = 6
# the mask's conductor cells (center radius on/outside the polyline
# radius, with the 1e-3*dr sliver -- exactly as ImplicitMHDWallMask
# builds it); the interface is the grid face above the last interior
# cell
r_centers = (np.arange(nr) + 0.5) * dr
masked = (r_centers + 1.0e-3 * dr) >= 0.065
n_interior = int(np.argmax(masked))
assert 2 <= n_interior < nr, "degenerate wall mask layout"

chi_wall_op = chi_op_par if mode == "parallel" else chi_op_perp

# specific internal energies of the uniform column and the wall bath:
# e = (q/m_i) T[eV] / (gamma - 1) (the density-free reservoir
# conversion; no temperature floors are set, so the pin target is the
# reservoir value exactly)
e_per_ev = (qe / proton_mass) / (gamma - 1.0)
e0 = e_per_ev * T0_ev
e_wall = e_per_ev * Twall_ev

FIELDS = (
    "implicit_mhd_mass_density",
    "implicit_mhd_electron_energy",
    "implicit_mhd_ion_energy",
    "implicit_mhd_momentum_r",
    "implicit_mhd_momentum_t",
    "implicit_mhd_momentum_z",
)


def load_state(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return {f: np.squeeze(data["boxlib", f].value) for f in FIELDS}


def profiles(state):
    """z-averaged radial profiles of the two specific internal energies
    (the ion channel subtracts the advective kinetic energy exactly as
    the solver's recovery does)."""
    rho = state["implicit_mhd_mass_density"]
    kinetic = (
        0.5
        * sum(state[f"implicit_mhd_momentum_{c}"] ** 2 for c in ("r", "t", "z"))
        / rho
    )
    e_electron = np.mean(state["implicit_mhd_electron_energy"] / rho, axis=1)
    e_ion = np.mean((state["implicit_mhd_ion_energy"] - kinetic) / rho, axis=1)
    return e_electron[:n_interior], e_ion[:n_interior]


def braginskii(temperature_ev):
    """Raw Braginskii Z = 1 chi_par/chi_perp of both species at the
    uniform column density, in the solver's OPERATOR convention
    (chi_code = (gamma - 1) * chi_Braginskii) -- the exact expressions
    the kernel evaluates."""
    tau_shared = np.pi**1.5 * eps0**2 / (qe**4 * coulomb_log)
    kt = qe * temperature_ev  # kB * T with T in kelvin
    tau_e = (
        6.0 * np.sqrt(2.0) * np.sqrt(electron_mass) * tau_shared * kt * np.sqrt(kt) / n0
    )
    tau_i = 12.0 * np.sqrt(proton_mass) * tau_shared * kt * np.sqrt(kt) / n0
    par_e = (gamma - 1.0) * 3.16 * kt * tau_e / electron_mass
    par_i = (gamma - 1.0) * 3.9 * kt * tau_i / proton_mass
    xe = ((qe / electron_mass) * tau_e * B0) ** 2
    xi = ((qe / proton_mass) * tau_i * B0) ** 2
    fit_e = (4.664 / 11.92 * xe + 1.0) / (
        ((1.0 / 3.7703) * xe + 14.79 / 3.7703) * xe + 1.0
    )
    fit_i = (2.0 / 2.645 * xi + 1.0) / (((1.0 / 0.677) * xi + 2.70 / 0.677) * xi + 1.0)
    return par_e, par_i, par_e * fit_e, par_i * fit_i


def column_model(wall_chi, steps=None):
    """Backward-Euler cylindrical diffusion column at the run's exact
    discretization: interior radial faces at chi_op_perp (b_n = 0, so
    the tensor nn projection is chi_perp), r_face/r_center divergence
    weights (the r = 0 face drops out), and the outer face of the last
    interior cell exchanging with the bath at 2 * wall_chi / dr (the
    half-cell Dirichlet pin: uncapped and ungated)."""
    steps = nsteps if steps is None else steps
    rate_interior = chi_op_perp * dt / dr**2
    rate_wall = wall_chi * dt / dr**2
    matrix = np.zeros((n_interior, n_interior))
    bath = np.zeros(n_interior)
    for k in range(n_interior):
        r_center = (k + 0.5) * dr
        weight_low = k * dr / r_center
        weight_high = (k + 1.0) * dr / r_center
        diagonal = 1.0
        if k > 0:
            matrix[k, k - 1] -= rate_interior * weight_low
            diagonal += rate_interior * weight_low
        if k < n_interior - 1:
            matrix[k, k + 1] -= rate_interior * weight_high
            diagonal += rate_interior * weight_high
        else:
            diagonal += 2.0 * rate_wall * weight_high
            bath[k] = 2.0 * rate_wall * weight_high * e_wall
        matrix[k, k] += diagonal
    history = [np.full(n_interior, e0)]
    for _ in range(steps):
        history.append(np.linalg.solve(matrix, history[-1] + bath))
    return np.array(history)


plotfiles = sorted(glob.glob(f"{diag_dir}/diag??????"))
assert len(plotfiles) == nsteps + 1, f"need one snapshot per step, got {len(plotfiles)}"
states = [load_state(p) for p in plotfiles]
for step, state in enumerate(states):
    for name, field in state.items():
        assert np.isfinite(field).all(), f"{name} not finite at step {step}"

measured = [profiles(s) for s in states]
model = column_model(chi_wall_op)

print(f"mode = {mode} (wall chi_op = {chi_wall_op:.6g} m^2/s)")

# ---- the clamps really are saturated, at the run's own face state ----
# the wall face's coefficient temperature is the reservoir average
# 0.5 * (T_interior + T_wall); take the coldest one the run reaches.
face_t_hot = 0.5 * (T0_ev + Twall_ev)
face_t_cold = 0.5 * (
    min(measured[-1][0][-1], measured[-1][1][-1]) / e_per_ev + Twall_ev
)
for label, face_t in (("hot", face_t_hot), ("cold", face_t_cold)):
    par_e, par_i, perp_e, perp_i = braginskii(face_t)
    margins = (
        par_e / chi_op_par,
        par_i / chi_op_par,
        perp_e / chi_op_perp,
        perp_i / chi_op_perp,
    )
    print(
        f"clamp margin ({label} face state {face_t:.2f} eV): "
        f"par_e {margins[0]:.1f}x  par_i {margins[1]:.1f}x  "
        f"perp_e {margins[2]:.1f}x  perp_i {margins[3]:.1f}x"
    )
    assert min(margins) > 3.0, (
        f"a Braginskii clamp came off its cap ({label}: {margins}) -- the "
        "analytic column below no longer applies"
    )

# ---- the initial column is uniform ----------------------------------
for name, initial in (("electron", measured[0][0]), ("ion", measured[0][1])):
    assert np.all(np.abs(initial / e0 - 1.0) < 1.0e-12), (
        f"{name} column not uniform at t = 0"
    )

# ---- FIRST-STEP drain: the cleanest read of the wall conductance -----
# The whole column starts uniform, so after ONE backward-Euler step the
# entire departure from e0 is the wall exchange propagated through the
# (state-independent) implicit operator: no accumulated advective
# response has had time to contaminate it. The measured drop is gated
# against the model's first step per channel, both at the wall ring and
# summed over the column.
model_first_drop = e0 - model[1][-1]
print(f"first-step wall-ring drain: model {model_first_drop / e_per_ev:.6e} eV")
first_drops = {}
for name, channel in (("electron", 0), ("ion", 1)):
    drop = e0 - measured[1][channel][-1]
    first_drops[name] = drop
    ratio = drop / model_first_drop
    column_ratio = np.sum(e0 - measured[1][channel]) / np.sum(e0 - model[1])
    print(
        f"  {mode}/{name}: measured {drop / e_per_ev:.6e} eV, "
        f"ring measured/model = {ratio:.5f}, column = {column_ratio:.5f}"
    )
    assert 0.98 < ratio < 1.02, (
        f"{mode} {name} first-step wall drain off its analytic value: {ratio:.5f}"
    )
    assert 0.98 < column_ratio < 1.02, (
        f"{mode} {name} first-step column drain off its analytic value: "
        f"{column_ratio:.5f}"
    )

# ---- whole-run column: the same discrete operator, integrated --------
for name, channel in (("electron", 0), ("ion", 1)):
    final = measured[-1][channel]
    ring_ratio = (e0 - final[-1]) / (e0 - model[-1][-1])
    column_ratio = np.sum(e0 - final) / np.sum(e0 - model[-1])
    print(
        f"  {mode}/{name} after {nsteps} steps: ring T = "
        f"{final[-1] / e_per_ev:.4f} eV (model "
        f"{model[-1][-1] / e_per_ev:.4f} eV), ring drop ratio "
        f"{ring_ratio:.5f}, column drop ratio {column_ratio:.5f}"
    )
    assert 0.98 < ring_ratio < 1.02, (
        f"{mode} {name} wall-ring cooling off its Dirichlet column: {ring_ratio:.5f}"
    )
    assert 0.98 < column_ratio < 1.02, (
        f"{mode} {name} column-integrated drain off its prediction: {column_ratio:.5f}"
    )
    # the drain is monotone and never overshoots the bath
    ring_trace = np.array([m[channel][-1] for m in measured])
    assert np.all(np.diff(ring_trace) < 0.0), (
        f"{mode} {name} wall ring not draining monotonically: {ring_trace}"
    )
    assert final[-1] > e_wall, f"{mode} {name} ring undershot the bath"

# ---- the masked band stayed a rigid, bit-static conductor ------------
conductor = np.broadcast_to(masked[:, None], states[0][FIELDS[0]].shape)
for f in FIELDS:
    scale = np.max(np.abs(states[1][f]))
    scale = scale if scale > 0.0 else 1.0
    drift = (
        np.max(np.abs(states[-1][f] - states[1][f]), initial=0.0, where=conductor)
        / scale
    )
    assert drift < 1.0e-12, f"masked band changed ({f}: {drift:.3e})"

# ---- the wall ledger closes against the interior loss ----------------
# rows of "step mass energy" in true J; the interior loss is the
# volume-weighted (2 pi r dr dz per z row) internal-energy drop of the
# unmasked cells, which under theta = 1 the accepted-state booking
# matches.
volume = 2.0 * np.pi * r_centers[:n_interior] * dr * (Lz / nz) * nz
rho0 = n0 * proton_mass
interior_loss = sum(
    float(np.sum((measured[0][channel] - measured[-1][channel]) * rho0 * volume))
    for channel in (0, 1)
)
ledger_rows = np.loadtxt(f"{diag_dir}/wall_ledger.txt", ndmin=2)
ledger_energy = float(ledger_rows[-1, 2])
closure = abs(ledger_energy - interior_loss) / abs(interior_loss)
print(
    f"  ledger {ledger_energy:.6e} J vs interior loss "
    f"{interior_loss:.6e} J, closure {closure:.3e}"
)
assert closure < 5.0e-3, f"wall ledger does not close ({closure:.3e})"

# ---- knob discrimination against the perp twin ----------------------
if mode == "parallel":
    perp_dir = sys.argv[3]
    perp_plotfiles = sorted(glob.glob(f"{perp_dir}/diag??????"))
    assert len(perp_plotfiles) == nsteps + 1
    perp_measured = [profiles(load_state(p)) for p in perp_plotfiles]
    # analytic first-step separation: the same column integrated with
    # each mode's own wall coefficient (a shade under the raw 1e4 clamp
    # ratio, because the parallel wall face is already backward-Euler
    # saturated on its first step while the perp one is not)
    model_ratio = (e0 - column_model(chi_op_par, 1)[1][-1]) / (
        e0 - column_model(chi_op_perp, 1)[1][-1]
    )
    for name, channel in (("electron", 0), ("ion", 1)):
        perp_drop = e0 - perp_measured[1][channel][-1]
        ratio = first_drops[name] / perp_drop
        print(
            f"  parallel/perp first-step drain ratio ({name}): "
            f"{ratio:.1f} (analytic {model_ratio:.1f})"
        )
        assert 0.97 < ratio / model_ratio < 1.03, (
            f"{name} parallel/perp drain separation off the analytic clamp "
            f"ratio: {ratio:.1f} vs {model_ratio:.1f}"
        )
        # ...and the perp twin is the old, essentially inert wall
        assert perp_drop / e0 < 1.0e-4, (
            f"{name} perp-mode wall drain unexpectedly large "
            f"({perp_drop / e0:.3e} of e0)"
        )

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert 1 <= newton_history[-1][2] <= 20

print(f"wall_conduction_scale ({mode}): all gates passed")
