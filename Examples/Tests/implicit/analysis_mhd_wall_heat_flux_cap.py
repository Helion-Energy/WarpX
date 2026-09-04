#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Wall heat-flux cap (implicit_mhd.wall_heat_flux_cap) of the conductive
wall exchanges for the theta-implicit RZ MHD recast: the analytic gate.

Two geometries share the model. The SHAPED-WALL deck
(inputs_test_rz_theta_implicit_mhd_wall_heat_flux_cap): a hot static
uniform two-temperature column (Te0 = 100 eV, Ti0 = 40 eV) inside a
straight cylinder wall under the dirichlet pin at T_wall = 2 eV with a
CONSTANT isotropic chi in both channels, so the uncapped half-cell wall
exchange is exactly q_unl = chi rho (e - e_wall) 2/dr per channel. The
Z-END deck (inputs_test_rz_theta_implicit_mhd_z_wall_conduction with
z_wall_conduction on, diagnostics every step): the same hot column
(T0 = 100 eV both species) between two z_wall_temperature ends whose
saturated-clamp Braginskii chi_par is the state-independent constant
(gamma - 1) chi_cap (the z_wall_conduction ctest asserts that
saturation), so q_unl = chi_op rho (e - e_wall) 2/dz at both end faces.

Every cap the ctests select is the smooth harmonic form

    q = q_unl / (1 + q_unl / q_cap),

  mode = "sonic":          q_cap = f n kB T_s c_s,
                           c_s^2 = gamma (kB Te + kB Ti)/m_i,
                           f = 2.5 unless given on the command line;
  mode = "free_streaming": q_cap = f n kB T_s v_th,s, v_th,s = sqrt(kB
                           T_s/m_s), f = 1 (the legacy factor with
                           conduction_flux_limit_factor unset) -- the
                           knob-unset dirichlet_limited pin, and the
                           knob's free_streaming override on the hard pin.

n, Te and Ti are the face state the cap reads: the interior density and
the reservoir-averaged temperatures (T_interior + T_wall)/2, at the
END-of-step state (theta = 1, conduction_coefficient_state = theta). So
the first implicit step of the capped column is a NONLINEAR backward-Euler
step in which the two channels couple through c_s. This analysis solves
it (a fixed-point iteration over the wall sinks around the exact linear
interior column, contraction factor ~ the half-cell diffusion number)
and gates, per channel, the measured first-step wall-ring and column
drains against it at 1 percent. It ALSO integrates the uncapped column
and a hard min(q_unl, q_cap) column and requires the measurement to be
well separated from both: the FORM of the cap is gated, not only its
magnitude. The remaining checks are the usual wall contract: monotone
ring cooling that never undershoots the bath, a bit-static masked band
(shaped wall) and a wall ledger that closes against the interior's
TOTAL fluid-energy loss (shaped wall: the ledger books everything that
crosses the interface, the capped conductive drain and the enthalpy the
acoustic response advects into the wall, so total energy is the exact
statement while internal energy alone is off by the p dV work).

Usage: analysis_mhd_wall_heat_flux_cap.py <diag_dir> <sonic|free_streaming> [factor] [--z-end] [--dt-factor=F]
"""

import glob
import sys

import numpy as np
import yt

yt.set_log_level(50)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
z_end = "--z-end" in sys.argv[1:]
# z-end only: the deck's conduction-paced dt = 0.5 dz^2/chi_op scaled by
# this factor on the command line (my_constants.dt=<factor>*0.5*dz^2/chi_op)
dt_factor = 1.0
for a in sys.argv[1:]:
    if a.startswith("--dt-factor="):
        dt_factor = float(a.split("=", 1)[1])
diag_dir = args[0]
mode = args[1]
assert mode in ("sonic", "free_streaming"), f"unknown mode {mode}"
factor = float(args[2]) if len(args) > 2 else (2.5 if mode == "sonic" else 1.0)

# WarpX PhysConst / ablastr::constant values (the parser's m_p, q_e, ...)
proton_mass = 1.67262192595e-27
electron_mass = 9.1093837139e-31
qe = 1.602176634e-19
gamma = 5.0 / 3.0

# ---- deck constants (kept in lockstep with the inputs files) ---------
if not z_end:
    n0 = 1.0e18
    Te0_ev, Ti0_ev, Twall_ev = 100.0, 40.0, 2.0
    Lr, Lz, nr, nz = 0.1, 0.1, 8, 8
    dn = Lr / nr  # face-normal cell size
    chi_op = 700.0  # constant isotropic operator diffusivity, both channels
    dt = 4.0e-9
    nsteps = 4
    # the mask's conductor cells, exactly as ImplicitMHDWallMask builds it;
    # the interface is the grid face above the last interior cell
    r_centers = (np.arange(nr) + 0.5) * dn
    masked = (r_centers + 1.0e-3 * dn) >= 0.065
    n_cells = int(np.argmax(masked))
    assert 2 <= n_cells < nr, "degenerate wall mask layout"
    # cylindrical divergence weights of the radial faces (r = 0 drops out)
    def face_weight(k_face, k_cell):
        return (k_face * dn) / ((k_cell + 0.5) * dn)
    wall_faces = [(n_cells - 1, face_weight(n_cells, n_cells - 1))]
else:
    n0 = 1.0e18
    Te0_ev, Ti0_ev, Twall_ev = 100.0, 100.0, 2.0
    Lz, nz = 1.0, 64
    dn = Lz / nz
    chi_cap = 4.0e4  # defined (kappa/(n kB)) convention, both species
    chi_op = (gamma - 1.0) * chi_cap  # saturated-clamp operator diffusivity
    dt = dt_factor * 0.5 * dn**2 / chi_op
    nsteps = 8
    n_cells = nz
    def face_weight(k_face, k_cell):  # noqa: E306 (plain axial faces)
        return 1.0
    wall_faces = [(0, 1.0), (nz - 1, 1.0)]

rho0 = n0 * proton_mass
e_per_ev = {  # specific internal energy per eV, e = (q/m) T[eV]/(gamma - 1)
    "electron": (qe / proton_mass) / (gamma - 1.0),
    "ion": (qe / proton_mass) / (gamma - 1.0),
}
T0 = {"electron": Te0_ev, "ion": Ti0_ev}
species_mass = {"electron": electron_mass, "ion": proton_mass}
channels = ("electron", "ion")
e0 = {s: e_per_ev[s] * T0[s] for s in channels}
e_wall = {s: e_per_ev[s] * Twall_ev for s in channels}

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
    """Specific internal energies per channel along the column normal
    (averaged over the uniform tangential direction; the ion channel
    subtracts the advective kinetic energy exactly as the solver's
    recovery does)."""
    rho = state["implicit_mhd_mass_density"]
    kinetic = (
        0.5
        * sum(state[f"implicit_mhd_momentum_{c}"] ** 2 for c in ("r", "t", "z"))
        / rho
    )
    e_electron = state["implicit_mhd_electron_energy"] / rho
    e_ion = (state["implicit_mhd_ion_energy"] - kinetic) / rho
    axis = 1 if not z_end else 0  # average over z (shaped) or r (z-end)
    e_electron = np.mean(e_electron, axis=axis)
    e_ion = np.mean(e_ion, axis=axis)
    return {"electron": e_electron[:n_cells], "ion": e_ion[:n_cells]}


# ---- the model --------------------------------------------------------
def interior_operator():
    """Backward-Euler matrix of the interior column (all faces at chi_op
    with the geometry's divergence weights; wall faces excluded)."""
    rate = chi_op * dt / dn**2
    matrix = np.eye(n_cells)
    for k in range(n_cells):
        if k > 0:
            w = rate * face_weight(k, k)
            matrix[k, k - 1] -= w
            matrix[k, k] += w
        if k < n_cells - 1:
            w = rate * face_weight(k + 1, k)
            matrix[k, k + 1] -= w
            matrix[k, k] += w
    return matrix


def temperature_ev(e_spec, s):
    return e_spec / e_per_ev[s]


def cap_flux(s, t_face):
    """q_cap of channel s at the face temperatures t_face[species] (eV)."""
    n_kb_t = n0 * qe * t_face[s]
    if mode == "sonic":
        speed = np.sqrt(gamma * qe * (t_face["electron"] + t_face["ion"]) / proton_mass)
    else:
        speed = np.sqrt(qe * t_face[s] / species_mass[s])
    return factor * n_kb_t * speed


def wall_flux(s, e_cell, t_face, form):
    """Wall exchange of channel s [W/m^2] for a cell energy e_cell."""
    q_unl = chi_op * rho0 * (e_cell - e_wall[s]) * 2.0 / dn
    if form == "uncapped":
        return q_unl
    q_cap = cap_flux(s, t_face)
    if form == "hard_min":
        return min(q_unl, q_cap) if q_unl >= 0.0 else max(q_unl, -q_cap)
    return q_unl / (1.0 + abs(q_unl) / q_cap)


def first_step(form):
    """One backward-Euler step of both channels from the uniform column:
    the wall sinks (evaluated at the END-of-step wall-cell energies and
    the end-of-step reservoir-averaged face temperatures, coupling the
    channels through c_s) are iterated to a fixed point around the exact
    linear interior solve."""
    matrix = interior_operator()
    solve = np.linalg.solve
    e_new = {s: np.full(n_cells, e0[s]) for s in channels}
    for _ in range(200):
        previous = {s: e_new[s].copy() for s in channels}
        for s in channels:
            rhs = np.full(n_cells, e0[s])
            for cell, weight in wall_faces:
                t_face = {
                    sp: 0.5 * (temperature_ev(e_new[sp][cell], sp) + Twall_ev)
                    for sp in channels
                }
                q = wall_flux(s, e_new[s][cell], t_face, form)
                rhs[cell] -= dt * weight / dn * q / rho0
            e_new[s] = solve(matrix, rhs)
        change = max(
            np.max(np.abs(e_new[s] - previous[s])) / e0[s] for s in channels
        )
        if change < 1.0e-14:
            break
    else:
        raise RuntimeError("wall-sink fixed point did not converge")
    return e_new


plotfiles = sorted(glob.glob(f"{diag_dir}/diag??????"))
assert len(plotfiles) == nsteps + 1, f"need one snapshot per step, got {len(plotfiles)}"
states = [load_state(p) for p in plotfiles]
for step, state in enumerate(states):
    for name, field in state.items():
        assert np.isfinite(field).all(), f"{name} not finite at step {step}"
measured = [profiles(s) for s in states]

print(f"mode = {mode}, f = {factor}, geometry = {'z-end' if z_end else 'shaped wall'}")

# ---- the initial column is uniform ----------------------------------
for s in channels:
    assert np.all(np.abs(measured[0][s] / e0[s] - 1.0) < 1.0e-12), (
        f"{s} column not uniform at t = 0"
    )

# ---- the cap is BINDING but not saturating at the initial state -----
# (so the harmonic value sits well away from both limits and the gate
# below discriminates the form of the cap)
t_face0 = {s: 0.5 * (T0[s] + Twall_ev) for s in channels}
for s in channels:
    q_unl0 = wall_flux(s, e0[s], t_face0, "uncapped")
    q_cap0 = cap_flux(s, t_face0)
    print(
        f"  {s}: q_unl = {q_unl0:.4e} W/m^2, q_cap = {q_cap0:.4e} W/m^2, "
        f"q_unl/q_cap = {q_unl0 / q_cap0:.3f}, harmonic/q_cap = "
        f"{wall_flux(s, e0[s], t_face0, 'harmonic') / q_cap0:.3f}"
    )

# ---- FIRST-STEP drain per channel against the three column models ---
model = {form: first_step(form) for form in ("harmonic", "uncapped", "hard_min")}
wall_cells = [cell for cell, _ in wall_faces]
for s in channels:
    drops = {
        form: np.array([e0[s] - model[form][s][c] for c in wall_cells])
        for form in model
    }
    measured_drop = np.array([e0[s] - measured[1][s][c] for c in wall_cells])
    column_drop = np.sum(e0[s] - measured[1][s])
    column_model = np.sum(e0[s] - model["harmonic"][s])
    ring_ratio = measured_drop / drops["harmonic"]
    column_ratio = column_drop / column_model
    print(
        f"  {s} first step: wall-cell drop measured "
        f"{measured_drop / e_per_ev[s]} eV, harmonic model "
        f"{drops['harmonic'] / e_per_ev[s]} eV; ratio {ring_ratio}, "
        f"column ratio {column_ratio:.5f}"
    )
    print(
        f"    separation: measured/uncapped = {measured_drop / drops['uncapped']}, "
        f"measured/hard_min = {measured_drop / drops['hard_min']}"
    )
    assert np.all(np.abs(ring_ratio - 1.0) < 0.01), (
        f"{s} first-step wall-cell drain off the capped model: {ring_ratio}"
    )
    assert abs(column_ratio - 1.0) < 0.01, (
        f"{s} first-step column drain off the capped model: {column_ratio:.5f}"
    )
    # the form is discriminated: the harmonic value sits at least 2.5
    # percent (2.5x the gate above) from the uncapped and the hard-min
    # columns for every wall cell -- two-fold apart on the shaped-wall
    # deck, 3 percent apart on the z-end deck whose saturated chi_par
    # drives q_unl/q_cap to ~30
    assert np.all(np.abs(measured_drop / drops["uncapped"] - 1.0) > 0.025), (
        f"{s}: the cap does not bind ({measured_drop / drops['uncapped']})"
    )
    assert np.all(np.abs(measured_drop / drops["hard_min"] - 1.0) > 0.025), (
        f"{s}: indistinguishable from a hard min ({measured_drop / drops['hard_min']})"
    )

# ---- whole run: monotone drain that never undershoots the bath ------
for s in channels:
    for c in wall_cells:
        trace = np.array([m[s][c] for m in measured])
        assert np.all(np.diff(trace) < 0.0), (
            f"{s} wall cell {c} not draining monotonically: {trace / e_per_ev[s]}"
        )
        assert trace[-1] > e_wall[s], f"{s} wall cell {c} undershot the bath"

if not z_end:
    # ---- the masked band stayed a rigid, bit-static conductor --------
    conductor = np.broadcast_to(masked[:, None], states[0][FIELDS[0]].shape)
    for f in FIELDS:
        scale = np.max(np.abs(states[1][f]))
        scale = scale if scale > 0.0 else 1.0
        drift = (
            np.max(np.abs(states[-1][f] - states[1][f]), initial=0.0, where=conductor)
            / scale
        )
        assert drift < 1.0e-12, f"masked band changed ({f}: {drift:.3e})"

    # ---- the wall ledger closes against the interior TOTAL loss ------
    # rows of "step mass energy" in true kg/J. The ledger books everything
    # that crosses the interface -- the capped conductive drain AND the
    # mass/enthalpy the acoustic response advects into the wall -- so the
    # exact statement is against the interior's TOTAL fluid energy
    # (electron energy + ion total energy, kinetic included), volume
    # weighted; with theta = 1 the accepted-state booking matches it to
    # solver tolerance.
    volume = 2.0 * np.pi * r_centers[:n_cells] * dn * Lz

    def interior_total(state):
        total = np.mean(
            state["implicit_mhd_electron_energy"] + state["implicit_mhd_ion_energy"],
            axis=1,
        )[:n_cells]
        return float(np.sum(total * volume))

    def interior_mass(state):
        return float(
            np.sum(np.mean(state["implicit_mhd_mass_density"], axis=1)[:n_cells] * volume)
        )

    interior_loss = interior_total(states[0]) - interior_total(states[-1])
    mass_loss = interior_mass(states[0]) - interior_mass(states[-1])
    ledger_rows = np.loadtxt(f"{diag_dir}/wall_ledger.txt", ndmin=2)
    ledger_mass, ledger_energy = float(ledger_rows[-1, 1]), float(ledger_rows[-1, 2])
    closure = abs(ledger_energy - interior_loss) / abs(interior_loss)
    mass_closure = abs(ledger_mass - mass_loss) / max(abs(mass_loss), 1.0e-300)
    print(
        f"  ledger {ledger_energy:.6e} J vs interior total loss "
        f"{interior_loss:.6e} J, closure {closure:.3e}; mass closure {mass_closure:.3e}"
    )
    assert closure < 1.0e-5, f"wall ledger does not close ({closure:.3e})"
    assert mass_closure < 1.0e-5, f"wall mass ledger does not close ({mass_closure:.3e})"

newton_history = np.atleast_2d(np.loadtxt(f"{diag_dir}/newton.txt"))
assert 1 <= newton_history[-1][2] <= 20

print(f"wall_heat_flux_cap ({mode}, f = {factor}): all gates passed")
