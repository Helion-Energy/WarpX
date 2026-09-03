#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Shaped-wall CORNER conductance contract for the theta-implicit RZ MHD
recast (implicit_mhd.wall_thermal_bc = dirichlet + z_wall_conduction).

A static UNIFORM hot column (T0 = 100 eV) sits inside a stepped
conducting wall against a 2 eV Dirichlet bath, with the conductive
z-end exchange enabled against the same bath.  Only isotropic
constant-chi ELECTRON conduction is alive (eta = 0, no Hall, no
electron-pressure Ohm term, no Joule heating, frozen ions, u = 0), so
the electron specific internal energy obeys EXACTLY the discrete heat
equation this script integrates.

THE CONTRACT.  The half-cell Dirichlet exchange

    q_f = chi rho (e_interior - e_wall) * 2/dn_f

is the discrete statement "the wall surface passes this cell at the
half-cell distance dn_f/2".  A live cell can carry SEVERAL such faces:

  * a staircase corner of the shaped wall (masked in +r and in +z), and
  * a shaped-wall face on a cell that also sits on a z_wall_conduction
    domain end face (the EB-meets-z_hi corner),

and the two exchanges are emitted by independent branches that know
nothing about each other.  Adding them as independent drains gives the
cell the divergence contribution of a SCALAR-summed wall area
S_r + S_z, i.e. a single exchange at the distance dn/(2M) for M equal
faces -- the wall is placed M times closer to the cell centre than any
wall can be, and the cell's wall diffusion number is inflated by M.
The staircase faces are the discrete image of ONE wall surface, whose
true area is the VECTOR sum of the face-area elements (the divergence
theorem: a closed surface's area vectors sum to zero, so the staircase
and the segment it represents share their vector area, while the
staircase inflates the scalar area).  The corner therefore has to
carry the geometrically consistent conductance

    w = sqrt(S_r^2 + S_z^2) / (S_r + S_z)   applied to every wall face
                                            of that cell,

which is exactly 1 -- bit-identical -- on every flat wall, and strictly
STRONGER than a flat wall at a corner (no capping, no bottling: the
corner still drains sqrt(2)x a flat face).

GATES.  Flat single-face reference cells sit at the SAME radius as each
corner, so the measured corner/flat drain ratios cancel every radius-
and metric-dependent factor:

    stair corner (i = 18, j = 7)  vs  r-face flat (i = 18, j = 4)
    z-end corner (i = 13, j = 15) vs  r-face flat (i = 13, j = 10)

Each ratio must match the vector-area prediction and must sit well
below the unguarded scalar-sum prediction.  The whole live field is
also gated against the exact discrete theta-staged operator with the
vector-area corner rule.

MONOTONICITY mode (theta = 1/2, larger step).  The theta update of a
cell whose wall exchange carries the diffusion number D is
e1 = bath + A (e0 - bath) with A = (1 - (1 - th) D)/(1 + th D), which
is monotone only while D <= 1/(1 - th).  Inflating a corner cell's D
by its wall-face count is exactly what carries it past that bound: at
chi dt/dz^2 = 0.6 a flat wall face (D = 1.2) is monotone in both
contracts, while the unguarded corner sum (D = 2.4) drives a live cell
BELOW its own 2 eV reservoir -- the seed of the negative corner
energies seen on the production formation arms.  A two-sided Dirichlet
exchange may never undershoot the bath it exchanges against, and the
vector-area contract keeps every live cell above it.

Usage: analysis_mhd_wall_corner_drain.py <diag_dir> [theta] [chi dt/dz^2]
                                         [nsteps]
"""

import glob
import sys

import numpy as np
import yt

yt.set_log_level(50)

diag_dir = sys.argv[1]
theta = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
diffusion_number = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
nsteps = int(sys.argv[4]) if len(sys.argv) > 4 else 2
# the ratio discriminator needs the LINEAR (small-D) regime; the
# monotonicity discriminator needs a step past the corner's theta bound
ratio_gates = theta == 1.0 and diffusion_number < 0.1

proton_mass = 1.67262192595e-27  # WarpX PhysConst::m_p (ablastr/constant.H)
qe = 1.602176634e-19
n0 = 1.0e20
rho0 = n0 * proton_mass
T0_ev = 100.0
Twall_ev = 2.0
gamma = 5.0 / 3.0
chi = 20.0
nr = 32
nz = 16
Lr = 0.5
Lz = 0.25
dr = Lr / nr
dz = Lz / nz
z_lo = -0.5 * Lz
dt = diffusion_number * dz**2 / chi

# specific internal energies (density-free reservoir conversion of the
# wall fill, e = (q/m) T[eV]/(gamma - 1); no temperature floors are set,
# so the bath is the reservoir exactly)
e0 = (qe / proton_mass) * T0_ev / (gamma - 1.0)
e_wall = (qe / proton_mass) * Twall_ev / (gamma - 1.0)

# ---------------------------------------------------------------- mask
# ImplicitMHDWallMask: first masked CELL-CENTERED radial index at the
# cell-centre axial position, r_w from the stepped polyline
# (wall_step_r30_r22.csv), tol_r = 1e-3 dr.
tol_r = 1.0e-3 * dr


def wall_radius(z):
    return 0.30 if z < 0.0 else 0.22


def first_masked(j):
    z = z_lo + (j + 0.5) * dz
    rw = wall_radius(z) - tol_r
    return int(np.ceil(rw / dr - 0.5 - 1.0e-12))


wall_fm = {j: first_masked(j) for j in range(-1, nz + 1)}

# the two corner cells and their flat single-face twins, at matched radii
J_STEP = next(j for j in range(1, nz) if wall_fm[j] < wall_fm[j - 1])
I_STAIR = wall_fm[J_STEP - 1] - 1
STAIR_CORNER = (I_STAIR, J_STEP - 1)
STAIR_FLAT = (I_STAIR, J_STEP - 4)
I_ZEND = wall_fm[nz - 1] - 1
ZEND_CORNER = (I_ZEND, nz - 1)
ZEND_FLAT = (I_ZEND, nz - 6)

assert wall_fm[0] == 19 and wall_fm[nz - 1] == 14, (
    f"unexpected wall mask: fm[0] = {wall_fm[0]}, fm[nz-1] = {wall_fm[nz - 1]}"
)
assert STAIR_CORNER == (18, 7) and ZEND_CORNER == (13, 15), (
    f"unexpected corner cells {STAIR_CORNER}, {ZEND_CORNER}"
)


def live(i, j):
    return 0 <= i < nr and 0 <= j < nz and i < wall_fm[j]


def wall_faces(i, j):
    """(S_r, S_z): per-axis summed wall-face area/volume elements of the
    live cell (i, j).  r faces carry the RZ area factor r_face/(r_cell
    dr); z faces carry 1/dz.  The +/-z faces include the
    z_wall_conduction domain end faces."""
    s_r = 0.0
    if i + 1 >= wall_fm[j]:
        s_r += (i + 1.0) / ((i + 0.5) * dr)
    s_z = 0.0
    if j + 1 > nz - 1 or i >= wall_fm[j + 1]:  # z_hi end face or +z ledge
        s_z += 1.0 / dz
    if j - 1 < 0 or i >= wall_fm[j - 1]:  # z_lo end face or -z ledge
        s_z += 1.0 / dz
    return s_r, s_z


def corner_weight(i, j, rule):
    s_r, s_z = wall_faces(i, j)
    total = s_r + s_z
    if total <= 0.0 or rule == "sum":
        return 1.0
    return np.hypot(s_r, s_z) / total


# --------------------------------------------------- discrete operator
index = {}
for j in range(nz):
    for i in range(nr):
        if live(i, j):
            index[(i, j)] = len(index)
ncell = len(index)


def build_operator(rule):
    """L, b such that de/dt = L e + b: exactly the solver's conductive
    divergence -div(F)/rho for the frozen uniform-density column, with
    F the centred flux on interior faces and the half-cell Dirichlet
    exchange (scaled by the corner weight) on wall / z-end faces."""
    matrix = np.zeros((ncell, ncell))
    source = np.zeros(ncell)
    weight = {cell: corner_weight(*cell, rule) for cell in index}

    for (i, j), row in index.items():
        r_cell = (i + 0.5) * dr
        w = weight[(i, j)]
        # ---- radial faces (area factor r_face/(r_cell dr)) ----
        for face, neighbor in ((i + 1, (i + 1, j)), (i, (i - 1, j))):
            area = face * dr / (r_cell * dr)  # r_face/(r_cell dr); 0 on axis
            if area == 0.0:
                continue
            if live(*neighbor):
                matrix[row, row] -= area * chi / dr
                matrix[row, index[neighbor]] += area * chi / dr
            else:
                matrix[row, row] -= area * 2.0 * chi * w / dr
                source[row] += area * 2.0 * chi * w * e_wall / dr
        # ---- axial faces (area factor 1/dz) ----
        for neighbor in ((i, j + 1), (i, j - 1)):
            if live(*neighbor):
                matrix[row, row] -= chi / dz**2
                matrix[row, index[neighbor]] += chi / dz**2
            else:
                matrix[row, row] -= 2.0 * chi * w / dz**2
                source[row] += 2.0 * chi * w * e_wall / dz**2
    return matrix, source


def integrate(rule):
    """Per-step fields, one per snapshot the run writes."""
    matrix, source = build_operator(rule)
    implicit = np.eye(ncell) - theta * dt * matrix
    explicit = np.eye(ncell) + (1.0 - theta) * dt * matrix
    state = np.full(ncell, e0)
    history = []
    for _ in range(nsteps):
        state = np.linalg.solve(implicit, explicit @ state + dt * source)
        field = np.full((nr, nz), np.nan)
        for cell, row in index.items():
            field[cell] = state[row]
        history.append(field)
    return history


# --------------------------------------------------------------- data
plotfiles = sorted(glob.glob(f"{diag_dir}/diag??????"))
assert len(plotfiles) >= 2, f"need initial+final snapshots, got {len(plotfiles)}"


def load(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    rho = np.squeeze(data["boxlib", "implicit_mhd_mass_density"].value)
    ue = np.squeeze(data["boxlib", "implicit_mhd_electron_energy"].value)
    return ue / rho


e_initial = load(plotfiles[0])
e_final = load(plotfiles[-1])
assert e_final.shape == (nr, nz)

# the column boots uniform on every live cell
for i, j in index:
    assert abs(e_initial[i, j] / e0 - 1.0) < 1.0e-12, (
        f"live cell ({i},{j}) did not boot uniform"
    )

history_vector = integrate("vector")
history_sum = integrate("sum")
model_vector = history_vector[-1]
model_sum = history_sum[-1]

# ------------------------------------------------------------- gate 1
# corner / flat drain ratios at matched radius (linear-regime mode only:
# at the monotonicity step the ratios saturate and stop discriminating)
scale = e0 - e_wall
for label, corner, flat in (
    ("stair", STAIR_CORNER, STAIR_FLAT),
    ("z-end", ZEND_CORNER, ZEND_FLAT),
):
    assert wall_faces(*corner)[0] > 0.0 and wall_faces(*corner)[1] > 0.0, (
        f"{label} corner {corner} is not a two-axis wall cell"
    )
    assert wall_faces(*flat)[0] > 0.0 and wall_faces(*flat)[1] == 0.0, (
        f"{label} flat twin {flat} is not a single r-face wall cell"
    )
    measured = (e0 - e_final[corner]) / (e0 - e_final[flat])
    predicted = (e0 - model_vector[corner]) / (e0 - model_vector[flat])
    unguarded = (e0 - model_sum[corner]) / (e0 - model_sum[flat])
    print(
        f"{label} corner {corner}: drain ratio vs flat {flat} = "
        f"{measured:.4f} (vector-area contract {predicted:.4f}, "
        f"unguarded double drain {unguarded:.4f})"
    )
    if not ratio_gates:
        continue
    assert unguarded - predicted > 0.3, (
        "the two contracts are not separated enough to discriminate: "
        f"{unguarded:.4f} vs {predicted:.4f}"
    )
    assert abs(measured - predicted) < 1.0e-3, (
        f"{label} corner drain ratio {measured:.4f} is off the "
        f"vector-area contract {predicted:.4f} (the unguarded sum of two "
        f"independent half-cell Dirichlet exchanges predicts "
        f"{unguarded:.4f})"
    )

# ------------------------------------------------------------- gate 2
# the whole live field on the exact discrete operator with the corner rule
error_vector = 0.0
error_sum = 0.0
for cell in index:
    error_vector = max(error_vector, abs(e_final[cell] - model_vector[cell]))
    error_sum = max(error_sum, abs(e_final[cell] - model_sum[cell]))
print(
    f"max |measured - model| / (e0 - e_wall): vector-area contract "
    f"{error_vector / scale:.3e}, unguarded double drain "
    f"{error_sum / scale:.3e}"
)
assert error_vector / scale < 1.0e-8, (
    f"the live field is off its discrete conduction column: {error_vector / scale:.3e}"
)

# ------------------------------------------------------------- gate 3
# a two-sided Dirichlet exchange may never undershoot its own bath: the
# theta update is monotone only while the cell's wall diffusion number
# stays under 1/(1 - theta), and the unguarded corner sum is what
# carries it past that bound
# the theta exchange oscillates rather than overshooting once, so the
# undershoot is scanned over EVERY snapshot, not only the last
measured_history = [load(plotfile) for plotfile in plotfiles[1:]]
assert len(measured_history) == nsteps, (
    f"expected {nsteps} evolved snapshots, got {len(measured_history)}"
)
worst_measured = min(
    (field[cell], cell, step)
    for step, field in enumerate(measured_history, start=1)
    for cell in index
)
worst_model_sum = min(field[cell] for field in history_sum for cell in index)
print(
    f"coldest live cell over the run: {worst_measured[0] / e0 * T0_ev:.4f} eV "
    f"at {worst_measured[1]} step {worst_measured[2]} "
    f"(bath {Twall_ev:.2f} eV; the unguarded double drain reaches "
    f"{worst_model_sum / e0 * T0_ev:.4f} eV)"
)
if not ratio_gates:
    # the step must actually exercise the monotonicity bound, or this
    # mode proves nothing
    assert worst_model_sum < e_wall, (
        "this step does not push the unguarded corner sum past its "
        f"theta monotonicity bound ({worst_model_sum / e0 * T0_ev:.4f} eV "
        f"vs bath {Twall_ev:.2f} eV): not a discriminator"
    )
assert worst_measured[0] >= e_wall - 1.0e-9 * scale, (
    f"live cell {worst_measured[1]} undershot the wall bath at step "
    f"{worst_measured[2]}: {worst_measured[0] / e0 * T0_ev:.4f} eV"
)

# ------------------------------------------------------------- gate 4
# the corner still drains STRICTLY HARDER than a flat face (the contract
# is a geometric reconciliation, not a cap)
for label, corner, flat in (
    ("stair", STAIR_CORNER, STAIR_FLAT),
    ("z-end", ZEND_CORNER, ZEND_FLAT),
):
    assert (e0 - e_final[corner]) > (1.2 if ratio_gates else 1.0) * (
        e0 - e_final[flat]
    ), (
        f"{label} corner stopped outrunning its flat twin -- the corner "
        f"contract must not bottle the wall heat flux"
    )

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert 1 <= newton_history[-1][2] <= 20

print("wall corner drain: all gates passed")
