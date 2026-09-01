#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Conductive z-end exchange (implicit_mhd.z_wall_conduction) for the
theta-implicit RZ MHD recast.

A hot static uniform column (T0 = 100 eV both species, B along z so
bn = 1 on the z end faces -> the PARALLEL Braginskii channel) sits
against cold wall_temperature ends (T_wall = 2 eV). The chi_par soft
cap is saturated (raw chi_par of both species stays >= 5x above it at
the coldest end-face temperature reached), so the effective diffusivity
is the state-independent constant (gamma - 1) chi_cap to machine
precision, and with theta = 1 each run's conduction subsystem is
EXACTLY a linear backward-Euler diffusion column. This analysis
integrates the identical discrete operator per mode and gates the
measured end-cell cooling against it, per channel:

mode = "off" (the null twin, implicit_mhd.z_wall_conduction=0 on the
CLI): the legacy end faces. The plain Braginskii tensor branch is LIVE
at the z domain boundary faces -- it differences the interior cell
against the wall_temperature ghost at the ONE-CELL Neumann-ghost
distance (flux chi rho (e - e_wall)/dz, empirically confirmed on this
branch 2026-08-31). The twin must match the one-cell-distance Dirichlet
column, which proves the knob's default preserves the legacy behavior
quantitatively.

mode = "on": the end faces REPLACE the plain branch with the hard
HALF-CELL Dirichlet exchange chi_nn rho (e - e_wall) 2/dz (the wall
value sits ON the face; no free-streaming cap): the run must match the
half-cell pin column -- a 2x faster end-face exchange whose end cell
lands well below the legacy twin's.

Usage: analysis_mhd_z_wall_conduction.py <diag_dir> off
       analysis_mhd_z_wall_conduction.py <diag_dir> on <off_diag_dir>
"""

import glob
import sys

import numpy as np
import yt

yt.set_log_level(50)

diag_dir = sys.argv[1]
mode = sys.argv[2]
assert mode in ("off", "on"), f"unknown mode {mode}"

proton_mass = 1.67262192595e-27  # WarpX PhysConst::m_p (ablastr/constant.H)
qe = 1.602176634e-19
n0 = 1.0e18
T0_ev = 100.0
Twall_ev = 2.0
gamma = 5.0 / 3.0
chi_cap = 4.0e4  # defined (kappa/(n kB)) convention
chi_op = (gamma - 1.0) * chi_cap  # operator diffusivity of the pinned cap
nz = 64
Lz = 1.0
dz = Lz / nz
dt = 0.5 * dz**2 / chi_op
nsteps = 8

# specific internal energies of the uniform column and the wall bath
# (the reservoir conversion of the ghost fill: e = (q/m) T[eV] /
# (gamma - 1); no temperature floors are set, so the bath is the
# reservoir exactly, and n0 kB T_wall sits far above the pressure
# floors, so the ghost image is unfloored)
e0 = (qe / proton_mass) * T0_ev / (gamma - 1.0)
e_wall = (qe / proton_mass) * Twall_ev / (gamma - 1.0)


def load_state(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    rho = np.squeeze(data["boxlib", "implicit_mhd_mass_density"].value)
    ue = np.squeeze(data["boxlib", "implicit_mhd_electron_energy"].value)
    ei = np.squeeze(data["boxlib", "implicit_mhd_ion_energy"].value)
    mom2 = sum(
        np.squeeze(data["boxlib", f"implicit_mhd_momentum_{c}"].value) ** 2
        for c in ("r", "t", "z")
    )
    # radially averaged specific internal energies (the column is
    # uniform in r; the ion channel subtracts the advective kinetic
    # energy exactly as the solver's recovery does)
    e_spec_e = np.mean(ue / rho, axis=0)
    e_spec_i = np.mean((ei - 0.5 * mom2 / rho) / rho, axis=0)
    return e_spec_e, e_spec_i


def column_model(end_face_factor):
    """Backward-Euler Dirichlet diffusion column at the run's exact
    discretization: interior faces chi_op/dz^2; the two end faces
    exchange against e_wall at end_face_factor * chi_op/dz^2 (1 = the
    legacy one-cell Neumann-ghost distance, 2 = the half-cell pin)."""
    rate = chi_op * dt / dz**2
    matrix = np.zeros((nz, nz))
    rhs_bath = np.zeros(nz)
    for k in range(nz):
        diagonal = 1.0
        if k > 0:
            matrix[k, k - 1] -= rate
            diagonal += rate
        if k < nz - 1:
            matrix[k, k + 1] -= rate
            diagonal += rate
        if k == 0 or k == nz - 1:
            diagonal += end_face_factor * rate
            rhs_bath[k] = end_face_factor * rate * e_wall
        matrix[k, k] += diagonal
    e_model = np.full(nz, e0)
    for _ in range(nsteps):
        e_model = np.linalg.solve(matrix, e_model + rhs_bath)
    return e_model


plotfiles = sorted(glob.glob(f"{diag_dir}/diag??????"))
assert len(plotfiles) >= 2, f"need initial+final snapshots, got {len(plotfiles)}"
ee_i, ei_i = load_state(plotfiles[0])
ee_f, ei_f = load_state(plotfiles[-1])
assert ee_i.shape == (nz,)

# initial state sanity: the uniform column
assert np.all(np.abs(ee_i / e0 - 1.0) < 1.0e-12)
assert np.all(np.abs(ei_i / e0 - 1.0) < 1.0e-12)

end_face_factor = 1.0 if mode == "off" else 2.0
e_model = column_model(end_face_factor)
model_drop_end = e0 - e_model[0]
model_drop_int = np.sum(e0 - e_model)
print(
    f"{mode} model (end-face factor {end_face_factor:.0f}): "
    f"T_end = {e_model[0] / e0 * T0_ev:.2f} eV, "
    f"end drop = {model_drop_end / e0:.4f} e0, "
    f"column drop = {model_drop_int / e0:.4f} e0"
)

for name, final in (("electron", ee_f), ("ion", ei_f)):
    measured_drop_end = e0 - 0.5 * (final[0] + final[-1])
    measured_drop_int = np.sum(e0 - final)
    end_ratio = measured_drop_end / model_drop_end
    int_ratio = measured_drop_int / model_drop_int
    print(
        f"{mode}/{name}: end-cell T = "
        f"{0.5 * (final[0] + final[-1]) / e0 * T0_ev:.2f} eV, "
        f"drop ratio (measured/model) end = {end_ratio:.4f}, "
        f"column = {int_ratio:.4f}"
    )
    # the run must land on ITS OWN Dirichlet column (the one-cell
    # legacy branch for "off", the half-cell pin for "on"); the band
    # absorbs the small acoustic response (c_s t / dz ~ 0.4) and the
    # coefficient-state temperature averaging
    assert 0.75 < end_ratio < 1.2, (
        f"{mode} {name} end-cell cooling off its Dirichlet-column "
        f"prediction: {end_ratio:.4f}"
    )
    assert 0.75 < int_ratio < 1.25, (
        f"{mode} {name} column-integrated drain off its prediction: "
        f"{int_ratio:.4f}"
    )
    # the midplane has not seen the front over this short run
    assert final[nz // 2] > 0.98 * e0

if mode == "on":
    # ---- knob discrimination against the null twin: the half-cell pin
    # is a strictly faster end-face exchange than the legacy one-cell
    # branch; the ON end cells must land materially below the OFF
    # twin's (model separation 13.6 eV; gate at half of it) ----
    off_dir = sys.argv[3]
    off_plotfiles = sorted(glob.glob(f"{off_dir}/diag??????"))
    assert len(off_plotfiles) >= 2
    ee_off, ei_off = load_state(off_plotfiles[-1])
    model_off = column_model(1.0)
    model_separation = model_off[0] - e_model[0]
    for name, on_final, off_final in (
        ("electron", ee_f, ee_off),
        ("ion", ei_f, ei_off),
    ):
        separation = 0.5 * (
            (off_final[0] - on_final[0]) + (off_final[-1] - on_final[-1])
        )
        print(
            f"on/{name}: end-cell separation off-minus-on = "
            f"{separation / e0 * T0_ev:.2f} eV "
            f"(model {model_separation / e0 * T0_ev:.2f} eV)"
        )
        assert separation > 0.5 * model_separation, (
            f"{name}: the pin did not outrun the legacy one-cell branch: "
            f"{separation / e0:.4f} e0 vs model "
            f"{model_separation / e0:.4f} e0"
        )
        # and the ON run drains MORE in total than the twin
        assert np.sum(e0 - on_final) > np.sum(e0 - off_final)

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert 1 <= newton_history[-1][2] <= 20

print(f"z_wall_conduction ({mode}): all gates passed")
