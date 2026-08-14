#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""z-end fluid boundary "wall_temperature" for the theta-implicit RZ MHD
recast.

mode = "cold": a hot static uniform slab (T0 = 100 eV) sits against cold
end walls (T_wall = 2 eV). The end ghosts copy the interior density and
momentum but carry the wall temperature, so the end faces exchange
advectively against a T_wall reservoir: the boundary cells must cool
substantially toward T_wall, the total fluid energy must drain strictly
monotonically through the ends (the plain Neumann ends are a zero-flux
energy cork), the drained energy must be a physically sensible fraction
of the free-streaming enthalpy exchange, and the midplane must stay hot
over this short run (the cooling front travels at ~ the sound speed).

mode = "equilibrium" (CLI override my_constants.Twall_ev=100.0): a slab
AT the wall temperature. The ghost state then equals the interior to
roundoff, the residual vanishes, and every fluid moment must stay
unchanged to solver roundoff.

The ion temperature is reconstructed from the dumps exactly as intended
for production use: T_i = (gamma - 1)(E_i - |m|^2/(2 rho)) / (n q_e)
with the momentum from the implicit_mhd_momentum_{r,t,z} views.

Usage: analysis_mhd_z_wall_temperature.py <diag_dir> <mode>
"""

import glob
import sys

import numpy as np
import yt

diag_dir = sys.argv[1]
mode = sys.argv[2]
assert mode in ("cold", "equilibrium"), f"unknown mode {mode}"

proton_mass = 1.67262192369e-27
qe = 1.602176634e-19
n0 = 1.0e20
rho0 = n0 * proton_mass
T0_ev = 100.0
Twall_ev = 2.0 if mode == "cold" else T0_ev
gamma = 5.0 / 3.0


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


def load_state(plotfile):
    _, data = get_data(plotfile)
    rho = np.squeeze(data["boxlib", "implicit_mhd_mass_density"].value)
    ue = np.squeeze(data["boxlib", "implicit_mhd_electron_energy"].value)
    ei = np.squeeze(data["boxlib", "implicit_mhd_ion_energy"].value)
    mom2 = sum(
        np.squeeze(data["boxlib", f"implicit_mhd_momentum_{c}"].value) ** 2
        for c in ("r", "t", "z")
    )
    return rho, ue, ei, mom2


def temperatures_ev(rho, ue, ei, mom2):
    number_density = rho / proton_mass
    te = (gamma - 1.0) * ue / (number_density * qe)
    ti = (gamma - 1.0) * (ei - 0.5 * mom2 / rho) / (number_density * qe)
    return te, ti


plotfiles = sorted(glob.glob(f"{diag_dir}/diag??????"))
assert len(plotfiles) >= 4, f"need at least 4 snapshots, got {len(plotfiles)}"

ds0, _ = get_data(plotfiles[0])
nr, nz = int(ds0.domain_dimensions[0]), int(ds0.domain_dimensions[1])
r_edges = np.linspace(
    float(ds0.domain_left_edge[0]), float(ds0.domain_right_edge[0]), nr + 1
)
radial_weight = 0.5 * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
dz = (float(ds0.domain_right_edge[1]) - float(ds0.domain_left_edge[1])) / nz
final_time = float(yt.load(plotfiles[-1]).current_time)

energies = []
for plotfile in plotfiles:
    rho, ue, ei, mom2 = load_state(plotfile)
    energies.append(
        float(np.sum(radial_weight[:, None] * (ue + ei)) * 2.0 * np.pi * dz)
    )
energies = np.array(energies)
print("energy ratios:", energies / energies[0])

rho_i, ue_i, ei_i, mom2_i = load_state(plotfiles[0])
rho_f, ue_f, ei_f, mom2_f = load_state(plotfiles[-1])

if mode == "equilibrium":
    # ghost state == interior to roundoff: nothing may change beyond
    # solver roundoff (measured ~1e-16 relative)
    for name, initial, final, scale in (
        ("mass density", rho_i, rho_f, rho0),
        ("electron energy", ue_i, ue_f, np.max(ue_i)),
        ("ion energy", ei_i, ei_f, np.max(ei_i)),
    ):
        drift = np.max(np.abs(final - initial)) / scale
        print(f"equilibrium {name} drift = {drift:.3e}")
        assert drift < 1.0e-11, f"{name} drifted at T_wall = T_slab: {drift:.3e}"
    print("PASS")
    sys.exit(0)

# --- cold-wall mode ---
interval_change = np.diff(energies) / energies[0]
assert np.all(interval_change < 1.0e-12), (
    f"energy did not drain monotonically: {interval_change}"
)

te_f, ti_f = temperatures_ev(rho_f, ue_f, ei_f, mom2_f)
edge_te = 0.5 * (np.mean(te_f[:, 0]) + np.mean(te_f[:, -1]))
edge_ti = 0.5 * (np.mean(ti_f[:, 0]) + np.mean(ti_f[:, -1]))
mid_te = np.mean(te_f[:, nz // 2])
mid_ti = np.mean(ti_f[:, nz // 2])
print(
    f"final edge T_e/T_i = {edge_te:.2f}/{edge_ti:.2f} eV, "
    f"midplane T_e/T_i = {mid_te:.2f}/{mid_ti:.2f} eV"
)

# the boundary cells relax substantially toward T_wall = 2 eV
# (measured edge T_e/T_i = 57/62 eV after 120 steps of advective
# exchange; the plain Neumann cork would leave them at T0) ...
assert edge_te < 0.65 * T0_ev, "edge electrons did not cool toward T_wall"
assert edge_ti < 0.65 * T0_ev, "edge ions did not cool toward T_wall"
assert edge_te > 0.25 * Twall_ev and edge_ti > 0.25 * Twall_ev
# ... while the midplane has not yet seen the cooling front
assert mid_te > 0.9 * T0_ev and mid_ti > 0.9 * T0_ev

# rate consistency with the advective exchange: the drained energy is
# bounded by (and a sensible fraction of) the free-streaming enthalpy
# exchange 2 A [gamma/(gamma-1)] (p_e0 + p_i0) c_s t through both ends
area = np.pi * r_edges[-1] ** 2
p0 = n0 * T0_ev * qe
sound_speed = np.sqrt(gamma * 2.0 * p0 / rho0)
free_streaming = (
    2.0 * area * (gamma / (gamma - 1.0)) * 2.0 * p0 * sound_speed * final_time
)
drained = energies[0] - energies[-1]
fraction = drained / free_streaming
print(
    f"drained {drained:.3e} J = {fraction:.3f} of the free-streaming enthalpy exchange"
)
assert 0.02 < fraction < 1.0, (
    f"drain rate inconsistent with advective exchange: {fraction:.3e}"
)

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
assert 1 <= newton_history[-1][2] <= 12

print("PASS")
