#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Temperature-primary nodal electron closure of the theta-implicit MHD solver.

The nodal electron pressure and temperature must be derived
TEMPERATURE-first: Te_cell = p_cell/(n_cell_f kB) natively on the cell
grid (the flux kernels' floored pressure recovery over the Ohm-floored
density), Te_nodal = Interp(Te_cell), and Pe_nodal = n_nodal_f kB
Te_nodal rebuilt co-located with the SAME floored discrete density the
Ohm grad-Pe/(e n) denominator uses. The pressure-primary alternative
(interpolate the n T product, divide by the interpolated density) is
wrong at density gradients -- Interp(n T) != Interp(n) Interp(T) -- and
its floor conventions do not co-locate, so the isothermal limit picks
up spurious electric fields exactly where n has structure.

The run carries a density CLIFF in z (4 decades, crossing the Ohm
n_floor mid-slope) with UNIFORM electron temperature Te0 (the pressure
parser builds p = max(n, n_floor) kB Te0, so Te is Te0 through the
sub-floor band too). Asserts:

* after initialization (FillFluidSources runs in Define):
  - Te_nodal == Te0 at every node, THROUGH the cliff and the floor
    band (pressure-primary leaves O(10%) errors at the band-edge
    nodes);
  - Pe_nodal == max(n_nodal, n_floor) kB Te0 (the floored co-location;
    pressure-primary gives Interp of the cell-floored product, which
    differs at band-edge nodes: mean-of-max != max-of-mean);
  - Pe == n_f kB Te holds in EVERY ghost ring too (the ghosts are
    rebuilt from the mirrored Te and n images, not mirrored directly).
* after one step with B = 0, u = 0, eta = 0 and the Ohm electron
  pressure term ON, the solved E_z must satisfy, node-exact THROUGH the
  floor band,
      E_z = -grad_z Pe / (e n_f)  ==  -kB Te0 grad_z(n_f) / (e n_f)
  in the exact discrete form of the Ohm stencil (UpwardDz of the nodal
  Pe over the edge-interpolated, then floored, density).
"""

import numpy as np

from pywarpx import picmi

constants = picmi.constants

# thermodynamics: uniform Te0, 4-decade density cliff crossing n_floor
Te0_ev = 50.0
Te0_K = constants.q_e * Te0_ev / constants.kb
n_hi = 1.0e20
n_lo = 1.0e15
n_floor = 1.0e18
gamma = 5.0 / 3.0

nr = 8
nz = 64
rmax = 0.1
zmax = 0.5
dz = 2.0 * zmax / nz

# cliff over ~a cell so band-edge nodes (one cell above the floor, one
# below) exist; centered off a node so the profile is generic
cliff_w = dz
n_expr = f"({n_lo} + {n_hi - n_lo}*0.5*(1 - tanh(z/{cliff_w})))"
rho_expr = f"{constants.m_p}*{n_expr}"
# p_e = max(n, n_floor) kB Te0: uniform Te0 through the sub-floor band
pe_expr = f"{constants.q_e * Te0_ev}*max({n_expr}, {n_floor})"

rho_hi = n_hi * constants.m_p
sound_speed = np.sqrt(gamma * n_hi * constants.q_e * Te0_ev / rho_hi)
# tiny step: the state barely moves, so the post-step isothermal
# reduction against the INITIAL Te0 stays sharp
dt = 1.0e-5 * dz / sound_speed

grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz],
    warpx_max_grid_size=64,
    lower_bound=[0.0, -zmax],
    upper_bound=[rmax, zmax],
    lower_boundary_conditions=["none", "none"],
    upper_boundary_conditions=["dirichlet", "none"],
    lower_boundary_conditions_particles=["none", "absorbing"],
    upper_boundary_conditions_particles=["reflecting", "absorbing"],
)

solver = picmi.HybridPICSolver(
    grid=grid,
    Te=Te0_ev,
    n0=n_hi,
    gamma=gamma,
    n_floor=n_floor,
    plasma_resistivity=0.0,
    include_hall_term=False,
    include_electron_pressure_term=True,
)

nonlinear_solver = picmi.NewtonNonlinearSolver(
    verbose=True,
    max_iterations=20,
    relative_tolerance=1.0e-10,
    absolute_tolerance=1.0e-12,
)

evolve_scheme = picmi.ThetaImplicitMHDEvolveScheme(
    nonlinear_solver=nonlinear_solver,
    theta=1.0,
    mass_density=rho_expr,
    electron_pressure=pe_expr,
    reference_mass_density=rho_hi,
    reference_magnetic_field=0.05,
    gamma_e=gamma,
    gamma_i=gamma,
    reference_ion_pressure=0.0,
    mass_density_floor=1.0e-4 * n_lo * constants.m_p,
    electron_pressure_floor=1.0e-3,
    fluid_flux="hllc",
)

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=1,
    verbose=1,
)
sim.evolve_scheme = evolve_scheme

sim.initialize_inputs()
sim.initialize_warpx()


def full_array(name, dir=None):
    """Return the squeezed (r, z) array of a register field with ghosts."""
    if dir is None:
        mf = sim.fields.get(name, level=0)
    else:
        mf = sim.fields.get(name, dir, level=0)
    ng = mf.n_grow_vect
    arr = np.squeeze(np.array(mf[()]))
    if arr.ndim == 3:
        # multi-component register field (e.g. rho_fp carries old/new
        # slices); the solver fills and reads component 0
        arr = arr[..., 0]
    return arr, (ng[0], ng[1])


def valid(arr, ng):
    return arr[ng[0] : arr.shape[0] - ng[0], ng[1] : arr.shape[1] - ng[1]]


q_e = constants.q_e
kb = constants.kb
rho_q_floor = q_e * n_floor

# ---- initialization-time derivation (FillFluidSources ran in Define) ----
te, ng_te = full_array("hybrid_electron_temperature_fp")
pe, ng_pe = full_array("hybrid_electron_pressure_fp")
rho_q, ng_rho = full_array("rho_fp")

te_v = valid(te, ng_te)
pe_v = valid(pe, ng_pe)
rho_v = valid(rho_q, ng_rho)

# the cliff must genuinely cross the floor (discriminating test)
n_nodal = rho_v / q_e
assert n_nodal.min() < 0.1 * n_floor and n_nodal.max() > 10.0 * n_floor, (
    "density cliff does not cross the Ohm n_floor; test misconfigured"
)

# (1) uniform Te0 at every node, THROUGH the cliff and the floor band
np.testing.assert_allclose(
    te_v,
    Te0_K,
    rtol=1.0e-12,
    err_msg="nodal Te is not the uniform cell ratio (pressure-primary residue)",
)

# (2) Pe co-located with the floored nodal density
pe_expected = np.maximum(rho_v, rho_q_floor) / q_e * kb * Te0_K
np.testing.assert_allclose(
    pe_v,
    pe_expected,
    rtol=1.0e-12,
    err_msg="nodal Pe is not n_f kB Te0 (floored co-location broken)",
)

# (3) Pe = n_f kB Te holds INCLUDING every ghost ring (rebuilt ghosts)
pe_identity = np.maximum(rho_q, rho_q_floor) / q_e * kb * te
np.testing.assert_allclose(
    pe, pe_identity, rtol=1.0e-14, err_msg="Pe = n_f kB Te broken in the ghosts"
)

# ---- one implicit step: Ohm E_z from grad Pe/(e n) ----------------------
sim.step(1)

te, _ = full_array("hybrid_electron_temperature_fp")
pe, _ = full_array("hybrid_electron_pressure_fp")
rho_q, _ = full_array("rho_fp")
ez, ng_ez = full_array("Efield_fp", "z")

te_v = valid(te, ng_te)
pe_v = valid(pe, ng_pe)
rho_v = valid(rho_q, ng_rho)
ez_v = valid(ez, ng_ez)

# post-step identity still exact everywhere (every residual evaluation
# refills the sources)
pe_identity = np.maximum(rho_q, rho_q_floor) / q_e * kb * te
np.testing.assert_allclose(
    pe, pe_identity, rtol=1.0e-14, err_msg="post-step Pe = n_f kB Te broken"
)

# exact discrete Ohm stencil: Ez lives at (r-node, z-face);
# grad_Pe = UpwardDz(Pe), denominator = floored edge-interpolated density.
# The r_max wall node is excluded: the PEC (dirichlet) boundary zeroes
# the tangential E there by construction.
grad_pe = (pe_v[:-1, 1:] - pe_v[:-1, :-1]) / dz
rho_edge = np.maximum(0.5 * (rho_v[:-1, 1:] + rho_v[:-1, :-1]), rho_q_floor)
ez_ohm = -grad_pe / rho_edge
ez_v = ez_v[:-1]

scale = np.abs(ez_ohm).max()
assert scale > 1.0e3, "no grad-Pe field at the cliff; test misconfigured"
print(f"max |E_z| = {scale:.6e} V/m")
print(f"Ohm-stencil deviation  = {np.abs(ez_v - ez_ohm).max() / scale:.3e} (of max)")

# (4) the solved E_z is the Ohm expression, node-exact (solver-tolerance)
np.testing.assert_allclose(
    ez_v,
    ez_ohm,
    atol=1.0e-8 * scale,
    rtol=0.0,
    err_msg="solved E_z is not grad Pe/(e n_f) at the nodes",
)

# (5) isothermal reduction with the INITIAL Te0, node-exact THROUGH the
# floor band: E_z = -kB Te0 grad_z(n_f)/(e n_f) discretely. The floored
# co-location makes this exact; the pressure-primary derivation leaves
# O(10%) violations at the band-edge nodes. (The tiny step moves the
# state by O(1e-5) relative, hence the tolerance.)
n_f = np.maximum(rho_v[:-1], rho_q_floor) / q_e
ez_isothermal = -kb * Te0_K * (n_f[:, 1:] - n_f[:, :-1]) / dz / rho_edge
print(
    f"isothermal deviation   = {np.abs(ez_v - ez_isothermal).max() / scale:.3e} (of max)"
)
np.testing.assert_allclose(
    ez_v,
    ez_isothermal,
    atol=1.0e-4 * scale,
    rtol=0.0,
    err_msg="E_z != kB Te0 grad(n_f)/(e n_f) through the floor band",
)

print("PASS")
