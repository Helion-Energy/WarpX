#!/usr/bin/env python3
"""Identity-gate reference run.

A small cylindrical hybrid case that exercises the same code path as a
production state: hybrid Ohm's-law solver, resistivity, a non-trivial
evolving B, one kinetic species.  There is deliberately NO external-field
register in play (the initial B is set through the grid-init style, which
writes the field state directly), so the field state IS the total field
and the remap has no total-vs-plasma-frame decision to get wrong.  That
keeps the identity gate measuring round-trip losses only.

Runs 2N steps, and at step N writes out the remap payload:
  * B at native staggering (.npy, from the field wrappers)
  * the particle list (openPMD, one iteration, one species)
"""

import argparse
import os

import numpy as np

from pywarpx import callbacks, fields, picmi

parser = argparse.ArgumentParser()
parser.add_argument("--nsplit", type=int, default=200)
parser.add_argument("--nsteps", type=int, default=400)
parser.add_argument("--outdir", default="payload")
parser.add_argument("--nr", type=int, default=64)
parser.add_argument("--nz", type=int, default=128)
parser.add_argument("--ppc", type=int, default=32)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

R_MAX, Z_MIN, Z_MAX = 0.10, -0.25, 0.25
N0, B0 = 1.0e19, 0.20
TE = 10.0

grid = picmi.CylindricalGrid(
    number_of_cells=[args.nr, args.nz],
    lower_bound=[0.0, Z_MIN],
    upper_bound=[R_MAX, Z_MAX],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    n_azimuthal_modes=1,
)

solver = picmi.HybridPICSolver(
    grid=grid, gamma=1.0, Te=TE, n0=N0,
    plasma_resistivity=1.0e-4, substeps=8,
)

# A structured, non-uniform profile so the field genuinely evolves and the
# comparison is not trivially satisfied by a static state.
dist = picmi.AnalyticDistribution(
    density_expression=f"{N0}*(1.0 + 0.6*exp(-((x*x+y*y)/(0.03*0.03))) "
                       f"* (1.0 + 0.3*sin(2*pi*z/0.5)))",
    rms_velocity=[2.0e5] * 3,
    directed_velocity=[0.0, 0.0, 0.0],
)
ions = picmi.Species(
    particle_type="proton", name="ions", initial_distribution=dist,
    warpx_reflection_model_xhi="0.0",
)

part_diag = picmi.ParticleDiagnostic(
    name="pdump", period=args.nsplit, species=[ions],
    data_list=["position", "momentum", "weighting"],
    write_dir=args.outdir, warpx_format="openpmd",
    warpx_openpmd_backend="h5",
)
fld_diag = picmi.FieldDiagnostic(
    name="fdump", grid=grid, period=args.nsplit,
    data_list=["B", "rho"],
    write_dir=args.outdir, warpx_format="openpmd",
    warpx_openpmd_backend="h5",
)

sim = picmi.Simulation(solver=solver, time_step_size=2.0e-11,
                       max_steps=args.nsteps, verbose=0)
sim.add_species(ions, layout=picmi.PseudoRandomLayout(
    n_macroparticles_per_cell=args.ppc, grid=grid))
sim.add_diagnostic(part_diag)
sim.add_diagnostic(fld_diag)
sim.initialize_inputs()

from pywarpx import amrex as amrex_bucket  # noqa: E402
from pywarpx import warpx as wx_bucket  # noqa: E402

amrex_bucket.the_arena_init_size = 2_000_000_000
wx_bucket.B_ext_grid_init_style = "constant"
wx_bucket.B_external_grid = [0.0, 0.0, B0]


def dump_b(tag):
    br = np.array(fields.BxFPWrapper(level=0)[...])
    bt = np.array(fields.ByFPWrapper(level=0)[...])
    bz = np.array(fields.BzFPWrapper(level=0)[...])
    np.savez(os.path.join(args.outdir, f"B_{tag}.npz"),
             br=br, bt=bt, bz=bz)
    print(f"[ref] wrote B_{tag}: br{br.shape} bt{bt.shape} bz{bz.shape}",
          flush=True)


def after_step():
    n = sim.extension.warpx.getistep(lev=0)
    if n == args.nsplit:
        dump_b("split")
    elif n == args.nsteps:
        dump_b("final")


callbacks.installafterstep(after_step)
sim.step(args.nsteps)
print("[ref] done", flush=True)
