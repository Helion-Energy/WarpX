#!/usr/bin/env python3
"""Identity-gate continuation run.

Loads a remapped state -- particles from the openPMD dump, B written
directly into the field state -- and continues.  With refinement ratio 1
the prolongation is the identity, so any difference from the
uninterrupted reference is a round-trip loss, not a resolution effect.
That is exactly what this gate isolates.
"""

import argparse
import os

import numpy as np

from pywarpx import callbacks, fields, picmi

parser = argparse.ArgumentParser()
parser.add_argument("--particles", required=True)
parser.add_argument("--bfield", required=True)
parser.add_argument("--nsteps", type=int, default=200)
parser.add_argument("--outdir", default="cont")
parser.add_argument("--nr", type=int, default=64)
parser.add_argument("--nz", type=int, default=128)
parser.add_argument("--ratio", type=int, default=1)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

R_MAX, Z_MIN, Z_MAX = 0.10, -0.25, 0.25
N0, B0, TE = 1.0e19, 0.20, 10.0

nr_f = args.nr * args.ratio
grid = picmi.CylindricalGrid(
    number_of_cells=[nr_f, args.nz],
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
ions = picmi.Species(
    particle_type="proton", name="ions",
    initial_distribution=picmi.FromFileDistribution(file_path=args.particles),
    warpx_reflection_model_xhi="0.0",
)
fld_diag = picmi.FieldDiagnostic(
    name="fdump", grid=grid, period=args.nsteps,
    data_list=["B", "rho"], write_dir=args.outdir,
    warpx_format="openpmd", warpx_openpmd_backend="h5",
)
sim = picmi.Simulation(solver=solver, time_step_size=2.0e-11,
                       max_steps=args.nsteps, verbose=0)
sim.add_species(ions, layout=None)
sim.add_diagnostic(fld_diag)
sim.initialize_inputs()

from pywarpx import amrex as amrex_bucket  # noqa: E402
from pywarpx import warpx as wx_bucket  # noqa: E402

amrex_bucket.the_arena_init_size = 2_000_000_000
# Grid-init style writes the FIELD STATE directly (not the hybrid external
# register), so no external field is added at first step and the array we
# write below is unambiguously the total field.
wx_bucket.B_ext_grid_init_style = "constant"
wx_bucket.B_external_grid = [0.0, 0.0, B0]

_payload = np.load(args.bfield)
_applied = {"done": False}


def apply_b():
    """Overwrite the field state with the remapped B, after init."""
    if _applied["done"]:
        return
    br, bt, bz = _payload["br"], _payload["bt"], _payload["bz"]
    wr = fields.BxFPWrapper(level=0)
    wt = fields.ByFPWrapper(level=0)
    wz = fields.BzFPWrapper(level=0)
    for w, a, nm in ((wr, br, "Br"), (wt, bt, "Bt"), (wz, bz, "Bz")):
        cur = np.array(w[...])
        if cur.shape != a.shape:
            msg = f"{nm}: state {cur.shape} vs payload {a.shape}"
            raise RuntimeError(msg)
        w[...] = a
    _applied["done"] = True
    print(f"[cont] applied B: br{br.shape} bt{bt.shape} bz{bz.shape}",
          flush=True)


def dump_final():
    if sim.extension.warpx.getistep(lev=0) != args.nsteps:
        return
    br = np.array(fields.BxFPWrapper(level=0)[...])
    bt = np.array(fields.ByFPWrapper(level=0)[...])
    bz = np.array(fields.BzFPWrapper(level=0)[...])
    np.savez(os.path.join(args.outdir, "B_final.npz"), br=br, bt=bt, bz=bz)
    print("[cont] wrote B_final", flush=True)


callbacks.installafterinit(apply_b)
callbacks.installafterstep(dump_final)
sim.step(args.nsteps)
print("[cont] done", flush=True)
