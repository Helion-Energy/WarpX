#!/usr/bin/env python3
"""Stage-resolved probe of the fine-seam div(B) injection on the REAL
liftoff MR config (divb_seam_diagnosis_notes.md, step 3).

Runs liftoff_mr's setup at small N for one step and measures, at every
available hook (after init, per-stage afterEpush/afterBpush, beforestep,
afterstep), the max |div(B)| of the fine level split by in-plane distance
to the patch boundary (layer 0 = outermost valid fine-cell ring).

Also computes the STATIC H2 discriminator right after init:
div_h(hybrid_B_fp_external[lev=1]) layer-by-layer. If that is nonzero on
layer 0, the once-per-step external subtract/add pair injects
(s_new - s_old) * div_h(curlA) there — no stepping needed to convict.

Usage:
  PYTHONPATH=<build>/lib/site-packages python divb_seam_stage_probe.py \
      [--resolution 48] [--steps 1] [--emf-matching 1]
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "liftoff"))

probe_parser = argparse.ArgumentParser()
probe_parser.add_argument("--resolution", type=int, default=48)
probe_parser.add_argument("--steps", type=int, default=1)
probe_parser.add_argument("--substeps", type=int, default=None)
probe_parser.add_argument("--emf-matching", type=int, default=1)
probe_parser.add_argument("--tau-ramp", type=float, default=20.0e-6)
probe_parser.add_argument("--nppc", type=int, default=4)
probe_parser.add_argument("--patch-half", type=float, default=None)
probe_parser.add_argument("--max-grid-size-xy", type=int, default=24)
probe_args = probe_parser.parse_args()
sys.argv = sys.argv[:1]

import liftoff_mr as lm  # noqa: E402

import pywarpx  # noqa: E402
from pywarpx import callbacks  # noqa: E402

args = argparse.Namespace(
    test=False,
    resolution=probe_args.resolution,
    nppc=probe_args.nppc,
    steps=probe_args.steps,
    diag_steps=probe_args.steps,
    refined_core=True,
    patch_half=(
        probe_args.patch_half
        if probe_args.patch_half is not None
        else lm.PATCH_HALF_TARGET
    ),
    no_divb_audit=False,
    mr_emf_matching=probe_args.emf_matching,
    substeps=probe_args.substeps,
    pin_numerics_n=96,
    eta_vac_frac=lm.ETA_VAC_FRAC,
    tau_ramp=probe_args.tau_ramp,
    seed=lm.SEED,
    holmstrom=True,
    max_grid_size_xy=probe_args.max_grid_size_xy,
    verbose=0,
)

sim, info = lm.setup_simulation(args)
print(
    "probe: N=%d substeps=%d dt=%.4e patch_half=%.4f (%d coarse cells)"
    % (
        args.resolution,
        info["substeps"],
        info["dt"],
        info["patch_half"],
        info["patch_cells"],
    ),
    flush=True,
)

sim.initialize_inputs()
pywarpx.warpx.random_seed = args.seed
pywarpx.hybridpicmodel.mr_check_div_b = 1
pywarpx.warpx.refine_plasma_init = 1
pywarpx.hybridpicmodel.mr_emf_matching = args.mr_emf_matching
sim.initialize_warpx()

from pywarpx._libwarpx import libwarpx  # noqa: E402

warpx = libwarpx.warpx
geom1 = warpx.Geom(1)
dx1 = np.array([geom1.data().CellSize(d) for d in range(3)])
IAMROOT = libwarpx.amr.ParallelDescriptor.IOProcessor()


def rprint(*a, **kw):
    if IAMROOT:
        print(*a, **kw)


def get_mf(name, idir, lev):
    reg = warpx.multifab_register()
    mf = reg.get(name, dir=libwarpx.libwarpx_so.Direction(idir), level=lev)
    mf.level = lev
    return mf


def gather_valid(name, idir, lev):
    mf = get_mf(name, idir, lev)
    arr = mf[:, :, :]
    if arr.ndim == 4:
        arr = arr[..., 0]
    return np.asarray(arr)


def divb_layers(name="Bfield_fp", lev=1, nlayers=6):
    """Max |div| per in-plane layer (0 = outermost valid cell ring of the
    patch), plus the interior max beyond nlayers."""
    bx = gather_valid(name, 0, lev)
    by = gather_valid(name, 1, lev)
    bz = gather_valid(name, 2, lev)
    div = (
        (bx[1:, :, :] - bx[:-1, :, :]) / dx1[0]
        + (by[:, 1:, :] - by[:, :-1, :]) / dx1[1]
        + (bz[:, :, 1:] - bz[:, :, :-1]) / dx1[2]
    )
    nx, ny, _ = div.shape
    ii = np.arange(nx)[:, None]
    jj = np.arange(ny)[None, :]
    dist = np.minimum(np.minimum(ii, nx - 1 - ii), np.minimum(jj, ny - 1 - jj))
    out = []
    for d in range(nlayers):
        m = dist == d
        out.append(np.abs(div[m, :]).max() if m.any() else 0.0)
    m = dist >= nlayers
    out.append(np.abs(div[m, :]).max() if m.any() else 0.0)
    return out


def dump_div(tag):
    """Save the full fine-valid div field + fine box layout for offline
    localization (rank 0 writes; gathers are collective)."""
    bx = gather_valid("Bfield_fp", 0, 1)
    by = gather_valid("Bfield_fp", 1, 1)
    bz = gather_valid("Bfield_fp", 2, 1)
    div = (
        (bx[1:, :, :] - bx[:-1, :, :]) / dx1[0]
        + (by[:, 1:, :] - by[:, :-1, :]) / dx1[1]
        + (bz[:, :, 1:] - bz[:, :, :-1]) / dx1[2]
    )
    if IAMROOT:
        ba = get_mf("Bfield_fp", 0, 1).box_array()
        boxes = []
        for ib in range(ba.size):
            b = ba[ib]
            boxes.append(list(b.small_end) + list(b.big_end))
        np.savez("divdump_%s.npz" % tag, div=div, boxes=np.array(boxes))
        print("DUMPED divdump_%s.npz" % tag, flush=True)


MYRANK = libwarpx.amr.ParallelDescriptor.MyProc()


def dump_local_fabs(tag):
    """Every rank dumps its LOCAL fabs (valid+ghost) of the fine-level
    B/E/J/rho for offline cross-copy consistency analysis."""
    fields = [
        ("Bfield_fp", 0),
        ("Bfield_fp", 1),
        ("Bfield_fp", 2),
        ("Efield_fp", 0),
        ("Efield_fp", 1),
        ("Efield_fp", 2),
        ("current_fp", 0),
        ("current_fp", 1),
        ("current_fp", 2),
        ("rho_fp", None),
        ("hybrid_current_fp_plasma", 0),
        ("hybrid_current_fp_plasma", 1),
        ("hybrid_current_fp_plasma", 2),
        ("hybrid_current_fp_temp", 0),
        ("hybrid_current_fp_temp", 1),
        ("hybrid_current_fp_temp", 2),
        ("hybrid_current_fp_external", 0),
        ("hybrid_current_fp_external", 1),
        ("hybrid_current_fp_external", 2),
        ("hybrid_B_fp_external", 0),
        ("hybrid_B_fp_external", 1),
        ("hybrid_B_fp_external", 2),
        ("hybrid_rho_fp_temp", None),
        ("hybrid_electron_pressure_fp", None),
    ]
    out = {}
    reg = warpx.multifab_register()
    for name, idir in fields:
        try:
            if idir is None:
                mf = reg.get(name, level=1)
            else:
                mf = reg.get(name, dir=libwarpx.libwarpx_so.Direction(idir), level=1)
        except RuntimeError:
            continue
        key = "%s_%s" % (name, "s" if idir is None else "xyz"[idir])
        ng = np.array(mf.n_grow_vect)
        for ifab, mfi in enumerate(mf):
            vbx = mfi.validbox()
            lo = np.array(vbx.small_end)
            hi = np.array(vbx.big_end)
            a = np.array(mf.array(mfi), copy=True)
            out["%s.fab%d.data" % (key, ifab)] = a
            out["%s.fab%d.lo" % (key, ifab)] = lo
            out["%s.fab%d.hi" % (key, ifab)] = hi
            out["%s.fab%d.ng" % (key, ifab)] = ng
    np.savez("fabs_%s_rank%d.npz" % (tag, MYRANK), **out)
    if IAMROOT:
        print("DUMPED local fabs %s" % tag, flush=True)


records = []
stage_counter = [0]


def fmt(layers):
    return " ".join("%.3e" % v for v in layers)


def probe(label):
    layers = divb_layers()
    records.append((label, list(layers)))
    rprint("PROBE %-24s L0..L5 int: %s" % (label, fmt(layers)), flush=True)


def after_epush():
    stage_counter[0] += 1
    probe("afterEpush.%03d" % stage_counter[0])
    if stage_counter[0] == 3:
        dump_div("afterEpush%03d" % stage_counter[0])
    if stage_counter[0] in (1, 2):
        dump_local_fabs("afterEpush%03d" % stage_counter[0])


def after_bpush():
    probe("afterBpush.%03d" % stage_counter[0])
    if stage_counter[0] in (2, 3):
        dump_div("afterBpush%03d" % stage_counter[0])
    if stage_counter[0] in (1, 2):
        dump_local_fabs("afterBpush%03d" % stage_counter[0])


def before_step():
    probe("beforestep")


def after_step():
    probe("afterstep")


# ---------------------------------------------------------------------------
# Static state right after init
# ---------------------------------------------------------------------------
rprint("\n--- static, after init ---", flush=True)
probe("init.Bfield_fp")
bext_layers = divb_layers(name="hybrid_B_fp_external", lev=1)
rprint(
    "PROBE %-24s L0..L5 int: %s" % ("init.div(B_ext[1])", fmt(bext_layers)), flush=True
)
bext_layers0 = divb_layers(name="hybrid_B_fp_external", lev=0)
rprint(
    "PROBE %-24s L0..L5 int: %s" % ("init.div(B_ext[0])", fmt(bext_layers0)), flush=True
)

callbacks.installafterEpush(after_epush)
callbacks.installafterBpush(after_bpush)
callbacks.installbeforestep(before_step)
callbacks.installafterstep(after_step)

rprint("\n--- stepping %d step(s) ---" % args.steps, flush=True)
sim.step(args.steps)

rprint("\n--- post-run B_ext check ---", flush=True)
bext_layers = divb_layers(name="hybrid_B_fp_external", lev=1)
rprint(
    "PROBE %-24s L0..L5 int: %s" % ("final.div(B_ext[1])", fmt(bext_layers)), flush=True
)
probe("final.Bfield_fp")
