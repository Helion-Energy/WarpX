#!/usr/bin/env python3
"""Unit repro for the fine-seam div(B) injection (divb_seam_diagnosis_notes.md §1.3).

Pure-geometry test of the fine-level direct-B writers: perturb ONLY the
coarse B with a manufactured discretely-div-free field, run exactly one
inert step (dt = 1e-25 s) of the 2-level t_smoke_inputs_3d config, and diff
the fine-level valid B faces before/after. Faraday and every other O(dt)
operator is quenched; any fine VALID face that moves by O(perturbation) was
directly overwritten by interlevel/comm machinery.

Usage:
  PYTHONPATH=<build>/lib/site-packages python divb_seam_unit_repro.py \
      [--amp 1e-3] [--control] [--emf-matching]
"""

import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--amp",
    type=float,
    default=1.0e-3,
    help="perturbation amplitude in T (0 = control leg)",
)
parser.add_argument(
    "--control", action="store_true", help="zero-perturbation control leg"
)
parser.add_argument(
    "--emf-matching",
    action="store_true",
    help="run with hybrid_pic_model.mr_emf_matching=1 (arm-M-like)",
)
parser.add_argument("--inputs", default="t_smoke_inputs_3d")
parser.add_argument("--substeps", type=int, default=20)
args = parser.parse_args()

AMP = 0.0 if args.control else args.amp
THRESH = 1.0e-13  # T; direct writes appear at O(AMP)=1e-3..1e-5, physics at O(dt)=1e-25

from pywarpx import geometry  # noqa: E402

geometry.dims = "3"  # select the 3D pybind module before loading

from pywarpx._libwarpx import libwarpx  # noqa: E402

argv = [
    "warpx",
    args.inputs,
    "max_step=1",
    "warpx.const_dt=1e-25",
    "hybrid_pic_model.substeps=%d" % args.substeps,
    "hybrid_pic_model.mr_check_div_b=1",
    "hybrid_pic_model.mr_emf_matching=%d" % (1 if args.emf_matching else 0),
    "diag1.intervals=100000",
    "field_energy.intervals=100000",
    "part_energy.intervals=100000",
    "ions.num_particles_per_cell=1",
    "amrex.verbose=1",
]
libwarpx.initialize(argv)
warpx = libwarpx.warpx

geom0 = warpx.Geom(0)
geom1 = warpx.Geom(1)
dx0 = np.array([geom0.data().CellSize(d) for d in range(3)])
dx1 = np.array([geom1.data().CellSize(d) for d in range(3)])
prob_lo = np.array([geom0.data().ProbLo(d) for d in range(3)])
prob_hi = np.array([geom0.data().ProbHi(d) for d in range(3)])
L = prob_hi - prob_lo

# ---------------------------------------------------------------------------
# Manufactured div-free perturbation: B_pert = curl_h(A), A analytic periodic,
# evaluated at exact Yee staggered positions => div_h(B_pert) = 0 identically
# on every cell all of whose faces are set consistently (all of them: valid
# and ghosts get the same closed formula; ghost indices evaluate the periodic
# image to roundoff).
# ---------------------------------------------------------------------------
kx, ky, kz = 2 * np.pi / L[0], 2 * np.pi / L[1], 2 * np.pi / L[2]
# A amplitude chosen so |B_pert| ~ AMP
ca = AMP / max(kx, ky, kz)


def a_x(x, y, z):
    return ca * np.sin(2 * ky * y) * np.cos(kz * z)


def a_y(x, y, z):
    return ca * np.sin(kz * z) * np.cos(2 * kx * x)


def a_z(x, y, z):
    return ca * np.sin(kx * x) * np.cos(2 * ky * y)


def bpert(comp, i, j, k, dx, plo):
    """Discrete curl of A on the Yee face (comp) at integer face index
    (i, j, k) of a level with spacing dx. Index convention: face 'x' at
    (i, j+1/2, k+1/2) etc.; A_x lives at (i+1/2, j, k) etc."""
    x0, y0, z0 = plo
    dxx, dyy, dzz = dx

    def Ax(ii, jj, kk):
        return a_x(x0 + (ii + 0.5) * dxx, y0 + jj * dyy, z0 + kk * dzz)

    def Ay(ii, jj, kk):
        return a_y(x0 + ii * dxx, y0 + (jj + 0.5) * dyy, z0 + kk * dzz)

    def Az(ii, jj, kk):
        return a_z(x0 + ii * dxx, y0 + jj * dyy, z0 + (kk + 0.5) * dzz)

    if comp == "x":  # dAz/dy - dAy/dz at (i, j+1/2, k+1/2)
        return (Az(i, j + 1, k) - Az(i, j, k)) / dyy - (
            Ay(i, j, k + 1) - Ay(i, j, k)
        ) / dzz
    if comp == "y":  # dAx/dz - dAz/dx at (i+1/2, j, k+1/2)
        return (Ax(i, j, k + 1) - Ax(i, j, k)) / dzz - (
            Az(i + 1, j, k) - Az(i, j, k)
        ) / dxx
    # z: dAy/dx - dAx/dy at (i+1/2, j+1/2, k)
    return (Ay(i + 1, j, k) - Ay(i, j, k)) / dxx - (Ax(i, j + 1, k) - Ax(i, j, k)) / dyy


COMPS = ["x", "y", "z"]
STAG = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def get_mf(comp, lev):
    reg = warpx.multifab_register()
    direction = libwarpx.libwarpx_so.Direction(COMPS.index(comp))
    mf = reg.get("Bfield_fp", dir=direction, level=lev)
    mf.level = lev
    return mf


def valid_arr(comp, lev):
    """Gather the VALID union of Bfield_fp[comp] on level lev plus its
    global index origin (from imesh)."""
    mf = get_mf(comp, lev)
    mf.level = lev
    arr = mf[:, :, :]
    if arr.ndim == 4:
        arr = arr[..., 0]
    lo = [int(mf.imesh(d, False)[0]) for d in range(3)]
    return np.array(arr, copy=True), lo


def full_arr(comp, lev):
    """Gather valid+ghost union of Bfield_fp[comp] on level lev."""
    mf = get_mf(comp, lev)
    arr = mf[()]
    if arr.ndim == 4:
        arr = arr[..., 0]
    return np.array(arr, copy=True)


# ---------------------------------------------------------------------------
# Snapshot fine level before
# ---------------------------------------------------------------------------
before = {}
before_full = {}
fine_lo = {}
for c in COMPS:
    before[c], fine_lo[c] = valid_arr(c, 1)
    before_full[c] = full_arr(c, 1)

# Perturb coarse B (valid + all ghosts) via the empty-tuple global index
for c in COMPS:
    mf = get_mf(c, 0)
    mf.level = 0
    ng = mf.n_grow_vect
    stag = STAG[c]
    lo = [int(mf.imesh(d, False)[0]) for d in range(3)]  # valid lo (0 for level 0)
    full = mf[()]
    if full.ndim == 4:
        shp = full.shape[:3]
    else:
        shp = full.shape
    idx = [np.arange(lo[d] - ng[d], lo[d] - ng[d] + shp[d]) for d in range(3)]
    ii, jj, kk = np.meshgrid(idx[0], idx[1], idx[2], indexing="ij")
    pert = bpert(c, ii, jj, kk, dx0, prob_lo)
    newv = np.array(full, copy=True)
    if newv.ndim == 4:
        newv[..., 0] += pert
    else:
        newv += pert
    mf[()] = newv

# sanity: report coarse divB of the perturbed state (should be machine zero)
bx0, _ = valid_arr("x", 0)
by0, _ = valid_arr("y", 0)
bz0, _ = valid_arr("z", 0)
div0 = (
    (bx0[1:, :, :] - bx0[:-1, :, :]) / dx0[0]
    + (by0[:, 1:, :] - by0[:, :-1, :]) / dx0[1]
    + (bz0[:, :, 1:] - bz0[:, :, :-1]) / dx0[2]
)
print("UNITREPRO coarse |divB| after perturbation: max = %.3e" % np.abs(div0).max())

# ---------------------------------------------------------------------------
# One inert step
# ---------------------------------------------------------------------------
warpx.evolve(1)

# ---------------------------------------------------------------------------
# Diff fine valid faces
# ---------------------------------------------------------------------------
after = {}
after_full = {}
for c in COMPS:
    after[c], _ = valid_arr(c, 1)
    after_full[c] = full_arr(c, 1)

# instrument sanity: the fine C-F GHOST faces must have moved by
# O(perturbation) in the perturbed leg (the fill prolongs the perturbed
# coarse field into them); report the ghost-region diff per component.
print("\n-------- fine ghost-region diff (instrument sanity) --------")
for c in COMPS:
    dfull = np.abs(after_full[c] - before_full[c])
    mf = get_mf(c, 1)
    ng = mf.n_grow_vect
    shell = np.array(dfull, copy=True)
    shell[
        ng[0] : dfull.shape[0] - ng[0],
        ng[1] : dfull.shape[1] - ng[1],
        ng[2] : dfull.shape[2] - ng[2],
    ] = 0.0
    print(
        "B%s: full-union max|dB| = %.3e, ghost-shell max|dB| = %.3e"
        % (c, dfull.max(), shell.max())
    )

print("\n================ UNIT REPRO VERDICT ================")
print("perturbation amplitude: %.3e T, threshold: %.1e T" % (AMP, THRESH))
any_valid_change = False
for c in COMPS:
    d = np.abs(after[c] - before[c])
    changed = d > THRESH
    nch = int(changed.sum())
    print(
        "\ncomponent B%s: valid shape %s, changed faces: %d, max|dB| = %.3e"
        % (c, d.shape, nch, d.max())
    )
    if nch == 0:
        continue
    any_valid_change = True
    # classify by layer offset from the patch surface, per side
    stag = STAG[c]
    n = d.shape
    for axis in range(3):
        w = np.where(changed)
        lo_off = w[axis].min()
        hi_off = (n[axis] - 1) - w[axis].max()
        print(
            "  axis %d: changed-face index range [%d, %d] of [0, %d] "
            "(offset from lo surface %d, from hi surface %d)"
            % (axis, w[axis].min(), w[axis].max(), n[axis] - 1, lo_off, hi_off)
        )
    # per-layer counts on the normal axis of this component and each axis
    for axis in range(3):
        layers = {}
        w = np.where(changed)
        for v in w[axis]:
            layers[int(v)] = layers.get(int(v), 0) + 1
        near_lo = {kk: vv for kk, vv in sorted(layers.items())[:3]}
        near_hi = {kk: vv for kk, vv in sorted(layers.items())[-3:]}
        print(
            "  axis %d layer histogram (lo-3): %s ... (hi-3): %s"
            % (axis, near_lo, near_hi)
        )


# fine divB before/after on valid cells
def divb(bx, by, bz, dx):
    return (
        (bx[1:, :, :] - bx[:-1, :, :]) / dx[0]
        + (by[:, 1:, :] - by[:, :-1, :]) / dx[1]
        + (bz[:, :, 1:] - bz[:, :, :-1]) / dx[2]
    )


div_before = divb(before["x"], before["y"], before["z"], dx1)
div_after = divb(after["x"], after["y"], after["z"], dx1)
ddiv = np.abs(div_after - div_before)
print(
    "\nfine valid divB: max before %.3e, max after %.3e, max delta %.3e"
    % (np.abs(div_before).max(), np.abs(div_after).max(), ddiv.max())
)
if ddiv.max() > 1e-10:
    w = np.unravel_index(np.argmax(ddiv), ddiv.shape)
    print("  worst cell (rel to fine valid lo): %s of shape %s" % (w, ddiv.shape))
    # layer histogram of cells with ddiv > 1e-10
    hot = np.where(ddiv > 1e-10)
    for axis in range(3):
        print(
            "  axis %d hot-cell index range: [%d, %d] of [0, %d]"
            % (axis, hot[axis].min(), hot[axis].max(), ddiv.shape[axis] - 1)
        )

print(
    "\nVERDICT: %s"
    % (
        "V-A: OUTERMOST-VALID FINE FACES WERE WRITTEN (H1-class confirmed)"
        if any_valid_change
        else "V-B: fine valid faces bit-clean (H1 falsified for the ungated path)"
    )
)
print("====================================================\n")

libwarpx.finalize()
