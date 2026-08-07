#!/usr/bin/env python3
"""
T0.2 -- RZ (m=0) flux-form divergence-free prolongation prototype
==================================================================

Standalone numpy prototype of a Balsara-style divergence-free face
prolongation / flux-sum restriction for the axisymmetric poloidal field
(B_r, B_z) on an RZ mesh, refinement ratio 2.  This is the design template
("contract") for the future C++ RZ face interpolater (P3).

Degrees of freedom: metric face FLUXES (per radian)
---------------------------------------------------
Cells (i,j): r in [r_i, r_{i+1}], z in [z_j, z_{j+1}], uniform (dr, dz).
  Phi_r(i,j) = int B_r r dz  over the r-face at r_i     [area r_i*dz]
  Phi_z(i,j) = int B_z r dr  over the z-face at z_j     [area (r_o^2-r_i^2)/2]
Discrete RZ divergence per cell = net outward flux (Gauss, per radian):
  D(i,j) = Phi_r(i+1,j) - Phi_r(i,j) + Phi_z(i,j+1) - Phi_z(i,j)
This is EXACTLY the metric divergence
  [ (r A B_r)_outer - (r A B_r)_inner ] + metric-area-weighted z-face
difference; no additional geometric factor appears once fluxes are the DOFs.

Constant-coefficient reformulation:  (u, w) = (r*B_r, r*B_z)
------------------------------------------------------------
The metric divergence  (1/r) d_r(r B_r) + d_z(B_z) = 0  multiplied by r is
  d_r(u) + d_z(w) = 0 ,
a CARTESIAN divergence in the (r,z) plane with constant coefficients.
Moreover the fluxes are plain (unweighted) line integrals of (u, w):
  Phi_r = int u dz   (r fixed on an r-face),
  Phi_z = int w dr .
So the entire 2D Cartesian Balsara construction of T0.1 applies verbatim to
(u, w): face means U = Phi_r/dz on r-faces, W = Phi_z/dr on z-faces,
  u(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*x*y      (x = local r, y = local z)
  w(x,y) = b0 + b1*x + b2*y + b3*x*y + b4*y^2
  constraints: a1 + b2 = coarse cell flux-div / (dr*dz)  [carried through]
               2*a3 + b3 = 0 ,  a4 + 2*b4 = 0
Fine faces are filled by EXACT metric-area integration of the polynomial:
on any face the polynomial is linear in the transverse coordinate, so the
exact sub-face integral is (midpoint value) * (sub-face length), i.e.
  Phi_r^fine = (u at sub-face midpoint) * dz/2 ,
  Phi_z^fine = (w at sub-face midpoint) * dr/2 .
Geometry (r values) never enters the prolongation itself -- only the
flux <-> B conversion and the axis parity below use r.

Restriction = flux sum (r-weighted face average of B):
  Phi_r^coarse(i,j) = Phi_rf(2i,2j) + Phi_rf(2i,2j+1)   (and analogously z),
which is exact for the reconstruction => R o P = identity on coarse fluxes.

Axis handling (r = 0)
---------------------
* The r-face at r=0 has zero area, hence zero flux: U(0,:) = 0 exactly on
  input, its transverse slope is 0, and the prolongation returns EXACTLY
  zero fine fluxes there -- automatically consistent, no special case.
* Parity: B_r is odd in r  => u = r*B_r is even;
          B_z is even in r => w = r*B_z is ODD, w(0,z) = 0.
  The only face-data slope taken in the r direction is that of W on
  z-faces.  For the axis cell the central difference uses a parity ghost
  cell mean: int_{-dr}^{0} w dr = -int_{0}^{dr} w dr (w odd), so
    sW(0,:) = (W(1,:) - W_ghost)/(2dr) = (W(1,:) + W(0,:)) / (2*dr).
* B_r "on the axis" is not a degree of freedom (zero-area face); fine
  values of B are recovered as flux/area only for faces of positive area.

Tests (all asserted in main):
  (a) machine-zero fine RZ discrete flux-div for random solenoidal fields,
      generated from the Stokes stream function psi = r*A_theta sampled at
      nodes: Phi_r = -(psi(z+) - psi(z-)), Phi_z = psi(r+) - psi(r-) are
      EXACT face-mean fluxes of B_r = -d_z A_theta, B_z = (1/r) d_r(r A_theta),
      and telescope to exactly solenoidal coarse data;
  (b) restriction o prolongation == identity on coarse fluxes;
  (c) order of accuracy vs an analytic A_theta (smooth, ~r near the axis),
      interior cells and axis-touching cells reported separately, both in
      B (flux/area) and in flux;
  (d) both a patch touching the axis (r in [0,1]) and a patch NOT touching
      it (r in [0.5,1.5]).

Run:  ~/.env/warpx/bin/python t0_2_rz_fluxform.py
"""

import os
import sys

import numpy as np

# ----------------------------------------------------------------------
# Core operators (flux DOFs)
# ----------------------------------------------------------------------


def div_rz(Phi_r, Phi_z):
    """Discrete RZ divergence per cell = net outward metric flux."""
    return (Phi_r[1:, :] - Phi_r[:-1, :]) + (Phi_z[:, 1:] - Phi_z[:, :-1])


def prolong_rz(Phi_r, Phi_z, dr, dz, axis_at_rlo):
    """Div-free RZ flux prolongation, ratio 2.

    Parameters
    ----------
    Phi_r : (Nr+1, Nz) coarse r-face metric fluxes
    Phi_z : (Nr, Nz+1) coarse z-face metric fluxes
    dr, dz : coarse spacings
    axis_at_rlo : True if the first r-face row is the axis r = 0
                  (activates the parity ghost for the W slope; the zero
                  axis flux itself needs no special case).

    Returns
    -------
    Phi_rf : (2Nr+1, 2Nz), Phi_zf : (2Nr, 2Nz+1) fine metric fluxes.
    """
    Nr, Nz = Phi_z.shape[0], Phi_r.shape[1]
    U = Phi_r / dz  # face means of u = r*B_r on r-faces
    W = Phi_z / dr  # face means of w = r*B_z on z-faces

    # Transverse slopes per FACE (unlimited central differences).
    sU = np.gradient(U, dz, axis=1, edge_order=2)  # d<u>/dz on r-faces
    sW = np.gradient(W, dr, axis=0, edge_order=2)  # d<w>/dr on z-faces
    if axis_at_rlo:
        # parity ghost: w odd in r => ghost cell mean = -W[0]
        sW[0, :] = (W[1, :] + W[0, :]) / (2.0 * dr)

    # Per-cell coefficients (same algebra as T0.1 2D with x -> r).
    Um, Up = U[:-1, :], U[1:, :]
    Sm, Sp = sU[:-1, :], sU[1:, :]
    Wm, Wp = W[:, :-1], W[:, 1:]
    Tm, Tp = sW[:, :-1], sW[:, 1:]

    _a1 = (Up - Um) / dr
    a2 = 0.5 * (Sp + Sm)
    a4 = (Sp - Sm) / dr
    _b2 = (Wp - Wm) / dz
    b1 = 0.5 * (Tp + Tm)
    b3 = (Tp - Tm) / dz
    a3 = -0.5 * b3
    b4 = -0.5 * a4
    a0 = 0.5 * (Up + Um) - 0.25 * a3 * dr * dr
    b0 = 0.5 * (Wp + Wm) - 0.25 * b4 * dz * dz

    drf, dzf = 0.5 * dr, 0.5 * dz
    Phi_rf = np.empty((2 * Nr + 1, 2 * Nz))
    Phi_zf = np.empty((2 * Nr, 2 * Nz + 1))

    # Fine r-faces (exact integral = midpoint value * sub-length).
    for oz, zb in ((0, -0.25 * dz), (1, +0.25 * dz)):
        Phi_rf[0::2, oz::2] = (U + sU * zb) * dzf  # on coarse planes
        Phi_rf[1::2, oz::2] = (a0 + a2 * zb) * dzf  # cell midplane r
    # Fine z-faces.
    for orr, rb in ((0, -0.25 * dr), (1, +0.25 * dr)):
        Phi_zf[orr::2, 0::2] = (W + sW * rb) * drf
        Phi_zf[orr::2, 1::2] = (b0 + b1 * rb) * drf

    return Phi_rf, Phi_zf


def restrict_rz(Phi_rf, Phi_zf):
    """Flux-sum restriction (= r-weighted face average of B)."""
    Phi_r = Phi_rf[0::2, 0::2] + Phi_rf[0::2, 1::2]
    Phi_z = Phi_zf[0::2, 0::2] + Phi_zf[1::2, 0::2]
    return Phi_r, Phi_z


# ----------------------------------------------------------------------
# Exact flux sampling from a stream function psi = r * A_theta
# ----------------------------------------------------------------------


def fluxes_from_psi(psi):
    """Exact metric face fluxes from node values of psi = r*A_theta.

    Phi through any surface of revolution spanning two circles equals the
    difference of psi (Stokes): Phi_r = -(psi(r,z+) - psi(r,z-)) and
    Phi_z = psi(r+,z) - psi(r-,z).  Coarse div telescopes to machine zero.
    """
    Phi_r = -(psi[:, 1:] - psi[:, :-1])
    Phi_z = psi[1:, :] - psi[:-1, :]
    return Phi_r, Phi_z


def B_from_fluxes(Phi_r, Phi_z, r_edges, dz):
    """Face-mean B from fluxes; B_r at r=0 (zero-area face) is set to 0."""
    rf = r_edges[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        Br = np.where(rf > 0.0, Phi_r / np.where(rf > 0.0, rf, 1.0) / dz, 0.0)
    area_z = 0.5 * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    Bz = Phi_z / area_z[:, None]
    return Br, Bz


def A_theta(r, z):
    """Smooth analytic A_theta, odd in r (=> B_z even, B_r odd; B ~ O(1))."""
    return r * np.exp(-(r**2)) * (
        1.0 + 0.5 * np.sin(2 * np.pi * z + 0.4)
    ) + 0.2 * r * np.exp(-2.0 * r**2) * np.cos(2 * np.pi * z - 0.3)


def analytic_fluxes(r_lo, r_hi, z_lo, z_hi, Nr, Nz):
    """Exact coarse fluxes of the analytic field via psi at the nodes."""
    r = np.linspace(r_lo, r_hi, Nr + 1)
    z = np.linspace(z_lo, z_hi, Nz + 1)
    psi = r[:, None] * A_theta(r[:, None], z[None, :])
    Phi_r, Phi_z = fluxes_from_psi(psi)
    return Phi_r, Phi_z, r, z


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

PASS = "PASS"


def _check(label, value, tol, results):
    ok = value < tol
    results.append((label, f"{value:.3e}", f"< {tol:.0e}", PASS if ok else "FAIL"))
    assert ok, f"{label}: {value:.3e} !< {tol:.0e}"


def _check_order(label, order, lo, results):
    ok = order > lo
    results.append((label, f"{order:.3f}", f"> {lo}", PASS if ok else "FAIL"))
    assert ok, f"{label}: order {order:.3f} !> {lo}"


def print_table(title, results):
    print(f"\n=== {title} ===")
    w0 = max(len(r[0]) for r in results) + 2
    print(f"{'check':<{w0}}{'value':>12}  {'criterion':<10}{'status'}")
    for r in results:
        print(f"{r[0]:<{w0}}{r[1]:>12}  {r[2]:<10}{r[3]}")


def _subset_l2(err_r, err_z, r_mask_r, r_mask_z):
    """Combined L2 over the selected radial rows of both components."""
    vals = np.concatenate([err_r[r_mask_r, :].ravel(), err_z[r_mask_z, :].ravel()])
    return np.sqrt(np.mean(vals**2))


def run_patch(r_lo, r_hi, tag, results, rng):
    axis = r_lo == 0.0
    z_lo, z_hi = 0.0, 1.0
    N = 24
    Nr = Nz = N
    dr, dz = (r_hi - r_lo) / Nr, (z_hi - z_lo) / Nz

    # (a) random solenoidal field: random psi nodes (psi = 0 on the axis).
    psi = rng.standard_normal((Nr + 1, Nz + 1))
    if axis:
        psi[0, :] = 0.0
    Phi_r, Phi_z = fluxes_from_psi(psi)
    scale = max(np.abs(Phi_r).max(), np.abs(Phi_z).max())
    Dc = np.abs(div_rz(Phi_r, Phi_z)).max()
    Phi_rf, Phi_zf = prolong_rz(Phi_r, Phi_z, dr, dz, axis)
    Df = np.abs(div_rz(Phi_rf, Phi_zf)).max()
    _check("(a) coarse flux-div (random, rel)", Dc / scale, 1e-13, results)
    _check("(a) fine flux-div (random, rel)", Df / scale, 1e-13, results)
    if axis:
        ax_flux = np.abs(Phi_rf[0, :]).max()
        results.append(
            (
                "    fine axis-face flux (exact 0)",
                f"{ax_flux:.1e}",
                "== 0",
                PASS if ax_flux == 0.0 else "FAIL",
            )
        )
        assert ax_flux == 0.0

    # (b) restriction o prolongation identity on coarse fluxes
    Phi_rr, Phi_zr = restrict_rz(Phi_rf, Phi_zf)
    idr = max(np.abs(Phi_rr - Phi_r).max(), np.abs(Phi_zr - Phi_z).max())
    _check("(b) R(P(.)) - I (random, rel)", idr / scale, 1e-13, results)

    # (c) order of accuracy vs the analytic field; interior vs axis cells.
    metrics = {}  # {label: [err(N1), err(N2)]}
    Ns = (24, 48)
    for Nc in Ns:
        drc, dzc = (r_hi - r_lo) / Nc, (z_hi - z_lo) / Nc
        Phi_r, Phi_z, r_c, _ = analytic_fluxes(r_lo, r_hi, z_lo, z_hi, Nc, Nc)
        Phi_rf, Phi_zf = prolong_rz(Phi_r, Phi_z, drc, dzc, axis)
        Phi_rE, Phi_zE, r_f, _ = analytic_fluxes(r_lo, r_hi, z_lo, z_hi, 2 * Nc, 2 * Nc)
        # sanity: prolonged fluxes stay solenoidal on the analytic case too
        assert np.abs(div_rz(Phi_rf, Phi_zf)).max() < 1e-13 * max(
            np.abs(Phi_rf).max(), 1.0
        )

        Br_p, Bz_p = B_from_fluxes(Phi_rf, Phi_zf, r_f, 0.5 * dzc)
        Br_e, Bz_e = B_from_fluxes(Phi_rE, Phi_zE, r_f, 0.5 * dzc)
        err_Br = np.abs(Br_p - Br_e)
        err_Bz = np.abs(Bz_p - Bz_e)
        # flux errors normalized by dr*dz (a fixed logical-area scale).
        # NOTE: relative to the B error this carries an extra ~r/h weight
        # (flux ~ r*h*B on r-faces), so its measured interior order is
        # (B order - 1); it is reported as a fixed-scale metric of the
        # flux DOFs themselves, not as a second estimate of the B order.
        err_Fr = np.abs(Phi_rf - Phi_rE) / (drc * dzc)
        err_Fz = np.abs(Phi_zf - Phi_zE) / (drc * dzc)

        nrf, nzf = Br_p.shape[0], Bz_p.shape[0]  # 2Nc+1 r-faces, 2Nc z-rows
        # axis subset: fine faces of the fine cells inside the first coarse
        # radial ring (fine cells i_f = 0,1). Skip the zero-area face i=0.
        rmask_r = np.zeros(nrf, dtype=bool)
        rmask_z = np.zeros(nzf, dtype=bool)
        rmask_r[1:3] = True
        rmask_z[0:2] = True
        int_r = ~rmask_r
        int_z = ~rmask_z
        if axis:
            int_r[0] = False  # zero-area face: not a DOF
        if axis:
            metrics.setdefault("B  axis-cells", []).append(
                _subset_l2(err_Br, err_Bz, rmask_r, rmask_z)
            )
            metrics.setdefault("B  interior", []).append(
                _subset_l2(err_Br, err_Bz, int_r, int_z)
            )
            metrics.setdefault("flux axis-cells", []).append(
                _subset_l2(err_Fr, err_Fz, rmask_r, rmask_z)
            )
            metrics.setdefault("flux interior", []).append(
                _subset_l2(err_Fr, err_Fz, int_r, int_z)
            )
        else:
            allr = np.ones(nrf, dtype=bool)
            allz = np.ones(nzf, dtype=bool)
            metrics.setdefault("B  interior", []).append(
                _subset_l2(err_Br, err_Bz, allr, allz)
            )
            metrics.setdefault("flux interior", []).append(
                _subset_l2(err_Fr, err_Fz, allr, allz)
            )

        # div/identity checks on the analytic case at the working resolution
        if Nc == Ns[0]:
            Phi_rr, Phi_zr = restrict_rz(Phi_rf, Phi_zf)
            idr = max(np.abs(Phi_rr - Phi_r).max(), np.abs(Phi_zr - Phi_z).max())
            sc = max(np.abs(Phi_r).max(), np.abs(Phi_z).max())
            _check("(b) R(P(.)) - I (analytic, rel)", idr / sc, 1e-13, results)

    orders = {}
    for label, errs in metrics.items():
        order = np.log2(errs[0] / errs[1])
        orders[label] = (errs, order)
        results.append((f"(c) L2 {label}  N={Ns[0]}", f"{errs[0]:.3e}", "", ""))
        results.append((f"(c) L2 {label}  N={Ns[1]}", f"{errs[1]:.3e}", "", ""))
        lo = 1.7 if "interior" in label else 0.7
        _check_order(f"(c) order {label}", order, lo, results)

    return orders


def save_error_map_png(outdir):
    """Optional: |B_z error| map for the axis-touching analytic case."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    Nc = 32
    r_lo, r_hi, z_lo, z_hi = 0.0, 1.0, 0.0, 1.0
    drc, dzc = (r_hi - r_lo) / Nc, (z_hi - z_lo) / Nc
    Phi_r, Phi_z, r_c, _ = analytic_fluxes(r_lo, r_hi, z_lo, z_hi, Nc, Nc)
    Phi_rf, Phi_zf = prolong_rz(Phi_r, Phi_z, drc, dzc, True)
    Phi_rE, Phi_zE, r_f, z_f = analytic_fluxes(r_lo, r_hi, z_lo, z_hi, 2 * Nc, 2 * Nc)
    Bz_p = B_from_fluxes(Phi_rf, Phi_zf, r_f, 0.5 * dzc)[1]
    Bz_e = B_from_fluxes(Phi_rE, Phi_zE, r_f, 0.5 * dzc)[1]
    err = np.abs(Bz_p - Bz_e)
    rc_f = 0.5 * (r_f[1:] + r_f[:-1])
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.pcolormesh(z_f, rc_f, err, cmap="Blues", shading="nearest")
    fig.colorbar(im, ax=ax, label=r"$|B_z^{prolonged} - B_z^{exact}|$")
    ax.set_xlabel("z (fine z-face planes)")
    ax.set_ylabel("r (fine cell centers)")
    ax.set_title(f"T0.2 RZ prolongation error in $B_z$, coarse N={Nc} (axis at r=0)")
    path = os.path.join(outdir, "t0_2_error_map_rz.png")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main():
    rng = np.random.default_rng(20260806)
    res_axis, res_off = [], []
    print("T0.2 RZ (m=0) flux-form div-free prolongation prototype")
    run_patch(0.0, 1.0, "axis", res_axis, rng)
    run_patch(0.5, 1.5, "offaxis", res_off, rng)
    print_table("T0.2 patch touching the axis: r in [0,1]", res_axis)
    print_table("T0.2 patch NOT touching the axis: r in [0.5,1.5]", res_off)
    outdir = os.path.dirname(os.path.abspath(__file__))
    png = save_error_map_png(outdir)
    if png:
        print(f"\nerror map: {png}")
    print("\nT0.2: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
