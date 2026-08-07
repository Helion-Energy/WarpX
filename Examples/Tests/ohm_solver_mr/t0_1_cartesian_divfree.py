#!/usr/bin/env python3
"""
T0.1 -- Cartesian divergence-free face prolongation / restriction reference
===========================================================================

Standalone numpy reference implementation (no WarpX / no pyamrex) of the
Balsara / Toth-Roe style divergence-free prolongation of face-centered
(Yee, MAC-staggered) magnetic-field data for a refinement ratio of 2, plus
its adjoint area-average restriction.  This is the *reference* the AMReX
``FaceDivFree`` interpolater will be checked against later, so the math is
written out explicitly.

Index-space conventions (2D)
----------------------------
Coarse grid of Nx x Ny cells, spacing (dx, dy).
  Bx : shape (Nx+1, Ny)  -- face-MEAN of B_x over each x-face (normal = x)
  By : shape (Nx,  Ny+1) -- face-MEAN of B_y over each y-face (normal = y)
Discrete divergence per cell (face means, so no area weights in Cartesian):
  D(i,j) = (Bx[i+1,j]-Bx[i,j])/dx + (By[i,j+1]-By[i,j])/dy

2D reconstruction inside a coarse cell (Balsara 2001, review-paper form)
------------------------------------------------------------------------
Local cell coordinates x in [-dx/2, dx/2], y in [-dy/2, dy/2]:

  B_x(x,y) = a0 + ax*x + ay*y + axx*x^2 + axy*x*y
  B_y(x,y) = b0 + bx*x + by*y + bxy*x*y + byy*y^2

Pointwise divergence:
  div = (ax + by) + x*(2*axx + bxy) + y*(axy + 2*byy)
so the div-free constraints are
  (C1)  ax + by         = 0        [holds iff the coarse cell is div-free;
                                    we do NOT force it -- the reconstruction
                                    carries the coarse cell's divergence
                                    through unchanged, i.e. div-PRESERVING]
  (C2)  2*axx + bxy     = 0
  (C3)  axy  + 2*byy    = 0

Face matching data (per coarse FACE, so the two cells sharing a face use
identical data => the reconstruction is normal-continuous across cells):
  * the face mean (the coarse DOF itself), and
  * the transverse slope of the face-mean profile, from simple UNLIMITED
    central differences of neighboring face means (one-sided, 2nd order, at
    the domain boundary):  e.g. on x-faces  s_y(i,j) ~ d<B_x>/dy.

Matching the y-averaged polynomial and its y-slope on the two x-faces:
  <B_x>(+-dx/2)      = a0 -+... :  ax  = (Bx+ - Bx-)/dx
                                   a0  = (Bx+ + Bx-)/2 - axx*dx^2/4
  d/dy B_x(+-dx/2,y) = s_y^+-  :   ay  = (s_y^+ + s_y^-)/2
                                   axy = (s_y^+ - s_y^-)/dx
and symmetrically for B_y.  (C2), (C3) then fix the quadratic coefficients:
  axx = -bxy/2 ,   byy = -axy/2 .

Fine faces are filled with the exact FACE AVERAGE of the polynomial.  Since
B_x contains no y^2 term it is linear in y on any x-face, so the exact face
average equals the value at the face midpoint (same for B_y / B_z below).
Fine faces lying ON a coarse face plane are filled directly from the
per-face data (mean + slope * transverse-midpoint-offset), which is
algebraically identical to evaluating either neighboring polynomial there.

3D extension (standard tensor generalization; verified numerically below)
--------------------------------------------------------------------------
  B_x = a0 + ax*x + ay*y + az*z + axx*x^2 + axy*x*y + axz*x*z
  B_y = b0 + bx*x + by*y + bz*z + byy*y^2 + bxy*x*y + byz*y*z
  B_z = c0 + cx*x + cy*y + cz*z + czz*z^2 + cxz*x*z + cyz*y*z
Pointwise divergence:
  div = (ax+by+cz) + x*(2axx+bxy+cxz) + y*(axy+2byy+cyz) + z*(axz+byz+2czz)
Constraint set used:
  (C1)  ax + by + cz        = 0   [coarse-cell divergence, carried through]
  (C2)  2*axx + bxy + cxz   = 0
  (C3)  axy + 2*byy + cyz   = 0
  (C4)  axz + byz + 2*czz   = 0
Face data: mean + TWO transverse slopes per face; the six cross coefficients
(axy, axz, bxy, byz, cxz, cyz) come from slope differences across the cell,
the three pure-quadratic ones from (C2)-(C4), the linear "own-direction"
coefficients from face-mean differences.  This is the minimal (no xyz term)
Balsara set consistent with faces that are linear in both transverse
directions.  ``verify_constraints_3d`` checks numerically, at random points
in every cell, that the pointwise polynomial divergence equals the coarse
discrete divergence of that cell (machine zero for solenoidal input).

Guaranteed properties (tested in main):
  (a) fine discrete div == coarse discrete div carried through (machine zero
      for solenoidal coarse data),
  (b) restriction o prolongation == identity on coarse faces,
  (c) 2nd-order convergence to smooth solenoidal fields,
  (d) exact reproduction of uniform and linear solenoidal fields.

Run:  ~/.env/warpx/bin/python t0_1_cartesian_divfree.py
"""

import os
import sys

import numpy as np

# ----------------------------------------------------------------------
# 2D operators
# ----------------------------------------------------------------------


def div_2d(Bx, By, dx, dy):
    """Discrete divergence per cell from face-mean data."""
    return (Bx[1:, :] - Bx[:-1, :]) / dx + (By[:, 1:] - By[:, :-1]) / dy


def prolong_2d(Bx, By, dx, dy):
    """Balsara div-free prolongation, ratio 2. Returns fine face means."""
    Nx, Ny = By.shape[0], Bx.shape[1]

    # Per-FACE transverse slopes (unlimited central differences; one-sided
    # 2nd-order at domain boundaries via edge_order=2).
    sBx = np.gradient(Bx, dy, axis=1, edge_order=2)  # d<Bx>/dy on x-faces
    sBy = np.gradient(By, dx, axis=0, edge_order=2)  # d<By>/dx on y-faces

    # Per-cell coefficient arrays, shape (Nx, Ny).
    Uxm, Uxp = Bx[:-1, :], Bx[1:, :]
    Sxm, Sxp = sBx[:-1, :], sBx[1:, :]
    Uym, Uyp = By[:, :-1], By[:, 1:]
    Sym, Syp = sBy[:, :-1], sBy[:, 1:]

    _ax = (Uxp - Uxm) / dx
    ay = 0.5 * (Sxp + Sxm)
    axy = (Sxp - Sxm) / dx
    _by = (Uyp - Uym) / dy
    bx = 0.5 * (Syp + Sym)
    bxy = (Syp - Sym) / dy
    axx = -0.5 * bxy  # (C2)
    byy = -0.5 * axy  # (C3)
    a0 = 0.5 * (Uxp + Uxm) - 0.25 * axx * dx * dx
    b0 = 0.5 * (Uyp + Uym) - 0.25 * byy * dy * dy

    Bxf = np.empty((2 * Nx + 1, 2 * Ny))
    Byf = np.empty((2 * Nx, 2 * Ny + 1))

    # Fine x-faces. Transverse (y) sub-face midpoints sit at -+dy/4.
    for oy, yb in ((0, -0.25 * dy), (1, +0.25 * dy)):
        # even fine i-index: on a coarse face plane -> per-face data
        Bxf[0::2, oy::2] = Bx + sBx * yb
        # odd fine i-index: cell midplane x=0 -> polynomial
        Bxf[1::2, oy::2] = a0 + ay * yb
    # Fine y-faces.
    for ox, xb in ((0, -0.25 * dx), (1, +0.25 * dx)):
        Byf[ox::2, 0::2] = By + sBy * xb
        Byf[ox::2, 1::2] = b0 + bx * xb

    return Bxf, Byf


def restrict_2d(Bxf, Byf):
    """Area-average (here: transverse-average) restriction, ratio 2."""
    Bx = 0.5 * (Bxf[0::2, 0::2] + Bxf[0::2, 1::2])
    By = 0.5 * (Byf[0::2, 0::2] + Byf[1::2, 0::2])
    return Bx, By


def two_sided_face_check_2d(Bx, By, dx, dy):
    """Max mismatch of the two neighboring polynomials on shared x-faces.

    The even-plane fine faces are filled from per-face data; this verifies
    that both adjacent cell polynomials reproduce that value (normal-
    component continuity of the reconstruction).
    """
    sBx = np.gradient(Bx, dy, axis=1, edge_order=2)
    sBy = np.gradient(By, dx, axis=0, edge_order=2)
    Uxm, Uxp = Bx[:-1, :], Bx[1:, :]
    Sxm, Sxp = sBx[:-1, :], sBx[1:, :]
    ax = (Uxp - Uxm) / dx
    ay = 0.5 * (Sxp + Sxm)
    axy = (Sxp - Sxm) / dx
    bxy = (sBy[:, 1:] - sBy[:, :-1]) / dy
    axx = -0.5 * bxy
    a0 = 0.5 * (Uxp + Uxm) - 0.25 * axx * dx * dx
    worst = 0.0
    for yb in (-0.25 * dy, +0.25 * dy):
        left = (
            a0 + ax * 0.5 * dx + axx * 0.25 * dx * dx + (ay + axy * 0.5 * dx) * yb
        )  # cell i at +dx/2
        right = (
            a0 - ax * 0.5 * dx + axx * 0.25 * dx * dx + (ay - axy * 0.5 * dx) * yb
        )  # cell i+1 at -dx/2
        worst = max(worst, np.abs(left[:-1, :] - right[1:, :]).max())
    return worst


# ----------------------------------------------------------------------
# 3D operators
# ----------------------------------------------------------------------


def div_3d(Bx, By, Bz, dx, dy, dz):
    return (
        (Bx[1:, :, :] - Bx[:-1, :, :]) / dx
        + (By[:, 1:, :] - By[:, :-1, :]) / dy
        + (Bz[:, :, 1:] - Bz[:, :, :-1]) / dz
    )


def _coeffs_3d(Bx, By, Bz, dx, dy, dz):
    """Per-cell polynomial coefficients (see module docstring)."""
    sxy = np.gradient(Bx, dy, axis=1, edge_order=2)
    sxz = np.gradient(Bx, dz, axis=2, edge_order=2)
    syx = np.gradient(By, dx, axis=0, edge_order=2)
    syz = np.gradient(By, dz, axis=2, edge_order=2)
    szx = np.gradient(Bz, dx, axis=0, edge_order=2)
    szy = np.gradient(Bz, dy, axis=1, edge_order=2)

    c = {}
    c["ax"] = (Bx[1:, :, :] - Bx[:-1, :, :]) / dx
    c["ay"] = 0.5 * (sxy[1:, :, :] + sxy[:-1, :, :])
    c["axy"] = (sxy[1:, :, :] - sxy[:-1, :, :]) / dx
    c["az"] = 0.5 * (sxz[1:, :, :] + sxz[:-1, :, :])
    c["axz"] = (sxz[1:, :, :] - sxz[:-1, :, :]) / dx

    c["by"] = (By[:, 1:, :] - By[:, :-1, :]) / dy
    c["bx"] = 0.5 * (syx[:, 1:, :] + syx[:, :-1, :])
    c["bxy"] = (syx[:, 1:, :] - syx[:, :-1, :]) / dy
    c["bz"] = 0.5 * (syz[:, 1:, :] + syz[:, :-1, :])
    c["byz"] = (syz[:, 1:, :] - syz[:, :-1, :]) / dy

    c["cz"] = (Bz[:, :, 1:] - Bz[:, :, :-1]) / dz
    c["cx"] = 0.5 * (szx[:, :, 1:] + szx[:, :, :-1])
    c["cxz"] = (szx[:, :, 1:] - szx[:, :, :-1]) / dz
    c["cy"] = 0.5 * (szy[:, :, 1:] + szy[:, :, :-1])
    c["cyz"] = (szy[:, :, 1:] - szy[:, :, :-1]) / dz

    # (C2)-(C4): pure quadratic coefficients from the cross terms.
    c["axx"] = -0.5 * (c["bxy"] + c["cxz"])
    c["byy"] = -0.5 * (c["axy"] + c["cyz"])
    c["czz"] = -0.5 * (c["axz"] + c["byz"])

    c["a0"] = 0.5 * (Bx[1:, :, :] + Bx[:-1, :, :]) - 0.25 * c["axx"] * dx * dx
    c["b0"] = 0.5 * (By[:, 1:, :] + By[:, :-1, :]) - 0.25 * c["byy"] * dy * dy
    c["c0"] = 0.5 * (Bz[:, :, 1:] + Bz[:, :, :-1]) - 0.25 * c["czz"] * dz * dz
    return c


def prolong_3d(Bx, By, Bz, dx, dy, dz):
    Nx, Ny, Nz = By.shape[0], Bz.shape[1], Bx.shape[2]
    sxy = np.gradient(Bx, dy, axis=1, edge_order=2)
    sxz = np.gradient(Bx, dz, axis=2, edge_order=2)
    syx = np.gradient(By, dx, axis=0, edge_order=2)
    syz = np.gradient(By, dz, axis=2, edge_order=2)
    szx = np.gradient(Bz, dx, axis=0, edge_order=2)
    szy = np.gradient(Bz, dy, axis=1, edge_order=2)
    c = _coeffs_3d(Bx, By, Bz, dx, dy, dz)

    Bxf = np.empty((2 * Nx + 1, 2 * Ny, 2 * Nz))
    Byf = np.empty((2 * Nx, 2 * Ny + 1, 2 * Nz))
    Bzf = np.empty((2 * Nx, 2 * Ny, 2 * Nz + 1))

    offs = ((0, -0.25), (1, +0.25))
    for oy, yb_ in offs:
        yb = yb_ * dy
        for oz, zb_ in offs:
            zb = zb_ * dz
            Bxf[0::2, oy::2, oz::2] = Bx + sxy * yb + sxz * zb
            Bxf[1::2, oy::2, oz::2] = c["a0"] + c["ay"] * yb + c["az"] * zb
    for ox, xb_ in offs:
        xb = xb_ * dx
        for oz, zb_ in offs:
            zb = zb_ * dz
            Byf[ox::2, 0::2, oz::2] = By + syx * xb + syz * zb
            Byf[ox::2, 1::2, oz::2] = c["b0"] + c["bx"] * xb + c["bz"] * zb
    for ox, xb_ in offs:
        xb = xb_ * dx
        for oy, yb_ in offs:
            yb = yb_ * dy
            Bzf[ox::2, oy::2, 0::2] = Bz + szx * xb + szy * yb
            Bzf[ox::2, oy::2, 1::2] = c["c0"] + c["cx"] * xb + c["cy"] * yb
    return Bxf, Byf, Bzf


def restrict_3d(Bxf, Byf, Bzf):
    Bx = 0.25 * (
        Bxf[0::2, 0::2, 0::2]
        + Bxf[0::2, 1::2, 0::2]
        + Bxf[0::2, 0::2, 1::2]
        + Bxf[0::2, 1::2, 1::2]
    )
    By = 0.25 * (
        Byf[0::2, 0::2, 0::2]
        + Byf[1::2, 0::2, 0::2]
        + Byf[0::2, 0::2, 1::2]
        + Byf[1::2, 0::2, 1::2]
    )
    Bz = 0.25 * (
        Bzf[0::2, 0::2, 0::2]
        + Bzf[1::2, 0::2, 0::2]
        + Bzf[0::2, 1::2, 0::2]
        + Bzf[1::2, 1::2, 0::2]
    )
    return Bx, By, Bz


def verify_constraints_3d(Bx, By, Bz, dx, dy, dz, rng, nsample=4):
    """Numerical check of the 3D constraint set.

    At random points inside every cell, the pointwise divergence of the
    reconstruction polynomial must equal the coarse discrete divergence of
    that cell (a constant per cell).  Returns the max deviation.
    """
    c = _coeffs_3d(Bx, By, Bz, dx, dy, dz)
    Dc = div_3d(Bx, By, Bz, dx, dy, dz)
    worst = 0.0
    for _ in range(nsample):
        x = rng.uniform(-0.5 * dx, 0.5 * dx)
        y = rng.uniform(-0.5 * dy, 0.5 * dy)
        z = rng.uniform(-0.5 * dz, 0.5 * dz)
        div_poly = (
            (c["ax"] + c["by"] + c["cz"])
            + x * (2 * c["axx"] + c["bxy"] + c["cxz"])
            + y * (c["axy"] + 2 * c["byy"] + c["cyz"])
            + z * (c["axz"] + c["byz"] + 2 * c["czz"])
        )
        worst = max(worst, np.abs(div_poly - Dc).max())
    return worst


# ----------------------------------------------------------------------
# Field generators (exact face means, exactly solenoidal by construction)
# ----------------------------------------------------------------------


def faces_from_Az_2d(Az, dx, dy):
    """Exact face means from a node-sampled vector potential A_z.

    Bx = dAz/dy, By = -dAz/dx; the exact mean of Bx over an x-face is the
    difference of Az at the face's end nodes / dy (fundamental theorem of
    calculus), so the coarse discrete div is EXACTLY (machine) zero.
    """
    Bx = (Az[:, 1:] - Az[:, :-1]) / dy
    By = -(Az[1:, :] - Az[:-1, :]) / dx
    return Bx, By


def faces_from_edges_3d(EIx, EIy, EIz, dx, dy, dz):
    """Exact face means from exact edge integrals of a vector potential.

    EIx[i,j,k] = int A.dl on the x-edge (cell i, nodes j,k), etc.
    Face flux = circulation around the face (Stokes) => discrete div of the
    fluxes telescopes to machine zero.
    """
    Phix = EIy[:, :, :-1] - EIy[:, :, 1:] + EIz[:, 1:, :] - EIz[:, :-1, :]
    Phiy = EIz[:-1, :, :] - EIz[1:, :, :] + EIx[:, :, 1:] - EIx[:, :, :-1]
    Phiz = EIx[:, :-1, :] - EIx[:, 1:, :] + EIy[1:, :, :] - EIy[:-1, :, :]
    return Phix / (dy * dz), Phiy / (dx * dz), Phiz / (dx * dy)


def analytic_Az_2d(x, y):
    """Smooth 2-mode vector potential (2D)."""
    return np.sin(2 * np.pi * x + 0.9) * np.cos(2 * np.pi * y + 0.5) / (
        2 * np.pi
    ) + 0.5 * np.sin(4 * np.pi * x + 0.2) * np.sin(2 * np.pi * y) / (4 * np.pi)


def analytic_faces_2d(N):
    """Exact coarse face means of the analytic 2D field on an N x N grid."""
    dx = dy = 1.0 / N
    xn = np.linspace(0.0, 1.0, N + 1)
    Az = analytic_Az_2d(xn[:, None], xn[None, :])
    Bx, By = faces_from_Az_2d(Az, dx, dy)
    return Bx, By, dx, dy


def analytic_faces_3d(N):
    """Exact face means of a smooth solenoidal 3D field on an N^3 grid.

    Uses A = (f(y,z), g(z,x), h(x,y)); each component is constant along its
    own direction, so exact edge integrals are pointwise values * edge
    length, and exact face means follow from the circulation.
    """
    d = 1.0 / N
    xn = np.linspace(0.0, 1.0, N + 1)

    def f(y, z):
        return np.sin(2 * np.pi * y + 0.3) * np.cos(2 * np.pi * z + 0.1) / (2 * np.pi)

    def g(z, x):
        return np.sin(2 * np.pi * z + 0.6) * np.cos(2 * np.pi * x + 0.2) / (2 * np.pi)

    def h(x, y):
        return np.sin(2 * np.pi * x + 0.9) * np.cos(2 * np.pi * y + 0.5) / (2 * np.pi)

    EIx = f(xn[None, :, None], xn[None, None, :]) * d * np.ones((N, 1, 1))
    EIy = g(xn[None, None, :], xn[:, None, None]) * d * np.ones((1, N, 1))
    EIz = h(xn[:, None, None], xn[None, :, None]) * d * np.ones((1, 1, N))
    Bx, By, Bz = faces_from_edges_3d(EIx, EIy, EIz, d, d, d)
    return Bx, By, Bz, d


def linear_faces_2d(N):
    """Exact face means of a linear solenoidal field (mean = centroid value)."""
    dx = dy = 1.0 / N
    xn = np.linspace(0.0, 1.0, N + 1)
    xc = 0.5 * (xn[1:] + xn[:-1])
    # Bx = 1 + 2x + 3y ; By = 4 + 5x - 2y  (div = 2 - 2 = 0)
    Bx = 1.0 + 2.0 * xn[:, None] + 3.0 * xc[None, :]
    By = 4.0 + 5.0 * xc[:, None] - 2.0 * xn[None, :]
    return Bx, By, dx, dy


def linear_faces_3d(N):
    d = 1.0 / N
    xn = np.linspace(0.0, 1.0, N + 1)
    xc = 0.5 * (xn[1:] + xn[:-1])
    # div = 1 + 5 - 6 = 0
    Bx = (
        1.0
        + 1.0 * xn[:, None, None]
        + 2.0 * xc[None, :, None]
        + 3.0 * xc[None, None, :]
    )
    By = (
        2.0
        + 4.0 * xc[:, None, None]
        + 5.0 * xn[None, :, None]
        + 6.0 * xc[None, None, :]
    )
    Bz = (
        3.0
        + 7.0 * xc[:, None, None]
        + 8.0 * xc[None, :, None]
        - 6.0 * xn[None, None, :]
    )
    return Bx, By, Bz, d


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


def run_2d(rng, results):
    N = 24
    dx = dy = 1.0 / N

    # (a) machine-zero fine div for a RANDOM solenoidal coarse field
    Az = rng.standard_normal((N + 1, N + 1))
    Bx, By = faces_from_Az_2d(Az, dx, dy)
    scale = max(np.abs(Bx).max(), np.abs(By).max()) / min(dx, dy)
    Dc = np.abs(div_2d(Bx, By, dx, dy)).max()
    Bxf, Byf = prolong_2d(Bx, By, dx, dy)
    Df = np.abs(div_2d(Bxf, Byf, 0.5 * dx, 0.5 * dy)).max()
    _check("(a) coarse div (random, rel)", Dc / scale, 1e-13, results)
    _check("(a) fine div (random, rel)", Df / scale, 1e-13, results)

    # shared-face two-sided consistency of the reconstruction
    ts = two_sided_face_check_2d(Bx, By, dx, dy)
    _check("    shared-face 2-sided diff (rel)", ts / np.abs(Bx).max(), 1e-13, results)

    # (b) restriction o prolongation == identity
    Bxr, Byr = restrict_2d(Bxf, Byf)
    idr = max(np.abs(Bxr - Bx).max(), np.abs(Byr - By).max())
    _check("(b) R(P(.)) - I (random, rel)", idr / np.abs(Bx).max(), 1e-13, results)

    # (d) exactness: uniform and linear solenoidal fields
    Bxu = np.full((N + 1, N), 1.7)
    Byu = np.full((N, N + 1), -0.6)
    Bxfu, Byfu = prolong_2d(Bxu, Byu, dx, dy)
    eu = max(np.abs(Bxfu - 1.7).max(), np.abs(Byfu + 0.6).max())
    _check("(d) uniform exactness (abs)", eu, 1e-13, results)

    BxL, ByL, dx, dy = linear_faces_2d(N)
    BxfL, ByfL = prolong_2d(BxL, ByL, dx, dy)
    BxE, ByE, _, _ = linear_faces_2d(2 * N)
    el = max(np.abs(BxfL - BxE).max(), np.abs(ByfL - ByE).max())
    _check("(d) linear exactness (rel)", el / np.abs(BxE).max(), 1e-13, results)

    # (c) convergence order vs analytic smooth solenoidal field
    errs = []
    Ns = (16, 32)
    for Nc in Ns:
        Bx, By, dx, dy = analytic_faces_2d(Nc)
        Bxf, Byf = prolong_2d(Bx, By, dx, dy)
        BxE, ByE, _, _ = analytic_faces_2d(2 * Nc)
        e2 = np.sqrt(
            np.mean(
                np.concatenate([(Bxf - BxE).ravel() ** 2, (Byf - ByE).ravel() ** 2])
            )
        )
        errs.append(e2)
    order = np.log2(errs[0] / errs[1])
    results.append((f"(c) L2 err  N={Ns[0]}->{2 * Ns[0]}", f"{errs[0]:.3e}", "", ""))
    results.append((f"(c) L2 err  N={Ns[1]}->{2 * Ns[1]}", f"{errs[1]:.3e}", "", ""))
    _check_order("(c) convergence order", order, 1.7, results)

    return errs, order


def run_3d(rng, results):
    N = 12
    d = 1.0 / N

    # (a) random solenoidal field from random exact edge integrals
    EIx = rng.standard_normal((N, N + 1, N + 1)) * d
    EIy = rng.standard_normal((N + 1, N, N + 1)) * d
    EIz = rng.standard_normal((N + 1, N + 1, N)) * d
    Bx, By, Bz = faces_from_edges_3d(EIx, EIy, EIz, d, d, d)
    scale = max(np.abs(Bx).max(), np.abs(By).max(), np.abs(Bz).max()) / d
    Dc = np.abs(div_3d(Bx, By, Bz, d, d, d)).max()
    Bxf, Byf, Bzf = prolong_3d(Bx, By, Bz, d, d, d)
    Df = np.abs(div_3d(Bxf, Byf, Bzf, d / 2, d / 2, d / 2)).max()
    _check("(a) coarse div (random, rel)", Dc / scale, 1e-13, results)
    _check("(a) fine div (random, rel)", Df / scale, 1e-13, results)

    # numerical verification of the 3D constraint set (docstring, C1-C4)
    dev = verify_constraints_3d(Bx, By, Bz, d, d, d, rng)
    _check("    constraint set check (rel)", dev / scale, 1e-13, results)

    # (b) identity
    Bxr, Byr, Bzr = restrict_3d(Bxf, Byf, Bzf)
    idr = max(np.abs(Bxr - Bx).max(), np.abs(Byr - By).max(), np.abs(Bzr - Bz).max())
    _check("(b) R(P(.)) - I (random, rel)", idr / np.abs(Bx).max(), 1e-13, results)

    # (d) uniform and linear exactness
    Bxu = np.full((N + 1, N, N), 0.4)
    Byu = np.full((N, N + 1, N), 1.3)
    Bzu = np.full((N, N, N + 1), -2.2)
    Bxfu, Byfu, Bzfu = prolong_3d(Bxu, Byu, Bzu, d, d, d)
    eu = max(
        np.abs(Bxfu - 0.4).max(), np.abs(Byfu - 1.3).max(), np.abs(Bzfu + 2.2).max()
    )
    _check("(d) uniform exactness (abs)", eu, 1e-13, results)

    BxL, ByL, BzL, d = linear_faces_3d(N)
    BxfL, ByfL, BzfL = prolong_3d(BxL, ByL, BzL, d, d, d)
    BxE, ByE, BzE, _ = linear_faces_3d(2 * N)
    el = max(
        np.abs(BxfL - BxE).max(), np.abs(ByfL - ByE).max(), np.abs(BzfL - BzE).max()
    )
    _check("(d) linear exactness (rel)", el / np.abs(BxE).max(), 1e-13, results)

    # (c) convergence
    errs = []
    Ns = (8, 16)
    for Nc in Ns:
        Bx, By, Bz, d = analytic_faces_3d(Nc)
        Bxf, Byf, Bzf = prolong_3d(Bx, By, Bz, d, d, d)
        BxE, ByE, BzE, _ = analytic_faces_3d(2 * Nc)
        e2 = np.sqrt(
            np.mean(
                np.concatenate(
                    [
                        (Bxf - BxE).ravel() ** 2,
                        (Byf - ByE).ravel() ** 2,
                        (Bzf - BzE).ravel() ** 2,
                    ]
                )
            )
        )
        errs.append(e2)
    order = np.log2(errs[0] / errs[1])
    results.append((f"(c) L2 err  N={Ns[0]}->{2 * Ns[0]}", f"{errs[0]:.3e}", "", ""))
    results.append((f"(c) L2 err  N={Ns[1]}->{2 * Ns[1]}", f"{errs[1]:.3e}", "", ""))
    _check_order("(c) convergence order", order, 1.7, results)

    return errs, order


def save_error_map_png(outdir):
    """Optional: error map of the prolonged Bx for the 2D analytic case."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    N = 32
    Bx, By, dx, dy = analytic_faces_2d(N)
    Bxf, _ = prolong_2d(Bx, By, dx, dy)
    BxE, _, _, _ = analytic_faces_2d(2 * N)
    err = np.abs(Bxf - BxE)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.pcolormesh(
        np.linspace(0, 1, 2 * N + 1),
        np.linspace(0, 1, 2 * N + 1)[:-1] + 0.25 / N,
        err.T,
        cmap="Blues",
        shading="nearest",
    )
    fig.colorbar(im, ax=ax, label=r"$|B_x^{prolonged} - B_x^{exact}|$")
    ax.set_xlabel("x (fine x-face centers)")
    ax.set_ylabel("y")
    ax.set_title(f"T0.1 2D prolongation error, coarse N={N}")
    path = os.path.join(outdir, "t0_1_error_map_2d.png")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main():
    rng = np.random.default_rng(20260806)
    results2, results3 = [], []
    run_2d(rng, results2)
    run_3d(rng, results3)
    print_table("T0.1 2D Cartesian div-free prolongation", results2)
    print_table("T0.1 3D Cartesian div-free prolongation", results3)
    outdir = os.path.dirname(os.path.abspath(__file__))
    png = save_error_map_png(outdir)
    if png:
        print(f"\nerror map: {png}")
    print("\nT0.1: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
