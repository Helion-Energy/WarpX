#!/usr/bin/env python3
"""Divergence-preserving grid remap for cylindrical (RZ) hybrid-PIC states.

Continuing a hybrid-PIC run on a finer mesh only requires two things,
because of what the model actually stores:

  * the magnetic field is the sole field state -- the electric field is
    re-derived from the generalized Ohm's law every step and the current
    from curl(B) plus the particle deposit, so neither has to be carried;
  * macro-particles are Lagrangian, hence grid-independent: the same
    particle list is a valid state on any mesh covering the same domain.

So the remap reduces to prolonging B onto the refined mesh and handing the
particle list over untouched.  The only delicate part is that the
prolongation must preserve div(B) = 0 *discretely*.  A component-wise
interpolation does not: it injects a divergence error that the field
advance then transports and amplifies.

Method
------
For an axisymmetric cylindrical Yee mesh the divergence of a cell is a
face-flux balance (there is no theta contribution when d/dtheta = 0):

    div(B) * V = 2 pi dz [ r_out B_r(out) - r_in B_r(in) ]
                 + pi (r_out^2 - r_in^2) [ B_z(top) - B_z(bottom) ]

Refining radially by an integer ratio, the prolongation is built so that
this balance is inherited exactly:

  1. B_r on r-faces that already exist is copied.  Those faces are
     geometrically unchanged, so their flux is unchanged.
  2. B_z on each z-face is split across the sub-annuli by an
     area-weighted reconstruction whose *flux sum is exactly the coarse
     flux* -- a slope may be applied, but it is then corrected so the
     conservation identity holds to roundoff.
  3. B_r on newly created r-faces is not interpolated at all.  It is
     *solved for*, by imposing div(B) = 0 on each new sub-cell, sweeping
     outward from the innermost face.

Step 3 is what makes the result exact: if the coarse cell satisfied the
balance and step 2 conserved the z-fluxes, then imposing zero divergence
on all but the outermost sub-cell forces it in the outermost one as well.
The prolonged field is therefore divergence-free to roundoff wherever the
coarse field was, independent of how smooth it is.

B_theta is cell-centred and does not enter the axisymmetric divergence at
all, so it is interpolated on its own (piecewise-constant by default,
optionally linear in r); the choice cannot affect div(B).

This module is deliberately free of simulation-framework imports so it can
be unit-tested on synthetic fields.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "CylindricalMesh",
    "divergence",
    "prolong_b",
    "refine_mesh",
]


class CylindricalMesh:
    """Axisymmetric Yee mesh descriptor.

    Cells are indexed ``(i, j)`` with ``i`` radial and ``j`` axial.  Field
    component shapes follow the staggering:

    ==========  ===================  ==============================
    component   location             shape
    ==========  ===================  ==============================
    ``B_r``     r-node, z-centre     ``(nr + 1, nz)``
    ``B_z``     r-centre, z-node     ``(nr, nz + 1)``
    ``B_t``     cell centre          ``(nr, nz)``
    ==========  ===================  ==============================
    """

    def __init__(self, nr, nz, r_min, r_max, z_min, z_max):
        if r_min < 0.0:
            msg = "cylindrical mesh cannot start at negative radius"
            raise ValueError(msg)
        self.nr = int(nr)
        self.nz = int(nz)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

    @property
    def dr(self):
        return (self.r_max - self.r_min) / self.nr

    @property
    def dz(self):
        return (self.z_max - self.z_min) / self.nz

    @property
    def r_nodes(self):
        """Radii of the r-faces, shape ``(nr + 1,)``."""
        return self.r_min + self.dr * np.arange(self.nr + 1)

    def __repr__(self):
        return (
            f"CylindricalMesh(nr={self.nr}, nz={self.nz}, "
            f"r=[{self.r_min}, {self.r_max}], z=[{self.z_min}, {self.z_max}])"
        )


def refine_mesh(mesh, ratio_r=2, ratio_z=1):
    """Return the mesh refined by the given integer ratios."""
    if ratio_z != 1:
        msg = (
            "only radial refinement is implemented; axial refinement needs "
            "the z-face split and the same divergence sweep applied in z"
        )
        raise NotImplementedError(msg)
    if ratio_r < 1:
        msg = "refinement ratio must be >= 1"
        raise ValueError(msg)
    return CylindricalMesh(
        mesh.nr * ratio_r, mesh.nz * ratio_z,
        mesh.r_min, mesh.r_max, mesh.z_min, mesh.z_max,
    )


def divergence(mesh, b_r, b_z):
    """Discrete div(B), one value per cell, shape ``(nr, nz)``.

    This is the face-flux balance divided by the cell volume, i.e. exactly
    the operator the staggered field advance conserves.  Returned in
    ``T/m`` so that meshes of different spacing are directly comparable.
    """
    r_n = mesh.r_nodes
    r_in = r_n[:-1][:, None]
    r_out = r_n[1:][:, None]
    dz = mesh.dz

    flux_r = 2.0 * np.pi * dz * (r_out * b_r[1:, :] - r_in * b_r[:-1, :])
    flux_z = np.pi * (r_out**2 - r_in**2) * (b_z[:, 1:] - b_z[:, :-1])
    volume = np.pi * (r_out**2 - r_in**2) * dz
    return (flux_r + flux_z) / volume


def _split_bz(mesh, b_z, ratio):
    """Split B_z across sub-annuli, conserving each z-face's flux exactly."""
    r_n = mesh.r_nodes
    r_in = r_n[:-1][:, None]
    r_out = r_n[1:][:, None]
    coarse_area = r_out**2 - r_in**2                      # / pi

    fine = np.empty((mesh.nr * ratio, b_z.shape[1]), dtype=b_z.dtype)
    sub_area = np.empty((mesh.nr * ratio, 1))
    dr_f = mesh.dr / ratio
    for k in range(ratio):
        a = r_in + k * dr_f
        b = r_in + (k + 1) * dr_f
        sub_area[k::ratio] = (b**2 - a**2)
        fine[k::ratio, :] = b_z                            # piecewise constant

    # Enforce the conservation identity to roundoff regardless of the
    # reconstruction used above: sum_k area_k * Bz_k == area_c * Bz_c.
    acc = np.zeros_like(b_z)
    for k in range(ratio):
        acc += sub_area[k::ratio] * fine[k::ratio, :]
    target = coarse_area * b_z
    # distribute the (zero, for piecewise-constant) residual by area share
    residual = target - acc
    total_area = np.zeros_like(coarse_area)
    for k in range(ratio):
        total_area += sub_area[k::ratio]
    for k in range(ratio):
        fine[k::ratio, :] += residual / total_area
    return fine


def prolong_b(mesh, b_r, b_z, b_t, ratio=2, theta_linear=False):
    """Prolong ``(B_r, B_z, B_theta)`` onto a radially refined mesh.

    Returns ``(fine_mesh, B_r, B_z, B_theta)``.  ``B_r`` on pre-existing
    faces is copied, ``B_z`` is split conservatively, and ``B_r`` on new
    faces is solved from the zero-divergence condition, which makes the
    result divergence-free to roundoff wherever the input was.
    """
    expected = (
        (mesh.nr + 1, mesh.nz),
        (mesh.nr, mesh.nz + 1),
        (mesh.nr, mesh.nz),
    )
    for name, arr, shape in zip(("B_r", "B_z", "B_theta"),
                                (b_r, b_z, b_t), expected):
        if arr.shape != shape:
            msg = f"{name} has shape {arr.shape}, expected {shape}"
            raise ValueError(msg)

    fine = refine_mesh(mesh, ratio_r=ratio)
    bz_f = _split_bz(mesh, b_z, ratio)

    # B_theta plays no part in the axisymmetric divergence.
    if theta_linear and mesh.nr > 1:
        centres = np.arange(mesh.nr) + 0.5
        fine_centres = (np.arange(fine.nr) + 0.5) / ratio
        bt_f = np.empty((fine.nr, mesh.nz), dtype=b_t.dtype)
        for j in range(mesh.nz):
            bt_f[:, j] = np.interp(fine_centres, centres, b_t[:, j])
    else:
        bt_f = np.repeat(b_t, ratio, axis=0)

    # Solve for B_r on every fine face by sweeping outward and imposing
    # zero divergence on each sub-cell.
    br_f = np.empty((fine.nr + 1, mesh.nz), dtype=b_r.dtype)
    br_f[0, :] = b_r[0, :]
    r_f = fine.r_nodes
    dz = mesh.dz
    for i in range(fine.nr):
        r_in = r_f[i]
        r_out = r_f[i + 1]
        d_bz = bz_f[i, 1:] - bz_f[i, :-1]
        # 2 dz (r_out Br_out - r_in Br_in) + (r_out^2 - r_in^2) dBz = 0
        numer = 2.0 * dz * r_in * br_f[i, :] - (r_out**2 - r_in**2) * d_bz
        if r_out == 0.0:
            br_f[i + 1, :] = 0.0
        else:
            br_f[i + 1, :] = numer / (2.0 * dz * r_out)
    return fine, br_f, bz_f, bt_f
