.. _examples-tests-open_bc_greens:

Green's-function open boundary for the RZ hybrid-PIC field solver
==================================================================

These tests exercise the OPEN (free-space) field boundary condition for the
RZ (axisymmetric, m = 0) hybrid-PIC / Ohm's-law B-field advance, selected
with ``boundary.field_hi = open ...`` on the radial face together with
``algo.maxwell_solver = hybrid``.

Method
------

By default the radial boundary of the hybrid solver acts as a perfect
conductor: the ghost values seen by the curl stencils carry image currents,
which is non-physical when the field sources (e.g. coils) are inside the
domain. The open BC instead fills the r_hi ghost values of B with the
free-space field of the interior sources:

* **Poloidal field (Br, Bz)**: the poloidal flux psi = r A_theta is
  evaluated at the ghost psi-points with the axisymmetric ring-current
  Green's function,

  .. math::

     \psi(x_b) = \int_\Omega G_0(x_b; x')\, J_\theta(x')\, \mathrm{d}A', \qquad
     G_0 = \frac{\mu_0}{4\pi} \sqrt{(z-z')^2 + (r+r')^2}\,
           \left[(2-k^2)K(k) - 2E(k)\right],

  with :math:`k^2 = 4 r r' / ((z-z')^2 + (r+r')^2)`, and the ghost B is
  produced by differencing psi **exactly as the interior Yee stencil
  would**. This keeps the ghost-region field discretely divergence-free by
  construction (constrained-transport principle), and corner ghosts are
  filled from the same psi table so adjacent faces agree identically.

* **Toroidal field (B_theta)**: Ampere's law. With no poloidal current
  outside the domain, :math:`r B_\theta` is constant beyond the face, so the
  ghosts continue :math:`r_c B_\theta` from the outermost valid cell --
  exactly :math:`\mu_0 I_{enc} / (2\pi r)`, with no image sum needed.

The source is the total azimuthal current
:math:`J_\theta = (\nabla\times B)_\theta / \mu_0` (plasma + external),
evaluated with the same discrete curl as the Ohm's-law solver, so the ghost
fill is an instantaneous *linear* map of the B state at the same time
level. It is applied through ``WarpX::ApplyBfieldBoundary``, i.e. inside
every RHS/stage evaluation of the subcycled B integrator (RK4/RKF45), and
therefore inherits the integrator's order with no added iteration.

Error controls
--------------

* ``boundary.open_bc_coarsening`` (default 4): the source is conservatively
  binned onto coarse cells of this linear size before the precomputed-kernel
  GEMV. Each bin carries its monopole (total ring current) and first (r, z)
  moments, and the kernel stores :math:`[G_0, \partial G_0/\partial r',
  \partial G_0/\partial z']` per (ghost-point, bin) pair. The dipole terms
  cancel the leading error of monopole-only binning, giving the
  Barnes-Hut/Salmon-Warren :math:`O((h/d)^2)` far-field error law
  (h = coarse-cell size, d = distance to the boundary) while the kernel
  stays a fixed precomputed matrix of a few MB.
* ``boundary.open_bc_image_sum_rtol`` (default 1e-6) and
  ``boundary.open_bc_max_images`` (default 200): for periodic z the kernel
  is assembled with a tolerance-driven image-ring sum (the per-image psi
  tail decays algebraically, ~1/n^3), at setup time only.

Tests
-----

All tests hold a prescribed external current fixed in a resistive vacuum
(Hall and electron-pressure terms disabled) and let the explicit hybrid
solver relax B resistively to the magnetostatic steady state
:math:`\nabla\times B = \mu_0 J_{ext}` consistent with the boundary
condition, mirroring the analytic verification-suite structure of the
PIXIE3D resistive-wall BC paper (arXiv:2606.05446):

* ``test_rz_open_bc_greens_ring``: Gaussian ring of azimuthal current. The
  relaxed field must match the analytic free-space (z-periodic) loop field
  -- reconstructed in the analysis by superposing exact loop fields of the
  discrete source at full resolution and Yee-differencing psi like the code
  -- to discretization + coarse-graining order, globally and in a
  near-wall band. The same metric evaluated on the PEC baseline
  (``test_rz_open_bc_greens_ring_pec``) measures the image-suppression
  factor of the open boundary.
* ``test_rz_open_bc_greens_btheta``: Gaussian axial current column. The
  relaxed B_theta must satisfy :math:`\mu_0 I_{enc}(r)/(2\pi r)` including
  at the outermost cells against the open face, and stay z-uniform.

Measured on the ring test grid (32x64): open-BC wall-band error 5.4e-3
vs analytic (PEC baseline: 2.1, i.e. a ~400x image suppression), and the
wall-band error vs coarsening factor is 3.8e-3 / 3.7e-3 / 5.4e-3 / 1.6e-2
for C = 1 / 2 / 4 / 8 -- at the discretization floor through C = 2-4 and
growing ~:math:`h^2` beyond, as the Barnes-Hut error law predicts.

Limitations (phase 1)
---------------------

RZ geometry with m = 0 only; hybrid-PIC solver only; single level; ``open``
allowed on the r_hi face only (z faces may be periodic or any non-open BC);
the domain must include the axis. The fill applies to the evolved
(plasma-response) field; split external fields provide their own ghost
values. E-field ghosts at the open face are not filled (the Ohm advance
does not consume them). Planned follow-ups: the resistive ring-decay
convergence test (design test 2) and the whistler/fast-pulse reflection
test at the open face.
