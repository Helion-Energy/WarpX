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

* ``boundary.open_bc_coarsening`` (default 4): interior linear size of the
  conservative source binning ahead of the precomputed-kernel GEMV. The
  bins are GRADED under a multipole acceptance criterion -- a bin whose
  outermost node is a distance d from the face may not be wider than d/4,
  so bins within a few cells of the open face are single nodes (exact) and
  only grow to the requested coarsening in the interior. Each bin carries
  its monopole (total ring current) and first (r, z) moments, and the
  kernel stores :math:`[G_0, \partial G_0/\partial r',
  \partial G_0/\partial z']` per (ghost-point, bin) pair. The dipole terms
  cancel the leading error of monopole-only binning, giving the
  Barnes-Hut/Salmon-Warren :math:`O((h/d)^2)` far-field error law
  (h = bin size, d = distance to the boundary), which the grading extends
  up to sources sitting against the face.
* ``boundary.open_bc_image_sum_rtol`` (default 1e-6, must be > 0) and
  ``boundary.open_bc_max_images`` (default 200, must be >= 1): for
  periodic z the kernel is assembled with a tolerance-driven image-ring
  sum (the per-image psi tail decays algebraically, ~1/n^3), at setup time
  only.

Cost: free space is translation-invariant in z, so the kernel is stored
per radial bin as a table over the psi-point/bin z-offset rather than as a
dense matrix -- O(n_radial_bins * nz) reals in total, e.g. ~40 MB at
512 x 2048 with the default coarsening (the dense form would be ~10 GB).
Each application is one small GEMV whose rows are distributed over the MPI
ranks, followed by an allreduce of the ~nz-long psi vector (alongside the
existing source-moment reduction).

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
* ``test_rz_open_bc_greens_ring_nearwall``: near-filament ring 4 cells
  from the open face at the worst in-bin phase. Guards the graded
  near-face binning: with uniform coarsening the face-adjacent bins
  violate the multipole acceptance criterion and the test fails by an
  order of magnitude (measured 1.1e-1 vs 1.3e-2 graded).
* ``test_rz_open_bc_greens_ring_offcenter``: near-filament ring at the
  worst in-bin phase of a 4x4 interior bin, 18 cells from the face.
  Guards the dipole (first-moment) machinery: monopole-only binning fails
  the test by ~3x (measured 8.8e-2 vs 1.4e-2 with the dipole terms).
* ``test_rz_open_bc_greens_transient_sub20`` / ``..._sub80``: the same
  mid-transient diffusion problem at two substep counts. Guards the
  TEMPORAL coupling (the ghost fill must run inside every RK stage, per
  the design): per-stage evaluation makes the transient
  substep-independent (measured ~3e-13 relative difference), while the
  forbidden lagged variant (ghosts frozen over a substep, which makes the
  boundary partially reflecting for wave-dominated fields) measures ~4e-6;
  the assert sits at 1e-7, well inside the seven-decade gap.
* ``test_rz_open_bc_greens_btheta``: Gaussian axial current column. The
  relaxed B_theta must satisfy :math:`\mu_0 I_{enc}(r)/(2\pi r)` including
  at the outermost cells against the open face, and stay z-uniform.
* ``test_rz_open_bc_greens_btheta_isoz``: same column with PEC z faces,
  exercising the isolated-z kernel branch (no image sum) and the
  non-periodic B_theta corner clamp of the ghost fill.

Measured on the ring test grid (32x64): open-BC wall-band error 4.9e-3
vs analytic (PEC baseline: 2.1, i.e. a ~440x image suppression), and the
wall-band error vs interior coarsening factor is
3.8e-3 / 3.7e-3 / 4.9e-3 / 5.4e-3 for C = 1 / 2 / 4 / 8 -- the grading
keeps the error near the discretization floor because the bin size is
capped by the distance-to-face criterion regardless of C.

Limitations (phase 1)
---------------------

RZ geometry with m = 0 only; hybrid-PIC solver only; single level; ``open``
allowed on the r_hi face only; the domain must include the axis. The fill
applies to the evolved (plasma-response) field: loading applied fields via
``warpx.B_ext_grid_init_style`` is rejected with an abort (a curl-free
field written into the evolved B would be erased at the open face and
drive a spurious wall current sheet) -- use the hybrid solver's split
external fields (``hybrid_pic_model.add_external_fields``), which provide
their own ghost values. E-field ghosts at the open face are not filled
(the Ohm advance does not consume them).

z faces: periodic z is the validated configuration (all psi-path tests).
Non-periodic z faces run with an isolated-z (free-space) kernel; the
toroidal-field path is covered by ``btheta_isoz``, but the poloidal ghost
fill with conducting z faces omits the z-wall image response, so its
accuracy near the corners is not validated -- treat open r_hi with
non-periodic z as experimental for poloidal fields.

Planned follow-ups: the resistive ring-decay convergence test (design
test 2) and the whistler/fast-pulse reflection measurement at the open
face (the temporal per-stage coupling that controls reflection is guarded
by the transient substep-invariance test above).
