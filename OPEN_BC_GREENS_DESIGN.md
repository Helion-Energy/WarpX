# Open (free-space) field boundary via Green's-function coupling
## Design for an independent WarpX PR, testable on the explicit hybrid solver

(2026-07-30. Motivated by the Stage-3 finding that a PEC wall near
in-domain coils produces a non-reciprocal image response — see
an internal wall-treatment note — but the deliverable is general: a
free-space boundary for the RZ hybrid/MHD field advance, matching the
reference vacuum-region closure. The same limitation affects the EXPLICIT
hybrid solver, so this lands as its own PR with explicit-solver tests,
before the implicit MHD/circuit work consumes it.)

## 1. What "open" means for this field system

The hybrid/MHD field advance in RZ (m=0) evolves B by Faraday from the
Ohm's-law E. The poloidal field is equivalent to the flux function
psi = r A_theta (B_r = -(1/r) dpsi/dz, B_z = (1/r) dpsi/dr), sourced by
the azimuthal current J_theta; the toroidal field B_theta is sourced by
the poloidal current. An OPEN boundary means the ghost values seen by
the curl stencils at the domain edge are those of the free-space field
of the interior sources — no image currents, no reflected flux:

    psi_ghost(x_b) = Int_domain G0(x_b; x') J_theta(x') dV',

with G0 the axisymmetric ring-current kernel (the same one used by
hetools CurrentLoop),

    G0 = (mu0/4pi) sqrt((z-z')^2 + (r+r')^2) [(2-k^2)K(k) - 2E(k)],
    k^2 = 4 r r' / ((z-z')^2 + (r+r')^2),

and, for the toroidal component, Ampere's law directly:
B_theta,ghost = mu0 I_pol,enc(z_b) / (2 pi r_b).

This is the time-stepping analog of a von Hagenow/James free-boundary
closure: we do not have an elliptic solve to piggyback on (the field is
advanced by Faraday, not solved), so the boundary values are produced by
direct source integration each time they are applied.

## 2. Discretization and cost control

- Ghost fill: for each ghost point of B_r (nodal r, cell z) and B_z
  (cell r, nodal z) on the open faces, evaluate psi at the two
  psi-points of its Yee difference and difference exactly as the
  interior stencil would — the ghost B is then discretely consistent
  with a free-space psi extension (same construction as the
  DivFreeFieldLoaderRZ psi-differencing, which keeps div B = 0).
- Direct kernel: K[N_ghost-psi-points x N_source-cells], applied as a
  GEMV per boundary application. Full-resolution K is too large
  (~GB); instead COARSE-GRAIN the source: restrict J_theta to a 4x-8x
  coarsened source grid (far-field smoothness makes the boundary error
  fall as (d_cell/d_boundary)^2 of the coarsening cell); kernel is then
  a few MB and the GEMV is microseconds on GPU. Coarsening operator =
  conservative r-weighted binning (sum of J dV per coarse cell at its
  centroid). Phase-2 option if ever needed: FMM / hierarchical kernels.
- z-topology: Phase 1 keeps the existing option space — for periodic z,
  use the periodically-imaged kernel (image sum converges geometrically;
  ~10 images at |Delta z| = L_z); for open z faces, the same machinery
  applies verbatim. This also creates the path to lifting the implicit
  MHD solver's periodic-z restriction.
- Frequency of application: every place the current BC is applied today
  (ApplyBfieldBoundary — per substep in the explicit BfieldEvolve
  subcycles, per residual evaluation in the implicit solve). The map is
  LINEAR and smooth in the state (J = curl B), so matrix-free Jacobian
  probes see it exactly; JFNK-safe by construction.

## 3. WarpX integration points

- New boundary handling under `Source/BoundaryConditions/`:
  `GreensFunctionOpenBC` (RZ, hybrid-family solvers first). Selected by
  `boundary.field_hi_r = open` (the `FieldBoundaryType::Open` enum value
  already exists for the electrostatic MLMG path; this PR extends its
  meaning to the FDTD/hybrid B advance in RZ rather than adding a new
  token).
- Hook: `WarpX::ApplyBfieldBoundary` (and the E-side tangential ghost
  fill where the Ohm solve needs it) dispatches to the Green's fill when
  the face is Open and the solver is hybrid/implicit-MHD.
- Precompute at init (and on regrid): coarsening operator + kernel
  matrices per open face, stored on device.
- The implicit MHD Define() asserts (PEC at r_hi, periodic z) relax to
  accept Open faces once the BC exists; the fluid reflecting wall
  becomes an outflow/vacuum-contact question that stays SEPARATE (Phase
  1 keeps the fluid boundary reflecting; the dust annulus means the
  fluid never dynamically reaches the boundary anyway).

## 4. Independent PR + explicit-solver tests

PR scope: the BC machinery + explicit-hybrid RZ tests only (no implicit
MHD, no circuit code). Tests (CTest, <30 s, CPU/GPU portable):
1. Static ring field: a prescribed toroidal current ring held in the
   domain (external current path of the hybrid solver); with Open BC the
   grid field must match the analytic loop field at interior probe
   points to discretization order, and show NO wall-image contribution
   (compare against the same run in a 2x larger domain: results must
   agree to the coarse-grain tolerance rather than differ by the image
   term).
2. Resistive decay of a current ring (explicit hybrid, resistive Ohm):
   the decay must be free of reflected-flux artifacts; convergence of
   the boundary error with coarsening factor.
3. B_theta open face: poloidal current column -> B_theta = mu0 I/(2 pi r)
   at the boundary, exact to quadrature.
Acceptance: image-term suppression >= 100x vs PEC at matched geometry;
no change to any existing test (BC off by default).

## 5. What this buys the circuit coupling

With true open faces the wall image disappears at the root: free-space
Green's reciprocity becomes VALID on the grid, so
- the reciprocity probe Int A^unit . J_p dV and the disk-flux probe
  agree (each within discretization), and either can drive the back-EMF;
- coil-coil and coil-plasma mutuals computed from geometry (hetools
  CurrentLoop / the planned coil-lattice reader) match the discrete
  model without calibration;
- physical conducting walls, when wanted, are added back as EXPLICIT
  shorted filament ports in the circuit (internal note, Sec.
  5/7) with geometry-computed mutuals — passive by construction — or as
  resistive-shell fluid regions, but never as an implicit image of the
  computational boundary.

Interim (until the PR lands): the implicit-MHD two-way coupling uses the
model-consistent disk-flux back-EMF (Faraday on the coil contour from
the model's own fields) + bank stray inductance, which is passive and
stable with the PEC wall in place — validated in RZ-mhd-drive.

## 6. Temporal coupling: how the closure enters the time advance
## (scoping addendum, 2026-07-30 — REQUIRED reading for implementation)

The Green's closure is an INSTANTANEOUS LINEAR CONSTRAINT,
psi_ghost = K . J_theta[B], not an evolution equation. That determines
the correct temporal treatment in each solver:

- IMPLICIT (theta-implicit MHD / hybrid): evaluate the ghost fill inside
  every residual evaluation. Newton then converges boundary and interior
  self-consistently; the map is linear and smooth, so matrix-free
  Jacobian probes see it exactly. No extra iteration machinery.

- EXPLICIT hybrid (THIS PR): evaluate the ghost fill INSIDE EVERY
  RHS/STAGE EVALUATION of the subcycled B integrator (each RK4/RKF45
  stage, and each Euler substep), i.e. treat it as part of the
  right-hand side like any state-dependent closure in method of lines.
  The RK stages already constitute the predictor-corrector structure, so
  this inherits the integrator's native order with NO added iteration;
  because the closure is linear in the state, a within-stage fixed-point
  loop would converge in one pass and is therefore pointless.
  DO NOT implement the lagged variant (ghosts filled once per substep
  from the pre-substep state) as the default: it makes the boundary a
  partially reflecting surface with reflection ~ O(substep CFL) for
  wave-dominated fields. It may exist as an option only for the
  reflection-comparison test below.

- Cost: with the coarse-grained source the fill is a small GEMV
  (~tens of MB kernel); ~4 stages x O(10) substeps x 2 halves = O(100)
  fills/step is negligible on GPU and acceptable on CPU test grids. If
  profiling ever disagrees, the documented escalation is a split
  refresh (far-field rows once per substep, near-boundary rows per
  stage) — not iteration.

- REQUIRED TEST (add to the test set): wave reflection at the open face.
  Launch a compact whistler/fast pulse toward the boundary; measure the
  reflected amplitude relative to the incident pulse, and compare
  against (a) the PEC baseline (~full reflection) and (b) the lagged
  Tier-0 variant. Acceptance: per-stage evaluation reflects <~ 1%;
  report the Tier-0 number for the record.

## 7. Axial (z) cap faces (implemented 2026-08; extends Sec. 2 "the same
## machinery applies verbatim")

Motivation: the hetools free-boundary GS 2x2 showed the z-cap treatment
is load-bearing -- a Neumann cap (the z-invariant-current asymptote)
biased every coil-held racetrack solve to a ballooned attractor, while
the Green's-matched cap recovered it. The WarpX time-domain analogs were
the frozen loader ghosts (explicit hybrid) and the per-solve Neumann
fill of ThetaImplicitMHD. `boundary.field_lo/hi[1] = open` (RZ hybrid)
now selects the Green's fill for the caps; any combination of r_hi,
z_lo, z_hi works.

- Toroidal closure (the z-face analog of the r-face r*Bt continuation):
  outside the cap J = 0, so d(r B_theta)/dz = mu0 r J_r = 0 at fixed r
  -- B_theta(r, z_ghost) = B_theta(r, z_lastvalid). This IS Ampere's
  law with the enclosed axial-current profile I_z(r) frozen at its
  boundary-plane value; for m = 0 with no radial current beyond the cap
  it is not an approximation. Exactness is asserted by
  test_rz_open_bc_greens_btheta_zcaps_picmi (ghost row == valid plane,
  bitwise).

- psi-table cap rows: the shared psi table (nodal points, one value per
  point regardless of which face's fill reads it) gains rows at ALL
  interior radii, i in [0, nr-1], j in the two nodal cap ghost bands
  [-ngz, 0] and [nz, nz+ngz]. The i >= nr corner columns are NOT
  duplicated: they live in the r_hi band (which always spans the full
  j range), so the corner ghosts of the r-face and cap fills difference
  the SAME psi values -- corner-ghost div B is machine zero by
  construction (measured 2.5e-18 relative at 32x32).

- Cap-row kernel layout (the decision Sec. 2 left open): the cap rows
  evaluate at a fixed nodal-j band while the source bins span all of z,
  so the r-band's z-offset translation tables buy nothing for them.
  Chosen: a DENSE block of nr * (ngz+1) rows per open cap by 3*nbins
  columns, aligned with the source-moment vector, with the monopole +
  dipole moments evaluated at each bin's actual (r_c, z_c) center --
  preferred over widening the translation tables to all interior radii
  (O(n_radial_bins * nr * nz), strictly larger and it re-imposes the
  nominal-center offset convention on rows that do not need it). Open z
  requires non-periodic z, so no image sum is involved. Measured cost
  at the production hold resolution (128 x 512, coarsening 4): cap
  kernel 18.5M reals = 141 MB (vs 2.7 MB for the r-band tables),
  one-time assembly +0.9 s (OMP host), steady per-application cost
  3.1 ms -> 9.4 ms per fill (8 ranks x 4 threads CPU). The memory
  scales as nr^2 nz / coarsening^2; the escalation path if it ever
  binds is radial aggregation of far bins for cap rows (Barnes-Hut in
  r), not FMM.

- Source support and the boundary-plane singularity: the j = 0 plane
  node STAYS in the curl-B deposit (its stencil reads the cap ghost row,
  i.e. the previous application's fill -- a per-stage-lagged contraction
  in the sense of Sec. 6). Dropping it was tried and REJECTED: a real
  boundary-plane current (the plugged racetrack's rotating
  open-field-line column crossing the cap) then becomes invisible to
  the fill while the plane-row curl and Ohm E keep acting on it -- an
  unstable feedback channel (observed boundary-plane runaway). Keeping
  it exposes the OTHER hazard: plane source nodes coincide with cap psi
  points and the filament kernel is log-singular there (K(m -> 1); raw
  entries reached 5e9 and the fill's own ghost-row seed exploded at
  ~e^15 per application). The cap kernel therefore regularizes sub-cell
  separations as an equivalent filament of minor radius
  a = min(dr,dz)/2 (bounding the self entry at the standard
  mu0 r (ln(8r/a) - 2) loop self-inductance scale). The r_hi band needs
  neither: its face node i = nr stays excluded, which is what keeps the
  translation tables singularity-free. Restarts are safe either way:
  checkpoints store ghost cells (VisMF), so the first post-restart
  deposit reads exactly the pre-checkpoint fill values.

- Explicit-path stability at plasma-crossed caps: with real (floor-
  density) plasma crossing an open cap, the explicit RK feedback of the
  per-stage fill admits a grid-scale (odd-even in z) whistler boundary
  layer that frozen ghosts do not (they pin the layer). It is inside
  the reach of the standard hybrid grid-scale dissipation: the plugged
  racetrack hold is clean at plasma_hyper_resistivity = 1e-6 (k^4
  selective; FRC decay time untouched) where 1e-8 ran away in
  ~0.5 t_ci. The theta-implicit path damps it natively (the implicit
  z-open outflow test needs no extra dissipation).

- Grading caveat (recorded, deliberate): the source binning stays
  graded against the r_hi face only. Bins within ~coarsening cells of a
  cap are up to coarsening wide in both directions, so a source
  concentration hugging an open cap is coarse-grained at O(1) in-bin
  phase there. The target physics (end-matched holds; separatrix ends
  away from the caps) puts no current there; if a use case does, the
  z-grading must mirror the radial one (bin-group machinery, kernel
  memory unchanged for the translation tables).

- Ordering: with a z cap open the fill runs at the TOP of
  ApplyBfieldBoundary (it reads only valid data), so the PEC wall and
  axis mirrors -- which read interior cap-ghost values at the corners
  -- see this application's values. With r_hi-only open the original
  tail position is kept bit-for-bit (verified: 32/32 plotfile binaries
  identical across the existing suite).

- Implicit path: the fluid moments keep the Neumann outflow ghosts; the
  field/current z-ghost fills defer to the Green's values on an open
  cap (ApplyNeumannZDomainGhosts keeps the outermost
  `open_face_keep_rows` ghost rows: all of B, the curl-of-Green's-B
  plasma-current row, the cell-centered interpolants' first ghost row,
  which FillCellCenteredElectromagneticFields now computes directly
  from the Green's-filled B). E keeps the plain Neumann clamp: it has
  no Green's counterpart and its cap ghosts feed no residual stencil
  (the ghost-B rows a curl-E write could reach are overwritten by the
  fill). JFNK exactness is preserved: the fill runs inside every
  residual evaluation, same as the r face.

- Particle boundaries at the caps are intentionally untouched
  (absorbing ends remain a separate physics decision), as is the fluid
  Neumann semantics when z is not open. 3D matched caps stay out of
  scope (vector kernel).
