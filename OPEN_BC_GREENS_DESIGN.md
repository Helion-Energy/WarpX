# Open (free-space) field boundary via Green's-function coupling
## Design for an independent WarpX PR, testable on the explicit hybrid solver

(2026-07-30. Motivated by the Stage-3 finding that a PEC wall near
in-domain coils produces a non-reciprocal image response — see
INTERNAL-REF_WALL_TREATMENT.md Sec. 6 — but the deliverable is general: a
free-space boundary for the RZ hybrid/MHD field advance, matching the
internal-ref vacuum-region closure. The same limitation affects the EXPLICIT
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
internal-ref and by hetools CurrentLoop),

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
  shorted filament ports in the circuit (INTERNAL-REF_WALL_TREATMENT.md Sec.
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
