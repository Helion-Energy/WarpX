# Preconditioning the hlld conservative-form theta-implicit MHD system

Decision (2026-08-05): the physics preconditioner comes BEFORE campaign
runs. Benchmark for assessment: the 0.5 B0 mirror squeeze ramp-and-hold
(10 us ramp / 20 us hold, RZ 128x512, 13,250 steps,
`RZ-mhd-drive/run_te_mirror_ramphold.sh`); unpreconditioned baseline
wall time / Newton / GMRES statistics recorded from the 2026-08-05 run.
Target: >= 10x wall-time reduction at identical physics (the paired
pc/no-pc CI harness asserts solution equality, not just convergence).

## What exists (scoping verified 2026-08-05)

`MHDBlockPC` + `MHDWaveOp` precondition the OLD E-based state
(E, rho, M, U_e): edge curl-curl `[I + d_* curlcurl]` field block,
momentum-side small-flow wave Schur `M - h^2 c_s^2 grad div M +
(h^2/mu0 rho) W_B M` (acoustic + magnetosonic + Alfven, centered,
constant-coefficient), algebraic drho/dU_e recovery, triangular
E<->M couplings, fixed-cycle MLMG throughout (plain GMRES stays
legal). Hard-blocked under hlld (`ThetaImplicitMHD.cpp` Define guard),
barotropic-only, no RZ metric anywhere, LinOpBC aborts on Open,
zero-writes unknown blocks (would be singular on the hlld state).

## Design constraints (Chacon takeaways, settled 2026-08-04)

- Upwinded advection lives INSIDE the PC only; the residual stays the
  smooth hlld fan.
- Schur parabolization keeps (B.grad) outermost (null-space property:
  no average effect along field lines).
- Fixed MG cycle counts, smoother bottoms, zero initial guess ->
  stationary PC, plain (non-flexible) GMRES stays valid.

## Staged port plan

Stage 0 — instrumentation (cheap, do first):
  newton.diagnostic_file on the benchmark; per-step Newton iters +
  cumulative GMRES iters as baseline columns. Re-run a short segment
  (e.g. 500 steps spanning ramp onset) if the full-run file is absent.

Stage 1 — non-singular block PC with signal diffusion (the 80% win):
  - Extend Define/Apply to the hlld block layout: B (faces) + rho, M,
    U_e (+ E_i or U_par/U_perp) cell-centered. IDENTITY on any block
    without an operator — never zero.
  - Cell-centered blocks: scalar "signal-diffusion" Helmholtz
    `[I + theta dt S_ref (h/2)(-lap)]` per block via MLABecLaplacian
    (natively RZ-capable in AMReX; the ablastr Poisson path is the
    precedent), S_ref = domain-reference fast speed + |u|_ref. This
    stands in for the O(theta dt |S|/h) HLLD upwind dissipation — the
    stiffest near-wall Jacobian content (scoping item 4).
  - Momentum additionally gets the MHDWaveOp wave Schur ONLY in 1D
    (where its kernels are valid); RZ waits for Stage 2 metrics.
  - B block Stage-1 treatment: identity + resistive scalar option per
    component where centering permits; the ideal-induction coupling is
    represented by the triangular Faraday corrector
    `bB - theta dt curl(dM/rho_ref x B_ref)` on faces.
  - LinOpBC: map Open -> Dirichlet(homogeneous) inside the PC (PC-only
    approximation; the residual keeps the true Green's BC).

Stage 2 — RZ wave Schur (the physics win at scale):
  - RZ (m=0) metric in the momentum wave Schur: port MHDWaveOp's
    W_B + grad-div forms to cylindrical components (1/r d_r(r .),
    -u_t/r, +u_r/r geometric terms exactly as the CGL work-term
    stencils in ThetaImplicitMHD.cpp already do), rediscretized per MG
    level; keep the exact 3x3 cell-block Jacobi smoother.
  - Reference-field selection: FRC domain-mean B_z is near zero
    (reversed core) — use RMS(|B|) or the external/vacuum field as
    B_ref instead of the mean (scoping gotcha), possibly a radial
    profile of separable references.
  - Upwinded advection blocks (first-order donor) on rho/M/U_e/E_i in
    the PC operators.

Stage 3 — B-side option + Hall (only if Stage 2 falls short):
  - Face-based Alfven/fast Schur on dB (custom MLLinOp a la MHDWaveOp;
    AMReX MLCurlCurl is edge-based and cannot be reused — spike says
    custom operator).
  - Whistler fourth-order Schur (MGSI two coupled second-order
    systems) when the Hall term joins hlld — the blueprint is the
    electron-inertia port plan.

## Assessment protocol

On the benchmark deck, pc off vs on: identical diags (bit-check +
books), wall time, total GMRES iters, worst-step time through the
ramp onset. Also rerun the 4 existing E-based PC CI pairs (must stay
green) and the full implicit suite.
