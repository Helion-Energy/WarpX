# QDSMC electron energy equation — 2nd order + conduction: campaign handoff

**Date:** 2026-08-04 · **Owner:** Eric Clark · **Status:** Thrusts A, B, C
complete and validated; C.7/G3a measurements and Thrust D are next.

This document lets a fresh session (or colleague) resume the campaign
without the original conversation. The full plan with all measured results
and the decision log is `QDSMC_ENERGY_2ND_ORDER_PLAN.md` next to this file —
read its Phase-0 results block and decision log first; this handoff is the
operational summary.

---

## 1. What this campaign is

Raise the QDSMC electron energy equation (PR #6982, in BLAST development
since 26.08) from first order to **second order in space and time**, and add
**Ito-process tensor thermal conduction** (anisotropic, flux-limited, with a
vacuum fast-front policy). All work is research-branch only; upstream
packaging is explicitly deferred (Eric, 2026-08-04).

## 2. Where everything lives

| Thing | Location |
|---|---|
| Work branch | `qdsmc_energy_leapfrog` @ Helion-Energy/WarpX (pushed) |
| Worktree | `~/src/WarpX-qdsmc` (off development @ d72f49d70 = 26.08) |
| Venv | `~/.env/warpx-qdsmc` (pywarpx pip-installed from worktree `build/`) |
| Build config | 2D, OMP, MPI=ON (openmpi5), openPMD (parallel HDF5 /usr/local/hdf5), EB=ON, QED=OFF |
| Plan + results + decisions | `Docs/qdsmc_energy/QDSMC_ENERGY_2ND_ORDER_PLAN.md` (this branch) |
| Harness | `Examples/Tests/ohm_solver_electron_energy_eq/convergence/` (see its README) |
| Results artifact (G3 report) | claude.ai artifact "QDSMC conduction — G3 gate" (Eric's account) |

Every run needs
`LD_LIBRARY_PATH=/usr/local/openmpi5/lib:/usr/local/hdf5/lib OMP_NUM_THREADS=2`.
Rebuild:
`PATH=/usr/local/openmpi5/bin:$PATH /usr/bin/cmake --build build -j 8 --target pip_install`.

## 3. Commit map (all pushed)

| Commit | Content |
|---|---|
| `296df8406` | B0: markers homed at grid nodes, unique seam/periodic ownership — kills the chi_num = dx²/(4dt) box-filter diffusion (E6) |
| `bc3cde26f` | Thrust A: `qdsmc_time_advance = euler | leapfrog | pc` bake-off switch; euler = #6982 bit-identical control; Pe FillBoundary ghost fix |
| `75dce1b08` | pc selected as default; B1 half-gradient-corrected deposit (spatial slope 0.74 → 1.92) |
| `b4d4672a7` | Bugfix: Evolve re-entry no longer wipes evolved Te/Pe (energy-eq path emits Pe from current Te at `HybridPICInitializeRhoJandB`) |
| `7eee78f40` | Thrust C: Ito tensor conduction substep (quiet GH daughters, grid kernel) |
| (this commit) | Docs + harness scripts committed for handoff |

Controls: `qdsmc_time_advance = euler` + `qdsmc_gradient_deposit = 0` +
kappa parser unset ⇒ recovers #6982 bit-for-bit.

## 4. State of the physics (headline numbers)

**Transport (advection) — 2nd order in space and time.**
- pc advance: O(dt²) (euler−pc isolates euler's O(dt) at slope 1.04;
  candidates agree at slope 1.88 after staggering-template subtraction);
  best stiff-Joule budget (−0.16% at 50× eta vs euler +3.9%).
- B1 gradient deposit: translate spatial slope **1.92** (was 0.74), 25×
  error cut at N=128; conservation exact (correction sums to zero).
- At-rest identity: relL2 3e-6 over 64 steps (was 3–7e-2 pre-B0).

**Conduction (Thrust C) — G3 PASSED.**
- Aligned-B Gaussian spread, parabolic refinement (dt ∝ dx²): slope
  **1.95** at npts=3, **1.93** at npts=2 (4.8× constant — cheap production
  default is viable). σ² growth exact (apparent 0.28% deficit = wrapped-
  moment estimator artifact; exact field scores identically).
- Tilted 30° full tensor: slope ~2.0; spurious χ⊥,num/χ∥ = **1.4e-4** at
  N=64 falling ~N⁻² (raw moment 1.7e-3 is the estimator floor — subtract
  the exact-field score).
- Correction-off control: slope 0.00, χ inflated 1.432× at ALL N — the
  daughter half-gradient correction is **mandatory** (hat-remap variance
  floor, conduction analogue of E6).
- Σ(Te) conserved to 2–3e-8 in every conduction run. CI matrix 9/9 after
  everything.

## 5. Architecture crib (what was built where)

`Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.{H,cpp}`:
- `AdvanceElectronEnergyQDSMC` (euler/leapfrog drivers; pc returns early),
  `AdvanceElectronEnergyQDSMC_PC` (called from
  `WarpXPushFieldsHybridPIC.cpp` BETWEEN the two B half-pushes, before the
  rho LinComb — B^{n+1/2} in register, rho_fp_temp still rho^n),
  `QdsmcTransportOnce`, `ApplyQdsmcEnergySources` (Strang halves),
  `QdsmcConductionOnce` (Thrust C), `ApplyQdsmcPeExtrapolation` (leapfrog
  only, before final E-solve).
- Conduction is a **grid kernel, not particles**: per owned node (same
  seam/periodic ownership trim as `InitParticles`), build u = 3/2 n kB Te,
  spawn quiet GH daughters along (b, e⊥1, e⊥2) with Ito drift
  [div D + D·∇ln n]dt, hat-deposit with source-node MC-limited slopes of u
  (B1 form), SumBoundary, recover Te = u/(3/2 kB n). Strang bracket
  C(dt/2)·[S A S]·C(dt/2) in pc/leapfrog, Lie in euler.
- Knobs (`hybrid_pic_model.*`): `qdsmc_kappa_par(n,Te,t)` [enables; n m⁻³,
  Te eV, κ W/(m·K)], `qdsmc_kappa_perp(n,Te,t)` [default 0],
  `qdsmc_conduction_quadrature_points = <par> [<perp>]` (GH 2..7 hardcoded,
  probabilists'), `qdsmc_conduction_flux_limit_factor` (0.1; ≤0 off),
  `qdsmc_conduction_max_hop` (2 cells; p=4 power soft-min — do NOT revert
  to harmonic, it biases χ ~20% at cap/4), `qdsmc_conduction_vacuum_fast_front`.

`Source/Fluids/QdsmcParticleContainer.{H,cpp}`: node-homed markers (B0),
slope attributes + MC limiter in `SetK`, `GatherVAtMidpoint` (RK2),
gradient-corrected `DepositScalar`.

## 6. Traps (will bite again)

1. **Host atomics aren't atomic**: `Gpu::Atomic::AddNoRet` is plain `+=` on
   CPU. Never OMP-thread a deposit loop whose reach crosses tiles
   (racing NaN → floor() → wild index → segfault).
2. **Evolve re-entry**: `HybridPICInitializeRhoJandB` runs at
   step==step_begin of EVERY `Evolve()` entry (each `sim.step()` segment).
   Fixed for Te/Pe; anything else initialized there has the same exposure.
3. **Whistler stability of test decks**: uniform-B harness decks need
   ω_wh(k_max)·dt_sub ≲ 0.1 (B0 = 2e-5 T for the standard deck; invariant
   under dt ∝ dx² sweeps). At B0 = 0.01 T machine-eps curl E seeds destroy
   the run by step ~4.
4. **Esirkepov + hybrid segfaults on multi-cell crossings** — keep ion and
   marker CFL < 0.6 in every sweep point.
5. **Leapfrog state ends at t−dt/2** — any fixed-time comparison needs the
   0.5·dt·∂tT offset handled (template-subtract). Leapfrog also leaks
   staggering into integer-time energy budgets (−17% joule closure): use pc.
6. **PICMI bucket writes**: conduction/advance knobs are not picmi kwargs —
   write `pywarpx.hybridpicmodel.*` (and `add_new_attr` for parser-named
   keys) AFTER `initialize_inputs()`; verify in `warpx_used_inputs`.
7. **Moment estimators on periodic boxes lie at the 1e-3 level** (wrapped
   tails). Always score the exact solution with the same estimator and
   subtract, or trust the L2-vs-exact metric.
8. **dt refinement at fixed dx cannot converge** for remap-every-step
   schemes; use parabolic (conduction) or fixed-CFL (advection) sweeps,
   or same-discretization A/B differences.

## 7. Open work queue (in order)

1. **C.7 / G3a measurements** on the Thrust-C prototype:
   - Sharma–Hammett ring test (hot patch on circular field lines,
     χ⊥=0): measures the curved-field-line leak χ⊥,num ~ χ∥(l_hop/R_c)²
     (trap 11) vs hop size and npts — the uniform-B tilted test cannot see
     this term. Needs a new deck (circular B from a line current or
     analytic A_θ).
   - Hop-cap transport deficit: capped run vs reference, quantify the
     slowdown when the cap engages.
   - Zeldovich nonlinear front (κ ~ T^{5/2} parser) vs self-similar
     solution; limiter off/on; front speed ≤ f·v_te when limited.
   - **Regime survey needs REAL deck numbers from Eric**
     (liftoff/annulus/compression dx, dt, n, B, R_c) —
     `regime_survey.py` rows are placeholders.
   - Gate G3a: choose explicit escalation (field-line-following hops →
     subcycling) only if measurements demand; elliptic parallel solve is
     LAST RESORT (Eric: avoid elliptic in the explicit advance at all costs).
2. **Thrust D — boundary conditions** (domain + EB): daughter/marker
   specular reflection (adiabatic, default), isothermal T_wall re-emission
   with wall-flux tally, prescribed-flux injection; replace the E7 clamp;
   EB via level-set reflection ([[eb-bc-design-preferences]]: check the reference algorithm
   first).
3. **Deferred fixes recorded, not started**: Te not checkpointed
   (FlushFormatCheckpoint has zero hybrid energy-eq fields — real restart
   bug, same family as re-entry); RZ support for conduction (guarded off);
   external-field B in b̂ (TODO comment in QdsmcConductionOnce).
4. **Combined G1 space-time demonstration** on a run with sources — nice-to-
   have for the eventual PR text.
5. **Upstream packaging** — deferred until Eric calls it; when slicing,
   strip leapfrog (per A.0), keep euler control + pc default, and the
   standalone bugfixes (re-entry, Pe ghosts, Te checkpoint) can go as their
   own small PRs.

## 8. Reproduce the G3 numbers

```bash
cd ~/src/WarpX-qdsmc/Examples/Tests/ohm_solver_electron_energy_eq/convergence
~/.env/warpx-qdsmc/bin/python3 run_conduction.py            # all four sections
~/.env/warpx-qdsmc/bin/python3 run_ci_matrix.py --outdir ci_matrix_check
```

(run_conduction.py caches per-case npz under `cond_out/` — delete to force
re-runs. Full sweep ~15 min on 2 threads/case.)
