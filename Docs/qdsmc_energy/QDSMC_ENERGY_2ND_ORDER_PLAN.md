# QDSMC Electron Energy Equation: Second-Order Accuracy + Ito Thermal Conduction

Research plan, drafted 2026-08-04. Baseline: `development` @ `d72f49d70` (26.08),
which contains PR #6982 (QDSMC entropy transport, Joule heating, Q_ei relaxation,
multi-species Ohm's law; Belyaev et al., Phys. Plasmas 31, 012902 (2024)).

Working branch for this effort: new branch off `development` (suggested:
`qdsmc_energy_leapfrog`). Do NOT base on `eb_ect_yee_followup`; the EB/ECT work
merges independently. This document stays untracked.

---

## 0. Baseline: what #6982 does and where first-order enters

Per-step sequence (`WarpX::HybridPICEvolveFields`, `HybridPICModel::AdvanceElectronEnergyQDSMC`):

1. Particles pushed to t^{n+1}; deposit rho^{n+1} (into `rho_fp`), J_i^{n+1/2}
   (into `current_fp`). `hybrid_rho_fp_temp` = rho^n, `hybrid_current_fp_temp` = J_i^{n-1/2}.
2. **Energy advance (documented as T_e^n -> T_e^{n+1}):**
   - `QDSMCInitializeUe`: V_e = -(J_plasma - J_i)/rho on nodes, from
     **J_i^{n-1/2}** (`current_fp_temp`, averaging to J_i^n happens LATER),
     J_plasma(B^n), rho^n.
   - `QDSMCInitializeKe`: K_e = T_e n_e^{1-gamma} (nodal, eV-scaled).
   - Markers (one per cell, home = **cell center**, fields **nodal**): gather V_e,
     K_e*N, N with linear weights; **forward-Euler** push x = home + V_e dt
     (clamped at domain walls -> entropy piles up at boundary nodes);
     linear scatter of K*N and N; T_e recovered with n_e^{n+1} from rho_fp.
   - Operator-split sources (Lie, full dt, applied after transport):
     Joule (from J_plasma^n, eta parser, optional T_e-threshold redirect to ions),
     Q_ei (T_i^n from the shape-aware temperature deposit), drag-diffusion ion kick.
   - Emit Pe = n_e k_B T_e; markers reset to home.
3. J_i^n = avg(J_i^{n±1/2}); B pushed n -> n+1/2 -> n+1 (both halves see the
   **same frozen Pe** from step 2); final E-solve at n+1 with extrapolated
   J_i^{n+1} = 3/2 J^{n+1/2} - 1/2 J^{n-1/2} — again with the same Pe.

**First-order (or worse) error sources, enumerated:**

| # | Source | Where |
|---|--------|-------|
| E1 | Forward-Euler marker push with beginning-of-interval V_e | `QdsmcParticleContainer::PushX` |
| E2 | V_e built from J_i^{n-1/2} (half-step stale) while B^n, rho^n | `QDSMCInitializeUe` + call order in `HybridPICEvolveFields` |
| E3 | Lie splitting of sources (full-dt kick after transport) | `AdvanceElectronEnergyQDSMC` steps 6–6c |
| E4 | Pe fed to the B-push halves and final E-solve at a single frozen time level | `HybridPICEvolveFields` |
| E5 | Per-step gather/scatter remap with linear shapes: hat-kernel convolution each step -> numerical diffusion ~ dx*|V_e|*(1-CFL)/2 (first order in dx) | `SetK`/`DepositK` |
| E6 | Home = cell center on a **nodal** grid: the at-rest gather is already a 2^D box filter, so the round trip smooths even at V_e = 0 (rate ~ dx^2/dt) | `InitParticles` vs nodal fields |
| E7 | Domain-wall clamp accumulates advected entropy at boundary nodes (uncontrolled BC) | `PushX` clamp |

E6 deserves a targeted check early: verify against Belyaev Sec. III.A whether
markers belong at nodes for nodal fields (at-rest round trip is then the
identity). If confirmed, it is a cheap, self-contained accuracy fix and a
possible upstream bugfix PR on its own.

**What is already right and must be preserved:** the scatter (deposit) form is
conservative in Sigma(K*N) and Sigma(N) by construction — no iteration needed.
This is precisely what the old gather-based layer method (monotone cubic
interpolants, Milstein-style) lacked. The plan keeps the scatter form everywhere
and gets second order by *time-centering the velocity* and *raising the
effective reconstruction order*, not by switching to backward semi-Lagrangian.

---

## Thrust A — Second-order time advance (candidates measured, best selected)

### A.0 Scheme selection by measurement (decided 2026-08-04)

Do not pre-commit: implement the time-centering candidates behind a runtime
switch (dev-only knob, e.g. `hybrid_pic_model.qdsmc_time_advance = euler |
leapfrog | pc`), measure on the shared harness, and select the winner. The
half-staggered leapfrog (A.1) is the on-paper favorite (one transport per
step, sources auto-centered); the predictor–corrector keeps T_e on integer
levels at the cost of a second transport pass.

Candidates:

- **`euler`** — the #6982 scheme, kept as the A/B control.
- **`leapfrog`** — half-staggered T_e^{n+1/2} + midpoint push (A.1–A.3).
  One transport/step; minimal restructuring (J_i LinComb reorder).
- **`pc`** — integer-level predictor–corrector: predictor transports
  T_e^n -> T_e* with V_e^n (midpoint push), emits Pe* for the first B
  half-push; corrector re-transports T_e^n -> T_e^{n+1} from the SAME initial
  markers using V_e^{n+1/2} built from B^{n+1/2}, J_i^{n+1/2}, rho^{n+1/2}
  after the first half-push. Two transports/step; restructures
  `HybridPICEvolveFields` (energy advance split across the B halves), but T_e
  stays at integer levels (diagnostics comparability, no half-level
  checkpoint state, no Pe extrapolation for the final E-solve).

Selection criteria (recorded with numbers, not vibes): measured dt-order on
the manufactured-advection and adiabatic-compression sweeps; conservation;
wall-clock overhead vs ion PIC cost; robustness on a stiff liftoff-like deck
(substepped/RKF45 B advance, Joule hot spots); implementation invasiveness /
restart-state complexity. Losing candidates are removed before the upstream
PR (the switch does not ship, or ships reduced to the winner + `euler`).

### A.1 Candidate `leapfrog`: reinterpret T_e as living at half-integer times

Synchronize the thermodynamic state with the ion PIC staggering
(x at integer, v and J_i at half-integer):

- **T_e^{n+1/2}** is the state variable; the QDSMC advance maps
  T_e^{n-1/2} -> T_e^{n+1/2} over one dt.
- The time-centered advection velocity for that interval is **V_e^n** — built
  from J_i^n = avg(J_i^{n±1/2}), J_plasma(B^n), rho^n. All three are available
  at the existing call site; the ONLY ordering change is to move the J_i
  averaging LinComb **before** the energy advance (fixes E2 for free).
- The existing sources are then *automatically midpoint-centered*: Joule uses
  J_plasma^n, Q_ei uses T_i^n (the temperature deposit is already at t^n).
  This is the elegant payoff of the half-staggering: no source extrapolation
  needed.
- Pe^{n+1/2} emitted by the advance is exactly time-centered for the full
  [n, n+1] B-push — better than today's centering for both halves (fixes E4
  for the halves). The final E-solve at t^{n+1} gets
  **Pe^{n+1} = 3/2 Pe^{n+1/2} - 1/2 Pe^{n-1/2}**, mirroring the existing
  J_i^{n+1} extrapolation (store one previous Pe or Te level, analogous to
  `current_fp_temp`).

### A.2 Midpoint spatial evaluation (fixes E1)

Time-centered V_e^n alone is not enough: the marker must sample it at the
trajectory midpoint. Two-stage push per marker:

```
x_mid  = home + (dt/2) * V_e^n(home)       # predictor half-push
x_new  = home +  dt    * V_e^n(x_mid)      # midpoint full push
```

Two gathers of the same grid field — cheap, no extra deposit, no iteration,
and conservation is untouched (deposit still moves the full carried content).
This is RK2-midpoint on the ODE dx/dt = V_e(x, t), globally O(dt^2) when V_e is
time-centered.

### A.3 Strang-split sources (fixes E3)

Apply half-dt Joule and Q_ei kicks **before** marker load and half-dt after
recovery: S(dt/2) A(dt) S(dt/2). All source operators are grid-local
(cheap); the ion drag-diffusion kick pairs with the Q_ei halves.
**Decided 2026-08-04: implement Strang from the start**; a Lie variant can be
added later as a cost/simplicity follow-on if measurements show the
commutator is negligible.

### A.4 Bookkeeping changes

- **Self-start:** first step advances T_e^0 -> T_e^{1/2} with dt/2 (same
  desynchronization pattern as the PIC velocity half-kick). On restart-from-
  closure the same half-kick applies.
- **Checkpoint/restart:** T_e^{n-1/2} (and the previous-level Pe/Te for the
  extrapolation) become genuine state. Audit against the rho_fp checkpoint gap
  (PR #7049) — T_e appears to NOT be checkpointed today, which is already a
  restart bug for #6982 with the energy equation on. Fix alongside, or fold
  into #7049's follow-up. Test names must not contain "_restart" (CI quirk).
- **Substep/RKF45 interplay:** Pe stays frozen across B substeps within each
  half — unchanged from today, but now the frozen value is time-centered.
  Document that Joule-stiff hot spots may want the substep controller to see
  Pe updates (out of scope; note as known limitation).

### A.4b Predictor–corrector specifics (candidate `pc`)

IMPLEMENTED 2026-08-04 (commit bc3cde26f), and it simplified: the predictor
transport is UNNECESSARY. The first B half-push needs a Pe at its start time
t^n — and the previous step's corrector already left Pe^n in the register, so
`pc` is a SINGLE corrector transport per step, placed between the two B
half-pushes: T_e^n -> T_e^{n+1} with V_e^{n+1/2}(J_i^{n+1/2} = current_fp,
B^{n+1/2} = register mid-push, rho^{n+1/2} averaged in-kernel), midpoint
push, Strang sources (Joule uses J_plasma^{n+1/2} — better centered than
euler's). It must run BEFORE the rho^{n+1/2} LinComb so rho_fp_temp still
holds rho^n for the K_e/N_e load. One extra `CalculatePlasmaCurrent` at
B^{n+1/2} per step. T_e stays at integer levels: no half-level checkpoint
state, no Pe extrapolation. Same transport count as leapfrog — the original
two-transport concern (corrector restart from initial marker state) is moot.

### A.5 Gate G1

- Manufactured advection test (prescribed V_e, frozen fields): L2(T_e) slope in
  dt >= 1.9 at fixed dx (dx chosen fine enough that E5/E6 don't floor the sweep)
  — for BOTH `leapfrog` and `pc`, with `euler` slope ~1 as the control.
- Adiabatic compression test (existing `analysis_adiabat.py` extended to a dt
  sweep): slope >= 1.9.
- Sigma(K*N), Sigma(N) conserved to round-off per step (periodic box).
- **Selection recorded** per A.0 criteria (order, cost, robustness,
  invasiveness); losing candidate stripped before upstreaming.
- Existing three CI tests re-blessed; qei/joule budgets unchanged in physics.

---

## Thrust B — Spatial accuracy of the remap (contingent, measure first)

Phase 0 measures the dx-order and the effective numerical diffusion chi_num of
the current scheme (at-rest test for E6; translating-Gaussian test for E5).
Options, in escalation order:

1. **B0 — node homing (E6 fix):** home markers at nodes so the at-rest round
   trip is the identity. One-line-ish change in `InitParticles` (plus box-
   ownership care at domain edges so shared nodes aren't double-initialized).
   Verify against the Belyaev paper's placement.
2. **B1 — slope-carrying markers (recommended target):** each marker carries
   limited gradients of its content (MUSCL/monotonized-central limited
   reconstruction of K and N at load time); the scatter integrates the linear
   sub-cell profile against the destination hat functions. Conservative by
   construction (integrals of the reconstruction are the carried totals),
   positivity-preserving via the limiter, and second-order in dx for CFL < 1.
   This is the conservative counterpart of the old monotone-cubic layer
   method — same reconstruction philosophy, but on the scatter side, so no
   iteration is needed.
3. **B2 — BFECC/MacCormack correction** on the deposited fields (advect
   forward, back, correct): 2–3x transport cost, second order, needs a limiter
   for positivity. Fallback if B1's limiter proves fussy near the EB.

Decision gate G2: pick B1 or B2 only if the measured dx-order after A+B0 caps
below ~1.7 on the translating-Gaussian and rigid-rotation tests at liftoff-
relevant CFL; otherwise defer (the dt fix may dominate practical error).

**G2 MEASURED 2026-08-04 — trigger TRIPPED, B1 is GO**: post-B0, with a
resolved blob (sigma = 0.08 L, translation vs exact, CFL 0.5, N = 32..192),
the spatial order climbs 0.61 -> 0.86 toward the expected FIRST-order remap
asymptote (the alpha(1-alpha) donor-style diffusion cap). Overall fit 0.74.
Full second order requires Thrust B; B1 (slope-carrying markers: limited
MUSCL reconstruction carried per marker, integrated on scatter) is the
target per the standing preference, with B2 (BFECC) as fallback.

**B1 IMPLEMENTED AND MEASURED 2026-08-04 — spatial slope 1.92** (same sweep;
local orders 1.98 -> 1.86, the tail dip being the MC limiter's standard
first-order clip at the blob extremum; 25x absolute error reduction at
N = 128). Design as-built: the **half-gradient-corrected deposit** — SetK
loads limited MC slopes (per grid dimension, boundary-safe one-sided at
non-periodic edges) of the transported node fields s = K*rho*V/q and
n = rho*V/q; the scatter deposits w*(A + 1/2 G.(x_dest - x_p)). The
correction sums to zero exactly per marker (hat first-moment property =>
conservation unchanged, verified ~1e-8), cancels the hat's second moment
(exact through quadratics), and reduces to the identity at rest (verified
2.8e-6, unchanged from B0). Knob: hybrid_pic_model.qdsmc_gradient_deposit,
default ON; euler + gradient_deposit=false recovers #6982 bit-for-bit.
Simpler than the full sub-cell-integration variant sketched above — the
half-gradient trick achieves the same order with a 2-point stencil deposit.

---

## Thrust C — Ito-process tensor diffusion: electron thermal conduction

### C.1 Physics target

Add heat conduction to the energy equation:
dU_e/dt += div(kappa . grad T_e), with anisotropic
kappa = kappa_par b b + kappa_perp (I - b b), b = B/|B| (from B^n),
kappa_par Spitzer ~ T_e^{5/2} (parser-driven, like eta), kappa_perp with its
own parser. **Decided 2026-08-04: the full tensor ships from day one —
anisotropic thermal conduction is a must-have** (kappa_perp = 0 is just the
trivial parser setting, not a deferred feature).

### C.2 SDE formulation (conservative form)

Work on the **energy density** u = (3/2) n_e k_B T_e during the conduction
substep (NOT on the entropy K — conduction is not adiabatic, and mapping the
Fourier flux into K-space drags in n-gradient corrections; see C.5). With
diffusivity tensor D = chi_par b b + chi_perp (I - b b), chi = kappa/((3/2) n_e k_B):

div(kappa grad T) in Fokker–Planck (Ito, conservative) form for u requires the
drift correction:

```
dX_i = [ d_j D_ij + D_ij d_j ln(n_e) ] dt + (sqrt(2D))_ij dW_j
sqrt(2D) = sqrt(2 chi_par) b b + sqrt(2 chi_perp) (I - b b)
```

(The `grad ln n_e` term converts Fickian diffusion of u into Fourier diffusion
of T; the `div D` term is the standard Ito drift for spatially varying D.
Both evaluated on the grid at t^n and gathered at the marker; n_e floored by
`qdsmc_n_floor` before the log-gradient.)

### C.3 Quiet (deterministic) sampling — no RNG noise

One marker per cell + random kicks = unacceptable noise. Use QDSMC-style quiet
daughters: replace the diffusive displacement by a deterministic Gauss–Hermite
quadrature of the Gaussian, per quadrature direction (the eigendirections of D:
b and the perpendicular pair).

**Number of quadrature points is an input knob, settable per direction.** The
quadrature axes are field-aligned — (b, e_perp1, e_perp2), i.e. the "cardinal"
directions of the quadrature are oriented with the parallel/perpendicular
eigendirections of D — so the parallel and perpendicular counts can differ:
`hybrid_pic_model.qdsmc_conduction_quadrature_points = <npts_par> <npts_perp>`
(scalar broadcasts to both). **Default 2 in each direction** (decided
2026-08-04). Raising npts_par alone buys fast parallel tails where Spitzer
chi_par is large, without paying for perpendicular daughters (see the
stiffness analysis, C.7). Normalized probabilists' GH abscissae/weights
hardcoded in a small table (2..~7 points; sigma = sqrt(2 chi dt) per
direction):

| npts | offsets (units of sigma) | weights | properties |
|---|---|---|---|
| 2 (default) | ±1 | 1/2, 1/2 | variance-exact; per-step 4th-moment defect => conduction operator is **weak order 1 in dt**; cheapest (2 daughters, parallel-only) |
| 3 | 0, ±sqrt(3) | 2/3, 1/6, 1/6 | matches through 5th moment => **weak order 2**; tails to sqrt(3) sigma |
| 4+ | standard GH roots (e.g. 4-pt: ±0.742, ±2.334) | GH weights | tail sampling out to larger multiples of sigma — for high-T / large-hop regimes where the capped Gaussian tail carries real flux |

Notes:
- All GH weights are positive => positivity of the deposited u is automatic at
  every npts.
- The default npts=2 caps the conduction operator's formal dt-order at 1; the
  research target of overall 2nd order is demonstrated at npts=3 (see G3), with
  npts=2 accepted as the cheap production setting. In practice conduction
  errors are subdominant where the limiter engages, and the knob is the escape
  hatch when tails matter (large T_e => large chi => hops comparable to
  gradients).
- Full tensor: tensor-product across eigendirections (npts^d daughters, exact
  covariance) or a sparse per-direction stencil (npts*d daughters, covariance-
  exact with rescaled offsets); default chi_perp=0 keeps it at npts daughters.
- Daughters are ephemeral: spawned at the (already advected) marker position,
  displaced, deposited, discarded. They carry energy content only (conduction
  transports heat, not electrons — do NOT diffuse the N weights).

### C.4 Splitting and recovery

Strang-compatible placement: C(dt/2) . [S(dt/2) A(dt) S(dt/2)] . C(dt/2), or
fold conduction into the same push as advection (single SDE — no A/C splitting
error at all) once the split version is validated. **Decided 2026-08-04:
split substep** (clean physics, independently testable, and the clean seam
for an elliptic parallel backend if C.7 demands one); unify later only if
profiling says so.

Recovery after the conduction deposit: T_e = u_dep / ((3/2) k_B n_e) with n_e
from rho_fp — no weight division, positivity guaranteed by positive quadrature
weights and positive u.

Conservation: Sigma(u) exact to round-off (deposit form + reflecting BCs).

### C.5 Recorded alternative (not first choice)

Unified single-push SDE acting on the entropy markers directly (diffusing K
with chi' and extra n-gradient drift terms derived from K = T n^{1-gamma}).
Saves one deposit pass but couples the limiter and the BCs into the adiabatic
carrier. Explicitly deferred 2026-08-04 (split substep chosen); revisit only
if the split substep is a measured bottleneck.

### C.6 Flux limiting and the vacuum policy (the user-specified knobs)

Two distinct caps on chi_par, applied per cell before the drift/offset build:

1. **Physical free-streaming limiter** (longitudinal only):
   `kappa_eff = kappa_Sp / (1 + |q_Sp| / (f q_fs))`, q_Sp = kappa_Sp |grad_par T_e|,
   q_fs = n_e k_B T_e v_te, flux-limit factor f ~ 0.1–0.3 (input knob).
   Needs grad_par T_e on the grid at t^n — one stencil pass.
2. **Numerical hop cap:** sqrt(2 chi_par dt) <= m_hop * dx (m_hop input, default
   ~2): guarantees daughters stay within the guard/Redistribute reach and keeps
   the deposit local. Where the cap engages, transport is slower than physical —
   accepted by design, exactly like the vacuum-resistivity ceiling turning
   vacuum into a fast-but-finite diffusion front. In floored/vacuum cells
   (n_e <= qdsmc_n_floor) chi is set to the capped ceiling value, not zero,
   so fronts propagate through low-density regions instead of stalling.

Smooth both caps (soft-min) to avoid kinks in div D feeding the drift term.

### C.7 Stiffness analysis — SDE arm vs elliptic parallel solve (added 2026-08-04)

Answers, with measurements, whether the explicit stack can carry Spitzer
parallel conduction or the parallel channel needs an implicit/elliptic solve.
Prior-art datum: the reference algorithm's implicit solver absorbed the full conductivity
stiffness in JFNK without trouble; the question is what the explicit hybrid
stack tolerates.

**Framing — where stiffness actually bites the SDE arm.** The quiet-daughter
kernel is NOT explicit-diffusion-CFL limited: for locally constant
coefficients the one-step Gaussian kernel is exact in time at any chi*dt, so
the classic S = chi_par dt/dx^2 bound does not apply to it. The real limits
are:

1. **Hop-cap engagement**: x_max sqrt(2 chi_par dt) > m_hop dx => transport
   artificially slowed. Accepted-by-design fallback, but the deficit must be
   quantified against reference solutions, not assumed benign.
2. **Geometry variation over the hop**: b changes over the hop length l_hop
   (curvature R_c, shear), so straight-line parallel hops leak cross-field:
   spurious chi_perp,num ~ chi_par (l_hop/R_c)^2. This — not instability —
   is the anisotropy-killer at high stiffness.
3. **Drift-term locality**: div D and grad ln n gathered at the marker are
   assumed constant over the hop.

**Analysis tasks** (before the C implementation hardens):

- **Regime survey** (pure script, no runs): tabulate S = chi_par dt/dx^2,
  l_hop/dx, and l_hop/R_c across the target decks (liftoff, annulus,
  compression) over the expected T_e range with Spitzer T^{5/2}.
- **Pollution measurement**: the ring test (C.8) swept over l_hop and
  (npts_par, npts_perp) => empirical max usable hop before
  chi_perp,num/chi_par crosses threshold; separately measure the hop-cap
  transport deficit.
- **Escalation options, evaluated cheapest-first**:
  - (a) **Anisotropic quadrature + field-line-following hops**: raise
    npts_par (fast tails, per-direction knob in C.3) and replace the straight
    parallel hop with an arclength displacement along the traced field line
    (RK2/RK4 trace of b per daughter). Kills the (l_hop/R_c)^2 leak for a few
    extra B gathers per daughter; stays explicit, conservative, positive.
  - (b) **Subcycling**, two flavors: (i) SDE hop subcycling — repeat smaller
    hops N_sub times per step with its own controller (reuse the B-advance
    substep-controller pattern, minus the #7091 ratchet); cost is N_sub
    deposits plus N_sub remap smoothings. (ii) Grid-form conduction folded
    into the existing RKF45 field advance as an RHS term — then RKF45's
    adaptive controller carries it, but explicit-parabolic stability forces
    N_sub ~ S, so the regime survey's S table answers feasibility directly
    (S ~ 10: fine; S ~ 10^3+: hopeless), and it couples the thermal and field
    substep controllers.
  - (c) **Elliptic parallel solve — LAST RESORT, not scoped up front**
    (decided 2026-08-04: avoid elliptic solves in the explicit advance at all
    costs; scope this only if G3a measurements prove (a)/(b) insufficient):
    backward-Euler (I - dt div(chi_par b b grad)) T^{n+1} = T^n with a
    Guenter-style symmetric anisotropic stencil to bound perpendicular
    pollution; AMReX GMRES with MLMG preconditioning (plain MLABecLaplacian
    is diagonal-beta only, so the b b cross terms need the custom stencil) or
    a JFNK port of the reference algorithm's approach. Perpendicular conduction and the
    flux limiter stay in the SDE arm; the split-substep decision (C.4) makes
    the handover a clean seam. Until G3a, the accepted answer to
    over-stiffness is the capped/limited fast-front transport, not an
    implicit solve.

**Gate G3a (decision)**: stiffness table + pollution/deficit measurements =>
choose the backend, strongly biased explicit: (a), then (a)+(b); (c) is
scoped only if the measurements prove the explicit options insufficient AND
the capped-transport fallback distorts the target physics. Record the
crossover S* either way. If (c) is ever selected, its implementation inserts
as Phase 3b before Phase 4.

### C.7 RESULTS (measured 2026-08-04)

**Regime survey with REAL liftoff deck numbers** (2026-08-04 deck defaults,
eb_ect_yee_followup: dx = 2/N m, dt = 0.01 t_ci(bz_rev=0.5 T) = 2.62e-9 s,
n_annulus = 1.59e20, fill 3e19, floor 7.5e18 m^-3; R_c is ASSUMED — 0.35 m
column scale, 0.10 m pessimistic near-null; annulus/compression rows still
placeholders): S_eff < 4 everywhere up to 3 keV, the f = 0.1 flux limiter
dominates Spitzer above ~10 eV, l_hop(npts=2) <= 2.8 dx at N=128 so the
m_hop = 2 cap barely engages below 1 keV. The raw-stiffness arm of the
question is settled: comfortable.

**Hop-cap transport deficit = the designed soft-min, exactly**
(run_hopcap.py, aligned N=64, npts=3, chi_cap/chi0 swept 16.7 -> 0.128):
chi_meas/chi0 tracks (1+(chi0/chi_cap)^4)^{-1/4} to 3-4 digits at every
point (e.g. 8x over cap: measured 0.1276, predicted 0.1276). No excess
deficit beyond design; capped transport is predictable.

**Curvature leak: the framing-item-2 model chi_perp,num ~
chi_par (l_hop/R_c)^2 is WRONG — ~1000x too optimistic at sigma/dx <~ 1.**
Instrument: qdsmc_wiggle_test.py + run_wiggle.py (fully periodic snaking
field B = B0 (eps sin kz, 0, 1), R_c = (1+eps^2)/(eps k); leak = growth of
Var_w of the exact flux function A = x + (eps/k) cos kz; eps = 0 floor
5e-6). Findings (all pc, chi0 = 1e3, wiggle_out/ + wiggle_sweep.log):
- leak/chi0 ~ eps^2.07, k^0.68, OSCILLATES with sigma/dx (aliasing
  structure, factor ~5 swings), and GROWS as dt shrinks at fixed dx: it is
  a PER-REMAP effect — the E6 remap floor dx^2/(4 dt) reopened by
  curvature — not the per-hop chord term (which is chi^2 dt kappa^2-sized
  and ~1e-3 here).
- Not fixed by quadrature choice (eps=.2, m=2, N=64: npts=2: 0.47,
  npts=3: 0.21, npts=5: 0.53 of chi0) and the B1 half-gradient correction
  barely participates (OFF: 0.26 vs ON: 0.21) — unlike the straight-field
  case where B1 removes the 43% floor. npts=2's zero chord variance NOT
  helping proves the leak is in the deposit/gather chain on curved b, so
  escalation (a)'s field-line-following hops alone will NOT remove it.
- **At the liftoff dimensionless point** (sigma/dx = 0.84,
  kappa dx = dx/R_c = 0.045, N=128): chi_perp_num/chi_par = **1.8 at
  npts=2 (the production default) and 0.56 at npts=3**, vs physical
  chi_perp/chi_par ~ 3e-9 there. Electron heat transport effectively
  isotropizes wherever field lines curve at machine scale; straight-field
  regions are clean (floor 5e-6).
- Breakdown regime mapped: hops spanning >~ 1 radian of field rotation
  (l_hop k >~ 1) fully isotropize (leak 2.3 chi0 at m=6).

**Zeldovich nonlinear front (kappa ~ Te^{5/2} parser)**: N=128, ns=256,
amp 15, chi0(Te0) = 3 vs a 1D flux-conservative reference with the same
T0-floored background: relL2(profile) = 0.13; error structure = core too
hot (+0.5 Te0), front tip too cold (-1.0), precursor pre-heat beyond the
front (+0.24) — the hop tails pre-heat ahead of the nonlinear front and
the MC limiter clips slopes at it; front half-width 11% ahead of
reference; front exponent 0.154 vs reference 0.206 (ZK cold-background
2/9). Conservation -2e-5. **The 13% error is a PLATEAU: N=192 parabolic
(fixed sigma/dx) gives relL2 = 0.134** — grid-sharp fronts are
first-order-limited by the front-foot limiter clip + hop precursor, the
same per-remap class as the curvature leak. Limiter legs (windowed front
speeds; the naive per-step tracker is dx/dt-quantized): unlimited front
runs at 3.75e4 m/s; production f = 0.1 leaves it untouched (20% of
allowance — no distortion of resolved fronts); even f = 0.01 only bites
~30% because q_Sp/q_fs ~ 5e-3 here — the strong-bind front-speed-bound
demonstration needs a dedicated q_Sp >> f q_fs parameter point (open
leg).

**Ring-test campaign (qdsmc_ring_test.py) — kept as the div-D validator,
superseded as the leak instrument.** Its four design iterations each hit a
scatter-form boundary pathology worth recording (they are Thrust-D
previews): (i) a periodic box cannot host phi_hat — the seam b-flip makes
a div-D kick line; (ii) blending b to zhat stirs O(1) when b rotates
within a hop length; (iii) **PEC/dirichlet walls zero rho on the wall
rows, the (default ON) vacuum fast-front then boosts their chi to the hop
cap, and the resulting D cliff drift-kicks row-1 daughters into the wall
where the floored-node recovery skip DELETES the energy from Te
(Te(row+-1) -> 0 in ONE substep, Sigma(Te) -6% even at chi = 0)** — any
Thrust-D wall BC must handle the rho=0-row + fast-front pairing;
(iv) sharp chi switches make the first chi=0 node a one-way accumulator
(receives spill, never re-emits). On the working (walled, annular-chi)
configuration the ring DID validate the drift/chord mean cancellation:
blob mean radius drift 2e-4 L over a full run (chi=0 identity: 1e-7).

**Gate G3a verdict (2026-08-04, pending Eric's sign-off)**: the explicit
SDE arm is validated for straight and gently-curved fields at liftoff
stiffness (S_eff small, cap predictable, conservation machine-level). It
is NOT ready for machine-scale-curved field regions (FRC closed-line
geometry): the remap-borne curvature leak isotropizes heat transport
there. The (a)/(b) escalation as scoped does not address the actual
mechanism; the follow-up is a **curved-deposit fix** (deposit/slope
correction in the field frame, or field-line-coordinate remap) BEFORE any
elliptic consideration. Mechanism pinning and the fix design are the next
scheme work; (c) elliptic remains unscoped.

### C.7b LAYER (GATHER) FORM — the curvature-leak fix (2026-08-05)

Eric's call ("it's just a layer method solution"): replace the scatter
deposit with the Milstein layer/gather transfer,
T^{n+1}(node) = sum_q w_q Interp(T^n)(foot_q). T satisfies the backward
equation with the SAME Ito drift (div D + D grad ln n), so the feet are
node + drift + quadrature offsets; pointwise interpolation carries no
deposit-moment frame, removing the remap leak by construction. Landed as
`qdsmc_conduction_form = scatter | layer` with
`qdsmc_conduction_interp = linear | monocubic | keys` (all interpolating
=> machine-exact at-rest identity; monocubic = MC-limited monotone
Hermite, positive and overshoot-free), `qdsmc_conduction_curved_feet`
(default OFF, see below) and `qdsmc_conduction_conserve_fixup` (global
proportional Sigma(rho T) restore). Gather is race-free => the layer loop
is OMP-threaded, unlike the scatter deposit. Scatter path untouched
(bit-identical defaults). Gauntlet: run_layer.py + layer_gauntlet.log /
layer_followup.log; production arm = **layer + monocubic + straight feet
(+ fixup where fronts are steep)**:

- Aligned parabolic order 1.97 (relL2 7.7e-5 at N=128) — no G3
  regression; monocubic == keys bit-for-bit on smooth data (MC slopes
  reduce to central differences; Catmull-Rom IS the Keys kernel).
- Straight-field floor 3.6e-6 (scatter 4.9e-6); linear-interp arm shows
  chi_par inflated 12% at N=64 (the uncorrected hat-adjoint remap floor,
  as predicted for the control); monocubic/keys read chi_par = 1.000.
- Curvature leak at the eps=0.2 wiggle point: **4.7e-4 vs scatter
  2.07e-1 (440x)**; dt-growth signature FLAT (4.41e-3 -> 4.44e-3 over
  ns 64 -> 256, vs scatter 0.21 -> 0.34): the per-remap mechanism is
  gone. Layer-linear control leaks 7.9e-2 (transfer-operator-borne,
  mechanism confirmed).
- **Liftoff dimensionless point (npts=2): 2.56e-4 vs scatter 1.81 —
  ~7000x**; npts-insensitive (npts=3: 4.0e-4). The chi_par ~ 0.89
  reading there is the z-vs-arc estimator factor (1+eps^2/4)^-2 at
  eps=0.45, not a transport deficit.
- **Curved feet (midpoint rotation) are HARMFUL as implemented**: the
  div-D drift already IS the mean-curvature correction for straight
  chords (Euler-Maruyama consistent pairing, ring-validated), so
  rotating the feet double-counts curvature: systematic cross-field
  drift (adrift -1e-2 to -2.4e-2 L) and 10-80x more leak. Default OFF;
  a future weak-2 scheme must re-derive the drift consistently.
- Zeldovich front: relL2 6.0e-2 without fixup but Sigma drift -3.2%
  (steep front + one-sided limiter clipping = the gather deficit,
  exactly where Eric's old layer method needed iteration). **With the
  global proportional fixup: relL2 4.2e-2, Sigma drift 7.8e-7** — the
  cheap fixup restores conservation AND improves the profile; no
  iteration needed at these scales. Front exponent 0.190 (reference
  0.206, scatter 0.154).
- Conservation elsewhere without fixup: 3e-8 (aligned), ~1e-5 (wiggle).

**G3a verdict amended (2026-08-05)**: the curved-field blocker is
RESOLVED by the layer form; explicit SDE arm now validated including
machine-scale-curved fields. Open: default-form decision (scatter stays
default until Eric flips it), Thrust-D BCs (reflected feet are the
natural gather-form wall treatment), a local (vs global) conservation
fixup if production decks show structured deficits, and the strong-bind
limiter demo.

### C.7c Conservative-scatter alternatives — measured and CLOSED (2026-08-05)

Eric's constraint: strict conservation is non-negotiable for the explicit
arm. Two conservative scatter upgrades were built and measured against
the layer form (knobs kept; run_keys.py, fct_out/):

**Keys deposit** (`qdsmc_conduction_deposit_kernel = keys`): the
interpolating zero-first-and-second-moment cubic-convolution kernel
(Keys 1981, a = -1/2; = Catmull-Rom; 4^d stencil). Conservation exact
(PoU), at-rest identity exact, aligned order 1.97. Curvature leak:
eps=0.2 point 7.9e-2 (2.6x better than hat), liftoff point 9.0e-2 (20x)
— but ~350x above layer-monocubic: with kernel moments identically zero,
the residual is the PUSHFORWARD term (the displacement field is
source-evaluated and varies across the cloud; per-hop, ~ sqrt(dt)), which
no fixed kernel can remove. Zeldovich: relL2 13.3% (no gain — the
plateau is front-foot physics, not kernel moments), front exponent 0.125
(worse than hat), and **undershoot -0.375 T0** at the front foot (2.5% of
the step height, textbook signed-kernel step response) — at a floor-
pinned plasma edge the 1/n Te recovery would amplify this into deeply
negative Te. Note (Eric): the SPH-deficiency framing does NOT apply
(daughters regenerate on the full grid every substep); the ringing is
plain step response, re-created per substep wherever the profile is
grid-sharp.

**Compensated hat / Boris-Book FCT**
(`qdsmc_conduction_compensate = 1`, requires hat + grad_deposit 0): hat
deposit plus an exactly-bookkept per-node covariance tally and one
Boris-Book-limited antidiffusive flux sweep per axis (destination-local
gradients; flux form => conservation exact). Measured: **monotonicity
perfect** (Zeldovich undershoot -3.5e-15), conservation exact, aligned
relL2 1.93e-4 at N=64/ns=128 — better than hat+B1 (3.03e-4) — but the
**curvature leak is UNTOUCHED: liftoff point 1.813 vs hat's 1.812**.
Mechanism, now measured: a cross-field-confined filament IS an extremum,
so the "no new extrema" limiter zeroes the antidiffusive fluxes exactly
at the crest — re-admitting exactly the hat diffusion that constitutes
the leak. Unlimited compensation would equal Keys (composition/
convolution identity) and ring.

**The structural conclusion**: conservative scatter is boxed in by
Godunov exactly at the structure that matters. "Add-then-remove"
(hat + any compensation) fails because removal is forbidden at extrema;
"zero-moment kernel" (Keys) evades the limiter and pays with ringing +
the pushforward wall. The two schemes that work are both "NEVER-ADD":
the layer/gather form (monotone interpolation adds no variance; limiter
only costs local order) and the flux-form conservative semi-Lagrangian
remap (Lin-Rood/SLICE class: limited RECONSTRUCTION = never-add
monotonicity, departure-volume integration = inherent conservation AND
kills the pushforward term). Decision reduced to: layer + LOCAL
conservative fixup (cheap, deficits measured tiny except steep fronts
where the global fixup already restores Sigma exactly) vs the flux-form
remap build (satisfies every constraint inherently; the
multidimensional/tensor-hop departure geometry is the cost). Side
finding: FCT-compensated hat beats hat+B1 on straight fields (1.6x,
monotone, no slope storage) — a candidate default for the scatter form
regardless.

### C.7d Esirkepov flux-form remap — the production path (decided 2026-08-05)

**Decision (Eric, 2026-08-05): layer is OFF the table as the production
method.** The production conduction transfer is the flux-form
conservative semi-Lagrangian remap ("Esirkepov form"):
`qdsmc_conduction_form = fluxform`. Layer and scatter remain in the tree
as measurement controls (layer = the leak reference arm, scatter = the
#6982-compatible control); neither is a production candidate.

**Why this form satisfies every standing constraint at once**
- *Strict conservation, structurally*: the update is
  u_i^{new} = u_i − (F_{i+1/2} − F_{i−1/2}) per axis; Σu telescopes to
  machine regardless of what F is. Limiting, wall handling, and tallies
  modify F and can never break conservation.
- *Never-add monotonicity*: each face flux is the integral of a
  MC-limited PLM reconstruction of u^n over the swept interval, so every
  branch remap is a mean of reconstruction values — bounded by the local
  data. Convex GH-weighted combination of branches preserves the bound.
  (The Godunov box of C.7c does not apply: limiting lives in the
  reconstruction, not in an antidiffusive correction.)
- *Kills the pushforward term*: the swept-interval integral is the exact
  pushforward of the reconstructed density under a displacement field
  interpolated CONTINUOUSLY between nodes — the source-evaluated,
  per-cloud displacement error that no fixed deposit kernel could remove
  (C.7c, Keys) is absent by construction.
- *Thrust D substrate*: face fluxes exist as first-class objects. Wall
  faces default to F = 0 = exact adiabatic (replaces the E7 clamp with
  the correct default BC, no marker reflection needed); isothermal and
  prescribed-flux BCs and the wall heat-flux tally become face-local
  bookkeeping on F.

**Operator layout** (per conduction substep, inside QdsmcConductionOnce,
after the existing Pass-1 cond build; scatter/layer paths untouched):

1. *kin field* (scratch MF, 8 comps, ghosts ceil(max_hop)+3): per node,
   the Ito drift a·dt (same clamped central differences and the same
   node-evaluated pairing as scatter/layer — trap 13 applies: the drift
   already carries the mean curvature, do NOT re-rotate), unit b, and
   sigma_par/perp = sqrt(2 chi dt). Computed on valid nodes, then
   FillBoundary (seam/periodic ghosts). Frame vectors e1, e2 are
   recomputed from b on the fly (deterministic), so the branch
   displacement field disp_q(x) = a dt + xq_par sig_par b
   + xq_perp sig_perp e1 + xq_perp sig_perp e2 is defined at every node
   and interpolates linearly along sweep lines.
2. *Branch loop* over the FULL GH lattice (nq_par x nq_perp x nq_perp;
   no per-node collapse logic — offsets with no grid projection drop out
   automatically because sweeps only read grid components of disp, and
   the collapsed-direction weights sum to 1).
3. *Per branch: dimensionally split 1D conservative remaps* (x-sweep
   then z-sweep in 2D, order alternating per substep call). Per face
   f = (n, n+1): departure point by two Picard iterations of
   s_dep + disp(s_dep) = s_f on the linearly-interpolated disp;
   F = integral of the PLM-MC reconstruction over [s_dep, s_f] (walks at
   most ceil(max_hop)+1 dual cells). Each node's update recomputes both
   of its face fluxes locally (deterministic recomputation = exact
   telescoping, no face scratch, no atomics, race-free => OMP-threaded,
   unlike the scatter deposit). Sequential oblique sweeps retain the
   D_xz cross term exactly at the hop level (mass from p lands at
   p + (dx,dz) after both sweeps).
4. *Accumulate* u_acc += w_branch * u_branch; recovery
   Te = u_acc/(3/2 kB ne) with the same rho pairing and floored-node
   skip as scatter Pass 3.

**Splitting-error ladder** (escalate only if gates demand): alternate
sweep order per substep (default, free) -> average both orderings (2x)
-> Lin-Rood inner-advection cross terms -> unsplit swept-parallelogram
(CSLAM-lite). The wiggle matrix and tilted-30deg tensor legs are the
adjudicators.

**Known traps recorded up front**
- *Map folding*: the remap assumes the per-branch map is monotone along
  the sweep line (d disp/ds > −1). Chi cliffs (vacuum_fast_front) can
  fold it — the hop cap and p=4 soft-min keep smooth decks safe, but a
  fold guard (departure-interval positivity check) must abort/donor-cell
  rather than silently produce negative u. Deferred to the first
  vacuum-facing run; gauntlet decks are smooth.
- *Frame-flip lines* (e1 fallback near b ~ yhat) make disp_q
  discontinuous per branch. In 2D with in-plane b, e1 = yhat exactly and
  projects out — no flip exists, all gauntlet decks unaffected. In 3D,
  fix by sign-continuity along each sweep line before trusting 3D runs.
- *Vacuum semantics match scatter for now*: u flowing into floored nodes
  is dropped by the recovery skip (the known Thrust-D deletion class).
  Flux form makes the eventual fix trivial (zero/reflect the face), but
  v1 reproduces scatter behavior away from walls.
- *Point-value vs dual-cell-average identification* of nodal u is
  O(dx^2) and uniform — does not cap the order-2 target.

**Gates (run with the existing drivers, `--form fluxform`)**
- GF1 *correctness/identity*: uniform-Te preservation to machine;
  eps=0 wiggle floor at the layer level (~4e-6); Sigma(rho Te) drift
  <= 1e-12 (structural — any violation is a bug, not a tolerance);
  Zeldovich undershoot at machine (never-add).
- GF2 *accuracy*: aligned parabolic order >= 1.9 with constant within
  ~2x of layer-monocubic; tilted-30deg order ~2 with chi_perp_num at the
  estimator floor.
- GF3 *the leak* (the reason this form exists): wiggle eps=0.2 and the
  liftoff point (sigma/dx=0.84, dx/R_c=0.045, npts=2) at layer level
  (2.6e-4-class, >= 100x below scatter); dt-growth FLAT.
- GF4 *integration*: Zeldovich relL2 <= layer+fixup (4.2e-2) with Sigma
  at machine and no ringing; CI matrix 9/9 with joule budgets at the pc
  baseline; walltime within ~2x of scatter at npts=2 (expected faster:
  no atomics, threaded sweeps).

Follow-ups staged behind the gates: PPM reconstruction knob if the PLM
constant disappoints vs layer-monocubic; face-level free-streaming
limiting and wall tallies (Thrust D); the advection remap on the same
flux substrate (single branch, disp = V_e dt) as a later unification.

**RESULTS (2026-08-05, three design iterations measured — run_fluxform.py,
fluxform_out/, logs fluxform_gauntlet*.log):**

*Iteration log (each fix was forced by a gauntlet failure):*
1. *Backward face-trace* (departure = face − Picard-traced displacement):
   aligned/floors clean, but (a) wiggle leak 5.7e-2, dt-FLAT, with a
   systematic adrift ~1e-2 — the second sweep evaluated its displacement
   at the destination grid point, re-introducing the source-vs-destination
   cross term through the dimensional splitting; and (b) the ZK front
   fired the map-fold trap exactly as recorded above (chi ~ Te^2.5 cliff
   gives d(disp)/ds < -1): departure intervals crossed, cells overdrawn to
   Te = -4.9e6 K, and the negativity poisoned the advection markers
   (Sigma drift -1.8e-3).
2. *Fold guard (trailing-window max monotonization) + Lin-Rood-style
   pull-back* (later sweeps evaluate their displacement at the
   backward-traced source point of the earlier sweeps): positivity
   restored (min Te +132 K) but the ZK front STARVED (front_exp -0.50,
   relL2 1.73) — a backward Picard trace converges to the NEAREST root of
   a folded map and never finds fast donors whose hop crosses the face,
   and the max-window then discards exactly the crossing mass. Backward
   tracing is the wrong member of the flux-form family at chi cliffs.
3. *Forward donor decomposition (the true Esirkepov construction,
   production form)*: donor dual cells map affinely by the
   face-interpolated (and pulled-back) displacement; face flux = sum over
   the donor window of (mass of image beyond face − pre-move mass beyond
   face). No Picard at faces, no fold guard needed — a folded donor's
   image degenerates to a point and the mass lands where the map says;
   positivity unconditional; zero displacement returns F = 0 exactly.
   Sweeps stay race-free (both neighbors sum the same donor window
   deterministically) and OMP-threaded, no atomics.

*Gate scorecard (v3 = forward donor form):*
- **GF1 PASS**: eps=0 wiggle floor 3.6e-6 (= layer); at-rest/zero-disp
  identity exact by construction; Sigma(rho Te) at the harness floor in
  every run — aligned 3e-8 (identical digits to scatter), wiggle 4e-9,
  liftoff 6e-9, ZK -7.3e-6 vs scatter's own -2e-5..-6e-5 on the same
  deck (the deck's advection front floor, i.e. fluxform is cleaner than
  the exactly-conservative scatter arm there). No fixup knob exists.
- **GF2 PASS**: aligned parabolic orders 1.93/1.96/1.97/1.96 (N=32..128)
  at SCATTER's error constant (7.16e-4 at N=32; 1.6x below layer);
  tilted-30deg orders 1.85/1.94 with raw chi_perp_num/chi0 1.7-2.0e-3 =
  the moment-estimator floor — the split remap carries the full tensor.
- **GF3 OPEN — the one red gate**: wiggle eps=0.2 point 1.69e-2
  (12x below scatter's 0.207, but 36x above layer's 4.7e-4); dt-growth
  mild (1.69e-2 -> 2.65e-2 for 4x steps: sub-linear, mixed
  per-remap/operator character); liftoff point npts=2 1.13e-1 / npts=3
  2.32e-2 (scatter 1.81/0.56, layer 2.56e-4). The pull-back cut the v1
  leak 3.4x and collapsed adrift 9.7e-3 -> -1.7e-4; the residual is the
  remaining splitting error of per-branch oblique hops swept
  axis-by-axis (plus a possible limiter-clipping component at the
  filament crest). Escalation rungs left: limiter-off diagnostic to
  split those two, ordering-average (cheap, likely small), UNSPLIT 2D
  donor images (corner transport — the expected real fix).
- **GF4 PASS**: ZK front relL2 **1.98e-2** — best of the whole campaign
  (layer+fixup 4.2e-2, scatter plateau 1.32e-1) — with front exponent
  0.204 vs reference 0.206, min(Te) pinned at ambient (no undershoot,
  -3e-3 relative), no fixup. CI matrix 9/9 PASS with joule budgets at
  the pc baselines to the decimal; scatter and layer control arms
  reproduce their recorded numbers exactly. Walltime ~3.5x scatter on
  the aligned deck at npts=3 (branch-lattice collapse included; the
  donor window is the next optimization target — GF4's 2x cost goal
  needs one more pass).

*Implementation notes (knobs and internals):* form value `fluxform` in
`qdsmc_conduction_form`; branch kinematics staged per node in a `kin` MF
(drift, b, sigmas), per-branch displacement staged in a `dsp` MF
(grid-dim components, index units, clamped at min(max_hop, 6) cells —
the internal `ff_rmax` bound keeps stencils fixed when decks set
max_hop large to park the chi soft-min cap); global branch-lattice
collapse (zero sigma, or 2D in-plane b => e1 = yhat exactly) replaces
the scatter form's per-node loop collapse; sweep order alternates by
(istep + Strang-half) parity.

### C.8 Gate G3

- 1D Gaussian spread along B parallel to z: sigma^2(t) = sigma_0^2 + 2 chi t to
  second order in dt and dx.
- Same with B tilted 30 deg to the grid (tensor rotation exercised): identical
  spread along b, measure spurious perpendicular spread => effective
  chi_perp,num / chi_par below a set threshold (target <= 1e-3 at N=64,
  refine goal after first measurements).
- Anisotropic ring test (Sharma–Hammett style): hot patch on circular field
  lines, chi_perp = 0 — heat stays on the flux surface.
- Zeldovich nonlinear front (kappa ~ T^{5/2}) against self-similar solution;
  limiter off/on comparison; front speed <= f v_te when limited.
- Energy budget closes to round-off with reflecting BCs.

---

## Thrust D — Boundary conditions (domain + EB), flux and temperature

Replace the E7 clamp with real BCs; the same machinery serves advection
markers and conduction daughters.

Per-domain-face and per-EB options (input-driven, parser for space/time
dependence where sensible):

| BC type | Marker rule | Conserves |
|---|---|---|
| Adiabatic / zero heat flux (default) | Specular reflection of the displacement across the face / level set | energy exactly |
| Isothermal T_wall (Dirichlet) | Crossing daughter's energy content reset to the T_wall value at the wall-intersection point (thermal-bath re-emission); tally the exchanged energy as wall heat flux diagnostic | budget via wall tally |
| Prescribed flux q_wall (Neumann != 0) | Adiabatic reflection + injection of energy markers at faces at rate q_wall * A * dt (EB: cut-face areas) | budget via source tally |

Notes:
- **Advection BC** stays as-is physically (V_e normal component at walls is
  governed by the Ohm's-law/EB E and J BCs already in place); what changes is
  that the *clamp* becomes reflection so entropy stops piling on boundary
  nodes. Outflow faces (if ever needed) delete content with a tally — markers
  are reset to home each step anyway, so deletion does not orphan cells.
- **EB geometry:** use the existing level set phi and distance machinery;
  reflection = mirror across the local normal, iterate the intersection to
  second order per the house EB-BC standard (level-set mirror ghosts). Check
  the reference algorithm for an existing marker-reflection implementation to port before
  writing a new one.
- **Cut-face areas** for EB flux injection can reuse the conformal-EB area
  fractions when `use_conformal_eb` is active; staircase areas otherwise.
- Wall-tally reduced diagnostics (net wall heat flux per boundary) come for
  free from the BC bookkeeping and should be exposed — they are the
  verification instrument for G4.

Gate G4: slab with two isothermal walls -> steady linear T profile (const
kappa) and correct q_wall; EB annulus with T(r_in)=T1, T(r_out)=T2 -> ln r
profile; budget closure |dE_plasma/dt - Q_wall| at round-off-adjacent levels.

---

## Verification harness (shared infrastructure, Phase 0)

A small research driver (untracked, alongside the existing
`Examples/Tests/ohm_solver_electron_energy_eq/`) that can:
- prescribe V_e analytically (bypass the Ohm's-law solve) for advection-only
  convergence sweeps (translation, rigid rotation, compression);
- sweep dt at fixed dx and dx at fixed CFL, emit L1/L2 orders;
- run the at-rest test (V_e = 0: T_e must be stationary — today it is not,
  E6) and report the per-step smoothing rate;
- budget audits: Sigma(K*N), Sigma(N), Sigma(u), wall tallies.

CI additions (each < 30 s, 2-core, CPU/GPU portable): one dt-order assert on
the advection test (3-point sweep, slope > 1.7 as a loose CI-proof bound), one
conduction spread test, one BC budget test. Existing adiabat/joule/qei tests
re-blessed once per behavior-changing phase (CHECKSUM_RESET procedure).

---

## Sequencing

| Phase | Content | Gate |
|---|---|---|
| 0 | Branch off development @ d72f49d70; build with energy eq; verification harness; measure baseline dt/dx orders, at-rest smoothing, chi_num; audit T_e restart | G0: baseline numbers recorded |

Phase 0 started 2026-08-04 (worktree `~/src/WarpX-qdsmc`, venv
`~/.env/warpx-qdsmc`, harness in
`Examples/Tests/ohm_solver_electron_energy_eq/convergence/`). Results:
- **T_e checkpoint gap CONFIRMED**: `FlushFormatCheckpoint.cpp` writes zero
  hybrid energy-equation fields (trap 2 is real).
- **E6 CONFIRMED and DOMINANT (G0 headline)**: at rest (V_e = 0, sum(Te)
  conserved to 1e-9), a Gaussian blob (sigma = 0.04L) loses half its
  amplitude in 64 steps: peak 20 -> 10.5 eV (N=32), -> 11.7 (N=64),
  -> 14.5 (N=128). The smoothing is per-step (dt-independent), so it is a
  numerical diffusion ~ dx^2/dt that grows unbounded as dt is refined.
- **Measured baseline spatial order: 0.2-0.4** (translation vs exact,
  CFL = 0.5, N = 32..128) — the E6 floor buries even the expected 1st-order
  remap error. The scheme is effectively sub-first-order in space at
  practical resolutions.
- **chi_num measured = dx^2/(4 dt) exactly**: added variance per step per
  direction = 0.50 dx^2 at N = 32, 64 and 128 (three-digit agreement) — the
  analytic [1/2,1/2]x[1/2,1/2] round-trip prediction for cell-center homes on
  a nodal grid. Scale check (liftoff placeholders, dx=1e-3, dt=1e-9):
  chi_num ~ 2.5e2 m^2/s vs flux-limited physical chi_eff ~ 1.3e3 at 10 eV —
  baseline numerical diffusion is ~20% of the conduction Thrust C would add.
  Physical conduction cannot be credibly added until B0 lands.
- **Consequence — plan reorder**: B0 (node homing) is promoted from Phase 2
  to a Phase 1 PREREQUISITE. No dt-order measurement is meaningful while E6
  dominates: finer dt = more remaps = more smoothing, so dt self-convergence
  slopes ~ 0 at baseline. G1's instrument after B0: same-discretization A/B
  between `euler`/`leapfrog`/`pc` + fixed-CFL combined refinement (pure-dt
  refinement at fixed dx cannot converge for a remap-every-step scheme —
  the donor-cell-limit diffusion dx*v*t*(1-CFL) grows as dt shrinks).
- **Ion CFL trap (harness)**: Esirkepov + hybrid segfaults on multi-cell
  crossings (WarpX warns at startup); keep ion and marker CFL < ~0.6 in all
  sweep points.
- **B0 LANDED (commit 296df8406 on qdsmc_energy_leapfrog)**: markers homed at
  grid nodes with unique seam/periodic ownership (box-seam nodes belong to
  the lower box; periodic top node gets no marker; non-periodic domain-top
  markers pulled 1e-6 dx inside). Post-B0 measurements:
  * at rest: relL2 3e-6 over 64 steps at every N (was 3-7e-2) — identity
    restored, residual is recovery-roundoff, resolution-independent.
  * translate spatial slope 0.40 overall, local order 0.29 -> 0.52 by N=128 —
    PRE-ASYMPTOTIC: the remap variance ~ dx*v*t*(1-CFL) is still comparable
    to the blob variance at N=128 (sb = 0.04L); the asymptotic 1st-order
    remap regime needs a wider blob or N >~ 512. Widen the blob for the
    Thrust-A instrument.
  * rotate dt self-convergence: slope 1.33 (local 1.16 -> 1.60) — the euler
    baseline platform number; mixed Euler + remap-difference contributions.
  * rotate sum(Te) drift ~ -6.6e-2 but IDENTICAL across step counts 64..1024
    => a property of the test setup (ballistic ion pattern evolution + the
    uniform-n proxy assumption), not a per-step numerical leak; at-rest and
    translate conserve to 1e-9. Use Sigma(entropy_fp) directly for the
    conservation gate in Thrust A.
- **Thrust A IMPLEMENTED (commit bc3cde26f)**: `qdsmc_time_advance` switch
  with all three schemes; euler verified BIT-IDENTICAL to the pre-restructure
  build (max |dTe| = 0 on the rotation test); knob verified in
  warpx_used_inputs; both new schemes conserve like euler. `pc` simplified to
  a single corrector transport (see A.4b); leapfrog carries pe_prev/pe_ext
  scratch fields and the dt/2 self-start. Also fixed in passing:
  QDSMCFillElectronPressureFromTe now FillBoundary's Pe (the E-solve reads
  grad Pe at box edges — latent multi-box ghost staleness in #6982).
- **G1 accuracy leg MEASURED (2026-08-04, rotate, N=128, CFL<=0.55)**:
  * Per-scheme errors vs a fine-dt reference all slope ~1.1 with candidates
    only 1.2-1.3x below euler — as predicted, that instrument is dominated by
    the scheme-independent remap-DIFFERENCE (itself O(dt) at fixed N).
  * Same-discretization differences (remap cancels): euler-leapfrog slope
    **1.04** = euler's isolated first-order time error. pc-leapfrog raw
    slope 1.01 is NOT an error: it is the half-step staggering offset
    (leapfrog's state ends at t - dt/2, so the pair is compared 0.5*dt
    apart). Subtracting a dt-scaled template of that offset leaves a
    residual at slope **1.88** — the two independently-staggered candidates
    agree to O(dt^2).
  * VERDICT (accuracy): both `leapfrog` and `pc` eliminate the O(dt)
    integrator error; at practical operating points their time error is
    already subdominant to spatial remap effects. The binding error is now
    SPATIAL (G2 territory), exactly as the plan structure anticipated.
  * Instrument note for the record: any fixed-time reference comparison of
    `leapfrog` carries the 0.5*dt*dTe/dt staggering offset; validate
    leapfrog against half-level-time references or template-subtract.
  * Remaining A.0 legs before selection: sources-on behavior (Strang wiring
    is in but untested — adiabat/joule tests), stiff-deck robustness
    (watch: pc's FIRST half-push uses Pe^n, dt/2 staler than leapfrog's
    centered Pe^{n+1/2} — the thing to probe on Joule-stiff decks),
    restart complexity (pc wins structurally: integer-time state, no
    extrapolation, no half-level checkpoint). Preliminary lean: **pc**,
    pending the sources/stiffness legs and Eric's call.
- **Stiff-robustness leg DONE (2026-08-04)**: Joule test at eta-scale 1000
  and 5000 (10x and 50x the CI value), all three schemes: NO instabilities
  anywhere — the pc Pe^n-lag concern does not manifest even at 50x. Budget
  closure at the stiffest point: euler +3.86% (its Lie full-dt source kick
  degrades with stiffness), leapfrog -1.00%, **pc -0.16%** — the Strang
  midpoint-valued sources win exactly where it matters. (Leapfrog's budget
  artifact shrinks with scale — consistent with the half-level-offset
  interpretation, relative to a growing dE_e.)
- **BAKE-OFF COMPLETE — RECOMMENDATION: select `pc`** (pending Eric's
  sign-off). Scorecard: accuracy tie with leapfrog (both O(dt^2)); cost tie;
  sources-on all-pass with pc budgets tracking euler at CI scale and best of
  all three under stiff heating; structurally simplest (integer-time state,
  no Pe extrapolation, no half-level checkpoint or diagnostic offsets).
  Leapfrog stays on the branch as a research alternative until upstreaming
  strips it per A.0.
- **Sources-on leg DONE (2026-08-04)**: all NINE case x scheme CI
  combinations PASS (adiabat / joule@eta-scale-100 / qei, run through a
  scheme-injection wrapper on the MPI+openPMD rebuild; results in
  convergence/ci_matrix/). Discriminator found: the Joule energy-budget
  closure is euler -5.9% / pc -6.0% / **leapfrog -16.4%** — the half-level
  T_e staggering leaks into integer-time energy accounting (adiabat and qei
  are scheme-equivalent, so the Strang/midpoint machinery itself is clean;
  the differentiator is purely the staggering). Operationally this means
  every downstream energy-budget diagnostic would need half-level-aware
  corrections under leapfrog — a second structural point for pc.
- **Survey (PLACEHOLDER deck numbers) preview of G3a**: raw Spitzer chi_par
  gives S_Sp up to 1e9 and hops of 30-1e5 cells — never carry it explicitly.
  Flux-limited (f=0.1, L_T=10dx) chi_eff collapses to S_eff ~ 0.5-23,
  l_hop(npts=2) ~ 1-7 dx, curvature leak (l_hop/R_c)^2 <~ 2% at npts=2 —
  i.e. the free-streaming limiter itself is what makes the explicit SDE arm
  viable; supports the no-elliptic preference. Re-run with real deck numbers
  before gating.
- **Thrust C IMPLEMENTED (2026-08-04)**: `QdsmcConductionOnce` — the
  daughters are EPHEMERAL, so no particle container: a grid kernel over
  owned nodes (same seam/periodic ownership trim as InitParticles) builds
  u = 3/2 n_e k_B T_e, spawns the quiet GH daughters along (b, e_perp1,
  e_perp2) with the Ito drift [div D + D.grad ln n] dt, hat-deposits into a
  scratch nodal MF (SumBoundary), and recovers T_e = u/(3/2 k_B n_e). Knobs:
  `qdsmc_kappa_par(n,Te,t)` (presence enables; n [m^-3], Te [eV], kappa
  [W/(m K)]), `qdsmc_kappa_perp(n,Te,t)` (default 0),
  `qdsmc_conduction_quadrature_points = <par> [<perp>]` (GH 2..7 hardcoded,
  probabilists' normalization), `qdsmc_conduction_flux_limit_factor`
  (default 0.1), `qdsmc_conduction_max_hop` (default 2),
  `qdsmc_conduction_vacuum_fast_front` (default on: floored cells get the
  capped ceiling chi, isotropic). Strang bracket
  C(dt/2).[S(dt/2) A(dt) S(dt/2)].C(dt/2) in pc and leapfrog (rho pairing:
  rho^n pre-transport, rho^{n+1} post), Lie C(dt) in euler; all no-ops with
  the parser unset, so the #6982 control stays bit-identical. Perp basis is
  chosen nearest yhat so the out-of-plane daughter loop collapses in 2D;
  zero-sigma / zero-projection loops collapse to a single point.
- **Daughter remap floor MEASURED — the half-gradient correction is
  MANDATORY, not an optimization**: without it (`qdsmc_gradient_deposit=0`)
  the aligned sweep is flat at slope **-0.01** with a resolution-INDEPENDENT
  spurious conduction chi_meas/chi0 = **1.432** at every N (24..128; the
  hat's alpha(1-alpha) dx^2 variance per deposit, constant in hop/dx under
  parabolic refinement — the conduction analogue of E6). With the
  correction (source-node MC slopes of u, same B1 identity: zero-sum =>
  conservation exact, cancels the hat second moment) the floor vanishes:
  chi recovery exact to ~1e-4.
- **G3 MEASURED (2026-08-04) — conduction is 2nd order**: aligned-B Gaussian
  spread, parabolic refinement (dt ~ dx^2, nsteps = 32(N/32)^2, chi0 =
  s0^2/2T, hops ~0.6 dx), L2 vs exact wrapped solution:
  * npts=3: N=24..128 slope **1.95** (local 1.86 -> 1.97), relL2 1.9e-4 at
    N=64. sigma^2 growth exact: the apparent 0.28% chi deficit is the
    wrapped-moment ESTIMATOR (the exact field scores 0.99717 under the same
    estimator; scheme-vs-estimator-consistent-reference agreement ~1e-4).
  * npts=2 (production default): slope **1.93**, constant 4.8x npts=3 — the
    weak-order-1 4th-moment defect scales as O(dt) = O(dx^2) under parabolic
    refinement, so 2nd order holds with a bigger constant, as designed.
  * tilted 30 deg (tensor rotation, chi_perp=0): relL2 6.4e-4 / 2.9e-4 /
    1.6e-4 at N=64/96/128 (local orders 1.96, 2.03) — 2nd order holds with
    the full rotated tensor. Perpendicular pollution: the raw moment reads
    1.7-1.8e-3, but the EXACT field scores 1.65e-3 under the same estimator
    (wrapped parallel tails leak into the tilted perp moment), so the
    scheme's actual spurious spread is the difference:
    **chi_perp_num/chi_par = 1.4e-4 at N=64, 4e-5 at N=96, 2e-5 at N=128**
    (~N^-2) — an order below the 1e-3 gate target and converging away.
    Uniform B => no curvature contribution; the curved-field-line leak
    (trap 11) remains the C.7 ring-test question.
  * Sigma(Te) conserved to 2-3e-8 in every run; at-rest/uniform identity
    preserved (chi=0 or uniform Te => exact by construction).
  * Hop-cap soft-min form matters: the harmonic soft-min chi*cap/(chi+cap)
    biases chi by ~20% already at chi = cap/4 — replaced by the p=4 power
    soft-min chi/(1+(chi/cap)^4)^(1/4) (<0.4% at cap/2, still smooth for
    div D).
- **Evolve re-entry T_e reset — pre-existing #6982 bug FOUND and FIXED**:
  `HybridPICInitializeRhoJandB` runs at step == step_begin of EVERY
  `Evolve()` entry (not just run start), and its closure
  `CalculateElectronPressure()` overwrote the evolved T_e/Pe with the
  polytropic value — any PICMI deck calling sim.step() in segments silently
  resets the energy equation's state each segment (found because the
  conduction probe scripts stepped one step at a time: the blob vanished at
  the second sim.step()). Fix on the branch: with the energy equation on,
  that entry point now emits Pe from the CURRENT T_e
  (QDSMCFillElectronPressureFromTe) instead of the closure — also the
  energy-consistent Pe^0 = n k_B Te0 seed on fresh start/restart. CI matrix
  re-run after the fix: 9/9 PASS (joule budgets shift by ~0.8 points from
  the changed first-step seed: euler -6.7 / pc -6.7 / leapfrog -17.2%).
  Same family as the T_e checkpoint gap (trap 2); fold both into the
  eventual restart fix.
- **Host-OMP deposit race trap (segfault diagnosed)**: on the HOST,
  `amrex::Gpu::Atomic::AddNoRet` is a PLAIN `+=` — any OMP-threaded MFIter
  loop whose deposits reach across tile boundaries races (NaN -> floor() ->
  wild index -> segfault). The daughter deposit loop is therefore
  unthreaded, same as DepositScalar. House rule: no `#pragma omp parallel`
  around host atomic-deposit kernels.
- **Conduction-harness whistler note**: the uniform-B decks must keep the
  explicit B-substep advance stable against machine-eps curl(E) seeds:
  omega_wh(k_max) dt_sub = (pi N/L)^2 B/(mu0 q n0) dt/substeps <~ 0.1
  (B0 = 2e-5 T for the standard parameters; the ratio is INVARIANT under
  the parabolic dt ~ dx^2 sweep, so one B0 serves the whole N range).
  At B0 = 0.01 T the instability seeds from rounding and destroys the run
  by step ~4 — harness scripts qdsmc_conduction_test.py / run_conduction.py
  encode this.
| 1 | Thrust A bake-off: J_i^n reorder + midpoint push (shared); `leapfrog` (half-staggered T_e, Pe extrapolation, self-start, checkpoint) and `pc` (split advance across B halves) behind the `qdsmc_time_advance` switch; Strang sources; measure and select | G1 |
| 2 | B0 node-homing; re-measure; decide on B1/B2 | G2 |
| 3 | Thrust C conduction (split substep, quiet daughters, full tensor D, per-direction quadrature, both limiters, vacuum policy) | G3 |
| 3a | C.7 stiffness analysis: regime survey (script), curvature-pollution + hop-cap-deficit measurements on the Phase-3 prototype; parallel-backend decision | G3a |
| 3b | (last resort; NOT scoped unless G3a proves explicit options insufficient) elliptic parallel solve | — |
| 4 | Thrust D BCs (domain + EB, Dirichlet/flux, tallies); remove E7 clamp | G4 |
| 5 | Integration: liftoff/annulus decks with conduction on (the GPU nodes for the big runs); perf target: energy-eq + conduction overhead < ~15% of ion PIC cost; docs (parameters.rst, theory section), PICMI knobs | G5 |

Upstream packaging is deferred (decided 2026-08-04): explore and test first;
revisit PR slicing once G1/G3 numbers exist. The E6 node-homing and T_e
checkpoint findings stay recorded as candidate standalone fixes for when that
conversation happens.

---

## Traps (numbered, house style)

1. **PICMI bucket-write clobber** — new knobs (flux-limit factor f, m_hop,
   kappa parsers, BC selections) must be verified in `warpx_used_inputs`, not
   assumed from the deck.
2. **T_e checkpoint gap** — energy-equation state is not checkpointed today
   (same class as the rho_fp gap, PR #7049). Leapfrog makes it worse (half-level
   state + previous Pe). Fix explicitly; test without "_restart" in the name.
3. **OMP deposition heap bug** — pre-existing (OMP<=2 workaround); the new
   conduction deposit adds another scatter path that may trip it. Keep the
   workaround in test decks.
4. **Redistribute reach** — conduction daughters hopping m_hop*dx can cross
   more than one box; ensure ghost/`RedistributeLocal` assumptions and the
   one-cell CFL comment in `PushX` are updated together.
5. **RZ volume weighting** — `DepositScalar` uses a constant cell_volume;
   markers move in the (x,y) plane with theta = 0 pinned. Validate RZ
   conservation explicitly before enabling conduction there (RCYLINDER/RSPHERE
   already refused at ReadParameters).
6. **Node double-ownership** — B0 node-homing puts markers on box-boundary
   nodes; ownership must be unique or Sigma(N) double-counts at seams.
7. **Drift-term kinks** — hard min() in the chi caps makes div D
   discontinuous; soft-min or the drift correction will inject spurious
   structure exactly at the limiter engagement front.
8. **Checksum re-bless discipline** — every phase that changes physics
   re-blesses; keep the per-phase re-bless commits separate from code commits.
9. **Frozen Pe across B substeps** — unchanged by this work but more visible
   once Joule + conduction sharpen hot spots; record as known limitation, do
   not silently "fix" inside the substep controller.
10. **Isolate with controls** — any instability during liftoff integration
    gets an A/B ladder (advection-only / +sources / +conduction / +BCs) before
    touching the scheme.
11. **Curved-field-line leak** — straight parallel hops across curved/sheared
    B leak cross-field as chi_par (l_hop/R_c)^2; every time the hop cap is
    raised (or npts_par extends the tails), re-check the ring-test pollution
    number. The failure is silent anisotropy loss, not a crash.

---

## Decision log

- **2026-08-04** — Conduction quadrature: Gauss–Hermite, **default 2 points
  per direction**, selectable via `qdsmc_conduction_quadrature_points` for
  high-T / tail-sampling regimes. npts=2 is variance-exact but weak order 1;
  the G3 order-2 demonstration runs at npts=3. (Eric)
- **2026-08-04** — Time advance: no pre-commitment to half-staggered leapfrog.
  Implement `leapfrog` and `pc` behind the `qdsmc_time_advance` switch (with
  `euler` as control), measure per A.0 criteria, select the best performing.
  (Eric)
- **2026-08-04** — Review resolutions: Strang first (Lie as follow-on);
  conduction as split substeps on u; **full tensor from day one** (anisotropic
  conduction is a must-have); G2 trigger 1.7 confirmed; upstream packaging
  deferred until after exploration. (Eric)
- **2026-08-04** — Quadrature counts become **per-direction**
  (`qdsmc_conduction_quadrature_points = <npts_par> <npts_perp>`), quadrature
  axes field-aligned, so the parallel direction can carry more points for
  fast tails. (Eric)
- **2026-08-04** — Added C.7 stiffness analysis / gate G3a: measure whether
  the explicit stack (quiet daughters, field-line hops, subcycling,
  RKF45-embedded grid form) carries Spitzer parallel conduction, or the
  parallel channel needs an elliptic implicit solve. The reference algorithm's JFNK precedent:
  the conductivity stiffness was acceptable implicitly. (Eric)
- **2026-08-04** — GO for Phase 0. Elliptic parallel solve demoted to last
  resort: avoid elliptic solves in the explicit advance at all costs; do not
  scope option (c) unless G3a proves (a)/(b) insufficient. Work branch
  `qdsmc_energy_leapfrog`, worktree `~/src/WarpX-qdsmc`. (Eric)
- **2026-08-04** — **Time advance SELECTED: `pc`** (G1 tie on accuracy/cost;
  9/9 sources-on matrix; best stiff-Joule budget −0.16% at 50x eta;
  structurally simplest). Default flipped to pc on the branch; euler kept as
  control; leapfrog kept until upstream packaging strips it per A.0. Proceed
  to Thrust B (B1 slope-carrying markers — G2 tripped at first-order remap
  cap 0.86). (Eric)
- **2026-08-04** — **Thrust C implementation form: grid-kernel ephemeral
  daughters** (spawn/displace/deposit/discard inside one ParallelFor over
  owned nodes) instead of an AMReX particle pass — no container, no
  Redistribute, reuses the validated seam-ownership and SumBoundary
  patterns. (Claude, measured)
- **2026-08-04** — **Daughters carry the B1 half-gradient correction —
  mandatory**: without it the hat remap adds a resolution-independent
  +43% spurious chi under parabolic refinement (slope 0.00); with it,
  chi exact to 1e-4 and slope 1.95. Gated by the same
  `qdsmc_gradient_deposit` knob as the advection markers. (measured)
- **2026-08-04** — **Hop-cap smoothing = p=4 power soft-min**
  chi/(1+(chi/cap)^4)^(1/4); the harmonic soft-min rejected (~20% chi bias
  already at cap/4 — would silently slow all conduction). (measured)
- **2026-08-04** — **G3 PASSED**: aligned slope 1.95 (npts=3) / 1.93
  (npts=2, 4.8x constant); tilted-30deg slope ~2.0 with
  chi_perp_num/chi_par = 1.4e-4 at N=64 falling ~N^-2 (estimator-floor
  subtracted). Zeldovich nonlinear front + limiter-on legs and the C.7
  ring test remain open.
- **2026-08-04** — **Evolve re-entry Pe/Te reset fixed** (pre-existing
  #6982 bug): with the energy equation on, HybridPICInitializeRhoJandB now
  emits Pe from the current T_e instead of the algebraic closure, so
  segmented PICMI sim.step() no longer wipes the evolved state. CI matrix
  9/9 after the fix.
- **2026-08-05** — **Production conduction form: Esirkepov flux-form
  remap** (`qdsmc_conduction_form = fluxform`, design C.7d). **Layer is
  off the table as a production method** (Eric); it stays in-tree only as
  the leak reference arm. Rationale: the flux form is the only scheme
  that gives strict conservation, never-add monotonicity, and the
  pushforward fix structurally at once, and its face fluxes are the
  Thrust-D substrate (adiabatic walls = F=0 by construction, wall-flux
  tallies for free). (Eric)

## Design questions — resolutions (2026-08-04 review)

1. **Leapfrog vs predictor–corrector** — test and choose: both behind the
   `qdsmc_time_advance` switch, measured on the A.0 criteria (A.0/A.4b).
2. **Strang vs Lie** — Strang first; Lie only as a possible follow-on
   simplification if the commutator measures negligible (A.3).
3. **Conduction carrier** — split substeps on u (C.4); unified single-push
   SDE (C.5) deferred.
4. **Tensor** — full tensor from day one; anisotropic thermal conduction is a
   must-have (C.1).
5. **Spatial thrust trigger** — G2 threshold 1.7 confirmed.
6. **Upstream packaging** — deferred; explore and test before planning any
   upstream work.

## Open question (new, 2026-08-04)

- **SDE arm vs elliptic parallel solve** for Spitzer-stiff parallel
  conduction — pending the C.7 stiffness analysis and gate G3a. Levers on the
  explicit side, in escalation order: field-aligned per-direction quadrature
  (npts_par > npts_perp for fast tails), field-line-following hops,
  subcycling (SDE-repeat or RKF45-embedded grid form, where N_sub ~ S =
  chi_par dt/dx^2 decides feasibility). The elliptic parallel solve is a
  LAST RESORT (avoid elliptic solves in the explicit advance at all costs)
  — the default answer to over-stiffness is capped/limited fast-front
  transport, and (c) gets scoped only if G3a proves that fallback distorts
  the target physics.
