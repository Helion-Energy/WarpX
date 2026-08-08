# Quarantine instrument: open-set contamination tally (design)

Item 6 of `DELIVERY_runaway_2026-08-08/prompts/redirect_drag_and_quarantine.md`.
Goal: measure, per step and per channel, the energy flux from sub-floor /
marooned cells into open-set (n > n_floor) cells — before/after the hardening
knobs — and verify the S0→SK degradation vector (fast-front + kappa_perp).

## Cell classes (computed once per source application, nodal, from rho_fp)

- OPEN: rho > rho_floor
- HALO: rho <= rho_floor (includes marooned/evaporated cells; a "marooned"
  sub-class — was OPEN within the last K steps — needs a persistent last-open
  step field, phase 2)

## Channels and tap points (all C++, one input gates all:
`hybrid_pic_model.qdsmc_contamination_tally = 0|1`)

1. **kappa_par / kappa_perp conduction** — in the fluxform sweep flux
   assembly: every 1D face flux with donor cell HALO and receiver OPEN
   accumulates |F|·dV into channel (par|perp, from the branch class).
   The split sweeps know their branch (par/perp) at assembly time; the
   per-face donor side is the upwind cell of the remap. Tally MF: nodal,
   4 comps (par_in, perp_in, par_out, out for symmetry checks).
2. **vacuum fast-front** — same tap, but tagged when the donor cell's chi
   was fast-front-boosted (floored cell, chi set to hop cap). Separate
   comp; the S0→SK hypothesis test is (fast-front + kappa_perp) vs controls.
3. **QDSMC advection** — in the transport deposit fold
   (QDSMCFoldInsulatingDeposits / DepositK): entropy deposited on OPEN
   nodes by markers whose HOME node was HALO. The spill-only fold already
   distinguishes these populations; add the tally before the fold decides.
4. **ion kicks** — in QDSMCApplyIonHeating's coefficient loop: redirected
   E_s staged in HALO cells × (3/2) n_s (the energy the OU kicks will carry
   into ions inside quarantined cells — these ions then free-stream into
   the open set; the grid-side proxy is the staging tally, the particle-side
   confirmation is a Ti-weighted flux at the class boundary, phase 2).

## Output

Same pattern as the dropped-energy tally: cumulative J per channel,
`[qdsmc] step N contamination_J: cond_par=... cond_perp=... fast_front=...
advect=... kicks=...` every joule_dropped_energy_print_interval steps
(shared cadence input), plus a registered per-cell contamination-density
field for forensics maps when a diag flag is on.

## Arms (the GPU node, slow-ramp liftoff, after SR2/SF verdicts)

| arm | knobs | expected |
|---|---|---|
| Q0 | tally only (S0 config) | baseline channel spectrum |
| Q1 | + vacuum_fast_front=0 | fast-front channel -> 0; does perp drop too? |
| Q2 | + joule_heating_eta spitzer | ignition-band source gone; contamination should collapse |
| Q3 | + kick cap + redirect gates (SR2 config) | kicks channel bounded by cap tally |
| QK | + kappa x4 (SK config) | the S0->SK degradation vector, quantified per channel |

Acceptance hook: open-set te_max bounded (<= few x100 eV transients) with the
energy audit <= 3%/100k-step window including dropped + contamination tallies.

## Notes / traps

- Tally MFs are per-application temporaries reduced with sum_unique (nodal)
  — same seam-redundancy pattern as the dropped-energy tally (verified).
- The conduction executor's Thrust-D face-local deterministic recomputation
  must stay atomics-free: tallies accumulate into a separate nodal MF in the
  same face loop (per-tile disjoint writes), NOT into shared counters.
- Do not tap the scatter/layer control forms (pre-EB behavior is a known
  perpetual bath donor next to a wall band; the tally is for the production
  fluxform path). Layer taps can come later if needed.
- Class boundary is evaluated at source-application time; a cell that
  evaporates mid-step is classed by the pre-application rho (document in
  the tally header).
