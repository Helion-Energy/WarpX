# theta_implicit_mhd — commit-history errata

This branch does not rewrite pushed history. Corrections to claims made in
commit messages are recorded here, per the 2026-08-29 adversarial audit
(full reports: pc-emhd-analysis/{COMMIT_AUDIT,CLAIMS_AUDIT,PARITY_AUDIT}.md).

## bc96d3c53 — "MHD: default fluid_flux = central; rename centered -> legacy_e_centered"
- The "Full implicit suite 55/55 with the flipped default" claim was inaccurate:
  the 55-test slice did not cover the stiff-alfven preconditioner pair, whose
  analysis still asserted the old name (fixed in a231b5cf9).
- Not disclosed: with the flip, a deck setting neither fluid_flux nor viscosity
  aborts at startup (central requires viscosity > 0, default 0). Remediated by
  a migration-aware abort message + docs in the parity-fix series.
- Docs (parameters.rst, theory docs) were not updated by the rename commit and
  taught the hard-error name as the default until the parity-fix series.

## df72965ac — "MHD: reference-matched per-component Braginskii conduction clamps"
- The claim "decks that set neither are bit-identical" is FALSE for decks
  running the quasi-shorting or halo chi_perp boosts (including the production
  standard): the par >= perp guard is unconditional and re-orders boosted
  chi_perp above raw chi_par. The GUARD ITSELF is parity-correct (the reference code
  ntb.f90:594 applies the same MAX after its clamps); only the bit-identical
  claim was wrong. Every 2026-08-28/29 formation arm on this tree carried the
  guard-modified chi_par relative to the pre-guard survivor arm — an
  undisclosed confound in those A/B comparisons.
- Shipped without tests or docs; remediated in the parity-fix series.

## 3e72ac642 — "MHD: wall_thermal_bc = dirichlet -- pin the wall temperature"
- The formation A/B paragraph ("the pin moved the late freeze +11.93 -> +14.99 us
  and removed EVERY wall-band cell from the terminal pin population") is
  RETRACTED: replica ensembles showed the late-freeze onset carries +/- 3 us
  run-to-run scatter and terminal pin locations are themselves draw-dependent;
  additionally 36 of the 52 terminal pins were never listed, so "EVERY
  wall-band cell" was unverifiable at commit time. The mode's contract
  correctness (its ctest gates) is unaffected.

## 6e6e17a90 — "MHD: enforce the absolute electron energy floor..."
- Additional undisclosed behavior change: decks WITH electron_temperature_floor
  set now restore against max(absolute floor, temperature image) where the
  temperature image alone was used before.
- "Full implicit suite green modulo ..." claims in this and other commits are
  post-hoc unfalsifiable (only the most recent ctest log survives). Landing
  practice from 2026-08-29 onward: archive the gate's ctest LastTest.log
  alongside the landing (scratch or CI), so suite claims stay auditable.

## Campaign-ledger corrections (recorded in the session ledger, summarized here)
- smoothic (2026-08-28 arm A) died on the ION energy floor, not electron; the
  "boost hits the electron channel first" gloss is withdrawn.
- "E2 ~19% tolerance-converged" was inflated ~17x: true tolerance-converged
  fraction 34/3144 = 1.1% (the 19% counted non-pegged exits, mostly
  frozen/stagnation-accepted).
- "E2R2 shows no wall pins" was false (final pin block contains (0.7148, 3.574));
  the associated retraction stands on replica-scatter grounds instead.
- The 2026-08-28 "clamp condemnation" is REVERSED: the reference code's clamp values are
  exonerated; the failures were WarpX realization defects (non-monotone
  Braginskii corner cross term; stair-step wall drain exposing the parallel
  clamp class through the nn projection; a (gamma-1) conduction-convention
  skew). See the parity-audit report and the parity-fix series.
