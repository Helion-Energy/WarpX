RKL2 conduction rollback and retry
=================================

A validated RKL stage that has a finite positive stability ceiling below the
selected requirement rejects its entire super-step. Intermediate RKL stages are
not physical-time solutions. The original guard (including its 64-epsilon
roundoff allowance) is unchanged. Nonfinite state/RHS and invalid ceilings are
terminal failures, including when a nonfinite RHS also violates the ceiling.

The integrator restores valid temperature and passive auxiliary values/rates at
the super-step entry. Previous completed super-steps survive. Its ``AttemptHooks``
save/restore the conduction caller's raw floor energy, floor count and floor mask.
Coefficient-cache validity is cleared; temperature ghosts, tensor, bulk/leg bounds,
flux-budget scratch and validation reductions are rebuilt before reuse. Static
geometry, density, ownership masks, bath pins and entry projection heat do not
change during an attempt. RKL recurrence scratch is overwritten at retry entry.
Test-only projection traces and attempted-work counters describe provisional work.
Callbacks must not publish provisional energy or otherwise perform irreversible
actions. Other callers supplying mutable callbacks must supply equivalent hooks.

Let ``tau`` be the failed duration, ``required`` the ceiling it required, and
``observed`` the smaller positive ceiling. With ``margin=min(0.9,safety)``, retry
uses persistent bounds::

    retry_tau = min(0.5*tau, tau * margin * observed/required)
    retry_cap = min(previous_retry_cap, margin*observed)

Stage sizing uses the smaller of the entry ceiling and ``retry_cap``, then obeys
the existing maximum-stage and remaining-work limits. The bounds survive both
retry re-entry and subsequent accepted super-steps within the call. The normal
no-rejection sizing and recurrence arithmetic remain unchanged. There are at most
32 consecutive retries of one super-step; the next rejection fails explicitly.
The existing per-call work limit includes every attempted stage evaluation,
including the failing evaluation. Nonpositive/nonfinite duration, an underflowed
first-stage increment or required ceiling, and inability to advance represented
time are explicit failures. Accepted stage counts, maximum stage count and timestep
statistics are committed only after a complete super-step succeeds. ``retries``
reports recoverable rejections, including a terminal rejection at the retry limit.

The conduction call still evolves a private temperature buffer. It commits the
entire temperature field and historical heat tallies only after the requested
interval completes and finite/accounting checks pass. On ultimate failure no
partial temperature or cumulative heat is published. The generic integrator's
caller continues to own this whole-call transaction.

``QdsmcRKStats`` and ``QdsmcConductionReport`` expose attempted/accepted work,
recoverable rejections, discarded work, current attempted duration and ceilings,
and the last rejected duration/required/observed ceiling. The optional
``WARPX_QDSMC_COND_REPORT`` JSON includes these fields; ordinary successful calls
produce no new logging. Failure messages include the same retry context.

Verification
------------

``test_rz_qdsmc_rkl_retry`` tests state-dependent tightening, rejection after four
completed stages with nonzero floor/projection and passive heat, repeated rejection,
retention of earlier completed super-steps, conservative-reference accuracy,
energy closure, work/retry exhaustion, timestep underflow, and nonfinite/invalid
input. Only the final MPI rank tightens the synthetic bound, so the two-rank test
exercises collective decisions and rollback. ``test_rz_qdsmc_rkl_retry_mpi`` is
registered for CPU MPI builds; run the same executable with a scheduler-supported
1/2-rank launch on CUDA hardware.

``test_rz_conduction_boundary_native_retry_floor``, ``retry_budget`` and
``retry_underflow`` exercise
the actual conduction callbacks and success-only commit. A test-build-only ceiling
injection on the fifth evaluation forces rollback after floor activity. Failure
checks include valid and ghost temperatures and every historical wall/leg/EB/floor
tally. The recovered floor solution is compared with a smaller-step native
reference at a relative temperature/energy tolerance of 1e-4, with the original
energy-accounting gate applied to both. ``gate_retry`` also exercises recovery
from physical boundary activation without injected ceilings. The existing
accounting and native conduction tests remain regression gates.

This is numerical recovery within a conduction call, not checkpoint recovery,
a relaxation of the conduction physics, or a production campaign qualification.
