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

Finite-temperature admissibility
--------------------------------

RKL conduction also validates every provisional temperature state, including
its final stage. A finite spectral-ceiling check alone does not establish that
the temperature trajectory is physically admissible. The check observes the
minimum after physical boundary constraints but before numerical floor clipping,
and the maximum after projection. A nonfinite raw state remains a hard failure,
even if a boundary constraint or floor could conceal it.

A whole-call temperature envelope is frozen after entry boundary projection.
It includes the initial active temperatures, prescribed wall temperatures, leg
reservoir/gate temperatures and the configured floor. At elapsed time ``t``, the
bounds are::

    lower(t) = initial_min - t * max_prescribed_cooling_rate
    upper(t) = initial_max + t * max_prescribed_heating_rate

The prescribed-flux rates use the same nodal heat capacity and physical wall
area/volume metric as the RHS, summing positive and negative contributions at
corners separately before taking global maxima. Thus imposed heat fluxes may
legitimately drive temperatures outside the initial range. Physical boundary
reactions and the floor values are unchanged. The envelope is not reset after
accepted super-steps, so repeated small excursions cannot ratchet up its bounds.
The roundoff allowance is ``128*epsilon*s*s*max(1 K,abs(lower),abs(upper))`` for
an ``s``-stage plan; it does not change the spectral-ceiling tolerance.

A finite violation rejects the whole super-step through the existing transaction,
including all provisional heat, floor energy/count/mask, and accepted-work
statistics. Duration is halved persistently and the stage plan is recomputed;
a temperature violation does not invent an observed spectral-ceiling violation.
The same consecutive-retry and attempted-work budgets apply. Nonfinite state,
invalid bounds and lack of represented progress remain explicit failures.
``admissibility_retries`` distinguishes these rejections from ceiling rejections;
optional reports record the last rejected temperature extrema and their bounds
in kelvin. The extrema share the existing projection reduction, followed by one
collective maximum per RKL stage. SSPRK2/RKF45 are unchanged.

``test_rz_qdsmc_rkl_admissibility`` covers finite upper/lower excursions, floor
concealment, last-stage rejection, repeated rejection, retry/work exhaustion and
nonfinite priority. Controlled excursions use the 168/168/176-stage plans and
three temperature-growth ratios from a failed application case; these are
transaction regressions, not an exact spatial replay. Native temperature tests
cover MPI-local upper/lower/final/late-stage rejection and failure atomicity.
Existing imposed-flux/floor cases and ``hot_bath`` check legitimate boundary
heating/cooling; the conservative-reference and energy-closure gates still apply.

This guard is not an error estimator or a change to the spatial operator. If the
spatial discretization itself violates the envelope even as duration shrinks,
the call fails within its finite budgets instead of accepting the excursion.
