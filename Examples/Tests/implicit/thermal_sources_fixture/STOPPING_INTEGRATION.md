# Pure accepted cell-U stopping transfer

`EulerianStoppingTransfer.{H,cpp}` uses the delivered `KineticThermalMoments`
physical source restriction. Link both translation units in the production
implicit-solver build. This worker patch does not edit model, collision,
Evolve, Theta, checkpoint, or production build-list methods.

Construct with the same normalized physical RZ Coord1 geometry, cell boxes
and distribution as the endpoint thermal state. Map options as follows:

- `gamma = model.m_gamma`;
- `raw_density_floor = model.m_n_floor` (not `m_qdsmc_n_floor`);
- `temperature_floor_ev = model.m_cond_te_floor`;
- propagate EB, physical-level and azimuthal-mode counts to the guards.

The first supported application is RZ m=0, one complete zero-based physical
mesh starting at r=0, no EB. The new helper rejects other geometry/capability
requests. Existing 3D methods remain unchanged; this translation unit compiles
in a 3D build but is not a new 3D stopping implementation.

## Inputs and scratch outputs

After the existing accepted endpoint collision pass, copy the nodal partial
staging and `SumBoundary(period)` that private copy once. Pass its signed
**J/m3 impulse** to

```cpp
bool ok = transfer.Evaluate(endpoint_moments, consolidated_impulse,
                            raw_endpoint_charge, endpoint_number_capacity,
                            accepted_cell_U);
```

Every physical input is const and one component. Nodal fields require the
cell-converted layout and matching distribution; only valid values are read.
The actual raw charge is C/m3; capacity is physical cell electron number
per m3 including the configured pedestal/floor. `endpoint_moments` must have
the same geometry/layout/periodicity. Its nonconst reference is needed only
for `RestrictNodalScalar`'s private scalar scratch; its density, temperature,
pressure and current outputs remain unchanged. Native charge quadrature
(axis3 or axis4) does not replace the physical impulse quadrature.

There is no dt/theta multiplier, no implicit-stage source evaluation and no
extra source consumption. Nonfinite/invalid U or capacity and materially
inconsistent duplicate node values return false. Copy/sum roundoff up to
64 machine epsilons times the nodal max norm is reconciled using private
owner synchronization; this never substitutes for summing partial deposits.
Input aliases of owned output storage and structural errors assert.

On success, read const `CandidateEnergy`, `CellImpulse` (actual stored delta U),
`RequestedCellImpulse` (eligible physical map before the floor),
`NodalEligibleImpulse`, `NodalDeclinedImpulse`, `CellFloorDeclinedImpulse`, and
`Ledger`. Outputs are invalid after a failed Evaluate. Calls overwrite private
scratch and ledger; they never accumulate accepted diagnostics. A/B/A is
bitwise reproducible in the fixture. Calling the same input repeatedly is
pure, not consumption: the caller must commit/clear exactly once.

The raw node gate is `rho > q_e*raw_density_floor`. A pedestal does not enable
an otherwise ineligible node and does not amplify physical impulse. After
mapping eligible impulse to cells, the primary cell floor is
`min(U_before, n_capacity*q_e*T_floor_eV/(gamma-1))`. Cooling that would cross
this value is declined locally; an already subfloor cell is never raised.
Nodal floor clipping alone is insufficient: U=[1,100], U_floor=1 and a shared
nodal impulse -49.5 can map to U=[-23.75,75.25]. The fixture includes that case.

## Commit and accounting remain caller-owned

Validate candidate U and its derived endpoint Te/Pe in scratch before copying
`CandidateEnergy` into accepted U. Then refresh the accepted thermodynamics
from that U and the chosen actual endpoint density, with all trial views
cleared. Book `Ledger` once and clear the original staging only after successful
commit. The caller owns rollback/restart between already changed ion momenta
and electron commit. A failed helper call alone cannot roll back collisions.

Ledger fields are all joules. `requested` is the signed staged ion kinetic
loss; `density_eligible` and `mapped` bracket physical restriction;
`density_declined` is signed raw-gate decline; `floor_declined` is nonnegative
cooling declined by the cell floor; `delivered` is the actual candidate-minus-
input cell energy integral. `map_defect` and `exchange_defect` expose map and
stored-arithmetic roundoff. If delta_Ki=-requested, then

`delta_U + delta_Ki = floor_declined - density_declined + exchange_defect`.

Do not call the old nodal-Te stopping consumer or a wall/marker integrator.
The cell method's reservoir/leg/flux boundary already lives in the timestep
thermal residual; a post-collision Te-only pin creates an inconsistent second
thermal state. No extra conduction or leg interval is applied here.

## Temporal and production qualification limits

This preserves the existing **accepted endpoint collision split**. The ions
and staging are produced once after the field/thermal step accepts, using the
prepared endpoint background. Stopping is not predicted inside the implicit
midpoint residual or differentiated in its Jv; no claim of a second-order
fully coupled stopping discretization is made. Preserve the existing DT
fusion/alpha collision ordering and every requested stopping species; this
helper does not replace those operators or create QDSMC markers.

The fixture uses the actual production moments restriction and physical
clipped-volume deposition, including native axis3/4, finite/periodic caps,
seams and MPI empty ranks. It tests signed measured-loss conservation, raw
floor equality, pedestal, active cooling floor, nodal-floor counterexample,
input/moment-output purity, caller-simulated once-only consumption, malformed
seams, invalid inputs and capability/alias rejection. Full runtime DT fusion,
measured alpha stopping, accepted state/RNG/tally checks and interrupted
restart remain required after parent wiring. CUDA evidence is compile-only.

For standalone reproduction, configure `THERMAL_MOMENTS_SOURCE_DIR` to the
integrated adapter or immutable b6e106d reference, build `test_stopping_transfer`,
and run `ctest --test-dir <build> -R '^stopping_' --output-on-failure`. Run each
physical case with MPI2 at `test.max_grid_size=4` and `64` as recorded in the
worker's exact command manifest.
