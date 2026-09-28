# Native activity and feasible trial initialization

This isolated native AMReX fixture exercises production `NativeConductionActivity`,
`ThermalStageInitialGuess`, `EulerianThermalStage`, `ThermalConductionPC`, and the
scalar Newton/FGMRES driver. Only the small WarpX layout facade is used. The legacy
`QdsmcConductionFDOperator` is called directly to check native nodal edge gates.
No GPU runtime, particle advance, callback side effects, or accepted energy clipping.

Configure with an RZ CPU AMReX installation:

```sh
cmake -S Examples/Tests/implicit/thermal_trial_fixture -B build-thermal-trial \
  -DAMReX_DIR=/path/to/amrex/lib/cmake/AMReX
cmake --build build-thermal-trial -j4
ctest --test-dir build-thermal-trial --output-on-failure
OMP_NUM_THREADS=1 mpirun --mca pml ob1 --mca btl self,tcp -np 2 \
  build-thermal-trial/test_thermal_trial test.case=masked_balance test.box=8
```

Cases `activity`, `floors`, `refinement`, `channels`, `native_edges`, `initializer`,
`compression`, and `masked_balance` run in single- and multi-box layouts. The
`reject` case checks that omitting an explicit cell-closure choice aborts.
`test.margin` overrides only the compression fixture's initial-guess margin.

## Activity contract

The maintained native gate is evaluated **at native nodes before restriction**:
`n_raw = rho/q_e + pedestal/q_e` and
`open = n_raw > (halo_unfreeze || pedestal_pointer ? 0 : qdsmc_n_floor)`.
A present zero-valued pedestal still selects the zero threshold. Each native edge
opens only when both endpoint nodes are open. The helper reproduces this graph and
reports both `max(n_raw, qdsmc_n_floor)` and `max(n_raw, n_floor)` capacities plus
their relative discrepancy, separately over all nodes and over active nodes.
It neither overwrites floors nor changes the thermal capacity density.

Callers must choose `ConductionCellClosure::AllContributors` explicitly. A thermal
cell opens only if every corner node is active; a thermal face opens only if both
adjacent cells are open. Physical faces use the nearest cell image. This is a
conservative finite-volume adaptation, not an identical native nodal discretization.
At smooth threshold boundaries it erodes the conducting domain by O(h). A channel
narrower than a cell can disappear. `OpenFraction` integrates physical nodal-dual
intersection volumes; it is a diagnostic, not a fractional conductivity coefficient
and not the corrected-axis particle deposition quadrature. `Measure` reports native
open volume, retained full-cell volume, mixed-cell volume, and lost open volume.

`Evaluate` validates all native inputs before publishing owned outputs, leaves
caller valid/ghost data unchanged, and preserves previous outputs on failure.
Only normalized RZ physical geometry, one level, and no EB are qualified here.
The resulting cell mask is ready for `ThermalFaceContext.conduction_active`; it
changes conductive faces only, not capacity, pressure, sources or advection.

## Trial seed contract

For `e_floor = q_e T_floor/(gamma-1)`, a trial stage seed must satisfy both
`U_stage/n_stage >= e_floor` and
`(U_stage-(1-theta) U_old)/(theta n_endpoint) >= e_floor`, with strict positivity
when the configured floor is zero. The point helper checks these actual divisions.
An already feasible input is returned bit-for-bit. Otherwise it moves above the
larger bound by configurable `relative_margin * max(U_old, bound)`; the default
1e-4 gives finite-difference probes room. A representability-only margin (64 eps)
was admissible but made this fixture's symmetric Jv cancellation dominated.
This margin is numerical initial-guess preparation; it does not change any floor,
residual, or accepted state. Overflow/unrepresentable guesses reject transactionally.

`MakeThermalStageInitialGuess(output, old, seed, stage_density, endpoint_density, options)`
uses matching one-component cell layouts, reads valid data only, permits in-place
`output == seed`, and forbids aliasing accepted old U or densities. Invoke only
before Newton, after trial densities are known and before pressure/source callbacks
require positive trial temperature. It is not a residual/Jv or acceptance repair.

The analytic compressive flow case starts from the repaired infeasible seed and
converges to the same root with and without the production thermal PC. Masked
conduction tests use physical 2 pi r dr dz extensive energy, exact closed-face flux,
unchanged old state, and the same nonlinear root at matched tolerances.
