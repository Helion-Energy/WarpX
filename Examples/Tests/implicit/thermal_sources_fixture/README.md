# Pure Eulerian thermal source assembly

This fixture links the production `EulerianThermalSources.cpp` against native
AMReX. It does not run a WarpX advance, particles, the field solver or QDSMC.
The two implicit modes still need the parent-owned runtime integration below.

## Physical contract

The output is nodal **W/m3**, with positive `ElectronTotal` heating cell Ue.
`Ledger` integrates the same returned rates over clipped physical nodal dual
volumes and the full physical dt. It returns values, without incrementing a
physical accumulator. Mapping `ElectronTotal` to cell U uses the independently
delivered `KineticThermalMoments::RestrictNodalScalar`, preserving units and the
physical integral. Do not multiply by electron heat capacity or gamma again.

The source state has borrowed const views. The caller explicitly supplies fresh
rho, B, J and applied work fields for each coupled residual. A segregated thermal
solve may hold its deposited species/current/field context fixed for that solve;
Te and every Te-dependent coefficient/work mirror remain live. A separate helper
instance or refreshed views supply the next stage/dt/time. Parser executors are
borrowed; their owning parsers must outlive the helper. Input nodal duplicates
and all Yee interpolation ghosts must already be consistent and physically
filled; this helper never sums partially deposited input or changes its ghosts.

* Joule power preserves the existing Darwin source law
  `Qs = Zs qe^2 eta_s ns ne |J_nodal/rho|^2`, with
  `ns = [rho_s_raw/max(sum_charged rho_s_raw,qe*n_floor)] ne/Zs`.
  The raw species charge includes all charged species, even those excluded from
  relaxation. Neutrals are omitted. Per-species rho and J remain unscaled RZ
  deposits; they are not mistaken for physical number/current densities.
* Global field eta with a temperature argument receives **Kelvin**; the separate
  heating eta receives **eV**; species overlays receive **Kelvin** and raw species
  moments. The optional additive nodal eta represents the exact end-region eta.
  Hard density gates, C1 gate/source tapers, threshold redirection and species
  exclusions are explicit. Redirected ion powers and the associated per-ion
  variance rates are exposed separately. An excluded species' redirected share
  is explicitly declined, not redistributed or silently delivered.
* Relaxation is the physical simultaneous rate
  `Qe = -sum_s 3 ns qe nu(rho,max(Te_eV,1e-3),Ti_eV,t)(Te_eV-Ti_eV)`.
  The old species Ti is supplied as a nodal field with its explicitly chosen
  moment/ghost convention. The same per-species `-Qe_s` goes into the exchange
  output. No sequential exponential temperature update runs inside the source.
* Viscous **work** and hyperresistive heat use the actual applied `EV_out` and
  `EH_out` Yee mirrors. Each edge product `Jc*Ec` is formed first. In RZ, radial
  and axial edge energies are split equally onto their adjacent physical nodal
  dual volumes; theta work is already nodal. This is the current Darwin
  `QDSMCDepositDragWork` stencil, including axis/cap volumes. Interpolating J and
  E separately would not give the same work. Strain heat is an explicitly
  separate optional live input, never added to work heat a second time.
  Signed work is retained. Availability-gated work remains visible in signed
  declined-work channels, so delivered plus declined work equals applied work.
* A parser sink receives `(rho,Te_eV,|B|,t)` in W/m3; positive sink removes Ue.
* Stopping accepts a **previously consolidated immutable impulse** in J/m3 and
  emits impulse/full_dt, never impulse/(theta*dt). It never sums or clears the
  input. Below-floor overlap is exposed as a signed declined impulse rate.
  This is an adapter for a known impulse, not an implicit stopping predictor.

Raw charge sets fluid availability and physical species density. A stationary
density pedestal changes the thermal capacity through the moment/U adapter; it
must not amplify Q. The old temperature-increment implementation mixes raw
stage density with endpoint/pedestal capacities for some sources. Importing
those dTe increments directly would not preserve the physical rates above.
Cooling is never clipped here. Nonfinite state/rates, negative resistivity,
negative collision frequency and negative strain heat return false. The full
thermal solver must reject inadmissible U rather than manufacture floor energy.

## Parent call sites

1. Add the new cpp to the parent's implicit solver CMake/Make source list.
   Construct options from the existing model flags and compiled parser executors.
   Explicitly route the correct density component (the state field uses component
   zero, so use a const stage alias when the registry has two rho components).
   Preserve actual implicit semantics: a legacy `Te_shunt_threshold` configured
   in a deck does not activate shunting; the old implicit path never called it.
2. Before each source callback, update the moment adapter from trial cell U and
   the current deposited/MM density. Pass `NodalTemperature()` in Kelvin.
   Supply pre-gathered old species Ti in eV. Provide raw species moments at the
   selected stage; do not silently freeze these across coupled field probes.
3. Obtain applied EV/EH in caller-owned scratch from the existing field solver
   kernels. EV/EH must match the force's stencil, boundary mask and live trial
   coefficients. Extract/refactor the existing viscous stress/strain precompute
   to accept explicit trial Te. Do not temporarily copy trial Te into the
   accepted field registry or call the old mutating thermal integrator.
   The present helper deliberately has no `WarpX`, registry or field-solver
   dependency and cannot prove freshness of supplied mirrors by itself.
4. Call `Evaluate(state,nodal_rates,ion_rates)`. On false reject the trial. On
   success, map `SourceComponent::ElectronTotal` with `RestrictNodalScalar` into
   the stage callback's cell Q. This **replaces** the prescribed source, as the
   transport worker's callback specifies. The callback's local dQ/dU may be zero
   initially as an explicitly labeled PC surrogate: nodal T restriction and
   work fields make the actual source response nonlocal. Full response remains
   in every residual and Jv. Do not freeze source Q to build the combined Jv.
5. Once all field/thermal/circuit constraints accept the same physical state,
   call `Ledger` and commit U, stopping consumption and each source account
   once. `ViscousWork`/`ViscousStrain` are diagnostic subaccounts, not extra heat.
   Do not additionally run the legacy Joule/relaxation/conduction source stage.
   Rejected residuals and retries do not touch accepted sources or ledgers.
6. Realize any accepted ion exchange once, outside residual/Jv. The legacy OU
   update is **not** an exact samplewise thermal-energy match: it also drags
   bulk velocity toward ue and has stochastic finite-particle variance. Measure
   realized ion energy and its defect from the deterministic exchange ledger;
   do not claim exact total conservation from the analytic rate pair. A future
   conservative particle exchange must qualify its momentum/energy correction.
   This helper provides collision frequency and redirect variance rate, not a
   replacement particle operator. If stopping exists only after an accepted
   endpoint collision, explicitly qualify that split or derive a pure trial
   predictor and reconcile its realization; this helper does not supply it.

## Bounds and audit differences

Supported helper geometry: one complete RZ m=0 physical mesh without EB.
Existing 3D code is unchanged and the helper TU compiles in 3D, but construction
fails explicitly there. AMR/EB, an *active* temperature shunt, pedestal-temperature
cap and redirect kick cap fail explicitly. A stationary density pedestal itself
is supported by the capacity-independent source contract. Full field/moment,
particle, mode/restart and application qualification remain parent work.

The immutable Darwin base is e3a20fc92c0cf94c120e06eecc1975a109edd5f7. Its source
rates and work stencils were audited in HybridPICModel.cpp. The explicit
reference is commit 792712724b11c0f73b8a997769e5a9185ddfe362. That reference
still has arithmetic nodal work interpolation/interior clipping rather than the
newer Darwin RZ physical-volume correction. The helper retains Darwin's corrected
stencil. Both retain distinct strain/work and separate heating-resistivity laws.

Final MHD reference 592e2b3dfe9469eabab7bd8c6696a07ce1d68277 supplies the cell-U
stage and transport, not a replacement kinetic source law. Its fluid Joule
quench/ion-energy splits cannot be substituted for Darwin's species/force paths.
When HasResistivity() and implicit_push_excludes_resistive_field activate the
Darwin push correction, it subtracts the full-minus-nores Ohm field. The actual
include_resistivity gate covers viscosity too, so this difference includes eta,
hyperresistivity AND viscosity. Without that correction the viscous electric
force remains in the ion push. The optional explicit resistive collision
uses gathered species drift Vs-Ve and an exponential bulk shift. The grid
Joule law above is the existing source convention; it is not proof of exact
particle/field work equality for multistream species, unequal heating/field eta,
particle gathers or finite dt. No total-energy claim is made here.

Two bookkeeping improvements are explicit: the frozen Darwin Joule implementation
returns at its heat gate before recording the withheld rate (the current explicit
reference already fixes that); the helper exposes that rate. It also exposes
redirects skipped for excluded species and signed availability-declined applied
work. These do not change delivered electron/ion source power.

## Reproduction

Configure this standalone CMake directory with an existing native AMReX package:

```sh
cmake -S Examples/Tests/implicit/thermal_sources_fixture -B build-thermal-sources \
  -DAMReX_DIR=/path/to/native/amrex/lib/cmake/AMReX
cmake --build build-thermal-sources -j 2
OMP_NUM_THREADS=1 ctest --test-dir build-thermal-sources --output-on-failure
```

The optional `THERMAL_MOMENTS_SOURCE_DIR` CMake path points to a separately
reviewed KineticThermalMoments source directory. When available, the work tests
also check the complete edge-to-node-to-cell physical energy integral at the
axis, both caps, and across MPI/box seams. This does not alter either source tree.

Tests cover mixed-Z raw species fractions, exclusions, parser units, physical
source/exchange balance, source gates/redirection, signed stopping and work,
wrong-product discrimination, live nonlinear directional derivatives, input and
ghost purity, A/B/A, negative/nonfinite coefficients, and capability/parser/alias
guards. Run each test case with 2 MPI ranks and box size 4 or 64; the latter has
one box and an empty rank. CUDA is compile-only in the delivered evidence.

The dissipation fixture also links the actual production `WarpX_PEC.cpp`
boundary routines. `pmc_projection` compares the helper's homogeneous electric
increments against that independent reference, including mixed PEC/PMC
corners; `pmc_sources` checks exact force/work composition using production
current images; `pmc_guards` verifies the absorbing-particle PMC parity and
the native hyper stencil's response to an input physical ghost. Radial-upper
PMC and transformed E solves remain guarded.

`test_stopping_transfer` is enabled when the delivered moment adapter is
available. It qualifies the pure accepted endpoint cell-U transfer separately
from the implicit stage source helper. See `STOPPING_INTEGRATION.md` for the
signed ledger, primary cell floor, caller-owned consolidation/commit and the
explicit endpoint collision split limitation.
