# Native per-species thermal source context

`KineticThermalSpecies` owns raw per-species nodal charge, optional raw Yee
current, and a fixed-old nodal ion-temperature snapshot. Every charged species
is retained in native order. A relaxation exclusion suppresses only its Ti and
exchange, not its charge in the Joule/relaxation fraction denominator.

`KineticThermalSpeciesNative.cpp` is the production bridge. Link both core and
native translation units. The standalone fixture links the core and actual
production charge, implicit-current and double-pass temperature kernels;
production RZ/3D syntax and CUDA checks separately compile the actual bridge.
No full WarpX runtime or GPU execution is claimed by this component fixture.

## Parent integration

1. At step entry, call `DescribeNativeThermalSpecies(mpc)`, which returns every
   charged species in native order, with name, Z and mass. Set each descriptor's
   `relaxation_excluded`, `has_resistivity_overlay` and compiled overlay parser
   from the model. Keep parser owners alive. Neutrals are omitted; do not omit
   an excluded alpha or replace the raw sum with physical rho.
2. Construct `KineticThermalSpecies(physical_geometry, cells, dm, ghosts,
   descriptors, options)`. The geometry is normalized RZ Coord1 with identical
   domain/bounds/periodicity to WarpX. The guard width must cover the actual
   native rho and J deposition widths; preserve the configured precision of
   communication. Options include relaxation and actual EB/level/mode counts.
3. Before any trial particle push, call
   `FreezeNativeOldSpeciesTemperature(context, mpc)` once. The native bridge
   directly invokes the actual `doVarianceDepositionShapeN` kernels with
   private double-pass arrays. It does not call the public registry-backed
   `AccumulateVelocitiesAndComputeTemperature`. The native n/((n-1)*sumw)
   correction, relativistic velocity, mass/kB conversion and configured
   temperature filter are retained. Only private fields are filtered. No RNG,
   particle values, accepted temperature, diagnostic fields or tallies change.
4. After EVERY full materialized trial push, including signed field/thermal
   Jacobian probes, call `RefreshNativeThermalSpecies(context, mpc,
   actual_implicit_deposition_dt, MaterializedSpeciesTrial::FullParticleState)`.
   Pass exactly the dt used by the native push/current deposit; the existing
   `WarpX::PushParticlesandDeposit` passes `WarpX::getdt(0)`. No theta multiplier
   is guessed by this adapter. Then set `source_state.species=context.Species()`.
   Failed calls invalidate the private views and must reject the evaluation.
5. A thermal-only subsolve may hold these species moments after its fixed field
   push. Old Ti remains fixed throughout the step. A new field residual must
   refresh the full trial species state before evaluating any source. Do not
   reuse the previous residual's deposit or infer fractions from aggregate J.

Raw charge is deposited at actual trial midpoint particle positions, using the
native local deposit without inverse-volume scaling, reflection or filtering;
all source/destination guards are summed once. This intentionally differs from
the physical endpoint-averaged continuity density used for electron capacity.
The legacy raw fraction is rho_s/max(sum_all_charged rho_s,q_e*m_n_floor).
The configured pedestal and physical rho must not be substituted into that sum.
Raw current is needed only for per-species resistivity overlays. The bridge
calls the per-tile native `DepositCurrent(...,PushType::Implicit)`, reading saved
x_n/u_n and actual materialized midpoint state. The public vector overload
hardcodes Explicit and is deliberately not used. Raw current receives the same
one guard sum and no physical volume or field-boundary projection. The old
implicit source refreshed raw charge but held these moments during MM probes;
this new path explicitly refreshes them from the full trial state.

Old temperature follows the existing two-stage map: native Yee T[K] ->
cell-centered trace/3 in eV -> nodal eV. Cell-centered physical exterior ghosts
are zero as in `QDSMCAddTemperatureRelaxation`; periodic and box ghosts are
filled. This source Ti map is intentionally separate from the electron
pressure/current boundary images. The adapter owns both cell and nodal old Ti
and never refreshes them because a probe changed current particle momenta.

## Mass-matrix fallback and bounded capabilities

`AggregateMassMatrixOnly` is explicitly rejected. An aggregate mass matrix
cannot identify individual species responses: the fixture swaps two species'
responses with identical total rho/J but a different relaxation source. Parent
must use a consistent FULL particle response for both field and thermal blocks
when these sources are active. Merely pushing species for Q while keeping an
unrelated aggregate-MM field response is not the qualified fallback. Existing
MM paths stay intact for configurations that do not need this context.
Per-species MM capture/projection remains a separate future implementation.

The bridge initially supports RZ m0, single level, no EB, ordinary physical
particle containers, fixed positive ion charge and non-shared Yee deposition.
Direct and Esirkepov implicit current are allowed; tests exercise both actual
current kernels. It rejects explicit subcycled containers and variable-charge
ionization. When an overlay needs J_s, it also rejects any nsuborbits>1: a final
midpoint cannot recreate its piecewise orbit-averaged current. That case needs
supplied actual trajectory deposits. Charge-only common-eta Joule/relaxation
remains available with implicit suborbit particles; no J_s is requested there.
New 3D runtime is guarded; existing 3D code is unchanged and the new translation
units compile in 3D. Accepted stochastic ion exchange remains parent-owned.

## Reproduce

Configure with an installed CPU AMReX package, build `test_thermal_species`, then
run `ctest --test-dir <build> --output-on-failure`. The matrix spans shape1/3,
box4/64, MPI2 seams/empty rank, native raw/current parity, analytic old Ti,
Joule floor/alpha denominator, source directional response, aggregate ambiguity,
A/B/A, invalid recovery and explicit MM/old-Ti/descriptor/capability guards.
The standalone particle arrays are genuine inputs to production deposition
kernels, not synthetic physical-rho fractions or a full-app particle advance.
