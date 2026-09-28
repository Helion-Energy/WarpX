# Accepted native ion exchange

`AcceptedIonExchange` prepares copied Eulerian source coefficients without RNG,
then stages and commits the native OU ion update once, after the caller has
accepted and materialized the actual endpoint particles. Link both
`AcceptedIonExchange.cpp` and `AcceptedIonExchangeNative.cpp` in production.
No model, Theta, Darwin advance, Evolve or build-list file is changed here.

## Integration and transaction

1. Construct one noncopyable object per attempted step with normalized physical
   RZ geometry, current cell BoxArray/DistributionMapping, **every charged
   species** in native order (`KineticSpeciesDescriptor`), and
   `IonExchangeOptions`. Pass physical full `dt`, `raw_density_floor=m_n_floor`,
   source time, relaxation/redirect flags and actual capability flags. Excluded
   alpha remains an indexed no-op. Neutrals are omitted. This consumer never
   synthesizes species fractions; requested rates already use the native raw
   all-species denominator from `EulerianThermalSources`.
2. Before endpoint updates, call
   `Prepare(source_state, converged_ion_rates, nodal_electron_drift)`.
   Drift is required: three **nodal cylindrical** `(Vr,Vtheta,Vz)` fields in
   m/s from the same converged midpoint state. Do not reconstruct it from
   plasma current alone or supply a prior-residual registry image. The source
   state supplies physical nodal charge and converged Te[K]. All needed
   coefficients and requested work are copied; input lifetimes end on return.
   Prepare is pure and repeatable before the commit attempt. A bad preparation
   invalidates outputs but may be retried with corrected inputs; it consumes
   no RNG and writes no particles, thermal fields, registry or diagnostics.
3. Nu is the converged `ion_rates.CollisionFrequency`, which already depends on
   fixed-old species Ti. Nu, Te, drift and redirect variance rate each use the
   **ordinary arithmetic nodal-corner average** to cell centers; cells with
   averaged physical rho <= q_e*m_n_floor remain inactive. This coefficient
   gather intentionally follows the legacy cell lookup, not the conservative
   physical work restriction. RedirectEnergy is full dt times the gathered
   RedirectVarianceRate, in J per physical ion (variance numerator). No extra
   dt, theta, RZ factor, species fraction or mass factor is inserted later.
4. An optional diagnostic oracle can be passed as
   `Prepare(state,rates,drift,old_cell_Ti,&legacy_nu_parser)`. It evaluates the
   old cell parser `(rho_CC,Te_CC[eV],Ti_old_CC[eV],time)` only to report its
   maximum difference from gathered source nu. It **never** sets the new
   update's nu. Old Ti may come from `KineticThermalSpecies::OldCellTemperature`.
   Inputs must not alias owned coefficient storage. Parser ownership is needed
   only through Prepare; no executor is used during commit.
5. Complete the deterministic particle endpoint update and redistribute.
   Invoke `CommitNativeIonExchangeOnce(exchange, mpc)` exactly once. Native
   bridge verifies full charged-species metadata, native geometry and particle
   layout, reads actual endpoint NGP cells/theta, and preflights all finite
   weights/momenta/coefficient variances before drawing. Invalid/deleted ids or
   nonfinite positions reject before integer cell conversion. The caller must
   own these particles exclusively during commit. All calls are MPI collective,
   including empty ranks.
6. The accepted path draws three shared normals per active ion, stages all
   species' momenta and measures candidate energy, globally validates every
   candidate, then copies all staged momenta. A second attempt is rejected.
   A preflight rejection writes no particles and consumes no RNG. A stochastic
   staging failure also writes no particles but **has consumed RNG**; its
   `FailedAfterDraw` state is terminal. A full-step retry requires parent RNG
   state handling. No automatic retry, U adjustment, physical tally update,
   marker call, boundary integrator or stopping impulse consumption occurs.
   After success, add `Ledger()` channels once in the parent's transaction.

## Native mathematics and energy accounting

The reference is `HybridPICModel.cpp::QDSMCApplyIonHeating`, lines 11630–11945
on this commit's base. It uses endpoint NGP lookup and

```
drag = -expm1(-nu*dt)
sigma^2 = (-kB*Te*expm1(-2*nu*dt) + RedirectEnergy) / mass
u_new = u_old + (-drag*(u_old - electron_drift) + sigma*normal)
```

Stored momenta are native proper velocities u=gamma*v. The update preserves
this legacy nonrelativistic OU convention; measured kinetic energy uses the
native stable relativistic `Algorithms::KineticEnergy`. The new RZ consumer
rotates `(Vr,Vtheta)` to Cartesian at the endpoint azimuth before applying the
update. The old implementation is unchanged. Noise is isotropic in Cartesian
coordinates; a rotated realization must rotate its normal vector when checking
pathwise covariance, while ordinary independently drawn realizations agree in
distribution. The fixture checks a 12-point rotated ring.

Per-species requested relaxation/redirect work is full dt times the **physical
nodal dual-volume integral** of converged power, in J. It is not obtained from
particle weights or the coefficient average. The ledger separately reports:

- finite-step nonrelativistic OU expectation using actual endpoint moments and
  macro weights, including drift work;
- realized relativistic kinetic-energy change using the actual stored candidate;
- expected-minus-requested and realized-minus-expected defects.

The last defect includes stochastic sampling **and** the nonrelativistic OU
versus relativistic energy convention. The shared-draw channel attribution is
explicit: relaxation uses the same normal with only its thermal variance;
redirect is K(combined variance) minus K(relaxation-only). They telescope to the
actual change, but are not two independently realized collisions. No exact
per-kick exchange conservation is claimed.

Even the uniform zero-drift nonrelativistic ensemble has a finite-step defect.
For DeltaT=Te_mid-Ti_old in eV and x=nu*dt,

```
requested ion gain = 3*n*q_e*DeltaT*x
OU mean ion gain   = 1.5*n*q_e*DeltaT*(1-exp(-2*x))
expected/requested = (1-exp(-2*x))/(2*x)
expected-requested = -3*n*q_e*DeltaT*x^2 + O(x^3)
```

At x=0.01, 0.1 and 1 the ratios are 0.9900663, 0.9063462 and 0.4323324. This is
an O(dt^2) per-step mismatch before RNG, drift, physical/native quadrature,
coefficient interpolation and endpoint sampling differences. A future paired
mean adapter might use `(1-exp(-2*nu*dt))/(2*dt)` in electron requested transfer
while retaining physical nu for OU, or consistently eliminate the ion midpoint.
Neither is implemented here. Temporal-order and paired-transfer qualification
must precede a full source-closure claim. Parent owns reconciliation policy.

## Bounds and validation

New runtime is RZ m0, one physical level, no EB, fixed positive-ion charge and
ordinary physical native containers; explicit subcycled containers, variable
charge, active kick caps and temperature shunts are guarded. Existing 3D paths
are unchanged; new TUs compile in 3D but their new runtime is guarded. A merely
configured inert shunt threshold is not an active request.

The native full-particle source adapter is still required on every field/source
probe. Parent has separately guarded per-species resistivity overlays pending a
self-consistent old raw-species seed and species/current closure in the force
prepush correction. Component source/current tests do not qualify that runtime
force closure. Common-eta source arms are unaffected by that overlay guard.

Configure with a CPU AMReX package, build `test_ion_exchange`, then run CTest.
The tests exercise pure/frozen preparation, nonlinear legacy-rate comparison,
exact native expm1 arithmetic, rotated ring, analytic OU expectation and
finite-step mismatch, native stored-energy accounting, one-time application,
individual channels, excluded alpha, empty rank/seams, input and endpoint
rejection, RNG purity, staged failure, private alias rejection and capability
guards. The production bridge is compiled separately against actual native
headers for RZ/3D/CUDA. The component fixture uses actual native momentum and
kinetic-energy conventions on owned particle arrays; it does not execute a full
WarpX application or GPU kernel.
