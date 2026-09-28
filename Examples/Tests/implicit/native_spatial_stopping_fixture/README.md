# Native spatial stopping event prerequisite

This fixture runs an actual finite native stopping kick with self-consistent electron reaction and a nonuniform transverse electromagnetic impulse. The initial domain is periodic Cartesian3D, uniform positive raw charge and Te, A=B=D=0 and accepted Je=-I from the native instantaneous deposit. Shape3 Esirkepov, native momentum-conserving electric gather, the repaired stopping velocity gather and its exact transpose are retained. No physical source guard, Evolve dispatch, stopping rate or legacy heat path is changed.

The mixed input produces BOTH magnetic response and nonzero longitudinal displacement. A constant control recovers the previously sealed harmonic limit. The field-only Fourier oracle checks native curl eigenvalue, curl work adjoint and longitudinal projection before the event. Every positive checks raw particle/field/RNG purity, stale attribute rejection, once-only commit, exact CPU rollback/retry, independent actual endpoint current redeposition and stable represented-input energy/momentum. A separately compiled wrong-B-sign variant must reject at the Faraday gate before publication. Budget, speed, nonuniform temperature and external-force capability negatives are separate.

## Build and run

From a coherent WarpX source tree, add this directory only when `TARGET lib_3d`; it links `lib_3d` and the explicitly listed context/transpose/event sources. The native stopping TU must include the sealed `NativeStoppingMap.H` factorization. Existing accepted-context/origin/transpose prerequisites must be present. The new event header changes no existing class layout. Do not combine an R53 event object with an older native WarpX library merely because both use3D.

A standalone importer supports an already coherent executable:

```
cmake -S Examples/Tests/implicit/native_spatial_stopping_fixture -B build-spatial-tests \
  -DNATIVE_SPATIAL_STOPPING_EXECUTABLE=/absolute/path/native_spatial_stopping \
  -DNATIVE_SPATIAL_WRONG_MAGNETIC_EXECUTABLE=/absolute/path/wrong/native_spatial_stopping \
  -DMPIEXEC_EXECUTABLE=/absolute/path/mpiexec
ctest --test-dir build-spatial-tests --output-on-failure -j2
```

The MPI tests use two ranks, with both real seams and an empty rank. `analyze.py RUN... --output RESULT.json` reconstructs all physical DOFs and represented particle inputs using70-digit Decimal arithmetic. It independently rebuilds the native MC auxiliary gather, native Ampere/Faraday differences, gradient-D check, compatible electron mass, magnetic/U/kinetic storage and the signed work decomposition. `check_parser.py GOOD_RUN NEW_OUTPUT_DIRECTORY` injects16 corruptions without rerunning physics. No tolerance is fitted to a measured defect.

## Equation and work contract

The unknowns are transverse electric impulse Psi [V s/m] and midpoint electron current. Each private ion map uses electric half/native stopping/electric half. Electric impulse uses the native Yee-to-nodal auxiliary interpolation plus cubic MC gather; stopping velocity uses its separate repaired Yee gather. Let S be the native proper-momentum drag impulse and b its mass-weighted reaction transpose. The electron response is Je1=Je0+M^-1(Psi+b), with the accepted compatible-Yee M=me/(e rho_edge). Solve P_T(KPsi+deltaI+deltaJe)=0 and P_LPsi=0, together with the midpoint relation. Then A1=-Psi, B1=-curl(Psi) and D1=-P_L(deltaI+deltaJe). Never assign the entire ion current jump to longitudinal D.

Relative drag heat is Q=-DeltaK_drag+sum(w S dot Ve_mid); no heat clamp or energy rescale. The native actual total DeltaKi+DeltaKe+DeltaWB+DeltaU remains reported with the independent relativistic convention, MC/current spatial transfer, current constraint, mean-current, transpose, heat-transfer, curl-adjoint and D dot Psi terms. Algebraic accounting closure is not zero physical total-energy defect. The two meshes are a bounded spatial check, not a temporal-order claim.

The periodic native divergence RHS mathematically has zero volume-weighted mean. Floating-point mean is removed only from this private divergence, and only after checking16 gamma(N+32) sum(2 max|F_c|/dx_c). Native projection remains rtol1e-12, atol0; the fresh coupled maximum-row gate remains1e-12. Signed removed means, positive input scales and bounds are recorded separately for current and impulse. A general incompatible scalar source is not projected away.

## Transaction and limitations

`Prepare()` owns private candidates and consumes no RNG. `CommitOnce(accepted_Je, nonthrowing_invalidator)` collectively preflights state, then publishes particle proper momenta, Je/U and A/B/D once. `Rollback()` restores these exact bytes until `Finalize()`; it requires no intervening topology, time or external-state change. The invalidator must discard caller source/stopping/MM/field caches. The caller still owns accepted Te/Pe mirrors, event metadata, interval-history policy and finite endpoint electric/rate recovery. This helper does not publish a complete production timestep.

The source amount uses a finite native stopping interval, while electromagnetic response is an ideal instantaneous impulse. It does not integrate a finite-duration Lorentz orbit through the newly generated B. There is no time-order qualification here. Ion relativistic-versus-NR impulse work has an explicit speed-based bound and declared accuracy budget; the electron bulk model is nonrelativistic. The tested proton speeds do not qualify3.5MeV alpha stopping.

Uniform positive density/temperature, physical me, no pedestal, fully periodic3D, fixed topology, zero initial A/B/D, default native centering2 and double precision are required. Circuit, prescribed/external field/current, particle external force/lattice, nonuniform density/temperature, RZ/walls, DT/fusion, stochastic events, AMR/EB and unqualified field algorithms remain excluded. Parent owns GPU execution. CUDA/RZ compilation is not a GPU event or RZ-physics claim. CPU exact output replay does not establish atomic-scatter byte identity on GPU.
