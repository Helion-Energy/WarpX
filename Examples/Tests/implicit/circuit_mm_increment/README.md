# Native circuit MM retained-increment regression

This fixture targets the guarded source-free positive-density RZ, native circuit,
periodic-z, direct-Yee/correlated interval path. It does not qualify sources,
temperature-dependent pushes, vacuum MM, 3D endpoint MM, or outer longitudinal Jv.

`density_main.cpp` uses actual native geometry and divergence. A current increment
below the represented base-current ULP still produces its nonzero continuity
response. Zero response and A/B/A are exact; recapture invalidates readiness;
stale reuse rejects; NaN must remain visible to residual rejection. It uses
accepted rho component 0 because the midpoint component is not initialized before
the first particle push. The pure action interval of 1 s amplifies its observable
density change; no particles or simulation time advance in that test.

`retry_main.cpp` injects invalidation only after a real companion Freeze and
retained current composition. The test-only replacement ImplicitSolver is made
by `prepare_retry_hook.py`, which asserts each exact insertion site. The rejected
step restores physical fields/particle attributes, time, step, RNG and native C3
state exactly; solver xn/un/chord/impulse scratch is classified explicitly.
A fresh retry must match every accepted byte of a clean solve, including scratch.
The production solver has no test hook or runtime failure option.

Configure this directory alone with `CIRCUIT_MM_DENSITY_EXECUTABLE`,
`CIRCUIT_MM_RETRY_EXECUTABLE`, `CIRCUIT_MM_PLUGIN_LIBRARY`,
`CIRCUIT_MM_PLUGIN_CONFIG` and optionally `MPIEXEC_EXECUTABLE`. Alternatively
add it under a coherent WarpX RZ build with `lib_rz`. There are eight registered
CPU tests with MPI. Use `ctest --output-on-failure -j1`; OMP is fixed to 1.
Each test retains fresh run directories, exact commands, hashes and results.
The supplied input requires the pinned native rate/clock/identity plugin and
one-port model; paths are explicitly overridden by the runner. Test the final
production source with full ImplicitSolver/Theta header dependency closure.

Pinned qualification: native-circuit-mm-r63-r01/production-r01/ctest-r06.
Final production executable SHA3bae8197b08e64bc737cf078fb0bd6104e3b8805d9ae1352e0c9fcee7b5e19a3.
The full evidence/report lives in artifacts/darwin-thermal-moments-20260925/
2026-09-26-native-circuit-mm-r63. Original weak stored-state Q6 failures and
pre-fix/fixture-initialization failures remain preserved there and in raw runs.
