# Pure implicit endpoint compatibility audit

`warpx::implicit::AuditImplicitParticleEndpoints(WarpX&, int lev,
EndpointAuditFields const&)` inspects current native halfstate particles and their
saved x_n/u_n. It returns global particle and per-issue counts, `ok()`, `count(issue)`
and `summary()`. It changes no particles, IDs, counters, RNG state or field data.
The bridge uses GPU-native reductions followed by MPI sums, with no full-particle
or field round trip to host.

The caller supplies the actual nodal endpoint density MultiFab, Yee current[3],
nodal MC gather_e[3]/gather_b[3], and explicitly qualified
`gather_filled_ghosts`/`current_deposit_ghosts`. Allocation alone does not prove that
gather ghosts contain valid images. The bridge checks the field shapes,
distribution, algorithm and configured guard widths. It is currently qualified
for a static single-level shape3 implicit Esirkepov/MC kinetic population in RZ
(axis-inclusive m=0, including WarpX's native Coord0 Geometry) or Cartesian 3D.
The standalone 3D counterpart does not qualify the entire Eulerian runtime in 3D.

Call this before virtual endpoint deposition and before irreversible acceptance.
The result reports nonfinite state, invalid IDs, out-of-domain saved starts,
absorbing/open, thermal or other unsupported endpoint boundary actions,
unresolved radial reflecting wall endpoints, and gather/current/endpoint/projection
reach violations. Reject with the parent field/circuit/U and exact particle
transaction while physical boundary actions, collisions, injections, sorting and
redistribution have not occurred. Field and particle BC selections are preserved.
Periodic endpoints may pass if their original-box support fits.

An RZ radial endpoint within a roundoff envelope of the reflecting wall is rejected
as unresolved. This includes every final pusher clamp outcome (>2-bounce safety
fallback) but also some valid wall-touching orbits. Post-reflection halfstate
cannot reveal bounce count or previous Picard clamp history; there is no exact
bounce-count claim. Non-radial reflected endpoints requiring a later boundary
position change are initially rejected. The gate is deliberately conservative at
physical and stencil thresholds.

A post-push audit cannot retrospectively protect intermediate Picard iterates or
an already performed deposition. Keep native pusher/deposition guards; this helper
is an acceptance/reach gate for the inspected state. Adding a gate inside every
unsafe kernel requires separate pusher integration. It also does not prove that
injection/scraper/collision hooks leave the endpoint population unchanged.

Pure endpoint rejection does not implement open-boundary energy closure. The
preceding native absorption fixture measured virtual charge1 versus actual
post-BC surviving charge0. Supporting such crossings requires matching trial
current divergence, endpoint number loss, thermal boundary energy, and accepted
loss accounting; late redeposition/Te refresh alone does not supply that closure.

Configure inside the source root using a dimension-matched AMReX package:

```sh
cmake -S Examples/Tests/implicit/endpoint_audit_fixture -B build-endpoint-rz \
  -DAMReX_DIR=/path/to/rz-amrex/lib/cmake/AMReX -DAUDIT_GEOMETRY=RZ
cmake --build build-endpoint-rz -j 2
ctest --test-dir build-endpoint-rz --output-on-failure
mpiexec -n 2 build-endpoint-rz/test_endpoint_audit \
  test.case=periodic test.max_grid_size=4
```

Use AUDIT_GEOMETRY=3D with a 3D AMReX build for the Cartesian fixture. The17 cases
run with many/single boxes (34 CTests per geometry): interior, axis, periodic,
absorbing, thermal, planar reflection, exact/near radial wall, interior reflected
endpoint, gather/current/endpoint/projection reach, nonfinite, identity, invalid
old state, and A/B/A ranges. Single box with MPI2 leaves one rank empty.

Tests compare all particle real bytes (including NaN payload), IDs, suborbit
counters and RNG sequence across pure evaluations. Accepted cases call actual
shape3 MC gather with NaNs outside filled ranges, native implicit Esirkepov
current deposition, and endpoint charge deposition. Native writes outside the
qualified ranges must stay zero; the gather must stay finite. The production
WarpX bridge is separately CPU/CUDA compiled, not instantiated by this fixture.
