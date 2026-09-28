# Rejected implicit-step particle rollback fixture

This standalone AMReX fixture compiles the production `ImplicitParticleRollback.H`
and actual WarpX Get/SetPosition, order-3 charge deposition, physical particle
boundary kernel, and nodal boundary-image implementations. It uses a native AMReX
SoA particle container with the same noncommunicating implicit attributes.

Configure inside the source root:

```sh
cmake -S Examples/Tests/implicit/particle_rollback_fixture -B build-rollback-rz \
  -DAMReX_DIR=/path/to/rz-amrex/lib/cmake/AMReX -DROLLBACK_GEOMETRY=RZ
cmake --build build-rollback-rz -j 2
ctest --test-dir build-rollback-rz --output-on-failure
mpiexec -n 2 build-rollback-rz/test_particle_rollback \
  test.case=restore test.max_grid_size=4 test.suborbits=1
```

Use a 3D AMReX package with `ROLLBACK_GEOMETRY=3D` for the Cartesian counterpart.
The 20 CTests combine restore, identity guard, physical endpoint, absorption, and
actual redistribution tests with many/single boxes and optional suborbit state.
Run the same tests with MPI2: the single-box case leaves one rank empty. The
identity guard changes only rank 0, and validity is reduced before any restore.

Restore tests compare all compile-time attributes, including native RZ r/theta,
exactly (value and signed-zero equality), and compare redeposited charge including
images bitwise. The RZ theta values deliberately exceed the atan2 principal branch.
Repeated A/B/A and idempotent restores reuse the fixed saved start. Migration tests
call real AMReX Redistribute and demonstrate that the saved topology is rejected.

The endpoint fixture uses the same standard species finish algebra as
`WarpXParticleContainer::FinishImplicitParticleUpdate`, with actual Get/SetPosition;
it does not instantiate the full WarpX application or its virtual dispatch. The
pre-boundary virtual and materialized deposits agree. For an absorbing PMC axial
cap crossing, the virtual charge is 1 and the actual charge after the physical
boundary kernel and Redistribute is 0. This is a demonstrated limitation of using
an unfiltered virtual endpoint as an accepted post-boundary density.

The rollback API is opt in: call `SaveParticlesAtImplicitStepStart(true)` before
rejectable work, `RestoreParticlesAtImplicitStepStart()` on rejection, and
`DiscardSavedImplicitParticleState()` after acceptance. Keep all rejection gates
before physical BCs, particle creation/deletion/reordering, collisions, and
redistribution. This is a bounded transaction helper, not a particle checkpoint.
The default Save call allocates no rollback buffers and preserves legacy behavior.

Payload per particle is `sizeof(uint64_t) + native_coordinates*sizeof(ParticleReal)
+ (has_suborbits ? sizeof(int) : 0)`: double RZ 24 bytes (28 with suborbits),
Cartesian 8 bytes (12 with suborbits), excluding per-tile allocator/map overhead.
Existing x_n/u_n storage is reused; z_n already stores RZ z exactly.
