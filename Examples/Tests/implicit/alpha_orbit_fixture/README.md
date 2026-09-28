# Native implicit alpha orbits

This fixture calls the production implicit Boris pusher with native alpha mass and
charge, 3.5 MeV proper momentum, third-order momentum-conserving gather, native
Esirkepov current deposition and the normal implicit endpoint conversion. It tests
prescribed fields only; it does not qualify fusion births, stopping, wall impacts,
or a self-consistent plasma stroke.

The five registered CTests cover RZ and 3D uniform-B gyration, RZ and 3D radial-E
circular orbits, and a 3D force-free crossed-field drift. Each uses a fixed physical
time of four initial gyroperiods and 16, 32, 64, 128 and 256 steps per gyroperiod.
Particle convergence remains at 50 iterations and 1e-10 tolerance. The analysis
checks analytic orbit error, second-order refinement, native continuity, exact
Boris electric work, finite contained gathers and accepted endpoints.

A diagnostic-only optional integer attribute, `alpha_orbit_picard_iterations`,
records the actual native iteration count. Ordinary simulations do not create it.
The qualification archive includes identical-physical-endpoint comparisons with
the unmodified native pusher, MPI seams and a single-box/empty-rank control.
The fixture explicitly resets this noncommunicated scratch before every push.

In a WarpX build with implicit tests enabled, run:

```sh
ctest --test-dir build -R alpha_orbit --output-on-failure
```

To register already-built immutable native probes without rebuilding WarpX:

```sh
cmake -S Examples/Tests/implicit/alpha_orbit_fixture -B build-alpha-ctest \
  -DALPHA_ORBIT_rz_BINARY=/absolute/path/to/rz/test_alpha_counts \
  -DALPHA_ORBIT_3d_BINARY=/absolute/path/to/3d/test_alpha_counts
ctest --test-dir build-alpha-ctest --output-on-failure -j 2
```

Each process retains a fresh run directory, runtime log, sampled trajectory and
hexadecimal accepted endpoints. `ALPHA_ORBIT_RESULT` reports physical error norms,
iteration counts, native layouts and containment. Charge/current/work are sampled
at the first, middle-third and final steps; native gather safety applies on every
inner iteration. The field arrays retain the runtime allocation; no ghost widening
or alternative integrator is used.

The RZ geometry stores native radius/theta but advances and measures Cartesian
trajectories. The first two uniform-B tracers orbit the axis with shape support
across the axis; other guiding centers cross native radial box/rank seams. Outer
walls and axial periodic boundaries are not struck. In 3D all traces remain inside
the periodic physical box. Boundary support therefore establishes contained
orbits, not a reflection or open-end qualification.
