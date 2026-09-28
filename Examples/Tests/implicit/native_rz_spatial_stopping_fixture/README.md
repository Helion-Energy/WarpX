# Native RZ fixed-density spatial stopping prerequisite

`NativeRZSpatialStoppingEvent` is a standalone accepted-event transaction. It does not enable stopping in the live endpoint driver. It supports single-level on-axis RZm0, radial PEC with reflecting particles, periodic z or two PMC reflecting caps, native positive deposited raw density above the inertia floor, uniform positive electron temperature, physical electron mass, shape3 Esirkepov current and momentum-conserving electric gather. Initial A/B/D are zero and initial accepted Je=-instantaneous native ion current. No pedestal, vacuum, external/drive/circuit fields, AMR/EB, stochastic event, topology/DT, or3.5MeV alpha qualification is included.

Unknowns are a transverse electric impulse and midpoint plasma electron current. Every charged positive physical species receives the two electromagnetic half impulses; the selected fast species receives the unchanged native finite stopping kick between them. The native stored-theta collision-frame rotations are applied privately. The stopping gather/reaction transpose and electric MC/current transfer remain distinct. Particle positions/IDs/weights/topology do not change. The caller must supply a materialized accepted population after native boundary handling/redistribution; this is not a trajectory or whole-collision-handler adapter.

`Prepare()` copies and checks actual native state, evaluates private fields/particles with no RNG calls, and computes the signed inventory. `CommitOnce()` requires a nonthrowing cache invalidator and publishes momenta, plasma Je, cellU and A/B/D only after collective stale-state preflight. `Rollback()` restores those exact accepted bytes until `Finalize()`. The caller owns Te/Pe mirrors, accepted epoch, longitudinal E/rate recovery and any boundary/circuit publication.

The returned `CandidateConductorCurrent()` is essential: radial rmax tangential theta and z electric rows are constrained, so their Ampere reaction is conductor current, not plasma Je. Normal radial plasma current is untouched. Axis theta regularity is a separate zero-work restriction of the stopping covector. The live expression C-Ji-D is insufficient after this event unless it subtracts the conductor reaction with a phase-consistent plasma-current contract. This fixture never hides Jwall in D, Ve or heat.

The inherited `electron_momentum_change[0,1]` ledger entries are cylindrical component integrals, not Cartesian net momentum. Only axial component2 has that meaning inRZm0. The fixture measures the axial electric MC/mass and stopping-image transfer separately and bounds only their accounting error. Arbitrary transverse particle momentum remains an explicitly unrepresented mode.

Heat is the actual relative drag work, `-DeltaK_drag + sum w*S dot Ve`, restricted from native linear nodal weights through the physical dual-volume map. Finite MC/current, coefficient, curl, mean-current, and constraint defects never enter heat. The actual full energy defect remains reported even when the signed accounting identity passes. The bounded proton tests do not establish alpha accuracy, temporal order or closed total energy.

The CPU importer accepts:

```
cmake -S . -B build \
  -DNATIVE_RZ_STOPPING_EXECUTABLE=/absolute/native_rz_event \
  -DNATIVE_RZ_WRONG_AXIS_EXECUTABLE=/absolute/wrong_axis \
  -DNATIVE_RZ_WRONG_WALL_EXECUTABLE=/absolute/wrong_wall \
  -DMPIEXEC_EXECUTABLE=/absolute/mpiexec
ctest --test-dir build --output-on-failure -j2
```

In a native tree, add this subdirectory only when `TARGET lib_rz`; a3D-only parent must skip it. The target compiles the two prerequisite context/transpose TUs plus the new RZ helper and links a coherent nativeRZ library. The retained Cartesian3D helper is unchanged, and the new helper itself compiles as a rejected/guarded3D counterpart. CUDA RZ helper and fixture compile, but GPU runtime and exact retry-output byte equality are not qualified here. CUDA CTest registration deliberately does not claim CPU deterministic retry semantics; parent-owned GPU execution needs its own reviewed scatter/root reproducibility contract.

`analyze.py` independently reconstructs exact represented inputs using70-digit Decimal: physical owner/axis/cap volumes, stable particle kinetic changes, native edge inertia mass, magnetic and cellU changes, cubic MC auxiliary gather, staggered stopping velocity gather, actual native finite kick, Ampere/Faraday/longitudinalD, conductor/axis work and axial transfer. Bounds use positive input/work scales and declared operation counts, never observed signed errors. The original residual1e-12, native scalar rtol1e-12/atol0, speed and NR limits are retained. `test_parser.py` injects13 independent corruptions.

`operator.cpp`/`analyze_operator.py` and `axis_transpose.cpp` preserve the prerequisite native metric/range and restricted-covector experiments. Their exact build/run plans are in the sealed artifact. They do not establish a live wall-plasma publication law.
