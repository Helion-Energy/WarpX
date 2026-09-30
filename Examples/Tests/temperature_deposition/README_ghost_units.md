# Native temperature ghost-unit regression

This private fixture exercises the actual `ConvertVarianceToTemperatureAndFilter`
entry and native filtering; the particle modes additionally call the supported
`MultiParticleContainer::DepositTemperatures` entry. It does not evolve particles
or fields. The source repair is a separate commit.

Normal WarpX builds register `test_temperature_ghost_units_rz` and
`test_temperature_ghost_units_3d`, plus26 small CTests, when the corresponding
libraries exist. For example:

```sh
cmake --build build --target test_temperature_ghost_units_rz -j 2
ctest --test-dir build -R '^test_rz_temperature_ghost_units_' --output-on-failure
```

Each registration invokes `analysis_ghost_units.py`, which retains every attempt,
checks the native exit/result, enforces complete-FAB finiteness, and checks an
independent exact two-particle oracle in particle modes. Its deliberate NaN case
must report finite=false, both error norms exactlyzero, and exit2; an unrelated
abort does not count as a pass. Each CTest is a single MPI process, with two OMP
threads. MPI seam qualifications below were additionally launched explicitly with
one and two ranks; the metadata records their original launch context.

The input files cover nonperiodic RZ axis/ends/wall and3D faces/edges/corners,
periodic and mixed boundaries, anisotropic ghosts, staggered components,
passes0/1/2, and actual filtering disabled. Native MFIter4-cell tiles ensure real
thread/tile boundaries are exercised. Analytic bounds are fixed at256epsilon*1e7K
for prescribed fields. Particle rtol5e-5 and atol1e-14K remain unchanged.

The two particles per species have weights1:9 and identical positions, so their
positive cubic support and shape weights vary but the exact unbiased weighted
variance is(v1-v2)^2/2 at each receiving point. The independent oracle analytically
convolves that support with the separable binomial kernel; it never reads deposited
moment buffers to construct the expected result. This is distinct from assuming
finite random samples must produce a uniform population temperature.

Qualification receipts and exact reproducible compiler/link/launcher commands:
`/home/st247c/src/agent_workspace/artifacts/circuit-rz-temperature-bias-20260929/temperature-ghost-units-r01/`.
`MATRIX_RAW.json` records52 before/after executions; `ANALYSIS.json` and
`PARTICLE_ORACLE.json` preserve all criteria and failures. `REBUILD_R03_PASS.json`
pins final fixture/binaries. `BUILD_PASS.json` pins copied exactcc2RZ and retained
77088f4e97d03a1553f66f9be672d09bd2701587-compatible3D libraries,AMReX/compiler/options.
Those archives are prerequisites for the receipt's incremental-only build recipe.
The isolated source/normal CMake target is portable to a complete WarpX build,
including CUDA via `setup_target_for_cuda_compilation`; GPU execution remains
unqualified until independently run. Multi-level, single/mixed precision and new
physical boundary parity are outside this regression's demonstrated scope.
