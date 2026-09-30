# Viscous source replay from common native particles and fields

This test checks whether the native viscous source books the physical electron
energy it claims, and whether its instrumentation perturbs the calculation. It
uses the existing P64, seed17 hot-PIC fixture on the production source; it does
not change a production equation or add marker smoothing.

A 20-step native run produces a WarpX checkpoint plus a test-only supplement:
all registered field FABs including their box-specific ghosts, every physical
particle attribute/ID, and every raw QDSMC marker attribute/ID. Replays read the
saved FABs directly (a grown `ParallelCopy` would mix overlapping ghosts), clear
the destination particle containers, then read their checkpoints. AMReX particle
`Restart` appends and must not be used here without that explicit clear.

The native checkpoint supplies the clock and ordinary restart metadata. The
supplement preserves the registered fields and particle/marker attributes;
uncheckpointed solver/cache state is reinitialized equally in both arms. This
is a controlled matched replay, not a claim that restarting reproduces the
uninterrupted trajectory. Native end-of-step `ResetParticles` zeroes raw-marker
entropy, weights and velocities; their home positions and IDs are retained.

From the identical saved state the fixture calls the actual
`ApplyQdsmcEnergySources` with `dt/2`, both with and without energy-budget
instrumentation. The instrumented arm separately calls the actual Ohm kernel
into private scratch arrays to isolate its viscous E contribution and integrate
staggered `J.Evisc`. It brackets the source with an independently constructed
`1.5*n_eff*kB*Te` thermal energy using physical dual volumes, and compares the
actual temperature change, native ledger change, staggered work and nodal work.
The existing bound is unchanged:

```
128 * epsilon * abs(initial_thermal_energy) + 2e-11 * abs(expected_work)
```

The Python analysis also integrates local temperature differences before
subtracting global totals, and applies the same bound in the axis four rows,
outer four rows, and interior. `Qnu` is reported separately as a strain
observable; it is not added again to the booked work. The frozen source has no
Joule heating, Qei, alpha stopping, pedestal, shunt or active floor band. A
nonzero clamp fails this fixture rather than inventing a spatial attribution.

The fixture then evolves two ten-step continuations with instrumentation off/on.
Every loaded field FAB, every real/integer particle component and every ID must
match the producer exactly. Each instrumentation pair must remain byte-identical,
including per-box ghosts. Clock and progress checks prevent a zero-step false
pass. All serialized fields and particle data must be finite. The MPI extension
replays the same serial-produced state on two ranks; source states must agree
exactly across layouts, while continuation layout differences are reported as
measured norms without defining a new tolerance.

`inputs_viscous_common_state` is the unchanged historical hot-PIC template.
The driver applies the same P64 override and changes `amr.max_grid_size` from 64
to 16 only to create eight boxes for MPI seams. Domain, resolution, particles,
transport, conduction, timestep, field solver and source physics are unchanged.

Build and run the registered serial control:

```
cmake --build BUILD --target test_viscous_common_state_rz
ctest --test-dir BUILD -R '^test_rz_viscous_common_state$' --output-on-failure
```

For the bounded serial plus two-rank matrix:

```
python3 Examples/Tests/implicit/analysis_viscous_common_state.py \
  --executable BUILD/Examples/Tests/implicit/test_viscous_common_state_rz \
  --output NEW_OUTPUT_DIRECTORY \
  --mpi-prefix-json '["mpiexec", "-n", "2"]'
```

Commands, input/executable hashes, logs, native snapshots and `ANALYSIS.json`
remain in the output directory. Existing output is never overwritten. Registered
CTest runs use fresh timestamped subdirectories. The optional reader mode
`--analyze-existing --output OUTPUT_DIRECTORY` repeats only the analysis.

The current qualified target is CPU RZ with double field and particle precision.
CUDA compilation is supported by the target declaration; GPU replay, single
precision and mixed precision are unqualified and are not registered as passing
controls. The fixture does not close the full-PIC second-order temporal gate or
FRC physical attribution. The drag field remains excluded from the ion push, so
passing this electron-source bookkeeping test is not full moving-plasma energy
conservation or a physical validation of the drag closure.
