# Native global eta-component kernel fixture

This calls the production `FiniteDifferenceSolver::HybridPICSolveE` after native
WarpX initialization, using private E, J, Ji, B, charge, pressure and trial-Te
MultiFabs. It does not advance particles, consume collision RNG, perform wrapper
boundary projection, or qualify a full Darwin energy balance.

Build `test_eta_component_native.cpp` against a full RZ or Cartesian 3D WarpX
static build with its normal compile definitions, generated headers, include
paths and link dependencies. The production `HybridPICSolveE.cpp` and its direct
`HybridPICModel.cpp` caller must both be rebuilt after the optional argument is
added; this changes the C++ symbol. Use the fixture object in place of the app's
`main.cpp.o` in the normal app link command. No new production source is added.

An optional immutable pre-change oracle is compiled with
`-DETA_CAPTURE_REFERENCE` and the original headers/library. Its native field
checksums are compared with the updated executable. Do not mix changed HPM class
headers with an old HPM library: use a coherent full build or a verified original
layout with only the FD kernel and unchanged caller replaced.

Run, for example:

```sh
python3 run_checks.py --rz-updated /absolute/rz/test-capture \
  --rz-original /absolute/rz/test-reference \
  --cartesian-updated /absolute/3d/test-capture \
  --cartesian-original /absolute/3d/test-reference \
  --mpi --output /absolute/new-results
```

The runner creates fresh directories, logs every command, and requires the exact
assertion reason for the alias/insufficient-guard negatives. It tests RZ Yee,
Cartesian Yee and Cartesian nodal kernels. The nodal arm uses native Cartesian
initialization, then constructs private collocated fields and the nodal FD
solver; it does not claim collocated Darwin particle-runtime qualification.
MPI arms use a multi-box layout and a single-box layout (an empty rank with two
ranks). `OMP_NUM_THREADS=1` is set for the matrix.

Each positive arm covers constant eta; raw rho, current magnitude and time;
Kelvin-dependent eta; updated and spatially varying borrowed trial Te; inactive
eta; push retention with temperature relaxation; Faraday; subfloor, zero and
negative raw charge; nonzero end-region eta; nonzero hyper-resistivity; and EB
row masks. The EB case toggles only the native kernel's `EB::enabled` predicate
after initialization and supplies manufactured private masks. It qualifies
skipped-row handling, not physical EB geometry or Eulerian-mode EB support.

The expected global component is checked on valid and ghost rows, including the
RZ theta-axis row. Scratch starts dirty before every call. The end-region test
requires its actual additional field to be 35 V/m at its maximum; the hyper test
requires nonzero applied EH. Capture-on/off E values and all field/scalar inputs
are compared. Original-library twins check unchanged physical field checksums.
The native fixture is GPU-compilable; CUDA compilation alone is not GPU execution
or a GPU arithmetic-equivalence claim.

The output is the **raw global** eta*J contribution in V/m on native E staggering.
It excludes end eta, per-species overlays, hyper-resistivity and viscosity. It
still needs every actual homogeneous wrapper boundary projection, auxiliary
centering and actual particle gather before it represents applied ion work.
The parent integration owns those operations. Spherical capture is explicitly
rejected; no spherical run is included in this matrix.
