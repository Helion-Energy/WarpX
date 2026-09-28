# Frozen local source diagonal for the thermal PC

`EulerianSourceDiagonal` supplies a partial same-cell derivative of the existing Eulerian Joule and electron-ion relaxation source laws. It does not change `EulerianThermalSources`, the physical residual, accepted electron U, particles, RNG, registries or any ledger. Five new helper/fixture files are the complete source patch; the parent owns build lists and runtime wiring.

## API and units

Construct with normalized physical RZ geometry, the synchronized native nodal layout, the SAME `ThermalSourceOptions` used by the physical source, and optional `SourceDiagonalOptions`. Parser owners must outlive the object. The options and their time are fixed at construction; construct a new instance for each physical step when time changes.

Call `Freeze(source_state, cell_number_capacity, gamma)` at the nonlinear state used to freeze the thermal PC. `source_state.temperature_kelvin` must be that state's live nodal T; rho, raw per-species rho/J, fixed-old Ti, J, B and additive eta must describe the same state. All charged species remain in the denominator; exclusions and parser units are inherited from the actual source. The capacity is the positive native-restricted cell number density including pedestal, with the exact cell box/distribution map. It is not physical nodal rho divided by q_e.

The helper runs the actual source evaluator at copied T+h and T-h in private scratch, with only Joule and relaxation channels active. The default h is `min(T/4,max(2e-5*abs(T),1e-6 K))`. Nonpositive T has zero probe width and follows the existing source's zero gates. Additive end eta is retained. No accepted object or diagnostic sum is touched. No stochastic operation is invoked.

On success, immutable views until the next Freeze are:

- `NodalDerivative()`: Joule, relaxation, total in W/(m^3 K).
- `KinkMask()`: Joule and relaxation channel masks, described below.
- `RawDiagonal()`: actual same-cell dQ_cell/dU_cell in 1/s for the frozen local map.
- `PCDiagonal()`: `min(0,RawDiagonal())`, in 1/s.

`Freeze` invalidates prior outputs immediately. A false result means unavailable PC data. The caller may use zero source derivative or decline PC construction; it must independently require a valid physical source and state. Accessors require successful Freeze. Owned output/input aliases, nonfinite capacity/T and nonpositive capacity are rejected. No references to input arrays are retained after Freeze.

## Exact transfer diagonal and PC sign

For the existing moment map, T_c=(gamma-1)*U_c/(k_B*n_c), nodal T is an ordinary four-cell average A with even physical-boundary and periodic images, and the physical source restriction is R. The partial diagonal is

```
d_c = (gamma-1)/(k_B*n_c) * sum_v R_cv * (dq_v/dT_v) * A_vc.
```

For zero-based radial cell i, the low-node radial R weight is `(i+1/4)/(2*i+1)` and the high-node weight is its complement; axial weights are one half. A counts repeated images explicitly at axis/caps and periodic boundaries. The physical work quadrature intentionally differs from the native axis charge quadrature. Each A_vc factor is required, and its boundary multiplicity matters.

With h=theta*dt>=0, the thermal PC mass uses `n*(1-h*d_pc)`. Clipping only the PC derivative to d_pc<=0 preserves a positive mass at least n. Positive source feedback remains in the true residual and Jv, but is omitted from this damping surrogate. This does not guarantee nonlinear convergence. Nonlocal thermal derivatives and all density/species/current/B responses also remain solely in the full residual/Jv.

When the finite-difference interval spans the known Joule redirect threshold, only its Joule slope is zeroed. When it spans the 1e-3 eV relaxation parser-argument floor, only the relaxation slope is zeroed. Masks report these omissions. Arbitrary user-parser nondifferentiability is not detected. Density gates/tapers are held fixed, so their zero regions inherit exact source zeros.

## Parent wiring

Link `EulerianSourceDiagonal.cpp` with the existing source helper. Build it from the same step source options. In the stage source callback, first evaluate the TRUE source at the current full context and restrict its ElectronTotal normally. Freeze this derivative with that same context and the stage's native cell number capacity; copy `PCDiagonal()` into callback `local_dQdU`. Copying is required: callback scratch and future source evaluations must not alias helper output. The conduction PC then snapshots density and this derivative at its normal Freeze. The derivative is a PC approximation only; do not substitute it for the full physical Jv or replace the true source Q by a linearized law.

Viscous/hyper applied work, strain heating, external sinks and stopping derivatives are omitted. Their physical sources are not disabled. Unsupported model transforms such as active shunts, caps, EB or multiple levels remain guarded. Per-species eta's local T response follows the existing formula, but this component does not qualify the parent's species-dependent prepush force fixed point. Retain that independent runtime guard.

The constructor is qualified only for zero-based RZ m0, r_min=0, one level and no EB. Existing 3D code remains unchanged and this TU compiles in 3D, but it must not be constructed there. A source-free Cartesian 3D lane should use Q=0 and dQdU=0 without constructing either RZ source helper, while preserving trial moment/pressure refresh and zero acceptance ledgers.

## Reproduction and coverage

Configure the fixture with an MPI-enabled RZ AMReX package and the separately delivered native moment adapter (qualified reference commit b6e106d17dcce5aa3fd5219afb65449fda97a6a6):

```sh
cmake -S Examples/Tests/implicit/source_diagonal_fixture -B build-source-diagonal \
  -DAMReX_DIR=/path/to/rz/amrex/lib/cmake/AMReX \
  -DTHERMAL_MOMENTS_SOURCE_DIR=/path/to/qualified/moment/headers-and-source
cmake --build build-source-diagonal -j 4
OMP_NUM_THREADS=1 ctest --test-dir build-source-diagonal --output-on-failure -j 4
OMP_NUM_THREADS=1 mpiexec -n 2 build-source-diagonal/test_source_diagonal \
  test.case=basis test.max_grid_size=4 test.periodic=0
```

There are 36 native cases: analytic, basis, gates, kink, clip, purity, invalid, kelvin, rawfloor; each uses box4/64 and periodic/nonperiodic axial boundaries. Repeat all with two MPI ranks; box64 includes an empty rank. Basis probes pass U through the actual KineticThermalMoments T map, actual source law, and physical RestrictNodalScalar at axis/caps/interior/seams. Finite-difference steps 2e-5 and 1e-5 and independent U-probe refinement are compared. Tests also cover exact alpha denominator/exclusion, Kelvin versus eV eta convention, physical gate/raw denominator floor distinction, zero regions, separate channel kink masks, positive-feedback clipping, copied/frozen output behavior, deterministic A/B/A replay, no RNG advance/input writes, invalid input and private alias rejection. RZ and 3D production-header syntax checks and CUDA sm86 compilation cover the new TU; no GPU execution is claimed.
