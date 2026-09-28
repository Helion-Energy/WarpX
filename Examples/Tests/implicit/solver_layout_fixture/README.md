# Solver vector layouts

This standalone CTest fixture compiles the actual `WarpXSolverVec`, `WarpXSolverDOF`
and `MultiFabRegister` implementations with native AMReX. Its small `WarpX.H`
facade supplies only geometry, the register, and the same `OwnerMask` route as
`WarpX::getFieldDotMaskPointer`. It does not construct a full WarpX simulation;
compile the production translation units against the real header as well.

Configure within the source tree, using a matching CPU AMReX package:

```sh
cmake -S Examples/Tests/implicit/solver_layout_fixture -B build-layout-rz \
  -DLAYOUT_GEOMETRY=RZ -DAMReX_DIR=/path/to/rz/amrex/lib/cmake/AMReX
cmake --build build-layout-rz -j 2
ctest --test-dir build-layout-rz --output-on-failure
OMP_NUM_THREADS=1 mpiexec -n 2 build-layout-rz/test_solver_layout \
  test.max_grid_size=4 test.periodic=1
```

Use `LAYOUT_GEOMETRY=3D` with a 3D AMReX package for the Cartesian fixture.
The `4`/`64` box sizes test multiple boxes and one box; periodicity applies only
in the final coordinate. An MPI run with one box also tests an empty local rank.
The tests cover concurrent E, phi, U and E+U layouts, a second two-component
nodal block, all vector algebra, independently scaled and unscaled norms,
local/global packing IDs, duplicate/periodic synchronization, source destruction,
move construction/assignment, interleaved simulations, frozen masks, registry
remakes, optional checkpoint roundtrip, and restart layout reconstruction.

`LAYOUT_REFERENCE_SOURCE=/read-only/baseline` additionally builds
`legacy_candidate` and `legacy_reference`. Run each with the same `test.kind`
(`Efield_fp` or `phi_fp`), `test.max_grid_size`, `test.periodic`, and MPI count,
using distinct `test.output` paths. The emitted `.rankN` files contain exact bit
representations of the norm and packed values after identical vector algebra.
Compare those files byte for byte. Each baseline process constructs one layout,
since the old implementation has one static DOF object.

## Optional thermal storage contract

The named block API follows `resistive_mhd` at
`592e2b3dfe9469eabab7bd8c6696a07ce1d68277`. It omits MHD's fused-vector and nonlinear
solver changes. A caller that owns an Eulerian thermal advance can explicitly
allocate `FieldType::hybrid_electron_energy_fp` as a **cell-centered, one-component
U_e in J/m^3**, with `checkpoint_restart=true`, then define:

```cpp
WarpXSolverVec thermal, coupled;
thermal.Define(warpx, "none", "none", {{"hybrid_electron_energy_fp", energy_scale}});
coupled.Define(warpx, "Efield_fp", "none",
               {{"hybrid_electron_energy_fp", energy_scale}}, electric_scale);
```

Names identify real registered MultiFabs. Each named block inherits its own
centering and component count. It receives its own periodic owner mask and DOF
map. Scales normalize packing and dot products; they do not alter stored physical
values. The old three-argument Define keeps unit scales. `Copy(FieldType...)`
keeps its field/scalar-only behavior; named register transfer is explicit through
`CopyMultiFabBlocksFromFields` / `CopyMultiFabBlocksToFields`. `Define(other)` shares
only the DOF/mask layout and allocates independent data from the source vector's
actual layout. Physical geometry/registry must outlive their solver vectors.

The enum addition does not allocate or advance a thermal field by default.
Existing `write_checkpoints` / `read_restarts` support opted-in named fields without
changes to the checkpoint format. The register reports which fields were read
and deliberately skips absent fields for legacy compatibility: a future new-mode
restart must check that U was loaded or perform an explicit, validated conversion.
This fixture does not certify full-application mode switching or checkpoint
metadata. No thermal scale/default, initialization from T/density, physical
inactive-cell constraint, residual, Jv, preconditioner, or mode selector is wired.
Owner masks remove duplicate storage; physical thermal constraints belong to the
future residual/active-set adapter and must not be inferred from E/phi parity.

The legacy E/scalar arithmetic retains its existing one-component contract;
named blocks support arbitrary component counts. Physical AMR, EB, full 3D
transport, and thermal boundary physics are outside this layout fixture.
