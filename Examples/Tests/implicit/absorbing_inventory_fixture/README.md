# Absorbing electron inventory

`RemapAbsorbingElectronInventory` is a pure cell candidate and accounting helper. It consumes virtual energy, virtual/survivor effective capacities from the same native moment options and pedestal, and `RestrictNativeMoment(lost_charge)/q_e`. The lost-charge input is linear: no pedestal or density floor belongs in it.

The output preserves cell temperature through `Us=Uv*(ns/nv)`. Equal capacities copy the energy bitwise, without arithmetic. Signed physical-volume ledgers obey `represented = raw_requested - floor_replacement`. A floor-bound cell can lose raw particle charge while retaining its represented energy; its positive replacement ledger records that distinction. A fixed pedestal is retained, not removed with the kinetic particles. Signed roundoff and `ns>nv` are measured without clipping. Absolute ledgers expose cancellation. False marks inadmissible inputs or unrepresentable results; output scratch must be discarded on failure.

The helper changes no particles, random state, model floor, epoch, or cumulative ledger. The caller must book it once after acceptance, separately from live floor-controller inventory, publish survivor charge consistently, and refresh accepted temperature/pressure from the survivor energy and density. The caller also owns the native charge-subset identity and the pure-loss consistency gate; this helper does not infer particle boundary actions.

Only new files are delivered. Parent registration needs `AbsorbingElectronInventory.cpp` in the implicit-solver `target_sources` list. This standalone fixture compiles the production helper and actual `KineticThermalMoments.cpp` against native AMReX:

```sh
cmake -S Examples/Tests/implicit/absorbing_inventory_fixture -B build-absorbing-rz \
  -DAMReX_DIR=/path/to/AMReX/lib/cmake/AMReX -DINVENTORY_GEOMETRY=RZ
cmake --build build-absorbing-rz -j4
ctest --test-dir build-absorbing-rz --output-on-failure
```

Use `INVENTORY_GEOMETRY=3D` for Cartesian cells. Tests cover signed accounting, cancellation, invalid data/overflow, zero loss, bitwise equal-capacity identity including signed zero, untouched ghosts, A/B/A and input purity, MPI seams/empty ranks, and actual native floor/pedestal maps with RZ axis3/axis4 and finite/periodic axial caps. These component fixtures do not qualify the caller's particle absorption/current continuity transaction.
