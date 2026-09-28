# Frozen pressure-to-field block fixture

This fixture validates `FrozenPressureFieldCoupling` against the production
kinetic pressure transfer and the actual native `HybridPICSolveE` kernel. It
changes no field residual, source callback, particle advance, or accepted state.

The upper triangular application is
`z_U = D^-1 r_U; z_E = A^-1 (r_E - B z_U)`.
`B` returns **physical V/m** for a cell perturbation in **J/m^3**. The sign is
positive because the physical field residual is `E_T + E_L - E_Ohm` and the
Ohm pressure term is `-grad(Pe)/rho`. Do not add theta, dt, or vector norm scales.

`Freeze(n_cell, n_node, weights)` owns copies of every coefficient and rejects
nonfinite/nonpositive densities or nonfinite/negative weights transactionally.
`Apply` uses allocated device scratch and neighbor communication, with no live
model access or coefficient reductions. It permits signed energy increments.
Its counters count successful freezes and applications, not field/thermal
inverse work. The caller retains separate counters for those inverses.

The exact transfer is `dT_cell=(gamma-1)*dU/(kB*n_cell)`, the native arithmetic
cell-to-node temperature prolongation, then `dPe_node=kB*n_node*dT_node` and
the native PEC pressure image. The distinct cell and nodal densities cannot be
replaced by a direct average of dU, especially at the RZ axis and outer wall.

Each supplied weight is `row_response * gate / rho_Ohm` on the native electric
Yee staggering. Interpolate raw nodal charge and pedestal separately BEFORE
the native smooth/hard floor; do not reuse the thermal advection face density.
`PressureResponseParameters` takes charge density in C/m^3 and absolute widths;
map the model's density floor by `q_e*n_floor` and its dimensionless floor and
Holmstrom widths by this charge floor. Pressure enable must match the actual
push/Faraday branch. Multiply by the native end-region factor and the actual
frozen prescribed/recovery row response. A replaced row has exact zero weight.
The helper's optional conductor raw gate reproduces the kernel algebra only;
it does not repair the currently unsupported mutating conductor-wall wrapper.

RZ m=0, one full physical mesh, normalized cylindrical geometry and no EB/AMR
are qualified. Other dimensions and transformed Ohm forms reject explicitly.
A tensor/curl-curl numerator solve needs its actual inverse after the pressure
numerator, not this E-form weight. Density/current feedback, Te-dependent
resistivity or viscosity, recovery-mask derivatives and the lower-left block
remain omitted PC terms. The full residual/Jv must retain all live responses.

Standalone CPU build, inside the source tree:

```sh
cmake -S Examples/Tests/implicit/pressure_field_fixture -B build-pressure-rz \
  -DAMReX_DIR=/path/to/amrex/lib/cmake/AMReX
cmake --build build-pressure-rz -j 4
ctest --test-dir build-pressure-rz --output-on-failure
```

A worker without the parent-owned moment source can set
`-DPRESSURE_MOMENT_SOURCE=/path/to/exact/moment/prerequisite`. No stub is used.
The matrix includes one and many boxes, periodic/PEC/PMC pressure images,
linearity, zero/constant states, variable-density constant-energy response,
A/B/A after live density/weight mutation, failed-freeze preservation and gate
algebra. `pressure_reject_transformed` checks the explicit unsupported guard.

The native oracle needs a completed matching WarpX RZ CPU Makefile build:

```sh
python3 Examples/Tests/implicit/pressure_field_fixture/build_native.py \
  --native-source /path/to/warpx --native-build /path/to/warpx/build \
  --output build-pressure-native
```

Run `build-pressure-native/test_pressure_field_native` with the absolute path
to `inputs_native_pressure` from a fresh working directory. Add
`boundary.field_lo="none pmc" boundary.field_hi="pec pmc"` and matching
reflecting particle ends for the nonperiodic native end-region case. Repeat
with `amr.max_grid_size=8` and `64`, and MPI2. The native oracle uses actual
`KineticThermalMoments(U +/- epsilon)` pressure and actual Ohm solves; its
expected field derivative is `-(E_plus-E_minus)/(2 epsilon)`. It also applies
the real PEC/PMC electric boundary operators. The separate recovery-mask case
qualifies an explicitly supplied row multiplier, not the live recovery solver.
There is no integrated Newton convergence or PC performance claim in this
bounded helper delivery; those require the parent's runtime hooks.

`pressure_oracle.unequal_density_components=1` adds the runtime two-component
charge register contract: component zero is deliberately different accepted
old charge and component one is live midpoint charge. Both the native Ohm
secant and the pressure coefficient adapter consume the midpoint alias. An
omission twin freezes the weight from old charge and must measurably disagree.
This exercises the production weight law, native pressure transfer, Ohm kernel
and boundary operators; it does not instantiate `DarwinThermalAdvance` or test
its private application wiring. Use the parent's runtime regression for that
last integration check. The default remains the equal-component control.

`pressure_oracle.recovery_rows=1` extends the native oracle through actual
`HybridPICModel::ApplyVacuumFaradayE` finite-difference probe replacement and
native PEC/PMC electric row projection. The Ohm denominator remains midpoint
charge. A live recovery mask instead uses component zero of the full native
charge register, matching the production replacement kernel and field PC.
Opposite old/midpoint threshold cases must distinguish a midpoint-mask omission
twin. Frozen masks, flux-only recovery, vacuum/transition masks and global masks
are controls. This tests the supplied pressure-block coefficients and the real
replacement kernel; it does not instantiate the private thermal-advance wiring.

Run with `pressure_oracle.unequal_density_components=1` as well. For a true MPI2
empty-rank control add `amr.max_grid_size=64 amr.refine_grid_layout=0
pressure_oracle.expect_empty_rank=1`; the fixture asserts that at least one rank
owns no boxes. No particle advance or nonlinear convergence threshold changes.

The native Ohm oracle now uses the same `PressureResponseStagger` helper and
`ablastr::coarsen::sample::Interp` as `DarwinThermalAdvance` for its raw-charge
and pedestal coefficients. A manually averaged coefficient alone cannot check
that assembly. `test_pressure_stagger` checks the helper on native RZ/3D Yee
centerings; padded storage exposes the historical RZ extra-plane error (50
for a plane difference of 100) without an invalid memory access. The full
pressure-PC RZ runtime guard is unchanged.
