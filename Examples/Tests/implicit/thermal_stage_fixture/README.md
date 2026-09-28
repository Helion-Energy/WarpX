# Native fixed-context thermal stage fixture

This fixture compiles the production `EulerianThermalStage`,
`ThermalConductionPC`, `ThermalStageSolver`, `WarpXSolverVec` and DOF/register
sources. It reuses the existing layout fixture's small WarpX geometry/register
facade. The Newton driver uses the production `FlexibleGMRES`; there is no toy
dense solver or globally gathered field. Real-header production compilation is
an additional check, not a full WarpX application link/run.

Configure inside a source checkout with a compatible RZ CPU AMReX build:

```sh
cmake -S Examples/Tests/implicit/thermal_stage_fixture -B build-thermal-rz \
  -DAMReX_DIR=/path/to/amrex/lib/cmake/AMReX -DCMAKE_BUILD_TYPE=Release
cmake --build build-thermal-rz -j 4
ctest --test-dir build-thermal-rz --output-on-failure
OMP_NUM_THREADS=1 mpiexec -n 2 build-thermal-rz/test_thermal_stage \
  test.case=density test.max_grid_size=8
```

Cases `eigen`, `constant`, `nonlinear`, `density`, `boundary`, `rows`, `rollback`,
`cap`, `source`, and `invalid` accept `test.max_grid_size=8` (multiple boxes) or `64`
(one box, including an empty rank with MPI2). All tests use native MultiFabs and
normal scalar MPI reductions. CTest also checks explicit rejection of the
unimplemented leg law, zero B-floor, nonpositive density and a reservoir on a
periodic boundary. The prior corrected-donor/mixed-stage kernel test is retained.

## Implemented equations and units

At fixed copied electron number density `n`, field `B`, source and old energy:

```
e = U/n
T[eV] = (gamma-1) U/(n q_e)
F(U_s) = U_s - U_n - theta dt (source - div_RZ(q))
U_end = (U_s - (1-theta) U_n)/theta
```

`U` is cell-centered J/m^3, `source` is W/m^3, and the conductivity expressions
accept `n,Te,t` and return W/(m K). This bounded API owns native AMReX parsers;
expressions must resolve their constants before construction. The coefficient
adapter uses arithmetic face physical kappa, `n_face=(n_L+n_R)/2`, and
`chi_face=(gamma-1) kappa_face/(n_face kB)`. Thus density changes alone never
create conduction at constant temperature. No mass-density or kinetic moment
approximation is hidden in this fixed-context API.

Each residual rebuilds kappa from trial U and uses the corrected MHD Chacon
FD2/FD4/SMART kernels. The old/live limited gradient brackets use the same live
coefficients and are mixed with `w=theta_c/theta`; the total flux is capped once
at the conduction-stage free-streaming limit. There is no RKL2, explicit thermal
time step, clipping, accepted-state import/export, tally or random draw.

At physical faces, adiabatic means zero q; prescribed flux is outward W/m^2;
a reservoir fixes T on the face and exchanges through the half-cell distance.
Reservoir flux is capped after old/live mixing, if enabled. Prescribed flux is
the requested physical flux and is not silently recapped. The r=0 axis has
zero area and must be adiabatic. End and radial wall faces are counted once;
there is no independent cell source for the same wall. A finite-length leg is
explicitly unsupported. Periodicity belongs to geometry, not a wall law.

The density-weighted MHD PC solves `(R + theta dt K) y = rhs`, `dU=R y`, using
the emitted frozen `n_face chi_nn w/cap^2` and RZ metric factors. Distributed
MLMG uses the MHD two-fixed-cycle, smoother-bottom configuration. The copied
MHD row emitter is unchanged. Max-order 2 makes the homogeneous reservoir
image exactly the residual's half-cell row; MHD documents its default-order-3
row as an approximation. The surrogate freezes physical coefficients,
limiter and cap temperature and drops wide FD4/cross terms. It is not claimed
to be the full nonlinear Jacobian. Row/MLMG/compact-Jacobian equality and
PC-on/off matched-root checks keep this distinction testable.

The transactional scalar driver uses the named M2 U vector and native
FGMRES. The field Newton/Jacobian templates instantiate field-specific PC,
mass-matrix and diagnostic hooks; this slice does not alter those paths.
Symmetric finite-difference Jv probes include live thermal coefficients and
limit the physical perturbation by the true positive endpoint. PC coefficients
freeze only at Newton bases. Armijo backtracking requires a finite decrease;
FGMRES checks its true linear residual; final acceptance recomputes the full
physical nonlinear residual and checks the endpoint. Every failure leaves the
stage and endpoint outputs, including ghosts, untouched. Floor-bound solves
may reject: no active-set/free-subspace acceptance or floor-energy repair is
implemented here.

## Scope limits

This is a working scalar solve for one RZ physical level without EB and fixed
prescribed context. It does not select `decoupled_jfnk` or `coupled_jfnk` in a
WarpX application. Central advection/compression, deposited/MM density/current
transfers, nodal pressure reconstruction, trial field/thermal cross derivatives,
Darwin source-work pairing, actual leg physics, mode/checkpoint metadata and
outer circuit/longitudinal closure still require parent integration. Existing
3D, field/phi, particle, implicit Je, CHIEF/Green and native circuit paths are
unchanged. The new operator explicitly rejects non-RZ construction; compiling
its translation units in 3D does not qualify a 3D thermal implementation.
