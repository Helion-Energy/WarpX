# Native FD4 conduction edge qualification

The normal-eligibility change is deliberately a small correction: a face's normal FD4 term now depends only on its normal stencil window. Mixed terms retain their transverse guards. It does not upgrade the radial axis/wall flux closure.

`test_conduction_order_rz.cpp` calls the real `QdsmcConductionOnceFDAtState` in RZ or 3D. It sets fixed physical temperature profiles, density, and magnetic field after native initialization. Test-only coefficients and boundary settings isolate the operator; campaign inputs and defaults are unchanged. No copied production stencil or test-only hook is used.

For a linear frozen operator, a single SSPRK2 update is quadratic in time. The driver obtains its spatial RHS as `[4*(T(dt/2)-T0) - (T(dt)-T0)]/dt`, and repeats at half dt. It verifies that each native call used exactly one accepted step, checks temporal sensitivity, and measures native energy independently of subtractive RHS inference. Spatial errors are evaluated at nodes rather than on averaged plotfile data.

## Build and run

Use a source worktree with AMReX pinned by its `cmake/dependencies.json`. Configure an in-tree build with `WarpX_DIMS=RZ` or `3`, MPI enabled, AMReX assertions and bound checks enabled. The native targets are `test_conduction_order_rz` and `test_conduction_order_3d`; CUDA uses the same source. These standalone native probes follow the existing implicit-test `add_test` convention because `add_warpx_test` launches `app_<dim>` or a pyWarpX script, not a custom native executable.

```sh
cmake --build build --target test_conduction_order_rz -j 8
ctest --test-dir build -R '^test_rz_conduction_fd4_normal_edge_order$' --output-on-failure
python3 Examples/Tests/implicit/analysis_conduction_order_rz.py \
  --executable build/bin/test_conduction_order_rz --output /tmp/fd4-rz-new-attempt
```

The registered CTests use a fresh output directory on every attempt. For two-rank checks add `--ranks 2 --max-grid 8`. For 3D use the 3D target, `--geometry 3d`, and `--inputs Examples/Tests/implicit/inputs_test_3d_conduction_order`. The thin 8x8 transverse domain tests normal accuracy at physical walls, box boundaries, and corners; it is not a general 3D anisotropic convergence test. Use `--baseline` with a probe linked to the pristine library to verify the old second-order axial edge behavior.

Optional fixtures include `--density-slope 0.5`, `--beta 0.25 --perpendicular 1.e-4`, and `--limiter 1.e-5`. The latter is the physical flux-limiter factor; the production transverse limiter is still `ssmart`. The analytic oracle is the unlimited continuum operator, so an active physical cap is a finite/conservation test rather than an accuracy test against that oracle. Non-axial nodal probes exclude the outer four radial nodes from accuracy norms where the analytic field is incompatible with the closed wall. This exclusion is explicit: those rows do not qualify wall accuracy.

## Scope of the gates

The axial cosine gate requires measured order at least 3.7 at the axis, adjacent row, and across the full native domain, temporal sensitivity below 3% of its finest axis error, no new axial extrema, and relative native energy drift below 1.e-11. The baseline gate expects second order at the axis. Other manufactured fields are diagnostic rows, not a full-closure pass.

On production baseline `59e904ce4467c732ad7318884f5e1c985ed399ab`, the axial axis-row order is 1.99965; the guard correction gives 3.99704 on N=32,64,128. The radial `r^2` node-2 error remains 1/12, independent of spacing. With regular Br=0.25r, Bz=1, chi_perp/chi_par=1.e-4, the mixed smooth field also retains an axis error approaching 0.314. Neither remaining defect is fixed or hidden by the axial regression.

## Full closure requires a compatible energy representation

The production state is nodal point temperature, while its thermal ledger uses a diagonal mass from clipped physical dual volumes. For closed `T=(1-r^2)^2`, summing the exact pointwise RHS with this unchanged mass gives `2*h^2` (excluding the common 2*pi factor), despite zero physical boundary flux. Therefore uniformly fourth-order pointwise rows and exact unchanged-mass conservation cannot both be obtained by adjusting edge stencils alone.

A compatible design is persistent finite-volume electron energy `U_i=integral(3/2*n*kB*T dV)` and particle number `N_i=integral(n dV)`, or an explicit high-order mass map with a defined inverse. Reconstruct energy and density to recover point temperature, evaluate one shared physical face flux, and divide flux differences by exact volume. The axis uses even scalar reconstruction; physical walls use inward stencils and the existing physical flux law. This changes the state/consumer interface and must not be inserted as a silent relabelling of `Te`.

Before production integration, pressure/conductivity/gathers, entropy recovery, heating and relaxation sources, alpha stopping, boundary/leg capacities, floor budgets, diagnostics and restart records must agree on that representation. Marker filtering is not part of this design. General mixed terms require proper face quadrature; their current midpoint arithmetic average is inconsistent at the RZ axis even for a regular field. Explicit timestep limits and limiter/positivity behavior must be requalified for the actual completed operator.

The guard correction and its native regression can be reviewed independently of that conversion. A short FRC comparison, general anisotropy and cap/leg accuracy for a new representation, and GPU 3D qualification remain separate acceptance gates.
