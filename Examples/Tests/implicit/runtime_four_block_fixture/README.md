# Actual runtime E/U directional derivatives

This Linux/GNU-linker fixture links matching native WarpX headers and libraries.
It wraps the public `SolveThermalSystem` entry reached by real
`WarpX::InitData` → `Evolve` → `ThetaImplicitHybrid::OneStep` →
`DarwinThermalAdvance::Solve`. The wrapper receives the real combined operator
and state after normal step setup. No production translation unit, class layout,
header, field map, or residual implementation is replaced.

```sh
python3 Examples/Tests/implicit/runtime_four_block_fixture/build_native.py \
  --native-source /path/to/snapshot/source \
  --native-build /path/to/snapshot/cpu \
  --output build-runtime-probe
python3 Examples/Tests/implicit/runtime_four_block_fixture/run_qualification.py \
  --binary build-runtime-probe/test_runtime_four_block \
  --output build-runtime-probe/results
```

The build snapshot requirements are the same as `mm_pc_particle_fixture`: native
CMake `flags.make` and `link.txt`, generated headers, matching static libraries
and unchanged dependency include paths. The wrapper labels explicitly use the
GCC/Itanium ABI; a different ABI needs its own linker adapter. The native CUDA
header compile is a separate compile-only gate, not GPU execution.

At fixed accepted step references and fixed outer longitudinal field, the test
forms E-only, U-only and mixed directions, then evaluates an epsilon ladder.
For every pair, `Residual(...,false)` supplies the full-particle reference.
A fresh physical base and `Freeze(...,false)` rebuild the mass matrices before
`Residual(...,true)` supplies the Jacobian route. The thermal PC is not applied.
Every nonlinear context remains live through the actual runtime callbacks.
Additional link wrappers count real whole-level particle pushes and
`PreRHSOp` stages; they forward every argument unchanged. These counts are per
rank. They do not count same-translation-unit calls to the internal MM action.

The electric direction has a 1000 V/m scale. The thermal direction is
`0.2*U_base*(1+0.3*cos(pi*r/R)*cos(2*pi*z/L))`; its temperature is always computed
from live density. E/U block norms omit solver reference scales and use the
native vector's duplicate-ownership masks. Their units remain V/m and J/m³.
Moment output reports componentwise infinity norms of current [A/m²], midpoint
charge [C/m³], actual trial temperature [K] and pressure [Pa]. No theta-dt or
norm-scale multiplier is added to the physical field residual.

`PROBES.jsonl` records the full and approximate derivatives, their differences,
physical moment responses, centered Taylor remainders, refinement differences,
zero-probe consistency, A/B/A and work counts. `analyze_case.py` checks roundoff
purity, mixed-direction linearity, quadratic thermal Taylor convergence,
nonzero pressure/thermal responses, the expected U-only particle response, and
exact full-reference equivalence when the MM Jacobian flag is off. Numerical
purity is bounded by 128 machine epsilons relative to the anchor residual; this
allows the observed one-roundoff MPI boundary-row difference. Accepted U must
remain exactly unchanged in the executable.

The Q6 smooth-window check uses epsilon `.00625` and reference `.0001` for
all cases and blocks. The sum of recorded adjacent derivative-difference norms
conservatively bounds their difference by the triangle inequality. Each block
normalizes by its own full-particle derivative norm, so weak cross blocks are
not hidden by the diagonal scale. Both this bound and the adjacent refinement
change must be below `1e-5`. This is a measured finite-difference consistency
check, not an exact analytic Jacobian oracle. The final `.00001` step is retained
as a subtraction-noise diagnostic; the zero-B E-to-U block exceeds `1e-5` there.
The nonlinear thermal centered differences exhibit second-order refinement in
the resolved window. Cap transitions, the temperature floor, active-density
transitions and source channels require separate Q6 cases.

The qualification runs eleven cases: zero-B/curl-free, constant eta, live
eta(Te), two timestep reductions, an explicit density-projection omission twin,
one-box, two-rank, full-particle-route control, and PMC-end serial/two-rank cases.
The varying static B supplies nonzero plasma current for the eta(Te) gate. Its
resistivity parser takes **Kelvin**; the nonlinear conductivity parser takes
**eV**. All heating/ion-exchange source channels are off. Nonlinear conduction,
central transport, compression, actual particle/field coupling and physical
moment transfers remain active.

`run_case.py` runs one configuration; its temperature-eta validation assumes a
nonzero plasma current, as supplied by the qualification's B profile. Outputs
must be new directories. The suite uses OMP=1 and Open MPI local TCP for MPI2.
Very tight push-context tolerance `1e-13` can reject particular reference
probes on r19; default `1e-12` controls complete. Those diagnostic failures are
retained separately and are not accepted residuals or missing Jv values.

After the probes, the fixture restores the input trial and throws a local
completion marker caught by its main. It exits before Newton or physical
acceptance. It therefore tests the actual combined residual and Jacobian route,
not the Newton driver's epsilon selection, a converged root, the longitudinal
outer solve, a field-PC inverse, source/species mass matrices, accepted particle
wall crossings, 3D, or GPU runtime. MM/full disagreement must be reported as an
approximate-Jacobian error unless an independent residual defect is established.
