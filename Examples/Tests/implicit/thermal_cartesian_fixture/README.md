# Cartesian electron thermal stage

The native AMReX fixture exercises production Stage, thermal PC, solver vector,
DOF and Newton/FGMRES driver code. A small WarpX geometry/registry facade is the
only replacement. No full Darwin advance, particles or source evaluator is run.

```sh
cmake -S Examples/Tests/implicit/thermal_cartesian_fixture -B build-cartesian3d \
  -DCMAKE_BUILD_TYPE=Release -DAMReX_DIR=/path/to/3d/amrex/lib/cmake/AMReX
cmake --build build-cartesian3d -j4
ctest --test-dir build-cartesian3d --output-on-failure
OMP_NUM_THREADS=1 mpirun --mca pml ob1 --mca btl self,tcp -np 2 \
  build-cartesian3d/test_thermal_cartesian test.case=nonlinear test.box=6
```

Cases cover constant temperature at variable density; full native 3D tensor
faces; FD2/FD4 spatial and theta temporal convergence; analytic advection and
diffusion; nonlinear smooth SMART, mixed theta, flux cap and mask; conservative
number-flux contact; actual residual directional derivatives and frozen-PC
invariance; all six physical wall faces; and signed compressive flow.
Each case runs as one box and many boxes, including MPI with an empty rank.

## Discretization

Cartesian heat flux is `q_d = -n_face F_d` in W/m², where each face bracket has
one normal term and **both** transverse tensor terms. The new 3D kernel follows
the native FD operator's full tensor sum while retaining the final MHD normal
FD2/FD4 composition, corrected smooth SMART donor, mask downgrades, trial-stage
conductivity and flux-before-cap mixing. Every face is shared by its two cells.
The divergence is the sum of face differences divided by their own `dx[d]`, and
the extensive volume is `dx*dy*dz`; no cylindrical radius enters any 3D row.
The compact frozen thermal PC similarly uses unit geometric weight in `a`, `b`,
RHS and recovery, plus all three signed central-advection/compression directions.
Its inner FGMRES remains nonsymmetric, with independently frozen density and
coefficients. Live full residual/Jv and source callback contracts are unchanged.

The actual FD4 normal composite is a seven-point second-derivative row:
`[-1/45, 1/20, 1, -37/18, 1, 1/20, -1/45]/h²`. It is fourth order and has an
exact constant-null derivative form; it is not the usual five-point stencil.
The Fourier-root oracle uses this actual symbol. Cross terms use the fourth-
order centered derivative in each of their two directions. The centered cross
control is used for analytic convergence; smooth SMART is a separately tested
nonlinear choice. The direct native FD face oracle uses an affine temperature
and its SH control, where the native and adapted full tensor fluxes coincide.
It does not claim the native nodal integrator and new FV residual are identical.

`THERMAL_GEOMETRY=RZ` builds a binary snapshot generator. With `THERMAL_SOURCE`
pointing at an exact pre-extension Stage/PC snapshot, it compares residual,
conduction/advection face arrays, wall derivatives, PC matrix-free action,
row action and inverse bytes against the extended implementation. Existing RZ
kernel code and numerical behavior are retained.

Physical adiabatic, prescribed outward flux, reservoir and nonlinear leg laws
apply independently to every nonperiodic Cartesian face. The same face electron
velocity enters central energy advection and `-P div(u)`. Conserved number flux
retains the explicit reconstruction=none / entropy-penalty=0 contract, with the
same arithmetic cell-density pair in all three directions.

This is a single-level, non-EB thermal operator qualification. Source-free 3D
application mode tests must bypass RZ-only source helpers only when all physical
source channels are inactive, and disable the separate RZ-only frozen pressure
coupling. Those application hooks remain the parent integrator's responsibility.

## Native activity counterpart

The separate `test_activity_cartesian` target checks the exact native gate,
all-eight-contributors cell closure, native edge and FV face masks, physical
`dx*dy*dz` open/mixed/lost volumes, intentional capacity-floor mismatch,
transactional invalid inputs and mapped-mask production energy conservation.
Run `test.case=gate`, `refinement` or `masked` in one/many boxes or MPI.
The cell closure remains an explicit conservative adaptation with O(h)
insulating erosion; no fractional conductivity or implicit OR policy is added.
