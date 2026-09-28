# Implicit gather safety

`doGatherShapeNImplicit` returns a `warpx::particles::ImplicitGatherResult`.
The native pusher checks that result for every initial and intermediate Picard
gather. A failed gather exits that particle kernel. The host reports typed failure
counts and aborts before ordinary current deposition or suborbit fallback. The
suborbit path also checks each gather before using its result. No invalid field
is replaced with zero, no ghost region is enlarged, and no rejected orbit is
marked converged.

`CheckMomentumGather` is a pure GPU-native pre-access test of the Direct/MC
stencil. It takes actual six `Array4` views, their individual centerings, the
native reference origin/lower index, inverse spacing, shape order and RZ mode
count. It reads metadata only. The Direct implicit gather calls it after the
existing PEC orbit crop, using the exact physical point sent to the native
kernel. It repeats the native `Real` coordinate transform and shape index rule:
truncation toward zero, half-cell shift for cell centering, half-index shift for
even orders, first index `anchor-order/2`, and all `order+1` loop entries.
Zero-valued weights still require allocated values because the native loop reads
them. The float-to-int cast occurs only after finite and integer-range checks;
wide integer arithmetic checks the complete support against every actual FAB.
All RZ mode components and unused spatial array dimensions are also validated.

Finite starting/trial coordinates, starting/current momenta, and gathered plus
external field values are checked before they can reach the next gather. Grid
NaNs are reported as `NonFiniteField`, before the momentum pusher is called.
This is the exact trigger in the matched parent r20 replay: all six gathered
fields are NaN at a finite, in-range first-particle point. Original-binary gdb
then observes indices near INT_MAX at the subsequent unsafe gather. The guard
addresses the memory safety failure; the upstream generation of invalid fields
is a separate nonlinear-solver defect to diagnose.

The fixture compiles actual production gather and shape headers. Admissible
results are bitwise compared with the original native direct gather, with
unchanged input fields. Tests cover orders1--4, nodal and Yee layouts, six
components and every spatial direction, RZ axis and modes, actual finite-cap
cropping, finite huge and nonfinite coordinates, NaN grid fields, bad component
count, A/B/A, exact-zero endpoint weights, multi-box seams and an empty MPI rank.
CUDA compilation uses the same kernels. `GATHER_SANITIZE=ON` adds CPU ASan,
UBSan and float-cast-overflow checks. Deliberate `test.unguarded=nan` and
`test.unguarded=range` counterfactuals call the native unchecked kernel and must
fail under sanitizers; the checked fixture must pass.

Example standalone builds (build directories stay inside the WarpX source):

```sh
cmake -S Examples/Tests/implicit/gather_safety_fixture -B build-gather-safety/rz \
  -DAMReX_DIR=/path/to/rz-amrex/lib/cmake/AMReX -DGATHER_GEOMETRY=RZ
cmake --build build-gather-safety/rz -j2
OMP_NUM_THREADS=1 ctest --test-dir build-gather-safety/rz --output-on-failure
mpiexec -n2 build-gather-safety/rz/test_gather_safety test.box=64
```

Limits: this is explicit fail-fast safety, not a recoverable transaction/retry.
Particle trials and earlier tile scratch may already have changed when the host
aborts; no accepted step or conservation claim follows. Exact spatial support
validation currently covers Direct/MC gathers. Existing Esirkepov/Villasenor
energy-adjoint gathers retain their stencil implementation; only their finite
input/output checks are added. Current/charge deposition reach and boundary loss
closure remain separate contracts. Actual allocated FAB bounds establish memory
safety, not physical validity of unfilled guards: the caller must still fill its
gather images. No GPU execution or performance qualification is claimed.
