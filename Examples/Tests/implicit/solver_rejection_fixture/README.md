# Invalid residual propagation

This fixture instantiates the production `AMReXGMRES` wrapper and AMReX's native
GMRES with distributed GPU-capable MultiFabs. It injects an invalid RHS, a rejected
third matrix action, a rejected preconditioner, and arithmetic overflow from finite
inputs. The rejected solve returns status 2 with `failedEvaluation()==true`, clears
its partial correction, and never evaluates an operator with an invalid input.
Only rank zero injects NaN, testing collective failure with seams and empty ranks.

Each case solves the same finite diagonal problem before and after rejection.
The two valid solutions, residuals, iteration counts and operator/preconditioner
call counts must equal unmodified native AMReX GMRES exactly. This also checks
reuse after an interrupted Arnoldi cycle. No tolerances are adjusted on failure.

Configure inside the WarpX source tree:

```sh
cmake -S Examples/Tests/implicit/solver_rejection_fixture -B build-rejection \
  -DAMReX_DIR=/path/to/amrex/lib/cmake/AMReX -DCMAKE_BUILD_TYPE=Release
cmake --build build-rejection -j 2
OMP_NUM_THREADS=1 ctest --test-dir build-rejection --output-on-failure
OMP_NUM_THREADS=1 mpiexec -n 2 build-rejection/test_solver_rejection \
  test.failure=action test.box=64
```

Repeat with two- and three-dimensional AMReX builds. The same source can be
compiled with the CUDA flags of a configured WarpX build. Tests use only local
kernels and collective norms; they do not copy full fields to the host.

The production Newton consumer checks `failedEvaluation()` and nonfinite residual
or correction norms before any update. Invalid evaluations return -6 even when
`newton.require_convergence=0` or a fixed iteration count was requested. Finite
iteration-limit behavior is preserved. The JFNK trial and nonlinear input are
checked before calling the physical residual. Exceptions are confined inside the
AMReX C++ wrapper and cannot cross a PETSc callback.

This is status propagation, not timestep retry or a change to particle/current
convergence. Application callers still own rollback of particle, field and circuit
scratch after a failure. The matched r14 r20 regression rejects a finite trial
whose inner radial-current consistency loop stalls above its unchanged tolerance;
the r21 physical controls must still finish normally.
