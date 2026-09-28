# Native Esirkepov completed-chord regression

Add this directory to the implicit CPU test suite. The pusher test includes the actual `ImplicitPushPX.cpp` to exercise its internal completed orbit, preserving the existing final-gather-record and orbit-iteration-count arguments. It links to the production `lib_rz`; no particle/model mock replaces the pusher.

The pusher test verifies that a chord at dt=1e-17 survives when adding it to the absolute position rounds to zero, that the real specular reflected chord is -0.0198 while midpoint radial momentum is zero, that reflection preserves endpoint kinetic energy, and that A/B/A repeats exactly. It supports serial and MPI execution. The shape test compares the production factorized difference with an independent quad-precision spline oracle across orders1–4 and both same-cell and changed-support cases.

The pusher regression runs on CPU because it uses a small host storage view to expose the native internal function. The full pusher's CUDA and Cartesian3D compile checks are separate gates. GNU C++ and quadmath are required for the shape oracle.

For a full CPU WarpX build, include this directory with `add_subdirectory(esirkepov_chord_fixture)` from the implicit tests. The targets inherit the production library configuration. A standalone CTest registration can reuse already compiled executables:

```sh
cmake -S path/to/esirkepov_chord_fixture -B build/chord-tests \
  -DESIRKEPOV_CHORD_BINARY=/absolute/path/test_esirkepov_chord_rz \
  -DESIRKEPOV_SHAPE_BINARY=/absolute/path/test_esirkepov_shape_difference
ctest --test-dir build/chord-tests --output-on-failure
```

Set `MPIEXEC_EXECUTABLE` and `MPIEXEC_NUMPROC_FLAG` to register the two-rank pusher test in standalone mode. Each test fixes `OMP_NUM_THREADS=1`.
