# Origin Holmstrom native attribution

`test_holmstrom_axis.cpp` evaluates the actual RZ Ohm and FD4 conduction APIs on a common manufactured state. It preserves the physical axis and PEC boundary closure, builds divergence-compatible B through the native curl of a regular analytic vector potential, and evaluates the embedded elliptic electron-inertia route. This is not a replacement for an evolved FRC or hot-PIC test.

The independent Python analysis checks the exact staggered density/radial tanh multiplier on Hall, pressure and ion-motional terms. It requires bitwise unchanged initial fields, viscous/hyper fields, strain heating and three conduction advances between ON and OFF. The dense control must also retain bitwise identical applied E. Signed J.E, Ji.E and magnetic-energy rates are reported separately; no sign-definite diffusion assumption is made.

ON uses `holmstrom_vacuum_region=1`, transition width0.5, radius0.006m and rolloff0.003m. OFF clears all four. A zero radius with region1 is a global gate and is not an OFF setting. The analysis audits the consumed native input diff, not only the supplied file.

The native fixture uses a single rank and box for lossless map output. CPU/MPI viscosity qualification lives in the separate `test_viscous_physics` matrix. This fixture and its CTest compile for CPU/CUDA. Its prescribed coefficient values, tight cold-start inertia solve and SSPRK2 diagnostic time extrapolation are stated in the inputs; production physics is never overridden by this fixture.

Run the complete source comparison with two separately linked native libraries:

```sh
python3 Examples/Tests/implicit/analysis_holmstrom_axis.py \
  --baseline /absolute/path/to/pristine/native_holmstrom \
  --candidate /absolute/path/to/guard/native_holmstrom \
  --output build/holmstrom-matrix --source-probes
```

For regression, CTest uses the current executable in both library slots to test the ON/OFF gate. `--output-parent` creates a unique directory per invocation; no prior outputs are overwritten. An analysis-only pass can use `--output <existing> --analyze-only`.

`holmstrom_test.source_probe=visc` independently exercises the actual conservative nodal work receiver without modifying temperature. `hyper` probes the same receiver for the separately unbooked hyper field. A deep sub-floor trough can trip the no-eligible-receiver assertion in that hypothetical hyper-booking path; preserve and label this result rather than changing the source floor or treating it as a live booked-source failure. Main work maps always retain the unmodified edge product before receiver redistribution.

The Joule/pedestal capacity defect reported by `analysis_viscous_physics.py` remains a distinct failed physics gate. These attribution checks do not certify a complete moving-plasma energy model, prove the cause of FRC axis heating, or qualify the still-open full fourth-order radial/mixed/wall closure.
