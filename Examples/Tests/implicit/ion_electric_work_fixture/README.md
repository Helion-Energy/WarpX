# Native implicit ion electric-work diagnostic

The helper measures a supplied electric component with the native shape-3,
nodal MomentumConserving/Direct gather and the relativistic Boris work velocity
`(u_old+u_end)/(gamma_old+gamma_end)`. Work and Cartesian impulse retain signs;
absolute work and measured kinetic change are separate. Every charged species,
including relaxation-excluded alpha, has a row. No eta law or species fraction
is guessed.

`RecordedFinalGather` requires the coordinates of the actual last Boris
gather, after any crop. `MeasureNativeImplicitIonElectricWork` reads the final
stored midpoint and always labels its result `FinalMidpointReconstruction`:
native position iteration updates that position after the final gather. The
reconstruction test demonstrates the resulting nonzero work defect.

Caller fields must be immutable copies of the actual projected electric
component after exactly the same auxiliary centering as the push. The optional
total electric field must also include any particle-external contribution
before claiming a Boris total-work identity. The public native API does not
expose all per-species no-push/no-gather/radiation/cropping flags, so the native
bridge requires an explicit caller attestation and guards the exposed geometry,
pusher, gather, shape, variable-charge and subcycled-container settings.

This is an acceptance-only diagnostic. It performs MPI scalar reductions and
must not be inserted into every residual/Jv. It never changes particles, fields,
RNG, or accepted tallies. Measure before endpoint extrapolation/absorption;
ions lost at the endpoint still experienced trajectory force work.

Configure with an existing double-precision AMReX build:

    cmake -S Examples/Tests/implicit/ion_electric_work_fixture \
      -B build-ion-electric-work -DAMReX_DIR=/path/to/amrex/lib/cmake/AMReX
    cmake --build build-ion-electric-work -j 2
    ctest --test-dir build-ion-electric-work --output-on-failure

For Cartesian 3D, use a 3D AMReX build and `-DWORK_GEOMETRY=3D`.
Run any fixture on two ranks with `mpirun -n 2`; box64 includes an empty rank.

The tests call the actual native gather and Boris pusher. They cover heating,
cooling, signed cancellation, rotated RZ states, all charged species, missing
total field, invalid input/guards, RNG and input purity, replay, temperature
coefficient differentiation, a shifted reconstruction negative, and the
single-species NR matched-rate limit. Synthetic test fields are not proof that
the application captured its eta force correctly; runtime capture/wiring is
still a separate gate.
