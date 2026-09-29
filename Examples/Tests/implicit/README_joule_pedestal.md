# Joule receiving capacity with a density pedestal

The additive state carries `Ue = 1.5 (ne + nped) kB Te` for gamma=5/3.
Joule physics uses the deposited plasma: `eta J^2`, species charge fractions,
relative drift, density gates, source tapers and ion redirection. Only energy
received by electrons must divide by the additive state's heat capacity.
The original PEDCOV contract explicitly deferred this conversion while Joule
heating was disabled; this change completes that narrow follow-up.

`test_joule_pedestal.cpp` calls the actual production Joule primitive with two
ion species (charge numbers1 and2, charge fractions0.6 and0.4). It independently
integrates the temperature change using physical radial dual volumes and the
receiving density. It also computes staged ion energy and declined heat, and
compares the independent temperature integral against the native class ledger.
The bound is declared in C++: `128 eps initial_U + 2e-11 source_J`.

Coverage includes no pedestal, zero/nonzero pedestal, variable density/capacity,
below/at the physical density gate, density and source tapers, full redirection,
redirected taper and redirect density rejection. The test performs no particle
advance. Redirected energy is staged; realized stochastic ion heating is not
claimed. The enabled relaxation parser satisfies the native redirection guard;
this fixture never calls relaxation. No clamp is relaxed.

Build and run inside the source worktree:

```sh
cmake --build build --parallel 8 --target test_joule_pedestal_rz
ctest --test-dir build -R '^test_rz_joule_pedestal_capacity$' --output-on-failure
python3 Examples/Tests/implicit/analysis_joule_pedestal.py \
  --executable build/bin/test_joule_pedestal_rz --output build/joule-mpi \
  --ranks 2 --mpiexec /path/to/compiler-matched/mpiexec
```

The CTest driver creates a new attempt directory on every invocation and retains
native logs, consumed input copies, exact commands and JSON results. The full
before/after qualification uses exactly the same test object linked against
pristine59e904ce and the separate capacity candidate. Five pedestal-aware
electron-receiving baseline cases fail; all candidate cases pass. Seven
unaffected serial controls have byte-identical temperature files. Separate
multi-box two-rank checks cover variable capacity, taper and redirection.

Scope is the source primitive with a matched coefficient and receiver density,
including the explicit production call. The theta-implicit final source stage
passes a midpoint coefficient density after restoring an endpoint temperature.
That pre-existing receiving-density mismatch is not changed or qualified by
this patch; no theta integration energy-closure claim follows from these tests.
This change does not alter Qei, hyper/viscous heat, marker transport, pedestal
tracking, floors, the field/heating Spitzer normalization, or any campaign input.
