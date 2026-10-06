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
the original capacity cases do not call relaxation. The raw-species scaling
cases also call the production electron relaxation primitive against its
analytic cold-ion exponential. No clamp is relaxed.

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
The pedestal-capacity correction does not alter Qei, hyper/viscous heat, marker
transport, pedestal tracking, floors, or the field/heating Spitzer normalization.
The separate raw-species normalization repair below also corrects the electron
relaxation rate; its tests do not certify the complete stochastic Qei exchange.

## Raw RZ species fractions

The per-species deposits have not received the total-density RZ volume scaling.
A physical density floor cannot be applied to their sum. Normalization now uses
the positive raw sum, so the species fractions sum to one at any radial volume;
empty raw deposits have no species source. Physical density gates still use the
scaled total density. This corrects Joule heating and the electron relaxation
rate, without changing the receiving heat capacity or the configured gates.

Native tests rescale both species deposits by 1e-6, 1, and 1e6 while holding the
physical density, current, fractions, and temperature fixed. Joule energy,
redirected energy, and the analytic electron relaxation rate must be invariant.
The latter verifies the electron primitive, not a particle-pair energy theorem.
The moving-edge resistive tests additionally close the full particle, magnetic,
bulk-electron, and thermal budget at two timesteps.

## Electron-ion exchange with the represented capacity

The physical relaxation source is `Qei = sum_s 3 ns kB nu (Te - Ti_s)`. Its
electron temperature rate divides by the represented capacity, including the
pedestal and the source-stage density. The parser and source gates retain the
physical deposited density. Using ion density alone in that denominator loses
energy even as the timestep tends to zero when a pedestal is present.

Eight native cases now call both the electron primitive and actual two-species
particle OU heating. They cover pedestal fractions0,0.2,5 and receiving-density
ratios0.2,1,10, including the production source-stage wrapper. The gate is1% of
the actual exchanged energy, and0.1% for the independently calculated OU
ensemble mean. The short step has nu*dt=1e-4; sequential source integration has
finite-time splitting error, and realized ion heating has sampling error. These
are reported separately. The earlier electron-primitive-only scope statements
apply to the raw-fraction scale tests, not these new paired exchange cases.
