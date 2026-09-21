# Native Darwin circuit trials

The supported native driver uses a compiled `ExternalCircuit` ABI2 plugin with
single-level, double-precision, three-dimensional segregated Darwin and disk
magnetic-flux probes. It measures the actual accepted field at the beginning of
each step. Every residual and Jacobian probe starts from that accepted circuit
state. The converged theta-stage EMF drives a full-step circuit candidate; exact
acceptance uses the same forcing and occurs once per step.

The default `circuit.trial_backend = host` is the reference implementation.
Plugins can additionally export the optional C interface in
`ExternalCircuitAffine.h` to support:

```
circuit.engine = external
circuit.trial_backend = device_affine
circuit.device_iterations = 20
```

This extension preserves the ABI2 virtual interface and factory. After
`BeginStep`, preparation supplies immutable host arrays for the interval's
current response `p0 + G*emf`, reference and entry currents, and all required
sign/cancellation guards. The driver validates the packet and copies it once to
the execution device before releasing the borrowed view. A plugin must refuse
intervals for which this guarded affine response is unavailable. There is no
host fallback within device residual trials.

During a residual evaluation, magnetic probes, imposed-field subtraction, EMF
filtering, response evaluation, scale convergence, and scale consumption stay on
the device. Both the external E/B assembly and Darwin boundary A read device
scale segments. Multiple GPU ranks require GPU-aware MPI; the coil reduction
uses device buffers and ordered point-to-point exchanges. CPU builds execute the
same response algorithm as a validation oracle.

`device_iterations` is a fixed launch budget, avoiding a host convergence poll.
The first candidate meeting the existing stage tolerance stays fixed for the
remaining passes. A fresh residual resets this latch. Exhausting the budget,
nonfinite arithmetic, and invalid response guards terminate the solve, including
release GPU builds. Acceptance remeasures the final fields, rechecks convergence,
reads the candidate once, and verifies the exact ABI2 accepting result against
the same stage tolerance. Field solver reductions may still synchronize with the
host; the circuit extension does not claim to remove those reductions.

The fixed budget can perform unnecessary field reconstructions. Transfer removal
alone does not establish a speedup. This backend remains an optional optimization
pending application-specific guard coverage, time integration, work/energy,
field agreement, and scaling qualification. It does not alter electric closure,
embedded-boundary physics, or existing solver restart restrictions.

Tests cover coupled response arithmetic, frozen filter memory, port ordering and
normalization, release-build refusals, residual replay, exact single acceptance,
host/device field comparisons, and direct E/B consumers with poisoned host scales.
