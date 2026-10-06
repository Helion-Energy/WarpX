# Nonperiodic momentum-conserving boundary probe

`test_rz_mc_boundary` advances real particles with explicit RZ hybrid PIC,
shape-3 Esirkepov deposition, MC gathering, the conservative filter, and all
three electron-inertia contributions. The cold counterstreaming fixture has
Neumann axial field boundaries, fractional particle absorption, a reflecting
radial PEC wall, and two full-radius Z slabs. It is deliberately smaller and
simpler than a production compression experiment.

The inventory CTests cover absorption probabilities 0, 0.05 and 1. They require
actual axial wall incidents, reflection where applicable, escape at both ends
where applicable, finite fields, conserved particle charge plus absorbed
charge, and agreement between the kinetic-energy decrement during boundary
handling and the native absorbed-energy tally. The relative inventory and
boundary-energy tolerance is 2e-12.

**A passing inventory test is not a continuity or energy-closure certificate.**
The probe also records the native deposition continuity residual, localized
norms, electron bulk/magnetic/thermal inventories, and the discrepancy between
particle kinetic gain plus escape and the postprocessed current's grid work.
The latter includes MC spatial mismatch, explicit time quadrature and boundary
trajectory reconstruction; it is not a pure gather error or a heat source.
The inventories occupy native staggered time levels and are not asserted to
form a closed total-energy budget.

Run a separate continuity gate with an explicit required tolerance:

```bash
OMP_NUM_THREADS=1 build-cpu/bin/test_rz_mc_boundary \
    Examples/Tests/implicit/inputs_test_rz_mc_boundary > boundary.log 2>&1
python3 Examples/Tests/implicit/analysis_rz_mc_boundary.py \
    boundary.log --continuity-tolerance 1e-12
```

The ef19-based nonperiodic implementation fails this continuity gate at actual
wall events. Do not turn that known failure into an expected-pass conservation
test or use inventory success to authorize a production energy-closure claim.
`warpx.use_filter=0`, probabilities 0 and 1, and a two-rank launch provide
controls separating filtering, absorption and MPI effects. Probability 0.05
uses stochastic wall decisions; rank-count changes need not reproduce the
same individual absorption history.

The recorded `continuity_seam_relative` strip includes its intersections with
the radial walls. Use `continuity_interior_relative` and deterministic
rank-count comparisons to distinguish physical-wall errors from MPI seams.

`analysis_rz_gather_configuration.py` is independent: its constructor-only
executable never calls `InitData`. It tests accepted periodic/MC settings and
rejection of incompatible adjoint settings before particle loading.
