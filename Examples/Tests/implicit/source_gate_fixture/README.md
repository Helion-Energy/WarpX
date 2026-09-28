# Native eta corner regression

`test_thermal_source_gate_rz` loads only the initial internal Btheta through the native beforeInitEsolve callback, clears that callback, and runs the normal one-step Darwin/Eulerian advance. It links the production library directly. No debug instrumentation or alternate particle/source update is used. The accepted production receipt uses the recorded final Boris gather coordinates.

The proton fixture has a fixed physical64x4x64 quiet lattice, periodic z, RZ axis, PEC rmax, common eta=.001 and nu=q*rho*eta/mp. Both layouts have identical cells, particles and equations. The radial continuum pressure gradient balances the continuum JxB at initialization; finite-grid startup is not claimed to be exact equilibrium. The test checks the same component work across single-box and box8, plus equal loading and total push/kinetic work. Before the sync repair the component mismatch is4.58e-5; afterward it is at rounding scale. The old single-box result is not an exact oracle either: its periodic/axis corners also needed repair.

CMake registers coupled/decoupled serial and MPI2 controls when RZ/MPI is built. amr.refine_grid_layout=0 preserves the single box (an empty rank for MPI2). The Python runner supports MPIEXEC/MPIEXEC_PREFLAGS and preserves every run in a new directory. Run directly:

    OMP_NUM_THREADS=1 python3 analysis_source_gate.py /path/test_thermal_source_gate_rz inputs_rz --mode coupled_jfnk --ranks 2

This is a double-precision communication regression. CUDA compilation is supported but GPU execution requires the caller's allocation. It is not a claim of full-method time order, total-field conservation, or D/T mechanical closure. The separate diagnostic source ladder records those limitations and configured-rate contributions.


The optional `source_gate.energy_audit=1` installs read-only `beforestep` and
`afterstep` callbacks for a full native energy inventory. It records the
accepted fine-patch B and longitudinal E, stable native relativistic ion kinetic
energy, and physical cell electron U. It separately compares native geometric
dual field quadrature against `FieldEnergy::ComputeNorm2` and records exact
clipped-volume quadrature. Native radial nodal outer half-ring weighting differs
from the exact clipped annulus by pi*dr^2/4; the active Btheta and ELr in this
case are radial cell centered and their dominant inventories agree.

`analysis_native_energy.py executable inputs_rz --with-mpi` runs the fixed
3x3 spatial/dt ladder, decoupled/layout/MPI controls and an audit-off twin. The
registered `test_rz_thermal_source_energy_inventory` checks independent particle,
thermal and field inventory identities, closed boundary accounts and byte-exact
accepted physics receipts with the callback on/off. It reports actual total
energy change, measured realization minus conditional NR expectation, the
remaining deterministic expected defect, and the rigorously bounded NR
convention interval. A passing inventory test is **not** a zero-total-energy or
whole-method second-order claim. The initial immutable R25 qualification leaves
a resolved O(dt^2) per-step expected defect; startup, accumulated stochastic
trajectories and maintained electron inertia are outside this bounded case.
No error tolerance, force, source, rate, floor or accepted RNG operation changes.

Receipt checking defaults to `--receipt-check=exact`, preserving the original
CPU text comparison. CUDA CTest explicitly selects `--receipt-check=roundoff`.
That mode adds one unchanged audit-on repeat and applies the same independently
derived numeric bounds to the audit-off twin and the repeat. It does not claim
GPU byte identity. All original native inventory identities and mode/layout
comparisons remain unchanged.

The numeric checker uses `gamma=k*eps/(1-k*eps)`, with
`k=32*(grid inventory visits+2*particle count)+256`, as in the inventory
analysis. Callback twins have identical grids and populations; their positive
receipt scales are evaluated separately. Work scales include positive kinetic
inventories, absolute OU/source constituents, and the actual absolute/Cauchy
eta and remainder work. Stored U and K use their old/new inventory scales;
particle weight uses its positive weight scale. Each comparison allows the
sum of the two independently derived allowances, never a value fitted to the
observed difference. These scales apply to this closed, cold, single-proton
fixture, not arbitrary signed multi-species source receipts.

Nonnumeric text, species/channel labels, numeric field sets, actual flags and
configured-zero channels stay exact. Nonfinite values are rejected. The
dimensionless endpoint work relative mismatch retains its native `1e-11`
quality limit in each run and uses a dimensionless gamma-based comparison;
it is not an exact-zero flag. The JSON report includes every measured signed
receipt difference, its bound, per-run positive scales, and the unchanged
repeat result. To analyze preserved runs in roundoff mode, the manifest must
also contain `full-energy-repeat`.


### Native electron-inertia inventory

`analysis_native_inertia.py <source_gate_executable> <inputs_rz> --with-mpi`
adds a separate two-step physical-electron-mass audit. Both energy modes use
production measured electron-current history, the two-point dJe-only E form,
and the same active Joule/relaxation population source as this closed proton
fixture. The zero-inertia pair changes only the inertia switch. No source,
particle, history, or endpoint staging rule is changed.

`source_gate.inertia_audit=1` registers a fixture-owned reduced diagnostic only
after initialization. Its regular intervals are empty (so no diagnostic
velocity synchronization is requested). The existing accepted midstep hook
copies the exact stage nodal inertia field/current, Yee plasma/ion current,
coefficient, and interval longitudinal-field origin. Private copies and
projections are reduced after endpoint acceptance. No residual observer,
accepted-state write, random draw, or lazy pedestal refill is used.

For A = 1(raw rho>0) taper m_e/[e floor(raw rho+pedestal)], Jv =
[Jtheta-(1-theta)J0]/theta, and the actually stored endpoint Js, the test checks

    dt Jtheta.Ei - Delta(0.5 A |Js|^2)
      = (theta-0.5) Atheta |Jv-J0|^2
        + 0.5(Atheta-A1)|Js|^2 + 0.5(A0-Atheta)|J0|^2
        + 0.5 Atheta (Jv-Js).(Jv+Js).

The density-coefficient and stored-history terms are distinct. Native
geometric trapezoid and physical clipped dual integrals are both reported;
RZ interpolation work and homogeneous PEC/PMC/axis projection work are separate
accounts. A private source exchange precedes corner projection. The original
accepted field/particle/U inventory arithmetic gates and bounds remain unchanged.
The new multi-step analyzer explicitly includes the signed accepted cell-U
solver residual in its reconstructed budget; it verifies this against the
accepted L1 residual and the independently integrated delivered U change.
The original one-step analyzer defaults to its original gate. The signed
residual remains in every reported actual/conditional total defect.
`EL_entry_jump` is already contained in the endpoint field difference and must
not be added to total energy again. The complete actual and conditional total
energy defects are reported both with stored electron inertia and with the
named projected inertia work; neither is required to vanish. The homogeneous
projection does not identify all vacuum-recovery/constraint contributions in
other decks, and this fixture does not establish whole-method temporal order.

CPU callback on/off inventory and source receipts must be text-exact. CUDA
CTest selects `--receipt-check=roundoff`, additionally runs unchanged audit-on
repeats, and uses the existing independent positive receipt/inventory bounds;
it makes no GPU byte-identity claim. The first deterministic stage alone is
compared across mode/box/MPI layouts, before layout-dependent OU sampling can
feed the next step. The JSON records every identity error and allowance, work
account, callback difference, and unresolved full-budget defect.
