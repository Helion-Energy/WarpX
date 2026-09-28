# Native harmonic stopping event

This fixture loads actual uniform fast/background proton populations into WarpX,
uses the actual native finite stopping operator as a zero-electric-impulse oracle,
and prepares/commits a paired electron/ion event through the new helper. It never
calls Evolve or opens production source/event guards.

The scope is uniform fixed-density, zero native B, fully periodic Cartesian3D,
physical electron mass and frozen input temperature. All charged ions receive
symmetric electric half impulses; only fast ions receive the existing native
stopping kick. The six harmonic equations couple the electric impulse and mean
electron current. Real native instantaneous Esirkepov deposition checks every
accepted current row. This is not a spatial/RZ/fusion/circuit event solver.

Heat is minus the native drag's relative work, `-(deltaK_drag-S dot Ve_mean)`.
The original native lab-loss heat remains unchanged in legacy dispatch. Actual
relativistic particle energy, compatible-Yee electron bulk energy, cell U and
Cartesian momentum are reconstructed independently from represented native data.
The finite electric/current kinematic convention defect is reported and bounded
by `sum w |q| |G Psi| Ucap^3/c^2`; it is never deposited as heat. Accounting closure
is not a claim of exact physical energy conservation or temporal order.

The positive cases cover one box, actual seams, MPI2 and an empty rank, plus a
zero event. Negative cases reject an event exceeding its declared convention budget, an inadequate
speed cap and nonuniform input before any accepted change. An isolated compiled
reaction-sign fault converges the field constraint but fails Cartesian momentum;
there is no production fault knob. Preparation, full raw particle/selected field
purity, required cache invalidation, stale-data rejection, once-only commit and
exact rollback/retry are checked. The cache hook deliberately invalidates the
borrowed stopping context after commit; rollback uses its independently saved
binding. Native E/B, density, temperature, all particle attributes/IDs and RNG are
unchanged except the explicitly committed momenta/current/U.

`analyze.py` uses Decimal arithmetic on exact represented double inputs. Native
IDCPU words identify particles; periodic owner masks identify each physical field
DOF once. It independently reconstructs stable ion kinetic change, electron mass
metric/storage, U, momentum and all reported work terms. Bounds use positive
input scales and global contributor counts, never observed discrepancies.

To add to the parent test tree, use `if(TARGET lib_3d)` before adding this directory.
A standalone importer accepts `NATIVE_UNIFORM_STOPPING_EXECUTABLE`, optional
`NATIVE_STOPPING_WRONG_REACTION_EXECUTABLE`, and `MPIEXEC_EXECUTABLE`. Each CTest
creates a fresh directory. CPU native proof is qualified; CUDA3D object compilation
alone is not a physical GPU event qualification. The fixture's exact native-map
and retry assertions remain unchanged until actual device evidence exists.

The helper uses private device scratch and only scalar reductions during its
accepted-event solve. It has no residual/Jv caller, no RNG draw and no host field
array bounce. Fixture-only observers copy native data for independent analysis.
The constructor borrows context/temperature/U through the transaction. Commit's
required nonthrowing invalidator must clear all caller particle/source/MM caches.
The driver must publish thermal mirrors and event/checkpoint metadata afterward;
this prerequisite does not implement that driver. Field/particle topology and
external state must stay fixed until Finalize or Rollback. Other event schemes,
outer collisions and production endpoint capabilities stay guarded.
