# Accepted-background stopping prerequisite

This CPU fixture applies the original zero-background native event to prepare a
real accepted A/B/D/Je/W background, recovers its native endpoint field, and then
uses the new explicit background-increment constructor. It preserves actual
accepted electron current and displacement; it never sets Je=-I at the event.
The generic accepted-event caller updates only the nodal `Je_n` view of the new
accepted current. It preserves `Je_nm1` and the completed interval's theta record.

Preparation is pure. The fixture checks exact raw field/particle/RNG bytes,
prepared A/B/A, duplicate rejection, and a controlled rejection after candidate
publication plus endpoint recovery. It restores the mirrors, rolls back the
candidate, retries, and compares exact CPU outcome bytes. This is an event-window
transaction; whole-step redistribution rollback is a separate prerequisite.

The default cases cover periodic/PMC caps and two subsequent native steps in
coupled/decoupled mode. MPI cases cover a real multibox seam and an empty rank.
Negative cases reject incompatible Ampere state, B versus curl(A), wall support,
aliasing and stale accepted fields before any commit. A wrong-sign work receipt
is also rejected in each positive case. The original zero-background API remains
separately checked against the immutable base library by the handoff controls.

`analyze.py` uses 70-digit Decimal arithmetic on represented field and particle
inputs to reconstruct stable relativistic delta K, electron mass storage, the
magnetic background cross and self terms, relative drag heat, and all nine work
terms. Its allowance is a count-based gamma bound on positive work operands.
Finite MC/Esirkepov spatial transfer remains reported; total PIC energy is not
required to vanish. No rate, native kick, gather or heat clamp changes.

The map is an instantaneous fixed-position source split with no additional
magnetic rotation. It has no temporal-order or maintained-alpha qualification.
It excludes vacuum, pedestal, external drive, circuit, topology changes, real
particle boundary events and live Evolve collision dispatch. All production
event guards stay closed. CUDA production/fixture compilation is separate from
runtime certification: GPU atomic recomputation is not assumed byte-identical.

Standalone registration accepts `-DNATIVE_BACKGROUND_STOPPING_EXECUTABLE=...`.
An integrated parent should add this subdirectory only under `if(TARGET lib_rz)`;
3D-only builds compile the guarded production helper but do not register RZ tests.
