Core Cartesian hybrid electron-energy repairs
=============================================

Cartesian reloads require consistent field guards, thermal stencils, particle
moments and accepted histories. This integration repairs those paths and retains
the existing RKL temperature-admissibility rollback and retry behavior.
Exactly stationary RKL stages preserve temperature and passive accounts without
roundoff from cancelling multiples of an unchanged baseline. The ordinary
recurrence and its accumulation order are retained for all other stages.

The changes include:

* Cancel the density factor analytically in Braginskii conductivity to avoid
  low-density overflow; report positive Hall overflow explicitly. Use the total
  magnetic field for explicit conduction and guard closed-node stencils.
* Extend compatible motional and hyper-resistive operators to Cartesian ECT,
  including cell vacuum seams and staircase walls. Bound extension initialization
  by allocated fields and cover hybrid guards with EB geometry and masks.
* Preserve accepted Cartesian moments, ion-temperature cadence, elliptic
  corrections, stopping velocity and particle-ID state across native restart.
  Optional native RNG checkpointing supports reproducibility checks.
* Synchronize restored total electric fields and accepted magnetic guards.
  Keep charge/current shapes consistent; preserve pedestal entropy and covered
  temperatures; stabilize temperature variance for dominant particle weights.
* Associate split-child IDs and thermal draws with particle identity, and add
  bounded failure observations and diagnostic-only source accounting.

The regression fixtures exercise thermal operators, EB geometry, restart state,
particle identity, weighted temperatures and rejected-stage rollback. Numerical
tolerances are unchanged by the integration scrub.

Scope
-----

This integration contains the core campaign changes through 883301005 plus the
EB standoff fixture correction and formatting/documentation cleanup. It is based
on the published RKL admissibility fix 2292edcad.

It excludes the later experimental paired electron-ion source, singleton-cell
refinement, finite-bath exchange map, retained-increment RKL variant, canonical
collision and boundary ordering, and optional fusion pair trace. It does not
establish the cause of the production 3D thermal runaway or qualify those runs
as stable.

Integration qualification limit
-------------------------------

A bounded Cartesian probe with an abrupt pinned-temperature wall exposes an
interaction between the fourth-order spatial operator and the existing RKL
temperature-admissibility envelope. With an 8-by-8-by-8 grid, 5 eV interior,
2 eV wall, conduction CFL 0.04 and a requested 1e-7 s advance, the fourth-order
case repeatedly rejects stages and exhausts its work budget. The corresponding
second-order case completes in two stages without retry. This is consistent
with a non-monotone wide spatial stencil conflicting with the maximum-principle
envelope; it requires separate operator/admissibility qualification.

The temperature guard is preserved. The successful core integration checks do
not qualify the fourth-order pinned-wall case or the full production 3D thermal
path. No acceptance tolerance is relaxed to make these probes pass.
