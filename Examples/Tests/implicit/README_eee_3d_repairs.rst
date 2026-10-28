Core Cartesian hybrid electron-energy repairs
=============================================

Cartesian reloads require consistent field guards, thermal stencils, particle
moments and accepted histories. This integration repairs those paths and retains
the existing RKL temperature-admissibility rollback and retry behavior.

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
