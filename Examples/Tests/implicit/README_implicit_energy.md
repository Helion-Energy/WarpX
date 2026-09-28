# Implicit electron energy development status

`hybrid_pic_model.electron_energy_mode` selects `coupled_jfnk` or
`decoupled_jfnk`. Both evolve cell internal energy and use the finite-difference
conduction operator inside the implicit residual. The legacy explicit
conduction integrator is not called by either JFNK mode. Conductivity, FD order,
limiter, flux limit and thermal boundary parameters still select the spatial
operator. The default mode remains `legacy`.

The coupled mode solves field and energy blocks together. The decoupled mode
alternates field and scalar thermal solves and verifies both residuals at the
same updated state. See `Docs/source/usage/parameters.rst` for parameter details,
block scales, acceptance gates and boundary restrictions.

## Material source qualification

The experimental `native_stopping_event.material_support=1` path with
`native_stopping_event.scheme=symmetric_midpoint` composes a source half-step,
field/transport step and second source half-step. It accounts for particle drag,
electron heating, field work and represented endpoint current before publishing
the accepted state. Unsupported configurations reject rather than silently use
a different source update.

This composition is currently guarded to precise CPU RZ, one level, periodic
axial boundaries, radial PEC, shape-three momentum-conserving gather and
Esirkepov deposition. Its complete combination with CUDA, mass matrices,
nonperiodic ends, a native circuit or restart is not yet qualified. Separate
operator or source-free tests of those features do not qualify their combination
with the material source. Embedded boundaries and mesh refinement are outside
this development scope. Cartesian implementations remain in place, with their
existing capability guards.

`implicit_evolve.darwin_current_absolute_tolerance` adds a fixed SI current
floor to local arithmetic error bounds. The material-source default derives
from its fixed nonlinear absolute tolerance and physical current scale; zero
restores arithmetic-only checking. It is frozen before iteration. Energy,
charge and nonlinear residual checks remain independent.

## Focused regressions

The implicit CTest suite includes conduction operator and thermal stage tests,
Esirkepov current/density response, gather order, Hall preconditioner
coefficients, matched stopping/viscous work, source corner decomposition,
source projection defect and current tolerance tests. Build their executable
targets against the same library before running CTest. Source acceptance must
also be exercised through accepted steps in the ordinary application with
nonzero transport and source terms. A near-zero source test establishes
arithmetic robustness, not large-source or production-scale qualification.
