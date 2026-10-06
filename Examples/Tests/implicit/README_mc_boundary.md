# RZ nonperiodic ion continuity

`test_rz_mc_boundary` advances actual particles through reflecting and absorbing
walls with MC gathering, the conservative RZ filter, and full electron inertia.
The fixture populates the axis, radial wall, both endcaps and an MPI Z seam.
The absorption fractions 0, 0.05 and 1 are strict native CTest cases. All local
and integrated charge checks use a tolerance of 1e-12, without fitting a current
correction to the measured residual.

## Boundary-current contract

The explicit hybrid advance retains each charged species' Esirkepov current
before reflection, absorption and redistribution change the particle trajectory.
The native boundary decision also records which particles were actually absorbed
and deposits their unreflected endpoint charge clouds. Surviving particle clouds
are mirrored into the plasma independently of the electromagnetic field parity.
Each removed endpoint cloud is transported normally to its absorbing endcap:
its cumulative nodal charge determines the added normal face current. This
preserves the final trajectory of a lost particle and makes the integrated
surface current equal the native particle-loss tally. It adds no heating term.

The filter acts on assembled, physically folded moments. Density uses A=I-DH
and normal current uses B=I-HD, so D B = A D. H vanishes at a physical surface;
the face current is preserved and filtered tangentially. In particular,
Jz_face=(Jz[-1]+Jz[0])/2 at the lower cap, and an exterior ghost is reconstructed
as 2*Jz_face minus its interior mirror. The radial rule preserves r*Jr flux.
After every pass, physical ghosts at internal box seams are rebuilt from the
exchanged interior and exchanged nodal endcap flux. Both loss components are
summed across overlaps. Repeated-pass and radial-plus-axial box-split tests
exercise these intersections. Charge quadrature uses the native deposition axis
volume and half volumes on
physical boundary nodes. These weights differ from the clipped geometric
volumes used by the field-energy diagnostic; the test keeps them separate.

The supported path is single-level staggered m=0 RZ, r_lo=0, explicit hybrid PIC,
Esirkepov deposition, PEC/reflecting radial wall, and reflecting, absorbing or
fractional_absorbing axial particles with PEC or PMC/Neumann fields. It is
selected by `warpx.rz_continuity_filter=1` with nonperiodic Z, even if
`warpx.use_filter=0`. Unsupported topology is rejected before particle loading.
The particle gathering algorithm is unchanged. Adjoint gathering still requires
its separately supported periodic-Z topology.

## Carried moments and restart

Absorbed-particle current cannot be reconstructed from surviving particles.
The completed-step total rho/Ji and any per-species moments are checkpointed
and retained across segmented `Evolve()` calls. A nonzero-step checkpoint
without these moments is rejected with an explanation. Fresh starts continue
to deposit the initial moments from the loaded particles.

## What the probe measures

The native checks cover local continuity, represented grid charge versus
particle charge, integrated continuity including the physical surface currents,
and each endcap's current versus its actual particle-loss tally. When the
thermal model needs per-species moments, each species gets its own local
continuity check. Particle charge plus absorbed charge and specular/absorbing
boundary kinetic-energy bookkeeping are also checked. Diagnostics read carried
solver state without recomputing or overwriting the plasma current.

The reported particle-field work discrepancy includes MC spatial discretization
and finite-step particle quadrature. It is not booked as heat. These tests do
not establish full hot-plasma, fusion, Robin-conduction or driven energy closure,
or qualify a production-size run. The nonperiodic repair does not change the
periodic-Z boundary/filter path.

For additional checks, set `mc_boundary.segment_steps`, enable checkpoint
output and continue with `amr.restart`, vary the timestep/grid/shape order, or
enable the electron energy equation. The artifact qualification report records
native inputs, exact source and executable hashes, CPU/CUDA decomposition,
refinement results and any remaining limits.
