# Pure trial dissipation integration

`EulerianDissipation.{H,cpp}` extracts the existing RZ velocity/stress/drag
chain and all three existing RZ hyperresistive operators into owned scratch.
No HybridPICModel, field solver, registry, boundary sum, particle or RNG is
modified. This is preparation for parent-owned runtime wiring, not that wiring.
The base is source-helper commit `31db03282356b88d04a0cf174bf782fe958949cd`.

## Exact input and option mapping

Construct one helper per complete physical level (currently require level zero,
RZ m=0, zero-based Yee domain, r=0, no EB). The BoxArray is cell-centered; the
DistributionMapping must match every borrowed input. Each field needs one
already-physical ghost layer. The helper copies component zero and synchronizes
private box seams; it never applies a deposited-charge fold to its inputs.

- `viscosity = model.m_visc_in_ohms_law`;
  `viscous = model.ViscosityPointParams(lev)` (its geometry is refreshed from
  the constructor). The existing point kernel remains the only constitutive
  implementation: Braginskii or parser `(n[m^-3],Te[eV],B[T])`, density gate,
  hard cap, taper, and parallel flux limit are unchanged. Matched drag requires
  centered gradients. Negative parser viscosity is clamped with separate
  parallel/perpendicular masks, overwritten on every call; no warning counter
  is mutated during residuals.
- If `!m_include_hyper_resistivity_term`, `hyper=Off`; else select
  `AmpereCurlCurl` when `m_hyper_resistivity_curlcurl`, `InteriorCurlCurl` when
  `m_hyper_res_curl_curl`, otherwise `Laplacian`. This precedence matches the
  actual FD addition branch. Use `m_eta_h` and
  `m_hyper_resistivity_has_B_dependence`. Its parser uses unfloored physical
  charge density, not an electron number density or a heat capacity.
- Map radial lower boundary to `Axis`, radial upper PEC/None to `PEC`/`Open`;
  axial PEC/PMC (including its Neumann alias)/Periodic/None to
  `PEC`/`PMC`/`Periodic`/`Open`. PMC uses the actual wrapper response
  `PEC::ApplyPECtoBfield`: normal electric ghosts are odd, tangential ghosts
  are even, and only a normal nodal component would be constrained to zero.
  The Yee axial normal component is cell centered, so its interior row is
  retained. Mixed PEC/PMC corners compose the two parities. Other boundary
  kinds and radial-upper PMC remain explicitly unqualified.
  This electric projection does not reconstruct input moment images: the
  caller supplies actual deposition/current guards. In particular, absorbing
  particles at a PMC cap still produce odd Jz/even Jr and Jtheta guards;
  reflecting particles at a radial PEC wall use the current PMC parity with
  the existing metric r*J image, despite the electric PEC projection.
- Set `transformed_electric_solve = m_esolve_tensor || m_esolve_curlcurl` and
  propagate the true EB/level/mode flags. These guards must not be bypassed.
  Those E solves transform a raw source through their response operator; raw
  EV/EH is not yet the applied response. Existing 3D code is unchanged and the
  new translation unit compiles there, but this helper explicitly rejects 3D.

At every relevant trial assemble `TrialDissipationState` from explicit const
views: current physical nodal charge, `ElectronTemperatureForSolve(lev)` (the
borrowed current-trial temperature), plasma Ampere current, the deliberately
selected current-stage ion current, and current trial B. They must be the
same moments/B that the Ohm force sees after its conductor-row moment closure.
Old or outer-iterate ion current is allowed only when deliberately selected by
the decoupled stage contract. Do not let `m_implicit_visc_current`,
`m_implicit_visc_temperature`, or `m_implicit_visc_magnetic` silently override
these arguments for an active new-mode trial.

`Evaluate` returns false for an invalid state/coefficient; discard the scratch
and reject that residual. Getters return const owned views. `NodalDrag` is the
weighted negative-transpose stress force in V/m; `StrainPower` is W/m3;
`Stress` packs xx,yy,zz,xy,xz,yz in Pa. `ViscousField` and `HyperField` are the
applied homogeneous-boundary increments in V/m. `RawHyperField` is an audit
view of valid pre-projection kernel values and must never be booked in place
of `HyperField`.

## Minimal force and source wiring

Add `EulerianDissipation.cpp` to the same production implicit-solver CMake and
GNUmake source lists used for `EulerianThermalSources.cpp`; the standalone
fixture already links both. No production build list was changed in this
scoped worker patch.

The smallest safe new-mode dispatch is to pass an optional borrowed const
`EulerianDissipation*` into the FD Ohm solve (or expose this explicit scratch
through a borrowed trial context). Parent owns that interface change.

1. Prepare/evaluate it from the same trial state **before** the force consumer.
   Require it when `HasElectronThermalTrial(lev)` and either enabled term needs
   it; do not silently fall back to the old frozen-coefficient implementation.
2. Only in this active trial branch, skip the old viscous nodal precompute and
   old hyperresistive preparation/addition. This includes BOTH the wrapper
   `hybrid_hyperres_curlJ_fp`/`hybrid_hyperres_E_fp` precompute and the internal
   FD `Kr/Kt/Kz` precompute. Preserve all old branches when the pointer is null.
   Keep the ordinary Ohm, Hall, pressure, resistivity, external and inertia
   terms, masks and boundary ordering intact.
3. At the existing per-component hyper/viscous accumulation sites, if
   `include_resistivity`, add **the supplied** `HyperField()[c]` and then
   `ViscousField()[c]` to E. Add neither on the no-resistivity pass. This
   preserves the actual existing `include_resistivity` gate on both terms.
   Populate `EH_out`/`EV_out` from those same values when requested. In
   particular the current outer `m_hyper_resistivity_curlcurl` addition omits
   `EH_out`; that omission is a bug to fix in the new branch, not a source to
   disable. No separately recomputed viscosity/hyper field may be used for
   the force. The helper's theta-axis and constrained PEC increments are zero.
4. Apply the existing full wrapper electric boundary stack to total E. For
   the qualified additive mode this is compatible with the helper's already
   projected increment. Standard PEC/PMC/axis images include one ghost layer.
   With a resistive shell, keep the same trial B in both force decompositions:
   its affine wall value cancels, so the incremental tangential wall value is
   zero and its ghost is minus the interior increment. Do not assign the shell
   wall work to EV/EH. Free physical boundary rows retain their signed work.
5. Pass `ViscousField`, `HyperField`, and `StrainPower` directly to
   `ThermalSourceState.viscous_electric`, `.hyper_electric`, and
   `.viscous_strain_power`. Map physical nodal Q to cell U with the previously
   delivered conservative restriction. Work and strain remain distinct;
   the source chooses the configured physical contribution once.

For the existing push correction, both actual calls use the same prepared
trial context: `E_full` includes these terms; `E_nores` omits them. Therefore
`E_iterate - (E_full - E_nores)` subtracts their supplied increments together
with the actual resistive difference. Preserve the existing activation
`HasResistivity() && m_implicit_push_excludes_resistive_field`; a configured
viscosity alone must not invent a new correction. An equivalent explicit
subtraction is allowed only after verifying its residual and boundary rows
against these actual full/no-term fields, including separate eta overlays and
`add_resistivity_push` behavior. Do not use a generic eta*J substitute.

The trial rho/T consistency before this pre-push correction remains the
parent's coupled-stage responsibility. Updating Te only after deposition or in
a later source callback is too late; A/B/A probes must reconstruct the same
pre-push inputs and force, not inherit previous-residual density.

## Qualification and remaining integration proof

The standalone fixture verifies physical-volume viscous adjoint power with
spatial density/Te, anisotropy and nonzero ion current; built-in Braginskii caps,
taper and flux limit; live Te derivatives; direct source-ledger composition;
native quadratic/axis hyper rows in each mode; spatial-coefficient curl
symmetry/positive closed-domain work; signed Laplacian work; affine shell
increment ghosts; direct comparison with the production PEC/PMC electric
boundary routines at caps and mixed corners; actual PMC deposition-current
images with absorbing particles; signed source composition in all three
hyper modes and exact sensitivity to a supplied physical current guard;
A/B/A and all input/ghost purity; negative-coefficient and unsupported-mode
guards. MPI includes a one-box empty rank. CUDA is
translation-unit compile-only, with no GPU execution.

After wiring, parent must compare the actual new-mode force increment and
EV/EH source views at every valid row, including boundary rows, and inject
mixed U/E perturbations at the real pre-push and post-deposit callsites.
Standalone extraction tests cannot qualify production particle/circuit/RNG
rollback, total energy or the consistency closure between trial rho and push.
The accepted stochastic ion transfer remains once-only and its realized energy
must be measured.
