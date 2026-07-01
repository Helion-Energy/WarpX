# Plasma-off-the-wall: MVP stand-in + the DielectricEB / sheath follow-ups

**Date:** 2026-07-01 · **Branch:** `eb_cylindrical_correction` · **Status:** MVP decisions + follow-up notes

This note records what the MVP ships for keeping the plasma off the metal wall, why the
heavier machinery is deferred, and the physics findings to revisit later (dielectric liner,
PEC sheath, electron-pressure BC). Companion: `DIELECTRIC_EB_CLASS_SCOPE.md` (the full
second-EB scoping study).

---

## 1. What the MVP ships

### 1a. Standoff = a Python particle scraper (not the C++ offset, not a second EB)
The liftoff deck (`Examples/Tests/ohm_solver_plasma_cylinder_liftoff/inputs_test_3d_...py`)
holds the plasma off the wall with `install_wall_scraper()`: a `callfromparticlescraper`
callback that flags ions with `sqrt(x^2+y^2) > r_standoff` invalid via the zero-copy pyAMReX
SoA / idcpu interface (`unpack_ids` → sign-flip → `pack_ids`), so `Redistribute` removes them
the same step. The `particlescraper` callback fires in `WarpX::Evolve` (`WarpXEvolve.cpp:720`)
**before** Redistribute (`:729/:748`), the C++ EB scrape (`:755`), and `deleteInvalidParticles`
(`:760`), so the flag is honoured that step. Backend-agnostic (numpy on CPU, cupy on GPU) via
`load_cupy()`; the id/cpu arrays alias WarpX's own buffers, so the flag writes back with no copy.

**Why not the C++ `warpx.eb_particle_scrape_offset`.** That path scrapes on
`distance_to_eb < offset` (`ParticleScraper.H:195`). But AMReX `FillSignedDistance`
(`AMReX_EB_utils.cpp`) only computes a true signed distance within a few cells of the EB; every
deep-fluid node is **clamped** to `+ls_roof = (flags.nGrow()+1) * min(dx)`. The field EB factory
is built with `nGrow = guard_cells.ng_FieldSolver.max()` (small — a few cells; `WarpX.cpp:2344`),
so `ls_roof` is only a few cells. Whenever the requested standoff `offset >= ls_roof`, the
clamped deep-fluid value `+ls_roof` trips `phi < offset` and **the entire interior annulus is
scraped** — measured as ~99.96 % plasma loss at `standoff=3` on a wide annulus (R_INNER=0.5),
while a narrow in-band annulus (R_INNER=0.7) survives because its plasma sits inside the roof.
Testing the radius directly in Python sidesteps the clamp entirely. The C++ offset remains valid
only for a narrow standoff (`offset < ls_roof`); it is left in place but the deck no longer uses it.

### 1b. Electron-pressure PEC BC: Neumann → Dirichlet
`hybrid_pic_model.eb_pe_dirichlet` (PICMI `eb_pe_dirichlet`), **default `true`** = Dirichlet.
Flipped the electron-pressure EB fill at `HybridPICModel::CalculateElectronPressure` from the
even (Neumann, zero-normal-gradient) reflection to the odd (Dirichlet-0) reflection in
`ApplyEBBoundaryToNodalScalar` (`EBJBoundary.cpp:2134`; `odd=true` ⇒ `f = (s/d_im)*f_im` → Pe→0
at the wall). **Rationale:** a PEC wall supports a normal (radial) E via surface charge, and the
Dirichlet pressure gives a non-zero `grad(Pe)` across the wall that supplies that allowable radial
field in Ohm's law. The previous Neumann reflection pinned `grad(Pe)·n = 0` and suppressed the
near-wall radial E. Either parity keeps `grad(Pe)` stencils off the nonpositive mirrored density
inside the conductor. Toggle with `eb_pe_dirichlet=False` to recover the old Neumann behaviour.

### 1c. Warning: plasma against the wall
`HybridPICModel::WarnIfPlasmaAgainstWall` emits a one-time warning if `|rho|` in the near-wall
fluid band `0 < distance_to_eb <= nox*dx` reaches a significant fraction of the peak `|rho|`
(scale-free test). Deposition reaches `nox` cells, so plasma that close lays current onto
wall-adjacent edges and drives near-wall current spikes. The message recommends scraping at
least `nox*dx` off the wall (e.g. `--standoff-cells >= particle shape order`). Cheap: a single
masked reduction on the first call, then latched.

---

## 2. Follow-up: DielectricEB objects for macroscopic phenomena

The scrape is a geometry hack; the physically-motivated path is a **second embedded boundary**
(a dielectric liner standing off inside the metal EB) so the standoff *and* macroscopic
dielectric physics live in the field solve. Full design + AMReX feasibility (two coexisting
`EB2::IndexSpace`s, the `top()`/regrid hazard, integration points) are in
`DIELECTRIC_EB_CLASS_SCOPE.md`. Deferred from this MVP because it is a net-new subsystem
(~250–400 LOC + build wiring + regrid hardening) and the scrape proves the point for now.

**eps_r = 1 for the first cut.** Do NOT add a dielectric electron-pressure BC yet: the pressure
gradient at the dielectric surface depends on the liner's polarizability, so a correct BC needs
the eps_r model first. Keep eps_r = 1 (a purely geometric standoff / particle reflector) until the
material model is in.

### What "PEC / dielectric" is in WarpX today (subagent audit, 2026-07-01)
WarpX has **no dielectric boundary of any kind** — no `pec_dielectric`/`dielectric` in the
`FieldBoundaryType` enum (`WarpXAlgorithmSelection.H:124-140`), no eps_r interface, no surface
charge, no D-normal continuity, anywhere. Every EB is a **perfect electric conductor**:
- **Electrostatic:** fixed-potential Dirichlet `phi|_EB = eb_potential(x,y,z,t)` via
  `MLEBNodeFDLaplacian::setEBDirichlet` (`PoissonSolver.H:364`) — Dirichlet-only, no Robin path wired.
- **FDTD/PSATD:** emergent PEC from cut-cell masking (`eb_update_E/B` skip).
- **Hybrid Ohm's law:** the `mirror_combine` image-point PEC (`EBJBoundary.cpp:710`): for E/J
  (`normal_odd=false`) `w_n=1` (normal zero-gradient), `w_t=s/d_im<0` (tangential → E_tan=0); for B
  (`normal_odd=true`) swapped → B_normal=0. A mathematically pure PEC.
- Relative permittivity exists only as a **bulk** FDTD medium (`MacroscopicProperties`,
  volumetric, no EB coupling); the EffectivePotential "dielectric function" is a numerical
  plasma-frequency dressing, not a material. Scraped charge is **diagnostics-only**
  (`ParticleBoundaryBuffer` → `BoundaryScrapingDiagnostics`), never fed back to the solve.

**So a dielectric liner in the hybrid solver is net-new physics**, not a config of existing code:
it needs (1) a non-PEC field-BC parity (the `mirror_combine` knob is only `normal_odd`, a PEC
parity swap — it cannot express an eps_r jump), (2) an `eps_r` field + surface-charge field `sigma_s`
with a D-normal-continuity update `eps0 E_n^plasma - eps_r eps0 E_n^wall = sigma_s`, and (3) a
surface-charge accumulation model feeding scraped charge onto the liner. The one reusable piece is
the nodal-scalar BC (`ApplyEBBoundaryToNodalScalar`), which already carries per-field odd/even
parity and could hold a scalar surface-charge condition; the vector-E dielectric jump is new work.

---

## 3. Finding to revisit at the implicit stage: the PEC sheath / wall function

Holding the plasma against a bare PEC wall is not just a numerical nuisance — it is **physically
under-modelled**. At a conducting wall, quasineutrality breaks down and a **sub-grid sheath**
forms (Debye-scale, unresolved on the hybrid grid). The hybrid Ohm's-law closure assumes
quasineutrality, so the near-wall layer is outside its validity; this is a **likely driver of the
observed near-wall / late-time wall instability** (the wall B/J growth seen in the liftoff, which
was shown NOT to be LSQ-specific — the masked path grows identically). The Dirichlet pressure BC
(§1b) is a crude improvement (it at least admits a radial E) but is not a sheath model.

The correct treatment is a **wall function / sheath boundary condition** (a sub-grid model for the
sheath potential drop and the ion/electron fluxes to the wall). This gets substantially more
complicated **when magnetized** (oblique B at the wall → magnetized-sheath / Chodura layer, with
the sheath structure depending on the field angle). **Deferred.** Revisit when doing the implicit
solver, where the near-wall stiffness this introduces is tractable. For now the MVP keeps the
plasma off the wall (the scrape) so we do not fight the sheath at all.

---

## 4. Summary of code touched (MVP)
- `Examples/.../ohm_solver_plasma_cylinder_liftoff/inputs_test_3d_...py`: `install_wall_scraper()`
  Python callback; deck wires it in place of `warpx.eb_particle_scrape_offset`.
- `HybridPICModel.{H,cpp}`: `m_eb_pe_dirichlet` (default true, ParmParse `eb_pe_dirichlet`);
  electron-pressure BC flipped to Dirichlet; `WarnIfPlasmaAgainstWall` + `m_checked_plasma_at_wall`.
- `Python/pywarpx/picmi.py`: `eb_pe_dirichlet` HybridPICSolver kwarg (mirrors `eb_rho_dirichlet`).
- No change to the C++ scraper (`ParticleScraper.H` / `eb_particle_scrape_offset`): left in place,
  documented limitation (narrow standoff only).
</content>
