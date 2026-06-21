# Dey–Mittra conformal B-advance for the hybrid EB solver — plan

Branch: `ohms_law_dey_mittra_eb` (off `ohms_law_conformal_EB`).
Status: **plan only** (no implementation yet). This documents the motivating
edge-order findings and a concrete plan to add a Dey–Mittra conformal magnetic
update at the embedded boundary.

> Reference note: the requested page
> (bohrium.com .../Dey-Mittra_conformal_method) refused programmatic fetch
> (HTTP 405). This plan is based on the standard reference — S. Dey & R. Mittra,
> *"A locally conformal finite-difference time-domain (FDTD) algorithm for
> modeling three-dimensional perfectly conducting objects,"* IEEE Microwave
> Guided Wave Lett. 7(9), 273 (1997) — and should be cross-checked against the
> page before implementation.

---

## 1. Motivation — the edge-order findings (the reason to try Dey–Mittra)

Measured on the `ohm_solver_eb_diffusion` resistive-decay convergence test
(interior L2 of `By` vs the analytic decaying eigenmode):

| treatment | flat tilted square | smooth cylinder (N=96→192→384) |
|---|---|---|
| stair-step | ~1.0 | **1.00, 1.00** |
| conformal **ECT** (Xiao–Liu enlarged cell) | **2.79** | **1.03, 0.99** (engages, ECT/stair L2 ≈ 0.60, but **1st order**) |
| collocated **mirror** fill | −0.86 (corner-limited) | grows, then **blows up at N=384** (7.5e+32) |

**Key findings:**
1. The hybrid conformal EB (the mature staggered **ECT**) is genuinely
   **1st-order on a smoothly-curved wall**, even though it is 2.79 on the
   flat/tilted square. So the conformal-EB 2nd order is a *flat/tilted-wall*
   property in this hybrid path, not a curved-wall one — **nothing currently
   reaches 2nd order on a curve**.
2. The collocated **mirror** fill is **unstable** on the curved wall at high
   resolution (blows up at N=384); it is curvature-limited and worse than
   merely non-convergent there. (It is stable in the lower-resolution formation
   runs — the instability is resolution/config dependent.)
3. The **EB-aware div(B)/div(J) clean** (committed on `ohms_law_conformal_EB`)
   is order-neutral (validated: div-free field → 0 injection; wall residual
   byte-identical clean-ON vs clean-OFF). It removes the clean's prior O(h)
   wall injection but does not change the underlying mirror order.
4. **C-ECT verdict:** since the staggered ECT (same conformal method, staggered)
   is only 1st-order on the curve, the collocated C-ECT analog cannot recover
   2nd order on a curve either → it does not meet the "recovers 2nd order"
   bar, so it was not revived.

**Implication:** a *different* conformal treatment is worth trying for genuine
2nd-order curved-wall accuracy. Dey–Mittra is the canonical area-scaled
cut-cell conformal FDTD; the ECT is "Dey–Mittra + small-cell enlargement". A
plausible hypothesis is that the ECT's enlargement/borrowing smears the
boundary and costs the order on a curve, and that **plain Dey–Mittra (no
enlargement) recovers 2nd order** — this plan builds it so we can test that
directly. (It is equally possible the 1st order is a deeper hybrid issue —
the test settles it.)

---

## 2. The Dey–Mittra method

For a cell face cut by a PEC boundary, replace the Faraday update of the normal
B component with the finite-volume (conformal) form:

```
dB_n/dt = -(1 / S_open) * ∮ E · dl       (integral over the UNCUT edges of the face)
```

- `S_open` — the open (uncut) area of the cell face (the fraction of the face
  in the fluid, times the physical face area).
- the line integral runs over the **partial (uncut) primary edge lengths**;
  the PEC condition `E_tangential = 0` on the boundary means the cut portion of
  the contour contributes nothing, so only the uncut edges carry E.
- regular (uncut) faces recover the standard Yee curl (`S_open` = full area,
  full edges); covered faces (`S_open = 0`) are skipped.

**Accuracy:** 2nd-order for smooth curved PEC boundaries (the area/edge-length
weighting captures the geometry to 2nd order), vs ~1st-order for staircase.

**Small-cut-cell stability problem:** the update has a `1/S_open` factor, so a
tiny cut face drives the explicit CFL limit to `dt ∝ S_open` → the smallest cut
cell collapses the global time step. Dey–Mittra's standard mitigations:
- **area thresholding** — faces with `S_open` below a fraction of the full
  area are excluded/merged (treated as covered or lumped into a neighbor); the
  threshold trades a small accuracy loss for a workable dt; or
- accept the reduced global dt set by the smallest retained cut cell.
The **ECT (Xiao–Liu)** is the alternative mitigation: *enlarge* small cut cells
by borrowing area from neighbors (the repo's
`Enlarged_cells_..._time_step_reduction.pdf`), keeping a full-cell dt. So:
- **Dey–Mittra** = area-scaled update + thresholding (simpler; small-cell dt
  sensitivity).
- **ECT** = area-scaled update + enlargement (no dt hit; more machinery; what
  we measured at 1st order on the curve here).

---

## 3. WarpX integration plan (staggered grid first)

The Dey–Mittra ingredients **already exist on the staggered conformal path**
(verified via code map):
- `face_areas` (FieldType) = the open (uncut) cut-face area `S_open`
  (`EmbeddedBoundaryInit.cpp` `ComputeFaceAreas` + `ScaleAreas`).
- `edge_lengths` = the partial uncut primary edge lengths.
- `ECTRhofield` already holds **exactly the Dey–Mittra area-scaled circulation**:
  `EvolveRhoCartesianECT` (`EvolveECTRho.cpp:70-174`) computes
  `Rho(face) = (∮ E·edge_length) / face_area`. It is precomputed each substep in
  `HybridPICModel::FieldPush` (`HybridPICModel.cpp:1351-1361`) on the staggered
  conformal path.
- These are built **only** on the staggered conformal path
  (`WarpXInitData.cpp:1616-1628`); the collocated path has only
  `distance_to_eb`. **So start on the staggered grid.**

### 3a. New B update routine `EvolveBCartesianDeyMittra`
Add alongside `EvolveBCartesianECT` in
`Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp` (`#ifdef AMREX_USE_EB`,
3D + XZ like ECT). Core update, for each B face with `S_open > S_thresh`:
```
B(i,j,k) -= dt * Rho(i,j,k)          // Rho = ∮E·dl_partial / S_open, already from EvolveECTRho
```
This is the ECT "stable un-intruded" update (`EvolveB.cpp:455`) applied to
**all** cut faces — i.e. Dey–Mittra without the enlargement / borrowing /
`area_mod` / `Venl` machinery. **Do not divide by `S` again** — `EvolveECTRho`
already divided the circulation by `face_area`.

Small-cell stability floor: a tunable `dey_mittra_area_threshold` (fraction of
the full face area, default e.g. 0.5). Faces with `S_open/S_full < threshold`
are either skipped (left to the regular Yee update of the host cell) or lumped —
start with "skip + fall back to stair for that face" and measure; refine to a
merge if needed. Report (log) the smallest retained `S_open` and the implied dt
margin so the dt restriction is visible.

### 3b. Dispatch + flag
- Add a selector `hybrid_pic_model.conformal_b_method = mirror | ect | dey_mittra`
  (or a boolean `dey_mittra_eb`) plumbed like `m_use_conformal_eb`
  (constructor `pp.query` in `HybridPICModel.cpp` → `m_*` member → read in
  `FieldPush`/`EvolveB`).
- New branch in `FiniteDifferenceSolver::EvolveB` (`EvolveB.cpp:108`, parallel
  to the ECT branch) selecting `EvolveBCartesianDeyMittra` when the flag is set.
- Keep `EvolveECTRho` running (it produces `Rho`); the Dey–Mittra routine
  consumes it. No collocated changes (deferred — see risks).

### 3c. Reuse vs new allocations
`face_areas`, `edge_lengths`, `ECTRhofield` are reused as-is. `area_mod`,
`Venl`, `m_borrowing`, the face-extension machinery are **not needed** for
Dey–Mittra (they are the enlargement) — a leaner alloc set can be introduced
later; initially just gate them off when `dey_mittra` is selected.

---

## 4. Validation plan

1. Add a `--dey-mittra` arm to
   `inputs_test_3d_ohm_solver_eb_diffusion_picmi.py` (parallel to `--conformal`)
   and route staggered + dey_mittra → the new path.
2. **Cylinder edge-order** (`cylinder_edge_order.py`, already on this branch):
   add a Dey–Mittra config and run N=96,192,384 on the GPU node alongside
   stair / ECT / mirror. **The decisive number:** does Dey–Mittra reach ~2nd
   order on the cylinder where ECT is 1.0? (interior L2 vs the Bessel mode).
3. **Flat square** (existing `eb_diffusion` `analysis.py`): confirm Dey–Mittra
   is ≥2nd order on the flat/tilted walls (sanity — it must at least match ECT's
   2.79 there).
4. **Stability sweep:** vary `dey_mittra_area_threshold` and resolution; record
   where the small-cut-cell dt restriction bites (the trade-off vs accuracy).
5. If Dey–Mittra reaches 2nd order on the curve: it belongs in the PR as the
   curved-wall B treatment; write the algorithm description (the user asked for
   one once a conformal method validates).

---

## 5. Risks / open questions

- **It may also be 1st order.** If the curved-wall 1st order is a deeper hybrid
  issue (the Ohm E-solve on the curve, the metric, the RK substepping) rather
  than the ECT enlargement, Dey–Mittra will not fix it. The cylinder test
  settles this before any PR commitment — *this is the whole point of the
  experiment.*
- **Small-cell dt collapse** is the central Dey–Mittra cost; the threshold is a
  tuning trade-off. ECT exists precisely to avoid it — if Dey–Mittra needs a
  large threshold to stay stable, it may lose the accuracy edge.
- **Collocated** is out of scope initially: `face_areas`/`edge_lengths` are not
  built on the collocated path (`WarpXInitData.cpp:1653` deliberately skips
  them). A collocated Dey–Mittra would first need those metrics built nodally —
  substantial extra work. Staggered first.
- **Dimensionality:** 3D + XZ only (the ECT metric machinery aborts otherwise).
- Cross-check the update formula and the threshold convention against the
  Dey–Mittra paper / the bohrium page before coding.

---

## 6. Files to touch (implementation, when greenlit)

- `Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp` — dispatch branch
  (~:108) + new `EvolveBCartesianDeyMittra` (alongside ECT ~:232).
- `Source/FieldSolver/FiniteDifferenceSolver/EvolveECTRho.cpp` — reuse as-is
  (circulation `Rho`).
- `Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.{H,cpp}`
  — the `conformal_b_method` selector + plumbing; route `EvolveECTRho` +
  `EvolveB` for the dey_mittra method.
- `Examples/Tests/ohm_solver_eb_diffusion/inputs_test_3d_ohm_solver_eb_diffusion_picmi.py`
  — `--dey-mittra` arm; `cylinder_edge_order.py` — Dey–Mittra config; that
  dir's `CMakeLists.txt` — a chained convergence CTest.
- `Docs/source/usage/parameters.rst` — document the new selector.
