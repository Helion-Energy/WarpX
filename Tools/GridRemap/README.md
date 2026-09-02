# Grid remap for cylindrical hybrid-PIC states

Continue a hybrid-PIC run on a refined mesh, so that a resolution study
costs the window of interest instead of the whole simulation.

Motivation: in a compression experiment a transition was observed at a
time when the radial density gradient in the low-density region measured
only 1.4-2.7 cells across, i.e. at or below what the mesh can represent.
Deciding whether such a transition is physical or an artifact of the
discretisation requires re-running it at higher resolution -- but a 2x
radial refine costs roughly 8x per unit physical time (twice the cells,
four times the field substeps, since the whistler substep limit scales
as the square of the cell size), so re-running from the start is
prohibitive. Remapping shortly before the event and running only across
it costs a small fraction of that.

## Why this is tractable at all

In the hybrid (generalized Ohm's law) model:

* **B is the only field state.** E is re-derived from Ohm's law every
  step and J from `curl(B)` plus the particle deposit, so neither has to
  be transferred.
* **Macro-particles are Lagrangian, hence grid-independent.** The same
  particle list is a valid state on any mesh covering the same domain.
  No resampling, no reconstruction.

So a remap is: prolong B divergence-free, carry the particles verbatim.

## The one delicate part

A component-wise interpolation of B does **not** preserve `div(B) = 0`.
The injected divergence is then transported and amplified by the field
advance. Measured on a production-shaped mesh (256x1024, refine 2x
radially), component-wise interpolation gives `max|div B| = 2.6e+04 T/m`
where the correct prolongation gives `1.0e-11 T/m` -- a factor of 2.5e15.

`grid_remap.py` implements a divergence-preserving prolongation
specialised to an axisymmetric cylindrical Yee mesh. For each coarse
cell the divergence is a face-flux balance (no theta term when
`d/dtheta = 0`):

```
div(B) V = 2 pi dz [r_out B_r(out) - r_in B_r(in)]
           + pi (r_out^2 - r_in^2) [B_z(top) - B_z(bot)]
```

and the prolongation inherits it exactly by construction:

1. `B_r` on r-faces that already exist is **copied** -- those faces are
   geometrically unchanged, so their flux is unchanged.
2. `B_z` on each z-face is **split across sub-annuli with the flux sum
   forced to equal the coarse flux** to roundoff.
3. `B_r` on newly created r-faces is **not interpolated**. It is solved
   for by imposing `div(B) = 0` on each new sub-cell, sweeping outward.

Given 1 and 2, imposing zero divergence on all but the outermost
sub-cell forces it in the outermost one too -- so the result is
divergence-free to roundoff wherever the input was, for arbitrarily
rough fields. `B_theta` is cell-centred and does not enter the
axisymmetric divergence, so its interpolation cannot affect the result.

## Validation

`test_grid_remap.py` builds test fields as the *discrete* curl of a
vector potential, which makes them divergence-free to roundoff by
construction, so any divergence measured afterwards is attributable to
the prolongation alone. Rough (white-noise) content is included so the
test is not restricted to smooth modes.

Representative results (256x1024, refine 2x radially):

| quantity | value |
|---|---|
| coarse `max|div B|` | 7.6e-12 T/m (roundoff) |
| prolonged `max|div B|` | 1.0e-11 T/m (roundoff) |
| naive component-wise | 2.6e+04 T/m |
| outer radial flux, coarse vs fine | rel. diff 1.3e-13 |
| axial field energy, coarse vs fine | rel. diff 2.4e-14 |

All cases pass at ratios 1, 2 and 4, on axis-touching and off-axis
meshes. Run `python3 test_grid_remap.py`.

## Cost at production scale

`bench_remap.py`. Field prolongation is negligible: 0.02 s for
256x1024 at ratio 2, 0.04 s for 512x2048. The binding constraint is
particles, not fields:

* carrying 128M particles costs 6.7 GB resident (7 doubles each);
* the engine's file-injection path reads the **entire list on a single
  rank into host vectors**, so it needs roughly twice that, ~13 GB, on
  one rank.

That fits on a large-memory node but does not scale, and the engine
source carries an explicit `TODO: Make changes for read/write in
multiple MPI ranks`. Treat single-rank memory as the ceiling when
sizing a remap.

## The particle round-trip, and what it loses

Injection from file requires an openPMD file with exactly one iteration
and one species, providing `position` and `positionOffset` (x/y/z as the
geometry requires), `momentum` (x/z, and y when present), and
`weighting`. `charge` and `mass` are optional and are overridden by an
explicit `charge`/`mass`/`species_type` in the deck.

**Only `weighting` is carried as a per-particle attribute.** The base
cylindrical attributes survive because they are reconstructed from
position and momentum, but any *runtime* per-particle component does
not round-trip. Anything a species accumulates per particle beyond
position, momentum and weight must be treated as reset by a remap. Check
the species' runtime components before trusting a remapped continuation,
and state in the run record that they were reset.

Particles outside the destination domain are silently dropped with a
warning ("Simulation box doesn't cover all particles") -- verify the
particle count and total weight across the remap rather than assuming.

## Loading the field back: two traps

1. **`LoadInitialFieldFromPython` targets the wrong array.** It sets
   `B/Efield_fp_external`, i.e. the *external* field, not the field
   state. Loading a remapped total B through it would leave the
   plasma-frame field at zero. Write the remapped B into the field state
   directly instead (via the field register / MultiFab wrappers) from an
   `afterinit` callback.
2. **Beware double-counting the external field.** The hybrid
   initialisation adds the external B into the field state at the first
   step of a run, and the field state holds the *total* between steps
   while the substep loop works in the plasma frame. Decide explicitly
   whether the remapped array is total or plasma-frame and make the
   load consistent with it, or the run starts with one extra copy of the
   applied field. This is worth an explicit boot check on the first
   dump: compare the loaded on-axis field against the source run's.

For a general field the parser route is unavailable in cylindrical
geometry regardless: the external-field parser hard-codes the initial
radial and azimuthal components to zero.

## Designing the comparison: the particles-per-cell confound

Refining the mesh at fixed particle count **halves the
particles-per-cell**, so a bare coarse-vs-fine comparison confounds
resolution with sampling noise (which grows as `1/sqrt(PPC)`, i.e. 1.41x
for a 2x refine). Any conclusion of the form "the feature changed when I
refined" is then unattributable.

Use three arms:

| arm | mesh | PPC | isolates |
|---|---|---|---|
| A | coarse | full | reference |
| B | coarse | half | sampling noise alone (A vs B) |
| C | fine | half | resolution alone (B vs C) |

`ppc_match.py` builds arm B by weight-preserving random decimation with
an explicit seed. **Do not** build the matched arm by splitting
particles into coincident copies: that restores the PPC count without
restoring the statistics, because copies at identical positions carry
identical deposition error. The matched arm must be made by removing
samples, not by duplicating them.

## Identity gate (performed)

The gate that licenses every later inference: remap at **refinement
ratio 1** -- where the prolongation is the identity -- continue, and
compare against the uninterrupted reference over the same interval.
Anything that shows up is a round-trip loss, not a resolution effect.

Fixture: cylindrical hybrid case, 64x128, 16 ppc (262k particles), one
kinetic species, evolving structured profile, split at step 100 and
continued 100 steps against a 200-step reference. Small, but it
exercises the whole path end to end -- openPMD particle round-trip,
field-state write, hybrid continuation -- which is what the gate tests.
Both arms use identical particle counts.

**Run single-threaded.** With `OMP_NUM_THREADS=8`, two bit-identical
reference decks diverge from *each other* by 4-9e-3 by step 200 --
threaded deposition ordering reseeds roundoff, which then amplifies.
That is the same magnitude as the effect being measured, so a threaded
gate reports a false failure. At `OMP_NUM_THREADS=1` the code is
bit-reproducible (two identical runs differ by exactly 0.0), and only
then does the gate mean anything.

| measurement | value |
|---|---|
| rho at load vs source state | **max rel 6.0e-16**, L2 1.3e-16 |
| total \|rho\| at load | difference **exactly 0** |
| div B, reference | 6.0e-13 T/m (rel 4.0e-15) |
| div B, continuation | 6.8e-13 T/m (rel 4.6e-15) |
| B after 100 steps | max rel **8.0e-4**, L2 4.2e-5 |
| rho after 100 steps | max rel 1.5e-4, L2 1.4e-5 |
| total \|rho\| after 100 steps | rel 5.6e-7 |
| field energy drift | rel 1.2e-6 |

**Verdict: the remap is faithful; the residual is not a round-trip
loss.** The state at the moment of load reproduces the source to
roundoff -- 6.0e-16 in deposited rho, with total weight identical to the
last bit. There is no systematic error: no velocity-centering offset, no
dropped particles, no weight loss. What remains is a roundoff-level seed
(particles return in file order rather than the internal sorted order,
so deposition sums in a different order) which then amplifies:

| step | difference |
|---|---|
| 0 (load) | 6.0e-16 |
| 20 | 1.8e-3 |
| 100 | 8.0e-4 |

Growth is ~1.4 per step until roughly step 20, then it **saturates** and
stops growing. That is chaotic amplification reaching the local noise
amplitude, not a defect -- and it is the same level two identical
threaded runs reach on their own.

### What this means for how the tool must be used

**Pointwise trajectory comparison is meaningless beyond ~20 steps here.**
The inference "the fine continuation differs from the coarse one,
therefore resolution matters" is *invalid* stated that way, because two
identical runs differ by the same amount after the Lyapunov time. Any
resolution study built on this tool must therefore:

1. compare **statistical** measures -- profiles, moments, scale lengths,
   spectra, time-averages -- not cell-by-cell field differences;
2. carry a **chaotic-floor control**: a same-resolution remapped
   continuation, whose divergence from the reference is the noise floor
   the resolution signal has to beat;
3. quote the resolution effect only where it **exceeds that floor**.

Combined with the three-arm particles-per-cell design above, a defensible
study is: coarse/full, coarse/half (sampling), fine/half (resolution),
plus a ratio-1 remap of the coarse arm (chaotic floor). A difference
matters when it is larger than both the sampling arm and the floor arm.

### Reproducing

```
cd identity
export OMP_NUM_THREADS=1          # required; see above
python3 ref.py  --nsplit 100 --nsteps 200 --nr 64 --nz 128 --ppc 16 --outdir id_ref
python3 cont.py --particles id_ref/pdump/openpmd_000100.h5 \
                --bfield    id_ref/B_split.npz --nsteps 100 --outdir id_cont
python3 compare.py --ref id_ref --cont id_cont --split 100 --final 200
```

`compare.py` reports B and rho differences, divergence cleanliness on
both arms, weight and energy drift, and classifies the outcome. Note it
labels anything above roundoff as needing attribution -- the attribution
for this fixture is above.

### Still untested

The external-field variant. This fixture sets the initial field through
the grid-init style, which writes the field state directly, so total and
plasma-frame B coincide and the decomposition cannot be got wrong. A
production state carrying a hybrid external-field register must decide
explicitly whether the remapped array is total or plasma-frame, and
verify it on the first dump, per the traps above.
