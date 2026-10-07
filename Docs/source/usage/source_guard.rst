Hybrid source eligibility (experimental)
---------------------------------------

The ``hybrid_pic_model.source_guard`` controls are default off. The initial
implementation requires single-level, single-mode RZ including the axis, explicit evolution,
and the electron energy equation. These controls gate modeled energy sources;
they do not change resistivity in Ohm's law, conduction, compression work,
fusion, particle orbits, charge deposition, or current deposition.

Example for a separately qualified D/T attribution run::

    hybrid_pic_model.source_guard.enabled = 1
    hybrid_pic_model.source_guard.background_species = ions_D ions_T
    hybrid_pic_model.source_guard.minimum_resident_markers = 16
    hybrid_pic_model.source_guard.n_min = 0
    hybrid_pic_model.source_guard.axis_cells = 1
    hybrid_pic_model.source_guard.ohmic = 1
    hybrid_pic_model.source_guard.thermal_relaxation = 1
    hybrid_pic_model.source_guard.alpha_stopping = 1
    hybrid_pic_model.source_guard.diagnostic_interval = 1
    hybrid_pic_model.source_guard.export_interval = 0
    hybrid_pic_model.source_guard.output_prefix = source_guard

The three channel switches default to one under the default-off master switch.
``background_species`` is required and cannot contain duplicates, neutral species,
or species excluded from thermal relaxation. Counts are computed separately for
each listed species. Alpha counts do not qualify the background; no minimum alpha
count is imposed. ``minimum_resident_markers`` must be a positive integer and
defaults to 16. This is a dust cutoff, not a validated statistical accuracy
threshold. It requires problem-specific 8/16/32 sensitivity checks.

``n_min`` must be finite and nonnegative. Each channel requires physical deposited
charge density divided by e strictly above the larger of this value and its
existing physical source floor. Ohmic heating also retains ``joule_heating_n_min``.
Neither a density floor nor a pedestal qualifies physical vacuum. ``axis_cells``
is nonnegative and defaults to one full cell. It must be smaller than the radial
domain length.

Native support
~~~~~~~~~~~~~~

A cell is rejected if any of its physical-density corner nodes is at/below the
channel's threshold, any configured background species has fewer resident
computational markers than the cutoff, or it lies in the excluded axis strip.
A node inherits the union of rejection reasons from all adjacent in-domain
cells, including periodic neighbors. Both nodes bounding the first radial cell
are therefore excluded. This construction widens exclusion around density and
sampling discontinuities. It does not represent a sharp cell-centered mask.

Stopping checks exactly the nonzero linear weights of its heat-deposition stencil.
If any receiving node is rejected, the whole particle interaction is skipped before
rate evaluation and momentum change. For example, excluding the first radial cell
also excludes stopping in the next cell's interior. A particle exactly on a node
ignores its zero-weight corners. The alpha still participates in all its other
configured processes.

The guarded electron relaxation retains the existing exponential temperature
update. It saves masked nodal ``nu`` and ``nu*T_bath`` before that update. The ion
OU leg averages those same coefficients to its native cell grid, using the ratio
of the averages for the bath temperature. A rejected cell has zero rate at every
bounding node, and hence zero ion relaxation. Neighboring cells can have partial
averaged rates. Redirected heat is controlled separately. Counts and masks refresh
at each source phase and stopping collision; native temperature moments refresh
at every guarded thermal source phase. Diagnostic cadence does not control these
refreshes.

Accepted stopping packets retain their signed lab-frame kinetic-energy transfer.
The predictor/corrector half-deposits consume that packet with their receiving
capacities. A later mask/density change does not cancel an already accepted
packet: its capacity is bounded below by the existing density floor. The existing
negative-heat temperature-floor limit still applies, and any declined removal
remains in the existing stopping-decline ledger. Inspect the stopping residual;
this guard is not a finite-electron-reservoir collision algorithm.

Diagnostics and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reason bits are vacuum=1, low count=2, axis=4. Coverage lines report disjoint
reason codes 0 through 7 so rejected volume/inventory is not counted twice, plus
mask transitions and count histograms. Nodal integrals use clipped RZ control
volumes and unique MPI owners. Marker inventory is the sum of macro weights.
Linear contributor diagnostics expose ``sum(w*S)`` and ``sum((w*S)^2)``;
``Neff=(sum a)^2/sum(a^2)`` refers to this linear source stencil, not the native
higher-order temperature estimator. Cloning does not create independent samples.

``diagnostic_interval`` is a positive outer-step interval (default one) for
coverage and phase extrema. ``export_interval`` is zero to disable field export,
or a positive outer-step interval. Exported AMReX MultiFabs contain the actual
cell/node reason masks, per-species resident counts and weight inventories, and
linear contributor moments, in the order of ``background_species``. Alpha collision
exports also contain exact accepted/rejected interaction counts aggregated by
resident cell, including zero-weight stencil boundary decisions. Paths include
step and source epoch. Use a fresh output prefix/run directory for each restart
segment; diagnostic counters/epochs restart with the process and are not a
cumulative checkpointed science budget.

Source-energy lines report actual electron changes and interval powers. The ion
ledger reports realized total energy change and a thermal-only counterfactual
using the same three random draws; the incremental redirect share includes its
noise cross term. These attributions sum to the realized ion change. Thermal
exchange has sampling and operator-splitting error and is not conserved exactly
by the stochastic OU model. A separate frozen-state nonrelativistic OU expectation
is reported to distinguish time/grid bias from realized sampling variation; it is
not an exact relativistic expectation. No realized-energy correction is introduced.

Declined modeled Ohmic heating is tallied before redirection. This is not total
field dissipation: heating and field resistivity parsers may differ. Vacuum or
nonfinite counterfactuals are counted as unevaluable species-node evaluations;
a zero reported declined energy does not bound their omitted power. Arbitrary
parser singularities remain a model/input responsibility.

Skipped thermal exchange reports a signed, frozen per-species electron update
only when the original physical density exceeds the solver floor and the native
temperatures, species normalization and receiving capacity are finite and valid.
Zero/invalid sampled Ti is explicitly unevaluable. The rate must also be finite
and nonnegative. These estimates do not mutate temperature or shared coefficients,
do not consume RNG draws, and do not predict the realized skipped OU exchange.
Species estimates use the bath at their actual source phase; summing them is not
a full unguarded trajectory with sequential hypothetical bath updates.

Skipped stopping reports the signed hypothetical lab-frame particle energy change
and conjugate electron packet, using the unchanged rejected particle and its
current background. Density, temperature, rate, velocity and candidate result are
validated before unsafe operations. Invalid counterfactuals are counted separately.
The export's first two components remain accepted/rejected counts, followed by
hypothetical signed electron energy[J], unevaluable count and evaluated count.
No rejected particle momentum, staged heat or RNG state changes. These frozen
estimates do not include subsequent electron capacity/floor clipping or establish
a bound on missing power from unevaluable interactions.

Eligible invalid states use a device error flag and host abort, including release
GPU builds where AMReX device assertions are disabled. Excluded invalid states
remain skipped. The guard does not replace invalid physical data with a zero bath.

Phase maximum records include coordinates, owning rank, physical and pedestal
densities, resident counts, linear effective counts, native sampled Ti, and the
most recent per-channel temperature changes. On the existing open-set temperature
abort, a bounded native neighborhood is saved before stopping. Existing physical
health thresholds must remain in place. Production startup, memory overhead,
source-power sensitivity, and fusion/confinement interpretation require separate
qualification by the run owner.
