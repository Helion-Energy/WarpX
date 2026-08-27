# c3 act-bank external-circuit plugin

The validated the reference code's act-bank solver (hetools-greens `Machines/<machine>/`
`the reference circuit module` + `c3_stepper.py`, the 0.0221-relL2-vs-alternate-case
configuration) and the production formation deck's full-mirror-machine adapter
(`c3_bank_source.py`) as a compiled WarpX external-circuit plugin behind
ABI v2 (`circuit.engine = external`, `circuit.plugin_library =
libc3circuit.so`).

Self-contained by construction: the plugin calls **no WarpX or AMReX
symbols** (the two ABI headers are vendored in `vendor/`, including a
double-precision `amrex::Real` shim), builds with a standalone
CMakeLists, and is dlopen'd `RTLD_LOCAL`.

```
vendor/    ExternalCircuit.H (verbatim ABI v2 copy) + AMReX_REAL.H shim
src/       the ports, named by their Python reference module:
           ReferenceCircuit.{H,cpp}   <- the reference circuit module (NetBuilder,
                                      AddCoil, SwdBank incl. the v7.2
                                      reverse-recovery companion) +
                                      c3_bank_source.add_passive_ring
           C3Stepper.{H,cpp}       <- c3_stepper.py (BE interval stepper,
                                      snapshot/restore)
           C3BankNetwork.{H,cpp}   <- c3_bank_source.build() +
                                      C3CircuitAdapter (mirror fold, west
                                      twins, D03/D04 loading, two-stage
                                      flux-lock swap, eps conventions)
           C3ExternalCircuit.{H,cpp}  the ABI implementation
           PluginFactory.cpp       extern "C" factory + ABI stamp
           DenseLU / YamlLite / C3Config   self-contained support
harness/   c3_harness.cpp: offline ABI driver (dlopen or direct)
validation/  export_case.py / reference_run.py / validate.py
```

## Build

```bash
cmake -S . -B build && cmake --build build -j
# -> build/libc3circuit.so, build/c3_harness
```

**Precision contract:** double throughout; the ABI version stamp does
NOT encode the `amrex::Real` width. Load only into a WarpX built with
`-DWarpX_PRECISION=DOUBLE`, same toolchain family. The build pins
`-ffp-contract=off` so every operation rounds exactly like the Python
reference (no FMA fusing).

## Config string (`circuit.plugin_config`)

Either inline `key=value,key=value,...` (lists use `+`) or a path to an
INI-style file of `key = value` lines:

| key        | default        | meaning                                          |
|------------|----------------|--------------------------------------------------|
| `yaml`     | *(required)*   | c3-schema circuit yaml (`make_circuit_yaml` output) |
| `matrix`   | *(required)*   | amp-turn M_EE/M_EW block file (see below)        |
| `rings`    | *(none)*       | ring spec file: `name R_ohm lock` per line (plate rings first, R = 0 locks last) |
| `dt`       | `5.0e-8`       | circuit BE step [s] (the deck's dt/subcycle)     |
| `t0`       | `-200.0e-6`    | machine glob t0 [s] (pre-roll / bias-soak start) |
| `shift`    | `2.5e-6`       | sim = machine + shift [s] (`BANK_T_SHIFT`)       |
| `extra`    | `D03+D04`      | loading-only east circuits appended after the Define coils (no eps, no pushed scale) |
| `eps_sign` | `1.0`          | Lenz A/B insurance on the fed-back EMF (`C3_EPS_SIGN`) |
| `mirror`   | `1`            | fold in the west twins (0 = east-only, e.g. the alternate reference case) |
| `preroll`  | `1`            | run the machine pre-roll inside Define; pass 0 on restart decks (ReadCheckpoint supersedes it) |
| `verbose`  | `1`            | stderr chatter                                   |

`Define` runs the whole boot: yaml parse, matrix fold, assembly, and the
machine pre-roll from `t0` to machine `-shift` (= WarpX t = 0), so the
plugin arrives at t = 0 already rolled. Coil names may carry the deck's
`coil_` prefix (stripped) or the pack `E`/`W` prefix (dropped on yaml
lookup). Port j of Define is east channel j; scales are
`s = I_el * d / I_ref` with `d = part_turns[0]` from the yaml.

### Matrix file (`c3-mel-blocks v1`)

Hexfloat text: `n`, the n east channel names (Define coils + extra +
rings, order-checked), then `M_EE` and `M_EW` as n x n row-major blocks
in the **amp-turn** convention [H] -- exactly the post-symmetrization,
post-ring-kernel-override blocks `c3_bank_source.build()` derives from
the deck's mesh probe. The plugin performs the mirror fold, the
D-folding `mel = M_full * outer(d, d)` and the SPD check (Cholesky) with
the same floating-point expressions as the reference, so `mel` is
bit-identical across the ports. `validation/export_case.py` writes the
file from the deck; the WarpX-side driver must generate it the same way
(see TODOs).

## Latch-point inventory (restore-and-re-advance)

1. **swd switch/diode transitions** -- every `AdvanceInterval` first
   restores the interval-entry snapshot, so switches are evaluated
   against the entry state; transitions occurring inside the interval
   persist ONLY when `accept = true` re-takes the entry snapshot.
   Non-accepted evaluations (any eps, any resized `t1`) leave no trace.
2. **LU refactorization** -- state-derived, never latched; a bit-identity
   LRU factor cache (same swd state => same matrix => same factors)
   absorbs the restore churn without touching trajectories.
3. **machine-0 lock-ring stage swap** -- fires only in `BeginStep`
   (between steps, never inside an evaluation) at the first boundary
   `t0 > 0` with `t0 - shift >= -dt/2`: the boundary nearest machine 0,
   the deck's `max(1, round(shift/dt))` schedule. Stage B inherits the
   live state on the index-identical shared prefix, locks close carrying
   zero current, the clock is continuous, and the locked fluxes +
   station sums are logged.
4. **eps hold** -- overwritten each `AdvanceInterval` from the argument
   (the coupler holds the predictor value); no plugin-side EMA, matching
   the hetools reference (`eps_elec = eps_sign * d * dPhi/dt`, west rows
   mirrored from east, extra/ring channels 0).
5. **checkpoint** -- `WriteCheckpoint` serializes the accepted entry
   snapshot (x, m, the six SwdBank arrays) + `lock_phase` + the live
   stepper clock `t0` in hexfloat: `ReadCheckpoint` restores bit-exactly
   into the matching stage on either side of the swap.

## Validation

```bash
HETOOLS_USE_GPU=0 ~/.env/warpx-mhd/bin/python3 validation/validate.py \
    --workdir /path/to/work            # cases a,b,c,d,e
```

Runs the plugin through the real dlopen ABI path with deliberately
perturbed non-accepted evaluations (midpoint-resized substep + scaled
eps) against the REAL Python machinery on bit-identical inputs
(`mcomb.npy` through a stub `probe_fn` into `c3_bank_source.build()`):

- (a) the reference shot pre-roll + machine [0, +12 us], eps = 0
- (b) sinusoidal eps on coil_F09/A07/C05
- (c) machine-0 lock swap: locked fluxes + station sums at arming,
  post-swap R = 0 linked-flux conservation, dlopen == direct
- (d) checkpoint round trip at machine +6 us (post-swap): bitwise
- (e) alternate-case circuits (reverse-recovery + segment-3/5 + reversed
  formation banks), machine [0, +100 us]

Bar: per-coil `max|ds| / max|s_ref| <= 1e-10` for (a/b/c); bitwise for
(d); median `<= 1e-10`, max `<= 1e-8` for (e). The residual difference
is pivot-order roundoff (DenseLU vs scipy splu) -- same algorithm, same
precision. Case (e)'s wider max bar is measured, not tuned: the 843
compression banks' reverse-recovery turn-off rides the ring current's
zero crossing, and the SAME Python stepper run with scipy dense
`lu_factor` instead of sparse `splu` diverges from itself by 1.1e-9 on
the same C coils -- the class the C++ port lands in (closer to
python-dense than the two Python variants are to each other).

Measured (2026-08-26, this tree): (a) 3.7e-12 / (b) 3.7e-12 /
(c) traj 3.4e-12, arming 5.2e-13, station sums A +68.9749 mWb and
B +249.7787 mWb on both sides, post-swap lock-flux drift 3.1e-12 /
(d) bitwise / (e) max 1.5e-9, median 1.5e-12.

## Remaining WarpX-side integration TODOs

Driver status (the in-tree `Source/Circuit` engine +
`implicit_mhd.circuit_driver = native`):

1. **Matrix-file generation** -- OPEN (deck-side export is canonical):
   `validation/export_case.py` writes the `c3-mel-blocks v1` file from
   the deck's mesh probe (`_bank_discrete_inductance_matrices` + the
   ring-kernel overrides); the deck passes its path through
   `circuit.plugin_config`. A WarpX-side generator would duplicate the
   deck's amp-turn/mirror conventions and is deferred.
2. **Ring/lock eps ports** -- MECHANISM LANDED, deck wiring open: the
   coupler measures ANY `circuit.coils` entry (batched probes, disk or
   reciprocity) and the plugin maps east channels by name, so rings are
   declared as coils with their own probes and `python_scale` fields.
   The deck-side ring declarations (positions, I_ref conventions) are
   the remaining piece.
3. **Painted scales for rings/extra** -- OPEN (deck-side conventions;
   D03/D04 painting and ring painted modes need deck ports + I_ref
   choices).
4. **Restart plumbing** -- CLOSED: `circuit.plugin_restart_config`
   replaces `plugin_config` on restart runs; set it to the same string
   plus `,preroll=0` (ReadCheckpoint supersedes the bias soak).
5. **Checkpoint I/O rank** -- CLOSED: the engine calls WriteCheckpoint
   on the I/O rank only and restores on every rank at InitData, next to
   the scale-segment checkpoint (`circuit_coupling.dat`).
6. **Performance**: the dense LU is ~6 s per factorization at the
   full-mirror 2464 unknowns (fine offline; the LRU cache keeps counts
   low). If in-loop cost matters, port the blocked/tiled factorization
   or a self-contained sparse LU before coupling long runs.
