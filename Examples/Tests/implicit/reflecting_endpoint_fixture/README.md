# Native RZ reflecting endpoint regression

The Eulerian acceptance path supports the existing native **endpoint-only**
radial reflection when the outer field boundary is PEC, the particle boundary
is reflecting, the domain includes the axis, and
`implicit_evolve.reflect_particles_at_rmax=0`. Shape-3 Esirkepov deposition,
MC gathering, and the original virtual interval current/density remain intact.
This does not qualify the separate segmented in-push reflector or infer a bounce
count from endpoint data.

`ImplicitRadialEndpoint.H` invokes the native scalar boundary kernel on private
values after the exact native Cartesian-to-stored-coordinate conversion. It
retains stored `(r, theta, z)` and Cartesian proper momentum, including
`reflect_all_velocities`. Native axial loss decisions use the original endpoint
momentum before radial reflection. The absorbing classifier regression includes
`reflection_model_zhi(u)=if(u>0,0,1)` with positive outgoing `uz` and
`reflect_all_velocities=true`; classifying after reversing `uz` fails this test.

The native position image is `r' = 2*rmax-r`, with unchanged theta. The native
kernel itself describes that position image as approximate; this regression is
not a segmented/specular orbit accuracy proof. Exact-wall endpoints, images
outside the domain, stochastic/thermal actions, unsupported field boundaries,
and unqualified gather/deposition reach still reject before acceptance.

## Why the original charge/current pair remains valid

The native reflecting deposition boundary combines an interior nodal charge
with its reflected guard contribution. `WarpX_PEC.cpp::ReflectJorRho` multiplies
the latter by its guard-to-interior radius ratio after inverse-volume scaling.
For a symmetric particle shape this gives the same folded density for endpoint
`r` and its image `2*rmax-r`: both contribute
`(S_i(r)+S_mirror(i)(r))/r_i`, with the existing native axis/cap quadrature.
No instantaneous reflected-current tangent is substituted for the original
Esirkepov interval current or mass-matrix density response.

The production audit checks the unreflected current/gather/projection support.
Private OU endpoint data use the reflected position/momentum and redistribute
only private particles. On acceptance, even a reflection-only step must execute
native particle boundaries before redistribution and OU; redistribution alone
could delete the outside endpoint. Native charge/count checks then compare the
actual accepted particles to the predicted survivor deposit.

## Tests

`test_reflecting_endpoint_native_rz` uses actual native initialization, implicit
shape-3 Esirkepov deposition, finisher, boundary action, charge synchronization,
private endpoint redistribution, and scrape/tally bookkeeping. The fixture
includes ions, excluded alpha particles, and neutral particles. It checks:

- virtual continuity and survivor-plus-loss continuity with native RZ metrics;
- native reflected stored coordinates/momentum by particle identity, bitwise;
- private versus actual endpoint NGP number and first/second momentum moments;
- input purity, RNG preservation, rollback and repeat before native boundaries;
- absorbed charge/weight/energy and scrape counts, without duplicate booking;
- reflection-only, mixed radial/cap corners, `reflect_all_velocities`, native
  axis quadratures, multiple boxes, MPI seams, and an empty rank.

The four registered CTests are `native_radial_{reflection,corner,axis4,all_velocities}_rz`.
The existing absorbing classifier fixture carries the pure negative/corner cases.
CUDA compilation is supported. GPU execution is a separate qualification gate.

Rollback is qualified only before native boundary tallies/buffers are mutated.
A failure after that explicit irreversibility barrier remains terminal. Radial
reflection supplies a wall momentum impulse and preserves kinetic energy up to
native trig roundoff. Subsequent expected OU receipts describe the later OU map;
they are not a complete wall/field energy conservation proof.

## Portable purity check

The fixture compares particle and registered-field **bytes**, not deposition
hashes or moments, around private endpoint evaluations. All species, identities,
real/integer components (including runtime attributes and neighbors), tile keys,
box ownership, and every registered local FAB including its guards are checked.
Rollback compares every particle byte except the intentionally populated
`wall_*` fixture outputs; recreated-midpoint comparison includes those outputs
and the field registry again. Allocator pointer/capacity is not physical state.
No RNG is invoked by the observer.

GPU atomics can reorder an unchanged positive deposit. Therefore CPU repeated
charge checks remain exactly zero, while GPU repeated-charge checks use the
following a priori bound. This applies only to independent repeat, post-private,
rollback, and retry deposits. **Continuity, partition, actual survivor density,
exact reflected image, charge, and boundary tally gates are unchanged. The
separate GPU signed-moment normalization correction is documented below.** The original strict GPU failure is retained as diagnostic
history, not a passing test.

Let `N` be the maximum actual number of depositing charged particles in a FAB,
summed over all of its species and tiles. Excluded alpha ions count; neutrals
have zero charge. Each shape-3 particle contributes at most one nonnegative term
to a given local node, so its local atomic sum has at most `N` additions. Let
`G` be the number of global FABs. In the asserted nonperiodic layout,
`SumBoundary` receives at most one contribution from each source FAB at a node,
including local and remote donors. Its packing is a copy; the local and received
terms are added once, and their combined count is bounded by `G`. MPI does not
add a second sum of the same terms.

The remaining arithmetic depth is conservatively fourteen operations:

- One positive axis fold and one inverse-volume division.
- Four possible physical-boundary side updates (two dimensions, two sides),
  each bounded by two multiplications and one addition: twelve operations.

Actual axis-none/radial-reflecting/PMC configurations use fewer side updates, but
the bound counts all four. Positive radial mirror ratios are treated as fixed
computed coefficients shared by both deposits. Physical guard replacement can multiply by a radial ratio, but those outside-
domain ghosts cannot feed the compared valid nodes through the final nonperiodic
`FillBoundary` (whose sources are valid nodes). Thus `k=N+G+14` bounds the
rounding depth along each path in the positive linear deposition map. This is a
count bound, not a fitted coefficient or a measured-error multiplier.

With `eps=std::numeric_limits<double>::epsilon()` (conservative relative to unit
roundoff), define `gamma=k*eps/(1-k*eps)`. Both repeated positive results differ
from their same exact positive sum `S` by at most `gamma*S`, and the reference
obeys `rho_ref >= (1-gamma)*S`. Each valid node must satisfy

```
abs(rho_again-rho_ref) <= (1+eps)*2*gamma/(1-gamma)*rho_ref .
```

The extra `(1+eps)` covers the measured subtraction; the bound itself is rounded
outward. Empty reference nodes require exact zero difference. Each receipt gives
`N`, `G`, `k`, the relative bound, maximum difference/bound, and maximum pointwise
fraction of the bound. A bound maximum is a diagnostic: acceptance is pointwise,
not a ratio of global maxima.

The helper explicitly restricts this proof to single-level, one-mode RZ,
shape 3, positive charged species without ionization, unfiltered ordinary charge
deposition, double communication, nonperiodic boundaries, radial PEC with native
reflecting particles, and even PMC caps. It checks actual old/virtual shape
summands are nonnegative and normal after the largest inverse-volume divisor;
nonfinite/cancelling/subnormal cases are outside this proof. This does not assert
portability for every possible deposition mode or boundary condition.

Independent negative controls are available:

```
radial_test.inject_byte_mutation=1
radial_test.inject_density_mutation=nonzero
radial_test.inject_density_mutation=empty
```

The first changes one representable weight per nonempty tile and must fail the
byte comparison. The latter two alter private comparison fields only and must
fail the nonzero-node/empty-node bounds, independently of the particle snapshots.
Each invocation is expected to exit nonzero before native boundary acceptance.

## Cancelling GPU NGP moments

The original corner test divides each private/native moment difference by the
norm of that **signed** moment. An unchanged GPU native redeposit alone can fail
this metric: a y-momentum sum differing by 4096 had a signed reference 155648,
while its absolute particle contributions were about 2.32115e20. The original
failure is retained. This is a correction to that GPU comparison, not a claim
that every original moment-norm assertion passed.

Before accepting any moment comparison, the fixture now requires exact sorted
multiset equality of all complete source tuples actually exposed by the private
trial: species, grid, NGP cell, theta, weight, and Cartesian proper momentum.
Private-refresh repeat and post-boundary native tuples must both agree byte for
byte on every rank. The existing actual native image test independently checks
stored position and momentum by particle identity. The private public API does
not expose IDs; no private-ID correspondence is claimed by the multiset test.

CPU retains the original combined `2e-13` moment ratio. GPU number and second
moments also retain that ratio. Signed first moments on GPU instead require an
a priori absolute summation bound, alongside the mandatory exact tuple check.
The same bound is checked on independently repeated private and native deposits.

For each species/cell/component, `n` is the exact source-tuple count and
`S=sum(abs(w*u^p))`, where `p=0,1,2` as appropriate. The error depth is **n+2**,
including both independently rounded products before atomic summation; equality
of the product bytes is not assumed. Using conservative machine epsilons,

```
gd = gamma(n+2, epsilon_double)
gh = gamma(n+2, epsilon_long_double)
S_upper = abs_sum_long_double / (1-gh)
abs(moment_private-moment_native) <= (1+epsilon_double)*2*gd*S_upper
```

Every bound arithmetic operation and the final conversion round outward.
Absolute products/sums and the bound must have a full binade of under/overflow
margin in double; nonfinite or extreme cases reject. Empty/zero-term channels
require both compared values to be exactly zero. MPI redistribution precedes
the NGP deposits, so each cell is owned locally and no MPI moment sum is omitted
from this count. Per-component receipts give counts, depths, differences,
bounds, and maximum pointwise fractions. The historical signed reference ratio
continues to be printed separately.

Two further negative controls change only private comparison fields and must
reject independently of the particle/tuple checks:

```
radial_test.inject_moment_mutation=nonzero
radial_test.inject_moment_mutation=empty
```

The first exceeds the per-cell number-moment bound by a factor of four; the
second writes a nonzero value in the excluded-species empty channel. Neither
changes native particles, forces, boundaries, RNG, or physical source laws.
