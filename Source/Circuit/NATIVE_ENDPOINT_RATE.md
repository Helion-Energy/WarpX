# Native endpoint circuit rate API

This is an optional accepted-boundary prerequisite. It does not open a Darwin
field/geometry capability guard or advance the circuit. The existing ABI2 virtual
layout, optional interval-affine ABI, and differential-rate v1 packet are unchanged.
Older plugins continue to use their existing interval path. `SupportsNativeEndpointRate`
requires both differential-rate and clock-certificate exports, double precision,
disk/none probes, and zero EMF lowpass time.

`ConfigureDarwinMagneticResponse` caches the existing free-space unit probe Q and
clears the endpoint boundary response H. The caller supplies H from the actual
boundary operator, in exact coil-by-all-external-field order. H and Q are not
interchangeable. The setter pins level-zero domain, physical bounds, periodicity,
coordinate convention, Bz layout and external-field order. Unit fields and coil
metadata must remain immutable. Regrid or replacement of those objects requires
explicit reconfiguration. Ordinary time/restart invalidation keeps H.

At a closed accepted circuit state, call `PrepareNativeEndpointRate(time,b,error)`.
The prescribed rate vector has every external-field entry and zero at coupled ports.
Preparation owns host plugin calls, binds clock token/request/state/dt, validates
finite clock operands and unique lattice bounds, maps signed I_ref and turns in
native coil order, copies/uploads the Schur maps, releases the lease, and builds
the native device probe weights. All ranks participate; preparation failure leaves
readiness false. Active borrowed interval scale views reject before host scale reads.
The optional certificate is required here even at zero shift. The lower-level
`DeviceCircuitRate::Prepare` keeps its old null-certificate unshifted-only interface.

The trusted provider certifies the operand-derived arithmetic enclosure; the consumer
independently checks native separated clock expressions, exact packet binding, the
requested-machine discrepancy, a positive adjacent-neighborhood gap, and a radius/gap
that cannot reach a neighboring represented lattice point. No fitted tolerance or
accepted-time relabeling is used. This proves clock compatibility, not circuit temporal
order or a forward-error bound for an ill-conditioned circuit/field Schur complement.

For each operator evaluation, pass homogeneous Bdot_z to `ApplyNativeEndpointRate`.
It uses the existing LinkageBatch device measurement and uploaded maps. CUDA/HIP has
no host coil-vector readback or plugin callback in Apply; multi-rank GPU use requires
existing GPU-aware MPI. The CPU implementation uses the established host oracle.
`affine=false` excludes p0 and prescribed rates; `affine=true` includes both. Outputs
are borrowed const device vectors and become usable only after Apply.

`NativeEndpointRateGeneration` advances on every invalidation and preparation entry;
overflow is rejected. An enclosing field context must pin the generation after a
successful Prepare and check both readiness and generation before using it. Begin,
trial mutation, accept, cancel, memory restore and Q reconfiguration invalidate
readiness. Direct mutations through `Plugin()` or ExternalVectorPotential outside
those paths require explicit invalidation. No plugin pointer or accepted-state lease
is consulted inside Apply. Output references expire at invalidation/next Apply;
the caller must complete asynchronous consumers before destroying the owner.

Validation uses a real native C3 two-port accepted packet with signed references,
turn normalization, permuted fields and a prescribed third field. Independent
long-double two-by-two elimination and separate DiskFluxLinkage calls check the
uncondensed rate/EMF equations. Shifted clocks, lag/corruption, A/B/A, input and
checkpoint purity, cancellation and reprepare, MPI seams/empty ranks, and deliberate
preparation/layout/output errors are covered. RZ tests instantiate the primitive
after native initialization; they do not bypass or qualify the existing RZ external
driver guard. Parent-owned field context qualification is separate.
