# Native retained C3 acceptance

This positive-density, source-free RZ fixture uses the actual native affine circuit provider, with the optional retained post-accept ABI. It runs native true acceptance and FinishStep before both endpoint validators, and keeps the provider, coupler, external segments, field/thermal/history/particle/RNG and midpoint diagnostic snapshots cancelable until endpoint success. Full native plugin and coupler checkpoint bytes are compared as well as the raw in-process oracle.

The initial inactive Eprev allocation is zeroed only after asserting have_prev=false, identically in every arm. Derived nine-band native mass-matrix tangent scratch is excluded by pointer identity; cancellation must invalidate its readiness and all dependent response/density caches. PC generation remains monotone. Accepted native receipts occur exactly once. Actual midstep Poynting data must be nonzero and is restored on decline.

The CPU runner compares retry, clean retained and nonretained controls exactly. GPU in-process byte restoration has the same contract, but cross-process arithmetic requires separate qualification. CMake only registers the CPU comparisons when NATIVE_RETAINED_C3_PLUGIN and NATIVE_RETAINED_C3_CONFIG are supplied; NATIVE_RETAINED_C3_OLD_PLUGIN enables the missing-capability negative. The configuration must match the pinned dt=1e-10, t0=-1e-9 native circuit baseline and its coil/matrix files.

The MPI ownership control moves every candidate particle to a valid box owned by another rank only after endpoint verification, then deliberately rejects. This proves whole-map cancellation through Redistribute; it does not qualify a physical orbit crossing. The empty-rank arm uses one box on two ranks and asserts exactly one rank is empty.

Combined vacuum+circuit, extra prescribed fields, stopping/fusion, topology-changing physical events and generic terminal MLMG/device convergence errors remain unsupported. There is no maintained-alpha or temporal-order claim.

The empty-rank setup explicitly sets amr.refine_grid_layout=0 so AMReX does not split its one box to fill both MPI ranks. Original attempted max_grid_size-only setup correctly failed its empty-rank assertion before any step. The combined-vacuum negative specifies its required native_edge_candidate policy so it reaches the retained-dispatch guard, rather than failing an unrelated initialization policy check.
