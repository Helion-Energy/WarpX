# Native circuit rejection before acceptance

This slice adds an optional in-memory plugin snapshot/cancel capability beside the unchanged ExternalCircuit ABI2. It preserves the state BEFORE BeginStep, including phase/topology changes driven by the attempted dt. ABI2 BeginStep or FinishStep cannot substitute for cancellation. The maintained RL fixture rejects reopening an open step, and the maintained C3 plugin can ArmLocks and replace its entry at BeginStep.

## Parent integration

The loader may resolve `warpx_external_circuit_rejection_api_v1` and pass its result to `coupler.SetRejectionCapability(api)` after the existing constructor. Missing export is represented by nullptr; existing ABI2/affine behavior remains unchanged. The setter rejects invalid size/version/callbacks. `SupportsNativeRejection()` reports the advertised capability, not an active transaction.

The caller sequence for a reversible native attempt is:

1. Require successful `SnapshotNativeStep(external_vector_potential)` BEFORE step-entry measurement and BeginStepMeasured. The external object and plugin/capability must outlive the attempt. All ranks must take the same lifecycle and collectively require success before proceeding.
2. Run ordinary BeginStepMeasured, optional affine preparation, and host/device residuals. No new field/circuit-vector transfer is added inside residual/Jv; snapshot metadata is host state and the plugin owns its own in-memory snapshot.
3. If the attempt is rejected before engine acceptance, call `CancelNativeStep()`. Success restores exact pre-BeginStep coupler interval/count/dt, latest/start/accepted linkage maps, filter memory, flags and all Python-scale segment values/times. It synchronizes device work at this boundary, clears borrowed device scale pointers and releases DeviceCircuit storage/readiness. The parent separately restores field/particle/thermal arrays and its own circuit-step-open flags. No field refresh is done here.
4. On acceptance, both host FireEngine and device CommitDarwinDeviceStep call the irreversibility barrier BEFORE the first native `AdvanceInterval(...,true,...)`. It releases the plugin/local snapshot before filter/accepted bookkeeping can move. FinishStep closes the accepted protocol. Any later failure is outside this rollback contract, even if an accepting call throws before returning. Parent must arrange all reversible checks before this point.

`NativeStepCancelable()` means a snapshot exists. `NativeRejectionFailed()` means a plugin cancellation/capability failure poisoned the transaction; fail-stop is required, with no retry claim. Missing capability or invalid preflight simply refuses reversible preparation. A failed cancellation may already have cleared trial device views but does not restore host metadata or claim engine restoration. Layout preflight failure retains the pending snapshot without calling the engine. Do not replace failure with FinishStep or checkpoint serialization.

The helper owns no accepted field arrays and never invokes a disk checkpoint. It snapshots all Python-scale segments, including nondriven external fields, and refuses to read a preexisting active device view. Geometry/unit-field probe caches are derived and remain reusable for unchanged geometry; field arrays themselves are parent-owned rollback state.

## External provider contract

`ExternalCircuitRejection.h` is a self-contained C header with snapshot/cancel/release callbacks, a version/size stamp and PRE_ACCEPT capability. Snapshot returns a nonzero instance-specific token and preserves accepted physics; failure must return token0 and own no snapshot. Cancel restores pre-BeginStep dynamics, topology/phase, held histories and protocol readiness, invalidates attempt-derived affine views, and consumes its snapshot on success. Failed cancellation is terminal. Release discards the snapshot without changing physics and is idempotent. Callbacks must not throw across the C boundary. The C++ wrapper contains exceptions and rejects malformed providers; it never fabricates an ABI2 lifecycle call.

The pinned maintained C3 library `e58f2215c2e89bce367775388812a5cd37876ccb92ccad58968471089f829d86` does NOT export this capability and is correctly reported non-retryable by this slice. Its separate provider implementation/lock-transition validation is required before claiming maintained C3 rollback. The new native RL oracle includes a synthetic dt-dependent phase/current mutation to exercise snapshot ownership; it does not model C3 network topology.

## Native tests and scope

`build_native.py` copies and hashes the selected native WarpX link libraries into the owned build directory, verifies source libraries did not change during copying, and compiles the actual modified CircuitCoupler.cpp/DarwinDeviceCoupling.cpp. The fixture executes actual native coupler methods and the actual native ExternalVectorPotential segment methods against a compiled C++ RL engine. It does not construct a full WarpX instance or advance fields. A fixed committed input file seeds filter/linkage memory once; no checkpoint file is used for cancellation and plugin checkpoint entry points deliberately throw.

```sh
python3 Examples/Tests/implicit/circuit_rejection_fixture/build_native.py \
  --native-build /path/to/completed/native/rz/build --output build-circuit-rejection
python3 Examples/Tests/implicit/circuit_rejection_fixture/run_tests.py \
  --build build-circuit-rejection --legacy-plugin /path/to/original/rl/plugin.so \
  --c3-plugin /path/to/pinned/maintained/libc3circuit.so \
  --output /path/to/owned/evidence --mpiexec /path/to/mpiexec
```

The original unmodified `Examples/Tests/implicit/rl_test_circuit.cpp` may be built as a shared library against the same Source and AMReX headers for the missing-capability control. Preserve that compile command, including OpenMP when AMReX requires it. Production-header RZ/3D syntax and CUDA compilation cover both changed translation units; `test_c_api.c` checks C11 compatibility.

Fourteen cases run with one and two MPI ranks: exact replay, smaller-dt retry, rejected phase mutation, device-view cleanup, missing capability, malformed API, failed snapshot, preexisting device view, terminal failed cancellation, accepting-call failure, wrong token, original RL and maintained C3 missing-export controls, and refusal of FinishStep before acceptance. The clean/native retry twins compare exact engine state and scale segments; a separate analytic RL/filter check prevents identical shared errors from passing. Initial phase mutation is restored before smaller-dt retry. Double snapshot/cancel and capability replacement are rejected. No GPU execution, full field advance, maintained C3 phase rollback or post-acceptance rollback is claimed.
