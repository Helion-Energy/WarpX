/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_EXTERNAL_CIRCUIT_REJECTION_V1_H
#define WARPX_EXTERNAL_CIRCUIT_REJECTION_V1_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define WARPX_CIRCUIT_REJECTION_API_V1 1u
#define WARPX_CIRCUIT_REJECTION_PRE_ACCEPT_V1 1ull

enum WarpxCircuitRejectionStatusV1 {
    WARPX_REJECTION_OK = 0,
    WARPX_REJECTION_UNSUPPORTED = 1,
    WARPX_REJECTION_BAD_REQUEST = 2,
    WARPX_REJECTION_INTERNAL_ERROR = 3
};
/* Optional capability beside the unchanged ExternalCircuit ABI2 vtable.
 * All operations are synchronous HOST step-boundary operations. No exception
 * may escape a callback. instance is the original factory's ExternalCircuit*.
 */
struct WarpxCircuitRejectionApiV1 {
    uint32_t struct_bytes;
    uint32_t api_version;
    uint64_t capabilities;
    /* Before BeginStep: own an immutable in-memory snapshot of accepted state
     * AND protocol readiness, topology/lock phase, histories and held memories.
     * BeginStep may change these based on attempted dt. Accepted physics is
     * untouched by snapshot. Success returns a nonzero instance-specific token.
     * Failure returns token0, owns no snapshot and changes no accepted state.
     */
    int32_t (*snapshot)(void* instance, uint64_t* token, char* error_buffer,
                        uint64_t error_capacity);
    /* Only before ANY accept=true call: restore exactly pre-BeginStep state,
     * including open-step flags, and invalidate attempt-derived affine views.
     * Success consumes token and snapshot. Failure is terminal: caller must
     * not retry or claim restoration. The caller does not use FinishStep as
     * cancel. No per-step checkpoint/disk serialization is permitted here.
     */
    int32_t (*cancel)(void* instance, uint64_t token, char* error_buffer,
                      uint64_t error_capacity);
    /* Discard only; never change physics. Idempotent for the token. Called
     * BEFORE the first accepting call (the irreversibility barrier), or while
     * destroying a failed/unused transaction. Instance must still exist.
     */
    void (*release)(void* instance, uint64_t token);
};
const struct WarpxCircuitRejectionApiV1*
warpx_external_circuit_rejection_api_v1 (void);
#ifdef __cplusplus
}
#endif
#endif
