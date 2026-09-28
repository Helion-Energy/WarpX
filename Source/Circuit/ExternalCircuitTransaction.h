/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_EXTERNAL_CIRCUIT_TRANSACTION_V1_H
#define WARPX_EXTERNAL_CIRCUIT_TRANSACTION_V1_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define WARPX_CIRCUIT_TRANSACTION_API_V1 1u
#define WARPX_CIRCUIT_TRANSACTION_POST_ACCEPT_RESTORE_V1 1ull

enum WarpxCircuitTransactionStatusV1 {
    WARPX_TRANSACTION_OK = 0,
    WARPX_TRANSACTION_UNSUPPORTED = 1,
    WARPX_TRANSACTION_BAD_REQUEST = 2,
    WARPX_TRANSACTION_INTERNAL_ERROR = 3
};
enum WarpxCircuitTransactionPhaseV1 {
    WARPX_TRANSACTION_SNAPSHOTTED = 1,
    WARPX_TRANSACTION_OPEN = 2,
    WARPX_TRANSACTION_ACCEPTED_OPEN = 3,
    WARPX_TRANSACTION_ACCEPTED_CLOSED = 4,
    WARPX_TRANSACTION_FAILED_CALL = 5
};
struct WarpxCircuitTransactionViewV1 {
    uint32_t struct_bytes;
    uint32_t phase;
    uint64_t token;
    uint32_t can_cancel;
    uint32_t can_finalize;
};
/* Optional synchronous HOST boundary capability beside unchanged ABI2 and
 * PRE_ACCEPT rejection V1. No callback throws. Only one retained attempt may
 * exist per instance, and it cannot overlap a PRE_ACCEPT snapshot. The caller
 * owns exclusive access, including no concurrent rate/pointer consumers.
 *
 * An ABI2 true advance and FinishStep inside this transaction create an
 * engine-accepted, APPLICATION-PROVISIONAL endpoint. Internal rate/clock setup
 * may read it, but the caller MUST quarantine returned scales, logs, callbacks
 * and all other accepted publications until finalize. Checkpoint write/read,
 * redefinition and another BeginStep are forbidden while a token is retained.
 * No per-step disk checkpoint is used for restoration.
 */
struct WarpxCircuitTransactionApiV1 {
    uint32_t struct_bytes;
    uint32_t api_version;
    uint64_t capabilities;
    /* Before BeginStep, own the complete native state, including both lock
     * stages, hidden storage, clock, held memories, counters and protocol.
     * Failure returns token0 without an owned snapshot or physical mutation.
     * Success invalidates borrowed rate/affine views; generations never rewind.
     */
    int32_t (*snapshot)(void* instance, uint64_t* token, char* error_buffer,
                        uint64_t error_capacity);
    /* Pure preflight. Failed/foreign/stale tokens return a cleared view when
     * struct_bytes matches; a mismatched-size object is not overwritten.
     * Caller must set struct_bytes. Readiness cannot change without a mutating
     * call by the exclusive owner. Use collective preflight before cancel or
     * finalize across ranks; no application validation may follow finalize.
     */
    int32_t (*query)(void* instance, uint64_t token,
                     struct WarpxCircuitTransactionViewV1* out,
                     char* error_buffer, uint64_t error_capacity);
    /* Restore pre-BeginStep state, including after true and FinishStep or a
     * caught failed call. Consume snapshot and discard its deferred logs.
     * No fallible allocation after valid-token preflight. Any failure is
     * terminal to the caller, never evidence of successful restoration.
     */
    int32_t (*cancel)(void* instance, uint64_t token, char* error_buffer,
                      uint64_t error_capacity);
    /* Only after exactly one successful true advance and FinishStep. Consume
     * the snapshot, invalidate borrowed views and flush deferred native logs.
     * No physics change or fallible allocation after valid-token preflight.
     * Double/stale finalize rejects; there is deliberately no discard/release
     * shortcut that can accidentally publish an unvalidated provisional state.
     */
    int32_t (*finalize)(void* instance, uint64_t token, char* error_buffer,
                        uint64_t error_capacity);
};
const struct WarpxCircuitTransactionApiV1*
warpx_external_circuit_transaction_api_v1(void);
#ifdef __cplusplus
}
#endif
#endif
