/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL. */
#ifndef WARPX_EXTERNAL_CIRCUIT_RATE_CLOCK_V1_H
#define WARPX_EXTERNAL_CIRCUIT_RATE_CLOCK_V1_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define WARPX_CIRCUIT_RATE_CLOCK_API_V1 1u
#define WARPX_CIRCUIT_RATE_CLOCK_F64 1u
#define WARPX_CIRCUIT_RATE_CLOCK_NATIVE_OPERAND_ENCLOSURE_V1 1u
#define WARPX_CIRCUIT_RATE_CLOCK_NATIVE_LATTICE_PROVENANCE_V1 2u

enum WarpxCircuitRateClockStatusV1 {
    WARPX_RATE_CLOCK_OK = 0,
    WARPX_RATE_CLOCK_UNSYNCHRONIZED = 1,
    WARPX_RATE_CLOCK_AMBIGUOUS = 2,
    WARPX_RATE_CLOCK_BAD_REQUEST = 3,
    WARPX_RATE_CLOCK_UNSUPPORTED_ARITHMETIC = 4,
    WARPX_RATE_CLOCK_INTERNAL_ERROR = 5
};
/* Additive optional capability: neither ExternalCircuit ABI2 nor rate API v1
 * changes. A value-only certificate for one existing accepted rate token.
 * The provider encloses the errors in the actual machine-clock expressions
 * time-shift and T0+double(step)*dt against exact arithmetic on their represented
 * operands (and exact integer step). No state time is relabeled or advanced.
 * OK requires agreement within sync_error_bound AND disjoint compatibility
 * neighborhoods of both adjacent lattice points. This is numerical clock
 * compatibility, not evidence of continuous-circuit temporal order.
 * Policy NATIVE_LATTICE_PROVENANCE additionally includes the configured native
 * origin construction/reset and canonical caller clock arithmetic. Simulation
 * time is a direct integer multiple or a grouping/sum of same-sign native-dt
 * increments from zero (integer-multiple field steps included). For n increments
 * its a priori rounding envelope uses at most n-1 binary64 rounding operations,
 * with gradual underflow; it does not infer a tolerance from observed mismatch.
 * Fractional/off-lattice endpoint rates and overlapping envelopes are refused.
 * The provider reconstructs origin provenance from immutable configuration and
 * existing serialized native origin; neither state_time nor state is projected.
 * On nonzero status the object is diagnostic only and must not be used.
 */
struct WarpxCircuitRateClockViewV1 {
    uint32_t struct_bytes;
    uint32_t scalar_kind;
    uint32_t clock_policy;
    uint32_t reserved;
    uint64_t token;
    int64_t step_index;
    double requested_time_sim;
    double state_time_sim;         /* unchanged rate-v1 reported state time */
    double lattice_dt;
    double time_origin_machine;
    double time_shift;
    double requested_time_machine;
    double state_time_machine;
    double sync_error_bound;       /* upward absolute bound [s] */
    double neighbor_separation;    /* downward gap of compatibility regions [s] */
};
struct WarpxCircuitRateClockApiV1 {
    uint32_t struct_bytes;
    uint32_t api_version;
    uint64_t capabilities;
    /* Read-only query outside field residual/Jv. Token and requested time must
     * match the current rate-v1 preparation exactly. All rate lease expiration
     * rules apply; querying does not invalidate or extend that lease. The caller
     * binds this certificate to that packet and its own accepted-state epoch.
     * No C++ exception crosses this ABI. No release is needed for value fields.
     */
    int32_t (*query)(void* instance, uint64_t rate_token, double requested_time_sim,
                    struct WarpxCircuitRateClockViewV1* out,
                    char* error_buffer, uint64_t error_capacity);
};
const struct WarpxCircuitRateClockApiV1* warpx_external_circuit_rate_clock_api_v1(void);
#ifdef __cplusplus
}
#endif
#endif
