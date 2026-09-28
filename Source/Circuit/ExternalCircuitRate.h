/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 * Optional accepted-state differential response; ExternalCircuit ABI2 is unchanged.
 */
#ifndef WARPX_EXTERNAL_CIRCUIT_RATE_V1_H
#define WARPX_EXTERNAL_CIRCUIT_RATE_V1_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define WARPX_CIRCUIT_RATE_API_V1 1u
#define WARPX_CIRCUIT_RATE_F64 1u
#define WARPX_CIRCUIT_RATE_FROZEN_ACCEPTED_COEFFICIENTS_V1 1u

enum WarpxCircuitRateStatusV1 {
    WARPX_RATE_OK = 0,
    WARPX_RATE_UNSUPPORTED = 1,
    WARPX_RATE_PHASE_UNSUPPORTED = 2,
    WARPX_RATE_BAD_REQUEST = 3,
    WARPX_RATE_INTERNAL_ERROR = 4
};

/* Pure accepted-state response dI_channel/dt = p0 + G*eps, with eps in
 * the same per-port V/turn convention/order as ABI2. It is NOT an interval
 * secant and advances neither clock nor device states. Algebraic variables
 * are solved consistently at held differential storage. Switch/recovery
 * coefficients are frozen at the accepted state, not reevaluated per trial.
 * All pointers are HOST memory: copy once outside field residual iterations.
 * No second-order temporal claim follows from this packet; the integrator's
 * accepted circuit lattice time is exposed separately from requested time. */
struct WarpxCircuitRateViewV1 {
    uint32_t struct_bytes;
    uint32_t scalar_kind;
    uint32_t coefficient_policy;
    uint32_t reserved;
    uint64_t token;
    uint64_t n_port;
    double requested_time_sim;
    double state_time_sim;
    double lattice_dt;
    const double* current; /* accepted painted channel current [A-turn] */
    const double* p0;      /* [n_port], current rate at zero port EMF */
    const double* g;       /* [n_port,n_port], current rate per port EMF */
    const double* i_ref;   /* s=I_channel/i_ref */
};
struct WarpxCircuitRateApiV1 {
    uint32_t struct_bytes;
    uint32_t api_version;
    uint64_t capabilities;
    /* Call only at a closed accepted boundary, after Define/ReadCheckpoint or
     * FinishStep. The returned state is accepted, never a trial's live scratch.
     * Views expire on release, another rate preparation or any ABI2 mutation.
     * Complete H2D copies before release. No C++ exception crosses this ABI. */
    int32_t (*prepare)(void* instance, double time_sim,
                      struct WarpxCircuitRateViewV1* out,
                      char* error_buffer, uint64_t error_capacity);
    void (*release)(void* instance, uint64_t token);
};
const struct WarpxCircuitRateApiV1* warpx_external_circuit_rate_api_v1(void);
#ifdef __cplusplus
}
#endif
#endif
