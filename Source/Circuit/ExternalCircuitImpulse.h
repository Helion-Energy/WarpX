/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_EXTERNAL_CIRCUIT_IMPULSE_V1_H
#define WARPX_EXTERNAL_CIRCUIT_IMPULSE_V1_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define WARPX_CIRCUIT_IMPULSE_API_V1 1u
#define WARPX_CIRCUIT_IMPULSE_SERIES_RLC_V1 1ull
#define WARPX_CIRCUIT_IMPULSE_F64 1u
enum WarpxCircuitImpulseStatusV1 {
    WARPX_IMPULSE_OK = 0, WARPX_IMPULSE_UNSUPPORTED = 1,
    WARPX_IMPULSE_BAD_REQUEST = 2, WARPX_IMPULSE_INTERNAL_ERROR = 3
};
enum WarpxCircuitImpulsePhaseV1 {
    WARPX_IMPULSE_PRE_FIELD = 1, WARPX_IMPULSE_POST_FIELD = 2
};
/* Optional HOST setup/publication capability beside unchanged ABI2. This
 * first topology has one unit-turn port and five native coordinates. Upload
 * g once; residual/Jv applications must use device coefficients and perform
 * no provider callback. chi is the oriented port voltage impulse [V s].
 *
 * delta_x is a direct homogeneous response, not a difference of full rates.
 * Differential slots have units of state per chi; algebraic slots are zero.
 * zeta contains time-integrated algebraic voltage per chi, not endpoint volts.
 * The finite algebraic endpoint is reclosed separately on publication.
 * Views expire on release, next impulse prepare, any ABI2 mutation, retained
 * cancel/finalize, or impulse commit. Generations never rewind on rollback.
 */
struct WarpxCircuitImpulseViewV1 {
    uint32_t struct_bytes, scalar_kind, phase, reserved;
    uint64_t token, transaction_token, n_port, n_state;
    double time_sim;
    const double* current;
    const double* g;
    const double* i_ref;
    const double* state;
    const double* delta_x;
    const double* zeta;
};
struct WarpxCircuitImpulseReceiptV1 {
    uint32_t struct_bytes, phase;
    uint64_t token, transaction_token;
    double time_sim;
    double coil_energy_change, local_inductor_energy_change, capacitor_energy_change;
    double port_work, resistor_impulse_work, energy_defect, arithmetic_bound;
};
struct WarpxCircuitImpulseApiV1 {
    uint32_t struct_bytes, api_version;
    uint64_t capabilities;
    /* Requires exact native clock synchronization and the original retained
     * transaction. PRE is before BeginStep; POST is after true+FinishStep.
     * Each phase permits one successful publication per retained transaction.
     * Preparation validates the actual accepted MNA storage and switch state.
     * No state, clock, ordinary source, resistor flow or switch is advanced. */
    int32_t (*prepare)(void* instance, uint64_t transaction_token, double time_sim,
                       uint32_t phase, struct WarpxCircuitImpulseViewV1* out,
                       char* error_buffer, uint64_t error_capacity);
    /* Small accepted-boundary call only. Stages and validates the full state
     * before publication; all failures leave physical state untouched. The
     * result remains application-provisional and cancellable by the original
     * retained token. It publishes no accepted logs/checkpoint by itself. */
    int32_t (*commit)(void* instance, uint64_t token, const double* chi, uint64_t n_port,
                      struct WarpxCircuitImpulseReceiptV1* out,
                      char* error_buffer, uint64_t error_capacity);
    void (*release)(void* instance, uint64_t token);
};
const struct WarpxCircuitImpulseApiV1* warpx_external_circuit_impulse_api_v1(void);
#ifdef __cplusplus
}
#endif
#endif
