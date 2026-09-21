/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 * Optional immutable affine response beside unchanged ExternalCircuit ABI2.
 */
#ifndef WARPX_EXTERNAL_CIRCUIT_AFFINE_V1_H
#define WARPX_EXTERNAL_CIRCUIT_AFFINE_V1_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define WARPX_CIRCUIT_AFFINE_API_V1 1u
#define WARPX_CIRCUIT_AFFINE_F64 1u
#define WARPX_CIRCUIT_AFFINE_SIGN_GUARD_V1 1u

enum WarpxCircuitAffineStatusV1 {
    WARPX_AFFINE_OK = 0,
    WARPX_AFFINE_UNSUPPORTED = 1,
    WARPX_AFFINE_INTERVAL_TRANSITION = 2,
    WARPX_AFFINE_PHASE_UNSUPPORTED = 3,
    WARPX_AFFINE_BAD_REQUEST = 4,
    WARPX_AFFINE_INTERNAL_ERROR = 5
};

/* All pointers below address HOST memory. Caller copies the immutable packet
 * once before device trials. No pointer is a C++ container or device pointer.
 * n_port and ordering equal the original Define call. Array products/counts
 * are checked for overflow before allocation. Token is opaque to the host. */
struct WarpxCircuitAffineViewV1 {
    uint32_t struct_bytes;
    uint32_t scalar_kind;        /* F64 only in v1 */
    uint32_t guard_kind;         /* SIGN_GUARD_V1 */
    uint32_t reserved;
    uint64_t token;
    uint64_t n_port;
    uint64_t n_guard;
    uint64_t circuit_substeps;
    double t0_sim;
    double t1_sim;
    double guard_relative_band;  /* nonnegative relative guard margin */
    double guard_absolute_band;  /* must be zero in v1 */
    const double* entry_current; /* [n_port], accepted channel current at t0 */
    const double* p0;            /* [n_port], raw published channel current */
    const double* g;             /* [n_port,n_port], row-major current / eps */
    const double* i_ref;         /* [n_port], scale = current / i_ref */
    const double* guard_v0;      /* [n_guard] */
    const double* guard_g;       /* [n_guard,n_port], row-major sense / eps */
    const int32_t* guard_sign;   /* [n_guard], -1/0/+1 */
    const uint64_t* guard_device;/* [n_guard], diagnostics only */
    const uint64_t* guard_substep;/* [n_guard], diagnostics only */
};

struct WarpxCircuitAffineApiV1 {
    uint32_t struct_bytes;
    uint32_t api_version;
    uint64_t capabilities;
    /* instance is the unchanged factory's ExternalCircuit* converted to void*.
     * BeginStep must already have been called exactly once. Prepare restores
     * that accepted entry and MAY alter scratch/caches, NEVER accepted state.
     * It does not call BeginStep, accept, FinishStep, or publish field scales.
     * Errors never throw across this C boundary; diagnostics use caller buffer.
     * out->struct_bytes is set by caller before entry. */
    int32_t (*prepare)(void* instance, double t0_sim, double t1_sim,
                      struct WarpxCircuitAffineViewV1* out,
                      char* error_buffer, uint64_t error_capacity);
    /* Views are valid until release or any mutating ABI2 instance operation.
     * Caller must finish asynchronous H2D copies BEFORE release. Accept/BeginStep,
     * checkpoint restore, topology change or target change invalidate all tokens.
     * release is idempotent for the current token and does not alter physics. */
    void (*release)(void* instance, uint64_t token);
};

/* Optional dlsym symbol. Missing = unsupported, never silent host fallback
 * in strict device mode. Factory/version symbols and ABI2 remain unchanged. */
const struct WarpxCircuitAffineApiV1*
warpx_external_circuit_affine_api_v1(void);

#ifdef __cplusplus
}
#endif
#endif
