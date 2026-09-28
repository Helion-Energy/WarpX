/* Additive immutable model identity capability. The virtual ABI2 and all
 * affine/rejection/rate/clock layouts remain unchanged. */
#ifndef WARPX_EXTERNAL_CIRCUIT_IDENTITY_H_
#define WARPX_EXTERNAL_CIRCUIT_IDENTITY_H_
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define WARPX_CIRCUIT_IDENTITY_API_V1 1u
#define WARPX_CIRCUIT_IDENTITY_IMMUTABLE_MODEL_V1 1u
/* Bytes include canonical format AND provider/model version. Exact comparison
 * is required; an opaque path or hash of a path is not a model identity.
 * Borrowed immutable storage remains valid until Define/destruction. Query is
 * observational and does not require or mutate an interval/rate token. */
typedef struct WarpxCircuitIdentityViewV1 {
    uint64_t struct_bytes;
    uint64_t data_bytes;
    const char* data;
} WarpxCircuitIdentityViewV1;
typedef struct WarpxCircuitIdentityApiV1 {
    uint64_t struct_bytes;
    uint32_t api_version;
    uint32_t flags;
    int32_t (*query)(void*, WarpxCircuitIdentityViewV1*, char*, uint64_t);
} WarpxCircuitIdentityApiV1;
/* Optional export: warpx_external_circuit_identity_api_v1. 0 means success;
 * nonzero is an invalid/undefined model or provider failure. */
#ifdef __cplusplus
}
#endif
#endif
