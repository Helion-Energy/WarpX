/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "Circuit/ExternalCircuitRejection.h"
#include <stddef.h>
_Static_assert(offsetof(struct WarpxCircuitRejectionApiV1, struct_bytes) == 0,
               "C capability starts with size");
static const struct WarpxCircuitRejectionApiV1 empty_api = {
    sizeof(struct WarpxCircuitRejectionApiV1), WARPX_CIRCUIT_REJECTION_API_V1,
    WARPX_CIRCUIT_REJECTION_PRE_ACCEPT_V1, 0, 0, 0};
const struct WarpxCircuitRejectionApiV1* test_c_rejection_api(void) {
    return &empty_api;
}
