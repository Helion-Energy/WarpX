/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_RETAINED_ACCEPTANCE_H_
#define WARPX_NATIVE_RETAINED_ACCEPTANCE_H_
class WarpX;
namespace warpx::implicit {
// Acceptance/rollback policy only. This does not enable a stopping/event model.
bool NativeRetainedAcceptanceEnabled();
enum class AcceptancePhase {
    BeforeEndpoint, AfterEndpoint,
    BeforePreSource, AfterPreSourcePrepare, AfterPreSourceCommit, AfterPreSourceReclosure,
    BeforeSymmetricField, AfterSymmetricField,
    BeforePostSource, AfterPostSourcePrepare, AfterPostSourceCommit, AfterPostSourceReclosure
};
// Optional native acceptance validator, outside residual/Jv. A validator may
// inspect/stage only state covered by the current transaction. It must not
// publish time/step, mutate external state or retain particle pointers across
// rollback. False is a collective decline; no accepted receipt is emitted.
// Empty means accept. Installation is quiescent, once per WarpX driver.
struct NativeAcceptanceValidator {
    bool (*check)(WarpX&, AcceptancePhase, void*) = nullptr;
    void* context = nullptr;
};
void SetNativeAcceptanceValidator(NativeAcceptanceValidator);
bool CheckNativeAcceptance(WarpX&, AcceptancePhase);
}
#endif
