/* Diagnostic adapter. Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL. */
#ifndef WARPX_NATIVE_INSTANTANEOUS_ION_CURRENT_H_
#define WARPX_NATIVE_INSTANTANEOUS_ION_CURRENT_H_
#include <ablastr/fields/MultiFabRegister.H>
class WarpX;
namespace warpx::particles {
enum class InstantaneousIonState { Current, Old, VirtualEndpoint };
/** Deposit the h->0 limit of the SAME native Esirkepov current into caller-owned
 * Yee scratch. Current reads stored positions/momenta. Old reads saved x_n/u_n.
 * VirtualEndpoint mirrors native finishing, including RZ set/get representation,
 * without mutating particles or their primary interval deposit. Caller owns
 * prior native reflection; no absorption, redistribution or fresh BC action is
 * inferred. false clears output for a nonfinite/unresolved/unsupported point.
 * This is an instantaneous companion only, never a continuity replacement.
 */
bool DepositNativeInstantaneousIonCurrent(WarpX&,InstantaneousIonState,
                                          ablastr::fields::VectorField const&);
}
#endif
