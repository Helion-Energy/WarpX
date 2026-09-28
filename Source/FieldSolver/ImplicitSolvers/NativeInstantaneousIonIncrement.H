/* Diagnostic prototype. Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL. */
#ifndef WARPX_NATIVE_INSTANTANEOUS_ION_INCREMENT_H_
#define WARPX_NATIVE_INSTANTANEOUS_ION_INCREMENT_H_
#include <ablastr/fields/MultiFabRegister.H>
class WarpX;
namespace warpx::particles {
/** Pure common-support finite current increment from saved native old phase,
 * final actual-force Boris kick and precise completed orbit chord. Base is the
 * existing same-Esirkepov point map. Delta describes the correlated finite
 * increment, not a subtraction of rounded virtual endpoint states. No primary
 * current, particle, history, redistribution or wall action is modified.
 * Missing/invalid record, support crossing or unsupported orbit clears both
 * caller outputs and returns false. Caller separately measures endpoint error.
 */
bool DepositNativeInstantaneousIonIncrement(WarpX&,
    ablastr::fields::VectorField const& base,
    ablastr::fields::VectorField const& delta);
/** Explicit pure adjacent-cell prototype using the same native old phase,
 * completed chord and finite kick. Old symbol/guard remain unchanged.
 * One cell per active direction, both endpoints outside the old knot margin,
 * full old/new support union. No retained/source/event caller selects this.
 * Unsupported input clears both outputs collectively, like the old API.
 */
bool DepositNativeAdjacentInstantaneousIonIncrement(WarpX&,
    ablastr::fields::VectorField const& base,
    ablastr::fields::VectorField const& delta);
}
#endif
