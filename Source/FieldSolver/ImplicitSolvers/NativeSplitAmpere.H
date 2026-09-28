/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_SPLIT_AMPERE_H_
#define WARPX_NATIVE_SPLIT_AMPERE_H_
#include "ablastr/fields/MultiFabRegister.H"
class WarpX;
namespace warpx::darwin {
/** Pure value-current map C(B_static)+C(curl(A)), using the native curl and
 * double-precision boundary exchange. A already has its native boundary images.
 * No B projection or change to the magnetic force field occurs. Inputs are
 * immutable. Only valid/current grow1 rows are overwritten, exactly as native
 * CalculateCurrentAmpere; outer output guards retain their entry values.
 * This is a VALUE map: never use it for a rate/directional B argument.
 */
void CalculateNativeSplitAmpere(WarpX& sim,
    const ablastr::fields::VectorField& conditioned_A,
    const ablastr::fields::VectorField& static_B,
    const ablastr::fields::VectorField& current);

/** Explicit opt-in value-current callsite. Returns false without field writes
 * when disabled or the model has no configured resistive/viscous force term.
 * On true, refreshes the registered plasma current from registered A/B_static.
 * The caller applies the existing displacement current afterwards. Restricted
 * to the guarded smooth-source compatible-Yee RZ prototype; rate routes remain
 * independent and source-free paths keep their original arithmetic.
 */
bool TryCalculateNativeSplitDarwinAmpere(WarpX& sim);
}
#endif
