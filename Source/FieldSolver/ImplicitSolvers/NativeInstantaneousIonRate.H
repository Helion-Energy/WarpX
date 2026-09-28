/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_INSTANTANEOUS_ION_RATE_H_
#define WARPX_NATIVE_INSTANTANEOUS_ION_RATE_H_
#include <ablastr/fields/MultiFabRegister.H>
class WarpX;
namespace warpx::particles {
/** Read-only particle rate used by the native constrained endpoint response.
 * Inputs are the already assembled native centered MC auxiliary fields. Outputs
 * are separate caller-owned native Yee current/rate arrays including deposition
 * guards. Units are A/m^2 and A/(m^2 s). No theta*dt or residual scale is applied.
 * linear_only keeps electric acceleration response at the same stored orbit;
 * the full call includes native magnetic/convective/geometric rate terms.
 * This bounded adapter shares the existing shape-3, m0 kernel and all its force
 * capability guards. It never replaces the primary Esirkepov interval deposit.
 */
void EvaluateNativeInstantaneousIonRate(
    WarpX&, ablastr::fields::VectorField const& electric_aux,
    ablastr::fields::VectorField const& magnetic_aux,
    ablastr::fields::VectorField const& current,
    ablastr::fields::VectorField const& rate, bool linear_only);
}
#endif
