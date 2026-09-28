/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_INSTANTANEOUS_FORCE_H_
#define WARPX_NATIVE_INSTANTANEOUS_FORCE_H_

#include "EulerianDissipation.H"
#include "ablastr/fields/MultiFabRegister.H"
#include <memory>
class WarpX;

namespace warpx::darwin {
/** Private dissipation with the exact live model parsers, floors and BCs.
 * No accepted thermal state, particle attributes, RNG or tallies are changed.
 * Parser owners must outlive the returned scratch. RZ source scope only.
 */
std::unique_ptr<thermal::EulerianDissipation> MakeNativeInstantaneousDissipation(WarpX& sim);

/** Capture the native full/nores Ohm pair at the caller's instantaneous state.
 * The caller must install the same Ji, Jp, B, rho, borrowed Te/Pe and inertia
 * that its force uses. This routine calls the level wrapper on registered
 * Efield_fp: that wrapper's electric BC acts on the registry, not its argument.
 * Entry E is restored exactly, including guards, before return. Outputs must
 * be distinct caller-owned scratch with the exact native E layout/guards.
 * Conductor/transformed and per-species closures are rejected. No callbacks,
 * particle/RNG operations or accepted-state publication occur. False means
 * nonfinite scratch; discard both outputs. Dissipation scratch is overwritten.
 */
bool CaptureNativeInstantaneousForce(
    WarpX& sim, const amrex::MultiFab& charge,
    const ablastr::fields::VectorField& full,
    const ablastr::fields::VectorField& nores,
    thermal::EulerianDissipation* dissipation);

/** Native subtraction/addition order P=(E-F)+N, on matching caller scratch.
 * Disabled is an exact no-op. The caller performs the actual native field
 * guard exchange, electric projection and auxiliary centering AFTER this map.
 * F/N stay fixed in a frozen-state electric solve; rebuild for each outer state.
 */
void ApplyNativeInstantaneousForce(
    const ablastr::fields::VectorField& electric,
    const ablastr::fields::VectorField& full,
    const ablastr::fields::VectorField& nores, bool enabled);
}
#endif
