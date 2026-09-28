/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL. */
#ifndef WARPX_NATIVE_BOUNDARY_FIELD_RATE_H_
#define WARPX_NATIVE_BOUNDARY_FIELD_RATE_H_
#include <ablastr/fields/MultiFabRegister.H>
class WarpX;
namespace warpx::darwin {
/** Exact linear derivative of the native A boundary and native double curl.
 * All inputs include their allocated guards. Output arrays are caller-owned,
 * pairwise distinct and disjoint from input/reference. No registered field,
 * external scale, circuit state or particle is modified. Geometry is fixed.
 * The imposed reference is an A rate, not a full-domain external E addition.
 */
void NativeBoundaryFieldRate(WarpX&,
    ablastr::fields::VectorField const& electric,
    ablastr::fields::VectorField const& reference_rate,
    ablastr::fields::VectorField out_potential_rate,
    ablastr::fields::VectorField out_magnetic_rate,
    ablastr::fields::VectorField out_current_rate);
}
#endif
