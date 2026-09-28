/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_HYBRID_OHM_DAMPING_FIELDS_H_
#define WARPX_HYBRID_OHM_DAMPING_FIELDS_H_
#include "ablastr/fields/MultiFabRegister.H"
namespace warpx::thermal {
/** Optional RAW same-kernel numerical end components [V/m]. Both vectors
 * require independent native E layouts/guards. The kernel clears all scratch.
 * End resistivity is captured only when present in the requested solve.
 * End Holmstrom is the signed AFTER-minus-BEFORE change of the geometric
 * drive multiplication, before inertia and resistivity are added. It is not
 * a nonnegative dissipation. Caller owns homogeneous BC projection; these
 * components must never be silently added to physical electron heating.
 */
struct HybridOhmDampingFields {
    ablastr::fields::VectorField end_resistivity;
    ablastr::fields::VectorField end_holmstrom;
};
}
#endif
