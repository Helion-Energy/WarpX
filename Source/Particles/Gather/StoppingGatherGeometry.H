/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#ifndef WARPX_STOPPING_GATHER_GEOMETRY_H_
#define WARPX_STOPPING_GATHER_GEOMETRY_H_
#include <AMReX_Box.H>

namespace warpx::particles {
/** Integer-origin translation for the native stopping field gather.
 * Compute_shape_factor uses integer truncation. A particle in the first half
 * cell of the original FAB would have a negative cell-centered coordinate.
 * Growing only the coordinate box makes every admitted coordinate positive;
 * the lower index shifts by the same integer, preserving the global stencil.
 * No field allocation, position, staggering or shape algorithm is changed.
 */
inline amrex::Box StoppingGatherBox (amrex::Box const& valid_cell_fab)
{
    AMREX_ALWAYS_ASSERT(valid_cell_fab.ixType().cellCentered());
    return amrex::grow(valid_cell_fab, 1);
}
} // namespace warpx::particles
#endif
