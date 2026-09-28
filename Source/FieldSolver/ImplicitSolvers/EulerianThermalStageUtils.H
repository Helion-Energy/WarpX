/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_EULERIAN_THERMAL_STAGE_UTILS_H_
#define WARPX_EULERIAN_THERMAL_STAGE_UTILS_H_
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>
namespace warpx::thermal::stage_detail {
using amrex::Real;
inline bool
finite (const amrex::MultiFab& x) {
    return !x.contains_nan(0, x.nComp(), 0) && !x.contains_inf(0, x.nComp(), 0);
}
inline void
check_layout (const amrex::MultiFab& x, const amrex::MultiFab& base,
              int components) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        x.is_cell_centered() && x.nComp() == components &&
            !x.hasEBFabFactory() && x.boxArray() == base.boxArray() &&
            x.DistributionMap() == base.DistributionMap(),
        "EulerianThermalStage requires matching cell-centred, non-EB "
        "single-level data");
}
inline void
fill_images (amrex::MultiFab& data, const amrex::Geometry& geometry) {
    data.FillBoundary(geometry.periodicity());
    auto const lo = geometry.Domain().smallEnd();
    auto const hi = geometry.Domain().bigEnd();
    auto const periodic = geometry.isPeriodicArray();
    // Physical images are insulated specific-energy extensions. Reservoir and
    // prescribed-flux laws are imposed directly ON faces, never by a ghost U
    // clamp. FD4 downgrades beside physical boundaries in the shared kernel.
    for (amrex::MFIter mfi(data); mfi.isValid(); ++mfi) {
        auto const a = data.array(mfi);
        amrex::ParallelFor(
            mfi.fabbox(), data.nComp(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                int src[3] = {i, j, k};
                bool image = false;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    if (!periodic[d] && (src[d] < lo[d] || src[d] > hi[d])) {
                        src[d] = src[d] < lo[d] ? 2 * lo[d] - src[d] - 1
                                                : 2 * hi[d] - src[d] + 1;
                        src[d] = amrex::max(lo[d], amrex::min(hi[d], src[d]));
                        image = true;
                    }
                }
                if (image) {
                    a(i, j, k, n) = a(src[0], src[1], src[2], n);
                }
            });
    }
}

} // namespace warpx::thermal::stage_detail
#endif
