/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "ExpectedIonExchangeChannel.H"
#include "Utils/WarpXConst.H"
#include <AMReX_MFIter.H>
#include <ablastr/coarsen/sample.H>
#include <cmath>
namespace warpx::thermal {
bool
BuildExpectedIonDrift (const amrex::Geometry& g, const amrex::MultiFab& rho,
                       const amrex::MultiFab* pedestal,
                       const std::array<const amrex::MultiFab*, 3>& ji,
                       const std::array<const amrex::MultiFab*, 3>& jp,
                       amrex::Real floor,
                       const std::array<amrex::MultiFab*, 3>& drift) {
    amrex::GpuArray<int, 3> nodes{}, one{1, 1, 1};
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
        nodes[d] = 1;
    amrex::Real const rf = PhysConst::q_e * floor;
    for (int d = 0; d < 3; ++d) {
        AMREX_ALWAYS_ASSERT(drift[d] && ji[d] && jp[d]);
        AMREX_ALWAYS_ASSERT(drift[d]->boxArray() == rho.boxArray() &&
                            drift[d]->DistributionMap() ==
                                rho.DistributionMap());
        AMREX_ALWAYS_ASSERT(ji[d]->boxArray() == jp[d]->boxArray() &&
                            ji[d]->nGrowVect().allGE(amrex::IntVect(1)) &&
                            jp[d]->nGrowVect().allGE(amrex::IntVect(1)));
        amrex::GpuArray<int, 3> stagger{};
        for (int k = 0; k < AMREX_SPACEDIM; ++k)
            stagger[k] = ji[d]->ixType()[k];
        drift[d]->setVal(0.);
        for (amrex::MFIter mfi(*drift[d], amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            auto out = drift[d]->array(mfi);
            auto r = rho.const_array(mfi), j = jp[d]->const_array(mfi),
                 i = ji[d]->const_array(mfi);
            auto p = pedestal ? pedestal->const_array(mfi)
                              : amrex::Array4<const amrex::Real>{};
            bool const ped = pedestal != nullptr;
            amrex::ParallelFor(
                mfi.tilebox(), [=] AMREX_GPU_DEVICE(int x, int y, int z) {
                    auto const raw = r(x, y, z);
                    if (!ped && raw <= rf)
                        return;
                    auto const capacity =
                        ped ? amrex::max(raw + p(x, y, z), rf) : raw;
                    auto const jn = ablastr::coarsen::sample::Interp(
                        j, stagger, nodes, one, x, y, z, 0);
                    auto const in = ablastr::coarsen::sample::Interp(
                        i, stagger, nodes, one, x, y, z, 0);
                    out(x, y, z) = -(jn - in) / capacity;
                });
        }
        drift[d]->FillBoundary(g.periodicity());
    }
    bool ok = true;
    for (auto* d : drift) {
        bool const finite = d->is_finite(0, 1, 0);
        ok = ok && finite;
    }
    return ok;
}
} // namespace warpx::thermal
