/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "ThermalStageInitialGuess.H"
#include <AMReX_MFIter.H>
namespace warpx::thermal {
bool
MakeThermalStageInitialGuess (amrex::MultiFab& output,
                              const amrex::MultiFab& old,
                              const amrex::MultiFab& seed,
                              const amrex::MultiFab& ns,
                              const amrex::MultiFab& ne,
                              ThermalStageGuessOptions options) {
    AMREX_ALWAYS_ASSERT(&output != &old && &output != &ns && &output != &ne);
    for (auto const* f : {&old, &seed, &ns, &ne,
                          static_cast<const amrex::MultiFab*>(&output)}) {
        AMREX_ALWAYS_ASSERT(f->boxArray() == old.boxArray() &&
                            f->DistributionMap() == old.DistributionMap() &&
                            f->nComp() == 1 && f->ixType().cellCentered() &&
                            !f->hasEBFabFactory());
    }
    amrex::MultiFab candidate(old.boxArray(), old.DistributionMap(), 1, 0);
    for (amrex::MFIter mfi(candidate, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
        auto const un = old.const_array(mfi), u = seed.const_array(mfi),
                   nst = ns.const_array(mfi), nen = ne.const_array(mfi);
        auto const out = candidate.array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                               int k) {
            auto const result = PrepareThermalStageGuess(
                un(i, j, k), u(i, j, k), nst(i, j, k), nen(i, j, k), options);
            out(i, j, k) = result.valid
                               ? result.energy
                               : std::numeric_limits<amrex::Real>::quiet_NaN();
        });
    }
    if (candidate.contains_nan() || candidate.contains_inf()) {
        return false;
    }
    amrex::MultiFab::Copy(output, candidate, 0, 0, 1, 0);
    return true;
}
} // namespace warpx::thermal
