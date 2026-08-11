#include "RhoFunctor.H"

#include "Diagnostics/ComputeDiagFunctors/ComputeDiagFunctor.H"
#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
    #include "FieldSolver/SpectralSolver/SpectralFieldData.H"
    #include "FieldSolver/SpectralSolver/SpectralSolverRZ.H"
#endif
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/EBJBoundary.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "Fluids/MultiFluidContainer.H"
#include "Fluids/WarpXFluidContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include <memory>

RhoFunctor::RhoFunctor (const int lev,
                        const amrex::IntVect crse_ratio,
                        bool apply_rz_psatd_filter,
                        const int species_index,
                        bool convertRZmodes2cartesian,
                        const int ncomp)
    : ComputeDiagFunctor(ncomp, crse_ratio),
      m_lev(lev),
      m_apply_rz_psatd_filter(apply_rz_psatd_filter),
      m_species_index(species_index),
      m_convertRZmodes2cartesian(convertRZmodes2cartesian)
{}

void
RhoFunctor::operator() ( amrex::MultiFab& mf_dst, const int dcomp, const int /*i_buffer*/ ) const
{
    auto& warpx = WarpX::GetInstance();
    std::unique_ptr<amrex::MultiFab> rho;
    bool fresh_deposit = false;

    // Total rho with the hybrid-PIC solver and mesh refinement: use the
    // solver's synchronized rho_fp instead of a fresh per-level deposit. The
    // per-level deposit only sees the particles owned by this level, so it
    // misses every cross-level contribution: the coarse level is empty under
    // a fine patch (its particles live on the fine level), and the fine
    // level's outermost cells lose the shape support of particles just
    // outside the patch (a factor ~1/2 at faces and ~1/4 at corners) -- a
    // depressed band at the seam that exists only in the diagnostic, not in
    // the solver state. rho_fp holds rho^{n+1} deposited at the same time,
    // already SyncRho'd (fine restricted onto the coarse level), filtered,
    // boundary-applied, coarse-fine ghost/band filled, and including any
    // fluid species contribution.
    if (m_species_index == -1 &&
        WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC &&
        warpx.finestLevel() > 0)
    {
        const amrex::MultiFab& rho_fp = *warpx.m_fields.get(warpx::fields::FieldType::rho_fp, m_lev);
        rho = std::make_unique<amrex::MultiFab>(
            rho_fp.boxArray(), rho_fp.DistributionMap(), rho_fp.nComp(), rho_fp.nGrowVect());
        amrex::MultiFab::Copy(*rho, rho_fp, 0, 0, rho_fp.nComp(), rho_fp.nGrowVect());
        // No ApplyFilterandSumBoundaryRho here: rho_fp is already filtered
        // and summed by SyncRho.
    }
    else
    {
        fresh_deposit = true;
        // Deposit charge density
        // Call this with local=true since the parallel transfers will be handled
        // by ApplyFilterandSumBoundaryRho

        // Dump total rho
        if (m_species_index == -1) {
            auto& mypc = warpx.GetPartContainer();
            rho = mypc.GetChargeDensity(m_lev, true);
            if (warpx.DoFluidSpecies()) {
                auto& myfl = warpx.GetFluidContainer();
                myfl.DepositCharge(warpx.m_fields, *rho, m_lev);
            }
        }
        // Dump rho per species
        else {
            auto& mypc = warpx.GetPartContainer().GetParticleContainer(m_species_index);
            rho = mypc.GetChargeDensity(m_lev, true);
        }

        // Handle the parallel transfers of guard cells and
        // apply the filtering if requested.
        warpx.ApplyFilterandSumBoundaryRho(m_lev, m_lev, *rho, 0, rho->nComp());
    }

#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
    // Apply k-space filtering when using the PSATD solver
    if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD)
    {
        if (WarpX::use_kspace_filter && m_apply_rz_psatd_filter) {
            auto & solver = warpx.get_spectral_solver_fp(m_lev);
            const SpectralFieldIndex& Idx = solver.m_spectral_index;
            solver.ForwardTransform(m_lev, *rho, Idx.rho_new);
            solver.ApplyFilter(m_lev, Idx.rho_new);
            solver.BackwardTransform(m_lev, *rho, Idx.rho_new);
        }
    }
#else
    amrex::ignore_unused(m_apply_rz_psatd_filter);
#endif

    // For the hybrid solver, fold the deposit collected by covered points
    // back across the embedded surface and enforce the embedded-boundary
    // Dirichlet condition on the freshly deposited charge density (the
    // density vanishes at a conducting wall), after the filter and
    // guard-cell sum so the mirrored values are not smeared into the
    // conductor, matching the solver's own rho treatment in
    // HybridPICDepositRhoAndJ. Skipped on the rho_fp copy path above: the
    // solver already applied exactly this treatment to rho_fp.
    if (fresh_deposit && EB::enabled() &&
        WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC)
    {
        warpx::hybrid::FoldEBDepositToNodalScalar(
            *rho,
            *warpx.m_fields.get(warpx::fields::FieldType::distance_to_eb, m_lev),
            warpx.Geom(m_lev));
        warpx::hybrid::ApplyEBBoundaryToNodalScalar(
            *rho,
            *warpx.m_fields.get(warpx::fields::FieldType::distance_to_eb, m_lev),
            warpx.Geom(m_lev),
            /*odd=*/true);  // Dirichlet: rho -> 0 at the PEC wall
    }

    InterpolateMFForDiag(mf_dst, *rho, dcomp, warpx.DistributionMap(m_lev),
                         m_convertRZmodes2cartesian);
}
