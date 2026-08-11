/* Copyright 2019 Andrew Myers, Ann Almgren, Axel Huebl
 * David Grote, Maxence Thevenet, Michael Rowan
 * Remi Lehe, Weiqun Zhang, levinem, Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#include "Diagnostics/MultiDiagnostics.H"
#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
#include "EmbeddedBoundary/Enabled.H"
#include "EmbeddedBoundary/WarpXFaceInfoBox.H"
#include "Fields.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/ExternalField.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/ParticleBoundaryBuffer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/profiler/ProfilerWrapper.H>

#include <AMReX.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FabFactory.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_IndexType.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MFIter.H>
#include <AMReX_MakeType.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParallelContext.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

using namespace amrex;

void
WarpX::CheckLoadBalance (int step)
{
    if (step > 0 && load_balance_intervals.contains(step+1))
    {
        LoadBalance();

        // Reset the costs to 0
        ResetCosts();
    }
    if (!costs.empty())
    {
        RescaleCosts(step);
    }
}

void
WarpX::LoadBalance ()
{
    ABLASTR_PROFILE_REGION("LoadBalance");
    ABLASTR_PROFILE("WarpX::LoadBalance()");

    AMREX_ALWAYS_ASSERT(!costs.empty());
    AMREX_ALWAYS_ASSERT(costs[0] != nullptr);

#ifdef AMREX_USE_MPI
    if (load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Heuristic)
    {
        // compute the costs on a per-rank basis
        ComputeCostsHeuristic(costs);
    }

    // By default, do not do a redistribute; this toggles to true if RemakeLevel
    // is called for any level
    int loadBalancedAnyLevel = false;

    const int nLevels = finestLevel();
    for (int lev = 0; lev <= nLevels; ++lev)
    {
        int doLoadBalance = false;

        // Compute the new distribution mapping
        DistributionMapping newdm;
        const amrex::Real nboxes = costs[lev]->size();
        const amrex::Real nprocs = ParallelContext::NProcsSub();
        const int nmax = static_cast<int>(std::ceil(nboxes/nprocs*load_balance_knapsack_factor));
        // These store efficiency (meaning, the  average 'cost' over all ranks,
        // normalized to max cost) for current and proposed distribution mappings
        amrex::Real currentEfficiency = 0.0;
        amrex::Real proposedEfficiency = 0.0;

        newdm = (load_balance_with_sfc)
            ? DistributionMapping::makeSFC(*costs[lev],
                                           currentEfficiency, proposedEfficiency,
                                           false,
                                           ParallelDescriptor::IOProcessorNumber())
            : DistributionMapping::makeKnapSack(*costs[lev],
                                                currentEfficiency, proposedEfficiency,
                                                nmax,
                                                false,
                                                ParallelDescriptor::IOProcessorNumber());
        // As specified in the above calls to makeSFC and makeKnapSack, the new
        // distribution mapping is NOT communicated to all ranks; the loadbalanced
        // dm is up-to-date only on root, and we can decide whether to broadcast
        if ((load_balance_efficiency_ratio_threshold > 0.0)
            && (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber()))
        {
            doLoadBalance = (proposedEfficiency > load_balance_efficiency_ratio_threshold*currentEfficiency);
        }

        ParallelDescriptor::Bcast(&doLoadBalance, 1,
                                  ParallelDescriptor::IOProcessorNumber());

        amrex::Print() << Utils::TextMsg::Info("current LB efficiency = " + std::to_string(currentEfficiency)
                          + " proposed LB efficiency = " + std::to_string(proposedEfficiency)
                          + " LoadBalance is set to : " + std::to_string(doLoadBalance) );

        if (doLoadBalance)
        {
            Vector<int> pmap;
            if (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber())
            {
                pmap = newdm.ProcessorMap();
            } else
            {
                pmap.resize(static_cast<std::size_t>(nboxes));
            }
            ParallelDescriptor::Bcast(pmap.data(), pmap.size(), ParallelDescriptor::IOProcessorNumber());

            if (ParallelDescriptor::MyProc() != ParallelDescriptor::IOProcessorNumber())
            {
                newdm = DistributionMapping(pmap);
            }

            RemakeLevel(lev, t_new[lev], boxArray(lev), newdm);

            // Record the load balance efficiency
            setLoadBalanceEfficiency(lev, proposedEfficiency);
        }

        loadBalancedAnyLevel = loadBalancedAnyLevel || doLoadBalance;
    }
    if (loadBalancedAnyLevel)
    {
        mypc->Redistribute();
        mypc->defineAllParticleTiles();

        // redistribute particle boundary buffer
        m_particle_boundary_buffer->redistribute();

        // diagnostics & reduced diagnostics
        // not yet needed:
        //multi_diags->LoadBalance();
        reduced_diags->LoadBalance();
    }
#endif
}

void
WarpX::RemakeLevel (int lev, Real /*time*/, const BoxArray& ba, const DistributionMapping& dm)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    const auto RemakeMultiFab = [&](auto& mf){
        if (mf == nullptr) { return; }
        const IntVect& ng = mf->nGrowVect();
        auto pmf = std::remove_reference_t<decltype(mf)>{};
        AllocInitMultiFab(pmf, mf->boxArray(), dm, mf->nComp(), ng, lev, mf->tags()[0]);
        *mf = std::move(*pmf);
    };

    bool const eb_enabled = EB::enabled();
    if (ba == boxArray(lev))
    {
        if (ParallelDescriptor::NProcs() == 1) { return; }

        m_fields.remake_level(lev, dm);

        // Fine patch
        ablastr::fields::MultiLevelVectorField const& Bfield_fp = m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level);
        for (int idim=0; idim < 3; ++idim)
        {
            if (eb_enabled) {
                RemakeMultiFab( m_eb_reduce_particle_shape[lev] );
                if (WarpX::electromagnetic_solver_id != ElectromagneticSolverAlgo::PSATD) {
                    RemakeMultiFab( m_eb_update_E[lev][idim] );
                    RemakeMultiFab( m_eb_update_B[lev][idim] );
                    if (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::ECT) {
                        m_borrowing[lev][idim] = std::make_unique<amrex::LayoutData<FaceInfoBox>>(amrex::convert(ba, Bfield_fp[lev][idim]->ixType().toIntVect()), dm);
                    }
                }
            }
        }

        if (eb_enabled) {
#ifdef AMREX_USE_EB
            int const max_guard = guard_cells.ng_FieldSolver.max();
            m_field_factory[lev] = amrex::makeEBFabFactory(Geom(lev), ba, dm,
                                                           {max_guard, max_guard, max_guard},
                                                           amrex::EBSupport::full);
#endif
            InitializeEBGridData(lev);
            if (electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC
                && max_level > 0) {
                // Re-run the EB clearance guard after the level remake: a
                // regrid that moves a fine patch into the wall band must
                // fail loudly (the check is cheap; verbose at init only).
                m_hybrid_pic_model->CheckMREBClearance(false);
            }
        } else {
            m_field_factory[lev] = std::make_unique<FArrayBoxFactory>();
        }

#ifdef WARPX_USE_FFT
        if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
            if (spectral_solver_fp[lev] != nullptr) {
                // Get the cell-centered box
                BoxArray realspace_ba = ba;   // Copy box
                realspace_ba.enclosedCells(); // Make it cell-centered
                auto ngEB = getngEB();
                auto dx = CellSize(lev);

#   ifdef WARPX_DIM_RZ
                if ( !fft_periodic_single_box ) {
                    realspace_ba.grow(1, ngEB[1]); // add guard cells only in z
                }
                if (field_boundary_hi[0] == FieldBoundaryType::PML && !do_pml_in_domain) {
                    // Extend region that is solved for to include the guard cells
                    // which is where the PML boundary is applied.
                    realspace_ba.growHi(0, pml_ncell);
                }
                AllocLevelSpectralSolverRZ(spectral_solver_fp,
                                           lev,
                                           realspace_ba,
                                           dm,
                                           dx);
#   else
                if ( !fft_periodic_single_box ) {
                    realspace_ba.grow(ngEB);   // add guard cells
                }
                bool const pml_flag_false = false;
                AllocLevelSpectralSolver(spectral_solver_fp,
                                         lev,
                                         realspace_ba,
                                         dm,
                                         dx,
                                         pml_flag_false);
#   endif
            }
        }
#endif

        // Coarse patch
        if (lev > 0) {

#ifdef WARPX_USE_FFT
            if (electromagnetic_solver_id == ElectromagneticSolverAlgo::PSATD) {
                if (spectral_solver_cp[lev] != nullptr) {
                    BoxArray cba = ba;
                    cba.coarsen(refRatio(lev-1));
                    const std::array<Real,3> cdx = CellSize(lev-1);

                    // Get the cell-centered box
                    BoxArray c_realspace_ba = cba;  // Copy box
                    c_realspace_ba.enclosedCells(); // Make it cell-centered

                    auto ngEB = getngEB();

#   ifdef WARPX_DIM_RZ
                    c_realspace_ba.grow(1, ngEB[1]); // add guard cells only in z
                    if (field_boundary_hi[0] == FieldBoundaryType::PML && !do_pml_in_domain) {
                        // Extend region that is solved for to include the guard cells
                        // which is where the PML boundary is applied.
                        c_realspace_ba.growHi(0, pml_ncell);
                    }
                    AllocLevelSpectralSolverRZ(spectral_solver_cp,
                                               lev,
                                               c_realspace_ba,
                                               dm,
                                               cdx);
#   else
                    c_realspace_ba.grow(ngEB);
                    bool const pml_flag_false = false;
                    AllocLevelSpectralSolver(spectral_solver_cp,
                                             lev,
                                             c_realspace_ba,
                                             dm,
                                             cdx,
                                             pml_flag_false);
#   endif
                }
            }
#endif
        }

        // Re-initialize the lattice element finder with the new ba and dm.
        m_accelerator_lattice[lev]->InitElementFinder(lev, gamma_boost, gett_new(), ba, dm);

        if (costs[lev] != nullptr)
        {
            costs[lev] = std::make_unique<LayoutData<Real>>(ba, dm);
            const auto iarr = costs[lev]->IndexArray();
            for (const auto& i : iarr)
            {
                (*costs[lev])[i] = 0.0;
                setLoadBalanceEfficiency(lev, -1);
            }
        }

        SetDistributionMap(lev, dm);

        if (lev > 0 && (n_field_gather_buffer > 0 || n_current_deposition_buffer > 0)) {
            if (current_buffer_masks[lev] || gather_buffer_masks[lev]) {
                if (current_buffer_masks[lev]) {
                    RemakeMultiFab( current_buffer_masks[lev] );
                }
                if (gather_buffer_masks[lev]) {
                    RemakeMultiFab( gather_buffer_masks[lev] );
                }
                BuildBufferMasks();
            }
        }

    } else
    {
        // Changed BoxArray (relocation of a refined patch): only reachable
        // through AmrCore::regrid, which WarpX only invokes from
        // HybridPICRegrid (hybrid-PIC dynamic mesh refinement). The EM/ES
        // solvers are static-MR and only support the same-BoxArray load
        // balancing path above.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC &&
            m_hybrid_regrid_in_progress && lev > 0,
            "RemakeLevel with a changed BoxArray is only implemented for "
            "refined levels of the hybrid-PIC solver, through "
            "warpx.regrid_int (the EM/ES solvers are static-MR).");

        // Stash the old-layout B solution: after the level is re-created on
        // the new layout, HybridPICRegrid seeds B by div-free prolongation
        // from the coarse level and copies this stash back over the overlap.
        // Everything else is recomputed (E, electron pressure) or
        // re-deposited and re-seeded (moments and their history fabs).
        auto& stash = m_regrid_stashed_B[lev];
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab* B = m_fields.get(FieldType::Bfield_fp, Direction{idim}, lev);
            stash[idim] = std::make_unique<MultiFab>(std::move(*B));
        }

        // Tear the level down and re-create it on the new layout, exactly
        // like a level created from scratch. AmrCore::regrid publishes the
        // new BoxArray/DistributionMapping after this hook returns.
        ClearLevelData(lev);
        AllocLevelData(lev, ba, dm);

#ifdef AMREX_USE_EB
        // Same as MakeNewLevelFromCoarse: fill the freshly allocated EB
        // solver masks of the re-created level (the load-balance branch
        // above does this through its own InitializeEBGridData call).
        if (eb_enabled) { InitializeEBGridData(lev); }
#endif

        m_regrid_relocated_levels.push_back(lev);
    }

    // Re-initialize diagnostic functors that stores pointers to the user-requested fields at level, lev.
    multi_diags->InitializeFieldFunctors( lev );

    // Reduced diagnostics
    // not needed yet
}

void
WarpX::ComputeCostsHeuristic (amrex::Vector<std::unique_ptr<amrex::LayoutData<amrex::Real> > >& a_costs)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        const auto & mypc_ref = GetPartContainer();
        const auto nSpecies = mypc_ref.nSpecies();

        // Species loop
        for (int i_s = 0; i_s < nSpecies; ++i_s)
        {
            auto & myspc = mypc_ref.GetParticleContainer(i_s);

            // Particle loop
            for (WarpXParIter pti(myspc, lev); pti.isValid(); ++pti)
            {
                (*a_costs[lev])[pti.index()] += costs_heuristic_particles_wt*pti.numParticles();
            }
        }

        // Cell loop
        MultiFab* Ex = m_fields.get(FieldType::Efield_fp, Direction{0}, lev);
        for (MFIter mfi(*Ex, false); mfi.isValid(); ++mfi)
        {
            const Box& gbx = mfi.growntilebox();
            (*a_costs[lev])[mfi.index()] += costs_heuristic_cells_wt*gbx.numPts();
        }
    }
}

void
WarpX::ResetCosts ()
{
    AMREX_ALWAYS_ASSERT(!costs.empty());
    AMREX_ALWAYS_ASSERT(costs[0] != nullptr);

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        const auto iarr = costs[lev]->IndexArray();
        for (const auto& i : iarr)
        {
            // Reset costs
            (*costs[lev])[i] = 0.0;
        }
    }
}

void
WarpX::RescaleCosts (int step)
{
    // rescale is only used for timers
    if (WarpX::load_balance_costs_update_algo != LoadBalanceCostsUpdateAlgo::Timers)
    {
        return;
    }

    // costs is sized max_level+1; with hybrid-PIC dynamic mesh refinement
    // finest_level can be smaller than max_level (deferred/removed levels).
    AMREX_ALWAYS_ASSERT(costs.size() > finest_level);

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (costs[lev])
        {
            // Perform running average of the costs
            // (Giving more importance to most recent costs; only needed
            // for timers update, heuristic load balance considers the
            // instantaneous costs)
            for (const auto& i : costs[lev]->IndexArray())
            {
                (*costs[lev])[i] *= (1._rt - 2._rt/load_balance_intervals.localPeriod(step+1));
            }
        }
    }
}

void
WarpX::HybridPICRegrid (int step, amrex::Real time)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC,
        "HybridPICRegrid: dynamic regridding is only implemented for the "
        "hybrid-PIC solver.");

    ABLASTR_PROFILE("WarpX::HybridPICRegrid()");

    const int old_finest = finest_level;

    m_regrid_created_levels.clear();
    m_regrid_relocated_levels.clear();
    m_regrid_stashed_B.clear();

    // Re-evaluate the refinement tags (ErrorEst at the current time) and let
    // AmrCore::regrid drive the MakeNewLevelFromCoarse / RemakeLevel /
    // ClearLevel hooks. Identical tags produce identical grids and the call
    // is a no-op.
    m_hybrid_regrid_in_progress = true;
    regrid(0, time);
    m_hybrid_regrid_in_progress = false;

    const bool level_created   = !m_regrid_created_levels.empty();
    const bool level_relocated = !m_regrid_relocated_levels.empty();
    const bool level_removed   = (finest_level < old_finest);

    if (!level_created && !level_relocated && !level_removed) {
        return;
    }

    {
        std::ostringstream oss;
        oss << "Hybrid-PIC regrid at step " << step
            << " (t = " << std::scientific << std::setprecision(6) << time
            << " s): finest_level " << old_finest << " -> " << finest_level
            << (level_created   ? "; level(s) created"   : "")
            << (level_relocated ? "; level(s) relocated" : "")
            << (level_removed   ? "; level(s) removed"   : "");
        amrex::Print() << Utils::TextMsg::Info(oss.str());
    }

    // Any hierarchy change invalidates the cached coarse-fine masks (they
    // are keyed to the layouts they were built from and rebuilt on demand).
    m_hybrid_pic_model->ClearMRMaskCache();

    // Re-run the EB clearance guard on the new hierarchy: a regrid that
    // creates or relocates a fine patch into the wall band must fail loudly
    // (no-op unless EB is enabled and a fine level exists).
    m_hybrid_pic_model->CheckMREBClearance(false);

    // Seed the fields of created/relocated levels, coarsest first, so a
    // multi-level cascade always prolongs from an already-seeded parent.
    for (int lev = 1; lev <= finest_level; ++lev)
    {
        const bool is_created = std::find(
            m_regrid_created_levels.begin(), m_regrid_created_levels.end(), lev)
            != m_regrid_created_levels.end();
        const bool is_relocated = std::find(
            m_regrid_relocated_levels.begin(), m_regrid_relocated_levels.end(), lev)
            != m_regrid_relocated_levels.end();
        if (!is_created && !is_relocated) { continue; }

        // B: divergence-free prolongation of the coarse solution over the
        // whole level (div(B) = 0 is inherited)...
        m_hybrid_pic_model->SeedBfieldFromCoarse(
            lev, guard_cells.ng_FieldSolver, WarpX::sync_nodal_points);

        if (is_relocated) {
            // ... overwritten with the old-layout fine solution wherever the
            // new layout overlaps it (valid region to valid region).
            auto& stash = m_regrid_stashed_B[lev];
            for (int idim = 0; idim < 3; ++idim) {
                MultiFab* B = m_fields.get(FieldType::Bfield_fp, Direction{idim}, lev);
                B->ParallelCopy(*stash[idim], 0, 0, 1,
                                IntVect(0), IntVect(0), Geom(lev).periodicity());
            }
            // Restore same-level ghosts around the copied overlap; the
            // coarse-fine ghosts keep the prolonged values.
            FillBoundaryB(lev, guard_cells.ng_FieldSolver, WarpX::sync_nodal_points);
        }

        // E: staggering-aware placeholder interpolation (recomputed from the
        // moments and B in the first Ohm's-law solve of this step).
        m_hybrid_pic_model->SeedEfieldFromCoarse(lev);

        // External sources resident on the level (external current, external
        // vector potential A and curl(A), electron-temperature fill value).
        m_hybrid_pic_model->ReinitLevelData(lev);
    }
    m_regrid_stashed_B.clear();

    // Particles move to their new home levels by position (weights are
    // unchanged; on removal they drop to the coarse level automatically).
    mypc->Redistribute();
    mypc->defineAllParticleTiles();
    m_particle_boundary_buffer->redistribute();

    // Gather/deposition buffer masks for the new hierarchy (the masks of
    // created/relocated levels were freshly allocated with the level).
    if (n_field_gather_buffer > 0 || n_current_deposition_buffer > 0) {
        BuildBufferMasks();
    }

    // Diagnostics follow the new hierarchy: level counts, per-level output
    // buffers, and field functors (BackTransformed diagnostics abort).
    multi_diags->HandleHierarchyChange();
    reduced_diags->LoadBalance();

    // Re-seed the Ohm's-law moment state on the new hierarchy, exactly like
    // the restart path: deposit rho^n and J^{n-1/2} from the particles at
    // (x^n, v^{n-1/2}) -- SyncCurrentAndRho and the coarse-fine ghost/band
    // fill run inside the deposit -- recompute the electron pressure, and
    // copy into the history fabs (hybrid_rho_fp_temp = rho^n,
    // hybrid_current_fp_temp = J^{n-1/2}). The deposit uses the standard
    // -dt/2 relative-time offset, so the seeded time centering matches the
    // regular scheme; on a freshly created level the first step is
    // nonetheless approximate: the particles that now populate it gathered
    // coarse-resolution fields before the regrid.
    HybridPICDepositRhoAndJ();
    m_hybrid_pic_model->CalculateElectronPressure();

    const ablastr::fields::MultiLevelScalarField rho_fp_temp =
        m_fields.get_mr_levels(FieldType::hybrid_rho_fp_temp, finest_level);
    const ablastr::fields::MultiLevelVectorField current_fp_temp =
        m_fields.get_mr_levels_alldirs(FieldType::hybrid_current_fp_temp, finest_level);
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        MultiFab::Copy(*rho_fp_temp[lev], *m_fields.get(FieldType::rho_fp, lev),
                       0, 0, 1, rho_fp_temp[lev]->nGrowVect());
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::Copy(*current_fp_temp[lev][idim],
                           *m_fields.get(FieldType::current_fp, Direction{idim}, lev),
                           0, 0, 1, current_fp_temp[lev][idim]->nGrowVect());
        }
    }
}
