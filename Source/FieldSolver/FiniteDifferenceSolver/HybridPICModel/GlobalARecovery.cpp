/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "GlobalARecovery.H"

#include "HybridPICModel.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/fields/VectorPoissonSolver.H>
#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/Communication.H>

#include <AMReX_MultiFab.H>
#ifdef AMREX_USE_EB
#   include <AMReX_EBFabFactory.H>
#endif

#include <algorithm>

using namespace amrex;
using warpx::fields::FieldType;

void GlobalARecoveryBoundaryHandler::define ()
{
    for (int adim = 0; adim < 3; adim++) {
        for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
            dirichlet_flag[adim][2*idim] = false;
            dirichlet_flag[adim][2*idim+1] = false;

            if (WarpX::field_boundary_lo[idim] == FieldBoundaryType::Periodic &&
                WarpX::field_boundary_hi[idim] == FieldBoundaryType::Periodic) {
                lobc[adim][idim] = LinOpBCType::Periodic;
                hibc[adim][idim] = LinOpBCType::Periodic;
                continue;
            }

            has_non_periodic = true;

            // The plasma-generated A decays to zero at the walls since the
            // external field contribution is split off, so all non-periodic
            // boundaries are homogeneous Dirichlet (matching the homogeneous
            // Dirichlet condition on embedded boundaries).
            lobc[adim][idim] = LinOpBCType::Dirichlet;
            dirichlet_flag[adim][2*idim] = true;
            hibc[adim][idim] = LinOpBCType::Dirichlet;
            dirichlet_flag[adim][2*idim+1] = true;

#if defined(WARPX_DIM_RZ)
            // The r=0 axis is not a physical boundary
            if (idim == 0 && WarpX::GetInstance().Geom(0).ProbLo(0) == 0.) {
                lobc[adim][0] = LinOpBCType::Neumann;
                dirichlet_flag[adim][0] = false;
            }
#endif
        }
    }
    bcs_set = true;
}

GlobalARecovery::GlobalARecovery (HybridPICModel const* hybrid_model)
    : m_hybrid_model(hybrid_model)
{
    ReadParameters();
}

void GlobalARecovery::ReadParameters ()
{
    const ParmParse pp_hybrid("hybrid_pic_model");

    utils::parser::queryWithParser(pp_hybrid, "global_A_rtol", m_rtol);
    utils::parser::queryWithParser(pp_hybrid, "global_A_atol", m_atol);
    utils::parser::queryWithParser(pp_hybrid, "global_A_max_iters", m_max_iters);
    utils::parser::queryWithParser(pp_hybrid, "global_A_verbosity", m_verbosity);
}

void GlobalARecovery::AllocateLevelMFs (
    ablastr::fields::MultiFabRegister & fields,
    int lev, const BoxArray& ba, const DistributionMapping& dm,
    const int ncomps,
    const IntVect& ngEB,
    const IntVect& ngRho,
    const IntVect& Ex_nodal_flag,
    const IntVect& Ey_nodal_flag,
    const IntVect& Ez_nodal_flag,
    const IntVect& Bx_nodal_flag,
    const IntVect& By_nodal_flag,
    const IntVect& Bz_nodal_flag) const
{
    using ablastr::fields::Direction;

    // The recovered A and its time history live on the edge (E) staggering
    // so that curl(A) lands on the B staggering and dA/dt matches the E
    // staggering without further interpolation.
    for (auto field_type : {FieldType::hybrid_A_fp,
                            FieldType::hybrid_A_fp_prev,
                            FieldType::hybrid_A_fp_old}) {
        fields.alloc_init(field_type, Direction{0},
            lev, amrex::convert(ba, Ex_nodal_flag),
            dm, ncomps, ngEB, 0.0_rt);
        fields.alloc_init(field_type, Direction{1},
            lev, amrex::convert(ba, Ey_nodal_flag),
            dm, ncomps, ngEB, 0.0_rt);
        fields.alloc_init(field_type, Direction{2},
            lev, amrex::convert(ba, Ez_nodal_flag),
            dm, ncomps, ngEB, 0.0_rt);
    }

    // curl(A) is stored on the B staggering before blending into B
    fields.alloc_init(FieldType::hybrid_B_fp_from_A, Direction{0},
        lev, amrex::convert(ba, Bx_nodal_flag),
        dm, ncomps, ngEB, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_B_fp_from_A, Direction{1},
        lev, amrex::convert(ba, By_nodal_flag),
        dm, ncomps, ngEB, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_B_fp_from_A, Direction{2},
        lev, amrex::convert(ba, Bz_nodal_flag),
        dm, ncomps, ngEB, 0.0_rt);

    // Nodal multifabs for the MLMG solve: the solution persists between
    // solves to serve as a warm-start guess, the source is scratch.
    const IntVect nodal_flag = IntVect::TheNodeVector();
    for (int adim = 0; adim < 3; ++adim) {
        fields.alloc_init(FieldType::hybrid_A_fp_nodal, Direction{adim},
            lev, amrex::convert(ba, nodal_flag),
            dm, ncomps, ngRho, 0.0_rt);
        fields.alloc_init(FieldType::hybrid_J_fp_nodal, Direction{adim},
            lev, amrex::convert(ba, nodal_flag),
            dm, ncomps, IntVect(1), 0.0_rt);
    }
}

void GlobalARecovery::InitData ()
{
    auto& warpx = WarpX::GetInstance();

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        warpx.finestLevel() == 0,
        "Global A recovery only works with a single level.");

#if defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::n_rz_azimuthal_modes == 1,
        "Global A recovery in RZ is only implemented for a single azimuthal mode.");
#elif !defined(WARPX_DIM_3D)
    WARPX_ABORT_WITH_MESSAGE(
        "Global A recovery is only implemented in 3D and RZ geometries.");
#endif

    m_boundary_handler.define();
}

void GlobalARecovery::InitialSolve ()
{
    ABLASTR_PROFILE("GlobalARecovery::InitialSolve()");

    using ablastr::fields::Direction;
    auto& warpx = WarpX::GetInstance();

    const ablastr::fields::MultiLevelVectorField Bfield =
        warpx.m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, warpx.finestLevel());
    const ablastr::fields::MultiLevelScalarField rhofield =
        warpx.m_fields.get_mr_levels(FieldType::rho_fp, warpx.finestLevel());

    m_hybrid_model->CalculatePlasmaCurrent(Bfield, warpx.GetEBUpdateEFlag());
    BuildMaskedNodalSource(rhofield);
    SolveA();
    InterpNodalAToEdge();

    // Seed the history with A^0 so the first step can use a first-order
    // dA/dt. The initial B field is left untouched (the user-specified
    // initial condition stays authoritative).
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        for (int adim = 0; adim < 3; ++adim) {
            auto* A = warpx.m_fields.get(FieldType::hybrid_A_fp, Direction{adim}, lev);
            auto* A_prev = warpx.m_fields.get(FieldType::hybrid_A_fp_prev, Direction{adim}, lev);
            auto* A_old = warpx.m_fields.get(FieldType::hybrid_A_fp_old, Direction{adim}, lev);
            MultiFab::Copy(*A_prev, *A, 0, 0, A_prev->nComp(), A_prev->nGrowVect());
            MultiFab::Copy(*A_old, *A, 0, 0, A_old->nComp(), A_old->nGrowVect());
        }
    }
    m_history_count = 1;
}

void GlobalARecovery::AdvanceHistory ()
{
    using ablastr::fields::Direction;
    auto& warpx = WarpX::GetInstance();

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        for (int adim = 0; adim < 3; ++adim) {
            auto* A = warpx.m_fields.get(FieldType::hybrid_A_fp, Direction{adim}, lev);
            auto* A_prev = warpx.m_fields.get(FieldType::hybrid_A_fp_prev, Direction{adim}, lev);
            auto* A_old = warpx.m_fields.get(FieldType::hybrid_A_fp_old, Direction{adim}, lev);
            MultiFab::Copy(*A_old, *A_prev, 0, 0, A_old->nComp(), A_old->nGrowVect());
            MultiFab::Copy(*A_prev, *A, 0, 0, A_prev->nComp(), A_prev->nGrowVect());
        }
    }
    m_history_count = std::min(m_history_count + 1, 3);
}

void GlobalARecovery::RecoverB (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelScalarField const& rho_blend,
    amrex::IntVect ng, std::optional<bool> nodal_sync)
{
    ABLASTR_PROFILE("GlobalARecovery::RecoverB()");

    auto& warpx = WarpX::GetInstance();

    // Calculate the plasma current J = curl(B)/mu0 - J_ext implied by the
    // current (plasma-only) B field, on the edge staggering
    m_hybrid_model->CalculatePlasmaCurrent(Bfield, warpx.GetEBUpdateEFlag());

    // Interpolate the plasma current to the nodal grid, masked to zero in
    // vacuum cells so only plasma currents source A
    BuildMaskedNodalSource(rho_blend);

    // Solve del^2 A = -mu0 J with homogeneous Dirichlet boundaries
    SolveA();

    // Interpolate the nodal solution onto the edge staggering
    InterpNodalAToEdge();

    // Compute B_from_A = curl(A) on the B staggering
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        ablastr::fields::VectorField B_from_A =
            warpx.m_fields.get_alldirs(FieldType::hybrid_B_fp_from_A, lev);
        const ablastr::fields::VectorField A_edge =
            warpx.m_fields.get_alldirs(FieldType::hybrid_A_fp, lev);
        warpx.get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(
            B_from_A, A_edge, warpx.GetEBUpdateBFlag()[lev], lev);
    }

    // Replace B with curl(A) in vacuum cells, keeping the integrated B in
    // the plasma
    BlendB(Bfield, rho_blend);

    warpx.FillBoundaryB(ng, nodal_sync);
}

void GlobalARecovery::ReconstructVacuumE (
    ablastr::fields::MultiLevelVectorField const& Efield,
    ablastr::fields::MultiLevelScalarField const& rhofield,
    amrex::Real dt,
    amrex::IntVect ng, std::optional<bool> nodal_sync)
{
    ABLASTR_PROFILE("GlobalARecovery::ReconstructVacuumE()");

    // A valid dA/dt requires at least two distinct time levels of A
    if (m_history_count < 2) { return; }
    const bool use_bdf2 = (m_history_count >= 3);

    auto& warpx = WarpX::GetInstance();

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        const ablastr::fields::VectorField A =
            warpx.m_fields.get_alldirs(FieldType::hybrid_A_fp, lev);
        const ablastr::fields::VectorField A_prev =
            warpx.m_fields.get_alldirs(FieldType::hybrid_A_fp_prev, lev);
        const ablastr::fields::VectorField A_old =
            warpx.m_fields.get_alldirs(FieldType::hybrid_A_fp_old, lev);
        auto const* Pe = warpx.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);

        warpx.get_pointer_fdtd_solver_fp(lev)->HybridPICVacuumEFromA(
            Efield[lev], A, A_prev, A_old, *rhofield[lev], *Pe,
            warpx.GetEBUpdateEFlag()[lev], lev, m_hybrid_model, dt, use_bdf2);
    }

    warpx.FillBoundaryE(ng, nodal_sync);
}

void GlobalARecovery::BuildMaskedNodalSource (
    ablastr::fields::MultiLevelScalarField const& rho_mask)
{
    using namespace ablastr::coarsen::sample;
    auto& warpx = WarpX::GetInstance();

    // Index types required for interpolating J from its staggering to nodes
    amrex::GpuArray<int, 3> const& Jx_stag = m_hybrid_model->Jx_IndexType;
    amrex::GpuArray<int, 3> const& Jy_stag = m_hybrid_model->Jy_IndexType;
    amrex::GpuArray<int, 3> const& Jz_stag = m_hybrid_model->Jz_IndexType;
    amrex::GpuArray<int, 3> const nodal{1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen{1, 1, 1};

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        const ablastr::fields::VectorField J_plasma =
            warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
        ablastr::fields::VectorField J_nodal =
            warpx.m_fields.get_alldirs(FieldType::hybrid_J_fp_nodal, lev);

#if defined(WARPX_DIM_RZ)
        // The vector Laplacian rows for the r and theta components are
        // zeroed on axis by the operator (the m=0 components of A_r and
        // A_theta vanish at r=0), so the source must also vanish there for
        // the system to be consistent.
        const bool zero_on_axis = (warpx.Geom(lev).ProbLo(0) == 0.);
        const int axis_index = warpx.Geom(lev).Domain().smallEnd(0);
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*J_nodal[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Array4<Real const> const& Jx = J_plasma[0]->const_array(mfi);
            Array4<Real const> const& Jy = J_plasma[1]->const_array(mfi);
            Array4<Real const> const& Jz = J_plasma[2]->const_array(mfi);
            Array4<Real> const& Jx_n = J_nodal[0]->array(mfi);
            Array4<Real> const& Jy_n = J_nodal[1]->array(mfi);
            Array4<Real> const& Jz_n = J_nodal[2]->array(mfi);
            Array4<Real const> const& rho = rho_mask[lev]->const_array(mfi);

            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                if (rho(i, j, k) <= 0._rt) {
                    // true vacuum: no plasma current sources A here
                    Jx_n(i, j, k) = 0._rt;
                    Jy_n(i, j, k) = 0._rt;
                    Jz_n(i, j, k) = 0._rt;
                } else {
                    Jx_n(i, j, k) = Interp(Jx, Jx_stag, nodal, coarsen, i, j, k, 0);
                    Jy_n(i, j, k) = Interp(Jy, Jy_stag, nodal, coarsen, i, j, k, 0);
                    Jz_n(i, j, k) = Interp(Jz, Jz_stag, nodal, coarsen, i, j, k, 0);
#if defined(WARPX_DIM_RZ)
                    if (zero_on_axis && i == axis_index) {
                        Jx_n(i, j, k) = 0._rt;
                        Jy_n(i, j, k) = 0._rt;
                    }
#endif
                }
            });
        }
    }
}

void GlobalARecovery::SolveA ()
{
    auto& warpx = WarpX::GetInstance();
    const int finest = warpx.finestLevel();

    // Make sure the warm-start guess satisfies the homogeneous Dirichlet
    // boundary values seen by the operator
    ZeroDirichletBoundaryNodes();

    amrex::Vector<amrex::Array<amrex::MultiFab*, 3>> sorted_curr;
    amrex::Vector<amrex::Array<amrex::MultiFab*, 3>> sorted_A;
    for (int lev = 0; lev <= finest; ++lev) {
        const ablastr::fields::VectorField J_nodal =
            warpx.m_fields.get_alldirs(FieldType::hybrid_J_fp_nodal, lev);
        const ablastr::fields::VectorField A_nodal =
            warpx.m_fields.get_alldirs(FieldType::hybrid_A_fp_nodal, lev);
        sorted_curr.emplace_back(amrex::Array<amrex::MultiFab*, 3>(
            {J_nodal[0], J_nodal[1], J_nodal[2]}));
        sorted_A.emplace_back(amrex::Array<amrex::MultiFab*, 3>(
            {A_nodal[0], A_nodal[1], A_nodal[2]}));
    }

#if defined(AMREX_USE_EB)
    std::optional<amrex::Vector<amrex::EBFArrayBoxFactory const *>> eb_farray_box_factory;
    if (EB::enabled()) {
        amrex::Vector<amrex::EBFArrayBoxFactory const *> factories;
        for (int lev = 0; lev <= finest; ++lev) {
            factories.push_back(&warpx.fieldEBFactory(lev));
        }
        eb_farray_box_factory = factories;
    }
#else
    const std::optional<amrex::Vector<amrex::FArrayBoxFactory const *>> eb_farray_box_factory;
#endif

    ablastr::fields::computeVectorPotential(
        sorted_curr,
        sorted_A,
        m_rtol,
        m_atol,
        m_max_iters,
        m_verbosity,
        warpx.Geom(),
        warpx.DistributionMap(),
        warpx.boxArray(),
        m_boundary_handler,
        EB::enabled(),
        WarpX::do_single_precision_comms,
        warpx.refRatio(),
        std::nullopt,
        warpx.gett_new(0),
        eb_farray_box_factory,
        // The bottom BiCGStab solver breaks down on the coarsened RZ vector
        // Laplacian (the -1/r^2 term for the r and theta components), so use
        // plain relaxation on the (small) coarsest level instead.
        amrex::BottomSolver::smoother
    );
}

void GlobalARecovery::ZeroDirichletBoundaryNodes ()
{
    using ablastr::fields::Direction;

    if (!m_boundary_handler.has_non_periodic) { return; }

    auto& warpx = WarpX::GetInstance();
    const auto dirichlet_flag = m_boundary_handler.dirichlet_flag;

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        amrex::Box domain = warpx.Geom(lev).Domain();
        domain.surroundingNodes();

        const ablastr::fields::VectorField A_nodal =
            warpx.m_fields.get_alldirs(FieldType::hybrid_A_fp_nodal, lev);

        for (int adim = 0; adim < 3; adim++) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(*A_nodal[adim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                auto A_arr = A_nodal[adim]->array(mfi);
                const Box& tb = mfi.tilebox(A_nodal[adim]->ixType().toIntVect());

                for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
                    if (!(dirichlet_flag[adim][2*idim] || dirichlet_flag[adim][2*idim+1])) { continue; }

                    if (!domain.strictly_contains(tb)) {
                        amrex::ParallelFor(tb,
                            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                                const IntVect iv(AMREX_D_DECL(i, j, k));

                                if (dirichlet_flag[adim][2*idim] && iv[idim] == domain.smallEnd(idim)) {
                                    A_arr(i, j, k) = 0._rt;
                                }

                                if (dirichlet_flag[adim][2*idim+1] && iv[idim] == domain.bigEnd(idim)) {
                                    A_arr(i, j, k) = 0._rt;
                                }
                            }
                        );
                    }
                }
            }
        }
    }
}

void GlobalARecovery::InterpNodalAToEdge ()
{
    using namespace ablastr::coarsen::sample;
    auto& warpx = WarpX::GetInstance();

    amrex::GpuArray<int, 3> const nodal{1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen{1, 1, 1};
    const amrex::GpuArray<amrex::GpuArray<int, 3>, 3> A_stag = {
        m_hybrid_model->Ex_IndexType,
        m_hybrid_model->Ey_IndexType,
        m_hybrid_model->Ez_IndexType};

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        const ablastr::fields::VectorField A_nodal =
            warpx.m_fields.get_alldirs(FieldType::hybrid_A_fp_nodal, lev);
        ablastr::fields::VectorField A_edge =
            warpx.m_fields.get_alldirs(FieldType::hybrid_A_fp, lev);

        for (int adim = 0; adim < 3; ++adim) {
            amrex::GpuArray<int, 3> const dst_stag = A_stag[adim];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(*A_edge[adim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                Array4<Real const> const& An = A_nodal[adim]->const_array(mfi);
                Array4<Real> const& Ae = A_edge[adim]->array(mfi);
                const Box& tb = mfi.tilebox(A_edge[adim]->ixType().toIntVect());

                amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    Ae(i, j, k) = Interp(An, nodal, dst_stag, coarsen, i, j, k, 0);
                });
            }

            ablastr::utils::communication::FillBoundary(
                *A_edge[adim], A_edge[adim]->nGrowVect(),
                WarpX::do_single_precision_comms,
                warpx.Geom(lev).periodicity());
        }
    }
}

void GlobalARecovery::BlendB (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    ablastr::fields::MultiLevelScalarField const& rho_blend)
{
    using namespace ablastr::coarsen::sample;
    auto& warpx = WarpX::GetInstance();

    const Real rho_floor = m_hybrid_model->m_n_floor * PhysConst::q_e;

    amrex::GpuArray<int, 3> const& Bx_stag = m_hybrid_model->Bx_IndexType;
    amrex::GpuArray<int, 3> const& By_stag = m_hybrid_model->By_IndexType;
    amrex::GpuArray<int, 3> const& Bz_stag = m_hybrid_model->Bz_IndexType;
    amrex::GpuArray<int, 3> const nodal{1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen{1, 1, 1};

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        const ablastr::fields::VectorField B_from_A =
            warpx.m_fields.get_alldirs(FieldType::hybrid_B_fp_from_A, lev);
        auto const& eb_update_B = warpx.GetEBUpdateBFlag()[lev];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*Bfield[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Array4<Real> const& Bx = Bfield[lev][0]->array(mfi);
            Array4<Real> const& By = Bfield[lev][1]->array(mfi);
            Array4<Real> const& Bz = Bfield[lev][2]->array(mfi);
            Array4<Real const> const& BxA = B_from_A[0]->const_array(mfi);
            Array4<Real const> const& ByA = B_from_A[1]->const_array(mfi);
            Array4<Real const> const& BzA = B_from_A[2]->const_array(mfi);
            Array4<Real const> const& rho = rho_blend[lev]->const_array(mfi);

            // Extract structures indicating where the fields
            // should be updated, given the position of the embedded boundaries.
            amrex::Array4<int> update_Bx_arr, update_By_arr, update_Bz_arr;
            if (EB::enabled()) {
                update_Bx_arr = eb_update_B[0]->array(mfi);
                update_By_arr = eb_update_B[1]->array(mfi);
                update_Bz_arr = eb_update_B[2]->array(mfi);
            }

            const Box& tbx = mfi.tilebox(Bfield[lev][0]->ixType().toIntVect());
            const Box& tby = mfi.tilebox(Bfield[lev][1]->ixType().toIntVect());
            const Box& tbz = mfi.tilebox(Bfield[lev][2]->ixType().toIntVect());

            amrex::ParallelFor(tbx, tby, tbz,

                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    // Skip field update in the embedded boundaries
                    if (update_Bx_arr && update_Bx_arr(i, j, k) == 0) { return; }

                    const Real rho_val = Interp(rho, nodal, Bx_stag, coarsen, i, j, k, 0);
                    if (rho_val < rho_floor) { Bx(i, j, k) = BxA(i, j, k); }
                },

                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    // Skip field update in the embedded boundaries
                    if (update_By_arr && update_By_arr(i, j, k) == 0) { return; }

                    const Real rho_val = Interp(rho, nodal, By_stag, coarsen, i, j, k, 0);
                    if (rho_val < rho_floor) { By(i, j, k) = ByA(i, j, k); }
                },

                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    // Skip field update in the embedded boundaries
                    if (update_Bz_arr && update_Bz_arr(i, j, k) == 0) { return; }

                    const Real rho_val = Interp(rho, nodal, Bz_stag, coarsen, i, j, k, 0);
                    if (rho_val < rho_floor) { Bz(i, j, k) = BzA(i, j, k); }
                }
            );
        }
    }
}
