/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FiniteDifferenceSolver.H"

#include "EmbeddedBoundary/Enabled.H"
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
#   include "FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#elif defined(WARPX_DIM_RSPHERE)
#   include "FiniteDifferenceAlgorithms/SphericalYeeAlgorithm.H"
#else
#   include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#endif
#include "HybridPICModel/HybridPICModel.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

using namespace amrex;

void FiniteDifferenceSolver::HybridPICVacuumEFromA (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Afield,
    ablastr::fields::VectorField const& Afield_prev,
    ablastr::fields::VectorField const& Afield_old,
    amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev, HybridPICModel const* hybrid_model,
    amrex::Real dt, bool use_bdf2 )
{
    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)

        HybridPICVacuumEFromACylindrical <CylindricalYeeAlgorithm> (
            Efield, Afield, Afield_prev, Afield_old, rhofield, Pefield,
            eb_update_E, lev, hybrid_model, dt, use_bdf2
        );

#elif defined(WARPX_DIM_RSPHERE)
        amrex::ignore_unused(Efield, Afield, Afield_prev, Afield_old,
            rhofield, Pefield, eb_update_E, lev, hybrid_model, dt, use_bdf2);
        amrex::Abort(Utils::TextMsg::Err(
            "HybridPICVacuumEFromA: Not implemented in RSPHERE geometry"));

#else
    if (WarpX::grid_type == GridType::Staggered)
    {
        HybridPICVacuumEFromACartesian <CartesianYeeAlgorithm> (
            Efield, Afield, Afield_prev, Afield_old, rhofield, Pefield,
            eb_update_E, lev, hybrid_model, dt, use_bdf2
        );
    } else {
        HybridPICVacuumEFromACartesian <CartesianNodalAlgorithm> (
            Efield, Afield, Afield_prev, Afield_old, rhofield, Pefield,
            eb_update_E, lev, hybrid_model, dt, use_bdf2
        );
    }
#endif
    } else {
        amrex::Abort(Utils::TextMsg::Err(
            "HybridPICVacuumEFromA: The hybrid-PIC electromagnetic solver algorithm must be used"));
    }
}

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
template<typename T_Algo>
void FiniteDifferenceSolver::HybridPICVacuumEFromACylindrical (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Afield,
    ablastr::fields::VectorField const& Afield_prev,
    ablastr::fields::VectorField const& Afield_old,
    amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev, HybridPICModel const* hybrid_model,
    amrex::Real dt, bool use_bdf2 )
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (m_nmodes == 1),
        "Global A recovery only supports m = 0 azimuthal mode at present.");

    amrex::ignore_unused(lev);

    using namespace ablastr::coarsen::sample;

    const auto rho_floor = hybrid_model->m_n_floor * PhysConst::q_e;

    // Index type required for interpolating the nodal charge density to the
    // E-field staggering
    amrex::GpuArray<int, 3> const& Er_stag = hybrid_model->Ex_IndexType;
    amrex::GpuArray<int, 3> const& Etheta_stag = hybrid_model->Ey_IndexType;
    amrex::GpuArray<int, 3> const& Ez_stag = hybrid_model->Ez_IndexType;
    amrex::GpuArray<int, 3> const nodal{1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen{1, 1, 1};

    // Time differencing weights such that
    // dA/dt = c0*A + c1*A_prev + c2*A_old
    // using BDF2 when three time levels of A are available and a first-order
    // backward difference otherwise.
    const Real c0 = (use_bdf2 ?  1.5_rt :  1.0_rt) / dt;
    const Real c1 = (use_bdf2 ? -2.0_rt : -1.0_rt) / dt;
    const Real c2 = (use_bdf2 ?  0.5_rt :  0.0_rt) / dt;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const& Er = Efield[0]->array(mfi);
        Array4<Real> const& Etheta = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real const> const& Ar = Afield[0]->const_array(mfi);
        Array4<Real const> const& At = Afield[1]->const_array(mfi);
        Array4<Real const> const& Az = Afield[2]->const_array(mfi);
        Array4<Real const> const& Ar_prev = Afield_prev[0]->const_array(mfi);
        Array4<Real const> const& At_prev = Afield_prev[1]->const_array(mfi);
        Array4<Real const> const& Az_prev = Afield_prev[2]->const_array(mfi);
        Array4<Real const> const& Ar_old = Afield_old[0]->const_array(mfi);
        Array4<Real const> const& At_old = Afield_old[1]->const_array(mfi);
        Array4<Real const> const& Az_old = Afield_old[2]->const_array(mfi);
        Array4<Real const> const& rho = rhofield.const_array(mfi);
        Array4<Real const> const& Pe = Pefield.const_array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries
        amrex::Array4<int> update_Er_arr, update_Etheta_arr, update_Ez_arr;
        if (EB::enabled()) {
            update_Er_arr = eb_update_E[0]->array(mfi);
            update_Etheta_arr = eb_update_E[1]->array(mfi);
            update_Ez_arr = eb_update_E[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_r = m_stencil_coefs_r.dataPtr();
        int const n_coefs_r = static_cast<int>(m_stencil_coefs_r.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        Box const& ter = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tet = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez = mfi.tilebox(Efield[2]->ixType().toIntVect());

        // Loop over the cells and replace E in the vacuum region
        amrex::ParallelFor(ter, tet, tez,

            // Er calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Er_arr && update_Er_arr(i, j, 0) == 0) { return; }

                // Interpolate to get the appropriate charge density in space
                const Real rho_val = Interp(rho, nodal, Er_stag, coarsen, i, j, 0, 0);

                // Keep the Ohm's law solution in the plasma region
                if (rho_val >= rho_floor) { return; }

                const Real dAdt = c0*Ar(i, j, 0) + c1*Ar_prev(i, j, 0) + c2*Ar_old(i, j, 0);
                const Real grad_Pe = T_Algo::UpwardDr(Pe, coefs_r, n_coefs_r, i, j, 0, 0);

                // The pressure term only contributes in the low-density
                // transition region; in true vacuum 1/rho is set to zero
                const Real irho = (rho_val > 0._rt) ? 1._rt/rho_val : 0._rt;

                Er(i, j, 0) = -dAdt - grad_Pe * irho;
            },

            // Etheta calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Etheta_arr && update_Etheta_arr(i, j, 0) == 0) { return; }

                // Interpolate to get the appropriate charge density in space
                const Real rho_val = Interp(rho, nodal, Etheta_stag, coarsen, i, j, 0, 0);

                // Keep the Ohm's law solution in the plasma region
                if (rho_val >= rho_floor) { return; }

                const Real dAdt = c0*At(i, j, 0) + c1*At_prev(i, j, 0) + c2*At_old(i, j, 0);

                // The azimuthal gradient of the electron pressure is zero
                // for m = 0
                Etheta(i, j, 0) = -dAdt;
            },

            // Ez calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Ez_arr && update_Ez_arr(i, j, 0) == 0) { return; }

                // Interpolate to get the appropriate charge density in space
                const Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, 0, 0);

                // Keep the Ohm's law solution in the plasma region
                if (rho_val >= rho_floor) { return; }

                const Real dAdt = c0*Az(i, j, 0) + c1*Az_prev(i, j, 0) + c2*Az_old(i, j, 0);
                const Real grad_Pe = T_Algo::UpwardDz(Pe, coefs_z, n_coefs_z, i, j, 0, 0);

                const Real irho = (rho_val > 0._rt) ? 1._rt/rho_val : 0._rt;

                Ez(i, j, 0) = -dAdt - grad_Pe * irho;
            }
        );
    }
}

#elif defined(WARPX_DIM_RSPHERE)

#else

template<typename T_Algo>
void FiniteDifferenceSolver::HybridPICVacuumEFromACartesian (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Afield,
    ablastr::fields::VectorField const& Afield_prev,
    ablastr::fields::VectorField const& Afield_old,
    amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev, HybridPICModel const* hybrid_model,
    amrex::Real dt, bool use_bdf2 )
{
    amrex::ignore_unused(lev);

    using namespace ablastr::coarsen::sample;

    const auto rho_floor = hybrid_model->m_n_floor * PhysConst::q_e;

    // Index type required for interpolating the nodal charge density to the
    // E-field staggering
    amrex::GpuArray<int, 3> const& Ex_stag = hybrid_model->Ex_IndexType;
    amrex::GpuArray<int, 3> const& Ey_stag = hybrid_model->Ey_IndexType;
    amrex::GpuArray<int, 3> const& Ez_stag = hybrid_model->Ez_IndexType;
    amrex::GpuArray<int, 3> const nodal{1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen{1, 1, 1};

    // Time differencing weights such that
    // dA/dt = c0*A + c1*A_prev + c2*A_old
    // using BDF2 when three time levels of A are available and a first-order
    // backward difference otherwise.
    const Real c0 = (use_bdf2 ?  1.5_rt :  1.0_rt) / dt;
    const Real c1 = (use_bdf2 ? -2.0_rt : -1.0_rt) / dt;
    const Real c2 = (use_bdf2 ?  0.5_rt :  0.0_rt) / dt;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        // Extract field data for this grid/tile
        Array4<Real> const& Ex = Efield[0]->array(mfi);
        Array4<Real> const& Ey = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real const> const& Ax = Afield[0]->const_array(mfi);
        Array4<Real const> const& Ay = Afield[1]->const_array(mfi);
        Array4<Real const> const& Az = Afield[2]->const_array(mfi);
        Array4<Real const> const& Ax_prev = Afield_prev[0]->const_array(mfi);
        Array4<Real const> const& Ay_prev = Afield_prev[1]->const_array(mfi);
        Array4<Real const> const& Az_prev = Afield_prev[2]->const_array(mfi);
        Array4<Real const> const& Ax_old = Afield_old[0]->const_array(mfi);
        Array4<Real const> const& Ay_old = Afield_old[1]->const_array(mfi);
        Array4<Real const> const& Az_old = Afield_old[2]->const_array(mfi);
        Array4<Real const> const& rho = rhofield.const_array(mfi);
        Array4<Real const> const& Pe = Pefield.const_array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries
        amrex::Array4<int> update_Ex_arr, update_Ey_arr, update_Ez_arr;
        if (EB::enabled()) {
            update_Ex_arr = eb_update_E[0]->array(mfi);
            update_Ey_arr = eb_update_E[1]->array(mfi);
            update_Ez_arr = eb_update_E[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        Box const& tex = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tey = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez = mfi.tilebox(Efield[2]->ixType().toIntVect());

        // Loop over the cells and replace E in the vacuum region
        amrex::ParallelFor(tex, tey, tez,

            // Ex calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Skip field update in the embedded boundaries
                if (update_Ex_arr && update_Ex_arr(i, j, k) == 0) { return; }

                // Interpolate to get the appropriate charge density in space
                const Real rho_val = Interp(rho, nodal, Ex_stag, coarsen, i, j, k, 0);

                // Keep the Ohm's law solution in the plasma region
                if (rho_val >= rho_floor) { return; }

                const Real dAdt = c0*Ax(i, j, k) + c1*Ax_prev(i, j, k) + c2*Ax_old(i, j, k);
                const Real grad_Pe = T_Algo::UpwardDx(Pe, coefs_x, n_coefs_x, i, j, k);

                // The pressure term only contributes in the low-density
                // transition region; in true vacuum 1/rho is set to zero
                const Real irho = (rho_val > 0._rt) ? 1._rt/rho_val : 0._rt;

                Ex(i, j, k) = -dAdt - grad_Pe * irho;
            },

            // Ey calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Skip field update in the embedded boundaries
                if (update_Ey_arr && update_Ey_arr(i, j, k) == 0) { return; }

                // Interpolate to get the appropriate charge density in space
                const Real rho_val = Interp(rho, nodal, Ey_stag, coarsen, i, j, k, 0);

                // Keep the Ohm's law solution in the plasma region
                if (rho_val >= rho_floor) { return; }

                const Real dAdt = c0*Ay(i, j, k) + c1*Ay_prev(i, j, k) + c2*Ay_old(i, j, k);
                const Real grad_Pe = T_Algo::UpwardDy(Pe, coefs_y, n_coefs_y, i, j, k);

                const Real irho = (rho_val > 0._rt) ? 1._rt/rho_val : 0._rt;

                Ey(i, j, k) = -dAdt - grad_Pe * irho;
            },

            // Ez calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Skip field update in the embedded boundaries
                if (update_Ez_arr && update_Ez_arr(i, j, k) == 0) { return; }

                // Interpolate to get the appropriate charge density in space
                const Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, k, 0);

                // Keep the Ohm's law solution in the plasma region
                if (rho_val >= rho_floor) { return; }

                const Real dAdt = c0*Az(i, j, k) + c1*Az_prev(i, j, k) + c2*Az_old(i, j, k);
                const Real grad_Pe = T_Algo::UpwardDz(Pe, coefs_z, n_coefs_z, i, j, k);

                const Real irho = (rho_val > 0._rt) ? 1._rt/rho_val : 0._rt;

                Ez(i, j, k) = -dAdt - grad_Pe * irho;
            }
        );
    }
}
#endif
