/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy, Inc.)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FiniteDifferenceSolver.H"

#ifdef WARPX_DIM_RZ
#   include "FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#else
#   include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#endif
#include "HybridPICModel/HybridPICModel.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/utils/Communication.H>

using namespace amrex;

void FiniteDifferenceSolver::HybridPICEvolveEDisplacement (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField& Jfield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Jextfield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    std::array< std::unique_ptr<amrex::iMultiFab>, 3 > const& eb_update_E,
    amrex::Real dt, int lev, HybridPICModel const* hybrid_model,
    const bool include_resistivity_term)
{
    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC) {
#ifdef WARPX_DIM_RZ

        HybridPICEvolveEDisplacementCylindrical <CylindricalYeeAlgorithm> (
            Efield, Jfield, Jifield, Jextfield, Bfield, rhofield, Pefield,
            eb_update_E, dt, lev, hybrid_model, include_resistivity_term
        );

#else

        HybridPICEvolveEDisplacementCartesian <CartesianYeeAlgorithm> (
            Efield, Jfield, Jifield, Jextfield, Bfield, rhofield, Pefield,
            eb_update_E, dt, lev, hybrid_model, include_resistivity_term
        );

#endif
    } else {
        amrex::Abort(Utils::TextMsg::Err(
            "HybridSolveE: The hybrid-PIC electromagnetic solver algorithm must be used"));
    }
}

#ifdef WARPX_DIM_RZ
template<typename T_Algo>
void FiniteDifferenceSolver::HybridPICEvolveEDisplacementCylindrical (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Jextfield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    std::array< std::unique_ptr<amrex::iMultiFab>, 3 > const& eb_update_E,
    Real dt, int lev, HybridPICModel const* hybrid_model,
    const bool include_resistivity_term )
{
#ifndef AMREX_USE_EB
    amrex::ignore_unused(eb_update_E);
#endif

    // Both steps below do not currently support m > 0 and should be
    // modified if such support wants to be added
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (m_nmodes == 1),
        "Ohm's law solver only support m = 0 azimuthal mode at present.");

    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    using namespace ablastr::coarsen::sample;

    const auto& warpx = WarpX::GetInstance();

    // get hybrid model parameters
    const auto eta = hybrid_model->m_eta;
    const auto eta_h = hybrid_model->m_eta_h;
    const auto rho_floor = hybrid_model->m_n_floor * PhysConst::q_e;
    const auto resistivity_has_J_dependence = hybrid_model->m_resistivity_has_J_dependence;
    const auto hyper_resistivity_has_B_dependence = hybrid_model->m_hyper_resistivity_has_B_dependence;
    const bool include_hyper_resistivity_term = hybrid_model->m_include_hyper_resistivity_term;

    //const bool include_hyper_resistivity_term = (eta_h > 0.0) && include_resistivity_term;

    // Index type required for interpolating fields from their respective
    // staggering to the Ex, Ey, Ez locations
    amrex::GpuArray<int, 3> const& Er_stag = hybrid_model->Ex_IndexType;
    amrex::GpuArray<int, 3> const& Et_stag = hybrid_model->Ey_IndexType;
    amrex::GpuArray<int, 3> const& Ez_stag = hybrid_model->Ez_IndexType;
    amrex::GpuArray<int, 3> const& Jr_stag = hybrid_model->Jx_IndexType;
    amrex::GpuArray<int, 3> const& Jt_stag = hybrid_model->Jy_IndexType;
    amrex::GpuArray<int, 3> const& Jz_stag = hybrid_model->Jz_IndexType;
    amrex::GpuArray<int, 3> const& Br_stag = hybrid_model->Bx_IndexType;
    amrex::GpuArray<int, 3> const& Bt_stag = hybrid_model->By_IndexType;
    amrex::GpuArray<int, 3> const& Bz_stag = hybrid_model->Bz_IndexType;

    // Parameters for `interp` that maps from Yee to nodal mesh and back
    amrex::GpuArray<int, 3> const& nodal = {1, 1, 1};
    // The "coarsening is just 1 i.e. no coarsening"
    amrex::GpuArray<int, 3> const& coarsen = {1, 1, 1};

    // The E-field calculation is done in 2 steps:
    // 1) The J x B term is calculated on a nodal mesh in order to ensure
    //    energy conservation.
    // 2) The nodal E-field values are averaged onto the Yee grid and the
    //    electron pressure & resistivity terms are added (these terms are
    //    naturally located on the Yee grid).

    // Create a temporary multifab to hold the nodal E-field values
    // Note the multifab has 3 values for Ex, Ey and Ez which we can do here
    // since all three components will be calculated on the same grid.
    // Also note that enE_nodal_mf does not need to have any guard cells since
    // these values will be interpolated to the Yee mesh which is contained
    // by the nodal mesh.
    auto const& ba = convert(rhofield.boxArray(), IntVect::TheNodeVector());
    MultiFab enE_nodal_mf(ba, rhofield.DistributionMap(), 3, IntVect::TheZeroVector());

    MultiFab grad_Pe_r_mf(Efield[0]->boxArray(), Efield[0]->DistributionMap(), 1, Efield[0]->nGrowVect());
    MultiFab grad_Pe_z_mf(Efield[2]->boxArray(), Efield[2]->DistributionMap(), 1, Efield[2]->nGrowVect());

    MultiFab nabla2_J_r_mf(Efield[0]->boxArray(), Efield[0]->DistributionMap(), 1, Efield[0]->nGrowVect());
    MultiFab nabla2_J_z_mf(Efield[2]->boxArray(), Efield[2]->DistributionMap(), 1, Efield[2]->nGrowVect());

    grad_Pe_r_mf.setVal(0.);
    grad_Pe_z_mf.setVal(0.);
    nabla2_J_r_mf.setVal(0.);
    nabla2_J_z_mf.setVal(0.);

    // Loop through the grids, and over the tiles within each grid for the
    // initial, nodal calculation of E
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(enE_nodal_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        Array4<Real> const& enE_nodal = enE_nodal_mf.array(mfi);
        Array4<Real const> const& Jr = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jt = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& Jir = Jifield[0]->const_array(mfi);
        Array4<Real const> const& Jit = Jifield[1]->const_array(mfi);
        Array4<Real const> const& Jiz = Jifield[2]->const_array(mfi);
        Array4<Real const> const& Jextr = Jextfield[0]->const_array(mfi);
        Array4<Real const> const& Jextt = Jextfield[1]->const_array(mfi);
        Array4<Real const> const& Jextz = Jextfield[2]->const_array(mfi);
        Array4<Real const> const& Br = Bfield[0]->const_array(mfi);
        Array4<Real const> const& Bt = Bfield[1]->const_array(mfi);
        Array4<Real const> const& Bz = Bfield[2]->const_array(mfi);

        // Loop over the cells and update the nodal E field
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

            // interpolate the total current to a nodal grid
            auto const jr_interp = Interp(Jr, Jr_stag, nodal, coarsen, i, j, 0, 0);
            auto const jt_interp = Interp(Jt, Jt_stag, nodal, coarsen, i, j, 0, 0);
            auto const jz_interp = Interp(Jz, Jz_stag, nodal, coarsen, i, j, 0, 0);

            // interpolate the ion current to a nodal grid
            auto const jir_interp = Interp(Jir, Jr_stag, nodal, coarsen, i, j, 0, 0);
            auto const jit_interp = Interp(Jit, Jt_stag, nodal, coarsen, i, j, 0, 0);
            auto const jiz_interp = Interp(Jiz, Jz_stag, nodal, coarsen, i, j, 0, 0);

            // interpolate the B field to a nodal grid
            auto const Br_interp = Interp(Br, Br_stag, nodal, coarsen, i, j, 0, 0);
            auto const Bt_interp = Interp(Bt, Bt_stag, nodal, coarsen, i, j, 0, 0);
            auto const Bz_interp = Interp(Bz, Bz_stag, nodal, coarsen, i, j, 0, 0);

            // calculate enE = (J - Ji) x B
            enE_nodal(i, j, 0, 0) = (
                (jt_interp - jit_interp - Jextt(i, j, 0)) * Bz_interp
                - (jz_interp - jiz_interp - Jextz(i, j, 0)) * Bt_interp
            );
            enE_nodal(i, j, 0, 1) = (
                (jz_interp - jiz_interp - Jextz(i, j, 0)) * Br_interp
                - (jr_interp - jir_interp - Jextr(i, j, 0)) * Bz_interp
            );
            enE_nodal(i, j, 0, 2) = (
                (jr_interp - jir_interp - Jextr(i, j, 0)) * Bt_interp
                - (jt_interp - jit_interp - Jextt(i, j, 0)) * Br_interp
            );
        });

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(grad_Pe_r_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        Array4<Real const> const& Pe = Pefield.const_array(mfi);
        Array4<Real> const& grad_Pe_r = grad_Pe_r_mf.array(mfi);
        Array4<Real> const& grad_Pe_z = grad_Pe_z_mf.array(mfi);
        Array4<Real const> const& Jr = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jt = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real> const& nabla2_J_r = nabla2_J_r_mf.array(mfi);
        Array4<Real> const& nabla2_J_z = nabla2_J_z_mf.array(mfi);

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_r = m_stencil_coefs_r.dataPtr();
        auto const n_coefs_r = static_cast<int>(m_stencil_coefs_r.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        Box const& ter  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        // Extract cylindrical specific parameters
        Real const dr = m_dr;
        Real const rmin = m_rmin;

        // Loop over the cells and update the gradPe and laplacian(J) fields
        amrex::ParallelFor(ter, tez, 
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                grad_Pe_r(i,j,0) = T_Algo::UpwardDr(Pe, coefs_r, n_coefs_r, i, j, 0, 0);

                // r on cell-centered point (Jr is cell-centered in r)
                Real const r = rmin + (i + 0.5_rt)*dr;
                nabla2_J_r(i,j,0) = T_Algo::Dr_rDr_over_r(Jr, r, dr, coefs_r, n_coefs_r, i, j, 0, 0);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                grad_Pe_z(i,j,0) = T_Algo::UpwardDz(Pe, coefs_z, n_coefs_z, i, j, 0, 0);
                nabla2_J_z(i,j,0) = T_Algo::Dzz(Jz, coefs_z, n_coefs_z, i, j, 0, 0);
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

    grad_Pe_r_mf.FillBoundary(warpx.Geom(lev).periodicity());
    grad_Pe_z_mf.FillBoundary(warpx.Geom(lev).periodicity());
    nabla2_J_r_mf.FillBoundary(warpx.Geom(lev).periodicity());
    nabla2_J_z_mf.FillBoundary(warpx.Geom(lev).periodicity());

    ablastr::utils::communication::FillBoundary(grad_Pe_xr_mf, WarpX::do_single_precision_comms, warpx.Geom(lev).periodicity(), true);
    ablastr::utils::communication::FillBoundary(grad_Pe_z_mf, WarpX::do_single_precision_comms, warpx.Geom(lev).periodicity(), true);
    ablastr::utils::communication::FillBoundary(nabla2_J_r_mf, WarpX::do_single_precision_comms, warpx.Geom(lev).periodicity(), true);
    ablastr::utils::communication::FillBoundary(nabla2_J_z_mf, WarpX::do_single_precision_comms, warpx.Geom(lev).periodicity(), true);
    
    amrex::Gpu::synchronize();
 


    // Loop through the grids, and over the tiles within each grid again
    // for the Yee grid calculation of the E field
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const& Er = Efield[0]->array(mfi);
        Array4<Real> const& Et = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real const> const& Jr = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jt = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& enE = enE_nodal_mf.const_array(mfi);
        Array4<Real const> const& rho = rhofield->const_array(mfi);
        Array4<Real const> const& grad_Pe_r = grad_Pe_r_mf.const_array(mfi);
        Array4<Real const> const& grad_Pe_z = grad_Pe_z_mf.const_array(mfi);
        Array4<Real const> const& nabla2_J_r = nabla2_J_r_mf.const_array(mfi);
        Array4<Real const> const& nabla2_J_z = nabla2_J_z_mf.const_array(mfi);
        Array4<Real const> const& Br = Bfield[0]->const_array(mfi);
        Array4<Real const> const& Bt = Bfield[1]->const_array(mfi);
        Array4<Real const> const& Bz = Bfield[2]->const_array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries
        amrex::Array4<int> update_Er_arr, update_Et_arr, update_Ez_arr;
        if (EB::enabled()) {
            update_Er_arr = eb_update_E[0]->array(mfi);
            update_Et_arr = eb_update_E[1]->array(mfi);
            update_Ez_arr = eb_update_E[2]->array(mfi);
        }

        // Extract cylindrical specific parameters
        Real const dr = m_dr;
        Real const rmin = m_rmin;

        Box const& ter  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tet  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        // Loop over the cells and update the E field
        amrex::ParallelFor(ter, tet, tez,

            // Er calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                
	        // Skip field update in the embedded boundaries
                if (update_Er_arr && update_Er_arr(i, j, 0) == 0) { return; }

	        // Interpolate to get the appropriate charge density in space
                Real rho_val = Interp(rho, nodal, Er_stag, coarsen, i, j, 0, 0);

                // Interpolate current to appropriate staggering to match E field
                const Real jr_val = Interp(Jr, Jr_stag, Er_stag, coarsen, i, j, 0, 0);
                const Real jt_val = Interp(Jt, Jt_stag, Er_stag, coarsen, i, j, 0, 0);
                const Real jz_val = Interp(Jz, Jz_stag, Er_stag, coarsen, i, j, 0, 0);
                const Real jtot_val = std::sqrt(jr_val*jr_val + jt_val*jt_val + jz_val*jz_val);

                // safety condition since we divide by rho_val later
                if (rho_val < rho_floor) { rho_val = rho_floor; }

                // Interpolate the electron pressure gradient to Ex staggering 
                auto grad_Pe_r_val = grad_Pe_r(i, j, 0);
                auto grad_Pe_z_val = Interp(grad_Pe_z, Ez_stag, Er_stag, coarsen, i, j, 0, 0);

                // interpolate the nodal enE values to the Yee grid for Er staggering
                auto enE_r = Interp(enE, nodal, Er_stag, coarsen, i, j, 0, 0);
                auto enE_t = Interp(enE, nodal, Er_stag, coarsen, i, j, 0, 1);
                auto enE_z = Interp(enE, nodal, Er_stag, coarsen, i, j, 0, 2);

                // Interpolate B values to Er staggering
                auto Br_val = Interp(Br, Br_stag, Er_stag, coarsen, i, j, 0, 0);
                auto Bt_val = Interp(Bt, Bt_stag, Er_stag, coarsen, i, j, 0, 0);
                auto Bz_val = Interp(Bz, Bz_stag, Er_stag, coarsen, i, j, 0, 0);
                const Real btot_val = std::sqrt(Br_val*Br_val + Bt_val*Bt_val + Bz_val*Bz_val);

                // Store old E values
                const Real Ero = Er(i,j,0);
                const Real Eto = Interp(Et, Et_stag, Er_stag, coarsen, i, j, 0, 0);
                const Real Ezo = Interp(Ez, Ez_stag, Er_stag, coarsen, i, j, 0, 0);

                Real Jtilde_r = enE_r - grad_Pe_r_val;
                Real Jtilde_t = enE_t;
                Real Jtilde_z = enE_z - grad_Pe_z_val;

                const auto eta_val = eta(rho_val, jtot_val);

                // Add resistivity only if E field value is used to update B
                if (include_resistivity_term) { 
                    Jtilde_r += rho_val * eta_val * jr_val; 
                    Jtilde_t += rho_val * eta_val * jt_val;
                    Jtilde_z += rho_val * eta_val * jz_val;
                }

                if (include_hyper_resistivity_term) {
                    auto nabla2_J_r_val = nabla2_J_r(i, j, 0);
                    auto nabla2_J_z_val = Interp(nabla2_J_z, Ez_stag, Er_stag, coarsen, i, j, 0, 0);

                    Jtilde_r -= rho_val * eta_h(rho_val, btot_val) * nabla2_J_r_val;
                    Jtilde_z -= rho_val * eta_h(rho_val, btot_val) * nabla2_J_z_val;
                }

                // Include explicit terms for displacement current
                Jtilde_r -= 0.5_rt * rho_val - PhysConst::ep0 / dt * (eta_val*Ero + Bz_val*Eto - Bt_val*Ezo); 
                Jtilde_t += PhysConst::ep0 / dt * (eta_val*Eto - Bz_val*Ero + Br_val*Ezo);
                Jtilde_z += PhysConst::ep0 / dt * (eta_val*Ezo + Bt_val*Ero - Br_val*Eto);

                // Calculate inverse for semi-implicit advance
                Real br = PhysConst::ep0 * Br_val / dt;
                Real bt = PhysConst::ep0 * Bt_val / dt;
                Real bz = PhysConst::ep0 * Bz_val / dt;
                Real d = 0.5_rt * rho_val + PhysConst::ep0 / dt * eta_val;
                Real coeff = 1.0_rt / (d * (d*d + br*br + bt*bt + bz*bz));

                Er(i,j,0) = coeff * (
                    (d*d + br*br) * Jtilde_r
                    + (br*bt - d*bz) * Jtilde_t
                    + (br*bz + d*bt) * Jtilde_z
                    );
            },

            // Et calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                
		// Skip field update in the embedded boundaries
                if (update_Er_arr && update_Er_arr(i, j, 0) == 0) { return; }
		    
		// r on a nodal grid (Et is nodal in r)
                Real const r = rmin + i*dr;
                // Mode m=0: // Ensure that Et remains 0 on axis
                if (r < 0.5_rt*dr) {
                    Et(i, j, 0, 0) = 0.;
                    return;
                }

                // Interpolate to get the appropriate charge density in space
                Real rho_val = Interp(rho, nodal, Et_stag, coarsen, i, j, 0, 0);

                // Interpolate current to appropriate staggering to match E field
                const Real jr_val = Interp(Jr, Jr_stag, Et_stag, coarsen, i, j, 0, 0);
                const Real jt_val = Interp(Jt, Jt_stag, Et_stag, coarsen, i, j, 0, 0);
                const Real jz_val = Interp(Jz, Jz_stag, Et_stag, coarsen, i, j, 0, 0);
                const Real jtot_val = std::sqrt(jr_val*jr_val + jt_val*jt_val + jz_val*jz_val);
                
                // safety condition since we divide by rho_val later
                if (rho_val < rho_floor) { rho_val = rho_floor; }

                // Interpolate the electron pressure gradient to Et staggering 
                auto grad_Pe_r_val = Interp(grad_Pe_r, Er_stag, Et_stag, coarsen, i, j, 0, 0);
                auto grad_Pe_z_val = Interp(grad_Pe_z, Ez_stag, Et_stag, coarsen, i, j, 0, 0);

                // interpolate the nodal enE values to the Yee grid for Et staggering
                auto enE_r = Interp(enE, nodal, Et_stag, coarsen, i, j, 0, 0);
                auto enE_t = Interp(enE, nodal, Et_stag, coarsen, i, j, 0, 1);
                auto enE_z = Interp(enE, nodal, Et_stag, coarsen, i, j, 0, 2);

                // Interpolate B values to Et staggering
                auto Br_val = Interp(Br, Br_stag, Et_stag, coarsen, i, j, 0, 0);
                auto Bt_val = Interp(Bt, Bt_stag, Et_stag, coarsen, i, j, 0, 0);
                auto Bz_val = Interp(Bz, Bz_stag, Et_stag, coarsen, i, j, 0, 0);
                const Real btot_val = std::sqrt(Br_val*Br_val + Bt_val*Bt_val + Bz_val*Bz_val);

                // Store old E values
                const Real Ero = Interp(Er, Er_stag, Et_stag, coarsen, i, j, 0, 0);
                const Real Eto = Et(i,j,0);
                const Real Ezo = Interp(Ez, Ez_stag, Et_stag, coarsen, i, j, 0, 0);

                Real Jtilde_r = enE_r - grad_Pe_r_val;
                Real Jtilde_t = enE_t;
                Real Jtilde_z = enE_z - grad_Pe_z_val;

                const auto eta_val = eta(rho_val, jtot_val);

                // Add resistivity only if E field value is used to update B
                if (include_resistivity_term) { 
                    Jtilde_r += rho_val * eta_val * jr_val; 
                    Jtilde_z += rho_val * eta_val * jz_val;
                }

                if (include_hyper_resistivity_term) {
                    auto nabla2_J_r_val = Interp(nabla2_J_r, Er_stag, Et_stag, coarsen, i, j, 0, 0);
                    auto nabla2_J_z_val = Interp(nabla2_J_z, Ez_stag, Et_stag, coarsen, i, j, 0, 0);

                    Jtilde_r -= rho_val * eta_h(rho_val, btot_val) * nabla2_J_r_val;
                    Jtilde_z -= rho_val * eta_h(rho_val, btot_val) * nabla2_J_z_val;
                }

                // Include explicit terms for displacement current
                Jtilde_r += PhysConst::ep0 / dt * (eta_val*Ero + Bz_val*Eto - Bt_val*Ezo); 
                Jtilde_t -= 0.5_rt * rho_val - PhysConst::ep0 / dt * (eta_val*Eto - Bz_val*Ero + Br_val*Ezo);
                Jtilde_z += PhysConst::ep0 / dt * (eta_val*Ezo + Bt_val*Ero - Br_val*Eto);

                // Calculate inverse for semi-implicit advance
                Real br = PhysConst::ep0 * Br_val / dt;
                Real bt = PhysConst::ep0 * Bt_val / dt;
                Real bz = PhysConst::ep0 * Bz_val / dt;
                Real d = 0.5_rt * rho_val + PhysConst::ep0 / dt * eta_val;
                Real coeff = 1.0_rt / (d * (d*d + br*br + bt*bt + bz*bz));

                Et(i,j,0) = coeff * (
                    (br*bt + d*bz) * Jtilde_r
                    + (d*d + bt*bt) * Jtilde_t
                    + (bt*bz - d*br) * Jtilde_z
                    );

                // Note: Hyper-resisitivity should be revisited here when modal decomposition is implemented
            },

            // Ez calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                
		// Skip field update in the embedded boundaries
                if (update_Ez_arr && update_Ez_arr(i, j, 0) == 0) { return; }
		    
		// Interpolate to get the appropriate charge density in space
                Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, 0, 0);

                // Interpolate current to appropriate staggering to match E field
                const Real jr_val = Interp(Jr, Jr_stag, Ez_stag, coarsen, i, j, 0, 0);
                const Real jt_val = Interp(Jt, Jt_stag, Ez_stag, coarsen, i, j, 0, 0);
                const Real jz_val = Interp(Jz, Jz_stag, Ez_stag, coarsen, i, j, 0, 0);
                const Real jtot_val = std::sqrt(jr_val*jr_val + jt_val*jt_val + jz_val*jz_val);

                // safety condition since we divide by rho_val later
                if (rho_val < rho_floor) { rho_val = rho_floor; }

                // Interpolate the electron pressure gradient to Ez staggering 
                auto grad_Pe_r_val = Interp(grad_Pe_r, Er_stag, Ez_stag, coarsen, i, j, 0, 0);
                auto grad_Pe_z_val = grad_Pe_z(i, j, 0);

                // interpolate the nodal enE values to the Yee grid for Ex staggering
                auto enE_r = Interp(enE, nodal, Ez_stag, coarsen, i, j, 0, 0);
                auto enE_t = Interp(enE, nodal, Ez_stag, coarsen, i, j, 0, 1);
                auto enE_z = Interp(enE, nodal, Ez_stag, coarsen, i, j, 0, 2);

                // Interpolate B values to Ex staggering
                auto Br_val = Interp(Br, Br_stag, Ez_stag, coarsen, i, j, 0, 0);
                auto Bt_val = Interp(Bt, Bt_stag, Ez_stag, coarsen, i, j, 0, 0);
                auto Bz_val = Interp(Bz, Bz_stag, Ez_stag, coarsen, i, j, 0, 0);
                const Real btot_val = std::sqrt(Br_val*Br_val + Bt_val*Bt_val + Bz_val*Bz_val);

                // Store old E values
                const Real Ero = Interp(Er, Er_stag, Ez_stag, coarsen, i, j, 0, 0);
                const Real Eto = Interp(Et, Et_stag, Ez_stag, coarsen, i, j, 0, 0);
                const Real Ezo = Ez(i,j,0);

                Real Jtilde_r = enE_r - grad_Pe_r_val;
                Real Jtilde_t = enE_t;
                Real Jtilde_z = enE_z - grad_Pe_z_val;

                const auto eta_val = eta(rho_val, jtot_val);

                // Add resistivity only if E field value is used to update B
                if (include_resistivity_term) { 
                    Jtilde_r += rho_val * eta_val * jr_val; 
                    Jtilde_t += rho_val * eta_val * jt_val;
                    Jtilde_z += rho_val * eta_val * jz_val;
                }

                if (include_hyper_resistivity_term) {
                    auto nabla2_J_r_val = Interp(nabla2_J_r, Er_stag, Ez_stag, coarsen, i, j, 0, 0);
                    auto nabla2_J_z_val = nabla2_J_z(i, j, 0);

                    Jtilde_r -= rho_val * eta_h(rho_val, btot_val) * nabla2_J_r_val;
                    Jtilde_z -= rho_val * eta_h(rho_val, btot_val) * nabla2_J_z_val;
                }

                // Include explicit terms for displacement current
                Jtilde_r += PhysConst::ep0 / dt * (eta_val*Ero + Bz_val*Eto - Bt_val*Ezo); 
                Jtilde_t += PhysConst::ep0 / dt * (eta_val*Eto - Bz_val*Ero + Br_val*Ezo);
                Jtilde_z -= 0.5_rt * rho_val - PhysConst::ep0 / dt * (eta_val*Ezo + Bt_val*Ero - Br_val*Eto);

                // Calculate inverse for semi-implicit advance
                Real br = PhysConst::ep0 * Br_val / dt;
                Real bt = PhysConst::ep0 * Bt_val / dt;
                Real bz = PhysConst::ep0 * Bz_val / dt;
                Real d = 0.5_rt * rho_val + PhysConst::ep0 / dt * eta_val;
                Real coeff = 1.0_rt / (d * (d*d + br*br + bt*bt + bz*bz));

                Ez(i,j,0) = coeff * (
                    (br*bz - d*bt) * Jtilde_r
                    + (bt*bz + d*br) * Jtilde_t
                    + (d*d + bz*bz) * Jtilde_z
                    );
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}

#else

template<typename T_Algo>
void FiniteDifferenceSolver::HybridPICEvolveEDisplacementCartesian (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Jextfield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    std::array< std::unique_ptr<amrex::iMultiFab>, 3 > const& eb_update_E,
    amrex::Real dt, int lev, HybridPICModel const* hybrid_model,
    const bool include_resistivity_term )
{
#ifndef AMREX_USE_EB
    amrex::ignore_unused(eb_update_E);
#endif

    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    using namespace ablastr::coarsen::sample;

    auto& warpx = WarpX::GetInstance();

    // get hybrid model parameters
    const auto eta = hybrid_model->m_eta;
    const auto eta_h = hybrid_model->m_eta_h;
    const auto rho_floor = hybrid_model->m_n_floor * PhysConst::q_e;
    const auto resistivity_has_J_dependence = hybrid_model->m_resistivity_has_J_dependence;
    const auto hyper_resistivity_has_B_dependence = hybrid_model->m_hyper_resistivity_has_B_dependence;
    const bool include_hyper_resistivity_term = hybrid_model->m_include_hyper_resistivity_term;
    
    //const bool include_hyper_resistivity_term = (eta_h > 0.) && include_resistivity_term;

    // Index type required for interpolating fields from their respective
    // staggering to the Ex, Ey, Ez locations
    amrex::GpuArray<int, 3> const& Ex_stag = hybrid_model->Ex_IndexType;
    amrex::GpuArray<int, 3> const& Ey_stag = hybrid_model->Ey_IndexType;
    amrex::GpuArray<int, 3> const& Ez_stag = hybrid_model->Ez_IndexType;
    amrex::GpuArray<int, 3> const& Jx_stag = hybrid_model->Jx_IndexType;
    amrex::GpuArray<int, 3> const& Jy_stag = hybrid_model->Jy_IndexType;
    amrex::GpuArray<int, 3> const& Jz_stag = hybrid_model->Jz_IndexType;
    amrex::GpuArray<int, 3> const& Bx_stag = hybrid_model->Bx_IndexType;
    amrex::GpuArray<int, 3> const& By_stag = hybrid_model->By_IndexType;
    amrex::GpuArray<int, 3> const& Bz_stag = hybrid_model->Bz_IndexType;

    // Parameters for `interp` that maps from Yee to nodal mesh and back
    amrex::GpuArray<int, 3> const& nodal = {1, 1, 1};
    // The "coarsening is just 1 i.e. no coarsening"
    amrex::GpuArray<int, 3> const& coarsen = {1, 1, 1};

    // The E-field calculation is done in 2 steps:
    // 1) The J x B term is calculated on a nodal mesh in order to ensure
    //    energy conservation.
    // 2) The nodal E-field values are averaged onto the Yee grid and the
    //    electron pressure & resistivity terms are added (these terms are
    //    naturally located on the Yee grid).

    // Create a temporary multifab to hold the nodal E-field values
    // Note the multifab has 3 values for Ex, Ey and Ez which we can do here
    // since all three components will be calculated on the same grid.
    // Also note that enE_nodal_mf does not need to have any guard cells since
    // these values will be interpolated to the Yee mesh which is contained
    // by the nodal mesh.
    auto const& ba = convert(rhofield.boxArray(), IntVect::TheNodeVector());
    MultiFab enE_nodal_mf(ba, rhofield.DistributionMap(), 3, IntVect::TheZeroVector());

    MultiFab grad_Pe_x_mf(Efield[0]->boxArray(), Efield[0]->DistributionMap(), 1, Efield[0]->nGrowVect());
    MultiFab grad_Pe_y_mf(Efield[1]->boxArray(), Efield[1]->DistributionMap(), 1, Efield[1]->nGrowVect());
    MultiFab grad_Pe_z_mf(Efield[2]->boxArray(), Efield[2]->DistributionMap(), 1, Efield[2]->nGrowVect());

    MultiFab nabla2_J_x_mf(Efield[0]->boxArray(), Efield[0]->DistributionMap(), 1, Efield[0]->nGrowVect());
    MultiFab nabla2_J_y_mf(Efield[1]->boxArray(), Efield[1]->DistributionMap(), 1, Efield[1]->nGrowVect());
    MultiFab nabla2_J_z_mf(Efield[2]->boxArray(), Efield[2]->DistributionMap(), 1, Efield[2]->nGrowVect());

    grad_Pe_x_mf.setVal(0.);
    grad_Pe_y_mf.setVal(0.);
    grad_Pe_z_mf.setVal(0.);
    nabla2_J_x_mf.setVal(0.);
    nabla2_J_y_mf.setVal(0.);
    nabla2_J_z_mf.setVal(0.);

    enE_nodal_mf.setVal(0.);

    // Set Jfield = 0. Calculate Jfield = (curl x B) / mu0 on Yee grid, then interpolate it to nodal grid and use it to calculate enE_nodal = (curl x B) x B / mu0
    MultiFab curlB_x_mf(Efield[0]->boxArray(), Efield[0]->DistributionMap(), 1, Efield[0]->nGrowVect());
    MultiFab curlB_y_mf(Efield[1]->boxArray(), Efield[1]->DistributionMap(), 1, Efield[1]->nGrowVect());
    MultiFab curlB_z_mf(Efield[2]->boxArray(), Efield[2]->DistributionMap(), 1, Efield[2]->nGrowVect());

    curlB_x_mf.setVal(0.);
    curlB_y_mf.setVal(0.);
    curlB_z_mf.setVal(0.);

        // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(curlB_x_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers) {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const &Jx = curlB_x_mf.array(mfi);
        Array4<Real> const &Jy = curlB_y_mf.array(mfi);
        Array4<Real> const &Jz = curlB_z_mf.array(mfi);
        Array4<Real const> const &Bx = Bfield[0]->const_array(mfi);
        Array4<Real const> const &By = Bfield[1]->const_array(mfi);
        Array4<Real const> const &Bz = Bfield[2]->const_array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries.
        // The plasma current is stored at the same locations as the E-field,
        // therefore the `eb_update_E` multifab also appropriately specifies
        // where the plasma current should be calculated.
        amrex::Array4<int> update_Jx_arr, update_Jy_arr, update_Jz_arr;
        if (EB::enabled()) {
            update_Jx_arr = eb_update_E[0]->array(mfi);
            update_Jy_arr = eb_update_E[1]->array(mfi);
            update_Jz_arr = eb_update_E[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract tileboxes for which to loop
        Box const& tjx  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tjy  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tjz  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        Real const one_over_mu0 = 1._rt / PhysConst::mu0;

        // Calculate the total current, using Ampere's law, on the same grid
        // as the E-field
        amrex::ParallelFor(tjx, tjy, tjz,

            // Jx calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Skip field update in the embedded boundaries
                if (update_Jx_arr && update_Jx_arr(i, j, k) == 0) { return; }

                Jx(i, j, k) = one_over_mu0 * (
                    - T_Algo::DownwardDz(By, coefs_z, n_coefs_z, i, j, k)
                    + T_Algo::DownwardDy(Bz, coefs_y, n_coefs_y, i, j, k)
                );
            },

            // Jy calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Skip field update in the embedded boundaries
                if (update_Jy_arr && update_Jy_arr(i, j, k) == 0) { return; }

                Jy(i, j, k) = one_over_mu0 * (
                    - T_Algo::DownwardDx(Bz, coefs_x, n_coefs_x, i, j, k)
                    + T_Algo::DownwardDz(Bx, coefs_z, n_coefs_z, i, j, k)
                );
            },

            // Jz calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Skip field update in the embedded boundaries
                if (update_Jz_arr && update_Jz_arr(i, j, k) == 0) { return; }

                Jz(i, j, k) = one_over_mu0 * (
                    - T_Algo::DownwardDy(Bx, coefs_y, n_coefs_y, i, j, k)
                    + T_Algo::DownwardDx(By, coefs_x, n_coefs_x, i, j, k)
                );
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

    // Loop through the grids, and over the tiles within each grid for the
    // initial, nodal calculation of E
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(enE_nodal_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        Array4<Real> const& enE_nodal = enE_nodal_mf.array(mfi);
        Array4<Real const> const& Jx = curlB_x_mf.const_array(mfi);
        Array4<Real const> const& Jy = curlB_y_mf.const_array(mfi);
        Array4<Real const> const& Jz = curlB_z_mf.const_array(mfi);
        Array4<Real const> const& Bx = Bfield[0]->const_array(mfi);
        Array4<Real const> const& By = Bfield[1]->const_array(mfi);
        Array4<Real const> const& Bz = Bfield[2]->const_array(mfi);

	// Loop over the cells and update the nodal E field
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k){

            // interpolate the total current to a nodal grid
            auto const jx_interp = Interp(Jx, Jx_stag, nodal, coarsen, i, j, k, 0);
            auto const jy_interp = Interp(Jy, Jy_stag, nodal, coarsen, i, j, k, 0);
            auto const jz_interp = Interp(Jz, Jz_stag, nodal, coarsen, i, j, k, 0);

            // interpolate the B field to a nodal grid
            auto const Bx_interp = Interp(Bx, Bx_stag, nodal, coarsen, i, j, k, 0);
            auto const By_interp = Interp(By, By_stag, nodal, coarsen, i, j, k, 0);
            auto const Bz_interp = Interp(Bz, Bz_stag, nodal, coarsen, i, j, k, 0);

            // calculate enE = J x B
            enE_nodal(i, j, k, 0) = jy_interp * Bz_interp - jz_interp * By_interp;
            enE_nodal(i, j, k, 1) = jz_interp * Bx_interp - jx_interp * Bz_interp;
            enE_nodal(i, j, k, 2) = jx_interp * By_interp - jy_interp * Bx_interp;

        });


        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }


#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(grad_Pe_x_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        Array4<Real const> const& Pe = Pefield.const_array(mfi);
        Array4<Real> const& grad_Pe_x = grad_Pe_x_mf.array(mfi);
        Array4<Real> const& grad_Pe_y = grad_Pe_y_mf.array(mfi);
        Array4<Real> const& grad_Pe_z = grad_Pe_z_mf.array(mfi);
        Array4<Real const> const& Jx = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jy = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real> const& nabla2_J_x = nabla2_J_x_mf.array(mfi);
        Array4<Real> const& nabla2_J_y = nabla2_J_y_mf.array(mfi);
        Array4<Real> const& nabla2_J_z = nabla2_J_z_mf.array(mfi);

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        Box const& tex  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tey  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        // Loop over the cells and update the gradPe and laplacian(J) fields
        amrex::ParallelFor(tex, tey, tez, 
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                grad_Pe_x(i,j,k) = T_Algo::UpwardDx(Pe, coefs_x, n_coefs_x, i, j, k);
                nabla2_J_x(i,j,k) = T_Algo::Dxx(Jx, coefs_x, n_coefs_x, i, j, k);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                grad_Pe_y(i,j,k) = T_Algo::UpwardDy(Pe, coefs_y, n_coefs_y, i, j, k);
                nabla2_J_y(i,j,k) = T_Algo::Dyy(Jy, coefs_y, n_coefs_y, i, j, k);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                grad_Pe_z(i,j,k) = T_Algo::UpwardDz(Pe, coefs_z, n_coefs_z, i, j, k);
                nabla2_J_z(i,j,k) = T_Algo::Dzz(Jz, coefs_z, n_coefs_z, i, j, k);
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

    //grad_Pe_x_mf.FillBoundary(warpx.Geom(lev).periodicity());
    //grad_Pe_y_mf.FillBoundary(warpx.Geom(lev).periodicity());
    //grad_Pe_z_mf.FillBoundary(warpx.Geom(lev).periodicity());
    //nabla2_J_x_mf.FillBoundary(warpx.Geom(lev).periodicity());
    //nabla2_J_y_mf.FillBoundary(warpx.Geom(lev).periodicity());
    //nabla2_J_z_mf.FillBoundary(warpx.Geom(lev).periodicity());

    ablastr::utils::communication::FillBoundary(grad_Pe_x_mf, WarpX::do_single_precision_comms, warpx.Geom(lev).periodicity(), true);
    ablastr::utils::communication::FillBoundary(grad_Pe_y_mf, WarpX::do_single_precision_comms, warpx.Geom(lev).periodicity(), true);
    ablastr::utils::communication::FillBoundary(grad_Pe_z_mf, WarpX::do_single_precision_comms, warpx.Geom(lev).periodicity(), true);
    ablastr::utils::communication::FillBoundary(nabla2_J_x_mf, WarpX::do_single_precision_comms, warpx.Geom(lev).periodicity(), true);
    ablastr::utils::communication::FillBoundary(nabla2_J_y_mf, WarpX::do_single_precision_comms, warpx.Geom(lev).periodicity(), true);
    ablastr::utils::communication::FillBoundary(nabla2_J_z_mf, WarpX::do_single_precision_comms, warpx.Geom(lev).periodicity(), true);

    amrex::Gpu::synchronize();


    // Loop through the grids, and over the tiles within each grid again
    // for the Yee grid calculation of the E field
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const& Ex = Efield[0]->array(mfi);
        Array4<Real> const& Ey = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real const> const& Bx = Bfield[0]->const_array(mfi);
        Array4<Real const> const& By = Bfield[1]->const_array(mfi);
        Array4<Real const> const& Bz = Bfield[2]->const_array(mfi);
        Array4<Real const> const& Jx = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jy = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& enE = enE_nodal_mf.const_array(mfi);
        Array4<Real const> const& rho = rhofield.const_array(mfi);
        Array4<Real const> const& grad_Pe_x = grad_Pe_x_mf.const_array(mfi);
        Array4<Real const> const& grad_Pe_y = grad_Pe_y_mf.const_array(mfi);
        Array4<Real const> const& grad_Pe_z = grad_Pe_z_mf.const_array(mfi);
        Array4<Real const> const& nabla2_J_x = nabla2_J_x_mf.const_array(mfi);
        Array4<Real const> const& nabla2_J_y = nabla2_J_y_mf.const_array(mfi);
        Array4<Real const> const& nabla2_J_z = nabla2_J_z_mf.const_array(mfi);

	// Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries
        amrex::Array4<int> update_Ex_arr, update_Ey_arr, update_Ez_arr;
        if (EB::enabled()) {
            update_Ex_arr = eb_update_E[0]->array(mfi);
            update_Ey_arr = eb_update_E[1]->array(mfi);
            update_Ez_arr = eb_update_E[2]->array(mfi);
        }

        Box const& tex  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tey  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        // Loop over the cells and update the E field
        amrex::ParallelFor(tex, tey, tez,

            // Ex calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                
	        // Skip field update in the embedded boundaries
                if (update_Ex_arr && update_Ex_arr(i, j, k) == 0) { return; }
	    
	        // Interpolate to get the appropriate charge density in space
                Real rho_val = Interp(rho, nodal, Ex_stag, coarsen, i, j, k, 0);

                // Interpolate current to appropriate staggering to match E field
                const Real jx_val = Interp(Jx, Jx_stag, Ex_stag, coarsen, i, j, k, 0);
                const Real jy_val = Interp(Jy, Jy_stag, Ex_stag, coarsen, i, j, k, 0);
                const Real jz_val = Interp(Jz, Jz_stag, Ex_stag, coarsen, i, j, k, 0);
                const Real jtot_val = std::sqrt(jx_val*jx_val + jy_val*jy_val + jz_val*jz_val);

                // safety condition since we divide by rho_val later
                if (rho_val < rho_floor) { rho_val = rho_floor; }

                // Interpolate the electron pressure gradient to Ex staggering 
                auto grad_Pe_x_val = grad_Pe_x(i, j, k);
                auto grad_Pe_y_val = Interp(grad_Pe_y, Ey_stag, Ex_stag, coarsen, i, j, k, 0);
                auto grad_Pe_z_val = Interp(grad_Pe_z, Ez_stag, Ex_stag, coarsen, i, j, k, 0);

                // interpolate the nodal enE values to the Yee grid for Ex staggering
                auto enE_x = Interp(enE, nodal, Ex_stag, coarsen, i, j, k, 0);
                auto enE_y = Interp(enE, nodal, Ex_stag, coarsen, i, j, k, 1);
                auto enE_z = Interp(enE, nodal, Ex_stag, coarsen, i, j, k, 2);

                // Interpolate B values to Ex staggering
                auto Bx_val = Interp(Bx, Bx_stag, Ex_stag, coarsen, i, j, k, 0);
                auto By_val = Interp(By, By_stag, Ex_stag, coarsen, i, j, k, 0);
                auto Bz_val = Interp(Bz, Bz_stag, Ex_stag, coarsen, i, j, k, 0);
                const Real btot_val = std::sqrt(Bx_val*Bx_val + By_val*By_val + Bz_val*Bz_val);

                // Store old E values
                const Real Exo = Ex(i,j,k);
                const Real Eyo = Interp(Ey, Ey_stag, Ex_stag, coarsen, i, j, k, 0);
                const Real Ezo = Interp(Ez, Ez_stag, Ex_stag, coarsen, i, j, k, 0);

                Real Jtilde_x = enE_x - grad_Pe_x_val;
                Real Jtilde_y = enE_y - grad_Pe_y_val;
                Real Jtilde_z = enE_z - grad_Pe_z_val;

                const auto eta_val = eta(rho_val, jtot_val);

                // Add resistivity only if E field value is used to update B
                if (include_resistivity_term) { 
                    Jtilde_x += rho_val * eta_val * jx_val; 
                    Jtilde_y += rho_val * eta_val * jy_val;
                    Jtilde_z += rho_val * eta_val * jz_val;
                }

                if (include_hyper_resistivity_term) {
                    auto nabla2_J_x_val = nabla2_J_x(i, j, k);
                    auto nabla2_J_y_val = Interp(nabla2_J_y, Ey_stag, Ex_stag, coarsen, i, j, k, 0);
                    auto nabla2_J_z_val = Interp(nabla2_J_z, Ez_stag, Ex_stag, coarsen, i, j, k, 0);

                    Jtilde_x -= rho_val * eta_h(rho_val, btot_val)* nabla2_J_x_val;
                    Jtilde_y -= rho_val * eta_h(rho_val, btot_val)* nabla2_J_y_val;
                    Jtilde_z -= rho_val * eta_h(rho_val, btot_val)* nabla2_J_z_val;
                }

                // Include explicit terms for displacement current
                //Jtilde_x -= 0.5_rt * rho_val - PhysConst::ep0 / dt * (eta_val*Exo + Bz_val*Eyo - By_val*Ezo); 
                //Jtilde_y += PhysConst::ep0 / dt * (eta_val*Eyo - Bz_val*Exo + Bx_val*Ezo);
                //Jtilde_z += PhysConst::ep0 / dt * (eta_val*Ezo + By_val*Exo - Bx_val*Eyo);

                Jtilde_x -= PhysConst::ep0 / dt * (-rho_val*eta_val*Exo - Bz_val*Eyo + By_val*Ezo) + 0.5_rt * rho_val * Exo; 
                Jtilde_y -= PhysConst::ep0 / dt * (Bz_val*Exo - rho_val*eta_val*Eyo -  Bx_val*Ezo) + 0.5_rt * rho_val * Eyo;
                Jtilde_z -= PhysConst::ep0 / dt * (-By_val*Exo + Bx_val*Eyo - rho_val*eta_val*Ezo) + 0.5_rt * rho_val * Ezo;

                // Calculate inverse for semi-implicit advance
                Real bx = PhysConst::ep0 * Bx_val / dt;
                Real by = PhysConst::ep0 * By_val / dt;
                Real bz = PhysConst::ep0 * Bz_val / dt;
                Real d = 0.5_rt * rho_val + PhysConst::ep0 / dt * eta_val;
                Real coeff = 1.0_rt / (d * (d*d + bx*bx + by*by + bz*bz));

                Ex(i,j,k) = coeff * (
                    (d*d + bx*bx) * Jtilde_x
                    + (bx*by - d*bz) * Jtilde_y
                    + (bx*bz + d*by) * Jtilde_z
                    );
//        	if(std::isnan(Ex(i,j,k))){
//        	if(i == 0 && j == 0 && Ex(i,j,k) != 0.0){
//        	if(i == 0 && j == 0 && (k == 120 || k == 121)){
//                     amrex::Print()<< "Ex("<< i << "," << j << ", " << k << ") = " << Ex(i, j, k) << ", Jtilde_x = " << Jtilde_x << ", Jtilde_y = " << Jtilde_x << ", Jtilde_z = " << Jtilde_z << std::endl;
//		     amrex::Print() << "d = " << d << ", bx = " << bx << ", by = " << by << ", bz = " << bz << std::endl;
//                }
	
            },

            // Ey calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                
		// Skip field update in the embedded boundaries
                if (update_Ey_arr && update_Ey_arr(i, j, k) == 0) { return; }
		    
		// Interpolate to get the appropriate charge density in space
                Real rho_val = Interp(rho, nodal, Ey_stag, coarsen, i, j, k, 0);

                // Interpolate current to appropriate staggering to match E field
                const Real jx_val = Interp(Jx, Jx_stag, Ey_stag, coarsen, i, j, k, 0);
                const Real jy_val = Interp(Jy, Jy_stag, Ey_stag, coarsen, i, j, k, 0);
                const Real jz_val = Interp(Jz, Jz_stag, Ey_stag, coarsen, i, j, k, 0);
                const Real jtot_val = std::sqrt(jx_val*jx_val + jy_val*jy_val + jz_val*jz_val);
                
                // safety condition since we divide by rho_val later
                if (rho_val < rho_floor) { rho_val = rho_floor; }

                // Interpolate the electron pressure gradient to Ey staggering 
                auto grad_Pe_x_val = Interp(grad_Pe_x, Ex_stag, Ey_stag, coarsen, i, j, k, 0);
                auto grad_Pe_y_val = grad_Pe_y(i, j, k);
                auto grad_Pe_z_val = Interp(grad_Pe_z, Ez_stag, Ey_stag, coarsen, i, j, k, 0);

                // interpolate the nodal enE values to the Yee grid for Ey staggering
                auto enE_x = Interp(enE, nodal, Ey_stag, coarsen, i, j, k, 0);
                auto enE_y = Interp(enE, nodal, Ey_stag, coarsen, i, j, k, 1);
                auto enE_z = Interp(enE, nodal, Ey_stag, coarsen, i, j, k, 2);

                // Interpolate B values to Ey staggering
                auto Bx_val = Interp(Bx, Bx_stag, Ey_stag, coarsen, i, j, k, 0);
                auto By_val = Interp(By, By_stag, Ey_stag, coarsen, i, j, k, 0);
                auto Bz_val = Interp(Bz, Bz_stag, Ey_stag, coarsen, i, j, k, 0);
                const Real btot_val = std::sqrt(Bx_val*Bx_val + By_val*By_val + Bz_val*Bz_val);

                // Store old E values
                const Real Exo = Interp(Ex, Ex_stag, Ey_stag, coarsen, i, j, k, 0);
                const Real Eyo = Ey(i,j,k);
                const Real Ezo = Interp(Ez, Ez_stag, Ey_stag, coarsen, i, j, k, 0);

                Real Jtilde_x = enE_x - grad_Pe_x_val;
                Real Jtilde_y = enE_y - grad_Pe_y_val;
                Real Jtilde_z = enE_z - grad_Pe_z_val;

                const auto eta_val = eta(rho_val, jtot_val);

                // Add resistivity only if E field value is used to update B
                if (include_resistivity_term) { 
                    Jtilde_x += rho_val * eta_val * jx_val; 
                    Jtilde_y += rho_val * eta_val * jy_val;
                    Jtilde_z += rho_val * eta_val * jz_val;
                }

                if (include_hyper_resistivity_term) {
                    auto nabla2_J_x_val = Interp(nabla2_J_x, Ex_stag, Ey_stag, coarsen, i, j, k, 0);
                    auto nabla2_J_y_val = nabla2_J_y(i, j, k);
                    auto nabla2_J_z_val = Interp(nabla2_J_z, Ez_stag, Ey_stag, coarsen, i, j, k, 0);

                    Jtilde_x -= rho_val * eta_h(rho_val, btot_val)* nabla2_J_x_val;
                    Jtilde_y -= rho_val * eta_h(rho_val, btot_val)* nabla2_J_y_val;
                    Jtilde_z -= rho_val * eta_h(rho_val, btot_val)* nabla2_J_z_val;
                }

                // Include explicit terms for displacement current
                //Jtilde_x += PhysConst::ep0 / dt * (eta_val*Exo + Bz_val*Eyo - By_val*Ezo); 
                //Jtilde_y -= 0.5_rt * rho_val - PhysConst::ep0 / dt * (eta_val*Eyo - Bz_val*Exo + Bx_val*Ezo);
                //Jtilde_z += PhysConst::ep0 / dt * (eta_val*Ezo + By_val*Exo - Bx_val*Eyo);

                Jtilde_x -= PhysConst::ep0 / dt * (-rho_val*eta_val*Exo - Bz_val*Eyo + By_val*Ezo) + 0.5_rt * rho_val * Exo; 
                Jtilde_y -= PhysConst::ep0 / dt * (Bz_val*Exo - rho_val*eta_val*Eyo -  Bx_val*Ezo) + 0.5_rt * rho_val * Eyo;
                Jtilde_z -= PhysConst::ep0 / dt * (-By_val*Exo + Bx_val*Eyo - rho_val*eta_val*Ezo) + 0.5_rt * rho_val * Ezo;

                // Calculate inverse for semi-implicit advance
                Real bx = PhysConst::ep0 * Bx_val / dt;
                Real by = PhysConst::ep0 * By_val / dt;
                Real bz = PhysConst::ep0 * Bz_val / dt;
                Real d = 0.5_rt * rho_val + PhysConst::ep0 / dt * eta_val;
                Real coeff = 1.0_rt / (d * (d*d + bx*bx + by*by + bz*bz));

                Ey(i,j,k) = coeff * (
                    (bx*by + d*bz) * Jtilde_x
                    + (d*d + by*by) * Jtilde_y
                    + (by*bz - d*bx) * Jtilde_z
                    );
            },

            // Ez calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
               
                // Skip field update in the embedded boundaries
                if (update_Ez_arr && update_Ez_arr(i, j, k) == 0) { return; }

                // Interpolate to get the appropriate charge density in space
                Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, k, 0);

                // Interpolate current to appropriate staggering to match E field
                const Real jx_val = Interp(Jx, Jx_stag, Ez_stag, coarsen, i, j, k, 0);
                const Real jy_val = Interp(Jy, Jy_stag, Ez_stag, coarsen, i, j, k, 0);
                const Real jz_val = Interp(Jz, Jz_stag, Ez_stag, coarsen, i, j, k, 0);
                const Real jtot_val = std::sqrt(jx_val*jx_val + jy_val*jy_val + jz_val*jz_val);

                // safety condition since we divide by rho_val later
                if (rho_val < rho_floor) { rho_val = rho_floor; }

                // Interpolate the electron pressure gradient to Ex staggering 
                auto grad_Pe_x_val = Interp(grad_Pe_x, Ex_stag, Ez_stag, coarsen, i, j, k, 0);
                auto grad_Pe_y_val = Interp(grad_Pe_y, Ey_stag, Ez_stag, coarsen, i, j, k, 0);
                auto grad_Pe_z_val = grad_Pe_z(i, j, k);

                // interpolate the nodal enE values to the Yee grid for Ex staggering
                auto enE_x = Interp(enE, nodal, Ez_stag, coarsen, i, j, k, 0);
                auto enE_y = Interp(enE, nodal, Ez_stag, coarsen, i, j, k, 1);
                auto enE_z = Interp(enE, nodal, Ez_stag, coarsen, i, j, k, 2);

                // Interpolate B values to Ex staggering
                auto Bx_val = Interp(Bx, Bx_stag, Ez_stag, coarsen, i, j, k, 0);
                auto By_val = Interp(By, By_stag, Ez_stag, coarsen, i, j, k, 0);
                auto Bz_val = Interp(Bz, Bz_stag, Ez_stag, coarsen, i, j, k, 0);
                const Real btot_val = std::sqrt(Bx_val*Bx_val + By_val*By_val + Bz_val*Bz_val);

                // Store old E values
                const Real Exo = Interp(Ex, Ex_stag, Ez_stag, coarsen, i, j, k, 0);
                const Real Eyo = Interp(Ey, Ey_stag, Ez_stag, coarsen, i, j, k, 0);
                const Real Ezo = Ez(i,j,k);

                Real Jtilde_x = enE_x - grad_Pe_x_val;
                Real Jtilde_y = enE_y - grad_Pe_y_val;
                Real Jtilde_z = enE_z - grad_Pe_z_val;

                const auto eta_val = eta(rho_val, jtot_val);

                // Add resistivity only if E field value is used to update B
                if (include_resistivity_term) { 
                    Jtilde_x += rho_val * eta_val * jx_val; 
                    Jtilde_y += rho_val * eta_val * jy_val;
                    Jtilde_z += rho_val * eta_val * jz_val;
                }

                if (include_hyper_resistivity_term) {
                    auto nabla2_J_x_val = Interp(nabla2_J_x, Ex_stag, Ez_stag, coarsen, i, j, k, 0);
                    auto nabla2_J_y_val = Interp(nabla2_J_y, Ey_stag, Ez_stag, coarsen, i, j, k, 0);
                    auto nabla2_J_z_val = nabla2_J_z(i, j, k);

                    Jtilde_x -= rho_val * eta_h(rho_val, btot_val)* nabla2_J_x_val;
                    Jtilde_y -= rho_val * eta_h(rho_val, btot_val)* nabla2_J_y_val;
                    Jtilde_z -= rho_val * eta_h(rho_val, btot_val)* nabla2_J_z_val;
                }

                // Include explicit terms for displacement current
                //Jtilde_x += PhysConst::ep0 / dt * (eta_val*Exo + Bz_val*Eyo - By_val*Ezo); 
                //Jtilde_y += PhysConst::ep0 / dt * (eta_val*Eyo - Bz_val*Exo + Bx_val*Ezo);
                //Jtilde_z -= 0.5_rt * rho_val - PhysConst::ep0 / dt * (eta_val*Ezo + By_val*Exo - Bx_val*Eyo);

                Jtilde_x -= PhysConst::ep0 / dt * (-rho_val*eta_val*Exo - Bz_val*Eyo + By_val*Ezo) + 0.5_rt * rho_val * Exo; 
                Jtilde_y -= PhysConst::ep0 / dt * (Bz_val*Exo - rho_val*eta_val*Eyo -  Bx_val*Ezo) + 0.5_rt * rho_val * Eyo;
                Jtilde_z -= PhysConst::ep0 / dt * (-By_val*Exo + Bx_val*Eyo - rho_val*eta_val*Ezo) + 0.5_rt * rho_val * Ezo;

                // Calculate inverse for semi-implicit advance
                Real bx = PhysConst::ep0 * Bx_val / dt;
                Real by = PhysConst::ep0 * By_val / dt;
                Real bz = PhysConst::ep0 * Bz_val / dt;
                Real d = 0.5_rt * rho_val + PhysConst::ep0 / dt * eta_val;
                Real coeff = 1.0_rt / (d * (d*d + bx*bx + by*by + bz*bz));

                Ez(i,j,k) = coeff * (
                    (bx*bz - d*by) * Jtilde_x
                    + (by*bz + d*bx) * Jtilde_y
                    + (d*d + bz*bz) * Jtilde_z
                    );
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}
#endif
