/* Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *          S. Eric Clark (Helion Energy)
 *          Prabhat Kumar (Helion Energy)
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
using warpx::fields::FieldType;

void FiniteDifferenceSolver::CalculateCurrentAmpere (
    ablastr::fields::VectorField & Jfield,
    ablastr::fields::VectorField const& Bfield,
    [[maybe_unused]]std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev )
{
    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
        CalculateCurrentAmpereCylindrical <CylindricalYeeAlgorithm> (
            Jfield, Bfield, eb_update_E, lev
        );

#elif defined(WARPX_DIM_RSPHERE)
        CalculateCurrentAmpereSpherical <SphericalYeeAlgorithm> (
            Jfield, Bfield, lev
        );

#else
    if (WarpX::grid_type == GridType::Staggered)
    {
        CalculateCurrentAmpereCartesian <CartesianYeeAlgorithm> (
            Jfield, Bfield, eb_update_E, lev
        );
    } else {
        CalculateCurrentAmpereCartesian <CartesianNodalAlgorithm> (
            Jfield, Bfield, eb_update_E, lev
        );
    }

#endif
    } else {
        amrex::Abort(Utils::TextMsg::Err(
            "CalculateCurrentAmpere: Unknown algorithm choice."));
    }
}

// /**
//   * \brief Calculate total current from Ampere's law without displacement
//   * current i.e. J = 1/mu_0 curl x B.
//   *
//   * \param[out] Jfield  vector of total current MultiFabs at a given level
//   * \param[in] Bfield   vector of magnetic field MultiFabs at a given level
//   * \param[in] eb_update_E specifies where the plasma current should be calculated.
//   * \param[in] lev refinement level
//   */
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
template<typename T_Algo>
void FiniteDifferenceSolver::CalculateCurrentAmpereCylindrical (
    ablastr::fields::VectorField& Jfield,
    ablastr::fields::VectorField const& Bfield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev
)
{
    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Jfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const& Jr = Jfield[0]->array(mfi);
        Array4<Real> const& Jtheta = Jfield[1]->array(mfi);
        Array4<Real> const& Jz = Jfield[2]->array(mfi);
        Array4<Real> const& Br = Bfield[0]->array(mfi);
        Array4<Real> const& Btheta = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries.
        // The plasma current is stored at the same locations as the E-field,
        // therefore the `eb_update_E` multifab also appropriately specifies
        // where the plasma current should be calculated.
        amrex::Array4<int> update_Jr_arr, update_Jtheta_arr, update_Jz_arr;
        if (EB::enabled()) {
            update_Jr_arr = eb_update_E[0]->array(mfi);
            update_Jtheta_arr = eb_update_E[1]->array(mfi);
            update_Jz_arr = eb_update_E[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_r = m_stencil_coefs_r.dataPtr();
        int const n_coefs_r = static_cast<int>(m_stencil_coefs_r.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract cylindrical specific parameters
        Real const dr = m_dr;
        int const nmodes = m_nmodes;
        Real const rmin = m_rmin;

        // Extract tileboxes for which to loop with 1 guard cell included
        Box const& tjr  = mfi.tilebox(Jfield[0]->ixType().toIntVect(), IntVect(1));
        Box const& tjtheta  = mfi.tilebox(Jfield[1]->ixType().toIntVect(), IntVect(1));
        Box const& tjz  = mfi.tilebox(Jfield[2]->ixType().toIntVect(), IntVect(1));

        Real const one_over_mu0 = 1._rt / PhysConst::mu0;

        // Calculate the total current, using Ampere's law, on the same grid
        // as the E-field
        amrex::ParallelFor(tjr, tjtheta, tjz,

            // Jr calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Jr_arr && update_Jr_arr(i, j, 0) == 0) { return; }

                // Mode m=0
                Jr(i, j, 0, 0) = one_over_mu0 * (
                    - T_Algo::DownwardDz(Btheta, coefs_z, n_coefs_z, i, j, 0, 0)
                );

                // Higher-order modes
                // r on cell-centered point (Jr is cell-centered in r)
                Real const r = rmin + (i + 0.5_rt)*dr;
                for (int m=1; m<nmodes; m++) {
                    Jr(i, j, 0, 2*m-1) = one_over_mu0 * (
                        - T_Algo::DownwardDz(Btheta, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                        + m * Bz(i, j, 0, 2*m  ) / r
                    );  // Real part
                    Jr(i, j, 0, 2*m  ) = one_over_mu0 * (
                        - T_Algo::DownwardDz(Btheta, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                        - m * Bz(i, j, 0, 2*m-1) / r
                    ); // Imaginary part
                }
            },

            // Jtheta calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Jtheta_arr && update_Jtheta_arr(i, j, 0) == 0) { return; }

                // r on a nodal point (Jtheta is nodal in r)
                Real const r = rmin + i*dr;
                // Off-axis, regular curl
                if (r > 0.5_rt*dr) {
                    // Mode m=0
                    Jtheta(i, j, 0, 0) = one_over_mu0 * (
                        - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 0)
                        + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 0)
                    );

                    // Higher-order modes
                    for (int m=1 ; m<nmodes ; m++) { // Higher-order modes
                        Jtheta(i, j, 0, 2*m-1) = one_over_mu0 * (
                            - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                            + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                        ); // Real part
                        Jtheta(i, j, 0, 2*m  ) = one_over_mu0 * (
                            - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 2*m  )
                            + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                        ); // Imaginary part
                    }
                // r==0: on-axis corrections
                } else {
                    // Ensure that Jtheta remains 0 on axis (except for m=1)
                    // Mode m=0
                    Jtheta(i, j, 0, 0) = 0.;
                    // Higher-order modes
                    for (int m=1; m<nmodes; m++) {
                        if (m == 1){
                            // The same logic as is used in the E-field update for the fully
                            // electromagnetic FDTD case is used here.
                            Jtheta(i,j,0,2*m-1) =  Jr(i,j,0,2*m  );
                            Jtheta(i,j,0,2*m  ) = -Jr(i,j,0,2*m-1);
                        } else {
                            Jtheta(i, j, 0, 2*m-1) = 0.;
                            Jtheta(i, j, 0, 2*m  ) = 0.;
                        }
                    }
                }
            },

            // Jz calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Jz_arr && update_Jz_arr(i, j, 0) == 0) { return; }

                // r on a nodal point (Jz is nodal in r)
                Real const r = rmin + i*dr;
                // Off-axis, regular curl
                if (r > 0.5_rt*dr) {
                    // Mode m=0
                    Jz(i, j, 0, 0) = one_over_mu0 * (
                       T_Algo::DownwardDrr_over_r(Btheta, r, dr, coefs_r, n_coefs_r, i, j, 0, 0)
                    );
                    // Higher-order modes
                    for (int m=1 ; m<nmodes ; m++) {
                        Jz(i, j, 0, 2*m-1) = one_over_mu0 * (
                            - m * Br(i, j, 0, 2*m  ) / r
                            + T_Algo::DownwardDrr_over_r(Btheta, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                        ); // Real part
                        Jz(i, j, 0, 2*m  ) = one_over_mu0 * (
                            m * Br(i, j, 0, 2*m-1) / r
                            + T_Algo::DownwardDrr_over_r(Btheta, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m  )
                        ); // Imaginary part
                    }
                // r==0: on-axis corrections
                } else {
                    // For m==0, Btheta is linear in r, for small r
                    // Therefore, the formula below regularizes the singularity
                    Jz(i, j, 0, 0) = one_over_mu0 * 4 * Btheta(i, j, 0, 0) / dr;
                    // Ensure that Jz remains 0 for higher-order modes
                    for (int m=1; m<nmodes; m++) {
                        Jz(i, j, 0, 2*m-1) = 0.;
                        Jz(i, j, 0, 2*m  ) = 0.;
                    }
                }
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

#elif defined(WARPX_DIM_RSPHERE)
template<typename T_Algo>
void FiniteDifferenceSolver::CalculateCurrentAmpereSpherical (
    ablastr::fields::VectorField& Jfield,
    ablastr::fields::VectorField const& Bfield,
    int lev
)
{
    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Jfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const& Jr = Jfield[0]->array(mfi);
        Array4<Real> const& Jtheta = Jfield[1]->array(mfi);
        Array4<Real> const& Jphi = Jfield[2]->array(mfi);
        Array4<Real> const& Btheta = Bfield[1]->array(mfi);
        Array4<Real> const& Bphi = Bfield[2]->array(mfi);

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_r = m_stencil_coefs_r.dataPtr();
        int const n_coefs_r = static_cast<int>(m_stencil_coefs_r.size());

        // Extract cylindrical specific parameters
        Real const dr = m_dr;
        Real const rmin = m_rmin;

        // Extract tileboxes for which to loop with 1 guard cell included
        Box const& tjr  = mfi.tilebox(Jfield[0]->ixType().toIntVect(), IntVect(1));
        Box const& tjtheta  = mfi.tilebox(Jfield[1]->ixType().toIntVect(), IntVect(1));
        Box const& tjphi  = mfi.tilebox(Jfield[2]->ixType().toIntVect(), IntVect(1));

        Real const one_over_mu0 = 1._rt / PhysConst::mu0;

        // Calculate the total current, using Ampere's law, on the same grid
        // as the E-field
        amrex::ParallelFor(tjr, tjtheta, tjphi,

            // Jr calculation
            [=] AMREX_GPU_DEVICE (int i, int /*j*/, int /*k*/){
                Jr(i, 0, 0, 0) = 0._rt;
            },

            // Jtheta calculation
            [=] AMREX_GPU_DEVICE (int i, int /*j*/, int /*k*/){
                // r on a nodal point (Jtheta is nodal in r)
                Real const r = rmin + i*dr;
                // Off-axis, regular curl
                if (r > 0.5_rt*dr) {
                    // Mode m=0
                    Jtheta(i, 0, 0, 0) = one_over_mu0 * (
                        - T_Algo::DownwardDrr_over_r(Bphi, r, dr, coefs_r, n_coefs_r, i, 0, 0, 0));
                } else { // r==0: on-axis corrections
                    // Ensure that Jtheta remains 0 on axis
                    Jtheta(i, 0, 0, 0) = 0.;
                }
            },

            // Jphi calculation
            [=] AMREX_GPU_DEVICE (int i, int /*j*/, int /*k*/){
                // r on a nodal point (Jphi is nodal in r)
                Real const r = rmin + i*dr;
                // Off-axis, regular curl
                if (r > 0.5_rt*dr) {
                    Jphi(i, 0, 0, 0) = one_over_mu0 * (
                       T_Algo::DownwardDrr_over_r(Btheta, r, dr, coefs_r, n_coefs_r, i, 0, 0, 0)
                    );
                // r==0: on-axis corrections
                } else {
                    // Btheta is linear in r, for small r
                    // Therefore, the formula below regularizes the singularity
                    Jphi(i, 0, 0, 0) = one_over_mu0 * 4 * Btheta(i, 0, 0, 0) / dr;
                }
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
void FiniteDifferenceSolver::CalculateCurrentAmpereCartesian (
    ablastr::fields::VectorField& Jfield,
    ablastr::fields::VectorField const& Bfield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev
)
{
    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Jfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers) {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const &Jx = Jfield[0]->array(mfi);
        Array4<Real> const &Jy = Jfield[1]->array(mfi);
        Array4<Real> const &Jz = Jfield[2]->array(mfi);
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

        // Extract tileboxes for which to loop with 1 guard cell included
        Box const& tjx = mfi.tilebox(Jfield[0]->ixType().toIntVect(), IntVect(1));
        Box const& tjy = mfi.tilebox(Jfield[1]->ixType().toIntVect(), IntVect(1));
        Box const& tjz = mfi.tilebox(Jfield[2]->ixType().toIntVect(), IntVect(1));

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
}
#endif


void FiniteDifferenceSolver::HybridPICSolveE (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField& Jfield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    [[maybe_unused]]std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev, HybridPICModel const* hybrid_model,
    const bool solve_for_Faraday, const bool include_resistivity)
{
    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)

        HybridPICSolveECylindrical <CylindricalYeeAlgorithm> (
            Efield, Jfield, Jifield, Bfield, rhofield, Pefield,
            eb_update_E, lev, hybrid_model, solve_for_Faraday, include_resistivity
        );

#elif defined(WARPX_DIM_RSPHERE)

        HybridPICSolveESpherical <SphericalYeeAlgorithm> (
            Efield, Jfield, Jifield, Bfield, rhofield, Pefield,
            lev, hybrid_model, solve_for_Faraday, include_resistivity
        );

#else
    if (WarpX::grid_type == GridType::Staggered)
    {
        HybridPICSolveECartesian <CartesianYeeAlgorithm> (
            Efield, Jfield, Jifield, Bfield, rhofield, Pefield,
            eb_update_E, lev, hybrid_model, solve_for_Faraday, include_resistivity
        );
    } else {
        HybridPICSolveECartesian <CartesianNodalAlgorithm> (
            Efield, Jfield, Jifield, Bfield, rhofield, Pefield,
            eb_update_E, lev, hybrid_model, solve_for_Faraday, include_resistivity
        );
    }
#endif
    } else {
        amrex::Abort(Utils::TextMsg::Err(
            "HybridSolveE: The hybrid-PIC electromagnetic solver algorithm must be used"));
    }
}

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
template<typename T_Algo>
void FiniteDifferenceSolver::HybridPICSolveECylindrical (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev, HybridPICModel const* hybrid_model,
    const bool solve_for_Faraday, const bool include_resistivity )
{
    // Both steps below do not currently support m > 0 and should be
    // modified if such support wants to be added
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (m_nmodes == 1),
        "Ohm's law solver only support m = 0 azimuthal mode at present.");

    // Je-form relax advance (esolve = je_form): the division-free electron-momentum
    // E advance (see the m_je_advance member doc). Requirements are checked
    // here; the advance itself runs after the nodal enE assembly below
    // and returns before the legacy divided-Ohm path.
    bool const je_relax_solve =
        hybrid_model->m_esolve_je && (hybrid_model->m_je_advance == 1);
    if (je_relax_solve) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::grid_type == ablastr::utils::enums::GridType::Collocated,
            "hybrid_pic_model.esolve = je_form (v1) requires "
            "warpx.grid_type = collocated");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !hybrid_model->m_add_external_fields ||
                hybrid_model->m_external_field_mode ==
                    HybridPICModel::ExternalFieldMode::Split,
            "esolve = je_form with external fields supports the Split "
            "external-field mode only on this fork (TotalAssembled and "
            "UnifiedA are untested with the relax advance)");
        // The RK substep STAGES and particle-E calls return the frozen
        // state (see the m_je_relax_advance member doc): only the
        // once-per-accepted-substep advance raised by BfieldEvolve
        // mutates (E, J_e). Returning here also skips the enE assembly
        // below on every no-op call.
        if (!hybrid_model->m_je_relax_advance) { return; }
    }

    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    using namespace ablastr::coarsen::sample;

    // get hybrid model parameters
    const auto eta = hybrid_model->m_eta;
    const auto eta_h = hybrid_model->m_eta_h;
    const auto rho_floor = hybrid_model->m_n_floor * PhysConst::q_e;
    const auto resistivity_has_J_dependence = hybrid_model->m_resistivity_has_J_dependence;
    const auto hyper_resistivity_has_B_dependence = hybrid_model->m_hyper_resistivity_has_B_dependence;
    const bool include_hyper_resistivity_term = hybrid_model->m_include_hyper_resistivity_term;
    const bool include_electron_inertia = hybrid_model->m_include_electron_inertia;

    const bool include_external_fields = hybrid_model->m_add_external_fields
        && !hybrid_model->m_external_unified;
    const bool include_hall_term = hybrid_model->m_include_hall_term;
    const bool include_electron_pressure_term =
        hybrid_model->m_include_electron_pressure_term;
    // The stored electric field follows the split-field convention in both
    // schemes: the inductive E_ext is subtracted from plasma cells (where
    // the generalized Ohm's law itself is the electric field and the
    // external drive must not be double-counted), and the caller re-adds
    // E_ext when assembling the particle-push field and advances the
    // external flux analytically from A(t). The modes differ only in the
    // Hall-term field: the explicit scheme keeps Bfield_fp plasma-only
    // during the advance so the kernels add B_ext here; the implicit
    // scheme hands the kernels the total field already.
    const bool external_split = hybrid_model->m_external_split;

    // External-field convention (see HybridPICModel::ExternalFieldMode):
    // only the Split mode assembles totals inside the kernels by adding
    // B_ext; TotalAssembled consumers hand the kernels total B already, and
    // UnifiedA consumers never touch the external-field registers. The
    // E_ext subtraction is gated per cell on rho >= rho_floor unless the
    // consumer requires the smooth unconditional form (matrix-free
    // implicit residuals).
    const bool external_b_split = include_external_fields &&
        (hybrid_model->m_external_field_mode ==
         HybridPICModel::ExternalFieldMode::Split);
    const bool include_E_ext = include_external_fields &&
        (hybrid_model->m_external_field_mode !=
         HybridPICModel::ExternalFieldMode::UnifiedA);
    const bool subtract_E_ext_everywhere = hybrid_model->m_subtract_E_ext_everywhere;

    const bool holmstrom_vacuum_region = hybrid_model->m_holmstrom_vacuum_region;
    // Smooth Hall/grad-Pe turn-off across the vacuum gate: the binary branch
    // makes the residual discontinuous in the state exactly where cells
    // straddle the gate (Newton limit-cycles and grid-scale E jumps at the
    // separatrix edge); a tanh blend over holmstrom_transition_width*n_floor
    // restores smoothness. Width 0 (default) keeps the hard branch.
    const Real holmstrom_inv_width =
        (hybrid_model->m_holmstrom_transition_width > 0._rt)
        ? 1._rt / (hybrid_model->m_holmstrom_transition_width * rho_floor)
        : 0._rt;
    const bool holmstrom_smooth =
        holmstrom_vacuum_region && (holmstrom_inv_width > 0._rt);

    // Energy-equation-era gating (see the drag/battery ledger in the
    // HybridPICModel docs):
    //  * grad Pe stays in the FARADAY solves too when the Biermann battery
    //    is kept (curl(grad Pe/(e n)) = (grad Pe x grad n)/(e n^2) != 0
    //    once Te decouples from n);
    //  * eta J enters the PUSH solve when the Q_ei drag operator is on --
    //    dropping it is itself the single-species ion-side friction, so
    //    E* + drag would book the friction twice. Hyper-resistivity is a
    //    numerical B smoother and stays Faraday-only in either mode.
    const bool add_grad_pe_faraday = hybrid_model->m_include_biermann_battery
        && hybrid_model->m_include_electron_pressure_term;
    const bool add_resistivity_push =
        hybrid_model->m_include_temperature_relaxation;

    auto & warpx = WarpX::GetInstance();
    const amrex::Real t_new = warpx.gett_new(lev);
    // Nodal electron-inertia field, assembled by the caller each
    // evaluation (theta-implicit hybrid only; stays zero elsewhere).
    amrex::MultiFab const * Ei_nodal_mf = include_electron_inertia
        ? warpx.m_fields.get("hybrid_E_inertial_nodal", lev) : nullptr;
    ablastr::fields::VectorField Bfield_external, Efield_external;
    if (include_external_fields) {
        Bfield_external = warpx.m_fields.get_alldirs(FieldType::hybrid_B_fp_external, 0); // lev=0
        Efield_external = warpx.m_fields.get_alldirs(FieldType::hybrid_E_fp_external, 0); // lev=0
    }

    // Index type required for interpolating fields from their respective
    // staggering to the Ex, Ey, Ez locations
    amrex::GpuArray<int, 3> const& Er_stag = hybrid_model->Ex_IndexType;
    amrex::GpuArray<int, 3> const& Etheta_stag = hybrid_model->Ey_IndexType;
    amrex::GpuArray<int, 3> const& Ez_stag = hybrid_model->Ez_IndexType;
    amrex::GpuArray<int, 3> const& Jr_stag = hybrid_model->Jx_IndexType;
    amrex::GpuArray<int, 3> const& Jtheta_stag = hybrid_model->Jy_IndexType;
    amrex::GpuArray<int, 3> const& Jz_stag = hybrid_model->Jz_IndexType;
    amrex::GpuArray<int, 3> const& Br_stag = hybrid_model->Bx_IndexType;
    amrex::GpuArray<int, 3> const& Btheta_stag = hybrid_model->By_IndexType;
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

    // Per-species resistive overlay added to Ohm's-law E alongside +eta_global J.
    // Computed once per step (HybridPICEvolveFields -> ComputeResistiveOverlay)
    // into the registered hybrid_eta_overlay_fp fields and only READ here, so
    // the subcycled E-solves share it instead of recomputing it. When no
    // per-species resistivity parser is registered the fields are not
    // allocated and the per-cell add is skipped (E += 0 is a no-op) --
    // bit-identical to the single-eta path.
    const bool has_eta_overlay = hybrid_model->m_has_per_species_eta;
    ablastr::fields::VectorField eta_overlay_mf = {nullptr, nullptr, nullptr};
    if (has_eta_overlay) {
        eta_overlay_mf = warpx.m_fields.get_alldirs("hybrid_eta_overlay_fp", lev);
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
        Real wt = static_cast<Real>(amrex::second());

        Array4<Real> const& enE_nodal = enE_nodal_mf.array(mfi);
        Array4<Real const> const& Jr = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jtheta = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& Jir = Jifield[0]->const_array(mfi);
        Array4<Real const> const& Jit = Jifield[1]->const_array(mfi);
        Array4<Real const> const& Jiz = Jifield[2]->const_array(mfi);
        Array4<Real const> const& Br = Bfield[0]->const_array(mfi);
        Array4<Real const> const& Btheta = Bfield[1]->const_array(mfi);
        Array4<Real const> const& Bz = Bfield[2]->const_array(mfi);

        Array4<Real> Br_ext, Btheta_ext, Bz_ext;
        if (external_b_split) {
            Br_ext = Bfield_external[0]->array(mfi);
            Btheta_ext = Bfield_external[1]->array(mfi);
            Bz_ext = Bfield_external[2]->array(mfi);
        }

        // Loop over the cells and update the nodal E field
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

            // interpolate the total current to a nodal grid
            auto const jr_interp = Interp(Jr, Jr_stag, nodal, coarsen, i, j, 0, 0);
            auto const jtheta_interp = Interp(Jtheta, Jtheta_stag, nodal, coarsen, i, j, 0, 0);
            auto const jz_interp = Interp(Jz, Jz_stag, nodal, coarsen, i, j, 0, 0);

            // interpolate the ion current to a nodal grid
            auto const jir_interp = Interp(Jir, Jr_stag, nodal, coarsen, i, j, 0, 0);
            auto const jit_interp = Interp(Jit, Jtheta_stag, nodal, coarsen, i, j, 0, 0);
            auto const jiz_interp = Interp(Jiz, Jz_stag, nodal, coarsen, i, j, 0, 0);

            // interpolate the B field to a nodal grid
            auto Br_interp = Interp(Br, Br_stag, nodal, coarsen, i, j, 0, 0);
            auto Btheta_interp = Interp(Btheta, Btheta_stag, nodal, coarsen, i, j, 0, 0);
            auto Bz_interp = Interp(Bz, Bz_stag, nodal, coarsen, i, j, 0, 0);

            if (external_b_split) {
                Br_interp += Interp(Br_ext, Br_stag, nodal, coarsen, i, j, 0, 0);
                Btheta_interp += Interp(Btheta_ext, Btheta_stag, nodal, coarsen, i, j, 0, 0);
                Bz_interp += Interp(Bz_ext, Bz_stag, nodal, coarsen, i, j, 0, 0);
            }

            // calculate enE = (J - Ji) x B (without the Hall term the total
            // current drops out and this is the ideal -u_i x B motional
            // field)
            const Real jer = (include_hall_term ? jr_interp : 0.0_rt) - jir_interp;
            const Real jet = (include_hall_term ? jtheta_interp : 0.0_rt) - jit_interp;
            const Real jez = (include_hall_term ? jz_interp : 0.0_rt) - jiz_interp;
            enE_nodal(i, j, 0, 0) = (
                jet * Bz_interp
                - jez * Btheta_interp
            );
            enE_nodal(i, j, 0, 1) = (
                jez * Br_interp
                - jer * Bz_interp
            );
            enE_nodal(i, j, 0, 2) = (
                jer * Btheta_interp
                - jet * Br_interp
            );
        });

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

    // ------------------------------------------------------------------
    // Je-form relax advance (esolve = je_form, je_advance = relax): J_e is a
    // persistent dynamical field and E advances by Ampere-Maxwell with
    // displacement current at the CFL-set artificial light speed; the
    // stiff LOCAL electron-momentum terms (E coupling, gyration,
    // friction, vacuum decay) are solved pointwise implicitly in closed
    // form -- division-free, no linear algebra, no density floor. E is
    // the full physical field state (grad Pe always included); with
    // Split external fields the state stays internal-only and the
    // physics uses E_state + E_ext (quasi-static over a substep).
    // Steady state recovers generalized Ohm + Ampere exactly. Reached
    // once per ACCEPTED substep (the advance gate was checked at entry);
    // returns before the legacy divided-Ohm path below. RZ mode 0: the
    // axis nodes carry the parity conditions E_r = E_theta = J_e,r =
    // J_e,theta = 0 (only the z components are regular at r = 0).
    if (je_relax_solve) {
        auto const dx_arr = warpx.Geom(lev).CellSizeArray();
        amrex::Real sum_inv_dx2 = 0.0_rt;
        amrex::Real dx_min = dx_arr[0];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            sum_inv_dx2 += 1.0_rt / (dx_arr[d] * dx_arr[d]);
            dx_min = amrex::min(dx_min, dx_arr[d]);
        }
        amrex::Real const dt_full = warpx.getdt(lev);
        amrex::Real const dt_je =
            (hybrid_model->m_je_dt_sub > 0.0_rt)
                ? hybrid_model->m_je_dt_sub : dt_full;
        amrex::Real const rho_1c =
            PhysConst::q_e * hybrid_model->m_je_n_min;
        // Eq-(30) vacuum damping: LOCAL current decay below the
        // one-count level, no CFL (see the member doc).
        amrex::Real const vac_gamma =
            (hybrid_model->m_je_vac_gamma_frac > 0.0_rt)
                ? hybrid_model->m_je_vac_gamma_frac / dt_full : 0.0_rt;
        bool const use_vac_damping = (vac_gamma > 0.0_rt);
        amrex::Real c_art = amrex::min(
            PhysConst::c,
            hybrid_model->m_je_c_frac
                / (dt_je * std::sqrt(sum_inv_dx2)));
        if (hybrid_model->m_je_c_max > 0.0_rt) {
            c_art = amrex::min(c_art, hybrid_model->m_je_c_max);
        }
        // Quasineutral relaxation toward the one-count-guarded Ohm
        // value (see the m_je_qn_frac member doc): pins the slow/
        // longitudinal sector algebraically, plasma-blended.
        amrex::Real const gam_qn =
            hybrid_model->m_je_qn_frac / dt_full;
        bool const use_qn = (gam_qn > 0.0_rt);
        amrex::Real const inv_eps_art = PhysConst::mu0 * c_art * c_art;
        amrex::Real const qm = PhysConst::q_e / PhysConst::m_e;
        bool const use_vm = hybrid_model->m_je_var_mass;
        amrex::Real const vm_alpha = hybrid_model->m_je_alpha;
        // Anomalous viscous friction in Eq-27-capped cells (see the
        // m_je_visc_frac member doc): gamma_visc = gam_vis0 *
        // (1 - m_e/m_e'), identically zero wherever electron physics
        // is resolved. gam_vis0 sets the capped grid mode's
        // dissipation length to je_visc_frac * dx.
        amrex::Real const visc_frac = hybrid_model->m_je_visc_frac;
        bool const use_visc = use_vm && (visc_frac > 0.0_rt);
        amrex::Real const gam_vis0 = use_visc
            ? 2.0_rt * vm_alpha / (visc_frac * dt_je) : 0.0_rt;
        amrex::Real const dx2_min = dx_min * dx_min;
        amrex::Real const mu0_l = PhysConst::mu0;
        MultiFab * qvisc_mf = nullptr;
        if (use_visc && hybrid_model->m_solve_electron_energy_equation) {
            qvisc_mf = warpx.m_fields.get("hybrid_je_qvisc_fp", lev);
        }

        MultiFab & Je_mf = *warpx.m_fields.get("hybrid_je_fp", lev);
        // Staggered-leapfrog phase (see the m_je_time_stagger member
        // doc): 0 = the combined BE advance, 1 = the E update, 2 =
        // the CN J_e update.
        int const stagger_phase = hybrid_model->m_je_stagger_phase;
        bool const seed_je = !hybrid_model->m_je_init;
        hybrid_model->m_je_init = true;
        if (seed_je) {
            amrex::Print() << "[je] relax boot: c_art = " << c_art
                << " m/s (CFL value "
                << hybrid_model->m_je_c_frac
                       / (dt_je * std::sqrt(sum_inv_dx2))
                << ", cap " << hybrid_model->m_je_c_max
                << "), gamma_qn = " << gam_qn
                << " 1/s, dt_sub = " << dt_je
                << " s, var_mass = "
                << (hybrid_model->m_je_var_mass ? "on" : "off")
                << " (alpha " << hybrid_model->m_je_alpha
                << "), split = "
                << (hybrid_model->m_je_time_stagger ? "stagger-cn"
                    : (hybrid_model->m_je_centered_split
                        ? "centered" : "lie"))
                << ", visc_frac = " << hybrid_model->m_je_visc_frac
                << ", n_min = " << hybrid_model->m_je_n_min
                << " m^-3 (must be the one-count level)\n";
        }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid();
             ++mfi) {
            Array4<Real> const& Er = Efield[0]->array(mfi);
            Array4<Real> const& Et = Efield[1]->array(mfi);
            Array4<Real> const& Ez = Efield[2]->array(mfi);
            Array4<Real> const& Je = Je_mf.array(mfi);
            Array4<Real const> const& enE = enE_nodal_mf.const_array(mfi);
            Array4<Real const> const& rho = rhofield.const_array(mfi);
            Array4<Real const> const& Pe  = Pefield.const_array(mfi);
            Array4<Real const> const& Jr = Jfield[0]->const_array(mfi);
            Array4<Real const> const& Jt = Jfield[1]->const_array(mfi);
            Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
            Array4<Real const> const& Jir = Jifield[0]->const_array(mfi);
            Array4<Real const> const& Jit = Jifield[1]->const_array(mfi);
            Array4<Real const> const& Jiz = Jifield[2]->const_array(mfi);
            Array4<Real const> const& Br = Bfield[0]->const_array(mfi);
            Array4<Real const> const& Bt = Bfield[1]->const_array(mfi);
            Array4<Real const> const& Bz = Bfield[2]->const_array(mfi);
            Array4<Real> Br_e, Bt_e, Bz_e;
            if (external_b_split) {
                Br_e = Bfield_external[0]->array(mfi);
                Bt_e = Bfield_external[1]->array(mfi);
                Bz_e = Bfield_external[2]->array(mfi);
            }
            Array4<Real> Er_e, Et_e, Ez_e;
            if (include_E_ext) {
                Er_e = Efield_external[0]->array(mfi);
                Et_e = Efield_external[1]->array(mfi);
                Ez_e = Efield_external[2]->array(mfi);
            }
            amrex::Array4<int> upd_r, upd_t, upd_z;
            if (EB::enabled()) {
                upd_r = eb_update_E[0]->array(mfi);
                upd_t = eb_update_E[1]->array(mfi);
                upd_z = eb_update_E[2]->array(mfi);
            }
            amrex::Array4<Real> qv;
            if (qvisc_mf) { qv = qvisc_mf->array(mfi); }
            Real const * const AMREX_RESTRICT coefs_r =
                m_stencil_coefs_r.dataPtr();
            auto const n_coefs_r =
                static_cast<int>(m_stencil_coefs_r.size());
            Real const * const AMREX_RESTRICT coefs_z =
                m_stencil_coefs_z.dataPtr();
            auto const n_coefs_z =
                static_cast<int>(m_stencil_coefs_z.size());
            Real const dr_l = m_dr;
            Real const rmin_l = m_rmin;
            amrex::Real const dt_r = dt_je;
            amrex::ParallelFor(mfi.tilebox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                bool const skip_r = upd_r && upd_r(i,j,k) == 0;
                bool const skip_t = upd_t && upd_t(i,j,k) == 0;
                bool const skip_z = upd_z && upd_z(i,j,k) == 0;
                if (skip_r && skip_t && skip_z) { return; }
                // RZ mode-0 axis parity (nodal r index): E_r, E_theta
                // and the J_e r/theta components vanish at r = 0.
                bool const on_axis =
                    (rmin_l + i*dr_l) < 0.5_rt*dr_l;
                amrex::Real const rho_v =
                    amrex::max(rho(i,j,k), 0.0_rt);
                amrex::Real br = Br(i,j,k), bt = Bt(i,j,k),
                            bz = Bz(i,j,k);
                amrex::Real etr = Er(i,j,k), ett = Et(i,j,k),
                            etz = Ez(i,j,k);
                if (external_b_split) {
                    br += Br_e(i,j,k); bt += Bt_e(i,j,k);
                    bz += Bz_e(i,j,k);
                }
                if (include_E_ext) {
                    etr += Er_e(i,j,k); ett += Et_e(i,j,k);
                    etz += Ez_e(i,j,k);
                }
                // Eq-(27) variable electron mass (see member doc):
                // per-cell inflation where the grid-whistler speed
                // would violate the substep CFL; inert (== m_e)
                // wherever the physics is resolved. Applies to every
                // 1/m_e in the J_e update below.
                amrex::Real qm_c = qm;
                if (use_vm) {
                    amrex::Real const B2t = br*br + bt*bt + bz*bz;
                    amrex::Real const me_v =
                        HybridPICModel::JeEffectiveElectronMass(
                            B2t, amrex::max(rho_v, rho_1c),
                            dt_r, dx_min, vm_alpha);
                    qm_c = PhysConst::q_e / me_v;
                }
                amrex::Real gam_vis = 0.0_rt;
                if (use_visc) {
                    // m_e / m_e' == qm_c / qm exactly
                    gam_vis = gam_vis0 * (1.0_rt - qm_c / qm);
                    // The drag reaches B as a resistive diffusion with
                    // diffusivity gam_vis * d_e'^2 through the EXPLICIT
                    // substep loop; cap it at that loop's diffusion CFL
                    // (d_e'^2 = 1/(mu0 qm_c rho_g)), else the target
                    // dissipation length destabilizes the sheet it is
                    // meant to protect (measured: 20x violation at a
                    // 0.2 dx target).
                    amrex::Real const gam_cfl = 0.25_rt * dx2_min
                        * mu0_l * qm_c * amrex::max(rho_v, rho_1c)
                        / dt_r;
                    gam_vis = amrex::min(gam_vis, gam_cfl);
                }
                amrex::Real const kE = qm_c * rho_v;
                amrex::Real const gpr = T_Algo::UpwardDr(
                    Pe, coefs_r, n_coefs_r, i, j, 0, 0);
                amrex::Real const gpt = 0.0_rt;   // mode 0
                amrex::Real const gpz = T_Algo::UpwardDz(
                    Pe, coefs_z, n_coefs_z, i, j, 0, 0);
                amrex::Real jtot_val = 0.0_rt;
                if (resistivity_has_J_dependence) {
                    jtot_val = std::sqrt(
                        Jr(i,j,k)*Jr(i,j,k) + Jt(i,j,k)*Jt(i,j,k)
                        + Jz(i,j,k)*Jz(i,j,k));
                }
                amrex::Real const eta_v = eta(rho_v, jtot_val, t_new);
                amrex::Real gam_v = 0.0_rt;
                if (use_vac_damping) {
                    gam_v = vac_gamma * 0.5_rt * (1.0_rt - std::tanh(
                        (rho_v - rho_1c) / (0.5_rt * rho_1c)));
                }
                // Ampere drive Jc - J_i (Jfield = curl B/mu0 - J_ext)
                amrex::Real const rcr = Jr(i,j,k) - Jir(i,j,k);
                amrex::Real const rct = Jt(i,j,k) - Jit(i,j,k);
                amrex::Real const rcz = Jz(i,j,k) - Jiz(i,j,k);
                // previous J_e (seeded from the Ampere closure on the
                // first call)
                amrex::Real const jor = seed_je ? rcr : Je(i,j,k,0);
                amrex::Real const jot = seed_je ? rct : Je(i,j,k,1);
                amrex::Real const joz = seed_je ? rcz : Je(i,j,k,2);
                if (stagger_phase == 1) {
                    // staggered E push (see the m_je_time_stagger
                    // member doc): E^k -> E^{k+1} by Ampere-Maxwell
                    // with the midpoint drive rc(B^{k+1/2}) and
                    // J_e^{k+1/2} -- no implicit solve, the
                    // staggering replaces the BE E-feedback. The qn
                    // pin acts on E and so lives in this phase.
                    if (seed_je) {
                        Je(i,j,k,0) = jor; Je(i,j,k,1) = jot;
                        Je(i,j,k,2) = joz;
                    }
                    amrex::Real ern = Er(i,j,k)
                        + dt_r * inv_eps_art * (rcr - jor);
                    amrex::Real etn = Et(i,j,k)
                        + dt_r * inv_eps_art * (rct - jot);
                    amrex::Real ezn = Ez(i,j,k)
                        + dt_r * inv_eps_art * (rcz - joz);
                    if (use_qn) {
                        amrex::Real const w_qn = 0.5_rt
                            * (1.0_rt + std::tanh(
                                  (rho_v - rho_1c) / (0.5_rt * rho_1c)));
                        amrex::Real const lam = dt_r * gam_qn * w_qn;
                        amrex::Real const rho_gq =
                            amrex::max(rho_v, rho_1c);
                        amrex::Real const inv_l =
                            1.0_rt / (1.0_rt + lam);
                        ern = (ern + lam * ((enE(i,j,k,0) - gpr
                            + rho_v * eta_v * Jr(i,j,k)) / rho_gq))
                            * inv_l;
                        etn = (etn + lam * ((enE(i,j,k,1) - gpt
                            + rho_v * eta_v * Jt(i,j,k)) / rho_gq))
                            * inv_l;
                        ezn = (ezn + lam * ((enE(i,j,k,2) - gpz
                            + rho_v * eta_v * Jz(i,j,k)) / rho_gq))
                            * inv_l;
                    }
                    if (on_axis) { ern = 0.0_rt; etn = 0.0_rt; }
                    if (!skip_r) { Er(i,j,k) = ern; }
                    if (!skip_t) { Et(i,j,k) = etn; }
                    if (!skip_z) { Ez(i,j,k) = ezn; }
                    return;
                }
                if (stagger_phase == 2) {
                    // staggered CN (Cayley) J_e update against the
                    // midpoint E^{k+1} (etr/ett/etz: the registry E,
                    // just advanced, with E_ext folded above) and the
                    // substep-average B^{k+1} (the registry B; see
                    // BfieldEvolve). Gyration is |J_e|-preserving;
                    // damping and drive are trapezoidal.
                    amrex::Real const gam_sum =
                        kE * eta_v + gam_v + gam_vis;
                    amrex::Real const acn =
                        1.0_rt + 0.5_rt * dt_r * gam_sum;
                    amrex::Real const am =
                        1.0_rt - 0.5_rt * dt_r * gam_sum;
                    amrex::Real const chr = 0.5_rt * qm_c * dt_r * br;
                    amrex::Real const cht = 0.5_rt * qm_c * dt_r * bt;
                    amrex::Real const chz = 0.5_rt * qm_c * dt_r * bz;
                    amrex::Real const cb2 = chr*chr + cht*cht + chz*chz;
                    amrex::Real const srr = am*jor + (cht*joz - chz*jot)
                        + dt_r * (kE * etr - kE * eta_v * Jir(i,j,k)
                                  + qm_c * gpr);
                    amrex::Real const srt = am*jot + (chz*jor - chr*joz)
                        + dt_r * (kE * ett - kE * eta_v * Jit(i,j,k)
                                  + qm_c * gpt);
                    amrex::Real const srz = am*joz + (chr*jot - cht*jor)
                        + dt_r * (kE * etz - kE * eta_v * Jiz(i,j,k)
                                  + qm_c * gpz);
                    amrex::Real const cbd = chr*srr + cht*srt + chz*srz;
                    amrex::Real const cinv =
                        1.0_rt / (acn * (acn*acn + cb2));
                    amrex::Real jnr = (acn*acn*srr
                        + acn*(cht*srz - chz*srt) + cbd*chr) * cinv;
                    amrex::Real jnt = (acn*acn*srt
                        + acn*(chz*srr - chr*srz) + cbd*cht) * cinv;
                    amrex::Real const jnz = (acn*acn*srz
                        + acn*(chr*srt - cht*srr) + cbd*chz) * cinv;
                    if (on_axis) { jnr = 0.0_rt; jnt = 0.0_rt; }
                    if (qv) {
                        amrex::Real const j2 =
                            (skip_r ? 0.0_rt : jnr*jnr)
                            + (skip_t ? 0.0_rt : jnt*jnt)
                            + (skip_z ? 0.0_rt : jnz*jnz);
                        qv(i,j,k) += gam_vis * dt_r * j2
                            / (qm_c * amrex::max(rho_v, rho_1c));
                    }
                    if (!skip_r) { Je(i,j,k,0) = jnr; }
                    if (!skip_t) { Je(i,j,k,1) = jnt; }
                    if (!skip_z) { Je(i,j,k,2) = jnz; }
                    return;
                }
                // implicit local solve:
                //   (a I - [beta]x) Je' = r,  beta = qm dt B_tot
                //   inverse = (a^2 I + a [beta]x + beta beta^T)
                //             / (a (a^2 + |beta|^2))
                amrex::Real const a = 1.0_rt
                    + dt_r * (dt_r * kE * inv_eps_art
                              + kE * eta_v + gam_v + gam_vis);
                amrex::Real const ber = qm_c * dt_r * br;
                amrex::Real const bet = qm_c * dt_r * bt;
                amrex::Real const bez = qm_c * dt_r * bz;
                amrex::Real const b2 = ber*ber + bet*bet + bez*bez;
                amrex::Real const rr = jor
                    + dt_r * kE * (etr + dt_r * inv_eps_art * rcr)
                    - dt_r * kE * eta_v * Jir(i,j,k)
                    + qm_c * dt_r * gpr;
                amrex::Real const rt = jot
                    + dt_r * kE * (ett + dt_r * inv_eps_art * rct)
                    - dt_r * kE * eta_v * Jit(i,j,k)
                    + qm_c * dt_r * gpt;
                amrex::Real const rz = joz
                    + dt_r * kE * (etz + dt_r * inv_eps_art * rcz)
                    - dt_r * kE * eta_v * Jiz(i,j,k)
                    + qm_c * dt_r * gpz;
                amrex::Real const bdotr = ber*rr + bet*rt + bez*rz;
                amrex::Real const inv = 1.0_rt / (a * (a*a + b2));
                amrex::Real jnr =
                    (a*a*rr + a*(bet*rz - bez*rt) + bdotr*ber) * inv;
                amrex::Real jnt =
                    (a*a*rt + a*(bez*rr - ber*rz) + bdotr*bet) * inv;
                amrex::Real const jnz =
                    (a*a*rz + a*(ber*rt - bet*rr) + bdotr*bez) * inv;
                amrex::Real ern = Er(i,j,k)
                    + dt_r * inv_eps_art * (rcr - jnr);
                amrex::Real etn = Et(i,j,k)
                    + dt_r * inv_eps_art * (rct - jnt);
                amrex::Real ezn = Ez(i,j,k)
                    + dt_r * inv_eps_art * (rcz - jnz);
                if (use_qn) {
                    // implicit relaxation toward the one-count-guarded
                    // Ohm value, plasma-blended (the OPTIONAL floored
                    // division of the qn pin -- see the member doc)
                    amrex::Real const w_qn = 0.5_rt
                        * (1.0_rt + std::tanh(
                              (rho_v - rho_1c) / (0.5_rt * rho_1c)));
                    amrex::Real const lam = dt_r * gam_qn * w_qn;
                    amrex::Real const rho_gq =
                        amrex::max(rho_v, rho_1c);
                    amrex::Real const inv_l = 1.0_rt / (1.0_rt + lam);
                    ern = (ern + lam * ((enE(i,j,k,0) - gpr
                        + rho_v * eta_v * Jr(i,j,k)) / rho_gq))
                        * inv_l;
                    etn = (etn + lam * ((enE(i,j,k,1) - gpt
                        + rho_v * eta_v * Jt(i,j,k)) / rho_gq))
                        * inv_l;
                    ezn = (ezn + lam * ((enE(i,j,k,2) - gpz
                        + rho_v * eta_v * Jz(i,j,k)) / rho_gq))
                        * inv_l;
                }
                if (on_axis) {
                    jnr = 0.0_rt; jnt = 0.0_rt;
                    ern = 0.0_rt; etn = 0.0_rt;
                }
                if (qv) {
                    // dissipated energy density this substep:
                    // gamma_visc * m_e' |J_e|^2 / (e^2 n_g) * dt,
                    // with m_e'/(e^2 n_g) = 1/(qm_c * rho_g)
                    amrex::Real const j2 =
                        (skip_r ? 0.0_rt : jnr*jnr)
                        + (skip_t ? 0.0_rt : jnt*jnt)
                        + (skip_z ? 0.0_rt : jnz*jnz);
                    qv(i,j,k) += gam_vis * dt_r * j2
                        / (qm_c * amrex::max(rho_v, rho_1c));
                }
                if (!skip_r) { Je(i,j,k,0) = jnr; Er(i,j,k) = ern; }
                if (!skip_t) { Je(i,j,k,1) = jnt; Et(i,j,k) = etn; }
                if (!skip_z) { Je(i,j,k,2) = jnz; Ez(i,j,k) = ezn; }
            });
        }
        return;
    }

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
        Array4<Real> const& Etheta = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real const> const& Jr = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jtheta = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& enE = enE_nodal_mf.const_array(mfi);
        Array4<Real const> eiN;
        if (Ei_nodal_mf) { eiN = Ei_nodal_mf->const_array(mfi); }
        Array4<Real const> const& rho = rhofield.const_array(mfi);
        Array4<Real const> const& Pe = Pefield.const_array(mfi);
        Array4<Real> const& Br = Bfield[0]->array(mfi);
        Array4<Real> const& Btheta = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);
        // Overlay arrays stay default-constructed (never indexed) when no
        // per-species resistivity is registered -- the kernels gate the read
        // on has_eta_overlay.
        Array4<Real const> eta_overlay_r, eta_overlay_t, eta_overlay_z;
        if (has_eta_overlay) {
            eta_overlay_r = eta_overlay_mf[0]->const_array(mfi);
            eta_overlay_t = eta_overlay_mf[1]->const_array(mfi);
            eta_overlay_z = eta_overlay_mf[2]->const_array(mfi);
        }

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries
        amrex::Array4<int> update_Er_arr, update_Etheta_arr, update_Ez_arr;
        if (EB::enabled()) {
            update_Er_arr = eb_update_E[0]->array(mfi);
            update_Etheta_arr = eb_update_E[1]->array(mfi);
            update_Ez_arr = eb_update_E[2]->array(mfi);
        }

        Array4<Real> Er_ext, Etheta_ext, Ez_ext;
        if (include_E_ext) {
            Er_ext = Efield_external[0]->array(mfi);
            Etheta_ext = Efield_external[1]->array(mfi);
            Ez_ext = Efield_external[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_r = m_stencil_coefs_r.dataPtr();
        int const n_coefs_r = static_cast<int>(m_stencil_coefs_r.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

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
                const Real rho_val = Interp(rho, nodal, Er_stag, coarsen, i, j, 0, 0);

                if (rho_val < rho_floor && holmstrom_vacuum_region && !holmstrom_smooth) {
                    Er(i, j, 0) = 0._rt;
                } else {
                    // Get the gradient of the electron pressure if the longitudinal part of
                    // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                    const Real grad_Pe =
                        (solve_for_Faraday ? add_grad_pe_faraday
                                           : include_electron_pressure_term) ?
                        T_Algo::UpwardDr(Pe, coefs_r, n_coefs_r, i, j, 0, 0)
                        : 0._rt;

                    // interpolate the nodal neE values to the Yee grid
                    const auto enE_r = Interp(enE, nodal, Er_stag, coarsen, i, j, 0, 0);

                    // safety condition since we divide by rho
                    const auto rho_val_limited = std::max(rho_val, rho_floor);

                    Real ohm_val = (enE_r - grad_Pe) / rho_val_limited;
                    if (holmstrom_smooth) {
                        ohm_val *= 0.5_rt * (1._rt + std::tanh(
                            (rho_val - rho_floor) * holmstrom_inv_width));
                    }
                    Er(i, j, 0) = ohm_val;
                }
                if (include_electron_inertia) {
                    Er(i, j, 0) += Interp(eiN, nodal, Er_stag, coarsen, i, j, 0, 0);
                }


                // Resistivity: whenever the caller kept eta in this solve
                // (always true for the Faraday solves; the push/stored-E
                // solve follows the caller), or when the Q_ei drag operator
                // carries the ion-side friction (dropping eta J from the
                // push field IS the friction, so E* + drag would book it
                // twice).
                if (include_resistivity || add_resistivity_push) {
                    Real jtot_val = 0._rt;
                    if (resistivity_has_J_dependence) {
                        // Interpolate current to appropriate staggering to match E field
                        const Real jr_val = Jr(i, j, 0);
                        const Real jtheta_val = Interp(Jtheta, Jtheta_stag, Er_stag, coarsen, i, j, 0, 0);
                        const Real jz_val = Interp(Jz, Jz_stag, Er_stag, coarsen, i, j, 0, 0);
                        jtot_val = std::sqrt(jr_val*jr_val + jtheta_val*jtheta_val + jz_val*jz_val);
                    }

                    Er(i, j, 0) += eta(rho_val, jtot_val, t_new) * Jr(i, j, 0);
                    // Per-species resistive overlay (Phys. Plasmas 31, 012902 (2024)); zero
                    // when no per-species eta is registered.
                    if (has_eta_overlay) { Er(i, j, 0) += eta_overlay_r(i, j, 0); }

                    if (include_hyper_resistivity_term && solve_for_Faraday) {

                        // Interpolate B field to appropriate staggering to match E field
                        Real btot_val = 0._rt;
                        if (hyper_resistivity_has_B_dependence) {
                            const Real br_val = Interp(Br, Br_stag, Er_stag, coarsen, i, j, 0, 0);
                            const Real bt_val = Interp(Btheta, Btheta_stag, Er_stag, coarsen, i, j, 0, 0);
                            const Real bz_val = Interp(Bz, Bz_stag, Er_stag, coarsen, i, j, 0, 0);
                            btot_val = std::sqrt(br_val*br_val + bt_val*bt_val + bz_val*bz_val);
                        }

                        // r on cell-centered point (Jr is cell-centered in r)
                        const Real r = rmin + (i + 0.5_rt)*dr;
                        auto nabla2Jr = T_Algo::Dr_rDr_over_r(Jr, r, dr, coefs_r, n_coefs_r, i, j, 0, 0)
                            + T_Algo::Dzz(Jr, coefs_z, n_coefs_z, i, j, 0, 0) - Jr(i, j, 0)/(r*r);

                        Er(i, j, 0) -= eta_h(rho_val, btot_val) * nabla2Jr;
                    }
                }

                if (include_E_ext && (subtract_E_ext_everywhere || rho_val >= rho_floor)) {
                    Er(i, j, 0) -= Er_ext(i, j, 0);
                }
            },

            // Etheta calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Etheta_arr && update_Etheta_arr(i, j, 0) == 0) { return; }

                // r on a nodal grid (Etheta is nodal in r)
                Real const r = rmin + i*dr;
                // Mode m=0: // Ensure that Etheta remains 0 on axis
                if (r < 0.5_rt*dr) {
                    Etheta(i, j, 0, 0) = 0.;
                    return;
                }

                // Interpolate to get the appropriate charge density in space
                const Real rho_val = Interp(rho, nodal, Etheta_stag, coarsen, i, j, 0, 0);

                if (rho_val < rho_floor && holmstrom_vacuum_region && !holmstrom_smooth) {
                    Etheta(i, j, 0) = 0._rt;
                } else {
                    // Get the gradient of the electron pressure
                    // -> d/dt = 0 for m = 0
                    const auto grad_Pe = 0.0_rt;

                    // interpolate the nodal neE values to the Yee grid
                    const auto enE_t = Interp(enE, nodal, Etheta_stag, coarsen, i, j, 0, 1);

                    // safety condition since we divide by rho
                    const auto rho_val_limited = std::max(rho_val, rho_floor);

                    Real ohm_val = (enE_t - grad_Pe) / rho_val_limited;
                    if (holmstrom_smooth) {
                        ohm_val *= 0.5_rt * (1._rt + std::tanh(
                            (rho_val - rho_floor) * holmstrom_inv_width));
                    }
                    Etheta(i, j, 0) = ohm_val;
                }
                if (include_electron_inertia) {
                    Etheta(i, j, 0) += Interp(eiN, nodal, Etheta_stag, coarsen, i, j, 0, 1);
                }


                // Resistivity: whenever the caller kept eta in this solve
                // (always true for the Faraday solves; the push/stored-E
                // solve follows the caller), or when the Q_ei drag operator
                // carries the ion-side friction (dropping eta J from the
                // push field IS the friction, so E* + drag would book it
                // twice).
                if (include_resistivity || add_resistivity_push) {
                    Real jtot_val = 0._rt;
                    if(resistivity_has_J_dependence) {
                        // Interpolate current to appropriate staggering to match E field
                        const Real jr_val = Interp(Jr, Jr_stag, Etheta_stag, coarsen, i, j, 0, 0);
                        const Real jtheta_val = Jtheta(i, j, 0);
                        const Real jz_val = Interp(Jz, Jz_stag, Etheta_stag, coarsen, i, j, 0, 0);
                        jtot_val = std::sqrt(jr_val*jr_val + jtheta_val*jtheta_val + jz_val*jz_val);
                    }

                    Etheta(i, j, 0) += eta(rho_val, jtot_val, t_new) * Jtheta(i, j, 0);
                    if (has_eta_overlay) { Etheta(i, j, 0) += eta_overlay_t(i, j, 0); }

                    if (include_hyper_resistivity_term && solve_for_Faraday) {

                        // Interpolate B field to appropriate staggering to match E field
                        Real btot_val = 0._rt;
                        if (hyper_resistivity_has_B_dependence) {
                            const Real br_val = Interp(Br, Br_stag, Etheta_stag, coarsen, i, j, 0, 0);
                            const Real bt_val = Interp(Btheta, Btheta_stag, Etheta_stag, coarsen, i, j, 0, 0);
                            const Real bz_val = Interp(Bz, Bz_stag, Etheta_stag, coarsen, i, j, 0, 0);
                            btot_val = std::sqrt(br_val*br_val + bt_val*bt_val + bz_val*bz_val);
                        }

                        // Special handling of the hyper-resistivity term on axis to avoid division by zero
                        // and ensure that Etheta remains 0 on axis for m=0 mode
                        auto nabla2Jtheta = 0.0_rt;
                        if (r > 0.0_rt) {
                            nabla2Jtheta = T_Algo::Dr_rDr_over_r(Jtheta, r, dr, coefs_r, n_coefs_r, i, j, 0, 0)
                                + T_Algo::Dzz(Jtheta, coefs_z, n_coefs_z, i, j, 0, 0) - Jtheta(i, j, 0)/(r*r);
                        }

                        Etheta(i, j, 0) -= eta_h(rho_val, btot_val) * nabla2Jtheta;
                    }
                }

                if (include_E_ext && (subtract_E_ext_everywhere || rho_val >= rho_floor)) {
                    Etheta(i, j, 0) -= Etheta_ext(i, j, 0);
                }
            },

            // Ez calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Ez_arr && update_Ez_arr(i, j, 0) == 0) { return; }

                // Interpolate to get the appropriate charge density in space
                const Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, 0, 0);

                if (rho_val < rho_floor && holmstrom_vacuum_region && !holmstrom_smooth) {
                    Ez(i, j, 0) = 0._rt;
                } else {
                    // Get the gradient of the electron pressure if the longitudinal part of
                    // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                    const Real grad_Pe =
                        (solve_for_Faraday ? add_grad_pe_faraday
                                           : include_electron_pressure_term) ?
                        T_Algo::UpwardDz(Pe, coefs_z, n_coefs_z, i, j, 0, 0)
                        : 0._rt;

                    // interpolate the nodal neE values to the Yee grid
                    const auto enE_z = Interp(enE, nodal, Ez_stag, coarsen, i, j, 0, 2);

                    // safety condition since we divide by rho
                    const auto rho_val_limited = std::max(rho_val, rho_floor);

                    Real ohm_val = (enE_z - grad_Pe) / rho_val_limited;
                    if (holmstrom_smooth) {
                        ohm_val *= 0.5_rt * (1._rt + std::tanh(
                            (rho_val - rho_floor) * holmstrom_inv_width));
                    }
                    Ez(i, j, 0) = ohm_val;
                }
                if (include_electron_inertia) {
                    Ez(i, j, 0) += Interp(eiN, nodal, Ez_stag, coarsen, i, j, 0, 2);
                }


                // Resistivity: whenever the caller kept eta in this solve
                // (always true for the Faraday solves; the push/stored-E
                // solve follows the caller), or when the Q_ei drag operator
                // carries the ion-side friction (dropping eta J from the
                // push field IS the friction, so E* + drag would book it
                // twice).
                if (include_resistivity || add_resistivity_push) {
                    Real jtot_val = 0._rt;
                    if (resistivity_has_J_dependence) {
                        // Interpolate current to appropriate staggering to match E field
                        const Real jr_val = Interp(Jr, Jr_stag, Ez_stag, coarsen, i, j, 0, 0);
                        const Real jtheta_val = Interp(Jtheta, Jtheta_stag, Ez_stag, coarsen, i, j, 0, 0);
                        const Real jz_val = Jz(i, j, 0);
                        jtot_val = std::sqrt(jr_val*jr_val + jtheta_val*jtheta_val + jz_val*jz_val);
                    }

                    Ez(i, j, 0) += eta(rho_val, jtot_val, t_new) * Jz(i, j, 0);
                    if (has_eta_overlay) { Ez(i, j, 0) += eta_overlay_z(i, j, 0); }

                    if (include_hyper_resistivity_term && solve_for_Faraday) {

                        // Interpolate B field to appropriate staggering to match E field
                        Real btot_val = 0._rt;
                        if (hyper_resistivity_has_B_dependence) {
                            const Real br_val = Interp(Br, Br_stag, Ez_stag, coarsen, i, j, 0, 0);
                            const Real bt_val = Interp(Btheta, Btheta_stag, Ez_stag, coarsen, i, j, 0, 0);
                            const Real bz_val = Interp(Bz, Bz_stag, Ez_stag, coarsen, i, j, 0, 0);
                            btot_val = std::sqrt(br_val*br_val + bt_val*bt_val + bz_val*bz_val);
                        }

                        // r on nodal point (Jz is nodal in r)
                        const Real r = rmin + i*dr;

                        auto nabla2Jz = T_Algo::Dzz(Jz, coefs_z, n_coefs_z, i, j, 0, 0);
                        if (r > 0.5_rt*dr) {
                            nabla2Jz += T_Algo::Dr_rDr_over_r(Jz, r, dr, coefs_r, n_coefs_r, i, j, 0, 0);
                        } else {
                            // Special handling of the hyper-resistivity term on axis to avoid division by zero
                            // and ensure that Jz remains well-behaved on axis for m=0 mode
                            // This works since there is a symmetry condition on axis that cancels the geometric 1/r term
                            nabla2Jz += T_Algo::Drr(Jz, coefs_r, n_coefs_r, i, j, 0, 0);
                        }

                        Ez(i, j, 0) -= eta_h(rho_val, btot_val) * nabla2Jz;
                    }
                }

                if (include_E_ext && (subtract_E_ext_everywhere || rho_val >= rho_floor)) {
                    Ez(i, j, 0) -= Ez_ext(i, j, 0);
                }
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

#elif defined(WARPX_DIM_RSPHERE)
template<typename T_Algo>
void FiniteDifferenceSolver::HybridPICSolveESpherical (
    ablastr::fields::VectorField const& /*Efield*/,
    ablastr::fields::VectorField const& /*Jfield*/,
    ablastr::fields::VectorField const& /*Jifield*/,
    ablastr::fields::VectorField const& /*Bfield*/,
    amrex::MultiFab const& /*rhofield*/,
    amrex::MultiFab const& /*Pefield*/,
    int /*lev*/, HybridPICModel const* /*hybrid_model*/,
    const bool /*solve_for_Faraday*/, const bool /*include_resistivity*/ )
{
    WARPX_ABORT_WITH_MESSAGE("HybridPICSolveESphrical not fully implemented");
}
#else

template<typename T_Algo>
void FiniteDifferenceSolver::HybridPICSolveECartesian (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev, HybridPICModel const* hybrid_model,
    const bool solve_for_Faraday, const bool include_resistivity )
{
    // Je-form relax advance (esolve = je_form): the division-free
    // electron-momentum E advance (see the m_je_advance member doc).
    // Requirements are checked here; the advance itself runs after the
    // nodal enE assembly below and returns before the legacy
    // divided-Ohm path.
    bool const je_relax_solve =
        hybrid_model->m_esolve_je && (hybrid_model->m_je_advance == 1);
    if (je_relax_solve) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::grid_type == ablastr::utils::enums::GridType::Collocated,
            "hybrid_pic_model.esolve = je_form (v1) requires "
            "warpx.grid_type = collocated");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !hybrid_model->m_add_external_fields ||
                hybrid_model->m_external_field_mode ==
                    HybridPICModel::ExternalFieldMode::Split,
            "esolve = je_form with external fields supports the Split "
            "external-field mode only on this fork (TotalAssembled and "
            "UnifiedA are untested with the relax advance)");
        // The RK substep STAGES and particle-E calls return the frozen
        // state (see the m_je_relax_advance member doc): only the
        // once-per-accepted-substep advance raised by BfieldEvolve
        // mutates (E, J_e). Returning here also skips the enE assembly
        // below on every no-op call.
        if (!hybrid_model->m_je_relax_advance) { return; }
    }

    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    using namespace ablastr::coarsen::sample;

    // get hybrid model parameters
    const auto eta = hybrid_model->m_eta;
    const auto eta_h = hybrid_model->m_eta_h;
    const auto rho_floor = hybrid_model->m_n_floor * PhysConst::q_e;
    const auto resistivity_has_J_dependence = hybrid_model->m_resistivity_has_J_dependence;
    const auto hyper_resistivity_has_B_dependence = hybrid_model->m_hyper_resistivity_has_B_dependence;
    const bool include_hyper_resistivity_term = hybrid_model->m_include_hyper_resistivity_term;
    const bool include_electron_inertia = hybrid_model->m_include_electron_inertia;

    const bool include_external_fields = hybrid_model->m_add_external_fields
        && !hybrid_model->m_external_unified;
    const bool include_hall_term = hybrid_model->m_include_hall_term;
    const bool include_electron_pressure_term =
        hybrid_model->m_include_electron_pressure_term;
    // The stored electric field follows the split-field convention in both
    // schemes: the inductive E_ext is subtracted from plasma cells (where
    // the generalized Ohm's law itself is the electric field and the
    // external drive must not be double-counted), and the caller re-adds
    // E_ext when assembling the particle-push field and advances the
    // external flux analytically from A(t). The modes differ only in the
    // Hall-term field: the explicit scheme keeps Bfield_fp plasma-only
    // during the advance so the kernels add B_ext here; the implicit
    // scheme hands the kernels the total field already.
    const bool external_split = hybrid_model->m_external_split;

    // External-field convention (see HybridPICModel::ExternalFieldMode):
    // only the Split mode assembles totals inside the kernels by adding
    // B_ext; TotalAssembled consumers hand the kernels total B already, and
    // UnifiedA consumers never touch the external-field registers. The
    // E_ext subtraction is gated per cell on rho >= rho_floor unless the
    // consumer requires the smooth unconditional form (matrix-free
    // implicit residuals).
    const bool external_b_split = include_external_fields &&
        (hybrid_model->m_external_field_mode ==
         HybridPICModel::ExternalFieldMode::Split);
    const bool include_E_ext = include_external_fields &&
        (hybrid_model->m_external_field_mode !=
         HybridPICModel::ExternalFieldMode::UnifiedA);
    const bool subtract_E_ext_everywhere = hybrid_model->m_subtract_E_ext_everywhere;

    const bool holmstrom_vacuum_region = hybrid_model->m_holmstrom_vacuum_region;
    // Smooth Hall/grad-Pe turn-off across the vacuum gate: the binary branch
    // makes the residual discontinuous in the state exactly where cells
    // straddle the gate (Newton limit-cycles and grid-scale E jumps at the
    // separatrix edge); a tanh blend over holmstrom_transition_width*n_floor
    // restores smoothness. Width 0 (default) keeps the hard branch.
    const Real holmstrom_inv_width =
        (hybrid_model->m_holmstrom_transition_width > 0._rt)
        ? 1._rt / (hybrid_model->m_holmstrom_transition_width * rho_floor)
        : 0._rt;
    const bool holmstrom_smooth =
        holmstrom_vacuum_region && (holmstrom_inv_width > 0._rt);

    // Energy-equation-era gating (see the drag/battery ledger in the
    // HybridPICModel docs):
    //  * grad Pe stays in the FARADAY solves too when the Biermann battery
    //    is kept (curl(grad Pe/(e n)) = (grad Pe x grad n)/(e n^2) != 0
    //    once Te decouples from n);
    //  * eta J enters the PUSH solve when the Q_ei drag operator is on --
    //    dropping it is itself the single-species ion-side friction, so
    //    E* + drag would book the friction twice. Hyper-resistivity is a
    //    numerical B smoother and stays Faraday-only in either mode.
    const bool add_grad_pe_faraday = hybrid_model->m_include_biermann_battery
        && hybrid_model->m_include_electron_pressure_term;
    const bool add_resistivity_push =
        hybrid_model->m_include_temperature_relaxation;

    auto & warpx = WarpX::GetInstance();
    const amrex::Real t_new = warpx.gett_new(lev);
    // Nodal electron-inertia field, assembled by the caller each
    // evaluation (theta-implicit hybrid only; stays zero elsewhere).
    amrex::MultiFab const * Ei_nodal_mf = include_electron_inertia
        ? warpx.m_fields.get("hybrid_E_inertial_nodal", lev) : nullptr;
    ablastr::fields::VectorField Bfield_external, Efield_external;
    if (include_external_fields) {
        Bfield_external = warpx.m_fields.get_alldirs(FieldType::hybrid_B_fp_external, 0); // lev=0
        Efield_external = warpx.m_fields.get_alldirs(FieldType::hybrid_E_fp_external, 0); // lev=0
    }

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

    // Per-species resistive overlay added to Ohm's-law E alongside +eta_global J.
    // Computed once per step into the registered hybrid_eta_overlay_fp fields
    // and only READ here; see HybridPICSolveECylindrical (RZ branch) for the
    // design notes. When no per-species parser is registered the fields are
    // not allocated and the per-cell add is skipped (bit-identical
    // single-eta path).
    const bool has_eta_overlay = hybrid_model->m_has_per_species_eta;
    ablastr::fields::VectorField eta_overlay_mf = {nullptr, nullptr, nullptr};
    if (has_eta_overlay) {
        eta_overlay_mf = warpx.m_fields.get_alldirs("hybrid_eta_overlay_fp", lev);
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
        Array4<Real const> const& Jx = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jy = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& Jix = Jifield[0]->const_array(mfi);
        Array4<Real const> const& Jiy = Jifield[1]->const_array(mfi);
        Array4<Real const> const& Jiz = Jifield[2]->const_array(mfi);
        Array4<Real const> const& Bx = Bfield[0]->const_array(mfi);
        Array4<Real const> const& By = Bfield[1]->const_array(mfi);
        Array4<Real const> const& Bz = Bfield[2]->const_array(mfi);

        Array4<Real> Bx_ext, By_ext, Bz_ext;
        if (external_b_split) {
            Bx_ext = Bfield_external[0]->array(mfi);
            By_ext = Bfield_external[1]->array(mfi);
            Bz_ext = Bfield_external[2]->array(mfi);
        }

        // Loop over the cells and update the nodal E field
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k){

            // interpolate the total plasma current to a nodal grid
            auto const jx_interp = Interp(Jx, Jx_stag, nodal, coarsen, i, j, k, 0);
            auto const jy_interp = Interp(Jy, Jy_stag, nodal, coarsen, i, j, k, 0);
            auto const jz_interp = Interp(Jz, Jz_stag, nodal, coarsen, i, j, k, 0);

            // interpolate the ion current to a nodal grid
            auto const jix_interp = Interp(Jix, Jx_stag, nodal, coarsen, i, j, k, 0);
            auto const jiy_interp = Interp(Jiy, Jy_stag, nodal, coarsen, i, j, k, 0);
            auto const jiz_interp = Interp(Jiz, Jz_stag, nodal, coarsen, i, j, k, 0);

            // interpolate the B field to a nodal grid
            auto Bx_interp = Interp(Bx, Bx_stag, nodal, coarsen, i, j, k, 0);
            auto By_interp = Interp(By, By_stag, nodal, coarsen, i, j, k, 0);
            auto Bz_interp = Interp(Bz, Bz_stag, nodal, coarsen, i, j, k, 0);

            if (external_b_split) {
                Bx_interp += Interp(Bx_ext, Bx_stag, nodal, coarsen, i, j, k, 0);
                By_interp += Interp(By_ext, By_stag, nodal, coarsen, i, j, k, 0);
                Bz_interp += Interp(Bz_ext, Bz_stag, nodal, coarsen, i, j, k, 0);
            }

            // calculate enE = (J - Ji) x B (without the Hall term the total
            // current drops out and this is the ideal -u_i x B motional
            // field)
            const Real jex = (include_hall_term ? jx_interp : 0.0_rt) - jix_interp;
            const Real jey = (include_hall_term ? jy_interp : 0.0_rt) - jiy_interp;
            const Real jez = (include_hall_term ? jz_interp : 0.0_rt) - jiz_interp;
            enE_nodal(i, j, k, 0) = (
                jey * Bz_interp
                - jez * By_interp
            );
            enE_nodal(i, j, k, 1) = (
                jez * Bx_interp
                - jex * Bz_interp
            );
            enE_nodal(i, j, k, 2) = (
                jex * By_interp
                - jey * Bx_interp
            );
        });

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

    // ------------------------------------------------------------------
    // Je-form relax advance (esolve = je_form, je_advance = relax): J_e
    // is a persistent dynamical field and E advances by Ampere-Maxwell
    // with displacement current at the CFL-set artificial light speed;
    // the stiff LOCAL electron-momentum terms (E coupling, gyration,
    // friction, vacuum decay) are solved pointwise implicitly in closed
    // form -- division-free, no linear algebra, no density floor. E is
    // the full physical field state (grad Pe always included); with
    // Split external fields the state stays internal-only and the
    // physics uses E_state + E_ext (quasi-static over a substep).
    // Steady state recovers generalized Ohm + Ampere exactly. Reached
    // once per ACCEPTED substep (the advance gate was checked at entry);
    // returns before the legacy divided-Ohm path below.
    if (je_relax_solve) {
        auto const dx_arr = warpx.Geom(lev).CellSizeArray();
        amrex::Real sum_inv_dx2 = 0.0_rt;
        amrex::Real dx_min = dx_arr[0];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            sum_inv_dx2 += 1.0_rt / (dx_arr[d] * dx_arr[d]);
            dx_min = amrex::min(dx_min, dx_arr[d]);
        }
        amrex::Real const dt_full = warpx.getdt(lev);
        amrex::Real const dt_je =
            (hybrid_model->m_je_dt_sub > 0.0_rt)
                ? hybrid_model->m_je_dt_sub : dt_full;
        amrex::Real const rho_1c =
            PhysConst::q_e * hybrid_model->m_je_n_min;
        // Eq-(30) vacuum damping: LOCAL current decay below the
        // one-count level, no CFL (see the member doc).
        amrex::Real const vac_gamma =
            (hybrid_model->m_je_vac_gamma_frac > 0.0_rt)
                ? hybrid_model->m_je_vac_gamma_frac / dt_full : 0.0_rt;
        bool const use_vac_damping = (vac_gamma > 0.0_rt);
        amrex::Real c_art = amrex::min(
            PhysConst::c,
            hybrid_model->m_je_c_frac
                / (dt_je * std::sqrt(sum_inv_dx2)));
        if (hybrid_model->m_je_c_max > 0.0_rt) {
            c_art = amrex::min(c_art, hybrid_model->m_je_c_max);
        }
        // Quasineutral relaxation toward the one-count-guarded Ohm
        // value (see the m_je_qn_frac member doc): pins the slow/
        // longitudinal sector algebraically, plasma-blended.
        amrex::Real const gam_qn =
            hybrid_model->m_je_qn_frac / dt_full;
        bool const use_qn = (gam_qn > 0.0_rt);
        amrex::Real const inv_eps_art = PhysConst::mu0 * c_art * c_art;
        amrex::Real const qm = PhysConst::q_e / PhysConst::m_e;
        bool const use_vm = hybrid_model->m_je_var_mass;
        amrex::Real const vm_alpha = hybrid_model->m_je_alpha;
        // Anomalous viscous friction in Eq-27-capped cells (see the
        // m_je_visc_frac member doc): gamma_visc = gam_vis0 *
        // (1 - m_e/m_e'), identically zero wherever electron physics
        // is resolved. gam_vis0 sets the capped grid mode's
        // dissipation length to je_visc_frac * dx.
        amrex::Real const visc_frac = hybrid_model->m_je_visc_frac;
        bool const use_visc = use_vm && (visc_frac > 0.0_rt);
        amrex::Real const gam_vis0 = use_visc
            ? 2.0_rt * vm_alpha / (visc_frac * dt_je) : 0.0_rt;
        amrex::Real const dx2_min = dx_min * dx_min;
        amrex::Real const mu0_l = PhysConst::mu0;
        MultiFab * qvisc_mf = nullptr;
        if (use_visc && hybrid_model->m_solve_electron_energy_equation) {
            qvisc_mf = warpx.m_fields.get("hybrid_je_qvisc_fp", lev);
        }

        MultiFab & Je_mf = *warpx.m_fields.get("hybrid_je_fp", lev);
        // Staggered-leapfrog phase (see the m_je_time_stagger member
        // doc): 0 = the combined BE advance, 1 = the E update, 2 =
        // the CN J_e update.
        int const stagger_phase = hybrid_model->m_je_stagger_phase;
        bool const seed_je = !hybrid_model->m_je_init;
        hybrid_model->m_je_init = true;
        if (seed_je) {
            amrex::Print() << "[je] relax boot: c_art = " << c_art
                << " m/s (CFL value "
                << hybrid_model->m_je_c_frac
                       / (dt_je * std::sqrt(sum_inv_dx2))
                << ", cap " << hybrid_model->m_je_c_max
                << "), gamma_qn = " << gam_qn
                << " 1/s, dt_sub = " << dt_je
                << " s, var_mass = "
                << (hybrid_model->m_je_var_mass ? "on" : "off")
                << " (alpha " << hybrid_model->m_je_alpha
                << "), split = "
                << (hybrid_model->m_je_time_stagger ? "stagger-cn"
                    : (hybrid_model->m_je_centered_split
                        ? "centered" : "lie"))
                << ", visc_frac = " << hybrid_model->m_je_visc_frac
                << ", n_min = " << hybrid_model->m_je_n_min
                << " m^-3 (must be the one-count level)\n";
        }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid();
             ++mfi) {
            Array4<Real> const& Ex = Efield[0]->array(mfi);
            Array4<Real> const& Ey = Efield[1]->array(mfi);
            Array4<Real> const& Ez = Efield[2]->array(mfi);
            Array4<Real> const& Je = Je_mf.array(mfi);
            Array4<Real const> const& enE = enE_nodal_mf.const_array(mfi);
            Array4<Real const> const& rho = rhofield.const_array(mfi);
            Array4<Real const> const& Pe  = Pefield.const_array(mfi);
            Array4<Real const> const& Jx = Jfield[0]->const_array(mfi);
            Array4<Real const> const& Jy = Jfield[1]->const_array(mfi);
            Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
            Array4<Real const> const& Jix = Jifield[0]->const_array(mfi);
            Array4<Real const> const& Jiy = Jifield[1]->const_array(mfi);
            Array4<Real const> const& Jiz = Jifield[2]->const_array(mfi);
            Array4<Real const> const& Bx = Bfield[0]->const_array(mfi);
            Array4<Real const> const& By = Bfield[1]->const_array(mfi);
            Array4<Real const> const& Bz = Bfield[2]->const_array(mfi);
            Array4<Real> Bx_e, By_e, Bz_e;
            if (external_b_split) {
                Bx_e = Bfield_external[0]->array(mfi);
                By_e = Bfield_external[1]->array(mfi);
                Bz_e = Bfield_external[2]->array(mfi);
            }
            Array4<Real> Ex_e, Ey_e, Ez_e;
            if (include_E_ext) {
                Ex_e = Efield_external[0]->array(mfi);
                Ey_e = Efield_external[1]->array(mfi);
                Ez_e = Efield_external[2]->array(mfi);
            }
            amrex::Array4<int> upd_x, upd_y, upd_z;
            if (EB::enabled()) {
                upd_x = eb_update_E[0]->array(mfi);
                upd_y = eb_update_E[1]->array(mfi);
                upd_z = eb_update_E[2]->array(mfi);
            }
            amrex::Array4<Real> qv;
            if (qvisc_mf) { qv = qvisc_mf->array(mfi); }
            Real const * const AMREX_RESTRICT coefs_x =
                m_stencil_coefs_x.dataPtr();
            auto const n_coefs_x =
                static_cast<int>(m_stencil_coefs_x.size());
            Real const * const AMREX_RESTRICT coefs_y =
                m_stencil_coefs_y.dataPtr();
            auto const n_coefs_y =
                static_cast<int>(m_stencil_coefs_y.size());
            Real const * const AMREX_RESTRICT coefs_z =
                m_stencil_coefs_z.dataPtr();
            auto const n_coefs_z =
                static_cast<int>(m_stencil_coefs_z.size());
            amrex::Real const dt_r = dt_je;
            amrex::ParallelFor(mfi.tilebox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                bool const skip_x = upd_x && upd_x(i,j,k) == 0;
                bool const skip_y = upd_y && upd_y(i,j,k) == 0;
                bool const skip_z = upd_z && upd_z(i,j,k) == 0;
                if (skip_x && skip_y && skip_z) { return; }
                amrex::Real const rho_v =
                    amrex::max(rho(i,j,k), 0.0_rt);
                amrex::Real bx = Bx(i,j,k), by = By(i,j,k),
                            bz = Bz(i,j,k);
                amrex::Real etx = Ex(i,j,k), ety = Ey(i,j,k),
                            etz = Ez(i,j,k);
                if (external_b_split) {
                    bx += Bx_e(i,j,k); by += By_e(i,j,k);
                    bz += Bz_e(i,j,k);
                }
                if (include_E_ext) {
                    etx += Ex_e(i,j,k); ety += Ey_e(i,j,k);
                    etz += Ez_e(i,j,k);
                }
                // Eq-(27) variable electron mass (see member doc):
                // per-cell inflation where the grid-whistler speed
                // would violate the substep CFL; inert (== m_e)
                // wherever the physics is resolved. Applies to every
                // 1/m_e in the J_e update below.
                amrex::Real qm_c = qm;
                if (use_vm) {
                    amrex::Real const B2t = bx*bx + by*by + bz*bz;
                    amrex::Real const me_v =
                        HybridPICModel::JeEffectiveElectronMass(
                            B2t, amrex::max(rho_v, rho_1c),
                            dt_r, dx_min, vm_alpha);
                    qm_c = PhysConst::q_e / me_v;
                }
                amrex::Real gam_vis = 0.0_rt;
                if (use_visc) {
                    // m_e / m_e' == qm_c / qm exactly
                    gam_vis = gam_vis0 * (1.0_rt - qm_c / qm);
                    // The drag reaches B as a resistive diffusion with
                    // diffusivity gam_vis * d_e'^2 through the EXPLICIT
                    // substep loop; cap it at that loop's diffusion CFL
                    // (d_e'^2 = 1/(mu0 qm_c rho_g)), else the target
                    // dissipation length destabilizes the sheet it is
                    // meant to protect (measured: 20x violation at a
                    // 0.2 dx target).
                    amrex::Real const gam_cfl = 0.25_rt * dx2_min
                        * mu0_l * qm_c * amrex::max(rho_v, rho_1c)
                        / dt_r;
                    gam_vis = amrex::min(gam_vis, gam_cfl);
                }
                amrex::Real const kE = qm_c * rho_v;
                amrex::Real const gpx = T_Algo::UpwardDx(
                    Pe, coefs_x, n_coefs_x, i, j, k);
#if defined(WARPX_DIM_3D)
                amrex::Real const gpy = T_Algo::UpwardDy(
                    Pe, coefs_y, n_coefs_y, i, j, k);
#else
                amrex::Real const gpy = 0.0_rt;
                amrex::ignore_unused(coefs_y, n_coefs_y);
#endif
                amrex::Real const gpz = T_Algo::UpwardDz(
                    Pe, coefs_z, n_coefs_z, i, j, k);
                amrex::Real jtot_val = 0.0_rt;
                if (resistivity_has_J_dependence) {
                    jtot_val = std::sqrt(
                        Jx(i,j,k)*Jx(i,j,k) + Jy(i,j,k)*Jy(i,j,k)
                        + Jz(i,j,k)*Jz(i,j,k));
                }
                amrex::Real const eta_v = eta(rho_v, jtot_val, t_new);
                amrex::Real gam_v = 0.0_rt;
                if (use_vac_damping) {
                    gam_v = vac_gamma * 0.5_rt * (1.0_rt - std::tanh(
                        (rho_v - rho_1c) / (0.5_rt * rho_1c)));
                }
                // Ampere drive Jc - J_i (Jfield = curl B/mu0 - J_ext)
                amrex::Real const rcx = Jx(i,j,k) - Jix(i,j,k);
                amrex::Real const rcy = Jy(i,j,k) - Jiy(i,j,k);
                amrex::Real const rcz = Jz(i,j,k) - Jiz(i,j,k);
                // previous J_e (seeded from the Ampere closure on the
                // first call)
                amrex::Real const jox = seed_je ? rcx : Je(i,j,k,0);
                amrex::Real const joy = seed_je ? rcy : Je(i,j,k,1);
                amrex::Real const joz = seed_je ? rcz : Je(i,j,k,2);
                if (stagger_phase == 1) {
                    // staggered E push (see the m_je_time_stagger
                    // member doc): E^k -> E^{k+1} by Ampere-Maxwell
                    // with the midpoint drive rc(B^{k+1/2}) and
                    // J_e^{k+1/2} -- no implicit solve, the
                    // staggering replaces the BE E-feedback. The qn
                    // pin acts on E and so lives in this phase.
                    if (seed_je) {
                        Je(i,j,k,0) = jox; Je(i,j,k,1) = joy;
                        Je(i,j,k,2) = joz;
                    }
                    amrex::Real exn = Ex(i,j,k)
                        + dt_r * inv_eps_art * (rcx - jox);
                    amrex::Real eyn = Ey(i,j,k)
                        + dt_r * inv_eps_art * (rcy - joy);
                    amrex::Real ezn = Ez(i,j,k)
                        + dt_r * inv_eps_art * (rcz - joz);
                    if (use_qn) {
                        amrex::Real const w_qn = 0.5_rt
                            * (1.0_rt + std::tanh(
                                  (rho_v - rho_1c) / (0.5_rt * rho_1c)));
                        amrex::Real const lam = dt_r * gam_qn * w_qn;
                        amrex::Real const rho_gq =
                            amrex::max(rho_v, rho_1c);
                        amrex::Real const inv_l =
                            1.0_rt / (1.0_rt + lam);
                        exn = (exn + lam * ((enE(i,j,k,0) - gpx
                            + rho_v * eta_v * Jx(i,j,k)) / rho_gq))
                            * inv_l;
                        eyn = (eyn + lam * ((enE(i,j,k,1) - gpy
                            + rho_v * eta_v * Jy(i,j,k)) / rho_gq))
                            * inv_l;
                        ezn = (ezn + lam * ((enE(i,j,k,2) - gpz
                            + rho_v * eta_v * Jz(i,j,k)) / rho_gq))
                            * inv_l;
                    }
                    if (!skip_x) { Ex(i,j,k) = exn; }
                    if (!skip_y) { Ey(i,j,k) = eyn; }
                    if (!skip_z) { Ez(i,j,k) = ezn; }
                    return;
                }
                if (stagger_phase == 2) {
                    // staggered CN (Cayley) J_e update against the
                    // midpoint E^{k+1} (etx/ety/etz: the registry E,
                    // just advanced, with E_ext folded above) and the
                    // substep-average B^{k+1} (the registry B; see
                    // BfieldEvolve). Gyration is |J_e|-preserving;
                    // damping and drive are trapezoidal.
                    amrex::Real const gam_sum =
                        kE * eta_v + gam_v + gam_vis;
                    amrex::Real const acn =
                        1.0_rt + 0.5_rt * dt_r * gam_sum;
                    amrex::Real const am =
                        1.0_rt - 0.5_rt * dt_r * gam_sum;
                    amrex::Real const chx = 0.5_rt * qm_c * dt_r * bx;
                    amrex::Real const chy = 0.5_rt * qm_c * dt_r * by;
                    amrex::Real const chz = 0.5_rt * qm_c * dt_r * bz;
                    amrex::Real const cb2 = chx*chx + chy*chy + chz*chz;
                    amrex::Real const srx = am*jox + (chy*joz - chz*joy)
                        + dt_r * (kE * etx - kE * eta_v * Jix(i,j,k)
                                  + qm_c * gpx);
                    amrex::Real const sry = am*joy + (chz*jox - chx*joz)
                        + dt_r * (kE * ety - kE * eta_v * Jiy(i,j,k)
                                  + qm_c * gpy);
                    amrex::Real const srz = am*joz + (chx*joy - chy*jox)
                        + dt_r * (kE * etz - kE * eta_v * Jiz(i,j,k)
                                  + qm_c * gpz);
                    amrex::Real const cbd = chx*srx + chy*sry + chz*srz;
                    amrex::Real const cinv =
                        1.0_rt / (acn * (acn*acn + cb2));
                    amrex::Real const jnxs = (acn*acn*srx
                        + acn*(chy*srz - chz*sry) + cbd*chx) * cinv;
                    amrex::Real const jnys = (acn*acn*sry
                        + acn*(chz*srx - chx*srz) + cbd*chy) * cinv;
                    amrex::Real const jnzs = (acn*acn*srz
                        + acn*(chx*sry - chy*srx) + cbd*chz) * cinv;
                    if (qv) {
                        amrex::Real const j2 =
                            (skip_x ? 0.0_rt : jnxs*jnxs)
                            + (skip_y ? 0.0_rt : jnys*jnys)
                            + (skip_z ? 0.0_rt : jnzs*jnzs);
                        qv(i,j,k) += gam_vis * dt_r * j2
                            / (qm_c * amrex::max(rho_v, rho_1c));
                    }
                    if (!skip_x) { Je(i,j,k,0) = jnxs; }
                    if (!skip_y) { Je(i,j,k,1) = jnys; }
                    if (!skip_z) { Je(i,j,k,2) = jnzs; }
                    return;
                }
                // implicit local solve:
                //   (a I - [beta]x) Je' = r,  beta = qm dt B_tot
                //   inverse = (a^2 I + a [beta]x + beta beta^T)
                //             / (a (a^2 + |beta|^2))
                amrex::Real const a = 1.0_rt
                    + dt_r * (dt_r * kE * inv_eps_art
                              + kE * eta_v + gam_v + gam_vis);
                amrex::Real const bex = qm_c * dt_r * bx;
                amrex::Real const bey = qm_c * dt_r * by;
                amrex::Real const bez = qm_c * dt_r * bz;
                amrex::Real const b2 = bex*bex + bey*bey + bez*bez;
                amrex::Real const rx = jox
                    + dt_r * kE * (etx + dt_r * inv_eps_art * rcx)
                    - dt_r * kE * eta_v * Jix(i,j,k)
                    + qm_c * dt_r * gpx;
                amrex::Real const ry = joy
                    + dt_r * kE * (ety + dt_r * inv_eps_art * rcy)
                    - dt_r * kE * eta_v * Jiy(i,j,k)
                    + qm_c * dt_r * gpy;
                amrex::Real const rz = joz
                    + dt_r * kE * (etz + dt_r * inv_eps_art * rcz)
                    - dt_r * kE * eta_v * Jiz(i,j,k)
                    + qm_c * dt_r * gpz;
                amrex::Real const bdotr = bex*rx + bey*ry + bez*rz;
                amrex::Real const inv = 1.0_rt / (a * (a*a + b2));
                amrex::Real const jnx =
                    (a*a*rx + a*(bey*rz - bez*ry) + bdotr*bex) * inv;
                amrex::Real const jny =
                    (a*a*ry + a*(bez*rx - bex*rz) + bdotr*bey) * inv;
                amrex::Real const jnz =
                    (a*a*rz + a*(bex*ry - bey*rx) + bdotr*bez) * inv;
                amrex::Real exn = Ex(i,j,k)
                    + dt_r * inv_eps_art * (rcx - jnx);
                amrex::Real eyn = Ey(i,j,k)
                    + dt_r * inv_eps_art * (rcy - jny);
                amrex::Real ezn = Ez(i,j,k)
                    + dt_r * inv_eps_art * (rcz - jnz);
                if (use_qn) {
                    // implicit relaxation toward the one-count-guarded
                    // Ohm value, plasma-blended (the OPTIONAL floored
                    // division of the qn pin -- see the member doc)
                    amrex::Real const w_qn = 0.5_rt
                        * (1.0_rt + std::tanh(
                              (rho_v - rho_1c) / (0.5_rt * rho_1c)));
                    amrex::Real const lam = dt_r * gam_qn * w_qn;
                    amrex::Real const rho_gq =
                        amrex::max(rho_v, rho_1c);
                    amrex::Real const inv_l = 1.0_rt / (1.0_rt + lam);
                    exn = (exn + lam * ((enE(i,j,k,0) - gpx
                        + rho_v * eta_v * Jx(i,j,k)) / rho_gq))
                        * inv_l;
                    eyn = (eyn + lam * ((enE(i,j,k,1) - gpy
                        + rho_v * eta_v * Jy(i,j,k)) / rho_gq))
                        * inv_l;
                    ezn = (ezn + lam * ((enE(i,j,k,2) - gpz
                        + rho_v * eta_v * Jz(i,j,k)) / rho_gq))
                        * inv_l;
                }
                if (qv) {
                    // dissipated energy density this substep:
                    // gamma_visc * m_e' |J_e|^2 / (e^2 n_g) * dt,
                    // with m_e'/(e^2 n_g) = 1/(qm_c * rho_g)
                    amrex::Real const j2 =
                        (skip_x ? 0.0_rt : jnx*jnx)
                        + (skip_y ? 0.0_rt : jny*jny)
                        + (skip_z ? 0.0_rt : jnz*jnz);
                    qv(i,j,k) += gam_vis * dt_r * j2
                        / (qm_c * amrex::max(rho_v, rho_1c));
                }
                if (!skip_x) { Je(i,j,k,0) = jnx; Ex(i,j,k) = exn; }
                if (!skip_y) { Je(i,j,k,1) = jny; Ey(i,j,k) = eyn; }
                if (!skip_z) { Je(i,j,k,2) = jnz; Ez(i,j,k) = ezn; }
            });
        }
        return;
    }

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
        Array4<Real const> const& Jx = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jy = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& enE = enE_nodal_mf.const_array(mfi);
        Array4<Real const> eiN;
        if (Ei_nodal_mf) { eiN = Ei_nodal_mf->const_array(mfi); }
        Array4<Real const> const& rho = rhofield.const_array(mfi);
        Array4<Real const> const& Pe = Pefield.array(mfi);
        Array4<Real> const& Bx = Bfield[0]->array(mfi);
        Array4<Real> const& By = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);
        // Overlay arrays stay default-constructed (never indexed) when no
        // per-species resistivity is registered -- the kernels gate the read
        // on has_eta_overlay.
        Array4<Real const> eta_overlay_x, eta_overlay_y, eta_overlay_z;
        if (has_eta_overlay) {
            eta_overlay_x = eta_overlay_mf[0]->const_array(mfi);
            eta_overlay_y = eta_overlay_mf[1]->const_array(mfi);
            eta_overlay_z = eta_overlay_mf[2]->const_array(mfi);
        }

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries
        amrex::Array4<int> update_Ex_arr, update_Ey_arr, update_Ez_arr;
        if (EB::enabled()) {
            update_Ex_arr = eb_update_E[0]->array(mfi);
            update_Ey_arr = eb_update_E[1]->array(mfi);
            update_Ez_arr = eb_update_E[2]->array(mfi);
        }

        Array4<Real> Ex_ext, Ey_ext, Ez_ext;
        if (include_E_ext) {
            Ex_ext = Efield_external[0]->array(mfi);
            Ey_ext = Efield_external[1]->array(mfi);
            Ez_ext = Efield_external[2]->array(mfi);
        }

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

        // Loop over the cells and update the E field
        // Ex calculation
        amrex::ParallelFor(tex, [=] AMREX_GPU_DEVICE (int i, int j, int k){

            // Skip field update in the embedded boundaries
            if (update_Ex_arr && update_Ex_arr(i, j, k) == 0) { return; }

            // Interpolate to get the appropriate charge density in space
            const Real rho_val = Interp(rho, nodal, Ex_stag, coarsen, i, j, k, 0);

            if (rho_val < rho_floor && holmstrom_vacuum_region && !holmstrom_smooth) {
                Ex(i, j, k) = 0._rt;
            } else {
                // Get the gradient of the electron pressure if the longitudinal part of
                // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                const Real grad_Pe =
                    (solve_for_Faraday ? add_grad_pe_faraday
                                       : include_electron_pressure_term) ?
                    T_Algo::UpwardDx(Pe, coefs_x, n_coefs_x, i, j, k)
                    : 0._rt;

                // interpolate the nodal neE values to the Yee grid
                const auto enE_x = Interp(enE, nodal, Ex_stag, coarsen, i, j, k, 0);

                // safety condition since we divide by rho
                const auto rho_val_limited = std::max(rho_val, rho_floor);

                Real ohm_val = (enE_x - grad_Pe) / rho_val_limited;
                if (holmstrom_smooth) {
                    ohm_val *= 0.5_rt * (1._rt + std::tanh(
                        (rho_val - rho_floor) * holmstrom_inv_width));
                }
                Ex(i, j, k) = ohm_val;
            }
            if (include_electron_inertia) {
                Ex(i, j, k) += Interp(eiN, nodal, Ex_stag, coarsen, i, j, k, 0);
            }


            // Resistivity: whenever the caller kept eta in this solve
            // (always true for the Faraday solves; the push/stored-E
            // solve follows the caller), or when the Q_ei drag operator
            // carries the ion-side friction (dropping eta J from the
            // push field IS the friction, so E* + drag would book it
            // twice).
            if (include_resistivity || add_resistivity_push) {
                Real jtot_val = 0._rt;
                if (resistivity_has_J_dependence) {
                    // Interpolate current to appropriate staggering to match E field
                    const Real jx_val = Jx(i, j, k);
                    const Real jy_val = Interp(Jy, Jy_stag, Ex_stag, coarsen, i, j, k, 0);
                    const Real jz_val = Interp(Jz, Jz_stag, Ex_stag, coarsen, i, j, k, 0);
                    jtot_val = std::sqrt(jx_val*jx_val + jy_val*jy_val + jz_val*jz_val);
                }

                Ex(i, j, k) += eta(rho_val, jtot_val, t_new) * Jx(i, j, k);
                if (has_eta_overlay) { Ex(i, j, k) += eta_overlay_x(i, j, k); }

                if (include_hyper_resistivity_term && solve_for_Faraday) {

                    // Interpolate B field to appropriate staggering to match E field
                    Real btot_val = 0._rt;
                    if (hyper_resistivity_has_B_dependence) {
                        const Real bx_val = Interp(Bx, Bx_stag, Ex_stag, coarsen, i, j, k, 0);
                        const Real by_val = Interp(By, By_stag, Ex_stag, coarsen, i, j, k, 0);
                        const Real bz_val = Interp(Bz, Bz_stag, Ex_stag, coarsen, i, j, k, 0);
                        btot_val = std::sqrt(bx_val*bx_val + by_val*by_val + bz_val*bz_val);
                    }

                    auto nabla2Jx = T_Algo::Dxx(Jx, coefs_x, n_coefs_x, i, j, k)
                        + T_Algo::Dyy(Jx, coefs_y, n_coefs_y, i, j, k)
                        + T_Algo::Dzz(Jx, coefs_z, n_coefs_z, i, j, k);

                    Ex(i, j, k) -= eta_h(rho_val, btot_val) * nabla2Jx;
                }
            }

            if (include_E_ext && (subtract_E_ext_everywhere || rho_val >= rho_floor)) {
                Ex(i, j, k) -= Ex_ext(i, j, k);
            }
        });

        // Ey calculation
        amrex::ParallelFor(tey, [=] AMREX_GPU_DEVICE (int i, int j, int k) {

            // Skip field update in the embedded boundaries
            if (update_Ey_arr && update_Ey_arr(i, j, k) == 0) { return; }

            // Interpolate to get the appropriate charge density in space
            const Real rho_val = Interp(rho, nodal, Ey_stag, coarsen, i, j, k, 0);

            if (rho_val < rho_floor && holmstrom_vacuum_region && !holmstrom_smooth) {
                Ey(i, j, k) = 0._rt;
            } else {
                // Get the gradient of the electron pressure if the longitudinal part of
                // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                const Real grad_Pe =
                    (solve_for_Faraday ? add_grad_pe_faraday
                                       : include_electron_pressure_term) ?
                    T_Algo::UpwardDy(Pe, coefs_y, n_coefs_y, i, j, k)
                    : 0._rt;

                // interpolate the nodal neE values to the Yee grid
                const auto enE_y = Interp(enE, nodal, Ey_stag, coarsen, i, j, k, 1);

                // safety condition since we divide by rho
                const auto rho_val_limited = std::max(rho_val, rho_floor);

                Real ohm_val = (enE_y - grad_Pe) / rho_val_limited;
                if (holmstrom_smooth) {
                    ohm_val *= 0.5_rt * (1._rt + std::tanh(
                        (rho_val - rho_floor) * holmstrom_inv_width));
                }
                Ey(i, j, k) = ohm_val;
            }
            if (include_electron_inertia) {
                Ey(i, j, k) += Interp(eiN, nodal, Ey_stag, coarsen, i, j, k, 1);
            }


            // Resistivity: whenever the caller kept eta in this solve
            // (always true for the Faraday solves; the push/stored-E
            // solve follows the caller), or when the Q_ei drag operator
            // carries the ion-side friction (dropping eta J from the
            // push field IS the friction, so E* + drag would book it
            // twice).
            if (include_resistivity || add_resistivity_push) {
                Real jtot_val = 0._rt;
                if (resistivity_has_J_dependence) {
                    // Interpolate current to appropriate staggering to match E field
                    const Real jx_val = Interp(Jx, Jx_stag, Ey_stag, coarsen, i, j, k, 0);
                    const Real jy_val = Jy(i, j, k);
                    const Real jz_val = Interp(Jz, Jz_stag, Ey_stag, coarsen, i, j, k, 0);
                    jtot_val = std::sqrt(jx_val*jx_val + jy_val*jy_val + jz_val*jz_val);
                }

                Ey(i, j, k) += eta(rho_val, jtot_val, t_new) * Jy(i, j, k);
                if (has_eta_overlay) { Ey(i, j, k) += eta_overlay_y(i, j, k); }

                if (include_hyper_resistivity_term && solve_for_Faraday) {

                    // Interpolate B field to appropriate staggering to match E field
                    Real btot_val = 0._rt;
                    if (hyper_resistivity_has_B_dependence) {
                        const Real bx_val = Interp(Bx, Bx_stag, Ey_stag, coarsen, i, j, k, 0);
                        const Real by_val = Interp(By, By_stag, Ey_stag, coarsen, i, j, k, 0);
                        const Real bz_val = Interp(Bz, Bz_stag, Ey_stag, coarsen, i, j, k, 0);
                        btot_val = std::sqrt(bx_val*bx_val + by_val*by_val + bz_val*bz_val);
                    }

                    auto nabla2Jy = T_Algo::Dxx(Jy, coefs_x, n_coefs_x, i, j, k)
                        + T_Algo::Dyy(Jy, coefs_y, n_coefs_y, i, j, k)
                        + T_Algo::Dzz(Jy, coefs_z, n_coefs_z, i, j, k);

                    Ey(i, j, k) -= eta_h(rho_val, btot_val) * nabla2Jy;
                }
            }

            if (include_E_ext && (subtract_E_ext_everywhere || rho_val >= rho_floor)) {
                Ey(i, j, k) -= Ey_ext(i, j, k);
            }
        });

        // Ez calculation
        amrex::ParallelFor(tez, [=] AMREX_GPU_DEVICE (int i, int j, int k){

            // Skip field update in the embedded boundaries
            if (update_Ez_arr && update_Ez_arr(i, j, k) == 0) { return; }

            // Interpolate to get the appropriate charge density in space
            const Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, k, 0);

            if (rho_val < rho_floor && holmstrom_vacuum_region && !holmstrom_smooth) {
                Ez(i, j, k) = 0._rt;
            } else {
                // Get the gradient of the electron pressure if the longitudinal part of
                // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                const Real grad_Pe =
                    (solve_for_Faraday ? add_grad_pe_faraday
                                       : include_electron_pressure_term) ?
                    T_Algo::UpwardDz(Pe, coefs_z, n_coefs_z, i, j, k)
                    : 0._rt;

                // interpolate the nodal neE values to the Yee grid
                const auto enE_z = Interp(enE, nodal, Ez_stag, coarsen, i, j, k, 2);

                // safety condition since we divide by rho
                const auto rho_val_limited = std::max(rho_val, rho_floor);

                Real ohm_val = (enE_z - grad_Pe) / rho_val_limited;
                if (holmstrom_smooth) {
                    ohm_val *= 0.5_rt * (1._rt + std::tanh(
                        (rho_val - rho_floor) * holmstrom_inv_width));
                }
                Ez(i, j, k) = ohm_val;
            }
            if (include_electron_inertia) {
                Ez(i, j, k) += Interp(eiN, nodal, Ez_stag, coarsen, i, j, k, 2);
            }


            // Resistivity: whenever the caller kept eta in this solve
            // (always true for the Faraday solves; the push/stored-E
            // solve follows the caller), or when the Q_ei drag operator
            // carries the ion-side friction (dropping eta J from the
            // push field IS the friction, so E* + drag would book it
            // twice).
            if (include_resistivity || add_resistivity_push) {
                Real jtot_val = 0._rt;
                if (resistivity_has_J_dependence) {
                    // Interpolate current to appropriate staggering to match E field
                    const Real jx_val = Interp(Jx, Jx_stag, Ez_stag, coarsen, i, j, k, 0);
                    const Real jy_val = Interp(Jy, Jy_stag, Ez_stag, coarsen, i, j, k, 0);
                    const Real jz_val = Jz(i, j, k);
                    jtot_val = std::sqrt(jx_val*jx_val + jy_val*jy_val + jz_val*jz_val);
                }

                Ez(i, j, k) += eta(rho_val, jtot_val, t_new) * Jz(i, j, k);
                if (has_eta_overlay) { Ez(i, j, k) += eta_overlay_z(i, j, k); }

                if (include_hyper_resistivity_term && solve_for_Faraday) {

                    // Interpolate B field to appropriate staggering to match E field
                    Real btot_val = 0._rt;
                    if (hyper_resistivity_has_B_dependence) {
                        const Real bx_val = Interp(Bx, Bx_stag, Ez_stag, coarsen, i, j, k, 0);
                        const Real by_val = Interp(By, By_stag, Ez_stag, coarsen, i, j, k, 0);
                        const Real bz_val = Interp(Bz, Bz_stag, Ez_stag, coarsen, i, j, k, 0);
                        btot_val = std::sqrt(bx_val*bx_val + by_val*by_val + bz_val*bz_val);
                    }

                    auto nabla2Jz = T_Algo::Dxx(Jz, coefs_x, n_coefs_x, i, j, k)
                        + T_Algo::Dyy(Jz, coefs_y, n_coefs_y, i, j, k)
                        + T_Algo::Dzz(Jz, coefs_z, n_coefs_z, i, j, k);

                    Ez(i, j, k) -= eta_h(rho_val, btot_val) * nabla2Jz;
                }
            }

            if (include_E_ext && (subtract_E_ext_everywhere || rho_val >= rho_floor)) {
                Ez(i, j, k) -= Ez_ext(i, j, k);
            }
        });

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}
#endif
