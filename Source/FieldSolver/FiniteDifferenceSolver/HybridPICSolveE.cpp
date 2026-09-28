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

#include "FieldSolver/ImplicitSolvers/NativeCoilField.H"
#include "FieldSolver/ImplicitSolvers/NativePECPlasma.H"
#include "FiniteDifferenceSolver.H"
#include "CompensatedTransverseOhm.H"
#include "HybridPICModel/HybridOhmDampingFields.H"
#include "FieldSolver/ImplicitSolvers/EulerianDissipation.H"

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
#include "HybridPICModel/QdsmcVolumeElement.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>
#include <cmath>

using namespace amrex;
using warpx::fields::FieldType;

namespace CompensatedOhm=warpx::ohm::compensated;


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
                if (update_Jx_arr && update_Jx_arr(i, j, k) == 0) {
                    // This is a freshly computed curl, not an incremental update.
                    // A skipped row must not retain a previous residual's
                    // longitudinal-current correction or external subtraction.
                    Jx(i, j, k) = 0._rt;
                    return;
                }

                Jx(i, j, k) = one_over_mu0 * (
                    - T_Algo::DownwardDz(By, coefs_z, n_coefs_z, i, j, k)
                    + T_Algo::DownwardDy(Bz, coefs_y, n_coefs_y, i, j, k)
                );
            },

            // Jy calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Skip field update in the embedded boundaries
                if (update_Jy_arr && update_Jy_arr(i, j, k) == 0) {
                    // This is a freshly computed curl, not an incremental update.
                    // A skipped row must not retain a previous residual's
                    // longitudinal-current correction or external subtraction.
                    Jy(i, j, k) = 0._rt;
                    return;
                }

                Jy(i, j, k) = one_over_mu0 * (
                    - T_Algo::DownwardDx(Bz, coefs_x, n_coefs_x, i, j, k)
                    + T_Algo::DownwardDz(Bx, coefs_z, n_coefs_z, i, j, k)
                );
            },

            // Jz calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Skip field update in the embedded boundaries
                if (update_Jz_arr && update_Jz_arr(i, j, k) == 0) {
                    // This is a freshly computed curl, not an incremental update.
                    // A skipped row must not retain a previous residual's
                    // longitudinal-current correction or external subtraction.
                    Jz(i, j, k) = 0._rt;
                    return;
                }

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

void
FiniteDifferenceSolver::HybridPICSolveE (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField& Jfield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Bfield, amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    [[maybe_unused]] std::array<std::unique_ptr<amrex::iMultiFab>, 3> const&
        eb_update_E,
    int lev, HybridPICModel const* hybrid_model, const bool solve_for_Faraday,
    const bool include_resistivity, ablastr::fields::VectorField const* EH_out,
    ablastr::fields::VectorField const* EV_out,
    const warpx::thermal::EulerianDissipation* trial_dissipation,
    ablastr::fields::VectorField const* ER_out,
    const warpx::thermal::HybridOhmDampingFields* damping_out,
    ablastr::fields::VectorField const* F_pressure_hall_out,
    ablastr::fields::VectorField const* transverse_offset) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!(F_pressure_hall_out && transverse_offset),
        "Pressure/Hall capture and compensated transverse assembly require separate calls");
    if (transverse_offset) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(hybrid_model->UseCompatibleYeeInertia() &&
            hybrid_model->m_include_electron_inertia && !solve_for_Faraday &&
            WarpX::grid_type==GridType::Staggered,
            "Transverse Ohm offset requires direct Yee inertia");
        for (int c=0;c<3;++c) {
            auto const* offset=(*transverse_offset)[c];
            AMREX_ALWAYS_ASSERT(offset && offset->boxArray()==Efield[c]->boxArray() &&
                offset->DistributionMap()==Efield[c]->DistributionMap() &&
                offset->nComp()==1 && Efield[c]->nComp()==1 &&
                offset->nGrowVect().allGE(Efield[c]->nGrowVect()));
            for (int d=0;d<3;++d) AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                offset!=Efield[d] && (!EH_out || offset!=(*EH_out)[d]) &&
                (!EV_out || offset!=(*EV_out)[d]) &&
                (!ER_out || offset!=(*ER_out)[d]) &&
                (!damping_out || (offset!=damping_out->end_resistivity[d] &&
                                 offset!=damping_out->end_holmstrom[d])),
                "Transverse Ohm offset must not alias a field or capture output");
        }
    }

#if !defined(WARPX_DIM_RZ)
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(trial_dissipation==nullptr,
        "Trial dissipative field application currently requires RZ");
#endif
    if (F_pressure_hall_out) {
        // This is the exact value already evaluated by the native kernel,
        // before inertia and before wrapper electric boundary replacement.
        // Its complete non-inertial meaning is deliberately restricted to
        // this pressure/Hall-only algebraic scope. Never subtract a rounded
        // inertial field from the final electric field to reconstruct it.
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            !hybrid_model->HasResistivity() &&
            !hybrid_model->m_include_electron_viscosity &&
            !hybrid_model->m_include_hyper_resistivity_term &&
            !hybrid_model->m_end_region.holmstrom &&
            !hybrid_model->m_holmstrom_vacuum_region &&
            !hybrid_model->m_esolve_tensor && !hybrid_model->m_esolve_curlcurl &&
            (!hybrid_model->m_add_external_fields ||
             (hybrid_model->m_external_unified && warpx::darwin::NativeCoilCurrentEnabled() &&
              warpx::darwin::NativePECPlasmaEnabled())) && !hybrid_model->m_pec_conductor_wall_rows &&
            !EB::enabled(),
            "Pressure/Hall capture excludes resistivity, viscosity, hyper, vacuum/end, transformed, external and EB closures");
        for (int c=0;c<3;++c) {
            auto const* out=(*F_pressure_hall_out)[c];
            bool valid=out && out!=&rhofield && out!=&Pefield;
            if (out) {
                valid=valid && out->boxArray()==Efield[c]->boxArray() &&
                    out->DistributionMap()==Efield[c]->DistributionMap() &&
                    out->nComp()==Efield[c]->nComp() &&
                    out->nGrowVect().allGE(Efield[c]->nGrowVect());
                for (int d=0;d<3;++d) {
                    valid=valid && out!=Efield[d] && out!=Jfield[d] &&
                        out!=Jifield[d] && out!=Bfield[d] &&
                        (!ER_out || out!=(*ER_out)[d]) &&
                        (!EH_out || out!=(*EH_out)[d]) &&
                        (!EV_out || out!=(*EV_out)[d]) &&
                        (!damping_out || (out!=damping_out->end_resistivity[d] &&
                                         out!=damping_out->end_holmstrom[d])) &&
                        (c==d || out!=(*F_pressure_hall_out)[d]);
                }
                valid=valid && out!=&hybrid_model->ElectronTemperatureForSolve(lev);
            }
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(valid,
                "Pressure/Hall capture requires independent matching E scratch and all guards");
        }
        for (auto* out:*F_pressure_hall_out) out->setVal(0._rt);
    }
    if (ER_out) {
        // Validate every output before clearing any scratch. In particular,
        // accepting an aliased E/J/rho would change the physical solve.
        for (int c = 0; c < 3; ++c) {
            auto const* out = (*ER_out)[c];
            bool valid = out && out != &rhofield && out != &Pefield;
            if (out) {
                valid = valid && out->boxArray() == Efield[c]->boxArray() &&
                    out->DistributionMap() == Efield[c]->DistributionMap() &&
                    out->nComp() == Efield[c]->nComp() &&
                    out->nGrowVect().allGE(Efield[c]->nGrowVect());
                for (int d = 0; d < 3; ++d) {
                    valid = valid && out != Efield[d] && out != Jfield[d] &&
                        out != Jifield[d] && out != Bfield[d] &&
                        (!EH_out || out != (*EH_out)[d]) &&
                        (!EV_out || out != (*EV_out)[d]) &&
                        (c == d || out != (*ER_out)[d]);
                }
                if (hybrid_model->m_resistivity_has_Te_dependence) {
                    valid = valid && out != &hybrid_model->ElectronTemperatureForSolve(lev);
                }
            }
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(valid,
                "Global eta field capture requires independent, matching E scratch and all guards");
        }
        for (int c = 0; c < 3; ++c) { (*ER_out)[c]->setVal(0._rt); }
    }
    if (damping_out) {
        // Validate ALL borrowed arrays before clearing any of them. The
        // wrapper owns no input mutation through a requested component.
        auto const& end=damping_out->end_resistivity;
        auto const& gate=damping_out->end_holmstrom;
        for (int c=0;c<3;++c) {
            for (auto const* out : {end[c],gate[c]}) {
                bool valid=out && out!=&rhofield && out!=&Pefield;
                if (out) {
                    valid=valid && out->boxArray()==Efield[c]->boxArray() &&
                        out->DistributionMap()==Efield[c]->DistributionMap() &&
                        out->nComp()==Efield[c]->nComp() &&
                        out->nGrowVect().allGE(Efield[c]->nGrowVect());
                    int occurrences=0;
                    for (int d=0;d<3;++d) {
                        valid=valid && out!=Efield[d] && out!=Jfield[d] &&
                            out!=Jifield[d] && out!=Bfield[d] &&
                            (!ER_out || out!=(*ER_out)[d]) &&
                            (!EH_out || out!=(*EH_out)[d]) &&
                            (!EV_out || out!=(*EV_out)[d]);
                        occurrences+=int(out==end[d])+int(out==gate[d]);
                    }
                    valid=valid && occurrences==1 &&
                        out!=&hybrid_model->ElectronTemperatureForSolve(lev);
                }
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(valid,
                    "Numerical damping capture needs independent matching E scratch and all guards");
            }
        }
        for (int c=0;c<3;++c) { end[c]->setVal(0._rt); gate[c]->setVal(0._rt); }
    }
    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)

        HybridPICSolveECylindrical<CylindricalYeeAlgorithm>(
            Efield, Jfield, Jifield, Bfield, rhofield, Pefield, eb_update_E,
            lev, hybrid_model, solve_for_Faraday, include_resistivity, EH_out,
            EV_out, trial_dissipation, ER_out, damping_out, F_pressure_hall_out, transverse_offset);

#elif defined(WARPX_DIM_RSPHERE)

        // The spherical kernels have no hyper-resistivity or viscous drag
        // term. Global eta capture is not implemented there either: reject
        // the request rather than returning an incomplete work component.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            EH_out == nullptr && EV_out == nullptr && ER_out == nullptr && damping_out == nullptr && F_pressure_hall_out == nullptr,
            "HybridPICSolveE: EH_out / EV_out / ER_out (hyper_resistivity_heating, "
            "qdsmc_viscosity_in_ohms_law / global eta / damping capture) are not available in the spherical "
            "geometry");
        HybridPICSolveESpherical <SphericalYeeAlgorithm> (
            Efield, Jfield, Jifield, Bfield, rhofield, Pefield,
            lev, hybrid_model, solve_for_Faraday, include_resistivity
        );

#else
    if (WarpX::grid_type == GridType::Staggered)
    {
        HybridPICSolveECartesian<CartesianYeeAlgorithm>(
            Efield, Jfield, Jifield, Bfield, rhofield, Pefield, eb_update_E,
            lev, hybrid_model, solve_for_Faraday, include_resistivity, EH_out,
            EV_out, ER_out, damping_out, F_pressure_hall_out, transverse_offset);
    } else {
        HybridPICSolveECartesian<CartesianNodalAlgorithm>(
            Efield, Jfield, Jifield, Bfield, rhofield, Pefield, eb_update_E,
            lev, hybrid_model, solve_for_Faraday, include_resistivity, EH_out,
            EV_out, ER_out, damping_out, F_pressure_hall_out, transverse_offset);
    }
#endif
    } else {
        amrex::Abort(Utils::TextMsg::Err(
            "HybridSolveE: The hybrid-PIC electromagnetic solver algorithm must be used"));
    }

    if (transverse_offset) {
        auto const Ei=WarpX::GetInstance().m_fields.get_alldirs("hybrid_E_inertial_fp",lev);
        for (int c=0;c<3;++c) {
            for (amrex::MFIter mfi(*Efield[c],amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
                auto const e=Efield[c]->array(mfi);
                auto const inertia=Ei[c]->const_array(mfi);
                amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    e(i,j,k)+=inertia(i,j,k);
                });
            }
        }
    }
}

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
template <typename T_Algo>
void
FiniteDifferenceSolver::HybridPICSolveECylindrical (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Bfield, amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_E,
    int lev, HybridPICModel const* hybrid_model, const bool solve_for_Faraday,
    const bool include_resistivity, ablastr::fields::VectorField const* EH_out,
    ablastr::fields::VectorField const* EV_out,
    const warpx::thermal::EulerianDissipation* trial_dissipation,
    ablastr::fields::VectorField const* ER_out,
    const warpx::thermal::HybridOhmDampingFields* damping_out,
    ablastr::fields::VectorField const* F_pressure_hall_out,
    ablastr::fields::VectorField const* transverse_offset) {
    bool const write_er = ER_out != nullptr;
    bool const write_damping = damping_out != nullptr;
    // Both steps below do not currently support m > 0 and should be
    // modified if such support wants to be added
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (m_nmodes == 1),
        "Ohm's law solver only support m = 0 azimuthal mode at present.");

    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    using namespace ablastr::coarsen::sample;

    // get hybrid model parameters
    const auto eta = hybrid_model->m_eta;
    const auto end_region = hybrid_model->EndRegion(lev);
    // End eta is numerical magnetic damping. The legacy thermal path heats
    // electrons with it; the opt-in edge-work path books its signed sink
    // separately. Full-minus-nores removes its ion force, and no physical
    // OU/drag rate includes this increment.
    const auto eta_te = hybrid_model->m_eta_te;
    const bool eta_has_Te = hybrid_model->m_resistivity_has_Te_dependence;
    amrex::MultiFab const* const te_mf = hybrid_model->ResistivityTe(lev);
    const auto eta_h = hybrid_model->m_eta_h;
    const auto rho_floor = hybrid_model->m_n_floor * PhysConst::q_e;
    const auto floor_w = hybrid_model->m_n_floor_smooth_width * rho_floor;
    const auto resistivity_has_J_dependence = hybrid_model->m_resistivity_has_J_dependence;
    const auto hyper_resistivity_has_B_dependence = hybrid_model->m_hyper_resistivity_has_B_dependence;
    const bool include_hyper_resistivity_term =
        hybrid_model->m_include_hyper_resistivity_term && trial_dissipation==nullptr;
    const bool include_electron_inertia = hybrid_model->m_include_electron_inertia && transverse_offset==nullptr;

    const bool include_external_fields = hybrid_model->m_add_external_fields
        && !hybrid_model->m_external_unified;
    const bool subtract_E_ext_everywhere =
        hybrid_model->m_external_e_subtraction_unconditional;
    const bool include_hall_term = hybrid_model->m_include_hall_term;
    const bool include_electron_pressure_term =
        hybrid_model->m_include_electron_pressure_term;

    // Conductor-row treatment of the r-max PEC wall: the wall node plane
    // carries no plasma Hall/motional force (see the member documentation
    // of m_pec_conductor_wall_rows).
    const bool conductor_wall_row = hybrid_model->m_pec_conductor_wall_rows
        && (WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC);
    const int iwall_node =
        WarpX::GetInstance().Geom(lev).Domain().bigEnd(0) + 1;
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
    // Radial confinement of the gate (see the knob doc): eligibility is
    // masked to r < axis_radius so sub-floor halo cells beyond it keep the
    // ungated legacy Ohm E exactly. The rolloff width smooths the mask's
    // own radial edge for the smooth gate; the hard rho-branch uses a hard
    // radial cutoff.
    const bool holmstrom_axis_confined =
        holmstrom_vacuum_region && (hybrid_model->m_holmstrom_axis_radius > 0._rt);
    const Real holmstrom_axis_r = hybrid_model->m_holmstrom_axis_radius;
    const Real holmstrom_axis_inv_w =
        (hybrid_model->m_holmstrom_axis_rolloff > 0._rt)
        ? 1._rt / hybrid_model->m_holmstrom_axis_rolloff
        : 0._rt;

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
    bool const direct_yee_inertia = hybrid_model->UseCompatibleYeeInertia();
    ablastr::fields::VectorField Ei_yee;
    if (direct_yee_inertia) {
        Ei_yee = warpx.m_fields.get_alldirs("hybrid_E_inertial_fp", lev);
    }
    // Density pedestal (change of variables, HybridPICModel::m_density_pedestal):
    // the Hall / grad Pe divisor is max(rho + rho_ped, rho_floor) instead of
    // max(rho, rho_floor); the Holmstrom gate, the external-E subtraction
    // and eta(rho, J, t) keep the deposited rho.
    amrex::MultiFab const* const rho_ped_mf = hybrid_model->DensityPedestal(lev);
    const bool use_pedestal = (rho_ped_mf != nullptr);
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

    // Adjoint-paired curl-curl hyper-resistivity (see the
    // m_hyper_res_curl_curl member doc): precompute K = eta_H (curl J) on
    // the B staggering with the native Yee edge->face curl; the Faraday
    // hyper term in the kernels below becomes +(curl K) via the native
    // face->edge curl (the Ampere forms, including the exact on-axis
    // regularization), replacing the component Laplacian at the same
    // accumulation site. K is zeroed and computed on valid faces only;
    // box-boundary ghosts are exchanged, non-periodic DOMAIN ghosts stay
    // zero and are never read: the kernels skip the one-node ring at
    // non-periodic domain faces, so the operator carries no wall flux.
    const bool use_hyper_cc = hybrid_model->m_hyper_res_curl_curl
        && include_hyper_resistivity_term && include_resistivity;
    MultiFab Kr_mf, Kt_mf, Kz_mf;
    Box hyper_cc_er_box, hyper_cc_et_box, hyper_cc_ez_box;
    if (use_hyper_cc) {
        Kr_mf.define(Bfield[0]->boxArray(), Bfield[0]->DistributionMap(),
                     1, IntVect(1));
        Kt_mf.define(Bfield[1]->boxArray(), Bfield[1]->DistributionMap(),
                     1, IntVect(1));
        Kz_mf.define(Bfield[2]->boxArray(), Bfield[2]->DistributionMap(),
                     1, IntVect(1));
        Kr_mf.setVal(0.0_rt);
        Kt_mf.setVal(0.0_rt);
        Kz_mf.setVal(0.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Kr_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Array4<Real> const& Kr = Kr_mf.array(mfi);
            Array4<Real> const& Kt = Kt_mf.array(mfi);
            Array4<Real> const& Kz = Kz_mf.array(mfi);
            Array4<Real const> const& Jr_a = Jfield[0]->const_array(mfi);
            Array4<Real const> const& Jtheta_a = Jfield[1]->const_array(mfi);
            Array4<Real const> const& Jz_a = Jfield[2]->const_array(mfi);
            Array4<Real const> const& Br_a = Bfield[0]->const_array(mfi);
            Array4<Real const> const& Btheta_a = Bfield[1]->const_array(mfi);
            Array4<Real const> const& Bz_a = Bfield[2]->const_array(mfi);
            Array4<Real const> const& rho_a = rhofield.const_array(mfi);
            Real const * const AMREX_RESTRICT coefs_r = m_stencil_coefs_r.dataPtr();
            int const n_coefs_r = static_cast<int>(m_stencil_coefs_r.size());
            Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
            int const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());
            Real const dr_l = m_dr;
            Real const rmin_l = m_rmin;
            Box const& tbr = mfi.tilebox(Bfield[0]->ixType().toIntVect());
            Box const& tbt = mfi.tilebox(Bfield[1]->ixType().toIntVect());
            Box const& tbz = mfi.tilebox(Bfield[2]->ixType().toIntVect());
            amrex::ParallelFor(tbr, tbt, tbz,
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) {
                    // K_r on the Br staggering: (curl J)_r = -dz Jtheta
                    Real btot = 0._rt;
                    if (hyper_resistivity_has_B_dependence) {
                        const Real br_v = Br_a(i, j, 0);
                        const Real bt_v = Interp(Btheta_a, Btheta_stag, Br_stag, coarsen, i, j, 0, 0);
                        const Real bz_v = Interp(Bz_a, Bz_stag, Br_stag, coarsen, i, j, 0, 0);
                        btot = std::sqrt(br_v*br_v + bt_v*bt_v + bz_v*bz_v);
                    }
                    const Real rho_v = Interp(rho_a, nodal, Br_stag, coarsen, i, j, 0, 0);
                    Kr(i, j, 0) = -eta_h(rho_v, btot)
                        * T_Algo::UpwardDz(Jtheta_a, coefs_z, n_coefs_z, i, j, 0, 0);
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) {
                    // K_theta on the Btheta staggering:
                    // (curl J)_theta = dz Jr - dr Jz
                    Real btot = 0._rt;
                    if (hyper_resistivity_has_B_dependence) {
                        const Real br_v = Interp(Br_a, Br_stag, Btheta_stag, coarsen, i, j, 0, 0);
                        const Real bt_v = Btheta_a(i, j, 0);
                        const Real bz_v = Interp(Bz_a, Bz_stag, Btheta_stag, coarsen, i, j, 0, 0);
                        btot = std::sqrt(br_v*br_v + bt_v*bt_v + bz_v*bz_v);
                    }
                    const Real rho_v = Interp(rho_a, nodal, Btheta_stag, coarsen, i, j, 0, 0);
                    Kt(i, j, 0) = eta_h(rho_v, btot)
                        * ( T_Algo::UpwardDz(Jr_a, coefs_z, n_coefs_z, i, j, 0, 0)
                          - T_Algo::UpwardDr(Jz_a, coefs_r, n_coefs_r, i, j, 0, 0) );
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) {
                    // K_z on the Bz staggering (cell-centered in r, no
                    // axis singularity): (curl J)_z = (1/r) dr (r Jtheta)
                    const Real r = rmin_l + (i + 0.5_rt)*dr_l;
                    Real btot = 0._rt;
                    if (hyper_resistivity_has_B_dependence) {
                        const Real br_v = Interp(Br_a, Br_stag, Bz_stag, coarsen, i, j, 0, 0);
                        const Real bt_v = Interp(Btheta_a, Btheta_stag, Bz_stag, coarsen, i, j, 0, 0);
                        const Real bz_v = Bz_a(i, j, 0);
                        btot = std::sqrt(br_v*br_v + bt_v*bt_v + bz_v*bz_v);
                    }
                    const Real rho_v = Interp(rho_a, nodal, Bz_stag, coarsen, i, j, 0, 0);
                    Kz(i, j, 0) = eta_h(rho_v, btot)
                        * T_Algo::UpwardDrr_over_r(Jtheta_a, r, dr_l, coefs_r, n_coefs_r, i, j, 0, 0);
                });
        }
        const auto& period = warpx.Geom(lev).periodicity();
        Kr_mf.FillBoundary(period);
        Kt_mf.FillBoundary(period);
        Kz_mf.FillBoundary(period);

        // One-node-inside interior boxes per E staggering: exclude the
        // r_max wall ring and, when z is non-periodic, both z-end rings
        // (the axis is not a wall and keeps its own handling).
        const Box& dom = warpx.Geom(lev).Domain();
        const bool z_periodic = warpx.Geom(lev).isPeriodic(1);
        auto interior = [&](amrex::IntVect const& iv) {
            Box b = amrex::convert(dom, iv);
            b.growHi(0, -1);
            if (!z_periodic) { b.grow(1, -1); }
            return b;
        };
        hyper_cc_er_box = interior(Efield[0]->ixType().toIntVect());
        hyper_cc_et_box = interior(Efield[1]->ixType().toIntVect());
        hyper_cc_ez_box = interior(Efield[2]->ixType().toIntVect());
    }

    // Exact-operator hyper-resistivity: E_H = +curl(eta_H curl J) with the
    // coefficient inside the outer curl (energy-sign-definite for varying
    // eta_H), precomputed by the caller (HybridPICModel::HybridPICSolveE)
    // from the same plasma current the kernels bind below.
    const bool hyperres_curlcurl = include_hyper_resistivity_term
        && hybrid_model->m_hyper_resistivity_curlcurl;
    ablastr::fields::VectorField eH_mf = {nullptr, nullptr, nullptr};
    if (hyperres_curlcurl) {
        eH_mf = warpx.m_fields.get_alldirs("hybrid_hyperres_E_fp", lev);
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
        if (include_external_fields) {
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

            if (include_external_fields && external_split) {
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

            if (conductor_wall_row && i == iwall_node) {
                enE_nodal(i, j, 0, 0) = 0._rt;
                enE_nodal(i, j, 0, 1) = 0._rt;
                enE_nodal(i, j, 0, 2) = 0._rt;
            }
        });

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

#if defined(WARPX_DIM_RZ)
    bool const include_visc_drag =
        hybrid_model->m_visc_in_ohms_law && include_resistivity && trial_dissipation==nullptr;
    auto const visc_edges = hybrid_model->ViscosityRZEdges(lev);
    bool const write_ev = EV_out != nullptr;
    if (write_ev) {
        for (int c = 0; c < 3; ++c) {
            (*EV_out)[c]->setVal(0.0_rt);
        }
    }
    MultiFab visc_force;
    if (include_visc_drag) {
        visc_force.define(
            amrex::convert(Efield[0]->boxArray(), IntVect::TheNodeVector()),
            Efield[0]->DistributionMap(), 3, IntVect(1));
        hybrid_model->ComputeViscousDragNodal(lev, visc_force, Jfield, Jifield,
                                              rhofield, Bfield);
    }
    auto const visc_vn =
        MakeQdsmcVolumeElement(WarpX::GetInstance().Geom(lev),
                               amrex::IndexType(IntVect::TheNodeVector()));
    auto const visc_vr = MakeQdsmcVolumeElement(WarpX::GetInstance().Geom(lev),
                                                Efield[0]->ixType());
    auto const visc_vz = MakeQdsmcVolumeElement(WarpX::GetInstance().Geom(lev),
                                                Efield[2]->ixType());

#else
    amrex::ignore_unused(EV_out);
#endif

    // Hyper-resistive field mirror (hyper_resistivity_heating): zeroed here so
    // that EB-masked points, the axis pin and the one-node-inside exclusions
    // -- every early return or skipped branch below -- read as "no E_H
    // applied", exactly what the booking must see there.
    const bool write_eh = (EH_out != nullptr);
    if (write_eh) {
        for (int d = 0; d < 3; ++d) {
            (*EH_out)[d]->setVal(0.0_rt);
        }
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
        Array4<Real const> longitudinal0, longitudinal1, longitudinal2;
        if (transverse_offset) {
            longitudinal0=(*transverse_offset)[0]->const_array(mfi);
            longitudinal1=(*transverse_offset)[1]->const_array(mfi);
            longitudinal2=(*transverse_offset)[2]->const_array(mfi);
        }
        Array4<Real const> eiY0, eiY1, eiY2;
        if (direct_yee_inertia) {
            eiY0 = Ei_yee[0]->const_array(mfi);
            eiY1 = Ei_yee[1]->const_array(mfi);
            eiY2 = Ei_yee[2]->const_array(mfi);
        }
        // curlcurl_form (division-free) toroidal sector: the numerator
        // capture target and the measured inertia numerator (see
        // HybridPICModel::SolveEThetaCurlCurlRZ; both nodal). The arrays stay
        // default-constructed on the e_form path and gate the kernel
        // branches.
        Array4<Real> cc_num, cc_num_r, cc_num_z;
        Array4<Real const> eiCC;
        if (hybrid_model->m_esolve_curlcurl) {
            cc_num = hybrid_model->m_num_theta->array(mfi);
            cc_num_r = hybrid_model->m_num_pol[0]->array(mfi);
            cc_num_z = hybrid_model->m_num_pol[1]->array(mfi);
            eiCC = hybrid_model->m_ei_curlcurl_theta->const_array(mfi);
        }
        Array4<Real const> const& rho = rhofield.const_array(mfi);
        Array4<Real const> rho_ped;
        if (rho_ped_mf)
        {
            rho_ped = rho_ped_mf->const_array(mfi);
        }
        Array4<Real const> te_K;
        if (te_mf)
        {
            te_K = te_mf->const_array(mfi);
        }
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
        Array4<Real const> eHr, eHt, eHz;
        if (hyperres_curlcurl) {
            eHr = eH_mf[0]->const_array(mfi);
            eHt = eH_mf[1]->const_array(mfi);
            eHz = eH_mf[2]->const_array(mfi);
        }

        // Curl-curl hyper-resistivity intermediates (default-constructed
        // and never indexed when the mode is off -- the kernels gate on
        // use_hyper_cc).
        Array4<Real const> Kr_cc, Kt_cc, Kz_cc;
        if (use_hyper_cc) {
            Kr_cc = Kr_mf.const_array(mfi);
            Kt_cc = Kt_mf.const_array(mfi);
            Kz_cc = Kz_mf.const_array(mfi);
        }
        // E_H mirror arrays (default-constructed and never indexed unless
        // EH_out was passed -- the kernels gate the write on the Array4).
#if defined(WARPX_DIM_RZ)
        Array4<Real const> visc;
        Array4<Real> ev_r, ev_t, ev_z;
        if (include_visc_drag) {
            visc = visc_force.const_array(mfi);
        }
        if (write_ev) {
            ev_r = (*EV_out)[0]->array(mfi);
            ev_t = (*EV_out)[1]->array(mfi);
            ev_z = (*EV_out)[2]->array(mfi);
        }
#endif
        Array4<Real> end_r, end_t, end_z, gate_r, gate_t, gate_z;
        if (write_damping) {
            end_r = damping_out->end_resistivity[0]->array(mfi);
            gate_r = damping_out->end_holmstrom[0]->array(mfi);
            end_t = damping_out->end_resistivity[1]->array(mfi);
            gate_t = damping_out->end_holmstrom[1]->array(mfi);
            end_z = damping_out->end_resistivity[2]->array(mfi);
            gate_z = damping_out->end_holmstrom[2]->array(mfi);
        }
        Array4<Real> ph_r, ph_t, ph_z;
        if (F_pressure_hall_out) {
            ph_r = (*F_pressure_hall_out)[0]->array(mfi);
            ph_t = (*F_pressure_hall_out)[1]->array(mfi);
            ph_z = (*F_pressure_hall_out)[2]->array(mfi);
        }
        Array4<Real> er_r, er_t, er_z;
        if (write_er) {
            er_r = (*ER_out)[0]->array(mfi);
            er_t = (*ER_out)[1]->array(mfi);
            er_z = (*ER_out)[2]->array(mfi);
        }
        Array4<Real> eh_r, eh_t, eh_z;
        if (write_eh) {
            eh_r = (*EH_out)[0]->array(mfi);
            eh_t = (*EH_out)[1]->array(mfi);
            eh_z = (*EH_out)[2]->array(mfi);
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
        if (include_external_fields) {
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
                if (update_Er_arr && update_Er_arr(i, j, 0) == 0) {
                    if (cc_num_r) { cc_num_r(i, j, 0) = 0._rt; }
                    return;
                }

                // Interpolate to get the appropriate charge density in space
                const Real rho_val = Interp(rho, nodal, Er_stag, coarsen, i, j, 0, 0);
                CompensatedOhm::Pair quotient_hall{0.,0.}, quotient_grad{0.,0.};
                CompensatedOhm::Pair quotient_density{1.,0.};
                Real quotient_gate=0._rt;

                // curlcurl_form poloidal sector: capture the multiplied-through
                // numerator (division-free) for the grad-div-completed
                // vector-Helmholtz solve; Er is zeroed so the resistive
                // blocks accumulate cleanly for the caller's fold.
                if (cc_num_r) {
                    const Real grad_Pe_a =
                        (!solve_for_Faraday && include_electron_pressure_term) ?
                        T_Algo::UpwardDr(Pe, coefs_r, n_coefs_r, i, j, 0, 0)
                        : 0._rt;
                    const auto enE_ra = Interp(enE, nodal, Er_stag, coarsen, i, j, 0, 0);
                    cc_num_r(i, j, 0) = enE_ra - grad_Pe_a
                        + Interp(eiCC, nodal, Er_stag, coarsen, i, j, 0, 0);
                    Er(i, j, 0) = 0._rt;
                } else {

                // Axis-confinement mask (1 = gate eligible); Er is
                // cell-centered in r.
                const Real r_gate_c = rmin + (i + 0.5_rt)*dr;
                const Real ax_mask = !holmstrom_axis_confined ? 1._rt
                    : (holmstrom_axis_inv_w > 0._rt
                        ? 0.5_rt * (1._rt - std::tanh(
                            (r_gate_c - holmstrom_axis_r) * holmstrom_axis_inv_w))
                        : (r_gate_c < holmstrom_axis_r ? 1._rt : 0._rt));

                if (rho_val < rho_floor && holmstrom_vacuum_region && !holmstrom_smooth
                    && (!holmstrom_axis_confined || r_gate_c < holmstrom_axis_r)) {
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
                    const auto rho_val_limited =
                        HybridSmoothFloor(rho_val + (use_pedestal ? Interp(rho_ped, nodal, Er_stag,
                                                                           coarsen, i, j, 0, 0)
                                                                  : 0.0_rt),
                                          rho_floor, floor_w);

                    Real ohm_val = (enE_r - grad_Pe) / rho_val_limited;
                    if (transverse_offset) {
                        int const axis=0;
                        quotient_hall=CompensatedOhm::Average(enE,i,j,0,0,axis);
                        quotient_grad=(solve_for_Faraday ? add_grad_pe_faraday
                            : include_electron_pressure_term)
                            ? CompensatedOhm::Gradient(Pe,i,j,0,axis,coefs_r[0])
                            : CompensatedOhm::Pair{0.,0.};
                        quotient_density={rho_val_limited,0.}; quotient_gate=1._rt;
                        if(floor_w==0. && rho_val_limited>rho_floor) {
                            auto exact_density=CompensatedOhm::Average(rho,i,j,0,0,axis);
                            if(use_pedestal) exact_density=CompensatedOhm::Add(exact_density,
                                CompensatedOhm::Average(rho_ped,i,j,0,0,axis));
                            quotient_density=CompensatedOhm::WithHigh(exact_density,rho_val_limited);
                        }
                    }
                    // Conductor-wall stack, gate-on-raw / divide-by-floored:
                    // where the RAW (unfloored) rho at this stencil location
                    // is not positive -- the wall node row is zeroed by
                    // convention and its ghosts odd-imaged, so interpolations
                    // at/beyond the wall land <= 0 -- the Hall/motional and
                    // grad-Pe force is identically zero instead of the
                    // 1/rho_floor-amplified enE/max(rho, rho_floor):
                    // where(rho_raw > 0, enE/max(rho, rho_floor), 0).
                    if (conductor_wall_row && !(rho_val > 0._rt)) {
                        ohm_val = 0._rt;
                        if (transverse_offset) { quotient_gate=0._rt; }
                    }
                    if (holmstrom_smooth) {
                        const Real g = 0.5_rt * (1._rt + std::tanh(
                            (rho_val - rho_floor) * holmstrom_inv_width));
                        // Legacy arithmetic when unconfined (bit-identical).
                        ohm_val *= (holmstrom_axis_confined
                                    ? 1._rt - (1._rt - g)*ax_mask : g);
                        if (transverse_offset) quotient_gate *= (holmstrom_axis_confined
                            ? 1._rt - (1._rt - g)*ax_mask : g);
                    }
                    Er(i, j, 0) = ohm_val;
                }
                if (end_region.holmstrom)
                {
                    Real const before_end = write_damping ? Er(i, j, 0) : 0._rt;
                    Er(i, j, 0) *= 1. - end_region.Weight(i, j, 0, Er_stag);
                    if (transverse_offset) quotient_gate *= 1. - end_region.Weight(i,j,0,Er_stag);
                    if (write_damping) { gate_r(i, j, 0) = Er(i, j, 0) - before_end; }
                }
                if (ph_r) { ph_r(i, j, 0) = Er(i, j, 0); }
                if (transverse_offset) {
                    Er(i,j,0)=CompensatedOhm::OffsetQuotient(quotient_hall,quotient_grad,
                        quotient_density,quotient_gate,longitudinal0(i,j,0));
                }
                if (include_electron_inertia) {
                    Er(i, j, 0) += direct_yee_inertia ? eiY0(i, j, 0)
                    : Interp(eiN, nodal, Er_stag, coarsen, i, j, 0, 0);
                }

                } // end !cc_num_r


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

                    if (write_er) {
                        Real const global_eta_field =
                            (eta_has_Te
                                 ? eta_te(rho_val, jtot_val,
                                          Interp(te_K, nodal, Er_stag, coarsen, i, j, 0, 0), t_new)
                                 : eta(rho_val, jtot_val, t_new)) *
                            Jr(i, j, 0);
                        Er(i, j, 0) += global_eta_field;
                        er_r(i, j, 0) = global_eta_field;
                    } else {
                        Er(i, j, 0) +=
                            (eta_has_Te
                                 ? eta_te(rho_val, jtot_val,
                                          Interp(te_K, nodal, Er_stag, coarsen, i, j, 0, 0), t_new)
                                 : eta(rho_val, jtot_val, t_new)) *
                            Jr(i, j, 0);
                    }
                    if (end_region.resistivity > 0. && include_resistivity)
                    {
                        if (write_damping) {
                            Real const end_field = end_region.resistivity *
                                end_region.Weight(i, j, 0, Er_stag) * Jr(i, j, 0);
                            Er(i, j, 0) += end_field;
                            end_r(i, j, 0) = end_field;
                        } else {
                        Er(i, j, 0) += end_region.resistivity *
                            end_region.Weight(i, j, 0, Er_stag) * Jr(i, j, 0);
                        }
                    }
                    // Per-species resistive overlay (Phys. Plasmas 31, 012902 (2024)); zero
                    // when no per-species eta is registered.
                    if (has_eta_overlay) { Er(i, j, 0) += eta_overlay_r(i, j, 0); }

                    if (hyperres_curlcurl && include_resistivity) {
                        // Exact operator: E_H = +curl(eta_H curl J), the
                        // coefficient inside the outer curl; composed by
                        // the caller (eta_H face-averaged there).
                        Er(i, j, 0) += eHr(i, j, 0);
                    } else if (include_hyper_resistivity_term && include_resistivity) {
                        // Truncated identity (-lap J = curl curl J only at
                        // div J = 0); trajectory-preserving default.

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
                        // E_H is formed as one value, added (E += eh, with
                        // eh = -X exact, so bit-identical to the former
                        // E -= X) and mirrored to EH_out for the booking.
                        Real eh = 0._rt;
                        if (use_hyper_cc) {
                            // Adjoint curl-curl form: E_H = +(curl K)_r
                            // = -dz K_theta; the one-node-inside box ends
                            // the operator off non-periodic walls.
                            if (hyper_cc_er_box.contains(
                                    amrex::IntVect(AMREX_D_DECL(i, j, 0)))) {
                                eh = -T_Algo::DownwardDz(Kt_cc, coefs_z,
                                                         n_coefs_z, i, j, 0, 0);
                            }
                        } else {
                        auto nabla2Jr = T_Algo::Dr_rDr_over_r(Jr, r, dr, coefs_r, n_coefs_r, i, j, 0, 0)
                            + T_Algo::Dzz(Jr, coefs_z, n_coefs_z, i, j, 0, 0) - Jr(i, j, 0)/(r*r);

                        eh = -eta_h(rho_val, btot_val) * nabla2Jr;
                        }
                        Er(i, j, 0) += eh;
                        if (eh_r) {
                            eh_r(i, j, 0) = eh;
                        }
                    }
                }

#if defined(WARPX_DIM_RZ)
                if (include_visc_drag && visc_edges.active(i, j, 0)) {
                    Real const ev = 0.5_rt *
                                    (visc_vn(i, j) * visc(i, j, 0, 0) +
                                     visc_vn(i + 1, j) * visc(i + 1, j, 0, 0)) /
                                    visc_vr(i, j);
                    Er(i, j, 0) += ev;
                    if (ev_r) {
                        ev_r(i, j, 0) = ev;
                    }
                }
#endif

                if (include_external_fields && !cc_num_r) {
                    const amrex::Real w_ext = subtract_E_ext_everywhere
                        ? 1._rt
                        : HybridExtSubWeight(rho_val, rho_floor, floor_w);
                    Er(i, j, 0) -= w_ext * Er_ext(i, j, 0);
                }
            },

            // Etheta calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Etheta_arr && update_Etheta_arr(i, j, 0) == 0) {
                    if (cc_num) { cc_num(i, j, 0) = 0._rt; }
                    return;
                }

                // r on a nodal grid (Etheta is nodal in r)
                Real const r = rmin + i*dr;
                // Mode m=0: // Ensure that Etheta remains 0 on axis
                if (r < 0.5_rt*dr) {
                    Etheta(i, j, 0, 0) = 0.;
                    if (cc_num) { cc_num(i, j, 0) = 0._rt; }
                    return;
                }

                // Interpolate to get the appropriate charge density in space
                const Real rho_val = Interp(rho, nodal, Etheta_stag, coarsen, i, j, 0, 0);
                CompensatedOhm::Pair quotient_hall{0.,0.}, quotient_grad{0.,0.};
                CompensatedOhm::Pair quotient_density{1.,0.};
                Real quotient_gate=0._rt;

                // curlcurl_form (division-free) toroidal sector: assemble the
                // multiplied-through numerator e n E_theta_num = the
                // (J - J_i) x B numerator plus the measured inertia
                // numerator (both division-free); Etheta is zeroed so the
                // resistive/hyper-resistive blocks below accumulate their
                // E-valued terms cleanly for the caller to fold into the
                // numerator (times e rho) ahead of the elliptic solve.
                // No vacuum branch and no floor exist on this path.
                if (cc_num) {
                    const auto enE_t = Interp(enE, nodal, Etheta_stag,
                                              coarsen, i, j, 0, 1);
                    cc_num(i, j, 0) = enE_t + eiCC(i, j, 0, 1);
                    Etheta(i, j, 0) = 0._rt;
                } else {

                // Axis-confinement mask (1 = gate eligible); reuses the
                // nodal r computed above.
                const Real ax_mask = !holmstrom_axis_confined ? 1._rt
                    : (holmstrom_axis_inv_w > 0._rt
                        ? 0.5_rt * (1._rt - std::tanh(
                            (r - holmstrom_axis_r) * holmstrom_axis_inv_w))
                        : (r < holmstrom_axis_r ? 1._rt : 0._rt));

                if (rho_val < rho_floor && holmstrom_vacuum_region && !holmstrom_smooth
                    && (!holmstrom_axis_confined || r < holmstrom_axis_r)) {
                    Etheta(i, j, 0) = 0._rt;
                }
                else
                {
                    // Get the gradient of the electron pressure
                    // -> d/dt = 0 for m = 0
                    const auto grad_Pe = 0.0_rt;

                    // interpolate the nodal neE values to the Yee grid
                    const auto enE_t = Interp(enE, nodal, Etheta_stag, coarsen, i, j, 0, 1);

                    // safety condition since we divide by rho
                    const auto rho_val_limited = HybridSmoothFloor(
                        rho_val + (use_pedestal
                                       ? Interp(rho_ped, nodal, Etheta_stag, coarsen, i, j, 0, 0)
                                       : 0.0_rt),
                        rho_floor, floor_w);

                    Real ohm_val = (enE_t - grad_Pe) / rho_val_limited;
                    if (transverse_offset) {
                        int const axis=-1;
                        quotient_hall=CompensatedOhm::Average(enE,i,j,0,1,axis);
                        quotient_grad=(solve_for_Faraday ? add_grad_pe_faraday
                            : include_electron_pressure_term)
                            ? CompensatedOhm::Gradient(Pe,i,j,0,axis,0._rt)
                            : CompensatedOhm::Pair{0.,0.};
                        quotient_density={rho_val_limited,0.}; quotient_gate=1._rt;
                        if(floor_w==0. && rho_val_limited>rho_floor) {
                            auto exact_density=CompensatedOhm::Average(rho,i,j,0,0,axis);
                            if(use_pedestal) exact_density=CompensatedOhm::Add(exact_density,
                                CompensatedOhm::Average(rho_ped,i,j,0,0,axis));
                            quotient_density=CompensatedOhm::WithHigh(exact_density,rho_val_limited);
                        }
                    }
                    // Conductor-wall stack, gate-on-raw / divide-by-floored
                    // (see the Er branch).
                    if (conductor_wall_row && !(rho_val > 0._rt)) {
                        ohm_val = 0._rt;
                        if (transverse_offset) { quotient_gate=0._rt; }
                    }
                    if (holmstrom_smooth) {
                        const Real g = 0.5_rt * (1._rt + std::tanh(
                            (rho_val - rho_floor) * holmstrom_inv_width));
                        // Legacy arithmetic when unconfined (bit-identical).
                        ohm_val *= (holmstrom_axis_confined
                                    ? 1._rt - (1._rt - g)*ax_mask : g);
                        if (transverse_offset) quotient_gate *= (holmstrom_axis_confined
                            ? 1._rt - (1._rt - g)*ax_mask : g);
                    }
                    Etheta(i, j, 0) = ohm_val;
                }
                if (end_region.holmstrom)
                {
                    Real const before_end = write_damping ? Etheta(i, j, 0) : 0._rt;
                    Etheta(i, j, 0) *= 1. - end_region.Weight(i, j, 0, Etheta_stag);
                    if (transverse_offset) quotient_gate *= 1. - end_region.Weight(i,j,0,Etheta_stag);
                    if (write_damping) { gate_t(i, j, 0) = Etheta(i, j, 0) - before_end; }
                }
                if (ph_t) { ph_t(i, j, 0) = Etheta(i, j, 0); }
                if (transverse_offset) {
                    Etheta(i,j,0)=CompensatedOhm::OffsetQuotient(quotient_hall,quotient_grad,
                        quotient_density,quotient_gate,longitudinal1(i,j,0));
                }
                if (include_electron_inertia) {
                    Etheta(i, j, 0) += direct_yee_inertia ? eiY1(i, j, 0)
                    : Interp(eiN, nodal, Etheta_stag, coarsen, i, j, 0, 1);
                }

                } // end !cc_num


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

                    if (write_er) {
                        Real const global_eta_field =
                            (eta_has_Te
                                 ? eta_te(rho_val, jtot_val,
                                          Interp(te_K, nodal, Etheta_stag, coarsen, i, j, 0, 0), t_new)
                                 : eta(rho_val, jtot_val, t_new)) *
                            Jtheta(i, j, 0);
                        Etheta(i, j, 0) += global_eta_field;
                        er_t(i, j, 0) = global_eta_field;
                    } else {
                        Etheta(i, j, 0) +=
                            (eta_has_Te
                                 ? eta_te(rho_val, jtot_val,
                                          Interp(te_K, nodal, Etheta_stag, coarsen, i, j, 0, 0), t_new)
                                 : eta(rho_val, jtot_val, t_new)) *
                            Jtheta(i, j, 0);
                    }
                    if (end_region.resistivity > 0. && include_resistivity)
                    {
                        if (write_damping) {
                            Real const end_field = end_region.resistivity *
                                end_region.Weight(i, j, 0, Etheta_stag) * Jtheta(i, j, 0);
                            Etheta(i, j, 0) += end_field;
                            end_t(i, j, 0) = end_field;
                        } else {
                        Etheta(i, j, 0) += end_region.resistivity *
                            end_region.Weight(i, j, 0, Etheta_stag) * Jtheta(i, j, 0);
                        }
                    }
                    if (has_eta_overlay) { Etheta(i, j, 0) += eta_overlay_t(i, j, 0); }

                    if (hyperres_curlcurl && include_resistivity) {
                        // Exact operator: E_H = +curl(eta_H curl J), the
                        // coefficient inside the outer curl; composed by
                        // the caller (the axis row returned early above,
                        // keeping Etheta = 0 there).
                        Etheta(i, j, 0) += eHt(i, j, 0);
                    } else if (include_hyper_resistivity_term && include_resistivity) {
                        // Truncated identity; trajectory-preserving default.

                        // Interpolate B field to appropriate staggering to match E field
                        Real btot_val = 0._rt;
                        if (hyper_resistivity_has_B_dependence) {
                            const Real br_val = Interp(Br, Br_stag, Etheta_stag, coarsen, i, j, 0, 0);
                            const Real bt_val = Interp(Btheta, Btheta_stag, Etheta_stag, coarsen, i, j, 0, 0);
                            const Real bz_val = Interp(Bz, Bz_stag, Etheta_stag, coarsen, i, j, 0, 0);
                            btot_val = std::sqrt(br_val*br_val + bt_val*bt_val + bz_val*bz_val);
                        }

                        // E_H as one value, added and mirrored (see Er).
                        Real eh = 0._rt;
                        if (use_hyper_cc) {
                            // Adjoint curl-curl form: E_H = +(curl K)_theta
                            // = dz K_r - dr K_z (this branch runs off-axis
                            // only -- the kernel returned at r < dr/2 with
                            // Etheta pinned to 0); one-node-inside box ends
                            // the operator off non-periodic walls.
                            if (hyper_cc_et_box.contains(
                                    amrex::IntVect(AMREX_D_DECL(i, j, 0)))) {
                                eh = T_Algo::DownwardDz(Kr_cc, coefs_z,
                                                        n_coefs_z, i, j, 0, 0) -
                                     T_Algo::DownwardDr(Kz_cc, coefs_r,
                                                        n_coefs_r, i, j, 0, 0);
                            }
                        } else {
                        // Special handling of the hyper-resistivity term on axis to avoid division by zero
                        // and ensure that Etheta remains 0 on axis for m=0 mode
                        auto nabla2Jtheta = 0.0_rt;
                        if (r > 0.0_rt) {
                            nabla2Jtheta = T_Algo::Dr_rDr_over_r(Jtheta, r, dr, coefs_r, n_coefs_r, i, j, 0, 0)
                                + T_Algo::Dzz(Jtheta, coefs_z, n_coefs_z, i, j, 0, 0) - Jtheta(i, j, 0)/(r*r);
                        }

                        eh = -eta_h(rho_val, btot_val) * nabla2Jtheta;
                        }
                        Etheta(i, j, 0) += eh;
                        if (eh_t) {
                            eh_t(i, j, 0) = eh;
                        }
                    }
                }

#if defined(WARPX_DIM_RZ)
                if (include_visc_drag && visc_edges.active(i, j, 1)) {
                    Real const ev = visc(i, j, 0, 1);
                    Etheta(i, j, 0) += ev;
                    if (ev_t) {
                        ev_t(i, j, 0) = ev;
                    }
                }
#endif

                if (include_external_fields && !cc_num) {
                    // curlcurl_form defers the (unconditional) E_ext_theta
                    // subtraction to after the elliptic solve -- there is
                    // no vacuum branch to gate on.
                    const amrex::Real w_ext = subtract_E_ext_everywhere
                        ? 1._rt
                        : HybridExtSubWeight(rho_val, rho_floor, floor_w);
                    Etheta(i, j, 0) -= w_ext * Etheta_ext(i, j, 0);
                }
            },

            // Ez calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Ez_arr && update_Ez_arr(i, j, 0) == 0) {
                    if (cc_num_z) { cc_num_z(i, j, 0) = 0._rt; }
                    return;
                }

                // Interpolate to get the appropriate charge density in space
                const Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, 0, 0);
                CompensatedOhm::Pair quotient_hall{0.,0.}, quotient_grad{0.,0.};
                CompensatedOhm::Pair quotient_density{1.,0.};
                Real quotient_gate=0._rt;

                // curlcurl_form poloidal sector: numerator capture (see the
                // Er branch).
                if (cc_num_z) {
                    const Real grad_Pe_a =
                        (!solve_for_Faraday && include_electron_pressure_term) ?
                        T_Algo::UpwardDz(Pe, coefs_z, n_coefs_z, i, j, 0, 0)
                        : 0._rt;
                    const auto enE_za = Interp(enE, nodal, Ez_stag, coarsen, i, j, 0, 2);
                    cc_num_z(i, j, 0) = enE_za - grad_Pe_a
                        + Interp(eiCC, nodal, Ez_stag, coarsen, i, j, 0, 2);
                    Ez(i, j, 0) = 0._rt;
                } else {

                // Axis-confinement mask (1 = gate eligible); Ez is nodal
                // in r.
                const Real r_gate_n = rmin + i*dr;
                const Real ax_mask = !holmstrom_axis_confined ? 1._rt
                    : (holmstrom_axis_inv_w > 0._rt
                        ? 0.5_rt * (1._rt - std::tanh(
                            (r_gate_n - holmstrom_axis_r) * holmstrom_axis_inv_w))
                        : (r_gate_n < holmstrom_axis_r ? 1._rt : 0._rt));

                if (rho_val < rho_floor && holmstrom_vacuum_region && !holmstrom_smooth
                    && (!holmstrom_axis_confined || r_gate_n < holmstrom_axis_r)) {
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
                    const auto rho_val_limited =
                        HybridSmoothFloor(rho_val + (use_pedestal ? Interp(rho_ped, nodal, Ez_stag,
                                                                           coarsen, i, j, 0, 0)
                                                                  : 0.0_rt),
                                          rho_floor, floor_w);

                    Real ohm_val = (enE_z - grad_Pe) / rho_val_limited;
                    if (transverse_offset) {
                        int const axis=AMREX_SPACEDIM==2 ? 1 : -1;
                        quotient_hall=CompensatedOhm::Average(enE,i,j,0,2,axis);
                        quotient_grad=(solve_for_Faraday ? add_grad_pe_faraday
                            : include_electron_pressure_term)
                            ? CompensatedOhm::Gradient(Pe,i,j,0,axis,axis<0 ? 0._rt : coefs_z[0])
                            : CompensatedOhm::Pair{0.,0.};
                        quotient_density={rho_val_limited,0.}; quotient_gate=1._rt;
                        if(floor_w==0. && rho_val_limited>rho_floor) {
                            auto exact_density=CompensatedOhm::Average(rho,i,j,0,0,axis);
                            if(use_pedestal) exact_density=CompensatedOhm::Add(exact_density,
                                CompensatedOhm::Average(rho_ped,i,j,0,0,axis));
                            quotient_density=CompensatedOhm::WithHigh(exact_density,rho_val_limited);
                        }
                    }
                    // Conductor-wall stack, gate-on-raw / divide-by-floored
                    // (see the Er branch).
                    if (conductor_wall_row && !(rho_val > 0._rt)) {
                        ohm_val = 0._rt;
                        if (transverse_offset) { quotient_gate=0._rt; }
                    }
                    if (holmstrom_smooth) {
                        const Real g = 0.5_rt * (1._rt + std::tanh(
                            (rho_val - rho_floor) * holmstrom_inv_width));
                        // Legacy arithmetic when unconfined (bit-identical).
                        ohm_val *= (holmstrom_axis_confined
                                    ? 1._rt - (1._rt - g)*ax_mask : g);
                        if (transverse_offset) quotient_gate *= (holmstrom_axis_confined
                            ? 1._rt - (1._rt - g)*ax_mask : g);
                    }
                    Ez(i, j, 0) = ohm_val;
                }
                if (end_region.holmstrom)
                {
                    Real const before_end = write_damping ? Ez(i, j, 0) : 0._rt;
                    Ez(i, j, 0) *= 1. - end_region.Weight(i, j, 0, Ez_stag);
                    if (transverse_offset) quotient_gate *= 1. - end_region.Weight(i,j,0,Ez_stag);
                    if (write_damping) { gate_z(i, j, 0) = Ez(i, j, 0) - before_end; }
                }
                if (ph_z) { ph_z(i, j, 0) = Ez(i, j, 0); }
                if (transverse_offset) {
                    Ez(i,j,0)=CompensatedOhm::OffsetQuotient(quotient_hall,quotient_grad,
                        quotient_density,quotient_gate,longitudinal2(i,j,0));
                }
                if (include_electron_inertia) {
                    Ez(i, j, 0) += direct_yee_inertia ? eiY2(i, j, 0)
                    : Interp(eiN, nodal, Ez_stag, coarsen, i, j, 0, 2);
                }

                } // end !cc_num_z


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

                    if (write_er) {
                        Real const global_eta_field =
                            (eta_has_Te
                                 ? eta_te(rho_val, jtot_val,
                                          Interp(te_K, nodal, Ez_stag, coarsen, i, j, 0, 0), t_new)
                                 : eta(rho_val, jtot_val, t_new)) *
                            Jz(i, j, 0);
                        Ez(i, j, 0) += global_eta_field;
                        er_z(i, j, 0) = global_eta_field;
                    } else {
                        Ez(i, j, 0) +=
                            (eta_has_Te
                                 ? eta_te(rho_val, jtot_val,
                                          Interp(te_K, nodal, Ez_stag, coarsen, i, j, 0, 0), t_new)
                                 : eta(rho_val, jtot_val, t_new)) *
                            Jz(i, j, 0);
                    }
                    if (end_region.resistivity > 0. && include_resistivity)
                    {
                        if (write_damping) {
                            Real const end_field = end_region.resistivity *
                                end_region.Weight(i, j, 0, Ez_stag) * Jz(i, j, 0);
                            Ez(i, j, 0) += end_field;
                            end_z(i, j, 0) = end_field;
                        } else {
                        Ez(i, j, 0) += end_region.resistivity *
                            end_region.Weight(i, j, 0, Ez_stag) * Jz(i, j, 0);
                        }
                    }
                    if (has_eta_overlay) { Ez(i, j, 0) += eta_overlay_z(i, j, 0); }

                    if (hyperres_curlcurl && include_resistivity) {
                        // Exact operator: E_H = +curl(eta_H curl J), the
                        // coefficient inside the outer curl; composed by
                        // the caller (the Ampere-closure stencil owns the
                        // axis row, with the correct axis limit).
                        Ez(i, j, 0) += eHz(i, j, 0);
                    } else if (include_hyper_resistivity_term && include_resistivity) {
                        // Truncated identity; trajectory-preserving default.

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

                        // E_H as one value, added and mirrored (see Er).
                        Real eh = 0._rt;
                        if (use_hyper_cc) {
                            // Adjoint curl-curl form: E_H = +(curl K)_z
                            // = (1/r) dr (r K_theta), with the exact
                            // Ampere-style on-axis regularization
                            // (K_theta is linear in r near the axis);
                            // one-node-inside box ends the operator off
                            // non-periodic walls.
                            if (hyper_cc_ez_box.contains(
                                    amrex::IntVect(AMREX_D_DECL(i, j, 0)))) {
                                if (r > 0.5_rt*dr) {
                                    eh = T_Algo::DownwardDrr_over_r(
                                        Kt_cc, r, dr, coefs_r, n_coefs_r, i, j,
                                        0, 0);
                                } else {
                                    eh = 4._rt * Kt_cc(i, j, 0) / dr;
                                }
                            }
                        } else {
                            auto nabla2Jz =
                                T_Algo::Dzz(Jz, coefs_z, n_coefs_z, i, j, 0, 0);
                            if (r > 0.5_rt * dr) {
                                nabla2Jz += T_Algo::Dr_rDr_over_r(
                                    Jz, r, dr, coefs_r, n_coefs_r, i, j, 0, 0);
                            } else {
                                // Control-volume axis limit. The Ampere current's
                                // negative-radius guard is not necessarily an even
                                // mirror, so do not differentiate through it.
                                nabla2Jz += 4.0_rt * (Jz(i+1,j,0,0)-Jz(i,j,0,0)) / (dr*dr);
                            }

                            eh = -eta_h(rho_val, btot_val) * nabla2Jz;
                        }
                        Ez(i, j, 0) += eh;
                        if (eh_z) {
                            eh_z(i, j, 0) = eh;
                        }
                    }
                }

#if defined(WARPX_DIM_RZ)
                if (include_visc_drag && visc_edges.active(i, j, 2)) {
                    Real const ev = 0.5_rt *
                                    (visc_vn(i, j) * visc(i, j, 0, 2) +
                                     visc_vn(i, j + 1) * visc(i, j + 1, 0, 2)) /
                                    visc_vz(i, j);
                    Ez(i, j, 0) += ev;
                    if (ev_z) {
                        ev_z(i, j, 0) = ev;
                    }
                }
#endif

                if (include_external_fields && !cc_num_z) {
                    const amrex::Real w_ext = subtract_E_ext_everywhere
                        ? 1._rt
                        : HybridExtSubWeight(rho_val, rho_floor, floor_w);
                    Ez(i, j, 0) -= w_ext * Ez_ext(i, j, 0);
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
    // Use exactly the current trial's projected force increments for the
    // thermal work. A null context preserves every legacy path; no-resistivity
    // evaluations omit both increments for the existing particle correction.
    if (trial_dissipation && include_resistivity) {
        auto const hyper=trial_dissipation->HyperField();
        auto const viscous=trial_dissipation->ViscousField();
        for (int c=0;c<3;++c) {
            amrex::MultiFab::Add(*Efield[c],*hyper[c],0,0,1,0);
            if (hybrid_model->m_visc_in_ohms_law) { amrex::MultiFab::Add(*Efield[c],*viscous[c],0,0,1,0); }
            if (EH_out) { amrex::MultiFab::Copy(*(*EH_out)[c],*hyper[c],0,0,1,0); }
            if (EV_out && hybrid_model->m_visc_in_ohms_law) { amrex::MultiFab::Copy(*(*EV_out)[c],*viscous[c],0,0,1,0); }
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

template <typename T_Algo>
void
FiniteDifferenceSolver::HybridPICSolveECartesian (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Bfield, amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update_E,
    int lev, HybridPICModel const* hybrid_model, const bool solve_for_Faraday,
    const bool include_resistivity, ablastr::fields::VectorField const* EH_out,
    ablastr::fields::VectorField const* EV_out,
    ablastr::fields::VectorField const* ER_out,
    const warpx::thermal::HybridOhmDampingFields* damping_out,
    ablastr::fields::VectorField const* F_pressure_hall_out,
    ablastr::fields::VectorField const* transverse_offset) {
    bool const write_er = ER_out != nullptr;
    bool const write_damping = damping_out != nullptr;
    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    using namespace ablastr::coarsen::sample;

    // get hybrid model parameters
    const auto eta = hybrid_model->m_eta;
    const auto end_region = hybrid_model->EndRegion(lev);
    // End eta is numerical magnetic damping. The legacy thermal path heats
    // electrons with it; the opt-in edge-work path books its signed sink
    // separately. Full-minus-nores removes its ion force, and no physical
    // OU/drag rate includes this increment.
    const auto eta_te = hybrid_model->m_eta_te;
    const bool eta_has_Te = hybrid_model->m_resistivity_has_Te_dependence;
    amrex::MultiFab const* const te_mf = hybrid_model->ResistivityTe(lev);
    const auto eta_h = hybrid_model->m_eta_h;
    const auto rho_floor = hybrid_model->m_n_floor * PhysConst::q_e;
    const auto floor_w = hybrid_model->m_n_floor_smooth_width * rho_floor;
    const auto resistivity_has_J_dependence = hybrid_model->m_resistivity_has_J_dependence;
    const auto hyper_resistivity_has_B_dependence = hybrid_model->m_hyper_resistivity_has_B_dependence;
    const bool include_hyper_resistivity_term = hybrid_model->m_include_hyper_resistivity_term;
    const bool include_electron_inertia = hybrid_model->m_include_electron_inertia && transverse_offset==nullptr;

    const bool include_external_fields = hybrid_model->m_add_external_fields
        && !hybrid_model->m_external_unified;
    const bool subtract_E_ext_everywhere =
        hybrid_model->m_external_e_subtraction_unconditional;
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
    bool const direct_yee_inertia = hybrid_model->UseCompatibleYeeInertia();
    ablastr::fields::VectorField Ei_yee;
    if (direct_yee_inertia) {
        Ei_yee = warpx.m_fields.get_alldirs("hybrid_E_inertial_fp", lev);
    }
    // Density pedestal (change of variables, HybridPICModel::m_density_pedestal):
    // the Hall / grad Pe divisor is max(rho + rho_ped, rho_floor) instead of
    // max(rho, rho_floor); the Holmstrom gate, the external-E subtraction
    // and eta(rho, J, t) keep the deposited rho.
    amrex::MultiFab const* const rho_ped_mf = hybrid_model->DensityPedestal(lev);
    const bool use_pedestal = (rho_ped_mf != nullptr);
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

    // Exact-operator hyper-resistivity: E_H = +curl(eta_H curl J) with the
    // coefficient inside the outer curl (energy-sign-definite for varying
    // eta_H), precomputed by the caller (HybridPICModel::HybridPICSolveE)
    // from the same plasma current the kernels bind below.
    const bool hyperres_curlcurl = include_hyper_resistivity_term
        && hybrid_model->m_hyper_resistivity_curlcurl;
    ablastr::fields::VectorField eH_mf = {nullptr, nullptr, nullptr};
    if (hyperres_curlcurl) {
        eH_mf = warpx.m_fields.get_alldirs("hybrid_hyperres_E_fp", lev);
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
        if (include_external_fields) {
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

            if (include_external_fields && external_split) {
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

    // Hyper-resistive field mirror (hyper_resistivity_heating): zeroed here so
    // that EB-masked points and the one-node-inside exclusions -- every early
    // return or skipped branch below -- read as "no E_H applied", exactly
    // what the booking must see there.
    const bool write_eh = (EH_out != nullptr);
    if (write_eh) {
        for (int d = 0; d < 3; ++d) {
            (*EH_out)[d]->setVal(0.0_rt);
        }
    }

    // Braginskii electron viscous DRAG (qdsmc_viscosity_in_ohms_law; see the
    // m_visc_in_ohms_law member doc): E_visc = -div(Pi_e)/rho added in the
    // Dissipative solves only (like eta_H, the push field excludes it). The
    // nodal drag F = -div_c(Pi_e)/max(rho, rho_floor) is precomputed from the
    // SAME u_e, strain and capped coefficients as the booked heating
    // (ComputeViscousDragNodal: centred divergence over the floored nodal
    // rho, one exchanged ghost layer); the kernels below AVERAGE it onto
    // each Yee edge -- the transpose of the edge-to-node average that built
    // u_e from J -- so the whole chain is the exact adjoint of the gradient
    // chain and Sum J . E_visc = Sum Q_nu discretely. 3D only (asserted at
    // input); the EV_out mirror feeds the work booking exactly as EH_out
    // feeds the hyper-resistive one.
    const bool include_visc_drag =
        hybrid_model->m_visc_in_ohms_law && include_resistivity;
    const bool write_ev = (EV_out != nullptr);
    if (write_ev) {
        for (int d = 0; d < 3; ++d) {
            (*EV_out)[d]->setVal(0.0_rt);
        }
    }
    MultiFab F_mf;
    if (include_visc_drag) {
        F_mf.define(amrex::convert(Efield[0]->boxArray(),
                                   amrex::IntVect::TheNodeVector()),
                    Efield[0]->DistributionMap(), 3, IntVect(1));
        hybrid_model->ComputeViscousDragNodal(lev, F_mf, Jfield, Jifield,
                                              rhofield, Bfield);
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
        Array4<Real const> longitudinal0, longitudinal1, longitudinal2;
        if (transverse_offset) {
            longitudinal0=(*transverse_offset)[0]->const_array(mfi);
            longitudinal1=(*transverse_offset)[1]->const_array(mfi);
            longitudinal2=(*transverse_offset)[2]->const_array(mfi);
        }
        Array4<Real const> eiY0, eiY1, eiY2;
        if (direct_yee_inertia) {
            eiY0 = Ei_yee[0]->const_array(mfi);
            eiY1 = Ei_yee[1]->const_array(mfi);
            eiY2 = Ei_yee[2]->const_array(mfi);
        }
        Array4<Real const> const& rho = rhofield.const_array(mfi);
        Array4<Real const> rho_ped;
        if (rho_ped_mf)
        {
            rho_ped = rho_ped_mf->const_array(mfi);
        }
        Array4<Real const> te_K;
        if (te_mf)
        {
            te_K = te_mf->const_array(mfi);
        }
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
        Array4<Real const> eHx, eHy, eHz;
        if (hyperres_curlcurl) {
            eHx = eH_mf[0]->const_array(mfi);
            eHy = eH_mf[1]->const_array(mfi);
            eHz = eH_mf[2]->const_array(mfi);
        }
        // E_H mirror arrays (default-constructed and never indexed unless
        // EH_out was passed -- the kernels gate the write on the Array4).
        Array4<Real> end_x, end_y, end_z, gate_x, gate_y, gate_z;
        if (write_damping) {
            end_x = damping_out->end_resistivity[0]->array(mfi);
            gate_x = damping_out->end_holmstrom[0]->array(mfi);
            end_y = damping_out->end_resistivity[1]->array(mfi);
            gate_y = damping_out->end_holmstrom[1]->array(mfi);
            end_z = damping_out->end_resistivity[2]->array(mfi);
            gate_z = damping_out->end_holmstrom[2]->array(mfi);
        }
        Array4<Real> ph_x, ph_y, ph_z;
        if (F_pressure_hall_out) {
            ph_x = (*F_pressure_hall_out)[0]->array(mfi);
            ph_y = (*F_pressure_hall_out)[1]->array(mfi);
            ph_z = (*F_pressure_hall_out)[2]->array(mfi);
        }
        Array4<Real> er_x, er_y, er_z;
        if (write_er) {
            er_x = (*ER_out)[0]->array(mfi);
            er_y = (*ER_out)[1]->array(mfi);
            er_z = (*ER_out)[2]->array(mfi);
        }
        Array4<Real> eh_x, eh_y, eh_z;
        if (write_eh) {
            eh_x = (*EH_out)[0]->array(mfi);
            eh_y = (*EH_out)[1]->array(mfi);
            eh_z = (*EH_out)[2]->array(mfi);
        }
        // Nodal viscous drag (3 comps) and the E_visc mirror arrays;
        // default-constructed and never indexed when the drag is off.
        Array4<Real const> F_a;
        if (include_visc_drag) {
            F_a = F_mf.const_array(mfi);
        }
        Array4<Real> ev_x, ev_y, ev_z;
        if (write_ev) {
            ev_x = (*EV_out)[0]->array(mfi);
            ev_y = (*EV_out)[1]->array(mfi);
            ev_z = (*EV_out)[2]->array(mfi);
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
        if (include_external_fields) {
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
            CompensatedOhm::Pair quotient_hall{0.,0.}, quotient_grad{0.,0.};
            CompensatedOhm::Pair quotient_density{1.,0.};
            Real quotient_gate=0._rt;

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
                const auto rho_val_limited = HybridSmoothFloor(
                    rho_val + (use_pedestal ? Interp(rho_ped, nodal, Ex_stag, coarsen, i, j, k, 0)
                                            : 0.0_rt),
                    rho_floor, floor_w);

                Real ohm_val = (enE_x - grad_Pe) / rho_val_limited;
                if (transverse_offset) {
                    int const axis=AMREX_SPACEDIM==1 ? -1 : 0;
                    quotient_hall=CompensatedOhm::Average(enE,i,j,k,0,axis);
                    quotient_grad=(solve_for_Faraday ? add_grad_pe_faraday
                        : include_electron_pressure_term)
                        ? CompensatedOhm::Gradient(Pe,i,j,k,axis,coefs_x[0])
                        : CompensatedOhm::Pair{0.,0.};
                    quotient_density={rho_val_limited,0.}; quotient_gate=1._rt;
                    if(floor_w==0. && rho_val_limited>rho_floor) {
                        auto exact_density=CompensatedOhm::Average(rho,i,j,k,0,axis);
                        if(use_pedestal) exact_density=CompensatedOhm::Add(exact_density,
                            CompensatedOhm::Average(rho_ped,i,j,k,0,axis));
                        quotient_density=CompensatedOhm::WithHigh(exact_density,rho_val_limited);
                    }
                }
                if (holmstrom_smooth) {
                    ohm_val *= 0.5_rt * (1._rt + std::tanh(
                        (rho_val - rho_floor) * holmstrom_inv_width));
                    if (transverse_offset) quotient_gate *= 0.5_rt * (1._rt + std::tanh(
                        (rho_val - rho_floor) * holmstrom_inv_width));
                }
                Ex(i, j, k) = ohm_val;
            }
            if (end_region.holmstrom)
            {
                Real const before_end = write_damping ? Ex(i, j, k) : 0._rt;
                Ex(i, j, k) *= 1. - end_region.Weight(i, j, k, Ex_stag);
                if (transverse_offset) quotient_gate *= 1. - end_region.Weight(i,j,k,Ex_stag);
                if (write_damping) { gate_x(i, j, k) = Ex(i, j, k) - before_end; }
            }
            if (ph_x) { ph_x(i, j, k) = Ex(i, j, k); }
            if (transverse_offset) {
                Ex(i,j,k)=CompensatedOhm::OffsetQuotient(quotient_hall,quotient_grad,
                    quotient_density,quotient_gate,longitudinal0(i,j,k));
            }
            if (include_electron_inertia) {
                Ex(i, j, k) += direct_yee_inertia ? eiY0(i, j, k)
                    : Interp(eiN, nodal, Ex_stag, coarsen, i, j, k, 0);
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

                if (write_er) {
                    Real const global_eta_field =
                        (eta_has_Te ? eta_te(rho_val, jtot_val,
                                             Interp(te_K, nodal, Ex_stag, coarsen, i, j, k, 0), t_new)
                                    : eta(rho_val, jtot_val, t_new)) *
                        Jx(i, j, k);
                    Ex(i, j, k) += global_eta_field;
                    er_x(i, j, k) = global_eta_field;
                } else {
                    Ex(i, j, k) +=
                        (eta_has_Te ? eta_te(rho_val, jtot_val,
                                             Interp(te_K, nodal, Ex_stag, coarsen, i, j, k, 0), t_new)
                                    : eta(rho_val, jtot_val, t_new)) *
                        Jx(i, j, k);
                }
                if (end_region.resistivity > 0. && include_resistivity)
                {
                    if (write_damping) {
                        Real const end_field = end_region.resistivity *
                            end_region.Weight(i, j, k, Ex_stag) * Jx(i, j, k);
                        Ex(i, j, k) += end_field;
                        end_x(i, j, k) = end_field;
                    } else {
                    Ex(i, j, k) += end_region.resistivity *
                        end_region.Weight(i, j, k, Ex_stag) * Jx(i, j, k);
                    }
                }
                if (has_eta_overlay) { Ex(i, j, k) += eta_overlay_x(i, j, k); }

                if (hyperres_curlcurl && include_resistivity) {
                    // Exact operator: E_H = +curl(eta_H curl J), the
                    // coefficient inside the outer curl; composed by the
                    // caller (eta_H face-averaged there).
                    Ex(i, j, k) += eHx(i, j, k);
                } else if (include_hyper_resistivity_term && include_resistivity) {
                    // Truncated identity (-lap J = curl curl J only at
                    // div J = 0); trajectory-preserving default.

                    // Interpolate B field to appropriate staggering to match E field
                    Real btot_val = 0._rt;
                    if (hyper_resistivity_has_B_dependence) {
                        const Real bx_val = Interp(Bx, Bx_stag, Ex_stag, coarsen, i, j, k, 0);
                        const Real by_val = Interp(By, By_stag, Ex_stag, coarsen, i, j, k, 0);
                        const Real bz_val = Interp(Bz, Bz_stag, Ex_stag, coarsen, i, j, k, 0);
                        btot_val = std::sqrt(bx_val*bx_val + by_val*by_val + bz_val*bz_val);
                    }

                    // Mirror the exact applied dissipative field for energy
                    // booking.
                    auto nabla2Jx = T_Algo::Dxx(Jx, coefs_x, n_coefs_x, i, j, k)
                        + T_Algo::Dyy(Jx, coefs_y, n_coefs_y, i, j, k)
                        + T_Algo::Dzz(Jx, coefs_z, n_coefs_z, i, j, k);

                    Real const eh = -eta_h(rho_val, btot_val) * nabla2Jx;
                    Ex(i, j, k) += eh;
                    if (eh_x) {
                        eh_x(i, j, k) = eh;
                    }
                }
            }
            if (include_visc_drag) {
                // E_visc,x = node->edge average of the nodal drag F_x (the
                // transpose of the edge->node average that built u_e).
                const Real ev =
                    Interp(F_a, nodal, Ex_stag, coarsen, i, j, k, 0);
                Ex(i, j, k) += ev;
                if (ev_x) {
                    ev_x(i, j, k) = ev;
                }
            }

            if (include_external_fields) {
                const amrex::Real w_ext = subtract_E_ext_everywhere
                    ? 1._rt
                    : HybridExtSubWeight(rho_val, rho_floor, floor_w);
                Ex(i, j, k) -= w_ext * Ex_ext(i, j, k);
            }
        });

        // Ey calculation
        amrex::ParallelFor(tey, [=] AMREX_GPU_DEVICE (int i, int j, int k) {

            // Skip field update in the embedded boundaries
            if (update_Ey_arr && update_Ey_arr(i, j, k) == 0) { return; }

            // Interpolate to get the appropriate charge density in space
            const Real rho_val = Interp(rho, nodal, Ey_stag, coarsen, i, j, k, 0);
            CompensatedOhm::Pair quotient_hall{0.,0.}, quotient_grad{0.,0.};
            CompensatedOhm::Pair quotient_density{1.,0.};
            Real quotient_gate=0._rt;

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
                const auto rho_val_limited = HybridSmoothFloor(
                    rho_val + (use_pedestal ? Interp(rho_ped, nodal, Ey_stag, coarsen, i, j, k, 0)
                                            : 0.0_rt),
                    rho_floor, floor_w);

                Real ohm_val = (enE_y - grad_Pe) / rho_val_limited;
                if (transverse_offset) {
                    int const axis=AMREX_SPACEDIM==3 ? 1 : -1;
                    quotient_hall=CompensatedOhm::Average(enE,i,j,k,1,axis);
                    quotient_grad=(solve_for_Faraday ? add_grad_pe_faraday
                        : include_electron_pressure_term)
                        ? CompensatedOhm::Gradient(Pe,i,j,k,axis,coefs_y[0])
                        : CompensatedOhm::Pair{0.,0.};
                    quotient_density={rho_val_limited,0.}; quotient_gate=1._rt;
                    if(floor_w==0. && rho_val_limited>rho_floor) {
                        auto exact_density=CompensatedOhm::Average(rho,i,j,k,0,axis);
                        if(use_pedestal) exact_density=CompensatedOhm::Add(exact_density,
                            CompensatedOhm::Average(rho_ped,i,j,k,0,axis));
                        quotient_density=CompensatedOhm::WithHigh(exact_density,rho_val_limited);
                    }
                }
                if (holmstrom_smooth) {
                    ohm_val *= 0.5_rt * (1._rt + std::tanh(
                        (rho_val - rho_floor) * holmstrom_inv_width));
                    if (transverse_offset) quotient_gate *= 0.5_rt * (1._rt + std::tanh(
                        (rho_val - rho_floor) * holmstrom_inv_width));
                }
                Ey(i, j, k) = ohm_val;
            }
            if (end_region.holmstrom)
            {
                Real const before_end = write_damping ? Ey(i, j, k) : 0._rt;
                Ey(i, j, k) *= 1. - end_region.Weight(i, j, k, Ey_stag);
                if (transverse_offset) quotient_gate *= 1. - end_region.Weight(i,j,k,Ey_stag);
                if (write_damping) { gate_y(i, j, k) = Ey(i, j, k) - before_end; }
            }
            if (ph_y) { ph_y(i, j, k) = Ey(i, j, k); }
            if (transverse_offset) {
                Ey(i,j,k)=CompensatedOhm::OffsetQuotient(quotient_hall,quotient_grad,
                    quotient_density,quotient_gate,longitudinal1(i,j,k));
            }
            if (include_electron_inertia) {
                Ey(i, j, k) += direct_yee_inertia ? eiY1(i, j, k)
                    : Interp(eiN, nodal, Ey_stag, coarsen, i, j, k, 1);
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

                if (write_er) {
                    Real const global_eta_field =
                        (eta_has_Te ? eta_te(rho_val, jtot_val,
                                             Interp(te_K, nodal, Ey_stag, coarsen, i, j, k, 0), t_new)
                                    : eta(rho_val, jtot_val, t_new)) *
                        Jy(i, j, k);
                    Ey(i, j, k) += global_eta_field;
                    er_y(i, j, k) = global_eta_field;
                } else {
                    Ey(i, j, k) +=
                        (eta_has_Te ? eta_te(rho_val, jtot_val,
                                             Interp(te_K, nodal, Ey_stag, coarsen, i, j, k, 0), t_new)
                                    : eta(rho_val, jtot_val, t_new)) *
                        Jy(i, j, k);
                }
                if (end_region.resistivity > 0. && include_resistivity)
                {
                    if (write_damping) {
                        Real const end_field = end_region.resistivity *
                            end_region.Weight(i, j, k, Ey_stag) * Jy(i, j, k);
                        Ey(i, j, k) += end_field;
                        end_y(i, j, k) = end_field;
                    } else {
                    Ey(i, j, k) += end_region.resistivity *
                        end_region.Weight(i, j, k, Ey_stag) * Jy(i, j, k);
                    }
                }
                if (has_eta_overlay) { Ey(i, j, k) += eta_overlay_y(i, j, k); }

                if (hyperres_curlcurl && include_resistivity) {
                    // Exact operator: E_H = +curl(eta_H curl J), the
                    // coefficient inside the outer curl; composed by the
                    // caller (eta_H face-averaged there).
                    Ey(i, j, k) += eHy(i, j, k);
                } else if (include_hyper_resistivity_term && include_resistivity) {
                    // Truncated identity; trajectory-preserving default.

                    // Interpolate B field to appropriate staggering to match E field
                    Real btot_val = 0._rt;
                    if (hyper_resistivity_has_B_dependence) {
                        const Real bx_val = Interp(Bx, Bx_stag, Ey_stag, coarsen, i, j, k, 0);
                        const Real by_val = Interp(By, By_stag, Ey_stag, coarsen, i, j, k, 0);
                        const Real bz_val = Interp(Bz, Bz_stag, Ey_stag, coarsen, i, j, k, 0);
                        btot_val = std::sqrt(bx_val*bx_val + by_val*by_val + bz_val*bz_val);
                    }

                    // Mirror the exact applied dissipative field for energy
                    // booking.
                    auto nabla2Jy = T_Algo::Dxx(Jy, coefs_x, n_coefs_x, i, j, k)
                        + T_Algo::Dyy(Jy, coefs_y, n_coefs_y, i, j, k)
                        + T_Algo::Dzz(Jy, coefs_z, n_coefs_z, i, j, k);

                    Real const eh = -eta_h(rho_val, btot_val) * nabla2Jy;
                    Ey(i, j, k) += eh;
                    if (eh_y) {
                        eh_y(i, j, k) = eh;
                    }
                }
            }
            if (include_visc_drag) {
                // E_visc,y = node->edge average of F_y (see Ex).
                const Real ev =
                    Interp(F_a, nodal, Ey_stag, coarsen, i, j, k, 1);
                Ey(i, j, k) += ev;
                if (ev_y) {
                    ev_y(i, j, k) = ev;
                }
            }

            if (include_external_fields) {
                const amrex::Real w_ext = subtract_E_ext_everywhere
                    ? 1._rt
                    : HybridExtSubWeight(rho_val, rho_floor, floor_w);
                Ey(i, j, k) -= w_ext * Ey_ext(i, j, k);
            }
        });

        // Ez calculation
        amrex::ParallelFor(tez, [=] AMREX_GPU_DEVICE (int i, int j, int k){

            // Skip field update in the embedded boundaries
            if (update_Ez_arr && update_Ez_arr(i, j, k) == 0) { return; }

            // Interpolate to get the appropriate charge density in space
            const Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, k, 0);
            CompensatedOhm::Pair quotient_hall{0.,0.}, quotient_grad{0.,0.};
            CompensatedOhm::Pair quotient_density{1.,0.};
            Real quotient_gate=0._rt;

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
                const auto rho_val_limited = HybridSmoothFloor(
                    rho_val + (use_pedestal ? Interp(rho_ped, nodal, Ez_stag, coarsen, i, j, k, 0)
                                            : 0.0_rt),
                    rho_floor, floor_w);

                Real ohm_val = (enE_z - grad_Pe) / rho_val_limited;
                if (transverse_offset) {
                    int const axis=AMREX_SPACEDIM-1;
                    quotient_hall=CompensatedOhm::Average(enE,i,j,k,2,axis);
                    quotient_grad=(solve_for_Faraday ? add_grad_pe_faraday
                        : include_electron_pressure_term)
                        ? CompensatedOhm::Gradient(Pe,i,j,k,axis,coefs_z[0])
                        : CompensatedOhm::Pair{0.,0.};
                    quotient_density={rho_val_limited,0.}; quotient_gate=1._rt;
                    if(floor_w==0. && rho_val_limited>rho_floor) {
                        auto exact_density=CompensatedOhm::Average(rho,i,j,k,0,axis);
                        if(use_pedestal) exact_density=CompensatedOhm::Add(exact_density,
                            CompensatedOhm::Average(rho_ped,i,j,k,0,axis));
                        quotient_density=CompensatedOhm::WithHigh(exact_density,rho_val_limited);
                    }
                }
                if (holmstrom_smooth) {
                    ohm_val *= 0.5_rt * (1._rt + std::tanh(
                        (rho_val - rho_floor) * holmstrom_inv_width));
                    if (transverse_offset) quotient_gate *= 0.5_rt * (1._rt + std::tanh(
                        (rho_val - rho_floor) * holmstrom_inv_width));
                }
                Ez(i, j, k) = ohm_val;
            }
            if (end_region.holmstrom)
            {
                Real const before_end = write_damping ? Ez(i, j, k) : 0._rt;
                Ez(i, j, k) *= 1. - end_region.Weight(i, j, k, Ez_stag);
                if (transverse_offset) quotient_gate *= 1. - end_region.Weight(i,j,k,Ez_stag);
                if (write_damping) { gate_z(i, j, k) = Ez(i, j, k) - before_end; }
            }
            if (ph_z) { ph_z(i, j, k) = Ez(i, j, k); }
            if (transverse_offset) {
                Ez(i,j,k)=CompensatedOhm::OffsetQuotient(quotient_hall,quotient_grad,
                    quotient_density,quotient_gate,longitudinal2(i,j,k));
            }
            if (include_electron_inertia) {
                Ez(i, j, k) += direct_yee_inertia ? eiY2(i, j, k)
                    : Interp(eiN, nodal, Ez_stag, coarsen, i, j, k, 2);
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

                if (write_er) {
                    Real const global_eta_field =
                        (eta_has_Te ? eta_te(rho_val, jtot_val,
                                             Interp(te_K, nodal, Ez_stag, coarsen, i, j, k, 0), t_new)
                                    : eta(rho_val, jtot_val, t_new)) *
                        Jz(i, j, k);
                    Ez(i, j, k) += global_eta_field;
                    er_z(i, j, k) = global_eta_field;
                } else {
                    Ez(i, j, k) +=
                        (eta_has_Te ? eta_te(rho_val, jtot_val,
                                             Interp(te_K, nodal, Ez_stag, coarsen, i, j, k, 0), t_new)
                                    : eta(rho_val, jtot_val, t_new)) *
                        Jz(i, j, k);
                }
                if (end_region.resistivity > 0. && include_resistivity)
                {
                    if (write_damping) {
                        Real const end_field = end_region.resistivity *
                            end_region.Weight(i, j, k, Ez_stag) * Jz(i, j, k);
                        Ez(i, j, k) += end_field;
                        end_z(i, j, k) = end_field;
                    } else {
                    Ez(i, j, k) += end_region.resistivity *
                        end_region.Weight(i, j, k, Ez_stag) * Jz(i, j, k);
                    }
                }
                if (has_eta_overlay) { Ez(i, j, k) += eta_overlay_z(i, j, k); }

                if (hyperres_curlcurl && include_resistivity) {
                    // Exact operator: E_H = +curl(eta_H curl J), the
                    // coefficient inside the outer curl; composed by the
                    // caller (eta_H face-averaged there).
                    Ez(i, j, k) += eHz(i, j, k);
                } else if (include_hyper_resistivity_term && include_resistivity) {
                    // Truncated identity; trajectory-preserving default.

                    // Interpolate B field to appropriate staggering to match E field
                    Real btot_val = 0._rt;
                    if (hyper_resistivity_has_B_dependence) {
                        const Real bx_val = Interp(Bx, Bx_stag, Ez_stag, coarsen, i, j, k, 0);
                        const Real by_val = Interp(By, By_stag, Ez_stag, coarsen, i, j, k, 0);
                        const Real bz_val = Interp(Bz, Bz_stag, Ez_stag, coarsen, i, j, k, 0);
                        btot_val = std::sqrt(bx_val*bx_val + by_val*by_val + bz_val*bz_val);
                    }

                    // Mirror the exact applied dissipative field for energy
                    // booking.
                    auto nabla2Jz = T_Algo::Dxx(Jz, coefs_x, n_coefs_x, i, j, k)
                        + T_Algo::Dyy(Jz, coefs_y, n_coefs_y, i, j, k)
                        + T_Algo::Dzz(Jz, coefs_z, n_coefs_z, i, j, k);

                    Real const eh = -eta_h(rho_val, btot_val) * nabla2Jz;
                    Ez(i, j, k) += eh;
                    if (eh_z) {
                        eh_z(i, j, k) = eh;
                    }
                }
            }
            if (include_visc_drag) {
                // E_visc,z = node->edge average of F_z (see Ex).
                const Real ev =
                    Interp(F_a, nodal, Ez_stag, coarsen, i, j, k, 2);
                Ez(i, j, k) += ev;
                if (ev_z) {
                    ev_z(i, j, k) = ev;
                }
            }

            if (include_external_fields) {
                const amrex::Real w_ext = subtract_E_ext_everywhere
                    ? 1._rt
                    : HybridExtSubWeight(rho_val, rho_floor, floor_w);
                Ez(i, j, k) -= w_ext * Ez_ext(i, j, k);
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
