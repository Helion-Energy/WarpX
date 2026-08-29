/* Copyright 2024 The WarpX Community
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

#include "Utils/TextMsg.H"
#include "WarpX.H"

using namespace amrex;

void FiniteDifferenceSolver::ComputeCurlA (
    ablastr::fields::VectorField& Bfield,
    ablastr::fields::VectorField const& Afield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update_B,
    int lev, const amrex::IntVect& ngrow )
{
    // Domain-ghost growth is only supported away from embedded boundaries:
    // the eb_update flags are not guaranteed to carry matching ghost data.
    const amrex::IntVect ng = EB::enabled() ? amrex::IntVect(0) : ngrow;

    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::Yee ||
        m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
        ComputeCurlACylindrical <CylindricalYeeAlgorithm> (
            Bfield, Afield, eb_update_B, lev, ng
        );

#elif defined(WARPX_DIM_RSPHERE)
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ngrow == amrex::IntVect::TheZeroVector(),
            "ComputeCurlA: ghost growth not implemented in spherical geometry");
        ComputeCurlASpherical <SphericalYeeAlgorithm> (
            Bfield, Afield, eb_update_B, lev, ng
        );

#else
    if (WarpX::grid_type == GridType::Staggered)
    {
        ComputeCurlACartesian <CartesianYeeAlgorithm> (
            Bfield, Afield, eb_update_B, lev, ng
        );
    } else {
        // The nodal algorithm uses centered stencils (reads one cell in
        // both directions), so the outermost requested ghost ring would
        // read past the ghost data of A: no domain-ghost growth here.
        ComputeCurlACartesian <CartesianNodalAlgorithm> (
            Bfield, Afield, eb_update_B, lev, amrex::IntVect(0)
        );
    }

#endif
    } else {
        amrex::Abort(Utils::TextMsg::Err(
            "ComputeCurl: Unknown algorithm choice."));
    }
}

// /**
//   * \brief Calculate B from the curl of A
//   * i.e. B = curl(A) output field on B field mesh staggering
//   *
//   * \param[out] Bfield  output of curl operation
//   * \param[in] Afield   input staggered field, should be on E/J/A mesh staggering
//   * \param[in] eb_update_B specifies where the plasma current should be calculated.
//   * \param[in] lev refinement level

//   */
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
template<typename T_Algo>
void FiniteDifferenceSolver::ComputeCurlACylindrical (
    ablastr::fields::VectorField& Bfield,
    ablastr::fields::VectorField const& Afield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update_B,
    int lev,
    const amrex::IntVect& ngrow
)
{
    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // reset Bfield
    Bfield[0]->setVal(0);
    Bfield[1]->setVal(0);
    Bfield[2]->setVal(0);

    // Evaluation region: the valid cells plus the requested growth into the
    // domain ghosts. A carries trusted ghost values (the analytic parser
    // fill / file interpolation covers the full grown box) and the Yee curl
    // reads at most one cell upward per direction while nodal-in-r/z A
    // extends one index past cell-centered B, so every grown ring is exact.
    // Non-periodic domain ghosts (e.g. the RZ r_max ring that wall-adjacent
    // stencils read) are thereby computed instead of staying at the
    // allocation value; grid-interior ghosts inside the growth are
    // recomputed to the very values FillBoundary would copy anyway. Growth
    // below the axis is suppressed: the mirrored A ghosts there carry no
    // azimuthal parity information and r < 0 breaks the radial metric.
    amrex::Box grow_region = WarpX::GetInstance().Geom(lev).Domain();
    grow_region.grow(ngrow);
    if (m_rmin == 0._rt) {
        grow_region.setSmall(0, WarpX::GetInstance().Geom(lev).Domain().smallEnd(0));
    }
    const amrex::Box allowed_r = amrex::convert(grow_region, Bfield[0]->ixType());
    const amrex::Box allowed_t = amrex::convert(grow_region, Bfield[1]->ixType());
    const amrex::Box allowed_z = amrex::convert(grow_region, Bfield[2]->ixType());

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Afield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<const Real> const& Ar = Afield[0]->const_array(mfi);
        Array4<const Real> const& At = Afield[1]->const_array(mfi);
        Array4<const Real> const& Az = Afield[2]->const_array(mfi);
        Array4<Real> const& Br = Bfield[0]->array(mfi);
        Array4<Real> const& Btheta = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries.
        amrex::Array4<int> update_Br_arr, update_Btheta_arr, update_Bz_arr;
        if (EB::enabled()) {
            update_Br_arr = eb_update_B[0]->array(mfi);
            update_Btheta_arr = eb_update_B[1]->array(mfi);
            update_Bz_arr = eb_update_B[2]->array(mfi);
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

        // Extract tileboxes for which to loop over (grown into the domain
        // ghosts where requested; tiles only grow at valid-box edges, so
        // no cell is visited twice within a FAB)
        Box const tbr = mfi.tilebox(Bfield[0]->ixType().toIntVect(), ngrow) & allowed_r;
        Box const tbt = mfi.tilebox(Bfield[1]->ixType().toIntVect(), ngrow) & allowed_t;
        Box const tbz = mfi.tilebox(Bfield[2]->ixType().toIntVect(), ngrow) & allowed_z;

        // Calculate the B-field from the A-field
        amrex::ParallelFor(tbr, tbt, tbz,

            // Br calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                // Skip field update in the embedded boundaries
                if (update_Br_arr && update_Br_arr(i, j, 0) == 0) { return; }

                Real const r = rmin + i*dr; // r on nodal point (Br is nodal in r)
                if (r != 0) { // Off-axis, regular Maxwell equations
                    Br(i, j, 0, 0) = - T_Algo::UpwardDz(At, coefs_z, n_coefs_z, i, j, 0, 0); // Mode m=0
                    for (int m=1; m<nmodes; m++) { // Higher-order modes
                        Br(i, j, 0, 2*m-1) = - (
                            T_Algo::UpwardDz(At, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                            - m * Az(i, j, 0, 2*m  )/r );  // Real part
                        Br(i, j, 0, 2*m  ) = - (
                            T_Algo::UpwardDz(At, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                            + m * Az(i, j, 0, 2*m-1)/r ); // Imaginary part
                    }
                } else { // r==0: On-axis corrections
                    // Ensure that Br remains 0 on axis (except for m=1)
                    Br(i, j, 0, 0) = 0.; // Mode m=0
                    for (int m=1; m<nmodes; m++) { // Higher-order modes
                        if (m == 1){
                            // For m==1, Bz is linear in r, for small r
                            // Therefore, the formula below regularizes the singularity
                            Br(i, j, 0, 2*m-1) = - (
                                T_Algo::UpwardDz(At, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                                - m * Az(i+1, j, 0, 2*m  )/dr );  // Real part
                            Br(i, j, 0, 2*m  ) = - (
                                T_Algo::UpwardDz(At, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                                + m * Az(i+1, j, 0, 2*m-1)/dr ); // Imaginary part
                        } else {
                            Br(i, j, 0, 2*m-1) = 0.;
                            Br(i, j, 0, 2*m  ) = 0.;
                        }
                    }
                }
            },

            // Btheta calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                // Skip field update in the embedded boundaries
                if (update_Btheta_arr && update_Btheta_arr(i, j, 0) == 0) { return; }

                Btheta(i, j, 0, 0) = - (
                    T_Algo::UpwardDr(Az, coefs_r, n_coefs_r, i, j, 0, 0)
                    - T_Algo::UpwardDz(Ar, coefs_z, n_coefs_z, i, j, 0, 0)); // Mode m=0
                for (int m=1 ; m<nmodes ; m++) { // Higher-order modes
                    Btheta(i, j, 0, 2*m-1) = - (
                        T_Algo::UpwardDr(Az, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                        - T_Algo::UpwardDz(Ar, coefs_z, n_coefs_z, i, j, 0, 2*m-1)); // Real part
                    Btheta(i, j, 0, 2*m  ) = - (
                        T_Algo::UpwardDr(Az, coefs_r, n_coefs_r, i, j, 0, 2*m  )
                        - T_Algo::UpwardDz(Ar, coefs_z, n_coefs_z, i, j, 0, 2*m  )); // Imaginary part
                }
            },

            // Bz calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                // Skip field update in the embedded boundaries
                if (update_Bz_arr && update_Bz_arr(i, j, 0) == 0) { return; }

                Real const r = rmin + (i + 0.5_rt)*dr; // r on a cell-centered grid (Bz is cell-centered in r)
                Bz(i, j, 0, 0) =  T_Algo::UpwardDrr_over_r(At, r, dr, coefs_r, n_coefs_r, i, j, 0, 0);
                for (int m=1 ; m<nmodes ; m++) { // Higher-order modes
                    Bz(i, j, 0, 2*m-1) = - ( m * Ar(i, j, 0, 2*m  )/r
                        - T_Algo::UpwardDrr_over_r(At, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m-1)); // Real part
                    Bz(i, j, 0, 2*m  ) = - (-m * Ar(i, j, 0, 2*m-1)/r
                        - T_Algo::UpwardDrr_over_r(At, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m  )); // Imaginary part
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
void FiniteDifferenceSolver::ComputeCurlASpherical (
    ablastr::fields::VectorField& Bfield,
    ablastr::fields::VectorField const& Afield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update_B,
    int lev,
    const amrex::IntVect& ngrow
)
{
    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // reset Bfield
    Bfield[0]->setVal(0);
    Bfield[1]->setVal(0);
    Bfield[2]->setVal(0);

    // Evaluation region grown into the domain ghosts (see the cylindrical
    // variant for the rationale); no growth below the axis.
    amrex::Box grow_region = WarpX::GetInstance().Geom(lev).Domain();
    grow_region.grow(ngrow);
    if (m_rmin == 0._rt) {
        grow_region.setSmall(0, WarpX::GetInstance().Geom(lev).Domain().smallEnd(0));
    }
    const amrex::Box allowed_r = amrex::convert(grow_region, Bfield[0]->ixType());
    const amrex::Box allowed_t = amrex::convert(grow_region, Bfield[1]->ixType());
    const amrex::Box allowed_p = amrex::convert(grow_region, Bfield[2]->ixType());

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Afield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<const Real> const& At = Afield[1]->const_array(mfi);
        Array4<const Real> const& Ap = Afield[2]->const_array(mfi);
        Array4<Real> const& Br = Bfield[0]->array(mfi);
        Array4<Real> const& Btheta = Bfield[1]->array(mfi);
        Array4<Real> const& Bphi = Bfield[2]->array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries.
        amrex::Array4<int> update_Br_arr, update_Btheta_arr, update_Bphi_arr;
        if (EB::enabled()) {
            update_Br_arr = eb_update_B[0]->array(mfi);
            update_Btheta_arr = eb_update_B[1]->array(mfi);
            update_Bphi_arr = eb_update_B[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_r = m_stencil_coefs_r.dataPtr();
        int const n_coefs_r = static_cast<int>(m_stencil_coefs_r.size());

        // Extract spherical specific parameters
        Real const dr = m_dr;
        Real const rmin = m_rmin;

        // Extract tileboxes for which to loop over (grown into the domain
        // ghosts where requested)
        Box const tbr = mfi.tilebox(Bfield[0]->ixType().toIntVect(), ngrow) & allowed_r;
        Box const tbt = mfi.tilebox(Bfield[1]->ixType().toIntVect(), ngrow) & allowed_t;
        Box const tbp = mfi.tilebox(Bfield[2]->ixType().toIntVect(), ngrow) & allowed_p;

        // Calculate the B-field from the A-field
        amrex::ParallelFor(tbr, tbt, tbp,

            // Br calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                // Skip field update in the embedded boundaries
                if (update_Br_arr && update_Br_arr(i, j, 0) == 0) { return; }
                Br(i, j, 0, 0) = 0._rt;
            },

            // Btheta calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                // Skip field update in the embedded boundaries
                if (update_Btheta_arr && update_Btheta_arr(i, j, 0) == 0) { return; }

                Real const r = rmin + (i + 0.5_rt)*dr; // r on a cell-centered grid (Btheta is cell-centered in r)
                Btheta(i, j, 0, 0) =  -T_Algo::UpwardDrr_over_r(Ap, r, dr, coefs_r, n_coefs_r, i, j, 0, 0);
            },

            // Bphi calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                // Skip field update in the embedded boundaries
                if (update_Bphi_arr && update_Bphi_arr(i, j, 0) == 0) { return; }

                Real const r = rmin + (i + 0.5_rt)*dr; // r on a cell-centered grid (Bphi is cell-centered in r)
                Bphi(i, j, 0, 0) =  T_Algo::UpwardDrr_over_r(At, r, dr, coefs_r, n_coefs_r, i, j, 0, 0);
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
void FiniteDifferenceSolver::ComputeCurlACartesian (
    ablastr::fields::VectorField & Bfield,
    ablastr::fields::VectorField const& Afield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update_B,
    int lev,
    const amrex::IntVect& ngrow
)
{
    using ablastr::fields::Direction;

    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // reset Bfield
    Bfield[0]->setVal(0);
    Bfield[1]->setVal(0);
    Bfield[2]->setVal(0);

    // Evaluation region grown into the domain ghosts (see the cylindrical
    // variant for the rationale).
    amrex::Box grow_region = WarpX::GetInstance().Geom(lev).Domain();
    grow_region.grow(ngrow);
    const amrex::Box allowed_x = amrex::convert(grow_region, Bfield[0]->ixType());
    const amrex::Box allowed_y = amrex::convert(grow_region, Bfield[1]->ixType());
    const amrex::Box allowed_z = amrex::convert(grow_region, Bfield[2]->ixType());

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Afield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers) {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const &Bx = Bfield[0]->array(mfi);
        Array4<Real> const &By = Bfield[1]->array(mfi);
        Array4<Real> const &Bz = Bfield[2]->array(mfi);
        Array4<Real const> const &Ax = Afield[0]->const_array(mfi);
        Array4<Real const> const &Ay = Afield[1]->const_array(mfi);
        Array4<Real const> const &Az = Afield[2]->const_array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries.
        amrex::Array4<int> update_Bx_arr, update_By_arr, update_Bz_arr;
        if (EB::enabled()) {
            update_Bx_arr = eb_update_B[0]->array(mfi);
            update_By_arr = eb_update_B[1]->array(mfi);
            update_Bz_arr = eb_update_B[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract tileboxes for which to loop (grown into the domain
        // ghosts where requested)
        Box const tbx = mfi.tilebox(Bfield[0]->ixType().toIntVect(), ngrow) & allowed_x;
        Box const tby = mfi.tilebox(Bfield[1]->ixType().toIntVect(), ngrow) & allowed_y;
        Box const tbz = mfi.tilebox(Bfield[2]->ixType().toIntVect(), ngrow) & allowed_z;

        // Calculate the curl of A
        amrex::ParallelFor(tbx, tby, tbz,

            // Bx calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field update in the embedded boundaries
                if (update_Bx_arr && update_Bx_arr(i, j, k) == 0) { return; }

                Bx(i, j, k) =  (
                    - T_Algo::UpwardDz(Ay, coefs_z, n_coefs_z, i, j, k)
                    + T_Algo::UpwardDy(Az, coefs_y, n_coefs_y, i, j, k)
                );
            },

            // By calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field update in the embedded boundaries
                if (update_By_arr && update_By_arr(i, j, k) == 0) { return; }

                By(i, j, k) = (
                    - T_Algo::UpwardDx(Az, coefs_x, n_coefs_x, i, j, k)
                    + T_Algo::UpwardDz(Ax, coefs_z, n_coefs_z, i, j, k)
                );
            },

            // Bz calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field update in the embedded boundaries
                if (update_Bz_arr && update_Bz_arr(i, j, k) == 0) { return; }

                Bz(i, j, k) = (
                    - T_Algo::UpwardDy(Ax, coefs_y, n_coefs_y, i, j, k)
                    + T_Algo::UpwardDx(Ay, coefs_x, n_coefs_x, i, j, k)
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
