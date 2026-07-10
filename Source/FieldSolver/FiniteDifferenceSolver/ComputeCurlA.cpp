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

#include <limits>

using namespace amrex;

void FiniteDifferenceSolver::ComputeCurlA (
    ablastr::fields::VectorField& Bfield,
    ablastr::fields::VectorField const& Afield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update_B,
    int lev, [[maybe_unused]] bool iso_curl )
{
    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::Yee ||
        m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
        ComputeCurlACylindrical <CylindricalYeeAlgorithm> (
            Bfield, Afield, eb_update_B, lev
        );

#elif defined(WARPX_DIM_RSPHERE)
        ComputeCurlASpherical <SphericalYeeAlgorithm> (
            Bfield, Afield, eb_update_B, lev
        );

#else
    if (WarpX::grid_type == GridType::Staggered)
    {
        ComputeCurlACartesian <CartesianYeeAlgorithm> (
            Bfield, Afield, eb_update_B, lev, iso_curl
        );
    } else {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!iso_curl,
            "The isotropized curl of A is only implemented for staggered grids");
        ComputeCurlACartesian <CartesianNodalAlgorithm> (
            Bfield, Afield, eb_update_B, lev, false
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
    int lev
)
{
    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // reset Bfield
    Bfield[0]->setVal(0);
    Bfield[1]->setVal(0);
    Bfield[2]->setVal(0);

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

        // Extract tileboxes for which to loop over
        Box const& tbr  = mfi.tilebox(Bfield[0]->ixType().toIntVect());
        Box const& tbt  = mfi.tilebox(Bfield[1]->ixType().toIntVect());
        Box const& tbz  = mfi.tilebox(Bfield[2]->ixType().toIntVect());

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
    int lev
)
{
    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // reset Bfield
    Bfield[0]->setVal(0);
    Bfield[1]->setVal(0);
    Bfield[2]->setVal(0);

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

        // Extract tileboxes for which to loop over
        Box const& tbr = mfi.tilebox(Bfield[0]->ixType().toIntVect());
        Box const& tbt = mfi.tilebox(Bfield[1]->ixType().toIntVect());
        Box const& tbp = mfi.tilebox(Bfield[2]->ixType().toIntVect());

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
    int lev, [[maybe_unused]] bool iso_curl
)
{
    using ablastr::fields::Direction;

    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // reset Bfield
    Bfield[0]->setVal(0);
    Bfield[1]->setVal(0);
    Bfield[2]->setVal(0);

#if defined(WARPX_DIM_3D)
    // Isotropized curl: the Yee half-cell derivative D_a has leading error
    // (dx_a^2/24) d^3/dx_a^3, so for an axisymmetric A the truncation error
    // of B = curl A contains a fourfold (m=4) azimuthal harmonic (it acts on
    // the m=2 strain field of A_theta through the anisotropic part of the
    // per-axis third derivatives). Preconditioning each A component with the
    // compact second difference along its OWN axis, S_a = 1 + delta_a^2/24,
    // turns the leading error of every B component into the isotropic
    // transverse-Laplacian smoothing (1 + (h^2/24) Lap) B, cancelling the
    // m=4 imprint at leading order. Since div(curl(.)) vanishes identically
    // on the staggered mesh for ANY edge field, the corrected B remains
    // exactly divergence-free. The correction is added below as a separate
    // term, D(delta^2 A)/24, so the default path is unchanged.
    //
    // Along non-periodic axes the correction is skipped within one cell of
    // the domain boundary (where its wider footprint would read boundary
    // guard data), and near an embedded boundary (nodal signed-distance
    // band), matching the near-edge downgrades of the interpolation options.
    amrex::GpuArray<amrex::Real, 3> inv_dx{};
    amrex::GpuArray<int, 3> isox_lo, isox_hi, isoy_lo, isoy_hi, isoz_lo, isoz_hi;
    amrex::Real d_iso_band = 0.0_rt;
    amrex::MultiFab const* iso_phi_mf = nullptr;
    if (iso_curl) {
        auto& warpx = WarpX::GetInstance();
        const auto dx_lev = warpx.Geom(lev).CellSizeArray();
        amrex::Real h_max = 0.0_rt;
        for (int d = 0; d < 3; ++d) {
            inv_dx[d] = 1.0_rt/dx_lev[d];
            h_max = std::max(h_max, dx_lev[d]);
        }
        for (int d = 0; d < 3; ++d) {
            isox_lo[d] = std::numeric_limits<int>::lowest()/2;
            isoy_lo[d] = std::numeric_limits<int>::lowest()/2;
            isoz_lo[d] = std::numeric_limits<int>::lowest()/2;
            isox_hi[d] = std::numeric_limits<int>::max()/2;
            isoy_hi[d] = std::numeric_limits<int>::max()/2;
            isoz_hi[d] = std::numeric_limits<int>::max()/2;
        }
        const amrex::Box& domain = warpx.Geom(lev).Domain();
        for (int d = 0; d < 3; ++d) {
            if (!warpx.Geom(lev).isPeriodic(d)) {
                // restrict the transverse axes of each B component (the
                // component's own axis carries no correction reads)
                if (d != 0) { isox_lo[d] = domain.smallEnd(d) + 1;
                              isox_hi[d] = domain.bigEnd(d) - 1; }
                if (d != 1) { isoy_lo[d] = domain.smallEnd(d) + 1;
                              isoy_hi[d] = domain.bigEnd(d) - 1; }
                if (d != 2) { isoz_lo[d] = domain.smallEnd(d) + 1;
                              isoz_hi[d] = domain.bigEnd(d) - 1; }
            }
        }
        d_iso_band = 2.0_rt * h_max;
        if (EB::enabled()) {
            iso_phi_mf = warpx.m_fields.get(warpx::fields::FieldType::distance_to_eb, lev);
        }
    }
#endif

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

#if defined(WARPX_DIM_3D)
        // Nodal signed distance for the near-EB downgrade of the
        // isotropized curl (empty Array4 when unused; every read is gated).
        amrex::Array4<amrex::Real const> iso_phi;
        if (iso_phi_mf) { iso_phi = iso_phi_mf->const_array(mfi); }
#endif

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract tileboxes for which to loop
        Box const& tbx  = mfi.tilebox(Bfield[0]->ixType().toIntVect());
        Box const& tby  = mfi.tilebox(Bfield[1]->ixType().toIntVect());
        Box const& tbz  = mfi.tilebox(Bfield[2]->ixType().toIntVect());

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

#if defined(WARPX_DIM_3D)
                // own-axis preconditioning terms of the isotropized curl
                if (iso_curl
                    && j >= isox_lo[1] && j <= isox_hi[1]
                    && k >= isox_lo[2] && k <= isox_hi[2]
                    && (!iso_phi || iso_phi(i, j, k) >= d_iso_band)) {
                    const amrex::Real d2zAz_j  = Az(i, j  , k+1) - 2.0_rt*Az(i, j  , k) + Az(i, j  , k-1);
                    const amrex::Real d2zAz_jp = Az(i, j+1, k+1) - 2.0_rt*Az(i, j+1, k) + Az(i, j+1, k-1);
                    const amrex::Real d2yAy_k  = Ay(i, j+1, k  ) - 2.0_rt*Ay(i, j, k  ) + Ay(i, j-1, k  );
                    const amrex::Real d2yAy_kp = Ay(i, j+1, k+1) - 2.0_rt*Ay(i, j, k+1) + Ay(i, j-1, k+1);
                    Bx(i, j, k) += (
                        (d2zAz_jp - d2zAz_j)*inv_dx[1]
                        - (d2yAy_kp - d2yAy_k)*inv_dx[2]
                    ) / 24.0_rt;
                }
#endif
            },

            // By calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field update in the embedded boundaries
                if (update_By_arr && update_By_arr(i, j, k) == 0) { return; }

                By(i, j, k) = (
                    - T_Algo::UpwardDx(Az, coefs_x, n_coefs_x, i, j, k)
                    + T_Algo::UpwardDz(Ax, coefs_z, n_coefs_z, i, j, k)
                );

#if defined(WARPX_DIM_3D)
                // own-axis preconditioning terms of the isotropized curl
                if (iso_curl
                    && i >= isoy_lo[0] && i <= isoy_hi[0]
                    && k >= isoy_lo[2] && k <= isoy_hi[2]
                    && (!iso_phi || iso_phi(i, j, k) >= d_iso_band)) {
                    const amrex::Real d2zAz_i  = Az(i  , j, k+1) - 2.0_rt*Az(i  , j, k) + Az(i  , j, k-1);
                    const amrex::Real d2zAz_ip = Az(i+1, j, k+1) - 2.0_rt*Az(i+1, j, k) + Az(i+1, j, k-1);
                    const amrex::Real d2xAx_k  = Ax(i+1, j, k  ) - 2.0_rt*Ax(i, j, k  ) + Ax(i-1, j, k  );
                    const amrex::Real d2xAx_kp = Ax(i+1, j, k+1) - 2.0_rt*Ax(i, j, k+1) + Ax(i-1, j, k+1);
                    By(i, j, k) += (
                        - (d2zAz_ip - d2zAz_i)*inv_dx[0]
                        + (d2xAx_kp - d2xAx_k)*inv_dx[2]
                    ) / 24.0_rt;
                }
#endif
            },

            // Bz calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field update in the embedded boundaries
                if (update_Bz_arr && update_Bz_arr(i, j, k) == 0) { return; }

                Bz(i, j, k) = (
                    - T_Algo::UpwardDy(Ax, coefs_y, n_coefs_y, i, j, k)
                    + T_Algo::UpwardDx(Ay, coefs_x, n_coefs_x, i, j, k)
                );

#if defined(WARPX_DIM_3D)
                // own-axis preconditioning terms of the isotropized curl
                if (iso_curl
                    && i >= isoz_lo[0] && i <= isoz_hi[0]
                    && j >= isoz_lo[1] && j <= isoz_hi[1]
                    && (!iso_phi || iso_phi(i, j, k) >= d_iso_band)) {
                    const amrex::Real d2xAx_j  = Ax(i+1, j  , k) - 2.0_rt*Ax(i, j  , k) + Ax(i-1, j  , k);
                    const amrex::Real d2xAx_jp = Ax(i+1, j+1, k) - 2.0_rt*Ax(i, j+1, k) + Ax(i-1, j+1, k);
                    const amrex::Real d2yAy_i  = Ay(i  , j+1, k) - 2.0_rt*Ay(i  , j, k) + Ay(i  , j-1, k);
                    const amrex::Real d2yAy_ip = Ay(i+1, j+1, k) - 2.0_rt*Ay(i+1, j, k) + Ay(i+1, j-1, k);
                    Bz(i, j, k) += (
                        - (d2xAx_jp - d2xAx_j)*inv_dx[1]
                        + (d2yAy_ip - d2yAy_i)*inv_dx[0]
                    ) / 24.0_rt;
                }
#endif
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
