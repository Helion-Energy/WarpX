/* Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *          S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ISOTROPIC_OPERATORS_H_
#define WARPX_ISOTROPIC_OPERATORS_H_

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Extension.H>
#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

/** Isotropic finite-difference operators for the Hybrid-PIC resistive E solve.
 *
 *  These device-inline helpers isotropize the hyper-resistivity Laplacian and
 *  the resistive diffusion of the out-of-plane B (corner-curl correction). They
 *  are header-only because they are evaluated inside GPU lambdas in
 *  HybridPICSolveE.cpp. They are guarded by the same dimensionality macros as
 *  the Cartesian solve (no definitions in RZ / RSPHERE / RCYLINDER builds).
 */
namespace warpx::hybrid_isotropic
{
#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)

    /** Isotropic discrete Laplacian for the hyper-resistivity term.
     *
     *  The cross stencil Dxx+Dyy+Dzz has a leading truncation error whose
     *  angular dependence is cos(4*theta): the damping rate of a mode at
     *  finite kh differs between the grid axes and the diagonals, which
     *  imprints fourfold "squaring" on diffusing structures. The Mehrstellen
     *  9-point (2D) and Patra-Karttunen 27-point (3D) stencils cancel that
     *  term: their leading error is proportional to the isotropic
     *  biharmonic. They are written here as the cross stencil plus compact
     *  cross-derivative corrections, which stays consistent (2nd order) on
     *  non-cubic cells; the error is fully isotropic for cubic cells.
     */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real LaplacianIsotropic (
        amrex::Array4<amrex::Real const> const& F,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2)
    {
        using namespace amrex::literals;
#if defined(WARPX_DIM_3D)
        amrex::Real const d2x = F(i+1,j,k) - 2._rt*F(i,j,k) + F(i-1,j,k);
        amrex::Real const d2y = F(i,j+1,k) - 2._rt*F(i,j,k) + F(i,j-1,k);
        amrex::Real const d2z = F(i,j,k+1) - 2._rt*F(i,j,k) + F(i,j,k-1);

        // undivided cross differences delta_a^2 delta_b^2 F
        auto const cxy_at = [&] (int kk) {
            return F(i+1,j+1,kk) + F(i+1,j-1,kk) + F(i-1,j+1,kk) + F(i-1,j-1,kk)
                 - 2._rt*(F(i+1,j,kk) + F(i-1,j,kk) + F(i,j+1,kk) + F(i,j-1,kk))
                 + 4._rt*F(i,j,kk);
        };
        amrex::Real const cxy = cxy_at(k);
        amrex::Real const cxz =
              F(i+1,j,k+1) + F(i+1,j,k-1) + F(i-1,j,k+1) + F(i-1,j,k-1)
            - 2._rt*(F(i+1,j,k) + F(i-1,j,k) + F(i,j,k+1) + F(i,j,k-1))
            + 4._rt*F(i,j,k);
        amrex::Real const cyz =
              F(i,j+1,k+1) + F(i,j+1,k-1) + F(i,j-1,k+1) + F(i,j-1,k-1)
            - 2._rt*(F(i,j+1,k) + F(i,j-1,k) + F(i,j,k+1) + F(i,j,k-1))
            + 4._rt*F(i,j,k);
        amrex::Real const txyz = cxy_at(k+1) - 2._rt*cxy_at(k) + cxy_at(k-1);

        return d2x*inv_h2[0] + d2y*inv_h2[1] + d2z*inv_h2[2]
             + ((h2[0] + h2[1])/12._rt)*cxy*inv_h2[0]*inv_h2[1]
             + ((h2[0] + h2[2])/12._rt)*cxz*inv_h2[0]*inv_h2[2]
             + ((h2[1] + h2[2])/12._rt)*cyz*inv_h2[1]*inv_h2[2]
             + ((h2[0]*h2[1] + h2[1]*h2[2] + h2[0]*h2[2])/90._rt)
               *txyz*inv_h2[0]*inv_h2[1]*inv_h2[2];
#else
        // 2D XZ: Mehrstellen 9-point (j is the z index)
        amrex::Real const d2x = F(i+1,j,k) - 2._rt*F(i,j,k) + F(i-1,j,k);
        amrex::Real const d2z = F(i,j+1,k) - 2._rt*F(i,j,k) + F(i,j-1,k);
        amrex::Real const cxz =
              F(i+1,j+1,k) + F(i+1,j-1,k) + F(i-1,j+1,k) + F(i-1,j-1,k)
            - 2._rt*(F(i+1,j,k) + F(i-1,j,k) + F(i,j+1,k) + F(i,j-1,k))
            + 4._rt*F(i,j,k);
        return d2x*inv_h2[0] + d2z*inv_h2[1]
             + ((h2[0] + h2[1])/12._rt)*cxz*inv_h2[0]*inv_h2[1];
#endif
    }

    // Corner-curl correction that isotropizes the in-plane resistive
    // diffusion of the out-of-plane B component (Bz in 3D, By in 2D XZ).
    //
    // The resistive E (eta*curl(B)/mu0) advanced by Faraday produces, for
    // divergence-free B, the cross-stencil in-plane Laplacian of the
    // out-of-plane B, whose leading error is cos(4*theta) anisotropic and
    // drives a grid m=4 mode. Adding the corrections below to the two
    // in-plane E components makes the standard Faraday curl emit the missing
    // Mehrstellen corner term (h^2/6) d2a d2b B, i.e. the in-plane isotropic
    // Laplacian. The correction enters only through E, so the Faraday curl is
    // unchanged and div(B) stays exactly zero. Returned values already carry
    // 1/mu0 (matching the resistive current); multiply by eta at the call.
    //
    // Coefficient: the in-plane Mehrstellen corner is
    // ((h2a+h2b)/12) * (d2a d2b B) * inv_h2a * inv_h2b. It is split
    // antisymmetrically between the two E components (half each), with each
    // half pre-differenced (backward) along the in-plane axis so that the
    // Faraday upward difference completes a centered second difference.
#if defined(WARPX_DIM_3D)
    // 3D: Bz corner delivered through Ex and Ey (curl_z). a=x, b=y.
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real CornerResistiveEx_3D (
        amrex::Array4<amrex::Real const> const& Bz,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2,
        amrex::GpuArray<amrex::Real, 3> const& dx,
        amrex::Real const inv_mu0)
    {
        using namespace amrex::literals;
        // half the corner coefficient, times dy (undone by Faraday UpwardDy)
        amrex::Real const cf = 0.5_rt*((h2[0]+h2[1])/12._rt)*inv_h2[0]*inv_h2[1]
                             * dx[1] * inv_mu0;
        auto const d2x = [&] (int jj) {
            return Bz(i+1,jj,k) - 2._rt*Bz(i,jj,k) + Bz(i-1,jj,k);
        };
        // backward y-difference of d2x(Bz); UpwardDy in Faraday -> d2y d2x Bz
        return cf * (d2x(j) - d2x(j-1));
    }
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real CornerResistiveEy_3D (
        amrex::Array4<amrex::Real const> const& Bz,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2,
        amrex::GpuArray<amrex::Real, 3> const& dx,
        amrex::Real const inv_mu0)
    {
        using namespace amrex::literals;
        amrex::Real const cf = -0.5_rt*((h2[0]+h2[1])/12._rt)*inv_h2[0]*inv_h2[1]
                             * dx[0] * inv_mu0;
        auto const d2y = [&] (int ii) {
            return Bz(ii,j+1,k) - 2._rt*Bz(ii,j,k) + Bz(ii,j-1,k);
        };
        // backward x-difference of d2y(Bz); UpwardDx in Faraday -> d2x d2y Bz
        return cf * (d2y(i) - d2y(i-1));
    }

    // Collocated (nodal) analogues. On a collocated grid the Faraday UpwardD is a
    // wide centered difference (CartesianNodalAlgorithm), so the uncorrected
    // resistive diffusion of the out-of-plane B is the WIDE cross stencil
    // -(sin^2(kx hx) + sin^2(ky hy))/h^2, whose leading cos(4*theta) anisotropy is
    // (h^2/3)(kx^4+ky^4) -- four times the compact-cross value. The Yee
    // backward-pre-difference trick does not apply (UpwardD is no longer one-sided),
    // so the corner is delivered as a compact 2nd difference along one in-plane axis
    // followed by the nodal centered difference along the other; the wide Faraday
    // UpwardD then completes the corner. A half (coefficient 1/3) goes through each
    // in-plane E component, which reproduces the (hxa^2+hxb^2)/3 corner for any cell
    // aspect ratio (isotropic for cubic cells, consistent otherwise). Returned values
    // carry 1/mu0; multiply by eta at the call.
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real CornerResistiveEx_3D_Nodal (
        amrex::Array4<amrex::Real const> const& Bz,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& dx,
        amrex::Real const inv_mu0)
    {
        using namespace amrex::literals;
        // dEx = (1/3)(1/mu0) Dn_y( d2x(Bz) ); Dn_y is the nodal centered difference.
        auto const d2x = [&] (int jj) {
            return Bz(i+1,jj,k) - 2._rt*Bz(i,jj,k) + Bz(i-1,jj,k);
        };
        return (inv_mu0/3._rt) * (0.5_rt/dx[1]) * (d2x(j+1) - d2x(j-1));
    }
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real CornerResistiveEy_3D_Nodal (
        amrex::Array4<amrex::Real const> const& Bz,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& dx,
        amrex::Real const inv_mu0)
    {
        using namespace amrex::literals;
        // dEy = -(1/3)(1/mu0) Dn_x( d2y(Bz) )
        auto const d2y = [&] (int ii) {
            return Bz(ii,j+1,k) - 2._rt*Bz(ii,j,k) + Bz(ii,j-1,k);
        };
        return -(inv_mu0/3._rt) * (0.5_rt/dx[0]) * (d2y(i+1) - d2y(i-1));
    }
#elif defined(WARPX_DIM_XZ)
    // 2D XZ: By corner delivered through Ex and Ez (curl_y). a=x, b=z (j index).
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real CornerResistiveEx_XZ (
        amrex::Array4<amrex::Real const> const& By,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2,
        amrex::GpuArray<amrex::Real, 3> const& dx,
        amrex::Real const inv_mu0)
    {
        using namespace amrex::literals;
        // curl_y = UpwardDz(Ex) - UpwardDx(Ez); this half goes via UpwardDz
        amrex::Real const cf = -0.5_rt*((h2[0]+h2[1])/12._rt)*inv_h2[0]*inv_h2[1]
                             * dx[1] * inv_mu0;
        auto const d2x = [&] (int jj) {
            return By(i+1,jj,k) - 2._rt*By(i,jj,k) + By(i-1,jj,k);
        };
        return cf * (d2x(j) - d2x(j-1));
    }
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real CornerResistiveEz_XZ (
        amrex::Array4<amrex::Real const> const& By,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& h2,
        amrex::GpuArray<amrex::Real, 3> const& inv_h2,
        amrex::GpuArray<amrex::Real, 3> const& dx,
        amrex::Real const inv_mu0)
    {
        using namespace amrex::literals;
        amrex::Real const cf = 0.5_rt*((h2[0]+h2[1])/12._rt)*inv_h2[0]*inv_h2[1]
                             * dx[0] * inv_mu0;
        auto const d2z = [&] (int ii) {
            return By(ii,j+1,k) - 2._rt*By(ii,j,k) + By(ii,j-1,k);
        };
        return cf * (d2z(i) - d2z(i-1));
    }

    // Collocated (nodal) analogues for 2D XZ (out-of-plane By, in-plane x and z).
    // Same wide-Faraday reasoning as the 3D nodal pair; the in-plane axes are x (i)
    // and z (j). Sign convention matches the Yee XZ pair (Ex half negative, Ez half
    // positive), set by the curl_y topology dBy/dt = UpwardDx(Ez) - UpwardDz(Ex).
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real CornerResistiveEx_XZ_Nodal (
        amrex::Array4<amrex::Real const> const& By,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& dx,
        amrex::Real const inv_mu0)
    {
        using namespace amrex::literals;
        // dEx = -(1/3)(1/mu0) Dn_z( d2x(By) ); the -UpwardDz(Ex) in Faraday -> +corner
        auto const d2x = [&] (int jj) {
            return By(i+1,jj,k) - 2._rt*By(i,jj,k) + By(i-1,jj,k);
        };
        return -(inv_mu0/3._rt) * (0.5_rt/dx[1]) * (d2x(j+1) - d2x(j-1));
    }
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real CornerResistiveEz_XZ_Nodal (
        amrex::Array4<amrex::Real const> const& By,
        int const i, int const j, int const k,
        amrex::GpuArray<amrex::Real, 3> const& dx,
        amrex::Real const inv_mu0)
    {
        using namespace amrex::literals;
        // dEz = +(1/3)(1/mu0) Dn_x( d2z(By) ); the +UpwardDx(Ez) in Faraday -> +corner
        auto const d2z = [&] (int ii) {
            return By(ii,j+1,k) - 2._rt*By(ii,j,k) + By(ii,j-1,k);
        };
        return (inv_mu0/3._rt) * (0.5_rt/dx[0]) * (d2z(i+1) - d2z(i-1));
    }
#endif

    /** Isotropic upward differences of a nodal scalar (electron pressure).
     *
     *  The bare two-point difference of grad(Pe) carries the same
     *  cos(4*theta) angular truncation error as the cross Laplacian: the
     *  gathered pressure electric field of a radial density feature is
     *  stronger along the grid axes than the diagonals, a fourfold (m=4)
     *  source in the ion push. Smoothing the bare difference with the
     *  binomial (1/4, 1/2, 1/4) weights in each transverse direction
     *  cancels the angular term on cubic cells (the gradient analogue of
     *  the Mehrstellen Laplacian). The stencil is expressed through
     *  T_Algo::UpwardD*, so it is correct for both the Yee (edge-centered
     *  E) and collocated (nodal E) staggerings.
     *
     *  3D only: in other dimensionalities these reduce to the bare
     *  difference (the 2D XZ plane has no azimuthal m=4 to cancel).
     */
    template <typename T_Algo>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real GradXIsotropic (
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_x, int const n_coefs_x,
        int const i, int const j, int const k)
    {
        using namespace amrex::literals;
#if defined(WARPX_DIM_3D)
        amrex::Real acc = 0._rt;
        for (int dj = -1; dj <= 1; ++dj) {
            const amrex::Real wj = (dj == 0) ? 0.5_rt : 0.25_rt;
            for (int dk = -1; dk <= 1; ++dk) {
                const amrex::Real wk = (dk == 0) ? 0.5_rt : 0.25_rt;
                acc += wj * wk * T_Algo::UpwardDx(F, coefs_x, n_coefs_x, i, j+dj, k+dk);
            }
        }
        return acc;
#else
        return T_Algo::UpwardDx(F, coefs_x, n_coefs_x, i, j, k);
#endif
    }

    template <typename T_Algo>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real GradYIsotropic (
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_y, int const n_coefs_y,
        int const i, int const j, int const k)
    {
        using namespace amrex::literals;
#if defined(WARPX_DIM_3D)
        amrex::Real acc = 0._rt;
        for (int di = -1; di <= 1; ++di) {
            const amrex::Real wi = (di == 0) ? 0.5_rt : 0.25_rt;
            for (int dk = -1; dk <= 1; ++dk) {
                const amrex::Real wk = (dk == 0) ? 0.5_rt : 0.25_rt;
                acc += wi * wk * T_Algo::UpwardDy(F, coefs_y, n_coefs_y, i+di, j, k+dk);
            }
        }
        return acc;
#else
        return T_Algo::UpwardDy(F, coefs_y, n_coefs_y, i, j, k);
#endif
    }

    template <typename T_Algo>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real GradZIsotropic (
        amrex::Array4<amrex::Real const> const& F,
        amrex::Real const * const coefs_z, int const n_coefs_z,
        int const i, int const j, int const k)
    {
        using namespace amrex::literals;
#if defined(WARPX_DIM_3D)
        amrex::Real acc = 0._rt;
        for (int di = -1; di <= 1; ++di) {
            const amrex::Real wi = (di == 0) ? 0.5_rt : 0.25_rt;
            for (int dj = -1; dj <= 1; ++dj) {
                const amrex::Real wj = (dj == 0) ? 0.5_rt : 0.25_rt;
                acc += wi * wj * T_Algo::UpwardDz(F, coefs_z, n_coefs_z, i+di, j+dj, k);
            }
        }
        return acc;
#else
        return T_Algo::UpwardDz(F, coefs_z, n_coefs_z, i, j, k);
#endif
    }

#endif // !RZ && !RCYLINDER && !RSPHERE
} // namespace warpx::hybrid_isotropic

#endif // WARPX_ISOTROPIC_OPERATORS_H_
