/* Copyright 2026 Prabhat Kumar
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 * Implementation detail included by HybridPICPC.H. See its provenance.
 */
#ifndef WARPX_HYBRID_PIC_PC_KERNELS_H_
#define WARPX_HYBRID_PIC_PC_KERNELS_H_

/** Per-level grid constants of the stencil kernels: the Cartesian 1/dx_d^2
 *  and 1/(2 dx_d) in the vector-component slots (unused slots zero), and
 *  the cylindrical radius data of the RZ stencils (index dims (r, z) =
 *  (i, j)). */
struct HybridPICPCGrid
{
    amrex::GpuArray<amrex::Real,3> invdx2 = {0.0, 0.0, 0.0};
    amrex::GpuArray<amrex::Real,3> inv2dx = {0.0, 0.0, 0.0};
    amrex::Real rmin = 0.0;
    amrex::Real dr = 0.0;
    amrex::Real inv_dr = 0.0;
    amrex::Real inv_dz = 0.0;
};

#if defined(WARPX_DIM_RZ)
/** RZ (m = 0) stencil helpers mirroring the residual's discrete operators
 *  on the collocated grid: the Faraday curl uses the CylindricalYee
 *  "upward" differences and the Ampere curl the "downward" ones, each with
 *  the residual's on-axis branches, and r evaluated at the
 *  staggering-assumed positions the kernels use (nodal for r/theta rows,
 *  cell-centered for the z-row of the Faraday curl and the r-row of the
 *  hyper-resistive Laplacian). Components are (r, theta, z) = (0, 1, 2);
 *  index dims are (r, z) = (i, j). */
namespace hybridpcrz {

    /** True curl of an E-like field, Faraday (upward) stencils. */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void CurlF (amrex::Array4<amrex::Real const> const& a,
                amrex::Real& cr, amrex::Real& ct, amrex::Real& cz,
                int i, int j, HybridPICPCGrid const& g)
    {
        using namespace amrex::literals;
        const amrex::Real rn = g.rmin + i*g.dr;             // nodal position
        const amrex::Real rc = g.rmin + (i + 0.5_rt)*g.dr;  // cell-centered
        // (curl a)_r = -d_z a_theta; the Faraday kernel zeroes B_r on axis
        cr = (rn == 0.0_rt) ? 0.0_rt
           : -g.inv_dz*(a(i,j+1,0,1) - a(i,j,0,1));
        // (curl a)_theta = d_z a_r - d_r a_z
        ct = g.inv_dz*(a(i,j+1,0,0) - a(i,j,0,0))
           - g.inv_dr*(a(i+1,j,0,2) - a(i,j,0,2));
        // (curl a)_z = (1/r) d_r (r a_theta), r at the cell center
        cz = (1.0_rt/rc)*g.inv_dr*((rc + 0.5_rt*g.dr)*a(i+1,j,0,1)
                                 - (rc - 0.5_rt*g.dr)*a(i,j,0,1));
    }

    /** True curl of a B-like field, Ampere (downward) stencils with the
     *  residual's on-axis regularizations. */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void CurlA (amrex::Array4<amrex::Real const> const& a,
                amrex::Real& cr, amrex::Real& ct, amrex::Real& cz,
                int i, int j, HybridPICPCGrid const& g)
    {
        using namespace amrex::literals;
        const amrex::Real rn = g.rmin + i*g.dr;
        // (curl a)_r = -d_z a_theta
        cr = -g.inv_dz*(a(i,j,0,1) - a(i,j-1,0,1));
        if (rn > 0.5_rt*g.dr) {
            // (curl a)_theta = d_z a_r - d_r a_z
            ct = g.inv_dz*(a(i,j,0,0) - a(i,j-1,0,0))
               - g.inv_dr*(a(i,j,0,2) - a(i-1,j,0,2));
            // (curl a)_z = (1/r) d_r (r a_theta), r at the node
            cz = (1.0_rt/rn)*g.inv_dr*((rn + 0.5_rt*g.dr)*a(i,j,0,1)
                                     - (rn - 0.5_rt*g.dr)*a(i-1,j,0,1));
        } else {
            ct = 0.0_rt;                          // J_theta = 0 on axis
            cz = 4.0_rt*a(i,j,0,1)*g.inv_dr;      // J_z axis regularization
        }
    }

    /** Component-wise hyper-resistive Laplacian of a J-like field, the
     *  residual's RZ forms (r-row at the cell-centered radius including the
     *  -1/r^2 curvature term; theta-row zero on axis; z-row switching to
     *  the plain second difference on axis). */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void LapJ (amrex::Array4<amrex::Real const> const& a,
               amrex::Real& lr, amrex::Real& lt, amrex::Real& lz,
               int i, int j, HybridPICPCGrid const& g)
    {
        using namespace amrex::literals;
        const amrex::Real rn = g.rmin + i*g.dr;
        const amrex::Real rc = g.rmin + (i + 0.5_rt)*g.dr;
        const amrex::Real inv_dr2 = g.inv_dr*g.inv_dr;
        const amrex::Real inv_dz2 = g.inv_dz*g.inv_dz;
        auto dzz = [&] (int c) {
            return inv_dz2*(a(i,j-1,0,c) - 2.0_rt*a(i,j,0,c) + a(i,j+1,0,c));
        };
        auto dr_rdr_over_r = [&] (amrex::Real r, int c) {
            return (1.0_rt/r)*inv_dr2*((r + 0.5_rt*g.dr)*(a(i+1,j,0,c) - a(i,j,0,c))
                                     - (r - 0.5_rt*g.dr)*(a(i,j,0,c) - a(i-1,j,0,c)));
        };
        lr = dr_rdr_over_r(rc, 0) + dzz(0) - a(i,j,0,0)/(rc*rc);
        lt = (rn > 0.0_rt)
           ? dr_rdr_over_r(rn, 1) + dzz(1) - a(i,j,0,1)/(rn*rn)
           : 0.0_rt;
        lz = dzz(2);
        if (rn > 0.5_rt*g.dr) {
            lz += dr_rdr_over_r(rn, 2);
        } else {
            lz += inv_dr2*(a(i-1,j,0,2) - 2.0_rt*a(i,j,0,2) + a(i+1,j,0,2));
        }
    }

} // namespace hybridpcrz
#endif // WARPX_DIM_RZ

/** Apply (I + S)^-1 to (vx, vy, vz) in place, where S u = (diag(w) u) x bhat
 *  is the deposited-mass-matrix ion response folded through the Hall term
 *  (w_c = Sigma_cc |B0| / rho_c; identity when w = 0). Closed-form 3x3
 *  adjugate inverse; reduces to the analytic rotation inverse
 *  (v - m v x bhat + m^2 (v.bhat) bhat)/(1 + m^2) when w is isotropic. */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void HybridPICPCInvIpS (amrex::Real bhx, amrex::Real bhy, amrex::Real bhz,
                        amrex::Real wx, amrex::Real wy, amrex::Real wz,
                        amrex::Real& vx, amrex::Real& vy, amrex::Real& vz)
{
    using namespace amrex::literals;
    // off-diagonal entries of A = I + S: (S u)_x = wy uy bhz - wz uz bhy, etc.
    const amrex::Real M01 =  wy*bhz;
    const amrex::Real M02 = -wz*bhy;
    const amrex::Real M10 = -wx*bhz;
    const amrex::Real M12 =  wz*bhx;
    const amrex::Real M20 =  wx*bhy;
    const amrex::Real M21 = -wy*bhx;
    const amrex::Real a00 = 1.0_rt  - M12*M21;
    const amrex::Real a01 = M02*M21 - M01;
    const amrex::Real a02 = M01*M12 - M02;
    const amrex::Real a10 = M12*M20 - M10;
    const amrex::Real a11 = 1.0_rt  - M02*M20;
    const amrex::Real a12 = M02*M10 - M12;
    const amrex::Real a20 = M10*M21 - M20;
    const amrex::Real a21 = M01*M20 - M21;
    const amrex::Real a22 = 1.0_rt  - M01*M10;
    const amrex::Real invdet = 1.0_rt / (a00 + M01*a10 + M02*a20);
    const amrex::Real ux = vx, uy = vy, uz = vz;
    vx = (a00*ux + a01*uy + a02*uz)*invdet;
    vy = (a10*ux + a11*uy + a12*uz)*invdet;
    vz = (a20*ux + a21*uy + a22*uz)*invdet;
}

/** Pair couplings at one node: inline evaluation of
 *      tc = (I+S)^-1 ( U(aW)/sW - w x F )   and   kk = sW * curl(F),
 *  with F = curl xE staged in a 3-comp array, aW the scaled auxiliary
 *  W-hat, w = theta*dt ue0 (coef comps 11-13) and sW the local W-hat scale
 *  (coef comp 14). Collocated grid: Cartesian central stencils, or the
 *  cylindrical Ampere curl and vector Laplacian in RZ; one call per node
 *  inside each consumer (smoother update, residual). */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void HybridPICPCPairTcKk (amrex::Array4<const amrex::Real> const& coef,
                          amrex::Array4<const amrex::Real> const& aW,
                          amrex::Array4<const amrex::Real> const& F,
                          int i, int j, int k,
                          bool ion,
                          HybridPICPCGrid const& g,
                          amrex::Real& tcx, amrex::Real& tcy, amrex::Real& tcz,
                          amrex::Real& kkx, amrex::Real& kky, amrex::Real& kkz)
{
    using namespace amrex::literals;

    const auto& invdx2 = g.invdx2;
    const auto& inv2dx = g.inv2dx;
    const amrex::Real bhx = coef(i,j,k,0);
    const amrex::Real bhy = coef(i,j,k,1);
    const amrex::Real bhz = coef(i,j,k,2);
    const amrex::Real cw = coef(i,j,k,3);
    const amrex::Real ae = coef(i,j,k,4);
    const amrex::Real ah = coef(i,j,k,5);
    const amrex::Real sW = coef(i,j,k,14);

    const amrex::Real ax = aW(i,j,k,0);
    const amrex::Real ay = aW(i,j,k,1);
    const amrex::Real az = aW(i,j,k,2);

    amrex::Real lx = 0.0_rt, ly = 0.0_rt, lz = 0.0_rt;
#if defined(WARPX_DIM_3D)
    lx = (aW(i+1,j,k,0) - 2.0_rt*ax + aW(i-1,j,k,0))*invdx2[0]
       + (aW(i,j+1,k,0) - 2.0_rt*ax + aW(i,j-1,k,0))*invdx2[1]
       + (aW(i,j,k+1,0) - 2.0_rt*ax + aW(i,j,k-1,0))*invdx2[2];
    ly = (aW(i+1,j,k,1) - 2.0_rt*ay + aW(i-1,j,k,1))*invdx2[0]
       + (aW(i,j+1,k,1) - 2.0_rt*ay + aW(i,j-1,k,1))*invdx2[1]
       + (aW(i,j,k+1,1) - 2.0_rt*ay + aW(i,j,k-1,1))*invdx2[2];
    lz = (aW(i+1,j,k,2) - 2.0_rt*az + aW(i-1,j,k,2))*invdx2[0]
       + (aW(i,j+1,k,2) - 2.0_rt*az + aW(i,j-1,k,2))*invdx2[1]
       + (aW(i,j,k+1,2) - 2.0_rt*az + aW(i,j,k-1,2))*invdx2[2];
    kkx = sW*((F(i,j+1,k,2) - F(i,j-1,k,2))*inv2dx[1]
            - (F(i,j,k+1,1) - F(i,j,k-1,1))*inv2dx[2]);
    kky = sW*((F(i,j,k+1,0) - F(i,j,k-1,0))*inv2dx[2]
            - (F(i+1,j,k,2) - F(i-1,j,k,2))*inv2dx[0]);
    kkz = sW*((F(i+1,j,k,1) - F(i-1,j,k,1))*inv2dx[0]
            - (F(i,j+1,k,0) - F(i,j-1,k,0))*inv2dx[1]);
#elif defined(WARPX_DIM_XZ)
    lx = (aW(i+1,j,k,0) - 2.0_rt*ax + aW(i-1,j,k,0))*invdx2[0]
       + (aW(i,j+1,k,0) - 2.0_rt*ax + aW(i,j-1,k,0))*invdx2[2];
    ly = (aW(i+1,j,k,1) - 2.0_rt*ay + aW(i-1,j,k,1))*invdx2[0]
       + (aW(i,j+1,k,1) - 2.0_rt*ay + aW(i,j-1,k,1))*invdx2[2];
    lz = (aW(i+1,j,k,2) - 2.0_rt*az + aW(i-1,j,k,2))*invdx2[0]
       + (aW(i,j+1,k,2) - 2.0_rt*az + aW(i,j-1,k,2))*invdx2[2];
    kkx = sW*(-(F(i,j+1,k,1) - F(i,j-1,k,1))*inv2dx[2]);
    kky = sW*((F(i,j+1,k,0) - F(i,j-1,k,0))*inv2dx[2]
            - (F(i+1,j,k,2) - F(i-1,j,k,2))*inv2dx[0]);
    kkz = sW*((F(i+1,j,k,1) - F(i-1,j,k,1))*inv2dx[0]);
#elif defined(WARPX_DIM_RZ)
    // cylindrical m = 0: the residual's vector Laplacian of the J-like W and
    // the Ampere (downward) curl of the B-like F, axis branches included
    amrex::ignore_unused(invdx2, inv2dx, k);
    hybridpcrz::LapJ(aW, lx, ly, lz, i, j, g);
    hybridpcrz::CurlA(F, kkx, kky, kkz, i, j, g);
    kkx *= sW;
    kky *= sW;
    kkz *= sW;
#else
    lx = (aW(i+1,j,k,0) - 2.0_rt*ax + aW(i-1,j,k,0))*invdx2[2];
    ly = (aW(i+1,j,k,1) - 2.0_rt*ay + aW(i-1,j,k,1))*invdx2[2];
    lz = (aW(i+1,j,k,2) - 2.0_rt*az + aW(i-1,j,k,2))*invdx2[2];
    kkx = sW*(-(F(i+1,j,k,1) - F(i-1,j,k,1))*inv2dx[2]);
    kky = sW*((F(i+1,j,k,0) - F(i-1,j,k,0))*inv2dx[2]);
    kkz = 0.0_rt;
#endif
    // U(aW)/sW - w x F, then the pointwise ion rotation
    tcx = (cw*(ay*bhz - az*bhy) + ae*ax - ah*lx)/sW;
    tcy = (cw*(az*bhx - ax*bhz) + ae*ay - ah*ly)/sW;
    tcz = (cw*(ax*bhy - ay*bhx) + ae*az - ah*lz)/sW;
    const amrex::Real ex = coef(i,j,k,11);
    const amrex::Real ey = coef(i,j,k,12);
    const amrex::Real ez = coef(i,j,k,13);
    tcx -= ey*F(i,j,k,2) - ez*F(i,j,k,1);
    tcy -= ez*F(i,j,k,0) - ex*F(i,j,k,2);
    tcz -= ex*F(i,j,k,1) - ey*F(i,j,k,0);
    if (ion) {
        HybridPICPCInvIpS(bhx, bhy, bhz,
                          coef(i,j,k,6), coef(i,j,k,7), coef(i,j,k,8),
                          tcx, tcy, tcz);
    }
}

/** Staggered (Yee) rotation legs of U at container index (i,j,k), mirroring
 *  the residual's Ohm structure: the Hall term is computed at NODES (each
 *  J-like component down-averaged along its cell-centered dimension), then
 *  up-averaged back to each E staggering; the resistive and hyper-resistive
 *  legs act at the component's own location with the node-stored
 *  coefficients up-averaged there. The caller passes the compact
 *  (index-space) Laplacian terms, already evaluated at the component
 *  locations, and applies the ion-response rotation afterwards. */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void HybridPICPCUStag (amrex::Array4<amrex::Real const> const& av,
                       amrex::Array4<amrex::Real const> const& coef,
                       int i, int j, int k,
                       amrex::Real lx, amrex::Real ly, amrex::Real lz,
                       amrex::Real& ux, amrex::Real& uy, amrex::Real& uz)
{
    using namespace amrex::literals;
    // c-component of c_w (v~ x bhat) at node (mi,mj,mk)
    auto ncross = [&] (int mi, int mj, int mk, int c) -> amrex::Real
    {
#if defined(WARPX_DIM_3D)
        const amrex::Real vx = 0.5_rt*(av(mi-1,mj,mk,0) + av(mi,mj,mk,0));
        const amrex::Real vy = 0.5_rt*(av(mi,mj-1,mk,1) + av(mi,mj,mk,1));
        const amrex::Real vz = 0.5_rt*(av(mi,mj,mk-1,2) + av(mi,mj,mk,2));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        const amrex::Real vx = 0.5_rt*(av(mi-1,mj,mk,0) + av(mi,mj,mk,0));
        const amrex::Real vy = av(mi,mj,mk,1);
        const amrex::Real vz = 0.5_rt*(av(mi,mj-1,mk,2) + av(mi,mj,mk,2));
#else // WARPX_DIM_1D_Z
        const amrex::Real vx = av(mi,mj,mk,0);
        const amrex::Real vy = av(mi,mj,mk,1);
        const amrex::Real vz = 0.5_rt*(av(mi-1,mj,mk,2) + av(mi,mj,mk,2));
#endif
        // like the residual's enE stage, the node cross carries NO
        // density: c_w rho undoes the 1/rho folded into coef(3), and the
        // caller divides by rho interpolated to the component location
        const amrex::Real cwr = coef(mi,mj,mk,3)*coef(mi,mj,mk,9);
        if (c == 0) {
            return cwr*(vy*coef(mi,mj,mk,2) - vz*coef(mi,mj,mk,1));
        } else if (c == 1) {
            return cwr*(vz*coef(mi,mj,mk,0) - vx*coef(mi,mj,mk,2));
        }
        return cwr*(vx*coef(mi,mj,mk,1) - vy*coef(mi,mj,mk,0));
    };
    // guarded divide: zero density (physical ghosts under the
    // zero-extension closure) means a zero Hall term, not NaN
    auto odiv = [] (amrex::Real num, amrex::Real den) {
        return (den > 0.0_rt) ? num/den : 0.0_rt;
    };
#if defined(WARPX_DIM_3D)
    ux = odiv(ncross(i,j,k,0) + ncross(i+1,j,k,0),
              coef(i,j,k,9) + coef(i+1,j,k,9))
       + 0.5_rt*(coef(i,j,k,4) + coef(i+1,j,k,4))*av(i,j,k,0)
       - 0.5_rt*(coef(i,j,k,5) + coef(i+1,j,k,5))*lx;
    uy = odiv(ncross(i,j,k,1) + ncross(i,j+1,k,1),
              coef(i,j,k,9) + coef(i,j+1,k,9))
       + 0.5_rt*(coef(i,j,k,4) + coef(i,j+1,k,4))*av(i,j,k,1)
       - 0.5_rt*(coef(i,j,k,5) + coef(i,j+1,k,5))*ly;
    uz = odiv(ncross(i,j,k,2) + ncross(i,j,k+1,2),
              coef(i,j,k,9) + coef(i,j,k+1,9))
       + 0.5_rt*(coef(i,j,k,4) + coef(i,j,k+1,4))*av(i,j,k,2)
       - 0.5_rt*(coef(i,j,k,5) + coef(i,j,k+1,5))*lz;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    ux = odiv(ncross(i,j,k,0) + ncross(i+1,j,k,0),
              coef(i,j,k,9) + coef(i+1,j,k,9))
       + 0.5_rt*(coef(i,j,k,4) + coef(i+1,j,k,4))*av(i,j,k,0)
       - 0.5_rt*(coef(i,j,k,5) + coef(i+1,j,k,5))*lx;
    uy = odiv(ncross(i,j,k,1), coef(i,j,k,9))
       + coef(i,j,k,4)*av(i,j,k,1) - coef(i,j,k,5)*ly;
    uz = odiv(ncross(i,j,k,2) + ncross(i,j+1,k,2),
              coef(i,j,k,9) + coef(i,j+1,k,9))
       + 0.5_rt*(coef(i,j,k,4) + coef(i,j+1,k,4))*av(i,j,k,2)
       - 0.5_rt*(coef(i,j,k,5) + coef(i,j+1,k,5))*lz;
#else // WARPX_DIM_1D_Z
    ux = odiv(ncross(i,j,k,0), coef(i,j,k,9))
       + coef(i,j,k,4)*av(i,j,k,0) - coef(i,j,k,5)*lx;
    uy = odiv(ncross(i,j,k,1), coef(i,j,k,9))
       + coef(i,j,k,4)*av(i,j,k,1) - coef(i,j,k,5)*ly;
    uz = odiv(ncross(i,j,k,2) + ncross(i+1,j,k,2),
              coef(i,j,k,9) + coef(i+1,j,k,9))
       + 0.5_rt*(coef(i,j,k,4) + coef(i+1,j,k,4))*av(i,j,k,2)
       - 0.5_rt*(coef(i,j,k,5) + coef(i+1,j,k,5))*lz;
#endif
}

/** Staggered (Yee) variant of HybridPICPCPairTcKk: same contract, with the
 *  three staggered replacements. (a) U(aW) goes through the
 *  HybridPICPCUStag node chain (the compact index-space Laplacian is legal
 *  at each component's own location). (b) kk = sW curl(F) uses the compact
 *  downward (Ampere) differences, landing on the E/J staggering. (c) the
 *  convection term w x F mirrors the Hall cross: F (B-staggered) is
 *  down-averaged to nodes over its cell-centered dimensions, crossed with
 *  the node-built w = theta*dt ue0 (coef comps 11-13), and up-averaged
 *  back to each E staggering. The ion rotation keeps the node-collocation
 *  shortcut: node coefficients applied to the edge-centered triple (an
 *  upper-bound treatment, safe for a PC). Cartesian only (the staggered RZ
 *  grid is rejected in Define). */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void HybridPICPCPairTcKkStag (amrex::Array4<const amrex::Real> const& coef,
                              amrex::Array4<const amrex::Real> const& aW,
                              amrex::Array4<const amrex::Real> const& F,
                              int i, int j, int k,
                              bool ion,
                              HybridPICPCGrid const& g,
                              amrex::Real& tcx, amrex::Real& tcy, amrex::Real& tcz,
                              amrex::Real& kkx, amrex::Real& kky, amrex::Real& kkz)
{
    using namespace amrex::literals;

#if defined(WARPX_DIM_RZ)
    amrex::ignore_unused(coef, aW, F, i, j, k, ion, g);
    tcx = tcy = tcz = 0.0_rt;
    kkx = kky = kkz = 0.0_rt;
#else
    const auto& invdx2 = g.invdx2;
    const auto& inv2dx = g.inv2dx;
    const amrex::Real sW = coef(i,j,k,14);
    const amrex::Real ax = aW(i,j,k,0);
    const amrex::Real ay = aW(i,j,k,1);
    const amrex::Real az = aW(i,j,k,2);

    // index-space Laplacian at each component's own staggering (all
    // samples of a component are dx-separated in every direction)
    amrex::Real lx = 0.0_rt, ly = 0.0_rt, lz = 0.0_rt;
#if defined(WARPX_DIM_3D)
    lx = (aW(i+1,j,k,0) - 2.0_rt*ax + aW(i-1,j,k,0))*invdx2[0]
       + (aW(i,j+1,k,0) - 2.0_rt*ax + aW(i,j-1,k,0))*invdx2[1]
       + (aW(i,j,k+1,0) - 2.0_rt*ax + aW(i,j,k-1,0))*invdx2[2];
    ly = (aW(i+1,j,k,1) - 2.0_rt*ay + aW(i-1,j,k,1))*invdx2[0]
       + (aW(i,j+1,k,1) - 2.0_rt*ay + aW(i,j-1,k,1))*invdx2[1]
       + (aW(i,j,k+1,1) - 2.0_rt*ay + aW(i,j,k-1,1))*invdx2[2];
    lz = (aW(i+1,j,k,2) - 2.0_rt*az + aW(i-1,j,k,2))*invdx2[0]
       + (aW(i,j+1,k,2) - 2.0_rt*az + aW(i,j-1,k,2))*invdx2[1]
       + (aW(i,j,k+1,2) - 2.0_rt*az + aW(i,j,k-1,2))*invdx2[2];
    {
        const amrex::Real fx = 2.0_rt*inv2dx[0];
        const amrex::Real fy = 2.0_rt*inv2dx[1];
        const amrex::Real fz = 2.0_rt*inv2dx[2];
        kkx = sW*((F(i,j,k,2) - F(i,j-1,k,2))*fy
                - (F(i,j,k,1) - F(i,j,k-1,1))*fz);
        kky = sW*((F(i,j,k,0) - F(i,j,k-1,0))*fz
                - (F(i,j,k,2) - F(i-1,j,k,2))*fx);
        kkz = sW*((F(i,j,k,1) - F(i-1,j,k,1))*fx
                - (F(i,j,k,0) - F(i,j-1,k,0))*fy);
    }
#elif defined(WARPX_DIM_XZ)
    lx = (aW(i+1,j,k,0) - 2.0_rt*ax + aW(i-1,j,k,0))*invdx2[0]
       + (aW(i,j+1,k,0) - 2.0_rt*ax + aW(i,j-1,k,0))*invdx2[2];
    ly = (aW(i+1,j,k,1) - 2.0_rt*ay + aW(i-1,j,k,1))*invdx2[0]
       + (aW(i,j+1,k,1) - 2.0_rt*ay + aW(i,j-1,k,1))*invdx2[2];
    lz = (aW(i+1,j,k,2) - 2.0_rt*az + aW(i-1,j,k,2))*invdx2[0]
       + (aW(i,j+1,k,2) - 2.0_rt*az + aW(i,j-1,k,2))*invdx2[2];
    {
        const amrex::Real fx = 2.0_rt*inv2dx[0];
        const amrex::Real fz = 2.0_rt*inv2dx[2];
        kkx = sW*(-(F(i,j,k,1) - F(i,j-1,k,1))*fz);
        kky = sW*((F(i,j,k,0) - F(i,j-1,k,0))*fz
                - (F(i,j,k,2) - F(i-1,j,k,2))*fx);
        kkz = sW*((F(i,j,k,1) - F(i-1,j,k,1))*fx);
    }
#else
    lx = (aW(i+1,j,k,0) - 2.0_rt*ax + aW(i-1,j,k,0))*invdx2[2];
    ly = (aW(i+1,j,k,1) - 2.0_rt*ay + aW(i-1,j,k,1))*invdx2[2];
    lz = (aW(i+1,j,k,2) - 2.0_rt*az + aW(i-1,j,k,2))*invdx2[2];
    {
        const amrex::Real fz = 2.0_rt*inv2dx[2];
        kkx = sW*(-(F(i,j,k,1) - F(i-1,j,k,1))*fz);
        kky = sW*((F(i,j,k,0) - F(i-1,j,k,0))*fz);
        kkz = 0.0_rt;
    }
#endif

    // U(aW) through the staggered Hall/resistive node chain
    amrex::Real ux, uy, uz;
    HybridPICPCUStag(aW, coef, i, j, k, lx, ly, lz, ux, uy, uz);

    // c-component of w x F at node (mi,mj,mk): F down-averaged to the node
    // over its cell-centered dimensions (the bn* pattern of the coefficient
    // build), w read at the node
    auto ncrossF = [&] (int mi, int mj, int mk, int c) -> amrex::Real
    {
#if defined(WARPX_DIM_3D)
        const amrex::Real fx = 0.25_rt*(F(mi,mj-1,mk-1,0) + F(mi,mj,mk-1,0)
                                      + F(mi,mj-1,mk,0)   + F(mi,mj,mk,0));
        const amrex::Real fy = 0.25_rt*(F(mi-1,mj,mk-1,1) + F(mi,mj,mk-1,1)
                                      + F(mi-1,mj,mk,1)   + F(mi,mj,mk,1));
        const amrex::Real fz = 0.25_rt*(F(mi-1,mj-1,mk,2) + F(mi,mj-1,mk,2)
                                      + F(mi-1,mj,mk,2)   + F(mi,mj,mk,2));
#elif defined(WARPX_DIM_XZ)
        const amrex::Real fx = 0.5_rt*(F(mi,mj-1,mk,0) + F(mi,mj,mk,0));
        const amrex::Real fy = 0.25_rt*(F(mi-1,mj-1,mk,1) + F(mi,mj-1,mk,1)
                                      + F(mi-1,mj,mk,1)   + F(mi,mj,mk,1));
        const amrex::Real fz = 0.5_rt*(F(mi-1,mj,mk,2) + F(mi,mj,mk,2));
#else
        const amrex::Real fx = 0.5_rt*(F(mi-1,mj,mk,0) + F(mi,mj,mk,0));
        const amrex::Real fy = 0.5_rt*(F(mi-1,mj,mk,1) + F(mi,mj,mk,1));
        const amrex::Real fz = F(mi,mj,mk,2);
#endif
        const amrex::Real wx = coef(mi,mj,mk,11);
        const amrex::Real wy = coef(mi,mj,mk,12);
        const amrex::Real wz = coef(mi,mj,mk,13);
        if (c == 0) { return wy*fz - wz*fy; }
        if (c == 1) { return wz*fx - wx*fz; }
        return wx*fy - wy*fx;
    };
    // up-average the node cross back to each E staggering
#if defined(WARPX_DIM_3D)
    const amrex::Real cvx = 0.5_rt*(ncrossF(i,j,k,0) + ncrossF(i+1,j,k,0));
    const amrex::Real cvy = 0.5_rt*(ncrossF(i,j,k,1) + ncrossF(i,j+1,k,1));
    const amrex::Real cvz = 0.5_rt*(ncrossF(i,j,k,2) + ncrossF(i,j,k+1,2));
#elif defined(WARPX_DIM_XZ)
    const amrex::Real cvx = 0.5_rt*(ncrossF(i,j,k,0) + ncrossF(i+1,j,k,0));
    const amrex::Real cvy = ncrossF(i,j,k,1);
    const amrex::Real cvz = 0.5_rt*(ncrossF(i,j,k,2) + ncrossF(i,j+1,k,2));
#else
    const amrex::Real cvx = ncrossF(i,j,k,0);
    const amrex::Real cvy = ncrossF(i,j,k,1);
    const amrex::Real cvz = 0.5_rt*(ncrossF(i,j,k,2) + ncrossF(i+1,j,k,2));
#endif

    tcx = ux/sW - cvx;
    tcy = uy/sW - cvy;
    tcz = uz/sW - cvz;
    if (ion) {
        HybridPICPCInvIpS(coef(i,j,k,0), coef(i,j,k,1), coef(i,j,k,2),
                          coef(i,j,k,6), coef(i,j,k,7), coef(i,j,k,8),
                          tcx, tcy, tcz);
    }
#endif // WARPX_DIM_RZ
}

#endif
