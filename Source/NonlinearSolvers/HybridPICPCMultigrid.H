/* Copyright 2026 Prabhat Kumar
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 * Implementation detail included by HybridPICPC.H. See its provenance.
 */
#ifndef WARPX_HYBRID_PIC_PC_MULTIGRID_H_
#define WARPX_HYBRID_PIC_PC_MULTIGRID_H_

template <class T, class Ops>
void HybridPICPC<T,Ops>::RestrictCoef ( int a_lev )
{
    using namespace amrex;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*m_coef[a_lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> const& c = m_coef[a_lev]->array(mfi);
        Array4<const Real> const& f = m_coef[a_lev-1]->const_array(mfi);
        amrex::ParallelFor(bx, kNCoef - 1,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int m)
            {
                // comps 0-9 (bhat, c_w, a_e, a_H, MM, rho) + the pair's
                // electron velocity (11-13) and local W-hat scale (14);
                // the level-0-only pe-row diagonal (10) is skipped
                const int n = (m < 10) ? m : (m + 1);
#if defined(WARPX_DIM_3D)
                c(i,j,k,n) = f(2*i,2*j,2*k,n);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                c(i,j,k,n) = f(2*i,2*j,k,n);
#else // WARPX_DIM_1D_Z
                c(i,j,k,n) = f(2*i,j,k,n);
#endif
            });
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::ComputeCurl ( const MF& a, MF& out, int a_lev, int a_grow,
                                       CurlKind a_kind, int a_scomp ) const
{
    using namespace amrex;

    const HybridPICPCGrid g = m_grid[a_lev];
#if defined(WARPX_DIM_RZ)
    const bool faraday = (a_kind == CurlKind::EtoB);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(out, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox(a_grow);
        Array4<const Real> const& av = a.const_array(mfi, a_scomp);
        Array4<Real> const& cv = out.array(mfi);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::ignore_unused(k);
                Real cr, ct, cz;
                if (faraday) {
                    hybridpcrz::CurlF(av, cr, ct, cz, i, j, g);
                } else {
                    hybridpcrz::CurlA(av, cr, ct, cz, i, j, g);
                }
                cv(i,j,0,0) = cr;
                cv(i,j,0,1) = ct;
                cv(i,j,0,2) = cz;
            });
    }
#else
    // staggered: compact up (E->B) / down (B->J) index-space differences
    const bool faraday = (a_kind == CurlKind::EtoB);
    const bool stag = m_staggered;
    const auto inv2dx = g.inv2dx;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(out, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox(a_grow);
        Array4<const Real> const& av = a.const_array(mfi, a_scomp);
        Array4<Real> const& cv = out.array(mfi);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
#if defined(WARPX_DIM_3D)
                if (stag) {
                    const int  s  = faraday ? 0 : -1;
                    const Real ix = 2.0_rt*inv2dx[0];
                    const Real iy = 2.0_rt*inv2dx[1];
                    const Real iz = 2.0_rt*inv2dx[2];
                    cv(i,j,k,0) = (av(i,j+1+s,k,2) - av(i,j+s,k,2))*iy
                                - (av(i,j,k+1+s,1) - av(i,j,k+s,1))*iz;
                    cv(i,j,k,1) = (av(i,j,k+1+s,0) - av(i,j,k+s,0))*iz
                                - (av(i+1+s,j,k,2) - av(i+s,j,k,2))*ix;
                    cv(i,j,k,2) = (av(i+1+s,j,k,1) - av(i+s,j,k,1))*ix
                                - (av(i,j+1+s,k,0) - av(i,j+s,k,0))*iy;
                } else {
                    const Real dyaz = (av(i,j+1,k,2) - av(i,j-1,k,2)) * inv2dx[1];
                    const Real dzay = (av(i,j,k+1,1) - av(i,j,k-1,1)) * inv2dx[2];
                    const Real dzax = (av(i,j,k+1,0) - av(i,j,k-1,0)) * inv2dx[2];
                    const Real dxaz = (av(i+1,j,k,2) - av(i-1,j,k,2)) * inv2dx[0];
                    const Real dxay = (av(i+1,j,k,1) - av(i-1,j,k,1)) * inv2dx[0];
                    const Real dyax = (av(i,j+1,k,0) - av(i,j-1,k,0)) * inv2dx[1];
                    cv(i,j,k,0) = dyaz - dzay;
                    cv(i,j,k,1) = dzax - dxaz;
                    cv(i,j,k,2) = dxay - dyax;
                }
#elif defined(WARPX_DIM_XZ)
                // (i,j) = (x,z); d/dy = 0
                if (stag) {
                    const int  s  = faraday ? 0 : -1;
                    const Real ix = 2.0_rt*inv2dx[0];
                    const Real iz = 2.0_rt*inv2dx[2];
                    cv(i,j,k,0) = -(av(i,j+1+s,k,1) - av(i,j+s,k,1))*iz;
                    cv(i,j,k,1) = (av(i,j+1+s,k,0) - av(i,j+s,k,0))*iz
                                - (av(i+1+s,j,k,2) - av(i+s,j,k,2))*ix;
                    cv(i,j,k,2) = (av(i+1+s,j,k,1) - av(i+s,j,k,1))*ix;
                } else {
                    const Real dzay = (av(i,j+1,k,1) - av(i,j-1,k,1)) * inv2dx[2];
                    const Real dzax = (av(i,j+1,k,0) - av(i,j-1,k,0)) * inv2dx[2];
                    const Real dxaz = (av(i+1,j,k,2) - av(i-1,j,k,2)) * inv2dx[0];
                    const Real dxay = (av(i+1,j,k,1) - av(i-1,j,k,1)) * inv2dx[0];
                    cv(i,j,k,0) = -dzay;
                    cv(i,j,k,1) = dzax - dxaz;
                    cv(i,j,k,2) = dxay;
                }
#else // WARPX_DIM_1D_Z
                // i = z; d/dx = d/dy = 0
                if (stag) {
                    const int  s  = faraday ? 0 : -1;
                    const Real iz = 2.0_rt*inv2dx[2];
                    cv(i,j,k,0) = -(av(i+1+s,j,k,1) - av(i+s,j,k,1))*iz;
                    cv(i,j,k,1) = (av(i+1+s,j,k,0) - av(i+s,j,k,0))*iz;
                    cv(i,j,k,2) = 0.0_rt;
                } else {
                    const Real dzay = (av(i+1,j,k,1) - av(i-1,j,k,1)) * inv2dx[2];
                    const Real dzax = (av(i+1,j,k,0) - av(i-1,j,k,0)) * inv2dx[2];
                    cv(i,j,k,0) = -dzay;
                    cv(i,j,k,1) = dzax;
                    cv(i,j,k,2) = 0.0_rt;
                }
#endif
            });
    }
#endif // WARPX_DIM_RZ
}



template <class T, class Ops>
void HybridPICPC<T,Ops>::StagePair ( int a_lev )
{
    // F = curl xE on grown(gU), so consumers can read its first ring for
    // the inline kk = sW curl F
    const int gU = kNGrow - 1;
    ComputeCurl(*m_xF[a_lev], *m_tmp[a_lev], a_lev, gU, CurlKind::EtoB);
    ApplyDomainClosure(*m_tmp[a_lev], a_lev, ClosureClass::BField);
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::SmoothLevel ( int a_lev, int a_nsweeps, bool a_x_is_zero )
{
    using namespace amrex;

    BL_PROFILE("HybridPICPC::SmoothLevel()");

    // Joint per-node 2x2 block update (see the class docs): tc and kk are
    // evaluated inline (HybridPICPCPairTcKk) from the staged F = curl xE
    // and the xW neighbors.
    //   rE = bF - xE - tc, rW = bW - xW + kk
    //   P dE = rE - (1/sW) U_loc rW ; dW = rW + sW l2 dE
    const RT l2 = m_l2[a_lev];
    const RT l1p = m_l1[a_lev];
    const RT sigma = m_sigma;
    const RT sigmaw = m_sigma_w;
    const Real wdl2 = static_cast<Real>(m_pair_wdf*l2);
    const bool ionp = m_have_mm;
    const HybridPICPCGrid g = m_grid[a_lev];

    for (int sweep = 0; sweep < a_nsweeps; sweep++) {

        const bool r_is_b = (sweep == 0 && a_x_is_zero);
        if (!r_is_b) {
            // one exchange per half feeds the whole sweep: W is delta-J-like
            // (parity ghosts, no wall write), E gets the E-field closure
            FillGhosts(*m_xW[a_lev], a_lev);
            ApplyDomainClosure(*m_xW[a_lev], a_lev, ClosureClass::JField);
            FillGhosts(*m_xF[a_lev], a_lev);
            ApplyDomainClosure(*m_xF[a_lev], a_lev, ClosureClass::EField);
            StagePair(a_lev);
        }

        if (m_staggered) {
            // NODE-SANDWICH block: on the Yee grid the pointwise block's
            // cross-couplings would act between samples h/2 apart -- in
            // Fourier space a phase e^{i(theta_a-theta_b)/2} the real
            // rotation cannot represent. Every component-coupling term
            // of the update therefore goes through the operator's own
            // down-average/cross/up-average chain (symbol c_a c_b: real,
            // damped, phase-free); every diagonal term stays pointwise.
            //
            // stage A: exact residuals rE -> m_rF, rW -> m_rW (valid),
            //          then one exchange each fills the gather ghosts
            for (MFIter mfi(*m_coef[a_lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.tilebox();
                Array4<const Real> const& coef = m_coef[a_lev]->const_array(mfi);
                Array4<const Real> const& bW = m_bW[a_lev]->const_array(mfi);
                Array4<const Real> const& bF = m_bF[a_lev]->const_array(mfi);
                Array4<const Real> const& Fc = m_tmp[a_lev]->const_array(mfi);
                Array4<const Real> const& xW = m_xW[a_lev]->const_array(mfi);
                Array4<const Real> const& xE = m_xF[a_lev]->const_array(mfi);
                Array4<Real> const& rE = m_rF[a_lev]->array(mfi);
                Array4<Real> const& rW = m_rW[a_lev]->array(mfi);
                amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        if (r_is_b) {
                            for (int c = 0; c < 3; c++) {
                                rE(i,j,k,c) = bF(i,j,k,c);
                                rW(i,j,k,c) = bW(i,j,k,c);
                            }
                            return;
                        }
                        Real tcx, tcy, tcz, kkx, kky, kkz;
                        HybridPICPCPairTcKkStag(coef, xW, Fc, i, j, k, ionp, g,
                                                tcx, tcy, tcz, kkx, kky, kkz);
                        rE(i,j,k,0) = bF(i,j,k,0) - xE(i,j,k,0) - tcx;
                        rE(i,j,k,1) = bF(i,j,k,1) - xE(i,j,k,1) - tcy;
                        rE(i,j,k,2) = bF(i,j,k,2) - xE(i,j,k,2) - tcz;
                        rW(i,j,k,0) = bW(i,j,k,0) - xW(i,j,k,0) + kkx;
                        rW(i,j,k,1) = bW(i,j,k,1) - xW(i,j,k,1) + kky;
                        rW(i,j,k,2) = bW(i,j,k,2) - xW(i,j,k,2) + kkz;
                    });
            }
            FillGhosts(*m_rF[a_lev], a_lev);
            FillGhosts(*m_rW[a_lev], a_lev);
            // stage B: node coupling q = Q sD~ + P^-1 sX~ -> m_tmp2 on
            // grow(1) (stage C reads q at the two nodes of each edge)
            for (MFIter mfi(*m_coef[a_lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.growntilebox(1);
                Array4<const Real> const& coef = m_coef[a_lev]->const_array(mfi);
                Array4<const Real> const& rE = m_rF[a_lev]->const_array(mfi);
                Array4<const Real> const& rW = m_rW[a_lev]->const_array(mfi);
                Array4<Real> const& qq = m_tmp2[a_lev]->array(mfi);
                amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        // A- gather of both residual halves to the node
                        // (each component averaged along its own
                        // cell-centered dimension)
                        Real gEx, gEy, gEz, gWx, gWy, gWz;
#if defined(WARPX_DIM_3D)
                        gEx = 0.5_rt*(rE(i-1,j,k,0) + rE(i,j,k,0));
                        gEy = 0.5_rt*(rE(i,j-1,k,1) + rE(i,j,k,1));
                        gEz = 0.5_rt*(rE(i,j,k-1,2) + rE(i,j,k,2));
                        gWx = 0.5_rt*(rW(i-1,j,k,0) + rW(i,j,k,0));
                        gWy = 0.5_rt*(rW(i,j-1,k,1) + rW(i,j,k,1));
                        gWz = 0.5_rt*(rW(i,j,k-1,2) + rW(i,j,k,2));
#elif defined(WARPX_DIM_XZ)
                        gEx = 0.5_rt*(rE(i-1,j,k,0) + rE(i,j,k,0));
                        gEy = rE(i,j,k,1);
                        gEz = 0.5_rt*(rE(i,j-1,k,2) + rE(i,j,k,2));
                        gWx = 0.5_rt*(rW(i-1,j,k,0) + rW(i,j,k,0));
                        gWy = rW(i,j,k,1);
                        gWz = 0.5_rt*(rW(i,j-1,k,2) + rW(i,j,k,2));
#else
                        gEx = rE(i,j,k,0);
                        gEy = rE(i,j,k,1);
                        gEz = 0.5_rt*(rE(i-1,j,k,2) + rE(i,j,k,2));
                        gWx = rW(i,j,k,0);
                        gWy = rW(i,j,k,1);
                        gWz = 0.5_rt*(rW(i-1,j,k,2) + rW(i,j,k,2));
#endif
                        const Real bhx = coef(i,j,k,0);
                        const Real bhy = coef(i,j,k,1);
                        const Real bhz = coef(i,j,k,2);
                        const Real cw = coef(i,j,k,3);
                        const Real aU = coef(i,j,k,4) + coef(i,j,k,5)*l2;
                        const Real uem = std::sqrt(
                            coef(i,j,k,11)*coef(i,j,k,11)
                          + coef(i,j,k,12)*coef(i,j,k,12)
                          + coef(i,j,k,13)*coef(i,j,k,13));
                        const Real alpha = 1.0_rt + l1p*uem + l2*aU;
                        const Real beta = l2*cw;
                        const Real isW = 1.0_rt/coef(i,j,k,14);
                        // sD = gE - (1/sW) aU gW ; sX = -(1/sW) cw gW x b
                        const Real sDx = gEx - isW*aU*gWx;
                        const Real sDy = gEy - isW*aU*gWy;
                        const Real sDz = gEz - isW*aU*gWz;
                        const Real sXx = -isW*cw*(gWy*bhz - gWz*bhy);
                        const Real sXy = -isW*cw*(gWz*bhx - gWx*bhz);
                        const Real sXz = -isW*cw*(gWx*bhy - gWy*bhx);
                        // q = Q sD + P^-1 sX, with t = sD + sX:
                        //   q = [alpha sX - beta (t x b)
                        //        + (beta^2/alpha)(t.b) b
                        //        - (beta^2/alpha) sD] / (alpha^2+beta^2)
                        const Real tx = sDx + sXx;
                        const Real ty = sDy + sXy;
                        const Real tz = sDz + sXz;
                        const Real txb_x = ty*bhz - tz*bhy;
                        const Real txb_y = tz*bhx - tx*bhz;
                        const Real txb_z = tx*bhy - ty*bhx;
                        const Real tdb = tx*bhx + ty*bhy + tz*bhz;
                        const Real b2oa = beta*beta/alpha;
                        const Real inv_a2b2 = 1.0_rt/(alpha*alpha + beta*beta);
                        qq(i,j,k,0) = inv_a2b2*(alpha*sXx - beta*txb_x
                                      + b2oa*(tdb*bhx - sDx));
                        qq(i,j,k,1) = inv_a2b2*(alpha*sXy - beta*txb_y
                                      + b2oa*(tdb*bhy - sDy));
                        qq(i,j,k,2) = inv_a2b2*(alpha*sXz - beta*txb_z
                                      + b2oa*(tdb*bhz - sDz));
                    });
            }
            // stage C: dE = sD/alpha + A+ q ; xE += sigma dE,
            // xW += sigma_w (rW + sW l2 dE)
            for (MFIter mfi(*m_coef[a_lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.tilebox();
                Array4<const Real> const& coef = m_coef[a_lev]->const_array(mfi);
                Array4<const Real> const& rE = m_rF[a_lev]->const_array(mfi);
                Array4<const Real> const& rW = m_rW[a_lev]->const_array(mfi);
                Array4<const Real> const& qq = m_tmp2[a_lev]->const_array(mfi);
                Array4<Real> const& xW = m_xW[a_lev]->array(mfi);
                Array4<Real> const& xE = m_xF[a_lev]->array(mfi);
                amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        const Real aU = coef(i,j,k,4) + coef(i,j,k,5)*l2;
                        const Real uem = std::sqrt(
                            coef(i,j,k,11)*coef(i,j,k,11)
                          + coef(i,j,k,12)*coef(i,j,k,12)
                          + coef(i,j,k,13)*coef(i,j,k,13));
                        const Real ialpha = 1.0_rt/(1.0_rt + l1p*uem + l2*aU);
                        const Real swl = coef(i,j,k,14);
                        const Real isW = 1.0_rt/swl;
                        const Real l2s = static_cast<Real>(l2)*swl;
                        // A+ scatter of q to each component's edge
                        Real qx, qy, qz;
#if defined(WARPX_DIM_3D)
                        qx = 0.5_rt*(qq(i,j,k,0) + qq(i+1,j,k,0));
                        qy = 0.5_rt*(qq(i,j,k,1) + qq(i,j+1,k,1));
                        qz = 0.5_rt*(qq(i,j,k,2) + qq(i,j,k+1,2));
#elif defined(WARPX_DIM_XZ)
                        qx = 0.5_rt*(qq(i,j,k,0) + qq(i+1,j,k,0));
                        qy = qq(i,j,k,1);
                        qz = 0.5_rt*(qq(i,j,k,2) + qq(i,j+1,k,2));
#else
                        qx = qq(i,j,k,0);
                        qy = qq(i,j,k,1);
                        qz = 0.5_rt*(qq(i,j,k,2) + qq(i+1,j,k,2));
#endif
                        const Real dEx = (rE(i,j,k,0) - isW*aU*rW(i,j,k,0))*ialpha + qx;
                        const Real dEy = (rE(i,j,k,1) - isW*aU*rW(i,j,k,1))*ialpha + qy;
                        const Real dEz = (rE(i,j,k,2) - isW*aU*rW(i,j,k,2))*ialpha + qz;
                        xE(i,j,k,0) += sigma*dEx;
                        xE(i,j,k,1) += sigma*dEy;
                        xE(i,j,k,2) += sigma*dEz;
                        xW(i,j,k,0) += sigmaw*(rW(i,j,k,0) + l2s*dEx);
                        xW(i,j,k,1) += sigmaw*(rW(i,j,k,1) + l2s*dEy);
                        xW(i,j,k,2) += sigmaw*(rW(i,j,k,2) + l2s*dEz);
                    });
            }
            continue;
        }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*m_coef[a_lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            Array4<const Real> const& coef = m_coef[a_lev]->const_array(mfi);
            Array4<const Real> const& bW = m_bW[a_lev]->const_array(mfi);
            Array4<const Real> const& bF = m_bF[a_lev]->const_array(mfi);
            Array4<const Real> const& Fc = m_tmp[a_lev]->const_array(mfi);
            // reads from the current iterate, writes into the ping-pong
            // partner (a pure Jacobi sweep; see m_xW2)
            Array4<const Real> const& xW = m_xW[a_lev]->const_array(mfi);
            Array4<const Real> const& xE = m_xF[a_lev]->const_array(mfi);
            Array4<Real> const& xWo = m_xW2[a_lev]->array(mfi);
            Array4<Real> const& xEo = m_xF2[a_lev]->array(mfi);
            amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    Real rEx, rEy, rEz, rWx, rWy, rWz;
                    // current iterate; zero on the first sweep of a zero
                    // initial guess (the level's x is NOT pre-zeroed, see
                    // m_zero_free)
                    Real xEx = 0.0_rt, xEy = 0.0_rt, xEz = 0.0_rt;
                    Real xWx = 0.0_rt, xWy = 0.0_rt, xWz = 0.0_rt;
                    if (r_is_b) {
                        rEx = bF(i,j,k,0); rEy = bF(i,j,k,1); rEz = bF(i,j,k,2);
                        rWx = bW(i,j,k,0); rWy = bW(i,j,k,1); rWz = bW(i,j,k,2);
                    } else {
                        Real tcx, tcy, tcz, kkx, kky, kkz;
                        HybridPICPCPairTcKk(coef, xW, Fc, i, j, k, ionp, g,
                                            tcx, tcy, tcz, kkx, kky, kkz);
                        xEx = xE(i,j,k,0); xEy = xE(i,j,k,1); xEz = xE(i,j,k,2);
                        xWx = xW(i,j,k,0); xWy = xW(i,j,k,1); xWz = xW(i,j,k,2);
                        rEx = bF(i,j,k,0) - xEx - tcx;
                        rEy = bF(i,j,k,1) - xEy - tcy;
                        rEz = bF(i,j,k,2) - xEz - tcz;
                        rWx = bW(i,j,k,0) - xWx + kkx;
                        rWy = bW(i,j,k,1) - xWy + kky;
                        rWz = bW(i,j,k,2) - xWz + kkz;
                    }
                    const Real bhx = coef(i,j,k,0);
                    const Real bhy = coef(i,j,k,1);
                    const Real bhz = coef(i,j,k,2);
                    const Real cw = coef(i,j,k,3);
                    const Real aU = coef(i,j,k,4) + coef(i,j,k,5)*l2;
                    // local W-hat scale (coef comp 14)
                    const Real swl = coef(i,j,k,14);
                    const Real isW = 1.0_rt/swl;
                    // s = rE - (1/sW) U_loc rW (the E-row couples to
                    // the scaled W-hat through U_loc/sW)
                    const Real sx = rEx - isW*(cw*(rWy*bhz - rWz*bhy) + aU*rWx);
                    const Real sy = rEy - isW*(cw*(rWz*bhx - rWx*bhz) + aU*rWy);
                    const Real sz = rEz - isW*(cw*(rWx*bhy - rWy*bhx) + aU*rWz);
                    // P^-1 s with the block defects: hyper-resistive at
                    // the Laplacian Nyquist symbol 2*l2, whistler at wdl2
                    const Real uem = std::sqrt(
                        coef(i,j,k,11)*coef(i,j,k,11)
                      + coef(i,j,k,12)*coef(i,j,k,12)
                      + coef(i,j,k,13)*coef(i,j,k,13));
                    const Real aUb = coef(i,j,k,4) + coef(i,j,k,5)*(2.0_rt*l2);
                    const Real alpha = 1.0_rt + l1p*uem + l2*aUb;
                    const Real beta = wdl2*cw;
                    const Real sxb_x = sy*bhz - sz*bhy;
                    const Real sxb_y = sz*bhx - sx*bhz;
                    const Real sxb_z = sx*bhy - sy*bhx;
                    const Real sdb = sx*bhx + sy*bhy + sz*bhz;
                    const Real inv_a2b2 = 1.0_rt / (alpha*alpha + beta*beta);
                    const Real b2oa = beta*beta/alpha;
                    const Real dEx = inv_a2b2*(alpha*sx - beta*sxb_x + b2oa*sdb*bhx);
                    const Real dEy = inv_a2b2*(alpha*sy - beta*sxb_y + b2oa*sdb*bhy);
                    const Real dEz = inv_a2b2*(alpha*sz - beta*sxb_z + b2oa*sdb*bhz);
                    xEo(i,j,k,0) = xEx + sigma*dEx;
                    xEo(i,j,k,1) = xEy + sigma*dEy;
                    xEo(i,j,k,2) = xEz + sigma*dEz;
                    // perp-b part of dE couples through the whistler
                    // defect, the par-b part through the full l2
                    const Real dEb = dEx*bhx + dEy*bhy + dEz*bhz;
                    const Real dl = static_cast<Real>(l2) - wdl2;
                    xWo(i,j,k,0) = xWx + sigmaw*(rWx + swl*(wdl2*dEx + dl*dEb*bhx));
                    xWo(i,j,k,1) = xWy + sigmaw*(rWy + swl*(wdl2*dEy + dl*dEb*bhy));
                    xWo(i,j,k,2) = xWz + sigmaw*(rWz + swl*(wdl2*dEz + dl*dEb*bhz));
                });
        }
        // the partner now holds the new iterate (valid region); its
        // ghosts are refilled by the next FillGhosts before any read
        std::swap(m_xW[a_lev], m_xW2[a_lev]);
        std::swap(m_xF[a_lev], m_xF2[a_lev]);
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::ResidualLevel ( int a_lev )
{
    using namespace amrex;

    BL_PROFILE("HybridPICPC::ResidualLevel()");

    const bool stag = m_staggered;
    const bool ionp = m_have_mm;
    const HybridPICPCGrid g = m_grid[a_lev];

    FillGhosts(*m_xW[a_lev], a_lev);
    ApplyDomainClosure(*m_xW[a_lev], a_lev, ClosureClass::JField);
    FillGhosts(*m_xF[a_lev], a_lev);
    ApplyDomainClosure(*m_xF[a_lev], a_lev, ClosureClass::EField);
    StagePair(a_lev);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*m_rW[a_lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<const Real> const& coef = m_coef[a_lev]->const_array(mfi);
        Array4<const Real> const& bW = m_bW[a_lev]->const_array(mfi);
        Array4<const Real> const& bF = m_bF[a_lev]->const_array(mfi);
        Array4<const Real> const& xW = m_xW[a_lev]->const_array(mfi);
        Array4<const Real> const& xE = m_xF[a_lev]->const_array(mfi);
        Array4<const Real> const& Fc = m_tmp[a_lev]->const_array(mfi);
        Array4<Real> const& rW = m_rW[a_lev]->array(mfi);
        Array4<Real> const& rF = m_rF[a_lev]->array(mfi);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Real tcx, tcy, tcz, kkx, kky, kkz;
                if (stag) {
                    HybridPICPCPairTcKkStag(coef, xW, Fc, i, j, k, ionp, g,
                                            tcx, tcy, tcz, kkx, kky, kkz);
                } else {
                    HybridPICPCPairTcKk(coef, xW, Fc, i, j, k, ionp, g,
                                        tcx, tcy, tcz, kkx, kky, kkz);
                }
                rF(i,j,k,0) = bF(i,j,k,0) - xE(i,j,k,0) - tcx;
                rF(i,j,k,1) = bF(i,j,k,1) - xE(i,j,k,1) - tcy;
                rF(i,j,k,2) = bF(i,j,k,2) - xE(i,j,k,2) - tcz;
                rW(i,j,k,0) = bW(i,j,k,0) - xW(i,j,k,0) + kkx;
                rW(i,j,k,1) = bW(i,j,k,1) - xW(i,j,k,1) + kky;
                rW(i,j,k,2) = bW(i,j,k,2) - xW(i,j,k,2) + kkz;
            });
    }
}


template <class T, class Ops>
void HybridPICPC<T,Ops>::RestrictLevel ( int a_lev )
{
    using namespace amrex;

    BL_PROFILE("HybridPICPC::RestrictLevel()");

    // nodal full weighting of the fine residual(s); fine ghosts are filled
    // (periodic/internal) or zero (physical: homogeneous Dirichlet). The
    // same index-space transfer serves the staggered components.
    auto restrict_fw = [&] (const MF& a_fine, MF& a_coarse)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(a_coarse, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            Array4<const Real> const& fr = a_fine.const_array(mfi);
            Array4<Real> const& cb = a_coarse.array(mfi);
            amrex::ParallelFor(bx, 3,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
#if defined(WARPX_DIM_3D)
                    Real acc = 0.0_rt;
                    for (int oz = -1; oz <= 1; oz++) {
                    for (int oy = -1; oy <= 1; oy++) {
                    for (int ox = -1; ox <= 1; ox++) {
                        const Real w = (ox == 0 ? 0.5_rt : 0.25_rt)
                                     * (oy == 0 ? 0.5_rt : 0.25_rt)
                                     * (oz == 0 ? 0.5_rt : 0.25_rt);
                        acc += w * fr(2*i+ox, 2*j+oy, 2*k+oz, n);
                    }}}
                    cb(i,j,k,n) = acc;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    Real acc = 0.0_rt;
                    for (int oz = -1; oz <= 1; oz++) {
                    for (int ox = -1; ox <= 1; ox++) {
                        const Real w = (ox == 0 ? 0.5_rt : 0.25_rt)
                                     * (oz == 0 ? 0.5_rt : 0.25_rt);
                        acc += w * fr(2*i+ox, 2*j+oz, k, n);
                    }}
                    cb(i,j,k,n) = acc;
#else // WARPX_DIM_1D_Z
                    cb(i,j,k,n) = 0.5_rt*fr(2*i,j,k,n)
                                + 0.25_rt*(fr(2*i-1,j,k,n) + fr(2*i+1,j,k,n));
#endif
                });
        }
    };

    restrict_fw(*m_rW[a_lev], *m_bW[a_lev+1]);
    restrict_fw(*m_rF[a_lev], *m_bF[a_lev+1]);
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::ProlongLevel ( int a_lev )
{
    using namespace amrex;

    BL_PROFILE("HybridPICPC::ProlongLevel()");

    // The nodal (tri)linear prolongation reads coarse nodes ic and ic+1 <=
    // N/2 only -- never a coarse ghost -- so the exchange is needed only on
    // the cell-indexed staggered container (fine cell N-1 reads coarse
    // cell N/2, the periodic image of cell 0). The closure still applies
    // the wall-plane rows of the coarse correction. Level-0 index space
    // starts at 0, so i/2 floors correctly.
    auto prolong_lin = [&] (const MF& a_coarse, MF& a_fine)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(a_fine, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            Array4<const Real> const& xc = a_coarse.const_array(mfi);
            Array4<Real> const& xf = a_fine.array(mfi);
            amrex::ParallelFor(bx, 3,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
#if defined(WARPX_DIM_3D)
                    const int ic = i/2; const int fx = i - 2*ic;
                    const int jc = j/2; const int fy = j - 2*jc;
                    const int kc = k/2; const int fz = k - 2*kc;
                    Real acc = 0.0_rt;
                    for (int c = 0; c <= fz; c++) {
                    for (int b = 0; b <= fy; b++) {
                    for (int a = 0; a <= fx; a++) {
                        acc += xc(ic+a, jc+b, kc+c, n);
                    }}}
                    xf(i,j,k,n) += acc * (fx ? 0.5_rt : 1.0_rt)
                                       * (fy ? 0.5_rt : 1.0_rt)
                                       * (fz ? 0.5_rt : 1.0_rt);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    const int ic = i/2; const int fx = i - 2*ic;
                    const int jc = j/2; const int fy = j - 2*jc;
                    Real acc = 0.0_rt;
                    for (int b = 0; b <= fy; b++) {
                    for (int a = 0; a <= fx; a++) {
                        acc += xc(ic+a, jc+b, k, n);
                    }}
                    xf(i,j,k,n) += acc * (fx ? 0.5_rt : 1.0_rt)
                                       * (fy ? 0.5_rt : 1.0_rt);
#else // WARPX_DIM_1D_Z
                    const int ic = i/2; const int fx = i - 2*ic;
                    Real acc = 0.0_rt;
                    for (int a = 0; a <= fx; a++) { acc += xc(ic+a, j, k, n); }
                    xf(i,j,k,n) += acc * (fx ? 0.5_rt : 1.0_rt);
#endif
                });
        }
    };

    if (m_staggered) {
        FillGhosts(*m_xW[a_lev+1], a_lev+1);
        FillGhosts(*m_xF[a_lev+1], a_lev+1);
    }
    ApplyDomainClosure(*m_xW[a_lev+1], a_lev+1, ClosureClass::JField);
    ApplyDomainClosure(*m_xF[a_lev+1], a_lev+1, ClosureClass::EField);
    prolong_lin(*m_xW[a_lev+1], *m_xW[a_lev]);
    prolong_lin(*m_xF[a_lev+1], *m_xF[a_lev]);
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::Vcycle ( int a_lev, bool a_x_is_zero )
{
    if (a_lev == m_nlevels - 1) {
        const int nb = (m_nlevels == 1) ? kNuSingle : kNuBottom;
        SmoothLevel(a_lev, nb, a_x_is_zero);
        return;
    }

    SmoothLevel(a_lev, kNuPre, a_x_is_zero);

    ResidualLevel(a_lev);
    FillGhosts(*m_rW[a_lev], a_lev);
    ApplyDomainClosure(*m_rW[a_lev], a_lev, ClosureClass::JField);
    FillGhosts(*m_rF[a_lev], a_lev);
    ApplyDomainClosure(*m_rF[a_lev], a_lev, ClosureClass::EField);
    RestrictLevel(a_lev);
    if (!(m_zero_free && m_eb_flag == nullptr)) {
        m_xW[a_lev+1]->setVal(0.0);
        m_xF[a_lev+1]->setVal(0.0);
    }
    Vcycle(a_lev + 1, true);
    ProlongLevel(a_lev);

    SmoothLevel(a_lev, kNuPost, false);
}

#endif
