/* Copyright 2026 Prabhat Kumar
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 * Implementation detail included by HybridPICPC.H. See its provenance.
 */
#ifndef WARPX_HYBRID_PIC_PC_ALGEBRA_H_
#define WARPX_HYBRID_PIC_PC_ALGEBRA_H_

template <class T, class Ops>
void HybridPICPC<T,Ops>::ApplyMToW ( const MF& a_blk, int a_scomp )
{
    // block input -> m_bW[0] + m_bF[0] in one kernel, then one V-cycle
    SplitBlock(a_blk, a_scomp);
    if (!(m_zero_free && m_eb_flag == nullptr)) {
        m_xW[0]->setVal(0.0);
        m_xF[0]->setVal(0.0);
    }
    Vcycle(0, true);
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::ApplyReducedOp ( const MF& a_blk, MF& a_out )
{
    using namespace amrex;

    // a_blk must be ghost-filled (kNGrow layers) by the caller; the W-half
    // is comps 0..2, the E-half comps 3..5. out_W = W - kk ; out_E = xE +
    // tc, with F = curl xE staged from the block's E-half
    const int gU = kNGrow - 1;
    ComputeCurl(a_blk, *m_tmp[0], 0, gU, CurlKind::EtoB, 3);
    ApplyDomainClosure(*m_tmp[0], 0, ClosureClass::BField);
    const bool ionp = m_have_mm;
    const bool stag = m_staggered;
    const HybridPICPCGrid g = m_grid[0];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(a_out, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<const Real> const& coef = m_coef[0]->const_array(mfi);
        Array4<const Real> const& av = a_blk.const_array(mfi);
        Array4<const Real> const& zE = a_blk.const_array(mfi, 3);
        Array4<const Real> const& Fc = m_tmp[0]->const_array(mfi);
        Array4<Real> const& ow = a_out.array(mfi);
        Array4<Real> const& oe = a_out.array(mfi, 3);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Real tcx, tcy, tcz, kkx, kky, kkz;
                if (stag) {
                    HybridPICPCPairTcKkStag(coef, av, Fc, i, j, k, ionp, g,
                                            tcx, tcy, tcz, kkx, kky, kkz);
                } else {
                    HybridPICPCPairTcKk(coef, av, Fc, i, j, k, ionp, g,
                                        tcx, tcy, tcz, kkx, kky, kkz);
                }
                ow(i,j,k,0) = av(i,j,k,0) - kkx;
                ow(i,j,k,1) = av(i,j,k,1) - kky;
                ow(i,j,k,2) = av(i,j,k,2) - kkz;
                oe(i,j,k,0) = zE(i,j,k,0) + tcx;
                oe(i,j,k,1) = zE(i,j,k,1) + tcy;
                oe(i,j,k,2) = zE(i,j,k,2) + tcz;
            });
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::SplitBlock ( const MF& a_blk, int a_scomp )
{
    using namespace amrex;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*m_bW[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real const> const& blk = a_blk.const_array(mfi, a_scomp);
        Array4<Real> const& bw = m_bW[0]->array(mfi);
        Array4<Real> const& bf = m_bF[0]->array(mfi);
        amrex::ParallelFor(bx, 3,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int c)
            {
                bw(i,j,k,c) = blk(i,j,k,c);
                bf(i,j,k,c) = blk(i,j,k,3+c);
            });
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::MergeBlock ( MF& a_blk ) const
{
    using namespace amrex;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(a_blk, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> const& blk = a_blk.array(mfi);
        Array4<Real const> const& xw = m_xW[0]->const_array(mfi);
        Array4<Real const> const& xf = m_xF[0]->const_array(mfi);
        amrex::ParallelFor(bx, 3,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int c)
            {
                blk(i,j,k,c) = xw(i,j,k,c);
                blk(i,j,k,3+c) = xf(i,j,k,c);
            });
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::BlockAxpy ( MF& a_dst, int a_dcomp, const MF* a_src, int a_nv,
                                     const amrex::GpuArray<amrex::Real, kMaxKry>& a_c,
                                     amrex::Real a_scale ) const
{
    using namespace amrex;
    const int nv = a_nv;
    constexpr int nkb = kNKB;
    const Real scale = a_scale;
    const bool have_src = (a_src != nullptr);
    const auto c = a_c;
    // Aliasing contract: the source is read at comps 0..nkb-1 and the
    // destination written at a_dcomp..a_dcomp+nkb-1, so in-place use is safe
    // only when both ranges coincide (a_dcomp == 0) or are disjoint; writing
    // into the Krylov block must not overlap the basis vectors being read.
    AMREX_ALWAYS_ASSERT(a_src != &a_dst || a_dcomp == 0 || a_dcomp >= nkb);
    AMREX_ALWAYS_ASSERT(&a_dst != m_kryB.get() || a_dcomp >= nv * nkb);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(a_dst, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> const& d = a_dst.array(mfi, a_dcomp);
        Array4<Real const> const& V = m_kryB->const_array(mfi);
        Array4<Real const> const s = have_src ? a_src->const_array(mfi)
                                              : Array4<Real const>{};
        amrex::ParallelFor(bx, nkb,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int cc)
            {
                // in-place use (dst == src, a_dcomp == 0) is safe: each
                // thread then reads and writes the same element
                Real t = have_src ? s(i,j,k,cc) : 0.0_rt;
                for (int n = 0; n < nv; ++n) {
                    t -= c[n] * V(i,j,k, n*nkb + cc);
                }
                d(i,j,k,cc) = scale * t;
            });
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::BatchDot ( const MF& a_w, int a_nv, bool a_with_ww, RT* a_out )
{
    using namespace amrex;
    using warpx::fields::FieldType;
    using ablastr::fields::Direction;

    constexpr int nkb = kNKB;
    const int nslots = a_nv + (a_with_ww ? 1 : 0);
    if (nslots == 0) { return; }
    // collocated grid: the nodal owner mask of Efield_fp (duplicated
    // periodic/box-shared nodes counted once); staggered containers are
    // cell-indexed and need no mask
    const bool use_mask = !m_staggered;
    const amrex::iMultiFab* mask = nullptr;
    if (use_mask) {
        mask = WarpX::GetInstance().getFieldDotMaskPointer(
            FieldType::Efield_fp, 0, Direction{0});
    }
    std::vector<amrex::Real> res(nslots, 0.0);

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    if (Gpu::inLaunchRegion()) {
        constexpr int NV = 8;   // basis vectors per register chunk
        const int nbt = m_dot_nblocks_total;
        amrex::Real* part = m_dot_dev.data();
        for (MFIter mfi(a_w, false); mfi.isValid(); ++mfi) {
            const Box bx = mfi.validbox();
            const int ib = mfi.LocalIndex();
            const int nblocks = m_dot_nblocks[ib];
            const int boff = m_dot_box_offset[ib];
            Array4<Real const> const w = a_w.const_array(mfi);
            Array4<Real const> const V = m_kryB->const_array(mfi);
            Array4<int const> const mk = use_mask ? mask->const_array(mfi)
                                                  : Array4<int const>{};
            const auto lo = amrex::lbound(bx);
            const auto len = amrex::length(bx);
            const Long npts = bx.numPts();
            const int nv = a_nv;
            const bool with_ww = a_with_ww;
            amrex::launch<kDotThreads>(nblocks, Gpu::gpuStream(),
                [=] AMREX_GPU_DEVICE () noexcept
                {
                    const int tid = static_cast<int>(threadIdx.x);
                    const Long stride = Long(blockDim.x) * Long(gridDim.x);
                    const Long start = Long(blockDim.x) * Long(blockIdx.x) + tid;
                    const Long lxy = Long(len.x) * Long(len.y);
                    // chunks of NV basis vectors; the <w,w> slot rides
                    // with the first chunk. Every thread executes every
                    // block reduction (uniform control flow).
                    const int nchunks = amrex::max(1, (nv + NV - 1) / NV);
                    for (int ch = 0; ch < nchunks; ++ch) {
                        const int n0 = ch * NV;
                        const bool do_ww = with_ww && (ch == 0);
                        Real s[NV];
                        for (int n = 0; n < NV; ++n) { s[n] = 0.0_rt; }
                        Real sww = 0.0_rt;
                        for (Long ic = start; ic < npts; ic += stride) {
                            const int k = static_cast<int>(ic / lxy);
                            const Long r = ic - Long(k) * lxy;
                            const int j = static_cast<int>(r / len.x);
                            const int i = static_cast<int>(r - Long(j) * len.x);
                            const int ii = i + lo.x;
                            const int jj = j + lo.y;
                            const int kk = k + lo.z;
                            if (use_mask && mk(ii,jj,kk) == 0) { continue; }
                            for (int c = 0; c < nkb; ++c) {
                                const Real wc = w(ii,jj,kk,c);
                                if (do_ww) { sww += wc * wc; }
                                for (int n = 0; n < NV; ++n) {
                                    if (n0 + n < nv) {
                                        s[n] += wc * V(ii,jj,kk,(n0 + n)*nkb + c);
                                    }
                                }
                            }
                        }
                        for (int n = 0; n < NV; ++n) {
                            const Real rs = Gpu::blockReduceSum<kDotThreads>(s[n]);
                            if (tid == 0 && n0 + n < nv) {
                                part[std::size_t(n0 + n) * nbt + boff + blockIdx.x] = rs;
                            }
                        }
                        if (do_ww) {
                            const Real rs = Gpu::blockReduceSum<kDotThreads>(sww);
                            if (tid == 0) {
                                part[std::size_t(nv) * nbt + boff + blockIdx.x] = rs;
                            }
                        }
                    }
                });
        }
        // the one host synchronization of the pass; fixed-order host sum
        const std::size_t ncopy = static_cast<std::size_t>(nslots) * nbt;
        Gpu::dtoh_memcpy(m_dot_host.data(), m_dot_dev.data(), ncopy * sizeof(amrex::Real));
        for (int n = 0; n < nslots; ++n) {
            amrex::Real ssum = 0.0_rt;
            for (int b = 0; b < nbt; ++b) {
                ssum += m_dot_host[static_cast<std::size_t>(n) * nbt + b];
            }
            res[n] = ssum;
        }
    } else
#endif
    {
        for (int n = 0; n < nslots; ++n) {
            const bool ww = (n == a_nv);
            const int vc = ww ? 0 : n * nkb;
            const MF& b = ww ? a_w : *m_kryB;
            res[n] = use_mask
                ? amrex::MultiFab::Dot(*mask, a_w, 0, b, vc, nkb, 0, true)
                : amrex::MultiFab::Dot(a_w, 0, b, vc, nkb, 0, true);
        }
    }
    if (amrex::ParallelContext::NProcsSub() > 1) {
        amrex::ParallelAllReduce::Sum(res.data(), nslots,
                                      amrex::ParallelContext::CommunicatorSub());
    }
    for (int n = 0; n < nslots; ++n) { a_out[n] = static_cast<RT>(res[n]); }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::RotateValid ( MF& a ) const
{
    using namespace amrex;
    // a <- (I+S)^-1 a pointwise (deposited-mass-matrix ion response, see
    // the class docs)
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(a, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<const Real> const& coef = m_coef[0]->const_array(mfi);
        Array4<Real> const& v = a.array(mfi);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Real vx = v(i,j,k,0);
                Real vy = v(i,j,k,1);
                Real vz = v(i,j,k,2);
                HybridPICPCInvIpS(coef(i,j,k,0), coef(i,j,k,1), coef(i,j,k,2),
                                  coef(i,j,k,6), coef(i,j,k,7), coef(i,j,k,8),
                                  vx, vy, vz);
                v(i,j,k,0) = vx;
                v(i,j,k,1) = vy;
                v(i,j,k,2) = vz;
            });
    }
}


template <class T, class Ops>
void HybridPICPC<T,Ops>::CopyCompIn ( const MF& a_src, MF& a_dst, int a_comp,
                                      bool a_add ) const
{
    using namespace amrex;
    // valid region of the container only; ghosts (including the top nodal
    // planes, which are ghost indices of the cell-indexed container) are
    // re-derived by FillBoundary and the closures
    const int nc = a_comp;
    const bool add = a_add;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(a_dst, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<const Real> const& sv = a_src.const_array(mfi);
        Array4<Real> const& dv = a_dst.array(mfi);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                dv(i,j,k,nc) = (add ? dv(i,j,k,nc) : 0.0_rt) + sv(i,j,k);
            });
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::WriteXOut ( const MF& a_src, T& a_x ) const
{
    using namespace amrex;
    for (int n = 0; n < 3; n++) {
        MF& x = *a_x.getArrayVec()[0][n];
        const int nc = n;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(x, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            Array4<const Real> const& sv = a_src.const_array(mfi);
            Array4<Real> const& xv = x.array(mfi);
            amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    xv(i,j,k) = sv(i,j,k,nc);
                });
        }
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::FillGhosts ( MF& a, int a_lev ) const
{
    using namespace amrex;

    const amrex::Geometry& geom = m_geom[a_lev];
    if (!m_fastfill_ok[a_lev]) {
        a.FillBoundary(geom.periodicity());
        return;
    }

    // Single domain-spanning box: every ghost cell that lies inside the
    // periodic domain image is the wrap of a valid cell. Index arithmetic
    // per dimension d with the CELL domain length N_d:
    //   in range [lo, hi]           -> unchanged (hi = lo+N nodal, lo+N-1 cell)
    //   out of range, periodic      -> lo + ((idx-lo) mod N)   in [lo, lo+N)
    //   out of range, not periodic  -> not written (closure territory)
    // A duplicated nodal plane index lo+N is mapped to lo so every ghost
    // derives from the primary plane. The valid region is never written.
    const int ncomp = a.nComp();
    GpuArray<int,3> per{0, 0, 0};
    GpuArray<int,3> len{1, 1, 1};
    for (int d = 0; d < AMREX_SPACEDIM; d++) {
        per[d] = geom.isPeriodic(d) ? 1 : 0;
        len[d] = geom.Domain().length(d);
    }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(a, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box vbx = mfi.validbox();
        // the grown tile (== the whole fab box without tiling, i.e. on GPU)
        const Box gbx = mfi.growntilebox();
        GpuArray<int,3> lo{0, 0, 0};
        GpuArray<int,3> hi{0, 0, 0};
        for (int d = 0; d < AMREX_SPACEDIM; d++) {
            lo[d] = vbx.smallEnd(d);
            hi[d] = vbx.bigEnd(d);
        }
        Array4<Real> const& v = a.array(mfi);
        amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                int s[3] = {i, j, k};
                bool ghost = false;
                for (int d = 0; d < AMREX_SPACEDIM; d++) {
                    if (s[d] < lo[d] || s[d] > hi[d]) {
                        if (!per[d]) { return; }
                        ghost = true;
                        int r = (s[d] - lo[d]) % len[d];
                        if (r < 0) { r += len[d]; }
                        s[d] = lo[d] + r;
                    } else if (per[d] && s[d] == lo[d] + len[d]) {
                        s[d] = lo[d];   // duplicated nodal plane -> primary
                    }
                }
                if (!ghost) { return; }
                for (int n = 0; n < ncomp; n++) {
                    v(i,j,k,n) = v(s[0],s[1],s[2],n);
                }
            });
    }
}

#endif
